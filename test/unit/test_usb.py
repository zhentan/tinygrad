import ctypes
from types import SimpleNamespace
import pytest
from tinygrad.runtime.autogen import libusb
from tinygrad.runtime.support import usb

@pytest.mark.parametrize("failure", [None, "open", "get_device_descriptor", "get_string_descriptor_ascii", "product",
  "kernel_driver_active", "detach_kernel_driver", "reset_device", "set_configuration", "claim_interface", "set_interface_alt_setting", "interrupt"])
def test_usb_constructor_handle_ownership(monkeypatch, failure):
  handle = ctypes.cast(ctypes.pointer(ctypes.c_ubyte()), ctypes.POINTER(libusb.struct_libusb_device_handle))
  device = ctypes.cast(ctypes.pointer(ctypes.c_ubyte()), ctypes.POINTER(libusb.struct_libusb_device))
  closed, claimed = [], []
  api = SimpleNamespace(**{name:getattr(libusb, name) for name in
    ("struct_libusb_device_handle", "struct_libusb_device_descriptor", "libusb_transfer_cb_fn")})

  def call(name, *args):
    if name == failure: return libusb.LIBUSB_ERROR_IO
    if name == "open":
      ctypes.cast(ctypes.byref(args[1]), ctypes.POINTER(type(handle)))[0] = handle
    elif name == "get_device": return device
    elif name == "get_string_descriptor_ascii":
      product = b"unexpected device" if failure == "product" else b"custom test"
      args[2][:len(product)] = product
      return len(product)
    elif name == "kernel_driver_active": return 1
    elif name == "claim_interface": claimed.append(args[0])
    elif name == "set_interface_alt_setting" and failure == "interrupt": raise KeyboardInterrupt("interrupted setup")
    elif name == "close": closed.append(ctypes.cast(args[0], ctypes.c_void_p).value)
    elif name == "strerror": return b"injected USB failure"
    return 0

  for name in ("open", "get_device", "get_device_descriptor", "get_string_descriptor_ascii", "kernel_driver_active", "detach_kernel_driver",
               "reset_device", "set_configuration", "claim_interface", "set_interface_alt_setting", "close", "strerror"):
    def fn(*args, name=name): return call(name, *args)
    fn.__name__ = f"libusb_{name}"
    setattr(api, fn.__name__, fn)
  # Replace the entire library boundary: an unexpected native operation cannot reach hardware.
  monkeypatch.setattr(usb, "libusb", api)

  if failure is None:
    opened = usb.USB3(device)
    assert opened.product == "custom test" and claimed
    assert ctypes.cast(opened.handle, ctypes.c_void_p).value == ctypes.cast(handle, ctypes.c_void_p).value
    assert not closed
  else:
    error = AssertionError if failure == "product" else KeyboardInterrupt if failure == "interrupt" else RuntimeError
    with pytest.raises(error, match=None if failure == "product" else "interrupted setup" if failure == "interrupt" else "injected USB failure"):
      usb.USB3(device)
    assert closed == ([] if failure == "open" else [ctypes.cast(handle, ctypes.c_void_p).value])
