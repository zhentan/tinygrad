import ctypes
from unittest.mock import Mock
from types import SimpleNamespace
import pytest
from tinygrad.runtime.autogen import libusb
from tinygrad.runtime.support import usb

def dma_client(monkeypatch, size=32):
  storage = (ctypes.c_ubyte * size)()
  ptr = ctypes.cast(storage, ctypes.POINTER(ctypes.c_ubyte))
  api = SimpleNamespace(libusb_dev_mem_alloc=Mock(return_value=ptr), libusb_dev_mem_free=Mock(return_value=0),
                        libusb_strerror=lambda _: b"injected DMA error")
  api.libusb_dev_mem_free.__name__ = "libusb_dev_mem_free"
  monkeypatch.setattr(usb, "libusb", api)
  client = object.__new__(usb.USB3)
  client.handle, client._dma_buffers = object(), []
  return client, api, storage

def test_dma_allocation_stays_owned_until_explicit_cleanup(monkeypatch):
  client, api, storage = dma_client(monkeypatch)
  view = client.alloc_dma(len(storage))
  view[3] = 123
  assert storage[3] == 123
  api.libusb_dev_mem_alloc.assert_called_once_with(client.handle, len(storage))
  ptr = api.libusb_dev_mem_alloc.return_value
  del view
  api.libusb_dev_mem_free.assert_not_called()
  client.free_dma_buffers()
  client.free_dma_buffers()
  api.libusb_dev_mem_free.assert_called_once_with(client.handle, ptr, len(storage))
  assert not client._dma_buffers

def test_dma_null_allocation_does_not_fall_back(monkeypatch):
  client, api, _ = dma_client(monkeypatch)
  api.libusb_dev_mem_alloc.return_value = ctypes.POINTER(ctypes.c_ubyte)()
  with pytest.raises(RuntimeError, match="USB DMA allocation failed"): client.alloc_dma(32)
  assert not client._dma_buffers
  client.free_dma_buffers()
  api.libusb_dev_mem_alloc.assert_called_once()
  api.libusb_dev_mem_free.assert_not_called()

def test_dma_free_failure_keeps_ownership_and_is_reported(monkeypatch):
  client, api, _ = dma_client(monkeypatch)
  client.alloc_dma(32)
  owned = list(client._dma_buffers)
  api.libusb_dev_mem_free.return_value = -1
  with pytest.raises(RuntimeError, match="injected DMA error"): client.free_dma_buffers()
  assert client._dma_buffers == owned
  api.libusb_dev_mem_free.assert_called_once()

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
