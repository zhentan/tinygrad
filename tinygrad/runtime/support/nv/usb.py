import struct
from tinygrad.runtime.autogen import libusb
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp
from tinygrad.runtime.support.hcq2 import HCQ_RUNTIME_DEV, ccall, ccheck, patch

def make_buf(devs, slot:int=0, tag:str="signal") -> UOp: return UOp.placeholder((1,), dtypes.uint64, slot, device=devs, volatile=True, tag=tag)

def _libusb(devs, dep:tuple[UOp, ...], fn:str, *args) -> UOp:
  ret = ccall(getattr(libusb, fn), make_buf(devs, tag="usb_handle").after(*dep).index(0).load(), *args)
  return ccheck(ret, args[-2] if fn == "libusb_control_transfer" else 0)

def usb_bulk(devs, dep, endpoint:int, data:UOp, length, timeout:int=1000) -> UOp:
  actual = UOp.placeholder((1,), dtypes.int32, device=HCQ_RUNTIME_DEV.value, volatile=True, tag="usb_scratch")
  done = _libusb(devs, dep, "libusb_bulk_transfer", endpoint, data, length, actual.index(0), timeout)
  return ccheck(actual.after(done).index(0).load(), length)

def usb_stream(devs, dep:tuple[UOp, ...], addr:UOp, data:UOp, nbytes:int, write:bool) -> UOp:
  hdr = patch(UOp.placeholder((12,), dtypes.uint8, device=HCQ_RUNTIME_DEV.value, tag="usb_scratch"), [(0, addr)],
              struct.pack('<QI', 0, nbytes // 4)).after(*dep)
  arm = _libusb(devs, (), "libusb_control_transfer",
                0x40, 0xF0, (addr < UOp.const(1 << 32, dtypes.uint64)).where(0, 0x20).cast(dtypes.int) | (0x40 if write else 0) | (0x0F << 8),
                1 if write else 2, hdr.index(0), 12, 5000)
  return usb_bulk(devs, (arm,), 0x02 if write else 0x81, data, nbytes)

