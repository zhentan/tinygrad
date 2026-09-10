# Native NV uses reserved firmware SRAM, with F0 readback by default and opt-in F2 DMA readback.
import ctypes, functools, struct
from typing import Any, cast
from tinygrad.runtime.autogen import libusb
from tinygrad.helpers import round_up, ceildiv, to_tuple
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp, UPat, Ops, PatternMatcher, KernelInfo
from tinygrad.device import Buffer, BufferSpec, Device
from tinygrad.runtime.support.hcq2 import HCQInfo, make_submit, HCQ_RUNTIME_DEV, ccall, ccheck, patch, timeline, timeline_value


# *****************

def make_buf(devs, slot:int=0, tag:str="signal") -> UOp: return UOp.placeholder((1,), dtypes.uint64, slot, device=devs, volatile=True, tag=tag)

def _libusb(devs, dep:tuple[UOp, ...], fn:str, *args) -> UOp:
  ret = ccall(getattr(libusb, fn), make_buf(devs, tag="usb_handle").after(*dep).index(0).load(), *args)
  return ccheck(ret, args[-2] if fn == "libusb_control_transfer" else 0)

def usb_bulk(devs, dep, endpoint:int, data:UOp, length, timeout:int=1000) -> UOp:
  actual = UOp.placeholder((1,), dtypes.int32, device=HCQ_RUNTIME_DEV.value, volatile=True, tag="usb_scratch")
  done = _libusb(devs, dep, "libusb_bulk_transfer", endpoint, data, length, actual.index(0), timeout)
  return ccheck(actual.after(done).index(0).load(), length)

def usb_stream(devs, dep:tuple[UOp, ...], addr:UOp, data:UOp, nbytes:int, write:bool) -> UOp:
  boundary = UOp.const(1 << 32, dtypes.uint64)
  checked_addr = ccheck(((addr < boundary) & (boundary < addr + UOp.const(nbytes, dtypes.uint64))).cast(dtypes.int32))
  hdr = patch(UOp.placeholder((12,), dtypes.uint8, device=HCQ_RUNTIME_DEV.value, tag="usb_scratch"), [(0, addr)],
              struct.pack('<QI', 0, nbytes // 4)).after(*dep, checked_addr)
  arm = _libusb(devs, (), "libusb_control_transfer",
                0x40, 0xF0, (addr < boundary).where(0, 0x20).cast(dtypes.int) | (0x40 if write else 0) | (0x0F << 8),
                1 if write else 2, hdr.index(0), 12, 5000)
  return usb_bulk(devs, (arm,), 0x02 if write else 0x81, data, nbytes)

def usb_view(b:UOp) -> tuple[UOp, int, tuple[UOp, ...]]:
  if b.op is Ops.AFTER:
    base, offset, deps = usb_view(b.src[0])
    return base, offset, deps + b.src[1:]
  if b.op is Ops.BITCAST: return usb_view(b.src[0])
  if b.op is Ops.SHRINK:
    base, offset, deps = usb_view(b.src[0])
    return base, offset + b.src[1].val * b.dtype.itemsize, deps
  return b, 0, ()

def usb_address(b:UOp, offset:int) -> UOp:
  cell = UOp.placeholder((1,), dtypes.uint64, device=HCQ_RUNTIME_DEV.value, tag="usb_scratch")
  address = b.getaddr((HCQ_RUNTIME_DEV.value,)) + UOp.const(offset, dtypes.uint64)
  return patch(cell, [(0, address)]).index(0).load()

def usb_load(b:UOp, idx:UOp, dt) -> UOp|None:
  base, offset, deps = usb_view(b)
  if base.op is not Ops.PARAM or all(d.startswith("CPU") for d in to_tuple(b.device)) or base.tag == "usb_handle" or \
     (isinstance(base.tag, tuple) and base.tag[0] == "hcq_host"): return None
  got = UOp.placeholder((1,), dt, device=HCQ_RUNTIME_DEV.value, tag="usb_scratch")
  addr = usb_address(base, offset) + (idx*dt.itemsize).cast(dtypes.uint64)
  devs = to_tuple(b.device)
  return got.after(usb_stream(devs, deps, addr, got.index(0), dt.itemsize, False)).index(0).load()

def usb_write_subdword(devs:tuple[str, ...], deps:tuple[UOp, ...], addr:UOp, lane:UOp, v:UOp) -> UOp:
  byte_en = (UOp.const((1 << v.dtype.itemsize) - 1, dtypes.int) << lane).cast(dtypes.int)
  payload = (v.cast(dtypes.uint32) << (lane * UOp.const(8, dtypes.int))).cast(dtypes.uint32)
  hdr = patch(UOp.placeholder((12,), dtypes.uint8, device=HCQ_RUNTIME_DEV.value, tag="usb_scratch"),
              [(0, addr - lane.cast(dtypes.uint64)), (8, payload)], bytes(12)).after(*deps)
  boundary = UOp.const(1 << 32, dtypes.uint64)
  return _libusb(devs, (), "libusb_control_transfer", 0x40, 0xF0,
                 (addr < boundary).where(0, 0x20).cast(dtypes.int) | 0x40 | (byte_en << UOp.const(8, dtypes.int)),
                 0, hdr.index(0), 12, 5000)

def usb_write_one(base:UOp, offset:int, deps:tuple[UOp, ...], devs:tuple[str, ...], idx:UOp, v:UOp) -> UOp:
  idx_offset = idx * v.dtype.itemsize
  addr = usb_address(base, offset) + idx_offset.cast(dtypes.uint64)
  if v.dtype.itemsize < 4:
    lane = (idx_offset.cast(dtypes.int) + UOp.const(offset, dtypes.int)) & UOp.const(3, dtypes.int)
    return usb_write_subdword(devs, deps, addr, lane, v)
  val = (s:=UOp.placeholder((1,), v.dtype, device=HCQ_RUNTIME_DEV.value, tag="usb_scratch")).after(s.index(0).store(v))
  return usb_stream(devs, deps, addr, val.index(0), v.dtype.itemsize, True)

def usb_write(b:UOp, idx:UOp, v:UOp) -> UOp|None:
  base, offset, deps = usb_view(b)
  if base.op is not Ops.PARAM or all(d.startswith("CPU") for d in to_tuple(b.device)) or base.tag == "usb_handle" or \
     (isinstance(base.tag, tuple) and base.tag[0] == "hcq_host"): return None
  devs = to_tuple(b.device)
  if idx.op is Ops.STACK or v.op is Ops.STACK:
    if idx.op is not Ops.STACK or v.op is not Ops.STACK or len(idx.src) != len(v.src):
      raise RuntimeError("USB HCQ scatter requires paired indices and values")
    for lane_idx, lane_value in zip(idx.src, v.src):
      deps = (usb_write_one(base, offset, deps, devs, lane_idx, lane_value),)
    return deps[0]
  return usb_write_one(base, offset, deps, devs, idx, v)

def usb_idle(devs) -> UOp:
  v = usb_load(timeline(devs).after(loop:=UOp.loop(0)), UOp.const(0, dtypes.int), dtypes.uint64)
  assert v is not None
  return v.end(loop, v < timeline_value(devs))

def usb_wait_value(b:UOp, value:int, deps:tuple[UOp, ...]) -> UOp:
  v = usb_load(b.after(*deps).after(loop:=UOp.loop(0)), UOp.const(0, dtypes.int), b.dtype)
  assert v is not None
  return v.end(loop, v < UOp.const(value, b.dtype))

def usb_scsi(devs, read:bool, nbytes:int, slot_start:int=0, deps:tuple[UOp, ...]=()) -> UOp:
  return _libusb(devs, (usb_idle(devs).after(*deps),), "libusb_control_transfer", 0x40, 0xF2,
                 ceildiv(nbytes, 512) | (0x8000 if read else 0),
                 (ceildiv(nbytes, 0x4000) & 0xFF) << 8 | slot_start, UOp.const(0, dtypes.uint64), 0, 1000)

def usb_stage_copy(dst:UOp, src:UOp) -> UOp|None:
  host_devices = {"CPU", "DISK", "NPY", "PYTHON"}
  def is_usb_device(buf:UOp) -> bool:
    return any(d.split(":")[0] not in host_devices and getattr(Device[d], "pm_stage_copy", None) is pm_usb_stage for d in to_tuple(buf.device))
  if (cin:=is_usb_device(dst)) == is_usb_device(src): return None

  usb_dev = cast(Any, Device[(devs:=to_tuple((dst if cin else src).device))[0]])
  sram_readback = not cin and getattr(usb_dev, "usb_sram_readback", False) is True
  win = usb_dev.usb_sram if cin or sram_readback else usb_dev.usb_readback
  # Stable scratch parameters let HCQ cache prepared copies; concrete inputs must not be retained.
  stage = _usb_stage_copy.__wrapped__ if any(u.op is Ops.BUFFER for u in UOp.sink(dst, src).toposort()) else _usb_stage_copy
  return stage(dst, src, devs, win, cin, sram_readback)

@functools.cache
def _usb_stage_copy(dst:UOp, src:UOp, devs:tuple[str, ...], win:Buffer, cin:bool, sram_readback:bool=False) -> UOp:
  total, ops = dst.nbytes(), []
  if cin or sram_readback:
    slot_start, slot_offset = divmod(win.host.addr - 0xf000, 0x4000)
    if slot_offset or not 0 <= slot_start <= 0xff: raise RuntimeError(f"invalid USB SRAM staging address {win.host.addr:#x}")
    if sram_readback and slot_start * 0x4000 + win.size > 0x80000: raise RuntimeError("USB SRAM readback exceeds controller memory")
  for off in range(0, total, win.size): # off and nb are bytes, the two ends of the copy can have different dtypes
    stage = UOp.from_buffer(win)[0:(nb:=min(win.size, total - off))]
    s, d = src[off // src.dtype.itemsize:(off + nb) // src.dtype.itemsize], dst[off // dst.dtype.itemsize:(off + nb) // dst.dtype.itemsize]
    if cin:
      prep:tuple[UOp, ...] = ()
      if not to_tuple(s.device)[0].startswith("CPU"):
        host = UOp.placeholder((round_up(nb, 512),), dtypes.uint8, device="CPU", tag="usb_staging")[0:nb]
        prep, s = (s.copy_to_device("CPU").call(host, s),), host
      push = usb_bulk(devs, (usb_scsi(devs, False, nb, slot_start),),
                      0x02, usb_address(s, 0), round_up(nb, 512), 10000)
      ops += [*prep, push.sink(arg=KernelInfo("hcq_copyin")).call(stage, s, name="hcq_copyin", aux=HCQInfo(devs)),
              stage.copy_to_device(d.device).call(d, stage)]
    elif sram_readback:
      # F2 releases from slot zero: transfer the reserved prefix and discard it on the host.
      wire = round_up((prefix:=slot_start * 0x4000) + nb, 512)
      pad = UOp.placeholder((wire,), dtypes.uint8, device="CPU", tag="usb_staging")
      cq = UOp.placeholder((0x1000,), dtypes.uint8, device=devs, tag="usb_read_cq")
      wait = UOp(Ops.INS, src=(timeline(devs), timeline_value(devs)), arg=("wait", dtypes.void))
      release = UOp(Ops.INS, src=(cq[12:16].bitcast(dtypes.uint32), UOp.const(0, dtypes.uint32)), arg=("store", dtypes.void))
      submit = make_submit(wait, s.copy_to_device(stage.device).call(stage, s), release, devs=devs, queue="COPY:0").after(
        usb_scsi(devs, True, wire))
      pull = usb_bulk(devs, (submit,), 0x81, usb_address(pad, 0), wire, 10000)
      ops += [pull.sink(arg=KernelInfo("hcq_copyout")).call(pad, stage, s, name="hcq_copyout", aux=HCQInfo(devs)),
              pad[prefix:prefix + nb].copy_to_device(d.device).call(d, pad[prefix:prefix + nb])]
    else:
      pad = UOp.placeholder((round_up(nb, 512),), dtypes.uint8, device="CPU", tag="usb_staging")[0:nb]
      done = UOp.placeholder((1,), dtypes.uint64, device=devs, tag="usb_readback_done")
      reset = usb_write_one(done, 0, (), devs, UOp.const(0, dtypes.int), UOp.const(0, dtypes.uint64))
      signal = UOp(Ops.INS, src=(done, UOp.const(1, dtypes.uint64)), arg=("store", dtypes.void))
      # This direct submit bypasses the batch scheduler: wait for prior compute before reading its output.
      wait = UOp(Ops.INS, src=(timeline(devs), timeline_value(devs)), arg=("wait", dtypes.void))
      submit = make_submit(wait, s.copy_to_device(stage.device).call(stage, s), signal, devs=devs, queue="COPY:0").after(reset)
      pull = usb_stream(devs, (usb_wait_value(done, 1, (submit,)),), usb_address(stage, 0), pad.index(0), round_up(nb, 4), False)
      ops += [pull.sink(arg=KernelInfo("hcq_copyout")).call(pad, stage, s, name="hcq_copyout", aux=HCQInfo(devs)),
              pad.copy_to_device(d.device).call(d, pad)]
  return UOp(Ops.LINEAR, src=tuple(ops))
pm_usb_stage = PatternMatcher([(UPat(Ops.CALL, src=(UPat(Ops.COPY), UPat(name="dst"), UPat(name="src"))), usb_stage_copy)])

pm_usb_hostio = PatternMatcher([
  (UPat(Ops.LOAD, src=(UPat(Ops.INDEX, src=(UPat(name="b"), UPat(name="idx"))),),
        name="ld"), lambda b, idx, ld: usb_load(b, idx, ld.dtype)),
  (UPat(Ops.STORE, src=(UPat(Ops.INDEX, src=(UPat(name="b"), UPat(name="idx"))), UPat(name="v"))), usb_write)])

def usb_handle_buffer(ctx:Any) -> Buffer:
  if (buf:=getattr(ctx, "_usb_hcq_handle", None)) is None:
    handle = ctypes.cast(ctx.iface.pci_dev.usb.usb.handle, ctypes.c_void_p).value
    if handle is None: raise RuntimeError("USB HCQ has no libusb handle")
    buf = Buffer("CPU", 1, dtypes.uint64, options=BufferSpec(nolru=True), preallocate=True)
    buf.host.view(fmt='Q')[0] = handle
    ctx._usb_hcq_handle = buf
  return cast(Buffer, buf)

pm_usb_bufferize = PatternMatcher([
  (UPat(Ops.PARAM, tag="usb_handle"), usb_handle_buffer),
  (UPat(Ops.PARAM, tag="usb_readback_done"), lambda ctx: ctx.usb_readback_done),
  (UPat(Ops.PARAM, tag="usb_read_cq"), lambda ctx: ctx.usb_read_cq),
])
