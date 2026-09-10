import unittest, contextlib, ctypes, gc, struct, numpy as np
from dataclasses import replace
from unittest.mock import Mock, patch
from tinygrad import Device, Tensor, TinyJit, Variable, dtypes, GlobalCounters
from tinygrad.device import Buffer, Compiled
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import Context, dedup, partition, unwrap
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher, KernelInfo
from tinygrad.engine.realize import compile_linear, link_linear, lower_and_compile, run_linear
from tinygrad.codegen import do_to_program
from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.renderer.llvmir import CPULLVMRenderer
from tinygrad.renderer import Estimates
from tinygrad.runtime.autogen import libc, libusb
from tinygrad.runtime.support.usb import USBMMIOInterface, CustomASM24Controller
from tinygrad.runtime.support.nv.usb import usb_stream, usb_write_one, usb_idle
from tinygrad.runtime.support.c import init_c_struct_t
import tinygrad.runtime.support.hcq2 as hcq2
from tinygrad.runtime.support.hcq2 import HCQ_DEVS, all_devices_in, hcq_compile_cache, link_linear_cache
from test.helpers import call_is_hcq
from test.mockgpu.usb import MockUSB

@contextlib.contextmanager
def rt_buffers():
  calls, orig = [], Compiled.rt_buffer
  def track(dev, *args, **kwargs):
    calls.append(dev)
    return orig(dev, *args, **kwargs)
  with patch.object(Compiled, "rt_buffer", track): yield calls

def chain(x:Tensor, n:int) -> Tensor:
  for _ in range(n): x = (x + 1).contiguous()
  return x

@contextlib.contextmanager
def encoded_batches():
  batches, orig = [], hcq2.lower_and_compile
  def track(l, *args, **kwargs):
    batches.extend(c.without_after for c in l.src if call_is_hcq(c))
    return orig(l, *args, **kwargs)
  with patch.object(hcq2, "lower_and_compile", track): yield batches

def eager_chain(x:Tensor, n:int=64) -> Tensor: # at hcq_compile's use_rt bound: an eager linear this big bakes its inputs and borrows ring slots
  for _ in range(n): x = (x + 1).contiguous()
  return x.realize()

def patch_words(batch:UOp) -> list[UOp]:
  return [w for s in batch.src[0].toposort() if s.op is Ops.STORE and s.src[0].op is Ops.INDEX and s.src[0].src[1].op is Ops.STACK
          and s.src[1].op is Ops.STACK for w in s.src[1].src]

def rt_params(batch:UOp) -> list[str]:
  return dedup([u.arg.name for w in patch_words(batch) for u in w.toposort() if u.op is Ops.PARAM and u.arg.addrspace is AddrSpace.GLOBAL])

def cpu_buf(size:int=1, dtype=dtypes.uint8, **kwargs) -> UOp: return UOp.placeholder((size,), dtype, device="CPU", **kwargs)

def lower_hcq(body:UOp) -> UOp:
  return unwrap(hcq2.lower_call(UOp.sink(body, arg=KernelInfo("test")).call(aux=hcq2.HCQInfo(("CPU",)))))

class TestHCQ2Deps(unittest.TestCase):
  def test_disjoint_write_preserves_dependencies(self):
    b = UOp.param(0, dtypes.uint8, 16, device="CPU")
    for write in ([], [0]):
      tracker = hcq2.HCQDepsTracker()
      tracker.access_resources([b.shrink(((0, 4),))], write, 0)
      self.assertEqual(tracker.access_resources([b.shrink(((4, 8),))], [0], 1), [])
      self.assertEqual(tracker.access_resources([b.shrink(((0, 4),))], [0], 2), [0])

  def test_partial_write_preserves_dependencies(self):
    b = UOp.param(0, dtypes.uint8, 16, device="CPU")
    for write in ([], [0]):
      tracker = hcq2.HCQDepsTracker()
      tracker.access_resources([b], write, 0)
      self.assertEqual(tracker.access_resources([b.shrink(((4, 12),))], [0], 1), [0])
      self.assertEqual(tracker.access_resources([b.shrink(((0, 4),))], [0], 2), [0])
      self.assertEqual(tracker.access_resources([b.shrink(((12, 16),))], [0], 3), [0])
      self.assertEqual(tracker.access_resources([b.shrink(((4, 12),))], [], 4), [1])

@unittest.skipUnless(all_devices_in(Device.DEFAULT, HCQ_DEVS), "hcq2 device required")
class TestHCQ2Schedule(unittest.TestCase):
  @staticmethod
  def input(value:int=2) -> Tensor: return Tensor.full((4,), value, dtype=dtypes.int32).contiguous().realize()

  def compiled(self, n:int, jit=False):
    x, inputs = self.input(), []
    if jit:
      f = TinyJit(lambda a: chain(a, n).realize())
      f(x)
      return f(x), f.captured._linear, [x.uop.base]
    out = chain(x, n)
    return out, compile_linear(out.schedule_linear(), input_uops=inputs, cache=True), inputs

  def test_jit_has_no_rt_buffers(self):
    dev = Device[Device.DEFAULT]
    rings = [dev.rt_buffer(True, host) for host in (False, True)]
    ranges = [(b._buf, b._buf + b.nbytes) for b in rings]
    for n in (1, 65):
      with self.subTest(kernels=n):
        x, f = self.input(), TinyJit(lambda a: chain(a, n).realize())
        for _ in range(2): f(x)
        for u in f.captured.linear.toposort():
          if u.op is Ops.BUFFER and (buf:=u.buffer).device == dev.device:
            addr = buf._buf
            self.assertFalse(any(addr < end and start < addr + buf.nbytes for start, end in ranges))

  def test_small_eager_cached(self):
    _, compiled, inputs = self.compiled(1)
    linked = link_linear(compiled, input_uops=inputs)
    self.assertIs(link_linear(compiled, input_uops=inputs), linked)

  def test_profile_slots_survive_indirect_access(self):
    pm = PatternMatcher([(UPat((Ops.LOAD, Ops.STORE), src=(UPat(Ops.INDEX, src=(UPat.var("buf"), UPat())),), allow_any_len=True),
                          lambda buf: hcq2.rt_addr(buf, "CPU") if hcq2.unwrap_view(buf)[0].tag == "slots" else None)])
    with patch.object(Device[Device.DEFAULT], "pm_lower", pm):
      compiled = compile_linear(Tensor.ones(4).contiguous().schedule_linear(), profile=True)
    self.assertFalse(any(param.op is Ops.PARAM and (param.arg.name or "").startswith("slots_")
                         for param in compiled.src[0].without_after.src[0].toposort()))
    call = link_linear(compiled).src[0].without_after
    ((device, index),) = call.arg.aux.slots
    self.assertEqual(device, Device.DEFAULT)
    self.assertEqual(call.src[1 + index].buffer.dtype, dtypes.uint64)

  def test_large_eager_not_cached(self):
    _, compiled, inputs = self.compiled(65)
    linked = link_linear(compiled, input_uops=inputs)
    self.assertIsNot(link_linear(compiled, input_uops=inputs), linked)
    self.assertNotIn(compiled, link_linear_cache)

  def test_double_compile(self):
    for n in (1, 65):
      for jit in (False, True):
        with self.subTest(kernels=n, jit=jit):
          out, compiled, inputs = self.compiled(n, jit=jit)
          linked = link_linear(compiled, input_uops=inputs, allow_cache=not jit)
          before = tuple(inputs)
          with rt_buffers() as borrowed:
            for linear in (compiled, linked):
              self.assertIs(compile_linear(linear, input_uops=inputs, cache=not jit), linear)
          self.assertEqual(tuple(inputs), before)
          self.assertFalse(borrowed)
          run_linear(linked, input_uops=inputs, jit=True, wait=True)
          self.assertEqual(out.tolist(), [2 + n] * 4)

  def test_double_link(self):
    for n in (1, 65):
      for jit in (False, True):
        with self.subTest(kernels=n, jit=jit):
          out, compiled, inputs = self.compiled(n, jit=jit)
          linked = link_linear(compiled, input_uops=inputs, allow_cache=not jit)
          with rt_buffers() as borrowed:
            again = link_linear(linked, input_uops=inputs, allow_cache=not jit)
          self.assertIs(again, linked)
          self.assertFalse(borrowed)
          run_linear(again, input_uops=inputs, jit=True, wait=True)
          self.assertEqual(out.tolist(), [2 + n] * 4)

  def test_jit_new_inputs_each_call(self):
    @TinyJit
    def f(a, b): return (a * b + a).contiguous().realize()
    ins = [(Tensor.full((23,), float(i)).contiguous().realize(), Tensor.full((23,), 2.0).contiguous().realize()) for i in range(6)]
    for a, b in ins[:3]: f(a, b).tolist() # warm the jit and the copyout

    before = len(hcq_compile_cache)
    self.assertEqual([f(a, b).tolist() for a, b in ins[3:]], [[i * 3.0] * 23 for i in range(3, 6)])
    self.assertEqual(len(hcq_compile_cache), before)

  def test_jit_symbolic(self):
    @TinyJit
    def f(a): return (a + 1).sum().contiguous().realize()
    a = Tensor.rand(3, 10).contiguous().realize()
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      np.testing.assert_allclose(f(a[:, :vi]).item(), (a[:, :i] + 1).sum().item(), atol=1e-5, rtol=1e-5)

  def test_map_cpu_buffer_preserves_contents(self):
    src = Buffer("CPU", 16, dtypes.uint8, preallocate=True)
    data = bytes(range(16))
    src.host[:] = data
    src.get_buf(Device.DEFAULT)
    self.assertEqual(bytes(src.as_memoryview()), data)

  def test_rt_patches_are_inputs_and_vars_only(self):
    x = Tensor.rand(17, 33).contiguous().realize()
    with encoded_batches() as batches:
      @TinyJit
      def f(a): return (a.sin() * 3).contiguous().realize()
      for _ in range(3): f(x)
      eager_chain(x)

    jit, eager = partition(batches, lambda c: c.arg.aux.table >= 0)
    self.assertTrue(jit and eager, f"want both kinds of batch, got {len(jit)} jit and {len(eager)} eager")
    for c in batches:
      self.assertTrue(all(n.startswith(("inputs_", "timeline_")) for n in rt_params(c)), f"runtime patch reads {rt_params(c)}")
      self.assertFalse([u for w in patch_words(c) for u in w.toposort() if u.op is Ops.GETADDR], "addresses bake at link time")
    self.assertTrue(any(n.startswith("inputs_") for c in jit for n in rt_params(c)), "the jit patches its input addresses in")
    self.assertFalse(any(n.startswith("inputs_") for c in eager for n in rt_params(c)), "eager bakes its input addresses")

  def test_programs_are_not_call_args(self):
    # a program is a link-time patch a cmdbuf word addresses: it rides inside that word, no arg or param of its own
    def nargs(n):
      x = Tensor.ones(16).contiguous().realize()
      with encoded_batches() as batches:
        @TinyJit
        def f(a):
          for i in range(n): a = (a * (i + 1.5)).contiguous()
          return a.realize()
        for _ in range(3): f(x)
      return max(c.arg.aux.nargs for c in batches)
    self.assertEqual(nargs(2), nargs(12))

  def test_caches_hold_no_buffers(self):
    # an eager template caches without its buffers and the jit's linear compiles once uncached: freeing the tensors frees the device memory
    def step(i):
      buf = Buffer("NPY", 1024, dtypes.float32, initial_value=struct.pack("f", i) * 1024)
      x = Tensor(UOp.from_buffer(buf)).to(Device.DEFAULT).realize()
      @TinyJit
      def f(a): return (a * 2 + 1).contiguous().realize()
      for _ in range(3): out = f(x)
      self.assertEqual(out.to("CPU").tolist(), [2.0 * i + 1] * 1024)
    step(1) # warms the programs, templates and rings
    gc.collect()
    used = GlobalCounters.mem_used
    for i in range(2, 5): step(i)
    gc.collect()
    self.assertEqual(GlobalCounters.mem_used, used)

  def test_device_state_survives_as_link_refs(self):
    # a buffer the commands only address, never a param of the body, is kept by the linked call as a ref of what its getaddr resolved into
    dev = Device[Device.DEFAULT]
    names = {"AMD": () if getattr(dev, "is_aql", False) else ("scratch",), # the aql descriptor holds the scratch, nothing addresses it
             "NV": ("timeline",), "QCOM": ("_stack", "dummy")}[Device.DEFAULT.split(":")[0]]
    @TinyJit
    def f(a): return (a * 2 + 1).contiguous().realize()
    x = Tensor.ones(16).contiguous().realize()
    for _ in range(3): f(x)
    call = f.captured.linear.src[0]
    self.assertIs(call.op, Ops.AFTER, "the linked call sits after its refs")
    refs = [u.buffer for u in call.src[1:] if u.op is Ops.BUFFER]
    for n in names: self.assertTrue(any(r is getattr(dev, n) for r in refs), f"{n} is not a ref of the call")

  def test_usb_renumbering(self):
    programs = []
    with Context(HCQ_RUNTIME_DEV="CPU"), patch("tinygrad.codegen.do_to_program", wraps=do_to_program) as build:
      for ids in ((0, 1, 2, 3), (2, 0, 3, 1), (1, 0, 2, 3), (0, 1, 3, 2), (100, 101, 102, 103)):
        with self.subTest(ids=ids):
          regs = [UOp.placeholder((1,), dtypes.uint32, slot=i, addrspace=AddrSpace.REG) for i in ids[:2]]
          a, b = [r.after(r.index(0).store(v)) for r, v in zip(regs, (3, 5))]
          i, j = [UOp.range(UOp(Ops.NOOP), n, dtype=dtypes.void, src=(a, b)) for n in ids[2:]]
          out = cpu_buf(dtype=dtypes.uint32, tag="out")
          body = out.index(0).store(a.after(i, j).index(0).load()*10 + b.index(0).load()).end(j, UOp.const(False)).end(i, UOp.const(False))
          compiled = lower_and_compile(UOp(Ops.LINEAR, src=(lower_hcq(body),)))
          programs.append(compiled.src[0].without_after.src[0])
          self.assertIs(programs[-1], programs[0])
          linear = hcq2.hcq_link(compiled, allow_cache=False)
          run_linear(linear, jit=True)
          self.assertEqual(linear.src[0].without_after.src[1].buffer.host.view(fmt='I')[0], 35)
      self.assertLessEqual(build.call_count, 1)

  def test_patched_view(self):
    with Context(HCQ_RUNTIME_DEV="CPU"):
      ctx = hcq2.EncodeCtx(("CPU",))
      inner = hcq2.patch(cpu_buf(8, tag="inner"), [(4, UOp.const(42, dtypes.uint32))], bytes(8))
      inner = unwrap(hcq2.hoist_links(ctx, inner))
      outer = hcq2.patch(cpu_buf(8, tag="outer"), [(0, inner[4:8].getaddr("CPU"))])
      with patch.object(hcq2, "EncodeCtx", return_value=ctx): call = lower_hcq(outer.bitcast(dtypes.uint64).index(0).load())
      self.assertEqual(call.without_after.arg.aux.nargs, 1)
      self.assertTrue(all(s.op is Ops.STORE for s in call.src[1:]))
      linked = hcq2.hcq_link(UOp(Ops.LINEAR, src=(call,)), allow_cache=False).src[0]
      inner_buf, outer_buf = linked.src[1].buffer, linked.without_after.src[1].buffer
      self.assertEqual(inner_buf.host.view(fmt='I')[1], 42)
      self.assertEqual(outer_buf.host.view(fmt='Q')[0], inner_buf._buf + 4)

class TestHCQ2Link(unittest.TestCase):
  def test_post_encode_rewrites_callback_output(self):
    emitted = UOp.placeholder((1,), dtypes.uint32, device="CPU", tag="emitted")
    encode = PatternMatcher([(UPat(Ops.CUSTOM_FUNCTION, arg="emit"), lambda emitted=emitted: emitted.index(0).load())])
    post = PatternMatcher([(UPat(Ops.LOAD, src=(UPat(Ops.INDEX),)), lambda: UOp.const(7, dtypes.uint32))])
    body = UOp.custom_function("emit")
    with patch.object(Device["CPU"], "pm_encode", encode), patch.object(Device["CPU"], "pm_lower", post):
      call = hcq2.lower_call(UOp.sink(body, arg=KernelInfo("test_post_encode")).call(aux=hcq2.HCQInfo(("CPU",))))
    assert call is not None
    self.assertEqual(call.without_after.arg.aux.nargs, 0)

  def test_single_link_word(self):
    buf, word = Buffer("CPU", 1, dtypes.uint64, preallocate=True), 0x12345678
    uop = UOp.from_buffer(buf)
    store = uop.index(UOp.const(0, dtypes.int)).store(UOp.const(word, dtypes.uint32).cast(dtypes.uint64))
    hcq2.hcq_link(UOp(Ops.LINEAR, src=(uop.after(store),)), allow_cache=False)
    self.assertEqual(buf.host.view(fmt='Q')[0], word)

  def test_usb_link_subdword_patch_preserves_neighboring_bytes(self):
    for address in (0x10000000, 0x800000000):
      for offset in (1, 2, 3, 132):
        with self.subTest(address=address, offset=offset):
          memory = bytearray([0xA5] * 256)
          controller = object.__new__(CustomASM24Controller)
          def control(request, value, mode, data, timeout):
            self.assertEqual((request, value & 0xFF, mode, timeout), (0xF0, 0x60 if address >> 32 else 0x40, 0, 5000))
            target, payload = struct.unpack('<QI', data)
            self.assertEqual(target & 3, 0)
            self.assertTrue(0 < (value >> 8) <= 15)
            for lane in range(4):
              if value >> (8 + lane) & 1: memory[target - address + lane] = payload >> (8 * lane) & 0xFF
          controller.usb = Mock(spec=['control_write'], control_write=control)
          buf = Buffer("CPU", 256, dtypes.uint8, preallocate=True)
          with patch.object(buf, "_storage", replace(buf.get_storage(), host=USBMMIOInterface(controller, address, 256, "B"))):
            patched = hcq2.patch(UOp.from_buffer(buf), [(offset, UOp.const(0xBEEF, dtypes.uint16))])
            hcq2.hcq_link(UOp(Ops.LINEAR, src=(patched,)), allow_cache=False)
          expected = bytearray([0xA5] * 256)
          expected[offset:offset + 2] = b'\xef\xbe'
          self.assertEqual(memory, expected)

  def test_failed_binary_upload_is_not_cached(self):
    for previous, restore_previous in ((None, False), (b'old!', False), (b'old!', True)):
      with self.subTest(previous=previous, restore_previous=restore_previous):
        buf, usb = Buffer("CPU", 4, dtypes.uint8, preallocate=True), MockUSB(bytearray(4))
        def link(blob):
          store = UOp.from_buffer(buf).store(UOp(Ops.BINARY, arg=blob).bitcast(dtypes.uint8))
          return hcq2.hcq_link(UOp(Ops.LINEAR, src=(store,)), allow_cache=False)
        def fail_write(address, data):
          usb.mem[address:address+2] = data[:2] # a failed USB transfer can have partially changed the buffer
          raise RuntimeError("injected USB transfer failure")
        with patch.object(buf, "_storage", replace(buf.get_storage(), host=USBMMIOInterface(usb, 0, 4, "B"))):
          if previous is not None: link(previous)
          with patch.object(usb, "pcie_mem_write", side_effect=fail_write) as failed:
            with self.assertRaisesRegex(RuntimeError, "injected USB transfer failure"): link(b'new!')
            failed.assert_called_once()
          self.assertEqual(usb.mem[:2], b'ne')
          expected = previous if restore_previous else b'new!'
          with patch.object(usb, "pcie_mem_write", wraps=usb.pcie_mem_write) as write:
            link(expected)
            self.assertEqual(usb.mem, expected)
            write.assert_called_once_with(0, expected)
            link(expected)
            write.assert_called_once() # only successful uploads may be reused

@unittest.skipUnless(isinstance(Device["CPU"].renderer, CStyleLanguage), "CALL is rendered in C style only")
class TestHCQ2FFI(unittest.TestCase):
  @staticmethod
  def _link(body:UOp) -> UOp:
    call = hcq2.lower_call(UOp.sink(body, arg=KernelInfo("test_ffi")).call(aux=hcq2.HCQInfo(("CPU",))))
    assert call is not None
    return hcq2.hcq_link(lower_and_compile(UOp(Ops.LINEAR, src=(call,))), allow_cache=False)

  @classmethod
  def _run(cls, body:UOp) -> list[Buffer]:
    linear = cls._link(body)
    run_linear(linear, jit=True)
    return [u.buffer for u in linear.src[0].without_after.src[1:] if u.op is Ops.BUFFER]

  def test_ffi_ccall(self):
    with Context(HCQ_RUNTIME_DEV="CPU"):
      out = cpu_buf(dtype=dtypes.int32, slot=1, volatile=True, tag="ffi_result")
      bufs = self._run(out.index(0).store(hcq2.ccall(libc.dll.ffs, 0x10)))
    self.assertEqual(next(b for b in bufs if b.dtype is dtypes.int).host.view(fmt='i')[0], 5)

  def test_fence_preserves_preincrement_timeline_snapshot(self):
    with Context(HCQ_RUNTIME_DEV="CPU", DEBUG=0, PROFILE=0):
      tl = hcq2.timeline(("CPU",))
      slots = cpu_buf(dtype=dtypes.uint64, volatile=True, tag="fence_slots")
      out = cpu_buf(2, dtype=dtypes.uint64, volatile=True, tag="fence_values")
      fence = UOp.custom_function("hcq_fence", slots)
      waited = out.after(fence).index(0).store(hcq2.timeline_value(("CPU",)))
      signaled = out.after(waited).index(1).store(hcq2.timeline_value(("CPU",)) + UOp.const(1, dtypes.uint64))
      body = tl.after(signaled).index(0).store(hcq2.timeline_value(("CPU",)) + UOp.const(1, dtypes.uint64))
      timeline = Buffer("CPU", 2, dtypes.uint64, preallocate=True)
      timeline.host.view(fmt='Q')[0] = timeline.host.view(fmt='Q')[1] = 7
      with patch.object(Device["CPU"], "timeline", timeline):
        linear = self._link(body)
        run_linear(linear, jit=True)
      result = next(u.buffer for u in linear.src[0].without_after.src[1:] if u.op is Ops.BUFFER and u.buffer.size == 2 and u.buffer is not timeline)
      self.assertEqual(result.host.view(fmt='Q')[:], [7, 8])
      self.assertEqual(timeline.host.view(fmt='Q')[:], [8, 8])

  def test_usb_idle_links_device_timeline(self):
    with Context(HCQ_RUNTIME_DEV="CPU", DEBUG=0, PROFILE=0):
      timeline = Buffer("CPU", 2, dtypes.uint64, preallocate=True)
      timeline.host.view(fmt='Q')[0] = timeline.host.view(fmt='Q')[1] = 7
      # Replace the USB read with a local read, retaining the real placeholder binding and compiled wait.
      with patch.object(Device["CPU"], "timeline", timeline), \
           patch("tinygrad.runtime.support.nv.usb.usb_load", side_effect=lambda b, idx, dt: b.index(idx).load()):
        marker = cpu_buf(dtype=dtypes.uint32, volatile=True, tag="idle_marker")
        linear = self._link(marker.after(usb_idle(("CPU",))).index(0).store(123))
      bufs = [u.buffer for u in linear.src[0].without_after.src[1:] if u.op is Ops.BUFFER]
      self.assertIn(timeline, bufs, "USB idle must wait on the device timeline, not an unrelated allocation")
      run_linear(linear, jit=True)
      self.assertEqual(next(b for b in bufs if b.dtype is dtypes.uint32).host.view(fmt='I')[0], 123)

  def test_hcq_call_accepts_buffer_view_args(self):
    with Context(HCQ_RUNTIME_DEV="CPU"):
      source = Buffer("CPU", 4, dtypes.uint32, preallocate=True)
      source.host.view(fmt='I')[1] = 123
      view = UOp.from_buffer(source)[1:3]
      out = cpu_buf(dtype=dtypes.uint32, volatile=True, tag="view_result")
      body = UOp.sink(out.index(0).store(view.index(0).load()), arg=KernelInfo("view_arg"))
      call = hcq2.lower_call(body.call(view, aux=hcq2.HCQInfo(("CPU",))))
      linear = hcq2.hcq_link(lower_and_compile(UOp(Ops.LINEAR, src=(call,))), allow_cache=False)
      run_linear(linear, jit=True)
      result = next(u.buffer for u in linear.src[0].without_after.src[1:] if u.buffer.size == 1)
      self.assertEqual(result.host.view(fmt='I')[0], 123)

  def test_getaddr_with_runtime_dependency_stays_in_submitter(self):
    for runtime in ("CPU", "PYTHON"):
      with self.subTest(runtime=runtime), Context(HCQ_RUNTIME_DEV=runtime, DEBUG=0, PROFILE=0):
        source = UOp.placeholder((2,), dtypes.uint32, device="CPU", volatile=True, tag="runtime_source")
        marker = UOp.placeholder((3,), dtypes.uint32, device="CPU", volatile=True, tag="runtime_marker")
        program = UOp.placeholder((5,), dtypes.uint8, device="CPU", tag="program")
        dependent = program.after(marker.index(0).store(source.index(0).load()))
        word = hcq2.patch(UOp.placeholder((1,), dtypes.uint64, device="CPU", volatile=True, tag="link_word"),
                          [(0, dependent.getaddr(("CPU",)))])
        linear = self._link(word.index(0).load())
        bufs = [u.buffer for u in linear.src[0].without_after.src[1:] if u.op is Ops.BUFFER]
        source_buf = next(b for b in bufs if b.dtype is dtypes.uint32 and b.size == 2)
        marker_buf = next(b for b in bufs if b.dtype is dtypes.uint32 and b.size == 3)
        program_buf = next(u.buffer for u in linear.src[0].src[1:]
                           if u.op is Ops.BUFFER and u.dtype is dtypes.uint8 and u.buffer.size == 5)
        source_buf.host.view(fmt='I')[0] = 123
        run_linear(linear, jit=True)
      self.assertEqual(marker_buf.host.view(fmt='I')[0], 123)
      expected = program_buf.get_buf("CPU")
      self.assertEqual([b.host.view(fmt='Q')[0] for b in bufs if b.dtype is dtypes.uint64 and b.size == 1], [expected, expected])

  def test_ffi_cstruct(self):
    struct_t = init_c_struct_t(16, (("u8", ctypes.c_uint8, 0), ("u16", ctypes.c_uint16, 2),
                                  ("u32", ctypes.c_uint32, 4), ("u64", ctypes.c_uint64, 8)))
    cpu_buf() # reserve slot zero for device-owned placeholders
    with Context(HCQ_RUNTIME_DEV="CPU"):
      s = hcq2.cstruct(struct_t, u8=0x12, u16=UOp.const(0x3456, dtypes.uint16), u32=0x789ABCDE, u64=0xFEDCBA9876543210)
      bufs = self._run(s.index(0).load())
    got = struct_t.from_buffer_copy(bytes(next(b for b in bufs if b.nbytes == ctypes.sizeof(struct_t)).host.view(fmt='B')))
    self.assertEqual((got.u8, got.u16, got.u32, got.u64), (0x12, 0x3456, 0x789ABCDE, 0xFEDCBA9876543210))

  def test_nested_cstruct_patches(self):
    with Context(HCQ_RUNTIME_DEV="CPU"):
      inner = hcq2.cstruct(init_c_struct_t(4, (("value", ctypes.c_uint32, 0),)), value=42)
      outer = hcq2.cstruct(init_c_struct_t(8, (("ptr", ctypes.c_uint64, 0),)), ptr=inner.getaddr("CPU"))
      out = cpu_buf(dtype=dtypes.uint32, tag="result")
      copied = hcq2.ccall(libc.memcpy, out.index(0), outer.bitcast(dtypes.uint64).index(0).load(), 4)
      bufs = self._run(out.after(copied).index(0).load())
    self.assertEqual(next(b for b in bufs if b.dtype is dtypes.uint32).host.view(fmt='I')[0], 42)

  def test_ffi_ccheck_replay(self):
    calls = []
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(lambda value: calls.append(value) or value)
    fn.__module__, fn.__name__ = "tinygrad.runtime.autogen.libc", "test_checked_call"
    ptr = Buffer("CPU", 1, dtypes.uint64, preallocate=True)
    ptr.host.view(fmt='Q')[0] = ctypes.cast(fn, ctypes.c_void_p).value
    value = UOp.variable("ffi_value", -128, 128, dtypes.int32, param=True)
    expected = UOp.variable("ffi_expected", 0, 128, dtypes.int32, param=True)
    later = UOp.variable("ffi_later", 0, 128, dtypes.int32, param=True)
    with Context(HCQ_RUNTIME_DEV="CPU", PROFILE=0, DEBUG=0), patch.object(Device["CPU"].pm_bufferize, "rewrite",
      side_effect=lambda b, ctx: ptr if b.tag == ("cfunc", "libc", "test_checked_call") else None):
      first = hcq2.ccheck(hcq2.ccall(fn, value), expected)
      second = hcq2.ccheck(hcq2.ccall(fn, expected.after(first) + 1), later)
      out = UOp.placeholder((1,), dtypes.uint32, device="CPU", volatile=True, tag="ffi_marker")
      linear = self._link(out.after(second).index(0).store(123))
      call = linear.src[0].without_after
      self.assertIn(ptr, [u.buffer for u in call.src[1:] if u.op is Ops.BUFFER])
      info = replace(call.arg.aux, kernels=((("CPU",), "ffi", Estimates(), (0, 1), b""),), slots=(("CPU", call.arg.aux.error),))
      linear = linear.substitute({call: call.replace(arg=replace(call.arg, aux=info))})
      error = call.src[1 + info.error].buffer.host.view(fmt='i')
      marker = next(u.buffer for u in call.src[1:] if u.op is Ops.BUFFER and u.dtype is dtypes.uint32).host.view(fmt='I')
      with patch.object(Device["CPU"], "synchronize", side_effect=AssertionError("failed calls must not wait for a GPU")) as sync:
        for got, want, last, wait in ((12, 12, 13, False), (0, 0, 1, False), (-7, 0, 1, False), (3, 4, 5, False), (-7, 0, 1, True),
                                     (3, 4, 5, True), (12, 12, 14, False), (12, 12, 14, True), (12, 12, 13, False)):
          with self.subTest(got=got, want=want, last=last, wait=wait):
            calls.clear()
            error[0], error[1], marker[0] = -99, 0, 0 # cached replay must clear the previous error, even if this call succeeds
            pair, vals = (got, want) if got != want else (want + 1, last), {"ffi_value": got, "ffi_expected": want, "ffi_later": last}
            if pair[0] == pair[1]: run_linear(linear, vals, jit=True, wait=wait)
            else:
              with self.assertRaisesRegex(RuntimeError, f"native call returned {pair[0]}, expected {pair[1]}"):
                run_linear(linear, vals, jit=True, wait=wait)
            self.assertEqual(calls, [got, want + 1] if got == want else [got])
            self.assertEqual(marker[0], 123 if pair[0] == pair[1] else 0)
            self.assertEqual(error[:], [0, 0] if pair[0] == pair[1] else list(pair))
            sync.assert_not_called()

  def test_ffi_ccheck_requires_cstyle(self):
    with Context(HCQ_RUNTIME_DEV="CPU"), patch.object(type(Device["CPU"]), "renderer", CPULLVMRenderer.__new__(CPULLVMRenderer)):
      with self.assertRaisesRegex(AssertionError, "C-style CPU renderer"): hcq2.ccheck(UOp.const(0, dtypes.int32))
    with Context(HCQ_RUNTIME_DEV="PYTHON"):
      with self.assertRaisesRegex(AssertionError, "C-style CPU renderer"): hcq2.ccheck(UOp.const(0, dtypes.int32))

  def test_ffi_usb_transfers(self):
    calls, result = [], {}
    def control(handle, reqtype, request, value, index, data, length, timeout):
      calls.append(("control", value, index, ctypes.string_at(data, length)))
      return result['control'] if len(calls) == 1 else length
    def bulk(handle, endpoint, data, length, actual, timeout):
      calls.append(("bulk", endpoint, length, ctypes.string_at(data, length)))
      actual[0] = result['actual'] if len(calls) == 2 else length
      if endpoint == 0x81: ctypes.memmove(data, b'ABCDEFGH', min(length, actual[0]))
      return result['bulk'] if len(calls) == 2 else 0
    funcs = {fn.__name__: ctypes.CFUNCTYPE(fn.restype, *fn.argtypes)(stub)
             for fn, stub in ((libusb.libusb_control_transfer, control), (libusb.libusb_bulk_transfer, bulk))}
    ptrs = {name: Buffer("CPU", 1, dtypes.uint64, preallocate=True) for name in funcs}
    for name, fn in funcs.items(): ptrs[name].host.view(fmt='Q')[0] = ctypes.cast(fn, ctypes.c_void_p).value
    addr = UOp.variable('usb_addr', 0, (1 << 36)-1, dtypes.uint64, param=True)
    with Context(HCQ_RUNTIME_DEV="CPU", DEBUG=0, PROFILE=0), patch.object(Device["CPU"].pm_bufferize, "rewrite",
      side_effect=lambda b, ctx: ptrs[b.tag[2]] if isinstance(b.tag, tuple) and b.tag[0] == "cfunc" else None):
      for write in (False, True):
        payload = UOp.placeholder((8,), dtypes.uint8, device="CPU", volatile=True, tag="usb_payload")
        first = usb_stream(('CPU',), (), addr, payload.index(0), 8, write)
        second = usb_stream(('CPU',), (first,), addr + 16, payload.index(0), 4, True)
        marker = UOp.placeholder((1,), dtypes.uint32, device="CPU", volatile=True, tag="usb_marker")
        linear = self._link(marker.after(second).index(0).store(123))
        call = linear.src[0].without_after
        self.assertGreaterEqual(call.arg.aux.error, 0, "native USB transfers must report errors")
        bufs = [u.buffer for u in call.src[1:] if u.op is Ops.BUFFER]
        for ptr in ptrs.values(): self.assertIn(ptr, bufs)
        data = next(b for b in bufs if b.dtype is dtypes.uint8 and b.size == 8).host
        done = next(b for b in bufs if b.dtype is dtypes.uint32).host.view(fmt='I')
        for control_rc, bulk_rc, actual in ((12, 0, 8), (-7, 0, 8), (11, 0, 8), (12, -4, 8), (12, 0, 0), (12, 0, 7), (12, 0, 9), (12, 0, 8)):
          with self.subTest(write=write, control=control_rc, bulk=bulk_rc, actual=actual):
            calls.clear()
            result.update(control=control_rc, bulk=bulk_rc, actual=actual)
            data[:], done[0] = b'abcdefgh', 0
            error = (control_rc, 12) if control_rc != 12 else (bulk_rc, 0) if bulk_rc else (actual, 8)
            if error[0] == error[1]: run_linear(linear, {'usb_addr': 0x800000000}, jit=True, wait=False)
            else:
              with self.assertRaisesRegex(RuntimeError, f"native call returned {error[0]}, expected {error[1]}"):
                run_linear(linear, {'usb_addr': 0x800000000}, jit=True, wait=False)
            self.assertEqual([c[0] for c in calls], ['control'] if control_rc != 12 else
                             ['control', 'bulk'] if error[0] != error[1] else ['control', 'bulk', 'control', 'bulk'])
            self.assertEqual(calls[0], ('control', 0xf20 | (0x40 if write else 0), 1 if write else 2, bytes.fromhex('000000000800000002000000')))
            self.assertEqual(done[0], 123 if error[0] == error[1] else 0)
            if error[0] == error[1]:
              self.assertEqual(calls[2], ('control', 0xf60, 1, bytes.fromhex('100000000800000001000000')))
              self.assertEqual(calls[-1], ('bulk', 0x02, 4, b'abcd' if write else b'ABCD'))
        for address in (0xfffffff8, 0x100000000):
          calls.clear()
          run_linear(linear, {'usb_addr': address}, jit=True, wait=False)
          self.assertEqual(calls[0][1], 0xf00 | (0x20 if address >= (1 << 32) else 0) | (0x40 if write else 0))
          self.assertEqual([c[0] for c in calls], ['control', 'bulk', 'control', 'bulk'])
        calls.clear()
        with self.assertRaisesRegex(RuntimeError, "native call returned 1, expected 0"):
          run_linear(linear, {'usb_addr': 0xfffffffc}, jit=True, wait=False)
        self.assertEqual(calls, [], "a stream crossing 4 GiB must fail before any USB request")

  def test_ffi_usb_subdword_writes(self):
    calls = []
    def control(handle, reqtype, request, value, index, data, length, timeout):
      calls.append(("control", value, index, ctypes.string_at(data, length)))
      return length
    def bulk(handle, endpoint, data, length, actual, timeout):
      calls.append(("bulk", endpoint, length, ctypes.string_at(data, length)))
      actual[0] = length
      return 0
    funcs = {fn.__name__: ctypes.CFUNCTYPE(fn.restype, *fn.argtypes)(stub)
             for fn, stub in ((libusb.libusb_control_transfer, control), (libusb.libusb_bulk_transfer, bulk))}
    ptrs = {name: Buffer("CPU", 1, dtypes.uint64, preallocate=True) for name in funcs}
    for name, fn in funcs.items(): ptrs[name].host.view(fmt='Q')[0] = ctypes.cast(fn, ctypes.c_void_p).value
    with Context(HCQ_RUNTIME_DEV="CPU", DEBUG=0, PROFILE=0), patch.object(Device["CPU"].pm_bufferize, "rewrite",
      side_effect=lambda b, ctx: ptrs[b.tag[2]] if isinstance(b.tag, tuple) and b.tag[0] == "cfunc" else None):
      for dt, index, value in ((dtypes.uint8, 3, 0xA5), (dtypes.uint16, 1, 0xBEEF)):
        target = UOp.placeholder((8,), dt, device="CPU", tag="target")
        linear = self._link(usb_write_one(target, 0, (), ("CPU",), UOp.const(index), UOp.const(value, dt)))
        target_buf = next(u.buffer for u in linear.src[0].src[1:] if u.op is Ops.BUFFER and u.dtype is dt)
        calls.clear()
        run_linear(linear, jit=True)
        address = target_buf.get_buf("CPU") + index * dt.itemsize
        lane = address & 3
        byte_en = ((1 << dt.itemsize) - 1) << lane
        header = struct.pack('<QI', address & ~3, value << (8 * lane))
        self.assertEqual(calls, [("control", 0x60 | (byte_en << 8), 0, header)])
      addressed = UOp.placeholder((5,), dtypes.uint8, device="CPU", tag="addressed")
      target = UOp.placeholder((8,), dtypes.uint16, device="CPU", tag="target")
      value = (addressed.getaddr(("CPU",)) >> 32).cast(dtypes.uint16)
      linear = self._link(usb_write_one(target, 0, (), ("CPU",), UOp.const(1), value))
      refs = [u.buffer for u in linear.src[0].src[1:] if u.op is Ops.BUFFER]
      target_buf, addressed_buf = next(b for b in refs if b.dtype is dtypes.uint16), next(b for b in refs if b.dtype is dtypes.uint8)
      calls.clear()
      run_linear(linear, jit=True)
      address, value = target_buf.get_buf("CPU") + 2, addressed_buf.get_buf("CPU") >> 32
      header = struct.pack('<QI', address & ~3, value << (8 * (address & 3)))
      self.assertEqual(calls, [("control", 0xC60, 0, header)])


if __name__ == "__main__":
  unittest.main()
