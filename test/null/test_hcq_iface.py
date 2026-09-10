import unittest, array, time, struct, ctypes
from types import SimpleNamespace
from unittest.mock import Mock, patch
from tinygrad.dtype import dtypes
from tinygrad.helpers import mv_address
from tinygrad.uop.ops import UOp, Ops, KernelInfo, graph_rewrite
from tinygrad.device import Device
from tinygrad.runtime.support import hcq2
from tinygrad.runtime.support.hcq import MMIOInterface
from tinygrad.runtime.support.memory import BumpAllocator
from tinygrad.runtime.support.system import System, USBPCIDevice
from tinygrad.runtime.support.usb import USB3, USBMMIOInterface, CustomASM24Controller
from tinygrad.runtime.support.nv.usb import usb_stream, usb_scsi, usb_stage_copy
from tinygrad.runtime.support.usb import alloc_cbuffer
from tinygrad.runtime.support.nv.usb import pm_usb_stage, pm_usb_hostio
from test.mockgpu.usb import MockUSB

class TestHCQIface(unittest.TestCase):
  def setUp(self):
    self.size = 4 << 10
    self.buffer = bytearray(self.size)
    self.mv = memoryview(self.buffer).cast('I')
    self.mmio = MMIOInterface(mv_address(self.mv), self.size, fmt='I')

  def test_getitem_setitem(self):
    self.mmio[1] = 0xdeadbeef
    self.assertEqual(self.mmio[1], 0xdeadbeef)
    values = array.array('I', [10, 20, 30, 40])
    self.mmio[2:6] = values
    read_slice = self.mmio[2:6]
    # self.assertIsInstance(read_slice, array.array)
    self.assertEqual(read_slice, values.tolist())
    self.assertEqual(self.mv[2:6].tolist(), values.tolist())

  def test_view(self):
    full = self.mmio.view()
    self.assertEqual(len(full), len(self.mmio))
    self.mmio[0] = 0x12345678
    self.assertEqual(full[0], 0x12345678)

    # offset-only view
    self.mmio[1] = 0xdeadbeef
    off = self.mmio.view(offset=4)
    self.assertEqual(off[0], 0xdeadbeef)

    # offset + size view: write into sub-view and confirm underlying buffer
    values = array.array('I', [11, 22, 33])
    sub = self.mmio.view(offset=8, size=12)
    sub[:] = values
    self.assertEqual(sub[:], values.tolist())
    self.assertEqual(self.mv[2:5].tolist(), values.tolist())

  def test_speed(self):
    start = time.perf_counter()
    for i in range(10000):
      self.mmio[3:100] = array.array('I', [i] * 97)
      _ = self.mmio[3:100]
    end = time.perf_counter()

    mvstart = time.perf_counter()
    for i in range(10000):
      self.mv[3:100] = array.array('I', [i] * 97)
      _ = self.mv[3:100].tolist()
    mvend = time.perf_counter()
    print(f"speed: hcq {end - start:.6f}s vs plain mv {mvend - mvstart:.6f}s")

class TestUSBMMIOInterface(unittest.TestCase):
  def setUp(self):
    self.size = 256
    self.buffer = bytearray(self.size)
    self.usb = MockUSB(self.buffer)
    self.mmio = USBMMIOInterface(self.usb, 0, self.size, fmt='B', pcimem=False)

  def test_access_does_not_wait_for_device(self):
    with patch('tinygrad.runtime.support.usb.Device', create=True) as device:
      device.__getitem__.return_value.synchronize.side_effect = AssertionError("MMIO cannot wait for the device whose timeline it reads")
      self.mmio[0] = 42
      self.assertEqual(self.mmio[0], 42)

  def test_getitem_setitem_byte(self):
    self.mmio[1] = 0xAB
    self.assertEqual(self.mmio[1], 0xAB)
    self.assertEqual(self.usb.mem[1], 0xAB)

  def test_slice_getitem_setitem(self):
    values = [1, 2, 3, 4]
    self.mmio[10:14] = values
    raw = self.mmio[10:14]
    self.assertIsInstance(raw, bytes)
    self.assertEqual(list(raw), values)
    self.assertEqual(list(self.usb.mem[10:14]), values)

  def test_view(self):
    self.mmio[0] = 5
    view = self.mmio.view(offset=1, size=3)
    self.assertEqual(view[0], self.usb.mem[1])
    view[:] = [7, 8, 9]
    self.assertEqual(list(self.usb.mem[1:4]), [7, 8, 9])
    full_view = self.mmio.view()
    self.assertEqual(len(full_view), len(self.mmio))
    self.mmio[2] = 0xFE
    self.assertEqual(full_view[2], 0xFE)

  def test_pcimem_dword(self):
    usb2 = MockUSB(bytearray(self.size))
    mmio_pci = USBMMIOInterface(usb2, 0, self.size, fmt='I', pcimem=True)
    mmio_pci[3] = 0x11223344
    self.assertEqual(mmio_pci[3], 0x11223344)
    self.assertEqual(usb2.mem[12:16], b'\x44\x33\x22\x11')

  def test_typed_slice(self):
    for pcimem in (False, True):
      for fmt in ('I', 'Q'):
        with self.subTest(pcimem=pcimem, fmt=fmt):
          mmio = USBMMIOInterface(self.usb, 0, self.size, fmt=fmt, pcimem=pcimem)
          values = array.array(fmt, [0x12345678, 0xabcdef01])
          mmio[1:3] = values
          self.assertEqual(mmio[1:3], values.tolist())
          self.assertEqual(array.array(fmt, mmio[1:3]), values)

  def test_pcimem_slice(self):
    usb3 = MockUSB(bytearray(self.size))
    mmio_pci = USBMMIOInterface(usb3, 0, self.size, fmt='B', pcimem=True)
    values = [2, 3, 4, 5]
    mmio_pci[4:8] = values
    raw = mmio_pci[4:8]
    self.assertIsInstance(raw, bytes)
    self.assertEqual(list(raw), values)
    self.assertEqual(list(usb3.mem[4:8]), values)

class TestUSBPCITransfers(unittest.TestCase):
  def test_copyout_waits_before_direct_pcie_read(self):
    view = SimpleNamespace(addr=0x4f000)
    window = Mock(device="NV", dtype=dtypes.uint8, size=0x40000)
    window.host = view
    reset, submitted, waited = (UOp.custom_function(x) for x in ("test_usb_reset", "test_usb_submit", "test_usb_wait"))
    seen = {}

    def write_one(base, offset, deps, devs, idx, value):
      seen["reset"] = (base, offset, deps, devs, idx, value)
      return reset
    def submit(*cmds, devs, queue):
      seen["submit"] = (cmds, devs, queue)
      return submitted
    def wait_value(base, value, deps):
      seen["wait"] = (base, value, deps)
      return waited
    def stream(devs, deps, addr, data, nbytes, write):
      seen["stream"] = (devs, deps, addr, data, nbytes, write)
      return UOp.custom_function("test_usb_stream", *deps)

    with patch('tinygrad.runtime.support.nv.usb.Device') as device, \
         patch('tinygrad.runtime.support.nv.usb.usb_scsi', side_effect=AssertionError("copyout must not arm F2")), \
         patch('tinygrad.runtime.support.nv.usb.usb_write_one', side_effect=write_one), \
         patch('tinygrad.runtime.support.nv.usb.make_submit', side_effect=submit), \
         patch('tinygrad.runtime.support.nv.usb.usb_wait_value', side_effect=wait_value), \
         patch('tinygrad.runtime.support.nv.usb.usb_stream', side_effect=stream):
      device.__getitem__.return_value.pm_stage_copy = pm_usb_stage
      device.__getitem__.return_value.usb_readback = window
      usb_stage_copy(UOp.placeholder((0x1000,), dtypes.uint8, device="CPU"),
                     UOp.placeholder((0x1000,), dtypes.uint8, device="NV"))

    done, offset, deps, reset_devs, idx, value = seen["reset"]
    self.assertEqual((done.tag, done.dtype, offset, deps, reset_devs, idx.val, value.val),
                     ("usb_readback_done", dtypes.uint64, 0, (), ("NV",), 0, 0))
    cmds, devs, queue = seen["submit"]
    self.assertEqual((devs, queue), (("NV",), "COPY:0"))
    self.assertEqual(len(cmds), 3)
    self.assertEqual((cmds[0].op, cmds[0].arg), (Ops.INS, ("wait", dtypes.void)))
    self.assertIs(cmds[0].src[0], hcq2.timeline(("NV",)))
    self.assertIs(cmds[0].src[1], hcq2.timeline_value(("NV",)))
    self.assertIs(cmds[1].op, Ops.CALL)
    self.assertEqual((cmds[2].arg, cmds[2].src[0].tag, cmds[2].src[1].val),
                     (("store", dtypes.void), "usb_readback_done", 1))
    waited_done, waited_value, waited_deps = seen["wait"]
    self.assertIs(waited_done, done)
    self.assertEqual(waited_value, 1)
    self.assertEqual(waited_deps[0].src, (submitted, reset))
    stream_devs, stream_deps, _, _, nbytes, write = seen["stream"]
    self.assertEqual((stream_devs, stream_deps, nbytes, write), (("NV",), (waited,), 0x1000, False))

  def test_staging_uses_lowerable_hcq_calls(self):
    view = SimpleNamespace(addr=0x4f000)
    window = Mock(device="NV", dtype=dtypes.uint8, size=0x40000)
    window.host = view
    for src_device, dst_device, name in (("CPU", "NV", "hcq_copyin"), ("PYTHON", "NV", "hcq_copyin"),
                                         ("NV", "CPU", "hcq_copyout"), ("NV", "PYTHON", "hcq_copyout")):
      with self.subTest(name=name), patch('tinygrad.runtime.support.nv.usb.Device') as device:
        device.__getitem__.return_value.pm_stage_copy = pm_usb_stage
        device.__getitem__.return_value.usb_sram = window
        device.__getitem__.return_value.usb_readback = window
        staged = usb_stage_copy(UOp.placeholder((0x1000,), dtypes.uint8, device=dst_device),
                                UOp.placeholder((0x1000,), dtypes.uint8, device=src_device))
      calls = [u for u in staged.src if u.op is Ops.CALL and u.arg.name == name]
      self.assertEqual(len(calls), 1)
      self.assertIs(calls[0].src[0].op, Ops.SINK)
      self.assertEqual(calls[0].src[0].arg, KernelInfo(name))
      self.assertFalse(any(u.op is Ops.BUFFER and u.device == "CPU" for u in calls[0].src[0].toposort()))
      if src_device == "PYTHON":
        copies = [u for u in staged.toposort() if u.op is Ops.CALL and u.src[0].op is Ops.COPY]
        self.assertIn(("CPU", "PYTHON"), [(u.src[1].device, u.src[2].device) for u in copies])

  def test_hcq_compile_stages_native_usb_copies_before_batching(self):
    # Prime the matcher before mocking devices, as an earlier CPU test would.
    warm_dst, warm_src = (UOp.placeholder((16,), dtypes.uint8, device=d) for d in ("CPU", "PYTHON"))
    hcq2.hcq_compile(UOp(Ops.LINEAR, src=(warm_src.copy_to_device("CPU").call(warm_dst, warm_src),)), None, False)
    window = Mock(device="NV", dtype=dtypes.uint8, size=0x40000, host=SimpleNamespace(addr=0x4f000))
    devices = {"NV": SimpleNamespace(pm_stage_copy=pm_usb_stage, usb_sram=window, usb_readback=window,
                                    has_copy_queue=True), "CPU": Device["CPU"]}
    for src_device, dst_device, name in (("CPU", "NV", "hcq_copyin"), ("NV", "CPU", "hcq_copyout")):
      # Compiled matchers retain their globals, so patch lookup on the original Device object.
      with self.subTest(name=name), patch.object(type(Device), "__getitem__", lambda _, device: devices[device]), \
           patch.object(hcq2, "sched_batches", return_value=UOp(Ops.LINEAR)) as batch, patch.dict(hcq2.hcq_compile_cache, clear=True):
        for _ in range(2):
          # New inputs of the same shape must reuse the staged template.
          dst, src = (UOp.from_buffer(Mock(device=d, dtype=dtypes.uint8, size=0x1000)) for d in (dst_device, src_device))
          hcq2.hcq_compile(UOp(Ops.LINEAR, src=(src.copy_to_device(dst.device).call(dst, src),)), [], False, cache=True)
        self.assertEqual(batch.call_count, 1)
      calls = batch.call_args.args[0].src
      self.assertEqual(sum(c.op is Ops.CALL and c.arg.name == name for c in calls), 1)
      for call in calls:
        if call.op is Ops.CALL and call.src[0].op is Ops.COPY:
          self.assertEqual(call.src[1].device, call.src[2].device)

  def test_compiled_scsi_uses_staging_slot(self):
    for read in (False, True):
      with self.subTest(read=read), patch('tinygrad.runtime.support.nv.usb._libusb', return_value=UOp(Ops.NOOP)) as call:
        usb_scsi(('NV',), read, 0x1000, slot_start=16)
      args = call.call_args.args
      self.assertEqual((args[2:7], args[8:]),
                       (('libusb_control_transfer', 0x40, 0xF2, 8 | (0x8000 if read else 0), 0x110), (0, 1000)))

  def test_hcq_hostio_mediates_device_params(self):
    buf = UOp.placeholder((4,), dtypes.uint64, device=("NV",), tag="ring_compute_0")
    cases = (buf, buf.after(UOp(Ops.NOOP)), buf.after(UOp(Ops.NOOP)).after(UOp(Ops.NOOP)),
             buf.after(UOp(Ops.NOOP))[1:2].bitcast(dtypes.uint32))
    for depth, based in enumerate(cases):
      load = based.index(0).load()
      store = based.index(0).store(UOp.const(0x1234, based.dtype))
      for op in (load, store):
        with self.subTest(depth=depth, op=op.op):
          rewritten = graph_rewrite(op, pm_usb_hostio, walk=True)
          self.assertIsNot(rewritten, op)
          self.assertTrue(any(u.op is Ops.CUSTOM_FUNCTION and str(u.arg).startswith("libusb_") for u in rewritten.toposort()))

    cpu = UOp.placeholder((1,), dtypes.uint64, device="CPU", tag="host")
    host_load = cpu.index(0).load()
    self.assertIs(graph_rewrite(host_load, pm_usb_hostio, walk=True), host_load)

  def test_hcq_hostio_scalarizes_scatter(self):
    buf = UOp.placeholder((8,), dtypes.uint32, device=("NV",), tag="cmdbuf_copy_0")
    indices = UOp.stack(*[UOp.const(i, dtypes.int) for i in (1, 3, 5, 7)])
    values = UOp.stack(*[UOp.const(i, dtypes.uint32) for i in (10, 30, 50, 70)])
    rewritten = graph_rewrite(buf.index(indices).store(values), pm_usb_hostio, walk=True)
    bulk_calls = [u for u in rewritten.toposort() if u.op is Ops.CALL and u.src[0].op is Ops.CUSTOM_FUNCTION
                  and u.src[0].arg == "libusb_bulk_transfer"]
    self.assertEqual(len(bulk_calls), 4)

  def test_xdata_bounds(self):
    controller = object.__new__(CustomASM24Controller)
    controller.usb = Mock()
    controller.usb.control_read.return_value = b'\xa5'
    cases = ((0xffff, 1, True), (0, 0, True), (0x10000, 0, True), (-1, 1, False), (-1, 0, False),
             (0x10000, 1, False), (0x10001, 0, False), (0xffff, 2, False), (0xff00, 0x101, False))
    for write in (False, True):
      for address, size, valid in cases + (() if write else ((0, -1, False),)):
        with self.subTest(write=write, address=address, size=size):
          controller.usb.reset_mock()
          op, arg = (controller.write, b'\xa5' * size) if write else (controller.read, size)
          if not valid:
            with self.assertRaisesRegex(AssertionError, 'XDATA range'): op(address, arg)
            self.assertEqual(controller.usb.mock_calls, [])
          else:
            result = op(address, arg)
            if not write: self.assertEqual(result, b'\xa5' * size)
            if not size: self.assertEqual(controller.usb.mock_calls, [])
            elif write: controller.usb.control_write.assert_called_once_with(0xE5, value=address, index=0xa5)
            else: controller.usb.control_read.assert_called_once_with(0xE4, 1, value=address)

  def test_bulk_read_requires_full_transfer(self):
    usb = object.__new__(USB3)
    usb.handle = None
    usb._bulk_buf, usb._bulk_mv = alloc_cbuffer(4)
    usb._bulk_mv[:] = b'abcd'
    for completed in (0, 3, 4):
      with self.subTest(completed=completed), patch('tinygrad.runtime.support.usb.libusb.libusb_bulk_transfer', return_value=0):
        usb._transferred = ctypes.c_int(completed)
        if completed == 4: self.assertEqual(bytes(usb.bulk_read(4)), b'abcd')
        else:
          with self.assertRaisesRegex(AssertionError, 'bulk IN short read'): usb.bulk_read(4)

  def test_bulk_write_chunks_payload_to_reusable_buffer(self):
    usb = object.__new__(USB3)
    usb.handle = None
    usb._bulk_buf, usb._bulk_mv = alloc_cbuffer(4)
    usb._transferred = ctypes.c_int(0)
    chunks = []
    def transfer(_handle, endpoint, data, length, transferred, timeout):
      self.assertEqual((endpoint, timeout), (0x02, 1234))
      chunks.append(bytes(data[:length]))
      transferred.value = length
      return 0

    with patch('tinygrad.runtime.support.usb.libusb.libusb_bulk_transfer', side_effect=transfer):
      usb.bulk_write(b'abcdefghij', timeout=1234)

    self.assertEqual(chunks, [b'abcd', b'efgh', b'ij'])

  def test_bulk_write_bounds_each_transfer(self):
    usb = object.__new__(USB3)
    usb.handle = None
    size = (256 << 10) + 4
    usb._bulk_buf, usb._bulk_mv = alloc_cbuffer(size)
    usb._transferred = ctypes.c_int(0)
    lengths = []
    def transfer(_handle, _endpoint, _data, length, transferred, _timeout):
      lengths.append(length)
      transferred.value = length
      return 0

    with patch('tinygrad.runtime.support.usb.libusb.libusb_bulk_transfer', side_effect=transfer):
      usb.bulk_write(bytes(size))

    self.assertEqual(lengths, [256 << 10, 4])

  def test_address_format(self):
    controller = object.__new__(CustomASM24Controller)
    controller.usb = Mock()
    for address, fmt in ((0, 0), (0x10000000, 0), (0xfffffffc, 0), (0x100000000, 0x20), (0x800000000, 0x20)):
      for write in (False, True):
        with self.subTest(address=address, write=write):
          controller.usb.reset_mock()
          if write: controller.pcie_mem_write(address, b'abcd')
          else: controller.pcie_mem_read(address, 4)
          controller.usb.control_write.assert_called_once_with(0xF0, 0xf00 | fmt | (0x40 if write else 0), 1 if write else 2,
            struct.pack('<III', address & 0xffffffff, address >> 32, 1), 5000)

  def test_compiled_address_format(self):
    addr = UOp.variable('usb_addr', 0, (1 << 36)-1, dtypes.uint64)
    for write in (False, True):
      with patch('tinygrad.runtime.support.nv.usb._libusb', return_value=UOp(Ops.NOOP)) as call:
        usb_stream(('NV',), (), addr, UOp.const(0, dtypes.uint64), 4, write)
      args = call.call_args_list[0].args
      self.assertEqual((args[2:5], args[6], args[8:]), (('libusb_control_transfer', 0x40, 0xF0), 1 if write else 2, (12, 5000)))
      for address, fmt in ((0x10000000, 0), (0xfffffffc, 0), (0x100000000, 0x20), (0x800000000, 0x20)):
        with self.subTest(address=address, write=write):
          value = args[5].substitute({addr:UOp.const(address, dtypes.uint64)}) if isinstance(args[5], UOp) else args[5]
          self.assertEqual(int(value), 0xf00 | fmt | (0x40 if write else 0))
          if isinstance(value, UOp): self.assertEqual(value.dtype, dtypes.int)

  def test_cross_4gib_rejected_before_io(self):
    controller = object.__new__(CustomASM24Controller)
    controller.usb = Mock()
    for write in (False, True):
      with self.subTest(write=write):
        with self.assertRaisesRegex(AssertionError, 'crosses 4 GiB'):
          if write: controller.pcie_mem_write(0xfffffffc, b'abcdefgh')
          else: controller.pcie_mem_read(0xfffffffc, 8)
        self.assertEqual(controller.usb.mock_calls, [])

class TestUSBPCIBars(unittest.TestCase):
  def test_sysmem_returns_every_page(self):
    dev = object.__new__(USBPCIDevice)
    dev.usb, dev.sram = Mock(), BumpAllocator(0x80000, wrap=False)
    dev.sram.alloc(0x40000)

    view, paddrs = dev.alloc_sysmem(0x3000)

    self.assertEqual((view.addr, view.nbytes), (0x4f000, 0x3000))
    self.assertEqual(paddrs, [0x240000, 0x241000, 0x242000])

  def test_resize_all_bars(self):
    for count in (0, 1, 3):
      with self.subTest(count=count):
        config = {0x100:0x20000001, 0x200:0x10015 if count else 0, **{off:1 for off in range(0x10, 0x28, 4)}}
        entries = [(0x100, 0x400 | (count << 5)), (0xffc00, 0x801), (0x200, 0x503)][:count]
        for i, (cap, ctrl) in enumerate(entries): config.update({0x204+8*i:cap, 0x208+8*i:ctrl | 0xa5000000})
        def cfg(offset, bus, dev, fn, size, value=None):
          self.assertEqual((bus, dev, fn), (0, 0, 0))
          if value is None: return config.get(offset, 0)
          config[offset] = value
        System.pci_setup_usb_bars(Mock(pcie_cfg_req=cfg), gpu_bus=0, mem_base=0x10000000, pref_mem_base=32 << 30)
        self.assertEqual([config[0x208+8*i] for i in range(count)],
                         [0xa5000400 | (count << 5), 0xa5000f01, 0xa5000503][:count])

if __name__ == "__main__":
  unittest.main()
