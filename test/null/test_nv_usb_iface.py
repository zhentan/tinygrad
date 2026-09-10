import ctypes, unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tinygrad.device import BufferStorage, Buffer, BufferSpec, Device
from tinygrad.dtype import dtypes
from tinygrad.engine.realize import ExecContext, exec_copy
from tinygrad.helpers import Context
from tinygrad.runtime import ops_nv
from tinygrad.runtime.support import hcq2
from tinygrad.runtime.support.memory import AddrSpace
from tinygrad.runtime.support.nv.usb import pm_usb_hostio, pm_usb_stage, usb_stage_copy
from tinygrad.runtime.support.nv import usb as nvusb
from tinygrad.uop.ops import Ops, PatternMatcher, UOp, graph_rewrite


class TestNVUSBIface(unittest.TestCase):
  def test_batched_host_copies_preserve_typed_buffers_and_offsets(self):
    cpu, window = Device["CPU"], 256 << 10
    win = Mock(device="NV", dtype=dtypes.uint8, size=window, host=SimpleNamespace(addr=0x4f000))
    dev = SimpleNamespace(pm_stage_copy=pm_usb_stage, usb_sram_readback=True, usb_sram=win)
    for cin in (False, True):
      batch = 32 * (window // 2 - 512) if cin else 64 * window
      for dtype in (dtypes.uint8, dtypes.float32, dtypes.uint64):
        for total in (dtype.itemsize, batch + dtype.itemsize):
          with self.subTest(upload=cin, dtype=dtype, total=total):
            payload = (bytes(range(256)) * ((total + 255) // 256))[:total]
            data = memoryview(bytearray(b'\xa5' * (total + 2 * dtype.itemsize)))
            if cin: data[dtype.itemsize:-dtype.itemsize] = payload
            host = UOp.from_buffer(Buffer("PYTHON", total // dtype.itemsize + 2, dtype, opaque=data))[1:total // dtype.itemsize + 1]
            gpu = UOp.placeholder((total,), dtypes.uint8, device="NV")
            with Context(HCQ_RUNTIME_DEV="CPU"), patch.object(type(Device), "__getitem__", lambda _, name: dev if name == "NV" else cpu):
              linear = usb_stage_copy(gpu, host) if cin else usb_stage_copy(host, gpu)
            copies = [u for u in linear.src if u.src[0].op is Ops.COPY]
            self.assertEqual(len(copies), (total + batch - 1) // batch)
            for i, copy in enumerate(copies):
              staged = next(u for u in copy.toposort() if u.op is Ops.PARAM and u.tag == "usb_staging")
              buf = Buffer("CPU", staged.numel(), staged.dtype, preallocate=True)
              expected = payload[i * batch:(i + 1) * batch]
              if not cin: buf.host.view()[:] = expected
              copy = copy.substitute({staged: UOp.from_buffer(buf)})
              exec_copy(ExecContext(), copy, copy.src[0])
              if cin: self.assertEqual(bytes(buf.host.view()), expected)
            self.assertEqual(bytes(data), b'\xa5' * dtype.itemsize + payload + b'\xa5' * dtype.itemsize)

  def test_upload_batches_keep_markers_fixed_and_wait_before_reuse(self):
    cpu, window, half = Device["CPU"], 256 << 10, 128 << 10
    win = Mock(device="NV", dtype=dtypes.uint8, size=window, host=SimpleNamespace(addr=0x4f000))
    dev = SimpleNamespace(pm_stage_copy=pm_usb_stage, usb_sram=win)
    for chunks, counts in ((1, (1,)), (3, (3,)), (32, (32,)), (33, (32, 1))):
      size = (chunks - 1) * (half - 512) + 3
      with self.subTest(chunks=chunks), Context(HCQ_RUNTIME_DEV="CPU"), \
           patch.object(type(Device), "__getitem__", lambda _, name: dev if name == "NV" else cpu), \
           patch.object(nvusb, "usb_wait_value", wraps=nvusb.usb_wait_value) as waited:
        linear = usb_stage_copy(UOp.placeholder((size,), dtypes.uint8, device="NV"),
                                UOp.placeholder((size,), dtypes.uint8, device="CPU"))
      submits = [u for u in linear.toposort() if u.op is Ops.CUSTOM_FUNCTION and u.arg == "submit_nv_copy"]
      self.assertEqual(len(submits), len(counts))
      self.assertEqual([c.args[1] for c in waited.call_args_list], [value for count in counts for value in (*range(1, count - 1), count)])
      for submit, count in zip(submits, counts):
        commands = submit.src[0].src
        self.assertIs(commands[0].src[0], hcq2.timeline(("NV",)))
        self.assertEqual(len(commands), 1 + 4 * count)
        for i in range(count):
          wait, copy, clear, release = commands[1 + 4 * i:5 + 4 * i]
          self.assertEqual((wait.arg, wait.src[1].val), (("wait", dtypes.void), i + 1))
          self.assertEqual(hcq2.unwrap_view(wait.src[0])[1], (i % 2 + 1) * half - 8)
          self.assertIs(copy.src[0].op, Ops.COPY)
          self.assertIs(clear.src[0], wait.src[0])
          self.assertEqual(clear.src[1].val, 0)
          self.assertEqual((release.src[0].tag, release.src[1].val), ("usb_readback_done", i + 1))
      ends = [u for u in linear.toposort() if u.op is Ops.END]
      self.assertEqual(len(ends), len({u.src[1] for u in ends}))

  def test_uploads_share_host_staging_without_borrowing_runtime_ring_space(self):
    dev, pad = object.__new__(ops_nv.NVDevice), object()
    with patch.object(ops_nv, "Buffer", return_value=pad) as allocated:
      for _ in range(300):
        b = UOp.placeholder((256 << 10,), dtypes.uint8, device=("NV",), tag=("hcq_host", "usb_upload"))
        self.assertIs(nvusb.pm_usb_bufferize.rewrite(b, ctx=dev), pad)
        store = b.index(0).store(UOp.const(1, dtypes.uint8))
        self.assertIs(graph_rewrite(store, pm_usb_hostio, walk=True), store)
    allocated.assert_called_once_with("CPU", 256 << 10, dtypes.uint8, preallocate=True)

  def test_sram_readback_bounds_batches_and_waits_for_each_host_ready(self):
    cpu, window = Device["CPU"], 256 << 10
    win = Mock(device="NV", dtype=dtypes.uint8, size=window, host=SimpleNamespace(addr=0x4f000))
    dev = SimpleNamespace(pm_stage_copy=pm_usb_stage, usb_sram_readback=True, usb_sram=win)
    for chunks, counts in ((2, (2,)), (64, (64,)), (65, (64, 1))):
      size = (chunks - 1) * window + 3
      with self.subTest(chunks=chunks), Context(HCQ_RUNTIME_DEV="CPU"), \
           patch.object(type(Device), "__getitem__", lambda _, name: dev if name == "NV" else cpu):
        linear = usb_stage_copy(UOp.placeholder((size,), dtypes.uint8, device="CPU"),
                                UOp.placeholder((size,), dtypes.uint8, device="NV"))
      submits = [u for u in linear.toposort() if u.op is Ops.CUSTOM_FUNCTION and u.arg == "submit_nv_copy"]
      self.assertEqual(len(submits), len(counts))
      for submit, count in zip(submits, counts):
        commands = submit.src[0].src
        self.assertEqual(len(commands), 1 + 3 * count)
        self.assertIs(commands[0].src[0], hcq2.timeline(("NV",)))
        for i in range(count):
          wait, copy, release = commands[1 + 3 * i:4 + 3 * i]
          self.assertEqual((wait.arg, wait.src[0].tag, wait.src[1].val), (("wait", dtypes.void), "usb_readback_done", i + 1))
          self.assertIs(copy.src[0].op, Ops.COPY)
          cq, offset = hcq2.unwrap_view(release.src[0])
          self.assertEqual((release.arg, cq.tag, offset, release.src[1].val), (("store", dtypes.void), "usb_read_cq", 12, 0))

  def test_sram_readback_transfers_and_discards_slot_zero_prefix(self):
    cpu, window = Device["CPU"], 256 << 10
    win = Mock(device="NV", dtype=dtypes.uint8, size=window, host=SimpleNamespace(addr=0x4f000))
    dev = SimpleNamespace(pm_stage_copy=pm_usb_stage, usb_sram_readback=True, usb_sram=win)
    for size, lengths in ((1, (262656,)), (4096, (266240,)), (window + 3, (524288, 262656))):
      arms, pulls = [], []
      def arm(devs, read, nbytes, slot_start=0, deps=(), *, idle=True):
        node = UOp.custom_function(f"test_arm_{len(arms)}", *deps)
        arms.append((read, nbytes, slot_start, idle, node))
        return node
      def bulk(devs, deps, endpoint, data, length, timeout=1000):
        node = UOp.custom_function(f"test_pull_{len(pulls)}", *deps)
        pulls.append((endpoint, length, node))
        return node
      with self.subTest(size=size), Context(HCQ_RUNTIME_DEV="CPU"), \
           patch.object(type(Device), "__getitem__", lambda _, name: dev if name == "NV" else cpu), \
           patch.object(nvusb, "usb_scsi", side_effect=arm), patch.object(nvusb, "usb_bulk", side_effect=bulk):
        linear = usb_stage_copy(UOp.placeholder((size,), dtypes.uint8, device="CPU"),
                                UOp.placeholder((size,), dtypes.uint8, device="NV"))
      self.assertEqual([row[:4] for row in arms], [(True, length, 0, False) for length in lengths])
      pulls = [row for row in pulls if row[1] > 8] # exclude the initial timeline read and 64-bit ready reset
      self.assertEqual([row[:2] for row in pulls], [(0x81, length) for length in lengths])
      copies = [u for u in linear.toposort() if u.op is Ops.CALL and u.src[0].op is Ops.CUSTOM_FUNCTION and u.src[0].arg == "memcpy"]
      self.assertEqual(len(copies), len(lengths))
      for i, copy in enumerate(copies):
        self.assertIn(arms[i][-1], pulls[i][-1].toposort())
        self.assertIn(pulls[i][-1], copy.toposort())
        if i: self.assertIn(copies[i - 1], arms[i][-1].toposort())
        source = copy.src[2]
        self.assertIs(source.op, Ops.INDEX)
        self.assertEqual(source.src[1].val, 262144)
        self.assertEqual(copy.src[3].val, min(window, size - i * window))

  def test_sram_readback_is_opt_in_at_device_initialization(self):
    iface = object.__new__(ops_nv.USBIface)
    def init_runtime(dev, _device): dev.pm_bufferize = PatternMatcher([])
    for enabled in (0, 1):
      with self.subTest(enabled=enabled), patch.object(ops_nv.NVDevice, "_select_iface", return_value=iface), \
           patch.object(ops_nv.NVDevice, "_init_runtime", init_runtime), patch.object(ops_nv, "getenv", return_value=enabled):
        dev = ops_nv.NVDevice("")
      self.assertIs(dev.usb_sram_readback, bool(enabled))

  def test_sram_readback_rejects_invalid_windows_before_arming(self):
    cpu = Device["CPU"]
    for address in (0xef00, 0x4f001, 0x53000):
      win = Mock(device="NV", dtype=dtypes.uint8, size=256 << 10, host=SimpleNamespace(addr=address))
      dev = SimpleNamespace(pm_stage_copy=pm_usb_stage, usb_sram_readback=True, usb_sram=win)
      with self.subTest(address=address), patch.object(type(Device), "__getitem__", lambda _, name: dev if name == "NV" else cpu), \
           patch.object(nvusb, "usb_scsi") as arm, self.assertRaisesRegex(RuntimeError, "USB SRAM"):
        usb_stage_copy(UOp.placeholder((4096,), dtypes.uint8, device="CPU"), UOp.placeholder((4096,), dtypes.uint8, device="NV"))
      arm.assert_not_called()

  def test_sram_completion_mapping_does_not_own_vram_or_alias_memory_handles(self):
    iface = object.__new__(ops_nv.USBIface)
    iface.dev_impl = Mock()
    iface.dev_impl.mm.alloc_vaddr.return_value = va = 0x1000000000
    iface.dev_impl.mm.map_range.return_value = mapping = SimpleNamespace(aspace=AddrSpace.SYS)
    iface._native_memory = {0: (sentinel:=object())}
    cq = iface.alloc_usb_read_cq()
    iface.dev_impl.mm.map_range.assert_called_once_with(va, 0x1000, [(0x828000, 0x1000)], aspace=AddrSpace.SYS, uncached=True)
    self.assertEqual((cq.buf, cq.meta.mapping, cq.meta.has_cpu_mapping, cq.meta.hMemory, cq.host), (va, mapping, False, va, None))
    iface.free(cq)
    iface.dev_impl.mm.vfree.assert_not_called()
    self.assertIs(iface._native_memory[0], sentinel)
    dev = object.__new__(ops_nv.NVDevice)
    dev.device, dev.iface = "NV", iface
    with patch.object(iface, "alloc_usb_read_cq", return_value=cq) as alloc, patch.object(ops_nv, "Buffer", return_value=object()) as buffer:
      self.assertIs(dev.usb_read_cq, dev.usb_read_cq)
    alloc.assert_called_once_with()
    buffer.assert_called_once_with("NV", 0x1000, dtypes.uint8, opaque=cq)

  def test_readback_waits_for_prior_compute_before_copying_each_chunk(self):
    cpu, window = Device["CPU"], 256 << 10
    dev = SimpleNamespace(pm_stage_copy=pm_usb_stage, usb_readback=Buffer("NV", window, dtypes.uint8))
    for size in (16, 2 * window + 3):
      with self.subTest(size=size), Context(HCQ_RUNTIME_DEV="CPU"), \
           patch.object(type(Device), "__getitem__", lambda _, name: dev if name == "NV" else cpu):
        src = UOp.placeholder((size,), dtypes.uint8, device="NV")
        dst = UOp.placeholder((size,), dtypes.uint8, device="CPU")
        linear = usb_stage_copy(dst, src)
        assert linear is not None
        submits = [u for u in linear.toposort() if u.op is Ops.CUSTOM_FUNCTION and u.arg == "submit_nv_copy"]
        self.assertEqual(len(submits), (size + window - 1) // window)
        for submit in submits:
          commands = submit.src[0].src
          self.assertEqual((commands[0].op, commands[0].arg), (Ops.INS, ("wait", dtypes.void)))
          self.assertIs(commands[0].src[0], hcq2.timeline(("NV",)))
          self.assertIs(commands[0].src[1], hcq2.timeline_value(("NV",)))
          self.assertIs(commands[1].src[0].op, Ops.COPY)

  def test_nv_usb_keeps_queue_put_counter_on_host(self):
    fifo = SimpleNamespace(entries=8, token=0, ring=object(), gpput=object(), doorbell=object(), put_value=object())
    queue = object.__new__(ops_nv.NVQueue)
    queue.dev, queue.devs, queue.queue = SimpleNamespace(fifos={"COPY:0": fifo}), ("NV",), "COPY:0"
    submitted = queue.submit(UOp.placeholder((4,), dtypes.uint32, device=("NV",), tag="cmdbuf_copy_0"))
    put = next(u for u in submitted.toposort() if u.op is Ops.PARAM and isinstance(u.tag, tuple) and u.tag[0] == "hcq_host")
    self.assertIs(ops_nv._queue_bufferizer(fifo, "COPY:0").rewrite(put, ctx=object()), fifo.put_value)
    load, store = put.index(0).load(), put.index(0).store(UOp.const(2, dtypes.uint64))
    self.assertIs(graph_rewrite(load, pm_usb_hostio, walk=True), load)
    self.assertIs(graph_rewrite(store, pm_usb_hostio, walk=True), store)

  def test_nv_usb_submits_previous_timeline_value(self):
    with patch.object(Device["CPU"], "timeline_submit_offset", -1):
      current = hcq2.timeline_value(("CPU",))
      self.assertEqual(hcq2.timeline_submit_value(("CPU",)), current + UOp.const(-1, dtypes.uint64))
      self.assertIs(hcq2.timeline_submit_value(("CPU",), 1), current)

  def test_usb_queue_entry_scratch_does_not_read_or_write_device_memory(self):
    fifo = SimpleNamespace(entries=8, token=0)
    queue = object.__new__(ops_nv.NVQueue)
    queue.dev, queue.devs, queue.queue = SimpleNamespace(fifos={"COPY:0": fifo}), ("NV",), "COPY:0"
    with Context(HCQ_RUNTIME_DEV="CPU"):
      submitted = queue.submit(UOp.placeholder((4,), dtypes.uint32, device=("NV",), tag="cmdbuf_copy_0"))
      entry = next(u for u in submitted.toposort() if u.op is Ops.PARAM and u.tag == "gpentry_copy_0")
      load, store = entry.index(0).load(), entry.index(0).store(UOp.const(0x200000000004, dtypes.uint64))
      for access in (load, store):
        self.assertIs(graph_rewrite(access, pm_usb_hostio, walk=True), access)
      ring = next(u for u in submitted.toposort() if u.op is Ops.PARAM and u.tag == "ring_copy_0")
      self.assertIsNot(graph_rewrite(ring.index(0).store(load), pm_usb_hostio, walk=True), ring.index(0).store(load))

  def test_nv_usb_installs_hcq_transport(self):
    iface = object.__new__(ops_nv.USBIface)
    iface.pci_dev = SimpleNamespace(usb=SimpleNamespace(usb=SimpleNamespace(handle=ctypes.c_void_p(0x1234))))
    base = PatternMatcher([])
    def init_runtime(dev, _device): dev.pm_bufferize = base
    with patch.object(ops_nv.NVDevice, "_select_iface", return_value=iface), \
         patch.object(ops_nv.NVDevice, "_init_runtime", init_runtime):
      dev = ops_nv.NVDevice("")

    self.assertIs(dev.pm_lower, ops_nv.pm_usb_hostio)
    self.assertIs(dev.pm_stage_copy, ops_nv.pm_usb_stage)
    self.assertEqual(dev.timeline_submit_offset, 0)
    handle = UOp.placeholder((1,), dtypes.uint64, device=("NV",), tag="usb_handle")
    resolved = dev.pm_bufferize.rewrite(handle, ctx=dev)
    self.assertIsInstance(resolved, Buffer)
    self.assertEqual((resolved.device, resolved.host.view(fmt='Q')[0]), ("CPU", 0x1234))
    dev.__dict__["usb_readback_done"] = done = object()
    signal = UOp.placeholder((1,), dtypes.uint64, device=("NV",), tag="usb_readback_done")
    self.assertIs(dev.pm_bufferize.rewrite(signal, ctx=dev), done)

  def test_usb_target_selects_nv_usb_interface(self):
    usb_iface = getattr(ops_nv, "USBIface", None)
    self.assertIsNotNone(usb_iface)
    self.assertIn(usb_iface, ops_nv.NVDevice.ifaces)
    with Context(DEV="USB+NV:CUDA"), patch.object(usb_iface, "__init__", return_value=None) as init:
      selected = object.__new__(ops_nv.NVDevice)._select_iface("")
    self.assertIsInstance(selected, usb_iface)
    init.assert_called_once_with(unittest.mock.ANY, 0)

  def test_nv_usb_interface_uses_native_chestnut_transport(self):
    usb_iface = getattr(ops_nv, "USBIface", None)
    self.assertIsNotNone(usb_iface)
    usb_device, pci_device, nvdev = object(), Mock(), Mock()
    with patch.object(ops_nv.USB3, "list_devices", return_value=[(usb_device, "usb:4-3")]) as listed, \
         patch.object(ops_nv, "USBPCIDevice", return_value=pci_device) as opened, \
         patch.object(ops_nv.System, "reserve_va") as reserved, patch.object(ops_nv, "NVNativeDev", return_value=nvdev):
      iface = usb_iface(Mock(), 0)
    listed.assert_called_once_with(0x3801, 0x0001)
    opened.assert_called_once_with("NV", usb_device, "usb:4-3")
    reserved.assert_called_once_with(ops_nv.NVMemoryManager.va_allocator.base, ops_nv.NVMemoryManager.va_allocator.size)
    self.assertIs(iface.pci_dev, pci_device)
    self.assertIs(iface.dev_impl, nvdev)
    self.assertEqual((iface.vram_bar, iface.count), (1, 1))
    self.assertEqual((iface.gpfifo_class, iface.compute_class, iface.dma_class, iface.viddec_class),
                     (ops_nv.nv_gpu.AMPERE_CHANNEL_GPFIFO_A, ops_nv.nv_gpu.AMPERE_COMPUTE_B, ops_nv.nv_gpu.AMPERE_DMA_COPY_B, None))

  def test_failed_native_initialization_resets_tunneled_device(self):
    usb_iface = getattr(ops_nv, "USBIface", None)
    self.assertIsNotNone(usb_iface)
    pci_device = Mock()
    with patch.object(ops_nv.USB3, "list_devices", return_value=[(object(), "usb:4-3")]), \
         patch.object(ops_nv, "USBPCIDevice", return_value=pci_device), patch.object(ops_nv.System, "reserve_va"), \
         patch.object(ops_nv, "NVNativeDev", side_effect=RuntimeError("native bootstrap failed")):
      with self.assertRaisesRegex(RuntimeError, "native bootstrap failed"):
        usb_iface(Mock(), 0)
    pci_device.reset.assert_called_once_with()

  def test_native_usb_allocates_logical_rm_prerequisites(self):
    iface = object.__new__(ops_nv.USBIface)
    iface._setup_native_nv()
    nvdevice = iface.rm_alloc(iface.root, ops_nv.nv_gpu.NV01_DEVICE_0)
    subdevice = iface.rm_alloc(nvdevice, ops_nv.nv_gpu.NV20_SUBDEVICE_0)
    virtmem = iface.rm_alloc(nvdevice, ops_nv.nv_gpu.NV01_MEMORY_VIRTUAL)
    vaspace = iface.rm_alloc(nvdevice, ops_nv.nv_gpu.FERMI_VASPACE_A)
    channel_group = iface.rm_alloc(nvdevice, ops_nv.nv_gpu.KEPLER_CHANNEL_GROUP_A)
    ctxshare = iface.rm_alloc(channel_group, ops_nv.nv_gpu.FERMI_CONTEXT_SHARE_A)
    self.assertEqual(len({nvdevice, subdevice, virtmem, vaspace, channel_group, ctxshare}), 6)
    with self.assertRaisesRegex(RuntimeError, "invalid parent"):
      iface.rm_alloc(channel_group, ops_nv.nv_gpu.AMPERE_COMPUTE_B)

  def test_native_usb_materializes_ampere_gpfifo_channel(self):
    iface, native = object.__new__(ops_nv.USBIface), Mock()
    iface.dev_impl = native
    iface._setup_native_nv()
    nvdevice = iface.rm_alloc(iface.root, ops_nv.nv_gpu.NV01_DEVICE_0)
    iface.rm_alloc(nvdevice, ops_nv.nv_gpu.NV20_SUBDEVICE_0)
    iface.rm_alloc(nvdevice, ops_nv.nv_gpu.NV01_MEMORY_VIRTUAL)
    vaspace = iface.rm_alloc(nvdevice, ops_nv.nv_gpu.FERMI_VASPACE_A)
    channel_group = iface.rm_alloc(nvdevice, ops_nv.nv_gpu.KEPLER_CHANNEL_GROUP_A)
    ctxshare = iface.rm_alloc(channel_group, ops_nv.nv_gpu.FERMI_CONTEXT_SHARE_A,
                              ops_nv.nv_gpu.NV_CTXSHARE_ALLOCATION_PARAMETERS(hVASpace=vaspace))
    gpfifo = BufferStorage(0x1000000000, meta=SimpleNamespace(hMemory=0x400000, mapping=SimpleNamespace(size=0x300000)))
    iface._native_memory[gpfifo.meta.hMemory] = gpfifo
    native.alloc_channel.return_value = channel = SimpleNamespace(token=0)
    params = ops_nv.nv_gpu.NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS(
      gpFifoOffset=gpfifo.buf, gpFifoEntries=0x10000, hContextShare=ctxshare,
      hObjectBuffer=gpfifo.meta.hMemory,
      hUserdMemory=(ctypes.c_uint32*8)(gpfifo.meta.hMemory),
      userdOffset=(ctypes.c_uint64*8)(0x80000))

    handle = iface.rm_alloc(channel_group, ops_nv.nv_gpu.AMPERE_CHANNEL_GPFIFO_A, params)

    native.alloc_channel.assert_called_once_with(gpfifo.buf, 0x10000, 0x480000)
    self.assertIs(iface._native_channels[handle], channel)
    self.assertEqual(iface._native_objects[handle], ops_nv.nv_gpu.AMPERE_CHANNEL_GPFIFO_A)
    token = ops_nv.nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS(workSubmitToken=-1)
    self.assertIs(iface.rm_control(handle, ops_nv.nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN, token), token)
    self.assertEqual(token.workSubmitToken, 0)

  def test_native_usb_binds_compute_context(self):
    iface, native = object.__new__(ops_nv.USBIface), Mock()
    iface.dev_impl = native
    iface._setup_native_nv()
    nvdevice = iface.rm_alloc(iface.root, ops_nv.nv_gpu.NV01_DEVICE_0)
    vaspace = iface.rm_alloc(nvdevice, ops_nv.nv_gpu.FERMI_VASPACE_A)
    channel_group = iface.rm_alloc(nvdevice, ops_nv.nv_gpu.KEPLER_CHANNEL_GROUP_A)
    ctxshare = iface.rm_alloc(channel_group, ops_nv.nv_gpu.FERMI_CONTEXT_SHARE_A,
                              ops_nv.nv_gpu.NV_CTXSHARE_ALLOCATION_PARAMETERS(hVASpace=vaspace))
    memory = BufferStorage(0x1000000000, meta=SimpleNamespace(hMemory=0x400000, mapping=SimpleNamespace(size=0x300000)))
    iface._native_memory[memory.meta.hMemory] = memory
    native.alloc_channel.return_value = channel = SimpleNamespace(token=0)
    params = ops_nv.nv_gpu.NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS(
      gpFifoOffset=memory.buf, gpFifoEntries=0x10000, hContextShare=ctxshare,
      hObjectBuffer=memory.meta.hMemory, hUserdMemory=(ctypes.c_uint32*8)(memory.meta.hMemory),
      userdOffset=(ctypes.c_uint64*8)(0x80000))
    gpfifo = iface.rm_alloc(channel_group, ops_nv.nv_gpu.AMPERE_CHANNEL_GPFIFO_A, params)

    compute = iface.rm_alloc(gpfifo, ops_nv.nv_gpu.AMPERE_COMPUTE_B)
    debugger = iface.rm_alloc(nvdevice, ops_nv.nv_gpu.GT200_DEBUGGER,
                              ops_nv.nv_gpu.NV83DE_ALLOC_PARAMETERS(hAppClient=iface.root, hClass3dObject=compute))

    native.gr.bind_compute_context.assert_called_once_with(channel)
    self.assertEqual(iface._native_objects[compute], ops_nv.nv_gpu.AMPERE_COMPUTE_B)
    self.assertEqual(iface._native_objects[debugger], ops_nv.nv_gpu.GT200_DEBUGGER)

  def test_native_usb_binds_copy_context(self):
    iface, native = object.__new__(ops_nv.USBIface), Mock()
    iface.dev_impl = native
    iface._setup_native_nv()
    nvdevice = iface.rm_alloc(iface.root, ops_nv.nv_gpu.NV01_DEVICE_0)
    vaspace = iface.rm_alloc(nvdevice, ops_nv.nv_gpu.FERMI_VASPACE_A)
    channel_group = iface.rm_alloc(nvdevice, ops_nv.nv_gpu.KEPLER_CHANNEL_GROUP_A)
    ctxshare = iface.rm_alloc(channel_group, ops_nv.nv_gpu.FERMI_CONTEXT_SHARE_A,
                              ops_nv.nv_gpu.NV_CTXSHARE_ALLOCATION_PARAMETERS(hVASpace=vaspace))
    memory = BufferStorage(0x1000000000, meta=SimpleNamespace(hMemory=0x400000, mapping=SimpleNamespace(size=0x300000)))
    iface._native_memory[memory.meta.hMemory] = memory
    native.alloc_channel.return_value = channel = SimpleNamespace(token=0)
    params = ops_nv.nv_gpu.NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS(
      gpFifoOffset=memory.buf, gpFifoEntries=0x10000, hContextShare=ctxshare,
      hObjectBuffer=memory.meta.hMemory, hUserdMemory=(ctypes.c_uint32*8)(memory.meta.hMemory),
      userdOffset=(ctypes.c_uint64*8)(0x80000))
    gpfifo = iface.rm_alloc(channel_group, ops_nv.nv_gpu.AMPERE_CHANNEL_GPFIFO_A, params)

    copy = iface.rm_alloc(gpfifo, ops_nv.nv_gpu.AMPERE_DMA_COPY_B)

    native.bind_copy_context.assert_called_once_with(channel)
    self.assertEqual(iface._native_objects[copy], ops_nv.nv_gpu.AMPERE_DMA_COPY_B)

  def test_native_usb_schedules_ampere_channel_group(self):
    iface, native = object.__new__(ops_nv.USBIface), Mock()
    iface.dev_impl = native
    iface._setup_native_nv()
    nvdevice = iface.rm_alloc(iface.root, ops_nv.nv_gpu.NV01_DEVICE_0)
    channel_group = iface.rm_alloc(nvdevice, ops_nv.nv_gpu.KEPLER_CHANNEL_GROUP_A)
    channels = [SimpleNamespace(chid=0), SimpleNamespace(chid=1)]
    for index, channel in enumerate(channels):
      handle = 0xD0000000 + index
      iface._native_objects[handle] = ops_nv.nv_gpu.AMPERE_CHANNEL_GPFIFO_A
      iface._native_parents[handle] = channel_group
      iface._native_channels[handle] = channel
    params = ops_nv.nv_gpu.NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS(bEnable=1)

    self.assertIs(iface.rm_control(channel_group, ops_nv.nv_gpu.NVA06C_CTRL_CMD_GPFIFO_SCHEDULE, params), params)

    native.schedule_channel_group.assert_called_once_with(channels)

  def test_native_usb_reports_ga102_info_and_accepts_optional_perf_boost(self):
    iface = object.__new__(ops_nv.USBIface)
    iface._setup_native_nv()
    nvdevice = iface.rm_alloc(iface.root, ops_nv.nv_gpu.NV01_DEVICE_0)
    subdevice = iface.rm_alloc(nvdevice, ops_nv.nv_gpu.NV20_SUBDEVICE_0)
    params = ops_nv.nv_gpu.NV2080_CTRL_PERF_BOOST_PARAMS(duration=0xffffffff)
    self.assertIs(iface.rm_control(subdevice, ops_nv.nv_gpu.NV2080_CTRL_CMD_PERF_BOOST, params), params)
    info = ops_nv.nv_gpu.NV2080_CTRL_INTERNAL_STATIC_KGR_GET_INFO_PARAMS()
    self.assertIs(iface.rm_control(subdevice, ops_nv.nv_gpu.NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_INFO, info), info)
    self.assertEqual({index:info.engineInfo[0].infoList[index].data for index in (12, 13, 20, 23, 32)},
                     {12:0x806, 13:48, 20:7, 23:6, 32:2})
    with self.assertRaisesRegex(RuntimeError, "native NV control .* is not implemented"):
      iface.rm_control(subdevice, ops_nv.nv_gpu.NV2080_CTRL_CMD_GR_GET_INFO)

  def test_native_usb_sleep_does_not_poll_uninitialized_gsp_queue(self):
    iface = object.__new__(ops_nv.USBIface)
    iface.dev_impl = Mock(spec=[])
    iface.sleep(200)

  def test_native_usb_forces_runtime_allocations_into_vram(self):
    iface = object.__new__(ops_nv.USBIface)
    iface._native_memory = {}
    result = SimpleNamespace(meta=SimpleNamespace(hMemory=0x123000))
    with patch.object(ops_nv.PCIIfaceBase, "alloc", autospec=True, return_value=result) as alloc:
      self.assertIs(iface.alloc(0x1234, host=True, uncached=True, cpu_access=True, contiguous=True,
                               force_devmem=False, zero=True, marker=7), result)
    alloc.assert_called_once_with(iface, 0x1234, host=False, uncached=True, cpu_access=True, contiguous=True,
                                  force_devmem=True, zero=True, marker=7)

  def test_native_usb_allocates_staging_window_in_controller_sram(self):
    iface = object.__new__(ops_nv.USBIface)
    result = object()
    with patch.object(ops_nv.PCIIfaceBase, "alloc", autospec=True, return_value=result) as alloc:
      self.assertIs(iface.alloc_usb_sram(256 << 10), result)
    alloc.assert_called_once_with(iface, 256 << 10, host=True, cpu_access=True, contiguous=True)

  def test_native_usb_reuses_freed_memory_handle(self):
    iface = object.__new__(ops_nv.USBIface)
    iface._setup_native_nv()
    iface.vram_bar, iface.pci_dev, iface.dev_impl = 1, Mock(), Mock()
    iface.pci_dev.bar_info.return_value = (0, 256 << 20)
    # The physical allocator may hand the same page back after it is freed.
    mapping = SimpleNamespace(va_addr=0x1000000000, paddrs=[(0x400000, 0x1000)], size=0x1000, aspace=AddrSpace.PHYS)
    iface.dev_impl.mm.valloc.return_value = mapping
    first = iface.alloc(32)
    iface.free(first)
    iface.dev_impl.mm.vfree.assert_called_once_with(mapping)
    second = iface.alloc(32)
    self.assertIsNot(first, second)
    self.assertEqual(first.meta.hMemory, second.meta.hMemory)
    self.assertIs(iface._native_memory[second.meta.hMemory], second)
    with self.assertRaisesRegex(RuntimeError, "memory handle collision"): iface.alloc(32)

  def test_nv_usb_readback_buffers_are_cpu_addressable_vram(self):
    dev = object.__new__(ops_nv.NVDevice)
    dev.device, dev.iface = "NV", object.__new__(ops_nv.USBIface)
    readback, done = object(), object()
    with patch.object(ops_nv, "Buffer", side_effect=(readback, done)) as buffer:
      self.assertIs(dev.usb_readback, readback)
      self.assertIs(dev.usb_readback_done, done)
    self.assertEqual(buffer.call_args_list, [
      unittest.mock.call("NV", 256 << 10, dtypes.uint8, options=BufferSpec(cpu_access=True, nolru=True), preallocate=True),
      unittest.mock.call("NV", 1, dtypes.uint64, options=BufferSpec(cpu_access=True, nolru=True), preallocate=True),
    ])

  def test_failed_nv_device_construction_finalizes_initialized_interface(self):
    iface, error = Mock(), RuntimeError("native allocation failed")
    iface.root, iface.gpu_instance, iface.rm_alloc.side_effect = 1, 0, error
    with patch.object(ops_nv.NVDevice, "_select_iface", return_value=iface):
      with self.assertRaises(RuntimeError) as raised:
        ops_nv.NVDevice("0")
    self.assertIs(raised.exception, error)
    iface.device_fini.assert_called_once_with()


if __name__ == "__main__":
  unittest.main()
