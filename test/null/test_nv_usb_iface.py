import ctypes, unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tinygrad.device import BufferStorage, Buffer, BufferSpec, Device
from tinygrad.dtype import dtypes
from tinygrad.helpers import Context
from tinygrad.runtime import ops_nv
from tinygrad.runtime.support import hcq2
from tinygrad.runtime.support.memory import AddrSpace
from tinygrad.runtime.support.nv.usb import pm_usb_hostio, pm_usb_stage, usb_stage_copy
from tinygrad.uop.ops import Ops, PatternMatcher, UOp, graph_rewrite


class TestNVUSBIface(unittest.TestCase):
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
