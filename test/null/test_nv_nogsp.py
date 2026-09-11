from __future__ import annotations

import hashlib, struct
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, call, patch

from tinygrad.runtime.support.nv.gr import GA102ContextBuffer, GA102Topology, NVNativeGR
from tinygrad.runtime.support.memory import AddrSpace, VirtMapping
from tinygrad.runtime.support.nv.ip import NV_FLCN
from tinygrad.runtime.support.nv.nogsp import WPR_ALLOCATION_BYTES, build_ga102_acr_package
from tinygrad.runtime.support.nv.nvdev import NVNativeBootstrap, NVNativeChannel, NVNativeDev


class TestGA102NativeACRPackage(unittest.TestCase):
  def test_exact_signed_package_for_2mib_shadow(self):
    package = build_ga102_acr_package(0x200000)
    self.assertEqual((package.shadow_start, package.protected_start, package.protected_end), (0x200000, 0x240000, 0x280000))
    self.assertEqual(len(package.wpr_allocation), WPR_ALLOCATION_BYTES)
    self.assertEqual(hashlib.sha256(package.wpr_shadow).hexdigest(),
                     "4ae4f10ba6842d7e18428876fb6c76ce857259f3a2fbf98d839f1c59ffff3312")
    self.assertEqual(hashlib.sha256(package.wpr_allocation).hexdigest(),
                     "bfc8a0fae81373d33e516920db06c3ddfbb2376a0cc700e47c585280ecff70f1")
    self.assertEqual(hashlib.sha256(package.system_stage).hexdigest(),
                     "8d8c676109bb0b0db4b32d6f367ec6bb06eab4770418859af03aa12a4c4f6e38")
    self.assertEqual(hashlib.sha256(package.gr_net).hexdigest(),
                     "56b3b302589b9e3bbb7d6ac8f5e36840e4297773c3900779c27c0505c1e605d5")
    self.assertEqual([(helper.name, helper.gpu_address, helper.controller_address, len(helper.payload)) for helper in package.helpers], [
      ("AHESASC", 0x224000, 0x33000, 55552),
      ("ASB", 0x232000, 0x41000, 27136),
      ("unload", 0x239000, 0x48000, 16384),
      ("VPR_SCRUBBER", 0x23D000, 0x4C000, 7168),
    ])

  def test_wpr_allocation_rejects_unaligned_or_out_of_vram_address(self):
    for shadow_start in (-0x40000, 1, (24 << 30) - 0x40000):
      with self.subTest(shadow_start=shadow_start), self.assertRaises(ValueError):
        build_ga102_acr_package(shadow_start)

  def test_signed_falcon_loader_can_read_coherent_system_memory(self):
    nvdev, route = MagicMock(), Mock()
    nvdev.NV_PFALCON_FBIF_TRANSCFG.with_base.return_value.__getitem__.return_value = route
    nvdev.NV_PFALCON_FBIF_TRANSCFG_TARGET_COHERENT_SYSMEM = 1
    nvdev.NV_PFALCON_FBIF_TRANSCFG_MEM_TYPE_PHYSICAL = 1
    command = nvdev.NV_PFALCON_FALCON_DMATRFCMD.with_base.return_value
    command.encode.side_effect = (0x614, 0x600)
    nvdev.NV_PFALCON_FALCON_MAILBOX0.with_base.return_value.read.return_value = 0
    nvdev.NV_PFALCON_FALCON_MAILBOX1.with_base.return_value.read.return_value = 0
    flcn = object.__new__(NV_FLCN)
    flcn.nvdev = nvdev
    dma_gate = Mock()
    with patch.object(flcn, "disable_ctx_req") as disable, patch.object(flcn, "execute_dma") as execute_dma, \
         patch.object(flcn, "start_cpu") as start_cpu, patch.object(flcn, "wait_cpu_halted") as wait_cpu_halted:
      result = flcn.execute_hs(0x840000, 0x224000, code_off=0x100, data_off=0x8100,
                               imemPa=0, imemVa=0x100, imemSz=0x8000, dmemPa=0, dmemVa=0, dmemSz=0x4E00,
                               pkc_off=0x4B90, engid=1, ucodeid=1, mailbox=0xCAFEBEEF, ctx_dma=5, target=1, dma_gate=dma_gate)
    disable.assert_called_once_with(0x840000)
    route.update.assert_called_once_with(target=1, mem_type=1)
    self.assertEqual(command.encode.call_args_list, [
      call(write=0, size=nvdev.NV_PFALCON_FALCON_DMATRFCMD_SIZE_256B, ctxdma=5, imem=1, sec=1),
      call(write=0, size=nvdev.NV_PFALCON_FALCON_DMATRFCMD_SIZE_256B, ctxdma=5, imem=0, sec=0),
    ])
    self.assertEqual(execute_dma.call_args_list, [
      call(0x840000, 0x614, dest=0, mem_off=0x100, src=0x224000, size=0x8000),
      call(0x840000, 0x600, dest=0, mem_off=0, src=0x22C100, size=0x4E00),
    ])
    start_cpu.assert_called_once_with(0x840000)
    wait_cpu_halted.assert_called_once_with(0x840000)
    self.assertEqual(dma_gate.call_args_list, [call(True), call(False)])
    self.assertEqual(result, (0, 0))

  def test_ampere_channel_instance_matches_nouveau(self):
    image = NVNativeDev._ga102_channel_image(root_paddr=0, va_limit=0x2000000000000,
                                             gpfifo_offset=0x1000000000, entries=0x10000,
                                             chid=0, nonstall_vector=0xA0)

    self.assertEqual(len(image), 0x1000)
    self.assertEqual(hashlib.sha256(image).hexdigest(), "1b7e48902b92183cb784c9d5c2f83ec02e368b01528cfc51470ccdaa147ecc4f")
    self.assertEqual({offset:struct.unpack_from("<I", image, offset)[0] for offset in
                      (0x010, 0x030, 0x048, 0x04C, 0x084, 0x094, 0x0E4, 0x0E8, 0x0F4, 0x0F8)}, {
      0x010:0x0000FACE, 0x030:0x7FFFF902, 0x048:0, 0x04C:0x00100010, 0x084:0x20400000,
      0x094:0x30000001, 0x0E4:0, 0x0E8:0, 0x0F4:0x00001000, 0x0F8:0x800000A0,
    })
    self.assertEqual(struct.unpack_from("<QQI", image, 0x200), (0xC00, 0x1FFFFFFFFFFFF, 0))
    self.assertEqual(struct.unpack_from("<II", image, 0x298), (1, 0))
    self.assertEqual(struct.unpack_from("<III", image, 0x2A0), (0xC00, 0, 0))
    self.assertTrue(all(struct.unpack_from("<III", image, 0x2A0 + index * 0x10) == (1, 1, 0) for index in range(1, 64)))

  def test_signed_loader_uses_nvdec_second_register_window(self):
    native = object.__new__(NVNativeDev)
    native.rreg, native.wreg = Mock(return_value=0), Mock()
    flcn = object.__new__(NV_FLCN)
    flcn.nvdev = native
    flcn.init_regs()
    with patch.object(flcn, "disable_ctx_req"), patch.object(flcn, "execute_dma"), \
         patch.object(flcn, "start_cpu"), patch.object(flcn, "wait_cpu_halted"):
      flcn.execute_hs(0x848000, 0x23D000, code_off=0x100, data_off=0x1500,
                      imemPa=0, imemVa=0x100, imemSz=0x1400, dmemPa=0, dmemVa=0, dmemSz=0x700,
                      pkc_off=0, engid=4, ucodeid=14, target=1, brom_offset=0x1C00)
    # GA102 NVDEC addr2=0x1c00, with authentication registers defined by ga102_flcn_fw_boot.
    self.assertEqual({address:value for (address, value), _ in native.wreg.call_args_list if address >= 0x849000},
                     {0x849E10:0, 0x849D9C:4, 0x849D98:14, 0x849D80:1})

  def test_ampere_channel_materialization_uses_live_runlist_metadata(self):
    regs = {0x224FC:39 << 20, 0xC00004:0x00C2000B, 0xC00008:0, 0xC00160:0xC00000A0, 0xC20000:0}
    top = [0x80000040, 0xC040000C, 0x00C00000] + [0] * 36
    regs.update({0x22800 + index * 4:value for index, value in enumerate(top)})
    writes = {}
    native = object.__new__(NVNativeDev)
    native.rreg = lambda address: regs[address]
    native.mm = SimpleNamespace(root_page_table=SimpleNamespace(paddr=0), palloc=Mock(return_value=0x4321000))
    native.vram = SimpleNamespace(__setitem__=lambda _, key, value: writes.__setitem__(key, bytes(value)))
    native._native_chid = 0

    with patch.object(NVNativeDev, "_write_vram", autospec=True) as write_vram:
      channel = native.alloc_channel(0x1000000000, 0x10000, 0x700000)

    self.assertEqual(channel, NVNativeChannel(chid=0, instance_paddr=0x4321000, runlist=0xC00000,
                                              channel_table=0xC20000, doorbell=0, nonstall_vector=0xA0,
                                              userd_paddr=0x700000))
    native.mm.palloc.assert_called_once_with(0x1000, align=0x1000, zero=False)
    self.assertEqual(write_vram.call_args_list, [
      call(native, 0x4321000, NVNativeDev._ga102_channel_image(root_paddr=0, va_limit=0x2000000000000,
                                                               gpfifo_offset=0x1000000000, entries=0x10000,
                                                               chid=0, nonstall_vector=0xA0)),
      *[call(native, 0x700000 + offset, bytes(4)) for offset in NVNativeDev.USERD_CLEAR_OFFSETS],
    ])
    self.assertEqual(native._native_chid, 1)

  def test_native_dev_schedules_ga102_channel_group(self):
    native = object.__new__(NVNativeDev)
    native.mm = Mock()
    native.gr = Mock()
    native.mm.palloc.return_value = 0xFB8000
    native._native_runlist_paddr = None
    native.regs, native.writes = {0xC00300:0x7FF, 0xC0008C:0, 0x70000:0}, []
    native.rreg = lambda address: native.regs.get(address, 0)
    def wreg(address, value):
      native.writes.append((address, value))
      native.regs[address] = 0 if address == 0x70000 else value
    native.wreg = wreg
    channels = [
      NVNativeChannel(chid=0, instance_paddr=0xE58000, runlist=0xC00000, channel_table=0xC20000,
                      doorbell=0, nonstall_vector=0xA0, userd_paddr=0xD00000),
      NVNativeChannel(chid=1, instance_paddr=0xFB1000, runlist=0xC00000, channel_table=0xC20000,
                      doorbell=0, nonstall_vector=0xA0, userd_paddr=0xD80000),
    ]

    with patch.object(native, "_write_vram") as write_vram:
      native.schedule_channel_group(channels)

    native.mm.palloc.assert_called_once_with(0x1000, align=0x1000, zero=True)
    image = write_vram.call_args.args[1]
    self.assertEqual(write_vram.call_args.args[0], 0xFB8000)
    self.assertEqual(len(image), 0x1000)
    self.assertEqual([struct.unpack_from("<4I", image, offset) for offset in (0, 0x10, 0x20)], [
      (0x80030001, 2, 0, 0),
      (0xD00000, 0, 0xE58000, 0),
      (0xD80000, 0, 0xFB1001, 0),
    ])
    self.assertFalse(any(image[0x30:]))
    self.assertEqual(native.writes, [
      (0x70000, 1), (0xC00088, 0), (0xC00098, 0), (0xC00300, 0x800007FF),
      (0xC20000, 2), (0xC00090, 0), (0xC20004, 2), (0xC00090, 1),
      (0xC00080, 0xFB8000), (0xC00084, 0), (0xC00088, 3),
    ])
    self.assertEqual(native._native_runlist_paddr, 0xFB8000)
    native.gr.bind_channel_group_context.assert_called_once_with(channels)

  def test_signed_falcon_loader_disables_bus_mastering_after_dma_failure(self):
    nvdev, dma_gate, error = MagicMock(), Mock(), RuntimeError("first DMA failed")
    nvdev.NV_PFALCON_FBIF_TRANSCFG_MEM_TYPE_PHYSICAL = 1
    nvdev.NV_PFALCON_FALCON_DMATRFCMD.with_base.return_value.encode.return_value = 0x614
    flcn = object.__new__(NV_FLCN)
    flcn.nvdev = nvdev
    with patch.object(flcn, "disable_ctx_req"), patch.object(flcn, "execute_dma", side_effect=error), \
         self.assertRaises(RuntimeError) as raised:
      flcn.execute_hs(0x840000, 0x224000, code_off=0x100, data_off=0x8100,
                      imemPa=0, imemVa=0x100, imemSz=0x8000, dmemPa=0, dmemVa=0, dmemSz=0x4E00,
                      pkc_off=0x4B90, engid=1, ucodeid=1, ctx_dma=5, target=1, dma_gate=dma_gate)
    self.assertIs(raised.exception, error)
    self.assertEqual(dma_gate.call_args_list, [call(True), call(False)])

  def test_native_bootstrap_stages_and_boots_both_signed_helpers(self):
    class TracedVRAM:
      def __init__(self): self.backing, self.writes = bytearray(0x280000), []
      def __getitem__(self, index): return memoryview(self.backing)[index]
      def __setitem__(self, index, data):
        self.writes.append((index, bytes(data)))
        self.backing[index] = data

    package, flcn, nvdev = build_ga102_acr_package(0x200000), Mock(), Mock()
    nvdev.fw_name, nvdev.chip_id, nvdev.vram = "ga102", 0xB71F00A1, TracedVRAM()
    nvdev.mm.palloc.return_value = 0x200000
    nvdev.pci_dev.sram.alloc.side_effect = (0, 0x24000)
    nvdev.pci_dev.usb.read.return_value = b"\0"
    command, command_writes = 0x403, []

    def read_config(offset, size):
      self.assertEqual(size, 2)
      return command if offset == 4 else 0x10

    def write_config_flush(offset, value, size):
      nonlocal command
      self.assertEqual((offset, size), (4, 2))
      command, _ = value, command_writes.append(value)

    def execute_hs(*args, **kwargs):
      kwargs["dma_gate"](True)
      kwargs["dma_gate"](False)
      return (0, 0)

    nvdev.pci_dev.read_config.side_effect = read_config
    nvdev.pci_dev.write_config_flush.side_effect = write_config_flush
    nvdev.rreg.side_effect = lambda offset: {0x1FA80C: 0x28, 0x1FA81C: 0x2400, 0x1FA820: 0x2600}[offset]
    flcn.execute_hs.side_effect = execute_hs
    with patch.object(NVNativeBootstrap, "_start_sec2") as start_sec2, \
         patch.object(NVNativeBootstrap, "_bootstrap_gr_falcon") as bootstrap_gr_falcon:
      NVNativeBootstrap(nvdev, flcn, package).init_hw()

    nvdev.mm.palloc.assert_called_once_with(WPR_ALLOCATION_BYTES, align=0x40000, zero=False)
    self.assertEqual(bytes(nvdev.vram[0x200000:0x280000]), package.wpr_allocation)
    self.assertEqual([(index.start, index.stop, hashlib.sha256(data).hexdigest()) for index, data in nvdev.vram.writes], [
      (0x200000, 0x280000, hashlib.sha256(bytes(WPR_ALLOCATION_BYTES)).hexdigest()),
      (0x200000, 0x240000, hashlib.sha256(package.wpr_shadow).hexdigest()),
    ])
    self.assertEqual(nvdev.pci_dev.sram.alloc.call_args_list, [call(0x24000), call(0x1C000)])
    nvdev.pci_dev.usb.scsi_write.assert_called_once_with(package.system_stage, slot_start=9)
    self.assertEqual(flcn.reset.call_args_list, [call(0x840000), call(0x110000)])
    self.assertEqual([args[0] for args, _kwargs in flcn.execute_hs.call_args_list], [0x840000, 0x110000])
    self.assertEqual([kwargs["img_paddr"] for _args, kwargs in flcn.execute_hs.call_args_list], [0x224000, 0x232000])
    self.assertEqual([kwargs["ctx_dma"] for _args, kwargs in flcn.execute_hs.call_args_list], [5, 5])
    self.assertEqual([kwargs["target"] for _args, kwargs in flcn.execute_hs.call_args_list], [1, 1])
    self.assertEqual(command_writes, [0x407, 0x403, 0x407, 0x403])
    self.assertEqual(command, 0x403)
    start_sec2.assert_called_once_with()
    self.assertEqual(bootstrap_gr_falcon.call_args_list, [call(2), call(3)])

  def test_native_bootstrap_reports_first_wpr_mismatch(self):
    class CorruptVRAM:
      def __init__(self): self.backing = bytearray(0x280000)
      def __setitem__(self, index, data): self.backing[index] = data
      def __getitem__(self, index):
        data = bytearray(self.backing[index])
        data[0x1234] ^= 0xFF
        return data

    package, flcn, nvdev = build_ga102_acr_package(0x200000), Mock(), Mock()
    nvdev.fw_name, nvdev.vram = "ga102", CorruptVRAM()
    nvdev.mm.palloc.return_value = 0x200000
    nvdev.pci_dev.read_config.return_value = 0x403
    with self.assertRaises(RuntimeError) as raised:
      NVNativeBootstrap(nvdev, flcn, package).init_hw()
    self.assertIn("offset 0x1234 (GPU 0x201234)", str(raised.exception))
    self.assertIn("actual_sha256=", str(raised.exception))
    self.assertIn("expected_sha256=bfc8a0fae81373d33e516920db06c3ddfbb2376a0cc700e47c585280ecff70f1", str(raised.exception))

  def test_native_bootstrap_initializes_locked_vram_before_acr(self):
    for initially_locked, scrub_succeeds in ((True, True), (True, False), (False, True)):
      with self.subTest(initially_locked=initially_locked, scrub_succeeds=scrub_succeeds):
        nvdev, flcn = Mock(), Mock()
        nvdev.fw_name, nvdev.chip_id, nvdev.vram = "ga102", 0xB71F00A1, bytearray(0x280000)
        nvdev.mm.palloc.return_value = 0x200000
        nvdev.pci_dev.sram.alloc.side_effect = (0, 0x24000)
        nvdev.pci_dev.usb.read.return_value = b"\0"
        nvdev.pci_dev.read_config.side_effect = lambda address, size: 0x403 if address == 4 else 0x10
        regs = {0x600:0xFFFFBFFF, 0x1FA80C:0x38 if initially_locked else 0x28, 0x1FA81C:0x2400, 0x1FA820:0x2600,
                0x848100:0x10, 0x848118:2}
        nvdev.rreg.side_effect = lambda address: regs.get(address, 0)
        nvdev.wreg.side_effect = lambda address, value: regs.__setitem__(address, value)

        def execute_hs(base, **kwargs):
          if base == 0x848000:
            self.assertEqual((kwargs['img_paddr'], kwargs['ctx_dma'], kwargs['target'], kwargs['brom_offset']),
                             (0x23D000, 0, 1, 0x1C00))
            if scrub_succeeds: regs[0x1FA80C] = 0x28
          else: self.assertFalse(regs[0x1FA80C] & 0x10, "ACR started before VRAM initialization completed")
          return (0, 0)

        flcn.execute_hs.side_effect = execute_hs
        bootstrap = NVNativeBootstrap(nvdev, flcn, build_ga102_acr_package(0x200000))
        with patch.object(bootstrap, "_start_sec2"), patch.object(bootstrap, "_bootstrap_gr_falcon"):
          if initially_locked and not scrub_succeeds:
            with self.assertRaisesRegex(RuntimeError, "VPR.*locked"): bootstrap.init_hw()
          else: bootstrap.init_hw()
        expected = ([0x848000] if initially_locked else []) + ([] if initially_locked and not scrub_succeeds else [0x840000, 0x110000])
        self.assertEqual([args[0] for args, _ in flcn.execute_hs.call_args_list], expected)
        self.assertEqual(regs[0x600], 0xFFFFBFFF)
        self.assertNotIn(0x1FA80C, [args[0] for args, _ in nvdev.wreg.call_args_list])

  def test_native_sec2_starts_and_bootstraps_gr_falcons(self):
    init = bytes.fromhex("014004000002e80300000001800000008000000180000001") + bytes([0xA5]) * 40
    responses = {
      0xC0: bytes.fromhex("0714000000550000000000000200000000000000"),
      # The three bytes after msg_type are C-struct padding and are not initialized consistently by SEC2.
      0xD4: bytes.fromhex("07140000009e0000000000000300000000000000"),
    }

    class Sec2Regs:
      def __init__(self):
        self.state = {0x840008:0, 0x84001C:0x208000F0, 0x840100:0x50,
                      0x840C00:0, 0x840C04:0, 0x840C80:0, 0x840C84:0}
        self.writes, self.emem, self.emem_pointer, self.gpccs_samples = [], bytearray(0x100), 0, 0
        self.emem[0x80:0xC0] = init
      def wreg(self, address, value):
        self.writes.append((address, value))
        if address == 0x840130:
          self.state.update({0x840008:0x40, 0x840100:0x20, 0x840C00:0x01000000, 0x840C04:0x01000000,
                             0x840C80:0x010000C0, 0x840C84:0x01000080})
        elif address == 0x840AC0: self.emem_pointer = value & 0xFFFFFF
        elif address == 0x840AC4:
          self.emem[self.emem_pointer:self.emem_pointer+4] = struct.pack("<I", value)
          self.emem_pointer += 4
        elif address == 0x840C00:
          self.state[0x840C00] = self.state[0x840C04] = value
          offset = 0xC0 if value == 0x01000018 else 0xD4
          self.emem[offset:offset+len(responses[offset])] = responses[offset]
          self.state[0x840C80], self.state[0x840008] = 0x01000000 + offset + len(responses[offset]), 0x40
        elif address == 0x840C84: self.state[address] = value
        elif address == 0x840004:
          self.state[0x840008] = 0
          if self.state[0x840C00] == 0x01000030:
            self.state[0x840008], self.state[0x840100], self.gpccs_samples = 0x2000, 0, 0
        else: self.state[address] = value
      def rreg(self, address):
        if address == 0x840008 and self.state[0x840C00] == 0x01000030:
          self.gpccs_samples += 1
          if self.gpccs_samples > 1: self.state[0x840008], self.state[0x840100] = 0, 0x20
        if address == 0x840AC4:
          value, = struct.unpack_from("<I", self.emem, self.emem_pointer)
          self.emem_pointer += 4
          return value
        return self.state.get(address, 0)

    regs = Sec2Regs()
    bootstrap = object.__new__(NVNativeBootstrap)
    bootstrap.nvdev = regs
    bootstrap._start_sec2()
    bootstrap._bootstrap_gr_falcon(2)
    bootstrap._bootstrap_gr_falcon(3)
    self.assertEqual(regs.gpccs_samples, 5)
    self.assertEqual(regs.writes, [
      (0x840014, 0xFFFFFFFF), (0x840130, 2), (0x840AC0, 0x02000080),
      (0x840C84, 0x010000C0), (0x840004, 0x40),
      (0x840AC0, 0x01000000), (0x840AC4, 0x00031807), (0x840AC4, 0), (0x840AC4, 0),
      (0x840AC4, 2), (0x840AC4, 0), (0x840AC4, 0), (0x840C00, 0x01000018),
      (0x840AC0, 0x020000C0), (0x840AC0, 0x020000C4), (0x840C84, 0x010000D4), (0x840004, 0x40),
      (0x840AC0, 0x01000018), (0x840AC4, 0x00031807), (0x840AC4, 0), (0x840AC4, 0),
      (0x840AC4, 3), (0x840AC4, 0), (0x840AC4, 0), (0x840C00, 0x01000030),
      (0x840AC0, 0x020000D4), (0x840AC0, 0x020000D8), (0x840C84, 0x010000E8), (0x840004, 0x40),
    ])

  def test_native_gr_initializes_ltc_zbc_tables(self):
    class Regs:
      def __init__(self): self.selector, self.writes, self.selector_reads = 0, [], 0
      def rreg(self, address):
        self.assert_selector(address)
        self.selector_reads += 1
        return self.selector
      def wreg(self, address, value):
        self.writes.append((address, value))
        if address == NVNativeGR.LTC_ZBC_SELECTOR: self.selector = value
      @staticmethod
      def assert_selector(address):
        if address != NVNativeGR.LTC_ZBC_SELECTOR: raise AssertionError(f"unexpected read {address:#x}")

    regs = Regs()
    NVNativeGR(regs, b"")._init_ltc_zbc()
    self.assertEqual((len(regs.writes), regs.selector_reads, regs.selector), (210, 62, 0x1F))
    self.assertEqual(hashlib.sha256(b"".join(struct.pack("<II", *item) for item in regs.writes)).hexdigest(),
                     "fbd0d1b56f79b3fc945395245365639081b3395b41359b0a75bbcd381132673d")
    self.assertEqual(regs.writes[:5], [(0x17E338, 1), (0x17E33C, 0), (0x17E340, 0), (0x17E344, 0), (0x17E348, 0)])
    self.assertEqual(regs.writes[-4:], [(0x17E338, 0x1F), (0x17E34C, 0), (0x17E338, 0x1F), (0x17E204, 0)])

  def test_native_gr_applies_ga102_net_noncontext_init(self):
    package = build_ga102_acr_package(0x200000)
    class Regs:
      def __init__(self):
        self.regs = {0x400500:0, 0x100C80:0x08018001, 0x100CC4:0, 0x100CC8:0, 0x100CCC:0,
                     0x400700:0x014C0001, 0x000200:0x40000000, 0x40060C:1}
        self.writes = []
      def rreg(self, address): return self.regs.get(address, 0)
      def wreg(self, address, value):
        self.writes.append((address, value))
        self.regs[address] = value

    regs = Regs()
    NVNativeGR(regs, package.gr_net)._init_net()
    self.assertEqual(len(regs.writes), 610)
    self.assertEqual(hashlib.sha256(b"".join(struct.pack("<II", *item) for item in regs.writes)).hexdigest(),
                     "7e4712e8471ceec820fcff52792dc6e2702b9e5c9f1fc168b9f1543a8781411a")
    self.assertEqual(regs.writes[:6], [(0x400500, 0), (0x418880, 0x08000001), (0x418894, 0),
                                      (0x4188B4, 0), (0x4188B8, 0), (0x4188B0, 0)])

  def test_native_gr_parses_exact_ga102_context_net_regions(self):
    package = build_ga102_acr_package(0x200000)

    regions = NVNativeGR(Mock(), package.gr_net)._parse_context_net()

    self.assertEqual({region:len(entries) for region, entries in regions.items()}, {4:935, 5:577, 7:1701, 28:22, 34:66})
    self.assertEqual(regions[4][:2], [(0x1000, 2), (0x6B1, 0x11)])
    self.assertEqual(regions[5][:2], [(0x404014, 0), (0x404018, 0)])
    self.assertEqual(regions[7][0], (0x0200C797, 0))
    self.assertEqual(regions[28][-1], (0x1E100, 0x02000001))
    self.assertEqual(regions[34][:2], [(0x1000, 4), (0x2020, 0)])

  def test_native_gr_context_write_plans_match_ga102(self):
    topology = GA102Topology((5, 6, 6, 6, 6, 6, 6), ((1, 10, 20),) + ((9, 18, 36),) * 6, (7,) * 7)
    gr = NVNativeGR(Mock(), b"")
    gr.context_buffers = {name:GA102ContextBuffer(size, paddr, VirtMapping(va, mapped, [(paddr, mapped)], AddrSpace.PHYS))
                          for name, size, paddr, va, mapped in (
      ("pagepool", 0x020000, 0x280000, 0x1000000000, 0x020000),
      ("bundle",   0x003000, 0x2A0000, 0x1000020000, 0x003000),
      ("attrib",   0x829200, 0x2A3000, 0x1000800000, 0x82A000),
      ("unknown",  0x080000, 0xACD000, 0x1000080000, 0x080000),
    )}

    patch_writes, floorsweep_writes = gr._context_patch_writes(topology), gr._floorsweep_writes(topology)

    self.assertEqual((len(patch_writes), gr._stream_digest(patch_writes)),
                     (149, "c9a8a16185c9ff5298162c7ba569820c9274bd0fcbcb9038f2b1073751717b97"))
    self.assertEqual((len(floorsweep_writes), gr._stream_digest(floorsweep_writes)),
                     (75, "9dbd5d223931365f48e7af0e240d5030acf9c896252ff6b7d9547653e3ecf5d8"))

  def test_native_gr_context_icmd_and_method_streams_match_ga102(self):
    package = build_ga102_acr_package(0x200000)
    class Regs:
      def __init__(self): self.writes = []
      def rreg(self, address): return 0
      def wreg(self, address, value): self.writes.append((address, value))

    regs, gr = Regs(), NVNativeGR(None, package.gr_net)
    gr.nvdev = regs
    regions = gr._parse_context_net()
    expected = {
      4:(1175, "996188246e6d858638b2fe2daf83d46d60c8d1cbb22ef3087c6ea18c5d2dd9b3"),
      28:(1356, "ddfff2107e5dc279cf53c384023d97ce9e1a53c40a3e8298828b8ef5c4c2009a"),
      34:(74, "f0ca70906b70b9972240558cc7066e919683ffd2fab93b54ef3623966db040c1"),
    }
    for region_id, kwargs in ((4, {}), (28, {"count":64, "pitch":0x100000}), (34, {"wide":True})):
      regs.writes = []
      gr._apply_icmd(regions[region_id], **kwargs)
      self.assertEqual((len(regs.writes), gr._stream_digest(regs.writes)), expected[region_id])
    regs.writes = []
    gr._apply_methods(regions[7])
    self.assertEqual((len(regs.writes), gr._stream_digest(regs.writes)),
                     (1971, "d27ffc7bb01552e860f66744aed6ebfefc83c5731598fa27d30e5e15a888d4d5"))

  def test_native_gr_applies_complete_ga102_golden_init_stream(self):
    package = build_ga102_acr_package(0x200000)
    class Regs:
      def __init__(self):
        self.regs = {0x400700:0, 0x200:0x40000000, 0x40060C:0, 0x419BD8:0x33F, 0x504728:0x0481EB60,
                     0x404154:0x7FFFFFFF, 0x41980C:0x10, 0x41BE08:0x24, 0x400088:0xFE06BFE7, 0x409604:7}
        for gpc in range(7):
          base = 0x500000 + gpc * 0x8000
          self.regs[base + 0x2608] = 5 if gpc == 0 else 6
          for ppc, mask in enumerate((1, 10, 20) if gpc == 0 else (9, 18, 36)):
            self.regs[base + 0x0C30 + ppc * 4] = mask
          self.regs[base + 0x0C50] = 7
        self.writes = []
      def rreg(self, address): return self.regs.get(address, 0)
      def wreg(self, address, value):
        self.writes.append((address, value))
        self.regs[address] = value

    regs, gr = Regs(), NVNativeGR(None, package.gr_net)
    gr.nvdev = regs
    gr.context_buffers = {name:GA102ContextBuffer(size, paddr, VirtMapping(va, mapped, [(paddr, mapped)], AddrSpace.PHYS))
                          for name, size, paddr, va, mapped in (
      ("pagepool", 0x020000, 0x280000, 0x1000000000, 0x020000),
      ("bundle",   0x003000, 0x2A0000, 0x1000020000, 0x003000),
      ("attrib",   0x829200, 0x2A3000, 0x1000800000, 0x82A000),
      ("unknown",  0x080000, 0xACD000, 0x1000080000, 0x080000),
    )}

    gr._generate_main()

    self.assertEqual((len(regs.writes), gr._stream_digest(regs.writes)),
                     (5385, "71a25ee90b1dfc1fed7820e261231f99fcbfef3567e3a16ec8bce08ffd92a899"))

  def test_native_gr_golden_context_uses_fecs_and_releases_scratch(self):
    class VRAM:
      def __init__(self): self.data = bytearray(0x400000)
      def __getitem__(self, key): return memoryview(self.data)[key]
      def __setitem__(self, key, value): self.data[key] = value
    class Native:
      def __init__(self):
        self.regs = {0x200:0x40000000, 0x40060C:0, 0x400700:0, 0x409100:0x60, 0x41A100:0x60,
                     0x409614:0x110, 0x409800:0x12500, 0x409B00:0}
        self.writes, self.vram, self.mm = [], VRAM(), Mock()
      def rreg(self, address): return self.regs.get(address, 0)
      def wreg(self, address, value):
        self.writes.append((address, value))
        self.regs[address] = 0 if address == 0x404170 else value
        if address == 0x409504 and value == 3: self.regs[0x409800] |= 0x10
        if address == 0x409504 and value == 9: self.regs[0x409800] = self.regs[0x409800] & ~3 | 1
      def _write_vram(self, paddr, data): self.vram[paddr:paddr + len(data)] = data

    native, mapping = Native(), VirtMapping(0x1000000000, 0x81000, [(0x200000, 0x81000)], AddrSpace.PHYS)
    native.mm.palloc.return_value = 0x200000
    native.mm.alloc_vaddr.return_value = mapping.va_addr
    native.mm.map_range.return_value = mapping
    gr = NVNativeGR(native, b"")
    gr.main_size = 0x1000
    golden = bytes(range(256)) * 16
    with patch.object(gr, "_generate_main", side_effect=lambda: native._write_vram(0x280000, golden)) as generate_main:
      self.assertEqual(gr.generate_golden(SimpleNamespace(instance_paddr=0x100000)), golden)

    generate_main.assert_called_once_with()
    native.mm.palloc.assert_called_once_with(0x81000, align=1, zero=True)
    native.mm.map_range.assert_called_once_with(mapping.va_addr, 0x81000, [(0x200000, 0x81000)], AddrSpace.PHYS, kind=0)
    native.mm.vfree.assert_called_once_with(mapping)
    self.assertEqual(bytes(native.vram[0x100210:0x100218]), bytes(8))
    self.assertIn((0x409504, 3), native.writes)
    self.assertIn((0x409504, 9), native.writes)

  def test_native_gr_binds_compute_context_from_golden_image(self):
    class VRAM:
      def __init__(self): self.data = bytearray(0x500000)
      def __getitem__(self, key): return memoryview(self.data)[key]
      def __setitem__(self, key, value): self.data[key] = value
    class Native:
      def __init__(self): self.mm, self.vram, self.regs, self.writes = Mock(), VRAM(), {}, []
      def _write_vram(self, paddr, data): self.vram[paddr:paddr + len(data)] = data
      def rreg(self, address): return self.regs.get(address, 0)
      def wreg(self, address, value):
        self.writes.append((address, value))
        self.regs[address] = 0 if address == 0x70000 else value

    native = Native()
    native.mm.palloc.side_effect = (0x200000, 0x201000)
    native.mm.alloc_vaddr.side_effect = (0x1000000000, 0x1000200000)
    mappings = (VirtMapping(0x1000000000, 0x1000, [(0x200000, 0x1000)], AddrSpace.PHYS, privileged=True, kind=0),
                VirtMapping(0x1000200000, 0x2000, [(0x201000, 0x2000)], AddrSpace.PHYS, privileged=True, kind=0))
    native.mm.map_range.side_effect = mappings
    channel = NVNativeChannel(chid=3, instance_paddr=0x100000, runlist=0xC00000,
                              channel_table=0xC20000, doorbell=0, nonstall_vector=0xA0)
    struct.pack_into("<I", native.vram.data, channel.instance_paddr + 0xAC, 0x20)
    golden, patches, topology = bytes([0xA5]) * 0x1556, [(0x40800C, 0x1234), (0x419004, 0x5678)], Mock()
    gr = NVNativeGR(native, b"")
    gr.main_size = len(golden)

    with patch.object(gr, "generate_golden", return_value=golden) as generate, \
         patch.object(gr, "_read_topology", return_value=topology) as read_topology, \
         patch.object(gr, "_context_patch_writes", return_value=patches) as context_patch_writes:
      gr.bind_compute_context(channel)

    generate.assert_called_once_with(channel)
    read_topology.assert_called_once_with()
    context_patch_writes.assert_called_once_with(topology)
    self.assertEqual(native.mm.palloc.call_args_list, [call(0x1000, align=0x100, zero=False),
                                                       call(0x2000, align=0x1000, zero=False)])
    self.assertEqual(native.mm.map_range.call_args_list, [
      call(0x1000000000, 0x1000, [(0x200000, 0x1000)], AddrSpace.PHYS, privileged=True, kind=0),
      call(0x1000200000, 0x2000, [(0x201000, 0x2000)], AddrSpace.PHYS, privileged=True, kind=0),
    ])
    self.assertEqual(bytes(native.vram[0x200000:0x200010]), struct.pack("<4I", 0x40800C, 0x1234, 0x419004, 0x5678))
    expected = bytearray(golden)
    struct.pack_into("<IIIIIIII", expected, 0x10, len(patches), 0, 0x10, 1, 0, 0xA5A5A5A5, 0, 0)
    struct.pack_into("<II", expected, 0xF4, 0, 0)
    self.assertEqual(bytes(native.vram[0x201000:0x201000 + len(golden)]), expected)
    self.assertEqual(bytes(native.vram[0x100210:0x100218]), struct.pack("<Q", 0x1000200004))
    self.assertEqual(struct.unpack_from("<I", native.vram.data, 0x1000AC)[0], 0x10020)
    self.assertEqual(native.writes, [(0x70000, 1)])
    self.assertEqual(gr.channel_contexts, {channel.chid:mappings})

  def test_native_gr_shares_compute_context_with_channel_group(self):
    class VRAM:
      def __init__(self): self.data = bytearray(0x4000)
      def __getitem__(self, key): return memoryview(self.data)[key]
      def __setitem__(self, key, value): self.data[key] = value

    native = SimpleNamespace(vram=VRAM(), regs={0x70000:0}, writes=[])
    native._write_vram = lambda paddr, data: native.vram.__setitem__(slice(paddr, paddr + len(data)), data)
    native.rreg = lambda address: native.regs.get(address, 0)
    def wreg(address, value):
      native.writes.append((address, value))
      native.regs[address] = 0 if address == 0x70000 else value
    native.wreg = wreg
    compute = NVNativeChannel(chid=0, instance_paddr=0x1000, runlist=0xC00000,
                              channel_table=0xC20000, doorbell=0, nonstall_vector=0xA0)
    copy = NVNativeChannel(chid=1, instance_paddr=0x2000, runlist=0xC00000,
                           channel_table=0xC20000, doorbell=0, nonstall_vector=0xA0)
    struct.pack_into("<I", native.vram.data, compute.instance_paddr + 0xAC, 0x10020)
    struct.pack_into("<Q", native.vram.data, compute.instance_paddr + 0x210, 0x1000500004)
    struct.pack_into("<I", native.vram.data, copy.instance_paddr + 0xAC, 0x20020)
    gr = NVNativeGR(native, b"")
    gr.channel_contexts[compute.chid] = (Mock(), Mock())

    gr.bind_channel_group_context([compute, copy])

    self.assertEqual(bytes(native.vram[copy.instance_paddr + 0x210:copy.instance_paddr + 0x218]), struct.pack("<Q", 0x1000500004))
    self.assertEqual(struct.unpack_from("<I", native.vram.data, copy.instance_paddr + 0xAC)[0], 0x30020)
    self.assertEqual(native.writes, [(0x70000, 1)])

  def test_native_dev_binds_copy_context_through_bar2(self):
    class VRAM:
      def __init__(self): self.data = bytearray(0x500000)
      def __getitem__(self, key): return memoryview(self.data)[key]
      def __setitem__(self, key, value): self.data[key] = value

    native = object.__new__(NVNativeDev)
    native.mm, native.pci_dev, native.vram = Mock(), Mock(), VRAM()
    native.mm.root_page_table.paddr = 0x700000
    native.mm.palloc.side_effect = (0x200000, 0x203000)
    context = VirtMapping(0x1000, 0x3000, [(0x200000, 0x3000)], AddrSpace.PHYS, kind=0)
    native.mm.map_range.return_value = context
    native.pci_dev.bar_info.return_value = (0x1000000000, 0x2000000)
    native._native_bar2_instance_paddr, native._native_bar2_next = None, 0x1000
    native._native_copy_contexts = {}
    native.regs, native.writes = {0x104028:0x3F, 0xB80F48:0x40000000, 0xB80F50:0, 0x70000:0}, []
    native.rreg = lambda address: native.regs.get(address, 0)
    def wreg(address, value):
      native.writes.append((address, value))
      native.regs[address] = 0 if address == 0x70000 else value
    native.wreg = wreg
    channel = NVNativeChannel(chid=3, instance_paddr=0x100000, runlist=0xC00000,
                              channel_table=0xC20000, doorbell=0, nonstall_vector=0xA0)
    struct.pack_into("<I", native.vram.data, channel.instance_paddr + 0xAC, 0x20)

    native.bind_copy_context(channel)

    self.assertEqual(native.mm.palloc.call_args_list, [call(0x3000, align=0x1000, zero=True),
                                                       call(0x1000, align=0x1000, zero=False)])
    native.mm.map_range.assert_called_once_with(0x1000, 0x3000, [(0x200000, 0x3000)], AddrSpace.PHYS, kind=0)
    self.assertEqual(bytes(native.vram[0x203200:0x203210]), struct.pack("<QQ", 0x700C00, 0x1FFFFFF))
    self.assertEqual(bytes(native.vram[0x203298:0x2032A0]), struct.pack("<Q", 1))
    self.assertEqual(bytes(native.vram[0x2032A0:0x2032B0]), struct.pack("<III", 0x700C00, 0, 0) + bytes(4))
    self.assertEqual(bytes(native.vram[0x100220:0x100228]), struct.pack("<Q", 0x1000))
    self.assertEqual(struct.unpack_from("<I", native.vram.data, 0x1000AC)[0], 0x20020)
    self.assertEqual(native.writes, [(0xB80F48, 0x80000203), (0x70000, 1)])
    self.assertEqual(native._native_copy_contexts, {channel.chid:context})
    self.assertEqual(native._native_bar2_instance_paddr, 0x203000)
    self.assertEqual(native._native_bar2_next, 0x4000)

  def test_native_gr_programs_ga102_topology(self):
    class Regs:
      def __init__(self):
        self.regs = {0x400500:0, 0x000200:0x40000000, 0x40060C:1, 0x409604:7, 0x100800:12}
        for gpc in range(7):
          base = 0x500000 + gpc * 0x8000
          self.regs[base + 0x2608] = 5 if gpc == 0 else 6
          for ppc, mask in enumerate((1, 10, 20) if gpc == 0 else (9, 18, 36)): self.regs[base + 0x0C30 + ppc * 4] = mask
          self.regs[base + 0x0C50] = 7
        self.writes = []
      def rreg(self, address): return self.regs.get(address, 0)
      def wreg(self, address, value):
        self.writes.append((address, value))
        self.regs[address] = value | 0x40000000 if address == 0x4188AC or \
          any(address == 0x500910 + gpc * 0x8000 for gpc in range(7)) else value

    regs = Regs()
    NVNativeGR(regs, b"")._init_topology()
    self.assertEqual(len(regs.writes), 116)
    self.assertEqual(hashlib.sha256(b"".join(struct.pack("<II", *item) for item in regs.writes)).hexdigest(),
                     "91043022cf4b12cb117d2588612de4409294c8d698bc590e20ec7efab347f2d0")
    self.assertEqual(regs.writes[:2], [(0x503018, 1), (0x418980, 0x10000000)])
    self.assertEqual([regs.regs[0x500910 + gpc * 0x8000] for gpc in range(7)], [0x40040029] * 7)
    self.assertEqual(regs.regs[0x4188AC], 0x4000000C)

  def test_native_gr_initializes_exceptions_and_zbc(self):
    class Regs:
      def __init__(self):
        self.regs = {0x400500:0, 0x000200:0x40000000, 0x40060C:1, 0x409604:7, 0x100800:12,
                     0x40584C:0x7F, 0x502C94:0, 0x17E338:0x1F, 0x41BCB4:1,
                     0x41814C:0, 0x418150:0, 0x418154:0, 0x418158:0,
                     0x418198:0, 0x41819C:0, 0x4181A0:0, 0x4181A4:0, 0x4188A4:0}
        for gpc in range(7):
          base = 0x500000 + gpc * 0x8000
          self.regs[base + 0x2608] = 5 if gpc == 0 else 6
          for ppc, mask in enumerate((1, 10, 20) if gpc == 0 else (9, 18, 36)): self.regs[base + 0x0C30 + ppc * 4] = mask
          self.regs[base + 0x0C50] = 7
        self.writes = []
      def rreg(self, address): return self.regs.get(address, 0)
      def wreg(self, address, value):
        self.writes.append((address, value))
        if address != 0x502C94: self.regs[address] = value

    regs = Regs()
    NVNativeGR(regs, b"")._init_exceptions_zbc()
    self.assertEqual(len(regs.writes), 630)
    self.assertEqual(hashlib.sha256(b"".join(struct.pack("<II", *item) for item in regs.writes)).hexdigest(),
                     "e0f3e51b1bff22cb4c29ac8dc5f85fc2c7bcc1d887f29b7dd808f7678c6129d5")
    self.assertEqual(regs.writes[:4], [(0x400500, 0x00010001), (0x400100, 0xFFFFFFFF),
                                      (0x40013C, 0xFFFFFFFF), (0x400124, 2)])
    self.assertEqual((regs.regs[0x41BCB4], regs.regs[0x17E338], regs.regs[0x4188A4]), (30, 3, 0x03000000))

  def test_native_gr_starts_context_control_and_discovers_sizes(self):
    class Regs:
      def __init__(self):
        self.regs = {0x400500:0x00010001, 0x400700:0x014C0901, 0x40060C:1,
                     0x409008:0, 0x409100:0x50, 0x40910C:0, 0x409130:0, 0x409500:0x80000E58,
                     0x409504:0xBADF5545, 0x409800:0, 0x409804:0, 0x409C18:0,
                     0x41A008:0, 0x41A100:0x50, 0x41A10C:0, 0x41A130:0, 0x41A800:0, 0x41A804:0}
        self.writes = []
      def rreg(self, address): return self.regs.get(address, 0)
      def wreg(self, address, value):
        self.writes.append((address, value))
        self.regs[address] = value
        if (address, value) == (0x41A130, 2): self.regs[0x41A100], self.regs[0x41A130] = 0x60, 0
        if (address, value) == (0x409130, 2):
          self.regs[0x409100], self.regs[0x409800], self.regs[0x409130] = 0x60, 1, 0
          self.regs[0x400700], self.regs[0x40060C] = 0, 0
        if address == 0x409504:
          self.regs[0x409504] = 0xBADF5545
          if value in (0x10, 0x16, 0x25): self.regs[0x409800] = {0x10:0x155600, 0x16:0x118E00, 0x25:0x12500}[value]

    regs = Regs()
    gr = NVNativeGR(regs, b"")
    gr._init_ctxctl()
    self.assertEqual((gr.main_size, gr.zcull_size, gr.pm_size), (0x155600, 0x118E00, 0x12500))
    self.assertEqual(regs.writes, list(NVNativeGR.CTXCTL_WRITES))
    self.assertEqual(hashlib.sha256(b"".join(struct.pack("<II", *item) for item in regs.writes)).hexdigest(),
                     "224bdca4eb39da6d5a2e3ea2dcce1a37dc47cccb069f7436b16f9b66ea793ee5")

  def test_native_gr_maps_shared_context_buffers_as_privileged_pitch_memory(self):
    mm = Mock()
    mm.palloc.side_effect = (0x300000, 0x320000, 0x323000, 0xB55000)
    mm.alloc_vaddr.side_effect = (0x1000000000, 0x1000020000, 0x1000023000, 0x100084D000)
    mappings = tuple(Mock() for _ in range(4))
    mm.map_range.side_effect = mappings
    gr = NVNativeGR(SimpleNamespace(mm=mm), b"")

    gr._alloc_ctx_buffers()

    self.assertEqual(mm.mock_calls, [
      call.palloc(0x020000, align=0x0100, zero=False),
      call.palloc(0x003000, align=0x0100, zero=False),
      call.palloc(0x829200, align=0x1000, zero=False),
      call.palloc(0x080000, align=0x0100, zero=False),
      call.alloc_vaddr(0x020000, 0x1000),
      call.map_range(0x1000000000, 0x020000, [(0x300000, 0x020000)], AddrSpace.PHYS, privileged=True, kind=0),
      call.alloc_vaddr(0x003000, 0x1000),
      call.map_range(0x1000020000, 0x003000, [(0x320000, 0x003000)], AddrSpace.PHYS, privileged=True, kind=0),
      call.alloc_vaddr(0x82A000, 0x1000),
      call.map_range(0x1000023000, 0x82A000, [(0x323000, 0x82A000)], AddrSpace.PHYS, privileged=True, kind=0),
      call.alloc_vaddr(0x080000, 0x1000),
      call.map_range(0x100084D000, 0x080000, [(0xB55000, 0x080000)], AddrSpace.PHYS, privileged=True, kind=0),
    ])
    self.assertEqual(tuple(gr.context_buffers), ("pagepool", "bundle", "attrib", "unknown"))
    self.assertEqual(tuple((buffer.requested_size, buffer.paddr, buffer.mapping) for buffer in gr.context_buffers.values()), (
      (0x020000, 0x300000, mappings[0]), (0x003000, 0x320000, mappings[1]),
      (0x829200, 0x323000, mappings[2]), (0x080000, 0xB55000, mappings[3]),
    ))

  def test_native_device_enables_bus_mastering_after_secure_bootstrap(self):
    pci_dev, flcn, package, bootstrap, gr = Mock(), object.__new__(NV_FLCN), Mock(), Mock(), Mock()
    events = []
    pci_dev.pcibus, pci_dev.read_config.side_effect = "usb:4-3", (0x407, 0x403, 0x403, 0x407)
    pci_dev.write_config_flush.side_effect = lambda offset, value, size: events.append(("pci_command", offset, value, size))
    bootstrap.init_hw.side_effect = lambda: events.append(("bootstrap",))
    gr.init_hw.side_effect = lambda: events.append(("gr",))

    def early_ip_init(native): native.flcn, native.fw_name = flcn, "ga102"

    with patch.object(NVNativeDev, "_early_ip_init", autospec=True, side_effect=early_ip_init) as early_ip, \
         patch.object(NVNativeDev, "_early_mmu_init", autospec=True) as early_mmu, \
         patch("tinygrad.runtime.support.nv.nvdev.build_ga102_acr_package", return_value=package) as build, \
         patch("tinygrad.runtime.support.nv.nvdev.NVNativeBootstrap", return_value=bootstrap), \
         patch("tinygrad.runtime.support.nv.nvdev.NVNativeGR", return_value=gr):
      native = NVNativeDev(pci_dev)

    pci_dev.map_bar.assert_called_once_with(0, fmt='I')
    early_ip.assert_called_once_with(native)
    early_mmu.assert_called_once_with(native)
    self.assertEqual(events, [("pci_command", 4, 0x403, 2), ("bootstrap",),
                              ("pci_command", 4, 0x407, 2), ("gr",)])
    build.assert_called_once_with(0x200000)
    bootstrap.init_hw.assert_called_once_with()
    gr.init_hw.assert_called_once_with()
    self.assertFalse(native.is_booting)

    native.fini()
    pci_dev.reset.assert_called_once_with()


if __name__ == "__main__":
  unittest.main()
