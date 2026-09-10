from __future__ import annotations

import hashlib, struct
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, call, patch

from tinygrad.runtime.support.nv.gr import GA102ContextBuffer, GA102Topology, NVNativeGR
from tinygrad.runtime.support.memory import AddrSpace, VirtMapping
from tinygrad.runtime.support.nv.ip import NV_FLCN
from tinygrad.runtime.support.nv.nogsp import WPR_ALLOCATION_BYTES, build_ga102_acr_package


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


if __name__ == "__main__":
  unittest.main()
