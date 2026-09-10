from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

from tinygrad.runtime.autogen.nv_regs import dev_mmu, dev_vm
from tinygrad.runtime.support.memory import AddrSpace, MemoryManager
from tinygrad.runtime.support.nv.nvdev import NVMemoryManager, NVPageTableEntry, NVReg


class FakeVRAM:
  def __init__(self): self.pages:dict[int, bytearray] = {}
  def view(self, paddr:int, size:int, fmt='B'):
    return memoryview(self.pages.setdefault(paddr, bytearray(size))).cast(fmt)
  def __setitem__(self, key:slice, value:bytes): self.view(key.start, key.stop - key.start)[:] = value


class FakeNVDev:
  def __init__(self, mmu_ver=2, root_paddr=0):
    self.mmu_ver, self.vram = mmu_ver, FakeVRAM()
    self.mm = SimpleNamespace(level_cnt=5, root_page_table=SimpleNamespace(paddr=root_paddr))
    regs = dev_mmu.gh100 if mmu_ver == 3 else dev_mmu.tu102
    self.pte_t = NVReg(self, *regs[f'NV_MMU_VER{mmu_ver}_PTE'])
    self.pde_t = NVReg(self, *regs[f'NV_MMU_VER{mmu_ver}_PDE'])
    self.dual_pde_t = NVReg(self, *regs[f'NV_MMU_VER{mmu_ver}_DUAL_PDE'])
    self.writes:list[tuple[int, int]] = []
    self.invalidate_reads:list[int] = []
    self.invalidate_default = 0
    for name in ('NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE_PDB', 'NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE_UPPER_PDB',
                 'NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE'):
      setattr(self, name, NVReg(self, *dev_vm.tu102[name]))

  def wreg(self, addr:int, value:int): self.writes.append((addr, value))
  def rreg(self, addr:int) -> int:
    if addr == 0xB830B0: return self.invalidate_reads.pop(0) if self.invalidate_reads else self.invalidate_default
    return 0


class TestNVPageTableEntry(unittest.TestCase):
  def test_default_pte_encoding_is_preserved(self):
    dev = FakeNVDev()
    pte = NVPageTableEntry(dev, 0xB51000, lv=4)
    pte.set_entry(0, 0x280000, aspace=AddrSpace.PHYS)
    self.assertEqual(pte.entry(0), 0x0600000000028001)

  def test_pitch_kind_pte_encoding(self):
    for version, expected in ((2, 0x28001), (3, 0x280001)):
      with self.subTest(mmu_ver=version):
        pte = NVPageTableEntry(FakeNVDev(mmu_ver=version), 0xB51000, lv=4)
        pte.set_entry(0, 0x280000, kind=0)
        self.assertEqual(pte.entry(0), expected)

  def test_ga102_privileged_pitch_pte_matches_nouveau(self):
    dev = FakeNVDev()
    pte = NVPageTableEntry(dev, 0xB51000, lv=4)
    pte.set_entry(0, 0x280000, kind=0, privileged=True)
    self.assertEqual(pte.entry(0), 0x28021)

  def test_privileged_pcf_is_encoded_on_mmu_ver3(self):
    dev = FakeNVDev(mmu_ver=3)
    pte = NVPageTableEntry(dev, 0xB51000, lv=4)
    pte.set_entry(0, 0x280000, kind=0, privileged=True)
    self.assertEqual(pte.entry(0), 0x280011)

  def test_ga102_vram_pdes_match_nouveau(self):
    dev = FakeNVDev()
    pde = NVPageTableEntry(dev, 0xB4E000, lv=1)
    pde.set_entry(0, 0xB4F000, table=True)
    self.assertEqual(pde.entry(0), 0xB4F02)

    dual = NVPageTableEntry(dev, 0xB50000, lv=3)
    dual.set_entry(0, 0xB51000, table=True)
    self.assertEqual(dual.entry(0), 0xB5102 << 64)

  def test_map_range_preserves_privileged_mapping(self):
    dev = FakeNVDev()
    dev.is_booting, dev.smi_dev, dev.devfmt = True, False, "test"
    mm = MemoryManager(dev, 16 << 20, boot_size=2 << 20, pt_t=NVPageTableEntry,
                       va_bits=48, va_shifts=[12, 21, 29, 38, 47], va_base=0,
                       palloc_ranges=[(4 << 10, 4 << 10)])
    dev.mm, dev.is_booting = mm, False

    mapping = mm.map_range(0x1000000000, 0x1000, [(0x280000, 0x1000)], AddrSpace.PHYS, privileged=True, kind=0)

    leaf = mm.page_tables(mapping.va_addr, mapping.size)[-1]
    entry = (mapping.va_addr >> 12) % mm.pte_cnt[-1]
    self.assertTrue(mapping.privileged)
    self.assertEqual((mapping.kind, leaf.entry(entry)), (0, 0x28021))


class TestNVMMUInvalidate(unittest.TestCase):
  @staticmethod
  def manager(dev:FakeNVDev) -> NVMemoryManager:
    mm = object.__new__(NVMemoryManager)
    mm.dev, mm.root_page_table = dev, dev.mm.root_page_table
    return mm

  def test_tu102_invalidate_selects_root(self):
    dev = FakeNVDev(root_paddr=0x12345000)

    self.manager(dev).on_range_mapped()

    self.assertEqual(dev.writes, [
      (0xB830A0, 0x123450),
      (0xB830A4, 0),
      (0xB830B0, 0x80000001),
    ])

  def test_mmu_ver3_invalidate_encoding_is_preserved(self):
    dev = FakeNVDev(mmu_ver=3)

    self.manager(dev).on_range_mapped()

    self.assertEqual(dev.writes, [(0xB830B0, 0x80000043)])

  def test_tu102_invalidate_waits_for_completion(self):
    dev = FakeNVDev()
    dev.invalidate_reads = [0x80000001, 0]

    self.manager(dev).on_range_mapped()

    self.assertFalse(dev.invalidate_reads)

  def test_tu102_invalidate_times_out_if_trigger_stays_set(self):
    dev = FakeNVDev()
    dev.invalidate_default = 0x80000001

    with mock.patch('tinygrad.helpers.time.perf_counter', side_effect=[0.0, 0.0, 2.001]), \
         self.assertRaisesRegex(TimeoutError, 'MMU invalidate did not complete'):
      self.manager(dev).on_range_mapped()

if __name__ == '__main__':
  unittest.main()
