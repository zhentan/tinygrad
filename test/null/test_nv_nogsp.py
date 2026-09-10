from __future__ import annotations

import hashlib
import unittest
from unittest.mock import MagicMock, Mock, call, patch

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


if __name__ == "__main__":
  unittest.main()
