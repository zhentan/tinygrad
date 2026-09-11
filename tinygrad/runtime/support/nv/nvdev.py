from __future__ import annotations
import time, functools, hashlib, struct, tinygrad.runtime.autogen.nv_regs
from dataclasses import dataclass
from tinygrad.helpers import getenv, DEBUG, getbits, round_up, wait_cond
from tinygrad.runtime.autogen import pci
from tinygrad.runtime.support.memory import TLSFAllocator, MemoryManager, AddrSpace, VirtMapping
from tinygrad.runtime.support.nv.ip import NV_FLCN, NV_FLCN_COT, NV_GSP
from tinygrad.runtime.support.nv.gr import NVNativeGR
from tinygrad.runtime.support.nv.nogsp import GA102ACRPackage, NativeHelper, WPR_ALLOCATION_BYTES, build_ga102_acr_package
from tinygrad.runtime.support.system import PCIDevice
from tinygrad.runtime.support.hcq import MMIOInterface

NV_DEBUG = getenv("NV_DEBUG", 0)


class NVNativeBootstrap:
  SEC2_IRQSCLR, SEC2_IRQSTAT, SEC2_IRQMCLR, SEC2_IRQDEST = 0x840004, 0x840008, 0x840014, 0x84001C
  SEC2_CPUCTL, SEC2_CPUCTL_ALIAS = 0x840100, 0x840130
  SEC2_EMEMC0, SEC2_EMEMD0 = 0x840AC0, 0x840AC4
  SEC2_CMDQ_HEAD, SEC2_CMDQ_TAIL, SEC2_MSGQ_HEAD, SEC2_MSGQ_TAIL = 0x840C00, 0x840C04, 0x840C80, 0x840C84
  SEC2_INIT_MESSAGE = bytes.fromhex("014004000002e80300000001800000008000000180000001") + bytes([0xA5]) * 40

  def __init__(self, nvdev, flcn:NV_FLCN, package:GA102ACRPackage):
    self.nvdev, self.flcn, self.package = nvdev, flcn, package
    self.flcn.init_regs()
    self.flcn.falcon, self.flcn.sec2 = 0x110000, 0x840000

  def _set_bme(self, enabled:bool):
    command = self.nvdev.pci_dev.read_config(pci.PCI_COMMAND, 2)
    requested = command | pci.PCI_COMMAND_MASTER if enabled else command & ~pci.PCI_COMMAND_MASTER
    self.nvdev.pci_dev.write_config_flush(pci.PCI_COMMAND, requested, 2)
    actual = self.nvdev.pci_dev.read_config(pci.PCI_COMMAND, 2)
    if actual != requested: raise RuntimeError(f"PCI command did not become {requested:#x}: {actual:#x}")

  def _require_wpr(self):
    raw_start, raw_limit = self.nvdev.rreg(0x1FA81C), self.nvdev.rreg(0x1FA820)
    expected = self.package.protected_start >> 8, (self.package.protected_end - 0x20000) >> 8
    if (raw_start, raw_limit) != expected: raise RuntimeError(f"native WPR range mismatch: {(raw_start, raw_limit)} != {expected}")

  def _sec2_sample(self) -> tuple[int, int, int, int, int, int, int]:
    irqstat, irqdest, cpuctl = self.nvdev.rreg(self.SEC2_IRQSTAT), self.nvdev.rreg(self.SEC2_IRQDEST), self.nvdev.rreg(self.SEC2_CPUCTL)
    effective = irqstat & irqdest & ~(irqdest >> 16) & 0xFFFFFFFF
    if irqstat & 0x10 or cpuctl & 0x10 or effective & ~0x40:
      raise RuntimeError(f"native SEC2 fault: irqstat={irqstat:#x} irqdest={irqdest:#x} cpuctl={cpuctl:#x} effective={effective:#x}")
    return effective, cpuctl, self.nvdev.rreg(self.SEC2_CMDQ_HEAD), self.nvdev.rreg(self.SEC2_CMDQ_TAIL), \
      self.nvdev.rreg(self.SEC2_MSGQ_HEAD), self.nvdev.rreg(self.SEC2_MSGQ_TAIL), irqstat

  def _wait_sec2(self, predicate, label:str, timeout_ms:int=2000) -> tuple[int, int, int, int, int, int, int]:
    deadline, sample = time.monotonic() + timeout_ms / 1000, None
    while time.monotonic() < deadline:
      sample = self._sec2_sample()
      if predicate(sample): return sample
      time.sleep(0.001)
    raise TimeoutError(f"native SEC2 {label} timed out: {sample}")

  def _wait_sec2_quiescent(self, cmd_pointer:int, msg_pointer:int, timeout_ms:int=2000):
    deadline, consecutive, sample = time.monotonic() + timeout_ms / 1000, 0, None
    while time.monotonic() < deadline:
      sample = self._sec2_sample()
      if sample == (0, 0x20, cmd_pointer, cmd_pointer, msg_pointer, msg_pointer, 0):
        consecutive += 1
        if consecutive == 4: return
      else: consecutive = 0
      time.sleep(0.002)
    raise TimeoutError(f"native SEC2 did not quiesce after GPCCS bootstrap: {sample}")

  def _read_sec2_emem(self, offset:int, size:int) -> bytes:
    if offset < 0 or offset + size > 0x1000000 or offset % 4 or size % 4:
      raise RuntimeError(f"invalid native SEC2 EMEM read: offset={offset:#x} size={size:#x}")
    self.nvdev.wreg(self.SEC2_EMEMC0, 0x02000000 | offset)
    return b"".join(struct.pack("<I", self.nvdev.rreg(self.SEC2_EMEMD0)) for _ in range(size // 4))

  def _start_sec2(self):
    if self.nvdev.rreg(self.SEC2_CPUCTL) != 0x50: raise RuntimeError("native SEC2 is not ready to start")
    self.nvdev.wreg(self.SEC2_IRQMCLR, 0xFFFFFFFF)
    self.nvdev.wreg(self.SEC2_CPUCTL_ALIAS, 2)
    sample = self._wait_sec2(lambda x: x[0] == 0x40 and x[4] == x[5] + 64, "initial message")
    _, _, cmd_head, cmd_tail, msg_head, msg_tail, _ = sample
    if (cmd_head, cmd_tail, msg_head, msg_tail) != (0x01000000, 0x01000000, 0x010000C0, 0x01000080):
      raise RuntimeError(f"native SEC2 initial queue pointers changed: {sample}")
    raw = self._read_sec2_emem(msg_tail & 0xFFFFFF, 64)
    if raw != self.SEC2_INIT_MESSAGE:
      raise RuntimeError(f"native SEC2 initial message changed: sha256={hashlib.sha256(raw).hexdigest()}")
    self.nvdev.wreg(self.SEC2_MSGQ_TAIL, msg_head)
    self.nvdev.wreg(self.SEC2_IRQSCLR, 0x40)
    self._wait_sec2(lambda x: x[0] == 0 and x[4] == x[5] == msg_head, "initial acknowledgement")
    self._sec2_cmd_head, self._sec2_msg_head = cmd_head, msg_head

  def _bootstrap_gr_falcon(self, falcon_id:int):
    if falcon_id not in (2, 3): raise RuntimeError(f"unsupported native GR falcon {falcon_id}")
    expected_cmd = 0x01000000 if falcon_id == 2 else 0x01000018
    expected_msg = 0x010000C0 if falcon_id == 2 else 0x010000D4
    if (self._sec2_cmd_head, self._sec2_msg_head) != (expected_cmd, expected_msg):
      raise RuntimeError(f"native SEC2 software queue state changed before falcon {falcon_id}")
    _, _, cmd_head, cmd_tail, msg_head, msg_tail, _ = self._sec2_sample()
    if (cmd_head, cmd_tail, msg_head, msg_tail) != (expected_cmd, expected_cmd, expected_msg, expected_msg):
      raise RuntimeError(f"native SEC2 hardware queue state changed before falcon {falcon_id}")

    command = struct.pack("<6I", 0x00031807, 0, 0, falcon_id, 0, 0)
    self.nvdev.wreg(self.SEC2_EMEMC0, 0x01000000 | (expected_cmd & 0xFFFFFF))
    for word, in struct.iter_unpack("<I", command): self.nvdev.wreg(self.SEC2_EMEMD0, word)
    next_cmd, next_msg = expected_cmd + len(command), expected_msg + 20
    self.nvdev.wreg(self.SEC2_CMDQ_HEAD, next_cmd)
    self._wait_sec2(lambda x: x[0] == 0x40 and x[2] == x[3] == next_cmd and x[4] == next_msg and x[5] == expected_msg,
                    f"falcon {falcon_id} response")
    header = self._read_sec2_emem(expected_msg & 0xFFFFFF, 4)
    response = header + self._read_sec2_emem((expected_msg + 4) & 0xFFFFFF, 16)
    response_fields = struct.unpack("<5B3x3I", response)
    expected_fields = (7, 20, 0, 0, 0, 0, falcon_id, 0)
    if response_fields != expected_fields:
      raise RuntimeError(f"native SEC2 falcon {falcon_id} response changed: {response.hex()}")
    self.nvdev.wreg(self.SEC2_MSGQ_TAIL, next_msg)
    self.nvdev.wreg(self.SEC2_IRQSCLR, 0x40)
    self._wait_sec2(lambda x: x[0] == 0 and x[2] == x[3] == next_cmd and x[4] == x[5] == next_msg,
                    f"falcon {falcon_id} acknowledgement")
    self._sec2_cmd_head, self._sec2_msg_head = next_cmd, next_msg
    if falcon_id == 3: self._wait_sec2_quiescent(next_cmd, next_msg)

  def _reset_nvdec(self):
    # GA102 NVDEC0 uses PMC reset bit 15, unlike the GSP/SEC2 engine reset registers.
    self.nvdev.wreg(0x848048, self.nvdev.rreg(0x848048) & ~3)
    self.nvdev.wreg(0x848014, 0xFFFFFFFF)
    self.nvdev.rreg(0x8480F4)
    self.nvdev.rreg(0x8480F4)  # USB reads exceed Nouveau's 150 us reset-preparation poll.
    master = self.nvdev.rreg(0x600)
    for value in (master & ~0x8000, master | 0x8000):
      self.nvdev.wreg(0x600, value)
      for _ in range(2):
        if self.nvdev.rreg(0x600) != value: raise RuntimeError("native NVDEC reset did not read back")
    self.nvdev.wreg(0x848040, self.nvdev.rreg(0x848040))
    wait_cond(lambda: self.nvdev.rreg(0x8480F4) & 0x1000, value=0, timeout_ms=100, msg="NVDEC memory scrub did not finish")
    self.nvdev.wreg(0x848084, self.nvdev.chip_id)
    if self.nvdev.rreg(0x848100) & 0x12 != 0x10 or self.nvdev.rreg(0x848118) & 3 != 2:
      raise RuntimeError("native NVDEC is not halted and DMA-idle after reset")

  def _boot_helper(self, base:int, helper:NativeHelper):
    if base == 0x848000: self._reset_nvdec()
    else: self.flcn.reset(base)
    mailbox = self.flcn.execute_hs(base, img_paddr=helper.gpu_address, code_off=helper.imem_source_offset,
      data_off=helper.dmem_source_offset, imemPa=0, imemVa=helper.imem_source_offset, imemSz=helper.imem_bytes,
      dmemPa=0, dmemVa=0, dmemSz=helper.dmem_bytes, pkc_off=helper.dmem_signature_offset,
      engid=helper.engine_id, ucodeid=helper.ucode_id, mailbox=0xCAFEBEEF, ctx_dma=0 if base == 0x848000 else 5, target=1,
      dma_gate=self._set_bme, brom_offset=0x1C00 if base == 0x848000 else 0x1000)
    if mailbox != (0, 0): raise RuntimeError(f"signed {helper.name} boot failed: mailbox={mailbox}")

  def init_hw(self):
    if self.nvdev.fw_name != "ga102": raise RuntimeError(f"native bootstrap requires GA102, got {self.nvdev.fw_name}")
    if self.nvdev.pci_dev.read_config(pci.PCI_COMMAND, 2) & pci.PCI_COMMAND_MASTER:
      raise RuntimeError("native bootstrap requires bus mastering disabled at entry")
    wpr_paddr = self.nvdev.mm.palloc(WPR_ALLOCATION_BYTES, align=0x40000, zero=False)
    if wpr_paddr != self.package.shadow_start:
      raise RuntimeError(f"native WPR allocator returned {wpr_paddr:#x}, expected {self.package.shadow_start:#x}")
    self.nvdev.vram[wpr_paddr:wpr_paddr + WPR_ALLOCATION_BYTES] = bytes(WPR_ALLOCATION_BYTES)
    self.nvdev.vram[wpr_paddr:wpr_paddr + len(self.package.wpr_shadow)] = self.package.wpr_shadow
    readback = bytes(self.nvdev.vram[wpr_paddr:wpr_paddr + WPR_ALLOCATION_BYTES])
    if readback != self.package.wpr_allocation:
      mismatch = next(i for i, (actual, expected) in enumerate(zip(readback, self.package.wpr_allocation)) if actual != expected)
      raise RuntimeError(f"native WPR readback mismatch at offset {mismatch:#x} (GPU {wpr_paddr + mismatch:#x}): "
                         f"actual={readback[mismatch]:#x} expected={self.package.wpr_allocation[mismatch]:#x}; "
                         f"actual_sha256={hashlib.sha256(readback).hexdigest()} "
                         f"expected_sha256={hashlib.sha256(self.package.wpr_allocation).hexdigest()}")

    reserved, stage_offset = self.nvdev.pci_dev.sram.alloc(0x24000), self.nvdev.pci_dev.sram.alloc(0x1C000)
    if (reserved, stage_offset) != (0, 0x24000):
      raise RuntimeError(f"native SYSTEM stage allocator mismatch: {(reserved, stage_offset)}")
    if self.nvdev.pci_dev.usb.read(0xC450, 1) != b"\0": raise RuntimeError("Chestnut controller is busy before native SYSTEM staging")
    self.nvdev.pci_dev.usb.scsi_write(self.package.system_stage, slot_start=9)
    if self.nvdev.pci_dev.usb.read(0xC450, 1) != b"\0": raise RuntimeError("Chestnut native SYSTEM staging did not complete")

    helpers = {helper.name:helper for helper in self.package.helpers}
    # Nouveau runs NVIDIA's signed scrubber before ACR when VPR requires memory initialization.
    if self.nvdev.rreg(0x1FA80C) & 0x10:
      try: self._boot_helper(0x848000, helpers["VPR_SCRUBBER"])
      except BaseException as error:
        try: self._reset_nvdec()
        except BaseException as cleanup_error: raise BaseExceptionGroup("NVDEC scrub and reset failed", [error, cleanup_error])
        raise
      else: self._reset_nvdec()
      if self.nvdev.rreg(0x1FA80C) & 0x10: raise RuntimeError("native VPR is still locked after signed memory scrub")
    self._boot_helper(self.flcn.sec2, helpers["AHESASC"])
    self._require_wpr()
    self._boot_helper(self.flcn.falcon, helpers["ASB"])
    self._require_wpr()
    self._start_sec2()
    self._bootstrap_gr_falcon(2)
    self._bootstrap_gr_falcon(3)
    if self.nvdev.pci_dev.read_config(pci.PCI_COMMAND, 2) & pci.PCI_COMMAND_MASTER:
      raise RuntimeError("native bootstrap left bus mastering enabled")
    status = self.nvdev.pci_dev.read_config(pci.PCI_STATUS, 2)
    if status & 0xF900: raise RuntimeError(f"PCI status fault after native bootstrap: {status:#x}")

class NVReg:
  def __init__(self, nvdev, base, off, fields=None): self.nvdev, self.base, self.off, self.fields = nvdev, base, off, fields

  def __getitem__(self, idx:int): return NVReg(self.nvdev, self.base, self.off(idx), fields=self.fields)

  def add_field(self, name:str, start:int, end:int): self.fields[name] = (start, end)
  def with_base(self, base:int): return NVReg(self.nvdev, base + self.base, self.off, self.fields)

  def read(self): return self.nvdev.rreg(self.base + self.off)
  def read_bitfields(self) -> dict[str, int]: return self.decode(self.read())

  def write(self, _ini_val:int=0, **kwargs): self.nvdev.wreg(self.base + self.off, _ini_val | self.encode(**kwargs))

  def update(self, **kwargs): self.write(self.read() & ~self.mask(*kwargs.keys()), **kwargs)

  def mask(self, *names):
    return functools.reduce(int.__or__, ((((1 << (self.fields[nm][1]-self.fields[nm][0] + 1)) - 1) << self.fields[nm][0]) for nm in names), 0)

  def encode(self, **kwargs) -> int: return functools.reduce(int.__or__, (value << self.fields[name][0] for name,value in kwargs.items()), 0)
  def decode(self, val: int) -> dict: return {name:getbits(val, start, end) for name,(start,end) in self.fields.items()}

class NVPageTableEntry:
  def __init__(self, nvdev, paddr, lv): self.nvdev, self.paddr, self.lv, self.entries = nvdev, paddr, lv, nvdev.vram.view(paddr, 0x1000, fmt='Q')

  def _is_dual_pde(self) -> bool: return self.lv == self.nvdev.mm.level_cnt - 2

  def set_entry(self, entry_id:int, paddr:int, table=False, uncached=False, aspace=AddrSpace.PHYS, snooped=False, frag=0, valid=True,
                kind=6, privileged=False):
    if not table:
      x = self.nvdev.pte_t.encode(valid=valid, address_sys=paddr >> 12, aperture=2 if aspace is AddrSpace.SYS else 0, kind=kind,
        **({'pcf': int(uncached) | (int(privileged) << 1)} if self.nvdev.mmu_ver == 3 else {'vol': uncached, 'privilege': privileged}))
    else:
      pde = self.nvdev.dual_pde_t if self._is_dual_pde() else self.nvdev.pde_t
      small, sys = ("_small" if self._is_dual_pde() else ""), "" if self.nvdev.mmu_ver == 3 else "_sys"
      x = pde.encode(is_pte=False, **{f'aperture{small}': 1 if valid else 0, f'address{small}{sys}': paddr >> 12},
        **({f'pcf{small}': 0b10} if self.nvdev.mmu_ver == 3 else {}))

    if self._is_dual_pde(): self.entries[2*entry_id], self.entries[2*entry_id+1] = x & 0xffffffffffffffff, x >> 64
    else: self.entries[entry_id] = x

  def entry(self, entry_id:int) -> int:
    return (self.entries[2*entry_id+1]<<64) | self.entries[2*entry_id] if self._is_dual_pde() else self.entries[entry_id]

  def read_fields(self, entry_id:int) -> dict:
    if self.is_page(entry_id): return self.nvdev.pte_t.decode(self.entry(entry_id))
    return (self.nvdev.dual_pde_t if self._is_dual_pde() else self.nvdev.pde_t).decode(self.entry(entry_id))

  def is_page(self, entry_id) -> bool: return (self.entry(entry_id) & 1 == 1) if self.lv < self.nvdev.mm.level_cnt - 1 else True
  def supports_huge_page(self, paddr:int): return self.lv >= self.nvdev.mm.level_cnt - 3 and paddr % self.nvdev.mm.pte_covers[self.lv] == 0

  def valid(self, entry_id):
    if self.is_page(entry_id): return self.read_fields(entry_id)['valid']
    return self.read_fields(entry_id)['aperture_small' if self._is_dual_pde() else 'aperture'] != 0

  def address(self, entry_id:int) -> int:
    small, sys = ("_small" if self._is_dual_pde() else ""), "_sys" if self.nvdev.mmu_ver == 2 or self.lv == self.nvdev.mm.level_cnt - 1 else ""
    return self.read_fields(entry_id)[f'address{small}{sys}'] << 12

class NVMemoryManager(MemoryManager):
  va_allocator = TLSFAllocator((1 << 44), base=0x1000000000) # global for all devices.

  def on_range_mapped(self):
    if self.dev.mmu_ver == 2:
      self.dev.NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE_PDB.write(addr=self.root_page_table.paddr >> 12)
      self.dev.NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE_UPPER_PDB.write(0)
      self.dev.NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE.write(all_va=1, trigger=1)
      wait_cond(lambda: self.dev.NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE.read() & (1 << 31), value=0, timeout_ms=2000,
                msg="MMU invalidate did not complete")
    else:
      self.dev.NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE.write((1 << 0) | (1 << 1) | (1 << 6) | (1 << 31))

class NVDev:
  def __init__(self, pci_dev:PCIDevice):
    self.pci_dev, self.devfmt, self.mmio = pci_dev, pci_dev.pcibus, pci_dev.map_bar(0, fmt='I')

    self.smi_dev, self.is_booting, self.is_err_state = False, True, False
    self._early_ip_init()
    self._early_mmu_init()

    # No booting state, gsp client is reinited every run.
    self.is_booting = False

    for ip in [self.flcn, self.gsp]: ip.init_sw()
    for ip in [self.flcn, self.gsp]: ip.init_hw()

  def fini(self):
    for ip in [self.gsp, self.flcn]: ip.fini_hw()

  def reg(self, reg:str) -> NVReg: return self.__dict__[reg]
  def wreg(self, addr:int, value:int):
    self.mmio[addr // 4] = value
    if NV_DEBUG >= 4: print(f"wreg: {hex(addr)} = {hex(value)}")
  def rreg(self, addr:int) -> int: return self.mmio[addr // 4]

  def _early_ip_init(self):
    self.reg_names:set[str] = set()
    self.reg_offsets:dict[str, tuple[int, int]] = {}

    self.include("nv_ref", "")
    self.include("dev_fb", "tu102")
    self.include("dev_gc6_island", "ga102")

    if self.reg("NV_PFB_PRI_MMU_WPR2_ADDR_HI").read() != 0:
      self.pci_dev.write_config_flush(pci.PCI_COMMAND, self.pci_dev.read_config(pci.PCI_COMMAND, 2) & ~pci.PCI_COMMAND_MASTER, 2)
      if DEBUG >= 2: print(f"nv {self.devfmt}: WPR2 is up. Issuing a full reset.", flush=True)
      self.pci_dev.reset()
      time.sleep(0.1) # wait until device can respond again

    self.pci_dev.write_config_flush(pci.PCI_COMMAND, self.pci_dev.read_config(pci.PCI_COMMAND, 2) | pci.PCI_COMMAND_MASTER, 2)
    self.chip_id = self.reg("NV_PMC_BOOT_0").read()
    self.chip_details = self.reg("NV_PMC_BOOT_42").read_bitfields()
    self.chip_name = {0x17: "GA1", 0x19: "AD1", 0x1b: "GB2"}[self.chip_details['architecture']] + f"{self.chip_details['implementation']:02d}"
    self.fw_name = {"GB2": "gb202", "AD1": "ad102", "GA1": "ga102"}[self.chip_name[:3]]
    self.mmu_ver, self.fmc_boot = (3, True) if self.chip_details['architecture'] >= 0x1a else (2, False)

    self.flcn:NV_FLCN|NV_FLCN_COT = NV_FLCN_COT(self) if self.fmc_boot else NV_FLCN(self)
    self.gsp:NV_GSP = NV_GSP(self)

    self.flcn.wait_for_reset()

  def _early_mmu_init(self):
    self.include("dev_vm", "tu102")

    # MMU Init
    self.include("dev_mmu", "gh100" if self.mmu_ver == 3 else "tu102")
    self.pte_t, self.pde_t, self.dual_pde_t = [self.__dict__[name] for name in [f'NV_MMU_VER{self.mmu_ver}_PTE', f'NV_MMU_VER{self.mmu_ver}_PDE',
                                                                                f'NV_MMU_VER{self.mmu_ver}_DUAL_PDE']]

    self.vram_size = self.reg("NV_PGC6_AON_SECURE_SCRATCH_GROUP_42").read() << 20

    self.vram, self.mmio = self.pci_dev.map_bar(1), self.pci_dev.map_bar(0, fmt='I')
    self.large_bar = self.vram.nbytes >= self.vram_size

    # UVM depth   HW level                            VA bits
    # 0           PDE4                                56:56 (hopper+)
    # 1           PDE3                                55:47
    # 2           PDE2                                46:38
    # 3           PDE1 (or 512M PTE)                  37:29
    # 4           PDE0 (dual 64k/4k PDE, or 2M PTE)   28:21
    # 5           PTE_64K / PTE_4K                    20:16 / 20:12
    bits, shifts = (56, [12, 21, 29, 38, 47, 56]) if self.mmu_ver == 3 else (48, [12, 21, 29, 38, 47])

    # tail vram reserved for falcon structs
    self.mm = NVMemoryManager(self, self.vram_size - (64 << 20), boot_size=(2 << 20), pt_t=NVPageTableEntry, va_bits=bits, va_shifts=shifts,
      va_base=0, palloc_ranges=[(x, x) for x in [512 << 20, 2 << 20, 4 << 10]], reserve_ptable=not self.large_bar)

  def _alloc_boot_mem(self, size:int, data:bytes|None=None, contiguous:bool=False, sysmem:bool|None=None) -> tuple[MMIOInterface,int|None,list[int]]:
    sz = round_up(size, 0x1000)
    if sysmem is True or (sysmem is None and not self.large_bar):
      view, sysaddr = self.pci_dev.alloc_sysmem(size, 0, contiguous=contiguous)
      paddr = None
    else:
      paddr = self.mm.palloc(sz, boot=False)
      view = self.vram.view(paddr, sz)
      sysaddr = [self.pci_dev.bar_info(1)[0] + paddr + i * 0x1000 for i in range(sz // 0x1000)]
    if data is not None: view[:size] = data
    return view, paddr, sysaddr

  def include(self, name:str, arch:str):
    for k,v in getattr(getattr(tinygrad.runtime.autogen.nv_regs, name), arch or 'regs').items():
      self.__dict__[k] = NVReg(self, *v) if isinstance(v, tuple) else v


class NVNativeDev(NVDev):
  def __init__(self, pci_dev:PCIDevice):
    self.pci_dev, self.devfmt, self.mmio = pci_dev, pci_dev.pcibus, pci_dev.map_bar(0, fmt='I')
    self.smi_dev, self.is_booting, self.is_err_state = False, True, False
    self._early_ip_init()
    self._early_mmu_init()
    self.is_booting = False

    command = self.pci_dev.read_config(pci.PCI_COMMAND, 2) & ~pci.PCI_COMMAND_MASTER
    self.pci_dev.write_config_flush(pci.PCI_COMMAND, command, 2)
    actual = self.pci_dev.read_config(pci.PCI_COMMAND, 2)
    if actual != command: raise RuntimeError(f"native NV initialization could not disable bus mastering: {actual:#x} != {command:#x}")

    if not isinstance(self.flcn, NV_FLCN): raise RuntimeError(f"native NV initialization requires GA102 Falcon, got {type(self.flcn).__name__}")
    package = build_ga102_acr_package(0x200000)
    self.native = NVNativeBootstrap(self, self.flcn, package)
    self.native.init_hw()

    command = self.pci_dev.read_config(pci.PCI_COMMAND, 2) | pci.PCI_COMMAND_MASTER
    self.pci_dev.write_config_flush(pci.PCI_COMMAND, command, 2)
    actual = self.pci_dev.read_config(pci.PCI_COMMAND, 2)
    if actual != command: raise RuntimeError(f"native NV initialization could not enable bus mastering: {actual:#x} != {command:#x}")

    self.gr = NVNativeGR(self, package.gr_net)
    self.gr.init_hw()

    self._native_chid = 0
    self._native_bar2_instance_paddr:int|None = None
    self._native_bar2_next = 0x1000
    self._native_copy_contexts:dict[int, VirtMapping] = {}
    self._native_runlist_paddr:int|None = None

  USERD_CLEAR_OFFSETS = (0x040, 0x044, 0x048, 0x04C, 0x050, 0x058, 0x05C, 0x060, 0x088, 0x08C)

  @staticmethod
  def _ga102_vmm_image(root_paddr:int, va_limit:int) -> bytes:
    if root_paddr & 0xFFF: raise ValueError(f"native channel root page is unaligned: {root_paddr:#x}")
    if not 0 < va_limit <= 1 << 64: raise ValueError(f"native channel VA limit is invalid: {va_limit:#x}")

    image, pdb = bytearray(0x1000), root_paddr | 0xC00
    struct.pack_into("<QQ", image, 0x200, pdb, va_limit - 1)
    struct.pack_into("<I", image, 0x21C, 0)
    for index in range(64):
      low, high = (pdb, 0) if index == 0 else (1, 1)
      struct.pack_into("<III", image, 0x2A0 + index * 0x10, low, high, 0)
    struct.pack_into("<II", image, 0x298, 1, 0)
    return bytes(image)

  @staticmethod
  def _ga102_channel_image(root_paddr:int, va_limit:int, gpfifo_offset:int, entries:int, chid:int, nonstall_vector:int) -> bytes:
    if entries <= 0 or entries & (entries - 1): raise ValueError(f"native channel GPFIFO entries are not a power of two: {entries}")
    if not 0 <= chid < 0x1000: raise ValueError(f"native channel ID is invalid: {chid}")
    if not 0 <= nonstall_vector < 0x1000: raise ValueError(f"native channel nonstall vector is invalid: {nonstall_vector:#x}")

    image = bytearray(NVNativeDev._ga102_vmm_image(root_paddr, va_limit))
    for offset, value in (
      (0x010, 0x0000FACE), (0x030, 0x7FFFF902),
      (0x048, gpfifo_offset & 0xFFFFFFFF),
      (0x04C, (gpfifo_offset >> 32) | ((entries.bit_length() - 1) << 16)),
      (0x084, 0x20400000), (0x094, 0x30000001), (0x0E4, 0), (0x0E8, chid),
      (0x0F4, 0x00001000), (0x0F8, 0x80000000 | nonstall_vector),
    ): struct.pack_into("<I", image, offset, value)
    return bytes(image)

  def _ga102_gr_runlist(self) -> tuple[int, int, int, int, int]:
    size = self.rreg(0x224FC) >> 20
    if size != 39: raise RuntimeError(f"native GA102 TOP table size changed: {size}")
    records:list[tuple[int, ...]] = []
    record:list[int] = []
    for index in range(size):
      word = self.rreg(0x22800 + index * 4)
      if not record and word == 0: continue
      record.append(word)
      if word & 0x80000000: continue
      if len(record) < 2: raise RuntimeError(f"native GA102 TOP record is short: {record}")
      records.append(tuple(record))
      record = []
    if record: raise RuntimeError(f"native GA102 TOP record is unterminated: {record}")
    matches = [item for item in records if (item[0] >> 24) & 0x3F == 0 and (item[0] >> 16) & 0xF == 0]
    if len(matches) != 1 or len(matches[0]) < 3: raise RuntimeError(f"native GA102 GR TOP record changed: {matches}")
    runlist = matches[0][2] & 0x00FFFC00
    chcfg, dbcfg, vector_cfg = (self.rreg(runlist + offset) for offset in (0x004, 0x008, 0x160))
    if (runlist, chcfg, dbcfg, vector_cfg) != (0xC00000, 0x00C2000B, 0, 0xC00000A0):
      raise RuntimeError(f"native GA102 GR runlist metadata changed: {(runlist, chcfg, dbcfg, vector_cfg)}")
    return runlist, chcfg & 0xFFFFFFF0, 1 << (chcfg & 0xF), dbcfg >> 16, vector_cfg & 0xFFF

  def _write_vram(self, paddr:int, data:bytes): self.vram[paddr:paddr + len(data)] = data

  def alloc_channel(self, gpfifo_offset:int, entries:int, userd_paddr:int) -> NVNativeChannel:
    runlist, channel_table, channel_count, doorbell, nonstall_vector = self._ga102_gr_runlist()
    chid = self._native_chid
    if chid >= channel_count: raise RuntimeError("native GA102 GR runlist has no free channel IDs")
    if (slot:=self.rreg(channel_table + chid * 4)) != 0:
      raise RuntimeError(f"native GA102 channel {chid} is not empty: {slot:#x}")
    instance_paddr = self.mm.palloc(0x1000, align=0x1000, zero=False)
    image = self._ga102_channel_image(self.mm.root_page_table.paddr, 0x2000000000000,
                                      gpfifo_offset, entries, chid, nonstall_vector)
    self._write_vram(instance_paddr, image)
    for offset in self.USERD_CLEAR_OFFSETS: self._write_vram(userd_paddr + offset, bytes(4))
    self._native_chid += 1
    return NVNativeChannel(chid, instance_paddr, runlist, channel_table, doorbell, nonstall_vector, userd_paddr)

  def schedule_channel_group(self, channels:list[NVNativeChannel]):
    if not channels: raise RuntimeError("native GA102 channel group is empty")
    if self._native_runlist_paddr is not None: raise RuntimeError("native GA102 channel group is already scheduled")
    channels = sorted(channels, key=lambda channel: channel.chid)
    runlist, channel_table = channels[0].runlist, channels[0].channel_table
    if len({channel.chid for channel in channels}) != len(channels): raise RuntimeError("native GA102 channel IDs are not unique")
    for channel in channels:
      if (channel.runlist, channel.channel_table) != (runlist, channel_table):
        raise RuntimeError("native GA102 channel group spans multiple runlists")
      if channel.userd_paddr & 3: raise RuntimeError(f"native GA102 channel {channel.chid} USERD is unaligned")
      if channel.runq not in (0, 1): raise RuntimeError(f"native GA102 channel {channel.chid} runqueue is invalid")
    if self.rreg(runlist + 0x08C) & 0x8000: raise RuntimeError("native GA102 runlist update was already pending")
    self.gr.bind_channel_group_context(channels)

    image = bytearray(0x1000)
    struct.pack_into("<4I", image, 0, 0x80030001, len(channels), channels[0].chid, 0)
    for index, channel in enumerate(channels, 1):
      struct.pack_into("<4I", image, index * 0x10,
        (channel.userd_paddr & 0xFFFFFFFF) | channel.runq << 1, channel.userd_paddr >> 32,
        (channel.instance_paddr & 0xFFFFFFFF) | channel.chid, channel.instance_paddr >> 32)
    self._native_runlist_paddr = runlist_paddr = self.mm.palloc(0x1000, align=0x1000, zero=True)
    self._write_vram(runlist_paddr, bytes(image))
    self.wreg(0x70000, 1)
    wait_cond(lambda: self.rreg(0x70000) & 2, value=0, timeout_ms=2000, msg="native GA102 runlist BAR flush did not complete")

    self.wreg(runlist + 0x088, 0)
    self.wreg(runlist + 0x098, 0)
    self.wreg(runlist + 0x300, self.rreg(runlist + 0x300) | 0x80000000)
    for channel in channels:
      self.wreg(channel_table + channel.chid * 4, 2)
      self.wreg(runlist + 0x090, channel.chid)
    self.wreg(runlist + 0x080, runlist_paddr & 0xFFFFFFFF)
    self.wreg(runlist + 0x084, runlist_paddr >> 32)
    self.wreg(runlist + 0x088, len(channels) + 1)
    wait_cond(lambda: self.rreg(runlist + 0x08C) & 0x8000, value=0, timeout_ms=2000,
              msg="native GA102 runlist commit did not complete")
    expected = (runlist_paddr & 0xFFFFFFFF, runlist_paddr >> 32, len(channels) + 1)
    actual = tuple(self.rreg(runlist + offset) for offset in (0x080, 0x084, 0x088))
    if actual != expected: raise RuntimeError(f"native GA102 runlist commit changed: {actual} != {expected}")

  def bind_copy_context(self, channel:NVNativeChannel):
    if channel.chid in self._native_copy_contexts: return
    pce_map = self.rreg(0x104028)
    if not 0 < pce_map < 1 << 32: raise RuntimeError(f"native GA102 PCE map is invalid: {pce_map:#x}")
    context_size = round_up(27 * 5 * (((9 + 1 + 3) * pce_map.bit_count()) + 2), 0x1000)
    try: bar2_size = self.pci_dev.bar_info(3)[1]
    except KeyError as error: raise RuntimeError("native GA102 instance BAR is unavailable") from error
    context_va = round_up(self._native_bar2_next, 0x1000)
    if context_va + context_size > bar2_size:
      raise RuntimeError(f"native GA102 instance BAR is too small: need {context_va + context_size:#x}, have {bar2_size:#x}")

    context_paddr = self.mm.palloc(context_size, align=0x1000, zero=True)
    context = self.mm.map_range(context_va, context_size, [(context_paddr, context_size)], AddrSpace.PHYS, kind=0)
    if self._native_bar2_instance_paddr is None:
      instance_paddr = self.mm.palloc(0x1000, align=0x1000, zero=False)
      self._write_vram(instance_paddr, self._ga102_vmm_image(self.mm.root_page_table.paddr, bar2_size))
      block = 0x80000000 | instance_paddr >> 12
      self.wreg(0xB80F48, block)
      wait_cond(lambda: self.rreg(0xB80F50) & 0xC, value=0, timeout_ms=2000, msg="native GA102 BAR2 bind did not complete")
      if (actual:=self.rreg(0xB80F48)) != block:
        raise RuntimeError(f"native GA102 BAR2 block did not latch: {actual:#x} != {block:#x}")
      self._native_bar2_instance_paddr = instance_paddr

    flags, = struct.unpack("<I", bytes(self.vram[channel.instance_paddr + 0xAC:channel.instance_paddr + 0xB0]))
    if flags & 0x20000: raise RuntimeError(f"native GA102 channel {channel.chid} already has a copy context")
    self._write_vram(channel.instance_paddr + 0x220, struct.pack("<Q", context_va))
    self._write_vram(channel.instance_paddr + 0xAC, struct.pack("<I", flags | 0x20000))
    self.wreg(0x70000, 1)
    wait_cond(lambda: self.rreg(0x70000) & 2, value=0, timeout_ms=2000, msg="native GA102 copy context BAR flush did not complete")
    self._native_copy_contexts[channel.chid] = context
    self._native_bar2_next = context_va + context_size

  def fini(self): self.pci_dev.reset()


@dataclass(frozen=True)
class NVNativeChannel:
  chid:int
  instance_paddr:int
  runlist:int
  channel_table:int
  doorbell:int
  nonstall_vector:int
  userd_paddr:int = 0
  runq:int = 0

  @property
  def token(self) -> int: return (self.doorbell << 16) | self.chid
