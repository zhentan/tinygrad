from __future__ import annotations
import hashlib, struct, time
from dataclasses import dataclass
from tinygrad.helpers import round_up
from tinygrad.runtime.support.memory import AddrSpace, VirtMapping


def _require(condition:bool, message:str):
  if not condition: raise RuntimeError(message)


def _gpc_unit(gpc:int, offset:int) -> int: return 0x500000 + gpc * 0x8000 + offset


@dataclass(frozen=True)
class GA102Topology:
  tpc_nr: tuple[int, ...]
  ppc_tpc_mask: tuple[tuple[int, ...], ...]
  swdx_pes: tuple[int, ...]

  @property
  def gpc_nr(self) -> int: return len(self.tpc_nr)
  @property
  def tpc_total(self) -> int: return sum(self.tpc_nr)
  @property
  def tpc_max(self) -> int: return max(self.tpc_nr)
  @property
  def ppc_nr(self) -> tuple[int, ...]: return tuple(sum(mask != 0 for mask in row) for row in self.ppc_tpc_mask)
  @property
  def ppc_total(self) -> int: return sum(self.ppc_nr)
  @property
  def ppc_tpc_min(self) -> int: return min(mask.bit_count() for row in self.ppc_tpc_mask for mask in row if mask)


@dataclass(frozen=True)
class _Operation:
  address: int
  value: int
  mask: int|None = None
  expected_before: int|None = None


@dataclass(frozen=True)
class GA102ContextBuffer:
  requested_size: int
  paddr: int
  mapping: VirtMapping


def _oneinit_tiles(topology:GA102Topology) -> tuple[int, list[int]]:
  primes = (3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61)
  special = {15:6, 14:5, 13:2, 11:7, 10:6, 7:1, 5:1, 3:2, 2:1, 1:1}
  row_offset = special.get(topology.tpc_total)
  if row_offset is None: row_offset = next(prime for prime in primes if topology.tpc_total % prime)
  gpc_map = list(range(topology.gpc_nr))
  sorted_map = False
  while not sorted_map:
    sorted_map = True
    for index in range(topology.gpc_nr - 1):
      if topology.tpc_nr[gpc_map[index + 1]] > topology.tpc_nr[gpc_map[index]]:
        gpc_map[index], gpc_map[index + 1], sorted_map = gpc_map[index + 1], gpc_map[index], False
  mul_factor = 2 if topology.gpc_nr * topology.tpc_max & 1 else 1
  denominator = topology.gpc_nr * topology.tpc_max * mul_factor
  fractions = [topology.tpc_nr[gpc_map[i]] * topology.gpc_nr * mul_factor for i in range(topology.gpc_nr)]
  errors = [fractions[i] + i * topology.tpc_max * mul_factor - denominator // 2 for i in range(topology.gpc_nr)]
  tiles: list[int] = []
  while len(tiles) < topology.tpc_total:
    for index in range(topology.gpc_nr):
      if errors[index] * 2 >= denominator:
        tiles.append(gpc_map[index])
        errors[index] += fractions[index] - denominator
      else: errors[index] += fractions[index]
  _require(len(tiles) == topology.tpc_total, "native GA102 tile map overshot the TPC total")
  return row_offset, tiles


def _estimate_perf(topology:GA102Topology, masks:list[int], disable_gpc:int, disable_tpc:int) -> int:
  scale, pixels, world = 512, 1024 * 1024, 1024
  pes_count, min_pixels, average_tpcs, max_tpcs = 0, scale, 0, 0
  counts: list[int] = []
  removed_gpc = removed_pes = False
  for gpc in range(topology.gpc_nr):
    mask = masks[gpc]
    if gpc == disable_gpc and mask & 1 << disable_tpc:
      _require(not removed_gpc, "native GA102 TPC removed twice from GPC mask")
      mask, removed_gpc = mask & ~(1 << disable_tpc), True
    count = mask.bit_count()
    counts.append(count)
    average_tpcs, max_tpcs = average_tpcs + count, max(max_tpcs, count)
    min_pixels = min(min_pixels, scale * count // topology.tpc_nr[gpc])
    for pes in range(topology.ppc_nr[gpc]):
      pes_mask = topology.ppc_tpc_mask[gpc][pes] & masks[gpc]
      if gpc == disable_gpc and pes_mask & 1 << disable_tpc:
        _require(not removed_pes, "native GA102 TPC removed twice from PPC mask")
        pes_mask, removed_pes = pes_mask & ~(1 << disable_tpc), True
      if pes_mask.bit_count(): pes_count += 1
  _require(removed_gpc and removed_pes, "native GA102 candidate TPC was absent from topology masks")
  if max_tpcs == 0: return 0
  world_perf = scale * pes_count // topology.ppc_total
  average_tpcs = scale * average_tpcs // topology.gpc_nr
  deviation = sum(abs(average_tpcs - scale * count) for count in counts) // topology.gpc_nr
  normalized_deviation = deviation // max_tpcs
  balance = scale - normalized_deviation
  _require(all(value <= scale for value in (balance, world_perf, min_pixels, normalized_deviation)),
           "native GA102 SCG estimate overflowed")
  return pixels * min_pixels + world * world_perf + balance


def _oneinit_sm_id(topology:GA102Topology) -> list[tuple[int, int]]:
  masks = [0] * topology.gpc_nr
  for gpc, row in enumerate(topology.ppc_tpc_mask):
    for mask in row: masks[gpc] |= mask
  result: list[tuple[int, int]] = []
  for _ in range(topology.tpc_total):
    maximum, selected = -1, None
    for gpc in range(topology.gpc_nr):
      for tpc in range(topology.tpc_nr[gpc]):
        if masks[gpc] & 1 << tpc and (performance:=_estimate_perf(topology, masks, gpc, tpc)) > maximum:
          maximum, selected = performance, (gpc, tpc)
    if selected is None: raise RuntimeError("native GA102 SCG selection exhausted TPCs early")
    result.append(selected)
    masks[selected[0]] &= ~(1 << selected[1])
  _require(not any(masks), "native GA102 SCG selection left a TPC enabled")
  return result


def _nonpes_aware_tpc(topology:GA102Topology, gpc:int, tpc:int) -> int:
  translated = 0
  for pes in range(topology.ppc_nr[gpc]):
    mask = topology.ppc_tpc_mask[gpc][pes]
    if mask & 1 << tpc: return translated + (((1 << tpc) - 1) & mask).bit_count()
    translated += mask.bit_count()
  raise RuntimeError(f"native GA102 GPC {gpc} TPC {tpc} has no PPC")


def _unit_a_writes(topology:GA102Topology, active_ltcs:int) -> list[tuple[int, int]]:
  row_offset, tiles, sm_order = *_oneinit_tiles(topology), _oneinit_sm_id(topology)
  magic = (0x00800000 + topology.tpc_total - 1) // topology.tpc_total
  writes: list[tuple[int, int]] = [(_gpc_unit(0, 0x3018), 1)]
  bank = [0] * topology.gpc_nr
  for index in range(0, topology.gpc_nr * topology.tpc_max, 8):
    data = 0
    for slot in range(8):
      if index + slot >= topology.tpc_total: break
      gpc = tiles[index + slot]
      data, bank[gpc] = data | bank[gpc] << (slot * 4), bank[gpc] + 1
    writes.append((0x418980 + index // 2, data))
  _require(tuple(bank) == topology.tpc_nr, "native GA102 ZCULL bank counters changed")
  for gpc in range(topology.gpc_nr):
    writes.extend(((_gpc_unit(gpc, 0x0914), row_offset << 8 | topology.tpc_nr[gpc]),
                   (_gpc_unit(gpc, 0x0910), 0x00040000 | topology.tpc_total), (_gpc_unit(gpc, 0x0918), magic)))
  writes.extend(((0x41BFD4, magic), (0x4188AC, active_ltcs),
                 (0x4181D0, sum(value << (gpc * 4) for gpc, value in enumerate(topology.swdx_pes)))))
  distribution, gpcs = [0] * ((topology.tpc_total + 3) // 4), [0] * 16
  for sm, (gpc, tpc) in enumerate(sm_order):
    distribution[sm // 4] |= ((gpc << 4) | tpc) << (sm % 4) * 8
    gpcs[gpc + 7 * (tpc // 4)] |= sm << (tpc % 4) * 8
  writes.extend((0x405B60 + index * 4, value) for index, value in enumerate(distribution))
  writes.extend((0x405BA0 + index * 4, value) for index, value in enumerate(gpcs))
  writes.append((0x405B00, topology.tpc_total << 8 | topology.gpc_nr))
  for sm, (gpc, tpc) in enumerate(sm_order): writes.append((_gpc_unit(gpc, 0x0C10 + _nonpes_aware_tpc(topology, gpc, tpc) * 4), sm))
  skip_bytes = [0] * 32
  for gpc, row in enumerate(topology.ppc_tpc_mask):
    for original in row:
      count, mask = original.bit_count(), original
      while count > topology.ppc_tpc_min: count, mask = count - 1, mask & (mask - 1)
      skip_bytes[gpc] |= mask ^ original
  for index in range(8):
    writes.append((0x4064D0 + index * 4, sum(skip_bytes[index * 4 + byte] << (byte * 8) for byte in range(4))))
  gpc = 0
  for index in range(4):
    data = 0
    for nibble in range(8):
      if gpc >= topology.gpc_nr: break
      data, gpc = data | topology.tpc_nr[gpc] << (nibble * 4), gpc + 1
    writes.extend(((0x406028 + index * 4, data), (0x405870 + index * 4, data)))
  return writes


def _ppc_unit(gpc:int, ppc:int, offset:int) -> int: return 0x503000 + gpc * 0x8000 + ppc * 0x200 + offset


def _tpc_unit(gpc:int, tpc:int, offset:int) -> int: return 0x504000 + gpc * 0x8000 + tpc * 0x800 + offset


def _unit_b_writes(topology:GA102Topology, rop_source:int) -> list[tuple[int, int]]:
  writes = [
    (0x400500, 0x00010001), (0x400100, 0xFFFFFFFF), (0x40013C, 0xFFFFFFFF), (0x400124, 2),
    (0x409C24, 0x006E0003), (0x40A790, 0xC0000000), (0x405848, 0xC0000000), (0x40584C, 0x0000007F),
    (0x404000, 0xC0000000), (0x404600, 0xC0000000), (0x408030, 0xC0000000), (0x406018, 0xC0000000),
    (0x404490, 0xC0000000), (0x407020, 0x40000000), (0x405840, 0xC0000000), (0x405844, 0x00FFFFFF),
  ]
  for gpc in range(topology.gpc_nr):
    for ppc in range(3):
      _require(topology.ppc_tpc_mask[gpc][ppc] != 0, "native GA102 Unit-B PPC topology changed")
      writes.append((_ppc_unit(gpc, ppc, 0x038), 0xC0000000))
  for gpc, count in enumerate(topology.tpc_nr):
    writes.extend((_gpc_unit(gpc, offset), 0xC0000000) for offset in (0x0420, 0x0900, 0x1028, 0x0824))
    for tpc in range(count):
      writes.extend(((_tpc_unit(gpc, tpc, 0x0508), 0xFFFFFFFF), (_tpc_unit(gpc, tpc, 0x050C), 0xFFFFFFFF),
                     (_tpc_unit(gpc, tpc, 0x0084), 0xC0000000), (_tpc_unit(gpc, tpc, 0x0430), 0x403F0000),
                     (_tpc_unit(gpc, tpc, 0x0610), 1), (_tpc_unit(gpc, tpc, 0x072C), 4),
                     (_tpc_unit(gpc, tpc, 0x0610), 1), (_tpc_unit(gpc, tpc, 0x07AC), 4)))
    writes.extend(((_gpc_unit(gpc, 0x2C90), 0xFFFFFFFF), (_gpc_unit(gpc, 0x2C94), 0xFFFFFFFF)))
  writes.extend(((0x41BCBC, 0x40000000), (0x41BC38, 0x40000000), (0x41AC94, rop_source),
                 (0x400108, 0xFFFFFFFF), (0x400138, 0xFFFFFFFF), (0x400118, 0xFFFFFFFF), (0x400130, 0xFFFFFFFF)))
  _require(len(writes) == 414 and writes[409] == (0x41AC94, rop_source), "native GA102 Unit-B write shape changed")
  return writes


def _zbc_operations(baselines:dict[int, int]) -> list[_Operation]:
  tracked, operations = baselines.copy(), []
  def write(address:int, value:int):
    operations.append(_Operation(address, value & 0xFFFFFFFF))
    tracked[address] = value & 0xFFFFFFFF
  def mask(address:int, bitmask:int, data:int):
    _require(address in tracked, f"native GA102 ZBC lacks RMW baseline at {address:#x}")
    before = tracked[address]
    value = (before & ~bitmask) | (data & bitmask)
    operations.append(_Operation(address, value, bitmask, before))
    tracked[address] = value

  colors = ((0, 0, 0, 0), (0xFFFFFFFF,) * 4, (0, 0, 0, 0), (0x3F800000,) * 4)
  for index, color in enumerate(colors, 1):
    mask(0x17E338, 0x1F, index)
    for component, value in enumerate(color): write(0x17E33C + component * 4, value)
    mask(0x41BCB4, 0x1F, index)
    for component, value in enumerate(color): write(0x41BCEC + component * 4, value)
  for index in range(5, 31):
    mask(0x41BCB4, 0x1F, index)
    for component in range(4): write(0x41BCEC + component * 4, 0)
  for index, value in ((1, 0), (2, 0x3F800000)):
    mask(0x17E338, 0x0F, index)
    write(0x17E34C, value)
    write(0x418110 + (index - 1) * 4, value)
    znum, fmt = index - 1, 1
    shift, address = znum % 4 * 7, 0x41814C + znum // 4 * 4
    mask(address, 0x7F << shift, fmt << shift)
  for index in range(3, 16):
    znum = index - 1
    shift, address = znum % 4 * 7, 0x41814C + znum // 4 * 4
    mask(address, 0x7F << shift, 0)
  for index, value in ((1, 0), (2, 1), (3, 0xFF)):
    mask(0x17E338, 0x0F, index)
    write(0x17E204, value)
    write(0x41815C + (index - 1) * 4, value)
    znum, fmt = index - 1, 1
    shift, address = znum % 4 * 7, 0x418198 + znum // 4 * 4
    mask(address, 0x7F << shift, fmt << shift)
  for index in range(4, 16):
    znum = index - 1
    shift, address = znum % 4 * 7, 0x418198 + znum // 4 * 4
    mask(address, 0x7F << shift, 0)
  _require(len(operations) == 215 and sum(operation.mask is not None for operation in operations) == 69,
           "native GA102 ZBC operation shape changed")
  mask(0x4188A4, 0x03000000, 0x03000000)
  return operations


class NVNativeGR:
  LTC_ZBC_SELECTOR = 0x17E338
  LTC_ZBC_COLOR = (0x17E33C, 0x17E340, 0x17E344, 0x17E348)
  LTC_ZBC_DEPTH, LTC_ZBC_STENCIL = 0x17E34C, 0x17E204
  GR_GATE, GR_STATUS_FLUSH, GR_BUSY, MASTER_ENABLE = 0x400500, 0x400700, 0x40060C, 0x000200
  FECS_IRQSTAT, FECS_CPUCTL, FECS_OS, FECS_START = 0x409008, 0x409100, 0x40910C, 0x409130
  FECS_METHOD_DATA, FECS_METHOD = 0x409500, 0x409504
  FECS_MAILBOX0, FECS_MAILBOX1, FECS_CTXCTL_IRQSTAT = 0x409800, 0x409804, 0x409C18
  GPCCS_IRQSTAT, GPCCS_CPUCTL, GPCCS_OS, GPCCS_START = 0x41A008, 0x41A100, 0x41A10C, 0x41A130
  GPCCS_MAILBOX0, GPCCS_MAILBOX1 = 0x41A800, 0x41A804
  CTXCTL_SIZES = (0x155600, 0x118E00, 0x12500)
  CONTEXT_BUFFER_LAYOUT = (
    ("pagepool", 0x020000, 0x0100),
    ("bundle",   0x003000, 0x0100),
    ("attrib",   0x829200, 0x1000),
    ("unknown",  0x080000, 0x0100),
  )
  CTXCTL_WRITES = (
    (FECS_MAILBOX0, 0), (GPCCS_OS, 0), (FECS_OS, 0), (GPCCS_START, 2), (FECS_START, 2),
    (FECS_MAILBOX0, 0), (FECS_METHOD_DATA, 0x7FFFFFFF), (FECS_METHOD, 0x21),
    (FECS_MAILBOX0, 0), (FECS_METHOD_DATA, 0), (FECS_METHOD, 0x10),
    (FECS_MAILBOX0, 0), (FECS_METHOD_DATA, 0), (FECS_METHOD, 0x16),
    (FECS_MAILBOX0, 0), (FECS_METHOD_DATA, 0), (FECS_METHOD, 0x25),
  )
  NET_REGIONS = (48, 49, 50, 51)
  NET_REGION_EXPECTED = {
    48:(1080, 72560, "8189767e143bc9db19850823ae65fc89850aa83b2330d5c45e119726d966c0e9"),
    49:(2944, 73640, "4d7e8b9c1cd5902ce6b4ab06838ec0f88b8961d74fb06580cee880c9625592ce"),
    50:(624, 76584, "9c448c9cb2c9ddc6b4f5e5a79e5b4af345200c175df407150ffdb994eb937e0b"),
    51:(184, 77208, "98f5c9860b26a1b2046de8ddcdd1d6305e7c54bfc9a2e2759e7972b48199e06f"),
  }
  CONTEXT_NET_EXPECTED = {
    4:(7480, 57184, "f95ddb179cbfb0dd258b8e00ffcd537f5d0024836177e18362eacac02fa908d9"),
    5:(6924, 65632, "d95971a71fc90cddbb7eeae31c6f4af2ee72853a225c5e019e897f1973b809cf"),
    7:(13608, 77392, "4220c46e0c30a3166178aa5710fcd8ffc883f343048c42f08e8a5a32e579941e"),
    28:(176, 64664, "f0bc69a638d4d128ecfa6f200d6b1ed75f8de37292b388d963da2884e98d3f9e"),
    34:(792, 64840, "618f659d9e693fcaca9b4ccdddb60f4bcb9312d6fca5479d5f78bc247bbf72f1"),
  }

  def __init__(self, nvdev, net_image:bytes):
    self.nvdev, self.net_image = nvdev, net_image
    self.golden_context: bytes|None = None
    self.channel_contexts: dict[int, tuple[VirtMapping, VirtMapping]] = {}

  @staticmethod
  def _stream_digest(writes:list[tuple[int, int]]) -> str:
    return hashlib.sha256(b"".join(struct.pack("<II", *item) for item in writes)).hexdigest()

  def _net_descriptors(self) -> dict[int, tuple[int, int]]:
    if len(self.net_image) != 146088 or hashlib.sha256(self.net_image).hexdigest() != \
       "56b3b302589b9e3bbb7d6ac8f5e36840e4297773c3900779c27c0505c1e605d5":
      raise RuntimeError("native GA102 NET image changed")
    version, count = struct.unpack_from("<II", self.net_image)
    if (version, count) != (0, 50): raise RuntimeError(f"native GA102 NET header changed: {(version, count)}")
    descriptors = {region_id:(size, offset) for region_id, size, offset in
                   struct.iter_unpack("<III", self.net_image[8:8 + count * 12])}
    if len(descriptors) != count: raise RuntimeError("native GA102 NET contains duplicate regions")
    return descriptors

  def _parse_net(self) -> list[tuple[int, int]]:
    descriptors = self._net_descriptors()
    writes: list[tuple[int, int]] = []
    for region_id in self.NET_REGIONS:
      size, offset = descriptors.get(region_id, (-1, -1))
      expected_size, expected_offset, expected_sha = self.NET_REGION_EXPECTED[region_id]
      if (size, offset) != (expected_size, expected_offset):
        raise RuntimeError(f"native GA102 NET region {region_id} changed")
      region = self.net_image[offset:offset + size]
      if hashlib.sha256(region).hexdigest() != expected_sha: raise RuntimeError(f"native GA102 NET region {region_id} changed")
      writes.extend(struct.iter_unpack("<II", region))
    if len(writes) != 604 or self._stream_digest(writes) != "39bf609258f5ca4a9012d0f7993a3ab2ac321e99fe3947ed6338d30a831eaf59":
      raise RuntimeError("native GA102 NET write stream changed")
    if any(address % 4 or address >= 0x01000000 or 0x409000 <= address < 0x40A000 or
           0x41A000 <= address < 0x41B000 or 0x840000 <= address < 0x841000 for address, _ in writes):
      raise RuntimeError("native GA102 NET write stream escapes graphics registers")
    return writes

  def _parse_context_net(self) -> dict[int, list[tuple[int, int]]]:
    descriptors, regions = self._net_descriptors(), {}
    for region_id, (expected_size, expected_offset, expected_sha) in self.CONTEXT_NET_EXPECTED.items():
      size, offset = descriptors.get(region_id, (-1, -1))
      if (size, offset) != (expected_size, expected_offset):
        raise RuntimeError(f"native GA102 context NET region {region_id} changed")
      raw = self.net_image[offset:offset + size]
      if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise RuntimeError(f"native GA102 context NET region {region_id} changed")
      if region_id == 5: regions[region_id] = [(address, data) for address, _, data in struct.iter_unpack("<III", raw)]
      elif region_id == 34:
        regions[region_id] = [(address, data_hi << 32 | data_lo) for address, data_hi, data_lo in struct.iter_unpack("<III", raw)]
      else: regions[region_id] = list(struct.iter_unpack("<II", raw))
    if {region_id:len(entries) for region_id, entries in regions.items()} != {4:935, 5:577, 7:1701, 28:22, 34:66}:
      raise RuntimeError("native GA102 context NET entry counts changed")
    return regions

  def _init_ltc_zbc(self):
    selector = self.nvdev.rreg(self.LTC_ZBC_SELECTOR)
    if selector != 0: raise RuntimeError(f"native LTC ZBC selector is not clear: {selector:#x}")
    for index in range(1, 31):
      observed = self.nvdev.rreg(self.LTC_ZBC_SELECTOR)
      if observed != selector: raise RuntimeError(f"native LTC ZBC selector changed before color index {index}: {observed:#x}")
      selector = (observed & ~0x1F) | index
      self.nvdev.wreg(self.LTC_ZBC_SELECTOR, selector)
      for address in self.LTC_ZBC_COLOR: self.nvdev.wreg(address, 0)
    for index in range(1, 16):
      observed = self.nvdev.rreg(self.LTC_ZBC_SELECTOR)
      if observed != selector: raise RuntimeError(f"native LTC ZBC selector changed before depth index {index}: {observed:#x}")
      selector = (observed & ~0x0F) | index
      self.nvdev.wreg(self.LTC_ZBC_SELECTOR, selector)
      self.nvdev.wreg(self.LTC_ZBC_DEPTH, 0)
      observed = self.nvdev.rreg(self.LTC_ZBC_SELECTOR)
      if observed != selector: raise RuntimeError(f"native LTC ZBC selector changed before stencil index {index}: {observed:#x}")
      selector = (observed & ~0x0F) | index
      self.nvdev.wreg(self.LTC_ZBC_SELECTOR, selector)
      self.nvdev.wreg(self.LTC_ZBC_STENCIL, 0)
    if (actual:=self.nvdev.rreg(self.LTC_ZBC_SELECTOR)) != 0x1F:
      raise RuntimeError(f"native LTC ZBC selector did not finish at 0x1f: {actual:#x}")

  def _init_net(self):
    gate = self.nvdev.rreg(self.GR_GATE)
    sources = (self.nvdev.rreg(0x100C80), self.nvdev.rreg(0x100CC4), self.nvdev.rreg(0x100CC8), self.nvdev.rreg(0x100CCC))
    if gate != 0 or sources != (0x08018001, 0, 0, 0):
      raise RuntimeError(f"native GA102 GR NET prestate changed: gate={gate:#x} sources={sources}")
    writes = [(self.GR_GATE, gate & ~0x00010001), (0x418880, sources[0] & 0xF8001FFF), (0x418894, 0),
              (0x4188B4, sources[2]), (0x4188B8, sources[3]), (0x4188B0, sources[1]), *self._parse_net()]
    if len(writes) != 610 or self._stream_digest(writes) != "7e4712e8471ceec820fcff52792dc6e2702b9e5c9f1fc168b9f1543a8781411a":
      raise RuntimeError("native GA102 GR NET initialization stream changed")
    for address, value in writes: self.nvdev.wreg(address, value)

    deadline, sample = time.monotonic() + 2, None
    while time.monotonic() < deadline:
      sample = (self.nvdev.rreg(self.GR_STATUS_FLUSH), self.nvdev.rreg(self.MASTER_ENABLE), self.nvdev.rreg(self.GR_BUSY))
      if not sample[1] & 0x1000 or not sample[2] & 1: break
      time.sleep(0.001)
    else: raise TimeoutError(f"native GA102 GR did not become idle after NET initialization: {sample}")
    actual = tuple(self.nvdev.rreg(address) for address in (0x418880, 0x418894, 0x4188B4, 0x4188B8, 0x4188B0))
    if actual != (0x08000001, 0, 0, 0, 0): raise RuntimeError(f"native GA102 GPC MMU readback changed: {actual}")

  def _read_topology(self) -> GA102Topology:
    gpc_nr = self.nvdev.rreg(0x409604) & 0x1F
    _require(gpc_nr == 7, f"native GA102 GPC count changed: {gpc_nr}")
    topology = GA102Topology(
      tuple(self.nvdev.rreg(_gpc_unit(gpc, 0x2608)) & 0xFF for gpc in range(gpc_nr)),
      tuple(tuple(self.nvdev.rreg(_gpc_unit(gpc, 0x0C30 + ppc * 4)) & 0xFF for ppc in range(3)) for gpc in range(gpc_nr)),
      tuple(self.nvdev.rreg(_gpc_unit(gpc, 0x0C50)) & 0xF for gpc in range(gpc_nr)))
    _require(topology.tpc_nr == (5, 6, 6, 6, 6, 6, 6) and
             topology.ppc_tpc_mask == ((1, 10, 20),) + ((9, 18, 36),) * 6 and topology.swdx_pes == (7,) * 7 and
             topology.tpc_total == 41 and topology.tpc_max == 6 and topology.ppc_nr == (3,) * 7 and
             topology.ppc_total == 21 and topology.ppc_tpc_min == 1, f"native GA102 topology changed: {topology}")
    for gpc, row in enumerate(topology.ppc_tpc_mask):
      _require(not (sum(row) & ~((1 << topology.tpc_nr[gpc]) - 1)) and sum(row).bit_count() == topology.tpc_nr[gpc] and
               all(not (left & right) for index, left in enumerate(row) for right in row[index + 1:]),
               f"native GA102 GPC {gpc} PPC masks are not an exact TPC partition")
    return topology

  def _mask(self, address:int, mask:int, value:int) -> int:
    before = self.nvdev.rreg(address)
    _require(before != 0xFFFFFFFF, f"native GA102 register {address:#x} returned all ones")
    after = before & ~mask | value & mask
    self.nvdev.wreg(address, after)
    return before

  def _wait_clear(self, label:str, address:int, mask:int):
    deadline, observed = time.monotonic() + 2, None
    while time.monotonic() < deadline:
      observed = self.nvdev.rreg(address)
      _require(observed != 0xFFFFFFFF, f"native GA102 {label} returned all ones")
      if not observed & mask: return
      time.sleep(0.001)
    raise TimeoutError(f"native GA102 {label} timed out: {observed:#x}")

  def _wait_idle(self, label:str):
    deadline, sample = time.monotonic() + 2, None
    while time.monotonic() < deadline:
      sample = self.nvdev.rreg(self.GR_STATUS_FLUSH), self.nvdev.rreg(self.MASTER_ENABLE), self.nvdev.rreg(self.GR_BUSY)
      if not sample[1] & 0x1000 or not sample[2] & 1: return
      time.sleep(0.001)
    raise TimeoutError(f"native GA102 GR did not become idle during {label}: {sample}")

  def _context_patch_writes(self, topology:GA102Topology) -> list[tuple[int, int]]:
    if tuple(self.context_buffers) != ("pagepool", "bundle", "attrib", "unknown"):
      raise RuntimeError("native GA102 context buffers are incomplete")
    pagepool, bundle, attrib, unknown = (self.context_buffers[name].mapping.va_addr for name in self.context_buffers)
    writes = [
      (0x40800C, pagepool >> 8), (0x408010, 0x8007D800), (0x419004, pagepool >> 8), (0x419008, 0),
      (0x408004, bundle >> 8), (0x408008, 0x80000030), (0x418E24, bundle >> 8), (0x418E28, 0x80000030),
      (0x4064C8, 0x01801140),
      (0x418810, 0x80000000 | attrib >> 12), (0x419848, 0x10000000 | attrib >> 12),
      (0x419C2C, 0x10000000 | attrib >> 12), (0x419E00, attrib >> 12), (0x419E04, 0x80000000 | 0x829200 >> 7),
      (0x405830, 0x4A1), (0x40585C, 0x800), (0x4064C4, 0x0200FFFF),
    ]
    ppc_tpc_max, alpha_offset, attrib_offset = max(mask.bit_count() for row in topology.ppc_tpc_mask for mask in row), 0, 0xC00 * topology.tpc_total
    _require(ppc_tpc_max == 2, f"native GA102 PPC TPC maximum changed: {ppc_tpc_max}")
    for gpc, row in enumerate(topology.ppc_tpc_mask):
      for ppc, mask in enumerate(row):
        tpcs, base = mask.bit_count(), _ppc_unit(gpc, ppc, 0)
        alpha_size, attrib_size, gfxp_size = 0x800 * tpcs, 0x4A1 * ppc_tpc_max, 0xD28 * ppc_tpc_max
        writes.extend(((base + 0xC0, gfxp_size), (base + 0xF4, attrib_offset), (base + 0xF0, attrib_size)))
        attrib_offset += gfxp_size
        writes.extend(((base + 0xE4, alpha_size), (base + 0xF8, alpha_offset),
                       (0x418EA0 + (gpc * 3 + ppc) * 4, attrib_size)))
        alpha_offset += 0xC00 * tpcs
    _require(alpha_offset == 0xC00 * topology.tpc_total, "native GA102 alpha context geometry changed")
    writes.extend(((0x4181E4, 0x100), (0x41BEFC, 0x100),
                   (0x408070, unknown >> 8), (0x408074, 0x800), (0x419034, unknown >> 8), (0x408078, 0)))
    _require(len(writes) == 149, f"native GA102 context patch count changed: {len(writes)}")
    return writes

  @staticmethod
  def _floorsweep_writes(topology:GA102Topology) -> list[tuple[int, int]]:
    row_offset, tiles = _oneinit_tiles(topology)
    writes = [(_tpc_unit(gpc, _nonpes_aware_tpc(topology, gpc, tpc), 0x608), sm)
              for sm, (gpc, tpc) in enumerate(_oneinit_sm_id(topology))]
    gpc = 0
    for index in range(4):
      data = 0
      for nibble in range(8):
        if gpc >= topology.gpc_nr: break
        data, gpc = data | topology.tpc_nr[gpc] << (nibble * 4), gpc + 1
      writes.append((0x405870 + index * 4, data))
    padded_tiles = tiles + [0xFF] * (42 - len(tiles))
    writes.append((0x418BB8, topology.tpc_total << 8 | row_offset))
    for index in range(7):
      data = sum((padded_tiles[index * 6 + slot] & 0x1F) << (slot * 5) for slot in range(6))
      writes.extend(((0x418B08 + index * 4, data), (0x41BF00 + index * 4, data), (0x40780C + index * 4, data)))
    writes.append((0x41BFD0, topology.tpc_total << 8 | row_offset))
    for index, exponent in enumerate(range(1, 21, 4)):
      values = tuple((1 << (exponent + offset)) % topology.tpc_total for offset in range(4))
      writes.append((0x41BFB0 + index * 4, values[3] << 24 | values[2] << 16 | values[1] << 8 | values[0]))
    writes.extend(((0x4078BC, topology.tpc_total << 8 | row_offset), (0x406500, 0)))
    _require(len(writes) == 75, f"native GA102 floorsweep write count changed: {len(writes)}")
    return writes

  def _apply_icmd(self, entries:list[tuple[int, int]], count:int=1, pitch:int=1, wide:bool=False):
    self.nvdev.wreg(0x400208, 0x80000000)
    previous = None
    for address, data in entries:
      if data != previous:
        self.nvdev.wreg(0x400204, data & 0xFFFFFFFF)
        if wide: self.nvdev.wreg(0x40020C, data >> 32)
        previous = data
      repetitions = 1 if count == 64 and address & 0xFFFF == 0xE100 else count
      for offset in range(repetitions):
        target = address + offset * pitch
        self.nvdev.wreg(0x400200, target)
        if target & 0xFFFF == 0xE100: self._wait_idle("GO_IDLE bundle")
        self._wait_clear("context bundle", self.GR_STATUS_FLUSH, 4)
    self.nvdev.wreg(0x400208, 0)

  def _apply_methods(self, entries:list[tuple[int, int]]):
    previous = None
    for address_and_class, data in entries:
      if data != previous:
        self.nvdev.wreg(0x40448C, data)
        previous = data
      self.nvdev.wreg(0x404488, 0x80000000 | address_and_class)

  def _generate_main(self):
    regions, topology = self._parse_context_net(), self._read_topology()
    for address, value in regions[5]: self.nvdev.wreg(address, value)
    self._mask(0x419BD8, 0x700, 0)
    self.nvdev.wreg(0x419EA8, self.nvdev.rreg(0x504728) | 0x08000000)
    self._wait_idle("context register initialization")
    idle_timeout = self.nvdev.rreg(0x404154)
    _require(idle_timeout != 0xFFFFFFFF, "native GA102 context idle timeout returned all ones")
    self.nvdev.wreg(0x404154, 0)
    for address, value in self._context_patch_writes(topology): self.nvdev.wreg(address, value)
    self._mask(0x41980C, 0x10, 0x10)
    self._mask(0x41BE08, 0x04, 0x04)
    for address, value in self._floorsweep_writes(topology): self.nvdev.wreg(address, value)
    self._wait_idle("context floorsweep initialization")
    self._mask(0x400088, 0x00060000, 0)
    self._apply_icmd(regions[4])
    self._apply_icmd(regions[28], count=64, pitch=0x100000)
    self._apply_icmd(regions[34], wide=True)
    self._mask(0x400088, 0x00060000, 0x00060000)
    self.nvdev.wreg(0x404154, idle_timeout)
    self._apply_methods(regions[7])
    self._wait_idle("context method initialization")

  def _init_topology(self):
    if self.nvdev.rreg(self.GR_GATE) & 0x00010001: raise RuntimeError("native GA102 GR gate enabled before topology init")
    master, busy = self.nvdev.rreg(self.MASTER_ENABLE), self.nvdev.rreg(self.GR_BUSY)
    if master & 0x1000 and busy & 1: raise RuntimeError(f"native GA102 GR is busy before topology init: {(master, busy)}")
    topology, active_ltcs = self._read_topology(), self.nvdev.rreg(0x100800)
    _require(active_ltcs == 12, f"native GA102 active LTC count changed: {active_ltcs}")
    writes = _unit_a_writes(topology, active_ltcs)
    _require(len(writes) == 116 and self._stream_digest(writes) ==
             "91043022cf4b12cb117d2588612de4409294c8d698bc590e20ec7efab347f2d0", "native GA102 topology stream changed")
    for address, value in writes[:29]: self.nvdev.wreg(address, value)
    _require(self.nvdev.rreg(0x100800) == active_ltcs, "native GA102 active LTC count changed during topology init")
    self.nvdev.wreg(*writes[29])
    swdx = tuple(self.nvdev.rreg(_gpc_unit(gpc, 0x0C50)) & 0xF for gpc in range(topology.gpc_nr))
    _require(swdx == topology.swdx_pes, f"native GA102 SWDX PES topology changed: {swdx}")
    self.nvdev.wreg(*writes[30])
    for address, value in writes[31:]: self.nvdev.wreg(address, value)
    _require(not self.nvdev.rreg(self.GR_GATE) & 0x00010001, "native GA102 GR gate enabled during topology init")
    expected = {address:value for address, value in writes if address != _gpc_unit(0, 0x3018)}
    for gpc in range(topology.gpc_nr):
      address = _gpc_unit(gpc, 0x0910)
      _require(expected[address] == 0x00040029, "native GA102 GPC ZCULL target plan changed")
      expected[address] |= 0x40000000
    _require(expected[0x4188AC] == 12, "native GA102 active-LTC target plan changed")
    expected[0x4188AC] |= 0x40000000
    actual = {address:self.nvdev.rreg(address) for address in expected}
    if actual != expected:
      address = next(address for address in expected if actual[address] != expected[address])
      raise RuntimeError(f"native GA102 topology readback mismatch at {address:#x}: {actual[address]:#x} != {expected[address]:#x}")

  def _init_exceptions_zbc(self):
    _require(self.nvdev.rreg(self.GR_GATE) == 0, "native GA102 GR gate changed before exception init")
    topology = self._read_topology()
    tracked = (0x17E338, 0x41BCB4, 0x41814C, 0x418150, 0x418154, 0x418158,
               0x418198, 0x41819C, 0x4181A0, 0x4181A4, 0x4188A4)
    baselines = {address:self.nvdev.rreg(address) for address in tracked}
    expected_baselines = {0x17E338:0x1F, 0x41BCB4:1, 0x41814C:0, 0x418150:0, 0x418154:0, 0x418158:0,
                          0x418198:0, 0x41819C:0, 0x4181A0:0, 0x4181A4:0, 0x4188A4:0}
    _require(baselines == expected_baselines, f"native GA102 ZBC baselines changed: {baselines}")
    unit_b, operations = _unit_b_writes(topology, 0), _zbc_operations(baselines)
    zbc_writes = [(operation.address, operation.value) for operation in operations]
    _require(self._stream_digest(unit_b) == "ec7e60b3f00f79d5bea3952e276fc84b0304ccccdd1c591c1843657ef6edf91a" and
             len(operations) == 216 and sum(operation.mask is not None for operation in operations) == 70 and
             self._stream_digest(zbc_writes) == "186449906e13bf5114af73b3ad11a653b86f0b291581f3f54ca88882268ed1d5" and
             self._stream_digest(unit_b + zbc_writes) == "e0f3e51b1bff22cb4c29ac8dc5f85fc2c7bcc1d887f29b7dd808f7678c6129d5",
             "native GA102 exception/ZBC stream changed")

    for address, value in unit_b[:7]: self.nvdev.wreg(address, value)
    _require(self.nvdev.rreg(0x40584C) == 0x7F, "native GA102 0x40584c RMW baseline changed")
    self.nvdev.wreg(*unit_b[7])
    for address, value in unit_b[8:409]: self.nvdev.wreg(address, value)
    rop_source = self.nvdev.rreg(0x502C94)
    unit_b[409] = (0x41AC94, rop_source)
    self.nvdev.wreg(*unit_b[409])
    for address, value in unit_b[410:]: self.nvdev.wreg(address, value)
    for operation in operations:
      if operation.mask is not None:
        observed = self.nvdev.rreg(operation.address)
        _require(observed == operation.expected_before,
                 f"native GA102 ZBC RMW mismatch at {operation.address:#x}: {observed:#x} != {operation.expected_before:#x}")
      self.nvdev.wreg(operation.address, operation.value)

    expected = {0x400500:0x00010001, 0x41AC94:rop_source, 0x4188A4:0x03000000, 0x41BCB4:30, 0x17E338:3,
                0x41814C:0x81, 0x418150:0, 0x418154:0, 0x418158:0,
                0x418198:0x4081, 0x41819C:0, 0x4181A0:0, 0x4181A4:0,
                0x41BCEC:0, 0x41BCF0:0, 0x41BCF4:0, 0x41BCF8:0,
                0x418110:0, 0x418114:0x3F800000, 0x17E204:0xFF,
                0x41815C:0, 0x418160:1, 0x418164:0xFF}
    actual = {address:self.nvdev.rreg(address) for address in expected}
    if actual != expected:
      address = next(address for address in expected if actual[address] != expected[address])
      raise RuntimeError(f"native GA102 exception/ZBC readback mismatch at {address:#x}: "
                         f"{actual[address]:#x} != {expected[address]:#x}")

  def _wait_fecs_mailbox(self, label:str, mask:int=0xFFFFFFFF) -> int:
    deadline, observed = time.monotonic() + 2, None
    while time.monotonic() < deadline:
      observed = self.nvdev.rreg(self.FECS_MAILBOX0)
      _require(observed != 0xFFFFFFFF, f"native GA102 FECS {label} returned all ones")
      if observed & mask: return observed
    raise TimeoutError(f"native GA102 FECS {label} timed out: {observed}")

  def _init_ctxctl(self):
    prestate = {
      self.GR_GATE:0x00010001, self.GR_STATUS_FLUSH:0x014C0901, self.GR_BUSY:1,
      self.FECS_IRQSTAT:0, self.FECS_CPUCTL:0x50, self.FECS_OS:0, self.FECS_START:0,
      self.FECS_METHOD:0xBADF5545, self.FECS_MAILBOX0:0, self.FECS_MAILBOX1:0, self.FECS_CTXCTL_IRQSTAT:0,
      self.GPCCS_IRQSTAT:0, self.GPCCS_CPUCTL:0x50, self.GPCCS_OS:0, self.GPCCS_START:0, self.GPCCS_MAILBOX0:0, self.GPCCS_MAILBOX1:0,
    }
    actual = {address:self.nvdev.rreg(address) for address in prestate}
    _require(actual == prestate, f"native GA102 context-control prestate changed: {actual}")
    writes = list(self.CTXCTL_WRITES)
    _require(len(writes) == 17 and self._stream_digest(writes) ==
             "224bdca4eb39da6d5a2e3ea2dcce1a37dc47cccb069f7436b16f9b66ea793ee5",
             "native GA102 context-control stream changed")

    for address, value in writes[:3]: self.nvdev.wreg(address, value)
    _require(self.nvdev.rreg(self.GPCCS_CPUCTL) == 0x50, "native GA102 GPCCS changed before context-control start")
    self.nvdev.wreg(*writes[3])
    _require(self.nvdev.rreg(self.FECS_CPUCTL) == 0x50, "native GA102 FECS changed before context-control start")
    self.nvdev.wreg(*writes[4])
    ready = self._wait_fecs_mailbox("startup", mask=1)
    _require(ready == 1, f"native GA102 FECS startup reply changed: {ready:#x}")

    for address, value in writes[5:8]: self.nvdev.wreg(address, value)
    sizes = []
    for start, stop, label in ((8, 11, "main context size"), (11, 14, "ZCULL context size"), (14, 17, "PerfMon context size")):
      for address, value in writes[start:stop]: self.nvdev.wreg(address, value)
      sizes.append(self._wait_fecs_mailbox(label))
    _require(tuple(sizes) == self.CTXCTL_SIZES, f"native GA102 context-control sizes changed: {sizes}")

    expected = {self.GR_GATE:0x00010001, self.GR_STATUS_FLUSH:0, self.GR_BUSY:0,
                self.FECS_IRQSTAT:0, self.FECS_OS:0, self.FECS_START:0, self.FECS_METHOD_DATA:0,
                self.FECS_METHOD:0xBADF5545, self.FECS_MAILBOX0:sizes[-1], self.FECS_MAILBOX1:0,
                self.FECS_CTXCTL_IRQSTAT:0, self.GPCCS_IRQSTAT:0, self.GPCCS_OS:0, self.GPCCS_START:0,
                self.GPCCS_MAILBOX0:0, self.GPCCS_MAILBOX1:0}
    actual = {address:self.nvdev.rreg(address) for address in expected}
    _require(actual == expected, f"native GA102 context-control readback changed: {actual}")
    cpuctl = self.nvdev.rreg(self.FECS_CPUCTL), self.nvdev.rreg(self.GPCCS_CPUCTL)
    _require(all(value & 0xFFFFFFDF == 0x40 for value in cpuctl), f"native GA102 context-control Falcons stopped: {cpuctl}")
    self.main_size, self.zcull_size, self.pm_size = sizes

  def _alloc_ctx_buffers(self):
    allocations = [(name, requested_size, round_up(requested_size, 0x1000),
                    self.nvdev.mm.palloc(requested_size, align=align, zero=False))
                   for name, requested_size, align in self.CONTEXT_BUFFER_LAYOUT]
    self.context_buffers = {}
    for name, requested_size, mapped_size, paddr in allocations:
      va_addr = self.nvdev.mm.alloc_vaddr(mapped_size, 0x1000)
      mapping = self.nvdev.mm.map_range(va_addr, mapped_size, [(paddr, mapped_size)], AddrSpace.PHYS, privileged=True, kind=0)
      self.context_buffers[name] = GA102ContextBuffer(requested_size, paddr, mapping)

  def _power_mode(self, value:int, label:str):
    self.nvdev.wreg(0x404170, value)
    self._wait_clear(label, 0x404170, 0x10)

  def _fecs_method(self, label:str, instance:int, clear_mask:int, success:int, failure:int):
    self._mask(self.FECS_MAILBOX0, clear_mask, 0)
    self.nvdev.wreg(self.FECS_METHOD_DATA, instance)
    self.nvdev.wreg(self.FECS_METHOD, {"bind pointer":3, "golden save":9}[label])
    deadline, observed = time.monotonic() + 2, None
    while time.monotonic() < deadline:
      observed = self.nvdev.rreg(self.FECS_MAILBOX0)
      _require(observed != 0xFFFFFFFF, f"native GA102 FECS {label} returned all ones")
      if observed & failure: raise RuntimeError(f"native GA102 FECS {label} failed: {observed:#x}")
      if observed & success: return
      time.sleep(0.001)
    raise TimeoutError(f"native GA102 FECS {label} timed out: {observed:#x}")

  def generate_golden(self, channel) -> bytes:
    if self.golden_context is not None: return self.golden_context
    _require(channel.instance_paddr & 0xFFF == 0, f"native GA102 channel instance is unaligned: {channel.instance_paddr:#x}")
    self._wait_idle("golden context preflight")
    cpuctl = self.nvdev.rreg(self.FECS_CPUCTL), self.nvdev.rreg(self.GPCCS_CPUCTL)
    _require(all(value & 0xFFFFFFDF == 0x40 for value in cpuctl), f"native GA102 context-control Falcons stopped: {cpuctl}")

    self._power_mode(0x12, "forced power mode")
    self.nvdev.wreg(0x409614, 0x10)
    self.nvdev.wreg(0x41A614, 0x20)
    time.sleep(0.00001)
    self.nvdev.wreg(0x409614, 0x110)
    self.nvdev.wreg(0x41A614, 0xA20)
    time.sleep(0.00001)
    _require(self.nvdev.rreg(0x409614) == 0x110, "native GA102 FECS reset did not latch")
    self._power_mode(0x10, "automatic power mode")
    self.nvdev.wreg(0x40802C, 1)

    scratch_size, mapping, pointer_set = round_up(0x80000 + self.main_size, 0x1000), None, False
    scratch_paddr = self.nvdev.mm.palloc(0x80000 + self.main_size, align=1, zero=True)
    try:
      scratch_va = self.nvdev.mm.alloc_vaddr(scratch_size, 0x1000)
      mapping = self.nvdev.mm.map_range(scratch_va, scratch_size, [(scratch_paddr, scratch_size)], AddrSpace.PHYS, kind=0)
      context_va = scratch_va + 0x80000
      self.nvdev._write_vram(channel.instance_paddr + 0x210, struct.pack("<II", (context_va | 4) & 0xFFFFFFFF, context_va >> 32))
      pointer_set = True
      instance = 0x80000000 | channel.instance_paddr >> 12
      self._fecs_method("bind pointer", instance, 0x30, 0x10, 0x20)
      for offset, value in ((0x1C, 1), (0x20, 0), (0x28, 0), (0x2C, 0)):
        self.nvdev._write_vram(scratch_paddr + offset, struct.pack("<I", value))
      self._generate_main()
      self._fecs_method("golden save", instance, 0x03, 0x01, 0x02)
      self._mask(0x409B00, 0x80000000, 0)
      golden = bytes(self.nvdev.vram[scratch_paddr + 0x80000:scratch_paddr + 0x80000 + self.main_size])
      _require(len(golden) == self.main_size and any(golden), "native GA102 FECS produced an empty golden context")
      self.golden_context = golden
      return golden
    finally:
      if pointer_set: self.nvdev._write_vram(channel.instance_paddr + 0x210, bytes(8))
      if mapping is not None: self.nvdev.mm.vfree(mapping)

  def bind_compute_context(self, channel):
    if channel.chid in self.channel_contexts: return
    golden = self.generate_golden(channel)
    _require(len(golden) == self.main_size, f"native GA102 golden context size changed: {len(golden):#x}")
    patches = self._context_patch_writes(self._read_topology())
    _require(len(patches) * 8 <= 0x1000, f"native GA102 context patch list is too large: {len(patches)}")

    mmio_paddr = self.nvdev.mm.palloc(0x1000, align=0x100, zero=False)
    mmio_va = self.nvdev.mm.alloc_vaddr(0x1000, 0x1000)
    mmio = self.nvdev.mm.map_range(mmio_va, 0x1000, [(mmio_paddr, 0x1000)], AddrSpace.PHYS, privileged=True, kind=0)
    context_size = round_up(self.main_size, 0x1000)
    context_paddr = self.nvdev.mm.palloc(context_size, align=0x1000, zero=False)
    context_va = self.nvdev.mm.alloc_vaddr(context_size, 0x1000)
    context = self.nvdev.mm.map_range(context_va, context_size, [(context_paddr, context_size)],
                                     AddrSpace.PHYS, privileged=True, kind=0)
    try:
      self.nvdev._write_vram(mmio_paddr, b"".join(struct.pack("<II", *entry) for entry in patches))
      image = bytearray(golden)
      struct.pack_into("<I", image, 0x10, len(patches))
      struct.pack_into("<Q", image, 0x14, mmio_va)
      for offset, value in ((0x1C, 1), (0x20, 0), (0x28, 0), (0x2C, 0), (0xF4, 0), (0xF8, 0)):
        struct.pack_into("<I", image, offset, value)
      self.nvdev._write_vram(context_paddr, image)

      flags, = struct.unpack("<I", bytes(self.nvdev.vram[channel.instance_paddr + 0xAC:channel.instance_paddr + 0xB0]))
      _require(not flags & 0x10000, f"native GA102 channel {channel.chid} already has a GR context")
      self.nvdev._write_vram(channel.instance_paddr + 0x210, struct.pack("<Q", context_va | 4))
      self.nvdev._write_vram(channel.instance_paddr + 0xAC, struct.pack("<I", flags | 0x10000))
      self.nvdev.wreg(0x70000, 1)
      self._wait_clear("context BAR flush", 0x70000, 2)
      self.channel_contexts[channel.chid] = (mmio, context)
    except BaseException:
      self.nvdev.mm.vfree(context)
      self.nvdev.mm.vfree(mmio)
      raise

  def bind_channel_group_context(self, channels):
    sources = [channel for channel in channels if channel.chid in self.channel_contexts]
    if not sources: return
    _require(len(sources) == 1, f"native GA102 channel group has {len(sources)} GR contexts")
    source = sources[0]
    source_flags, = struct.unpack("<I", bytes(self.nvdev.vram[source.instance_paddr + 0xAC:source.instance_paddr + 0xB0]))
    source_pointer = bytes(self.nvdev.vram[source.instance_paddr + 0x210:source.instance_paddr + 0x218])
    pointer, = struct.unpack("<Q", source_pointer)
    _require(source_flags & 0x10000 and pointer & 4,
             f"native GA102 channel {source.chid} has an invalid GR context: flags={source_flags:#x}, pointer={pointer:#x}")

    changed = False
    for channel in channels:
      flags, = struct.unpack("<I", bytes(self.nvdev.vram[channel.instance_paddr + 0xAC:channel.instance_paddr + 0xB0]))
      existing, = struct.unpack("<Q", bytes(self.nvdev.vram[channel.instance_paddr + 0x210:channel.instance_paddr + 0x218]))
      if channel is source: continue
      _require(not flags & 0x10000 and existing == 0,
               f"native GA102 channel {channel.chid} already has a GR context: flags={flags:#x}, pointer={existing:#x}")
      self.nvdev._write_vram(channel.instance_paddr + 0x210, source_pointer)
      self.nvdev._write_vram(channel.instance_paddr + 0xAC, struct.pack("<I", flags | 0x10000))
      changed = True
    if changed:
      self.nvdev.wreg(0x70000, 1)
      self._wait_clear("channel-group context BAR flush", 0x70000, 2)

  def init_hw(self):
    self._init_ltc_zbc()
    self._init_net()
    self._init_topology()
    self._init_exceptions_zbc()
    self._init_ctxctl()
    self._alloc_ctx_buffers()
