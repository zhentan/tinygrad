from __future__ import annotations

import dataclasses, struct
from typing import Any, Callable

from tinygrad.helpers import fetch_fw


WPR_HEADER_SIZE, LSB_HEADER_SIZE, BL_DESC_SIZE = 32, 4944, 84
WPR_RAW_BYTES, WPR_HALF_BYTES, WPR_ALLOCATION_BYTES = 0x2C600, 0x40000, 0x80000
SYSTEM_GPU_BASE, SYSTEM_CONTROLLER_BASE, SYSTEM_STAGE_BYTES = 0x224000, 0x33000, 0x1C000
WPR_RELOCATIONS = (45604, 45632, 72996, 73024, 181540, 181568)

GA102_FIRMWARE = {
  "acr/ucode_ahesasc.bin": "ca69d947ccfa5d29e45e54825d0b6e9a41c591ad83784519568eb2eda94daa0e",
  "acr/ucode_asb.bin": "27ee1438ff9eb11b8ade47afdebcecaa135c3e52e98e164b3820d7de0ecc2666",
  "acr/ucode_unload.bin": "ef413c92a1cfbe645c4ab2e8130df79e7d778999a58994d5a5303e380996256b",
  "sec2/desc.bin": "1fd6fd67dc81d975cb7b129e2a405f61009a6665b8faf747aeb1bfc5c1223701",
  "sec2/hs_bl_sig.bin": "1e24000c438e12a85558aaab86e514f814bdd89a4ee294dd3425eb27a32ab436",
  "sec2/image.bin": "e103b57312fdfc9a8b6a9f67f8389b7da0f214a5c478c2898e3fb434bde54dce",
  "sec2/sig.bin": "d2485ddbdaeca1f9a4cd4a49fc2bf19f4331f47a1f86d1b97cb9cf18c6407fe6",
  "gr/NET_img.bin": "56b3b302589b9e3bbb7d6ac8f5e36840e4297773c3900779c27c0505c1e605d5",
  "gr/fecs_bl.bin": "b1153ed2bb2c7b3b19a2299d6f9fda5715b8b2833389ef07f8f268c41f904233",
  "gr/fecs_sig.bin": "ecb366986095eeb5f7326bfa21397b40991d0615e80cdbef7bb63691da93bcfc",
  "gr/gpccs_bl.bin": "aa2ad555bc8c88396b3bcc6cec9644ecd5b114c4b974b8f887f572da3287b22f",
  "gr/gpccs_sig.bin": "c298ede1d14a589e64760e6723bf826e00592a6d235b48422dc45f45cac390c8",
  "nvdec/scrubber.bin": "abb38039f8195852a912fca64b2e2c97c17fa71197d57166190f0081d717afbd",
}


@dataclasses.dataclass(frozen=True)
class NativeHelper:
  name: str
  payload: bytes
  allocated_bytes: int
  gpu_address: int
  controller_address: int
  falcon: str
  imem_source_offset: int
  imem_bytes: int
  dmem_source_offset: int
  dmem_bytes: int
  dmem_signature_offset: int
  boot_address: int
  engine_id: int
  ucode_id: int


@dataclasses.dataclass(frozen=True)
class GA102ACRPackage:
  shadow_start: int
  protected_start: int
  protected_end: int
  wpr_shadow: bytes
  wpr_allocation: bytes
  system_stage: bytes
  helpers: tuple[NativeHelper, ...]
  gr_net: bytes


def _require(condition: bool, message: str) -> None:
  if not condition: raise ValueError(message)


def _align(value: int, boundary: int) -> int:
  _require(boundary > 0 and boundary & (boundary - 1) == 0, "invalid alignment")
  return (value + boundary - 1) & -boundary


def _checked_range(offset: int, size: int, total: int, label: str) -> None:
  _require(offset >= 0 and size >= 0 and offset + size <= total,
           f"{label} outside blob: offset={offset:#x} size={size:#x} total={total:#x}")


def load_ga102_firmware(loader: Callable[[str, str, str], bytes]=fetch_fw) -> dict[str, bytes]:
  return {name: loader(f"nvidia/ga102/{name.rsplit('/', 1)[0]}", name.rsplit('/', 1)[1], digest)
          for name, digest in GA102_FIRMWARE.items()}


def _parse_netlist(blob: bytes) -> dict[int, bytes]:
  version, count = struct.unpack_from("<II", blob)
  _require(version == 0 and count == 50, f"unexpected NET header: version={version} count={count}")
  _checked_range(8, count * 12, len(blob), "NET region table")
  regions: dict[int, bytes] = {}
  for index in range(count):
    region_id, size, offset = struct.unpack_from("<III", blob, 8 + index * 12)
    _require(region_id not in regions, f"duplicate NET region {region_id}")
    _checked_range(offset, size, len(blob), f"NET region {region_id}")
    regions[region_id] = blob[offset:offset + size]
  _require({0, 1, 2, 3}.issubset(regions), "NET lacks FECS/GPCCS regions")
  return regions


def _build_gr_image(bootloader: bytes, inst: bytes, data: bytes) -> tuple[bytes, dict[str, int]]:
  magic, version, bin_size, header_offset, header_size = struct.unpack_from("<5I", bootloader)
  _require((magic, version, bin_size, header_offset, header_size) == (0x3B1D14F0, 1, 0, 32, 256),
           "unexpected GR bootloader header")
  bootloader_size, code_size, data_size = _align(header_size, 256), _align(len(inst), 256), _align(len(data), 256)
  _checked_range(header_offset, bootloader_size, len(bootloader), "GR bootloader payload")
  image = bytearray(bootloader_size + code_size + data_size)
  image[:bootloader_size] = bootloader[header_offset:header_offset + bootloader_size]
  image[bootloader_size:bootloader_size + len(inst)] = inst
  image[bootloader_size + code_size:bootloader_size + code_size + len(data)] = data
  return bytes(image), {
    "bootloader_size": bootloader_size, "bootloader_imem_offset": 0, "app_start_offset": bootloader_size,
    "app_size": code_size + data_size, "app_imem_entry": 0, "app_resident_code_offset": 0,
    "app_resident_code_size": code_size, "app_resident_data_offset": code_size, "app_resident_data_size": data_size,
    "app_imem_offset": 0, "app_dmem_offset": 0, "ucode_size": bootloader_size + code_size, "data_size": data_size,
  }


def _parse_sec2_descriptor(blob: bytes, image_size: int) -> dict[str, int]:
  _require(len(blob) >= 656 and not any(blob[656:]), "unexpected SEC2 descriptor size/trailing bytes")
  descriptor_size, described_size = struct.unpack_from("<II", blob)
  _require(descriptor_size == 656 and described_size == image_size, "unexpected SEC2 descriptor header")
  (secure_bootloader, bootloader_start_offset, bootloader_size, bootloader_imem_offset, bootloader_entry_point,
   app_start_offset, app_size, app_imem_offset, app_imem_entry, app_dmem_offset, resident_code_offset,
   resident_code_size, resident_data_offset, resident_data_size) = struct.unpack_from("<14I", blob, 80)
  for offset, size, label in ((bootloader_start_offset, bootloader_size, "SEC2 bootloader"),
                              (app_start_offset, app_size, "SEC2 app"),
                              (app_start_offset + resident_code_offset, resident_code_size, "SEC2 resident code"),
                              (app_start_offset + resident_data_offset, resident_data_size, "SEC2 resident data")):
    _checked_range(offset, size, image_size, label)
  aligned_bootloader, aligned_app = _align(bootloader_size, 256), _align(app_size, 256)
  ucode_size = _align(resident_data_offset, 256) + aligned_bootloader
  return {
    "secure_bootloader": secure_bootloader, "bootloader_size": aligned_bootloader,
    "bootloader_imem_offset": bootloader_imem_offset, "bootloader_entry_point": bootloader_entry_point,
    "app_start_offset": app_start_offset, "app_size": aligned_app, "app_imem_entry": app_imem_entry,
    "app_resident_code_offset": resident_code_offset, "app_resident_code_size": resident_code_size,
    "app_resident_data_offset": resident_data_offset, "app_resident_data_size": resident_data_size,
    "app_imem_offset": app_imem_offset, "app_dmem_offset": app_dmem_offset, "ucode_size": ucode_size,
    "data_size": aligned_app + aligned_bootloader - ucode_size,
  }


def _parse_hs_container(blob: bytes, name: str) -> dict[str, Any]:
  magic, version, bin_size, header_offset, data_offset, data_size = struct.unpack_from("<6I", blob)
  _require(magic == 0x10DE and version == 1 and bin_size == len(blob), f"invalid HS container {name}")
  _checked_range(data_offset, data_size, len(blob), f"{name} payload")
  (sig_prod_offset, sig_prod_size, patch_loc_ptr, patch_sig_ptr, metadata_offset, metadata_size,
   num_sig_ptr, load_header_offset, load_header_size) = struct.unpack_from("<9I", blob, header_offset)
  patch_loc, = struct.unpack_from("<I", blob, patch_loc_ptr)
  patch_sig, = struct.unpack_from("<I", blob, patch_sig_ptr)
  signature_count, = struct.unpack_from("<I", blob, num_sig_ptr)
  _require(signature_count > 0 and sig_prod_size % signature_count == 0, f"invalid signatures {name}")
  signature_size = sig_prod_size // signature_count
  _checked_range(sig_prod_offset + patch_sig, sig_prod_size, len(blob), f"{name} signatures")
  _checked_range(metadata_offset, metadata_size, len(blob), f"{name} metadata")
  fuse_version, engine_id, ucode_id = struct.unpack_from("<3I", blob, metadata_offset)
  os_code_offset, os_code_size, os_data_offset, os_data_size, app_count = struct.unpack_from("<5I", blob, load_header_offset)
  _require(app_count == 1 and load_header_size >= 36, f"unexpected app count/header size {name}")
  app_offset, app_size, app_data_offset, app_data_size = struct.unpack_from("<4I", blob, load_header_offset + 20)
  for offset, size, label in ((os_code_offset, os_code_size, "OS code"), (os_data_offset, os_data_size, "OS data"),
                              (app_offset, app_size, "app"), (app_data_offset, app_data_size, "app data"),
                              (patch_loc, signature_size, "signature destination")):
    _checked_range(offset, size, data_size, f"{name} {label}")
  return {
    "payload": blob[data_offset:data_offset + data_size], "signatures": blob[sig_prod_offset + patch_sig:sig_prod_offset + patch_sig + sig_prod_size],
    "signature_count": signature_count, "signature_size": signature_size, "patch_offset": patch_loc, "fuse_version": fuse_version,
    "engine_id": engine_id, "ucode_id": ucode_id, "imem_source_offset": app_offset, "imem_bytes": app_size,
    "dmem_source_offset": os_data_offset, "dmem_bytes": os_data_size, "dmem_signature_offset": patch_loc - os_data_offset,
    "boot_address": app_offset,
  }


def _patch_hs_signature(parsed: dict[str, Any], signature_index: int=0) -> bytes:
  _require(0 <= signature_index < parsed["signature_count"], "HS signature index outside range")
  start, size = signature_index * parsed["signature_size"], parsed["signature_size"]
  payload, signature = bytearray(parsed["payload"]), parsed["signatures"][start:start + size]
  _require(len(signature) == size, "truncated HS signature")
  payload[parsed["patch_offset"]:parsed["patch_offset"] + size] = signature
  return bytes(payload)


def _parse_sec2_hsbl(blob: bytes, signature_index: int=0) -> dict[str, Any]:
  magic, version, bin_size, header_offset = struct.unpack_from("<4I", blob)
  _require(magic == 0x10DE and version == 1 and bin_size == len(blob), "invalid SEC2 HSBL")
  sig_prod_offset, sig_prod_size, _patch_loc_ptr, patch_sig_ptr, metadata_offset, metadata_size, num_sig_ptr = \
    struct.unpack_from("<7I", blob, header_offset)
  patch_sig, = struct.unpack_from("<I", blob, patch_sig_ptr)
  count, = struct.unpack_from("<I", blob, num_sig_ptr)
  _require(count > 0 and sig_prod_size % count == 0, "invalid SEC2 HSBL signatures")
  signature_size = sig_prod_size // count
  _checked_range(sig_prod_offset + patch_sig, sig_prod_size, len(blob), "SEC2 HSBL signatures")
  _checked_range(metadata_offset, metadata_size, len(blob), "SEC2 HSBL metadata")
  _require(0 <= signature_index < count, "SEC2 HSBL signature index outside range")
  start, fuse_version, engine_id, ucode_id = sig_prod_offset + patch_sig + signature_index * signature_size, \
    *struct.unpack_from("<3I", blob, metadata_offset)
  return {"signature": blob[start:start + signature_size], "fuse_version": fuse_version, "engine_id": engine_id, "ucode_id": ucode_id}


def _pack_wpr_header(falcon_id: int, lsb_offset: int, bin_version: int) -> bytes:
  return struct.pack("<HHI6I", 2, 2, WPR_HEADER_SIZE, falcon_id, lsb_offset, 1, 1, bin_version, 1)


def _pack_bl_descriptor(entry: dict[str, int], image_offset: int, ctx_dma: int, argc: int=0, argv: int=0) -> bytes:
  code_base = image_offset + entry["app_start_offset"] + entry["app_resident_code_offset"]
  data_base = image_offset + entry["app_start_offset"] + entry["app_resident_data_offset"]
  descriptor = struct.pack("<8IIQ5IQ3I", *([0] * 8), ctx_dma, code_base, entry["app_resident_code_offset"],
                           entry["app_resident_code_size"], 0, 0, entry["app_imem_entry"], data_base,
                           entry["app_resident_data_size"], argc, argv)
  _require(len(descriptor) == BL_DESC_SIZE, "bootloader descriptor size mismatch")
  return descriptor


def _build_lsb(signature: bytes, entry: dict[str, int], image_offset: int, bld_offset: int,
               flags: int, secure_hsbl: dict[str, Any]|None) -> bytes:
  _require(len(signature) == 2248, "LSF signature must be 2248 bytes")
  lsb = bytearray(LSB_HEADER_SIZE)
  struct.pack_into("<HHI", lsb, 0, 4, 2, LSB_HEADER_SIZE)
  lsb[8:8 + len(signature)] = signature
  struct.pack_into("<18I", lsb, 2256, image_offset, entry["ucode_size"], entry["data_size"], entry["bootloader_size"],
                   entry["bootloader_imem_offset"], bld_offset, _align(BL_DESC_SIZE, 256), 0,
                   entry["app_start_offset"] + entry["app_resident_code_offset"], _align(entry["app_resident_code_size"], 256),
                   entry["app_start_offset"] + entry["app_resident_data_offset"], _align(entry["app_resident_data_size"], 256),
                   entry["app_imem_offset"], entry["app_dmem_offset"], flags, 0, 0, 0)
  if secure_hsbl is not None:
    struct.pack_into("<B3xHHIII", lsb, 2328, 1, 0, 1, secure_hsbl["engine_id"], secure_hsbl["ucode_id"], secure_hsbl["fuse_version"])
    signature = secure_hsbl["signature"]
    _require(len(signature) <= 512, "SEC2 HSBL signature exceeds PKC field")
    lsb[2348:2348 + len(signature)] = signature
  return bytes(lsb)


def _build_relative_wpr(blobs: dict[str, bytes]) -> bytes:
  regions = _parse_netlist(blobs["gr/NET_img.bin"])
  fecs_image, fecs = _build_gr_image(blobs["gr/fecs_bl.bin"], regions[1], regions[0])
  gpccs_image, gpccs = _build_gr_image(blobs["gr/gpccs_bl.bin"], regions[3], regions[2])
  fecs["bootloader_imem_offset"], gpccs["bootloader_imem_offset"] = 0x7E00, 0x3400
  sec2_image = blobs["sec2/image.bin"]
  sec2 = _parse_sec2_descriptor(blobs["sec2/desc.bin"], len(sec2_image))
  sec2_hsbl = _parse_sec2_hsbl(blobs["sec2/hs_bl_sig.bin"])
  _require((sec2_hsbl["fuse_version"], sec2_hsbl["engine_id"], sec2_hsbl["ucode_id"]) == (1, 1, 5),
           "unexpected SEC2 secure bootloader identity")
  specs = ((2, fecs_image, fecs, blobs["gr/fecs_sig.bin"], 0, 0, 0, None),
           (3, gpccs_image, gpccs, blobs["gr/gpccs_sig.bin"], 0x8, 0, 0, None),
           (7, sec2_image, sec2, blobs["sec2/sig.bin"], 0, 6, 1, sec2_hsbl))
  cursor, layouts = _align(21 * WPR_HEADER_SIZE, 256) + 0x100, []
  for falcon_id, image, entry, signature, flags, ctx_dma, argc, hsbl in specs:
    lsb_offset = _align(cursor, 256)
    image_offset = _align(lsb_offset + LSB_HEADER_SIZE, 4096)
    bld_offset = _align(image_offset + len(image), 256)
    cursor = bld_offset + _align(BL_DESC_SIZE, 256)
    layouts.append((falcon_id, image, entry, signature, flags, ctx_dma, argc, hsbl, lsb_offset, image_offset, bld_offset))
  _require(cursor == WPR_RAW_BYTES, f"raw WPR size mismatch: {cursor:#x}")
  wpr = bytearray(WPR_HALF_BYTES)
  for index, item in enumerate(layouts):
    wpr[index * WPR_HEADER_SIZE:(index + 1) * WPR_HEADER_SIZE] = _pack_wpr_header(item[0], item[8], struct.unpack_from("<I", item[3], 2176)[0])
  wpr[3 * WPR_HEADER_SIZE:4 * WPR_HEADER_SIZE] = _pack_wpr_header(0xFFFFFFFF, layouts[-1][8], struct.unpack_from("<I", layouts[-1][3], 2176)[0])
  struct.pack_into("<5I", wpr, 0x300, 0x00020003, 0x14, 0xFFFFFFFF, 0, 0)
  for falcon_id, image, entry, signature, flags, ctx_dma, argc, hsbl, lsb_offset, image_offset, bld_offset in layouts:
    lsb = _build_lsb(signature, entry, image_offset, bld_offset, flags, hsbl)
    descriptor = _pack_bl_descriptor(entry, image_offset, ctx_dma, argc, 0x01000000 if falcon_id == 7 else 0)
    wpr[lsb_offset:lsb_offset + len(lsb)], wpr[image_offset:image_offset + len(image)] = lsb, image
    wpr[bld_offset:bld_offset + len(descriptor)] = descriptor
  _require(not any(wpr[WPR_RAW_BYTES:]), "nonzero data beyond raw WPR")
  return bytes(wpr)


def _build_helpers(blobs: dict[str, bytes]) -> tuple[tuple[NativeHelper, ...], bytes]:
  specs = (("AHESASC", "acr/ucode_ahesasc.bin", 0xE000, "SEC2"), ("ASB", "acr/ucode_asb.bin", 0x7000, "GSP"),
           ("unload", "acr/ucode_unload.bin", 0x4000, "SEC2"), ("VPR_SCRUBBER", "nvdec/scrubber.bin", 0x2000, "NVDEC0"))
  helpers, gpu_address, controller_address = [], SYSTEM_GPU_BASE, SYSTEM_CONTROLLER_BASE
  for name, firmware, allocated_bytes, falcon in specs:
    parsed = _parse_hs_container(blobs[firmware], name)
    _require(parsed["fuse_version"] == 2, f"unexpected {name} fuse version")
    payload = _patch_hs_signature(parsed)
    _require(len(payload) <= allocated_bytes, f"{name} exceeds allocation")
    helpers.append(NativeHelper(name, payload, allocated_bytes, gpu_address, controller_address, falcon,
                                parsed["imem_source_offset"], parsed["imem_bytes"], parsed["dmem_source_offset"],
                                parsed["dmem_bytes"], parsed["dmem_signature_offset"], parsed["boot_address"],
                                parsed["engine_id"], parsed["ucode_id"]))
    gpu_address, controller_address = gpu_address + allocated_bytes, controller_address + allocated_bytes
  _require((gpu_address + 0x1000, controller_address + 0x1000) == (0x240000, 0x4F000), "SYSTEM allocation extent mismatch")
  stage = bytearray(SYSTEM_STAGE_BYTES)
  for helper in helpers:
    offset = helper.gpu_address - SYSTEM_GPU_BASE
    stage[offset:offset + len(helper.payload)] = helper.payload
  return tuple(helpers), bytes(stage)


def _allocation_bounds(shadow_start: int) -> tuple[int, int, int]:
  _require(shadow_start >= 0 and shadow_start % WPR_HALF_BYTES == 0, "shadow WPR must be nonnegative and 256 KiB aligned")
  protected_start, protected_end = shadow_start + WPR_HALF_BYTES, shadow_start + WPR_ALLOCATION_BYTES
  _require(protected_end <= 24 << 30, "WPR allocation escaped 24 GiB VRAM")
  return shadow_start, protected_start, protected_end


def build_ga102_acr_package(shadow_start: int, blobs: dict[str, bytes]|None=None) -> GA102ACRPackage:
  shadow_start, protected_start, protected_end = _allocation_bounds(shadow_start)
  blobs = load_ga102_firmware() if blobs is None else blobs
  _require(set(blobs) == set(GA102_FIRMWARE), "incomplete or unexpected GA102 firmware set")
  shadow = bytearray(_build_relative_wpr(blobs))
  for offset in WPR_RELOCATIONS:
    value, = struct.unpack_from("<Q", shadow, offset)
    _require(value + protected_start < 1 << 64, "WPR relocation overflow")
    struct.pack_into("<Q", shadow, offset, value + protected_start)
  helpers, stage_bytes = _build_helpers(blobs)
  stage, fields = bytearray(stage_bytes), 33024 + 528
  struct.pack_into("<I", stage, fields, 1)
  struct.pack_into("<I", stage, fields + 12, 2)
  struct.pack_into("<7I", stage, fields + 16, protected_start >> 8, protected_end >> 8, 1, 0xF, 0xC, 0x2, shadow_start >> 8)
  return GA102ACRPackage(shadow_start, protected_start, protected_end, bytes(shadow), bytes(shadow) + bytes(WPR_HALF_BYTES),
                         bytes(stage), helpers, blobs["gr/NET_img.bin"])
