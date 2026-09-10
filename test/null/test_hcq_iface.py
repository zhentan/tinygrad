import unittest, array, time, struct, ctypes
from unittest.mock import Mock, patch
from tinygrad.dtype import dtypes
from tinygrad.helpers import mv_address
from tinygrad.uop.ops import UOp, Ops
from tinygrad.runtime.support.hcq import MMIOInterface
from tinygrad.runtime.support.memory import BumpAllocator
from tinygrad.runtime.support.system import System, USBPCIDevice
from tinygrad.runtime.support.usb import USB3, USBMMIOInterface, CustomASM24Controller, alloc_cbuffer
from tinygrad.runtime.support.nv.usb import usb_stream
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

  def test_sysmem_returns_every_page(self):
    dev = object.__new__(USBPCIDevice)
    dev.usb, dev.sram = Mock(), BumpAllocator(0x80000, wrap=False)
    dev.sram.alloc(0x40000)

    view, paddrs = dev.alloc_sysmem(0x3000)

    self.assertEqual((view.addr, view.nbytes), (0x4f000, 0x3000))
    self.assertEqual(paddrs, [0x240000, 0x241000, 0x242000])

if __name__ == "__main__":
  unittest.main()
