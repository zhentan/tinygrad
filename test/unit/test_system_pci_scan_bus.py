import sys
import pytest
from unittest.mock import Mock

@pytest.mark.skipif(sys.platform != "linux", reason="uses linux sysfs layout")
def test_pci_scan_bus_filters_vendor(monkeypatch):
  import tinygrad.runtime.support.system as system

  fake = {
    "/sys/bus/pci/devices/0000:00:01.0/vendor": "0x1234",
    "/sys/bus/pci/devices/0000:00:01.0/device": "0x1111",
    "/sys/bus/pci/devices/0000:00:02.0/vendor": "0xabcd",
    "/sys/bus/pci/devices/0000:00:02.0/device": "0x1111",
  }

  class FakeFileIOInterface:
    def __init__(self, path, *args, **kwargs):
      self.path = path

    def listdir(self):
      assert self.path == "/sys/bus/pci/devices"
      return ["0000:00:01.0", "0000:00:02.0"]

    def read(self, *args, **kwargs):
      return fake[self.path]

  monkeypatch.setattr(system, "FileIOInterface", FakeFileIOInterface)

  assert system.System.pci_scan_bus(0x1234, devices=[(0xffff, [0x1111])]) == ["0000:00:01.0"]

@pytest.mark.parametrize("gpu_bus,fail_resize", [(2, False), (2, True)])
def test_usb_bar_setup(gpu_bus, fail_resize):
  from tinygrad.runtime.autogen import pci
  from tinygrad.runtime.support.system import System

  # Chestnut bridges on buses 0/1, multifunction 3090 on bus 2: BAR0/1/3 are resizable.
  config = [{pci.PCI_HEADER_TYPE:1}, {pci.PCI_HEADER_TYPE:1},
            {pci.PCI_HEADER_TYPE:0x80, pci.PCI_COMMAND:0x407, 0x10:0, 0x14:0xc, 0x18:8, 0x1c:0xc, 0x20:8, 0x24:1,
             0x100:0x10015, 0x104:0x100, 0x108:0x460, 0x10c:0xffc00, 0x110:0x801, 0x114:0x200, 0x118:0x503}]
  writes = []
  def cfg(offset, bus, dev, fn, size, value=None):
    assert (dev, fn) == (0, 0)
    if value is None:
      if bus == 2 and config[bus].get(offset) == 0xffffffff and 0x10 <= offset < 0x24:
        bar1_mask = -(1 << (20 + ((config[2][0x110] >> 8) & 31)))
        masks = {0x10:0xff000000, 0x14:(bar1_mask & 0xfffffff0) | 0xc, 0x18:(bar1_mask >> 32) & 0xffffffff,
                 0x1c:0xfe00000c, 0x20:0xffffffff}
        return masks[offset]
      return config[bus].get(offset, 0)
    writes.append((bus, offset, value))
    if bus == 2 and (0x10 <= offset < 0x24 or offset in (0x108, 0x110, 0x118)):
      assert not config[2][pci.PCI_COMMAND] & pci.PCI_COMMAND_MEMORY
      if fail_resize and offset == 0x110: raise RuntimeError("injected resize failure")
    config[bus][offset] = (config[bus].get(offset, 0) & ~((1 << (size * 8)) - 1)) | value

  if fail_resize:
    with pytest.raises(RuntimeError, match="injected resize failure"):
      System.pci_setup_usb_bars(Mock(pcie_cfg_req=cfg), gpu_bus, 0x10000000, 32 << 30)
    assert config[2][pci.PCI_COMMAND] == 0x405
  else:
    bars = System.pci_setup_usb_bars(Mock(pcie_cfg_req=cfg), gpu_bus, 0x10000000, 32 << 30)
    assert bars == {0:(0x10000000, 16 << 20), 1:(32 << 30, 32 << 30), 3:(64 << 30, 32 << 20)}
    assert (config[2][0x18] << 32) | (config[2][0x14] & ~0xf) == bars[1][0]
    assert (config[2][0x20] << 32) | (config[2][0x1c] & ~0xf) == bars[3][0]
    assert config[2][pci.PCI_COMMAND] == 0x407
