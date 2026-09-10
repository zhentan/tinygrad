import sys
import pytest
from unittest.mock import Mock, call, patch

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

@pytest.mark.parametrize("gpu_bus,fail_resize", [(2, False), (4, False), (2, True)])
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

  if gpu_bus == 4 or fail_resize:
    with pytest.raises((AssertionError, RuntimeError), match="PCI bridge|injected resize failure"):
      System.pci_setup_usb_bars(Mock(pcie_cfg_req=cfg), gpu_bus, 0x10000000, 32 << 30)
    if gpu_bus == 4: assert not any(bus == 2 for bus, _, _ in writes)
    else: assert config[2][pci.PCI_COMMAND] == 0x405
  else:
    bars = System.pci_setup_usb_bars(Mock(pcie_cfg_req=cfg), gpu_bus, 0x10000000, 32 << 30)
    assert bars == {0:(0x10000000, 16 << 20), 1:(32 << 30, 32 << 30), 3:(64 << 30, 32 << 20)}
    assert (config[2][0x18] << 32) | (config[2][0x14] & ~0xf) == bars[1][0]
    assert (config[2][0x20] << 32) | (config[2][0x1c] & ~0xf) == bars[3][0]
    assert config[2][pci.PCI_COMMAND] == 0x407
  assert [config[bus][pci.PCI_PRIMARY_BUS] for bus in (0, 1)] == [gpu_bus << 16 | 0x100, gpu_bus << 16 | 0x201]

@pytest.mark.parametrize("bridges,cold,header,vendor", [(2, True, 0x80, 0x10de), (4, True, 0, 0x1002),
  (2, False, 0x80, 0x10de), (4, False, 0, 0x1002), (2, True, 0xff, 0xffff), (2, True, 2, 0x10de),
  (2, True, 0, 0), (256, True, 0, 0x10de)])
def test_usb_pci_discovery(monkeypatch, bridges, cold, header, vendor):
  from tinygrad.runtime.autogen import pci
  import tinygrad.runtime.support.system as system

  buses = [0 if cold else bus | (bus+1) << 8 | bridges << 16 for bus in range(bridges)]
  writes, setup = [], Mock(return_value={})
  def cfg(offset, bus, dev, fn, size, value=None):
    assert (dev, fn) == (0, 0)
    assert bus <= bridges and bus < 256
    # An unnumbered bridge must be configured before its child can be reached.
    if any((buses[i] >> 8) & 0xff != i+1 or (buses[i] >> 16) & 0xff < bus for i in range(bus)):
      return (1 << (size * 8)) - 1
    if value is None:
      if offset == pci.PCI_HEADER_TYPE: return 1 if bus < bridges else header
      if offset == pci.PCI_VENDOR_ID: return vendor
      return 0
    writes.append((bus, offset, value))
    if bus < bridges:
      assert (offset, size) == (pci.PCI_PRIMARY_BUS, 4)
      assert value == bus | (bus+1) << 8 | 0xff0000
      buses[bus] = value
    else: assert setup.called
  usb = Mock(pcie_cfg_req=cfg)
  monkeypatch.setattr(system, "USB3", Mock(return_value=Mock(product="custom test")))
  monkeypatch.setattr(system, "CustomASM24Controller", Mock(return_value=usb))
  monkeypatch.setattr(system.System, "flock_acquire", Mock(return_value=0))
  monkeypatch.setattr(system.System, "pci_setup_usb_bars", setup)
  if header & 0x7f or not vendor or bridges == 256:
    with pytest.raises(AssertionError, match="PCI bridge|PCI endpoint"):
      system.USBPCIDevice("AM", None, "usb:mock")
    setup.assert_not_called()
    assert all(bus < bridges for bus, _, _ in writes)
  else:
    dev = system.USBPCIDevice("AM", None, "usb:mock")
    setup.assert_called_once_with(usb, gpu_bus=bridges, mem_base=0x10000000, pref_mem_base=32 << 30)
    assert dev.read_config(pci.PCI_VENDOR_ID, 2) == vendor
    dev.write_config(pci.PCI_COMMAND, 2, 2)
    assert writes[-1] == (bridges, pci.PCI_COMMAND, 2)

def test_usb_pci_reset_uses_downstream_bridge(monkeypatch):
  from tinygrad.runtime.autogen import pci
  import tinygrad.runtime.support.system as system

  dev = object.__new__(system.USBPCIDevice)
  dev.gpu_bus, dev.pcibus = 2, "usb:4-3"
  dev._mem_base, dev._pref_mem_base = 0x10000000, 32 << 30
  dev._bar_info = {0:(0x10000000, 16 << 20), 1:(32 << 30, 32 << 30), 3:(64 << 30, 32 << 20)}
  dev.sram = system.BumpAllocator(0x80000, wrap=False)
  dev.sram.alloc(0x24000)
  writes, command, bridge_control = [], 0x407, 0

  def config(offset, bus=1, dev=0, fn=0, value=None, size=4):
    nonlocal command, bridge_control
    if value is not None:
      writes.append((bus, fn, offset, value, size))
      if (bus, fn, offset, size) == (2, 0, pci.PCI_COMMAND, 2): command = value
      if (bus, fn, offset, size) == (1, 0, pci.PCI_BRIDGE_CONTROL, 2): bridge_control = value
      return None
    if (bus, fn, offset, size) == (2, 0, pci.PCI_COMMAND, 2): return command
    if (bus, fn, offset, size) == (2, 0, pci.PCI_STATUS, 2): return 0x10
    if (bus, fn, offset, size) == (1, 0, pci.PCI_HEADER_TYPE, 1): return pci.PCI_HEADER_TYPE_BRIDGE
    if (bus, fn, offset, size) == (1, 0, pci.PCI_BRIDGE_CONTROL, 2): return bridge_control
    if (bus, fn, offset, size) == (2, 0, pci.PCI_VENDOR_ID, 4): return 0x220410DE
    raise AssertionError((offset, bus, dev, fn, value, size))

  def restore_bars(usb, gpu_bus, mem_base, pref_mem_base):
    nonlocal command
    assert (usb, gpu_bus, mem_base, pref_mem_base) == (dev.usb, 2, 0x10000000, 32 << 30)
    command = 0x407
    return {0:(0x10000000, 16 << 20), 1:(32 << 30, 32 << 30), 3:(64 << 30, 32 << 20)}

  dev.usb = Mock(pcie_cfg_req=Mock(side_effect=config))
  clock = Mock(monotonic=Mock(side_effect=(0.0, 0.1)), sleep=Mock())
  monkeypatch.setattr(system, "time", clock, raising=False)
  with patch.object(system.os, "system") as sysfs_reset, patch.object(system.System, "pci_setup_usb_bars", side_effect=restore_bars) as setup:
    dev.reset()

  sysfs_reset.assert_not_called()
  setup.assert_called_once_with(dev.usb, gpu_bus=2, mem_base=0x10000000, pref_mem_base=32 << 30)
  assert writes == [
    (2, 0, pci.PCI_COMMAND, 0x403, 2),
    (1, 0, pci.PCI_BRIDGE_CONTROL, pci.PCI_BRIDGE_CTL_BUS_RESET, 2),
    (1, 0, pci.PCI_BRIDGE_CONTROL, 0, 2),
    (2, 0, pci.PCI_COMMAND, 0x403, 2),
  ]
  assert clock.sleep.call_args_list == [call(0.002), call(0.1)]
  assert command == 0x403 and dev.sram.ptr == 0
