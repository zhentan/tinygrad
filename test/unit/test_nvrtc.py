import ctypes, sysconfig, unittest
from pathlib import Path

from tinygrad.runtime.autogen import nvrtc


class TestNVRTCDiscovery(unittest.TestCase):
  def test_nvidia_wheel_library_path(self):
    wheel_lib = Path(sysconfig.get_path("purelib")) / "nvidia" / "cuda_nvrtc" / "lib"
    self.assertIn(str(wheel_lib), nvrtc.NVRTC_LIB_PATHS)
    if not wheel_lib.is_dir(): self.skipTest("NVIDIA NVRTC wheel is not installed")
    major, minor = ctypes.c_int(), ctypes.c_int()
    self.assertEqual(nvrtc.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor)), 0)
    self.assertGreater((major.value, minor.value), (0, 0))


if __name__ == "__main__": unittest.main()
