import unittest
from unittest import mock

import mlx_autoquant.hardware as hardware


class TestDetectHardware(unittest.TestCase):
    def test_detects_apple_silicon_mac(self) -> None:
        with (
            mock.patch("platform.system", return_value="Darwin"),
            mock.patch("platform.machine", return_value="arm64"),
            mock.patch("platform.processor", return_value=None),
            mock.patch(
                "mlx_autoquant.hardware.subprocess.check_output",
                side_effect=["Apple M4", "34359738368"],
            ),
        ):
            profile = hardware.detect_hardware()
        self.assertEqual(profile.chip, "Apple M4")
        self.assertEqual(profile.memory_bytes, 34_359_738_368)
        self.assertTrue(profile.is_apple_silicon)

    def test_flags_non_apple_silicon(self) -> None:
        with (
            mock.patch("platform.system", return_value="Linux"),
            mock.patch("platform.machine", return_value="x86_64"),
            mock.patch("platform.processor", return_value=None),
            mock.patch(
                "mlx_autoquant.hardware.subprocess.check_output", side_effect=["Intel", "16777216"]
            ),
        ):
            profile = hardware.detect_hardware()
        self.assertFalse(profile.is_apple_silicon)

    def test_raises_when_memory_unknown(self) -> None:
        with (
            mock.patch("platform.system", return_value="Darwin"),
            mock.patch("platform.machine", return_value="arm64"),
            mock.patch("platform.processor", return_value=None),
            mock.patch("mlx_autoquant.hardware.subprocess.check_output", return_value=""),
            self.assertRaises(RuntimeError),
        ):
            hardware.detect_hardware()

    def test_sysctl_failure_falls_back(self) -> None:
        def sysctl(args, text=False):
            if any("brand_string" in str(arg) for arg in args):
                raise OSError("no sysctl")
            return "34359738368"

        with (
            mock.patch("platform.system", return_value="Darwin"),
            mock.patch("platform.machine", return_value="arm64"),
            mock.patch("platform.processor", return_value="Apple M4"),
            mock.patch("mlx_autoquant.hardware.subprocess.check_output", side_effect=sysctl),
        ):
            profile = hardware.detect_hardware()
        self.assertEqual(profile.chip, "Apple M4")

    def test_to_dict_rounds_memory(self) -> None:
        profile = hardware.HardwareProfile("Apple M4", 34_359_738_368, True)
        self.assertEqual(profile.to_dict()["memory_gib"], 32.0)


if __name__ == "__main__":
    unittest.main()
