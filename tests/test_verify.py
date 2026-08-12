import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from mlx_autoquant.verify import verify_model


class TestVerifyModel(unittest.TestCase):
    def test_loads_generates_and_reports_peak_memory(self) -> None:
        fake_metal = types.SimpleNamespace(
            reset_peak_memory=mock.Mock(), get_peak_memory=mock.Mock(return_value=2 * 1024**3)
        )
        fake_core = types.ModuleType("mlx.core")
        fake_core.metal = fake_metal
        fake_mlx = types.ModuleType("mlx")
        fake_mlx.core = fake_core
        tokenizer = mock.Mock()
        tokenizer.encode.return_value = [1, 2, 3]
        fake_lm = types.ModuleType("mlx_lm")
        fake_lm.load = mock.Mock(return_value=(object(), tokenizer))
        fake_lm.generate = mock.Mock(return_value="Generated answer")
        with mock.patch.dict(
            sys.modules, {"mlx": fake_mlx, "mlx.core": fake_core, "mlx_lm": fake_lm}
        ):
            result = verify_model(Path("/tmp/model"), max_tokens=8)
        self.assertEqual(result.generated_tokens, 3)
        self.assertEqual(result.peak_memory_gib, 2.0)
        fake_lm.load.assert_called_once_with("/tmp/model")
        fake_lm.generate.assert_called_once()

    def test_rejects_empty_generation(self) -> None:
        fake_metal = types.SimpleNamespace(reset_peak_memory=mock.Mock())
        fake_core = types.ModuleType("mlx.core")
        fake_core.metal = fake_metal
        fake_mlx = types.ModuleType("mlx")
        fake_mlx.core = fake_core
        fake_lm = types.ModuleType("mlx_lm")
        fake_lm.load = mock.Mock(return_value=(object(), object()))
        fake_lm.generate = mock.Mock(return_value=" ")
        with (
            mock.patch.dict(
                sys.modules, {"mlx": fake_mlx, "mlx.core": fake_core, "mlx_lm": fake_lm}
            ),
            self.assertRaisesRegex(RuntimeError, "empty response"),
        ):
            verify_model(Path("/tmp/model"))


if __name__ == "__main__":
    unittest.main()
