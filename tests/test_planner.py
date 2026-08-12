import unittest

from mlx_autoquant.hardware import HardwareProfile
from mlx_autoquant.model import ModelProfile
from mlx_autoquant.planner import choose_quantization, estimated_model_gib


def model(parameters: int = 7_000_000_000) -> ModelProfile:
    return ModelProfile("example/model", parameters, 32, 4096, 8, 128)


class TestPlanner(unittest.TestCase):
    def test_prefers_best_precision_that_fits(self) -> None:
        plan = choose_quantization(HardwareProfile("M4 Max", 64 * 1024**3, True), model())
        self.assertEqual(plan.bits, 8)
        self.assertLessEqual(plan.estimated_model_gib, plan.available_for_model_gib)

    def test_falls_back_on_small_memory(self) -> None:
        plan = choose_quantization(HardwareProfile("M2", 16 * 1024**3, True), model(20_000_000_000))
        self.assertEqual(plan.bits, 3)

    def test_model_estimate_gets_smaller_with_fewer_bits(self) -> None:
        self.assertLess(
            estimated_model_gib(1_000_000_000, 4),
            estimated_model_gib(1_000_000_000, 8),
        )
