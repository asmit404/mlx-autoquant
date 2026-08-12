import json
import tempfile
import unittest
from pathlib import Path

from mlx_autoquant.model import profile_config


def write_config(tmp: Path, config: dict) -> Path:
    path = tmp / "config.json"
    path.write_text(json.dumps(config))
    return path


QWEN = {
    "hidden_size": 3584,
    "num_hidden_layers": 28,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "vocab_size": 152064,
    "intermediate_size": 18944,
}

GPT2 = {
    "n_embd": 768,
    "n_layer": 12,
    "n_head": 12,
    "vocab_size": 50257,
}


class TestProfileConfig(unittest.TestCase):
    def test_estimates_qwen_style_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = profile_config("Qwen/Qwen2.5-7B-Instruct", write_config(Path(tmp), QWEN))
        self.assertEqual(profile.parameters, 7_615_283_200)
        self.assertEqual(profile.layers, 28)
        self.assertEqual(profile.num_key_value_heads, 4)
        self.assertEqual(profile.head_dim, 128)
        self.assertFalse(profile.parameters_exact)

    def test_estimates_gpt2_style_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = profile_config("gpt2", write_config(Path(tmp), GPT2))
        self.assertEqual(profile.parameters, 190_440_960)
        self.assertEqual(profile.hidden_size, 768)

    def test_uses_exact_count_when_provided(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = profile_config("example/model", write_config(Path(tmp), QWEN), 7_000_000_000)
        self.assertEqual(profile.parameters, 7_000_000_000)
        self.assertTrue(profile.parameters_exact)

    def test_falls_back_when_exact_count_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = write_config(Path(tmp), QWEN)
            profile = profile_config("example/model", config, None)
        self.assertEqual(profile.parameters, 7_615_283_200)
        self.assertFalse(profile.parameters_exact)

    def test_estimates_moe_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = write_config(
                Path(tmp),
                {
                    "hidden_size": 1536,
                    "num_hidden_layers": 24,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 16,
                    "head_dim": 128,
                    "vocab_size": 157184,
                    "intermediate_size": 4608,
                    "num_experts": 128,
                    "num_shared_experts": 1,
                    "moe_intermediate_size": 512,
                },
            )
            profile = profile_config("example/moe", config)
        # Attention + embeddings dominate the non-expert share; the estimate must
        # stay in the low billions, not the trillions a dense reading would imply.
        self.assertLess(profile.parameters, 10_000_000_000)
        self.assertGreater(profile.parameters, 1_000_000_000)
        self.assertFalse(profile.parameters_exact)

    def test_to_dict_includes_exact_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = profile_config("example/model", write_config(Path(tmp), QWEN))
        self.assertIn("parameters_exact", profile.to_dict())


if __name__ == "__main__":
    unittest.main()
