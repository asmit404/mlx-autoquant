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

    def test_uses_exact_count_from_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            config = write_config(tmp, QWEN)
            index = tmp / "model.safetensors.index.json"
            index.write_text(json.dumps({"metadata": {"total_size": 14_000_000_000}}))
            profile = profile_config("example/model", config, index)
        self.assertEqual(profile.parameters, 7_000_000_000)
        self.assertTrue(profile.parameters_exact)

    def test_index_count_respects_fp32_dtype(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            config = write_config(tmp, {**QWEN, "torch_dtype": "float32"})
            index = tmp / "model.safetensors.index.json"
            index.write_text(json.dumps({"metadata": {"total_size": 28_000_000_000}}))
            profile = profile_config("example/model", config, index)
        self.assertEqual(profile.parameters, 7_000_000_000)
        self.assertTrue(profile.parameters_exact)

    def test_falls_back_when_index_total_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            config = write_config(tmp, QWEN)
            index = tmp / "model.safetensors.index.json"
            index.write_text(json.dumps({"metadata": {}}))
            profile = profile_config("example/model", config, index)
        self.assertEqual(profile.parameters, 7_615_283_200)
        self.assertFalse(profile.parameters_exact)

    def test_falls_back_when_index_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = write_config(Path(tmp), QWEN)
            profile = profile_config("example/model", config, None)
        self.assertFalse(profile.parameters_exact)

    def test_to_dict_includes_exact_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = profile_config("example/model", write_config(Path(tmp), QWEN))
        self.assertIn("parameters_exact", profile.to_dict())


if __name__ == "__main__":
    unittest.main()
