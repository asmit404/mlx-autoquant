import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from mlx_autoquant.errors import InsufficientMemoryError, MetadataError, UnsupportedModelError
from mlx_autoquant.hardware import HardwareProfile
from mlx_autoquant.preflight import download_snapshot, preflight

CONFIG = {
    "architectures": ["Qwen2ForCausalLM"],
    "model_type": "qwen2",
    "hidden_size": 3584,
    "num_hidden_layers": 28,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "vocab_size": 152064,
    "intermediate_size": 18944,
    "max_position_embeddings": 32768,
}


class TestPreflight(unittest.TestCase):
    def test_resolves_revision_and_fetches_metadata_without_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metadata_dir = Path(tmp) / "metadata"
            metadata_dir.mkdir()
            (metadata_dir / "config.json").write_text(json.dumps(CONFIG))
            info = SimpleNamespace(
                sha="abc123",
                safetensors=SimpleNamespace(total=900),
                siblings=[
                    SimpleNamespace(rfilename="config.json", size=100),
                    SimpleNamespace(rfilename="tokenizer.json", size=200),
                    SimpleNamespace(rfilename="model.safetensors.index.json", size=50),
                    SimpleNamespace(rfilename="model-00001.safetensors", size=900),
                ],
            )
            api = mock.Mock()
            api.model_info.return_value = info
            with (
                mock.patch(
                    "mlx_autoquant.preflight._api",
                    return_value=(api, mock.Mock(return_value=str(metadata_dir))),
                ),
                mock.patch(
                    "mlx_autoquant.preflight.shutil.disk_usage",
                    return_value=SimpleNamespace(free=100 * 1024**3),
                ),
            ):
                result = preflight(
                    "example/model",
                    "main",
                    HardwareProfile("Apple M4", 32 * 1024**3, True),
                    4096,
                    Path(tmp) / "cache",
                    Path(tmp) / "output",
                )
        self.assertEqual(result.resolved_revision, "abc123")
        self.assertEqual(result.context_length, 4096)
        self.assertEqual(result.source_weight_bytes, 900)
        self.assertEqual(result.required_metadata_bytes, 350)
        self.assertTrue(result.model.parameters_exact)
        call = api.model_info.call_args
        self.assertEqual(call.kwargs["revision"], "main")

    def test_rejects_unknown_architecture_before_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metadata_dir = Path(tmp) / "metadata"
            metadata_dir.mkdir()
            (metadata_dir / "config.json").write_text(json.dumps({"model_type": "vision"}))
            info = SimpleNamespace(
                sha="abc123",
                siblings=[SimpleNamespace(rfilename="model.safetensors", size=900)],
            )
            api = mock.Mock()
            api.model_info.return_value = info
            with (
                mock.patch(
                    "mlx_autoquant.preflight._api",
                    return_value=(api, mock.Mock(return_value=str(metadata_dir))),
                ),
                self.assertRaises(UnsupportedModelError),
            ):
                preflight(
                    "example/model",
                    None,
                    HardwareProfile("Apple M4", 32 * 1024**3, True),
                    4096,
                    Path(tmp) / "cache",
                    Path(tmp) / "output",
                )

    def test_rejects_non_causal_qwen_architecture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metadata_dir = Path(tmp) / "metadata"
            metadata_dir.mkdir()
            (metadata_dir / "config.json").write_text(
                json.dumps(
                    {"model_type": "qwen2", "architectures": ["Qwen2ForSequenceClassification"]}
                )
            )
            api = mock.Mock()
            api.model_info.return_value = SimpleNamespace(
                sha="abc123",
                siblings=[SimpleNamespace(rfilename="model.safetensors", size=900)],
            )
            with (
                mock.patch(
                    "mlx_autoquant.preflight._api",
                    return_value=(api, mock.Mock(return_value=str(metadata_dir))),
                ),
                self.assertRaises(UnsupportedModelError),
            ):
                preflight(
                    "example/model",
                    None,
                    HardwareProfile("Apple M4", 32 * 1024**3, True),
                    4096,
                    Path(tmp) / "cache",
                    Path(tmp) / "output",
                )

    def test_rejects_malformed_config_as_metadata_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metadata_dir = Path(tmp) / "metadata"
            metadata_dir.mkdir()
            (metadata_dir / "config.json").write_text("not json")
            api = mock.Mock()
            api.model_info.return_value = SimpleNamespace(
                sha="abc123",
                siblings=[SimpleNamespace(rfilename="model.safetensors", size=900)],
            )
            with (
                mock.patch(
                    "mlx_autoquant.preflight._api",
                    return_value=(api, mock.Mock(return_value=str(metadata_dir))),
                ),
                self.assertRaises(MetadataError),
            ):
                preflight(
                    "example/model",
                    None,
                    HardwareProfile("Apple M4", 32 * 1024**3, True),
                    4096,
                    Path(tmp) / "cache",
                    Path(tmp) / "output",
                )

    def test_reports_no_fitting_candidate_as_memory_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metadata_dir = Path(tmp) / "metadata"
            metadata_dir.mkdir()
            (metadata_dir / "config.json").write_text(json.dumps(CONFIG))
            api = mock.Mock()
            api.model_info.return_value = SimpleNamespace(
                sha="abc123",
                siblings=[SimpleNamespace(rfilename="model.safetensors", size=900)],
            )
            with (
                mock.patch(
                    "mlx_autoquant.preflight._api",
                    return_value=(api, mock.Mock(return_value=str(metadata_dir))),
                ),
                mock.patch(
                    "mlx_autoquant.preflight.shutil.disk_usage",
                    return_value=SimpleNamespace(free=100 * 1024**3),
                ),
                self.assertRaises(InsufficientMemoryError),
            ):
                preflight(
                    "example/model",
                    None,
                    HardwareProfile("Apple M4", 4 * 1024**3, True),
                    4096,
                    Path(tmp) / "cache",
                    Path(tmp) / "output",
                )

    def test_rejects_unknown_required_file_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metadata_dir = Path(tmp) / "metadata"
            metadata_dir.mkdir()
            (metadata_dir / "config.json").write_text(json.dumps(CONFIG))
            api = mock.Mock()
            api.model_info.return_value = SimpleNamespace(
                sha="abc123",
                siblings=[
                    SimpleNamespace(rfilename="model-00001.safetensors", size=None),
                    SimpleNamespace(rfilename="model-00002.safetensors", size=900),
                ],
            )
            with (
                mock.patch(
                    "mlx_autoquant.preflight._api",
                    return_value=(api, mock.Mock(return_value=str(metadata_dir))),
                ),
                self.assertRaises(MetadataError),
            ):
                preflight(
                    "example/model",
                    None,
                    HardwareProfile("Apple M4", 32 * 1024**3, True),
                    4096,
                    Path(tmp) / "cache",
                    Path(tmp) / "output",
                )

    def test_accepts_config_estimate_for_bin_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metadata_dir = Path(tmp) / "metadata"
            metadata_dir.mkdir()
            (metadata_dir / "config.json").write_text(json.dumps(CONFIG))
            info = SimpleNamespace(
                sha="abc123",
                siblings=[SimpleNamespace(rfilename="pytorch_model.bin", size=900)],
            )
            api = mock.Mock()
            api.model_info.return_value = info
            with (
                mock.patch(
                    "mlx_autoquant.preflight._api",
                    return_value=(api, mock.Mock(return_value=str(metadata_dir))),
                ),
                mock.patch(
                    "mlx_autoquant.preflight.shutil.disk_usage",
                    return_value=SimpleNamespace(free=100 * 1024**3),
                ),
            ):
                result = preflight(
                    "example/model",
                    None,
                    HardwareProfile("Apple M4", 32 * 1024**3, True),
                    4096,
                    Path(tmp) / "cache",
                    Path(tmp) / "output",
                )
        self.assertEqual(result.source_weight_bytes, 900)
        self.assertFalse(result.model.parameters_exact)

    def test_download_uses_resolved_revision_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = mock.Mock(return_value=tmp)
            result = mock.Mock(model_id="example/model", resolved_revision="abc123")
            with mock.patch("mlx_autoquant.preflight._api", return_value=(mock.Mock(), snapshot)):
                download_snapshot(result, Path(tmp) / "cache")
            kwargs = snapshot.call_args.kwargs
        self.assertEqual(kwargs["revision"], "abc123")
        self.assertIn("*.safetensors", kwargs["allow_patterns"])
        self.assertIn("*.md", kwargs["ignore_patterns"])


if __name__ == "__main__":
    unittest.main()
