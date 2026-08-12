import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mlx_autoquant import cli
from mlx_autoquant.hardware import HardwareProfile
from mlx_autoquant.model import ModelProfile
from mlx_autoquant.planner import choose_quantization

HW = HardwareProfile("Apple M4", 34_359_738_368, True)


def metadata(tmp: Path, params: int = 7_000_000_000, exact: bool = True):
    config = tmp / "config.json"
    config.write_text(
        json.dumps(
            {
                "hidden_size": 3584,
                "num_hidden_layers": 28,
                "num_attention_heads": 28,
                "num_key_value_heads": 4,
                "head_dim": 128,
                "vocab_size": 152064,
                "intermediate_size": 18944,
            }
        )
    )
    return config, params if exact else None


def patch_metadata(params: int = 7_000_000_000, exact: bool = True):
    tmp = tempfile.TemporaryDirectory()
    return tmp, mock.patch(
        "mlx_autoquant.cli._fetch_metadata", return_value=metadata(Path(tmp.name), params, exact)
    )


class TestCli(unittest.TestCase):
    def test_json_dry_run(self) -> None:
        tmp, patched = patch_metadata()
        with patched, mock.patch("mlx_autoquant.cli.detect_hardware", return_value=HW):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = cli.main(["example/model", "--dry-run", "--json"])
        tmp.cleanup()
        self.assertEqual(rc, 0)
        data = json.loads(out.getvalue())
        self.assertEqual(data["plan"]["bits"], 8)
        self.assertEqual(data["model"]["parameters"], 7_000_000_000)
        self.assertTrue(data["model"]["parameters_exact"])

    def test_summary_dry_run_labels_exact_count(self) -> None:
        tmp, patched = patch_metadata()
        with patched, mock.patch("mlx_autoquant.cli.detect_hardware", return_value=HW):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = cli.main(["example/model", "--dry-run"])
        tmp.cleanup()
        self.assertEqual(rc, 0)
        self.assertIn("7B (from safetensors metadata)", out.getvalue())

    def test_bits_override(self) -> None:
        tmp, patched = patch_metadata()
        with patched, mock.patch("mlx_autoquant.cli.detect_hardware", return_value=HW):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = cli.main(["example/model", "--dry-run", "--json", "--bits", "4"])
        tmp.cleanup()
        self.assertEqual(rc, 0)
        data = json.loads(out.getvalue())
        self.assertEqual(data["plan"]["bits"], 4)
        self.assertLess(data["plan"]["estimated_model_gib"], 7.0)

    def test_error_message_to_stderr(self) -> None:
        tmp, patched = patch_metadata()
        with patched, mock.patch("mlx_autoquant.cli.detect_hardware", return_value=HW):
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rc = cli.main(["example/model", "--output", str(Path(tmp.name))])
        tmp.cleanup()
        self.assertEqual(rc, 1)
        self.assertIn("not empty", err.getvalue())

    def test_keyboard_interrupt_exit_code(self) -> None:
        with (
            mock.patch("mlx_autoquant.cli._run", side_effect=KeyboardInterrupt),
            mock.patch("mlx_autoquant.cli.detect_hardware", return_value=HW),
        ):
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rc = cli.main(["example/model"])
        self.assertEqual(rc, 130)

    def test_unexpected_error_reported_cleanly(self) -> None:
        with (
            mock.patch(
                "mlx_autoquant.cli._run",
                side_effect=ValueError("Received 64 parameters not in model"),
            ),
            mock.patch("mlx_autoquant.cli.detect_hardware", return_value=HW),
        ):
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rc = cli.main(["example/model"])
        self.assertEqual(rc, 1)
        self.assertIn("error: ValueError: Received 64 parameters", err.getvalue())

    def test_clean_error_message_on_stdout_errors(self) -> None:
        with (
            mock.patch(
                "mlx_autoquant.cli._fetch_metadata",
                side_effect=RuntimeError("Could not fetch metadata for 'nope'"),
            ),
            mock.patch("mlx_autoquant.cli.detect_hardware", return_value=HW),
        ):
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = cli.main(["nope", "--dry-run"])
        self.assertEqual(rc, 1)
        self.assertIn("error: Could not fetch metadata", err.getvalue())
        self.assertEqual(out.getvalue(), "")

    def test_write_report_flags_estimated_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = ModelProfile("example/model", 7_000_000_000, 28, 3584, 4, 128, False)
            plan = choose_quantization(HW, model)
            report = cli._write_report(Path(tmp), HW, model, plan, None)
            data = json.loads(Path(report).read_text())
        self.assertTrue(data["parameter_count_is_estimated_from_config"])

    def test_write_report_flags_exact_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = ModelProfile("example/model", 7_000_000_000, 28, 3584, 4, 128, True)
            plan = choose_quantization(HW, model)
            report = cli._write_report(Path(tmp), HW, model, plan, "main")
            data = json.loads(Path(report).read_text())
        self.assertFalse(data["parameter_count_is_estimated_from_config"])
        self.assertEqual(data["revision"], "main")

    def test_format_params(self) -> None:
        self.assertEqual(cli._format_params(7_000_000_000), "7B")
        self.assertEqual(cli._format_params(7_615_283_200), "7.6B")
        self.assertEqual(cli._format_params(405_000_000_000), "405B")
        self.assertEqual(cli._format_params(125_000_000), "125M")
        self.assertEqual(cli._format_params(55_000_000), "55M")
        self.assertEqual(cli._format_params(500), "500")

    def test_download_weights_downloads_every_file(self) -> None:
        files = ["config.json", "model-00001-of-00002.safetensors", "model.safetensors.index.json"]
        infos = [mock.Mock(size=100), mock.Mock(size=900), mock.Mock(size=50)]
        api = mock.Mock()
        api.list_repo_files.return_value = files
        api.get_paths_info.return_value = infos
        with (
            mock.patch("huggingface_hub.HfApi", return_value=api),
            mock.patch("huggingface_hub.hf_hub_download") as download,
        ):
            cli._download_weights("example/model", "main")
        download.assert_has_calls(
            [
                mock.call("example/model", "config.json", revision="main"),
                mock.call("example/model", "model-00001-of-00002.safetensors", revision="main"),
                mock.call("example/model", "model.safetensors.index.json", revision="main"),
            ]
        )

    def test_download_weights_clean_error(self) -> None:
        api = mock.Mock()
        api.list_repo_files.side_effect = Exception("Repository Not Found for url")
        with (
            mock.patch("huggingface_hub.HfApi", return_value=api),
            self.assertRaisesRegex(RuntimeError, "Could not list files"),
        ):
            cli._download_weights("nope", None)

    def test_activity_bar_starts_and_stops(self) -> None:
        with mock.patch("tqdm.tqdm"), cli._activity_bar("Quantizing model"):
            pass


if __name__ == "__main__":
    unittest.main()
