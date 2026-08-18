import contextlib
import inspect
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mlx_autoquant import cli
from mlx_autoquant.errors import MetadataError
from mlx_autoquant.hardware import HardwareProfile
from mlx_autoquant.model import ModelProfile, profile_config
from mlx_autoquant.planner import choose_quantization
from mlx_autoquant.preflight import PreflightResult

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
                "max_position_embeddings": 32768,
            }
        )
    )
    return config, params if exact else None


def patch_metadata(params: int = 7_000_000_000, exact: bool = True):
    tmp = tempfile.TemporaryDirectory()
    config, exact_count = metadata(Path(tmp.name), params, exact)
    model = profile_config("example/model", config, exact_count)
    result = PreflightResult(
        "example/model",
        None,
        "main",
        Path(tmp.name),
        model,
        8192,
        1_000_000_000,
        100,
        100 * 1024**3,
        1_000_000_000,
        100 * 1024**3,
        100 * 1024**3,
        1_000_000_000,
        1_000_000_000,
        "supported",
    )
    return tmp, mock.patch("mlx_autoquant.cli.preflight", return_value=result)


class TestCli(unittest.TestCase):
    def test_mlx_lm_converter_contract(self) -> None:
        from mlx_lm import convert

        parameters = inspect.signature(convert).parameters
        self.assertTrue({"hf_path", "mlx_path", "revision"}.issubset(parameters))
        self.assertTrue({"q_group_size", "q_bits", "q_mode"}.issubset(parameters))

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
        self.assertEqual(data["plan"]["context_length"], 8192)
        self.assertEqual(len(data["options"]), 7)

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
        self.assertIn("already exists", err.getvalue())

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
                "mlx_autoquant.cli.preflight",
                side_effect=MetadataError("Could not fetch metadata for 'nope'"),
            ),
            mock.patch("mlx_autoquant.cli.detect_hardware", return_value=HW),
        ):
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = cli.main(["nope", "--dry-run"])
        self.assertEqual(rc, 1)
        self.assertIn("error [metadata]: Could not fetch metadata", err.getvalue())
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

    def test_activity_bar_starts_and_stops(self) -> None:
        with mock.patch("tqdm.tqdm"), cli._activity_bar("Quantizing model"):
            pass

    def test_conversion_failure_writes_diagnostic(self) -> None:
        tmp, patched = patch_metadata()
        output = Path(tmp.name) / "model"
        snapshot = Path(tmp.name) / "snapshot"
        snapshot.mkdir()
        with (
            patched,
            mock.patch("mlx_autoquant.cli.detect_hardware", return_value=HW),
            mock.patch("mlx_autoquant.cli.download_snapshot", return_value=snapshot),
            mock.patch("mlx_lm.convert", side_effect=ValueError("converter failed")),
        ):
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rc = cli.main(["example/model", "--output", str(output), "--yes"])
        diagnostics = list(Path(tmp.name).glob(".model.staging-*-diagnostic.json"))
        staging = list(Path(tmp.name).glob(".model.staging-*"))
        tmp.cleanup()
        self.assertEqual(rc, 1)
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(staging, diagnostics)
        self.assertIn("Diagnostic report", err.getvalue())

    def test_keyboard_interrupt_cleans_staging_and_writes_diagnostic(self) -> None:
        tmp, patched = patch_metadata()
        output = Path(tmp.name) / "model"
        snapshot = Path(tmp.name) / "snapshot"
        snapshot.mkdir()
        with (
            patched,
            mock.patch("mlx_autoquant.cli.detect_hardware", return_value=HW),
            mock.patch("mlx_autoquant.cli.download_snapshot", return_value=snapshot),
            mock.patch("mlx_lm.convert", side_effect=KeyboardInterrupt),
        ):
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rc = cli.main(["example/model", "--output", str(output), "--yes"])
        diagnostics = list(Path(tmp.name).glob(".model.staging-*-diagnostic.json"))
        staging = list(Path(tmp.name).glob(".model.staging-*"))
        tmp.cleanup()
        self.assertEqual(rc, 1)
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(staging, diagnostics)
        self.assertIn("cancellation", err.getvalue())


if __name__ == "__main__":
    unittest.main()
