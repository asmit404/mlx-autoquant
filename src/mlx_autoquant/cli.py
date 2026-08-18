from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
import uuid
from contextlib import contextmanager, redirect_stdout
from dataclasses import replace
from pathlib import Path
from typing import Any

from .errors import (
    AutoQuantError,
    CancellationError,
    ConversionError,
    InsufficientDiskError,
    InsufficientMemoryError,
)
from .hardware import HardwareProfile, detect_hardware
from .model import ModelProfile
from .planner import (
    QuantizationOption,
    QuantizationPlan,
    choose_quantization,
    estimated_model_gib,
    quantization_options,
)
from .preflight import PreflightResult, download_snapshot, preflight
from .verify import VerificationResult, verify_model


@contextmanager
def _activity_bar(desc: str) -> Any:
    from tqdm import tqdm

    bar = tqdm(
        total=100, desc=desc, leave=False, mininterval=0.05, bar_format="{l_bar}{bar}| {elapsed}"
    )
    stop = threading.Event()

    def spin() -> None:
        while not stop.is_set():
            bar.n = (bar.n + 1) % 100
            bar.refresh()
            time.sleep(0.08)

    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
        bar.close()


def _write_report(
    destination: Path,
    hardware: Any,
    model: Any,
    plan: Any,
    revision: str | None,
    options: tuple[QuantizationOption, ...] = (),
    verification: VerificationResult | None = None,
    preflight_result: PreflightResult | None = None,
) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    report = destination / "autoquant-report.json"
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "hardware": hardware.to_dict(),
                "model": model.to_dict(),
                "plan": plan.to_dict(),
                "options": [option.to_dict() for option in options],
                "revision": revision,
                "parameter_count_is_estimated_from_config": not model.parameters_exact,
                "verification": verification.to_dict() if verification else {"skipped": True},
                "preflight": preflight_result.to_dict() if preflight_result else None,
            },
            indent=2,
        )
        + "\n"
    )
    with report.open("w") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return report


def _write_diagnostic(stage: Path, error: Exception, preflight_result: PreflightResult) -> Path:
    diagnostic = stage.with_name(f"{stage.name}-diagnostic.json")
    code = error.code if isinstance(error, AutoQuantError) else "conversion"
    diagnostic.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "failed",
                "error": {"code": code, "message": str(error)},
                "preflight": preflight_result.to_dict(),
            },
            indent=2,
        )
        + "\n"
    )
    return diagnostic


def _cleanup_failed_stage(stage: Path, error: Exception, preflight_result: PreflightResult) -> None:
    try:
        diagnostic = _write_diagnostic(stage, error, preflight_result)
        print(f"Diagnostic report: {diagnostic}", file=sys.stderr)
    except OSError as diagnostic_error:
        print(f"Could not write diagnostic report: {diagnostic_error}", file=sys.stderr)
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _format_params(parameters: int) -> str:
    for unit, threshold in (("T", 10**12), ("B", 10**9), ("M", 10**6), ("K", 10**3)):
        if parameters >= threshold:
            value = parameters / threshold
            text = f"{value:.1f}"
            return f"{text[:-2]}{unit}" if text.endswith(".0") else f"{text}{unit}"
    return str(parameters)


def _print_summary(
    hardware: HardwareProfile,
    model: ModelProfile,
    plan: QuantizationPlan,
    options: tuple[QuantizationOption, ...] = (),
) -> None:
    count_label = (
        f"{_format_params(model.parameters)} (from safetensors metadata)"
        if model.parameters_exact
        else f"~{_format_params(model.parameters)} (estimated from config.json)"
    )
    lines = [
        "Machine",
        f"  Chip:            {hardware.chip}",
        f"  Unified memory:  {hardware.memory_gib:.2f} GiB",
        "",
        "Model",
        f"  ID:              {model.model_id}",
        f"  Parameters:      {count_label}",
        f"  Layers:          {model.layers}",
        f"  Context length:  {plan.context_length} tokens",
        "",
        "Plan",
        f"  Quantization:    {plan.bits}-bit {plan.mode} (group size {plan.group_size})",
        f"  Model weights:   {plan.estimated_model_gib:.2f} GiB",
        f"  KV cache:        {plan.estimated_kv_cache_gib:.2f} GiB at {plan.context_length} tokens",
        f"  Available:       {plan.available_for_model_gib:.2f} GiB",
        f"  Rationale:       {plan.rationale}",
    ]
    if options:
        lines.extend(["", "Options"])
        lines.extend(
            f"  {option.bits}-bit:           {option.estimated_model_gib:.2f} GiB  "
            f"{'fits' if option.fits else 'does not fit'}"
            f"{'  <- selected' if option.bits == plan.bits else ''}"
            for option in options
        )
    print("\n".join(lines))


def _run(args: argparse.Namespace) -> int:
    hardware = detect_hardware()
    hf_home = os.environ.get("HF_HOME")
    cache_dir = Path(hf_home) / "hub" if hf_home else Path.home() / ".cache" / "huggingface" / "hub"
    cache_dir.mkdir(parents=True, exist_ok=True)
    preflight_result = preflight(
        args.model,
        args.revision,
        hardware,
        args.context_length,
        cache_dir,
        args.output,
    )
    model = preflight_result.model
    context_length = preflight_result.context_length
    options = quantization_options(hardware, model, context_length)
    plan = choose_quantization(hardware, model, context_length)
    if args.bits:
        forced_size = estimated_model_gib(model.parameters, args.bits)
        if forced_size > plan.available_for_model_gib:
            raise InsufficientMemoryError(
                f"{args.bits}-bit weights need {forced_size:.2f} GiB, "
                f"but only {plan.available_for_model_gib:.2f} GiB is available.",
                "Choose a smaller bit-width, model, or context length.",
            )
        forced_output_bytes = int(forced_size * 1024**3)
        if (
            preflight_result.required_cache_bytes > preflight_result.available_cache_bytes
            or forced_output_bytes > preflight_result.available_output_bytes
        ):
            raise InsufficientDiskError(
                f"{args.bits}-bit conversion needs "
                f"{preflight_result.required_cache_bytes / 1024**3:.2f} GiB in the cache "
                f"and {forced_output_bytes / 1024**3:.2f} GiB in the output filesystem.",
                "Free disk space or choose a smaller model.",
            )
        plan = replace(
            plan,
            bits=args.bits,
            estimated_model_gib=round(forced_size, 2),
            rationale="User-selected bit-width; fit has not been overridden by the planner.",
        )
    if args.dry_run:
        result = {
            "hardware": hardware.to_dict(),
            "model": model.to_dict(),
            "plan": plan.to_dict(),
            "options": [option.to_dict() for option in options],
            "preflight": preflight_result.to_dict(),
        }
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            _print_summary(hardware, model, plan, options)
        if not args.json:
            print()
            print("Dry run complete. Re-run without --dry-run to convert the model.")
        return 0
    if not args.json:
        _print_summary(hardware, model, plan, options)
    if args.output.exists():
        raise AutoQuantError(
            f"Output path already exists: {args.output}",
            "Choose a new --output path or remove the existing output first.",
        )
    if not args.yes:
        try:
            answer = input("Proceed with downloading and converting this model? [y/N] ")
        except EOFError as error:
            raise AutoQuantError(
                "Conversion requires confirmation in a non-interactive terminal.",
                "Pass --yes for scripts and CI.",
            ) from error
        if answer.strip().lower() not in {"y", "yes"}:
            print("Conversion cancelled.")
            return 0
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise AutoQuantError(
            f"Could not create output directory: {args.output.parent}",
            "Choose an output path you can write to, then retry.",
        ) from error
    stage = args.output.parent / f".{args.output.name}.staging-{uuid.uuid4().hex[:8]}"
    try:
        from mlx_lm import convert
    except ImportError as error:
        raise AutoQuantError(
            "MLX dependencies are not installed.",
            "Install with `pip install mlx-autoquant` on an Apple-silicon Mac.",
        ) from error
    try:
        status_stream = sys.stderr if args.json else sys.stdout
        print("Downloading model weights...", file=status_stream)
        snapshot = download_snapshot(preflight_result, cache_dir)
        print("Quantizing model...", file=status_stream)
        with (
            _activity_bar("Quantizing model"),
            redirect_stdout(sys.stderr if args.json else sys.stdout),
        ):
            convert(
                str(snapshot),
                mlx_path=str(stage),
                quantize=True,
                q_group_size=plan.group_size,
                q_bits=plan.bits,
                q_mode=plan.mode,
                revision=None,
                trust_remote_code=False,
            )
        verification = (
            None if args.no_verify else verify_model(stage, max_tokens=args.verify_tokens)
        )
        report = _write_report(
            stage,
            hardware,
            model,
            plan,
            preflight_result.resolved_revision,
            options,
            verification,
            preflight_result,
        )
        stage.replace(args.output)
        report = args.output / report.name
    except AutoQuantError as error:
        _cleanup_failed_stage(stage, error, preflight_result)
        raise
    except KeyboardInterrupt as error:
        cancelled = CancellationError(
            "Conversion cancelled before the model was promoted.",
            "Retry with the same model and a new output path.",
        )
        _cleanup_failed_stage(stage, cancelled, preflight_result)
        raise cancelled from error
    except Exception as error:
        converted = ConversionError(
            f"Conversion failed for {args.model!r}: "
            f"{str(error).splitlines()[0] or error.__class__.__name__}",
            f"Inspect the staging directory {stage} and retry with a new --output path.",
        )
        _cleanup_failed_stage(stage, converted, preflight_result)
        raise converted from error
    if args.json:
        print(
            json.dumps(
                {
                    "status": "success",
                    "hardware": hardware.to_dict(),
                    "model": model.to_dict(),
                    "plan": plan.to_dict(),
                    "options": [option.to_dict() for option in options],
                    "preflight": preflight_result.to_dict(),
                    "verification": verification.to_dict() if verification else {"skipped": True},
                    "output": str(args.output),
                    "report": str(report),
                },
                indent=2,
            )
        )
        return 0
    if verification:
        peak_memory = (
            f"{verification.peak_memory_gib:.2f} GiB"
            if verification.peak_memory_gib is not None
            else "unavailable"
        )
        print(
            f"Verification passed; generated {verification.generated_tokens} tokens, "
            f"peak memory: {peak_memory}."
        )
    if args.no_verify:
        print("Warning: output was not verified (--no-verify).")
    print(f"Saved quantized model to {args.output}; report: {report}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Choose an MLX quantization that fits this Mac, "
        "then convert a Hugging Face model."
    )
    parser.add_argument("model", help="Hugging Face model ID, e.g. Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output", type=Path, default=Path("./mlx-model"))
    parser.add_argument("--revision")
    parser.add_argument(
        "--context-length",
        type=int,
        help="Expected context length (default: model config, capped at 8192).",
    )
    parser.add_argument(
        "--bits",
        type=int,
        choices=range(2, 9),
        help="Request a bit-width; rejected when it does not fit the detected budget.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the metadata preflight without downloading tensor weights.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON instead of the summary."
    )
    parser.add_argument(
        "--no-verify", action="store_true", help="Skip the post-conversion generation smoke test."
    )
    parser.add_argument(
        "--yes", action="store_true", help="Skip conversion confirmation (for scripts and CI)."
    )
    parser.add_argument(
        "--verify-tokens",
        type=int,
        default=8,
        help="Tokens to generate during verification (default: 8).",
    )
    args = parser.parse_args(argv)
    if args.context_length is not None and args.context_length <= 0:
        parser.error("--context-length must be positive")
    if args.verify_tokens <= 0:
        parser.error("--verify-tokens must be positive")
    try:
        return _run(args)
    except AutoQuantError as error:
        print(f"error [{error.code}]: {error}", file=sys.stderr)
        if error.hint:
            print(f"next: {error.hint}", file=sys.stderr)
        return 1
    except Exception as error:
        message = str(error).splitlines()[0] or error.__class__.__name__
        print(f"error: {error.__class__.__name__}: {message}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
