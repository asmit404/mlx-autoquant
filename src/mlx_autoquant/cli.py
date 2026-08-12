from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from .hardware import HardwareProfile, detect_hardware
from .model import ModelProfile, profile_config
from .planner import QuantizationPlan, choose_quantization, estimated_model_gib


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


def _download_weights(model_id: str, revision: str | None) -> None:
    """Download every repo file into the default HF cache behind one progress bar."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
        from huggingface_hub.utils.tqdm import disable_progress_bars
        from tqdm import tqdm
    except ImportError as error:
        raise RuntimeError("Install dependencies with: pip install -e .") from error
    disable_progress_bars()
    try:
        api = HfApi()
        files = sorted(api.list_repo_files(model_id, revision=revision))
        paths_info = api.get_paths_info(model_id, files, revision=revision)
        entries = []
        for name, info in zip(files, paths_info, strict=True):
            size = getattr(info, "size", None)
            if size is None:
                continue
            entries.append((name, int(size)))
    except Exception as error:
        message = str(error).splitlines()[0] or error.__class__.__name__
        raise RuntimeError(f"Could not list files for {model_id!r}: {message}") from error
    total_bytes = sum(size for _, size in entries)
    with tqdm(total=total_bytes, unit="B", unit_scale=True, desc="Downloading weights") as bar:
        for name, size in entries:
            try:
                hf_hub_download(model_id, name, revision=revision)
            except Exception as error:
                message = str(error).splitlines()[0] or error.__class__.__name__
                raise RuntimeError(f"Could not download {name!r}: {message}") from error
            bar.update(size)


def _fetch_metadata(
    model_id: str, revision: str | None, cache_dir: Path
) -> tuple[Path, int | None]:
    try:
        from huggingface_hub import HfApi, snapshot_download
        from huggingface_hub.utils.tqdm import disable_progress_bars
    except ImportError as error:
        raise RuntimeError("Install dependencies with: pip install -e .") from error
    disable_progress_bars()
    try:
        config = (
            Path(
                snapshot_download(
                    repo_id=model_id,
                    revision=revision,
                    allow_patterns=["config.json"],
                    cache_dir=str(cache_dir),
                )
            )
            / "config.json"
        )
        info = HfApi().model_info(model_id, revision=revision, files_metadata=True)
    except Exception as error:
        message = str(error).splitlines()[0] or error.__class__.__name__
        raise RuntimeError(f"Could not fetch metadata for {model_id!r}: {message}") from error
    if not config.exists():
        raise RuntimeError(
            "The repository has no config.json; it is not a supported Transformers checkpoint."
        )
    safetensors = getattr(info, "safetensors", None)
    exact = getattr(safetensors, "total", None) if safetensors is not None else None
    return config, int(exact) if exact else None


def _write_report(
    destination: Path, hardware: Any, model: Any, plan: Any, revision: str | None
) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    report = destination / "autoquant-report.json"
    report.write_text(
        json.dumps(
            {
                "hardware": hardware.to_dict(),
                "model": model.to_dict(),
                "plan": plan.to_dict(),
                "revision": revision,
                "parameter_count_is_estimated_from_config": not model.parameters_exact,
            },
            indent=2,
        )
        + "\n"
    )
    return report


def _format_params(parameters: int) -> str:
    for unit, threshold in (("T", 10**12), ("B", 10**9), ("M", 10**6), ("K", 10**3)):
        if parameters >= threshold:
            value = parameters / threshold
            text = f"{value:.1f}"
            return f"{text[:-2]}{unit}" if text.endswith(".0") else f"{text}{unit}"
    return str(parameters)


def _print_summary(hardware: HardwareProfile, model: ModelProfile, plan: QuantizationPlan) -> None:
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
        "",
        "Plan",
        f"  Quantization:    {plan.bits}-bit {plan.mode} (group size {plan.group_size})",
        f"  Model weights:   {plan.estimated_model_gib:.2f} GiB",
        f"  KV cache:        {plan.estimated_kv_cache_gib:.2f} GiB at {plan.context_length} tokens",
        f"  Available:       {plan.available_for_model_gib:.2f} GiB",
        f"  Rationale:       {plan.rationale}",
    ]
    print("\n".join(lines))


def _run(args: argparse.Namespace) -> int:
    hardware = detect_hardware()
    config, exact = _fetch_metadata(
        args.model, args.revision, Path.home() / ".cache" / "mlx-autoquant"
    )
    model = profile_config(args.model, config, exact)
    plan = choose_quantization(hardware, model, args.context_length)
    if args.bits:
        plan = replace(
            plan,
            bits=args.bits,
            estimated_model_gib=round(estimated_model_gib(model.parameters, args.bits), 2),
            rationale="User-selected bit-width; fit has not been overridden by the planner.",
        )
    if args.json:
        print(
            json.dumps(
                {"hardware": hardware.to_dict(), "model": model.to_dict(), "plan": plan.to_dict()},
                indent=2,
            )
        )
    else:
        _print_summary(hardware, model, plan)
    if args.dry_run:
        if not args.json:
            print()
            print("Dry run complete. Re-run without --dry-run to convert the model.")
        return 0
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {args.output}")
    try:
        from mlx_lm import convert
    except ImportError as error:
        raise RuntimeError(
            "Install dependencies on an Apple-silicon Mac: pip install -e ."
        ) from error
    print()
    print("Downloading model weights...")
    _download_weights(args.model, args.revision)
    print()
    print("Quantizing model...")
    with _activity_bar("Quantizing model"):
        convert(
            args.model,
            mlx_path=str(args.output),
            quantize=True,
            q_group_size=plan.group_size,
            q_bits=plan.bits,
            q_mode=plan.mode,
            revision=args.revision,
            trust_remote_code=args.trust_remote_code,
        )
    report = _write_report(args.output, hardware, model, plan, args.revision)
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
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument(
        "--bits", type=int, choices=range(2, 9), help="Override the automatic bit-width."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Download only config.json and print the decision."
    )
    parser.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON instead of the summary."
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args(argv)
    if args.context_length <= 0:
        parser.error("--context-length must be positive")
    try:
        return _run(args)
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
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
