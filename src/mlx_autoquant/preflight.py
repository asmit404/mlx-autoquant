from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .errors import (
    AuthenticationError,
    AutoQuantError,
    InsufficientDiskError,
    InsufficientMemoryError,
    MetadataError,
    NetworkError,
    UnsupportedModelError,
)
from .hardware import HardwareProfile
from .model import ModelProfile, profile_config
from .planner import estimated_model_gib, quantization_options

METADATA_PATTERNS = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "*.index.json",
)
WEIGHT_SUFFIXES = (".safetensors",)


@dataclass(frozen=True)
class PreflightResult:
    model_id: str
    requested_revision: str | None
    resolved_revision: str
    metadata_dir: Path
    model: ModelProfile
    context_length: int
    source_weight_bytes: int
    required_metadata_bytes: int
    available_disk_bytes: int
    estimated_temporary_bytes: int
    available_cache_bytes: int
    available_output_bytes: int
    required_cache_bytes: int
    required_output_bytes: int
    compatibility: str
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["metadata_dir"] = str(self.metadata_dir)
        data["model"] = self.model.to_dict()
        data["warnings"] = list(self.warnings)
        for key in (
            "source_weight_bytes",
            "required_metadata_bytes",
            "available_disk_bytes",
            "estimated_temporary_bytes",
        ):
            data[key] = int(data[key])
        return data


def _api():
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as error:
        raise MetadataError(
            "Hugging Face support is not installed.",
            "Install the package with `pip install mlx-autoquant`.",
        ) from error
    return HfApi(), snapshot_download


def _classify_hub_error(error: Exception, model_id: str) -> AutoQuantError:
    text = str(error).splitlines()[0] or error.__class__.__name__
    if "not found" in text.lower() or "repositorynotfound" in error.__class__.__name__.lower():
        return MetadataError(
            f"Hugging Face model {model_id!r} was not found: {text}",
            "Check the model ID and revision, then retry.",
        )
    if "401" in text or "403" in text or "gated" in text.lower() or "token" in text.lower():
        return AuthenticationError(
            f"Cannot access Hugging Face model {model_id!r}: {text}",
            "Set HF_TOKEN for private or gated models, then retry.",
        )
    if "timeout" in text.lower() or "connection" in text.lower():
        return NetworkError(
            f"Could not reach Hugging Face for {model_id!r}: {text}",
            "Check your network connection and retry.",
        )
    return MetadataError(
        f"Could not fetch metadata for {model_id!r}: {text}",
        "Check the model ID and revision, then retry.",
    )


def _required_file_sizes(info: Any) -> tuple[int, int, bool]:
    weight_bytes = 0
    metadata_bytes = 0
    unknown_size = False
    for sibling in getattr(info, "siblings", ()) or ():
        name = getattr(sibling, "rfilename", "")
        size = getattr(sibling, "size", None)
        if name.endswith("/"):
            continue
        required = (
            name.endswith(".index.json")
            or name.endswith(WEIGHT_SUFFIXES)
            or name in METADATA_PATTERNS
            or (name.endswith(".json") and "tokenizer" in name)
        )
        if size is None:
            unknown_size |= required
            continue
        if name.endswith(".index.json"):
            metadata_bytes += int(size)
        elif name.endswith(WEIGHT_SUFFIXES):
            weight_bytes += int(size)
        elif name in METADATA_PATTERNS or (name.endswith(".json") and "tokenizer" in name):
            metadata_bytes += int(size)
    return weight_bytes, metadata_bytes, unknown_size


def _exact_parameters(info: Any) -> int | None:
    safetensors = getattr(info, "safetensors", None)
    total = getattr(safetensors, "total", None) if safetensors is not None else None
    return int(total) if total else None


def _cached_size(root: Path, suffixes: tuple[str, ...]) -> int:
    return sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file() and path.name.endswith(suffixes)
    )


def _compatibility(config_path: Path) -> str:
    config = json.loads(config_path.read_text())
    architectures = set(config.get("architectures", ()))
    model_type = config.get("model_type")
    supported = {
        "LlamaForCausalLM",
        "Qwen2ForCausalLM",
    }
    supported_types = {"llama", "qwen2"}
    if architectures:
        return "supported" if architectures & supported else "unsupported"
    if model_type in supported_types:
        return "supported"
    if not architectures and not model_type:
        return "needs_review"
    return "unsupported"


def preflight(
    model_id: str,
    revision: str | None,
    hardware: HardwareProfile,
    context_length: int | None,
    cache_dir: Path,
    output: Path,
) -> PreflightResult:
    api, snapshot_download = _api()
    try:
        info = api.model_info(model_id, revision=revision, files_metadata=True)
        resolved_revision = getattr(info, "sha", None) or revision or "main"
        metadata_dir = Path(
            snapshot_download(
                repo_id=model_id,
                revision=resolved_revision,
                allow_patterns=list(METADATA_PATTERNS),
                ignore_patterns=["*.safetensors"],
                cache_dir=str(cache_dir),
            )
        )
    except Exception as error:
        raise _classify_hub_error(error, model_id) from error

    config_path = metadata_dir / "config.json"
    if not config_path.exists():
        raise MetadataError(
            f"Model {model_id!r} has no config.json.",
            "Use a standard Transformers text-generation checkpoint.",
        )
    try:
        compatibility = _compatibility(config_path)
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise MetadataError(
            f"Could not parse config.json for {model_id!r}: {error}",
            "Use a valid Transformers config.json and retry.",
        ) from error
    if compatibility != "supported":
        raise UnsupportedModelError(
            f"Model {model_id!r} is not supported before conversion ({compatibility}).",
            "Use a tested Llama or Qwen causal checkpoint, or add a support fixture.",
        )
    config = json.loads(config_path.read_text())
    if not any(key in config for key in ("num_attention_heads", "num_heads", "n_head")):
        raise MetadataError(
            f"Model {model_id!r} is missing attention head metadata.",
            "Use a complete causal text-generation config.json.",
        )
    try:
        model = profile_config(model_id, config_path, _exact_parameters(info))
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise MetadataError(
            f"Could not profile model metadata for {model_id!r}: {error}",
            "Check that config.json describes a causal text-generation model.",
        ) from error
    required_dimensions = {
        "hidden_size": model.hidden_size,
        "layers": model.layers,
        "num_attention_heads": model.num_attention_heads,
        "num_key_value_heads": model.num_key_value_heads,
        "head_dim": model.head_dim,
        "parameters": model.parameters,
    }
    missing_dimensions = [name for name, value in required_dimensions.items() if value <= 0]
    if missing_dimensions:
        raise MetadataError(
            f"Model {model_id!r} is missing positive dimensions: {', '.join(missing_dimensions)}.",
            "Use a complete causal text-generation config.json.",
        )

    source_weight_bytes, metadata_bytes, unknown_size = _required_file_sizes(info)
    if unknown_size:
        raise MetadataError(
            f"Hugging Face did not provide sizes for required files in {model_id!r}.",
            "Retry later or use a repository with complete Hub file metadata.",
        )
    if source_weight_bytes <= 0:
        raise MetadataError(
            f"Model {model_id!r} has no sized model weight files.",
            "Use a checkpoint with sized safetensors weights.",
        )
    effective_context = context_length or model.recommended_context_length
    options = quantization_options(hardware, model, effective_context)
    selected = next((option for option in options if option.fits), None)
    if selected is None:
        raise InsufficientMemoryError(
            "No supported quantization candidate fits the detected memory budget.",
            "Choose a smaller model or context length.",
        )
    output_bytes = int(estimated_model_gib(model.parameters, selected.bits) * 1024**3)
    cached_weight_bytes = _cached_size(metadata_dir, WEIGHT_SUFFIXES)
    cached_metadata_bytes = _cached_size(metadata_dir, (".json",))
    required_cache_bytes = max(0, source_weight_bytes - cached_weight_bytes) + max(
        0, metadata_bytes - cached_metadata_bytes
    )
    temporary_bytes = required_cache_bytes + output_bytes
    cache_root = cache_dir
    while not cache_root.exists() and cache_root != cache_root.parent:
        cache_root = cache_root.parent
    cache_free = shutil.disk_usage(cache_root).free
    output_parent = output.parent
    while not output_parent.exists() and output_parent != output_parent.parent:
        output_parent = output_parent.parent
    output_free = shutil.disk_usage(output_parent).free
    same_filesystem = cache_root.stat().st_dev == output_parent.stat().st_dev
    available = min(cache_free, output_free)
    insufficient_space = (
        required_cache_bytes + output_bytes > cache_free
        if same_filesystem
        else required_cache_bytes > cache_free or output_bytes > output_free
    )
    if insufficient_space:
        raise InsufficientDiskError(
            f"Conversion needs {required_cache_bytes / 1024**3:.2f} GiB in the cache "
            f"and {output_bytes / 1024**3:.2f} GiB in the output filesystem; "
            f"free space is {cache_free / 1024**3:.2f} GiB and "
            f"{output_free / 1024**3:.2f} GiB.",
            "Free disk space or choose a smaller model/context.",
        )
    return PreflightResult(
        model_id,
        revision,
        resolved_revision,
        metadata_dir,
        model,
        effective_context,
        source_weight_bytes,
        metadata_bytes,
        available,
        temporary_bytes,
        cache_free,
        output_free,
        required_cache_bytes,
        output_bytes,
        compatibility,
    )


def download_snapshot(result: PreflightResult, cache_dir: Path) -> Path:
    _, snapshot_download = _api()
    try:
        return Path(
            snapshot_download(
                repo_id=result.model_id,
                revision=result.resolved_revision,
                allow_patterns=list(METADATA_PATTERNS) + ["*.safetensors"],
                ignore_patterns=["*.md", "*.png", "*.jpg"],
                cache_dir=str(cache_dir),
            )
        )
    except Exception as error:
        raise _classify_hub_error(error, result.model_id) from error
