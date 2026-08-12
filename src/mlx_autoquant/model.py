from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelProfile:
    model_id: str
    parameters: int
    layers: int
    hidden_size: int
    num_key_value_heads: int
    head_dim: int
    parameters_exact: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _first(config: dict[str, Any], *names: str, default: int = 0) -> int:
    for name in names:
        if config.get(name) is not None:
            return int(config[name])
    return default


def _count_from_index(index_path: Path, bytes_per_element: int) -> int | None:
    try:
        data = json.loads(index_path.read_text())
        total_bytes = data.get("metadata", {}).get("total_size")
        return int(total_bytes) // bytes_per_element if total_bytes else None
    except (OSError, ValueError, TypeError):
        return None


def _bytes_per_element(config: dict[str, Any]) -> int:
    dtype = str(config.get("torch_dtype", "bfloat16")).lower()
    return 4 if dtype in {"float32", "fp32", "float"} else 2


def _estimate_parameters(config: dict[str, Any]) -> int:
    hidden = _first(config, "hidden_size", "n_embd", "d_model")
    layers = _first(config, "num_hidden_layers", "n_layer", "num_layers")
    heads = _first(config, "num_attention_heads", "n_head", "num_heads", default=1)
    kv_heads = _first(config, "num_key_value_heads", "num_kv_heads", default=heads)
    head_dim = _first(config, "head_dim", default=hidden // heads if heads else 0)
    vocab = _first(config, "vocab_size")
    intermediate = _first(config, "intermediate_size", "n_inner", "ffn_dim", default=4 * hidden)

    # Decoder-only estimate: embeddings + each block's attention and MLP weights.
    attention = 2 * hidden * hidden + 2 * hidden * kv_heads * head_dim
    mlp = 3 * hidden * intermediate  # SwiGLU family; close for common decoder models
    return vocab * hidden + layers * (attention + mlp) + hidden * vocab


def profile_config(
    model_id: str, config_path: Path, index_path: Path | None = None
) -> ModelProfile:
    """Profile a model from config.json, preferring exact counts from the weight index."""
    config = json.loads(config_path.read_text())
    hidden = _first(config, "hidden_size", "n_embd", "d_model")
    layers = _first(config, "num_hidden_layers", "n_layer", "num_layers")
    heads = _first(config, "num_attention_heads", "n_head", "num_heads", default=1)
    kv_heads = _first(config, "num_key_value_heads", "num_kv_heads", default=heads)
    head_dim = _first(config, "head_dim", default=hidden // heads if heads else 0)

    exact = (
        _count_from_index(index_path, _bytes_per_element(config))
        if index_path is not None
        else None
    )
    parameters = exact if exact is not None else _estimate_parameters(config)
    return ModelProfile(model_id, parameters, layers, hidden, kv_heads, head_dim, exact is not None)
