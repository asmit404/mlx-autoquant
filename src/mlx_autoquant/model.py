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
    num_experts = _first(config, "num_experts", "num_local_experts")
    if num_experts:
        # MoE: routed experts plus (optional) shared experts.
        expert_size = _first(
            config, "moe_intermediate_size", "intermediate_size", default=4 * hidden
        )
        shared_size = _first(
            config, "shared_expert_intermediate_size", "intermediate_size", default=expert_size
        )
        shared_count = _first(config, "num_shared_experts", default=1)
        mlp = 3 * hidden * (shared_count * shared_size + num_experts * expert_size)
    else:
        mlp = 3 * hidden * intermediate  # SwiGLU family; close for common decoder models
    return vocab * hidden + layers * (attention + mlp) + hidden * vocab


def profile_config(
    model_id: str, config_path: Path, exact_parameters: int | None = None
) -> ModelProfile:
    """Profile a model from config.json, using an exact count when available."""
    config = json.loads(config_path.read_text())
    hidden = _first(config, "hidden_size", "n_embd", "d_model")
    layers = _first(config, "num_hidden_layers", "n_layer", "num_layers")
    heads = _first(config, "num_attention_heads", "n_head", "num_heads", default=1)
    kv_heads = _first(config, "num_key_value_heads", "num_kv_heads", default=heads)
    head_dim = _first(config, "head_dim", default=hidden // heads if heads else 0)

    parameters = exact_parameters if exact_parameters is not None else _estimate_parameters(config)
    return ModelProfile(
        model_id, parameters, layers, hidden, kv_heads, head_dim, exact_parameters is not None
    )
