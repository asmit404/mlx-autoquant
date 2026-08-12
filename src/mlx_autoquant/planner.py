from __future__ import annotations

from dataclasses import asdict, dataclass

from .hardware import HardwareProfile
from .model import ModelProfile


@dataclass(frozen=True)
class QuantizationPlan:
    bits: int
    group_size: int
    mode: str
    estimated_model_gib: float
    estimated_kv_cache_gib: float
    available_for_model_gib: float
    context_length: int
    rationale: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class QuantizationOption:
    bits: int
    estimated_model_gib: float
    fits: bool

    def to_dict(self) -> dict:
        return asdict(self)


SUPPORTED_BITS = (8, 7, 6, 5, 4, 3, 2)


def estimated_model_gib(parameters: int, bits: int, group_size: int = 64) -> float:
    # Packed weights + fp16 scale/bias per group, then a 15% runtime/shard allowance.
    bytes_per_weight = bits / 8 + 4 / group_size
    return parameters * bytes_per_weight * 1.15 / 1024**3


def estimated_kv_cache_gib(model: ModelProfile, context_length: int) -> float:
    values = 2 * model.layers * model.num_key_value_heads * model.head_dim * context_length
    return values * 2 / 1024**3  # fp16 K and V


def _model_budget(
    hardware: HardwareProfile, model: ModelProfile, context_length: int
) -> tuple[float, float]:
    kv_gib = estimated_kv_cache_gib(model, context_length)
    reserve_gib = max(4.0, hardware.memory_gib * 0.25)
    return kv_gib, hardware.memory_gib - reserve_gib - kv_gib


def quantization_options(
    hardware: HardwareProfile,
    model: ModelProfile,
    context_length: int = 4096,
    group_size: int = 64,
) -> tuple[QuantizationOption, ...]:
    if not hardware.is_apple_silicon:
        raise RuntimeError("MLX quantization requires an Apple-silicon Mac.")
    _, budget_gib = _model_budget(hardware, model, context_length)
    return tuple(
        QuantizationOption(
            bits,
            round(estimated_model_gib(model.parameters, bits, group_size), 2),
            estimated_model_gib(model.parameters, bits, group_size) <= budget_gib,
        )
        for bits in SUPPORTED_BITS
    )


def choose_quantization(
    hardware: HardwareProfile,
    model: ModelProfile,
    context_length: int = 4096,
    group_size: int = 64,
) -> QuantizationPlan:
    kv_gib, budget_gib = _model_budget(hardware, model, context_length)
    reserve_gib = hardware.memory_gib - kv_gib - budget_gib
    for option in quantization_options(hardware, model, context_length, group_size):
        if option.fits:
            return QuantizationPlan(
                option.bits,
                group_size,
                "affine",
                option.estimated_model_gib,
                round(kv_gib, 2),
                round(budget_gib, 2),
                context_length,
                f"Highest tested precision fitting {budget_gib:.2f} GiB after a "
                f"{reserve_gib:.2f} GiB system reserve and KV cache.",
            )
    smallest = estimated_model_gib(model.parameters, 2, group_size)
    raise RuntimeError(
        f"Even 2-bit weights need {smallest:.2f} GiB, "
        f"but only {budget_gib:.2f} GiB is safely available. Choose a smaller model or context."
    )
