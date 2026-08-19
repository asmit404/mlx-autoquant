from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .errors import VerificationError


@dataclass(frozen=True)
class VerificationResult:
    prompt: str
    generated_text: str
    generated_tokens: int
    peak_memory_gib: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def verify_model(
    model_path: Path,
    prompt: str = "Write one short sentence about Apple silicon.",
    max_tokens: int = 8,
) -> VerificationResult:
    """Load a converted model and generate a short response as a smoke test."""
    if max_tokens <= 0:
        raise VerificationError("Verification token count must be positive.")

    try:
        import mlx.core as mx
        from mlx_lm import generate, load
    except ImportError as error:
        raise VerificationError(
            "MLX verification dependencies are not installed.",
            "Install with `pip install mlx-autoquant` on an Apple-silicon Mac.",
        ) from error

    reset_peak_memory = (
        mx.reset_peak_memory if hasattr(mx, "reset_peak_memory") else mx.metal.reset_peak_memory
    )
    reset_peak_memory()
    model, tokenizer = load(str(model_path))
    generated = generate(model, tokenizer, prompt, max_tokens=max_tokens, verbose=False)
    if not generated.strip():
        raise VerificationError(
            "Verification generated an empty response.",
            "Retry with a different prompt or inspect the converted model "
            "in the diagnostic report.",
        )

    get_peak_memory = (
        mx.get_peak_memory if hasattr(mx, "get_peak_memory") else mx.metal.get_peak_memory
    )
    peak_memory = get_peak_memory()
    peak_memory_gib = peak_memory / 1024**3 if peak_memory and peak_memory > 0 else None
    generated_tokens = len(tokenizer.encode(generated))
    return VerificationResult(
        prompt=prompt,
        generated_text=generated,
        generated_tokens=generated_tokens,
        peak_memory_gib=round(peak_memory_gib, 2)
        if peak_memory_gib and peak_memory_gib >= 0.01
        else None,
    )
