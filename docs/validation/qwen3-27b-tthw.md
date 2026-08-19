# Qwen3 27B Validation

Status: BLOCKED_PENDING_HUGGING_FACE_ACCESS
Target: Qwen3 27B on a 32 GB Apple Silicon Mac

## Goal

Measure the complete beginner path from install to verified output, including:

- Setup time and Python environment used
- Preflight duration and resolved Hub revision
- Selected quantization, estimated model size, KV-cache estimate, and disk budget
- Source download size and conversion duration
- Conversion peak memory and verification peak memory when available
- Whether the model loads and generates non-empty output
- Every decision or error that is not explained by the CLI

## Reproduction

Run from an Apple Silicon Mac with the tested Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
time mlx-autoquant Qwen/Qwen3-27B --dry-run
time mlx-autoquant Qwen/Qwen3-27B --output ./Qwen3-27B-MLX --yes
```

Record the output, report path, free disk before and after, and the model revision. Do not start the conversion unless the preflight succeeds and enough disk is available.

## Current Blocker

On 2026-08-19, both the CLI preflight and direct Hub access returned:

```text
401 Client Error for Qwen/Qwen3-27B
```

No large model download was attempted. Re-run this checklist after authenticating to Hugging Face or confirming the public model identifier. A successful API compatibility probe is covered by `tests/test_cli.py::TestCli::test_mlx_lm_converter_contract`.
