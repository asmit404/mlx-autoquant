# MLX AutoQuant

[![CI](https://github.com/asmit404/mlx-autoquant/actions/workflows/ci.yml/badge.svg)](https://github.com/asmit404/mlx-autoquant/actions/workflows/ci.yml)

`mlx-autoquant` converts a Hugging Face Transformers checkpoint to MLX, choosing the highest quantization precision that safely fits the current Apple-silicon Mac.

It reads your unified-memory capacity, profiles `config.json` and Hugging Face safetensors metadata, reserves memory for macOS and a KV cache, then calls the maintained `mlx_lm.convert` API.

## Install

Install the released package from PyPI:

```bash
pip install mlx-autoquant
```

For local development:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Use

Preview the decision without downloading weights. The context default is read from the model config and capped at 8192 tokens; pass `--context-length` to override it:

```bash
mlx-autoquant Qwen/Qwen2.5-7B-Instruct --dry-run
mlx-autoquant Qwen/Qwen2.5-7B-Instruct --dry-run --context-length 16384
```

Convert using the automatic decision:

```bash
mlx-autoquant Qwen/Qwen2.5-7B-Instruct --output ./Qwen2.5-7B-MLX
```

Use a larger expected context window, or take responsibility for a fixed precision:

```bash
mlx-autoquant Qwen/Qwen2.5-7B-Instruct --context-length 16384
mlx-autoquant Qwen/Qwen2.5-7B-Instruct --bits 4 --output ./Qwen-4bit
```

Add `--json` for machine-readable output. It includes every tested bit-width and whether it fits:

```bash
mlx-autoquant Qwen/Qwen2.5-7B-Instruct --dry-run --json
```

Conversions run an 8-token generation smoke test after quantization. Use
`--no-verify` to skip it, or `--verify-tokens 16` to generate more tokens.

Each completed conversion writes `autoquant-report.json` next to the MLX model. It records the detected machine, the model dimensions, the selected bits, and the memory assumptions. Parameter counts come from Hugging Face's safetensors metadata; when a repository has no safetensors weights, the tool falls back to an estimate from `config.json` and labels it as such.

Errors are printed to stderr with a non-zero exit code instead of a traceback. Downloads are cached under `~/.cache/mlx-autoquant`; set `HF_HOME` to move the Hugging Face cache, and `HF_TOKEN` to authenticate private or gated models.

During a real conversion you get two progress indicators: an aggregate bar while the weights download, then an activity bar while the model is quantized. The verification result and peak MLX memory are written to `autoquant-report.json`.

## Design boundaries

- Apple silicon only: MLX does not run on other hardware.
- Supports standard Transformers checkpoints with `config.json`; remote code is opt-in.
- A 15% conversion/sharding allowance and a 25% (minimum 4 GiB) system reserve make the automatic choice conservative.
- The user can always choose `--bits`, but the command intentionally displays the resulting plan first.
- Without safetensors weights, parameter counts are estimates; the report and summary mark them as estimated.

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

```bash
ruff check src tests          # lint
ruff format --check src tests # format
mypy src/mlx_autoquant        # type check
pytest                        # tests
make coverage                 # test coverage
```
