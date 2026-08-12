# MLX AutoQuant

[![CI](https://github.com/asmit404/mlx-autoquant/actions/workflows/ci.yml/badge.svg)](https://github.com/asmit404/mlx-autoquant/actions/workflows/ci.yml)

`mlx-autoquant` converts a Hugging Face Transformers checkpoint to MLX, choosing the highest quantization precision that safely fits the current Apple-silicon Mac.

It reads your unified-memory capacity, downloads only `config.json` and `model.safetensors.index.json` to profile the model, reserves memory for macOS and a KV cache, then calls the maintained `mlx_lm.convert` API. MLX-LM is responsible for downloading and converting the full checkpoint.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Use

Preview the decision without downloading weights:

```bash
mlx-autoquant Qwen/Qwen2.5-7B-Instruct --dry-run
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

Add `--json` for machine-readable output (progress bars are suppressed during downloads):

```bash
mlx-autoquant Qwen/Qwen2.5-7B-Instruct --dry-run --json
```

Each completed conversion writes `autoquant-report.json` next to the MLX model. It records the detected machine, the model dimensions, the selected bits, and the memory assumptions. Parameter counts come from Hugging Face's safetensors metadata; when a repository has no safetensors weights, the tool falls back to an estimate from `config.json` and labels it as such.

Errors are printed to stderr with a non-zero exit code instead of a traceback. Downloads are cached under `~/.cache/mlx-autoquant`; set `HF_HOME` to move the Hugging Face cache, and `HF_TOKEN` to authenticate private or gated models.

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
```
