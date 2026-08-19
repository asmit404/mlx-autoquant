# MLX AutoQuant

[![CI](https://github.com/asmit404/mlx-autoquant/actions/workflows/ci.yml/badge.svg)](https://github.com/asmit404/mlx-autoquant/actions/workflows/ci.yml)

`mlx-autoquant` converts a Hugging Face Transformers checkpoint to MLX, choosing the highest quantization precision that safely fits the current Apple-silicon Mac.

It reads your unified-memory capacity, profiles `config.json` and Hugging Face safetensors metadata, reserves memory for macOS and a KV cache, then calls the maintained `mlx_lm.convert` API.

## Requirements

- Apple Silicon Mac running the macOS version supported by the installed `mlx-lm` release.
- Python 3.10 or newer.
- Enough free disk space for the source weights, converted model, and temporary conversion allowance. The preflight reports this before downloading tensor shards.
- A Hugging Face token in `HF_TOKEN` for private or gated repositories.

## Install

Recommended isolated install:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install mlx-autoquant
```

If you already manage Python environments, the short form is `python -m pip install mlx-autoquant`.

For local development:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Use

Preview the decision without downloading tensor weights. This is the fastest way to know whether a model is supported and what will fit:

```bash
mlx-autoquant Qwen/Qwen2.5-7B-Instruct --dry-run
```

The summary includes the detected Mac, model parameters, selected bits, estimated model size, KV-cache estimate, available budget, and rationale. A successful preflight ends with `Dry run complete`; it should take under five minutes on a normal connection.

Typical output looks like this:

```text
Machine
  Chip:            Apple M4
  Unified memory:  32.00 GiB
Plan
  Quantization:    8-bit affine (group size 64)
  Model weights:   8.67 GiB
  Available:       23.56 GiB
  Rationale:       Highest tested precision fitting 23.56 GiB after system reserve and KV cache.
Dry run complete. Re-run without --dry-run to convert the model.
```

Convert using the automatic decision:

```bash
mlx-autoquant Qwen/Qwen2.5-7B-Instruct --output ./Qwen2.5-7B-MLX
```

The command prints the preflight first, then asks for confirmation before downloading tensor shards. It reports required disk space, shows download and conversion progress, runs a short verification generation, and writes `autoquant-report.json` beside the verified model. Use `--yes` for CI or scripts. Large-model conversion time depends on model size, network, disk, and Mac hardware.

Use a larger expected context window, or request a fixed precision that still fits the detected budget:

```bash
mlx-autoquant Qwen/Qwen2.5-7B-Instruct --context-length 16384
mlx-autoquant Qwen/Qwen2.5-7B-Instruct --bits 4 --output ./Qwen-4bit
```

Add `--json` for machine-readable output. It includes preflight data, every tested bit-width, and whether each candidate fits:

```bash
mlx-autoquant Qwen/Qwen2.5-7B-Instruct --dry-run --json
```

Conversions run an 8-token generation smoke test after quantization. Use
`--verify-tokens 16` to generate more tokens. `--no-verify` is an expert escape hatch: the output is explicitly marked unverified in the report and terminal output.

Each completed conversion writes `autoquant-report.json` next to the MLX model. It records the detected machine, the model dimensions, the selected bits, and the memory assumptions. Parameter counts come from Hugging Face's safetensors metadata; when that exact count is unavailable, the tool falls back to an estimate from `config.json` and labels it as such. Checkpoints must provide safetensors weights.

Errors are printed to stderr with a non-zero exit code instead of a traceback. Expected errors include a category, cause, next action, and diagnostic report when conversion has started. Downloads use one Hugging Face cache root; set `HF_HOME` to move it, and set `HF_TOKEN` to authenticate private or gated models.

During a real conversion you get two progress indicators: an aggregate bar while the weights download, then an activity bar while the model is quantized. The verification result and peak MLX memory are written to `autoquant-report.json`.

## Design boundaries

- Apple silicon only: MLX does not run on other hardware.
- Supports tested causal text-generation Transformers checkpoints with `config.json`; arbitrary remote code is not supported in the first release.
- A 15% conversion/sharding allowance and a 25% (minimum 4 GiB) system reserve make the automatic choice conservative.
- The user can request `--bits`, but a forced value is rejected if it does not fit the same budget.
- Without safetensors weights, parameter counts are estimates; the report and summary mark them as estimated.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for the fixture, support-matrix, and macOS integration workflow.

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
