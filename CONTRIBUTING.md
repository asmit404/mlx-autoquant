# Contributing

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

The runtime and integration tests require an Apple Silicon Mac. Unit tests can run elsewhere when MLX-dependent paths are mocked.

## Checks

```bash
ruff check src tests
ruff format --check src tests
mypy src/mlx_autoquant
pytest
make coverage
```

## Model Support

New model-family support requires lightweight metadata tests plus a pinned small conversion fixture. The fixture must load, convert, generate non-empty output, and write a report. Add the family to the support matrix only after the macOS integration check passes.

Update `docs/support-matrix.md` with the fixture repository and immutable revision when adding or changing support coverage.

The first implementation slice does not execute arbitrary remote code. Do not add a model requiring `trust_remote_code` without a separate security and compatibility review.

## Pull Requests

Describe the model family, MLX-LM version, hardware used for validation, expected disk usage, and verification result. Include the failure path when changing preflight, conversion cleanup, or report behavior.
