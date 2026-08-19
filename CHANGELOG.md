# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-08-19

### Added

- Metadata preflight now checks model support, attention metadata, memory, and disk space before tensor downloads.
- Conversions now produce verified reports and preserve actionable diagnostics when conversion fails.
- Added pinned Qwen2 support fixtures and contributor guidance for macOS integration checks.

### Changed

- The CLI supports machine-readable preflight and conversion results, explicit confirmation, and `--yes` automation.
- Only validated Llama and Qwen2 safetensors checkpoints are admitted by automatic preflight.
- MLX-LM and Hugging Face Hub dependencies remain bounded to tested major/minor ranges, with remote code disabled.

### Fixed

- Forced bit-widths now use the same memory and filesystem safety checks as automatic selection.
- Incomplete model metadata, unsupported architectures, and conversion races now fail before partial output can be promoted.
- CI and integration fixtures now exercise the pinned conversion and verification contract.

## [Unreleased]

## [0.2.0] - 2026-08-13

### Added

- Exact parameter counts from Hugging Face safetensors metadata (fallback to a
  config-based estimate, including MoE-aware estimation, for repos without
  safetensors weights).
- Human-readable CLI summary with `--json` for machine-readable output.
- Two-phase progress bars: an aggregate download bar for the weights, then an
  activity bar during quantization.
- Clean error handling: errors go to stderr with a non-zero exit code.
- CI (lint, format, type check, tests on macOS) and a tag-triggered
  PyPI release workflow using Trusted Publishing.
- `python -m mlx_autoquant` entry point, single-sourced version, `py.typed` marker.
- Post-conversion generation smoke test with peak MLX memory reporting.
- Config-aware context defaults capped at 8192 tokens.
- Dry-run comparison of every tested bit-width.
- Coverage tooling and a real-model macOS integration job.

### Changed

- Migrated to a `src/` layout with packaging tooling (ruff, mypy, pytest).

## [0.1.0] - 2026-08-13

### Added

- Initial release: hardware detection, config-based model profiling, automatic
  bit-width selection, and `mlx_lm.convert` integration.
