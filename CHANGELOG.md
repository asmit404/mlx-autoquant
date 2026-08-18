# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Pinned metadata preflight with support checks, disk estimates, and a confirmation boundary before tensor downloads.
- Transactional conversion reports with verification status and diagnostic reports for failed conversions.
- Contributor guidance for model fixtures and macOS integration checks.

### Changed

- MLX-LM and Hugging Face Hub dependencies are bounded to tested major/minor ranges.
- Remote-code conversion is no longer enabled in the first supported workflow.

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
