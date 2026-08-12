# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Exact parameter counts from `model.safetensors.index.json` (fallback to a
  config-based estimate when no index exists).
- Human-readable CLI summary with `--json` for machine-readable output.
- Clean error handling: errors go to stderr with a non-zero exit code.
- CI (lint, format, type check, tests on macOS and Linux) and a tag-triggered
  PyPI release workflow using Trusted Publishing.
- `python -m mlx_autoquant` entry point, single-sourced version, `py.typed` marker.

### Changed

- Migrated to a `src/` layout with packaging tooling (ruff, mypy, pytest).

## [0.1.0] - 2026-08-13

### Added

- Initial release: hardware detection, config-based model profiling, automatic
  bit-width selection, and `mlx_lm.convert` integration.
