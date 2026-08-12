.PHONY: install lint format type test check build smoke

install:
	python3 -m venv .venv
	.venv/bin/pip install -e ".[dev]"

lint:
	.venv/bin/ruff check src tests

format:
	.venv/bin/ruff format --check src tests

format-fix:
	.venv/bin/ruff check --fix src tests
	.venv/bin/ruff format src tests

type:
	.venv/bin/mypy src/mlx_autoquant

test:
	.venv/bin/python -m pytest

check: lint format type test

build:
	.venv/bin/pip wheel --no-deps -w dist .

smoke:
	.venv/bin/mlx-autoquant --help
