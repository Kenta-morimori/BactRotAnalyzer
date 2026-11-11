PYTHON ?= .venv/bin/python
export MPLBACKEND ?= Agg
export MPLCONFIGDIR ?= /tmp/bactrot-matplotlib

.PHONY: check format-check lint test test-runtime typecheck

check: format-check lint test

format-check:
	$(PYTHON) -m black . --check --diff
	$(PYTHON) -m isort . --check --diff

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m pytest -q

test-runtime:
	$(PYTHON) -m pytest -q tests/test_run_context.py

typecheck:
	$(PYTHON) -m mypy .
