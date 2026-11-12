PY ?= python

.PHONY: gate gate-real install install-models test lint format typecheck smoke experiments demo serve loadtest check

install:
	$(PY) -m pip install -e ".[dev]"

install-models:
	$(PY) -m pip install -e ".[dev,models]"

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

typecheck:
	$(PY) -m mypy

check: lint typecheck test

# Fake backends only. Writes to reports/smoke/, never touches README results.
smoke:
	$(PY) scripts/run_experiments.py --smoke

# Real models (sentence-transformers + your LLM from .env). Fills the README results block.
experiments:
	$(PY) scripts/run_experiments.py

demo:
	$(PY) scripts/demo.py

serve:
	$(PY) -m uvicorn --factory semcache.api:create_app --port 8000

loadtest:
	$(PY) loadtest/run_load.py

# Eval gates: fail if measured numbers regress (eval_gate.json)
gate:
	$(PY) scripts/eval_gate.py --mode smoke

gate-real:
	$(PY) scripts/eval_gate.py --mode real
