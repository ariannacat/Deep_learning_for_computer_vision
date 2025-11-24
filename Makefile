.PHONY: setup install test lint run parse decide clean

# --- Setup environment and install dependencies ---
setup:
	python -m pip install -U pip
	pip install -e ".[dev]"
	pre-commit install

# --- Install only runtime deps (no dev tools) ---
install:
	pip install -e .

# --- Lint the code ---
lint:
	ruff check src cli

# --- Run all tests ---
test:
	pytest -q

# --- End-to-end run examples ---
run:
	pokeai run data/samples/example.png --config configs/default.yaml

parse:
	pokeai parse data/samples/example.png

decide:
	pokeai decide --state examples/sample_outputs/state.json

# --- Clean build and cache files ---
clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache build dist *.egg-info

