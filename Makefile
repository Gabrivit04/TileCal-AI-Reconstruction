.PHONY: install lint test demo

install:
	pip install -e .[dev]

lint:
	ruff check .

test:
	pytest

demo:
	python scripts/run_dummy_training.py
