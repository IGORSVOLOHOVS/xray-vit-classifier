.PHONY: install lint format test clean help

# Variables
PYTHON = python
PIP = pip
RUFF = ruff
MYPY = mypy
PYTEST = pytest

help:
	@echo "Available commands:"
	@echo "  install    Install dependencies and project in editable mode"
	@echo "  lint       Run Ruff and Mypy checks"
	@echo "  format     Format code with Ruff"
	@echo "  test       Run tests with Pytest"
	@echo "  clean      Remove temporary files and caches"

install:
	$(PIP) install -e .

lint:
	$(RUFF) check .
	$(MYPY) .

format:
	$(RUFF) format .

test:
	$(PYTEST) tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf *.egg-info
