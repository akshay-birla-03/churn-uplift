.PHONY: install test lint format run

install:
	pip install --break-system-packages -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests

format:
	ruff check --fix src tests

run:
	upliftkit
