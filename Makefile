# Trade platform — developer entry points.
# Every command below is also what CI runs, so a green `make ci` locally means
# GitHub Actions is highly likely to pass too.

.PHONY: help install lint format type test test-cov ci hello dev-up dev-down dev-logs clean

help:
	@echo "install    Install the package with dev extras into the active env"
	@echo "lint       Run ruff check + format check"
	@echo "format     Run ruff --fix and ruff format"
	@echo "type       Run mypy on src/"
	@echo "test       Run pytest with coverage"
	@echo "ci         lint + type + test (same as CI)"
	@echo "hello      Manual smoke: fetch a few BTCUSDT/ETHUSDT klines from Bybit"
	@echo "dev-up     Start the local docker-compose dev stack"
	@echo "dev-down   Stop the local dev stack"
	@echo "dev-logs   Tail the dev stack logs"
	@echo "clean      Remove caches and build artifacts"

install:
	uv pip install -e ".[dev]"

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff check --fix src tests
	ruff format src tests

type:
	mypy src

test:
	pytest

test-cov:
	pytest --cov-report=xml --cov-report=html

ci: lint type test

hello:
	python scripts/hello_bybit.py

dev-up:
	docker compose -f ops/docker-compose.yml up -d

dev-down:
	docker compose -f ops/docker-compose.yml down

dev-logs:
	docker compose -f ops/docker-compose.yml logs -f

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml build dist
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
