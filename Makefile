.PHONY: setup lint format typecheck test check data

data: ## Download the historical dataset from Hugging Face (~6.5 GB, idempotent)
	uv run python -m dbahn_delay.data.download

setup: ## Install dependencies and git hooks
	uv sync
	uv run pre-commit install

lint: ## Check code style without modifying files
	uv run ruff check .
	uv run ruff format --check .

format: ## Auto-fix style issues and reformat
	uv run ruff check --fix .
	uv run ruff format .

typecheck: ## Run strict static type checks
	uv run mypy

test: ## Run the test suite
	uv run pytest

check: lint typecheck test ## Run everything CI runs
