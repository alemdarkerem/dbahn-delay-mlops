.PHONY: setup lint format typecheck test check data validate ingest features train mlflow-ui serve docker-build docker-run

features: ## Build the model-ready feature frame from the canonical dataset
	uv run python -m dbahn_delay.features.build

train: ## Run walk-forward CV + final model training (logs to MLflow)
	uv run python -m dbahn_delay.models.train

mlflow-ui: ## Browse experiment runs at http://localhost:5000
	uv run mlflow ui

serve: ## Run the API locally (newest bundle in models/ unless DBAHN_MODEL_DIR set)
	uv run uvicorn dbahn_delay.serving.app:app --reload --port 8000

docker-build: ## Build the production image
	docker build -t dbahn-delay-api .

docker-run: ## Run the production image with the local models volume
	docker compose up --build

live-fetch: ## One live cycle: fetch upcoming stops, log predictions + changes
	uv run python -m dbahn_delay.live.fetch

live-eval: ## Evaluate yesterday's predictions against observed outcomes
	uv run python -m dbahn_delay.live.evaluate_day

data: ## Download the historical dataset from Hugging Face (~6.5 GB, idempotent)
	uv run python -m dbahn_delay.data.download

validate: ## Validate all raw monthly files (tolerant raw profile)
	uv run python -m dbahn_delay.data.validate

ingest: ## Build the canonical stops dataset from raw files
	uv run python -m dbahn_delay.data.ingest

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
