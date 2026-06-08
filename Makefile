.DEFAULT_GOAL := help
API_URL       := http://localhost:8080
QUERY         ?= "What are my open tasks?"
MODE          ?= factual

.PHONY: help install run test test-all lint format type-check \
        up down pull-models ingest health query clean build

help:
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ── Dev setup ────────────────────────────────────────────────────────────────

install: ## Install all dependencies (including dev)
	poetry install

# ── Running ──────────────────────────────────────────────────────────────────

run: ## Start the FastAPI server with hot-reload (requires .env)
	poetry run uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --reload

# ── Testing ──────────────────────────────────────────────────────────────────

test: ## Run unit tests
	poetry run pytest tests/unit/ -v

test-all: ## Run all tests (unit + integration)
	poetry run pytest tests/ -v

# ── Code quality ─────────────────────────────────────────────────────────────

lint: ## Check code with ruff
	poetry run ruff check src/ tests/

format: ## Auto-format with ruff
	poetry run ruff format src/ tests/
	poetry run ruff check --fix src/ tests/

type-check: ## Run mypy
	poetry run mypy src/ --ignore-missing-imports

# ── Infrastructure ───────────────────────────────────────────────────────────

up: ## Start Qdrant + Ollama via Docker Compose
	docker compose up qdrant ollama -d

down: ## Stop all Docker Compose services
	docker compose down

pull-models: ## Pull required Ollama models (one-time, ~4 GB)
	ollama pull nomic-embed-text
	ollama pull llama3.2:3b

# ── API operations ───────────────────────────────────────────────────────────

health: ## Check service health
	@curl -s $(API_URL)/health | python3 -m json.tool

ingest: ## Trigger a full vault re-index
	@curl -s -X POST $(API_URL)/api/v1/ingest | python3 -m json.tool

query: ## Send a query (override: make query QUERY="..." MODE=synthesis)
	@curl -s -X POST $(API_URL)/api/v1/query \
		-H "Content-Type: application/json" \
		-d '{"query": $(QUERY), "mode": "$(MODE)"}' | python3 -m json.tool

# ── Docker ───────────────────────────────────────────────────────────────────

build: ## Build the Docker image
	docker build -t vigilant-rag:latest .

# ── Cleanup ──────────────────────────────────────────────────────────────────

clean: ## Remove Python caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache  -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache  -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
