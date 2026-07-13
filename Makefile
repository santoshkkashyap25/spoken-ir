# TransNLP — convenience targets for Docker workflow.
# Requires: Docker Desktop with Compose V2 (docker compose, not docker-compose).
#
# Windows users: run these from Git Bash, WSL2, or PowerShell with 'make'
# installed (e.g. via Chocolatey: choco install make).

.PHONY: help build up down logs shell-backend shell-frontend clean nuke

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Core lifecycle ────────────────────────────────────────────────────────────

build: ## Build (or rebuild) the Docker image
	docker compose build

up: ## Start all services in the background (builds if needed)
	docker compose up --build -d
	@echo ""
	@echo "  ✅  Backend  → http://localhost:8000"
	@echo "  ✅  Frontend → http://localhost:8501"
	@echo ""
	@echo "  Run 'make logs' to tail the logs."

down: ## Stop and remove containers (keeps volumes/images)
	docker compose down

logs: ## Tail logs for all services (Ctrl-C to exit)
	docker compose logs -f

logs-backend: ## Tail backend logs only
	docker compose logs -f backend

logs-frontend: ## Tail frontend logs only
	docker compose logs -f frontend

# ── Debugging ─────────────────────────────────────────────────────────────────

shell-backend: ## Open a bash shell in the running backend container
	docker compose exec backend /bin/bash

shell-frontend: ## Open a bash shell in the running frontend container
	docker compose exec frontend /bin/bash

health: ## Check backend /health endpoint
	curl -s http://localhost:8000/health | python -m json.tool

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean: ## Remove stopped containers and dangling images
	docker compose down --remove-orphans
	docker image prune -f

nuke: ## ⚠️  Remove ALL transnlp containers, images, and build cache
	docker compose down --rmi all --volumes --remove-orphans
	docker builder prune -f
