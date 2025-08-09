# APG Capability Registry - Development and Deployment Makefile
# Provides convenient commands for development, testing, and deployment

.PHONY: help install install-dev clean test test-cov lint format type-check security-check
.PHONY: dev dev-stop dev-logs dev-shell migrate migrate-create migrate-upgrade migrate-downgrade
.PHONY: docker-build docker-run docker-stop docker-clean
.PHONY: k8s-deploy k8s-delete k8s-status k8s-logs
.PHONY: backup restore health-check load-test docs docs-serve
.PHONY: ci pre-commit release

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[36m
GREEN := \033[32m
RED := \033[31m
YELLOW := \033[33m
NC := \033[0m # No Color

# Variables
PYTHON := python3.11
PIP := pip
DOCKER := docker
DOCKER_COMPOSE := docker-compose
KUBECTL := kubectl
PROJECT_NAME := apg-capability-registry
IMAGE_NAME := datacraft/$(PROJECT_NAME)
VERSION := $(shell grep -E '^__version__' capability_registry/__init__.py | cut -d'"' -f2)
DATABASE_URL := postgresql://registry_user:dev_password@localhost:5432/apg_capability_registry
TEST_DATABASE_URL := postgresql://registry_user:test_password@localhost:5432/apg_registry_test

help: ## Show this help message
	@echo "$(BLUE)APG Capability Registry - Development Commands$(NC)"
	@echo ""
	@echo "$(GREEN)Available commands:$(NC)"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ { printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# =============================================================================
# Environment Setup
# =============================================================================

install: ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt

install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements-dev.txt
	pre-commit install

clean: ## Clean up temporary files and caches
	@echo "$(BLUE)Cleaning up...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	rm -rf .mypy_cache/
	rm -rf logs/*.log

# =============================================================================
# Code Quality
# =============================================================================

lint: ## Run linting checks
	@echo "$(BLUE)Running linting checks...$(NC)"
	flake8 capability_registry/ tests/
	pylint capability_registry/

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	black capability_registry/ tests/
	isort capability_registry/ tests/

type-check: ## Run type checking with mypy
	@echo "$(BLUE)Running type checks...$(NC)"
	mypy capability_registry/

security-check: ## Run security checks
	@echo "$(BLUE)Running security checks...$(NC)"
	bandit -r capability_registry/
	safety check
	pip-audit

pre-commit: ## Run pre-commit hooks
	@echo "$(BLUE)Running pre-commit hooks...$(NC)"
	pre-commit run --all-files

# =============================================================================
# Testing
# =============================================================================

test: ## Run tests
	@echo "$(BLUE)Running tests...$(NC)"
	pytest tests/ -v

test-cov: ## Run tests with coverage
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	pytest tests/ -v --cov=capability_registry --cov-report=html --cov-report=term

test-ci: ## Run tests for CI/CD
	@echo "$(BLUE)Running CI tests...$(NC)"
	pytest tests/ci/ -v --cov=capability_registry --cov-report=xml

test-integration: ## Run integration tests
	@echo "$(BLUE)Running integration tests...$(NC)"
	pytest tests/ -v -m integration

test-performance: ## Run performance tests
	@echo "$(BLUE)Running performance tests...$(NC)"
	pytest tests/ -v -m performance

# =============================================================================
# Development Environment
# =============================================================================

dev: ## Start development environment
	@echo "$(BLUE)Starting development environment...$(NC)"
	$(DOCKER_COMPOSE) up -d

dev-build: ## Build and start development environment
	@echo "$(BLUE)Building and starting development environment...$(NC)"
	$(DOCKER_COMPOSE) up --build -d

dev-stop: ## Stop development environment
	@echo "$(BLUE)Stopping development environment...$(NC)"
	$(DOCKER_COMPOSE) down

dev-logs: ## Show development environment logs
	@echo "$(BLUE)Showing development logs...$(NC)"
	$(DOCKER_COMPOSE) logs -f

dev-shell: ## Open shell in development container
	@echo "$(BLUE)Opening shell in development container...$(NC)"
	$(DOCKER_COMPOSE) exec app bash

dev-clean: ## Clean development environment
	@echo "$(BLUE)Cleaning development environment...$(NC)"
	$(DOCKER_COMPOSE) down -v --remove-orphans
	$(DOCKER) system prune -f

# =============================================================================
# Database Management
# =============================================================================

migrate: ## Run database migrations
	@echo "$(BLUE)Running database migrations...$(NC)"
	alembic upgrade head

migrate-create: ## Create new migration
	@echo "$(BLUE)Creating new migration...$(NC)"
	@read -p "Migration message: " message; \
	alembic revision --autogenerate -m "$$message"

migrate-upgrade: ## Upgrade to specific migration
	@echo "$(BLUE)Upgrading to migration...$(NC)"
	@read -p "Migration revision: " revision; \
	alembic upgrade $$revision

migrate-downgrade: ## Downgrade to previous migration
	@echo "$(BLUE)Downgrading migration...$(NC)"
	alembic downgrade -1

migrate-history: ## Show migration history
	@echo "$(BLUE)Migration history:$(NC)"
	alembic history --verbose

migrate-current: ## Show current migration
	@echo "$(BLUE)Current migration:$(NC)"
	alembic current

# =============================================================================
# Docker Operations
# =============================================================================

docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	$(DOCKER) build -t $(IMAGE_NAME):$(VERSION) -t $(IMAGE_NAME):latest \
		--build-arg BUILD_DATE=$(shell date -u +'%Y-%m-%dT%H:%M:%SZ') \
		--build-arg VERSION=$(VERSION) \
		--build-arg VCS_REF=$(shell git rev-parse --short HEAD) \
		.

docker-run: ## Run Docker container
	@echo "$(BLUE)Running Docker container...$(NC)"
	$(DOCKER) run -d \
		--name $(PROJECT_NAME) \
		--env-file .env \
		-p 8000:8000 \
		$(IMAGE_NAME):latest

docker-stop: ## Stop Docker container
	@echo "$(BLUE)Stopping Docker container...$(NC)"
	$(DOCKER) stop $(PROJECT_NAME) || true
	$(DOCKER) rm $(PROJECT_NAME) || true

docker-clean: ## Clean Docker images and containers
	@echo "$(BLUE)Cleaning Docker resources...$(NC)"
	$(DOCKER) system prune -f
	$(DOCKER) image prune -f

docker-push: ## Push Docker image to registry
	@echo "$(BLUE)Pushing Docker image...$(NC)"
	$(DOCKER) push $(IMAGE_NAME):$(VERSION)
	$(DOCKER) push $(IMAGE_NAME):latest

# =============================================================================
# Kubernetes Operations
# =============================================================================

k8s-deploy: ## Deploy to Kubernetes
	@echo "$(BLUE)Deploying to Kubernetes...$(NC)"
	$(KUBECTL) apply -f k8s/

k8s-delete: ## Delete from Kubernetes
	@echo "$(BLUE)Deleting from Kubernetes...$(NC)"
	$(KUBECTL) delete -f k8s/

k8s-status: ## Check Kubernetes deployment status
	@echo "$(BLUE)Kubernetes deployment status:$(NC)"
	$(KUBECTL) get pods -n apg-registry
	$(KUBECTL) get services -n apg-registry
	$(KUBECTL) get ingress -n apg-registry

k8s-logs: ## Show Kubernetes logs
	@echo "$(BLUE)Kubernetes logs:$(NC)"
	$(KUBECTL) logs -f deployment/registry-app -n apg-registry

k8s-shell: ## Open shell in Kubernetes pod
	@echo "$(BLUE)Opening shell in Kubernetes pod...$(NC)"
	$(KUBECTL) exec -it deployment/registry-app -n apg-registry -- bash

# =============================================================================
# Database Operations
# =============================================================================

backup: ## Create database backup
	@echo "$(BLUE)Creating database backup...$(NC)"
	mkdir -p backups
	pg_dump $(DATABASE_URL) > backups/registry_backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "$(GREEN)Backup created in backups/ directory$(NC)"

restore: ## Restore database from backup
	@echo "$(BLUE)Restoring database from backup...$(NC)"
	@echo "$(YELLOW)Warning: This will drop the existing database!$(NC)"
	@read -p "Backup file: " backup_file; \
	read -p "Are you sure? (y/N): " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		dropdb --if-exists $(shell echo $(DATABASE_URL) | sed 's/.*\///'); \
		createdb $(shell echo $(DATABASE_URL) | sed 's/.*\///'); \
		psql $(DATABASE_URL) < $$backup_file; \
		echo "$(GREEN)Database restored successfully$(NC)"; \
	else \
		echo "$(YELLOW)Restore cancelled$(NC)"; \
	fi

# =============================================================================
# Monitoring and Health Checks
# =============================================================================

health-check: ## Check application health
	@echo "$(BLUE)Checking application health...$(NC)"
	curl -f http://localhost:8000/api/health || echo "$(RED)Health check failed$(NC)"

load-test: ## Run load tests
	@echo "$(BLUE)Running load tests...$(NC)"
	locust -f tests/load/locustfile.py --host=http://localhost:8000

performance-test: ## Run performance benchmarks
	@echo "$(BLUE)Running performance benchmarks...$(NC)"
	pytest tests/ -v --benchmark-only

# =============================================================================
# Documentation
# =============================================================================

docs: ## Build documentation
	@echo "$(BLUE)Building documentation...$(NC)"
	sphinx-build -b html docs/ docs/_build/html

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8080$(NC)"
	cd docs/_build/html && python -m http.server 8080

docs-clean: ## Clean documentation build
	@echo "$(BLUE)Cleaning documentation...$(NC)"
	rm -rf docs/_build/

# =============================================================================
# Release Management
# =============================================================================

release: ## Create a new release
	@echo "$(BLUE)Creating new release...$(NC)"
	@echo "Current version: $(VERSION)"
	@read -p "New version: " new_version; \
	sed -i "s/__version__ = \".*\"/__version__ = \"$$new_version\"/" capability_registry/__init__.py; \
	git add capability_registry/__init__.py; \
	git commit -m "Bump version to $$new_version"; \
	git tag -a "v$$new_version" -m "Release version $$new_version"; \
	echo "$(GREEN)Release $$new_version created. Push with: git push origin main --tags$(NC)"

changelog: ## Generate changelog
	@echo "$(BLUE)Generating changelog...$(NC)"
	git log --pretty=format:"- %s" $(shell git describe --tags --abbrev=0)..HEAD > CHANGELOG.tmp
	@echo "$(GREEN)Changelog generated in CHANGELOG.tmp$(NC)"

# =============================================================================
# CI/CD
# =============================================================================

ci: clean install-dev lint type-check security-check test-ci ## Run full CI pipeline
	@echo "$(GREEN)CI pipeline completed successfully$(NC)"

ci-docker: docker-build ## Build Docker image for CI
	@echo "$(GREEN)Docker CI build completed$(NC)"

# =============================================================================
# Utility Commands
# =============================================================================

logs: ## Show application logs
	@echo "$(BLUE)Application logs:$(NC)"
	tail -f logs/registry.log

monitor: ## Monitor system resources
	@echo "$(BLUE)System monitoring:$(NC)"
	htop

redis-cli: ## Connect to Redis CLI
	@echo "$(BLUE)Connecting to Redis...$(NC)"
	redis-cli -h localhost -p 6379

psql: ## Connect to PostgreSQL
	@echo "$(BLUE)Connecting to PostgreSQL...$(NC)"
	psql $(DATABASE_URL)

env-check: ## Check environment variables
	@echo "$(BLUE)Environment variables:$(NC)"
	@echo "DATABASE_URL: $(DATABASE_URL)"
	@echo "Version: $(VERSION)"
	@env | grep -E "(ENVIRONMENT|DEBUG|LOG_LEVEL)" || true

# =============================================================================
# Development Shortcuts
# =============================================================================

quick-start: dev migrate ## Quick start for new developers
	@echo "$(GREEN)Development environment is ready!$(NC)"
	@echo "$(BLUE)Access the application at: http://localhost:8000$(NC)"
	@echo "$(BLUE)API documentation at: http://localhost:8000/api/docs$(NC)"
	@echo "$(BLUE)Grafana dashboard at: http://localhost:3000$(NC)"

full-test: clean install-dev lint type-check security-check test-cov ## Run complete test suite
	@echo "$(GREEN)All tests passed!$(NC)"

dev-reset: dev-clean dev migrate ## Reset development environment
	@echo "$(GREEN)Development environment reset complete!$(NC)"