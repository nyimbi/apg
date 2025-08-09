# APG API Service Mesh - Makefile
#
# Comprehensive build, test, deployment, and maintenance automation
# for the API Service Mesh with support for multiple environments.
#
# © 2025 Datacraft. All rights reserved.
# Author: Nyimbi Odero <nyimbi@gmail.com>

.PHONY: help install install-dev test test-unit test-integration test-performance \
        lint format type-check security-check clean build docker-build docker-run \
        docker-stop deploy deploy-dev deploy-staging deploy-prod rollback status \
        logs backup docs serve-docs validate migrate seed monitor

# =============================================================================
# Configuration
# =============================================================================

# Default environment
ENV ?= development

# Python and package management
PYTHON ?= python3
PIP ?= pip
PYTEST ?= pytest
DOCKER ?= docker
DOCKER_COMPOSE ?= docker-compose

# Project information
PROJECT_NAME := apg-service-mesh
VERSION := $(shell cat VERSION 2>/dev/null || echo "1.0.0")
IMAGE_NAME := datacraft/$(PROJECT_NAME)
IMAGE_TAG ?= $(VERSION)

# Directories
SRC_DIR := .
TEST_DIR := tests
DOCS_DIR := docs
BUILD_DIR := build
DIST_DIR := dist

# Docker Compose files
COMPOSE_FILE := docker-compose.yml
COMPOSE_PROD_FILE := docker-compose.prod.yml

# Colors for output
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
RESET := \033[0m

# =============================================================================
# Help and Information
# =============================================================================

help: ## Show this help message
	@echo "$(BLUE)APG API Service Mesh - Build and Deployment Automation$(RESET)"
	@echo ""
	@echo "$(YELLOW)Available commands:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(YELLOW)Environment variables:$(RESET)"
	@echo "  ENV          Environment (development|staging|production) [default: development]"
	@echo "  IMAGE_TAG    Docker image tag [default: $(VERSION)]"
	@echo "  PYTHON       Python executable [default: python3]"
	@echo ""
	@echo "$(YELLOW)Examples:$(RESET)"
	@echo "  make install                    # Install dependencies"
	@echo "  make test                       # Run all tests"
	@echo "  make ENV=production deploy      # Deploy to production"
	@echo "  make docker-build IMAGE_TAG=v1.2.0  # Build specific version"

version: ## Show version information
	@echo "$(BLUE)APG API Service Mesh$(RESET)"
	@echo "Version: $(VERSION)"
	@echo "Environment: $(ENV)"
	@echo "Image: $(IMAGE_NAME):$(IMAGE_TAG)"

# =============================================================================
# Development Environment Setup
# =============================================================================

install: ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(RESET)"
	$(PIP) install -r requirements.txt

install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(RESET)"
	$(PIP) install -r requirements-dev.txt
	pre-commit install

install-prod: ## Install production dependencies (optimized)
	@echo "$(BLUE)Installing production dependencies (optimized)...$(RESET)"
	$(PIP) install --no-deps -r requirements-prod.txt

setup-env: ## Setup development environment
	@echo "$(BLUE)Setting up development environment...$(RESET)"
	cp .env.example .env
	@echo "$(YELLOW)Please edit .env file with your configuration$(RESET)"

# =============================================================================
# Code Quality and Testing
# =============================================================================

lint: ## Run code linting
	@echo "$(BLUE)Running code linting...$(RESET)"
	flake8 $(SRC_DIR) $(TEST_DIR)
	pylint $(SRC_DIR)
	bandit -r $(SRC_DIR) -f json -o build/bandit-report.json || true

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(RESET)"
	black $(SRC_DIR) $(TEST_DIR)
	isort $(SRC_DIR) $(TEST_DIR)

format-check: ## Check code formatting
	@echo "$(BLUE)Checking code formatting...$(RESET)"
	black --check $(SRC_DIR) $(TEST_DIR)
	isort --check-only $(SRC_DIR) $(TEST_DIR)

type-check: ## Run type checking with mypy
	@echo "$(BLUE)Running type checking...$(RESET)"
	mypy $(SRC_DIR)

security-check: ## Run security checks
	@echo "$(BLUE)Running security checks...$(RESET)"
	safety check
	bandit -r $(SRC_DIR)

test: test-unit test-integration ## Run all tests

test-unit: ## Run unit tests
	@echo "$(BLUE)Running unit tests...$(RESET)"
	$(PYTEST) -v -m "unit" --cov=$(SRC_DIR) --cov-report=html --cov-report=xml

test-integration: ## Run integration tests
	@echo "$(BLUE)Running integration tests...$(RESET)"
	$(PYTEST) -v -m "integration" --cov=$(SRC_DIR) --cov-append

test-performance: ## Run performance tests
	@echo "$(BLUE)Running performance tests...$(RESET)"
	$(PYTEST) -v -m "performance" --benchmark-only

test-api: ## Run API tests
	@echo "$(BLUE)Running API tests...$(RESET)"
	$(PYTEST) -v -m "api"

test-websocket: ## Run WebSocket tests
	@echo "$(BLUE)Running WebSocket tests...$(RESET)"
	$(PYTEST) -v -m "websocket"

test-coverage: ## Generate test coverage report
	@echo "$(BLUE)Generating test coverage report...$(RESET)"
	$(PYTEST) --cov=$(SRC_DIR) --cov-report=html --cov-report=xml --cov-report=term

test-watch: ## Run tests in watch mode
	@echo "$(BLUE)Running tests in watch mode...$(RESET)"
	$(PYTEST) -f

# =============================================================================
# Build and Package
# =============================================================================

clean: ## Clean build artifacts
	@echo "$(BLUE)Cleaning build artifacts...$(RESET)"
	rm -rf $(BUILD_DIR) $(DIST_DIR)
	rm -rf *.egg-info
	rm -rf .pytest_cache .coverage htmlcov
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete

build: clean ## Build the package
	@echo "$(BLUE)Building package...$(RESET)"
	mkdir -p $(BUILD_DIR)
	$(PYTHON) -m build

# =============================================================================
# Docker Operations
# =============================================================================

docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image $(IMAGE_NAME):$(IMAGE_TAG)...$(RESET)"
	$(DOCKER) build -t $(IMAGE_NAME):$(IMAGE_TAG) .
	$(DOCKER) tag $(IMAGE_NAME):$(IMAGE_TAG) $(IMAGE_NAME):latest

docker-build-dev: ## Build development Docker image
	@echo "$(BLUE)Building development Docker image...$(RESET)"
	$(DOCKER) build -f Dockerfile.dev -t $(IMAGE_NAME):dev .

docker-run: ## Run Docker container
	@echo "$(BLUE)Running Docker container...$(RESET)"
	$(DOCKER) run -d \
		--name $(PROJECT_NAME) \
		-p 8000:8000 \
		-e ENVIRONMENT=$(ENV) \
		$(IMAGE_NAME):$(IMAGE_TAG)

docker-run-dev: ## Run development Docker container
	@echo "$(BLUE)Running development Docker container...$(RESET)"
	$(DOCKER_COMPOSE) up -d

docker-stop: ## Stop Docker containers
	@echo "$(BLUE)Stopping Docker containers...$(RESET)"
	$(DOCKER_COMPOSE) down

docker-logs: ## View Docker container logs
	@echo "$(BLUE)Viewing Docker container logs...$(RESET)"
	$(DOCKER_COMPOSE) logs -f

docker-shell: ## Open shell in Docker container
	@echo "$(BLUE)Opening shell in Docker container...$(RESET)"
	$(DOCKER) exec -it $(PROJECT_NAME) /bin/bash

docker-clean: ## Clean Docker images and containers
	@echo "$(BLUE)Cleaning Docker images and containers...$(RESET)"
	$(DOCKER) system prune -f
	$(DOCKER) image prune -f

# =============================================================================
# Database Operations
# =============================================================================

migrate: ## Run database migrations
	@echo "$(BLUE)Running database migrations...$(RESET)"
	alembic upgrade head

migrate-create: ## Create new migration
	@echo "$(BLUE)Creating new migration...$(RESET)"
	@read -p "Migration message: " message; \
	alembic revision --autogenerate -m "$$message"

migrate-rollback: ## Rollback last migration
	@echo "$(BLUE)Rolling back last migration...$(RESET)"
	alembic downgrade -1

seed: ## Seed database with sample data
	@echo "$(BLUE)Seeding database with sample data...$(RESET)"
	$(PYTHON) scripts/seed_data.py

# =============================================================================
# Deployment Operations
# =============================================================================

deploy: ## Deploy to specified environment
	@echo "$(BLUE)Deploying to $(ENV) environment...$(RESET)"
	./scripts/deploy.sh -e $(ENV) deploy

deploy-dev: ## Deploy to development environment
	@echo "$(BLUE)Deploying to development environment...$(RESET)"
	ENV=development $(MAKE) deploy

deploy-staging: ## Deploy to staging environment
	@echo "$(BLUE)Deploying to staging environment...$(RESET)"
	ENV=staging $(MAKE) deploy

deploy-prod: ## Deploy to production environment
	@echo "$(BLUE)Deploying to production environment...$(RESET)"
	ENV=production $(MAKE) deploy

deploy-k8s: ## Deploy to Kubernetes
	@echo "$(BLUE)Deploying to Kubernetes...$(RESET)"
	./scripts/deploy.sh -t kubernetes deploy

rollback: ## Rollback deployment
	@echo "$(BLUE)Rolling back deployment...$(RESET)"
	./scripts/deploy.sh rollback

status: ## Check deployment status
	@echo "$(BLUE)Checking deployment status...$(RESET)"
	./scripts/deploy.sh status

logs: ## View deployment logs
	@echo "$(BLUE)Viewing deployment logs...$(RESET)"
	./scripts/deploy.sh logs

# =============================================================================
# Monitoring and Maintenance
# =============================================================================

monitor: ## Start monitoring stack
	@echo "$(BLUE)Starting monitoring stack...$(RESET)"
	$(DOCKER_COMPOSE) --profile monitoring up -d

health-check: ## Perform health check
	@echo "$(BLUE)Performing health check...$(RESET)"
	curl -f http://localhost:8000/api/health || exit 1

load-test: ## Run load tests
	@echo "$(BLUE)Running load tests...$(RESET)"
	$(DOCKER_COMPOSE) --profile testing up locust

backup: ## Backup database
	@echo "$(BLUE)Creating database backup...$(RESET)"
	./scripts/backup.sh

restore: ## Restore database from backup
	@echo "$(BLUE)Restoring database from backup...$(RESET)"
	./scripts/restore.sh

# =============================================================================
# Documentation
# =============================================================================

docs: ## Build documentation
	@echo "$(BLUE)Building documentation...$(RESET)"
	mkdocs build

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8001...$(RESET)"
	mkdocs serve -a localhost:8001

docs-deploy: ## Deploy documentation
	@echo "$(BLUE)Deploying documentation...$(RESET)"
	mkdocs gh-deploy

# =============================================================================
# Validation and Verification
# =============================================================================

validate: ## Validate deployment configuration
	@echo "$(BLUE)Validating deployment configuration...$(RESET)"
	./scripts/deploy.sh validate

verify: ## Verify installation and setup
	@echo "$(BLUE)Verifying installation and setup...$(RESET)"
	$(PYTHON) --version
	$(PIP) --version
	$(DOCKER) --version
	$(DOCKER_COMPOSE) --version

pre-commit: ## Run pre-commit checks
	@echo "$(BLUE)Running pre-commit checks...$(RESET)"
	pre-commit run --all-files

ci: lint format-check type-check security-check test ## Run CI pipeline

# =============================================================================
# Convenience Targets
# =============================================================================

dev: install-dev setup-env ## Setup complete development environment
	@echo "$(GREEN)Development environment ready!$(RESET)"
	@echo "$(YELLOW)Next steps:$(RESET)"
	@echo "  1. Edit .env file with your configuration"
	@echo "  2. Run 'make docker-run-dev' to start services"
	@echo "  3. Run 'make test' to verify everything works"

quick-start: dev docker-run-dev migrate seed ## Quick start for development
	@echo "$(GREEN)Quick start completed!$(RESET)"
	@echo "$(YELLOW)Services are running at:$(RESET)"
	@echo "  API: http://localhost:8000"
	@echo "  Docs: http://localhost:8000/api/docs"
	@echo "  Grafana: http://localhost:3000"

prod-ready: lint type-check security-check test build ## Prepare for production
	@echo "$(GREEN)Production readiness checks completed!$(RESET)"

all: clean install-dev lint format type-check security-check test build docs ## Run all tasks

# Default target
.DEFAULT_GOAL := help