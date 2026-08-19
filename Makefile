TESTS_DIR := tests

help:  ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z0-9_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
.PHONY: help

##@ Development

test: ## Run tests
	uv run pytest --cov=quaternion_neural_networks $(TESTS_DIR)/
.PHONY: test

lint: ## Lint the code
	uv run ruff check ./quaternion_neural_networks ./tests
.PHONY: lint

typecheck: ## Run type checking
	uv run mypy ./quaternion_neural_networks ./tests
.PHONY: typecheck

test-all: test lint typecheck ## Run all tests, linting, and type checking
.PHONY: test-all

format: ## Format the code
	uv run ruff format .
.PHONY: format

##@ Installation

edit-install: ## Install the package in editable mode
	uv sync --all-extras
.PHONY: edit-install

install: ## Install the package
	uv sync
.PHONY: install

##@ Release

bump-major: ## Bump the major version
	uv run bump-my-version bump major
.PHONY: bump-major

bump-minor: ## Bump the minor version
	uv run bump-my-version bump minor
.PHONY: bump-minor

bump-patch: ## Bump the patch version
	uv run bump-my-version bump patch
.PHONY: bump-patch

##@ ML

train: ## Run training
	uv run python -m quaternion_neural_networks.train

experiment: ## Run specific experiment
	uv run python -m quaternion_neural_networks.train experiment=$(EXP)
.PHONY: train experiment