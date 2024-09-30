# Variables
PROJECT = aintelope
TESTS = tests
CODEBASE = ${PROJECT} ${TESTS}
VENV = venv_$(PROJECT)

run-training-baseline: ## run baseline experiment
	python -m ${PROJECT} hparams.agent_id=example_agent

run-pipeline: ## run pipeline
	python -m ${PROJECT}

# ---------- installation and environment ----------
.PHONY: venv clean-venv install install-dev install-all build-local

venv: ## create virtual environment
	@if [ ! -f "$(VENV)/bin/activate" ]; then python3 -m venv $(VENV) ; fi;

clean-venv: ## remove virtual environment
	if [ -d $(VENV) ]; then rm -r $(VENV) ; fi;

install: ## Install packages
	pip uninstall ai_safety_gridworlds
	pip install -r requirements/api.txt

install-dev: ## Install development packages
	pip install -r requirements/dev.txt

install-all: install install-dev ## install all packages

build-local: ## install the project locally
	pip install -e .

# ---------- testing ----------
.PHONY: tests-local
tests-local: ## Run tests locally
	python -m pytest --tb=native --cov=$(CODEBASE)

# ---------- type checking ----------
.PHONY: typecheck-local
typecheck-local: ## Local typechecking
	mypy $(CODEBASE)

# ---------- formatting ----------
.PHONY: isort isort-check format format-check
format: ## apply automatic code formatter to repository
	black $(CODEBASE)

format-check: ## check formatting
	black --check $(CODEBASE)

isort: ## Sort python imports
	isort $(CODEBASE)

isort-check: ## check import order
	isort --check $(CODEBASE)

# ---------- linting ----------
.PHONY: flake8
flake8: ## check code style
	flake8 $(CODEBASE)

# ---------- cleaning ----------
.PHONY: clean
clean:
	rm -rf *.egg-info
	rm -rf .mypy_cache
	rm -rf .pytest_cache

# ---------- help ----------
.PHONY: help
help: ## Show this help
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)
