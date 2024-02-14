# Variables
PROJECT = aintelope
TESTS = tests
SITE = site
CODEBASE = ${PROJECT} ${TESTS} ${SITE}
VENV = venv_$(PROJECT)

run-training-baseline: ## run baseline experiment
	python -m ${PROJECT} hparams.agent_id=q_agent hparams.agent_params.target_instincts=[]

run-training-instinct: ## run instinct agent experiment
	python -m ${PROJECT} hparams.agent_id=instinct_agent hparams.agent_params.target_instincts=[smell]

run-pipeline:
	python aintelope/pipeline.py

# ---------- installation and environment ----------
.PHONY: venv clean-venv install install-dev install-all build-local

venv: ## create virtual environment
	@if [ ! -f "$(VENV)/bin/activate" ]; then python3 -m venv $(VENV) ; fi;

clean-venv: ## remove virtual environment
	if [ -d $(VENV) ]; then rm -r $(VENV) ; fi;

install: ## Install packages
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
.PHONY: format format-check isort isort-check
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
