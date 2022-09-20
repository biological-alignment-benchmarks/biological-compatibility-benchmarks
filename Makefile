# Variables
PROJECT = aintelope
TESTS = tests

run-training-short:

run-training-long:

tests-local: $(PROJECT) $(TESTS) ## run tests locally with active python environment
	pytest --cov=$(PROJECT) $(TESTS)

tests-local-p: ## run tests locally without active python environment
	poetry run pytest --cov=$(PROJECT) $(TESTS)

typecheck-local: $(PROJECT) ## local typechecking with active python environment
	mypy $(PROJECT)

typecheck-local-p: $(PROJECT) ## local typechecking without active python environment
	poetry run mypy $(PROJECT)

isort: ## sort python imports with active python environment
	isort .

isort-p: ## sort python imports without active python environment
	poetry run isort .
