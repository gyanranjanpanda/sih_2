# Developer entry points. `make verify` is the gate before every commit.
PY       := $(CURDIR)/.venv/bin/python
PIP      := $(PY) -m pip
BACKEND  := $(CURDIR)/backend
FRONTEND := $(CURDIR)/frontend

.PHONY: help venv install lint format typecheck test test-fast coverage scenario \
        smoke verify data train frontend-install frontend-build frontend-dev \
        api clean docker-up docker-down

help:
	@echo "make install        create the virtualenv and install pinned dependencies"
	@echo "make lint           run ruff"
	@echo "make format         run ruff with --fix"
	@echo "make typecheck      run mypy"
	@echo "make test           run the full test suite"
	@echo "make test-fast      run the test suite without the slow tests"
	@echo "make coverage       run tests with coverage on twin/ and optimize/"
	@echo "make data           generate the synthetic dataset"
	@echo "make train          train the machine learning models"
	@echo "make scenario       run the baseline versus optimized scenario"
	@echo "make smoke          start the API and check /health"
	@echo "make verify         lint, typecheck, test, mini scenario and API smoke test"
	@echo "make docker-up      build and start the full stack with Docker Compose"

venv:
	python3.11 -m venv .venv
	$(PIP) install --upgrade pip setuptools wheel

install: venv
	$(PIP) install -r $(BACKEND)/requirements-dev.txt

lint:
	cd $(BACKEND) && $(PY) -m ruff check .
	cd $(BACKEND) && $(PY) -m ruff format --check .

format:
	cd $(BACKEND) && $(PY) -m ruff check --fix .
	cd $(BACKEND) && $(PY) -m ruff format .

typecheck:
	cd $(BACKEND) && $(PY) -m mypy app

test:
	cd $(BACKEND) && $(PY) -m pytest

test-fast:
	cd $(BACKEND) && $(PY) -m pytest -m "not slow"

coverage:
	cd $(BACKEND) && $(PY) -m pytest --cov=app/twin --cov=app/optimize \
		--cov-report=term-missing --cov-fail-under=80

data:
	$(PY) scripts/generate_data.py

train:
	$(PY) scripts/train_models.py

scenario:
	$(PY) scripts/run_scenario.py

smoke:
	$(PY) scripts/smoke_api.py

verify: lint typecheck test scenario smoke
	@echo "verify: all checks passed"

frontend-install:
	cd $(FRONTEND) && npm install

frontend-build: frontend-install
	cd $(FRONTEND) && npm run build

frontend-dev:
	cd $(FRONTEND) && npm run dev

api:
	cd $(BACKEND) && $(PY) -m uvicorn app.api.main:app --reload --port 8000

docker-up:
	docker compose up --build

docker-down:
	docker compose down -v

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache backend/.pytest_cache backend/.mypy_cache
