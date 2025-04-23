# Makefile for setting up a Python virtual environment with uv and registering a Jupyter kernel.

PYTHON_VERSION ?= 3.10.0
VENV_DIR       ?= .venv
KERNEL_NAME    ?= trash-classifier

uv-venv:
	@echo "🚀 Creating virtual environment with uv..."
	@if ! command -v uv &> /dev/null; then \
        echo "❌ uv is not installed. Please install it with: pip install uv"; \
        exit 1; \
    fi
	@if [ ! -d "$(VENV_DIR)" ]; then \
        uv venv $(VENV_DIR) --python=$(PYTHON_VERSION); \
    else \
        echo "✅ Virtual environment already exists."; \
    fi
	@echo "📦 Installing dependencies with uv pip..."
	@. $(VENV_DIR)/bin/activate && uv pip install -r requirements.txt && uv pip install ipykernel
	@. $(VENV_DIR)/bin/activate && uv pip install tensorflow uvicorn fastapi python-multipart mlflow
	@echo "🔌 Registering Jupyter kernel..."
	@$(VENV_DIR)/bin/python -m ipykernel install --user --name=$(KERNEL_NAME) --display-name="Python (uv)"
	@echo "✅ uv virtual environment ready for Jupyter Notebook."

train:
	@echo "🚀 Training the model..."
	@. $(VENV_DIR)/bin/activate && $(VENV_DIR)/bin/python scripts/train.py --config config/config.yaml
	@echo "✅ Model training complete."

# Run the FastAPI API using the virtual environment
run-api:
	@echo "🚀 Starting the FastAPI API with uvicorn..."
	@. $(VENV_DIR)/bin/activate && $(VENV_DIR)/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Run MLflow UI
mlflow-ui:
	@echo "🚀 Starting the MLflow UI with SQLite backend..."
	@. $(VENV_DIR)/bin/activate && $(VENV_DIR)/bin/mlflow ui --backend-store-uri sqlite:///mlruns.db
	@echo "✅ MLflow UI is running at http://127.0.0.1:5000"

###############################################################################
# CLEANUP
###############################################################################

# Code Quality
autoflake:
	@autoflake . --check --recursive --remove-all-unused-imports --remove-unused-variables --exclude .venv;

black:
	@black . --check --exclude '.venv|build|target|dist|.cache|node_modules';

isort:
	@isort . --check-only;

lint: black isort autoflake

lint-fix:
	@black . --exclude '.venv|build|target|dist';
	@isort .;
	@autoflake . --in-place --recursive --exclude .venv --remove-all-unused-imports --remove-unused-variables;

# Docker targets for API service
.PHONY: docker-build-api docker-run-api

# Build Docker image for FastAPI API
docker-build-api:
	@echo "🔨 Building Docker image for FastAPI service..."
	@docker build -t garbage-api:latest -f api/Dockerfile .

# Run Docker container for FastAPI API
docker-run-api:
	@echo "▶️ Running Docker container for FastAPI service..."
	@docker run --rm -p 8000:8000 garbage-api:latest