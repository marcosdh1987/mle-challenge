# Roadmap for Productionizing the Garbage Classification Model

This document prioritizes and organizes the key steps to strengthen and bring the current model to production, focusing on modularity, deployment, and engineering best practices.

## 1. Modularization and Parameterization of the Training Pipeline
- **Description**: Refactor the training pipeline to remove hardcoded values and tightly coupled scripts. Use configuration files (YAML/JSON) and a single entrypoint to orchestrate data loading, model building, training, evaluation, and artifact management.
- **Benefit**: Improves reproducibility, enables experiment traceability, and simplifies hyperparameter tuning and automation.

## 2. Model Exposure via Real-Time Inference API
- **Description**: Create an API with FastAPI application (api/main.py) that provides:
  - `/predict` POST endpoint accepting JPEG/PNG file uploads only
  - Input validation of content type and preprocessing with Keras `preprocess_input`
  - JSON response with `predicted_class`, `confidence`, and `all_confidences`
  - `/health` GET endpoint reporting service status and model load state
  - Structured logging at INFO level for key events
  - Add support for image URLs or base64-encoded inputs
  - Enhance OpenAPI documentation with request/response examples and parameter details
  - Containerize the API (Docker).
- **Benefit**: Enables reliable real-time predictions, clear API docs, and easy deployment workflows.

## 3. Experiment Tracking and Visualization
- **Description**: Uses MLflow with a local SQLite backend (`mlruns.db`). Through provided `Makefile` targets (e.g., `make mlflow-ui`), you can:
  - Launch the MLflow UI locally to browse experiments, runs, metrics, and artifacts
  - Compare model versions and transition a model’s stage between Staging and Production
- **Benefit**: Simplifies local experiment auditing, model registry workflows, and rapid iteration without external services.

## 4. Unit and Integration Tests
- **Description**: Implement tests with pytest for preprocessing functions, model integrity, and (mocked) training loops. Validate expected outputs, shapes, and failure modes.
- **Benefit**: Ensures reliability, prevents regressions, and enables safe refactoring.

## 5. CI/CD
- **Description**: Create a pipeline (GitHub Actions or similar) to automate linting, tests, Docker builds, and deployment (to staging or model registry).
- **Benefit**: Accelerates delivery, enforces quality, and maintains consistency across environments.

## 6. Model Performance Monitoring and Alerting
- **Description**: Implement monitoring (e.g., Prometheus/Grafana) for inference latency, error rates, and accuracy on new data. Set up alerts for degradation.
- **Benefit**: Ensures SLA compliance and early detection of production issues.