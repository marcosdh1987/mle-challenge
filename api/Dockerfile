# Dockerfile for Garbage Classification FastAPI service
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY api/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV ENV=prod

# Expose port for the API
EXPOSE 8000

# Start the FastAPI service; use module path including 'api'
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
