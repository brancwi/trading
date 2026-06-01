# ============================================================
# Dockerfile — Trading Engine V4.1
# ============================================================

FROM python:3.11-slim

WORKDIR /app

# System deps for compilation (some Python packages need gcc)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure data directory exists (will be overridden by volume in compose)
RUN mkdir -p /app/data

# Install the package in editable mode
RUN pip install -e .

# Runtime deps that may have been missed in requirements
RUN pip install --no-cache-dir pandas xgboost

# Expose FastAPI port
EXPOSE 8000

# Default: run API (override in compose for other services)
CMD ["python", "scripts/run_api.py"]
