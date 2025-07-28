# APG Time & Attendance Capability Docker Image
# Multi-stage build for production-ready deployment
# Copyright © 2025 Datacraft

FROM python:3.12-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    postgresql-client \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN useradd --create-home --shell /bin/bash ta_user

# Set work directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt requirements-prod.txt ./

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy application code
COPY --chown=ta_user:ta_user . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data && \
    chown -R ta_user:ta_user /app

USER ta_user

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/human_capital_management/time_attendance/health || exit 1

# Default command for development
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base as production

# Install only production dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements-prod.txt && \
    pip install gunicorn

# Copy application code
COPY --chown=ta_user:ta_user . .

# Create necessary directories and set permissions
RUN mkdir -p /app/logs /app/data && \
    chown -R ta_user:ta_user /app && \
    chmod +x /app/scripts/*.sh

USER ta_user

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/human_capital_management/time_attendance/health || exit 1

# Production command with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--timeout", "120", "--keep-alive", "5", "--max-requests", "1000", "--max-requests-jitter", "100", "api:app"]