# APG Payroll Management - Production Docker Configuration
# © 2025 Datacraft. All rights reserved.
# Author: Nyimbi Odero | APG Platform Architect

FROM python:3.12-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN useradd --create-home --shell /bin/bash apg

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy application code
COPY . .

# Change ownership to apg user
RUN chown -R apg:apg /app

# Switch to apg user
USER apg

# Development stage
FROM base as development
ENV FLASK_ENV=development
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt
EXPOSE 5000
CMD ["python", "run.py"]

# Production stage
FROM base as production

# Install production dependencies only
ENV FLASK_ENV=production \
    PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run production server
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "gevent", "--worker-connections", "1000", "--timeout", "120", "--keepalive", "2", "--max-requests", "1000", "--max-requests-jitter", "100", "--access-logfile", "-", "--error-logfile", "-", "run:app"]