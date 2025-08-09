# APG Event Streaming Bus - Production Docker Image
# Multi-stage build for optimized production deployment
# © 2025 Datacraft. All rights reserved.

# Stage 1: Base Python Environment
FROM python:3.11-slim as base

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Dependencies
FROM base as dependencies

# Create application directory
WORKDIR /app

# Copy dependency files
COPY requirements.txt requirements-prod.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements-prod.txt

# Stage 3: Application
FROM dependencies as application

# Create non-root user
RUN groupadd -r apg && useradd -r -g apg apg

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=apg:apg . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/tmp && \
    chown -R apg:apg /app

# Install application in development mode
RUN pip install -e .

# Switch to non-root user
USER apg

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${API_PORT:-8080}/health || exit 1

# Expose ports
EXPOSE 8080 9090

# Default command
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]

# Stage 4: Production (final stage)
FROM application as production

# Production-specific configurations
ENV ENV=production \
    LOG_LEVEL=INFO \
    API_WORKERS=4 \
    METRICS_ENABLED=true

# Copy production configuration
COPY docker/prod/config/ /app/config/

# Final security hardening
USER apg

# Production entrypoint
COPY docker/entrypoint.sh /app/entrypoint.sh
USER root
RUN chmod +x /app/entrypoint.sh
USER apg

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["api"]