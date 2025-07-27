# APG Multi-language Localization - Production Dockerfile
# 
# Multi-stage build for optimized production deployment
# Supports async Python, PostgreSQL, Redis, and enterprise features
#
# Author: Nyimbi Odero
# Company: Datacraft
# Copyright: © 2025 Datacraft. All rights reserved.

# Build stage
FROM python:3.11-slim as builder

# Set build environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
COPY requirements-prod.txt /tmp/requirements-prod.txt

RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /tmp/requirements.txt && \
    pip install -r /tmp/requirements-prod.txt

# Production stage
FROM python:3.11-slim as production

# Set production environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libffi8 \
    libssl3 \
    curl \
    dumb-init \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create application user and directories
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /bin/bash appuser
RUN mkdir -p /app/logs /app/data /app/uploads && \
    chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . .

# Copy production configuration
COPY --chown=appuser:appuser docker/entrypoint.sh /entrypoint.sh
COPY --chown=appuser:appuser docker/healthcheck.py /healthcheck.py
COPY --chown=appuser:appuser docker/wait-for-postgres.py /wait-for-postgres.py

# Make scripts executable
RUN chmod +x /entrypoint.sh /healthcheck.py /wait-for-postgres.py

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python /healthcheck.py

# Expose application port
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["dumb-init", "--"]
CMD ["/entrypoint.sh"]

# Labels for metadata
LABEL \
    maintainer="Nyimbi Odero <nyimbi@gmail.com>" \
    org.opencontainers.image.title="APG Multi-language Localization" \
    org.opencontainers.image.description="Enterprise i18n/l10n service for APG platform" \
    org.opencontainers.image.vendor="Datacraft" \
    org.opencontainers.image.version="1.0.0" \
    org.opencontainers.image.url="https://www.datacraft.co.ke" \
    org.opencontainers.image.source="https://github.com/datacraft/apg" \
    org.opencontainers.image.licenses="Proprietary"