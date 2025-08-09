# APG API Service Mesh - Production Dockerfile
#
# Multi-stage build for optimized production container with comprehensive
# security, performance, and monitoring capabilities.
#
# © 2025 Datacraft. All rights reserved.
# Author: Nyimbi Odero <nyimbi@gmail.com>

# =============================================================================
# Build Stage
# =============================================================================

FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=1.0.0

# Set build environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app user and directories
RUN groupadd -r appgroup && useradd -r -g appgroup appuser
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt requirements-prod.txt ./
RUN pip install --user --no-warn-script-location -r requirements-prod.txt

# =============================================================================
# Production Stage
# =============================================================================

FROM python:3.11-slim as production

# Set production labels
LABEL maintainer="Nyimbi Odero <nyimbi@gmail.com>" \
      org.label-schema.name="APG API Service Mesh" \
      org.label-schema.description="Intelligent API orchestration and service mesh networking" \
      org.label-schema.url="https://datacraft.co.ke" \
      org.label-schema.vcs-url="https://github.com/datacraft/apg-service-mesh" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.version=$VERSION \
      org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.schema-version="1.0"

# Set production environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH=/home/appuser/.local/bin:$PATH \
    ENVIRONMENT=production \
    LOG_LEVEL=INFO \
    WORKERS=4 \
    HOST=0.0.0.0 \
    PORT=8000

# Install production runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    dumb-init \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app user and directories
RUN groupadd -r appgroup && useradd -r -g appgroup appuser \
    && mkdir -p /app /app/logs /app/data \
    && chown -R appuser:appgroup /app

# Copy Python dependencies from builder
COPY --from=builder --chown=appuser:appgroup /root/.local /home/appuser/.local

# Switch to app user
USER appuser
WORKDIR /app

# Copy application code
COPY --chown=appuser:appgroup . .

# Copy entrypoint script
COPY --chown=appuser:appgroup docker/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

# Expose ports
EXPOSE 8000 9090

# Set entrypoint
ENTRYPOINT ["dumb-init", "--"]
CMD ["./entrypoint.sh"]