# APG Connection Management Capability - Production Docker Image
# Multi-stage build for optimized production deployment
#
# Author: Nyimbi Odero
# Company: Datacraft
# Copyright: © 2025

# Build stage
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry for dependency management
RUN pip install poetry

# Set work directory
WORKDIR /build

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Configure Poetry
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --only=main --no-dev

# Production stage
FROM python:3.11-slim

# Create non-root user for security
RUN groupadd -r apg && useradd -r -g apg -s /bin/false apg

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set work directory
WORKDIR /app

# Copy application code
COPY capabilities/common/conn/ ./

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/tmp && \
    chown -R apg:apg /app

# Copy startup scripts
COPY deployment/docker/entrypoint.sh /entrypoint.sh
COPY deployment/docker/healthcheck.sh /healthcheck.sh
RUN chmod +x /entrypoint.sh /healthcheck.sh

# Environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    APG_ENV=production \
    APG_LOG_LEVEL=INFO \
    APG_WORKERS=4 \
    APG_PORT=8000 \
    APG_HOST=0.0.0.0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD /healthcheck.sh

# Switch to non-root user
USER apg

# Expose port
EXPOSE 8000

# Use tini for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

# Start application
CMD ["/entrypoint.sh"]

# Labels for metadata
LABEL maintainer="Nyimbi Odero <nyimbi@gmail.com>" \
      version="1.0.0" \
      description="APG Connection Management Capability" \
      org.opencontainers.image.source="https://github.com/datacraft/apg" \
      org.opencontainers.image.documentation="https://docs.datacraft.co.ke/apg/capabilities/conn" \
      org.opencontainers.image.vendor="Datacraft" \
      org.opencontainers.image.licenses="Proprietary"