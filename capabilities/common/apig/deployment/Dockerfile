# APG Intelligent Gateway - Production Dockerfile
# Version: 1.0.0  
# Date: August 11, 2025
# Multi-stage build for optimized production image

# Build Stage
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Labels for metadata
LABEL maintainer="APG Platform Team <support@datacraft.co.ke>" \
      org.label-schema.name="APG Intelligent Gateway" \
      org.label-schema.description="Revolutionary API Gateway with AI intelligence and edge computing" \
      org.label-schema.url="https://apg.datacraft.co.ke" \
      org.label-schema.vcs-url="https://github.com/datacraft-co-ke/apg" \
      org.label-schema.version=${VERSION} \
      org.label-schema.build-date=${BUILD_DATE} \
      org.label-schema.vcs-ref=${VCS_REF} \
      org.label-schema.schema-version="1.0"

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gcc \
    git \
    pkg-config \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /build

# Copy requirements first for better caching
COPY deployment/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip list --format=freeze > installed_packages.txt

# Production Stage
FROM python:3.11-slim as production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    dumb-init \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r apig && useradd -r -g apig -d /app -s /sbin/nologin apig

# Set work directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /build/installed_packages.txt /app/

# Copy application code
COPY apg_clients.py .
COPY ollama_client.py .
COPY wasm_runtime.py .
COPY edge_engine_production.py .
COPY service_production.py .
COPY control_plane.py .
COPY models.py .
COPY __init__.py .

# Copy configuration templates
COPY deployment/config /app/config
COPY deployment/scripts /app/scripts

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/wasm_modules /app/ssl /app/temp && \
    chown -R apig:apig /app

# Set environment variables
ENV PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ENVIRONMENT="production" \
    LOG_LEVEL="INFO"

# Health check
COPY deployment/scripts/healthcheck.sh /usr/local/bin/healthcheck
RUN chmod +x /usr/local/bin/healthcheck

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /usr/local/bin/healthcheck

# Expose ports
EXPOSE 8080 9090

# Switch to non-root user
USER apig

# Use dumb-init for proper signal handling
ENTRYPOINT ["dumb-init", "--"]

# Start the application
CMD ["python", "-m", "service_production"]