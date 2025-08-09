# APG Capability Registry - Production Dockerfile
# Multi-stage build for optimized production image

# Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION
ARG VCS_REF

# Set metadata
LABEL maintainer="Nyimbi Odero <nyimbi@gmail.com>" \
      org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="APG Capability Registry" \
      org.label-schema.description="Foundation infrastructure for APG capability discovery and orchestration" \
      org.label-schema.url="https://www.datacraft.co.ke" \
      org.label-schema.vcs-url="https://github.com/datacraft/apg-capability-registry" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vendor="Datacraft" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"

# Set working directory
WORKDIR /build

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt .
COPY requirements-prod.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Set build arguments for production stage
ARG BUILD_DATE
ARG VERSION
ARG VCS_REF

# Copy metadata labels
LABEL maintainer="Nyimbi Odero <nyimbi@gmail.com>" \
      org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="APG Capability Registry" \
      org.label-schema.description="Foundation infrastructure for APG capability discovery and orchestration" \
      org.label-schema.url="https://www.datacraft.co.ke" \
      org.label-schema.vcs-url="https://github.com/datacraft/apg-capability-registry" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vendor="Datacraft" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    dumb-init \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -u 1000 appuser

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/tmp \
    && chown -R appuser:appuser /app \
    && chmod -R 755 /app

# Copy entrypoint script
COPY scripts/docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Switch to non-root user
USER appuser

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Use dumb-init as entrypoint for proper signal handling
ENTRYPOINT ["dumb-init", "--", "/usr/local/bin/docker-entrypoint.sh"]

# Default command
CMD ["uvicorn", "capability_registry.api:api_app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]