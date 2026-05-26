# APG Registry (regy) - Production Docker Image
# =============================================
# Multi-stage Docker build for the APG Registry capability
# Optimized for production with security and performance

# Build arguments
ARG PYTHON_VERSION=3.11
ARG APG_VERSION=1.0.0
ARG BUILD_DATE
ARG VCS_REF

# Base Python image
FROM python:${PYTHON_VERSION}-slim-bookworm as base

# Metadata
LABEL maintainer="APG Platform Team <team@datacraft.co.ke>"
LABEL description="APG Registry - Intelligent Service Discovery & Management"
LABEL version="${APG_VERSION}"
LABEL build-date="${BUILD_DATE}"
LABEL vcs-ref="${VCS_REF}"

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=random
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_DEFAULT_TIMEOUT=100

# Create non-root user
RUN groupadd --gid 1000 apg && \
    useradd --uid 1000 --gid apg --shell /bin/bash --create-home apg

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libpq-dev \
        libffi-dev \
        libssl-dev \
        pkg-config \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ============================================================================
# Development stage
# ============================================================================
FROM base as development

# Install development dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        postgresql-client \
        redis-tools \
        vim \
        htop \
        procps \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/
COPY requirements-dev.txt /app/

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install -r requirements-dev.txt

# Copy source code
COPY . /app/

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models /app/config && \
    chown -R apg:apg /app

# Switch to non-root user
USER apg

# Expose ports
EXPOSE 5000 5001 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/api/regy/v1/status || exit 1

# Default development command
CMD ["python", "-m", "apg.capabilities.common.regy.api"]

# ============================================================================
# Production builder stage
# ============================================================================
FROM base as builder

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt /app/

# Install build dependencies and Python packages
RUN pip install --upgrade pip setuptools wheel && \
    pip install --user --no-cache-dir -r requirements.txt

# ============================================================================
# Production stage
# ============================================================================
FROM base as production

# Install runtime dependencies only
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        postgresql-client \
        curl \
        ca-certificates \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/apg/.local
ENV PATH=/home/apg/.local/bin:$PATH

# Copy application source
COPY --chown=apg:apg . /app/

# Remove development files
RUN rm -rf /app/tests /app/docs /app/.git* /app/requirements-dev.txt

# Create necessary directories with proper permissions
RUN mkdir -p \
        /app/logs \
        /app/data \
        /app/models \
        /app/config \
        /app/tmp \
        /var/log/apg && \
    chown -R apg:apg /app /var/log/apg && \
    chmod -R 755 /app && \
    chmod -R 777 /app/logs /app/data /app/tmp /var/log/apg

# Copy production configuration
COPY --chown=apg:apg docker/config/production.yaml /app/config/config.yaml

# Install ML models (if available)
RUN if [ -d "/app/ml-models" ]; then \
        cp -r /app/ml-models/* /app/models/ && \
        chown -R apg:apg /app/models; \
    fi

# Switch to non-root user
USER apg

# Set environment variables for production
ENV FLASK_ENV=production
ENV APG_CONFIG_FILE=/app/config/config.yaml
ENV APG_LOG_LEVEL=INFO

# Expose ports
EXPOSE 5000 5001 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/api/regy/v1/status || exit 1

# Production startup script
COPY --chown=apg:apg docker/scripts/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Default production command
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["gunicorn", "--config", "/app/docker/gunicorn.conf.py", "apg.capabilities.common.regy.api:app"]

# ============================================================================
# Testing stage
# ============================================================================
FROM development as testing

# Copy test configuration
COPY --chown=apg:apg docker/config/testing.yaml /app/config/config.yaml

# Install additional test dependencies
RUN pip install pytest-html pytest-json-report

# Set test environment
ENV FLASK_ENV=testing
ENV APG_CONFIG_FILE=/app/config/config.yaml
ENV APG_LOG_LEVEL=DEBUG

# Run tests by default
CMD ["python", "-m", "pytest", "tests/", "-v", "--cov=apg.capabilities.common.regy", "--html=reports/pytest-report.html", "--json-report", "--json-report-file=reports/pytest-report.json"]

# ============================================================================
# CI/CD stage
# ============================================================================
FROM testing as ci

# Install additional CI/CD tools
RUN pip install \
        black \
        isort \
        flake8 \
        mypy \
        bandit \
        safety \
        pytest-xdist

# Copy CI configuration files
COPY --chown=apg:apg .github/workflows/quality-checks.yml /app/
COPY --chown=apg:apg pyproject.toml /app/
COPY --chown=apg:apg .flake8 /app/
COPY --chown=apg:apg mypy.ini /app/

# CI/CD entrypoint
COPY --chown=apg:apg docker/scripts/ci-entrypoint.sh /app/ci-entrypoint.sh
RUN chmod +x /app/ci-entrypoint.sh

ENTRYPOINT ["/app/ci-entrypoint.sh"]

# ============================================================================
# Multi-architecture support
# ============================================================================
# This Dockerfile supports multiple architectures:
# - linux/amd64 (Intel/AMD 64-bit)
# - linux/arm64 (ARM 64-bit, including Apple M1/M2)
# - linux/arm/v7 (ARM 32-bit)

# Build commands for different architectures:
# docker buildx build --platform linux/amd64,linux/arm64 -t datacraft/apg-registry:1.0.0 .

# ============================================================================
# Build targets
# ============================================================================
# Development: docker build --target development -t apg-registry:dev .
# Production:  docker build --target production -t apg-registry:prod .
# Testing:     docker build --target testing -t apg-registry:test .
# CI/CD:       docker build --target ci -t apg-registry:ci .

# ============================================================================
# Security considerations
# ============================================================================
# 1. Non-root user (apg:1000)
# 2. Minimal base image (Python slim)
# 3. No sensitive data in layers
# 4. Regular security updates
# 5. Health checks enabled
# 6. Proper file permissions
# 7. Secrets via environment variables or mounted volumes