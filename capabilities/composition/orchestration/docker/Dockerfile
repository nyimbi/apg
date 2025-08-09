# Multi-stage build for APG Workflow Orchestration
FROM python:3.11-slim as base

# Build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=1.0.0

# Labels
LABEL maintainer="Nyimbi Odero <nyimbi@gmail.com>" \
      org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.vcs-url="https://github.com/datacraft-ke/apg-workflow" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0" \
      org.label-schema.name="APG Workflow Orchestration" \
      org.label-schema.description="Advanced workflow orchestration and ML-driven automation system" \
      org.label-schema.vendor="Datacraft"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r apg && useradd -r -g apg -u 1000 apg

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt requirements-prod.txt ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements-prod.txt

# Development stage
FROM base as development

# Install development dependencies
COPY requirements-dev.txt ./
RUN pip install -r requirements-dev.txt

# Install additional development tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    procps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set development environment
ENV FLASK_ENV=development \
    DEBUG=true \
    LOG_LEVEL=DEBUG

# Copy source code
COPY . .

# Change ownership to apg user
RUN chown -R apg:apg /app

USER apg

# Expose ports
EXPOSE 8080 8081 9090

# Development command
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=8080", "--reload"]

# Production build stage
FROM base as builder

# Copy source code
COPY . .

# Build optimizations
RUN python -m compileall . && \
    find . -name "*.pyc" -delete && \
    find . -name "__pycache__" -type d -exec rm -rf {} + || true

# Install only production dependencies
RUN pip install --no-deps -r requirements-prod.txt

# Production stage
FROM python:3.11-slim as production

# Copy labels from base
LABEL maintainer="Nyimbi Odero <nyimbi@gmail.com>" \
      org.label-schema.name="APG Workflow Orchestration" \
      org.label-schema.description="Advanced workflow orchestration and ML-driven automation system"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLASK_ENV=production \
    DEBUG=false \
    LOG_LEVEL=INFO \
    WORKERS=4 \
    TIMEOUT=300 \
    KEEPALIVE=5 \
    MAX_REQUESTS=1000 \
    MAX_REQUESTS_JITTER=100

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libpq5 \
    libffi8 \
    libssl3 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r apg && useradd -r -g apg -u 1000 apg

# Create necessary directories
RUN mkdir -p /app /app/data /app/models /app/logs /app/temp && \
    chown -R apg:apg /app

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code from builder
COPY --from=builder --chown=apg:apg /app .

# Create startup script
RUN cat > /app/start.sh << 'EOF'
#!/bin/bash
set -e

# Wait for database
if [ -n "$DATABASE_HOST" ]; then
    echo "Waiting for database..."
    while ! curl -f -s "$DATABASE_HOST:${DATABASE_PORT:-5432}" > /dev/null 2>&1; do
        echo "Database not ready, waiting..."
        sleep 2
    done
    echo "Database is ready!"
fi

# Wait for Redis
if [ -n "$REDIS_HOST" ]; then
    echo "Waiting for Redis..."
    while ! curl -f -s "$REDIS_HOST:${REDIS_PORT:-6379}" > /dev/null 2>&1; do
        echo "Redis not ready, waiting..."
        sleep 2
    done
    echo "Redis is ready!"
fi

# Run database migrations
echo "Running database migrations..."
python -c "
try:
    from service import initialize_database
    initialize_database()
    print('Database initialized successfully')
except Exception as e:
    print(f'Database initialization failed: {e}')
    exit(1)
"

# Start the application
echo "Starting APG Workflow Orchestration..."
exec gunicorn \
    --bind 0.0.0.0:8080 \
    --workers $WORKERS \
    --timeout $TIMEOUT \
    --keep-alive $KEEPALIVE \
    --max-requests $MAX_REQUESTS \
    --max-requests-jitter $MAX_REQUESTS_JITTER \
    --worker-class gevent \
    --worker-connections 1000 \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    --capture-output \
    --enable-stdio-inheritance \
    --preload \
    "api:app"
EOF

RUN chmod +x /app/start.sh

# Health check script
RUN cat > /app/healthcheck.sh << 'EOF'
#!/bin/bash
set -e

# Check main application
curl -f http://localhost:8080/health || exit 1

# Check WebSocket endpoint
curl -f http://localhost:8081/ws/health || exit 1

# Check metrics endpoint
curl -f http://localhost:9090/metrics || exit 1

echo "All health checks passed"
EOF

RUN chmod +x /app/healthcheck.sh

# Switch to non-root user
USER apg

# Create volume mount points
VOLUME ["/app/data", "/app/models", "/app/logs"]

# Expose ports
EXPOSE 8080 8081 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /app/healthcheck.sh

# Default command
CMD ["/app/start.sh"]

# Testing stage
FROM production as testing

# Switch back to root for package installation
USER root

# Install testing dependencies
RUN apt-get update && apt-get install -y \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy test requirements
COPY requirements-test.txt ./
RUN pip install -r requirements-test.txt

# Copy test files
COPY tests/ ./tests/
COPY pytest.ini ./

# Change ownership back to apg user
RUN chown -R apg:apg /app

USER apg

# Override entrypoint for testing
ENTRYPOINT ["python", "-m", "pytest"]
CMD ["tests/", "-v", "--tb=short"]

# CI/CD stage
FROM testing as ci

# Install additional CI tools
USER root
RUN pip install \
    coverage \
    pytest-cov \
    pytest-html \
    pytest-xdist \
    bandit \
    safety \
    black \
    flake8 \
    mypy

USER apg

# CI command
CMD ["python", "-m", "pytest", "tests/", "-v", "--cov=.", "--cov-report=xml", "--cov-report=html", "--html=report.html"]

# Security scanning stage
FROM alpine:3.18 as security

# Install security scanning tools
RUN apk add --no-cache \
    python3 \
    py3-pip \
    && pip3 install \
    bandit \
    safety \
    semgrep

COPY . /app
WORKDIR /app

# Security scan command
CMD ["sh", "-c", "bandit -r . -f json -o bandit-report.json && safety check --json --output safety-report.json && semgrep --config=auto --json --output=semgrep-report.json ."]