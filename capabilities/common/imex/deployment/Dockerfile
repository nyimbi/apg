# APG Import/Export (IMEX) Capability Production Dockerfile
#
# Purpose: Production-ready Docker container for APG IMEX capability
# Base: Python 3.11 with production optimizations
# Security: Non-root user, minimal attack surface
# Performance: Multi-stage build, optimized layers

# Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=1.0.0

# Labels for metadata
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="APG IMEX Capability" \
      org.label-schema.description="Production-grade Import/Export capability for APG platform" \
      org.label-schema.url="https://www.datacraft.co.ke" \
      org.label-schema.vcs-url="https://github.com/datacraft/apg" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vendor="Datacraft" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0" \
      maintainer="nyimbi@gmail.com"

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better layer caching
COPY requirements.txt /tmp/
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /tmp/requirements.txt

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN groupadd --gid 1000 apg && \
    useradd --uid 1000 --gid apg --shell /bin/bash --create-home apg

# Set up application directories
RUN mkdir -p /opt/apg/imex /opt/apg/uploads /opt/apg/logs /tmp/apg && \
    chown -R apg:apg /opt/apg /tmp/apg

# Set working directory
WORKDIR /opt/apg/imex

# Copy application code
COPY --chown=apg:apg . .

# Create additional required directories
RUN mkdir -p templates/imex static logs && \
    chown -R apg:apg .

# Switch to non-root user
USER apg

# Set environment variables
ENV APG_ENVIRONMENT=production \
    PYTHONPATH=/opt/apg/imex \
    APG_LOG_LEVEL=INFO \
    APG_WORKER_PROCESSES=4 \
    APG_WORKER_THREADS=2 \
    APG_UPLOAD_FOLDER=/opt/apg/uploads \
    APG_TEMP_FOLDER=/tmp/apg \
    APG_LOG_FOLDER=/opt/apg/logs

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Create startup script
RUN cat > start.sh << 'EOF'
#!/bin/bash
set -e

# Wait for dependencies
echo "Waiting for dependencies..."
while ! nc -z ${DB_HOST:-localhost} ${DB_PORT:-5432}; do
    echo "Waiting for database..."
    sleep 2
done

while ! nc -z ${REDIS_HOST:-localhost} ${REDIS_PORT:-6379}; do
    echo "Waiting for Redis..."
    sleep 2
done

echo "Starting APG IMEX capability..."

# Run database migrations if needed
if [ "$RUN_MIGRATIONS" = "true" ]; then
    echo "Running database migrations..."
    python -c "
import asyncio
from database import DatabaseManager, DatabaseConfig

async def migrate():
    config = DatabaseConfig(
        host='${DB_HOST}',
        port=int('${DB_PORT}'),
        database='${DB_NAME}',
        user='${DB_USER}',
        password='${DB_PASSWORD}'
    )
    db = DatabaseManager(config)
    await db.initialize()
    print('Database migrations completed')

if __name__ == '__main__':
    asyncio.run(migrate())
"
fi

# Start the application
exec gunicorn \
    --bind 0.0.0.0:8000 \
    --workers ${APG_WORKER_PROCESSES:-4} \
    --threads ${APG_WORKER_THREADS:-2} \
    --timeout ${APG_WORKER_TIMEOUT:-300} \
    --keep-alive 2 \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --preload \
    --access-logfile - \
    --error-logfile - \
    --log-level ${APG_LOG_LEVEL:-info} \
    --capture-output \
    --enable-stdio-inheritance \
    "wsgi:application"
EOF

RUN chmod +x start.sh

# Default command
CMD ["./start.sh"]

# Build information
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=1.0.0

# Runtime labels
LABEL build_date=$BUILD_DATE \
      vcs_ref=$VCS_REF \
      version=$VERSION