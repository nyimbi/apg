# APG RAG Production Dockerfile
# Multi-stage build for optimized production deployment

# Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=1.0.0

# Set build environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libffi-dev \
    libpq-dev \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create app user and directory
RUN useradd --create-home --shell /bin/bash app
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --user --no-warn-script-location -r requirements.txt

# Development stage
FROM builder as development

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install --user --no-warn-script-location -r requirements-dev.txt

# Copy application code
COPY . .

# Set development environment
ENV ENVIRONMENT=development \
    LOG_LEVEL=DEBUG \
    ENABLE_RELOAD=true

# Expose ports
EXPOSE 5000 9090

# Development command
CMD ["python", "-m", "uvicorn", "capabilities.common.rag.service:app", "--host", "0.0.0.0", "--port", "5000", "--reload"]

# Production stage
FROM python:3.11-slim as production

# Set production environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production \
    LOG_LEVEL=INFO \
    WORKERS=4 \
    WORKER_CLASS=gevent \
    WORKER_CONNECTIONS=1000 \
    MAX_REQUESTS=1000 \
    MAX_REQUESTS_JITTER=100 \
    TIMEOUT=300 \
    KEEPALIVE=5 \
    PRELOAD_APP=true

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libpq5 \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app user and directories
RUN useradd --create-home --shell /bin/bash --uid 1000 app && \
    mkdir -p /app/data /app/logs /app/temp /app/certs && \
    chown -R app:app /app

# Copy Python dependencies from builder
COPY --from=builder --chown=app:app /root/.local /home/app/.local

# Switch to app user
USER app
WORKDIR /app

# Update PATH to include user-installed packages
ENV PATH=/home/app/.local/bin:$PATH

# Copy application code
COPY --chown=app:app . .

# Create necessary directories
RUN mkdir -p data logs temp config

# Set file permissions
RUN chmod +x scripts/*.sh 2>/dev/null || true

# Add health check script
COPY --chown=app:app <<EOF /app/healthcheck.py
#!/usr/bin/env python3
import asyncio
import httpx
import sys

async def health_check():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:5000/api/v1/rag/health", timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                if data.get("success") and data.get("data", {}).get("service_status") == "running":
                    sys.exit(0)
        sys.exit(1)
    except Exception:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(health_check())
EOF

RUN chmod +x /app/healthcheck.py

# Add container labels
LABEL maintainer="Datacraft <nyimbi@gmail.com>" \
      version="${VERSION}" \
      description="APG RAG - Enterprise Retrieval-Augmented Generation Platform" \
      build-date="${BUILD_DATE}" \
      vcs-ref="${VCS_REF}" \
      org.opencontainers.image.title="APG RAG" \
      org.opencontainers.image.description="Enterprise-grade RAG platform" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.vendor="Datacraft" \
      org.opencontainers.image.authors="Nyimbi Odero <nyimbi@gmail.com>" \
      org.opencontainers.image.url="https://www.datacraft.co.ke" \
      org.opencontainers.image.source="https://github.com/datacraft/apg" \
      org.opencontainers.image.licenses="Proprietary"

# Expose ports
EXPOSE 5000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python /app/healthcheck.py

# Production startup script
COPY --chown=app:app <<EOF /app/start.sh
#!/bin/bash
set -e

# Wait for dependencies
echo "Waiting for dependencies..."
python -c "
import asyncio
import asyncpg
import httpx
import os
import sys
import time

async def wait_for_postgres():
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            conn = await asyncpg.connect(os.getenv('DATABASE_URL'))
            await conn.fetchval('SELECT 1')
            await conn.close()
            print('✓ PostgreSQL is ready')
            return True
        except Exception as e:
            print(f'Waiting for PostgreSQL... attempt {attempt + 1}/{max_attempts}')
            await asyncio.sleep(2)
    return False

async def wait_for_ollama():
    max_attempts = 60  # Ollama takes longer to start
    ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://ollama:11434')
    
    for attempt in range(max_attempts):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f'{ollama_url}/api/tags', timeout=5.0)
                if response.status_code == 200:
                    print('✓ Ollama is ready')
                    return True
        except Exception as e:
            print(f'Waiting for Ollama... attempt {attempt + 1}/{max_attempts}')
            await asyncio.sleep(2)
    return False

async def main():
    postgres_ready = await wait_for_postgres()
    ollama_ready = await wait_for_ollama()
    
    if not postgres_ready:
        print('❌ PostgreSQL not available after 60 seconds')
        sys.exit(1)
    
    if not ollama_ready:
        print('❌ Ollama not available after 120 seconds')
        sys.exit(1)
    
    print('✓ All dependencies are ready')

asyncio.run(main())
"

# Run database migrations if needed
echo "Running database initialization..."
python -c "
import asyncio
import asyncpg
import os

async def init_db():
    conn = await asyncpg.connect(os.getenv('DATABASE_URL'))
    
    # Check if tables exist
    tables = await conn.fetch(\"
        SELECT tablename FROM pg_tables 
        WHERE schemaname = 'public' AND tablename LIKE 'apg_rag_%'
    \")
    
    if len(tables) == 0:
        print('⚠️  RAG tables not found. Please run database migration first.')
    else:
        print(f'✓ Found {len(tables)} RAG tables')
    
    await conn.close()

asyncio.run(init_db())
"

# Start the application
echo "Starting APG RAG Service..."
exec gunicorn \
    --bind 0.0.0.0:5000 \
    --workers \$WORKERS \
    --worker-class \$WORKER_CLASS \
    --worker-connections \$WORKER_CONNECTIONS \
    --max-requests \$MAX_REQUESTS \
    --max-requests-jitter \$MAX_REQUESTS_JITTER \
    --timeout \$TIMEOUT \
    --keepalive \$KEEPALIVE \
    --preload \
    --access-logfile - \
    --error-logfile - \
    --log-level \${LOG_LEVEL,,} \
    --capture-output \
    capabilities.common.rag.service:app
EOF

RUN chmod +x /app/start.sh

# Default command
CMD ["/app/start.sh"]

# Production optimized stage with security hardening
FROM production as production-hardened

# Switch back to root for security hardening
USER root

# Security hardening
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Remove unnecessary packages and clean up
RUN apt-get purge -y \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Set security-focused filesystem permissions
RUN find /app -type f -exec chmod 644 {} \; && \
    find /app -type d -exec chmod 755 {} \; && \
    chmod 755 /app/start.sh /app/healthcheck.py && \
    chmod 600 /app/certs/* 2>/dev/null || true

# Create non-root user with minimal privileges
RUN usermod --shell /bin/false app && \
    usermod --home /app app

# Switch back to app user
USER app

# Security labels
LABEL security.scan="true" \
      security.hardened="true" \
      security.non-root="true"

# Final production stage selector
FROM production as final