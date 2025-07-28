# APG Financial Management General Ledger - Production Dockerfile
# Revolutionary AI-powered General Ledger System
# © 2025 Datacraft. All rights reserved.

FROM python:3.11-slim AS builder

# Set build arguments
ARG BUILD_ENV=production
ARG VERSION=1.0.0

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install UV for fast Python package management
RUN pip install uv

# Create application user
RUN groupadd -r gluser && useradd -r -g gluser gluser

# Set working directory
WORKDIR /app

# Copy dependency files
COPY requirements.txt requirements-prod.txt ./

# Install Python dependencies
RUN uv pip install --system -r requirements-prod.txt

# Copy application code
COPY . .

# Set ownership
RUN chown -R gluser:gluser /app

# Remove build dependencies
RUN apt-get purge -y build-essential && apt-get autoremove -y

# Production stage
FROM python:3.11-slim AS production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN groupadd -r gluser && useradd -r -g gluser gluser

# Create directories
RUN mkdir -p /app /var/log/gl /var/lib/gl && \
    chown -R gluser:gluser /app /var/log/gl /var/lib/gl

# Copy from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder --chown=gluser:gluser /app /app

# Switch to application user
USER gluser

# Set working directory
WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Set default environment variables
ENV PYTHONPATH=/app \
    LOG_LEVEL=INFO \
    PORT=8000 \
    WORKERS=4

# Default command
CMD ["python", "run.py", "--port", "8000", "--workers", "4"]

# Labels for metadata
LABEL maintainer="Nyimbi Odero <nyimbi@gmail.com>" \
      version="${VERSION}" \
      description="APG Financial Management General Ledger - Revolutionary AI-powered GL system" \
      org.opencontainers.image.title="APG General Ledger" \
      org.opencontainers.image.description="Revolutionary AI-powered General Ledger system" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.vendor="Datacraft" \
      org.opencontainers.image.licenses="Proprietary" \
      org.opencontainers.image.source="https://github.com/datacraft/apg-general-ledger"