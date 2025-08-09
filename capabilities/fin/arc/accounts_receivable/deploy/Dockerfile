# APG Accounts Receivable - Production Dockerfile
# Enterprise-grade containerization for AR capability

FROM python:3.11-slim as base

# Copyright and metadata
LABEL org.opencontainers.image.title="APG Accounts Receivable"
LABEL org.opencontainers.image.description="AI-Powered Accounts Receivable Management for APG Platform"
LABEL org.opencontainers.image.vendor="Datacraft"
LABEL org.opencontainers.image.authors="Nyimbi Odero <nyimbi@gmail.com>"
LABEL org.opencontainers.image.version="1.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
	PYTHONDONTWRITEBYTECODE=1 \
	PIP_NO_CACHE_DIR=1 \
	PIP_DISABLE_PIP_VERSION_CHECK=1 \
	DEBIAN_FRONTEND=noninteractive

# Create app user for security
RUN groupadd -r apg && useradd -r -g apg apg

# Install system dependencies
RUN apt-get update && apt-get install -y \
	build-essential \
	curl \
	libpq-dev \
	postgresql-client \
	redis-tools \
	netcat-traditional \
	procps \
	&& rm -rf /var/lib/apt/lists/*

# Install UV for faster Python package management
RUN pip install uv

# Create directory structure
WORKDIR /opt/apg
RUN mkdir -p \
	/opt/apg/ar \
	/opt/apg/logs \
	/opt/apg/data \
	/var/log/ar \
	/etc/apg/ar

# Copy requirements and install dependencies
COPY requirements.txt /opt/apg/
RUN uv pip install --system -r requirements.txt

# Production stage
FROM base as production

# Copy application code
COPY capabilities/core_financials/accounts_receivable/ /opt/apg/ar/
COPY capabilities/__init__.py /opt/apg/
COPY capabilities/core_financials/__init__.py /opt/apg/

# Copy configuration files
COPY deploy/config/ /etc/apg/ar/
COPY deploy/scripts/ /opt/apg/scripts/

# Set proper permissions
RUN chown -R apg:apg /opt/apg /var/log/ar /etc/apg/ar && \
	chmod +x /opt/apg/scripts/*.sh

# Health check script
COPY deploy/healthcheck.py /opt/apg/
RUN chmod +x /opt/apg/healthcheck.py

# Switch to non-root user
USER apg

# Expose ports
EXPOSE 8000 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
	CMD python /opt/apg/healthcheck.py

# Default command
CMD ["/opt/apg/scripts/start.sh"]

# Development stage
FROM base as development

# Install development dependencies
COPY requirements-dev.txt /opt/apg/
RUN uv pip install --system -r requirements-dev.txt

# Copy application code
COPY . /opt/apg/

# Install in development mode
RUN uv pip install --system -e .

# Development user with sudo access
USER root
RUN apt-get update && apt-get install -y sudo && \
	echo "apg ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
	rm -rf /var/lib/apt/lists/*

USER apg

# Development command
CMD ["/opt/apg/scripts/dev-start.sh"]