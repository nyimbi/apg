#!/bin/bash
# APG API Service Mesh - Container Entrypoint
#
# Production-ready entrypoint script with health checks, graceful shutdown,
# and comprehensive initialization procedures.
#
# © 2025 Datacraft. All rights reserved.
# Author: Nyimbi Odero <nyimbi@gmail.com>

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

# Default values
ENVIRONMENT="${ENVIRONMENT:-production}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-4}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
DATABASE_URL="${DATABASE_URL:-}"
REDIS_URL="${REDIS_URL:-}"

# Application settings
APP_MODULE="api_service_mesh.api:api_app"
LOG_CONFIG="docker/logging.conf"

# =============================================================================
# Logging Functions
# =============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2
}

log_info() {
    log "INFO: $*"
}

log_error() {
    log "ERROR: $*"
}

log_warn() {
    log "WARNING: $*"
}

# =============================================================================
# Health Check Functions
# =============================================================================

check_database() {
    log_info "Checking database connectivity..."
    
    if [ -z "$DATABASE_URL" ]; then
        log_error "DATABASE_URL not set"
        return 1
    fi
    
    # Extract components from DATABASE_URL
    local db_host=$(echo "$DATABASE_URL" | sed -n 's/.*@\([^:]*\):.*/\1/p')
    local db_port=$(echo "$DATABASE_URL" | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
    
    # Wait for database to be ready
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if nc -z "$db_host" "$db_port" 2>/dev/null; then
            log_info "Database is ready"
            return 0
        fi
        
        log_warn "Database not ready, attempt $attempt/$max_attempts"
        sleep 2
        attempt=$((attempt + 1))
    done
    
    log_error "Database failed to become ready after $max_attempts attempts"
    return 1
}

check_redis() {
    log_info "Checking Redis connectivity..."
    
    if [ -z "$REDIS_URL" ]; then
        log_error "REDIS_URL not set"
        return 1
    fi
    
    # Extract components from REDIS_URL
    local redis_host=$(echo "$REDIS_URL" | sed -n 's/.*:\/\/\([^:]*\):.*/\1/p')
    local redis_port=$(echo "$REDIS_URL" | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
    
    # Wait for Redis to be ready
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if nc -z "$redis_host" "$redis_port" 2>/dev/null; then
            log_info "Redis is ready"
            return 0
        fi
        
        log_warn "Redis not ready, attempt $attempt/$max_attempts"
        sleep 2
        attempt=$((attempt + 1))
    done
    
    log_error "Redis failed to become ready after $max_attempts attempts"
    return 1
}

# =============================================================================
# Database Migration
# =============================================================================

run_migrations() {
    log_info "Running database migrations..."
    
    if command -v alembic >/dev/null 2>&1; then
        alembic upgrade head
        log_info "Database migrations completed"
    else
        log_warn "Alembic not found, skipping migrations"
    fi
}

# =============================================================================
# Application Initialization
# =============================================================================

initialize_app() {
    log_info "Initializing application..."
    
    # Create necessary directories
    mkdir -p /app/logs /app/data
    
    # Set proper permissions
    chmod 755 /app/logs /app/data
    
    # Initialize application data if needed
    if [ -f "scripts/init_app.py" ]; then
        python scripts/init_app.py
        log_info "Application initialization completed"
    fi
}

# =============================================================================
# Signal Handlers
# =============================================================================

cleanup() {
    log_info "Received shutdown signal, performing cleanup..."
    
    # Kill child processes
    if [ -n "${PID:-}" ]; then
        kill -TERM "$PID" 2>/dev/null || true
        wait "$PID" 2>/dev/null || true
    fi
    
    log_info "Cleanup completed"
    exit 0
}

# Trap signals
trap cleanup SIGTERM SIGINT SIGQUIT

# =============================================================================
# Main Execution
# =============================================================================

main() {
    log_info "Starting APG API Service Mesh (Environment: $ENVIRONMENT)"
    log_info "Configuration: Host=$HOST, Port=$PORT, Workers=$WORKERS, LogLevel=$LOG_LEVEL"
    
    # Perform health checks
    check_database || exit 1
    check_redis || exit 1
    
    # Run migrations
    run_migrations
    
    # Initialize application
    initialize_app
    
    # Determine startup command based on environment
    if [ "$ENVIRONMENT" = "development" ]; then
        log_info "Starting development server with hot reload..."
        exec python -m uvicorn \
            "$APP_MODULE" \
            --host "$HOST" \
            --port "$PORT" \
            --log-level "$(echo "$LOG_LEVEL" | tr '[:upper:]' '[:lower:]')" \
            --reload \
            --reload-dir /app
    else
        log_info "Starting production server with Gunicorn..."
        exec gunicorn \
            "$APP_MODULE" \
            --bind "$HOST:$PORT" \
            --workers "$WORKERS" \
            --worker-class uvicorn.workers.UvicornWorker \
            --log-level "$(echo "$LOG_LEVEL" | tr '[:upper:]' '[:lower:]')" \
            --log-config "$LOG_CONFIG" \
            --access-logfile /app/logs/access.log \
            --error-logfile /app/logs/error.log \
            --capture-output \
            --enable-stdio-inheritance \
            --preload \
            --max-requests 1000 \
            --max-requests-jitter 100 \
            --timeout 30 \
            --keep-alive 2 \
            --graceful-timeout 30 &
        
        PID=$!
        log_info "Application started with PID $PID"
        
        # Wait for the process to complete
        wait "$PID"
    fi
}

# Run main function
main "$@"