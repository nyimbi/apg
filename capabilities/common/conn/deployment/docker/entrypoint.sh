#!/bin/bash
set -e

# APG Connection Management - Docker Entrypoint
# Production startup script with health checks and graceful shutdown
#
# Author: Nyimbi Odero
# Company: Datacraft
# Copyright: © 2025

# Colors for logging
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] APG-CONN:${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] APG-CONN WARNING:${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] APG-CONN ERROR:${NC} $1"
}

# Function to wait for database
wait_for_db() {
    log "Waiting for database connection..."

    # Default database connection settings
    DB_HOST=${APG_DB_HOST:-localhost}
    DB_PORT=${APG_DB_PORT:-5432}
    DB_NAME=${APG_DB_NAME:-apg}
    DB_USER=${APG_DB_USER:-apg}

    # Wait for database to be ready
    timeout=60
    count=0

    while ! pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" > /dev/null 2>&1; do
        count=$((count + 1))
        if [ $count -gt $timeout ]; then
            error "Database connection timeout after ${timeout} seconds"
            exit 1
        fi
        log "Database not ready, waiting... (${count}/${timeout})"
        sleep 1
    done

    log "Database connection established"
}

# Function to run database migrations
run_migrations() {
    if [ "${APG_RUN_MIGRATIONS:-true}" = "true" ]; then
        log "Running database migrations..."

        # Run Alembic migrations if available
        if [ -f "alembic.ini" ]; then
            alembic upgrade head
        else
            warn "No alembic.ini found, skipping migrations"
        fi

        log "Migrations completed"
    else
        log "Skipping migrations (APG_RUN_MIGRATIONS=false)"
    fi
}

# Function to initialize application
init_app() {
    log "Initializing APG Connection Management capability..."

    # Create necessary directories
    mkdir -p /app/logs /app/data /app/tmp

    # Set up logging configuration
    export APG_LOG_FILE="${APG_LOG_FILE:-/app/logs/apg-conn.log}"
    export APG_LOG_LEVEL="${APG_LOG_LEVEL:-INFO}"

    # Validate configuration
    python -c "
import sys
sys.path.insert(0, '/app')
try:
    from service import ConnectionManager
    from error_handling import global_error_handler
    from monitoring import global_metrics_collector
    print('✓ Configuration validation successful')
except Exception as e:
    print(f'✗ Configuration validation failed: {e}')
    sys.exit(1)
"

    log "Application initialization completed"
}

# Function to start the application
start_app() {
    log "Starting APG Connection Management capability..."

    # Application configuration
    HOST=${APG_HOST:-0.0.0.0}
    PORT=${APG_PORT:-8000}
    WORKERS=${APG_WORKERS:-4}

    # Start with Gunicorn for production
    if [ "${APG_ENV}" = "production" ]; then
        log "Starting with Gunicorn (${WORKERS} workers on ${HOST}:${PORT})"

        exec gunicorn \
            --bind "${HOST}:${PORT}" \
            --workers $WORKERS \
            --worker-class uvicorn.workers.UvicornWorker \
            --worker-connections 1000 \
            --timeout 120 \
            --keepalive 2 \
            --max-requests 1000 \
            --max-requests-jitter 100 \
            --preload \
            --access-logfile /app/logs/access.log \
            --error-logfile /app/logs/error.log \
            --log-level info \
            --capture-output \
            --enable-stdio-inheritance \
            --user apg \
            --group apg \
            --pid /app/gunicorn.pid \
            "app:create_app()"
    else
        log "Starting in development mode"

        # Development mode with auto-reload
        exec python -m uvicorn app:create_app \
            --host "$HOST" \
            --port "$PORT" \
            --reload \
            --log-level debug
    fi
}

# Function to handle shutdown
shutdown_handler() {
    log "Received shutdown signal, gracefully shutting down..."

    # Kill Gunicorn master process if exists
    if [ -f /app/gunicorn.pid ]; then
        PID=$(cat /app/gunicorn.pid)
        log "Sending SIGTERM to Gunicorn master process (PID: $PID)"
        kill -TERM $PID

        # Wait for graceful shutdown
        timeout=30
        count=0
        while kill -0 $PID 2>/dev/null && [ $count -lt $timeout ]; do
            count=$((count + 1))
            log "Waiting for graceful shutdown... (${count}/${timeout})"
            sleep 1
        done

        # Force kill if still running
        if kill -0 $PID 2>/dev/null; then
            warn "Graceful shutdown timeout, force killing process"
            kill -KILL $PID
        fi

        rm -f /app/gunicorn.pid
    fi

    log "Shutdown completed"
    exit 0
}

# Set up signal handlers for graceful shutdown
trap shutdown_handler SIGTERM SIGINT

# Main execution
main() {
    log "APG Connection Management Capability starting up..."
    log "Environment: ${APG_ENV:-development}"
    log "Python version: $(python --version)"
    log "Working directory: $(pwd)"

    # Pre-flight checks
    if [ "${APG_SKIP_DB_WAIT:-false}" != "true" ]; then
        wait_for_db
    fi

    # Initialize application
    init_app

    # Run migrations
    run_migrations

    # Start the application
    start_app
}

# Execute main function
main "$@"