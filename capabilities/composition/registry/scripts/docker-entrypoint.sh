#!/bin/bash
# APG Capability Registry - Docker Entrypoint Script
# Handles initialization, migrations, and startup for containerized deployments

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Default values
DEFAULT_WORKERS=4
DEFAULT_HOST="0.0.0.0"
DEFAULT_PORT=8000
DEFAULT_LOG_LEVEL="info"

# Environment variables with defaults
WORKERS=${WORKERS:-$DEFAULT_WORKERS}
HOST=${HOST:-$DEFAULT_HOST}
PORT=${PORT:-$DEFAULT_PORT}
LOG_LEVEL=${LOG_LEVEL:-$DEFAULT_LOG_LEVEL}
ENVIRONMENT=${ENVIRONMENT:-production}

# Function to wait for service availability
wait_for_service() {
    local service_name=$1
    local host=$2
    local port=$3
    local timeout=${4:-60}
    
    log "Waiting for $service_name at $host:$port..."
    
    local count=0
    until nc -z "$host" "$port" 2>/dev/null; do
        if [ $count -ge $timeout ]; then
            error "$service_name at $host:$port is not available after ${timeout}s"
        fi
        
        count=$((count + 1))
        sleep 1
    done
    
    log "$service_name is available"
}

# Function to extract host and port from URL
extract_db_info() {
    local db_url=$1
    
    # Extract host and port from DATABASE_URL
    # Format: postgresql://user:pass@host:port/dbname
    if [[ $db_url =~ postgresql://[^@]+@([^:]+):([0-9]+)/ ]]; then
        DB_HOST="${BASH_REMATCH[1]}"
        DB_PORT="${BASH_REMATCH[2]}"
    else
        warn "Could not parse DATABASE_URL format"
        DB_HOST="localhost"
        DB_PORT="5432"
    fi
}

# Function to extract Redis info
extract_redis_info() {
    local redis_url=$1
    
    # Extract host and port from REDIS_URL
    # Format: redis://[:password@]host:port[/db]
    if [[ $redis_url =~ redis://[^@]*@?([^:]+):([0-9]+) ]]; then
        REDIS_HOST="${BASH_REMATCH[1]}"
        REDIS_PORT="${BASH_REMATCH[2]}"
    else
        warn "Could not parse REDIS_URL format"
        REDIS_HOST="localhost"
        REDIS_PORT="6379"
    fi
}

# Function to run database migrations
run_migrations() {
    log "Running database migrations..."
    
    if ! python -m alembic upgrade head; then
        error "Database migrations failed"
    fi
    
    log "Database migrations completed successfully"
}

# Function to create initial data
create_initial_data() {
    if [ "$SKIP_INITIAL_DATA" = "true" ]; then
        log "Skipping initial data creation (SKIP_INITIAL_DATA=true)"
        return
    fi
    
    log "Creating initial data..."
    
    if [ -f "/app/scripts/create_initial_data.py" ]; then
        python /app/scripts/create_initial_data.py
        log "Initial data created successfully"
    else
        warn "Initial data script not found, skipping"
    fi
}

# Function to validate environment
validate_environment() {
    log "Validating environment configuration..."
    
    # Check required environment variables
    local required_vars=(
        "DATABASE_URL"
        "REDIS_URL"
        "SECRET_KEY"
        "JWT_SECRET_KEY"
    )
    
    local missing_vars=()
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            missing_vars+=("$var")
        fi
    done
    
    if [ ${#missing_vars[@]} -ne 0 ]; then
        error "Missing required environment variables: ${missing_vars[*]}"
    fi
    
    # Validate SECRET_KEY length
    if [ ${#SECRET_KEY} -lt 32 ]; then
        error "SECRET_KEY must be at least 32 characters long"
    fi
    
    log "Environment validation passed"
}

# Function to setup logging directory
setup_logging() {
    log "Setting up logging..."
    
    # Create logs directory if it doesn't exist
    mkdir -p /app/logs
    
    # Set proper permissions
    chmod 755 /app/logs
    
    log "Logging setup completed"
}

# Function to check health of dependencies
check_dependencies() {
    log "Checking dependencies..."
    
    # Extract database connection info
    if [ -n "$DATABASE_URL" ]; then
        extract_db_info "$DATABASE_URL"
        wait_for_service "PostgreSQL" "$DB_HOST" "$DB_PORT"
    fi
    
    # Extract Redis connection info
    if [ -n "$REDIS_URL" ]; then
        extract_redis_info "$REDIS_URL"
        wait_for_service "Redis" "$REDIS_HOST" "$REDIS_PORT"
    fi
    
    log "All dependencies are available"
}

# Function to optimize for container environment
optimize_container() {
    log "Optimizing for container environment..."
    
    # Set timezone if specified
    if [ -n "$TZ" ]; then
        ln -snf /usr/share/zoneinfo/$TZ /etc/localtime
        echo $TZ > /etc/timezone
        log "Timezone set to $TZ"
    fi
    
    # Configure Python optimizations
    export PYTHONUNBUFFERED=1
    export PYTHONDONTWRITEBYTECODE=1
    
    log "Container optimizations applied"
}

# Function to handle different startup modes
handle_startup_mode() {
    local mode=${1:-"server"}
    
    case $mode in
        "server")
            log "Starting APG Capability Registry server..."
            exec uvicorn capability_registry.api:api_app \
                --host "$HOST" \
                --port "$PORT" \
                --workers "$WORKERS" \
                --log-level "$LOG_LEVEL" \
                --access-log \
                --use-colors
            ;;
        "worker")
            log "Starting Celery worker..."
            exec celery -A capability_registry.tasks worker \
                --loglevel="$LOG_LEVEL" \
                --concurrency="${CELERY_CONCURRENCY:-4}" \
                --hostname="worker@%h"
            ;;
        "beat")
            log "Starting Celery beat scheduler..."
            exec celery -A capability_registry.tasks beat \
                --loglevel="$LOG_LEVEL" \
                --schedule="/app/celerybeat-schedule"
            ;;
        "flower")
            log "Starting Flower monitoring..."
            exec celery -A capability_registry.tasks flower \
                --port="${FLOWER_PORT:-5555}" \
                --basic_auth="${FLOWER_BASIC_AUTH:-admin:admin}"
            ;;
        "migrate")
            log "Running migrations only..."
            run_migrations
            log "Migration completed, exiting"
            exit 0
            ;;
        "shell")
            log "Starting interactive Python shell..."
            exec python -i -c "
import asyncio
from capability_registry.service import get_registry_service
from capability_registry.models import *
print('APG Capability Registry Shell')
print('Available imports: asyncio, get_registry_service, models')
"
            ;;
        *)
            error "Unknown startup mode: $mode"
            ;;
    esac
}

# Function to cleanup on exit
cleanup() {
    log "Cleaning up..."
    
    # Clean up temporary files
    rm -rf /tmp/prometheus_multiproc_dir/*
    
    log "Cleanup completed"
}

# Trap cleanup function on exit
trap cleanup EXIT

# Main execution
main() {
    log "APG Capability Registry starting up..."
    log "Environment: $ENVIRONMENT"
    log "Workers: $WORKERS"
    log "Host: $HOST"
    log "Port: $PORT"
    
    # Perform startup checks
    validate_environment
    optimize_container
    setup_logging
    
    # Only check dependencies if not in migration-only mode
    if [ "$1" != "migrate" ]; then
        check_dependencies
    fi
    
    # Run migrations unless explicitly skipped
    if [ "$SKIP_MIGRATIONS" != "true" ]; then
        run_migrations
    else
        log "Skipping migrations (SKIP_MIGRATIONS=true)"
    fi
    
    # Create initial data unless explicitly skipped
    if [ "$1" = "server" ]; then
        create_initial_data
    fi
    
    # Handle the startup mode
    handle_startup_mode "$1"
}

# Execute main function with all arguments
main "$@"