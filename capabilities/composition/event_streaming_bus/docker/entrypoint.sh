#!/bin/bash
# APG Event Streaming Bus - Docker Entrypoint Script
# Handles initialization, health checks, and graceful startup
# © 2025 Datacraft. All rights reserved.

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] ESB: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ESB ERROR: $1${NC}" >&2
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ESB SUCCESS: $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ESB WARNING: $1${NC}"
}

# Environment setup
setup_environment() {
    log "Setting up environment..."
    
    # Create necessary directories
    mkdir -p /app/logs /app/data /app/tmp
    
    # Set default values if not provided
    export ENV=${ENV:-development}
    export LOG_LEVEL=${LOG_LEVEL:-INFO}
    export API_PORT=${API_PORT:-8080}
    export API_WORKERS=${API_WORKERS:-4}
    export METRICS_PORT=${METRICS_PORT:-9090}
    export METRICS_ENABLED=${METRICS_ENABLED:-true}
    
    log "Environment: $ENV"
    log "Log level: $LOG_LEVEL"
    log "API port: $API_PORT"
    log "API workers: $API_WORKERS"
}

# Wait for dependencies
wait_for_dependencies() {
    log "Waiting for dependencies..."
    
    # Wait for PostgreSQL
    if [ -n "$DATABASE_URL" ]; then
        log "Waiting for PostgreSQL..."
        while ! python -c "
import psycopg2
import os
from urllib.parse import urlparse
url = urlparse(os.environ['DATABASE_URL'])
try:
    conn = psycopg2.connect(
        host=url.hostname,
        port=url.port or 5432,
        user=url.username,
        password=url.password,
        database=url.path[1:],
        connect_timeout=5
    )
    conn.close()
    print('PostgreSQL is ready')
except Exception as e:
    print(f'PostgreSQL not ready: {e}')
    exit(1)
" 2>/dev/null; do
            warn "PostgreSQL is not ready. Waiting..."
            sleep 2
        done
        success "PostgreSQL is ready"
    fi
    
    # Wait for Redis
    if [ -n "$REDIS_URL" ]; then
        log "Waiting for Redis..."
        while ! python -c "
import redis
import os
from urllib.parse import urlparse
url = urlparse(os.environ['REDIS_URL'])
try:
    r = redis.Redis(
        host=url.hostname,
        port=url.port or 6379,
        db=url.path[1:] if url.path else 0,
        socket_connect_timeout=5
    )
    r.ping()
    print('Redis is ready')
except Exception as e:
    print(f'Redis not ready: {e}')
    exit(1)
" 2>/dev/null; do
            warn "Redis is not ready. Waiting..."
            sleep 2
        done
        success "Redis is ready"
    fi
    
    # Wait for Kafka
    if [ -n "$KAFKA_BOOTSTRAP_SERVERS" ]; then
        log "Waiting for Kafka..."
        while ! python -c "
from kafka import KafkaProducer
import os
try:
    producer = KafkaProducer(
        bootstrap_servers=os.environ['KAFKA_BOOTSTRAP_SERVERS'].split(','),
        request_timeout_ms=5000
    )
    producer.close()
    print('Kafka is ready')
except Exception as e:
    print(f'Kafka not ready: {e}')
    exit(1)
" 2>/dev/null; do
            warn "Kafka is not ready. Waiting..."
            sleep 2
        done
        success "Kafka is ready"
    fi
}

# Initialize database
initialize_database() {
    log "Initializing database..."
    
    if [ -n "$DATABASE_URL" ]; then
        # Run database migrations
        log "Running database migrations..."
        python -m alembic upgrade head
        
        if [ $? -eq 0 ]; then
            success "Database migrations completed"
        else
            error "Database migrations failed"
            exit 1
        fi
    else
        warn "No DATABASE_URL provided, skipping database initialization"
    fi
}

# Initialize Kafka topics
initialize_kafka() {
    log "Initializing Kafka topics..."
    
    if [ -n "$KAFKA_BOOTSTRAP_SERVERS" ]; then
        # Create default topics
        python -c "
from kafka.admin import KafkaAdminClient, NewTopic
import os

admin_client = KafkaAdminClient(
    bootstrap_servers=os.environ['KAFKA_BOOTSTRAP_SERVERS'].split(','),
    client_id='esb_admin'
)

topics = [
    NewTopic(name='apg-events', num_partitions=6, replication_factor=1),
    NewTopic(name='apg-events-dlq', num_partitions=3, replication_factor=1),
    NewTopic(name='apg-metrics', num_partitions=3, replication_factor=1),
    NewTopic(name='apg-audit', num_partitions=3, replication_factor=1)
]

try:
    admin_client.create_topics(topics)
    print('Kafka topics created successfully')
except Exception as e:
    print(f'Kafka topic creation: {e}')
finally:
    admin_client.close()
"
        success "Kafka topics initialized"
    else
        warn "No KAFKA_BOOTSTRAP_SERVERS provided, skipping Kafka initialization"
    fi
}

# Health check
health_check() {
    log "Running health check..."
    
    # Check if the application is responding
    timeout 30 bash -c 'until curl -f http://localhost:$API_PORT/health 2>/dev/null; do sleep 1; done'
    
    if [ $? -eq 0 ]; then
        success "Health check passed"
        return 0
    else
        error "Health check failed"
        return 1
    fi
}

# Signal handlers for graceful shutdown
cleanup() {
    log "Received shutdown signal, cleaning up..."
    
    if [ -n "$API_PID" ]; then
        log "Stopping API server (PID: $API_PID)..."
        kill -TERM "$API_PID" 2>/dev/null || true
        wait "$API_PID" 2>/dev/null || true
    fi
    
    if [ -n "$WORKER_PID" ]; then
        log "Stopping background workers (PID: $WORKER_PID)..."
        kill -TERM "$WORKER_PID" 2>/dev/null || true
        wait "$WORKER_PID" 2>/dev/null || true
    fi
    
    success "Cleanup completed"
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Start services based on command
start_api() {
    log "Starting API server..."
    
    exec python -m uvicorn api:app \
        --host 0.0.0.0 \
        --port "$API_PORT" \
        --workers "$API_WORKERS" \
        --log-level "${LOG_LEVEL,,}" \
        --access-log \
        --loop uvloop &
    
    API_PID=$!
    log "API server started with PID: $API_PID"
    
    # Wait for the process
    wait $API_PID
}

start_worker() {
    log "Starting background worker..."
    
    exec python -m celery worker \
        --app=worker.celery \
        --loglevel="${LOG_LEVEL,,}" \
        --concurrency=4 \
        --queue=events,processing,notifications &
    
    WORKER_PID=$!
    log "Background worker started with PID: $WORKER_PID"
    
    # Wait for the process
    wait $WORKER_PID
}

start_scheduler() {
    log "Starting task scheduler..."
    
    exec python -m celery beat \
        --app=worker.celery \
        --loglevel="${LOG_LEVEL,,}" \
        --schedule=/app/tmp/celerybeat-schedule &
    
    SCHEDULER_PID=$!
    log "Task scheduler started with PID: $SCHEDULER_PID"
    
    # Wait for the process
    wait $SCHEDULER_PID
}

start_monitoring() {
    log "Starting monitoring services..."
    
    # Start metrics exporter
    python -m prometheus_client \
        --port "$METRICS_PORT" &
    
    METRICS_PID=$!
    log "Metrics exporter started with PID: $METRICS_PID"
    
    # Wait for the process
    wait $METRICS_PID
}

# Main execution
main() {
    log "Starting APG Event Streaming Bus..."
    log "Version: $(python -c 'import pkg_resources; print(pkg_resources.get_distribution("event-streaming-bus").version)' 2>/dev/null || echo 'development')"
    
    # Setup
    setup_environment
    wait_for_dependencies
    initialize_database
    initialize_kafka
    
    # Determine what to start based on command
    case "${1:-api}" in
        "api")
            start_api
            ;;
        "worker")
            start_worker
            ;;
        "scheduler")
            start_scheduler
            ;;
        "monitoring")
            start_monitoring
            ;;
        "all")
            log "Starting all services..."
            start_api &
            API_PID=$!
            start_worker &
            WORKER_PID=$!
            start_scheduler &
            SCHEDULER_PID=$!
            start_monitoring &
            METRICS_PID=$!
            
            # Wait for any process to exit
            wait -n
            ;;
        "migrate")
            log "Running database migrations only..."
            initialize_database
            success "Migrations completed"
            exit 0
            ;;
        "shell")
            log "Starting interactive shell..."
            exec python
            ;;
        *)
            error "Unknown command: $1"
            echo "Usage: $0 {api|worker|scheduler|monitoring|all|migrate|shell}"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"