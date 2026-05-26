#!/bin/bash

# APG Connection Management - Health Check Script
# Docker health check for production deployments
#
# Author: Nyimbi Odero
# Company: Datacraft
# Copyright: © 2025

set -e

# Configuration
HOST=${APG_HOST:-localhost}
PORT=${APG_PORT:-8000}
TIMEOUT=${APG_HEALTHCHECK_TIMEOUT:-10}

# Health check endpoint
HEALTH_URL="http://${HOST}:${PORT}/monitoring/api/health"

# Perform health check
check_health() {
    # Use curl to check health endpoint
    if curl -f -s --max-time "$TIMEOUT" "$HEALTH_URL" > /dev/null 2>&1; then
        echo "✓ Health check passed"
        return 0
    else
        echo "✗ Health check failed - endpoint not responding"
        return 1
    fi
}

# Check if application process is running
check_process() {
    if pgrep -f "gunicorn.*app:create_app" > /dev/null; then
        echo "✓ Application process running"
        return 0
    else
        echo "✗ Application process not found"
        return 1
    fi
}

# Check database connectivity
check_database() {
    DB_HOST=${APG_DB_HOST:-localhost}
    DB_PORT=${APG_DB_PORT:-5432}
    DB_NAME=${APG_DB_NAME:-apg}
    DB_USER=${APG_DB_USER:-apg}

    if pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" > /dev/null 2>&1; then
        echo "✓ Database connectivity OK"
        return 0
    else
        echo "✗ Database connectivity failed"
        return 1
    fi
}

# Check disk space
check_disk_space() {
    # Check if disk usage is under 90%
    DISK_USAGE=$(df /app | awk 'NR==2 {print $5}' | sed 's/%//')

    if [ "$DISK_USAGE" -lt 90 ]; then
        echo "✓ Disk space OK (${DISK_USAGE}%)"
        return 0
    else
        echo "✗ Disk space critical (${DISK_USAGE}%)"
        return 1
    fi
}

# Check memory usage
check_memory() {
    # Get memory usage percentage
    MEMORY_USAGE=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')

    if [ "$MEMORY_USAGE" -lt 90 ]; then
        echo "✓ Memory usage OK (${MEMORY_USAGE}%)"
        return 0
    else
        echo "✗ Memory usage high (${MEMORY_USAGE}%)"
        return 1
    fi
}

# Main health check function
main() {
    echo "APG Connection Management - Health Check"
    echo "========================================"

    local exit_code=0

    # Basic process check
    if ! check_process; then
        exit_code=1
    fi

    # Application health endpoint
    if ! check_health; then
        exit_code=1
    fi

    # Database connectivity (if not skipped)
    if [ "${APG_SKIP_DB_HEALTHCHECK:-false}" != "true" ]; then
        if ! check_database; then
            exit_code=1
        fi
    fi

    # System resources
    if ! check_disk_space; then
        exit_code=1
    fi

    if ! check_memory; then
        exit_code=1
    fi

    # Summary
    echo "========================================"
    if [ $exit_code -eq 0 ]; then
        echo "✓ Overall health check PASSED"
    else
        echo "✗ Overall health check FAILED"
    fi

    return $exit_code
}

# Execute main function
main "$@"