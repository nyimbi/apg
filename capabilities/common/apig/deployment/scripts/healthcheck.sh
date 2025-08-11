#!/bin/bash
# APG Intelligent Gateway - Health Check Script
# Version: 1.0.0
# Date: August 11, 2025

set -euo pipefail

# Configuration
HEALTH_URL="http://localhost:8080/health"
TIMEOUT=10
MAX_RETRIES=3
RETRY_DELAY=2

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Health check function
check_health() {
    local url="$1"
    local timeout="$2"
    
    # Use curl to check health endpoint
    if command -v curl >/dev/null 2>&1; then
        response=$(curl -s -o /dev/null -w "%{http_code}" --max-time "$timeout" "$url" 2>/dev/null || echo "000")
        if [[ "$response" == "200" ]]; then
            return 0
        else
            log "${RED}Health check failed: HTTP $response${NC}"
            return 1
        fi
    else
        log "${RED}curl not found, cannot perform health check${NC}"
        return 1
    fi
}

# Component-specific health checks
check_component_health() {
    local component="$1"
    local endpoint="$2"
    local timeout="$3"
    
    if command -v curl >/dev/null 2>&1; then
        response=$(curl -s --max-time "$timeout" "$endpoint" 2>/dev/null || echo "ERROR")
        if [[ "$response" != "ERROR" ]] && echo "$response" | grep -q '"status".*"healthy"'; then
            log "${GREEN}$component: healthy${NC}"
            return 0
        else
            log "${YELLOW}$component: unhealthy or unreachable${NC}"
            return 1
        fi
    else
        log "${YELLOW}Cannot check $component health (curl not available)${NC}"
        return 1
    fi
}

# Main health check
main() {
    log "Starting APIG health check..."
    
    local success=true
    
    # Basic health endpoint check
    log "Checking main health endpoint..."
    if check_health "$HEALTH_URL" "$TIMEOUT"; then
        log "${GREEN}Main health check: PASSED${NC}"
    else
        log "${RED}Main health check: FAILED${NC}"
        success=false
    fi
    
    # Component health checks
    log "Checking component health..."
    
    # Check detailed status endpoint
    if check_component_health "Core Service" "http://localhost:8080/status" "$TIMEOUT"; then
        echo > /dev/null
    else
        success=false
    fi
    
    # Check metrics endpoint
    if command -v curl >/dev/null 2>&1; then
        metrics_response=$(curl -s --max-time 5 "http://localhost:9090/metrics" 2>/dev/null || echo "ERROR")
        if [[ "$metrics_response" != "ERROR" ]] && echo "$metrics_response" | grep -q "apig_"; then
            log "${GREEN}Metrics endpoint: healthy${NC}"
        else
            log "${YELLOW}Metrics endpoint: unhealthy${NC}"
            # Don't fail on metrics endpoint issues
        fi
    fi
    
    # Check readiness endpoint
    if command -v curl >/dev/null 2>&1; then
        ready_response=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "http://localhost:8080/ready" 2>/dev/null || echo "000")
        if [[ "$ready_response" == "200" ]]; then
            log "${GREEN}Readiness check: PASSED${NC}"
        else
            log "${YELLOW}Readiness check: FAILED (HTTP $ready_response)${NC}"
            # Don't fail on readiness issues during startup
        fi
    fi
    
    # Memory check (basic)
    if command -v free >/dev/null 2>&1; then
        memory_usage=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
        if (( $(echo "$memory_usage > 90.0" | bc -l) )); then
            log "${YELLOW}High memory usage: ${memory_usage}%${NC}"
        else
            log "${GREEN}Memory usage: ${memory_usage}%${NC}"
        fi
    fi
    
    # Disk space check (basic)
    if command -v df >/dev/null 2>&1; then
        disk_usage=$(df /app | tail -1 | awk '{print $5}' | sed 's/%//')
        if [[ "$disk_usage" -gt 80 ]]; then
            log "${YELLOW}High disk usage: ${disk_usage}%${NC}"
        else
            log "${GREEN}Disk usage: ${disk_usage}%${NC}"
        fi
    fi
    
    # Final result
    if $success; then
        log "${GREEN}APIG health check: PASSED${NC}"
        exit 0
    else
        log "${RED}APIG health check: FAILED${NC}"
        exit 1
    fi
}

# Retry logic for health checks
retry_health_check() {
    local attempt=1
    
    while [[ $attempt -le $MAX_RETRIES ]]; do
        log "Health check attempt $attempt/$MAX_RETRIES"
        
        if main; then
            exit 0
        fi
        
        if [[ $attempt -lt $MAX_RETRIES ]]; then
            log "Waiting ${RETRY_DELAY}s before retry..."
            sleep "$RETRY_DELAY"
        fi
        
        ((attempt++))
    done
    
    log "${RED}Health check failed after $MAX_RETRIES attempts${NC}"
    exit 1
}

# Check if we're in retry mode
if [[ "${1:-}" == "--retry" ]]; then
    retry_health_check
else
    main
fi