#!/bin/bash

# APG Billing System Monitoring Script
# Supports configuration by composition/central_configuration

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load configuration from central configuration if available
if [ -f "/etc/apg/composition/central_configuration" ]; then
    source /etc/apg/composition/central_configuration
    MONITOR_CONFIG_SOURCE="central_configuration"
elif [ -f "$PROJECT_DIR/../../../composition/central_configuration" ]; then
    source "$PROJECT_DIR/../../../composition/central_configuration"
    MONITOR_CONFIG_SOURCE="composition_relative"
elif [ -f "$PROJECT_DIR/.env" ]; then
    export $(grep -v '^#' "$PROJECT_DIR/.env" | xargs)
    MONITOR_CONFIG_SOURCE="local_env"
else
    echo "No configuration found - using defaults"
    MONITOR_CONFIG_SOURCE="defaults"
fi

# Default configuration (can be overridden by central config)
SERVICE_URL="${APG_SERVICE_URL:-http://localhost:5000}"
HEALTH_ENDPOINT="${APG_HEALTH_ENDPOINT:-/billing/health}"
CHECK_INTERVAL="${APG_MONITOR_INTERVAL:-60}"
ALERT_THRESHOLD="${APG_ALERT_THRESHOLD:-3}"
DATABASE_URL="${DATABASE_URL:-postgresql://postgres:postgres@localhost:5432/apg_billing}"
REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"
SLACK_WEBHOOK="${APG_SLACK_WEBHOOK:-}"
EMAIL_ALERTS="${APG_EMAIL_ALERTS:-}"
PAGERDUTY_KEY="${APG_PAGERDUTY_KEY:-}"

# Thresholds
CPU_THRESHOLD="${APG_CPU_THRESHOLD:-80}"
MEMORY_THRESHOLD="${APG_MEMORY_THRESHOLD:-85}"
DISK_THRESHOLD="${APG_DISK_THRESHOLD:-90}"
RESPONSE_TIME_THRESHOLD="${APG_RESPONSE_TIME_THRESHOLD:-5000}"  # milliseconds

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

# Send alert notifications
send_alert() {
    local severity="$1"
    local message="$2"
    local details="$3"
    
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local full_message="APG Billing Alert [$severity] at $timestamp: $message"
    
    if [ -n "$details" ]; then
        full_message="$full_message\n\nDetails:\n$details"
    fi
    
    # Slack notification
    if [ -n "$SLACK_WEBHOOK" ]; then
        local color="warning"
        case "$severity" in
            "CRITICAL") color="danger" ;;
            "HIGH") color="danger" ;;
            "MEDIUM") color="warning" ;;
            "LOW") color="good" ;;
        esac
        
        curl -X POST -H 'Content-type: application/json' \
            --data "{
                \"text\": \"$full_message\",
                \"attachments\": [{
                    \"color\": \"$color\",
                    \"fields\": [{
                        \"title\": \"Severity\",
                        \"value\": \"$severity\",
                        \"short\": true
                    }, {
                        \"title\": \"Service\",
                        \"value\": \"APG Billing\",
                        \"short\": true
                    }]
                }]
            }" \
            "$SLACK_WEBHOOK" >/dev/null 2>&1 || true
    fi
    
    # Email notification
    if [ -n "$EMAIL_ALERTS" ]; then
        echo -e "$full_message" | mail -s "APG Billing Alert [$severity]" "$EMAIL_ALERTS" 2>/dev/null || true
    fi
    
    # PagerDuty notification for critical alerts
    if [ -n "$PAGERDUTY_KEY" ] && [[ "$severity" == "CRITICAL" || "$severity" == "HIGH" ]]; then
        curl -X POST "https://events.pagerduty.com/v2/enqueue" \
            -H "Content-Type: application/json" \
            -d "{
                \"routing_key\": \"$PAGERDUTY_KEY\",
                \"event_action\": \"trigger\",
                \"dedup_key\": \"apg_billing_$(echo "$message" | tr ' ' '_')\",
                \"payload\": {
                    \"summary\": \"$message\",
                    \"severity\": \"$(echo "$severity" | tr '[:upper:]' '[:lower:]')\",
                    \"source\": \"APG Billing Monitoring\",
                    \"component\": \"billing_system\",
                    \"custom_details\": {
                        \"details\": \"$details\",
                        \"config_source\": \"$MONITOR_CONFIG_SOURCE\"
                    }
                }
            }" >/dev/null 2>&1 || true
    fi
}

# Check service health
check_service_health() {
    local start_time=$(date +%s%3N)
    local response=$(curl -s -w "%{http_code}" -o /tmp/health_response "$SERVICE_URL$HEALTH_ENDPOINT" 2>/dev/null || echo "000")
    local end_time=$(date +%s%3N)
    local response_time=$((end_time - start_time))
    
    local http_code="${response: -3}"
    
    if [ "$http_code" = "200" ]; then
        if [ $response_time -gt $RESPONSE_TIME_THRESHOLD ]; then
            log_warning "Service responding slowly: ${response_time}ms"
            return 1
        else
            log_success "Service health check passed (${response_time}ms)"
            return 0
        fi
    else
        log_error "Service health check failed (HTTP $http_code)"
        return 1
    fi
}

# Check database connectivity
check_database() {
    if [[ "$DATABASE_URL" == postgresql://* ]]; then
        # PostgreSQL check
        if timeout 10 pg_isready -d "$DATABASE_URL" >/dev/null 2>&1; then
            log_success "Database connectivity check passed"
            return 0
        else
            log_error "Database connectivity check failed"
            return 1
        fi
    elif [[ "$DATABASE_URL" == sqlite://* ]]; then
        # SQLite check
        local db_file="${DATABASE_URL#sqlite:///}"
        if [ -f "$db_file" ] && [ -r "$db_file" ]; then
            log_success "Database file accessible"
            return 0
        else
            log_error "Database file not accessible: $db_file"
            return 1
        fi
    else
        log_warning "Unknown database type, skipping check"
        return 0
    fi
}

# Check Redis connectivity
check_redis() {
    if [[ "$REDIS_URL" == redis://* ]]; then
        local redis_host_port="${REDIS_URL#redis://}"
        local redis_host="${redis_host_port%:*}"
        local redis_port="${redis_host_port##*:}"
        redis_port="${redis_port%/*}"
        
        if timeout 5 redis-cli -h "$redis_host" -p "$redis_port" ping >/dev/null 2>&1; then
            log_success "Redis connectivity check passed"
            return 0
        else
            log_error "Redis connectivity check failed"
            return 1
        fi
    else
        log_warning "Redis not configured, skipping check"
        return 0
    fi
}

# Check system resources
check_system_resources() {
    local alerts=()
    
    # CPU usage
    if command -v top >/dev/null 2>&1; then
        local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//' | cut -d'%' -f1)
        if [ -n "$cpu_usage" ] && (( $(echo "$cpu_usage > $CPU_THRESHOLD" | bc -l) )); then
            alerts+=("High CPU usage: ${cpu_usage}%")
        fi
    fi
    
    # Memory usage
    if command -v free >/dev/null 2>&1; then
        local memory_usage=$(free | awk 'NR==2{printf "%.1f", $3*100/$2}')
        if [ -n "$memory_usage" ] && (( $(echo "$memory_usage > $MEMORY_THRESHOLD" | bc -l) )); then
            alerts+=("High memory usage: ${memory_usage}%")
        fi
    fi
    
    # Disk usage
    local disk_usage=$(df / | awk 'NR==2{print $5}' | sed 's/%//')
    if [ -n "$disk_usage" ] && [ "$disk_usage" -gt "$DISK_THRESHOLD" ]; then
        alerts+=("High disk usage: ${disk_usage}%")
    fi
    
    if [ ${#alerts[@]} -eq 0 ]; then
        log_success "System resources check passed"
        return 0
    else
        for alert in "${alerts[@]}"; do
            log_warning "$alert"
        done
        return 1
    fi
}

# Check Docker containers (if using Docker)
check_docker_containers() {
    if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
        local failing_containers=$(docker ps --filter "status=exited" --filter "label=apg-billing" --format "{{.Names}}" 2>/dev/null)
        
        if [ -n "$failing_containers" ]; then
            log_error "Failed containers: $failing_containers"
            return 1
        else
            log_success "Docker containers check passed"
            return 0
        fi
    else
        log_info "Docker not available, skipping container check"
        return 0
    fi
}

# Check background tasks
check_background_tasks() {
    local health_response=$(cat /tmp/health_response 2>/dev/null || echo "{}")
    
    # Extract background task status from health response
    local renewals_status=$(echo "$health_response" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print(data.get('background_tasks', {}).get('renewals', 'unknown'))
except:
    print('unknown')
" 2>/dev/null)
    
    if [ "$renewals_status" = "running" ]; then
        log_success "Background tasks check passed"
        return 0
    else
        log_error "Background tasks not running properly"
        return 1
    fi
}

# Check API endpoints
check_api_endpoints() {
    local endpoints=(
        "/api/v1/billing/health"
        "/api/v1/billing/plans"
        "/api/v1/billing/customers"
    )
    
    local failed_endpoints=()
    
    for endpoint in "${endpoints[@]}"; do
        local response=$(curl -s -w "%{http_code}" -o /dev/null "$SERVICE_URL$endpoint" 2>/dev/null || echo "000")
        
        if [[ "$response" != "200" && "$response" != "401" ]]; then  # 401 is OK for protected endpoints
            failed_endpoints+=("$endpoint")
        fi
    done
    
    if [ ${#failed_endpoints[@]} -eq 0 ]; then
        log_success "API endpoints check passed"
        return 0
    else
        log_error "Failed API endpoints: ${failed_endpoints[*]}"
        return 1
    fi
}

# Generate system report
generate_system_report() {
    local report_file="/tmp/apg_billing_system_report.txt"
    
    cat > "$report_file" << EOF
APG Billing System Status Report
Generated: $(date)
Configuration Source: $MONITOR_CONFIG_SOURCE

=== Service Health ===
$(check_service_health 2>&1)

=== Database Status ===
$(check_database 2>&1)

=== Cache Status ===
$(check_redis 2>&1)

=== System Resources ===
CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' 2>/dev/null || echo "N/A")
Memory: $(free -h | awk 'NR==2{printf "Used: %s/%s (%.1f%%)", $3,$2,$3*100/$2}' 2>/dev/null || echo "N/A")
Disk: $(df -h / | awk 'NR==2{printf "%s used (%s)", $5, $3}' 2>/dev/null || echo "N/A")

=== Docker Containers ===
$(docker ps --format "table {{.Names}}\t{{.Status}}" 2>/dev/null || echo "Docker not available")

=== Recent Logs ===
$(tail -n 20 "$PROJECT_DIR/logs/apg_billing.log" 2>/dev/null || echo "No logs available")
EOF
    
    echo "$report_file"
}

# Run all checks
run_all_checks() {
    local failures=0
    local checks=(
        "check_service_health"
        "check_database" 
        "check_redis"
        "check_system_resources"
        "check_docker_containers"
        "check_background_tasks"
        "check_api_endpoints"
    )
    
    log_info "Running comprehensive health checks..."
    
    for check in "${checks[@]}"; do
        if ! $check; then
            ((failures++))
        fi
    done
    
    if [ $failures -eq 0 ]; then
        log_success "All health checks passed"
        return 0
    else
        log_error "$failures health check(s) failed"
        return 1
    fi
}

# Continuous monitoring mode
monitor_continuous() {
    local consecutive_failures=0
    
    log_info "Starting continuous monitoring (interval: ${CHECK_INTERVAL}s, alert threshold: $ALERT_THRESHOLD)"
    log_info "Configuration source: $MONITOR_CONFIG_SOURCE"
    
    while true; do
        if run_all_checks >/dev/null 2>&1; then
            consecutive_failures=0
            log_success "Health check cycle completed successfully"
        else
            ((consecutive_failures++))
            log_warning "Health check cycle failed ($consecutive_failures/$ALERT_THRESHOLD)"
            
            if [ $consecutive_failures -ge $ALERT_THRESHOLD ]; then
                local report_file=$(generate_system_report)
                local report_content=$(cat "$report_file")
                
                send_alert "CRITICAL" "APG Billing System health checks failing" "$report_content"
                
                log_error "Alert threshold reached - notification sent"
                consecutive_failures=0  # Reset to avoid spam
            fi
        fi
        
        sleep $CHECK_INTERVAL
    done
}

# Performance test
run_performance_test() {
    log_info "Running performance test..."
    
    local start_time=$(date +%s)
    local total_requests=100
    local concurrent_requests=10
    local successful_requests=0
    
    # Simple load test using curl
    for i in $(seq 1 $concurrent_requests); do
        (
            for j in $(seq 1 $((total_requests / concurrent_requests))); do
                if curl -s -f "$SERVICE_URL$HEALTH_ENDPOINT" >/dev/null 2>&1; then
                    echo "success"
                fi
            done
        ) &
    done
    
    wait
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local requests_per_second=$((total_requests / duration))
    
    log_info "Performance test completed:"
    log_info "  Total requests: $total_requests"
    log_info "  Duration: ${duration}s"
    log_info "  Requests/second: $requests_per_second"
    
    if [ $requests_per_second -lt 10 ]; then
        log_warning "Low performance detected"
        return 1
    else
        log_success "Performance test passed"
        return 0
    fi
}

# Command handling
case "${1:-check}" in
    "check"|"")
        run_all_checks
        ;;
    "monitor")
        monitor_continuous
        ;;
    "health")
        check_service_health
        ;;
    "database")
        check_database
        ;;
    "redis")
        check_redis
        ;;
    "resources")
        check_system_resources
        ;;
    "docker")
        check_docker_containers
        ;;
    "tasks")
        check_background_tasks
        ;;
    "api")
        check_api_endpoints
        ;;
    "performance")
        run_performance_test
        ;;
    "report")
        local report_file=$(generate_system_report)
        cat "$report_file"
        log_info "Full report saved to: $report_file"
        ;;
    "alert-test")
        send_alert "MEDIUM" "Test alert from monitoring script" "This is a test alert to verify notification systems"
        log_info "Test alert sent"
        ;;
    *)
        echo "Usage: $0 {check|monitor|health|database|redis|resources|docker|tasks|api|performance|report|alert-test}"
        echo ""
        echo "Commands:"
        echo "  check        - Run all health checks once"
        echo "  monitor      - Continuous monitoring mode"
        echo "  health       - Check service health endpoint"
        echo "  database     - Check database connectivity"
        echo "  redis        - Check Redis connectivity" 
        echo "  resources    - Check system resources"
        echo "  docker       - Check Docker containers"
        echo "  tasks        - Check background tasks"
        echo "  api          - Check API endpoints"
        echo "  performance  - Run performance test"
        echo "  report       - Generate detailed system report"
        echo "  alert-test   - Send test alert"
        exit 1
        ;;
esac