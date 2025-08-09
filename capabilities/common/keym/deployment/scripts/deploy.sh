#!/bin/bash
#
# APG Key Management - Production Deployment Script
# Author: Nyimbi Odero
# Copyright: © 2025 Datacraft
#

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPLOYMENT_DIR="${PROJECT_ROOT}/deployment"

# Default configuration
ENVIRONMENT="${ENVIRONMENT:-production}"
DEPLOY_MODE="${DEPLOY_MODE:-rolling}"  # rolling, blue_green, canary
DRY_RUN="${DRY_RUN:-false}"
SKIP_TESTS="${SKIP_TESTS:-false}"
SKIP_BACKUP="${SKIP_BACKUP:-false}"
FORCE_DEPLOY="${FORCE_DEPLOY:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARN: $1${NC}" >&2
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" >&2
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS: $1${NC}"
}

# Print usage information
usage() {
    cat << EOF
APG Key Management Deployment Script

Usage: $0 [OPTIONS]

OPTIONS:
    -e, --environment ENV       Target environment (default: production)
    -m, --mode MODE            Deployment mode: rolling, blue_green, canary (default: rolling)
    -d, --dry-run              Perform dry run without making changes
    -t, --skip-tests           Skip pre-deployment tests
    -b, --skip-backup          Skip pre-deployment backup
    -f, --force                Force deployment even if checks fail
    -h, --help                 Show this help message

EXAMPLES:
    # Standard production deployment
    $0 -e production

    # Blue-green deployment with dry run
    $0 -e production -m blue_green -d

    # Force deployment skipping tests and backup
    $0 -e production -f -t -b

ENVIRONMENT VARIABLES:
    KEYM_DATABASE_URL          Database connection string
    KEYM_CACHE_URL            Redis cache connection string
    KEYM_ENCRYPTION_KEY       Master encryption key
    KEYM_HSM_PRIMARY_PIN      Primary HSM PIN
    AWS_ACCESS_KEY_ID         AWS access key for multi-cloud
    AZURE_CLIENT_SECRET       Azure client secret
    GCP_PROJECT_ID            Google Cloud project ID

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -m|--mode)
                DEPLOY_MODE="$2"
                shift 2
                ;;
            -d|--dry-run)
                DRY_RUN="true"
                shift
                ;;
            -t|--skip-tests)
                SKIP_TESTS="true"
                shift
                ;;
            -b|--skip-backup)
                SKIP_BACKUP="true"
                shift
                ;;
            -f|--force)
                FORCE_DEPLOY="true"
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Validate environment and prerequisites
validate_environment() {
    log "Validating deployment environment..."
    
    # Check required environment variables
    local required_vars=(
        "KEYM_DATABASE_URL"
        "KEYM_CACHE_URL" 
        "KEYM_ENCRYPTION_KEY"
        "APG_TENANT_ID"
    )
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            error "Required environment variable $var is not set"
            return 1
        fi
    done
    
    # Check deployment mode
    case "$DEPLOY_MODE" in
        rolling|blue_green|canary)
            log "Deployment mode: $DEPLOY_MODE"
            ;;
        *)
            error "Invalid deployment mode: $DEPLOY_MODE"
            return 1
            ;;
    esac
    
    # Check if target environment is accessible
    if ! check_target_environment; then
        error "Cannot access target environment: $ENVIRONMENT"
        return 1
    fi
    
    success "Environment validation completed"
}

# Check target environment accessibility
check_target_environment() {
    log "Checking target environment accessibility..."
    
    # Test database connection
    if ! python3 -c "
import asyncpg
import asyncio
async def test():
    try:
        conn = await asyncpg.connect('${KEYM_DATABASE_URL}')
        await conn.fetchval('SELECT 1')
        await conn.close()
        return True
    except Exception as e:
        print(f'Database connection failed: {e}')
        return False
result = asyncio.run(test())
exit(0 if result else 1)
"; then
        error "Database connection test failed"
        return 1
    fi
    
    # Test cache connection  
    if ! python3 -c "
import redis
try:
    r = redis.from_url('${KEYM_CACHE_URL}')
    r.ping()
    print('Cache connection successful')
except Exception as e:
    print(f'Cache connection failed: {e}')
    exit(1)
"; then
        error "Cache connection test failed"
        return 1
    fi
    
    log "Target environment is accessible"
    return 0
}

# Run pre-deployment tests
run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        warn "Skipping pre-deployment tests"
        return 0
    fi
    
    log "Running pre-deployment tests..."
    
    # Unit tests
    log "Running unit tests..."
    if ! python3 -m pytest "${PROJECT_ROOT}/tests/unit" -v --tb=short; then
        error "Unit tests failed"
        return 1
    fi
    
    # Integration tests
    log "Running integration tests..."
    if ! python3 -m pytest "${PROJECT_ROOT}/tests/integration" -v --tb=short; then
        error "Integration tests failed"
        return 1
    fi
    
    # Security tests
    log "Running security tests..."
    if ! python3 -m pytest "${PROJECT_ROOT}/tests/security" -v --tb=short; then
        error "Security tests failed"
        return 1
    fi
    
    # Configuration validation
    log "Validating configuration..."
    if ! python3 -m keym.config.validate --config "${DEPLOYMENT_DIR}/${ENVIRONMENT}/config.yaml"; then
        error "Configuration validation failed"
        return 1
    fi
    
    success "All pre-deployment tests passed"
}

# Create deployment backup
create_backup() {
    if [[ "$SKIP_BACKUP" == "true" ]]; then
        warn "Skipping pre-deployment backup"
        return 0
    fi
    
    log "Creating pre-deployment backup..."
    
    local backup_dir="/backup/keym/deployments/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Database backup
    log "Backing up database..."
    pg_dump "${KEYM_DATABASE_URL}" | gzip > "${backup_dir}/database.sql.gz"
    
    # Configuration backup
    log "Backing up configuration..."
    cp -r "${DEPLOYMENT_DIR}/${ENVIRONMENT}" "${backup_dir}/config"
    
    # Application state backup
    log "Backing up application state..."
    if [[ -d "/opt/keym/data" ]]; then
        tar -czf "${backup_dir}/app_data.tar.gz" -C "/opt/keym" data
    fi
    
    # Create backup manifest
    cat > "${backup_dir}/manifest.json" << EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "environment": "${ENVIRONMENT}",
  "version": "$(git rev-parse HEAD)",
  "files": [
    "database.sql.gz",
    "config/",
    "app_data.tar.gz"
  ]
}
EOF
    
    success "Backup created: $backup_dir"
    echo "$backup_dir" > /tmp/keym_backup_path
}

# Build and prepare deployment artifacts
build_deployment() {
    log "Building deployment artifacts..."
    
    # Create build directory
    local build_dir="${PROJECT_ROOT}/build/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$build_dir"
    
    # Copy application code
    log "Copying application code..."
    rsync -av --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
          "${PROJECT_ROOT}/keym/" "${build_dir}/keym/"
    
    # Copy deployment configuration
    log "Copying deployment configuration..."
    cp -r "${DEPLOYMENT_DIR}/${ENVIRONMENT}" "${build_dir}/config"
    
    # Install dependencies
    log "Installing dependencies..."
    python3 -m pip install -r "${PROJECT_ROOT}/requirements.txt" --target "${build_dir}/vendor"
    
    # Compile Python files
    log "Compiling Python files..."
    python3 -m compileall "${build_dir}/keym"
    
    # Create deployment package
    log "Creating deployment package..."
    tar -czf "${build_dir}.tar.gz" -C "${build_dir}" .
    
    success "Deployment package created: ${build_dir}.tar.gz"
    echo "${build_dir}.tar.gz" > /tmp/keym_deployment_package
}

# Perform rolling deployment
deploy_rolling() {
    log "Performing rolling deployment..."
    
    local package_path="$(cat /tmp/keym_deployment_package)"
    local deployment_target="/opt/keym"
    
    # Stop services gracefully
    log "Stopping services gracefully..."
    systemctl stop keym || warn "Failed to stop keym service"
    
    # Extract new version
    log "Extracting new version..."
    rm -rf "${deployment_target}.new"
    mkdir -p "${deployment_target}.new"
    tar -xzf "$package_path" -C "${deployment_target}.new"
    
    # Update configuration
    log "Updating configuration..."
    cp "${deployment_target}.new/config/config.yaml" "/etc/keym/config.yaml"
    
    # Run database migrations
    log "Running database migrations..."
    cd "${deployment_target}.new"
    python3 -m alembic upgrade head
    
    # Atomic deployment switch
    log "Switching to new version..."
    if [[ -d "$deployment_target" ]]; then
        mv "$deployment_target" "${deployment_target}.old"
    fi
    mv "${deployment_target}.new" "$deployment_target"
    
    # Start services
    log "Starting services..."
    systemctl start keym
    
    # Wait for service to be ready
    log "Waiting for service to be ready..."
    local retries=30
    while [[ $retries -gt 0 ]]; do
        if curl -f http://localhost:8080/health >/dev/null 2>&1; then
            success "Service is ready"
            break
        fi
        ((retries--))
        sleep 2
    done
    
    if [[ $retries -eq 0 ]]; then
        error "Service failed to start"
        return 1
    fi
    
    # Cleanup old version after successful deployment
    if [[ -d "${deployment_target}.old" ]]; then
        rm -rf "${deployment_target}.old"
    fi
    
    success "Rolling deployment completed"
}

# Perform blue-green deployment
deploy_blue_green() {
    log "Performing blue-green deployment..."
    
    local package_path="$(cat /tmp/keym_deployment_package)"
    local blue_target="/opt/keym-blue"
    local green_target="/opt/keym-green"
    local current_link="/opt/keym"
    
    # Determine current and target environments
    local current_env=""
    local target_env=""
    
    if [[ -L "$current_link" ]]; then
        local current_target="$(readlink "$current_link")"
        if [[ "$current_target" == "$blue_target" ]]; then
            current_env="blue"
            target_env="green"
        else
            current_env="green" 
            target_env="blue"
        fi
    else
        current_env="none"
        target_env="blue"
    fi
    
    log "Current environment: $current_env, Target environment: $target_env"
    
    # Deploy to target environment
    local target_path=""
    if [[ "$target_env" == "blue" ]]; then
        target_path="$blue_target"
    else
        target_path="$green_target"
    fi
    
    log "Deploying to $target_path..."
    rm -rf "$target_path"
    mkdir -p "$target_path"
    tar -xzf "$package_path" -C "$target_path"
    
    # Start target environment
    log "Starting target environment..."
    KEYM_HOME="$target_path" systemctl start "keym-$target_env"
    
    # Health check target environment
    log "Running health checks on target environment..."
    local target_port=""
    if [[ "$target_env" == "blue" ]]; then
        target_port="8080"
    else
        target_port="8081"
    fi
    
    local retries=30
    while [[ $retries -gt 0 ]]; do
        if curl -f "http://localhost:$target_port/health" >/dev/null 2>&1; then
            success "Target environment is healthy"
            break
        fi
        ((retries--))
        sleep 2
    done
    
    if [[ $retries -eq 0 ]]; then
        error "Target environment failed health check"
        systemctl stop "keym-$target_env"
        return 1
    fi
    
    # Switch traffic to target environment
    log "Switching traffic to target environment..."
    ln -sfn "$target_path" "$current_link"
    
    # Update load balancer configuration
    log "Updating load balancer configuration..."
    sed -i "s/:808[0-1]/:$target_port/g" /etc/nginx/sites-available/keym
    nginx -s reload
    
    # Stop old environment
    if [[ "$current_env" != "none" ]]; then
        log "Stopping old environment..."
        systemctl stop "keym-$current_env"
    fi
    
    success "Blue-green deployment completed"
}

# Perform canary deployment
deploy_canary() {
    log "Performing canary deployment..."
    
    local package_path="$(cat /tmp/keym_deployment_package)"
    local canary_target="/opt/keym-canary"
    local production_target="/opt/keym"
    
    # Deploy canary version
    log "Deploying canary version..."
    rm -rf "$canary_target"
    mkdir -p "$canary_target"
    tar -xzf "$package_path" -C "$canary_target"
    
    # Start canary service
    log "Starting canary service..."
    KEYM_HOME="$canary_target" systemctl start keym-canary
    
    # Configure load balancer for canary traffic (5%)
    log "Configuring load balancer for canary traffic..."
    cat > /etc/nginx/conf.d/keym-canary.conf << EOF
upstream keym_canary {
    server 127.0.0.1:8082 weight=5;   # 5% traffic to canary
    server 127.0.0.1:8080 weight=95;  # 95% traffic to production
}
EOF
    nginx -s reload
    
    # Monitor canary for specified duration
    local monitor_duration=300  # 5 minutes
    log "Monitoring canary deployment for $monitor_duration seconds..."
    
    local end_time=$(($(date +%s) + monitor_duration))
    local error_count=0
    local total_requests=0
    
    while [[ $(date +%s) -lt $end_time ]]; do
        # Check canary health
        if ! curl -f http://localhost:8082/health >/dev/null 2>&1; then
            ((error_count++))
        fi
        
        # Get request metrics
        local current_requests=$(curl -s http://localhost:8082/metrics | grep 'keym_requests_total' | awk '{sum += $2} END {print sum}')
        total_requests=${current_requests:-0}
        
        sleep 10
    done
    
    # Calculate error rate
    local error_rate=0
    if [[ $total_requests -gt 0 ]]; then
        error_rate=$(echo "scale=4; $error_count / $total_requests" | bc)
    fi
    
    log "Canary monitoring results: Error rate: $error_rate, Total requests: $total_requests"
    
    # Decide whether to promote or rollback
    local error_threshold="0.01"  # 1% error rate threshold
    if (( $(echo "$error_rate <= $error_threshold" | bc -l) )); then
        log "Canary deployment successful, promoting to production..."
        
        # Stop production service
        systemctl stop keym
        
        # Replace production with canary
        rm -rf "${production_target}.old"
        mv "$production_target" "${production_target}.old"
        mv "$canary_target" "$production_target"
        
        # Start production service
        systemctl start keym
        
        # Remove canary configuration
        rm -f /etc/nginx/conf.d/keym-canary.conf
        nginx -s reload
        
        success "Canary deployment promoted to production"
    else
        error "Canary deployment failed, rolling back..."
        
        # Stop canary service
        systemctl stop keym-canary
        
        # Remove canary files
        rm -rf "$canary_target"
        
        # Remove canary configuration
        rm -f /etc/nginx/conf.d/keym-canary.conf
        nginx -s reload
        
        return 1
    fi
}

# Perform post-deployment verification
verify_deployment() {
    log "Performing post-deployment verification..."
    
    # Health check
    log "Running health check..."
    if ! curl -f http://localhost:8080/health; then
        error "Health check failed"
        return 1
    fi
    
    # API functionality test
    log "Testing API functionality..."
    local test_response=$(curl -s -o /dev/null -w "%{http_code}" \
                         -H "Authorization: Bearer test-token" \
                         http://localhost:8080/keym/api/v1/keys)
    
    if [[ "$test_response" != "200" ]] && [[ "$test_response" != "401" ]]; then
        error "API test failed with response code: $test_response"
        return 1
    fi
    
    # Database connectivity test
    log "Testing database connectivity..."
    if ! python3 -c "
import asyncio
import asyncpg
async def test():
    conn = await asyncpg.connect('${KEYM_DATABASE_URL}')
    result = await conn.fetchval('SELECT COUNT(*) FROM km_keys')
    await conn.close()
    print(f'Database test successful: {result} keys found')
asyncio.run(test())
"; then
        error "Database connectivity test failed"
        return 1
    fi
    
    # Cache connectivity test
    log "Testing cache connectivity..."
    if ! python3 -c "
import redis
r = redis.from_url('${KEYM_CACHE_URL}')
r.set('test_key', 'test_value', ex=60)
value = r.get('test_key')
assert value == b'test_value'
r.delete('test_key')
print('Cache test successful')
"; then
        error "Cache connectivity test failed"
        return 1
    fi
    
    # HSM connectivity test (if enabled)
    if [[ "${KEYM_HSM_ENABLED:-false}" == "true" ]]; then
        log "Testing HSM connectivity..."
        if ! python3 -c "
from keym.hsm_integration import HSMManager
import asyncio
async def test():
    hsm_manager = HSMManager()
    await hsm_manager.initialize()
    status = await hsm_manager.get_hsm_status('primary-hsm')
    print(f'HSM test successful: {status}')
asyncio.run(test())
"; then
            warn "HSM connectivity test failed (non-critical)"
        fi
    fi
    
    success "Post-deployment verification completed"
}

# Cleanup deployment artifacts
cleanup() {
    log "Cleaning up deployment artifacts..."
    
    # Remove build artifacts
    if [[ -f /tmp/keym_deployment_package ]]; then
        local package_path="$(cat /tmp/keym_deployment_package)"
        rm -f "$package_path"
        rm -f /tmp/keym_deployment_package
    fi
    
    # Remove temporary files
    rm -f /tmp/keym_backup_path
    
    # Clean old docker images (if using docker)
    if command -v docker &> /dev/null; then
        docker image prune -f
    fi
    
    success "Cleanup completed"
}

# Rollback deployment
rollback() {
    error "Deployment failed, initiating rollback..."
    
    # Restore from backup
    if [[ -f /tmp/keym_backup_path ]]; then
        local backup_dir="$(cat /tmp/keym_backup_path)"
        log "Restoring from backup: $backup_dir"
        
        # Stop current service
        systemctl stop keym || true
        
        # Restore database
        log "Restoring database..."
        gunzip -c "${backup_dir}/database.sql.gz" | psql "${KEYM_DATABASE_URL}"
        
        # Restore configuration
        log "Restoring configuration..."
        cp -r "${backup_dir}/config/." "${DEPLOYMENT_DIR}/${ENVIRONMENT}/"
        
        # Restore application data
        if [[ -f "${backup_dir}/app_data.tar.gz" ]]; then
            log "Restoring application data..."
            tar -xzf "${backup_dir}/app_data.tar.gz" -C "/opt/keym"
        fi
        
        # Restart service
        systemctl start keym
        
        success "Rollback completed"
    else
        error "No backup found for rollback"
    fi
}

# Send deployment notifications
send_notifications() {
    local status="$1"
    local message="$2"
    
    log "Sending deployment notifications..."
    
    # Slack notification
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        curl -X POST -H 'Content-type: application/json' \
             --data "{\"text\":\"🚀 KEYM Deployment $status: $message\"}" \
             "$SLACK_WEBHOOK_URL" || warn "Failed to send Slack notification"
    fi
    
    # Email notification
    if command -v mail &> /dev/null; then
        echo "$message" | mail -s "KEYM Deployment $status" security-team@datacraft.co.ke || \
            warn "Failed to send email notification"
    fi
    
    # PagerDuty notification (for failures)
    if [[ "$status" == "FAILED" ]] && [[ -n "${PAGERDUTY_SERVICE_KEY:-}" ]]; then
        curl -X POST -H 'Content-Type: application/json' \
             -d "{
                \"service_key\": \"$PAGERDUTY_SERVICE_KEY\",
                \"event_type\": \"trigger\",
                \"description\": \"KEYM Deployment Failed: $message\"
             }" \
             https://events.pagerduty.com/generic/2010-04-15/create_event.json || \
            warn "Failed to send PagerDuty notification"
    fi
}

# Main deployment function
main() {
    local start_time=$(date +%s)
    
    log "Starting APG Key Management deployment..."
    log "Environment: $ENVIRONMENT"
    log "Mode: $DEPLOY_MODE"
    log "Dry run: $DRY_RUN"
    
    # Trap for cleanup on exit
    trap cleanup EXIT
    trap 'rollback; exit 1' ERR
    
    # Validate environment
    if ! validate_environment; then
        send_notifications "FAILED" "Environment validation failed"
        exit 1
    fi
    
    # Run tests
    if ! run_tests; then
        if [[ "$FORCE_DEPLOY" != "true" ]]; then
            send_notifications "FAILED" "Pre-deployment tests failed"
            exit 1
        else
            warn "Tests failed but continuing due to force flag"
        fi
    fi
    
    # Create backup
    if ! create_backup; then
        if [[ "$FORCE_DEPLOY" != "true" ]]; then
            send_notifications "FAILED" "Backup creation failed"
            exit 1
        else
            warn "Backup failed but continuing due to force flag"
        fi
    fi
    
    # Exit if dry run
    if [[ "$DRY_RUN" == "true" ]]; then
        success "Dry run completed successfully"
        exit 0
    fi
    
    # Build deployment
    if ! build_deployment; then
        send_notifications "FAILED" "Deployment build failed"
        exit 1
    fi
    
    # Perform deployment based on mode
    case "$DEPLOY_MODE" in
        rolling)
            if ! deploy_rolling; then
                send_notifications "FAILED" "Rolling deployment failed"
                exit 1
            fi
            ;;
        blue_green)
            if ! deploy_blue_green; then
                send_notifications "FAILED" "Blue-green deployment failed"
                exit 1
            fi
            ;;
        canary)
            if ! deploy_canary; then
                send_notifications "FAILED" "Canary deployment failed"
                exit 1
            fi
            ;;
    esac
    
    # Verify deployment
    if ! verify_deployment; then
        send_notifications "FAILED" "Post-deployment verification failed"
        exit 1
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    success "Deployment completed successfully in ${duration} seconds"
    send_notifications "SUCCESS" "Deployment completed successfully in ${duration} seconds"
}

# Parse arguments and run main function
parse_args "$@"
main