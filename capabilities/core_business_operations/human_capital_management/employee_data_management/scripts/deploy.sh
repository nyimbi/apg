#!/bin/bash
# APG Employee Data Management - Production Deployment Script

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT=${1:-production}
BRANCH=${2:-main}
BACKUP_ENABLED=${3:-true}
HEALTH_CHECK_RETRIES=${4:-10}
ROLLBACK_ON_FAILURE=${5:-true}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if running as correct user
    if [ "$USER" != "apg" ]; then
        log_error "This script should be run as the 'apg' user"
        exit 1
    fi
    
    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        log_error "Docker is not running or not accessible"
        exit 1
    fi
    
    # Check if required environment files exist
    if [ ! -f "$PROJECT_DIR/.env.$ENVIRONMENT" ]; then
        log_error "Environment file .env.$ENVIRONMENT not found"
        exit 1
    fi
    
    # Check disk space
    AVAILABLE_SPACE=$(df "$PROJECT_DIR" | awk 'NR==2 {print $4}')
    if [ "$AVAILABLE_SPACE" -lt 5242880 ]; then  # 5GB in KB
        log_error "Insufficient disk space. At least 5GB required"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Create backup
create_backup() {
    if [ "$BACKUP_ENABLED" = "true" ]; then
        log_info "Creating backup..."
        
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        BACKUP_DIR="$PROJECT_DIR/backups/deployment_$TIMESTAMP"
        mkdir -p "$BACKUP_DIR"
        
        # Backup database
        if docker-compose -f docker-compose.prod.yml ps postgres | grep -q "Up"; then
            log_info "Backing up database..."
            docker-compose -f docker-compose.prod.yml exec -T postgres pg_dump -U apg_user apg_hr | gzip > "$BACKUP_DIR/database_backup.sql.gz"
            
            if [ $? -eq 0 ]; then
                log_success "Database backup created: $BACKUP_DIR/database_backup.sql.gz"
            else
                log_error "Database backup failed"
                exit 1
            fi
        else
            log_warning "PostgreSQL container not running, skipping database backup"
        fi
        
        # Backup configuration
        log_info "Backing up configuration..."
        tar -czf "$BACKUP_DIR/config_backup.tar.gz" \
            .env.$ENVIRONMENT \
            docker-compose.prod.yml \
            nginx/ \
            monitoring/ \
            2>/dev/null || true
        
        # Store backup location
        echo "$BACKUP_DIR" > "$PROJECT_DIR/.last_backup"
        
        log_success "Backup completed: $BACKUP_DIR"
    else
        log_info "Backup disabled, skipping..."
    fi
}

# Pull latest code
update_code() {
    log_info "Updating code..."
    
    cd "$PROJECT_DIR"
    
    # Stash any local changes
    git stash push -u -m "Pre-deployment stash $(date +%Y%m%d_%H%M%S)" || true
    
    # Fetch and checkout
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
    
    log_success "Code updated to latest $BRANCH"
}

# Build and deploy containers
deploy_containers() {
    log_info "Building and deploying containers..."
    
    cd "$PROJECT_DIR"
    
    # Copy environment file
    cp ".env.$ENVIRONMENT" .env
    
    # Build new images
    log_info "Building Docker images..."
    docker-compose -f docker-compose.prod.yml build --no-cache app
    
    if [ $? -ne 0 ]; then
        log_error "Docker build failed"
        exit 1
    fi
    
    # Run database migrations
    log_info "Running database migrations..."
    docker-compose -f docker-compose.prod.yml run --rm app alembic upgrade head
    
    if [ $? -ne 0 ]; then
        log_error "Database migration failed"
        exit 1
    fi
    
    # Deploy services
    log_info "Deploying services..."
    docker-compose -f docker-compose.prod.yml up -d
    
    if [ $? -ne 0 ]; then
        log_error "Container deployment failed"
        exit 1
    fi
    
    log_success "Containers deployed successfully"
}

# Health check
perform_health_check() {
    log_info "Performing health check..."
    
    local retries=0
    local max_retries=$HEALTH_CHECK_RETRIES
    local health_url="http://localhost:8000/api/v1/health"
    
    while [ $retries -lt $max_retries ]; do
        log_info "Health check attempt $((retries + 1))/$max_retries"
        
        if curl -f -s "$health_url" > /dev/null; then
            log_success "Health check passed"
            return 0
        fi
        
        retries=$((retries + 1))
        if [ $retries -lt $max_retries ]; then
            log_info "Waiting 30 seconds before next attempt..."
            sleep 30
        fi
    done
    
    log_error "Health check failed after $max_retries attempts"
    return 1
}

# Rollback deployment
rollback_deployment() {
    if [ "$ROLLBACK_ON_FAILURE" = "true" ] && [ -f "$PROJECT_DIR/.last_backup" ]; then
        log_warning "Rolling back deployment..."
        
        BACKUP_DIR=$(cat "$PROJECT_DIR/.last_backup")
        
        if [ -d "$BACKUP_DIR" ]; then
            # Stop current containers
            docker-compose -f docker-compose.prod.yml down
            
            # Restore configuration
            if [ -f "$BACKUP_DIR/config_backup.tar.gz" ]; then
                tar -xzf "$BACKUP_DIR/config_backup.tar.gz" -C "$PROJECT_DIR"
            fi
            
            # Restore database
            if [ -f "$BACKUP_DIR/database_backup.sql.gz" ]; then
                log_info "Restoring database..."
                docker-compose -f docker-compose.prod.yml up -d postgres
                sleep 30
                gunzip -c "$BACKUP_DIR/database_backup.sql.gz" | docker-compose -f docker-compose.prod.yml exec -T postgres psql -U apg_user apg_hr
            fi
            
            # Restart services
            docker-compose -f docker-compose.prod.yml up -d
            
            log_success "Rollback completed"
        else
            log_error "Backup directory not found, cannot rollback"
        fi
    else
        log_warning "Rollback disabled or no backup available"
    fi
}

# Clean up old images and containers
cleanup() {
    log_info "Cleaning up old Docker resources..."
    
    # Remove old images
    docker image prune -f
    
    # Remove old containers
    docker container prune -f
    
    # Clean up old backups (keep last 5)
    if [ -d "$PROJECT_DIR/backups" ]; then
        find "$PROJECT_DIR/backups" -name "deployment_*" -type d | sort -r | tail -n +6 | xargs rm -rf
    fi
    
    log_success "Cleanup completed"
}

# Send deployment notification
send_notification() {
    local status=$1
    local message=$2
    
    if [ -n "$SLACK_WEBHOOK_URL" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"APG Employee Management Deployment [$ENVIRONMENT]: $status - $message\"}" \
            "$SLACK_WEBHOOK_URL" 2>/dev/null || true
    fi
    
    if [ -n "$TEAMS_WEBHOOK_URL" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"APG Employee Management Deployment [$ENVIRONMENT]: $status - $message\"}" \
            "$TEAMS_WEBHOOK_URL" 2>/dev/null || true
    fi
}

# Main deployment function
main() {
    local start_time=$(date +%s)
    
    log_info "Starting APG Employee Management deployment to $ENVIRONMENT environment"
    log_info "Branch: $BRANCH"
    log_info "Backup enabled: $BACKUP_ENABLED"
    log_info "Rollback on failure: $ROLLBACK_ON_FAILURE"
    
    # Load environment variables
    if [ -f "$PROJECT_DIR/.env.$ENVIRONMENT" ]; then
        set -a
        source "$PROJECT_DIR/.env.$ENVIRONMENT"
        set +a
    fi
    
    # Deployment steps
    check_prerequisites
    create_backup
    update_code
    deploy_containers
    
    # Health check and potential rollback
    if perform_health_check; then
        cleanup
        
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        log_success "Deployment completed successfully in ${duration}s"
        send_notification "SUCCESS" "Deployment completed in ${duration}s"
        
        # Display service status
        log_info "Service status:"
        docker-compose -f docker-compose.prod.yml ps
        
        # Display URLs
        echo ""
        log_info "Service URLs:"
        echo "  API: https://api.company.com"
        echo "  Health Check: https://api.company.com/api/v1/health"
        echo "  API Docs: https://api.company.com/api/v1/docs"
        echo "  Grafana: http://localhost:3000"
        echo "  Prometheus: http://localhost:9090"
        
    else
        log_error "Deployment failed health check"
        send_notification "FAILED" "Health check failed"
        
        rollback_deployment
        exit 1
    fi
}

# Trap errors and cleanup
trap 'log_error "Deployment failed due to error on line $LINENO"' ERR

# Run main function
main "$@"