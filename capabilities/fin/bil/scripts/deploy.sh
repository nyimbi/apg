#!/bin/bash

# APG Billing System Deployment Script
# Usage: ./scripts/deploy.sh [environment] [version]

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT="${1:-production}"
VERSION="${2:-latest}"

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
    
    # Check if Docker is installed and running
    if ! docker --version >/dev/null 2>&1; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check if Docker Compose is available
    if ! docker compose version >/dev/null 2>&1; then
        log_error "Docker Compose is not available"
        exit 1
    fi
    
    # Check if .env file exists
    if [ ! -f "$PROJECT_DIR/.env" ]; then
        log_warning ".env file not found, using .env.example"
        cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
        log_warning "Please update .env file with your configuration before running again"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Load environment variables
load_environment() {
    log_info "Loading environment configuration..."
    
    if [ -f "$PROJECT_DIR/.env" ]; then
        export $(grep -v '^#' "$PROJECT_DIR/.env" | xargs)
        log_success "Environment variables loaded"
    else
        log_error ".env file not found"
        exit 1
    fi
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    cd "$PROJECT_DIR"
    
    # Build with version tag
    docker build -t apg-billing:$VERSION .
    docker tag apg-billing:$VERSION apg-billing:latest
    
    log_success "Docker images built successfully"
}

# Run database migrations
run_migrations() {
    log_info "Running database migrations..."
    
    cd "$PROJECT_DIR"
    
    # Start only the database service
    docker compose up -d postgres
    
    # Wait for database to be ready
    log_info "Waiting for database to be ready..."
    sleep 10
    
    # Run migrations
    docker compose exec postgres psql -U postgres -d apg_billing -f /docker-entrypoint-initdb.d/01-schema.sql || true
    
    log_success "Database migrations completed"
}

# Deploy services
deploy_services() {
    log_info "Deploying services..."
    
    cd "$PROJECT_DIR"
    
    # Deploy with Docker Compose
    docker compose down
    docker compose up -d
    
    log_info "Waiting for services to start..."
    sleep 30
    
    # Health check
    if curl -f http://localhost:5000/billing/health >/dev/null 2>&1; then
        log_success "Services deployed and healthy"
    else
        log_error "Services deployment failed - health check failed"
        docker compose logs
        exit 1
    fi
}

# Backup current deployment
backup_deployment() {
    log_info "Creating backup of current deployment..."
    
    BACKUP_DIR="$PROJECT_DIR/backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # Backup database
    if docker compose ps postgres | grep -q "Up"; then
        docker compose exec postgres pg_dump -U postgres apg_billing > "$BACKUP_DIR/database_backup.sql"
        log_success "Database backup created: $BACKUP_DIR/database_backup.sql"
    fi
    
    # Backup application data
    if [ -d "$PROJECT_DIR/data" ]; then
        cp -r "$PROJECT_DIR/data" "$BACKUP_DIR/"
        log_success "Application data backup created"
    fi
    
    log_success "Backup completed: $BACKUP_DIR"
}

# Rollback deployment
rollback_deployment() {
    local backup_dir="$1"
    
    if [ -z "$backup_dir" ]; then
        log_error "Backup directory not specified for rollback"
        exit 1
    fi
    
    log_warning "Rolling back to: $backup_dir"
    
    # Stop current services
    docker compose down
    
    # Restore database
    if [ -f "$backup_dir/database_backup.sql" ]; then
        docker compose up -d postgres
        sleep 10
        docker compose exec -T postgres psql -U postgres apg_billing < "$backup_dir/database_backup.sql"
    fi
    
    # Restore application data
    if [ -d "$backup_dir/data" ]; then
        rm -rf "$PROJECT_DIR/data"
        cp -r "$backup_dir/data" "$PROJECT_DIR/"
    fi
    
    # Start services
    docker compose up -d
    
    log_success "Rollback completed"
}

# Post-deployment verification
verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check service health
    if ! curl -f http://localhost:5000/billing/health >/dev/null 2>&1; then
        log_error "Health check failed"
        return 1
    fi
    
    # Check database connectivity
    if ! docker compose exec postgres pg_isready -U postgres >/dev/null 2>&1; then
        log_error "Database connectivity check failed"
        return 1
    fi
    
    # Check Redis connectivity
    if ! docker compose exec redis redis-cli ping >/dev/null 2>&1; then
        log_error "Redis connectivity check failed"
        return 1
    fi
    
    log_success "Deployment verification passed"
    return 0
}

# Monitor deployment
monitor_deployment() {
    log_info "Monitoring deployment for 5 minutes..."
    
    for i in {1..10}; do
        if curl -f http://localhost:5000/billing/health >/dev/null 2>&1; then
            log_success "Health check $i/10 passed"
        else
            log_error "Health check $i/10 failed"
            docker compose logs --tail=50
            return 1
        fi
        sleep 30
    done
    
    log_success "Deployment monitoring completed successfully"
}

# Main deployment function
main_deploy() {
    log_info "Starting deployment for environment: $ENVIRONMENT, version: $VERSION"
    
    check_prerequisites
    load_environment
    backup_deployment
    build_images
    run_migrations
    deploy_services
    
    if verify_deployment; then
        monitor_deployment
        log_success "Deployment completed successfully!"
        
        # Display service URLs
        echo ""
        log_info "Service URLs:"
        echo "  Health Check: http://localhost:5000/billing/health"
        echo "  Dashboard:    http://localhost:5000/billing/dashboard"
        echo "  API:          http://localhost:5000/api/v1/billing"
        echo ""
        
        # Display useful commands
        log_info "Useful commands:"
        echo "  View logs:    docker compose logs -f"
        echo "  Stop services: docker compose down"
        echo "  Restart:      docker compose restart"
        echo ""
    else
        log_error "Deployment verification failed"
        
        # Get latest backup for rollback option
        LATEST_BACKUP=$(ls -t "$PROJECT_DIR/backups" | head -n1)
        if [ -n "$LATEST_BACKUP" ]; then
            log_warning "To rollback, run: $0 rollback $PROJECT_DIR/backups/$LATEST_BACKUP"
        fi
        exit 1
    fi
}

# Command handling
case "${1:-deploy}" in
    "deploy"|"")
        main_deploy
        ;;
    "rollback")
        if [ -z "$2" ]; then
            log_error "Usage: $0 rollback <backup_directory>"
            exit 1
        fi
        rollback_deployment "$2"
        ;;
    "verify")
        load_environment
        verify_deployment
        ;;
    "backup")
        load_environment
        backup_deployment
        ;;
    "logs")
        docker compose logs -f
        ;;
    "status")
        docker compose ps
        ;;
    "stop")
        docker compose down
        ;;
    "restart")
        docker compose restart
        ;;
    *)
        echo "Usage: $0 {deploy|rollback|verify|backup|logs|status|stop|restart}"
        echo ""
        echo "Commands:"
        echo "  deploy [env] [version]  - Deploy the application (default: production latest)"
        echo "  rollback <backup_dir>   - Rollback to a previous backup"
        echo "  verify                  - Verify current deployment"
        echo "  backup                  - Create a backup of current deployment"
        echo "  logs                    - Show application logs"
        echo "  status                  - Show service status"
        echo "  stop                    - Stop all services"
        echo "  restart                 - Restart all services"
        exit 1
        ;;
esac