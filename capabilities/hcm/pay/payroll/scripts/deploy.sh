#!/bin/bash

# APG Payroll Management - Production Deployment Script
# © 2025 Datacraft. All rights reserved.
# Author: Nyimbi Odero | APG Platform Architect

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
APP_NAME="apg-payroll-management"
VERSION="${1:-v2.0.0}"
ENVIRONMENT="${2:-production}"
REGISTRY="${DOCKER_REGISTRY:-registry.datacraft.co.ke}"

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
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check if kubectl is installed (for Kubernetes deployment)
    if ! command -v kubectl &> /dev/null; then
        log_warning "kubectl is not installed. Kubernetes deployment will be skipped."
    fi
    
    # Check if required environment variables are set
    if [[ "$ENVIRONMENT" == "production" ]]; then
        required_vars=("POSTGRES_PASSWORD" "APG_SECRET_KEY" "SECURITY_PASSWORD_SALT")
        for var in "${required_vars[@]}"; do
            if [[ -z "${!var}" ]]; then
                log_error "Required environment variable $var is not set"
                exit 1
            fi
        done
    fi
    
    log_success "Prerequisites check completed"
}

# Build Docker image
build_image() {
    log_info "Building Docker image for $APP_NAME:$VERSION..."
    
    cd "$PROJECT_DIR"
    
    # Build the production image
    docker build \
        --target production \
        --tag "$APP_NAME:$VERSION" \
        --tag "$APP_NAME:latest" \
        --build-arg VERSION="$VERSION" \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        .
    
    if [[ $? -eq 0 ]]; then
        log_success "Docker image built successfully"
    else
        log_error "Failed to build Docker image"
        exit 1
    fi
}

# Tag and push image to registry
push_image() {
    if [[ -n "$REGISTRY" ]]; then
        log_info "Tagging and pushing image to registry..."
        
        # Tag for registry
        docker tag "$APP_NAME:$VERSION" "$REGISTRY/$APP_NAME:$VERSION"
        docker tag "$APP_NAME:latest" "$REGISTRY/$APP_NAME:latest"
        
        # Push to registry
        docker push "$REGISTRY/$APP_NAME:$VERSION"
        docker push "$REGISTRY/$APP_NAME:latest"
        
        log_success "Image pushed to registry"
    else
        log_info "No registry specified, skipping push"
    fi
}

# Deploy using Docker Compose
deploy_docker_compose() {
    log_info "Deploying using Docker Compose..."
    
    cd "$PROJECT_DIR"
    
    # Create necessary directories
    mkdir -p logs uploads
    
    # Generate .env file for production
    cat > .env << EOF
# APG Payroll Management - Production Environment
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
APG_SECRET_KEY=${APG_SECRET_KEY}
SECURITY_PASSWORD_SALT=${SECURITY_PASSWORD_SALT}
OPENAI_API_KEY=${OPENAI_API_KEY:-}
SENTRY_DSN=${SENTRY_DSN:-}
GRAFANA_PASSWORD=${GRAFANA_PASSWORD:-admin123}
EOF

    # Deploy services
    docker-compose -f docker-compose.yml up -d
    
    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    sleep 30
    
    # Check service health
    if docker-compose -f docker-compose.yml ps | grep -q "unhealthy\|Exit"; then
        log_error "Some services failed to start properly"
        docker-compose -f docker-compose.yml logs
        exit 1
    fi
    
    log_success "Docker Compose deployment completed"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    if command -v kubectl &> /dev/null; then
        log_info "Deploying to Kubernetes..."
        
        # Apply Kubernetes manifests
        kubectl apply -f "$PROJECT_DIR/k8s/"
        
        # Wait for deployment to be ready
        kubectl rollout status deployment/payroll-app -n apg-payroll --timeout=300s
        
        # Get service endpoints
        log_info "Getting service information..."
        kubectl get services -n apg-payroll
        kubectl get ingress -n apg-payroll
        
        log_success "Kubernetes deployment completed"
    else
        log_warning "kubectl not found, skipping Kubernetes deployment"
    fi
}

# Run database migrations
run_migrations() {
    log_info "Running database migrations..."
    
    if docker-compose -f docker-compose.yml ps payroll-app | grep -q "Up"; then
        # Run migrations using Docker Compose
        docker-compose -f docker-compose.yml exec -T payroll-app python -c "
from run import app
with app.app_context():
    from flask_migrate import upgrade
    upgrade()
"
    elif command -v kubectl &> /dev/null && kubectl get deployment payroll-app -n apg-payroll &> /dev/null; then
        # Run migrations using Kubernetes
        kubectl exec -n apg-payroll deployment/payroll-app -- python -c "
from run import app
with app.app_context():
    from flask_migrate import upgrade
    upgrade()
"
    else
        log_warning "No running application found to run migrations"
    fi
    
    log_success "Database migrations completed"
}

# Validate deployment
validate_deployment() {
    log_info "Validating deployment..."
    
    # Check if using Docker Compose
    if docker-compose -f docker-compose.yml ps payroll-app | grep -q "Up"; then
        local health_url="http://localhost:8000/health"
        local api_url="http://localhost:8000/api/v1/payroll"
        
        # Test health endpoint
        if curl -f -s "$health_url" > /dev/null; then
            log_success "Health check passed"
        else
            log_error "Health check failed"
            return 1
        fi
        
        # Test API endpoint
        if curl -f -s "$api_url" > /dev/null; then
            log_success "API endpoint accessible"
        else
            log_warning "API endpoint may require authentication"
        fi
        
    # Check if using Kubernetes
    elif command -v kubectl &> /dev/null && kubectl get deployment payroll-app -n apg-payroll &> /dev/null; then
        # Port forward for testing
        kubectl port-forward -n apg-payroll service/payroll-app-service 8080:80 &
        local port_forward_pid=$!
        sleep 5
        
        # Test endpoints
        if curl -f -s "http://localhost:8080/health" > /dev/null; then
            log_success "Kubernetes health check passed"
        else
            log_error "Kubernetes health check failed"
        fi
        
        # Cleanup port forward
        kill $port_forward_pid 2>/dev/null || true
    fi
    
    log_success "Deployment validation completed"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    
    # Clean up old Docker images
    docker image prune -f
    
    # Remove dangling images
    docker images -f "dangling=true" -q | xargs -r docker rmi
    
    log_success "Cleanup completed"
}

# Main deployment function
main() {
    log_info "Starting APG Payroll Management deployment..."
    log_info "Version: $VERSION"
    log_info "Environment: $ENVIRONMENT"
    
    # Run deployment steps
    check_prerequisites
    build_image
    push_image
    
    # Choose deployment method
    if [[ "${DEPLOY_METHOD:-docker-compose}" == "kubernetes" ]]; then
        deploy_kubernetes
    else
        deploy_docker_compose
    fi
    
    run_migrations
    validate_deployment
    cleanup
    
    log_success "APG Payroll Management deployment completed successfully!"
    log_info "Access the application at:"
    
    if [[ "${DEPLOY_METHOD:-docker-compose}" == "kubernetes" ]]; then
        log_info "  - Kubernetes: https://payroll.apg.datacraft.co.ke"
        log_info "  - Local port-forward: kubectl port-forward -n apg-payroll service/payroll-app-service 8080:80"
    else
        log_info "  - Docker Compose: http://localhost:8000"
        log_info "  - Analytics: http://localhost:3000 (Grafana)"
        log_info "  - Monitoring: http://localhost:9090 (Prometheus)"
    fi
}

# Show usage information
show_usage() {
    echo "Usage: $0 [VERSION] [ENVIRONMENT]"
    echo ""
    echo "Arguments:"
    echo "  VERSION      Docker image version (default: v2.0.0)"
    echo "  ENVIRONMENT  Deployment environment (default: production)"
    echo ""
    echo "Environment Variables:"
    echo "  POSTGRES_PASSWORD      - PostgreSQL password (required for production)"
    echo "  APG_SECRET_KEY        - Application secret key (required for production)"
    echo "  SECURITY_PASSWORD_SALT - Password salt (required for production)"
    echo "  OPENAI_API_KEY        - OpenAI API key (optional)"
    echo "  SENTRY_DSN           - Sentry DSN for error tracking (optional)"
    echo "  DOCKER_REGISTRY      - Docker registry URL (optional)"
    echo "  DEPLOY_METHOD        - Deployment method: docker-compose or kubernetes (default: docker-compose)"
    echo ""
    echo "Examples:"
    echo "  $0                           # Deploy latest version to production"
    echo "  $0 v2.1.0 staging          # Deploy specific version to staging"
    echo "  DEPLOY_METHOD=kubernetes $0  # Deploy to Kubernetes"
}

# Handle command line arguments
case "${1:-}" in
    -h|--help)
        show_usage
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac