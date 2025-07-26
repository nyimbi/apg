#!/bin/bash
# APG API Service Mesh - Deployment Script
#
# Comprehensive deployment script supporting multiple environments,
# rollback capabilities, health checks, and deployment validation.
#
# © 2025 Datacraft. All rights reserved.
# Author: Nyimbi Odero <nyimbi@gmail.com>

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DOCKER_COMPOSE_FILE="docker-compose.yml"
DOCKER_COMPOSE_PROD_FILE="docker-compose.prod.yml"
K8S_DIR="$PROJECT_ROOT/k8s"

# Default values
ENVIRONMENT="${ENVIRONMENT:-development}"
DEPLOY_TYPE="${DEPLOY_TYPE:-docker}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
NAMESPACE="${NAMESPACE:-apg-service-mesh}"
TIMEOUT="${TIMEOUT:-300}"
HEALTH_CHECK_RETRIES="${HEALTH_CHECK_RETRIES:-30}"
ROLLBACK_ON_FAILURE="${ROLLBACK_ON_FAILURE:-true}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Logging Functions
# =============================================================================

log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log_info() {
    log "${BLUE}INFO:${NC} $*"
}

log_success() {
    log "${GREEN}SUCCESS:${NC} $*"
}

log_warning() {
    log "${YELLOW}WARNING:${NC} $*"
}

log_error() {
    log "${RED}ERROR:${NC} $*"
}

# =============================================================================
# Utility Functions
# =============================================================================

usage() {
    cat << EOF
APG API Service Mesh Deployment Script

Usage: $0 [OPTIONS] COMMAND

Commands:
    deploy          Deploy the service mesh
    rollback        Rollback to previous version
    status          Check deployment status
    logs            View deployment logs
    cleanup         Clean up deployment resources
    validate        Validate deployment configuration

Options:
    -e, --environment ENV       Environment (development|staging|production) [default: development]
    -t, --type TYPE            Deployment type (docker|kubernetes) [default: docker]
    -i, --image-tag TAG        Docker image tag [default: latest]
    -n, --namespace NS         Kubernetes namespace [default: apg-service-mesh]
    -T, --timeout SECONDS      Deployment timeout [default: 300]
    -r, --retries COUNT        Health check retries [default: 30]
    --no-rollback              Disable automatic rollback on failure
    -h, --help                 Show this help message

Examples:
    $0 deploy                                    # Deploy to development
    $0 -e production -t kubernetes deploy        # Deploy to production with k8s
    $0 -i v1.2.0 deploy                        # Deploy specific image version
    $0 rollback                                 # Rollback to previous version
    $0 status                                   # Check deployment status
    $0 cleanup                                  # Clean up resources

Environment Variables:
    ENVIRONMENT                Set deployment environment
    DEPLOY_TYPE               Set deployment type
    IMAGE_TAG                 Set Docker image tag
    NAMESPACE                 Set Kubernetes namespace
    DATABASE_URL              Database connection URL
    REDIS_URL                 Redis connection URL
    SECRET_KEY                Application secret key

EOF
}

check_dependencies() {
    local missing_deps=()

    if [[ "$DEPLOY_TYPE" == "docker" ]]; then
        command -v docker >/dev/null 2>&1 || missing_deps+=("docker")
        command -v docker-compose >/dev/null 2>&1 || missing_deps+=("docker-compose")
    elif [[ "$DEPLOY_TYPE" == "kubernetes" ]]; then
        command -v kubectl >/dev/null 2>&1 || missing_deps+=("kubectl")
        command -v helm >/dev/null 2>&1 || missing_deps+=("helm")
    fi

    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_error "Please install the required tools and try again"
        exit 1
    fi
}

validate_environment() {
    case "$ENVIRONMENT" in
        development|staging|production)
            log_info "Deploying to $ENVIRONMENT environment"
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT"
            log_error "Supported environments: development, staging, production"
            exit 1
            ;;
    esac
}

# =============================================================================
# Docker Deployment Functions
# =============================================================================

docker_deploy() {
    log_info "Starting Docker deployment..."
    
    local compose_file="$DOCKER_COMPOSE_FILE"
    if [[ "$ENVIRONMENT" == "production" ]]; then
        compose_file="$DOCKER_COMPOSE_PROD_FILE"
    fi
    
    # Build and start services
    log_info "Building and starting services..."
    docker-compose -f "$compose_file" build
    docker-compose -f "$compose_file" up -d
    
    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    local retry_count=0
    local max_retries=$HEALTH_CHECK_RETRIES
    
    while [[ $retry_count -lt $max_retries ]]; do
        if docker_health_check; then
            log_success "All services are healthy!"
            return 0
        fi
        
        log_info "Health check failed, retrying in 10 seconds... ($((retry_count + 1))/$max_retries)"
        sleep 10
        retry_count=$((retry_count + 1))
    done
    
    log_error "Health checks failed after $max_retries attempts"
    if [[ "$ROLLBACK_ON_FAILURE" == "true" ]]; then
        docker_rollback
    fi
    return 1
}

docker_health_check() {
    local services=("api-service-mesh" "postgres" "redis")
    
    for service in "${services[@]}"; do
        local health_status
        health_status=$(docker-compose ps -q "$service" | xargs docker inspect --format='{{.State.Health.Status}}' 2>/dev/null || echo "unhealthy")
        
        if [[ "$health_status" != "healthy" ]] && [[ "$health_status" != "starting" ]]; then
            log_warning "Service $service is not healthy (status: $health_status)"
            return 1
        fi
    done
    
    # Test API endpoint
    if ! curl -f -s http://localhost:8000/api/health >/dev/null 2>&1; then
        log_warning "API health endpoint not responding"
        return 1
    fi
    
    return 0
}

docker_rollback() {
    log_warning "Rolling back Docker deployment..."
    
    # Stop current services
    docker-compose down
    
    # Start previous version (if available)
    if docker images | grep -q "apg-service-mesh:previous"; then
        log_info "Starting previous version..."
        IMAGE_TAG="previous" docker-compose up -d
    else
        log_warning "No previous version found for rollback"
    fi
}

docker_status() {
    log_info "Docker deployment status:"
    docker-compose ps
    
    echo
    log_info "Service health status:"
    docker_health_check && log_success "All services healthy" || log_warning "Some services are unhealthy"
}

docker_logs() {
    log_info "Docker deployment logs:"
    docker-compose logs -f --tail=100
}

docker_cleanup() {
    log_info "Cleaning up Docker deployment..."
    
    # Stop and remove containers
    docker-compose down -v
    
    # Remove unused images
    docker image prune -f
    
    # Remove unused volumes
    docker volume prune -f
    
    log_success "Docker cleanup completed"
}

# =============================================================================
# Kubernetes Deployment Functions
# =============================================================================

k8s_deploy() {
    log_info "Starting Kubernetes deployment..."
    
    # Create namespace if it doesn't exist
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply configurations in order
    log_info "Applying Kubernetes configurations..."
    
    kubectl apply -f "$K8S_DIR/namespace.yaml"
    kubectl apply -f "$K8S_DIR/configmap.yaml"
    kubectl apply -f "$K8S_DIR/secret.yaml"
    kubectl apply -f "$K8S_DIR/pvc.yaml"
    kubectl apply -f "$K8S_DIR/service.yaml"
    kubectl apply -f "$K8S_DIR/deployment.yaml"
    kubectl apply -f "$K8S_DIR/ingress.yaml"
    
    # Wait for deployment to be ready
    log_info "Waiting for deployment to be ready..."
    if kubectl rollout status deployment/apg-service-mesh -n "$NAMESPACE" --timeout="${TIMEOUT}s"; then
        log_success "Deployment completed successfully!"
        k8s_health_check
    else
        log_error "Deployment failed or timed out"
        if [[ "$ROLLBACK_ON_FAILURE" == "true" ]]; then
            k8s_rollback
        fi
        return 1
    fi
}

k8s_health_check() {
    log_info "Performing Kubernetes health checks..."
    
    # Check pod status
    local ready_pods
    ready_pods=$(kubectl get pods -n "$NAMESPACE" -l app=apg-service-mesh --field-selector=status.phase=Running --no-headers 2>/dev/null | wc -l)
    local total_pods
    total_pods=$(kubectl get pods -n "$NAMESPACE" -l app=apg-service-mesh --no-headers 2>/dev/null | wc -l)
    
    log_info "Ready pods: $ready_pods/$total_pods"
    
    # Check service endpoint
    local service_ip
    service_ip=$(kubectl get service apg-service-mesh -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}' 2>/dev/null || echo "")
    
    if [[ -n "$service_ip" ]]; then
        log_info "Service available at: $service_ip"
    else
        log_warning "Service not found or not ready"
        return 1
    fi
    
    return 0
}

k8s_rollback() {
    log_warning "Rolling back Kubernetes deployment..."
    
    # Rollback to previous revision
    kubectl rollout undo deployment/apg-service-mesh -n "$NAMESPACE"
    
    # Wait for rollback to complete
    kubectl rollout status deployment/apg-service-mesh -n "$NAMESPACE" --timeout="${TIMEOUT}s"
    
    log_success "Rollback completed"
}

k8s_status() {
    log_info "Kubernetes deployment status:"
    
    echo "Namespace: $NAMESPACE"
    kubectl get all -n "$NAMESPACE"
    
    echo
    log_info "Pod details:"
    kubectl describe pods -n "$NAMESPACE" -l app=apg-service-mesh
    
    echo
    log_info "Events:"
    kubectl get events -n "$NAMESPACE" --sort-by='.lastTimestamp' | tail -10
}

k8s_logs() {
    log_info "Kubernetes deployment logs:"
    kubectl logs -n "$NAMESPACE" -l app=apg-service-mesh -f --tail=100
}

k8s_cleanup() {
    log_info "Cleaning up Kubernetes deployment..."
    
    # Delete all resources in namespace
    kubectl delete all --all -n "$NAMESPACE"
    kubectl delete pvc --all -n "$NAMESPACE"
    kubectl delete configmap --all -n "$NAMESPACE"
    kubectl delete secret --all -n "$NAMESPACE"
    
    # Optionally delete namespace
    read -p "Delete namespace $NAMESPACE? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        kubectl delete namespace "$NAMESPACE"
    fi
    
    log_success "Kubernetes cleanup completed"
}

# =============================================================================
# Main Functions
# =============================================================================

deploy() {
    log_info "Deploying APG API Service Mesh"
    log_info "Environment: $ENVIRONMENT"
    log_info "Deployment Type: $DEPLOY_TYPE"
    log_info "Image Tag: $IMAGE_TAG"
    
    case "$DEPLOY_TYPE" in
        docker)
            docker_deploy
            ;;
        kubernetes)
            k8s_deploy
            ;;
        *)
            log_error "Invalid deployment type: $DEPLOY_TYPE"
            exit 1
            ;;
    esac
}

rollback() {
    log_info "Rolling back APG API Service Mesh deployment"
    
    case "$DEPLOY_TYPE" in
        docker)
            docker_rollback
            ;;
        kubernetes)
            k8s_rollback
            ;;
        *)
            log_error "Invalid deployment type: $DEPLOY_TYPE"
            exit 1
            ;;
    esac
}

status() {
    case "$DEPLOY_TYPE" in
        docker)
            docker_status
            ;;
        kubernetes)
            k8s_status
            ;;
        *)
            log_error "Invalid deployment type: $DEPLOY_TYPE"
            exit 1
            ;;
    esac
}

logs() {
    case "$DEPLOY_TYPE" in
        docker)
            docker_logs
            ;;
        kubernetes)
            k8s_logs
            ;;
        *)
            log_error "Invalid deployment type: $DEPLOY_TYPE"
            exit 1
            ;;
    esac
}

cleanup() {
    read -p "Are you sure you want to cleanup the deployment? This will remove all data! (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Cleanup cancelled"
        exit 0
    fi
    
    case "$DEPLOY_TYPE" in
        docker)
            docker_cleanup
            ;;
        kubernetes)
            k8s_cleanup
            ;;
        *)
            log_error "Invalid deployment type: $DEPLOY_TYPE"
            exit 1
            ;;
    esac
}

validate() {
    log_info "Validating deployment configuration..."
    
    # Check required files
    local required_files=()
    
    if [[ "$DEPLOY_TYPE" == "docker" ]]; then
        required_files+=("$DOCKER_COMPOSE_FILE")
        if [[ "$ENVIRONMENT" == "production" ]]; then
            required_files+=("$DOCKER_COMPOSE_PROD_FILE")
        fi
    elif [[ "$DEPLOY_TYPE" == "kubernetes" ]]; then
        required_files+=(
            "$K8S_DIR/namespace.yaml"
            "$K8S_DIR/deployment.yaml"
            "$K8S_DIR/service.yaml"
        )
    fi
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            log_error "Required file not found: $file"
            exit 1
        fi
    done
    
    log_success "All required files found"
    
    # Validate environment variables
    local required_env_vars=()
    
    if [[ "$ENVIRONMENT" == "production" ]]; then
        required_env_vars+=(
            "DATABASE_URL"
            "REDIS_URL"
            "SECRET_KEY"
        )
    fi
    
    for var in "${required_env_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log_error "Required environment variable not set: $var"
            exit 1
        fi
    done
    
    log_success "Deployment configuration is valid"
}

# =============================================================================
# Argument Parsing
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -t|--type)
            DEPLOY_TYPE="$2"
            shift 2
            ;;
        -i|--image-tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -T|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -r|--retries)
            HEALTH_CHECK_RETRIES="$2"
            shift 2
            ;;
        --no-rollback)
            ROLLBACK_ON_FAILURE="false"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        deploy|rollback|status|logs|cleanup|validate)
            COMMAND="$1"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# =============================================================================
# Main Execution
# =============================================================================

# Check if command was provided
if [[ -z "${COMMAND:-}" ]]; then
    log_error "No command provided"
    usage
    exit 1
fi

# Change to project root directory
cd "$PROJECT_ROOT"

# Validate configuration
check_dependencies
validate_environment

# Execute command
case "$COMMAND" in
    deploy)
        validate
        deploy
        ;;
    rollback)
        rollback
        ;;
    status)
        status
        ;;
    logs)
        logs
        ;;
    cleanup)
        cleanup
        ;;
    validate)
        validate
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        usage
        exit 1
        ;;
esac

log_success "Command '$COMMAND' completed successfully"