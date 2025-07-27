#!/bin/bash
#
# APG Multi-language Localization - Production Deployment Script
#
# Automated deployment script for the multi-language localization capability
# Supports Docker, Kubernetes, and cloud platform deployments with validation
#
# Author: Nyimbi Odero
# Company: Datacraft
# Copyright: © 2025 Datacraft. All rights reserved.
#

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Default configuration
DEPLOYMENT_TYPE="kubernetes"
ENVIRONMENT="production"
NAMESPACE="localization"
IMAGE_TAG="latest"
VALIDATE_DEPLOYMENT="true"
BACKUP_BEFORE_DEPLOY="true"
ROLLBACK_ON_FAILURE="true"
WAIT_FOR_READY="true"
VERBOSE="false"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" >&2
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" >&2
}

log_warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

log_verbose() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "${BLUE}[VERBOSE]${NC} $1" >&2
    fi
}

# Help function
show_help() {
    cat << EOF
APG Multi-language Localization - Deployment Script

USAGE:
    $0 [OPTIONS]

OPTIONS:
    -t, --type TYPE         Deployment type: docker|kubernetes|helm (default: kubernetes)
    -e, --environment ENV   Target environment: development|staging|production (default: production)
    -n, --namespace NS      Kubernetes namespace (default: localization)
    -i, --image-tag TAG     Docker image tag (default: latest)
    --no-validate          Skip post-deployment validation
    --no-backup            Skip pre-deployment backup
    --no-rollback          Don't rollback on failure
    --no-wait              Don't wait for services to be ready
    -v, --verbose          Enable verbose output
    -h, --help             Show this help message

EXAMPLES:
    # Deploy to production Kubernetes
    $0 --type kubernetes --environment production --image-tag v1.2.3

    # Deploy to staging with Docker Compose
    $0 --type docker --environment staging --no-backup

    # Deploy with Helm chart
    $0 --type helm --namespace localization-prod --image-tag stable

ENVIRONMENT VARIABLES:
    KUBECONFIG            Path to kubectl configuration file
    DOCKER_REGISTRY       Docker registry URL
    HELM_CHART_REPO       Helm chart repository URL
    API_BASE_URL          Base URL for validation
    DATABASE_URL          Database connection URL for validation
    REDIS_URL             Redis connection URL for validation
    API_KEY               API key for validation

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--type)
                DEPLOYMENT_TYPE="$2"
                shift 2
                ;;
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -i|--image-tag)
                IMAGE_TAG="$2"
                shift 2
                ;;
            --no-validate)
                VALIDATE_DEPLOYMENT="false"
                shift
                ;;
            --no-backup)
                BACKUP_BEFORE_DEPLOY="false"
                shift
                ;;
            --no-rollback)
                ROLLBACK_ON_FAILURE="false"
                shift
                ;;
            --no-wait)
                WAIT_FOR_READY="false"
                shift
                ;;
            -v|--verbose)
                VERBOSE="true"
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Validate prerequisites
validate_prerequisites() {
    log_info "Validating prerequisites..."
    
    case "$DEPLOYMENT_TYPE" in
        docker)
            if ! command -v docker &> /dev/null; then
                log_error "Docker is not installed or not in PATH"
                exit 1
            fi
            if ! command -v docker-compose &> /dev/null; then
                log_error "Docker Compose is not installed or not in PATH"
                exit 1
            fi
            ;;
        kubernetes)
            if ! command -v kubectl &> /dev/null; then
                log_error "kubectl is not installed or not in PATH"
                exit 1
            fi
            if ! kubectl cluster-info &> /dev/null; then
                log_error "Cannot connect to Kubernetes cluster"
                exit 1
            fi
            ;;
        helm)
            if ! command -v helm &> /dev/null; then
                log_error "Helm is not installed or not in PATH"
                exit 1
            fi
            if ! command -v kubectl &> /dev/null; then
                log_error "kubectl is not installed or not in PATH"
                exit 1
            fi
            ;;
        *)
            log_error "Invalid deployment type: $DEPLOYMENT_TYPE"
            exit 1
            ;;
    esac
    
    # Check required files
    case "$DEPLOYMENT_TYPE" in
        docker)
            if [[ ! -f "$PROJECT_ROOT/docker-compose.yml" ]]; then
                log_error "docker-compose.yml not found in project root"
                exit 1
            fi
            ;;
        kubernetes|helm)
            if [[ ! -d "$PROJECT_ROOT/k8s" ]]; then
                log_error "Kubernetes manifests directory (k8s/) not found"
                exit 1
            fi
            ;;
    esac
    
    log_success "Prerequisites validated"
}

# Create backup
create_backup() {
    if [[ "$BACKUP_BEFORE_DEPLOY" != "true" ]]; then
        log_info "Skipping backup (disabled)"
        return 0
    fi
    
    log_info "Creating pre-deployment backup..."
    
    case "$DEPLOYMENT_TYPE" in
        kubernetes|helm)
            # Create namespace backup
            BACKUP_DIR="/tmp/localization-backup-$TIMESTAMP"
            mkdir -p "$BACKUP_DIR"
            
            # Backup current deployments
            if kubectl get namespace "$NAMESPACE" &> /dev/null; then
                kubectl get all -n "$NAMESPACE" -o yaml > "$BACKUP_DIR/current-state.yaml" || true
                kubectl get configmaps -n "$NAMESPACE" -o yaml > "$BACKUP_DIR/configmaps.yaml" || true
                kubectl get secrets -n "$NAMESPACE" -o yaml > "$BACKUP_DIR/secrets.yaml" || true
                kubectl get pvc -n "$NAMESPACE" -o yaml > "$BACKUP_DIR/pvcs.yaml" || true
                
                log_success "Kubernetes backup created at $BACKUP_DIR"
            else
                log_warn "Namespace $NAMESPACE does not exist, skipping backup"
            fi
            ;;
        docker)
            # Backup Docker volumes
            BACKUP_DIR="/tmp/localization-backup-$TIMESTAMP"
            mkdir -p "$BACKUP_DIR"
            
            # Get list of volumes
            cd "$PROJECT_ROOT"
            docker-compose ps -q | xargs docker inspect --format='{{range .Mounts}}{{if eq .Type "volume"}}{{.Name}}{{end}}{{end}}' | sort -u > "$BACKUP_DIR/volumes.txt" || true
            
            log_success "Docker backup metadata created at $BACKUP_DIR"
            ;;
    esac
}

# Build and push images
build_images() {
    log_info "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build main image
    IMAGE_NAME="datacraft/apg-localization:$IMAGE_TAG"
    
    if [[ -n "${DOCKER_REGISTRY:-}" ]]; then
        IMAGE_NAME="$DOCKER_REGISTRY/datacraft/apg-localization:$IMAGE_TAG"
    fi
    
    log_verbose "Building image: $IMAGE_NAME"
    docker build -t "$IMAGE_NAME" .
    
    # Push to registry if specified
    if [[ -n "${DOCKER_REGISTRY:-}" ]]; then
        log_info "Pushing image to registry..."
        docker push "$IMAGE_NAME"
        log_success "Image pushed successfully"
    fi
    
    log_success "Docker images built successfully"
}

# Deploy with Docker Compose
deploy_docker() {
    log_info "Deploying with Docker Compose..."
    
    cd "$PROJECT_ROOT"
    
    # Set environment variables
    export IMAGE_TAG="$IMAGE_TAG"
    export ENVIRONMENT="$ENVIRONMENT"
    
    # Load environment file if exists
    if [[ -f ".env.$ENVIRONMENT" ]]; then
        log_verbose "Loading environment file: .env.$ENVIRONMENT"
        set -a
        source ".env.$ENVIRONMENT"
        set +a
    elif [[ -f ".env" ]]; then
        log_verbose "Loading default environment file: .env"
        set -a
        source ".env"
        set +a
    fi
    
    # Stop existing services
    docker-compose down || true
    
    # Start services
    log_info "Starting services..."
    docker-compose up -d
    
    if [[ "$WAIT_FOR_READY" == "true" ]]; then
        wait_for_docker_services
    fi
    
    log_success "Docker deployment completed"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    # Create namespace if it doesn't exist
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_info "Creating namespace: $NAMESPACE"
        kubectl apply -f "$PROJECT_ROOT/k8s/namespace.yaml"
    fi
    
    # Update image tags in deployments
    if [[ "$IMAGE_TAG" != "latest" ]]; then
        log_verbose "Updating image tags to: $IMAGE_TAG"
        sed -i.bak "s|datacraft/apg-localization:latest|datacraft/apg-localization:$IMAGE_TAG|g" "$PROJECT_ROOT/k8s/deployment.yaml"
    fi
    
    # Apply configurations in order
    log_info "Applying Kubernetes manifests..."
    
    # Apply in dependency order
    kubectl apply -f "$PROJECT_ROOT/k8s/namespace.yaml"
    kubectl apply -f "$PROJECT_ROOT/k8s/configmap.yaml"
    kubectl apply -f "$PROJECT_ROOT/k8s/secrets.yaml"
    kubectl apply -f "$PROJECT_ROOT/k8s/pvc.yaml" || true  # PVC might not exist
    kubectl apply -f "$PROJECT_ROOT/k8s/deployment.yaml"
    kubectl apply -f "$PROJECT_ROOT/k8s/service.yaml"
    kubectl apply -f "$PROJECT_ROOT/k8s/ingress.yaml" || true  # Ingress might not exist
    kubectl apply -f "$PROJECT_ROOT/k8s/hpa.yaml"
    
    # Restore original deployment file
    if [[ -f "$PROJECT_ROOT/k8s/deployment.yaml.bak" ]]; then
        mv "$PROJECT_ROOT/k8s/deployment.yaml.bak" "$PROJECT_ROOT/k8s/deployment.yaml"
    fi
    
    if [[ "$WAIT_FOR_READY" == "true" ]]; then
        wait_for_kubernetes_services
    fi
    
    log_success "Kubernetes deployment completed"
}

# Deploy with Helm
deploy_helm() {
    log_info "Deploying with Helm..."
    
    HELM_RELEASE_NAME="localization"
    HELM_CHART_PATH="$PROJECT_ROOT/helm/localization"
    
    # Check if Helm chart exists
    if [[ ! -d "$HELM_CHART_PATH" ]]; then
        log_error "Helm chart not found at: $HELM_CHART_PATH"
        exit 1
    fi
    
    # Prepare values
    VALUES_FILE="$PROJECT_ROOT/helm/values-$ENVIRONMENT.yaml"
    if [[ ! -f "$VALUES_FILE" ]]; then
        VALUES_FILE="$PROJECT_ROOT/helm/values.yaml"
    fi
    
    # Install or upgrade
    if helm list -n "$NAMESPACE" | grep -q "$HELM_RELEASE_NAME"; then
        log_info "Upgrading Helm release..."
        helm upgrade "$HELM_RELEASE_NAME" "$HELM_CHART_PATH" \
            --namespace "$NAMESPACE" \
            --values "$VALUES_FILE" \
            --set image.tag="$IMAGE_TAG" \
            --set environment="$ENVIRONMENT" \
            --wait --timeout=600s
    else
        log_info "Installing Helm release..."
        helm install "$HELM_RELEASE_NAME" "$HELM_CHART_PATH" \
            --namespace "$NAMESPACE" \
            --create-namespace \
            --values "$VALUES_FILE" \
            --set image.tag="$IMAGE_TAG" \
            --set environment="$ENVIRONMENT" \
            --wait --timeout=600s
    fi
    
    log_success "Helm deployment completed"
}

# Wait for Docker services to be ready
wait_for_docker_services() {
    log_info "Waiting for Docker services to be ready..."
    
    local max_attempts=60
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        log_verbose "Health check attempt $attempt/$max_attempts"
        
        if curl -sf http://localhost:8000/health &> /dev/null; then
            log_success "Services are ready"
            return 0
        fi
        
        sleep 5
        ((attempt++))
    done
    
    log_error "Services failed to become ready within timeout"
    return 1
}

# Wait for Kubernetes services to be ready
wait_for_kubernetes_services() {
    log_info "Waiting for Kubernetes services to be ready..."
    
    # Wait for deployments to be ready
    kubectl wait --for=condition=available --timeout=600s deployment -l app.kubernetes.io/name=multi-language-localization -n "$NAMESPACE"
    
    # Wait for pods to be ready
    kubectl wait --for=condition=ready --timeout=300s pod -l app.kubernetes.io/name=multi-language-localization -n "$NAMESPACE"
    
    log_success "Kubernetes services are ready"
}

# Run post-deployment validation
run_validation() {
    if [[ "$VALIDATE_DEPLOYMENT" != "true" ]]; then
        log_info "Skipping validation (disabled)"
        return 0
    fi
    
    log_info "Running post-deployment validation..."
    
    # Determine API URL based on deployment type
    local api_url
    case "$DEPLOYMENT_TYPE" in
        docker)
            api_url="${API_BASE_URL:-http://localhost:8000}"
            ;;
        kubernetes|helm)
            if [[ -n "${API_BASE_URL:-}" ]]; then
                api_url="$API_BASE_URL"
            else
                # Try to get ingress URL
                local ingress_host
                ingress_host=$(kubectl get ingress -n "$NAMESPACE" -o jsonpath='{.items[0].spec.rules[0].host}' 2>/dev/null || echo "")
                if [[ -n "$ingress_host" ]]; then
                    api_url="https://$ingress_host"
                else
                    # Port forward for validation
                    log_info "Setting up port forward for validation..."
                    kubectl port-forward -n "$NAMESPACE" svc/localization-api 8000:8000 &
                    local port_forward_pid=$!
                    sleep 5
                    api_url="http://localhost:8000"
                fi
            fi
            ;;
    esac
    
    # Run validation script
    if [[ -f "$SCRIPT_DIR/production_validation.py" ]]; then
        log_verbose "Running validation script with API URL: $api_url"
        
        python3 "$SCRIPT_DIR/production_validation.py" \
            --api-url "$api_url" \
            --database-url "${DATABASE_URL:-postgresql://localhost:5432/localization}" \
            --redis-url "${REDIS_URL:-redis://localhost:6379/0}" \
            --api-key "${API_KEY:-test-api-key}" \
            --output "/tmp/validation-report-$TIMESTAMP.json"
        
        local validation_exit_code=$?
        
        # Kill port forward if we started it
        if [[ -n "${port_forward_pid:-}" ]]; then
            kill $port_forward_pid 2>/dev/null || true
        fi
        
        if [[ $validation_exit_code -eq 0 ]]; then
            log_success "Validation passed"
        elif [[ $validation_exit_code -eq 2 ]]; then
            log_warn "Validation passed with warnings"
        else
            log_error "Validation failed"
            return 1
        fi
    else
        log_warn "Validation script not found, skipping validation"
    fi
}

# Rollback deployment
rollback_deployment() {
    if [[ "$ROLLBACK_ON_FAILURE" != "true" ]]; then
        log_info "Rollback disabled, manual intervention required"
        return 1
    fi
    
    log_warn "Rolling back deployment..."
    
    case "$DEPLOYMENT_TYPE" in
        docker)
            cd "$PROJECT_ROOT"
            docker-compose down
            log_info "Docker services stopped"
            ;;
        kubernetes)
            if [[ -f "/tmp/localization-backup-$TIMESTAMP/current-state.yaml" ]]; then
                kubectl apply -f "/tmp/localization-backup-$TIMESTAMP/current-state.yaml" || true
                log_info "Kubernetes state restored from backup"
            else
                kubectl rollout undo deployment -n "$NAMESPACE" -l app.kubernetes.io/name=multi-language-localization || true
                log_info "Kubernetes deployment rolled back"
            fi
            ;;
        helm)
            helm rollback "$HELM_RELEASE_NAME" -n "$NAMESPACE" || true
            log_info "Helm release rolled back"
            ;;
    esac
    
    log_warn "Rollback completed"
}

# Cleanup function
cleanup() {
    log_verbose "Running cleanup..."
    
    # Kill any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    
    # Clean up temporary files
    find /tmp -name "localization-*-$TIMESTAMP*" -type f -mmin +60 -delete 2>/dev/null || true
}

# Main deployment function
main() {
    log_info "Starting APG Multi-language Localization deployment"
    log_info "Configuration:"
    log_info "  Type: $DEPLOYMENT_TYPE"
    log_info "  Environment: $ENVIRONMENT"
    log_info "  Namespace: $NAMESPACE"
    log_info "  Image Tag: $IMAGE_TAG"
    log_info "  Validation: $VALIDATE_DEPLOYMENT"
    log_info "  Backup: $BACKUP_BEFORE_DEPLOY"
    log_info "  Rollback: $ROLLBACK_ON_FAILURE"
    
    # Set up cleanup trap
    trap cleanup EXIT
    
    # Run deployment steps
    validate_prerequisites
    create_backup
    
    # Build images if needed
    if [[ "$DEPLOYMENT_TYPE" == "docker" ]] || [[ -n "${DOCKER_REGISTRY:-}" ]]; then
        build_images
    fi
    
    # Deploy based on type
    case "$DEPLOYMENT_TYPE" in
        docker)
            deploy_docker
            ;;
        kubernetes)
            deploy_kubernetes
            ;;
        helm)
            deploy_helm
            ;;
    esac
    
    # Run validation
    if ! run_validation; then
        log_error "Deployment validation failed"
        rollback_deployment
        exit 1
    fi
    
    log_success "Deployment completed successfully!"
    log_info "Timestamp: $TIMESTAMP"
    
    # Show deployment info
    case "$DEPLOYMENT_TYPE" in
        docker)
            log_info "Docker services status:"
            docker-compose ps
            ;;
        kubernetes|helm)
            log_info "Kubernetes resources in namespace '$NAMESPACE':"
            kubectl get all -n "$NAMESPACE"
            ;;
    esac
}

# Parse arguments and run main function
parse_args "$@"
main