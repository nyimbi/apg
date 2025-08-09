#!/bin/bash

# APG Central Configuration - Production Deployment Script
# Comprehensive deployment automation for Kubernetes and Docker Compose

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOYMENT_TYPE="${DEPLOYMENT_TYPE:-docker-compose}"
ENVIRONMENT="${ENVIRONMENT:-production}"
NAMESPACE="${NAMESPACE:-central-config}"
REGISTRY="${REGISTRY:-}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
SKIP_BUILD="${SKIP_BUILD:-false}"
SKIP_TESTS="${SKIP_TESTS:-false}"
DRY_RUN="${DRY_RUN:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Helper functions
check_dependencies() {
    log "Checking dependencies..."
    
    local deps=("docker" "docker-compose")
    
    if [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]]; then
        deps+=("kubectl" "helm")
    fi
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            error "$dep is required but not installed"
        fi
    done
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
    fi
    
    log "✅ All dependencies satisfied"
}

check_environment() {
    log "Checking environment configuration..."
    
    # Required environment variables
    local required_vars=()
    
    if [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]]; then
        required_vars+=("KUBECONFIG")
    fi
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            error "Required environment variable $var is not set"
        fi
    done
    
    # Check environment files
    local env_files=(".env" ".env.${ENVIRONMENT}")
    for env_file in "${env_files[@]}"; do
        if [[ -f "$PROJECT_ROOT/$env_file" ]]; then
            info "Loading environment from $env_file"
            source "$PROJECT_ROOT/$env_file"
        fi
    done
    
    log "✅ Environment configuration checked"
}

run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        warn "Skipping tests (SKIP_TESTS=true)"
        return
    fi
    
    log "Running test suite..."
    
    cd "$PROJECT_ROOT"
    
    # Unit tests
    log "Running unit tests..."
    if [[ -f "pytest.ini" ]] || [[ -f "pyproject.toml" ]]; then
        python -m pytest tests/ -v --tb=short --cov=. --cov-report=term-missing
    else
        warn "No pytest configuration found, skipping unit tests"
    fi
    
    # Type checking
    if command -v pyright &> /dev/null; then
        log "Running type checks..."
        pyright .
    elif command -v mypy &> /dev/null; then
        log "Running type checks with mypy..."
        mypy .
    else
        warn "No type checker available, skipping type checks"
    fi
    
    # Security scanning
    if command -v bandit &> /dev/null; then
        log "Running security scan..."
        bandit -r . -f json -o security-report.json || warn "Security scan found issues"
    fi
    
    log "✅ Tests completed successfully"
}

build_images() {
    if [[ "$SKIP_BUILD" == "true" ]]; then
        warn "Skipping build (SKIP_BUILD=true)"
        return
    fi
    
    log "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build API image
    log "Building API image..."
    docker build \
        --target production \
        --tag "central-config-api:${IMAGE_TAG}" \
        --tag "central-config-api:latest" \
        --file Dockerfile \
        .
    
    # Build Web image
    log "Building Web interface image..."
    docker build \
        --target production \
        --tag "central-config-web:${IMAGE_TAG}" \
        --tag "central-config-web:latest" \
        --file Dockerfile.web \
        .
    
    # Tag for registry if specified
    if [[ -n "$REGISTRY" ]]; then
        log "Tagging images for registry $REGISTRY..."
        docker tag "central-config-api:${IMAGE_TAG}" "${REGISTRY}/central-config-api:${IMAGE_TAG}"
        docker tag "central-config-web:${IMAGE_TAG}" "${REGISTRY}/central-config-web:${IMAGE_TAG}"
        
        # Push to registry
        log "Pushing images to registry..."
        docker push "${REGISTRY}/central-config-api:${IMAGE_TAG}"
        docker push "${REGISTRY}/central-config-web:${IMAGE_TAG}"
    fi
    
    log "✅ Images built successfully"
}

setup_ollama_models() {
    log "Setting up Ollama AI models..."
    
    # Start Ollama container if not running
    if ! docker ps | grep -q ollama; then
        log "Starting Ollama container..."
        docker run -d \
            --name ollama-setup \
            --gpus all \
            -p 11434:11434 \
            -v ollama_data:/root/.ollama \
            ollama/ollama:latest
        
        # Wait for Ollama to be ready
        sleep 30
    fi
    
    # Pull required models
    local models=("llama3.2:3b" "codellama:7b" "nomic-embed-text")
    
    for model in "${models[@]}"; do
        log "Pulling model: $model"
        docker exec ollama-setup ollama pull "$model" || warn "Failed to pull $model"
    done
    
    log "✅ Ollama models setup completed"
}

deploy_docker_compose() {
    log "Deploying with Docker Compose..."
    
    cd "$PROJECT_ROOT"
    
    # Generate environment file
    cat > .env.deploy <<EOF
# Generated deployment configuration
ENVIRONMENT=${ENVIRONMENT}
IMAGE_TAG=${IMAGE_TAG}
POSTGRES_PASSWORD=$(openssl rand -base64 32)
REDIS_PASSWORD=$(openssl rand -base64 32)
JWT_SECRET_KEY=$(openssl rand -base64 64)
ENCRYPTION_KEY=$(openssl rand -base64 32)
EOF
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "DRY RUN: Would execute docker-compose deployment"
        docker-compose --env-file .env.deploy config
        return
    fi
    
    # Pull external images
    log "Pulling external images..."
    docker-compose --env-file .env.deploy pull postgres redis ollama prometheus grafana
    
    # Deploy services
    log "Starting services..."
    docker-compose --env-file .env.deploy up -d
    
    # Wait for services to be healthy
    log "Waiting for services to be healthy..."
    local max_attempts=30
    local attempt=0
    
    while [[ $attempt -lt $max_attempts ]]; do
        if docker-compose --env-file .env.deploy ps | grep -q "Up (healthy)"; then
            log "✅ Services are healthy"
            break
        fi
        
        ((attempt++))
        info "Waiting for services... (attempt $attempt/$max_attempts)"
        sleep 10
    done
    
    if [[ $attempt -eq $max_attempts ]]; then
        error "Services failed to become healthy"
    fi
    
    # Run database migrations
    log "Running database migrations..."
    docker-compose --env-file .env.deploy exec -T cc_api python -m alembic upgrade head
    
    log "✅ Docker Compose deployment completed"
}

deploy_kubernetes() {
    log "Deploying to Kubernetes..."
    
    cd "$PROJECT_ROOT"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "DRY RUN: Would execute Kubernetes deployment"
        kubectl apply --dry-run=client -k k8s/
        return
    fi
    
    # Create namespace
    log "Creating namespace..."
    kubectl apply -f k8s/namespace.yaml
    
    # Apply ConfigMaps and Secrets
    log "Applying configuration..."
    kubectl create secret generic central-config-secrets \
        --namespace="$NAMESPACE" \
        --from-literal=database-url="postgresql+asyncpg://cc_admin:$(openssl rand -base64 32)@postgres-service:5432/central_config" \
        --from-literal=redis-url="redis://:$(openssl rand -base64 32)@redis-service:6379/0" \
        --from-literal=secret-key="$(openssl rand -base64 64)" \
        --from-literal=jwt-secret-key="$(openssl rand -base64 64)" \
        --from-literal=web-secret-key="$(openssl rand -base64 64)" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy PostgreSQL
    log "Deploying PostgreSQL..."
    kubectl apply -f k8s/postgres/
    
    # Deploy Redis
    log "Deploying Redis..."
    kubectl apply -f k8s/redis/
    
    # Deploy Ollama
    log "Deploying Ollama..."
    kubectl apply -f k8s/ollama/
    
    # Wait for dependencies
    log "Waiting for dependencies..."
    kubectl wait --for=condition=ready pod -l app=postgres --namespace="$NAMESPACE" --timeout=300s
    kubectl wait --for=condition=ready pod -l app=redis --namespace="$NAMESPACE" --timeout=300s
    kubectl wait --for=condition=ready pod -l app=ollama --namespace="$NAMESPACE" --timeout=600s
    
    # Deploy application
    log "Deploying application..."
    kubectl apply -f k8s/deployment.yaml
    kubectl apply -f k8s/service.yaml
    kubectl apply -f k8s/ingress.yaml
    
    # Deploy monitoring
    log "Deploying monitoring stack..."
    kubectl apply -f k8s/monitoring/
    
    # Wait for deployment
    log "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available deployment/central-config-api --namespace="$NAMESPACE" --timeout=600s
    kubectl wait --for=condition=available deployment/central-config-web --namespace="$NAMESPACE" --timeout=300s
    
    # Run database migrations
    log "Running database migrations..."
    kubectl exec -n "$NAMESPACE" deployment/central-config-api -- python -m alembic upgrade head
    
    # Get service URLs
    log "Getting service information..."
    kubectl get ingress -n "$NAMESPACE"
    kubectl get services -n "$NAMESPACE"
    
    log "✅ Kubernetes deployment completed"
}

run_post_deployment_tests() {
    log "Running post-deployment tests..."
    
    local base_url
    if [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]]; then
        base_url="http://$(kubectl get ingress central-config-ingress -n "$NAMESPACE" -o jsonpath='{.spec.rules[0].host}')"
    else
        base_url="http://localhost:8000"
    fi
    
    # Health check
    log "Testing health endpoint..."
    local max_attempts=20
    local attempt=0
    
    while [[ $attempt -lt $max_attempts ]]; do
        if curl -f -s "$base_url/health" > /dev/null; then
            log "✅ Health check passed"
            break
        fi
        
        ((attempt++))
        info "Waiting for health check... (attempt $attempt/$max_attempts)"
        sleep 15
    done
    
    if [[ $attempt -eq $max_attempts ]]; then
        error "Health check failed after deployment"
    fi
    
    # API functionality test
    log "Testing API functionality..."
    local health_response
    health_response=$(curl -s "$base_url/health" | jq -r '.status' 2>/dev/null || echo "unknown")
    
    if [[ "$health_response" == "healthy" ]]; then
        log "✅ API is responding correctly"
    else
        warn "API health status: $health_response"
    fi
    
    # Load test (optional)
    if [[ -f "$PROJECT_ROOT/tests/load/api_load_test.js" ]] && command -v k6 &> /dev/null; then
        log "Running smoke load test..."
        k6 run --env BASE_URL="$base_url" --env SCENARIO=smoke "$PROJECT_ROOT/tests/load/api_load_test.js"
    fi
    
    log "✅ Post-deployment tests completed"
}

cleanup() {
    log "Cleaning up temporary files..."
    rm -f "$PROJECT_ROOT/.env.deploy"
    rm -f "$PROJECT_ROOT/security-report.json"
}

show_deployment_info() {
    log "=== Deployment Information ==="
    echo "Environment: $ENVIRONMENT"
    echo "Deployment Type: $DEPLOYMENT_TYPE"
    echo "Image Tag: $IMAGE_TAG"
    echo "Namespace: $NAMESPACE"
    
    if [[ "$DEPLOYMENT_TYPE" == "docker-compose" ]]; then
        echo ""
        echo "Services:"
        docker-compose ps --format table
        echo ""
        echo "Access URLs:"
        echo "  API: http://localhost:8000"
        echo "  Web UI: http://localhost:8080"
        echo "  Prometheus: http://localhost:9090"
        echo "  Grafana: http://localhost:3000"
    elif [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]]; then
        echo ""
        echo "Kubernetes Resources:"
        kubectl get all -n "$NAMESPACE"
        echo ""
        echo "Ingress:"
        kubectl get ingress -n "$NAMESPACE"
    fi
    
    log "=== Deployment Complete ==="
}

# Main execution
main() {
    log "🚀 Starting APG Central Configuration Deployment"
    log "Deployment Type: $DEPLOYMENT_TYPE"
    log "Environment: $ENVIRONMENT"
    
    # Trap for cleanup
    trap cleanup EXIT
    
    # Execute deployment steps
    check_dependencies
    check_environment
    run_tests
    build_images
    setup_ollama_models
    
    case "$DEPLOYMENT_TYPE" in
        "docker-compose")
            deploy_docker_compose
            ;;
        "kubernetes"|"k8s")
            deploy_kubernetes
            ;;
        *)
            error "Unknown deployment type: $DEPLOYMENT_TYPE"
            ;;
    esac
    
    run_post_deployment_tests
    show_deployment_info
    
    log "🎉 Deployment completed successfully!"
}

# Script usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

APG Central Configuration Deployment Script

OPTIONS:
    -t, --type TYPE         Deployment type (docker-compose|kubernetes) [default: docker-compose]
    -e, --env ENV           Environment (development|staging|production) [default: production]
    -n, --namespace NS      Kubernetes namespace [default: central-config]
    -r, --registry REG      Docker registry URL
    -i, --image-tag TAG     Docker image tag [default: latest]
    --skip-build           Skip Docker image building
    --skip-tests           Skip running tests
    --dry-run              Show what would be deployed without executing
    -h, --help             Show this help message

EXAMPLES:
    $0                                          # Deploy with Docker Compose
    $0 -t kubernetes -e production             # Deploy to Kubernetes
    $0 --skip-tests --dry-run                  # Show deployment plan
    $0 -r registry.example.com -i v1.2.3      # Deploy specific version to registry

ENVIRONMENT VARIABLES:
    DEPLOYMENT_TYPE         Same as --type
    ENVIRONMENT            Same as --env
    NAMESPACE              Same as --namespace
    REGISTRY               Same as --registry
    IMAGE_TAG              Same as --image-tag
    SKIP_BUILD             Skip building (true|false)
    SKIP_TESTS             Skip tests (true|false)
    DRY_RUN                Dry run mode (true|false)

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            DEPLOYMENT_TYPE="$2"
            shift 2
            ;;
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -i|--image-tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD="true"
            shift
            ;;
        --skip-tests)
            SKIP_TESTS="true"
            shift
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# Run main function
main "$@"