#!/bin/bash

# APG Financial Management General Ledger - Deployment Script
# Revolutionary AI-powered General Ledger System
# © 2025 Datacraft. All rights reserved.
# Author: Nyimbi Odero <nyimbi@gmail.com>

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VERSION="${VERSION:-1.0.0}"
ENVIRONMENT="${ENVIRONMENT:-production}"
NAMESPACE="${NAMESPACE:-apg-general-ledger}"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    # Check if kubectl is installed (for Kubernetes deployment)
    if [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]]; then
        if ! command -v kubectl &> /dev/null; then
            print_error "kubectl is not installed. Please install kubectl first."
            exit 1
        fi
        
        # Check if cluster is accessible
        if ! kubectl cluster-info &> /dev/null; then
            print_error "Cannot connect to Kubernetes cluster. Please check your configuration."
            exit 1
        fi
    fi
    
    # Check if required environment variables are set
    if [[ -z "${DATABASE_URL:-}" ]]; then
        print_warning "DATABASE_URL not set. Using default configuration."
    fi
    
    if [[ -z "${REDIS_URL:-}" ]]; then
        print_warning "REDIS_URL not set. Using default configuration."
    fi
    
    print_success "Prerequisites check completed."
}

# Function to build Docker image
build_image() {
    print_status "Building Docker image..."
    
    cd "$PROJECT_ROOT"
    
    # Build the image
    docker build \
        --build-arg BUILD_ENV="$ENVIRONMENT" \
        --build-arg VERSION="$VERSION" \
        -t "apg/general-ledger:$VERSION" \
        -t "apg/general-ledger:latest" \
        .
    
    print_success "Docker image built successfully."
}

# Function to run tests
run_tests() {
    print_status "Running tests..."
    
    # Create test network if it doesn't exist
    docker network create apg-test-network 2>/dev/null || true
    
    # Start test database
    docker run -d \
        --name apg-gl-test-db \
        --network apg-test-network \
        -e POSTGRES_DB=apg_gl_test \
        -e POSTGRES_USER=test_user \
        -e POSTGRES_PASSWORD=test_password \
        postgres:14-alpine
    
    # Wait for database to be ready
    sleep 10
    
    # Run tests
    docker run --rm \
        --network apg-test-network \
        -e DATABASE_URL="postgresql://test_user:test_password@apg-gl-test-db:5432/apg_gl_test" \
        -e ENVIRONMENT=test \
        "apg/general-ledger:$VERSION" \
        python -m pytest tests/ -v --tb=short
    
    # Cleanup
    docker stop apg-gl-test-db
    docker rm apg-gl-test-db
    docker network rm apg-test-network
    
    print_success "Tests completed successfully."
}

# Function to deploy with Docker Compose
deploy_docker_compose() {
    print_status "Deploying with Docker Compose..."
    
    cd "$PROJECT_ROOT"
    
    # Create necessary directories
    mkdir -p backups logs
    
    # Generate environment file if it doesn't exist
    if [[ ! -f .env ]]; then
        cat > .env << EOF
# Database Configuration
DB_NAME=apg_gl
DB_USER=gl_user
DB_PASSWORD=$(openssl rand -base64 32)

# Application Secrets
SECRET_KEY=$(openssl rand -base64 32)
JWT_SECRET=$(openssl rand -base64 32)

# APG Platform Integration
APG_PLATFORM_URL=https://platform.company.com
APG_API_KEY=your-apg-api-key

# AI Configuration
OPENAI_API_KEY=your-openai-api-key
AI_MODEL=gpt-4
AI_CONFIDENCE_THRESHOLD=0.8

# Environment
ENVIRONMENT=production
LOG_LEVEL=INFO

# Monitoring
GRAFANA_PASSWORD=$(openssl rand -base64 16)
EOF
        print_warning "Created .env file with default values. Please update it with your actual configuration."
    fi
    
    # Deploy services
    docker-compose up -d
    
    # Wait for services to be healthy
    print_status "Waiting for services to be healthy..."
    timeout 300 bash -c 'until docker-compose ps | grep -q "Up (healthy)"; do sleep 5; done'
    
    # Run database migrations
    print_status "Running database migrations..."
    docker-compose exec gl-service python scripts/migrate.py
    
    print_success "Docker Compose deployment completed."
    print_status "Application is available at: http://localhost"
    print_status "Grafana dashboard: http://localhost:3000 (admin/admin)"
    print_status "Kibana logs: http://localhost:5601"
}

# Function to deploy to Kubernetes
deploy_kubernetes() {
    print_status "Deploying to Kubernetes..."
    
    cd "$PROJECT_ROOT"
    
    # Create namespace
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply secrets (update with actual values)
    kubectl apply -f k8s/secrets.yaml -n "$NAMESPACE"
    
    # Apply configuration
    kubectl apply -f k8s/deployment.yaml -n "$NAMESPACE"
    
    # Wait for deployment to be ready
    print_status "Waiting for deployment to be ready..."
    kubectl rollout status deployment/general-ledger -n "$NAMESPACE" --timeout=300s
    
    # Run database migrations
    print_status "Running database migrations..."
    kubectl run migrate-job \
        --image="apg/general-ledger:$VERSION" \
        --restart=Never \
        --rm -i \
        --namespace="$NAMESPACE" \
        -- python scripts/migrate.py
    
    # Get service information
    SERVICE_IP=$(kubectl get svc general-ledger -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [[ -z "$SERVICE_IP" ]]; then
        SERVICE_IP=$(kubectl get svc general-ledger -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
        print_status "Application is available at: http://$SERVICE_IP"
        print_warning "Service is using ClusterIP. You may need to set up an ingress or port-forward."
    else
        print_status "Application is available at: http://$SERVICE_IP"
    fi
    
    print_success "Kubernetes deployment completed."
}

# Function to verify deployment
verify_deployment() {
    print_status "Verifying deployment..."
    
    local health_url
    if [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]]; then
        # Port forward for health check
        kubectl port-forward svc/general-ledger 8080:80 -n "$NAMESPACE" &
        local port_forward_pid=$!
        sleep 5
        health_url="http://localhost:8080/health"
    else
        health_url="http://localhost/health"
    fi
    
    # Check health endpoint
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f -s "$health_url" > /dev/null; then
            print_success "Health check passed."
            break
        fi
        
        print_status "Attempt $attempt/$max_attempts: Waiting for service to be ready..."
        sleep 10
        ((attempt++))
    done
    
    if [[ $attempt -gt $max_attempts ]]; then
        print_error "Health check failed after $max_attempts attempts."
        
        # Clean up port forward if used
        if [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]]; then
            kill $port_forward_pid 2>/dev/null || true
        fi
        
        exit 1
    fi
    
    # Clean up port forward if used
    if [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]]; then
        kill $port_forward_pid 2>/dev/null || true
    fi
    
    # Test AI functionality
    print_status "Testing AI functionality..."
    local ai_test_response
    ai_test_response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d '{"description": "Test payment for office rent", "amount": "1000.00"}' \
        "$health_url/../ai/process-natural-language" || echo "AI test failed")
    
    if [[ "$ai_test_response" != "AI test failed" ]]; then
        print_success "AI functionality test passed."
    else
        print_warning "AI functionality test failed. Check configuration."
    fi
    
    print_success "Deployment verification completed."
}

# Function to display usage information
usage() {
    cat << EOF
APG General Ledger Deployment Script

Usage: $0 [OPTIONS] DEPLOYMENT_TYPE

DEPLOYMENT_TYPE:
    docker-compose    Deploy using Docker Compose
    kubernetes       Deploy to Kubernetes cluster

OPTIONS:
    -h, --help       Show this help message
    -v, --version    Set version tag (default: 1.0.0)
    -e, --env        Set environment (default: production)
    -n, --namespace  Set Kubernetes namespace (default: apg-general-ledger)
    --skip-tests     Skip running tests
    --skip-build     Skip building Docker image
    --skip-verify    Skip deployment verification

Examples:
    $0 docker-compose
    $0 kubernetes --version 1.1.0 --env staging
    $0 kubernetes --namespace my-namespace --skip-tests

Environment Variables:
    DATABASE_URL     Database connection string
    REDIS_URL        Redis connection string
    OPENAI_API_KEY   OpenAI API key for AI features
    APG_API_KEY      APG Platform API key

EOF
}

# Parse command line arguments
SKIP_TESTS=false
SKIP_BUILD=false
SKIP_VERIFY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -v|--version)
            VERSION="$2"
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
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --skip-verify)
            SKIP_VERIFY=true
            shift
            ;;
        docker-compose|kubernetes)
            DEPLOYMENT_TYPE="$1"
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate deployment type
if [[ -z "${DEPLOYMENT_TYPE:-}" ]]; then
    print_error "Deployment type is required."
    usage
    exit 1
fi

if [[ "$DEPLOYMENT_TYPE" != "docker-compose" && "$DEPLOYMENT_TYPE" != "kubernetes" ]]; then
    print_error "Invalid deployment type: $DEPLOYMENT_TYPE"
    usage
    exit 1
fi

# Main deployment flow
main() {
    print_status "Starting APG General Ledger deployment..."
    print_status "Version: $VERSION"
    print_status "Environment: $ENVIRONMENT"
    print_status "Deployment Type: $DEPLOYMENT_TYPE"
    
    if [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]]; then
        print_status "Namespace: $NAMESPACE"
    fi
    
    # Check prerequisites
    check_prerequisites
    
    # Build Docker image
    if [[ "$SKIP_BUILD" == "false" ]]; then
        build_image
    else
        print_warning "Skipping Docker image build."
    fi
    
    # Run tests
    if [[ "$SKIP_TESTS" == "false" ]]; then
        run_tests
    else
        print_warning "Skipping tests."
    fi
    
    # Deploy based on type
    case "$DEPLOYMENT_TYPE" in
        docker-compose)
            deploy_docker_compose
            ;;
        kubernetes)
            deploy_kubernetes
            ;;
    esac
    
    # Verify deployment
    if [[ "$SKIP_VERIFY" == "false" ]]; then
        verify_deployment
    else
        print_warning "Skipping deployment verification."
    fi
    
    print_success "🎉 APG General Ledger deployment completed successfully!"
    print_status "The revolutionary AI-powered General Ledger is now running."
}

# Trap to handle script interruption
trap 'print_error "Script interrupted. Cleaning up..."; exit 130' INT

# Run main function
main "$@"