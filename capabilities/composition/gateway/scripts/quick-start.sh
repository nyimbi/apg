#!/bin/bash

# =============================================================================
# APG API Service Mesh - Quick Start Script
# Revolutionary Service Mesh Deployment in Under 5 Minutes
# © 2025 Datacraft. All rights reserved.
# Author: Nyimbi Odero <nyimbi@gmail.com>
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Banner
echo -e "${PURPLE}"
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                🚀 APG API SERVICE MESH - QUICK START 🚀                     ║"
echo "║                                                                              ║"
echo "║           Revolutionary AI-Powered Service Mesh Deployment                  ║"
echo "║                        Ready in Under 5 Minutes!                           ║"
echo "║                                                                              ║"
echo "║  • Natural Language Policies  • Voice Control  • 3D Visualization          ║"
echo "║  • Autonomous Self-Healing   • AI Optimization • Zero Configuration        ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Configuration
APG_MESH_VERSION="2.1.0"
DEPLOYMENT_MODE=""
CLUSTER_NAME=""
NAMESPACE="apg-service-mesh"
ENABLE_AI="true"
ENABLE_VOICE="true"
ENABLE_3D="true"
ENABLE_FEDERATION="false"
DOMAIN_NAME="localhost"

# Functions
print_step() {
    echo -e "${CYAN}[STEP] $1${NC}"
}

print_info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

print_success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

print_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

show_help() {
    echo "APG Service Mesh Quick Start Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --mode MODE           Deployment mode: docker, kubernetes, or local"
    echo "  -c, --cluster NAME        Cluster name (default: auto-generated)"
    echo "  -n, --namespace NAME      Kubernetes namespace (default: apg-service-mesh)"
    echo "  -d, --domain DOMAIN       Domain name (default: localhost)"
    echo "  --no-ai                   Disable AI features"
    echo "  --no-voice                Disable voice control"
    echo "  --no-3d                   Disable 3D visualization"
    echo "  --enable-federation       Enable multi-cluster federation"
    echo "  -h, --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -m docker              # Deploy with Docker Compose"
    echo "  $0 -m kubernetes          # Deploy to Kubernetes"
    echo "  $0 -m local               # Run locally for development"
    echo "  $0 --enable-federation    # Enable multi-cluster capabilities"
}

check_prerequisites() {
    print_step "Checking prerequisites..."
    
    local missing_deps=()
    
    # Check required tools based on deployment mode
    case $DEPLOYMENT_MODE in
        "docker")
            if ! command -v docker &> /dev/null; then
                missing_deps+=("docker")
            fi
            if ! command -v docker-compose &> /dev/null; then
                missing_deps+=("docker-compose")
            fi
            ;;
        "kubernetes")
            if ! command -v kubectl &> /dev/null; then
                missing_deps+=("kubectl")
            fi
            if ! command -v helm &> /dev/null; then
                missing_deps+=("helm")
            fi
            ;;
        "local")
            if ! command -v python3 &> /dev/null; then
                missing_deps+=("python3")
            fi
            if ! command -v pip3 &> /dev/null; then
                missing_deps+=("pip3")
            fi
            ;;
    esac
    
    # Check common dependencies
    if ! command -v curl &> /dev/null; then
        missing_deps+=("curl")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing required dependencies: ${missing_deps[*]}"
        echo "Please install the missing dependencies and try again."
        exit 1
    fi
    
    print_success "All prerequisites satisfied"
}

setup_ollama() {
    print_step "Setting up Ollama AI models..."
    
    # Check if Ollama is running
    if ! curl -s http://localhost:11434/api/version &> /dev/null; then
        print_info "Starting Ollama service..."
        
        case $DEPLOYMENT_MODE in
            "docker")
                print_info "Ollama will be started via Docker Compose"
                ;;
            "kubernetes")
                print_info "Ollama will be deployed to Kubernetes"
                ;;
            "local")
                # Try to start Ollama locally
                if command -v ollama &> /dev/null; then
                    ollama serve &
                    sleep 5
                else
                    print_warning "Ollama not found locally. Please install Ollama from https://ollama.ai"
                    print_info "Continuing without local Ollama - using containerized version"
                fi
                ;;
        esac
    fi
    
    print_success "Ollama setup completed"
}

generate_configuration() {
    print_step "Generating configuration files..."
    
    # Generate cluster ID if not provided
    if [ -z "$CLUSTER_NAME" ]; then
        CLUSTER_NAME="apg-mesh-$(date +%s)"
    fi
    
    # Create configuration directory
    mkdir -p ./config
    
    # Generate main configuration
    cat > ./config/apg-mesh-config.yaml << EOF
# APG Service Mesh Configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: apg-mesh-config
  namespace: ${NAMESPACE}
data:
  cluster_name: "${CLUSTER_NAME}"
  cluster_region: "us-west-2"
  domain_name: "${DOMAIN_NAME}"
  enable_ai_features: "${ENABLE_AI}"
  enable_voice_control: "${ENABLE_VOICE}"
  enable_3d_visualization: "${ENABLE_3D}"
  enable_federation: "${ENABLE_FEDERATION}"
  log_level: "INFO"
  metrics_enabled: "true"
  tracing_enabled: "true"
EOF
    
    # Generate secrets
    DB_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
    REDIS_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
    GRAFANA_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
    JWT_SECRET=$(openssl rand -base64 32)
    
    cat > ./config/secrets.env << EOF
# APG Service Mesh Secrets
DB_PASSWORD=${DB_PASSWORD}
REDIS_PASSWORD=${REDIS_PASSWORD}
GRAFANA_PASSWORD=${GRAFANA_PASSWORD}
JWT_SECRET=${JWT_SECRET}
CLUSTER_ID=${CLUSTER_NAME}
CLUSTER_REGION=us-west-2
DOMAIN_NAME=${DOMAIN_NAME}
EOF
    
    print_success "Configuration files generated"
}

deploy_docker() {
    print_step "Deploying APG Service Mesh with Docker Compose..."
    
    # Copy deployment files
    cp deployment/docker-compose.prod.yml ./docker-compose.yml
    
    # Update environment variables
    export $(cat ./config/secrets.env | xargs)
    
    # Start services
    print_info "Starting APG Service Mesh stack..."
    docker-compose up -d
    
    # Wait for services to be ready
    print_info "Waiting for services to initialize..."
    sleep 30
    
    # Check service health
    print_info "Checking service health..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -sf http://localhost:8080/health &> /dev/null; then
            print_success "APG Service Mesh is ready!"
            break
        fi
        
        print_info "Waiting for services to start... (attempt $attempt/$max_attempts)"
        sleep 10
        ((attempt++))
    done
    
    if [ $attempt -gt $max_attempts ]; then
        print_error "Services failed to start within expected time"
        print_info "Check service logs with: docker-compose logs"
        exit 1
    fi
    
    print_success "Docker deployment completed"
}

deploy_kubernetes() {
    print_step "Deploying APG Service Mesh to Kubernetes..."
    
    # Create namespace
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    # Create secrets
    kubectl create secret generic apg-mesh-secrets \
        --namespace=$NAMESPACE \
        --from-env-file=./config/secrets.env \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply configuration
    kubectl apply -f ./config/apg-mesh-config.yaml
    
    # Deploy APG Service Mesh
    kubectl apply -f deployment/kubernetes/production-deployment.yaml
    
    # Wait for deployment
    print_info "Waiting for deployment to be ready..."
    kubectl wait --namespace=$NAMESPACE \
        --for=condition=available \
        --timeout=600s \
        deployment/apg-service-mesh
    
    # Get service URLs
    print_info "Getting service access information..."
    
    if kubectl get service apg-service-mesh-lb -n $NAMESPACE &> /dev/null; then
        EXTERNAL_IP=$(kubectl get service apg-service-mesh-lb -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
        if [ -n "$EXTERNAL_IP" ]; then
            print_success "External LoadBalancer IP: $EXTERNAL_IP"
        fi
    fi
    
    # Port forward for local access
    print_info "Setting up port forwarding for local access..."
    kubectl port-forward -n $NAMESPACE service/apg-service-mesh 8080:8080 &
    kubectl port-forward -n $NAMESPACE service/grafana 3000:3000 &
    
    print_success "Kubernetes deployment completed"
}

deploy_local() {
    print_step "Setting up APG Service Mesh for local development..."
    
    # Install Python dependencies
    print_info "Installing Python dependencies..."
    pip3 install -r requirements.txt
    
    # Setup local database
    print_info "Setting up local database..."
    python3 -c "
from sqlalchemy import create_engine
from models import Base
engine = create_engine('sqlite:///apg_mesh_local.db')
Base.metadata.create_all(engine)
print('Database initialized')
"
    
    # Start Redis (if available)
    if command -v redis-server &> /dev/null; then
        print_info "Starting Redis server..."
        redis-server --daemonize yes --port 6379
    else
        print_warning "Redis not found. Some features may be limited."
    fi
    
    # Start APG Service Mesh
    print_info "Starting APG Service Mesh..."
    export $(cat ./config/secrets.env | xargs)
    export ENVIRONMENT=development
    export DB_URL=sqlite:///apg_mesh_local.db
    export REDIS_URL=redis://localhost:6379
    
    python3 -m uvicorn api:app --host 0.0.0.0 --port 8080 &
    APG_MESH_PID=$!
    
    # Wait for startup
    sleep 10
    
    print_success "Local development setup completed"
}

setup_ai_models() {
    print_step "Setting up AI models..."
    
    local ollama_url="http://localhost:11434"
    
    # Wait for Ollama to be ready
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -sf $ollama_url/api/version &> /dev/null; then
            break
        fi
        print_info "Waiting for Ollama service... (attempt $attempt/$max_attempts)"
        sleep 10
        ((attempt++))
    done
    
    if [ $attempt -gt $max_attempts ]; then
        print_warning "Ollama service not ready. AI features may be limited."
        return
    fi
    
    # Pull required models
    print_info "Downloading AI models (this may take a few minutes)..."
    
    models=("llama3.2:3b" "codellama:7b" "nomic-embed-text")
    
    for model in "${models[@]}"; do
        print_info "Pulling model: $model"
        if ! curl -sf -X POST $ollama_url/api/pull -d "{\"name\":\"$model\"}" &> /dev/null; then
            print_warning "Failed to pull model: $model"
        fi
    done
    
    print_success "AI models setup completed"
}

show_access_info() {
    print_step "APG Service Mesh is ready! 🎉"
    
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                    🚀 APG SERVICE MESH - ACCESS INFORMATION 🚀              ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    case $DEPLOYMENT_MODE in
        "docker"|"local")
            echo -e "${CYAN}🌐 Main Dashboard:${NC}     http://localhost:8080"
            echo -e "${CYAN}📊 Grafana Monitoring:${NC} http://localhost:3000"
            echo -e "${CYAN}🔍 API Documentation:${NC}  http://localhost:8080/docs"
            ;;
        "kubernetes")
            echo -e "${CYAN}🌐 Main Dashboard:${NC}     http://localhost:8080 (port-forwarded)"
            echo -e "${CYAN}📊 Grafana Monitoring:${NC} http://localhost:3000 (port-forwarded)"
            echo -e "${CYAN}🔍 API Documentation:${NC}  http://localhost:8080/docs"
            
            if [ -n "$EXTERNAL_IP" ]; then
                echo -e "${CYAN}🌍 External Access:${NC}    http://$EXTERNAL_IP"
            fi
            ;;
    esac
    
    echo ""
    echo -e "${YELLOW}🎤 VOICE COMMANDS:${NC}"
    echo "  • \"Show me the 3D topology of all services\""
    echo "  • \"What services are failing right now?\""
    echo "  • \"Scale the user service to 5 replicas\""
    echo "  • \"Create a rate limit policy for the payment API\""
    echo ""
    
    echo -e "${PURPLE}🎮 3D VISUALIZATION:${NC}"
    echo "  • Navigate to the dashboard and click \"3D View\""
    echo "  • Use VR headset for immersive debugging"
    echo "  • Drag and drop services to reconfigure topology"
    echo ""
    
    echo -e "${BLUE}🧠 AI FEATURES:${NC}"
    echo "  • Natural language policies: \"Rate limit payment service to 1000 RPS\""
    echo "  • Autonomous healing: AI automatically fixes issues"
    echo "  • Predictive scaling: AI scales services before demand"
    echo ""
    
    echo -e "${GREEN}📱 QUICK ACTIONS:${NC}"
    echo "  • View logs:    docker-compose logs (Docker) | kubectl logs -n $NAMESPACE (K8s)"
    echo "  • Stop mesh:    docker-compose down (Docker) | kubectl delete -f deployment/"
    echo "  • Scale mesh:   Edit replicas in deployment files"
    echo ""
    
    if [ "$ENABLE_FEDERATION" = "true" ]; then
        echo -e "${CYAN}🌐 FEDERATION:${NC}"
        echo "  • This cluster is federation-ready"
        echo "  • Connect additional clusters with: apg-mesh join-federation"
        echo ""
    fi
    
    echo -e "${GREEN}✨ Welcome to the future of service mesh technology! ✨${NC}"
}

cleanup_on_exit() {
    print_info "Cleaning up..."
    
    case $DEPLOYMENT_MODE in
        "local")
            if [ -n "$APG_MESH_PID" ]; then
                kill $APG_MESH_PID 2>/dev/null || true
            fi
            ;;
    esac
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--mode)
            DEPLOYMENT_MODE="$2"
            shift 2
            ;;
        -c|--cluster)
            CLUSTER_NAME="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -d|--domain)
            DOMAIN_NAME="$2"
            shift 2
            ;;
        --no-ai)
            ENABLE_AI="false"
            shift
            ;;
        --no-voice)
            ENABLE_VOICE="false"
            shift
            ;;
        --no-3d)
            ENABLE_3D="false"
            shift
            ;;
        --enable-federation)
            ENABLE_FEDERATION="true"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Prompt for deployment mode if not specified
if [ -z "$DEPLOYMENT_MODE" ]; then
    echo ""
    echo -e "${CYAN}Select deployment mode:${NC}"
    echo "1) Docker Compose (Recommended for local development)"
    echo "2) Kubernetes (Recommended for production)"
    echo "3) Local Python (Development only)"
    echo ""
    read -p "Enter choice [1-3]: " choice
    
    case $choice in
        1) DEPLOYMENT_MODE="docker" ;;
        2) DEPLOYMENT_MODE="kubernetes" ;;
        3) DEPLOYMENT_MODE="local" ;;
        *) 
            print_error "Invalid choice"
            exit 1
            ;;
    esac
fi

# Validate deployment mode
if [[ ! "$DEPLOYMENT_MODE" =~ ^(docker|kubernetes|local)$ ]]; then
    print_error "Invalid deployment mode: $DEPLOYMENT_MODE"
    print_info "Valid modes: docker, kubernetes, local"
    exit 1
fi

# Set up cleanup on exit
trap cleanup_on_exit EXIT

# Main deployment flow
print_info "Starting APG Service Mesh deployment in $DEPLOYMENT_MODE mode..."

check_prerequisites
generate_configuration
setup_ollama

case $DEPLOYMENT_MODE in
    "docker")
        deploy_docker
        ;;
    "kubernetes") 
        deploy_kubernetes
        ;;
    "local")
        deploy_local
        ;;
esac

# Setup AI models (if enabled)
if [ "$ENABLE_AI" = "true" ]; then
    setup_ai_models
fi

# Show access information
show_access_info

print_success "APG Service Mesh deployment completed successfully! 🎉"

# Keep script running for local mode
if [ "$DEPLOYMENT_MODE" = "local" ]; then
    print_info "Press Ctrl+C to stop the service mesh"
    wait
fi