#!/bin/bash
#
# APG System Health Management (HLTH) - Production Deployment Script
# Copyright © 2025 Datacraft - www.datacraft.co.ke
# Author: Nyimbi Odero <nyimbi@gmail.com>
#
# This script deploys the HLTH capability to a Kubernetes cluster
#

set -euo pipefail

# Configuration
NAMESPACE="hlth"
ENVIRONMENT="${ENVIRONMENT:-production}"
KUBECTL_TIMEOUT="${KUBECTL_TIMEOUT:-300s}"
DRY_RUN="${DRY_RUN:-false}"

# Color codes for output
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
    
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    if ! command -v helm &> /dev/null; then
        log_warning "helm is not installed - some features may not work"
    fi
    
    # Check kubectl connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "kubectl cannot connect to cluster"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Create namespace if it doesn't exist
create_namespace() {
    log_info "Creating namespace: $NAMESPACE"
    
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_warning "Namespace $NAMESPACE already exists"
    else
        if [ "$DRY_RUN" = "true" ]; then
            log_info "[DRY RUN] Would create namespace: $NAMESPACE"
        else
            kubectl apply -f ../kubernetes/namespace.yaml
            log_success "Namespace $NAMESPACE created"
        fi
    fi
}

# Deploy secrets
deploy_secrets() {
    log_info "Deploying secrets..."
    
    # Check if secrets already exist
    if kubectl get secret hlth-secrets -n "$NAMESPACE" &> /dev/null; then
        log_warning "Secrets already exist. Skipping secret deployment."
        log_warning "If you need to update secrets, delete them first:"
        log_warning "kubectl delete secret hlth-secrets hlth-notification-secrets hlth-integration-secrets -n $NAMESPACE"
        return
    fi
    
    if [ "$DRY_RUN" = "true" ]; then
        log_info "[DRY RUN] Would deploy secrets"
    else
        kubectl apply -f ../kubernetes/secrets.yaml
        log_success "Secrets deployed"
    fi
}

# Deploy ConfigMaps
deploy_configmaps() {
    log_info "Deploying ConfigMaps..."
    
    if [ "$DRY_RUN" = "true" ]; then
        log_info "[DRY RUN] Would deploy ConfigMaps"
    else
        kubectl apply -f ../kubernetes/configmap.yaml
        log_success "ConfigMaps deployed"
    fi
}

# Deploy database
deploy_database() {
    log_info "Deploying database..."
    
    if [ "$DRY_RUN" = "true" ]; then
        log_info "[DRY RUN] Would deploy database"
    else
        kubectl apply -f ../kubernetes/database.yaml
        
        # Wait for database to be ready
        log_info "Waiting for database to be ready..."
        kubectl wait --for=condition=ready pod -l app=postgres -n "$NAMESPACE" --timeout="$KUBECTL_TIMEOUT"
        
        # Wait for Redis to be ready
        log_info "Waiting for Redis to be ready..."
        kubectl wait --for=condition=ready pod -l app=redis -n "$NAMESPACE" --timeout="$KUBECTL_TIMEOUT"
        
        log_success "Database deployed and ready"
    fi
}

# Deploy core services
deploy_services() {
    log_info "Deploying HLTH services..."
    
    if [ "$DRY_RUN" = "true" ]; then
        log_info "[DRY RUN] Would deploy services"
    else
        kubectl apply -f ../kubernetes/services.yaml
        
        # Wait for services to be ready
        log_info "Waiting for services to be ready..."
        kubectl wait --for=condition=available deployment -l app.kubernetes.io/name=apg-hlth -n "$NAMESPACE" --timeout="$KUBECTL_TIMEOUT"
        
        log_success "Services deployed and ready"
    fi
}

# Deploy monitoring
deploy_monitoring() {
    log_info "Deploying monitoring stack..."
    
    if [ "$DRY_RUN" = "true" ]; then
        log_info "[DRY RUN] Would deploy monitoring"
    else
        kubectl apply -f ../kubernetes/monitoring.yaml
        
        # Wait for monitoring services
        log_info "Waiting for monitoring services to be ready..."
        kubectl wait --for=condition=available deployment -l app.kubernetes.io/part-of=monitoring -n "$NAMESPACE" --timeout="$KUBECTL_TIMEOUT"
        
        log_success "Monitoring deployed and ready"
    fi
}

# Deploy ingress
deploy_ingress() {
    log_info "Deploying ingress..."
    
    if [ "$DRY_RUN" = "true" ]; then
        log_info "[DRY RUN] Would deploy ingress"
    else
        kubectl apply -f ../kubernetes/ingress.yaml
        log_success "Ingress deployed"
    fi
}

# Run health checks
run_health_checks() {
    log_info "Running health checks..."
    
    if [ "$DRY_RUN" = "true" ]; then
        log_info "[DRY RUN] Would run health checks"
        return
    fi
    
    # Check all pods are running
    local failed_pods
    failed_pods=$(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase!=Running --no-headers 2>/dev/null | wc -l)
    
    if [ "$failed_pods" -gt 0 ]; then
        log_error "Some pods are not running:"
        kubectl get pods -n "$NAMESPACE" --field-selector=status.phase!=Running
        log_error "Deployment may have issues. Check pod logs for details."
        return 1
    fi
    
    # Test API Gateway health endpoint
    log_info "Testing API Gateway health endpoint..."
    if kubectl exec -n "$NAMESPACE" deployment/hlth-api-gateway -- curl -f http://localhost:8080/health > /dev/null 2>&1; then
        log_success "API Gateway health check passed"
    else
        log_error "API Gateway health check failed"
        return 1
    fi
    
    log_success "All health checks passed"
}

# Display deployment status
show_status() {
    log_info "Deployment Status:"
    echo ""
    
    log_info "Pods:"
    kubectl get pods -n "$NAMESPACE" -o wide
    echo ""
    
    log_info "Services:"
    kubectl get services -n "$NAMESPACE"
    echo ""
    
    log_info "Ingress:"
    kubectl get ingress -n "$NAMESPACE"
    echo ""
    
    # Show important connection information
    log_info "Access Information:"
    echo "  API Endpoint: https://hlth.your-domain.com/api/v1/"
    echo "  Health Check: https://hlth.your-domain.com/health"
    echo "  Dashboard: https://hlth.your-domain.com/dashboard"
    echo "  Grafana: https://grafana.hlth.your-domain.com (admin/[check grafana-admin-secret])"
    echo "  Prometheus: https://prometheus.hlth.your-domain.com"
    echo "  Jaeger: https://jaeger.hlth.your-domain.com"
    echo ""
    
    log_info "Next Steps:"
    echo "  1. Update DNS records to point to your ingress controller"
    echo "  2. Configure TLS certificates (cert-manager recommended)"
    echo "  3. Update secrets with production values"
    echo "  4. Configure monitoring alerts"
    echo "  5. Set up backup procedures"
}

# Cleanup function for failed deployments
cleanup() {
    if [ "$DRY_RUN" = "true" ]; then
        log_info "[DRY RUN] Would run cleanup"
        return
    fi
    
    log_warning "Deployment failed. Cleaning up..."
    kubectl delete namespace "$NAMESPACE" --ignore-not-found=true
    log_info "Cleanup completed"
}

# Main deployment function
main() {
    log_info "Starting HLTH deployment to $ENVIRONMENT environment"
    
    if [ "$DRY_RUN" = "true" ]; then
        log_warning "DRY RUN MODE - No changes will be made"
    fi
    
    # Trap cleanup on failure
    trap cleanup ERR
    
    check_prerequisites
    create_namespace
    deploy_secrets
    deploy_configmaps
    deploy_database
    deploy_services
    deploy_monitoring
    deploy_ingress
    
    if [ "$DRY_RUN" != "true" ]; then
        run_health_checks
    fi
    
    show_status
    
    log_success "HLTH deployment completed successfully!"
}

# Help function
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Deploy APG System Health Management (HLTH) to Kubernetes"
    echo ""
    echo "Options:"
    echo "  --dry-run          Perform a dry run without making changes"
    echo "  --environment ENV  Set deployment environment (default: production)"
    echo "  --namespace NS     Set Kubernetes namespace (default: hlth)"
    echo "  --timeout TIME     Set kubectl timeout (default: 300s)"
    echo "  --help            Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  DRY_RUN           Set to 'true' for dry run mode"
    echo "  ENVIRONMENT       Deployment environment"
    echo "  KUBECTL_TIMEOUT   Kubectl operation timeout"
    echo ""
    echo "Examples:"
    echo "  $0                          # Normal deployment"
    echo "  $0 --dry-run               # Dry run"
    echo "  $0 --environment staging   # Deploy to staging"
    echo "  DRY_RUN=true $0            # Dry run using environment variable"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        --environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --timeout)
            KUBECTL_TIMEOUT="$2"
            shift 2
            ;;
        --help)
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

# Run main function
main