#!/bin/bash
# APG Event Streaming Bus - Deployment Script
# Production deployment automation and management
# © 2025 Datacraft. All rights reserved.

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOYMENT_CONFIG="$PROJECT_ROOT/deployment/config"

# Default values
ENVIRONMENT="staging"
NAMESPACE=""
CLUSTER=""
REGION="us-west-2"
IMAGE_TAG=""
DRY_RUN=false
FORCE=false
ROLLBACK=false
BACKUP=true
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARN: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" >&2
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS: $1${NC}"
}

debug() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "${PURPLE}[$(date +'%Y-%m-%d %H:%M:%S')] DEBUG: $1${NC}"
    fi
}

# Usage information
usage() {
    cat << EOF
APG Event Streaming Bus Deployment Script

Usage: $0 [OPTIONS]

OPTIONS:
    -e, --environment    Target environment (staging|production) [default: staging]
    -n, --namespace      Kubernetes namespace [auto-detected from environment]
    -c, --cluster        Kubernetes cluster name [auto-detected from environment]
    -r, --region         AWS region [default: us-west-2]
    -t, --image-tag      Docker image tag [auto-detected from git]
    -d, --dry-run        Show what would be deployed without making changes
    -f, --force          Force deployment without confirmation
    --rollback           Rollback to previous deployment
    --no-backup          Skip backup before deployment
    -v, --verbose        Enable verbose logging
    -h, --help           Show this help message

EXAMPLES:
    # Deploy to staging
    $0 --environment staging

    # Deploy specific version to production
    $0 --environment production --image-tag v1.2.3

    # Dry run deployment
    $0 --environment production --dry-run

    # Rollback deployment
    $0 --environment production --rollback

    # Force deployment without prompts
    $0 --environment production --force

ENVIRONMENT VARIABLES:
    AWS_ACCESS_KEY_ID       AWS access key
    AWS_SECRET_ACCESS_KEY   AWS secret key
    KUBECONFIG             Kubernetes config file path
    DOCKER_REGISTRY        Docker registry URL
    SLACK_WEBHOOK_URL      Slack notification webhook

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
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -c|--cluster)
                CLUSTER="$2"
                shift 2
                ;;
            -r|--region)
                REGION="$2"
                shift 2
                ;;
            -t|--image-tag)
                IMAGE_TAG="$2"
                shift 2
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -f|--force)
                FORCE=true
                shift
                ;;
            --rollback)
                ROLLBACK=true
                shift
                ;;
            --no-backup)
                BACKUP=false
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
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

# Validate environment
validate_environment() {
    log "Validating environment: $ENVIRONMENT"
    
    case $ENVIRONMENT in
        staging|production)
            ;;
        *)
            error "Invalid environment: $ENVIRONMENT. Must be 'staging' or 'production'"
            exit 1
            ;;
    esac
    
    # Set environment-specific defaults
    if [[ -z "$NAMESPACE" ]]; then
        NAMESPACE="apg-event-streaming-bus-$ENVIRONMENT"
    fi
    
    if [[ -z "$CLUSTER" ]]; then
        CLUSTER="apg-$ENVIRONMENT-cluster"
    fi
    
    debug "Environment: $ENVIRONMENT"
    debug "Namespace: $NAMESPACE"
    debug "Cluster: $CLUSTER"
    debug "Region: $REGION"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check required tools
    local required_tools=("kubectl" "aws" "docker" "jq" "yq")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            error "Required tool not found: $tool"
            exit 1
        fi
        debug "Found tool: $tool"
    done
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        error "AWS credentials not configured or expired"
        exit 1
    fi
    debug "AWS credentials validated"
    
    # Check Docker registry access
    if ! docker info &> /dev/null; then
        error "Docker daemon not running or not accessible"
        exit 1
    fi
    debug "Docker access validated"
    
    success "Prerequisites validated"
}

# Configure Kubernetes access
configure_kubernetes() {
    log "Configuring Kubernetes access..."
    
    # Update kubeconfig
    aws eks update-kubeconfig --region "$REGION" --name "$CLUSTER"
    
    # Test cluster access
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot access Kubernetes cluster: $CLUSTER"
        exit 1
    fi
    
    # Check namespace
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        warn "Namespace $NAMESPACE does not exist. Creating..."
        kubectl apply -f "$PROJECT_ROOT/k8s/namespace.yaml"
    fi
    
    success "Kubernetes access configured"
}

# Get current deployment info
get_current_deployment() {
    log "Getting current deployment information..."
    
    local current_image=""
    local current_replicas=""
    
    if kubectl get deployment event-streaming-bus -n "$NAMESPACE" &> /dev/null; then
        current_image=$(kubectl get deployment event-streaming-bus -n "$NAMESPACE" -o jsonpath='{.spec.template.spec.containers[0].image}')
        current_replicas=$(kubectl get deployment event-streaming-bus -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')
        
        debug "Current image: $current_image"
        debug "Current replicas: $current_replicas"
        
        echo "$current_image" > "/tmp/esb-current-image-$ENVIRONMENT"
        echo "$current_replicas" > "/tmp/esb-current-replicas-$ENVIRONMENT"
    else
        warn "No existing deployment found"
    fi
}

# Determine image tag
determine_image_tag() {
    if [[ -z "$IMAGE_TAG" ]]; then
        if [[ "$ROLLBACK" == "true" ]]; then
            error "Image tag required for rollback operation"
            exit 1
        fi
        
        # Auto-detect from git
        if git rev-parse --git-dir > /dev/null 2>&1; then
            IMAGE_TAG=$(git rev-parse --short HEAD)
            if [[ "$ENVIRONMENT" == "production" ]]; then
                # For production, prefer tags over commit hashes
                local git_tag=$(git describe --tags --exact-match 2>/dev/null || echo "")
                if [[ -n "$git_tag" ]]; then
                    IMAGE_TAG="$git_tag"
                fi
            fi
        else
            IMAGE_TAG="latest"
        fi
    fi
    
    log "Using image tag: $IMAGE_TAG"
}

# Backup current deployment
backup_deployment() {
    if [[ "$BACKUP" == "false" ]]; then
        debug "Skipping backup as requested"
        return
    fi
    
    log "Creating deployment backup..."
    
    local backup_dir="$PROJECT_ROOT/backups/$(date +%Y%m%d-%H%M%S)-$ENVIRONMENT"
    mkdir -p "$backup_dir"
    
    # Backup Kubernetes resources
    local resources=("deployment" "service" "configmap" "secret" "ingress")
    for resource in "${resources[@]}"; do
        if kubectl get "$resource" -n "$NAMESPACE" &> /dev/null; then
            kubectl get "$resource" -n "$NAMESPACE" -o yaml > "$backup_dir/$resource.yaml"
            debug "Backed up $resource"
        fi
    done
    
    # Backup database (if applicable)
    if [[ "$ENVIRONMENT" == "production" ]]; then
        log "Creating database backup..."
        # This would be implemented based on your database backup strategy
        # kubectl exec -n "$NAMESPACE" postgres-0 -- pg_dump ... > "$backup_dir/database.sql"
    fi
    
    success "Backup created: $backup_dir"
}

# Deploy application
deploy_application() {
    log "Deploying APG Event Streaming Bus..."
    
    local registry="${DOCKER_REGISTRY:-ghcr.io/datacraft}"
    local full_image="$registry/apg-event-streaming-bus:$IMAGE_TAG"
    
    debug "Full image: $full_image"
    
    # Verify image exists
    if ! docker manifest inspect "$full_image" &> /dev/null; then
        error "Docker image not found: $full_image"
        exit 1
    fi
    
    # Apply Kubernetes manifests
    local manifests=(
        "namespace.yaml"
        "configmap.yaml" 
        "secret.yaml"
        "deployment.yaml"
        "service.yaml"
        "ingress.yaml"
    )
    
    for manifest in "${manifests[@]}"; do
        local manifest_path="$PROJECT_ROOT/k8s/$manifest"
        if [[ -f "$manifest_path" ]]; then
            if [[ "$DRY_RUN" == "true" ]]; then
                log "Would apply: $manifest"
                kubectl apply -f "$manifest_path" -n "$NAMESPACE" --dry-run=client
            else
                log "Applying: $manifest"
                kubectl apply -f "$manifest_path" -n "$NAMESPACE"
            fi
        else
            warn "Manifest not found: $manifest_path"
        fi
    done
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "Dry run completed - no changes made"
        return
    fi
    
    # Update deployment image
    log "Updating deployment image to: $full_image"
    kubectl set image deployment/event-streaming-bus \
        event-streaming-bus="$full_image" \
        -n "$NAMESPACE"
    
    # Update worker deployment if it exists
    if kubectl get deployment event-streaming-bus-worker -n "$NAMESPACE" &> /dev/null; then
        kubectl set image deployment/event-streaming-bus-worker \
            event-streaming-bus-worker="$full_image" \
            -n "$NAMESPACE"
    fi
}

# Wait for deployment
wait_for_deployment() {
    if [[ "$DRY_RUN" == "true" ]]; then
        return
    fi
    
    log "Waiting for deployment to complete..."
    
    # Wait for main deployment
    if ! kubectl rollout status deployment/event-streaming-bus \
        -n "$NAMESPACE" \
        --timeout=600s; then
        error "Deployment failed or timed out"
        
        # Show pod status for debugging
        kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=event-streaming-bus
        kubectl describe pods -n "$NAMESPACE" -l app.kubernetes.io/name=event-streaming-bus
        
        exit 1
    fi
    
    # Wait for worker deployment if it exists
    if kubectl get deployment event-streaming-bus-worker -n "$NAMESPACE" &> /dev/null; then
        kubectl rollout status deployment/event-streaming-bus-worker \
            -n "$NAMESPACE" \
            --timeout=300s
    fi
    
    success "Deployment completed successfully"
}

# Verify deployment
verify_deployment() {
    if [[ "$DRY_RUN" == "true" ]]; then
        return
    fi
    
    log "Verifying deployment..."
    
    # Check pod status
    local running_pods=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=event-streaming-bus --field-selector=status.phase=Running --no-headers | wc -l)
    debug "Running pods: $running_pods"
    
    if [[ "$running_pods" -eq 0 ]]; then
        error "No running pods found"
        exit 1
    fi
    
    # Health check
    log "Running health checks..."
    local health_check_passed=false
    for i in {1..10}; do
        if kubectl run health-check-$RANDOM --rm -i --restart=Never --image=curlimages/curl:latest -- \
            curl -f "http://event-streaming-bus.$NAMESPACE:8080/health" &> /dev/null; then
            health_check_passed=true
            break
        fi
        debug "Health check attempt $i failed, retrying in 10 seconds..."
        sleep 10
    done
    
    if [[ "$health_check_passed" == "false" ]]; then
        error "Health checks failed"
        exit 1
    fi
    
    # Get service endpoints
    local api_url=""
    if [[ "$ENVIRONMENT" == "production" ]]; then
        api_url="https://api.event-streaming-bus.datacraft.co.ke"
    else
        api_url="https://staging.event-streaming-bus.datacraft.co.ke"
    fi
    
    success "Deployment verified successfully"
    success "API URL: $api_url"
}

# Rollback deployment
rollback_deployment() {
    log "Rolling back deployment..."
    
    if [[ ! -f "/tmp/esb-current-image-$ENVIRONMENT" ]]; then
        error "No previous deployment found for rollback"
        exit 1
    fi
    
    local previous_image=$(cat "/tmp/esb-current-image-$ENVIRONMENT")
    local previous_replicas=$(cat "/tmp/esb-current-replicas-$ENVIRONMENT" 2>/dev/null || echo "3")
    
    log "Rolling back to: $previous_image"
    
    kubectl set image deployment/event-streaming-bus \
        event-streaming-bus="$previous_image" \
        -n "$NAMESPACE"
    
    kubectl scale deployment/event-streaming-bus \
        --replicas="$previous_replicas" \
        -n "$NAMESPACE"
    
    wait_for_deployment
    verify_deployment
    
    success "Rollback completed successfully"
}

# Send notifications
send_notifications() {
    if [[ -z "${SLACK_WEBHOOK_URL:-}" ]]; then
        debug "No Slack webhook configured, skipping notifications"
        return
    fi
    
    local status="$1"
    local message="$2"
    local color=""
    local emoji=""
    
    case $status in
        success)
            color="good"
            emoji="✅"
            ;;
        failure)
            color="danger"
            emoji="❌"
            ;;
        warning)
            color="warning"
            emoji="⚠️"
            ;;
    esac
    
    local payload=$(cat << EOF
{
    "attachments": [
        {
            "color": "$color",
            "title": "$emoji APG Event Streaming Bus Deployment",
            "fields": [
                {
                    "title": "Environment",
                    "value": "$ENVIRONMENT",
                    "short": true
                },
                {
                    "title": "Image Tag",
                    "value": "$IMAGE_TAG",
                    "short": true
                },
                {
                    "title": "Namespace",
                    "value": "$NAMESPACE",
                    "short": true
                },
                {
                    "title": "Status",
                    "value": "$message",
                    "short": false
                }
            ],
            "footer": "APG Event Streaming Bus",
            "ts": $(date +%s)
        }
    ]
}
EOF
)
    
    curl -X POST -H 'Content-type: application/json' \
        --data "$payload" \
        "$SLACK_WEBHOOK_URL" &> /dev/null || true
}

# Cleanup temporary files
cleanup() {
    debug "Cleaning up temporary files..."
    rm -f "/tmp/esb-current-image-$ENVIRONMENT" "/tmp/esb-current-replicas-$ENVIRONMENT" 2>/dev/null || true
}

# Main deployment workflow
main() {
    log "Starting APG Event Streaming Bus deployment..."
    
    parse_args "$@"
    validate_environment
    check_prerequisites
    configure_kubernetes
    get_current_deployment
    determine_image_tag
    
    # Confirmation prompt for production
    if [[ "$ENVIRONMENT" == "production" && "$FORCE" == "false" && "$DRY_RUN" == "false" ]]; then
        echo
        warn "You are about to deploy to PRODUCTION environment!"
        warn "Environment: $ENVIRONMENT"
        warn "Cluster: $CLUSTER"
        warn "Namespace: $NAMESPACE"
        warn "Image Tag: $IMAGE_TAG"
        echo
        read -p "Are you sure you want to continue? (yes/no): " -r
        if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
            log "Deployment cancelled by user"
            exit 0
        fi
    fi
    
    if [[ "$ROLLBACK" == "true" ]]; then
        rollback_deployment
        send_notifications "success" "Rollback completed successfully"
    else
        backup_deployment
        deploy_application
        wait_for_deployment
        verify_deployment
        send_notifications "success" "Deployment completed successfully"
    fi
    
    cleanup
    success "APG Event Streaming Bus deployment completed successfully!"
}

# Error handling
trap 'error "Deployment failed at line $LINENO"; send_notifications "failure" "Deployment failed"; cleanup; exit 1' ERR
trap 'cleanup' EXIT

# Run main function
main "$@"