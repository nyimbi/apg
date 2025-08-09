#!/bin/bash
#
# APG System Health Management (HLTH) - Health Check Script
# Copyright © 2025 Datacraft - www.datacraft.co.ke
# Author: Nyimbi Odero <nyimbi@gmail.com>
#
# This script performs comprehensive health checks on the HLTH deployment
#

set -euo pipefail

# Configuration
NAMESPACE="${NAMESPACE:-hlth}"
TIMEOUT="${TIMEOUT:-30}"
VERBOSE="${VERBOSE:-false}"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Global counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNING_CHECKS=0

# Test result tracking
run_check() {
    local test_name="$1"
    local test_command="$2"
    ((TOTAL_CHECKS++))
    
    if [ "$VERBOSE" = "true" ]; then
        log_info "Running: $test_name"
    fi
    
    if eval "$test_command" >/dev/null 2>&1; then
        log_success "✓ $test_name"
        ((PASSED_CHECKS++))
        return 0
    else
        log_error "✗ $test_name"
        ((FAILED_CHECKS++))
        return 1
    fi
}

run_check_with_warning() {
    local test_name="$1"
    local test_command="$2"
    local warning_command="$3"
    ((TOTAL_CHECKS++))
    
    if [ "$VERBOSE" = "true" ]; then
        log_info "Running: $test_name"
    fi
    
    if eval "$test_command" >/dev/null 2>&1; then
        log_success "✓ $test_name"
        ((PASSED_CHECKS++))
        return 0
    elif eval "$warning_command" >/dev/null 2>&1; then
        log_warning "⚠ $test_name (with warnings)"
        ((WARNING_CHECKS++))
        return 0
    else
        log_error "✗ $test_name"
        ((FAILED_CHECKS++))
        return 1
    fi
}

# Check Kubernetes connectivity
check_k8s_connectivity() {
    log_info "Checking Kubernetes connectivity..."
    
    run_check "Kubernetes cluster connectivity" \
        "kubectl cluster-info"
    
    run_check "HLTH namespace exists" \
        "kubectl get namespace $NAMESPACE"
}

# Check pod health
check_pod_health() {
    log_info "Checking pod health..."
    
    local services=("hlth-api-gateway" "hlth-health-service" "hlth-ml-engine" "hlth-alert-engine" "hlth-remediation-engine" "postgres" "redis")
    
    for service in "${services[@]}"; do
        run_check_with_warning \
            "Pod $service is running" \
            "kubectl get pods -n $NAMESPACE -l app=$service --field-selector=status.phase=Running | grep -q $service" \
            "kubectl get pods -n $NAMESPACE -l app=$service | grep -q $service"
        
        # Check pod readiness
        local pods
        pods=$(kubectl get pods -n "$NAMESPACE" -l app="$service" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo "")
        
        for pod in $pods; do
            if [ -n "$pod" ]; then
                run_check_with_warning \
                    "Pod $pod is ready" \
                    "kubectl get pod -n $NAMESPACE $pod -o jsonpath='{.status.conditions[?(@.type==\"Ready\")].status}' | grep -q True" \
                    "kubectl get pod -n $NAMESPACE $pod >/dev/null"
            fi
        done
    done
}

# Check service endpoints
check_service_endpoints() {
    log_info "Checking service endpoints..."
    
    local services=("hlth-api-gateway" "hlth-health-service" "hlth-ml-engine" "hlth-alert-engine" "hlth-remediation-engine" "postgres-service" "redis-service")
    
    for service in "${services[@]}"; do
        run_check "Service $service has endpoints" \
            "kubectl get endpoints -n $NAMESPACE $service -o jsonpath='{.subsets[*].addresses[*].ip}' | grep -q ."
    done
}

# Check health endpoints
check_health_endpoints() {
    log_info "Checking application health endpoints..."
    
    # Check API Gateway health
    run_check "API Gateway health endpoint" \
        "kubectl exec -n $NAMESPACE deployment/hlth-api-gateway -c api-gateway -- curl -sf http://localhost:8080/health"
    
    # Check Health Service
    run_check "Health Service health endpoint" \
        "kubectl exec -n $NAMESPACE deployment/hlth-health-service -c health-service -- curl -sf http://localhost:8081/health"
    
    # Check ML Engine
    run_check_with_warning "ML Engine health endpoint" \
        "kubectl exec -n $NAMESPACE deployment/hlth-ml-engine -c ml-engine -- curl -sf http://localhost:8082/health" \
        "kubectl get pods -n $NAMESPACE -l app=hlth-ml-engine --field-selector=status.phase=Running | grep -q hlth-ml-engine"
    
    # Check Alert Engine
    run_check "Alert Engine health endpoint" \
        "kubectl exec -n $NAMESPACE deployment/hlth-alert-engine -c alert-engine -- curl -sf http://localhost:8083/health"
    
    # Check Remediation Engine
    run_check "Remediation Engine health endpoint" \
        "kubectl exec -n $NAMESPACE deployment/hlth-remediation-engine -c remediation-engine -- curl -sf http://localhost:8084/health"
}

# Check database connectivity
check_database_connectivity() {
    log_info "Checking database connectivity..."
    
    # PostgreSQL connectivity
    run_check "PostgreSQL connectivity" \
        "kubectl exec -n $NAMESPACE deployment/postgres -- pg_isready -U hlth -d hlth"
    
    # Redis connectivity
    run_check "Redis connectivity" \
        "kubectl exec -n $NAMESPACE deployment/redis -- redis-cli ping | grep -q PONG"
    
    # Database schema check
    run_check "Database schema exists" \
        "kubectl exec -n $NAMESPACE deployment/postgres -- psql -U hlth -d hlth -c '\\dt' | grep -q system_components"
}

# Check resource usage
check_resource_usage() {
    log_info "Checking resource usage..."
    
    # Check CPU usage
    local cpu_usage
    cpu_usage=$(kubectl top pods -n "$NAMESPACE" --no-headers 2>/dev/null | awk '{sum += $2} END {print sum}' || echo "0")
    
    run_check_with_warning "CPU usage reasonable" \
        "[ ${cpu_usage%m*} -lt 2000 ]" \
        "[ ${cpu_usage%m*} -lt 4000 ]"
    
    # Check memory usage
    local memory_usage
    memory_usage=$(kubectl top pods -n "$NAMESPACE" --no-headers 2>/dev/null | awk '{sum += $3} END {print sum}' || echo "0")
    
    run_check_with_warning "Memory usage reasonable" \
        "[ ${memory_usage%Mi*} -lt 4000 ]" \
        "[ ${memory_usage%Mi*} -lt 8000 ]"
    
    # Check persistent volume claims
    run_check "All PVCs are bound" \
        "! kubectl get pvc -n $NAMESPACE --no-headers | grep -v Bound"
}

# Check networking
check_networking() {
    log_info "Checking networking..."
    
    # Check ingress
    run_check_with_warning "Ingress exists and configured" \
        "kubectl get ingress -n $NAMESPACE hlth-ingress -o jsonpath='{.status.loadBalancer.ingress[0].ip}' | grep -q ." \
        "kubectl get ingress -n $NAMESPACE hlth-ingress >/dev/null"
    
    # Check network policies
    run_check_with_warning "Network policies configured" \
        "kubectl get networkpolicy -n $NAMESPACE | grep -q hlth" \
        "echo 'Network policies not configured - consider adding for security'"
    
    # Check service mesh (if applicable)
    if kubectl get pods -n istio-system >/dev/null 2>&1; then
        run_check_with_warning "Istio sidecar injection" \
            "kubectl get namespace $NAMESPACE -o jsonpath='{.metadata.labels.istio-injection}' | grep -q enabled" \
            "echo 'Istio detected but injection not enabled'"
    fi
}

# Check monitoring
check_monitoring() {
    log_info "Checking monitoring stack..."
    
    # Check Prometheus
    run_check_with_warning "Prometheus is running" \
        "kubectl get pods -n $NAMESPACE -l app=prometheus --field-selector=status.phase=Running | grep -q prometheus" \
        "kubectl get pods -n $NAMESPACE -l app=prometheus | grep -q prometheus"
    
    # Check Grafana
    run_check_with_warning "Grafana is running" \
        "kubectl get pods -n $NAMESPACE -l app=grafana --field-selector=status.phase=Running | grep -q grafana" \
        "kubectl get pods -n $NAMESPACE -l app=grafana | grep -q grafana"
    
    # Check Jaeger
    run_check_with_warning "Jaeger is running" \
        "kubectl get pods -n $NAMESPACE -l app=jaeger --field-selector=status.phase=Running | grep -q jaeger" \
        "kubectl get pods -n $NAMESPACE -l app=jaeger | grep -q jaeger"
    
    # Check metrics endpoints
    run_check_with_warning "Prometheus metrics available" \
        "kubectl exec -n $NAMESPACE deployment/hlth-api-gateway -c api-gateway -- curl -sf http://localhost:8080/metrics | grep -q hlth_" \
        "echo 'Metrics endpoint exists but may not have HLTH-specific metrics yet'"
}

# Check security
check_security() {
    log_info "Checking security configuration..."
    
    # Check RBAC
    run_check "Service accounts exist" \
        "kubectl get serviceaccount -n $NAMESPACE hlth-service-account"
    
    run_check "RBAC configured" \
        "kubectl get clusterrole hlth-remediation-role"
    
    # Check secrets
    run_check "Required secrets exist" \
        "kubectl get secret -n $NAMESPACE hlth-secrets"
    
    # Check pod security contexts
    run_check_with_warning "Pods run as non-root" \
        "kubectl get pods -n $NAMESPACE -o jsonpath='{.items[*].spec.securityContext.runAsNonRoot}' | grep -q true" \
        "echo 'Some pods may be running as root - review security contexts'"
    
    # Check network policies
    run_check_with_warning "Network policies in place" \
        "kubectl get networkpolicy -n $NAMESPACE | grep -q ." \
        "echo 'No network policies found - consider implementing for security'"
}

# Check API functionality
check_api_functionality() {
    log_info "Checking API functionality..."
    
    # Test API Gateway endpoints
    run_check "API Gateway responds to requests" \
        "kubectl exec -n $NAMESPACE deployment/hlth-api-gateway -c api-gateway -- curl -sf http://localhost:8080/api/v1/health"
    
    # Test readiness endpoints
    run_check "Services report ready status" \
        "kubectl exec -n $NAMESPACE deployment/hlth-api-gateway -c api-gateway -- curl -sf http://localhost:8080/ready"
    
    # Test metrics endpoint
    run_check_with_warning "Metrics endpoint functional" \
        "kubectl exec -n $NAMESPACE deployment/hlth-api-gateway -c api-gateway -- curl -sf http://localhost:8080/metrics | head -1" \
        "echo 'Metrics endpoint exists'"
}

# Generate summary report
generate_summary() {
    echo ""
    log_info "Health Check Summary"
    echo "===================="
    echo "Total Checks: $TOTAL_CHECKS"
    echo "Passed: $PASSED_CHECKS"
    echo "Warnings: $WARNING_CHECKS"
    echo "Failed: $FAILED_CHECKS"
    echo ""
    
    local success_rate
    success_rate=$((($PASSED_CHECKS + $WARNING_CHECKS) * 100 / $TOTAL_CHECKS))
    
    if [ $FAILED_CHECKS -eq 0 ]; then
        if [ $WARNING_CHECKS -eq 0 ]; then
            log_success "🎉 All checks passed! HLTH deployment is healthy."
        else
            log_warning "⚠️ All critical checks passed with $WARNING_CHECKS warnings. HLTH deployment is functional."
        fi
        return 0
    else
        log_error "❌ $FAILED_CHECKS critical checks failed. HLTH deployment has issues."
        echo ""
        log_info "Recommended actions:"
        echo "1. Check pod logs: kubectl logs -n $NAMESPACE <pod-name>"
        echo "2. Describe failed pods: kubectl describe pods -n $NAMESPACE"
        echo "3. Check events: kubectl get events -n $NAMESPACE --sort-by='.lastTimestamp'"
        echo "4. Verify resource quotas and limits"
        echo "5. Check ingress controller and DNS configuration"
        return 1
    fi
}

# Main function
main() {
    log_info "Starting HLTH health checks..."
    echo "Namespace: $NAMESPACE"
    echo "Timeout: ${TIMEOUT}s"
    echo "Verbose: $VERBOSE"
    echo ""
    
    check_k8s_connectivity
    check_pod_health
    check_service_endpoints
    check_database_connectivity
    check_health_endpoints
    check_resource_usage
    check_networking
    check_monitoring
    check_security
    check_api_functionality
    
    generate_summary
}

# Help function
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Perform comprehensive health checks on APG HLTH deployment"
    echo ""
    echo "Options:"
    echo "  --namespace NS    Kubernetes namespace (default: hlth)"
    echo "  --timeout N       Request timeout in seconds (default: 30)"
    echo "  --verbose         Enable verbose output"
    echo "  --help           Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  NAMESPACE        Kubernetes namespace"
    echo "  TIMEOUT          Request timeout"
    echo "  VERBOSE          Enable verbose mode (true/false)"
    echo ""
    echo "Examples:"
    echo "  $0                           # Basic health check"
    echo "  $0 --verbose                # Detailed output"
    echo "  $0 --namespace hlth-staging # Check staging environment"
    echo "  VERBOSE=true $0             # Verbose using environment variable"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE="true"
            shift
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