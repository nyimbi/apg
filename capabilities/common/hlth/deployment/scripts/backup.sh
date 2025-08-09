#!/bin/bash
#
# APG System Health Management (HLTH) - Database Backup Script
# Copyright © 2025 Datacraft - www.datacraft.co.ke
# Author: Nyimbi Odero <nyimbi@gmail.com>
#
# This script creates backups of the HLTH database and configuration
#

set -euo pipefail

# Configuration
NAMESPACE="${NAMESPACE:-hlth}"
BACKUP_DIR="${BACKUP_DIR:-/tmp/hlth-backups}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
S3_BUCKET="${S3_BUCKET:-}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Create backup directory
create_backup_dir() {
    log_info "Creating backup directory: $BACKUP_DIR"
    mkdir -p "$BACKUP_DIR"
    
    # Create subdirectories
    mkdir -p "$BACKUP_DIR/database"
    mkdir -p "$BACKUP_DIR/config"
    mkdir -p "$BACKUP_DIR/logs"
}

# Backup PostgreSQL database
backup_database() {
    log_info "Starting database backup..."
    
    local backup_file="$BACKUP_DIR/database/hlth_db_$TIMESTAMP.sql"
    local postgres_pod
    
    # Find PostgreSQL pod
    postgres_pod=$(kubectl get pod -n "$NAMESPACE" -l app=postgres -o jsonpath='{.items[0].metadata.name}')
    
    if [ -z "$postgres_pod" ]; then
        log_error "PostgreSQL pod not found"
        return 1
    fi
    
    log_info "Backing up database from pod: $postgres_pod"
    
    # Create database backup
    kubectl exec -n "$NAMESPACE" "$postgres_pod" -- pg_dump -U hlth hlth > "$backup_file"
    
    # Compress backup
    gzip "$backup_file"
    log_success "Database backup created: ${backup_file}.gz"
    
    # Create schema-only backup for reference
    local schema_file="$BACKUP_DIR/database/hlth_schema_$TIMESTAMP.sql"
    kubectl exec -n "$NAMESPACE" "$postgres_pod" -- pg_dump -U hlth -s hlth > "$schema_file"
    gzip "$schema_file"
    log_success "Schema backup created: ${schema_file}.gz"
}

# Backup Redis data
backup_redis() {
    log_info "Starting Redis backup..."
    
    local redis_pod
    local backup_file="$BACKUP_DIR/database/redis_dump_$TIMESTAMP.rdb"
    
    # Find Redis pod
    redis_pod=$(kubectl get pod -n "$NAMESPACE" -l app=redis -o jsonpath='{.items[0].metadata.name}')
    
    if [ -z "$redis_pod" ]; then
        log_error "Redis pod not found"
        return 1
    fi
    
    log_info "Backing up Redis from pod: $redis_pod"
    
    # Force Redis to save current state
    kubectl exec -n "$NAMESPACE" "$redis_pod" -- redis-cli BGSAVE
    
    # Wait for background save to complete
    sleep 5
    
    # Copy RDB file
    kubectl cp -n "$NAMESPACE" "$redis_pod:/data/dump.rdb" "$backup_file"
    
    # Compress backup
    gzip "$backup_file"
    log_success "Redis backup created: ${backup_file}.gz"
}

# Backup Kubernetes configurations
backup_configs() {
    log_info "Starting configuration backup..."
    
    local config_dir="$BACKUP_DIR/config"
    
    # Backup all HLTH resources
    kubectl get all,configmap,secret,pvc,ingress -n "$NAMESPACE" -o yaml > "$config_dir/k8s_resources_$TIMESTAMP.yaml"
    
    # Backup namespace definition
    kubectl get namespace "$NAMESPACE" -o yaml > "$config_dir/namespace_$TIMESTAMP.yaml"
    
    # Backup RBAC
    kubectl get clusterrole,clusterrolebinding,role,rolebinding -o yaml | grep -A999 -B999 "hlth" > "$config_dir/rbac_$TIMESTAMP.yaml" || true
    
    # Compress configs
    tar -czf "$config_dir/configs_$TIMESTAMP.tar.gz" -C "$config_dir" ./*.yaml
    rm "$config_dir"/*.yaml
    
    log_success "Configuration backup created: $config_dir/configs_$TIMESTAMP.tar.gz"
}

# Backup application logs
backup_logs() {
    log_info "Starting logs backup..."
    
    local logs_dir="$BACKUP_DIR/logs"
    local services=("hlth-api-gateway" "hlth-health-service" "hlth-ml-engine" "hlth-alert-engine" "hlth-remediation-engine")
    
    for service in "${services[@]}"; do
        log_info "Backing up logs for $service..."
        
        local pods
        pods=$(kubectl get pods -n "$NAMESPACE" -l app="$service" -o jsonpath='{.items[*].metadata.name}')
        
        for pod in $pods; do
            kubectl logs -n "$NAMESPACE" "$pod" > "$logs_dir/${service}_${pod}_$TIMESTAMP.log" 2>/dev/null || true
            
            # Get previous logs if available
            kubectl logs -n "$NAMESPACE" "$pod" --previous > "$logs_dir/${service}_${pod}_previous_$TIMESTAMP.log" 2>/dev/null || true
        done
    done
    
    # Compress logs
    tar -czf "$logs_dir/logs_$TIMESTAMP.tar.gz" -C "$logs_dir" ./*.log
    rm "$logs_dir"/*.log
    
    log_success "Logs backup created: $logs_dir/logs_$TIMESTAMP.tar.gz"
}

# Upload to S3 (if configured)
upload_to_s3() {
    if [ -z "$S3_BUCKET" ]; then
        log_info "S3_BUCKET not configured, skipping S3 upload"
        return
    fi
    
    if ! command -v aws &> /dev/null; then
        log_warning "AWS CLI not installed, skipping S3 upload"
        return
    fi
    
    log_info "Uploading backups to S3 bucket: $S3_BUCKET"
    
    local s3_path="s3://$S3_BUCKET/hlth-backups/$TIMESTAMP/"
    
    # Upload all backup files
    aws s3 cp "$BACKUP_DIR" "$s3_path" --recursive --quiet
    
    log_success "Backups uploaded to: $s3_path"
}

# Clean old backups
cleanup_old_backups() {
    log_info "Cleaning up backups older than $RETENTION_DAYS days..."
    
    # Local cleanup
    find "$BACKUP_DIR" -type f -mtime "+$RETENTION_DAYS" -delete 2>/dev/null || true
    find "$BACKUP_DIR" -type d -empty -delete 2>/dev/null || true
    
    # S3 cleanup (if configured)
    if [ -n "$S3_BUCKET" ] && command -v aws &> /dev/null; then
        local cutoff_date
        cutoff_date=$(date -d "$RETENTION_DAYS days ago" '+%Y-%m-%d')
        
        aws s3 ls "s3://$S3_BUCKET/hlth-backups/" --recursive | \
        awk '{print $1, $2, $4}' | \
        while read -r date time file; do
            if [[ "$date" < "$cutoff_date" ]]; then
                aws s3 rm "s3://$S3_BUCKET/$file" --quiet
                log_info "Removed old S3 backup: $file"
            fi
        done
    fi
    
    log_success "Cleanup completed"
}

# Verify backup integrity
verify_backups() {
    log_info "Verifying backup integrity..."
    
    local errors=0
    
    # Check database backup
    if [ -f "$BACKUP_DIR/database/hlth_db_$TIMESTAMP.sql.gz" ]; then
        if gzip -t "$BACKUP_DIR/database/hlth_db_$TIMESTAMP.sql.gz"; then
            log_success "Database backup integrity: OK"
        else
            log_error "Database backup integrity: FAILED"
            ((errors++))
        fi
    fi
    
    # Check Redis backup
    if [ -f "$BACKUP_DIR/database/redis_dump_$TIMESTAMP.rdb.gz" ]; then
        if gzip -t "$BACKUP_DIR/database/redis_dump_$TIMESTAMP.rdb.gz"; then
            log_success "Redis backup integrity: OK"
        else
            log_error "Redis backup integrity: FAILED"
            ((errors++))
        fi
    fi
    
    # Check config backup
    if [ -f "$BACKUP_DIR/config/configs_$TIMESTAMP.tar.gz" ]; then
        if tar -tzf "$BACKUP_DIR/config/configs_$TIMESTAMP.tar.gz" >/dev/null 2>&1; then
            log_success "Config backup integrity: OK"
        else
            log_error "Config backup integrity: FAILED"
            ((errors++))
        fi
    fi
    
    # Check logs backup
    if [ -f "$BACKUP_DIR/logs/logs_$TIMESTAMP.tar.gz" ]; then
        if tar -tzf "$BACKUP_DIR/logs/logs_$TIMESTAMP.tar.gz" >/dev/null 2>&1; then
            log_success "Logs backup integrity: OK"
        else
            log_error "Logs backup integrity: FAILED"
            ((errors++))
        fi
    fi
    
    if [ $errors -eq 0 ]; then
        log_success "All backups verified successfully"
    else
        log_error "$errors backup(s) failed verification"
        return 1
    fi
}

# Generate backup report
generate_report() {
    local report_file="$BACKUP_DIR/backup_report_$TIMESTAMP.txt"
    
    {
        echo "APG HLTH Backup Report"
        echo "======================"
        echo "Timestamp: $TIMESTAMP"
        echo "Namespace: $NAMESPACE"
        echo "Backup Directory: $BACKUP_DIR"
        echo ""
        echo "Backup Files Created:"
        find "$BACKUP_DIR" -name "*$TIMESTAMP*" -type f -exec basename {} \; | sort
        echo ""
        echo "Backup Sizes:"
        find "$BACKUP_DIR" -name "*$TIMESTAMP*" -type f -exec ls -lh {} \; | awk '{print $5, $9}' | sort
        echo ""
        echo "Total Backup Size:"
        find "$BACKUP_DIR" -name "*$TIMESTAMP*" -type f -exec du -ch {} + | tail -1
        echo ""
        echo "S3 Upload: $([ -n "$S3_BUCKET" ] && echo "Enabled ($S3_BUCKET)" || echo "Disabled")"
        echo "Retention: $RETENTION_DAYS days"
    } > "$report_file"
    
    log_success "Backup report created: $report_file"
    
    # Display summary
    log_info "Backup Summary:"
    cat "$report_file"
}

# Main backup function
main() {
    log_info "Starting HLTH backup process..."
    
    # Check prerequisites
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is required but not installed"
        exit 1
    fi
    
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    create_backup_dir
    backup_database
    backup_redis
    backup_configs
    backup_logs
    verify_backups
    upload_to_s3
    cleanup_old_backups
    generate_report
    
    log_success "HLTH backup completed successfully!"
}

# Help function
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Create backups of APG HLTH database and configuration"
    echo ""
    echo "Options:"
    echo "  --namespace NS        Kubernetes namespace (default: hlth)"
    echo "  --backup-dir DIR      Backup directory (default: /tmp/hlth-backups)"
    echo "  --retention-days N    Backup retention in days (default: 30)"
    echo "  --s3-bucket BUCKET    S3 bucket for remote storage"
    echo "  --help               Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  NAMESPACE            Kubernetes namespace"
    echo "  BACKUP_DIR           Local backup directory"
    echo "  RETENTION_DAYS       Backup retention period"
    echo "  S3_BUCKET           S3 bucket name"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Basic backup"
    echo "  $0 --s3-bucket my-backups           # Backup with S3 upload"
    echo "  $0 --retention-days 7               # Keep backups for 7 days"
    echo "  NAMESPACE=hlth-staging $0            # Backup staging environment"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --backup-dir)
            BACKUP_DIR="$2"
            shift 2
            ;;
        --retention-days)
            RETENTION_DAYS="$2"
            shift 2
            ;;
        --s3-bucket)
            S3_BUCKET="$2"
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