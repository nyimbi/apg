#!/bin/bash

# APG Central Configuration - Backup and Disaster Recovery Script
# Comprehensive backup automation for production environments

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKUP_TYPE="${BACKUP_TYPE:-full}"
BACKUP_RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-30}"
BACKUP_DIR="${BACKUP_DIR:-/var/backups/central-config}"
S3_BUCKET="${S3_BUCKET:-}"
ENCRYPTION_ENABLED="${ENCRYPTION_ENABLED:-true}"
COMPRESSION_ENABLED="${COMPRESSION_ENABLED:-true}"
NOTIFICATION_WEBHOOK="${NOTIFICATION_WEBHOOK:-}"

# Database connection
DB_HOST="${DB_HOST:-postgres}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-central_config}"
DB_USER="${DB_USER:-cc_admin}"
DB_PASSWORD="${DB_PASSWORD:-}"

# Redis connection
REDIS_HOST="${REDIS_HOST:-redis}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_PASSWORD="${REDIS_PASSWORD:-}"

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
    send_notification "❌ Backup Error" "$1"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Notification function
send_notification() {
    local title="$1"
    local message="$2"
    
    if [[ -n "$NOTIFICATION_WEBHOOK" ]]; then
        curl -X POST "$NOTIFICATION_WEBHOOK" \
            -H "Content-Type: application/json" \
            -d "{\"title\": \"$title\", \"message\": \"$message\", \"timestamp\": \"$(date -Iseconds)\"}" \
            || warn "Failed to send notification"
    fi
}

# Setup backup environment
setup_backup_environment() {
    log "Setting up backup environment..."
    
    # Create backup directory
    mkdir -p "$BACKUP_DIR"
    mkdir -p "$BACKUP_DIR/database"
    mkdir -p "$BACKUP_DIR/redis"
    mkdir -p "$BACKUP_DIR/files"
    mkdir -p "$BACKUP_DIR/configs"
    mkdir -p "$BACKUP_DIR/logs"
    
    # Check dependencies
    local deps=("pg_dump" "redis-cli" "tar" "gzip")
    
    if [[ "$ENCRYPTION_ENABLED" == "true" ]]; then
        deps+=("gpg")
    fi
    
    if [[ -n "$S3_BUCKET" ]]; then
        deps+=("aws")
    fi
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            error "$dep is required but not installed"
        fi
    done
    
    log "✅ Backup environment ready"
}

# Database backup
backup_database() {
    log "Starting database backup..."
    
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local backup_file="$BACKUP_DIR/database/central_config_${timestamp}.sql"
    
    # Set password for pg_dump
    export PGPASSWORD="$DB_PASSWORD"
    
    # Create database dump
    pg_dump \
        --host="$DB_HOST" \
        --port="$DB_PORT" \
        --username="$DB_USER" \
        --dbname="$DB_NAME" \
        --verbose \
        --clean \
        --if-exists \
        --create \
        --format=custom \
        --compress=9 \
        --file="$backup_file"
    
    if [[ $? -eq 0 ]]; then
        log "✅ Database backup completed: $(basename "$backup_file")"
        
        # Get backup size
        local size=$(du -h "$backup_file" | cut -f1)
        info "Database backup size: $size"
        
        # Verify backup integrity
        pg_restore --list "$backup_file" > /dev/null
        if [[ $? -eq 0 ]]; then
            log "✅ Database backup integrity verified"
        else
            error "Database backup integrity check failed"
        fi
    else
        error "Database backup failed"
    fi
    
    # Clean password from environment
    unset PGPASSWORD
    
    echo "$backup_file"
}

# Redis backup
backup_redis() {
    log "Starting Redis backup..."
    
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local backup_file="$BACKUP_DIR/redis/redis_${timestamp}.rdb"
    
    # Create Redis backup using BGSAVE
    if [[ -n "$REDIS_PASSWORD" ]]; then
        redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" BGSAVE
    else
        redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" BGSAVE
    fi
    
    # Wait for background save to complete
    while true; do
        if [[ -n "$REDIS_PASSWORD" ]]; then
            save_status=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" LASTSAVE)
        else
            save_status=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" LASTSAVE)
        fi
        
        sleep 2
        
        if [[ -n "$REDIS_PASSWORD" ]]; then
            current_save=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" LASTSAVE)
        else
            current_save=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" LASTSAVE)
        fi
        
        if [[ "$current_save" -gt "$save_status" ]]; then
            break
        fi
    done
    
    # Copy RDB file from Redis container
    if command -v docker &> /dev/null; then
        docker cp redis:/data/dump.rdb "$backup_file"
    else
        # For Kubernetes deployments
        kubectl cp redis-pod:/data/dump.rdb "$backup_file"
    fi
    
    if [[ -f "$backup_file" ]]; then
        log "✅ Redis backup completed: $(basename "$backup_file")"
        local size=$(du -h "$backup_file" | cut -f1)
        info "Redis backup size: $size"
    else
        error "Redis backup failed"
    fi
    
    echo "$backup_file"
}

# Configuration files backup
backup_configuration_files() {
    log "Starting configuration files backup..."
    
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local backup_file="$BACKUP_DIR/configs/configs_${timestamp}.tar.gz"
    
    # Files to backup
    local config_files=(
        "$PROJECT_ROOT/.env"
        "$PROJECT_ROOT/.env.production"
        "$PROJECT_ROOT/docker-compose.yml"
        "$PROJECT_ROOT/k8s/"
        "$PROJECT_ROOT/monitoring/"
        "$PROJECT_ROOT/scripts/"
    )
    
    # Create tar archive
    tar -czf "$backup_file" \
        --directory="$PROJECT_ROOT" \
        --exclude="*.pyc" \
        --exclude="__pycache__" \
        --exclude=".git" \
        --exclude="node_modules" \
        "${config_files[@]}" 2>/dev/null || true
    
    if [[ -f "$backup_file" ]]; then
        log "✅ Configuration files backup completed: $(basename "$backup_file")"
        local size=$(du -h "$backup_file" | cut -f1)
        info "Configuration backup size: $size"
    else
        error "Configuration files backup failed"
    fi
    
    echo "$backup_file"
}

# Application data backup
backup_application_data() {
    log "Starting application data backup..."
    
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local backup_file="$BACKUP_DIR/files/app_data_${timestamp}.tar.gz"
    
    # Directories to backup
    local data_dirs=()
    
    # Check for persistent volumes or data directories
    if [[ -d "/var/lib/central-config" ]]; then
        data_dirs+=("/var/lib/central-config")
    fi
    
    if [[ -d "/opt/central-config/data" ]]; then
        data_dirs+=("/opt/central-config/data")
    fi
    
    if [[ ${#data_dirs[@]} -gt 0 ]]; then
        tar -czf "$backup_file" "${data_dirs[@]}"
        
        if [[ -f "$backup_file" ]]; then
            log "✅ Application data backup completed: $(basename "$backup_file")"
            local size=$(du -h "$backup_file" | cut -f1)
            info "Application data backup size: $size"
        else
            error "Application data backup failed"
        fi
    else
        info "No application data directories found to backup"
        touch "$backup_file"  # Create empty file to maintain consistency
    fi
    
    echo "$backup_file"
}

# Logs backup
backup_logs() {
    log "Starting logs backup..."
    
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local backup_file="$BACKUP_DIR/logs/logs_${timestamp}.tar.gz"
    
    # Log directories to backup
    local log_dirs=()
    
    if [[ -d "/var/log/central-config" ]]; then
        log_dirs+=("/var/log/central-config")
    fi
    
    # Container logs (last 7 days)
    if command -v docker &> /dev/null; then
        # Export container logs
        local containers=("cc_api" "cc_web" "postgres" "redis" "ollama")
        local logs_dir="$BACKUP_DIR/logs/container_logs_${timestamp}"
        mkdir -p "$logs_dir"
        
        for container in "${containers[@]}"; do
            if docker ps -q -f name="$container" > /dev/null; then
                docker logs --since="7d" "$container" > "$logs_dir/${container}.log" 2>&1 || true
            fi
        done
        
        log_dirs+=("$logs_dir")
    fi
    
    if [[ ${#log_dirs[@]} -gt 0 ]]; then
        tar -czf "$backup_file" "${log_dirs[@]}"
        
        if [[ -f "$backup_file" ]]; then
            log "✅ Logs backup completed: $(basename "$backup_file")"
            local size=$(du -h "$backup_file" | cut -f1)
            info "Logs backup size: $size"
        else
            warn "Logs backup failed, but continuing"
            touch "$backup_file"
        fi
    else
        info "No log directories found to backup"
        touch "$backup_file"
    fi
    
    echo "$backup_file"
}

# Encrypt backup files
encrypt_backup() {
    local file="$1"
    local encrypted_file="${file}.gpg"
    
    if [[ "$ENCRYPTION_ENABLED" == "true" ]]; then
        log "Encrypting backup file: $(basename "$file")"
        
        # Use symmetric encryption with password
        local backup_password="${BACKUP_ENCRYPTION_PASSWORD:-$(openssl rand -base64 32)}"
        
        gpg --batch --yes --passphrase "$backup_password" \
            --cipher-algo AES256 \
            --compress-algo 1 \
            --symmetric \
            --output "$encrypted_file" \
            "$file"
        
        if [[ -f "$encrypted_file" ]]; then
            rm "$file"  # Remove unencrypted file
            log "✅ Backup encrypted: $(basename "$encrypted_file")"
            echo "$encrypted_file"
        else
            error "Backup encryption failed"
        fi
    else
        echo "$file"
    fi
}

# Upload to cloud storage
upload_to_cloud() {
    local file="$1"
    
    if [[ -n "$S3_BUCKET" ]]; then
        log "Uploading backup to S3: $(basename "$file")"
        
        local s3_key="central-config/$(date +%Y/%m/%d)/$(basename "$file")"
        
        aws s3 cp "$file" "s3://$S3_BUCKET/$s3_key" \
            --storage-class STANDARD_IA \
            --server-side-encryption AES256
        
        if [[ $? -eq 0 ]]; then
            log "✅ Backup uploaded to S3: s3://$S3_BUCKET/$s3_key"
        else
            error "Failed to upload backup to S3"
        fi
    fi
}

# Clean old backups
cleanup_old_backups() {
    log "Cleaning up old backups (retention: ${BACKUP_RETENTION_DAYS} days)..."
    
    find "$BACKUP_DIR" -type f -mtime +${BACKUP_RETENTION_DAYS} -delete
    
    # Clean up empty directories
    find "$BACKUP_DIR" -type d -empty -delete 2>/dev/null || true
    
    log "✅ Old backups cleaned up"
}

# Create backup manifest
create_backup_manifest() {
    local manifest_file="$BACKUP_DIR/backup_manifest_$(date +%Y%m%d_%H%M%S).json"
    
    cat > "$manifest_file" <<EOF
{
    "backup_info": {
        "timestamp": "$(date -Iseconds)",
        "type": "$BACKUP_TYPE",
        "hostname": "$(hostname)",
        "version": "1.0.0"
    },
    "database": {
        "host": "$DB_HOST",
        "port": $DB_PORT,
        "name": "$DB_NAME",
        "backup_file": "$(basename "$db_backup_file")"
    },
    "redis": {
        "host": "$REDIS_HOST",
        "port": $REDIS_PORT,
        "backup_file": "$(basename "$redis_backup_file")"
    },
    "files": {
        "configs_backup": "$(basename "$configs_backup_file")",
        "app_data_backup": "$(basename "$app_data_backup_file")",
        "logs_backup": "$(basename "$logs_backup_file")"
    },
    "settings": {
        "encryption_enabled": $ENCRYPTION_ENABLED,
        "compression_enabled": $COMPRESSION_ENABLED,
        "retention_days": $BACKUP_RETENTION_DAYS
    }
}
EOF
    
    log "✅ Backup manifest created: $(basename "$manifest_file")"
}

# Disaster recovery test
test_disaster_recovery() {
    log "Testing disaster recovery procedures..."
    
    # Test database restore (dry run)
    if [[ -f "$db_backup_file" ]]; then
        pg_restore --list "$db_backup_file" > /dev/null
        if [[ $? -eq 0 ]]; then
            log "✅ Database restore test passed"
        else
            error "Database restore test failed"
        fi
    fi
    
    # Test backup file integrity
    local backup_files=("$db_backup_file" "$redis_backup_file" "$configs_backup_file")
    
    for file in "${backup_files[@]}"; do
        if [[ -f "$file" ]]; then
            if [[ "$file" == *.tar.gz ]]; then
                tar -tzf "$file" > /dev/null
                if [[ $? -eq 0 ]]; then
                    log "✅ Archive integrity test passed: $(basename "$file")"
                else
                    error "Archive integrity test failed: $(basename "$file")"
                fi
            fi
        fi
    done
    
    log "✅ Disaster recovery test completed"
}

# Main backup function
perform_backup() {
    local start_time=$(date +%s)
    
    log "🚀 Starting APG Central Configuration Backup"
    log "Backup Type: $BACKUP_TYPE"
    log "Backup Directory: $BACKUP_DIR"
    
    send_notification "🔄 Backup Started" "Central Configuration backup process initiated"
    
    # Setup environment
    setup_backup_environment
    
    # Perform backups
    case "$BACKUP_TYPE" in
        "full")
            db_backup_file=$(backup_database)
            redis_backup_file=$(backup_redis)
            configs_backup_file=$(backup_configuration_files)
            app_data_backup_file=$(backup_application_data)
            logs_backup_file=$(backup_logs)
            ;;
        "database")
            db_backup_file=$(backup_database)
            redis_backup_file=""
            configs_backup_file=""
            app_data_backup_file=""
            logs_backup_file=""
            ;;
        "incremental")
            # Simplified incremental backup (configurations only)
            configs_backup_file=$(backup_configuration_files)
            logs_backup_file=$(backup_logs)
            db_backup_file=""
            redis_backup_file=""
            app_data_backup_file=""
            ;;
        *)
            error "Unknown backup type: $BACKUP_TYPE"
            ;;
    esac
    
    # Encrypt backups
    if [[ "$ENCRYPTION_ENABLED" == "true" ]]; then
        [[ -n "$db_backup_file" ]] && db_backup_file=$(encrypt_backup "$db_backup_file")
        [[ -n "$redis_backup_file" ]] && redis_backup_file=$(encrypt_backup "$redis_backup_file")
        [[ -n "$configs_backup_file" ]] && configs_backup_file=$(encrypt_backup "$configs_backup_file")
        [[ -n "$app_data_backup_file" ]] && app_data_backup_file=$(encrypt_backup "$app_data_backup_file")
        [[ -n "$logs_backup_file" ]] && logs_backup_file=$(encrypt_backup "$logs_backup_file")
    fi
    
    # Upload to cloud
    if [[ -n "$S3_BUCKET" ]]; then
        [[ -n "$db_backup_file" && -f "$db_backup_file" ]] && upload_to_cloud "$db_backup_file"
        [[ -n "$redis_backup_file" && -f "$redis_backup_file" ]] && upload_to_cloud "$redis_backup_file"
        [[ -n "$configs_backup_file" && -f "$configs_backup_file" ]] && upload_to_cloud "$configs_backup_file"
        [[ -n "$app_data_backup_file" && -f "$app_data_backup_file" ]] && upload_to_cloud "$app_data_backup_file"
        [[ -n "$logs_backup_file" && -f "$logs_backup_file" ]] && upload_to_cloud "$logs_backup_file"
    fi
    
    # Create manifest
    create_backup_manifest
    
    # Test disaster recovery
    test_disaster_recovery
    
    # Cleanup old backups
    cleanup_old_backups
    
    # Calculate duration
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Get total backup size
    local total_size=$(du -sh "$BACKUP_DIR" | cut -f1)
    
    log "✅ Backup completed successfully!"
    log "Duration: ${duration}s"
    log "Total backup size: $total_size"
    
    send_notification "✅ Backup Completed" "Central Configuration backup completed successfully in ${duration}s. Total size: $total_size"
}

# Restore function
restore_backup() {
    local backup_date="$1"
    
    if [[ -z "$backup_date" ]]; then
        error "Backup date required for restore operation"
    fi
    
    log "🔄 Starting disaster recovery restore"
    log "Backup date: $backup_date"
    
    warn "THIS WILL OVERWRITE EXISTING DATA. Continue? (y/N)"
    read -r confirmation
    
    if [[ "$confirmation" != "y" && "$confirmation" != "Y" ]]; then
        log "Restore operation cancelled"
        exit 0
    fi
    
    # Find backup files
    local db_backup=$(find "$BACKUP_DIR/database" -name "*${backup_date}*" -type f | head -1)
    local redis_backup=$(find "$BACKUP_DIR/redis" -name "*${backup_date}*" -type f | head -1)
    
    if [[ -z "$db_backup" ]]; then
        error "Database backup not found for date: $backup_date"
    fi
    
    # Restore database
    log "Restoring database from: $(basename "$db_backup")"
    
    # Decrypt if necessary
    if [[ "$db_backup" == *.gpg ]]; then
        local decrypted_file="${db_backup%.gpg}"
        gpg --batch --yes --passphrase "$BACKUP_ENCRYPTION_PASSWORD" \
            --decrypt "$db_backup" > "$decrypted_file"
        db_backup="$decrypted_file"
    fi
    
    export PGPASSWORD="$DB_PASSWORD"
    pg_restore \
        --host="$DB_HOST" \
        --port="$DB_PORT" \
        --username="$DB_USER" \
        --dbname="$DB_NAME" \
        --clean \
        --if-exists \
        --verbose \
        "$db_backup"
    
    if [[ $? -eq 0 ]]; then
        log "✅ Database restore completed"
    else
        error "Database restore failed"
    fi
    
    unset PGPASSWORD
    
    # Restore Redis if backup exists
    if [[ -n "$redis_backup" ]]; then
        log "Restoring Redis from: $(basename "$redis_backup")"
        
        # Decrypt if necessary
        if [[ "$redis_backup" == *.gpg ]]; then
            local decrypted_file="${redis_backup%.gpg}"
            gpg --batch --yes --passphrase "$BACKUP_ENCRYPTION_PASSWORD" \
                --decrypt "$redis_backup" > "$decrypted_file"
            redis_backup="$decrypted_file"
        fi
        
        # Copy RDB file to Redis container
        if command -v docker &> /dev/null; then
            docker cp "$redis_backup" redis:/data/dump.rdb
            docker restart redis
        fi
        
        log "✅ Redis restore completed"
    fi
    
    log "🎉 Disaster recovery restore completed!"
    send_notification "✅ Restore Completed" "Central Configuration restore completed successfully"
}

# Usage function
usage() {
    cat << EOF
Usage: $0 [COMMAND] [OPTIONS]

APG Central Configuration Backup and Disaster Recovery

COMMANDS:
    backup              Perform backup operation
    restore DATE        Restore from backup (format: YYYYMMDD)
    list                List available backups
    test                Test backup integrity
    cleanup             Clean old backups

OPTIONS:
    --type TYPE         Backup type (full|database|incremental) [default: full]
    --retention DAYS    Backup retention in days [default: 30]
    --encrypt           Enable encryption [default: true]
    --s3-bucket BUCKET  S3 bucket for cloud backup
    --help              Show this help message

EXAMPLES:
    $0 backup --type full --s3-bucket my-backups
    $0 restore 20250130
    $0 cleanup --retention 7

ENVIRONMENT VARIABLES:
    BACKUP_TYPE                 Backup type
    BACKUP_RETENTION_DAYS       Retention period
    BACKUP_DIR                  Local backup directory
    S3_BUCKET                   S3 bucket name
    ENCRYPTION_ENABLED          Enable encryption
    BACKUP_ENCRYPTION_PASSWORD  Encryption password
    NOTIFICATION_WEBHOOK        Webhook for notifications

EOF
}

# Parse command line arguments
COMMAND="${1:-backup}"

case "$COMMAND" in
    "backup")
        shift
        while [[ $# -gt 0 ]]; do
            case $1 in
                --type)
                    BACKUP_TYPE="$2"
                    shift 2
                    ;;
                --retention)
                    BACKUP_RETENTION_DAYS="$2"
                    shift 2
                    ;;
                --encrypt)
                    ENCRYPTION_ENABLED="true"
                    shift
                    ;;
                --s3-bucket)
                    S3_BUCKET="$2"
                    shift 2
                    ;;
                *)
                    usage
                    exit 1
                    ;;
            esac
        done
        perform_backup
        ;;
    "restore")
        if [[ -z "${2:-}" ]]; then
            error "Backup date required for restore"
        fi
        restore_backup "$2"
        ;;
    "list")
        log "Available backups:"
        find "$BACKUP_DIR" -name "*.sql" -o -name "*.rdb" -o -name "*.tar.gz" | sort
        ;;
    "test")
        test_disaster_recovery
        ;;
    "cleanup")
        cleanup_old_backups
        ;;
    "help"|"--help")
        usage
        ;;
    *)
        error "Unknown command: $COMMAND"
        ;;
esac