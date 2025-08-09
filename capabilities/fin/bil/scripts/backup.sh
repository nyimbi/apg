#!/bin/bash

# APG Billing System Backup Script
# Supports configuration by composition/central_configuration

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load configuration from central configuration if available
if [ -f "/etc/apg/composition/central_configuration" ]; then
    source /etc/apg/composition/central_configuration
    BACKUP_CONFIG_SOURCE="central_configuration"
elif [ -f "$PROJECT_DIR/../../../composition/central_configuration" ]; then
    source "$PROJECT_DIR/../../../composition/central_configuration"
    BACKUP_CONFIG_SOURCE="composition_relative"
elif [ -f "$PROJECT_DIR/.env" ]; then
    export $(grep -v '^#' "$PROJECT_DIR/.env" | xargs)
    BACKUP_CONFIG_SOURCE="local_env"
else
    echo "No configuration found - using defaults"
    BACKUP_CONFIG_SOURCE="defaults"
fi

# Default configuration (can be overridden by central config)
BACKUP_BASE_DIR="${APG_BACKUP_BASE_DIR:-$PROJECT_DIR/backups}"
BACKUP_RETENTION_DAYS="${APG_BACKUP_RETENTION_DAYS:-30}"
DATABASE_URL="${DATABASE_URL:-postgresql://postgres:postgres@localhost:5432/apg_billing}"
S3_BACKUP_BUCKET="${APG_S3_BACKUP_BUCKET:-}"
BACKUP_ENCRYPTION_KEY="${APG_BACKUP_ENCRYPTION_KEY:-}"
SLACK_BACKUP_WEBHOOK="${APG_SLACK_BACKUP_WEBHOOK:-}"
EMAIL_BACKUP_ALERTS="${APG_EMAIL_BACKUP_ALERTS:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# Send notification
send_notification() {
    local message="$1"
    local status="$2"
    
    # Slack notification
    if [ -n "$SLACK_BACKUP_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"APG Billing Backup [$status]: $message\"}" \
            "$SLACK_BACKUP_WEBHOOK" >/dev/null 2>&1 || true
    fi
    
    # Email notification (if configured)
    if [ -n "$EMAIL_BACKUP_ALERTS" ]; then
        echo "$message" | mail -s "APG Billing Backup [$status]" "$EMAIL_BACKUP_ALERTS" 2>/dev/null || true
    fi
}

# Create backup directory
create_backup_dir() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_dir="$BACKUP_BASE_DIR/$timestamp"
    
    mkdir -p "$backup_dir"
    echo "$backup_dir"
}

# Backup database
backup_database() {
    local backup_dir="$1"
    local db_backup_file="$backup_dir/database_backup.sql"
    
    log_info "Backing up database..."
    
    if [[ "$DATABASE_URL" == postgresql://* ]]; then
        # Extract PostgreSQL connection details
        local db_url_no_protocol="${DATABASE_URL#postgresql://}"
        local user_pass_host="${db_url_no_protocol%/*}"
        local database="${db_url_no_protocol##*/}"
        local user_pass="${user_pass_host%@*}"
        local host_port="${user_pass_host##*@}"
        local user="${user_pass%:*}"
        local password="${user_pass##*:}"
        local host="${host_port%:*}"
        local port="${host_port##*:}"
        
        # Use pg_dump for PostgreSQL
        PGPASSWORD="$password" pg_dump -h "$host" -p "$port" -U "$user" -d "$database" > "$db_backup_file"
        
    elif [[ "$DATABASE_URL" == sqlite://* ]]; then
        # SQLite backup
        local db_file="${DATABASE_URL#sqlite:///}"
        if [ -f "$db_file" ]; then
            cp "$db_file" "$backup_dir/sqlite_backup.db"
        fi
    else
        log_error "Unsupported database type: $DATABASE_URL"
        return 1
    fi
    
    # Compress database backup
    gzip "$db_backup_file"
    
    log_success "Database backup completed: ${db_backup_file}.gz"
}

# Backup application data
backup_application_data() {
    local backup_dir="$1"
    
    log_info "Backing up application data..."
    
    # Backup configuration files
    if [ -f "$PROJECT_DIR/.env" ]; then
        cp "$PROJECT_DIR/.env" "$backup_dir/env_backup"
    fi
    
    # Backup logs (last 7 days)
    if [ -d "$PROJECT_DIR/logs" ]; then
        find "$PROJECT_DIR/logs" -name "*.log" -mtime -7 -exec cp {} "$backup_dir/" \;
    fi
    
    # Backup static files and uploads
    if [ -d "$PROJECT_DIR/static/uploads" ]; then
        cp -r "$PROJECT_DIR/static/uploads" "$backup_dir/"
    fi
    
    # Backup migrations
    if [ -d "$PROJECT_DIR/migrations" ]; then
        cp -r "$PROJECT_DIR/migrations" "$backup_dir/"
    fi
    
    # Backup custom configuration from composition layer
    if [ -d "$PROJECT_DIR/../../../composition" ]; then
        mkdir -p "$backup_dir/composition"
        cp -r "$PROJECT_DIR/../../../composition/billing" "$backup_dir/composition/" 2>/dev/null || true
    fi
    
    log_success "Application data backup completed"
}

# Create backup manifest
create_backup_manifest() {
    local backup_dir="$1"
    local manifest_file="$backup_dir/backup_manifest.json"
    
    cat > "$manifest_file" << EOF
{
    "backup_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "backup_version": "1.0",
    "config_source": "$BACKUP_CONFIG_SOURCE",
    "database_url": "${DATABASE_URL%%:*}://***",
    "hostname": "$(hostname)",
    "apg_version": "$(cat $PROJECT_DIR/VERSION 2>/dev/null || echo 'unknown')",
    "files": [
        $(find "$backup_dir" -type f -printf '"%P",\n' | sed '$s/,$//')
    ]
}
EOF
    
    log_success "Backup manifest created"
}

# Encrypt backup
encrypt_backup() {
    local backup_dir="$1"
    
    if [ -n "$BACKUP_ENCRYPTION_KEY" ]; then
        log_info "Encrypting backup..."
        
        cd "$(dirname "$backup_dir")"
        local backup_name="$(basename "$backup_dir")"
        
        tar czf "${backup_name}.tar.gz" "$backup_name"
        
        # Encrypt with OpenSSL
        openssl enc -aes-256-cbc -salt -in "${backup_name}.tar.gz" -out "${backup_name}.tar.gz.enc" -k "$BACKUP_ENCRYPTION_KEY"
        
        # Remove unencrypted files
        rm -rf "$backup_name" "${backup_name}.tar.gz"
        
        log_success "Backup encrypted: ${backup_name}.tar.gz.enc"
        echo "${backup_dir}.tar.gz.enc"
    else
        # Just compress without encryption
        cd "$(dirname "$backup_dir")"
        local backup_name="$(basename "$backup_dir")"
        tar czf "${backup_name}.tar.gz" "$backup_name"
        rm -rf "$backup_name"
        
        log_success "Backup compressed: ${backup_name}.tar.gz"
        echo "${backup_dir}.tar.gz"
    fi
}

# Upload to S3
upload_to_s3() {
    local backup_file="$1"
    
    if [ -n "$S3_BACKUP_BUCKET" ] && command -v aws >/dev/null 2>&1; then
        log_info "Uploading backup to S3..."
        
        local s3_key="apg-billing/$(basename "$backup_file")"
        aws s3 cp "$backup_file" "s3://$S3_BACKUP_BUCKET/$s3_key"
        
        log_success "Backup uploaded to S3: s3://$S3_BACKUP_BUCKET/$s3_key"
    fi
}

# Clean old backups
cleanup_old_backups() {
    log_info "Cleaning up old backups (older than $BACKUP_RETENTION_DAYS days)..."
    
    # Local cleanup
    find "$BACKUP_BASE_DIR" -name "*.tar.gz*" -mtime +$BACKUP_RETENTION_DAYS -delete 2>/dev/null || true
    
    # S3 cleanup (if configured)
    if [ -n "$S3_BACKUP_BUCKET" ] && command -v aws >/dev/null 2>&1; then
        local cutoff_date=$(date -d "$BACKUP_RETENTION_DAYS days ago" +%Y-%m-%d)
        aws s3 ls "s3://$S3_BACKUP_BUCKET/apg-billing/" | while read -r line; do
            local file_date=$(echo "$line" | awk '{print $1}')
            local file_name=$(echo "$line" | awk '{print $4}')
            
            if [[ "$file_date" < "$cutoff_date" ]]; then
                aws s3 rm "s3://$S3_BACKUP_BUCKET/apg-billing/$file_name"
                log_info "Deleted old S3 backup: $file_name"
            fi
        done
    fi
    
    log_success "Old backup cleanup completed"
}

# Verify backup integrity
verify_backup() {
    local backup_file="$1"
    
    log_info "Verifying backup integrity..."
    
    if [[ "$backup_file" == *.enc ]]; then
        # Verify encrypted backup
        if [ -n "$BACKUP_ENCRYPTION_KEY" ]; then
            openssl enc -aes-256-cbc -d -in "$backup_file" -k "$BACKUP_ENCRYPTION_KEY" | tar -tzf - >/dev/null
            if [ $? -eq 0 ]; then
                log_success "Encrypted backup verification passed"
                return 0
            else
                log_error "Encrypted backup verification failed"
                return 1
            fi
        else
            log_error "Cannot verify encrypted backup: no encryption key provided"
            return 1
        fi
    else
        # Verify compressed backup
        tar -tzf "$backup_file" >/dev/null
        if [ $? -eq 0 ]; then
            log_success "Backup verification passed"
            return 0
        else
            log_error "Backup verification failed"
            return 1
        fi
    fi
}

# Restore from backup
restore_backup() {
    local backup_file="$1"
    local restore_dir="${2:-$PROJECT_DIR}"
    
    if [ ! -f "$backup_file" ]; then
        log_error "Backup file not found: $backup_file"
        exit 1
    fi
    
    log_warning "Restoring from backup: $backup_file"
    log_warning "This will overwrite current data. Continue? (y/N)"
    read -r confirmation
    
    if [[ "$confirmation" != "y" && "$confirmation" != "Y" ]]; then
        log_info "Restore cancelled"
        exit 0
    fi
    
    # Create temporary restore directory
    local temp_dir=$(mktemp -d)
    cd "$temp_dir"
    
    if [[ "$backup_file" == *.enc ]]; then
        # Decrypt and extract
        openssl enc -aes-256-cbc -d -in "$backup_file" -k "$BACKUP_ENCRYPTION_KEY" | tar -xzf -
    else
        # Extract compressed backup
        tar -xzf "$backup_file"
    fi
    
    # Find the backup directory
    local backup_dir=$(find . -maxdepth 1 -type d -name "20*" | head -n1)
    
    if [ -z "$backup_dir" ]; then
        log_error "No backup directory found in archive"
        rm -rf "$temp_dir"
        exit 1
    fi
    
    # Restore database
    if [ -f "$backup_dir/database_backup.sql.gz" ]; then
        log_info "Restoring database..."
        gunzip -c "$backup_dir/database_backup.sql.gz" | psql "$DATABASE_URL"
        log_success "Database restored"
    fi
    
    # Restore application files
    if [ -f "$backup_dir/env_backup" ]; then
        cp "$backup_dir/env_backup" "$restore_dir/.env"
        log_success "Configuration restored"
    fi
    
    if [ -d "$backup_dir/uploads" ]; then
        cp -r "$backup_dir/uploads" "$restore_dir/static/"
        log_success "Uploads restored"
    fi
    
    # Cleanup
    rm -rf "$temp_dir"
    
    log_success "Restore completed"
}

# Main backup function
main_backup() {
    log_info "Starting APG Billing System backup..."
    log_info "Configuration source: $BACKUP_CONFIG_SOURCE"
    
    local backup_dir=$(create_backup_dir)
    
    # Perform backup operations
    backup_database "$backup_dir"
    backup_application_data "$backup_dir"
    create_backup_manifest "$backup_dir"
    
    # Encrypt and compress
    local backup_file=$(encrypt_backup "$backup_dir")
    
    # Verify backup
    if verify_backup "$backup_file"; then
        # Upload to remote storage
        upload_to_s3 "$backup_file"
        
        # Cleanup old backups
        cleanup_old_backups
        
        # Send success notification
        send_notification "Backup completed successfully: $(basename "$backup_file")" "SUCCESS"
        
        log_success "Backup completed successfully: $backup_file"
    else
        send_notification "Backup verification failed" "ERROR"
        log_error "Backup failed verification"
        exit 1
    fi
}

# Command handling
case "${1:-backup}" in
    "backup"|"")
        main_backup
        ;;
    "restore")
        if [ -z "$2" ]; then
            log_error "Usage: $0 restore <backup_file> [restore_directory]"
            exit 1
        fi
        restore_backup "$2" "$3"
        ;;
    "list")
        log_info "Available backups:"
        ls -la "$BACKUP_BASE_DIR"/*.tar.gz* 2>/dev/null || log_info "No backups found"
        ;;
    "cleanup")
        cleanup_old_backups
        ;;
    "verify")
        if [ -z "$2" ]; then
            log_error "Usage: $0 verify <backup_file>"
            exit 1
        fi
        verify_backup "$2"
        ;;
    *)
        echo "Usage: $0 {backup|restore|list|cleanup|verify}"
        echo ""
        echo "Commands:"
        echo "  backup                    - Create a new backup"
        echo "  restore <file> [dir]      - Restore from backup file"
        echo "  list                      - List available backups"
        echo "  cleanup                   - Remove old backups"
        echo "  verify <file>            - Verify backup integrity"
        exit 1
        ;;
esac