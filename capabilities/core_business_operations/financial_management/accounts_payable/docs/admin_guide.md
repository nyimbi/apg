# APG Accounts Payable - Administrator Guide

**Version**: 1.0  
**Last Updated**: January 2025  
**© 2025 Datacraft. All rights reserved.**

## Table of Contents

1. [Administrator Overview](#administrator-overview)
2. [Initial System Setup](#initial-system-setup)
3. [User Management](#user-management)
4. [Security Configuration](#security-configuration)
5. [Workflow Configuration](#workflow-configuration)
6. [Integration Management](#integration-management)
7. [Performance Monitoring](#performance-monitoring)
8. [Backup and Recovery](#backup-and-recovery)
9. [Compliance Management](#compliance-management)
10. [Troubleshooting](#troubleshooting)
11. [Maintenance Procedures](#maintenance-procedures)

---

## Administrator Overview

### Prerequisites

**System Requirements**:
- APG Platform version 3.0+
- PostgreSQL 13+ database
- Redis 6+ for caching
- Minimum 8GB RAM (16GB recommended)
- 100GB storage space (SSD recommended)
- Network bandwidth: 100 Mbps minimum

**Administrative Permissions**:
- APG Platform Administrator
- Accounts Payable Admin (`ap.admin`)
- System Configuration (`system.config`)
- User Management (`user.admin`)
- Security Administration (`security.admin`)

### Administrator Dashboard

Access the administrator dashboard via:
```
https://your-apg-instance.com/admin/capabilities/accounts_payable
```

**Dashboard Sections**:
- **System Health**: Performance metrics and status indicators
- **User Activity**: Active sessions and recent actions
- **Configuration Status**: Current settings and pending changes
- **Integration Status**: External service connectivity
- **Security Alerts**: Authentication and authorization events
- **Compliance Dashboard**: Regulatory compliance status

---

## Initial System Setup

### 1. Capability Installation

**Install via APG Platform**:
```bash
# Navigate to APG admin console
cd /opt/apg/admin

# Install accounts payable capability
./apg-admin install-capability core_financials.accounts_payable

# Verify installation
./apg-admin list-capabilities | grep accounts_payable
```

**Configuration File Location**:
```
/opt/apg/config/capabilities/accounts_payable/
├── app_config.yaml
├── database_config.yaml
├── security_config.yaml
├── integration_config.yaml
└── workflow_config.yaml
```

### 2. Database Configuration

**PostgreSQL Setup**:
```sql
-- Create database and user
CREATE DATABASE apg_accounts_payable;
CREATE USER ap_service WITH PASSWORD 'secure_password_here';
GRANT ALL PRIVILEGES ON DATABASE apg_accounts_payable TO ap_service;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "hstore";
```

**Database Configuration** (`database_config.yaml`):
```yaml
database:
  host: localhost
  port: 5432
  name: apg_accounts_payable
  user: ap_service
  password: ${DB_PASSWORD} # Use environment variable
  pool_size: 20
  max_overflow: 30
  echo: false
  ssl_mode: require

cache:
  redis_url: redis://localhost:6379/0
  ttl_seconds: 3600
  max_connections: 50
```

### 3. Initial Data Migration

**Run Migration Scripts**:
```bash
# Execute database migrations
./apg-admin migrate accounts_payable

# Load reference data
./apg-admin load-reference-data accounts_payable

# Verify installation
./apg-admin verify-installation accounts_payable
```

**Reference Data Includes**:
- Chart of accounts templates
- Payment method configurations
- Country and currency data
- Tax code mappings
- Default workflow templates

### 4. Security Configuration

**Authentication Setup** (`security_config.yaml`):
```yaml
authentication:
  method: apg_integrated
  session_timeout: 3600
  max_concurrent_sessions: 5
  password_policy:
    min_length: 12
    require_uppercase: true
    require_lowercase: true
    require_numbers: true
    require_special_chars: true
  
multi_factor_auth:
  enabled: true
  methods: ['totp', 'sms', 'email']
  grace_period_hours: 24

encryption:
  at_rest:
    algorithm: AES-256-GCM
    key_rotation_days: 90
  in_transit:
    tls_version: 1.3
    cipher_suites: ['TLS_AES_256_GCM_SHA384']
```

---

## User Management

### User Roles Configuration

**Default Roles**:

| Role | Permissions | Description |
|------|-------------|-------------|
| `ap_admin` | Full access | System administration |
| `ap_manager` | Read, Write, Approve | Operations management |
| `ap_supervisor` | Read, Write, Limited Approve | Team supervision |
| `ap_clerk` | Read, Write | Data entry and processing |
| `ap_viewer` | Read only | Reporting and viewing |
| `ap_approver` | Read, Approve | Approval workflows only |

**Custom Role Creation**:
```yaml
# roles_config.yaml
custom_roles:
  ap_audit_reviewer:
    name: "AP Audit Reviewer"
    description: "Audit and compliance review access"
    permissions:
      - "ap.read"
      - "ap.audit.access"
      - "ap.compliance.review"
    restrictions:
      - "no_data_modification"
      - "read_only_financial_data"
```

### User Provisioning

**Bulk User Import**:
```bash
# Prepare CSV file with user data
# Format: username,email,first_name,last_name,role,department

# Import users
./apg-admin import-users accounts_payable users.csv

# Verify import
./apg-admin list-users accounts_payable
```

**Individual User Creation**:
```bash
# Create user via CLI
./apg-admin create-user \
  --capability accounts_payable \
  --username john.doe \
  --email john.doe@company.com \
  --role ap_manager \
  --department finance
```

### Permission Management

**Permission Matrix**:
```yaml
# permissions_config.yaml
permission_matrix:
  vendor_management:
    ap.vendor.create: ['ap_admin', 'ap_manager']
    ap.vendor.read: ['ap_admin', 'ap_manager', 'ap_clerk', 'ap_viewer']
    ap.vendor.update: ['ap_admin', 'ap_manager']
    ap.vendor.delete: ['ap_admin']
  
  invoice_processing:
    ap.invoice.create: ['ap_admin', 'ap_manager', 'ap_clerk']
    ap.invoice.approve: ['ap_admin', 'ap_manager', 'ap_approver']
    ap.invoice.read: ['ap_admin', 'ap_manager', 'ap_clerk', 'ap_viewer']
  
  payment_processing:
    ap.payment.create: ['ap_admin', 'ap_manager']
    ap.payment.approve: ['ap_admin', 'ap_manager']
    ap.payment.process: ['ap_admin']
```

---

## Security Configuration

### Access Control

**IP Restrictions**:
```yaml
# security_config.yaml
access_control:
  ip_restrictions:
    enabled: true
    whitelist:
      - "192.168.1.0/24"    # Internal network
      - "10.0.0.0/8"        # Corporate VPN
      - "203.0.113.100"     # Specific admin IP
    blacklist:
      - "192.0.2.0/24"      # Known malicious range
  
  geo_restrictions:
    enabled: true
    allowed_countries: ['US', 'CA', 'GB', 'AU']
    block_tor_exits: true
    block_vpn_providers: false
```

**Session Management**:
```yaml
session:
  timeout_minutes: 60
  absolute_timeout_hours: 8
  concurrent_limit: 3
  idle_warning_minutes: 5
  secure_cookies: true
  same_site: "Strict"
```

### Data Protection

**Field-Level Encryption**:
```yaml
encryption:
  sensitive_fields:
    - "vendor.tax_id"
    - "vendor.bank_account_number"
    - "vendor.routing_number"
    - "payment.account_details"
  
  encryption_keys:
    primary: ${ENCRYPTION_KEY_PRIMARY}
    backup: ${ENCRYPTION_KEY_BACKUP}
    rotation_schedule: "monthly"
```

**Data Masking Rules**:
```yaml
data_masking:
  tax_id:
    pattern: "XX-XXXXXX{last_3}"
    roles_exempt: ['ap_admin', 'compliance_officer']
  
  bank_account:
    pattern: "XXXXXX{last_4}"
    roles_exempt: ['ap_admin', 'payment_processor']
  
  email:
    pattern: "{first_char}***@{domain}"
    roles_exempt: ['ap_admin', 'ap_manager']
```

### Audit Configuration

**Audit Trail Settings**:
```yaml
audit:
  enabled: true
  log_level: "detailed"
  retention_days: 2555  # 7 years
  
  events:
    authentication: true
    authorization: true
    data_access: true
    data_modification: true
    system_changes: true
    security_events: true
  
  storage:
    type: "database"
    encryption: true
    backup_enabled: true
    
  real_time_alerts:
    failed_logins_threshold: 5
    privilege_escalation: true
    data_export_large: true
    after_hours_access: true
```

---

## Workflow Configuration

### Approval Workflows

**Standard Invoice Approval**:
```yaml
# workflow_config.yaml
invoice_approval:
  name: "Standard Invoice Approval"
  trigger: "invoice_created"
  
  steps:
    - step: 1
      name: "Supervisor Review"
      conditions:
        amount_range: [0, 1000]
      approvers:
        type: "hierarchy"
        level: "supervisor"
      timeout_hours: 24
      escalation_target: "manager"
    
    - step: 2
      name: "Manager Approval"
      conditions:
        amount_range: [1001, 10000]
      approvers:
        type: "hierarchy"
        level: "manager"
      timeout_hours: 48
      escalation_target: "director"
    
    - step: 3
      name: "Director Approval"
      conditions:
        amount_range: [10001, 100000]
      approvers:
        type: "role"
        required_role: "director"
      timeout_hours: 72
      escalation_target: "vp_finance"
```

**Payment Approval Workflow**:
```yaml
payment_approval:
  name: "Payment Authorization"
  trigger: "payment_created"
  
  fraud_check:
    enabled: true
    risk_threshold: 0.7
    auto_approve_low_risk: true
    require_dual_approval_high_risk: true
  
  steps:
    - step: 1
      name: "Payment Review"
      conditions:
        payment_method: ["ach", "check"]
        amount_range: [0, 50000]
      approvers:
        type: "segregation_of_duties"
        exclude_creator: true
      parallel_approval: false
    
    - step: 2
      name: "High Value Authorization"
      conditions:
        amount_range: [50001, 999999999]
      approvers:
        type: "dual_approval"
        required_count: 2
        required_roles: ["manager", "director"]
```

### Workflow Customization

**Department-Specific Workflows**:
```yaml
department_workflows:
  IT:
    invoice_approval:
      additional_step:
        name: "Technical Review"
        required_for_categories: ["software", "hardware"]
        approver: "it_manager"
        timeout_hours: 48
  
  Marketing:
    payment_approval:
      expedited_process:
        enabled: true
        conditions:
          vendor_category: "media_agency"
          urgency: "high"
        reduced_approval_levels: 1
```

### Escalation Rules

**Escalation Configuration**:
```yaml
escalation:
  triggers:
    timeout: true
    manual_escalation: true
    system_alerts: true
  
  levels:
    - level: 1
      timeout_hours: 24
      escalate_to: "immediate_supervisor"
      notification_methods: ["email", "sms"]
    
    - level: 2
      timeout_hours: 48
      escalate_to: "department_head"
      notification_methods: ["email", "sms", "phone"]
    
    - level: 3
      timeout_hours: 72
      escalate_to: "executive_team"
      notification_methods: ["email", "executive_dashboard"]
  
  emergency_override:
    enabled: true
    required_role: "cfo"
    audit_mandatory: true
    reason_required: true
```

---

## Integration Management

### Banking Integration

**Supported Banking Partners**:
```yaml
# integration_config.yaml
banking:
  primary_provider: "bank_api_provider"
  backup_provider: "secondary_bank"
  
  ach_processing:
    provider: "nacha_certified_provider"
    settlement_days: 2
    cutoff_time: "14:00:00"
    batch_processing: true
  
  wire_transfers:
    provider: "swift_network"
    same_day_cutoff: "15:00:00"
    international_enabled: true
    compliance_screening: true
  
  real_time_payments:
    rtp_enabled: true
    fednow_enabled: true
    processing_limits:
      per_transaction: 1000000
      daily_limit: 5000000
```

**Connection Configuration**:
```yaml
bank_connections:
  primary_bank:
    connection_type: "api"
    endpoint: "https://api.primarybank.com/v2"
    authentication:
      type: "oauth2"
      client_id: ${BANK_CLIENT_ID}
      client_secret: ${BANK_CLIENT_SECRET}
    
    monitoring:
      health_check_interval: 300  # seconds
      timeout_seconds: 30
      retry_attempts: 3
      circuit_breaker_enabled: true
```

### ERP Integration

**SAP Integration**:
```yaml
erp_integration:
  sap:
    enabled: true
    version: "S/4HANA"
    connection:
      host: "sap.company.com"
      client: "100"
      user: ${SAP_USER}
      password: ${SAP_PASSWORD}
    
    data_sync:
      vendors: true
      gl_accounts: true
      cost_centers: true
      purchase_orders: true
      sync_frequency: "hourly"
```

**Oracle Integration**:
```yaml
  oracle:
    enabled: true
    version: "R12"
    connection:
      jdbc_url: "jdbc:oracle:thin:@oracle.company.com:1521:PROD"
      username: ${ORACLE_USER}
      password: ${ORACLE_PASSWORD}
    
    integration_points:
      ap_invoices: true
      vendor_master: true
      payment_batches: true
      gl_posting: true
```

### AI Service Integration

**Computer Vision Configuration**:
```yaml
ai_services:
  computer_vision:
    provider: "apg_cv_service"
    endpoint: "https://cv.apg.company.com/v1"
    
    ocr_settings:
      confidence_threshold: 0.95
      language_support: ["en", "es", "fr"]
      document_types: ["pdf", "png", "jpg", "tiff"]
      max_file_size_mb: 10
    
    processing_options:
      async_processing: true
      batch_processing: true
      quality_enhancement: true
      table_extraction: true
```

**Federated Learning Configuration**:
```yaml
  federated_learning:
    provider: "apg_fl_service"
    endpoint: "https://fl.apg.company.com/v1"
    
    models:
      cash_flow_forecast:
        model_id: "cash_flow_v2.1"
        update_frequency: "daily"
        confidence_threshold: 0.85
      
      fraud_detection:
        model_id: "fraud_detect_v1.3"
        update_frequency: "real_time"
        risk_threshold: 0.7
```

---

## Performance Monitoring

### System Metrics

**Performance Targets**:
```yaml
# monitoring_config.yaml
performance_targets:
  api_response_time: 200ms
  invoice_processing_time: 2s
  concurrent_users: 1000
  throughput: 500  # operations per minute
  uptime: 99.9%
  data_backup_time: 30min
```

**Monitoring Configuration**:
```yaml
monitoring:
  prometheus:
    enabled: true
    port: 9090
    metrics_path: "/metrics"
    scrape_interval: "15s"
  
  custom_metrics:
    - name: "ap_invoice_processing_time"
      type: "histogram"
      description: "Time to process invoices"
    
    - name: "ap_payment_success_rate"
      type: "gauge"
      description: "Payment processing success rate"
    
    - name: "ap_fraud_detection_rate"
      type: "counter"
      description: "Fraud detection events"
```

### Alerting

**Alert Rules**:
```yaml
alerts:
  critical:
    - name: "System Down"
      condition: "up == 0"
      duration: "1m"
      
    - name: "High Error Rate"
      condition: "error_rate > 0.05"
      duration: "5m"
      
    - name: "Database Connection Failed"
      condition: "db_connections_active == 0"
      duration: "30s"
  
  warning:
    - name: "High Response Time"
      condition: "response_time_95th > 500ms"
      duration: "10m"
      
    - name: "Low Disk Space"
      condition: "disk_usage > 0.8"
      duration: "5m"
```

**Notification Channels**:
```yaml
notifications:
  email:
    enabled: true
    smtp_server: "smtp.company.com"
    recipients: ["admin@company.com", "devops@company.com"]
  
  slack:
    enabled: true
    webhook_url: ${SLACK_WEBHOOK_URL}
    channel: "#apg-alerts"
  
  pagerduty:
    enabled: true
    service_key: ${PAGERDUTY_SERVICE_KEY}
    severity_mapping:
      critical: "critical"
      warning: "warning"
```

### Performance Optimization

**Database Optimization**:
```sql
-- Index optimization for common queries
CREATE INDEX CONCURRENTLY idx_invoices_vendor_date 
ON ap_invoices(vendor_id, invoice_date DESC);

CREATE INDEX CONCURRENTLY idx_payments_status_date 
ON ap_payments(status, payment_date DESC);

-- Partition tables for better performance
CREATE TABLE ap_invoices_2025 PARTITION OF ap_invoices
FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');
```

**Cache Configuration**:
```yaml
caching:
  redis:
    cluster_mode: true
    nodes:
      - "redis-1.company.com:6379"
      - "redis-2.company.com:6379"
      - "redis-3.company.com:6379"
  
  cache_strategies:
    vendor_data:
      ttl: 3600  # 1 hour
      invalidation: "on_update"
    
    exchange_rates:
      ttl: 1800  # 30 minutes
      refresh_strategy: "background"
    
    user_permissions:
      ttl: 900   # 15 minutes
      invalidation: "on_role_change"
```

---

## Backup and Recovery

### Backup Configuration

**Database Backup**:
```yaml
# backup_config.yaml
database_backup:
  schedule: "0 2 * * *"  # Daily at 2 AM
  retention_days: 30
  compression: true
  encryption: true
  
  backup_types:
    full_backup:
      frequency: "weekly"
      day: "sunday"
    
    incremental_backup:
      frequency: "daily"
      exclude_weekends: false
    
    transaction_log_backup:
      frequency: "every_15_minutes"
      retention_hours: 72
```

**File System Backup**:
```yaml
filesystem_backup:
  paths:
    - "/opt/apg/config/accounts_payable/"
    - "/opt/apg/logs/accounts_payable/"
    - "/var/apg/uploads/invoices/"
  
  schedule: "0 3 * * *"  # Daily at 3 AM
  destination: "s3://company-backups/apg/accounts_payable/"
  encryption: true
  compression: true
```

### Disaster Recovery

**Recovery Procedures**:
```bash
#!/bin/bash
# disaster_recovery.sh

# 1. Database Recovery
echo "Starting database recovery..."
pg_restore -h backup-server -U ap_service \
  -d apg_accounts_payable \
  /backups/ap_latest.sql

# 2. File System Recovery
echo "Restoring configuration files..."
aws s3 sync s3://company-backups/apg/accounts_payable/config/ \
  /opt/apg/config/accounts_payable/

# 3. Verify Recovery
echo "Verifying system health..."
./apg-admin health-check accounts_payable

# 4. Restart Services
echo "Restarting services..."
systemctl restart apg-accounts-payable
```

**RTO/RPO Targets**:
```yaml
disaster_recovery:
  rto: "4 hours"      # Recovery Time Objective
  rpo: "15 minutes"   # Recovery Point Objective
  
  testing:
    frequency: "quarterly"
    automated_tests: true
    documentation_required: true
  
  failover:
    automatic: false
    approval_required: true
    rollback_plan: true
```

---

## Compliance Management

### GDPR Configuration

**Data Protection Settings**:
```yaml
# compliance_config.yaml
gdpr:
  data_retention:
    vendor_data: "7 years"
    invoice_data: "7 years"
    payment_data: "7 years"
    user_activity_logs: "2 years"
    audit_trails: "indefinite"
  
  data_subject_rights:
    access_request_sla: "30 days"
    deletion_request_sla: "30 days"
    portability_format: "json"
    
  consent_management:
    required_for: ["marketing", "analytics"]
    withdrawal_mechanism: "self_service"
    audit_trail: true
```

### SOX Compliance

**Internal Controls**:
```yaml
sox_controls:
  segregation_of_duties:
    enabled: true
    prevent_self_approval: true
    require_different_users:
      - "invoice_creation_approval"
      - "payment_creation_approval"
      - "vendor_creation_approval"
  
  audit_requirements:
    digital_signatures: true
    immutable_records: true
    management_certifications: "quarterly"
    
  financial_reporting:
    cutoff_procedures: true
    accrual_validation: true
    variance_analysis: true
```

### Compliance Monitoring

**Automated Compliance Checks**:
```yaml
compliance_monitoring:
  daily_checks:
    - "segregation_of_duties_violations"
    - "unauthorized_access_attempts"
    - "data_retention_policy_compliance"
    - "audit_trail_integrity"
  
  weekly_reports:
    - "access_review_summary"
    - "privilege_changes_report"
    - "exception_handling_review"
  
  monthly_assessments:
    - "full_compliance_review"
    - "risk_assessment_update"
    - "policy_effectiveness_review"
```

---

## Troubleshooting

### Common Issues

**Database Connection Issues**:
```bash
# Check database connectivity
pg_isready -h localhost -p 5432 -U ap_service

# Check active connections
SELECT count(*) FROM pg_stat_activity WHERE datname = 'apg_accounts_payable';

# Monitor slow queries
SELECT query, mean_time, calls 
FROM pg_stat_statements 
WHERE mean_time > 1000 
ORDER BY mean_time DESC;
```

**Performance Issues**:
```bash
# Check system resources
htop
iostat -x 1
sar -u 1

# APG-specific checks
./apg-admin performance-check accounts_payable
./apg-admin cache-stats accounts_payable
./apg-admin connection-pool-status accounts_payable
```

**Integration Issues**:
```bash
# Test banking integration
./apg-admin test-integration banking --provider primary_bank

# Test AI services
./apg-admin test-integration ai --service computer_vision

# Check webhook status
./apg-admin webhook-status accounts_payable
```

### Log Analysis

**Log Locations**:
```
/opt/apg/logs/accounts_payable/
├── application.log        # Main application logs
├── error.log             # Error logs
├── access.log            # API access logs
├── audit.log             # Audit trail
├── performance.log       # Performance metrics
└── integration.log       # External service calls
```

**Log Analysis Commands**:
```bash
# Error analysis
grep "ERROR" /opt/apg/logs/accounts_payable/application.log | tail -50

# Performance issues
grep "slow_query" /opt/apg/logs/accounts_payable/performance.log

# Security events
grep "authentication_failure" /opt/apg/logs/accounts_payable/audit.log
```

### Support Escalation

**Escalation Levels**:

| Level | Response Time | Expertise | Contact Method |
|-------|---------------|-----------|----------------|
| L1 | 4 hours | General support | support@datacraft.co.ke |
| L2 | 2 hours | Technical specialists | tech-support@datacraft.co.ke |
| L3 | 1 hour | Senior engineers | emergency@datacraft.co.ke |
| L4 | 30 minutes | Architecture team | critical@datacraft.co.ke |

---

## Maintenance Procedures

### Regular Maintenance Tasks

**Daily Tasks**:
```bash
#!/bin/bash
# daily_maintenance.sh

# Check system health
./apg-admin health-check accounts_payable

# Update exchange rates
./apg-admin update-exchange-rates

# Process scheduled workflows
./apg-admin process-scheduled-tasks accounts_payable

# Clean temporary files
find /tmp/apg-ap-* -type f -mtime +1 -delete
```

**Weekly Tasks**:
```bash
#!/bin/bash
# weekly_maintenance.sh

# Database maintenance
./apg-admin database-maintenance accounts_payable \
  --vacuum --analyze --reindex

# Clear old cache entries
./apg-admin cache-cleanup accounts_payable

# Generate performance reports
./apg-admin generate-reports performance --week
```

**Monthly Tasks**:
```bash
#!/bin/bash
# monthly_maintenance.sh

# Full system backup verification
./apg-admin verify-backups accounts_payable

# Security assessment
./apg-admin security-scan accounts_payable

# Performance optimization
./apg-admin optimize-performance accounts_payable

# License and capacity review
./apg-admin capacity-planning accounts_payable
```

### Updates and Patches

**Update Process**:
```bash
# 1. Pre-update checklist
./apg-admin pre-update-check accounts_payable

# 2. Backup system
./apg-admin backup accounts_payable --full

# 3. Apply update
./apg-admin update accounts_payable --version latest

# 4. Post-update verification
./apg-admin post-update-verify accounts_payable

# 5. Restart services
systemctl restart apg-accounts-payable
```

**Rollback Procedure**:
```bash
# Emergency rollback
./apg-admin rollback accounts_payable --to-version previous

# Restore from backup if needed
./apg-admin restore accounts_payable --backup-date yesterday
```

---

**Administrator Support:**
- **Documentation**: Updated with each release
- **Training**: Available through APG Learning Center
- **Technical Support**: admin-support@datacraft.co.ke
- **Emergency Contact**: Available 24/7 for critical issues

**© 2025 Datacraft. All rights reserved.**