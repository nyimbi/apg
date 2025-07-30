# APG Document Management - Administrator Guide

**Complete Administrator Guide for Revolutionary Document Management System**

Copyright © 2025 Datacraft  
Author: Nyimbi Odero <nyimbi@gmail.com>  
Website: www.datacraft.co.ke

## Table of Contents

1. [System Overview](#system-overview)
2. [Installation & Setup](#installation--setup)
3. [Configuration Management](#configuration-management)
4. [User & Access Management](#user--access-management)
5. [Security Administration](#security-administration)
6. [Compliance Management](#compliance-management)
7. [Performance Monitoring](#performance-monitoring)
8. [Backup & Recovery](#backup--recovery)
9. [Troubleshooting](#troubleshooting)
10. [Maintenance Procedures](#maintenance-procedures)

## System Overview

### Architecture Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    APG Document Management                       │
├─────────────────────────────────────────────────────────────────┤
│  Load Balancer  │  Web Servers  │  Application Servers          │
├─────────────────────────────────────────────────────────────────┤
│                   Service Layer                                 │
├─────────────┬───────────────┬───────────────┬──────────────────┤
│ IDP Engine  │ Search Engine │ GenAI Engine  │ Predictive Engine│
├─────────────┼───────────────┼───────────────┼──────────────────┤
│Classification│ Retention    │ Content Fabric│  DLP Engine      │
│   Engine    │   Engine     │    Engine     │                  │
├─────────────┴───────────────┴───────────────┴──────────────────┤
│                   APG Integration Layer                         │
├─────────────────────────────────────────────────────────────────┤
│ PostgreSQL │ Redis Cache │ File Storage │ Vector DB │ Blockchain│
└─────────────────────────────────────────────────────────────────┘
```

### Core Services

- **Document Management Service**: Core CRUD operations
- **Intelligent Document Processing (IDP)**: OCR and content extraction
- **Semantic Search Engine**: AI-powered search capabilities
- **Classification Engine**: Automatic document categorization
- **Retention Engine**: Compliance and lifecycle management
- **Generative AI Engine**: Content interaction and enhancement
- **Predictive Analytics**: Risk and value assessment
- **Data Loss Prevention**: Security and compliance monitoring

## Installation & Setup

### Prerequisites

#### System Requirements
- **OS**: Ubuntu 20.04+ / RHEL 8+ / Windows Server 2019+
- **CPU**: 8+ cores (16+ recommended for production)
- **RAM**: 32GB minimum (64GB+ recommended)
- **Storage**: 500GB+ SSD (varies by document volume)
- **Network**: Gigabit Ethernet minimum

#### Software Dependencies
- **Python**: 3.12+
- **PostgreSQL**: 15+
- **Redis**: 7+
- **Nginx**: 1.20+ (load balancer/proxy)
- **Docker**: 24+ (containerized deployment)
- **Kubernetes**: 1.28+ (orchestration, optional)

### Installation Methods

#### Docker Deployment (Recommended)

```bash
# Clone repository
git clone <repository-url>
cd apg/capabilities/general_cross_functional/document_content_management

# Configure environment
cp .env.example .env
nano .env  # Edit configuration

# Deploy services
docker-compose up -d

# Initialize database
docker-compose exec app python scripts/init_db.py

# Create admin user
docker-compose exec app python scripts/create_admin.py
```

#### Manual Installation

```bash
# Install system dependencies
sudo apt update
sudo apt install python3.12 python3.12-venv postgresql redis-server nginx

# Create application user
sudo useradd -m -s /bin/bash apg-docmgmt
sudo su - apg-docmgmt

# Setup application
git clone <repository-url> /home/apg-docmgmt/app
cd /home/apg-docmgmt/app
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure database
sudo -u postgres createdb apg_docmgmt
sudo -u postgres createuser apg_docmgmt
# Set database permissions

# Initialize application
python scripts/init_db.py
python scripts/create_admin.py

# Configure systemd service
sudo cp scripts/apg-docmgmt.service /etc/systemd/system/
sudo systemctl enable apg-docmgmt
sudo systemctl start apg-docmgmt
```

#### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/postgresql.yaml
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/app.yaml
kubectl apply -f k8s/ingress.yaml

# Check deployment status
kubectl get pods -n apg-docmgmt
kubectl get services -n apg-docmgmt
```

## Configuration Management

### Environment Variables

#### Core Settings
```bash
# Database Configuration
DATABASE_URL=postgresql://user:pass@localhost/docmgmt
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_POOL_SIZE=20

# File Storage
STORAGE_BACKEND=filesystem  # or s3, azure, gcp
STORAGE_PATH=/var/lib/apg-docmgmt/documents
MAX_FILE_SIZE=104857600  # 100MB in bytes

# Security
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret
ENCRYPTION_KEY=your-encryption-key

# APG Integration
APG_AI_ENDPOINT=http://apg-ai:8000
APG_RAG_ENDPOINT=http://apg-rag:8001
APG_GENAI_ENDPOINT=http://apg-genai:8002
APG_ML_ENDPOINT=http://apg-ml:8003
APG_BLOCKCHAIN_ENDPOINT=http://apg-blockchain:8004
```

#### Feature Flags
```bash
# AI Features
ENABLE_IDP=true
ENABLE_SEMANTIC_SEARCH=true
ENABLE_CLASSIFICATION=true
ENABLE_GENAI=true
ENABLE_PREDICTIVE_ANALYTICS=true

# Security Features
ENABLE_BLOCKCHAIN_PROVENANCE=true
ENABLE_DLP=true
ENABLE_AUDIT_LOGGING=true

# Performance Features
ENABLE_CACHING=true
ENABLE_ASYNC_PROCESSING=true
CACHE_TTL=3600
```

#### Performance Tuning
```bash
# Processing Limits
MAX_CONCURRENT_UPLOADS=10
MAX_CONCURRENT_PROCESSING=5
IDP_TIMEOUT=300
SEARCH_TIMEOUT=30

# Memory Limits
MAX_MEMORY_PER_WORKER=2048MB
WORKER_PROCESSES=4
WORKER_CONNECTIONS=1000

# Cache Settings
REDIS_CACHE_TTL=3600
QUERY_CACHE_SIZE=1000
VECTOR_CACHE_SIZE=10000
```

### Database Configuration

#### Connection Settings
```sql
-- postgresql.conf optimizations
max_connections = 200
shared_buffers = 8GB
effective_cache_size = 24GB
work_mem = 256MB
maintenance_work_mem = 2GB
wal_buffers = 64MB
checkpoint_completion_target = 0.9
random_page_cost = 1.1
```

#### Table Partitioning
```sql
-- Partition large tables by date
CREATE TABLE dcm_documents_2025 PARTITION OF dcm_documents
FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

-- Partition audit logs by month
CREATE TABLE dcm_audit_log_2025_01 PARTITION OF dcm_audit_log
FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
```

#### Indexing Strategy
```sql
-- Search performance indexes
CREATE INDEX CONCURRENTLY idx_documents_fts ON dcm_documents USING gin(to_tsvector('english', title || ' ' || description || ' ' || keywords));
CREATE INDEX CONCURRENTLY idx_documents_tenant_status ON dcm_documents(tenant_id, status) WHERE is_active = true;
CREATE INDEX CONCURRENTLY idx_documents_created_on ON dcm_documents(created_on DESC);

-- Audit log indexes
CREATE INDEX CONCURRENTLY idx_audit_log_tenant_action ON dcm_audit_log(tenant_id, action, created_at DESC);
CREATE INDEX CONCURRENTLY idx_audit_log_user_date ON dcm_audit_log(user_id, created_at DESC);
```

## User & Access Management

### User Administration

#### Creating Users
```python
# Using admin interface
from service import DocumentManagementService

service = DocumentManagementService()

# Create user
user_data = {
    'username': 'john.doe',
    'email': 'john.doe@company.com',
    'first_name': 'John',
    'last_name': 'Doe',
    'role': 'document_user',
    'tenant_id': 'company_tenant'
}

user = await service.create_user(user_data)
```

#### User Roles
- **System Admin**: Full system administration
- **Tenant Admin**: Tenant-level administration
- **Document Admin**: Document management administration
- **Power User**: Advanced document operations
- **Standard User**: Basic document operations
- **Read Only**: View-only access

#### Bulk User Management
```bash
# Import users from CSV
python scripts/import_users.py --file users.csv --tenant company_tenant

# Export user list
python scripts/export_users.py --tenant company_tenant --format csv

# Bulk role updates
python scripts/update_user_roles.py --file role_updates.csv
```

### Access Control

#### Permission Matrix
| Operation | Read Only | Standard | Power User | Doc Admin | Tenant Admin | System Admin |
|-----------|-----------|----------|------------|-----------|--------------|--------------|
| View Documents | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Upload Documents | ❌ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Edit Metadata | ❌ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Delete Documents | ❌ | Own Only | ✓ | ✓ | ✓ | ✓ |
| Manage Workflows | ❌ | ❌ | ✓ | ✓ | ✓ | ✓ |
| Admin Functions | ❌ | ❌ | ❌ | ✓ | ✓ | ✓ |
| Tenant Config | ❌ | ❌ | ❌ | ❌ | ✓ | ✓ |
| System Config | ❌ | ❌ | ❌ | ❌ | ❌ | ✓ |

#### Row-Level Security
```sql
-- Enable RLS on documents table
ALTER TABLE dcm_documents ENABLE ROW LEVEL SECURITY;

-- Policy for tenant isolation
CREATE POLICY dcm_documents_tenant_policy ON dcm_documents
    FOR ALL TO application_role
    USING (tenant_id = current_setting('app.current_tenant')::text);

-- Policy for user access
CREATE POLICY dcm_documents_user_policy ON dcm_documents
    FOR ALL TO application_role
    USING (
        created_by = current_setting('app.current_user')::text OR
        EXISTS (
            SELECT 1 FROM dcm_permissions p
            WHERE p.document_id = dcm_documents.id
            AND p.user_id = current_setting('app.current_user')::text
            AND p.can_read = true
        )
    );
```

### Multi-Tenancy

#### Tenant Configuration
```python
# Tenant setup
tenant_config = {
    'tenant_id': 'company_tenant',
    'name': 'Company Inc.',
    'domain': 'company.com',
    'settings': {
        'max_storage_gb': 1000,
        'max_users': 500,
        'retention_policy': 'company_standard',
        'security_classification': 'confidential',
        'features_enabled': [
            'ai_processing',
            'semantic_search',
            'blockchain_provenance',
            'dlp_scanning'
        ]
    }
}

await service.create_tenant(tenant_config)
```

#### Tenant Isolation
- **Database**: Row-level security policies
- **Storage**: Separate folder structures
- **Processing**: Tenant-aware queues
- **Analytics**: Isolated metrics and reporting

## Security Administration

### Authentication & Authorization

#### Single Sign-On (SSO)
```yaml
# SAML Configuration
saml:
  enabled: true
  idp_url: https://company.okta.com/app/saml
  sp_entity_id: urn:apg:docmgmt
  certificate_path: /etc/ssl/certs/saml.crt
  private_key_path: /etc/ssl/private/saml.key

# OAuth2/OIDC Configuration
oauth2:
  enabled: true
  provider: azure
  client_id: your-client-id
  client_secret: your-client-secret
  discovery_url: https://login.microsoftonline.com/tenant-id/v2.0/.well-known/openid_configuration
```

#### Multi-Factor Authentication
```python
# Enable MFA for roles
mfa_config = {
    'required_roles': ['system_admin', 'tenant_admin'],
    'optional_roles': ['document_admin', 'power_user'],
    'methods': ['totp', 'sms', 'email'],
    'grace_period_days': 7
}
```

### Data Security

#### Encryption Configuration
```bash
# Document encryption
DOCUMENT_ENCRYPTION=AES-256-GCM
ENCRYPTION_KEY_ROTATION_DAYS=90
KEY_MANAGEMENT_SERVICE=vault  # or aws-kms, azure-keyvault

# Database encryption
DATABASE_SSL_MODE=require
DATABASE_ENCRYPTION_AT_REST=true

# Network encryption
TLS_MIN_VERSION=1.2
HSTS_MAX_AGE=31536000
```

#### Data Loss Prevention (DLP)
```python
# DLP rule configuration
dlp_rules = [
    {
        'name': 'PII Detection',
        'patterns': [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}-\d{4}-\d{4}-\d{4}\b',  # Credit Card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email
        ],
        'actions': ['alert', 'quarantine'],
        'severity': 'high'
    },
    {
        'name': 'Confidential Content',
        'keywords': ['confidential', 'proprietary', 'trade secret'],
        'actions': ['alert', 'restrict_sharing'],
        'severity': 'medium'
    }
]
```

### Audit & Compliance

#### Audit Log Configuration
```python
# Audit events to log
audit_events = [
    'document_created',
    'document_viewed',
    'document_modified',
    'document_deleted',
    'document_shared',
    'user_login',
    'user_logout',
    'permission_granted',
    'permission_revoked',
    'export_initiated',
    'bulk_operation',
    'admin_action'
]

# Audit retention policy
audit_retention = {
    'standard_events': '7_years',
    'security_events': '10_years',
    'admin_events': 'permanent',
    'archival_schedule': 'quarterly'
}
```

#### Compliance Frameworks
```python
# GDPR compliance settings
gdpr_config = {
    'enabled': True,
    'data_retention_days': 2555,  # 7 years
    'right_to_erasure': True,
    'data_portability': True,
    'consent_tracking': True,
    'privacy_by_design': True
}

# HIPAA compliance settings
hipaa_config = {
    'enabled': True,
    'minimum_necessary': True,
    'access_logging': True,
    'encryption_required': True,
    'breach_notification': True
}
```

## Compliance Management

### Retention Policies

#### Policy Configuration
```python
# Standard retention policies
retention_policies = [
    {
        'name': 'Financial Records',
        'document_types': ['invoice', 'receipt', 'financial_report'],
        'retention_years': 7,
        'regulatory_basis': 'SOX',
        'auto_archive': True,
        'auto_delete': False
    },
    {
        'name': 'HR Documents',
        'document_types': ['employee_record', 'payroll'],
        'retention_years': 5,
        'regulatory_basis': 'FLSA',
        'auto_archive': True,
        'auto_delete': True
    },
    {
        'name': 'Healthcare Records',
        'document_types': ['patient_record', 'medical_report'],
        'retention_years': 6,
        'regulatory_basis': 'HIPAA',
        'auto_archive': True,
        'auto_delete': False
    }
]
```

#### Automated Compliance
```bash
# Daily compliance check
0 2 * * * /usr/local/bin/python /opt/apg-docmgmt/scripts/compliance_check.py

# Weekly retention processing
0 3 * * 0 /usr/local/bin/python /opt/apg-docmgmt/scripts/process_retention.py

# Monthly compliance report
0 9 1 * * /usr/local/bin/python /opt/apg-docmgmt/scripts/compliance_report.py
```

### Legal Hold Management

#### Legal Hold Procedures
```python
# Create legal hold
legal_hold = {
    'name': 'Litigation ABC vs Company',
    'description': 'Preserve all documents related to ABC matter',
    'custodians': ['user1@company.com', 'user2@company.com'],
    'keywords': ['ABC Corporation', 'Contract 2023'],
    'date_range': {
        'start': '2023-01-01',
        'end': '2024-12-31'
    },
    'preserve_system_metadata': True,
    'auto_preservation': True
}

await service.create_legal_hold(legal_hold)
```

## Performance Monitoring

### System Metrics

#### Key Performance Indicators
```python
# Performance thresholds
performance_thresholds = {
    'document_upload_time': 30,  # seconds
    'search_response_time': 2,   # seconds
    'ai_processing_time': 60,    # seconds
    'concurrent_users': 1000,
    'storage_utilization': 80,   # percent
    'database_connections': 150,
    'memory_usage': 85,          # percent
    'cpu_utilization': 80        # percent
}
```

#### Monitoring Tools
```yaml
# Prometheus configuration
prometheus:
  enabled: true
  metrics_endpoint: /metrics
  scrape_interval: 15s
  
grafana:
  enabled: true
  dashboard_config: /etc/grafana/dashboards/
  
alertmanager:
  enabled: true
  webhook_url: https://hooks.slack.com/your-webhook
```

### Health Checks

#### Service Health Endpoints
```python
# Health check configuration
health_checks = {
    'database': {
        'query': 'SELECT 1',
        'timeout': 5,
        'critical': True
    },
    'redis': {
        'command': 'ping',
        'timeout': 3,
        'critical': True
    },
    'storage': {
        'test_file': '/tmp/health_check',
        'timeout': 10,
        'critical': True
    },
    'ai_services': {
        'endpoints': [
            'http://apg-ai:8000/health',
            'http://apg-rag:8001/health'
        ],
        'timeout': 15,
        'critical': False
    }
}
```

### Performance Optimization

#### Database Optimization
```sql
-- Regular maintenance
VACUUM ANALYZE dcm_documents;
REINDEX INDEX CONCURRENTLY idx_documents_fts;

-- Statistics update
ANALYZE dcm_documents;
ANALYZE dcm_audit_log;

-- Connection pooling optimization
SELECT * FROM pg_stat_activity WHERE state = 'active';
```

#### Cache Optimization
```python
# Redis cache configuration
cache_config = {
    'search_results_ttl': 300,    # 5 minutes
    'document_metadata_ttl': 3600, # 1 hour
    'user_sessions_ttl': 86400,    # 24 hours
    'ai_predictions_ttl': 7200,    # 2 hours
    'max_memory_policy': 'allkeys-lru'
}
```

## Backup & Recovery

### Backup Strategy

#### Database Backups
```bash
#!/bin/bash
# Full database backup
pg_dump -h localhost -U postgres -d apg_docmgmt > /backup/db/full_$(date +%Y%m%d_%H%M%S).sql

# Incremental WAL backup
pg_receivewal -h localhost -U postgres -D /backup/wal/

# Backup retention
find /backup/db/ -name "*.sql" -mtime +30 -delete
```

#### Document Storage Backup
```bash
#!/bin/bash
# Rsync to backup location
rsync -av --delete /var/lib/apg-docmgmt/documents/ /backup/documents/

# S3 sync for cloud backup
aws s3 sync /var/lib/apg-docmgmt/documents/ s3://backup-bucket/documents/
```

#### Configuration Backup
```bash
#!/bin/bash
# Backup configuration files
tar czf /backup/config/config_$(date +%Y%m%d).tar.gz \
    /etc/apg-docmgmt/ \
    /opt/apg-docmgmt/.env \
    /etc/nginx/sites-available/apg-docmgmt
```

### Disaster Recovery

#### Recovery Procedures
```bash
# Database restore
pg_restore -h localhost -U postgres -d apg_docmgmt /backup/db/full_20250129_120000.sql

# Document restore
rsync -av /backup/documents/ /var/lib/apg-docmgmt/documents/

# Configuration restore
tar xzf /backup/config/config_20250129.tar.gz -C /
```

#### High Availability Setup
```yaml
# Docker Swarm configuration
version: '3.8'
services:
  app:
    image: apg-docmgmt:latest
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
```

## Troubleshooting

### Common Issues

#### Performance Issues
```bash
# Check system resources
top
iostat -x 1
df -h

# Database performance
SELECT * FROM pg_stat_activity WHERE state = 'active';
SELECT * FROM pg_stat_user_tables WHERE relname = 'dcm_documents';

# Application logs
tail -f /var/log/apg-docmgmt/app.log
grep ERROR /var/log/apg-docmgmt/app.log
```

#### Storage Issues
```bash
# Check disk usage
du -sh /var/lib/apg-docmgmt/documents/
find /var/lib/apg-docmgmt/documents/ -type f -size +100M

# Clean temporary files
find /tmp/ -name "apg-docmgmt-*" -mtime +1 -delete

# Check file permissions
ls -la /var/lib/apg-docmgmt/documents/
```

#### AI Service Issues
```bash
# Check AI service connectivity
curl -f http://apg-ai:8000/health
curl -f http://apg-rag:8001/health
curl -f http://apg-genai:8002/health

# Check processing queues
redis-cli -h localhost -p 6379 llen ai_processing_queue
redis-cli -h localhost -p 6379 llen search_indexing_queue
```

### Log Analysis

#### Log Locations
```bash
# Application logs
/var/log/apg-docmgmt/app.log
/var/log/apg-docmgmt/worker.log
/var/log/apg-docmgmt/scheduler.log

# System logs
/var/log/nginx/apg-docmgmt.access.log
/var/log/nginx/apg-docmgmt.error.log
/var/log/postgresql/postgresql.log
```

#### Log Analysis Commands
```bash
# Error analysis
grep ERROR /var/log/apg-docmgmt/app.log | tail -20
awk '/ERROR/ {print $1, $2, $NF}' /var/log/apg-docmgmt/app.log

# Performance analysis
grep "slow query" /var/log/postgresql/postgresql.log
awk '$9 > 2000 {print $0}' /var/log/nginx/apg-docmgmt.access.log

# User activity analysis
grep "login" /var/log/apg-docmgmt/app.log | wc -l
grep "upload" /var/log/apg-docmgmt/app.log | tail -10
```

## Maintenance Procedures

### Regular Maintenance

#### Daily Tasks
```bash
#!/bin/bash
# Daily maintenance script

# Check disk space
df -h | grep -E '9[0-9]%|100%' && echo "Disk space warning"

# Check services
systemctl is-active apg-docmgmt || echo "Service down"

# Process queues
python /opt/apg-docmgmt/scripts/process_queues.py

# Update statistics
python /opt/apg-docmgmt/scripts/update_stats.py
```

#### Weekly Tasks
```bash
#!/bin/bash
# Weekly maintenance script

# Database maintenance
sudo -u postgres vacuumdb -z apg_docmgmt

# Log rotation
logrotate /etc/logrotate.d/apg-docmgmt

# Security scan
python /opt/apg-docmgmt/scripts/security_scan.py

# Performance report
python /opt/apg-docmgmt/scripts/performance_report.py
```

#### Monthly Tasks
```bash
#!/bin/bash
# Monthly maintenance script

# Full database backup
pg_dump apg_docmgmt > /backup/monthly/apg_docmgmt_$(date +%Y%m).sql

# Storage cleanup
python /opt/apg-docmgmt/scripts/storage_cleanup.py

# Compliance report
python /opt/apg-docmgmt/scripts/compliance_report.py

# Update dependencies
pip list --outdated
```

### System Updates

#### Application Updates
```bash
# Backup before update
./scripts/backup.sh full

# Download update
git fetch origin
git checkout v2.1.0

# Update dependencies
pip install -r requirements.txt

# Run migrations
python manage.py migrate

# Restart services
systemctl restart apg-docmgmt

# Verify update
curl -f http://localhost:8000/health
```

#### Security Updates
```bash
# System updates
apt update && apt upgrade -y

# Python package updates
pip install --upgrade pip
pip install -r requirements.txt --upgrade

# Security scan
python scripts/security_scan.py --full
```

---

**Support & Documentation**

- 📧 Admin Support: admin-support@datacraft.co.ke
- 🌐 Admin Portal: [admin.datacraft.co.ke](https://admin.datacraft.co.ke)
- 📖 Technical Docs: [docs.datacraft.co.ke/admin](https://docs.datacraft.co.ke/admin)

**Powered by APG Platform** | **Built with ❤️ by Datacraft**