# Enterprise Asset Management - Deployment Guide

## Overview

This guide provides comprehensive deployment instructions for the Enterprise Asset Management (EAM) capability within the APG platform ecosystem.

## Prerequisites

### System Requirements

- **Python**: 3.12+ with async support
- **Database**: PostgreSQL 14+ with multi-tenant support
- **Cache**: Redis 6+ for session and query caching
- **Container Runtime**: Docker 20+ or Kubernetes 1.24+
- **APG Platform**: Core services (auth_rbac, audit_compliance, composition_engine)

### Hardware Specifications

#### Minimum Requirements
- **CPU**: 4 cores, 2.4GHz
- **Memory**: 8GB RAM
- **Storage**: 100GB SSD
- **Network**: 1Gbps connection

#### Recommended for Production
- **CPU**: 16 cores, 3.0GHz+
- **Memory**: 32GB RAM
- **Storage**: 500GB NVMe SSD with 10K IOPS
- **Network**: 10Gbps connection with redundancy

## Environment Setup

### 1. Database Preparation

```sql
-- Create EAM database
CREATE DATABASE eam_production 
WITH 
    ENCODING = 'UTF8'
    LC_COLLATE = 'en_US.UTF-8'
    LC_CTYPE = 'en_US.UTF-8'
    TEMPLATE = template0;

-- Create dedicated user
CREATE USER eam_service WITH ENCRYPTED PASSWORD 'secure_password_here';
GRANT ALL PRIVILEGES ON DATABASE eam_production TO eam_service;

-- Enable required extensions
\c eam_production
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
```

### 2. Redis Configuration

```redis
# /etc/redis/redis.conf
bind 127.0.0.1 10.0.0.100
port 6379
maxmemory 4gb
maxmemory-policy allkeys-lru
appendonly yes
appendfsync everysec

# Multi-tenant key namespacing
keyspace-events-config KEA
```

### 3. Environment Variables

```bash
# Database Configuration
export EAM_DATABASE_URL="postgresql://eam_service:secure_password@localhost:5432/eam_production"
export EAM_DATABASE_POOL_SIZE=20
export EAM_DATABASE_MAX_OVERFLOW=40

# Redis Configuration  
export EAM_REDIS_URL="redis://localhost:6379/0"
export EAM_CACHE_PREFIX="eam:prod"
export EAM_SESSION_TIMEOUT=3600

# APG Integration
export APG_COMPOSITION_ENGINE_URL="https://composition.apg.datacraft.co.ke"
export APG_AUTH_SERVICE_URL="https://auth.apg.datacraft.co.ke"
export APG_AUDIT_SERVICE_URL="https://audit.apg.datacraft.co.ke"
export APG_NOTIFICATION_SERVICE_URL="https://notifications.apg.datacraft.co.ke"

# Security
export EAM_SECRET_KEY="your-256-bit-secret-key-here"
export EAM_JWT_SECRET="your-jwt-signing-key-here"
export EAM_ENCRYPTION_KEY="your-aes-256-encryption-key"

# Performance Tuning
export EAM_MAX_WORKERS=8
export EAM_WORKER_TIMEOUT=300
export EAM_MAX_REQUESTS=1000
export EAM_PRELOAD_APP=true

# Feature Flags
export EAM_ENABLE_DIGITAL_TWINS=true
export EAM_ENABLE_PREDICTIVE_MAINTENANCE=true
export EAM_ENABLE_REAL_TIME_COLLABORATION=true
export EAM_ENABLE_MOBILE_SYNC=true

# Monitoring
export EAM_LOG_LEVEL=INFO
export EAM_METRICS_ENABLED=true
export EAM_TRACING_ENABLED=true
export OTEL_SERVICE_NAME="eam-capability"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://jaeger:14268/api/traces"
```

## Deployment Methods

### Method 1: Docker Compose (Development/Staging)

#### docker-compose.yml

```yaml
version: '3.8'

services:
  eam-api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://eam_service:password@postgres:5432/eam_db
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  eam-worker:
    build:
      context: .
      dockerfile: Dockerfile.worker
    environment:
      - DATABASE_URL=postgresql://eam_service:password@postgres:5432/eam_db
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: eam_db
      POSTGRES_USER: eam_service
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./sql/init.sql:/docker-entrypoint-initdb.d/01-init.sql
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - eam-api

volumes:
  postgres_data:
  redis_data:
```

#### Dockerfile

```dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY capabilities/general_cross_functional/enterprise_asset_management ./eam
COPY config/ ./config
COPY migrations/ ./migrations

# Create non-root user
RUN useradd --create-home --shell /bin/bash eam_user
RUN chown -R eam_user:eam_user /app
USER eam_user

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "eam.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Method 2: Kubernetes (Production)

#### Namespace and Resources

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: eam-production
  labels:
    app.kubernetes.io/name: enterprise-asset-management
    app.kubernetes.io/version: "1.0.0"

---
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: eam-config
  namespace: eam-production
data:
  APG_COMPOSITION_ENGINE_URL: "https://composition.apg.datacraft.co.ke"
  APG_AUTH_SERVICE_URL: "https://auth.apg.datacraft.co.ke"
  EAM_LOG_LEVEL: "INFO"
  EAM_MAX_WORKERS: "8"
  EAM_ENABLE_DIGITAL_TWINS: "true"

---
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: eam-secrets
  namespace: eam-production
type: Opaque
data:
  database-url: <base64-encoded-database-url>
  redis-url: <base64-encoded-redis-url>
  secret-key: <base64-encoded-secret-key>
  jwt-secret: <base64-encoded-jwt-secret>
```

#### Deployment Configuration

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: eam-api
  namespace: eam-production
  labels:
    app: eam-api
    version: "1.0.0"
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: eam-api
  template:
    metadata:
      labels:
        app: eam-api
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: eam-api
        image: datacraft/eam-capability:1.0.0
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: eam-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: eam-secrets
              key: redis-url
        envFrom:
        - configMapRef:
            name: eam-config
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        volumeMounts:
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: logs
        emptyDir: {}

---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: eam-api-service
  namespace: eam-production
  labels:
    app: eam-api
spec:
  selector:
    app: eam-api
  ports:
  - name: http
    port: 80
    targetPort: 8000
    protocol: TCP
  type: ClusterIP

---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: eam-api-ingress
  namespace: eam-production
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
  - hosts:
    - eam.apg.datacraft.co.ke
    secretName: eam-tls-cert
  rules:
  - host: eam.apg.datacraft.co.ke
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: eam-api-service
            port:
              number: 80
```

#### Background Workers

```yaml
# worker-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: eam-worker
  namespace: eam-production
spec:
  replicas: 2
  selector:
    matchLabels:
      app: eam-worker
  template:
    metadata:
      labels:
        app: eam-worker
    spec:
      containers:
      - name: eam-worker
        image: datacraft/eam-capability:1.0.0
        command: ["python", "-m", "eam.worker"]
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: eam-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: eam-secrets
              key: redis-url
        envFrom:
        - configMapRef:
            name: eam-config
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "1000m"

---
# cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: eam-maintenance-scheduler
  namespace: eam-production
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: maintenance-scheduler
            image: datacraft/eam-capability:1.0.0
            command: ["python", "-m", "eam.scheduler", "maintenance"]
            env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: eam-secrets
                  key: database-url
          restartPolicy: OnFailure
```

## Database Migrations

### Initial Setup

```bash
# Install Alembic for migrations
pip install alembic

# Initialize migration repository
alembic init migrations

# Generate initial migration
alembic revision --autogenerate -m "Initial EAM schema"

# Apply migrations
alembic upgrade head
```

### Migration Script Example

```python
# migrations/versions/001_initial_schema.py
"""Initial EAM schema

Revision ID: 001
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Create ea_location table
    op.create_table('ea_location',
        sa.Column('location_id', sa.String(36), nullable=False),
        sa.Column('tenant_id', sa.String(36), nullable=False),
        sa.Column('location_code', sa.String(50), nullable=False),
        sa.Column('location_name', sa.String(200), nullable=False),
        sa.Column('location_type', sa.String(50), nullable=False),
        sa.Column('parent_location_id', sa.String(36), nullable=True),
        sa.Column('created_on', sa.DateTime(timezone=True), nullable=False),
        sa.Column('changed_on', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('location_id')
    )
    
    # Create indexes
    op.create_index('idx_ea_location_tenant', 'ea_location', ['tenant_id'])
    op.create_index('idx_ea_location_code', 'ea_location', ['location_code'])
    
    # Create ea_asset table
    op.create_table('ea_asset',
        sa.Column('asset_id', sa.String(36), nullable=False),
        sa.Column('tenant_id', sa.String(36), nullable=False),
        sa.Column('asset_number', sa.String(50), nullable=False),
        sa.Column('asset_name', sa.String(200), nullable=False),
        sa.Column('asset_type', sa.String(50), nullable=False),
        sa.Column('status', sa.String(20), nullable=False),
        sa.Column('health_score', sa.Numeric(5, 2), nullable=True),
        sa.Column('location_id', sa.String(36), nullable=True),
        sa.Column('created_on', sa.DateTime(timezone=True), nullable=False),
        sa.Column('changed_on', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('asset_id'),
        sa.ForeignKeyConstraint(['location_id'], ['ea_location.location_id'])
    )
    
    # Create performance indexes
    op.create_index('idx_ea_asset_tenant_type', 'ea_asset', ['tenant_id', 'asset_type'])
    op.create_index('idx_ea_asset_health_score', 'ea_asset', ['health_score'])
    
def downgrade():
    op.drop_table('ea_asset')
    op.drop_table('ea_location')
```

## Performance Optimization

### Database Tuning

```sql
-- PostgreSQL configuration optimizations
-- postgresql.conf

# Memory Settings
shared_buffers = 8GB                    # 25% of total RAM
effective_cache_size = 24GB             # 75% of total RAM
work_mem = 64MB                         # Per connection sort/hash memory
maintenance_work_mem = 1GB              # Maintenance operations

# Connection Settings
max_connections = 200
max_prepared_transactions = 200

# Query Planner
random_page_cost = 1.1                  # SSD optimization
effective_io_concurrency = 200          # SSD concurrency
default_statistics_target = 100         # Query planning statistics

# Write Ahead Logging
wal_buffers = 16MB
checkpoint_completion_target = 0.7
wal_compression = on

# Performance Monitoring
shared_preload_libraries = 'pg_stat_statements'
track_activity_query_size = 2048
```

### Application Caching

```python
# Redis caching configuration
CACHE_CONFIG = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379/1',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'PARSER_CLASS': 'redis.connection.HiredisParser',
            'PICKLE_VERSION': 2,
        },
        'TIMEOUT': 300,
        'KEY_PREFIX': 'eam:cache',
        'VERSION': 1,
    },
    'sessions': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379/2',
        'TIMEOUT': 3600,
        'KEY_PREFIX': 'eam:session',
    }
}

# Cache warming strategy
async def warm_cache():
    """Warm frequently accessed cache entries"""
    # Dashboard metrics
    await cache_dashboard_metrics()
    
    # Asset hierarchies
    await cache_asset_hierarchies()
    
    # User permissions
    await cache_user_permissions()
```

## Monitoring and Alerting

### Prometheus Metrics

```python
# Custom metrics for EAM capability
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
eam_requests_total = Counter('eam_requests_total', 'Total EAM API requests', ['method', 'endpoint', 'status'])
eam_request_duration = Histogram('eam_request_duration_seconds', 'EAM request duration')

# Business metrics
eam_assets_total = Gauge('eam_assets_total', 'Total number of assets', ['tenant_id', 'asset_type'])
eam_work_orders_pending = Gauge('eam_work_orders_pending', 'Pending work orders', ['tenant_id', 'priority'])
eam_inventory_stockouts = Gauge('eam_inventory_stockouts', 'Items out of stock', ['tenant_id'])

# Health metrics
eam_asset_health_average = Gauge('eam_asset_health_average', 'Average asset health score', ['tenant_id'])
eam_maintenance_overdue = Gauge('eam_maintenance_overdue', 'Overdue maintenance tasks', ['tenant_id'])
```

### Grafana Dashboards

```json
{
  "dashboard": {
    "title": "EAM Capability - Operations Dashboard",
    "panels": [
      {
        "title": "API Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(eam_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Asset Health Distribution",
        "type": "piechart",
        "targets": [
          {
            "expr": "eam_assets_total",
            "legendFormat": "{{asset_type}}"
          }
        ]
      },
      {
        "title": "Work Order Status",
        "type": "stat",
        "targets": [
          {
            "expr": "eam_work_orders_pending",
            "legendFormat": "{{priority}} Priority"
          }
        ]
      }
    ]
  }
}
```

### Alert Rules

```yaml
# alerting-rules.yml
groups:
- name: eam_capability
  rules:
  - alert: EAMHighErrorRate
    expr: rate(eam_requests_total{status=~"5.."}[5m]) > 0.05
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High error rate in EAM capability"
      description: "Error rate is {{ $value }} errors per second"

  - alert: EAMAssetHealthCritical
    expr: eam_asset_health_average < 60
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Critical asset health detected"
      description: "Average asset health is {{ $value }}% for tenant {{ $labels.tenant_id }}"

  - alert: EAMMaintenanceOverdue
    expr: eam_maintenance_overdue > 10
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "Multiple overdue maintenance tasks"
      description: "{{ $value }} maintenance tasks are overdue for tenant {{ $labels.tenant_id }}"
```

## Security Hardening

### SSL/TLS Configuration

```nginx
# nginx.conf - SSL configuration
server {
    listen 443 ssl http2;
    server_name eam.apg.datacraft.co.ke;

    ssl_certificate /etc/nginx/ssl/eam.crt;
    ssl_certificate_key /etc/nginx/ssl/eam.key;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    
    location / {
        proxy_pass http://eam-api-service;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Network Security

```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: eam-network-policy
  namespace: eam-production
spec:
  podSelector:
    matchLabels:
      app: eam-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: apg-core
    ports:
    - protocol: TCP
      port: 443
  - to: []
    ports:
    - protocol: TCP
      port: 5432  # PostgreSQL
    - protocol: TCP
      port: 6379  # Redis
```

## Backup and Recovery

### Database Backup Strategy

```bash
#!/bin/bash
# backup-eam-db.sh

DB_NAME="eam_production"
BACKUP_DIR="/backups/eam"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/eam_backup_$DATE.sql"

# Create backup directory
mkdir -p $BACKUP_DIR

# Perform backup with compression
pg_dump -h localhost -U eam_service -d $DB_NAME \
    --verbose --clean --no-owner --no-privileges \
    --compress=9 > $BACKUP_FILE

# Verify backup
if [ $? -eq 0 ]; then
    echo "Backup completed successfully: $BACKUP_FILE"
    
    # Remove backups older than 30 days
    find $BACKUP_DIR -name "eam_backup_*.sql" -mtime +30 -delete
else
    echo "Backup failed!"
    exit 1
fi
```

### Disaster Recovery

```yaml
# disaster-recovery-plan.yml
recovery_procedures:
  database_restore:
    steps:
      1. "Stop all EAM services"
      2. "Create new database instance"
      3. "Restore from latest backup"
      4. "Update connection strings"
      5. "Start services and verify"
    
  complete_rebuild:
    steps:
      1. "Deploy infrastructure from IaC"
      2. "Restore database from backup"
      3. "Deploy application containers"
      4. "Restore Redis cache data"
      5. "Update DNS and load balancer"
      6. "Perform health checks"

  rollback_procedure:
    steps:
      1. "Identify problematic deployment"
      2. "Scale down new version"
      3. "Scale up previous version"
      4. "Update load balancer weights"
      5. "Verify system functionality"
```

## Troubleshooting

### Common Issues and Solutions

#### Performance Issues

```bash
# Check database performance
psql -d eam_production -c "SELECT query, calls, total_time, mean_time FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"

# Monitor connection pool
curl http://eam-api:8000/debug/pool-status

# Check Redis memory usage
redis-cli info memory
```

#### Connectivity Issues

```bash
# Test APG service connectivity
curl -v https://auth.apg.datacraft.co.ke/health
curl -v https://composition.apg.datacraft.co.ke/health

# Check network policies
kubectl describe networkpolicy eam-network-policy -n eam-production

# Verify DNS resolution
nslookup eam.apg.datacraft.co.ke
```

#### Authentication Problems

```python
# Debug authentication flow
import logging
logging.getLogger('eam.auth').setLevel(logging.DEBUG)

# Test permission checking
await auth_service.check_permission(user_id, "eam.asset.create", tenant_id)

# Verify JWT token
import jwt
decoded = jwt.decode(token, verify=False)
print(f"Token claims: {decoded}")
```

## Maintenance Procedures

### Regular Maintenance Tasks

```bash
# Weekly maintenance script
#!/bin/bash

echo "Starting EAM maintenance procedures..."

# Database maintenance
psql -d eam_production -c "VACUUM ANALYZE;"
psql -d eam_production -c "REINDEX DATABASE eam_production;"

# Clear old cache entries
redis-cli FLUSHDB

# Rotate logs
logrotate /etc/logrotate.d/eam

# Check disk space
df -h /var/lib/postgresql/data
df -h /var/log

# Update statistics
psql -d eam_production -c "ANALYZE;"

echo "Maintenance completed successfully."
```

### Health Check Procedures

```python
async def comprehensive_health_check():
    """Comprehensive health check for EAM capability"""
    
    health_status = {
        "database": await check_database_connectivity(),
        "redis": await check_redis_connectivity(),
        "apg_services": await check_apg_service_health(),
        "disk_space": check_disk_space(),
        "memory_usage": check_memory_usage(),
        "active_connections": get_active_connections(),
        "queue_lengths": get_queue_lengths()
    }
    
    overall_health = all(status.get("healthy", False) for status in health_status.values())
    
    return {
        "healthy": overall_health,
        "timestamp": datetime.utcnow().isoformat(),
        "details": health_status
    }
```

## Contact and Support

### Emergency Contacts

- **Primary**: nyimbi@gmail.com
- **Website**: www.datacraft.co.ke
- **On-call**: +254-XXX-XXXXXX

### Support Channels

- **Documentation**: [APG EAM Docs](https://docs.apg.datacraft.co.ke/eam)
- **Issue Tracking**: [GitHub Issues](https://github.com/datacraft/apg/issues)
- **Community**: [APG Community Forum](https://community.apg.datacraft.co.ke)

---

*This deployment guide is maintained by the APG EAM team. Last updated: 2024-01-01*