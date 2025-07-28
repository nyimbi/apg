# APG Sustainability & ESG Management - Installation Guide

**Version:** 1.0.0  
**Last Updated:** January 2025  
**Minimum APG Version:** 2.0.0

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [System Requirements](#system-requirements)
3. [Installation Methods](#installation-methods)
4. [APG Marketplace Installation](#apg-marketplace-installation)
5. [Manual Installation](#manual-installation)
6. [Docker Installation](#docker-installation)
7. [Kubernetes Deployment](#kubernetes-deployment)
8. [Database Setup](#database-setup)
9. [Configuration](#configuration)
10. [Integration Setup](#integration-setup)
11. [Security Configuration](#security-configuration)
12. [Performance Optimization](#performance-optimization)
13. [Verification](#verification)
14. [Troubleshooting](#troubleshooting)
15. [Upgrade Guide](#upgrade-guide)

---

## Prerequisites

### APG Platform Requirements

- **APG Platform:** Version 2.0.0 or higher
- **APG Capabilities Required:**
  - `auth_rbac` (Authentication & Authorization)
  - `audit_compliance` (Audit Logging)
  - `ai_orchestration` (AI/ML Services)
  - `real_time_collaboration` (Real-time Features)
  - `document_content_management` (Document Storage)

### System Access Requirements

- APG Administrator privileges
- Database administrator access (PostgreSQL)
- Network access to APG services
- SSL certificate for HTTPS (production)

---

## System Requirements

### Minimum Requirements

#### Hardware
- **CPU:** 2 cores, 2.4 GHz
- **RAM:** 4 GB
- **Storage:** 20 GB free space
- **Network:** 10 Mbps bandwidth

#### Software
- **OS:** Linux (Ubuntu 20.04+, CentOS 8+, RHEL 8+)
- **Python:** 3.11 or higher
- **PostgreSQL:** 13 or higher
- **Redis:** 6.0 or higher (for caching)
- **Node.js:** 18+ (for web assets)

### Recommended Requirements

#### Hardware
- **CPU:** 4 cores, 3.0 GHz
- **RAM:** 16 GB
- **Storage:** 100 GB SSD
- **Network:** 100 Mbps bandwidth

#### Software
- **OS:** Ubuntu 22.04 LTS or RHEL 9
- **Python:** 3.12
- **PostgreSQL:** 15
- **Redis:** 7.0
- **Node.js:** 20 LTS

### Production Requirements

#### Hardware
- **CPU:** 8+ cores, 3.2+ GHz
- **RAM:** 32+ GB
- **Storage:** 500+ GB NVMe SSD
- **Network:** 1 Gbps bandwidth
- **Load Balancer:** HAProxy or equivalent

#### Software
- **Container Platform:** Docker 24+ or Kubernetes 1.28+
- **Database:** PostgreSQL 15+ with streaming replication
- **Cache:** Redis Cluster 7.0+
- **Monitoring:** Prometheus + Grafana
- **Log Aggregation:** ELK Stack or equivalent

---

## Installation Methods

The APG Sustainability & ESG Management capability can be installed using several methods:

1. **APG Marketplace** (Recommended)
2. **Manual Installation**
3. **Docker Container**
4. **Kubernetes Deployment**

---

## APG Marketplace Installation

### Step 1: Access APG Marketplace

1. Log into your APG platform as an administrator
2. Navigate to **APG Marketplace** → **Capabilities**
3. Search for **"Sustainability & ESG Management"**
4. Select the official capability by Datacraft

### Step 2: Review Capability Details

```yaml
Capability: sustainability_esg_management
Version: 1.0.0
Publisher: Datacraft
License: Commercial
Dependencies:
  - auth_rbac: ">=2.0.0"
  - audit_compliance: ">=2.0.0"
  - ai_orchestration: ">=2.0.0"
  - real_time_collaboration: ">=2.0.0"
  - document_content_management: ">=2.0.0"
```

### Step 3: Install via APG CLI

```bash
# Login to APG CLI
apg auth login

# Install the capability
apg capability install sustainability_esg_management

# Verify installation
apg capability list | grep sustainability_esg_management
```

### Step 4: Configure Capability

```bash
# Configure initial settings
apg capability configure sustainability_esg_management \
  --database-url "postgresql://esg_user:password@localhost:5432/esg_db" \
  --ai-enabled true \
  --real-time-enabled true

# Enable for tenants
apg capability enable sustainability_esg_management --tenant-id your_tenant_id
```

---

## Manual Installation

### Step 1: Download and Extract

```bash
# Create installation directory
sudo mkdir -p /opt/apg/capabilities
cd /opt/apg/capabilities

# Download capability package
curl -O https://releases.apg.platform/capabilities/sustainability_esg_management-1.0.0.tar.gz

# Extract package
tar -xzf sustainability_esg_management-1.0.0.tar.gz
cd sustainability_esg_management
```

### Step 2: Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y \
  postgresql-client \
  redis-tools \
  nodejs \
  npm

# Install JavaScript dependencies
npm install
```

### Step 3: Database Setup

```bash
# Create database user and database
sudo -u postgres psql << EOF
CREATE USER esg_user WITH PASSWORD 'secure_password';
CREATE DATABASE esg_db OWNER esg_user;
GRANT ALL PRIVILEGES ON DATABASE esg_db TO esg_user;
EOF

# Run database migrations
python manage.py migrate
```

### Step 4: Configure Environment

```bash
# Copy configuration template
cp config/settings.example.py config/settings.py

# Edit configuration
nano config/settings.py
```

### Step 5: Register with APG

```bash
# Register capability with APG platform
python manage.py register_capability \
  --apg-url "https://your-apg-platform.com" \
  --admin-token "your_admin_token"
```

---

## Docker Installation

### Step 1: Pull Docker Image

```bash
# Pull the official image
docker pull apg/sustainability-esg-management:1.0.0

# Or build from source
git clone https://github.com/apg-platform/sustainability-esg-management.git
cd sustainability-esg-management
docker build -t apg/sustainability-esg-management:1.0.0 .
```

### Step 2: Create Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  esg_management:
    image: apg/sustainability-esg-management:1.0.0
    container_name: esg_management
    ports:
      - "8080:8000"
    environment:
      - DATABASE_URL=postgresql://esg_user:password@postgres:5432/esg_db
      - REDIS_URL=redis://redis:6379/0
      - APG_PLATFORM_URL=https://your-apg-platform.com
      - SECRET_KEY=your_secret_key_here
      - DEBUG=false
    depends_on:
      - postgres
      - redis
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    container_name: esg_postgres
    environment:
      - POSTGRES_DB=esg_db
      - POSTGRES_USER=esg_user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    container_name: esg_redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

### Step 3: Start Services

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f esg_management

# Check status
docker-compose ps
```

---

## Kubernetes Deployment

### Step 1: Create Namespace

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: apg-esg-management
  labels:
    name: apg-esg-management
```

```bash
kubectl apply -f namespace.yaml
```

### Step 2: Configure Secrets

```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: esg-secrets
  namespace: apg-esg-management
type: Opaque
data:
  database-url: cG9zdGdyZXNxbDovL2VzZ191c2VyOnBhc3N3b3JkQHBvc3RncmVzOjU0MzIvZXNnX2Ri
  secret-key: eW91cl9zZWNyZXRfa2V5X2hlcmU=
  apg-admin-token: eW91cl9hcGdfYWRtaW5fdG9rZW4=
```

```bash
kubectl apply -f secrets.yaml
```

### Step 3: Deploy Application

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: esg-management
  namespace: apg-esg-management
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: esg-management
  template:
    metadata:
      labels:
        app: esg-management
    spec:
      containers:
      - name: esg-management
        image: apg/sustainability-esg-management:1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: esg-secrets
              key: database-url
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: esg-secrets
              key: secret-key
        - name: REDIS_URL
          value: "redis://redis-service:6379/0"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /api/v1/esg/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/v1/esg/health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: esg-management-service
  namespace: apg-esg-management
spec:
  selector:
    app: esg-management
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```

```bash
kubectl apply -f deployment.yaml
```

### Step 4: Configure Ingress

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: esg-management-ingress
  namespace: apg-esg-management
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - esg.your-domain.com
    secretName: esg-management-tls
  rules:
  - host: esg.your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: esg-management-service
            port:
              number: 80
```

```bash
kubectl apply -f ingress.yaml
```

---

## Database Setup

### PostgreSQL Configuration

#### Create Database and User

```sql
-- Connect as postgres superuser
CREATE USER esg_user WITH PASSWORD 'secure_password_here';
CREATE DATABASE esg_db OWNER esg_user;
GRANT ALL PRIVILEGES ON DATABASE esg_db TO esg_user;

-- Connect to esg_db as esg_user
\c esg_db esg_user

-- Create required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
```

#### Optimize PostgreSQL Settings

```ini
# postgresql.conf
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
work_mem = 4MB
min_wal_size = 1GB
max_wal_size = 4GB
```

#### Run Database Migrations

```bash
# Apply database schema
python manage.py migrate

# Create initial data
python manage.py create_initial_data

# Create indexes for performance
python manage.py create_performance_indexes
```

### Database Security

```sql
-- Revoke public access
REVOKE ALL ON SCHEMA public FROM PUBLIC;
GRANT USAGE ON SCHEMA public TO esg_user;

-- Enable row level security
ALTER TABLE esg_tenant ENABLE ROW LEVEL SECURITY;
ALTER TABLE esg_metric ENABLE ROW LEVEL SECURITY;
ALTER TABLE esg_target ENABLE ROW LEVEL SECURITY;
ALTER TABLE esg_stakeholder ENABLE ROW LEVEL SECURITY;

-- Create RLS policies (example for esg_metric)
CREATE POLICY esg_metric_tenant_isolation ON esg_metric
  FOR ALL TO esg_user
  USING (tenant_id = current_setting('app.current_tenant_id'));
```

---

## Configuration

### Main Configuration File

```python
# config/settings.py

# APG Integration
APG_PLATFORM_URL = "https://your-apg-platform.com"
APG_CAPABILITY_NAME = "sustainability_esg_management"
APG_CAPABILITY_VERSION = "1.0.0"

# Database Configuration
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'esg_db',
        'USER': 'esg_user',
        'PASSWORD': 'secure_password',
        'HOST': 'localhost',
        'PORT': '5432',
        'OPTIONS': {
            'sslmode': 'require',
        }
    }
}

# Cache Configuration
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': 'redis://localhost:6379/0',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        }
    }
}

# ESG Specific Settings
ESG_SETTINGS = {
    'AI_ENABLED': True,
    'REAL_TIME_ENABLED': True,
    'AUTO_MEASUREMENTS': True,
    'STAKEHOLDER_PORTAL_ENABLED': True,
    'SUPPLY_CHAIN_MONITORING': True,
    'REGULATORY_MONITORING': True,
    'DATA_RETENTION_DAYS': 2555,  # 7 years
    'MAX_MEASUREMENTS_PER_METRIC': 10000,
    'AI_PREDICTION_HORIZON_MONTHS': 12,
    'REPORT_GENERATION_TIMEOUT': 300,  # 5 minutes
}

# AI/ML Configuration
AI_ORCHESTRATION_CONFIG = {
    'PREDICTION_MODELS': {
        'ENVIRONMENTAL_METRICS': 'lstm_environmental_v1',
        'SOCIAL_METRICS': 'transformer_social_v1',
        'GOVERNANCE_METRICS': 'ensemble_governance_v1',
    },
    'CONFIDENCE_THRESHOLD': 0.8,
    'RETRAIN_INTERVAL_DAYS': 30,
}

# Real-time Configuration
REAL_TIME_CONFIG = {
    'WEBSOCKET_ENABLED': True,
    'SERVER_SENT_EVENTS': True,
    'PUSH_NOTIFICATIONS': True,
    'CHANNELS': [
        'metrics_updates',
        'targets_progress',
        'ai_insights',
        'stakeholder_activities',
    ]
}

# Security Configuration
SECURITY_SETTINGS = {
    'ENCRYPT_SENSITIVE_DATA': True,
    'AUDIT_ALL_CHANGES': True,
    'SESSION_TIMEOUT_MINUTES': 60,
    'PASSWORD_POLICY': 'strict',
    'TWO_FACTOR_AUTH': False,  # Can be enabled
}

# File Storage Configuration
FILE_STORAGE = {
    'BACKEND': 'django.core.files.storage.FileSystemStorage',
    'LOCATION': '/opt/apg/capabilities/sustainability_esg_management/media',
    'BASE_URL': '/media/',
    'MAX_FILE_SIZE_MB': 100,
    'ALLOWED_EXTENSIONS': ['.pdf', '.xlsx', '.csv', '.png', '.jpg'],
}

# Email Configuration
EMAIL_CONFIG = {
    'BACKEND': 'django.core.mail.backends.smtp.EmailBackend',
    'HOST': 'smtp.your-domain.com',
    'PORT': 587,
    'USE_TLS': True,
    'HOST_USER': 'esg-notifications@your-domain.com',
    'HOST_PASSWORD': 'email_password',
    'DEFAULT_FROM_EMAIL': 'ESG Management <esg-notifications@your-domain.com>',
}

# Logging Configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': '/opt/apg/capabilities/sustainability_esg_management/logs/esg.log',
            'formatter': 'verbose',
        },
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
    },
    'loggers': {
        'sustainability_esg_management': {
            'handlers': ['file', 'console'],
            'level': 'DEBUG',
            'propagate': True,
        },
    },
}
```

### Environment Variables

```bash
# .env
DATABASE_URL=postgresql://esg_user:password@localhost:5432/esg_db
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=your_secret_key_here_make_it_long_and_random
DEBUG=false
APG_PLATFORM_URL=https://your-apg-platform.com
APG_ADMIN_TOKEN=your_apg_admin_token

# Optional AI Configuration
OPENAI_API_KEY=your_openai_api_key  # If using OpenAI models
HUGGINGFACE_API_KEY=your_hf_api_key  # If using HuggingFace models

# Optional External Integrations
WEATHER_API_KEY=your_weather_api_key
EMISSIONS_DB_API_KEY=your_emissions_db_key
```

---

## Integration Setup

### APG Service Integration

#### Authentication & Authorization (auth_rbac)

```python
# Configure ESG-specific roles and permissions
python manage.py setup_esg_permissions

# Create default roles
python manage.py create_esg_roles \
  --roles "esg_admin,esg_manager,esg_analyst,esg_viewer,stakeholder"
```

#### Audit & Compliance (audit_compliance)

```python
# Configure audit settings
python manage.py configure_audit \
  --audit-level "detailed" \
  --retention-days 2555
```

#### AI Orchestration (ai_orchestration)

```python
# Register AI models
python manage.py register_ai_models \
  --model-config config/ai_models.yaml

# Test AI integration
python manage.py test_ai_integration
```

#### Real-time Collaboration (real_time_collaboration)

```python
# Setup real-time channels
python manage.py setup_realtime_channels

# Test WebSocket connectivity
python manage.py test_websocket_connection
```

### External Service Integration

#### Environmental Data APIs

```yaml
# config/external_apis.yaml
environmental_apis:
  weather_service:
    url: "https://api.weatherapi.com/v1"
    api_key: "${WEATHER_API_KEY}"
    endpoints:
      current: "/current.json"
      forecast: "/forecast.json"
  
  emissions_database:
    url: "https://api.epa.gov/easiur"
    api_key: "${EPA_API_KEY}"
    rate_limit: 1000  # requests per hour
```

#### IoT Sensor Integration

```python
# Setup IoT data ingestion
python manage.py setup_iot_integration \
  --mqtt-broker "mqtt.your-company.com" \
  --topics "facilities/+/energy,facilities/+/water,facilities/+/waste"
```

---

## Security Configuration

### SSL/TLS Configuration

```nginx
# nginx configuration for SSL termination
server {
    listen 443 ssl http2;
    server_name esg.your-domain.com;
    
    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    location /ws/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### Firewall Configuration

```bash
# UFW firewall rules
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow from 10.0.0.0/8 to any port 5432  # PostgreSQL (internal network only)
sudo ufw allow from 10.0.0.0/8 to any port 6379  # Redis (internal network only)
sudo ufw --force enable
```

### Database Security

```sql
-- Enable SSL for PostgreSQL connections
ALTER SYSTEM SET ssl = on;
ALTER SYSTEM SET ssl_cert_file = '/path/to/server.crt';
ALTER SYSTEM SET ssl_key_file = '/path/to/server.key';
SELECT pg_reload_conf();

-- Configure connection limits
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET idle_in_transaction_session_timeout = '10min';
```

---

## Performance Optimization

### Database Optimization

```sql
-- Create performance indexes
CREATE INDEX CONCURRENTLY idx_esg_metric_tenant_type ON esg_metric(tenant_id, metric_type);
CREATE INDEX CONCURRENTLY idx_esg_measurement_metric_date ON esg_measurement(metric_id, measurement_date DESC);
CREATE INDEX CONCURRENTLY idx_esg_target_status_date ON esg_target(status, target_date);
CREATE INDEX CONCURRENTLY idx_esg_stakeholder_tenant_type ON esg_stakeholder(tenant_id, stakeholder_type);

-- Analyze tables for query planning
ANALYZE esg_metric;
ANALYZE esg_measurement;
ANALYZE esg_target;
ANALYZE esg_stakeholder;
```

### Application Optimization

```python
# config/performance.py

# Django optimization settings
USE_I18N = False  # If not using internationalization
USE_L10N = False  # If not using localization
USE_TZ = True     # Keep timezone support

# Database connection pooling
DATABASES['default']['CONN_MAX_AGE'] = 60
DATABASES['default']['OPTIONS']['MAX_CONNS'] = 20

# Cache optimization
CACHE_MIDDLEWARE_SECONDS = 300
CACHE_MIDDLEWARE_KEY_PREFIX = 'esg'

# Session optimization
SESSION_ENGINE = 'django.contrib.sessions.backends.cached_db'
SESSION_CACHE_ALIAS = 'default'

# Template optimization
TEMPLATES[0]['OPTIONS']['cached'] = True
```

### Monitoring Setup

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources

volumes:
  grafana_data:
```

---

## Verification

### Installation Verification

```bash
# Check service status
systemctl status esg-management

# Test API endpoints
curl -H "Authorization: Bearer $TOKEN" \
     -H "X-Tenant-ID: $TENANT_ID" \
     https://your-domain.com/api/v1/esg/health

# Verify database connectivity
python manage.py dbshell -c "SELECT count(*) FROM esg_tenant;"

# Test AI integration
python manage.py test_ai_predictions
```

### Functional Testing

```bash
# Run capability tests
python manage.py test sustainability_esg_management.tests

# Run integration tests
python manage.py test sustainability_esg_management.tests.test_integration

# Run performance tests
python manage.py test sustainability_esg_management.tests.test_performance --keepdb
```

### APG Integration Verification

```bash
# Verify APG registration
apg capability status sustainability_esg_management

# Test capability health
apg capability health-check sustainability_esg_management

# Verify permissions
apg auth list-permissions | grep esg
```

---

## Troubleshooting

### Common Installation Issues

#### Database Connection Issues

```bash
# Problem: Cannot connect to PostgreSQL
# Solution: Check connection parameters and firewall
sudo -u postgres psql -c "SELECT version();"
netstat -an | grep 5432
sudo ufw status | grep 5432

# Test connection
psql "postgresql://esg_user:password@localhost:5432/esg_db" -c "SELECT NOW();"
```

#### APG Integration Issues

```bash
# Problem: APG capability registration fails
# Solution: Check APG platform connectivity and tokens
curl -H "Authorization: Bearer $APG_ADMIN_TOKEN" \
     $APG_PLATFORM_URL/api/v1/capabilities

# Verify network connectivity
telnet your-apg-platform.com 443
```

#### Permission Issues

```bash
# Problem: Permission denied errors
# Solution: Check file ownership and APG roles
sudo chown -R esg_user:esg_group /opt/apg/capabilities/sustainability_esg_management
chmod -R 755 /opt/apg/capabilities/sustainability_esg_management

# Check APG permissions
apg auth check-permission --user $USER_ID --resource esg_metrics --action read
```

#### Performance Issues

```bash
# Problem: Slow API responses
# Solution: Check database performance and indexes
psql esg_db -c "SELECT schemaname, tablename, attname, n_distinct, correlation FROM pg_stats WHERE tablename LIKE 'esg_%';"

# Check for missing indexes
psql esg_db -c "SELECT schemaname, tablename FROM pg_tables WHERE tablename LIKE 'esg_%';"
psql esg_db -c "\\di esg_*"
```

### Log Analysis

```bash
# Application logs
tail -f /opt/apg/capabilities/sustainability_esg_management/logs/esg.log

# Database logs
sudo tail -f /var/log/postgresql/postgresql-15-main.log

# System logs
journalctl -u esg-management -f

# Docker logs (if using Docker)
docker logs -f esg_management
```

### Recovery Procedures

#### Database Recovery

```bash
# Backup current database
pg_dump -h localhost -U esg_user esg_db > esg_backup_$(date +%Y%m%d_%H%M%S).sql

# Restore from backup
psql -h localhost -U esg_user esg_db < esg_backup_20250128_103000.sql

# Verify data integrity
python manage.py check_data_integrity
```

#### Service Recovery

```bash
# Restart services
sudo systemctl restart esg-management
sudo systemctl restart postgresql
sudo systemctl restart redis

# Check service dependencies
systemctl list-dependencies esg-management
```

---

## Upgrade Guide

### Preparing for Upgrade

```bash
# Backup current installation
tar -czf esg_backup_$(date +%Y%m%d).tar.gz \
  /opt/apg/capabilities/sustainability_esg_management

# Backup database
pg_dump -h localhost -U esg_user esg_db > esg_db_backup_$(date +%Y%m%d).sql

# Check current version
apg capability info sustainability_esg_management
```

### Upgrade Process

```bash
# Download new version
apg capability upgrade sustainability_esg_management --version 1.1.0

# Run database migrations
python manage.py migrate

# Update configuration if needed
python manage.py update_config --version 1.1.0

# Restart services
sudo systemctl restart esg-management
```

### Post-Upgrade Verification

```bash
# Verify upgrade
apg capability info sustainability_esg_management

# Run health checks
python manage.py health_check

# Test critical functionality
python manage.py test_core_features
```

### Rollback Procedure

```bash
# If upgrade fails, rollback
apg capability rollback sustainability_esg_management --to-version 1.0.0

# Restore database if needed
psql -h localhost -U esg_user esg_db < esg_db_backup_20250128.sql

# Restart services
sudo systemctl restart esg-management
```

---

## Support and Resources

### Documentation Links

- **APG Platform Documentation:** https://docs.apg.platform
- **Capability API Reference:** [API Reference](./api_reference.md)
- **User Guide:** [User Guide](./user_guide.md)
- **Developer Guide:** [Developer Guide](./developer_guide.md)

### Support Channels

- **Technical Support:** support@datacraft.co.ke
- **Community Forum:** https://community.apg.platform/sustainability-esg
- **GitHub Issues:** https://github.com/apg-platform/sustainability-esg-management/issues

### Emergency Contacts

- **Critical Issues:** +254-XXX-XXXX (24/7)
- **Email:** emergency@datacraft.co.ke
- **Slack:** #esg-support (for enterprise customers)

---

**Copyright © 2025 Datacraft - All rights reserved.**  
**Author: Nyimbi Odero <nyimbi@gmail.com>**  
**Website: www.datacraft.co.ke**