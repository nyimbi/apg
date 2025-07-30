# APG Biometric Authentication - Deployment Guide

Comprehensive deployment guide for production installation of the APG Biometric Authentication capability in enterprise environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation Options](#installation-options)
3. [Production Deployment](#production-deployment)
4. [Configuration Management](#configuration-management)
5. [Security Setup](#security-setup)
6. [Monitoring and Logging](#monitoring-and-logging)
7. [Scaling and Performance](#scaling-and-performance)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

#### Minimum Requirements
- **OS**: Linux (Ubuntu 20.04+, CentOS 8+, RHEL 8+)
- **CPU**: 4 cores, 2.4GHz
- **RAM**: 8GB
- **Storage**: 50GB SSD
- **Network**: 1Gbps
- **Python**: 3.8+
- **Database**: PostgreSQL 12+

#### Recommended Requirements
- **OS**: Ubuntu 22.04 LTS or RHEL 9
- **CPU**: 8+ cores, 3.0GHz+
- **RAM**: 16GB+
- **Storage**: 100GB+ NVMe SSD
- **Network**: 10Gbps
- **Python**: 3.11+
- **Database**: PostgreSQL 15+

#### High-Availability Requirements
- **Instances**: 3+ application servers
- **Load Balancer**: HAProxy, Nginx, or cloud LB
- **Database**: PostgreSQL cluster with replication
- **Storage**: Shared storage or distributed filesystem
- **Monitoring**: Prometheus + Grafana

### Dependencies

#### System Dependencies
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    python3.11 python3.11-dev python3.11-venv \
    postgresql-client libpq-dev \
    build-essential cmake \
    libopencv-dev \
    libdlib-dev \
    portaudio19-dev \
    redis-server \
    nginx

# RHEL/CentOS
sudo dnf install -y \
    python3.11 python3.11-devel \
    postgresql-devel \
    gcc gcc-c++ cmake \
    opencv-devel \
    dlib-devel \
    portaudio-devel \
    redis \
    nginx
```

#### Python Dependencies
```bash
# Core requirements
pip install -r requirements.txt

# Production requirements
pip install -r requirements-prod.txt

# Optional GPU acceleration
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Installation Options

### Option 1: Package Installation

```bash
# Install from PyPI
pip install apg-biometric-auth

# Or install specific version
pip install apg-biometric-auth==1.0.0
```

### Option 2: Source Installation

```bash
# Clone repository
git clone https://github.com/datacraft/apg-biometric.git
cd apg-biometric

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e .

# Install production dependencies
pip install -r requirements-prod.txt
```

### Option 3: Docker Installation

```bash
# Pull official image
docker pull datacraft/apg-biometric:1.0.0

# Or build from source
docker build -t apg-biometric .
```

### Option 4: Kubernetes Deployment

```bash
# Deploy using Helm
helm repo add datacraft https://charts.datacraft.co.ke
helm install apg-biometric datacraft/apg-biometric

# Or apply manifests directly
kubectl apply -f k8s/
```

## Production Deployment

### Database Setup

#### PostgreSQL Configuration

```sql
-- Create database and user
CREATE DATABASE apg_biometric;
CREATE USER apg_biometric_user WITH PASSWORD 'secure_password_here';
GRANT ALL PRIVILEGES ON DATABASE apg_biometric TO apg_biometric_user;

-- Enable required extensions
\c apg_biometric;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create schema
CREATE SCHEMA IF NOT EXISTS biometric AUTHORIZATION apg_biometric_user;
```

#### Database Optimization

```sql
-- postgresql.conf optimizations
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
```

### Application Configuration

#### Environment Configuration

```bash
# /etc/apg-biometric/config.env
export APG_BIOMETRIC_ENV=production
export APG_BIOMETRIC_SECRET_KEY="your-secret-key-here"
export APG_BIOMETRIC_DATABASE_URL="postgresql://apg_biometric_user:password@localhost:5432/apg_biometric"
export APG_BIOMETRIC_REDIS_URL="redis://localhost:6379/0"

# Security settings
export APG_BIOMETRIC_ENCRYPTION_KEY="your-encryption-key-here"
export APG_BIOMETRIC_JWT_SECRET="your-jwt-secret-here"

# Performance settings
export APG_BIOMETRIC_WORKERS=4
export APG_BIOMETRIC_MAX_CONNECTIONS=100
export APG_BIOMETRIC_TIMEOUT=30

# Biometric engine settings
export APG_BIOMETRIC_FINGERPRINT_QUALITY_THRESHOLD=0.8
export APG_BIOMETRIC_FACE_QUALITY_THRESHOLD=0.85
export APG_BIOMETRIC_VOICE_QUALITY_THRESHOLD=0.75
```

#### Application Configuration File

```yaml
# /etc/apg-biometric/config.yaml
database:
  url: ${APG_BIOMETRIC_DATABASE_URL}
  pool_size: 20
  max_overflow: 30
  pool_timeout: 30
  pool_recycle: 3600

redis:
  url: ${APG_BIOMETRIC_REDIS_URL}
  pool_size: 10
  socket_timeout: 5
  socket_connect_timeout: 5

biometric_engines:
  fingerprint:
    quality_threshold: 0.8
    processing_timeout: 5.0
    template_encryption: true
    liveness_detection: true
  
  iris:
    quality_threshold: 0.85
    processing_timeout: 4.0
    segmentation_accuracy: high
  
  palm:
    quality_threshold: 0.8
    hand_detection_confidence: 0.9
  
  voice:
    quality_threshold: 0.75
    anti_spoofing: true
    noise_reduction: true
  
  gait:
    quality_threshold: 0.7
    analysis_window: 15.0

security:
  encryption_algorithm: AES-256-GCM
  key_rotation_days: 90
  session_timeout: 3600
  max_failed_attempts: 3
  lockout_duration: 900

performance:
  max_workers: 8
  worker_timeout: 30
  queue_size: 1000
  cache_ttl: 300

logging:
  level: INFO
  format: json
  file: /var/log/apg-biometric/app.log
  max_size: 100MB
  backup_count: 10

monitoring:
  metrics_enabled: true
  metrics_port: 9090
  health_check_interval: 30
  performance_tracking: true
```

### Service Configuration

#### Systemd Service

```ini
# /etc/systemd/system/apg-biometric.service
[Unit]
Description=APG Biometric Authentication Service
After=network.target postgresql.service redis.service
Wants=postgresql.service redis.service

[Service]
Type=notify
User=apg-biometric
Group=apg-biometric
WorkingDirectory=/opt/apg-biometric
Environment=PYTHONPATH=/opt/apg-biometric
EnvironmentFile=/etc/apg-biometric/config.env
ExecStart=/opt/apg-biometric/venv/bin/python -m apg_biometric.server
ExecReload=/bin/kill -HUP $MAINPID
KillMode=mixed
TimeoutStopSec=5
PrivateTmp=true
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### Nginx Configuration

```nginx
# /etc/nginx/sites-available/apg-biometric
upstream apg_biometric {
    least_conn;
    server 127.0.0.1:8000 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8001 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8002 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8003 max_fails=3 fail_timeout=30s;
}

server {
    listen 443 ssl http2;
    server_name biometric.yourdomain.com;

    ssl_certificate /etc/ssl/certs/biometric.yourdomain.com.pem;
    ssl_certificate_key /etc/ssl/private/biometric.yourdomain.com.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;

    client_max_body_size 10M;
    
    location / {
        proxy_pass http://apg_biometric;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }
    
    location /api/v1/biometric/health {
        proxy_pass http://apg_biometric;
        access_log off;
    }
    
    location /metrics {
        proxy_pass http://apg_biometric;
        allow 10.0.0.0/8;
        allow 172.16.0.0/12;
        allow 192.168.0.0/16;
        deny all;
    }
}

server {
    listen 80;
    server_name biometric.yourdomain.com;
    return 301 https://$server_name$request_uri;
}
```

### Docker Deployment

#### Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: apg_biometric
      POSTGRES_USER: apg_biometric_user
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    restart: unless-stopped

  apg-biometric:
    image: datacraft/apg-biometric:1.0.0
    environment:
      APG_BIOMETRIC_ENV: production
      APG_BIOMETRIC_DATABASE_URL: postgresql://apg_biometric_user:${POSTGRES_PASSWORD}@postgres:5432/apg_biometric
      APG_BIOMETRIC_REDIS_URL: redis://redis:6379/0
      APG_BIOMETRIC_SECRET_KEY: ${SECRET_KEY}
      APG_BIOMETRIC_ENCRYPTION_KEY: ${ENCRYPTION_KEY}
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis
    volumes:
      - ./config.yaml:/app/config.yaml
      - biometric_templates:/app/data
    restart: unless-stopped
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
          cpus: '1.0'

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl
    depends_on:
      - apg-biometric
    restart: unless-stopped

volumes:
  postgres_data:
  biometric_templates:
```

### Kubernetes Deployment

#### Helm Values

```yaml
# values.prod.yaml
replicaCount: 3

image:
  repository: datacraft/apg-biometric
  tag: "1.0.0"
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 8000

ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/client-max-body-size: "10m"
  hosts:
    - host: biometric.yourdomain.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: biometric-tls
      hosts:
        - biometric.yourdomain.com

postgresql:
  enabled: true
  auth:
    database: apg_biometric
    username: apg_biometric_user
    password: "secure_password_here"
  primary:
    persistence:
      size: 100Gi
    resources:
      requests:
        memory: 512Mi
        cpu: 500m
      limits:
        memory: 1Gi
        cpu: 1000m

redis:
  enabled: true
  auth:
    enabled: false
  master:
    persistence:
      size: 10Gi

resources:
  requests:
    memory: 1Gi
    cpu: 500m
  limits:
    memory: 2Gi
    cpu: 1000m

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

nodeSelector: {}

tolerations: []

affinity:
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      podAffinityTerm:
        labelSelector:
          matchExpressions:
          - key: app.kubernetes.io/name
            operator: In
            values:
            - apg-biometric
        topologyKey: kubernetes.io/hostname
```

## Configuration Management

### Environment-Specific Configurations

#### Development Environment

```yaml
# config/development.yaml
database:
  url: postgresql://dev_user:dev_pass@localhost:5432/apg_biometric_dev
  echo: true

logging:
  level: DEBUG
  format: text

biometric_engines:
  fingerprint:
    quality_threshold: 0.6
  iris:
    quality_threshold: 0.7

security:
  encryption_algorithm: AES-256-GCM
  session_timeout: 7200
```

#### Testing Environment

```yaml
# config/testing.yaml
database:
  url: postgresql://test_user:test_pass@localhost:5432/apg_biometric_test

biometric_engines:
  fingerprint:
    quality_threshold: 0.5
    processing_timeout: 1.0

performance:
  max_workers: 2
  worker_timeout: 10
```

#### Production Environment

```yaml
# config/production.yaml
database:
  url: ${APG_BIOMETRIC_DATABASE_URL}
  pool_size: 50
  max_overflow: 100

logging:
  level: INFO
  format: json

biometric_engines:
  fingerprint:
    quality_threshold: 0.8
    template_encryption: true
    liveness_detection: true

security:
  encryption_algorithm: AES-256-GCM
  key_rotation_days: 30
  session_timeout: 1800

performance:
  max_workers: 16
  worker_timeout: 30
```

### Configuration Validation

```python
# config_validator.py
import yaml
from jsonschema import validate, ValidationError

CONFIG_SCHEMA = {
    "type": "object",
    "required": ["database", "biometric_engines", "security"],
    "properties": {
        "database": {
            "type": "object",
            "required": ["url"],
            "properties": {
                "url": {"type": "string"},
                "pool_size": {"type": "integer", "minimum": 1},
                "max_overflow": {"type": "integer", "minimum": 0}
            }
        },
        "biometric_engines": {
            "type": "object",
            "properties": {
                "fingerprint": {
                    "type": "object",
                    "properties": {
                        "quality_threshold": {"type": "number", "minimum": 0, "maximum": 1}
                    }
                }
            }
        }
    }
}

def validate_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    try:
        validate(instance=config, schema=CONFIG_SCHEMA)
        print("Configuration is valid")
        return True
    except ValidationError as e:
        print(f"Configuration validation error: {e.message}")
        return False
```

## Security Setup

### SSL/TLS Configuration

#### Generate SSL Certificates

```bash
# Using Let's Encrypt with Certbot
sudo certbot certonly --nginx -d biometric.yourdomain.com

# Or using OpenSSL for self-signed (development only)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout /etc/ssl/private/biometric.key \
  -out /etc/ssl/certs/biometric.crt
```

#### SSL Configuration Best Practices

```nginx
# Strong SSL configuration
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384;
ssl_prefer_server_ciphers off;
ssl_session_cache shared:SSL:10m;
ssl_session_timeout 10m;
ssl_session_tickets off;

# HSTS
add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";

# Security headers
add_header X-Frame-Options DENY;
add_header X-Content-Type-Options nosniff;
add_header X-XSS-Protection "1; mode=block";
add_header Referrer-Policy "strict-origin-when-cross-origin";
```

### Database Security

#### Encryption at Rest

```sql
-- Enable encryption at rest
ALTER SYSTEM SET ssl = on;
ALTER SYSTEM SET ssl_cert_file = '/etc/ssl/certs/server.crt';
ALTER SYSTEM SET ssl_key_file = '/etc/ssl/private/server.key';

-- Restart PostgreSQL
SELECT pg_reload_conf();
```

#### Row-Level Security

```sql
-- Enable RLS for biometric templates
ALTER TABLE biometric.templates ENABLE ROW LEVEL SECURITY;

-- Create policy for user isolation
CREATE POLICY user_isolation ON biometric.templates
    FOR ALL
    TO apg_biometric_role
    USING (user_id = current_setting('app.current_user_id'));
```

### Application Security

#### API Key Management

```python
# api_key_manager.py
import secrets
import hashlib
from datetime import datetime, timedelta

class APIKeyManager:
    def generate_api_key(self, user_id: str, permissions: list) -> dict:
        """Generate secure API key with permissions"""
        key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        return {
            'key': key,
            'key_hash': key_hash,
            'user_id': user_id,
            'permissions': permissions,
            'created_at': datetime.utcnow(),
            'expires_at': datetime.utcnow() + timedelta(days=365)
        }
    
    def validate_api_key(self, key: str) -> bool:
        """Validate API key against stored hash"""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        # Check against database
        return self.check_key_hash(key_hash)
```

#### Template Encryption

```python
# template_encryption.py
from cryptography.fernet import Fernet
import base64

class TemplateEncryption:
    def __init__(self, encryption_key: str):
        self.cipher_suite = Fernet(encryption_key.encode())
    
    def encrypt_template(self, template_data: bytes) -> str:
        """Encrypt biometric template"""
        encrypted_data = self.cipher_suite.encrypt(template_data)
        return base64.b64encode(encrypted_data).decode()
    
    def decrypt_template(self, encrypted_template: str) -> bytes:
        """Decrypt biometric template"""
        encrypted_data = base64.b64decode(encrypted_template.encode())
        return self.cipher_suite.decrypt(encrypted_data)
```

### Access Control

#### Role-Based Access Control (RBAC)

```yaml
# rbac_config.yaml
roles:
  admin:
    permissions:
      - biometric:*
      - user:*
      - system:*
  
  security_analyst:
    permissions:
      - biometric:read
      - biometric:verify
      - analytics:read
      - user:read
  
  operator:
    permissions:
      - biometric:verify
      - user:read
  
  viewer:
    permissions:
      - analytics:read
      - user:read

policies:
  department_isolation:
    effect: allow
    condition: user.department == resource.department
  
  time_based_access:
    effect: allow
    condition: time.hour >= 9 AND time.hour <= 17
```

## Monitoring and Logging

### Application Monitoring

#### Prometheus Metrics

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define metrics
verification_requests = Counter('biometric_verification_requests_total', 
                               'Total verification requests', ['modality', 'status'])

verification_duration = Histogram('biometric_verification_duration_seconds',
                                 'Verification processing time', ['modality'])

active_sessions = Gauge('biometric_active_sessions',
                       'Number of active biometric sessions')

template_storage = Gauge('biometric_templates_stored_total',
                        'Total number of stored templates', ['modality'])

# Usage in application
def record_verification(modality: str, duration: float, success: bool):
    status = 'success' if success else 'failure'
    verification_requests.labels(modality=modality, status=status).inc()
    verification_duration.labels(modality=modality).observe(duration)
```

#### Health Check Endpoints

```python
# health_check.py
from flask import jsonify
import asyncio

async def health_check():
    """Comprehensive health check"""
    checks = {
        'database': await check_database_connection(),
        'redis': await check_redis_connection(),
        'biometric_engines': await check_biometric_engines(),
        'disk_space': check_disk_space(),
        'memory_usage': check_memory_usage()
    }
    
    overall_status = 'healthy' if all(checks.values()) else 'unhealthy'
    
    return {
        'status': overall_status,
        'timestamp': datetime.utcnow().isoformat(),
        'checks': checks,
        'version': '1.0.0'
    }

async def check_biometric_engines():
    """Check biometric engine health"""
    try:
        from biometric_engines import BiometricProcessor
        processor = BiometricProcessor()
        
        # Test each engine with dummy data
        test_results = await processor.run_health_checks()
        return all(test_results.values())
    except Exception:
        return False
```

### Logging Configuration

#### Structured Logging

```python
# logging_config.py
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)

# Configure logging
def setup_logging():
    logger = logging.getLogger('apg_biometric')
    logger.setLevel(logging.INFO)
    
    handler = logging.FileHandler('/var/log/apg-biometric/app.log')
    handler.setFormatter(JSONFormatter())
    
    logger.addHandler(handler)
    return logger
```

#### Log Rotation

```bash
# /etc/logrotate.d/apg-biometric
/var/log/apg-biometric/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
    size 100M
}
```

### Monitoring Setup

#### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "APG Biometric Authentication",
    "panels": [
      {
        "title": "Verification Requests",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(biometric_verification_requests_total[5m])",
            "legendFormat": "{{modality}} - {{status}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, biometric_verification_duration_seconds_bucket)",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "rate(biometric_verification_requests_total{status=\"failure\"}[5m]) / rate(biometric_verification_requests_total[5m])",
            "format": "percent"
          }
        ]
      }
    ]
  }
}
```

#### Alerting Rules

```yaml
# prometheus_alerts.yml
groups:
  - name: apg_biometric
    rules:
      - alert: HighErrorRate
        expr: rate(biometric_verification_requests_total{status="failure"}[5m]) / rate(biometric_verification_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: High error rate in biometric verification
          description: "Error rate is {{ $value | humanizePercentage }}"
      
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, biometric_verification_duration_seconds_bucket) > 1.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: High response time for biometric verification
          description: "95th percentile response time is {{ $value }}s"
      
      - alert: ServiceDown
        expr: up{job="apg-biometric"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: APG Biometric service is down
          description: "Service has been down for more than 1 minute"
```

## Scaling and Performance

### Horizontal Scaling

#### Load Balancer Configuration

```nginx
# nginx load balancer
upstream apg_biometric_cluster {
    least_conn;
    
    # Application servers
    server 10.0.1.10:8000 max_fails=3 fail_timeout=30s weight=1;
    server 10.0.1.11:8000 max_fails=3 fail_timeout=30s weight=1;
    server 10.0.1.12:8000 max_fails=3 fail_timeout=30s weight=1;
    server 10.0.1.13:8000 max_fails=3 fail_timeout=30s weight=1;
    
    # Backup server
    server 10.0.1.20:8000 backup;
    
    keepalive 32;
}

# Health check
location /health {
    proxy_pass http://apg_biometric_cluster;
    proxy_set_header Host $host;
    access_log off;
}
```

#### Auto-scaling Configuration

```yaml
# kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: apg-biometric-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: apg-biometric
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

### Performance Optimization

#### Database Optimization

```sql
-- Create indexes for performance
CREATE INDEX CONCURRENTLY idx_biometric_templates_user_modality 
ON biometric.templates (user_id, modality);

CREATE INDEX CONCURRENTLY idx_verification_history_user_date 
ON biometric.verification_history (user_id, created_at);

CREATE INDEX CONCURRENTLY idx_audit_logs_date 
ON biometric.audit_logs (created_at) 
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days';

-- Partitioning for audit logs
CREATE TABLE biometric.audit_logs_2025_01 PARTITION OF biometric.audit_logs
FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

-- Query optimization
ANALYZE biometric.templates;
ANALYZE biometric.verification_history;
```

#### Caching Strategy

```python
# redis_cache.py
import redis
import json
from datetime import timedelta

class BiometricCache:
    def __init__(self, redis_url: str):
        self.redis_client = redis.from_url(redis_url)
    
    async def cache_template(self, user_id: str, modality: str, template: dict):
        """Cache biometric template for fast retrieval"""
        key = f"template:{user_id}:{modality}"
        value = json.dumps(template)
        await self.redis_client.setex(key, timedelta(hours=1), value)
    
    async def get_cached_template(self, user_id: str, modality: str):
        """Retrieve cached template"""
        key = f"template:{user_id}:{modality}"
        cached_value = await self.redis_client.get(key)
        
        if cached_value:
            return json.loads(cached_value)
        return None
    
    async def cache_verification_result(self, verification_id: str, result: dict):
        """Cache verification result for audit trail"""
        key = f"verification:{verification_id}"
        value = json.dumps(result)
        await self.redis_client.setex(key, timedelta(days=1), value)
```

## Troubleshooting

### Common Issues

#### Issue: High Memory Usage

**Symptoms:**
- Memory usage > 80%
- Slow response times
- Out of memory errors

**Solutions:**
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head

# Monitor biometric processes
top -p $(pgrep -f apg-biometric)

# Adjust worker settings
export APG_BIOMETRIC_WORKERS=2  # Reduce workers
export APG_BIOMETRIC_MAX_CONNECTIONS=50  # Reduce connections

# Enable garbage collection tuning
export PYTHONHASHSEED=1
export PYTHONASYNCIODEBUG=1
```

#### Issue: Database Connection Pool Exhaustion

**Symptoms:**
- "Pool has reached maximum size" errors
- Connection timeouts
- Service unavailability

**Solutions:**
```python
# Increase pool size
database:
  pool_size: 50
  max_overflow: 100
  pool_timeout: 30
  pool_recycle: 3600

# Monitor connections
SELECT count(*) FROM pg_stat_activity WHERE state = 'active';

# Kill idle connections
SELECT pg_terminate_backend(pid) 
FROM pg_stat_activity 
WHERE state = 'idle' 
AND state_change < current_timestamp - interval '1 hour';
```

#### Issue: Biometric Engine Failures

**Symptoms:**
- Template enrollment failures
- Verification errors
- Engine initialization errors

**Solutions:**
```bash
# Check engine dependencies
python -c "import cv2; print(cv2.__version__)"
python -c "import dlib; print(dlib.__version__)"
python -c "import librosa; print(librosa.__version__)"

# Verify engine configuration
python -m apg_biometric.engines.test_engines

# Check log files
tail -f /var/log/apg-biometric/app.log | grep ERROR
```

### Diagnostic Tools

#### Health Check Script

```bash
#!/bin/bash
# health_check.sh

echo "=== APG Biometric Health Check ==="

# Check service status
echo "Service Status:"
systemctl is-active apg-biometric

# Check ports
echo "Port Status:"
netstat -tlnp | grep :8000

# Check database connection
echo "Database Connection:"
pg_isready -h localhost -p 5432 -U apg_biometric_user

# Check Redis connection
echo "Redis Connection:"
redis-cli ping

# Check disk space
echo "Disk Space:"
df -h /var/log/apg-biometric

# Check memory usage
echo "Memory Usage:"
free -h

# Check recent errors
echo "Recent Errors:"
journalctl -u apg-biometric --since "1 hour ago" --grep ERROR
```

#### Performance Monitoring Script

```python
#!/usr/bin/env python3
# performance_monitor.py

import asyncio
import aiohttp
import time
from statistics import mean

async def test_verification_performance():
    """Test verification endpoint performance"""
    url = "https://biometric.yourdomain.com/api/v1/biometric/health"
    response_times = []
    
    async with aiohttp.ClientSession() as session:
        for i in range(10):
            start_time = time.time()
            
            try:
                async with session.get(url) as response:
                    await response.text()
                    response_time = time.time() - start_time
                    response_times.append(response_time)
                    print(f"Request {i+1}: {response_time:.3f}s")
            
            except Exception as e:
                print(f"Request {i+1} failed: {e}")
            
            await asyncio.sleep(1)
    
    if response_times:
        print(f"\nAverage response time: {mean(response_times):.3f}s")
        print(f"Min response time: {min(response_times):.3f}s")
        print(f"Max response time: {max(response_times):.3f}s")

if __name__ == "__main__":
    asyncio.run(test_verification_performance())
```

### Emergency Procedures

#### Service Recovery

```bash
#!/bin/bash
# emergency_recovery.sh

echo "=== Emergency Recovery Procedure ==="

# Step 1: Stop service
echo "Stopping APG Biometric service..."
systemctl stop apg-biometric

# Step 2: Clear temporary files
echo "Clearing temporary files..."
rm -rf /tmp/apg-biometric-*

# Step 3: Check database
echo "Checking database..."
systemctl status postgresql
pg_isready -h localhost -p 5432

# Step 4: Clear Redis cache
echo "Clearing Redis cache..."
redis-cli FLUSHDB

# Step 5: Restart service
echo "Starting APG Biometric service..."
systemctl start apg-biometric

# Step 6: Wait and check
sleep 10
systemctl is-active apg-biometric

echo "Recovery procedure completed"
```

#### Backup and Restore

```bash
#!/bin/bash
# backup_restore.sh

backup_database() {
    echo "Backing up database..."
    pg_dump -h localhost -U apg_biometric_user apg_biometric > \
        "/backup/apg_biometric_$(date +%Y%m%d_%H%M%S).sql"
}

restore_database() {
    echo "Restoring database from $1..."
    psql -h localhost -U apg_biometric_user apg_biometric < "$1"
}

backup_templates() {
    echo "Backing up biometric templates..."
    tar -czf "/backup/templates_$(date +%Y%m%d_%H%M%S).tar.gz" \
        /var/lib/apg-biometric/templates/
}

case "$1" in
    backup)
        backup_database
        backup_templates
        ;;
    restore)
        restore_database "$2"
        ;;
    *)
        echo "Usage: $0 {backup|restore backup_file}"
        exit 1
        ;;
esac
```

---

*This deployment guide provides comprehensive instructions for production deployment of the APG Biometric Authentication capability. For additional support or custom deployment scenarios, contact our professional services team.*