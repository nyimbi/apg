# APG Capability Registry - Deployment Guide

Complete deployment guide for the APG Capability Registry with production-ready configurations, monitoring, and best practices.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Database Configuration](#database-configuration)
- [Application Configuration](#application-configuration)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Load Balancing](#load-balancing)
- [Monitoring and Logging](#monitoring-and-logging)
- [Security Configuration](#security-configuration)
- [Performance Optimization](#performance-optimization)
- [Backup and Recovery](#backup-and-recovery)
- [Troubleshooting](#troubleshooting)

## 🔧 Prerequisites

### System Requirements

#### Minimum Requirements
- **CPU**: 2 cores, 2.4 GHz
- **Memory**: 4 GB RAM
- **Storage**: 50 GB SSD
- **Network**: 1 Gbps connection

#### Recommended Production Requirements
- **CPU**: 8 cores, 3.0 GHz
- **Memory**: 16 GB RAM
- **Storage**: 200 GB NVMe SSD
- **Network**: 10 Gbps connection

### Software Dependencies

#### Core Dependencies
- **Python**: 3.11+
- **PostgreSQL**: 14+
- **Redis**: 6+
- **Node.js**: 18+ (for UI components)

#### Optional Dependencies
- **Elasticsearch**: 8+ (for advanced search)
- **Prometheus**: Latest (for metrics)
- **Grafana**: Latest (for dashboards)
- **Jaeger**: Latest (for tracing)

### Infrastructure Requirements

#### Cloud Providers
- **AWS**: EC2, RDS, ElastiCache, S3
- **Azure**: VM, Database, Cache, Storage
- **GCP**: Compute Engine, Cloud SQL, Memorystore, Storage

#### On-Premises
- **Virtualization**: VMware vSphere 7+
- **Container Platform**: Docker 20+, Kubernetes 1.24+
- **Load Balancer**: HAProxy, NGINX, or cloud LB

## 🌍 Environment Setup

### Development Environment

```bash
# Clone repository
git clone https://github.com/datacraft/apg-capability-registry.git
cd apg-capability-registry

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your configuration

# Install pre-commit hooks
pre-commit install

# Start development services
docker-compose up -d postgres redis

# Run database migrations
alembic upgrade head

# Start development server
uvicorn capability_registry.api:api_app --reload --host 0.0.0.0 --port 8000
```

### Staging Environment

```bash
# Use staging configuration
export ENVIRONMENT=staging
export DATABASE_URL="postgresql://user:pass@staging-db/registry"
export REDIS_URL="redis://staging-redis:6379"

# Deploy with Docker Compose
docker-compose -f docker-compose.staging.yml up -d
```

### Production Environment

```bash
# Use production configuration
export ENVIRONMENT=production
export DATABASE_URL="postgresql://user:pass@prod-db/registry"
export REDIS_URL="redis://prod-redis:6379"

# Deploy with Kubernetes
kubectl apply -f k8s/production/
```

## 🗄️ Database Configuration

### PostgreSQL Setup

#### Installation (Ubuntu/Debian)
```bash
# Install PostgreSQL
sudo apt update
sudo apt install postgresql postgresql-contrib

# Start and enable service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database and user
sudo -u postgres psql
```

```sql
-- Create database and user
CREATE DATABASE apg_capability_registry;
CREATE USER registry_user WITH ENCRYPTED PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE apg_capability_registry TO registry_user;

-- Grant necessary permissions
\c apg_capability_registry
GRANT ALL ON SCHEMA public TO registry_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO registry_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO registry_user;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

\q
```

#### PostgreSQL Configuration (postgresql.conf)
```ini
# Connection settings
listen_addresses = '*'
port = 5432
max_connections = 200

# Memory settings
shared_buffers = 4GB
effective_cache_size = 12GB
work_mem = 64MB
maintenance_work_mem = 1GB

# WAL settings
wal_buffers = 16MB
checkpoint_segments = 32
checkpoint_completion_target = 0.9

# Query planner
random_page_cost = 1.1
effective_io_concurrency = 200

# Logging
log_destination = 'stderr'
logging_collector = on
log_directory = 'pg_log'
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
log_statement = 'mod'
log_min_duration_statement = 1000
```

#### Authentication (pg_hba.conf)
```ini
# TYPE  DATABASE        USER            ADDRESS                 METHOD
local   all             postgres                                peer
local   all             all                                     md5
host    all             all             127.0.0.1/32            md5
host    all             all             ::1/128                 md5
host    apg_capability_registry registry_user 10.0.0.0/8       md5
```

### Database Migration

```bash
# Run migrations
alembic upgrade head

# Create custom migration
alembic revision --autogenerate -m "Add new feature"

# Rollback if needed
alembic downgrade -1
```

### Redis Configuration

#### Installation
```bash
# Ubuntu/Debian
sudo apt install redis-server

# Configure Redis
sudo nano /etc/redis/redis.conf
```

#### Redis Configuration (redis.conf)
```ini
# Network
bind 0.0.0.0
port 6379
protected-mode yes
requirepass your_secure_password

# Memory
maxmemory 2gb
maxmemory-policy allkeys-lru

# Persistence
save 900 1
save 300 10
save 60 10000

# Security
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command DEBUG ""
rename-command CONFIG ""
```

## ⚙️ Application Configuration

### Environment Variables

Create `.env` file:

```bash
# Application
ENVIRONMENT=production
DEBUG=False
LOG_LEVEL=INFO
SECRET_KEY=your-secret-key-here

# Database
DATABASE_URL=postgresql://registry_user:secure_password@localhost:5432/apg_capability_registry
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Redis
REDIS_URL=redis://:password@localhost:6379/0
REDIS_POOL_SIZE=20

# APG Platform Integration
APG_PLATFORM_URL=https://apg.platform.url
APG_COMPOSITION_ENGINE_URL=https://composition.apg.platform.url
APG_DISCOVERY_SERVICE_URL=https://discovery.apg.platform.url
APG_TENANT_ID=production-tenant
APG_API_KEY=your-apg-api-key

# Security
JWT_SECRET_KEY=your-jwt-secret-key
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
ENCRYPTION_KEY=your-encryption-key

# Performance
ASYNC_POOL_SIZE=100
MAX_CONNECTIONS_PER_HOST=20
REQUEST_TIMEOUT=30

# Features
ENABLE_MARKETPLACE_INTEGRATION=true
ENABLE_REAL_TIME_COLLABORATION=true
ENABLE_ANALYTICS_DASHBOARD=true
ENABLE_MOBILE_API=true
ENABLE_WEBHOOKS=true

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
ENABLE_TRACING=true
JAEGER_ENDPOINT=http://jaeger:14268/api/traces

# External Services
ELASTICSEARCH_URL=http://elasticsearch:9200
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password

# File Storage
STORAGE_BACKEND=s3
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_S3_BUCKET=apg-registry-files
AWS_REGION=us-east-1

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=100
RATE_LIMIT_BURST=20
```

### Configuration Classes

Create `config.py`:

```python
import os
from typing import Optional

class BaseConfig:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_RECORD_QUERIES = True
    
    # Redis
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    
    # APG Integration
    APG_PLATFORM_URL = os.environ.get('APG_PLATFORM_URL')
    APG_API_KEY = os.environ.get('APG_API_KEY')
    
    # Security
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', SECRET_KEY)
    JWT_ALGORITHM = 'HS256'
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 30
    
    # Performance
    DATABASE_POOL_SIZE = int(os.environ.get('DATABASE_POOL_SIZE', '20'))
    DATABASE_MAX_OVERFLOW = int(os.environ.get('DATABASE_MAX_OVERFLOW', '30'))
    
    # Features
    ENABLE_MARKETPLACE_INTEGRATION = os.environ.get('ENABLE_MARKETPLACE_INTEGRATION', 'false').lower() == 'true'
    ENABLE_REAL_TIME_COLLABORATION = os.environ.get('ENABLE_REAL_TIME_COLLABORATION', 'false').lower() == 'true'

class DevelopmentConfig(BaseConfig):
    """Development configuration"""
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'postgresql://localhost/apg_registry_dev')
    LOG_LEVEL = 'DEBUG'

class StagingConfig(BaseConfig):
    """Staging configuration"""
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
    LOG_LEVEL = 'INFO'

class ProductionConfig(BaseConfig):
    """Production configuration"""
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
    LOG_LEVEL = 'WARNING'
    
    # Security enhancements
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'

config = {
    'development': DevelopmentConfig,
    'staging': StagingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
```

## 🐳 Docker Deployment

### Dockerfile

```dockerfile
# Multi-stage build for production
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
COPY requirements-prod.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-prod.txt

# Production stage
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy application code
COPY . .

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Start application
CMD ["uvicorn", "capability_registry.api:api_app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Docker Compose (Production)

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://registry_user:${DB_PASSWORD}@postgres:5432/apg_capability_registry
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
      - ENVIRONMENT=production
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./logs:/app/logs
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
    restart: unless-stopped

  postgres:
    image: postgres:14
    environment:
      POSTGRES_DB: apg_capability_registry
      POSTGRES_USER: registry_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./postgres.conf:/etc/postgresql/postgresql.conf
    command: postgres -c config_file=/etc/postgresql/postgresql.conf
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U registry_user -d apg_capability_registry"]
      interval: 10s
      timeout: 5s
      retries: 5
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 4G
    restart: unless-stopped

  redis:
    image: redis:6-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD} --maxmemory 2gb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  default:
    driver: bridge
```

### Environment File (.env)

```bash
# Database
DB_PASSWORD=secure_database_password

# Redis
REDIS_PASSWORD=secure_redis_password

# Grafana
GRAFANA_PASSWORD=secure_grafana_password

# Application secrets
SECRET_KEY=your-super-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-key-here
```

## ☸️ Kubernetes Deployment

### Namespace

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: apg-registry
  labels:
    name: apg-registry
```

### ConfigMap

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: registry-config
  namespace: apg-registry
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  ENABLE_METRICS: "true"
  ENABLE_TRACING: "true"
  DATABASE_POOL_SIZE: "20"
  REDIS_POOL_SIZE: "20"
```

### Secrets

```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: registry-secrets
  namespace: apg-registry
type: Opaque
data:
  DATABASE_URL: cG9zdGdyZXNxbDovL3VzZXI6cGFzc0Bwb3N0Z3Jlcy9kYg==  # base64 encoded
  REDIS_URL: cmVkaXM6Ly86cGFzc0ByZWRpczozNjM5LzA=  # base64 encoded
  SECRET_KEY: eW91ci1zZWNyZXQta2V5LWhlcmU=  # base64 encoded
  JWT_SECRET_KEY: eW91ci1qd3Qtc2VjcmV0LWtleS1oZXJl  # base64 encoded
```

### Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: registry-app
  namespace: apg-registry
  labels:
    app: registry-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: registry-app
  template:
    metadata:
      labels:
        app: registry-app
    spec:
      containers:
      - name: registry-app
        image: datacraft/apg-capability-registry:latest
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 9090
          name: metrics
        envFrom:
        - configMapRef:
            name: registry-config
        - secretRef:
            name: registry-secrets
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /api/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /api/health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
        securityContext:
          runAsNonRoot: true
          runAsUser: 1000
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
        volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: tmp
        emptyDir: {}
      - name: logs
        emptyDir: {}
      securityContext:
        fsGroup: 1000
```

### Service

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: registry-service
  namespace: apg-registry
  labels:
    app: registry-app
spec:
  selector:
    app: registry-app
  ports:
  - name: http
    port: 80
    targetPort: 8000
    protocol: TCP
  - name: metrics
    port: 9090
    targetPort: 9090
    protocol: TCP
  type: ClusterIP
```

### Ingress

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: registry-ingress
  namespace: apg-registry
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-burst: "20"
spec:
  tls:
  - hosts:
    - api.apg.datacraft.co.ke
    secretName: registry-tls
  rules:
  - host: api.apg.datacraft.co.ke
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: registry-service
            port:
              number: 80
```

### HorizontalPodAutoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: registry-hpa
  namespace: apg-registry
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: registry-app
  minReplicas: 3
  maxReplicas: 10
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
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
      - type: Pods
        value: 2
        periodSeconds: 60
      selectPolicy: Max
```

### PostgreSQL StatefulSet

```yaml
# k8s/postgres-statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: apg-registry
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:14
        env:
        - name: POSTGRES_DB
          value: apg_capability_registry
        - name: POSTGRES_USER
          value: registry_user
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi
      storageClassName: fast-ssd
```

### Deployment Script

```bash
#!/bin/bash
# deploy.sh

set -e

NAMESPACE="apg-registry"
ENVIRONMENT=${1:-production}

echo "Deploying APG Capability Registry to $ENVIRONMENT environment..."

# Create namespace
kubectl apply -f k8s/namespace.yaml

# Apply secrets and configmaps
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/configmap.yaml

# Deploy database
kubectl apply -f k8s/postgres-statefulset.yaml
kubectl apply -f k8s/postgres-service.yaml

# Wait for database to be ready
echo "Waiting for PostgreSQL to be ready..."
kubectl wait --for=condition=ready pod -l app=postgres -n $NAMESPACE --timeout=300s

# Run database migrations
kubectl run migration-job --image=datacraft/apg-capability-registry:latest \
  --restart=Never \
  --env-from=configmap/registry-config \
  --env-from=secret/registry-secrets \
  --command -- alembic upgrade head \
  -n $NAMESPACE

# Deploy application
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

# Deploy ingress
kubectl apply -f k8s/ingress.yaml

# Wait for deployment
echo "Waiting for deployment to be ready..."
kubectl rollout status deployment/registry-app -n $NAMESPACE

echo "Deployment completed successfully!"
echo "Application available at: https://api.apg.datacraft.co.ke"
```

## ⚖️ Load Balancing

### NGINX Configuration

```nginx
# nginx.conf
upstream registry_backend {
    least_conn;
    server app1:8000 max_fails=3 fail_timeout=30s;
    server app2:8000 max_fails=3 fail_timeout=30s;
    server app3:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name api.apg.datacraft.co.ke;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.apg.datacraft.co.ke;

    ssl_certificate /etc/nginx/ssl/fullchain.pem;
    ssl_certificate_key /etc/nginx/ssl/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 10240;
    gzip_types text/plain text/css text/xml text/javascript application/json application/javascript application/xml+rss application/atom+xml image/svg+xml;

    location / {
        proxy_pass http://registry_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
        
        proxy_buffering on;
        proxy_buffer_size 8k;
        proxy_buffers 16 8k;
    }

    location /api/ws/ {
        proxy_pass http://registry_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }

    location /metrics {
        proxy_pass http://registry_backend;
        allow 10.0.0.0/8;
        deny all;
    }

    location /health {
        proxy_pass http://registry_backend;
        access_log off;
    }
}
```

### HAProxy Configuration

```ini
# haproxy.cfg
global
    daemon
    user haproxy
    group haproxy
    chroot /var/lib/haproxy
    stats socket /run/haproxy/admin.sock mode 660 level admin
    stats timeout 30s

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms
    option httplog
    option dontlognull
    option http-server-close
    option forwardfor except 127.0.0.0/8
    option redispatch
    retries 3

frontend registry_frontend
    bind *:80
    bind *:443 ssl crt /etc/ssl/certs/registry.pem
    redirect scheme https if !{ ssl_fc }
    
    # Rate limiting
    stick-table type ip size 100k expire 30s store http_req_rate(10s)
    http-request track-sc0 src
    http-request deny if { sc_http_req_rate(0) gt 20 }
    
    default_backend registry_backend

backend registry_backend
    balance leastconn
    option httpchk GET /api/health
    http-check expect status 200
    
    server app1 10.0.1.10:8000 check
    server app2 10.0.1.11:8000 check
    server app3 10.0.1.12:8000 check

listen stats
    bind *:8404
    stats enable
    stats uri /stats
    stats refresh 30s
    stats admin if TRUE
```

## 📊 Monitoring and Logging

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "registry_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'registry-app'
    static_configs:
      - targets: ['app:9090']
    metrics_path: /metrics
    scrape_interval: 15s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx-exporter:9113']
```

### Alert Rules

```yaml
# registry_rules.yml
groups:
- name: registry_alerts
  rules:
  - alert: RegistryDown
    expr: up{job="registry-app"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Registry service is down"
      description: "Registry service has been down for more than 1 minute"

  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} errors per second"

  - alert: HighResponseTime
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High response time"
      description: "95th percentile response time is {{ $value }}s"

  - alert: DatabaseConnectionIssue
    expr: postgresql_up == 0
    for: 30s
    labels:
      severity: critical
    annotations:
      summary: "Database connection issue"
      description: "Cannot connect to PostgreSQL database"
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "APG Capability Registry",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{status}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Active Connections",
        "type": "stat",
        "targets": [
          {
            "expr": "postgres_stat_database_numbackends",
            "legendFormat": "Database Connections"
          }
        ]
      }
    ]
  }
}
```

### Logging Configuration

```python
# logging_config.py
import logging
import sys
from pythonjsonlogger import jsonlogger

def setup_logging():
    log_format = "%(asctime)s %(name)s %(levelname)s %(message)s"
    
    # JSON formatter for structured logging
    json_formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(json_formatter)
    
    # File handler
    file_handler = logging.FileHandler('/app/logs/registry.log')
    file_handler.setFormatter(json_formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Specific loggers
    logging.getLogger('uvicorn').setLevel(logging.INFO)
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
    
    return root_logger
```

## 🔒 Security Configuration

### SSL/TLS Configuration

#### Generate SSL Certificate
```bash
# Using Let's Encrypt
certbot certonly --nginx -d api.apg.datacraft.co.ke

# Or generate self-signed certificate for testing
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout registry.key -out registry.crt \
  -subj "/C=US/ST=State/L=City/O=Datacraft/CN=api.apg.datacraft.co.ke"
```

### Firewall Configuration

```bash
# UFW (Ubuntu)
sudo ufw enable
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow from 10.0.0.0/8 to any port 5432  # Database
sudo ufw allow from 10.0.0.0/8 to any port 6379  # Redis
sudo ufw allow from 10.0.0.0/8 to any port 9090  # Metrics

# iptables rules
iptables -A INPUT -p tcp --dport 80 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -j ACCEPT
iptables -A INPUT -p tcp --dport 22 -j ACCEPT
iptables -A INPUT -p tcp --dport 5432 -s 10.0.0.0/8 -j ACCEPT
iptables -A INPUT -p tcp --dport 6379 -s 10.0.0.0/8 -j ACCEPT
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
iptables -A INPUT -j DROP
```

### Security Headers

```python
# security_middleware.py
from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        
        # Security headers
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        return response
```

## ⚡ Performance Optimization

### Database Optimization

```sql
-- Create indexes for better performance
CREATE INDEX CONCURRENTLY idx_capabilities_category ON capabilities(category);
CREATE INDEX CONCURRENTLY idx_capabilities_status ON capabilities(status);
CREATE INDEX CONCURRENTLY idx_capabilities_search ON capabilities USING gin(to_tsvector('english', capability_name || ' ' || description));
CREATE INDEX CONCURRENTLY idx_compositions_type ON compositions(composition_type);
CREATE INDEX CONCURRENTLY idx_dependencies_capability ON dependencies(dependent_id, dependency_id);

-- Analyze tables
ANALYZE capabilities;
ANALYZE compositions;
ANALYZE dependencies;

-- Update statistics
UPDATE pg_stat_user_tables SET n_tup_ins = 0, n_tup_upd = 0, n_tup_del = 0;
```

### Application Optimization

```python
# performance_config.py
import asyncio
from functools import lru_cache
from cachetools import TTLCache
import redis.asyncio as redis

# Connection pools
DATABASE_POOL_SIZE = 20
DATABASE_MAX_OVERFLOW = 30
REDIS_POOL_SIZE = 20

# Caching
@lru_cache(maxsize=1000)
def get_capability_metadata(capability_id: str):
    """Cache capability metadata"""
    pass

# Redis caching
redis_client = redis.Redis.from_url(
    "redis://localhost:6379",
    max_connections=REDIS_POOL_SIZE,
    health_check_interval=30
)

# Memory cache for frequently accessed data
memory_cache = TTLCache(maxsize=10000, ttl=300)  # 5 minutes TTL

# Async optimizations
async def batch_process_capabilities(capability_ids: list):
    """Process capabilities in batches for better performance"""
    batch_size = 50
    results = []
    
    for i in range(0, len(capability_ids), batch_size):
        batch = capability_ids[i:i + batch_size]
        batch_results = await asyncio.gather(*[
            process_capability(cap_id) for cap_id in batch
        ])
        results.extend(batch_results)
    
    return results
```

### Caching Strategy

```python
# caching.py
import json
import pickle
from typing import Any, Optional
from redis.asyncio import Redis

class CacheManager:
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            value = await self.redis.get(key)
            if value:
                return pickle.loads(value)
        except Exception as e:
            print(f"Cache get error: {e}")
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache with TTL"""
        try:
            await self.redis.setex(
                key, 
                ttl, 
                pickle.dumps(value)
            )
        except Exception as e:
            print(f"Cache set error: {e}")
    
    async def delete(self, key: str):
        """Delete value from cache"""
        try:
            await self.redis.delete(key)
        except Exception as e:
            print(f"Cache delete error: {e}")
    
    def cache_key(self, prefix: str, *args) -> str:
        """Generate cache key"""
        return f"{prefix}:{'_'.join(str(arg) for arg in args)}"

# Usage
cache = CacheManager(redis_client)

async def get_capability_cached(capability_id: str):
    cache_key = cache.cache_key("capability", capability_id)
    
    # Try cache first
    cached_result = await cache.get(cache_key)
    if cached_result:
        return cached_result
    
    # Fetch from database
    result = await fetch_capability_from_db(capability_id)
    
    # Cache result
    await cache.set(cache_key, result, ttl=1800)  # 30 minutes
    
    return result
```

## 💾 Backup and Recovery

### Database Backup

```bash
#!/bin/bash
# backup.sh

DB_NAME="apg_capability_registry"
DB_USER="registry_user"
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/registry_backup_$DATE.sql"

# Create backup directory
mkdir -p $BACKUP_DIR

# Perform backup
pg_dump -h localhost -U $DB_USER -d $DB_NAME -f $BACKUP_FILE

# Compress backup
gzip $BACKUP_FILE

# Upload to S3 (optional)
aws s3 cp $BACKUP_FILE.gz s3://apg-backups/database/

# Clean old backups (keep last 7 days)
find $BACKUP_DIR -name "registry_backup_*.sql.gz" -mtime +7 -delete

echo "Backup completed: $BACKUP_FILE.gz"
```

### Automated Backup with Cron

```bash
# Add to crontab
# crontab -e

# Daily backup at 2 AM
0 2 * * * /opt/scripts/backup.sh

# Weekly full backup on Sunday at 3 AM
0 3 * * 0 /opt/scripts/full_backup.sh
```

### Recovery Procedure

```bash
#!/bin/bash
# restore.sh

BACKUP_FILE=$1
DB_NAME="apg_capability_registry"
DB_USER="registry_user"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

# Stop application
docker-compose stop app

# Drop and recreate database
dropdb -h localhost -U $DB_USER $DB_NAME
createdb -h localhost -U $DB_USER $DB_NAME

# Restore from backup
if [[ $BACKUP_FILE == *.gz ]]; then
    gunzip -c $BACKUP_FILE | psql -h localhost -U $DB_USER -d $DB_NAME
else
    psql -h localhost -U $DB_USER -d $DB_NAME -f $BACKUP_FILE
fi

# Run migrations (if needed)
alembic upgrade head

# Start application
docker-compose start app

echo "Recovery completed"
```

## 🔧 Troubleshooting

### Common Issues

#### Application Won't Start

```bash
# Check logs
docker logs apg-registry-app
kubectl logs -f deployment/registry-app -n apg-registry

# Check database connection
psql -h localhost -U registry_user -d apg_capability_registry -c "SELECT 1;"

# Check Redis connection
redis-cli -h localhost -p 6379 ping

# Check environment variables
env | grep -E "(DATABASE_URL|REDIS_URL|SECRET_KEY)"
```

#### High Memory Usage

```bash
# Check memory usage
docker stats
kubectl top pods -n apg-registry

# Analyze memory leaks
python -m memory_profiler app.py

# Check for runaway queries
SELECT query, state, query_start 
FROM pg_stat_activity 
WHERE state = 'active' 
ORDER BY query_start;
```

#### Database Connection Issues

```sql
-- Check active connections
SELECT count(*) FROM pg_stat_activity;

-- Check connection limits
SHOW max_connections;

-- Kill idle connections
SELECT pg_terminate_backend(pid) 
FROM pg_stat_activity 
WHERE state = 'idle' 
AND query_start < now() - interval '1 hour';
```

#### Performance Issues

```bash
# Check system resources
top
htop
iotop
nethogs

# Check database performance
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;

# Check slow queries
SELECT query, total_time, mean_time, calls
FROM pg_stat_statements
WHERE mean_time > 1000
ORDER BY mean_time DESC;
```

### Health Checks

```bash
# Application health
curl -f http://localhost:8000/api/health

# Database health
pg_isready -h localhost -p 5432

# Redis health
redis-cli ping

# Full system check
python scripts/health_check.py
```

### Log Analysis

```bash
# Search for errors
grep -i error /app/logs/registry.log

# Search for specific patterns
grep "HTTP 5" /var/log/nginx/access.log

# Monitor logs in real-time
tail -f /app/logs/registry.log | jq '.'

# Analyze performance
grep "response_time" /app/logs/registry.log | awk '{print $8}' | sort -n
```

---

This comprehensive deployment guide covers all aspects of deploying the APG Capability Registry in production environments. For additional support, contact: nyimbi@gmail.com