# APG Employee Data Management - Deployment Guide

## 📋 Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Database Configuration](#database-configuration)
- [Application Deployment](#application-deployment)
- [Container Deployment](#container-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Monitoring & Observability](#monitoring--observability)
- [Security Configuration](#security-configuration)
- [Performance Optimization](#performance-optimization)
- [Backup & Disaster Recovery](#backup--disaster-recovery)

## 🎯 Overview

This guide provides comprehensive instructions for deploying the APG Employee Data Management capability across different environments: development, staging, and production.

### Deployment Architecture

```
                    ┌─────────────┐
                    │   Load      │
                    │  Balancer   │
                    │  (nginx)    │
                    └─────┬───────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
   ┌────▼────┐      ┌────▼────┐      ┌────▼────┐
   │   App   │      │   App   │      │   App   │
   │Instance │      │Instance │      │Instance │
   │    1    │      │    2    │      │    3    │
   └────┬────┘      └────┬────┘      └────┬────┘
        │                │                │
        └─────────────────┼─────────────────┘
                          │
   ┌──────────────────────┼──────────────────────┐
   │                      │                      │
┌──▼────┐         ┌─────▼─────┐         ┌──────▼──────┐
│ Postgres │      │   Redis   │         │  External   │
│Database  │      │   Cache   │         │  Services   │
│          │      │           │         │  (AI/ML)    │
└─────────┘      └───────────┘         └─────────────┘
```

### Deployment Options

| Option | Use Case | Complexity | Scalability |
|--------|----------|-------------|-------------|
| **Local Docker** | Development | Low | Low |
| **Docker Compose** | Testing/Staging | Medium | Medium |
| **Kubernetes** | Production | High | High |
| **Cloud Native** | Enterprise | High | Very High |

## 🔧 Prerequisites

### System Requirements

#### Minimum Requirements
- **CPU**: 2 cores
- **RAM**: 4GB
- **Storage**: 50GB SSD
- **Network**: 100 Mbps

#### Recommended Requirements
- **CPU**: 4+ cores
- **RAM**: 8GB+
- **Storage**: 100GB+ SSD
- **Network**: 1 Gbps

### Software Dependencies

#### Required Software
```bash
# Python 3.11+
python --version  # >= 3.11.0

# PostgreSQL with extensions
psql --version    # >= 14.0
# Required extensions: vector, pg_trgm, uuid-ossp

# Redis
redis-cli --version  # >= 7.0

# nginx (for production)
nginx -v  # >= 1.20
```

#### Optional Software
```bash
# Docker & Docker Compose
docker --version  # >= 20.10
docker-compose --version  # >= 2.0

# Kubernetes
kubectl version  # >= 1.24

# Cloud CLI tools
aws --version     # AWS CLI
gcloud version    # Google Cloud SDK
az --version      # Azure CLI
```

### Network Requirements

#### Ports
| Service | Port | Purpose |
|---------|------|---------|
| Application | 8000 | Main API server |
| PostgreSQL | 5432 | Database connection |
| Redis | 6379 | Cache and sessions |
| nginx | 80/443 | Load balancer/proxy |
| Monitoring | 9090/3000 | Prometheus/Grafana |

#### External Services
- **OpenAI API**: api.openai.com:443
- **APG Platform**: Internal network or VPN
- **Email Service**: SMTP server
- **File Storage**: S3/GCS/Azure Blob

## 🌍 Environment Setup

### Environment Variables

Create environment files for each deployment stage:

#### `.env.development`
```bash
# Application
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# Database
DATABASE_URL=postgresql+asyncpg://postgres:devpass@localhost:5432/apg_hr_dev
DATABASE_POOL_SIZE=5
DATABASE_ECHO=true

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_PREFIX=apg_hr_dev

# Security
SECRET_KEY=dev-secret-key-change-in-production
JWT_ALGORITHM=HS256
JWT_EXPIRY=86400

# AI Services
OPENAI_API_KEY=sk-dev-key
AI_MODEL_PROVIDER=openai
AI_MODEL_NAME=gpt-3.5-turbo
ENABLE_AI_FEATURES=true

# External Integrations
ENABLE_EXTERNAL_INTEGRATIONS=false
WEBHOOK_SECRET=dev-webhook-secret

# Monitoring
ENABLE_METRICS=true
SENTRY_DSN=https://your-sentry-dsn-here
```

#### `.env.staging`
```bash
# Application
ENVIRONMENT=staging
DEBUG=false
LOG_LEVEL=INFO

# Database
DATABASE_URL=postgresql+asyncpg://apg_user:staging_pass@db-staging:5432/apg_hr_staging
DATABASE_POOL_SIZE=10
DATABASE_ECHO=false

# Redis
REDIS_URL=redis://redis-staging:6379/0
REDIS_PREFIX=apg_hr_staging

# Security
SECRET_KEY=${STAGING_SECRET_KEY}
JWT_ALGORITHM=HS256
JWT_EXPIRY=3600

# AI Services
OPENAI_API_KEY=${STAGING_OPENAI_KEY}
AI_MODEL_PROVIDER=openai
AI_MODEL_NAME=gpt-4
ENABLE_AI_FEATURES=true

# External Integrations
ENABLE_EXTERNAL_INTEGRATIONS=true
WEBHOOK_SECRET=${STAGING_WEBHOOK_SECRET}

# Monitoring
ENABLE_METRICS=true
SENTRY_DSN=${STAGING_SENTRY_DSN}
```

#### `.env.production`
```bash
# Application
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING

# Database
DATABASE_URL=postgresql+asyncpg://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:5432/${DB_NAME}
DATABASE_POOL_SIZE=20
DATABASE_ECHO=false

# Redis
REDIS_URL=redis://${REDIS_HOST}:6379/0
REDIS_PREFIX=apg_hr_prod

# Security
SECRET_KEY=${PRODUCTION_SECRET_KEY}
JWT_ALGORITHM=HS256
JWT_EXPIRY=3600
ENCRYPTION_KEY=${PRODUCTION_ENCRYPTION_KEY}

# AI Services
OPENAI_API_KEY=${PRODUCTION_OPENAI_KEY}
AI_MODEL_PROVIDER=openai
AI_MODEL_NAME=gpt-4
ENABLE_AI_FEATURES=true

# External Integrations
ENABLE_EXTERNAL_INTEGRATIONS=true
WEBHOOK_SECRET=${PRODUCTION_WEBHOOK_SECRET}

# Monitoring
ENABLE_METRICS=true
SENTRY_DSN=${PRODUCTION_SENTRY_DSN}
NEW_RELIC_LICENSE_KEY=${NEW_RELIC_KEY}
```

### Secrets Management

#### Using Docker Secrets
```bash
# Create secrets
echo "super-secret-key" | docker secret create apg_secret_key -
echo "openai-api-key" | docker secret create apg_openai_key -

# Reference in docker-compose.yml
secrets:
  apg_secret_key:
    external: true
  apg_openai_key:
    external: true
```

#### Using Kubernetes Secrets
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: apg-employee-secrets
type: Opaque
data:
  secret-key: <base64-encoded-secret>
  openai-api-key: <base64-encoded-key>
  database-password: <base64-encoded-password>
```

#### Using Cloud Secret Managers
```bash
# AWS Secrets Manager
aws secretsmanager create-secret \
  --name "apg-employee-secrets" \
  --secret-string '{"SECRET_KEY":"...","OPENAI_API_KEY":"..."}'

# Google Secret Manager
gcloud secrets create apg-secret-key --data-file=secret.txt

# Azure Key Vault
az keyvault secret set \
  --vault-name "apg-vault" \
  --name "secret-key" \
  --value "super-secret-key"
```

## 🗄️ Database Configuration

### PostgreSQL Setup

#### Installation & Extensions
```bash
# Install PostgreSQL 14+
sudo apt-get install postgresql-14 postgresql-14-contrib

# Install vector extension
sudo apt-get install postgresql-14-pgvector

# Connect to PostgreSQL
sudo -u postgres psql

# Create database and user
CREATE DATABASE apg_hr_production;
CREATE USER apg_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE apg_hr_production TO apg_user;

# Enable extensions
\c apg_hr_production
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
```

#### Configuration (`postgresql.conf`)
```ini
# Memory settings
shared_buffers = 2GB
effective_cache_size = 6GB
work_mem = 64MB
maintenance_work_mem = 1GB

# Connection settings
max_connections = 200
max_prepared_transactions = 200

# WAL settings
wal_buffers = 16MB
checkpoint_timeout = 10min
checkpoint_completion_target = 0.9

# Logging
log_statement = 'mod'
log_min_duration_statement = 1000
log_checkpoints = on
log_connections = on
log_disconnections = on

# Performance
random_page_cost = 1.1
seq_page_cost = 1.0
default_statistics_target = 100

# Vector extension settings
shared_preload_libraries = 'pg_stat_statements,vector'
```

#### Performance Tuning
```sql
-- Analyze tables for better query planning
ANALYZE;

-- Create performance indexes
CREATE INDEX CONCURRENTLY idx_hr_employees_department_status 
ON hr_employees (department_id, employment_status) 
WHERE employment_status = 'Active';

CREATE INDEX CONCURRENTLY idx_hr_employees_search_trgm 
ON hr_employees USING gin(first_name gin_trgm_ops, last_name gin_trgm_ops);

-- Vacuum and reindex regularly
VACUUM ANALYZE hr_employees;
REINDEX INDEX CONCURRENTLY idx_hr_employees_search;
```

### Database Migration

#### Alembic Configuration
```python
# alembic.ini
[alembic]
script_location = alembic
sqlalchemy.url = driver://user:pass@localhost/dbname

[post_write_hooks]
hooks = black
black.type = console_scripts
black.entrypoint = black
black.options = -l 79 REVISION_SCRIPT_FILENAME
```

#### Migration Scripts
```bash
# Create migration
alembic revision --autogenerate -m "Add employee AI profiles"

# Review migration
alembic show head

# Apply migration
alembic upgrade head

# Rollback if needed
alembic downgrade -1
```

### Redis Configuration

#### Installation
```bash
# Install Redis
sudo apt-get install redis-server

# Configure Redis
sudo nano /etc/redis/redis.conf
```

#### Configuration (`redis.conf`)
```ini
# Network
bind 127.0.0.1
port 6379
protected-mode yes

# Memory
maxmemory 2gb
maxmemory-policy allkeys-lru

# Persistence
save 900 1
save 300 10
save 60 10000

# Security
requirepass your_redis_password

# Logging
loglevel notice
logfile /var/log/redis/redis-server.log
```

## 🚀 Application Deployment

### Manual Deployment

#### Preparation
```bash
# Clone repository
git clone https://github.com/datacraft/apg-platform.git
cd apg-platform/capabilities/employee_data_management

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements-prod.txt

# Set environment variables
export $(cat .env.production | xargs)

# Run database migrations
alembic upgrade head

# Collect static files (if applicable)
python manage.py collectstatic --noinput
```

#### Start Application
```bash
# Using Gunicorn (recommended)
gunicorn \
  --bind 0.0.0.0:8000 \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --worker-connections 1000 \
  --max-requests 1000 \
  --max-requests-jitter 100 \
  --timeout 30 \
  --keepalive 2 \
  --log-level info \
  --access-logfile - \
  --error-logfile - \
  main:app

# Using systemd service
sudo nano /etc/systemd/system/apg-employee.service
```

#### Systemd Service Configuration
```ini
[Unit]
Description=APG Employee Data Management
After=network.target postgresql.service redis.service

[Service]
Type=forking
User=apg
Group=apg
WorkingDirectory=/opt/apg/employee_data_management
Environment=PATH=/opt/apg/employee_data_management/venv/bin
EnvironmentFile=/opt/apg/employee_data_management/.env.production
ExecStart=/opt/apg/employee_data_management/venv/bin/gunicorn \
  --bind 0.0.0.0:8000 \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --daemon \
  --pid /var/run/apg-employee.pid \
  main:app
ExecReload=/bin/kill -s HUP $MAINPID
KillMode=mixed
TimeoutStopSec=5
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

### Automated Deployment

#### Deployment Script
```bash
#!/bin/bash
# deploy.sh

set -e

ENVIRONMENT=${1:-staging}
BRANCH=${2:-main}
BACKUP_ENABLED=${3:-true}

echo "Deploying APG Employee Management to $ENVIRONMENT..."

# Backup database if enabled
if [ "$BACKUP_ENABLED" = "true" ]; then
    echo "Creating database backup..."
    pg_dump apg_hr_$ENVIRONMENT > backup_$(date +%Y%m%d_%H%M%S).sql
fi

# Pull latest code
git fetch origin
git checkout $BRANCH
git pull origin $BRANCH

# Install dependencies
pip install -r requirements-prod.txt

# Run migrations
alembic upgrade head

# Restart services
sudo systemctl restart apg-employee
sudo systemctl restart nginx

# Health check
sleep 10
curl -f http://localhost:8000/api/v1/health || exit 1

echo "Deployment completed successfully!"
```

#### CI/CD Pipeline (GitHub Actions)
```yaml
name: Deploy to Production

on:
  push:
    branches: [main]
    paths: ['capabilities/employee_data_management/**']

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: production
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup SSH
      uses: webfactory/ssh-agent@v0.5.4
      with:
        ssh-private-key: ${{ secrets.DEPLOY_SSH_KEY }}
    
    - name: Deploy to server
      run: |
        ssh -o StrictHostKeyChecking=no ${{ secrets.DEPLOY_USER }}@${{ secrets.DEPLOY_HOST }} << 'EOF'
          cd /opt/apg/employee_data_management
          ./deploy.sh production main true
        EOF
    
    - name: Notify deployment
      uses: 8398a7/action-slack@v3
      with:
        status: ${{ job.status }}
        webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

## 🐳 Container Deployment

### Docker Configuration

#### Multi-stage Dockerfile
```dockerfile
# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements-prod.txt .
RUN pip install --no-cache-dir --user -r requirements-prod.txt

# Production stage
FROM python:3.11-slim as production

# Create app user
RUN useradd --create-home --shell /bin/bash app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /root/.local /home/app/.local

# Set work directory
WORKDIR /app

# Copy application code
COPY --chown=app:app . .

# Switch to app user
USER app

# Make sure scripts are executable
RUN chmod +x scripts/*.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Expose port
EXPOSE 8000

# Set environment
ENV PATH="/home/app/.local/bin:$PATH"
ENV PYTHONPATH="/app"

# Start application
CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:8000", "--worker-class", "uvicorn.workers.UvicornWorker", "--workers", "4"]
```

#### Docker Compose for Production
```yaml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
      target: production
    image: apg/employee-api:latest
    container_name: apg-employee-api
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+asyncpg://apg_user:${DB_PASSWORD}@postgres:5432/apg_hr
      - REDIS_URL=redis://redis:6379/0
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - SECRET_KEY=${SECRET_KEY}
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./logs:/app/logs
      - ./uploads:/app/uploads
    networks:
      - apg-network
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.apg-api.rule=Host(`api.company.com`)"
      - "traefik.http.routers.apg-api.tls.certresolver=letsencrypt"

  postgres:
    image: pgvector/pgvector:pg14
    container_name: apg-postgres
    restart: unless-stopped
    environment:
      - POSTGRES_DB=apg_hr
      - POSTGRES_USER=apg_user
      - POSTGRES_PASSWORD=${DB_PASSWORD}
      - POSTGRES_INITDB_ARGS=--encoding=UTF-8 --lc-collate=C --lc-ctype=C
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
      - ./postgresql.conf:/etc/postgresql/postgresql.conf
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U apg_user -d apg_hr"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - apg-network

  redis:
    image: redis:7-alpine
    container_name: apg-redis
    restart: unless-stopped
    command: redis-server --requirepass ${REDIS_PASSWORD} --appendonly yes
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
      - ./redis.conf:/usr/local/etc/redis/redis.conf
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - apg-network

  nginx:
    image: nginx:alpine
    container_name: apg-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
      - ./static:/var/www/static
    depends_on:
      - app
    networks:
      - apg-network

  prometheus:
    image: prom/prometheus:latest
    container_name: apg-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    networks:
      - apg-network

  grafana:
    image: grafana/grafana:latest
    container_name: apg-grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
    networks:
      - apg-network

volumes:
  postgres_data:
    driver: local
  redis_data:
    driver: local
  prometheus_data:
    driver: local
  grafana_data:
    driver: local

networks:
  apg-network:
    driver: bridge
```

#### nginx Configuration
```nginx
upstream apg_backend {
    least_conn;
    server app:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name api.company.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.company.com;

    # SSL Configuration
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;

    # Security Headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains";

    # Gzip Compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css application/json application/javascript;

    # Rate Limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;

    # API Proxy
    location /api/ {
        proxy_pass http://apg_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
        
        # Buffer settings
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }

    # Health check endpoint (no auth required)
    location /api/v1/health {
        proxy_pass http://apg_backend;
        access_log off;
    }

    # Static files
    location /static/ {
        alias /var/www/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }

    # API Documentation
    location /docs {
        proxy_pass http://apg_backend;
        proxy_set_header Host $host;
    }
}
```

## ☸️ Kubernetes Deployment

### Namespace and Resources

#### Namespace
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: apg-employee
  labels:
    name: apg-employee
    environment: production
```

#### ConfigMap
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: apg-employee-config
  namespace: apg-employee
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  DATABASE_POOL_SIZE: "20"
  REDIS_PREFIX: "apg_hr_prod"
  AI_MODEL_PROVIDER: "openai"
  AI_MODEL_NAME: "gpt-4"
  ENABLE_AI_FEATURES: "true"
```

#### Secrets
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: apg-employee-secrets
  namespace: apg-employee
type: Opaque
data:
  SECRET_KEY: <base64-encoded-secret>
  DATABASE_URL: <base64-encoded-db-url>
  REDIS_URL: <base64-encoded-redis-url>
  OPENAI_API_KEY: <base64-encoded-openai-key>
  ENCRYPTION_KEY: <base64-encoded-encryption-key>
```

### Database Deployment

#### PostgreSQL StatefulSet
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: apg-employee
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
        image: pgvector/pgvector:pg14
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_DB
          value: apg_hr
        - name: POSTGRES_USER
          value: apg_user
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        - name: PGDATA
          value: /var/lib/postgresql/data/pgdata
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        - name: postgres-config
          mountPath: /etc/postgresql
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - apg_user
            - -d
            - apg_hr
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - apg_user
            - -d
            - apg_hr
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: postgres-config
        configMap:
          name: postgres-config
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: "fast-ssd"
      resources:
        requests:
          storage: 100Gi
```

#### Redis Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: apg-employee
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        command:
        - redis-server
        - /etc/redis/redis.conf
        - --requirepass
        - $(REDIS_PASSWORD)
        env:
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: redis-secret
              key: password
        volumeMounts:
        - name: redis-config
          mountPath: /etc/redis
        - name: redis-data
          mountPath: /data
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          exec:
            command:
            - redis-cli
            - ping
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command:
            - redis-cli
            - ping
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: redis-config
        configMap:
          name: redis-config
      - name: redis-data
        emptyDir: {}
```

### Application Deployment

#### Main Application
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: apg-employee-api
  namespace: apg-employee
  labels:
    app: apg-employee-api
    version: v1.0.0
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 1
  selector:
    matchLabels:
      app: apg-employee-api
  template:
    metadata:
      labels:
        app: apg-employee-api
        version: v1.0.0
    spec:
      containers:
      - name: apg-employee-api
        image: apg/employee-api:v1.0.0
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: apg-employee-secrets
              key: DATABASE_URL
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: apg-employee-secrets
              key: REDIS_URL
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: apg-employee-secrets
              key: SECRET_KEY
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: apg-employee-secrets
              key: OPENAI_API_KEY
        envFrom:
        - configMapRef:
            name: apg-employee-config
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        volumeMounts:
        - name: logs
          mountPath: /app/logs
        - name: uploads
          mountPath: /app/uploads
      volumes:
      - name: logs
        emptyDir: {}
      - name: uploads
        persistentVolumeClaim:
          claimName: uploads-pvc
      initContainers:
      - name: migrate
        image: apg/employee-api:v1.0.0
        command: ["alembic", "upgrade", "head"]
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: apg-employee-secrets
              key: DATABASE_URL
```

#### Service and Ingress
```yaml
apiVersion: v1
kind: Service
metadata:
  name: apg-employee-api-service
  namespace: apg-employee
spec:
  selector:
    app: apg-employee-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: apg-employee-api-ingress
  namespace: apg-employee
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - api.company.com
    secretName: apg-api-tls
  rules:
  - host: api.company.com
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: apg-employee-api-service
            port:
              number: 80
```

#### Horizontal Pod Autoscaler
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: apg-employee-api-hpa
  namespace: apg-employee
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: apg-employee-api
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

## ☁️ Cloud Deployment

### AWS Deployment

#### ECS with Fargate
```yaml
# task-definition.json
{
  "family": "apg-employee-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::123456789012:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "apg-employee-api",
      "image": "123456789012.dkr.ecr.us-west-2.amazonaws.com/apg-employee-api:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "ENVIRONMENT", "value": "production"},
        {"name": "LOG_LEVEL", "value": "INFO"}
      ],
      "secrets": [
        {
          "name": "DATABASE_URL",
          "valueFrom": "arn:aws:secretsmanager:us-west-2:123456789012:secret:apg/database-url"
        },
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-west-2:123456789012:secret:apg/openai-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/apg-employee-api",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/api/v1/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

#### RDS Configuration
```bash
# Create RDS instance
aws rds create-db-instance \
  --db-instance-identifier apg-employee-db \
  --db-instance-class db.r5.large \
  --engine postgres \
  --engine-version 14.9 \
  --master-username apg_admin \
  --master-user-password $DB_PASSWORD \
  --allocated-storage 100 \
  --storage-type gp2 \
  --vpc-security-group-ids sg-12345678 \
  --db-subnet-group-name apg-db-subnet-group \
  --backup-retention-period 7 \
  --multi-az \
  --storage-encrypted \
  --kms-key-id arn:aws:kms:us-west-2:123456789012:key/12345678-1234-1234-1234-123456789012
```

#### ElastiCache Configuration
```bash
# Create Redis cluster
aws elasticache create-cache-cluster \
  --cache-cluster-id apg-employee-redis \
  --cache-node-type cache.r5.large \
  --engine redis \
  --engine-version 7.0 \
  --num-cache-nodes 1 \
  --cache-parameter-group default.redis7 \
  --cache-subnet-group-name apg-cache-subnet-group \
  --security-group-ids sg-87654321 \
  --at-rest-encryption-enabled \
  --transit-encryption-enabled \
  --auth-token $REDIS_AUTH_TOKEN
```

### Google Cloud Deployment

#### Cloud Run
```yaml
# service.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: apg-employee-api
  namespace: default
  annotations:
    run.googleapis.com/ingress: all
    run.googleapis.com/cpu-throttling: "false"
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "2"
        autoscaling.knative.dev/maxScale: "100"
        run.googleapis.com/cpu: "2"
        run.googleapis.com/memory: "4Gi"
        run.googleapis.com/execution-environment: gen2
    spec:
      containerConcurrency: 1000
      timeoutSeconds: 300
      containers:
      - image: gcr.io/project-id/apg-employee-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: apg-secrets
              key: database-url
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: apg-secrets
              key: openai-key
        resources:
          limits:
            cpu: "2"
            memory: "4Gi"
        livenessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

#### Cloud SQL Configuration
```bash
# Create Cloud SQL instance
gcloud sql instances create apg-employee-db \
  --database-version=POSTGRES_14 \
  --tier=db-custom-4-16384 \
  --region=us-central1 \
  --storage-type=SSD \
  --storage-size=100GB \
  --storage-auto-increase \
  --backup-start-time=03:00 \
  --backup-location=us-central1 \
  --maintenance-window-day=SUN \
  --maintenance-window-hour=04 \
  --maintenance-release-channel=production \
  --deletion-protection

# Create database
gcloud sql databases create apg_hr --instance=apg-employee-db

# Create user
gcloud sql users create apg_user \
  --instance=apg-employee-db \
  --password=$DB_PASSWORD
```

### Azure Deployment

#### Container Apps
```yaml
# container-app.yaml
apiVersion: app.containerapp.azure.com/v1
kind: ContainerApp
metadata:
  name: apg-employee-api
  resourceGroup: apg-employee-rg
spec:
  location: East US
  properties:
    managedEnvironmentId: /subscriptions/subscription-id/resourceGroups/apg-employee-rg/providers/Microsoft.App/managedEnvironments/apg-env
    configuration:
      ingress:
        external: true
        targetPort: 8000
        traffic:
        - weight: 100
          latestRevision: true
      secrets:
      - name: database-url
        value: postgresql://...
      - name: openai-key
        value: sk-...
    template:
      containers:
      - name: apg-employee-api
        image: acrregistry.azurecr.io/apg-employee-api:latest
        env:
        - name: DATABASE_URL
          secretRef: database-url
        - name: OPENAI_API_KEY
          secretRef: openai-key
        resources:
          cpu: 1.0
          memory: 2Gi
        probes:
        - type: liveness
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
      scale:
        minReplicas: 2
        maxReplicas: 10
        rules:
        - name: http-rule
          http:
            metadata:
              concurrentRequests: "10"
```

## 📊 Monitoring & Observability

### Prometheus Metrics

#### Application Metrics
```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import time

# Request metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

# Business metrics
EMPLOYEE_COUNT = Gauge(
    'apg_employees_total',
    'Total number of employees',
    ['tenant_id', 'status']
)

AI_ANALYSIS_COUNT = Counter(
    'apg_ai_analysis_total',
    'Total AI analyses performed',
    ['tenant_id', 'analysis_type']
)

AI_ANALYSIS_DURATION = Histogram(
    'apg_ai_analysis_duration_seconds',
    'AI analysis duration',
    ['analysis_type']
)

# Database metrics
DB_CONNECTIONS = Gauge(
    'apg_db_connections_active',
    'Active database connections'
)

# Middleware for request metrics
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    return response
```

#### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'apg-employee-api'
    static_configs:
      - targets: ['app:8000']
    metrics_path: '/metrics'
    scrape_interval: 15s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
    metrics_path: '/metrics'

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
    metrics_path: '/metrics'
```

### Grafana Dashboards

#### Main Dashboard Configuration
```json
{
  "dashboard": {
    "title": "APG Employee Management",
    "tags": ["apg", "employee", "api"],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
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
        "title": "Error Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5.*\"}[5m]) / rate(http_requests_total[5m])",
            "legendFormat": "Error Rate"
          }
        ]
      },
      {
        "title": "Active Employees",
        "type": "stat",
        "targets": [
          {
            "expr": "apg_employees_total{status=\"Active\"}",
            "legendFormat": "{{tenant_id}}"
          }
        ]
      }
    ]
  }
}
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
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if hasattr(record, 'tenant_id'):
            log_entry['tenant_id'] = record.tenant_id
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)

# Configure logging
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'json': {
            '()': JSONFormatter
        },
        'console': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'console',
            'level': 'INFO'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'logs/apg-employee.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'json',
            'level': 'INFO'
        }
    },
    'loggers': {
        'apg.employee': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False
        },
        'sqlalchemy.engine': {
            'handlers': ['file'],
            'level': 'WARNING',
            'propagate': False
        }
    },
    'root': {
        'handlers': ['console'],
        'level': 'WARNING'
    }
}
```

## 🔒 Security Configuration

### SSL/TLS Configuration

#### SSL Certificate Management
```bash
# Using Let's Encrypt with Certbot
sudo apt-get install certbot python3-certbot-nginx

# Generate certificate
sudo certbot --nginx -d api.company.com

# Auto-renewal
echo "0 2 * * * root certbot renew --quiet" | sudo tee -a /etc/crontab
```

#### nginx SSL Configuration
```nginx
# Strong SSL configuration
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
ssl_prefer_server_ciphers off;
ssl_session_cache shared:SSL:10m;
ssl_session_timeout 10m;

# HSTS
add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";

# Certificate transparency
ssl_ct on;
ssl_ct_static_scts /etc/nginx/ssl/scts;
```

### Firewall Configuration

#### UFW Rules
```bash
# Enable UFW
sudo ufw enable

# SSH access
sudo ufw allow ssh

# HTTP/HTTPS
sudo ufw allow 80
sudo ufw allow 443

# Database (only from application servers)
sudo ufw allow from 10.0.1.0/24 to any port 5432

# Redis (only from application servers)
sudo ufw allow from 10.0.1.0/24 to any port 6379

# Monitoring
sudo ufw allow from 10.0.1.0/24 to any port 9090
sudo ufw allow from 10.0.1.0/24 to any port 3000

# Check status
sudo ufw status verbose
```

#### iptables Rules
```bash
# Flush existing rules
iptables -F

# Default policies
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# Allow loopback
iptables -A INPUT -i lo -j ACCEPT

# Allow established connections
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# SSH
iptables -A INPUT -p tcp --dport 22 -j ACCEPT

# HTTP/HTTPS
iptables -A INPUT -p tcp --dport 80 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# Database (from app servers only)
iptables -A INPUT -p tcp -s 10.0.1.0/24 --dport 5432 -j ACCEPT

# Save rules
iptables-save > /etc/iptables/rules.v4
```

### Security Hardening

#### System Hardening
```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install security updates automatically
echo 'Unattended-Upgrade::Automatic-Reboot "false";' | sudo tee -a /etc/apt/apt.conf.d/50unattended-upgrades

# Disable root login
sudo sed -i 's/PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config

# Require key authentication
sudo sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config

# Restart SSH
sudo systemctl restart ssh

# Install fail2ban
sudo apt-get install fail2ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

#### Application Security
```python
# security.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import secrets

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://app.company.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"]
)

# Trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["api.company.com", "*.company.com"]
)

# Security headers middleware
@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    response.headers["X-Request-ID"] = secrets.token_hex(16)
    
    return response
```

## ⚡ Performance Optimization

### Database Optimization

#### Connection Pooling
```python
# database.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

# Optimized engine configuration
engine = create_async_engine(
    DATABASE_URL,
    # Connection pool settings
    poolclass=QueuePool,
    pool_size=20,                    # Base connections
    max_overflow=30,                 # Additional connections
    pool_recycle=3600,              # Recycle connections hourly
    pool_pre_ping=True,             # Validate connections
    
    # Performance settings
    connect_args={
        "server_settings": {
            "application_name": "apg_employee_api",
            "timezone": "UTC",
        },
        "command_timeout": 30,
        "statement_timeout": "30s"
    },
    
    # Async settings
    future=True,
    echo=False
)
```

#### Query Optimization
```sql
-- Performance monitoring queries
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    rows
FROM pg_stat_statements 
ORDER BY total_time DESC 
LIMIT 10;

-- Index usage analysis
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes 
ORDER BY idx_scan DESC;

-- Table statistics
SELECT 
    schemaname,
    tablename,
    n_tup_ins,
    n_tup_upd,
    n_tup_del,
    n_live_tup,
    n_dead_tup,
    last_vacuum,
    last_analyze
FROM pg_stat_user_tables;
```

### Caching Strategy

#### Multi-Level Caching
```python
# cache.py
import asyncio
from typing import Any, Optional, Union
import redis.asyncio as redis
import pickle
import json
from functools import wraps

class MultiLevelCache:
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
        self.local_cache = {}
        self.local_cache_size = 1000
        
    async def get(self, key: str, use_local: bool = True) -> Optional[Any]:
        # Try local cache first
        if use_local and key in self.local_cache:
            return self.local_cache[key]
        
        # Try Redis
        value = await self.redis.get(key)
        if value:
            deserialized = pickle.loads(value)
            
            # Store in local cache
            if use_local:
                self._update_local_cache(key, deserialized)
            
            return deserialized
        
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600, use_local: bool = True) -> bool:
        # Store in Redis
        serialized = pickle.dumps(value)
        result = await self.redis.setex(key, ttl, serialized)
        
        # Store in local cache
        if use_local:
            self._update_local_cache(key, value)
        
        return result
    
    def _update_local_cache(self, key: str, value: Any):
        if len(self.local_cache) >= self.local_cache_size:
            # Remove oldest item (simple LRU)
            oldest_key = next(iter(self.local_cache))
            del self.local_cache[oldest_key]
        
        self.local_cache[key] = value

# Cache decorator with TTL and invalidation
def cached_with_invalidation(
    key_pattern: str, 
    ttl: int = 3600,
    invalidate_on: list = None
):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = key_pattern.format(*args, **kwargs)
            
            # Try to get from cache
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Store in cache
            await cache.set(cache_key, result, ttl)
            
            return result
        
        # Add invalidation methods
        wrapper.invalidate = lambda *args, **kwargs: cache.delete(
            key_pattern.format(*args, **kwargs)
        )
        
        return wrapper
    return decorator
```

### Load Testing

#### Performance Testing Script
```python
# load_test.py
import asyncio
import aiohttp
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
import json

class LoadTester:
    def __init__(self, base_url: str, auth_token: str):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {auth_token}"}
        self.results = []
    
    async def run_load_test(
        self, 
        endpoint: str, 
        method: str = "GET",
        payload: dict = None,
        concurrent_users: int = 50,
        requests_per_user: int = 10,
        ramp_up_time: int = 10
    ):
        """Run load test with specified parameters"""
        
        print(f"Starting load test: {concurrent_users} users, {requests_per_user} requests each")
        
        # Calculate delay between user startups
        user_delay = ramp_up_time / concurrent_users
        
        # Create tasks for each user
        tasks = []
        for user_id in range(concurrent_users):
            task = asyncio.create_task(
                self._simulate_user(user_id, endpoint, method, payload, requests_per_user)
            )
            tasks.append(task)
            
            # Ramp up delay
            if user_id < concurrent_users - 1:
                await asyncio.sleep(user_delay)
        
        # Wait for all users to complete
        await asyncio.gather(*tasks)
        
        # Generate report
        self._generate_report()
    
    async def _simulate_user(self, user_id: int, endpoint: str, method: str, payload: dict, request_count: int):
        """Simulate a single user making multiple requests"""
        
        async with aiohttp.ClientSession() as session:
            for request_id in range(request_count):
                start_time = time.time()
                
                try:
                    async with session.request(
                        method=method,
                        url=f"{self.base_url}{endpoint}",
                        json=payload,
                        headers=self.headers
                    ) as response:
                        response_time = time.time() - start_time
                        
                        self.results.append({
                            'user_id': user_id,
                            'request_id': request_id,
                            'status_code': response.status,
                            'response_time': response_time,
                            'success': response.status < 400
                        })
                        
                except Exception as e:
                    response_time = time.time() - start_time
                    self.results.append({
                        'user_id': user_id,
                        'request_id': request_id,
                        'status_code': 0,
                        'response_time': response_time,
                        'success': False,
                        'error': str(e)
                    })
    
    def _generate_report(self):
        """Generate performance test report"""
        
        total_requests = len(self.results)
        successful_requests = sum(1 for r in self.results if r['success'])
        failed_requests = total_requests - successful_requests
        
        response_times = [r['response_time'] for r in self.results if r['success']]
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            median_response_time = statistics.median(response_times)
            p95_response_time = sorted(response_times)[int(len(response_times) * 0.95)]
            p99_response_time = sorted(response_times)[int(len(response_times) * 0.99)]
        else:
            avg_response_time = median_response_time = p95_response_time = p99_response_time = 0
        
        # Calculate throughput
        test_duration = max(r['response_time'] for r in self.results)
        throughput = successful_requests / test_duration if test_duration > 0 else 0
        
        report = {
            'summary': {
                'total_requests': total_requests,
                'successful_requests': successful_requests,
                'failed_requests': failed_requests,
                'success_rate': (successful_requests / total_requests) * 100,
                'throughput_rps': throughput
            },
            'response_times': {
                'average_ms': avg_response_time * 1000,
                'median_ms': median_response_time * 1000,
                'p95_ms': p95_response_time * 1000,
                'p99_ms': p99_response_time * 1000
            },
            'errors': [r for r in self.results if not r['success']]
        }
        
        print(json.dumps(report, indent=2))
        return report

# Usage example
async def main():
    tester = LoadTester(
        base_url="https://api.company.com",
        auth_token="your_jwt_token"
    )
    
    # Test employee listing endpoint
    await tester.run_load_test(
        endpoint="/api/v1/employees",
        concurrent_users=50,
        requests_per_user=20,
        ramp_up_time=30
    )

if __name__ == "__main__":
    asyncio.run(main())
```

## 💾 Backup & Disaster Recovery

### Database Backup

#### Automated Backup Script
```bash
#!/bin/bash
# backup_database.sh

set -e

# Configuration
DB_HOST="localhost"
DB_PORT="5432"
DB_NAME="apg_hr_production"
DB_USER="apg_user"
BACKUP_DIR="/backup/postgresql"
RETENTION_DAYS=30
S3_BUCKET="apg-backups"

# Create backup directory
mkdir -p $BACKUP_DIR

# Generate backup filename
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="$BACKUP_DIR/apg_hr_backup_$TIMESTAMP.sql.gz"

# Create database backup
echo "Creating database backup..."
pg_dump -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME | gzip > $BACKUP_FILE

# Verify backup
if [ -f "$BACKUP_FILE" ] && [ -s "$BACKUP_FILE" ]; then
    echo "Backup created successfully: $BACKUP_FILE"
    
    # Upload to S3
    aws s3 cp $BACKUP_FILE s3://$S3_BUCKET/postgresql/
    
    # Clean up local old backups
    find $BACKUP_DIR -name "*.sql.gz" -mtime +$RETENTION_DAYS -delete
    
    echo "Backup completed and uploaded to S3"
else
    echo "Backup failed!"
    exit 1
fi
```

#### Point-in-Time Recovery
```bash
#!/bin/bash
# restore_database.sh

set -e

BACKUP_FILE=$1
TARGET_DB_NAME=${2:-"apg_hr_restored"}

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file> [target_db_name]"
    exit 1
fi

# Create new database
createdb -U apg_user $TARGET_DB_NAME

# Restore from backup
echo "Restoring database from $BACKUP_FILE..."
gunzip -c $BACKUP_FILE | psql -U apg_user -d $TARGET_DB_NAME

echo "Database restored to $TARGET_DB_NAME"
```

### Application Backup

#### Configuration Backup
```bash
#!/bin/bash
# backup_config.sh

BACKUP_DIR="/backup/config"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
CONFIG_BACKUP="$BACKUP_DIR/config_backup_$TIMESTAMP.tar.gz"

mkdir -p $BACKUP_DIR

# Backup configuration files
tar -czf $CONFIG_BACKUP \
    /etc/nginx/nginx.conf \
    /etc/postgresql/14/main/postgresql.conf \
    /etc/redis/redis.conf \
    /opt/apg/employee_data_management/.env.production \
    /opt/apg/employee_data_management/docker-compose.yml

# Upload to S3
aws s3 cp $CONFIG_BACKUP s3://apg-backups/config/

echo "Configuration backup completed: $CONFIG_BACKUP"
```

### Disaster Recovery Plan

#### Recovery Procedures
```markdown
# Disaster Recovery Procedures

## 1. Database Recovery

### Scenario: Complete database loss

1. Provision new database server
2. Install PostgreSQL and extensions
3. Download latest backup from S3:
   ```bash
   aws s3 cp s3://apg-backups/postgresql/latest.sql.gz /tmp/
   ```
4. Restore database:
   ```bash
   gunzip /tmp/latest.sql.gz
   psql -U apg_user -d apg_hr < /tmp/latest.sql
   ```
5. Update application configuration
6. Restart application services

### Scenario: Partial data corruption

1. Stop application services
2. Create corrupted database backup
3. Restore from point-in-time backup
4. Apply transaction logs if available
5. Verify data integrity
6. Restart services

## 2. Application Recovery

### Scenario: Complete server loss

1. Provision new server infrastructure
2. Deploy from Docker images or git repository
3. Restore configuration from backup
4. Connect to restored database
5. Perform health checks
6. Update DNS/load balancer

### Scenario: Container failure

1. Check container logs for errors
2. Restart failed containers:
   ```bash
   docker-compose restart app
   ```
3. If restart fails, redeploy:
   ```bash
   docker-compose pull app
   docker-compose up -d app
   ```

## 3. Redis Recovery

### Scenario: Cache server failure

1. Redis failures are non-critical (cache only)
2. Application will continue without cache
3. Deploy new Redis instance
4. Update connection configuration
5. Cache will rebuild automatically

## 4. Load Balancer Recovery

### Scenario: nginx failure

1. Check nginx logs
2. Validate configuration
3. Restart nginx service:
   ```bash
   sudo systemctl restart nginx
   ```
4. If configuration corrupt, restore from backup

## 5. SSL Certificate Issues

### Scenario: Certificate expiry

1. Renew certificate:
   ```bash
   sudo certbot renew
   ```
2. Restart nginx:
   ```bash
   sudo systemctl restart nginx
   ```

## Recovery Time Objectives (RTO)

- Database recovery: 30 minutes
- Application recovery: 15 minutes
- Full system recovery: 1 hour
- Cache recovery: 5 minutes

## Recovery Point Objectives (RPO)

- Database: 1 hour (hourly backups)
- Configuration: 24 hours (daily backups)
- Application code: Real-time (git repository)
```

---

## 📞 Support & Troubleshooting

### Common Issues

#### Database Connection Issues
```bash
# Check database connectivity
pg_isready -h localhost -p 5432 -U apg_user

# Check active connections
psql -U apg_user -c "SELECT count(*) FROM pg_stat_activity;"

# Kill idle connections
psql -U apg_user -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'idle' AND state_change < now() - interval '1 hour';"
```

#### Performance Issues
```bash
# Check system resources
htop
iotop
free -h
df -h

# Check application logs
tail -f /var/log/apg-employee/app.log

# Check database performance
psql -U apg_user -c "SELECT query, state, query_start FROM pg_stat_activity WHERE state != 'idle';"
```

#### SSL/TLS Issues
```bash
# Test SSL configuration
openssl s_client -connect api.company.com:443 -servername api.company.com

# Check certificate expiry
openssl x509 -in /etc/ssl/certs/api.company.com.crt -text -noout | grep "Not After"

# Verify certificate chain
openssl verify -CAfile /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/api.company.com.crt
```

### Monitoring Alerts

#### Alert Rules (Prometheus)
```yaml
# alerts.yml
groups:
- name: apg-employee-alerts
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.*"}[5m]) / rate(http_requests_total[5m]) > 0.05
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} over the last 5 minutes"

  - alert: HighResponseTime
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1.0
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High response time detected"
      description: "95th percentile response time is {{ $value }}s"

  - alert: DatabaseConnectionsHigh
    expr: apg_db_connections_active > 80
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High database connection usage"
      description: "{{ $value }} active database connections"

  - alert: ServiceDown
    expr: up{job="apg-employee-api"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Service is down"
      description: "APG Employee API is not responding"
```

---

© 2025 Datacraft. All rights reserved.  
Deployment Guide Version 1.0 | Last Updated: January 2025