# Deployment Guide

Comprehensive guide for deploying APG Vendor Management to production environments with high availability, security, and performance optimizations.

## 📋 Table of Contents

- [Deployment Overview](#deployment-overview)
- [Infrastructure Requirements](#infrastructure-requirements)
- [Environment Setup](#environment-setup)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Database Setup](#database-setup)
- [Security Configuration](#security-configuration)
- [Monitoring & Logging](#monitoring--logging)
- [Performance Optimization](#performance-optimization)
- [Backup & Recovery](#backup--recovery)
- [Maintenance & Updates](#maintenance--updates)

## 🏗️ Deployment Overview

APG Vendor Management supports multiple deployment strategies:

### Deployment Options
- **Docker Compose**: Single-server development and small production
- **Kubernetes**: Scalable production with container orchestration
- **Traditional Server**: VM-based deployment with system services
- **Cloud Platforms**: AWS, Azure, GCP with managed services
- **Hybrid**: On-premise with cloud integrations

### Architecture Components
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │  Web Application│    │    Database     │
│     (nginx)     │◄──►│   (Flask/API)   │◄──►│  (PostgreSQL)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     CDN/Cache   │    │  Background     │    │   File Storage  │
│     (Redis)     │    │    Workers      │    │  (Object Store) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🖥️ Infrastructure Requirements

### Minimum Production Requirements

#### Application Server
```yaml
compute:
  cpu: 4 cores (Intel Xeon or AMD EPYC)
  memory: 8GB RAM
  storage: 100GB SSD
  network: 1Gbps connection

os:
  - Ubuntu 20.04+ LTS
  - RHEL 8+ / CentOS 8+
  - Amazon Linux 2
  - Debian 11+
```

#### Database Server
```yaml
database:
  cpu: 4 cores
  memory: 16GB RAM
  storage: 500GB SSD (with automatic backup)
  iops: 3000+ provisioned IOPS
  
database_engine:
  postgresql: 14.0+
  extensions:
    - pg_stat_statements
    - pgcrypto
    - uuid-ossp
```

#### Cache/Session Store
```yaml
cache:
  cpu: 2 cores
  memory: 4GB RAM
  storage: 50GB SSD
  
cache_engine:
  redis: 6.0+
  configuration: clustered for HA
```

### Recommended Production Setup

#### High Availability Configuration
```yaml
load_balancer:
  instances: 2 (active/passive)
  cpu: 2 cores each
  memory: 4GB each

application:
  instances: 3+ (horizontal scaling)
  cpu: 8 cores each
  memory: 16GB each
  storage: 200GB SSD each

database:
  primary: 1 instance
  replicas: 2+ read replicas
  cpu: 8 cores each
  memory: 32GB each
  storage: 1TB SSD with auto-scaling

cache:
  cluster: 3 nodes
  cpu: 4 cores each  
  memory: 8GB each
```

### Cloud Provider Recommendations

#### AWS
```yaml
compute:
  application: c5.2xlarge (8 vCPU, 16GB RAM)
  database: db.r5.2xlarge (8 vCPU, 64GB RAM)
  cache: cache.r6g.xlarge (4 vCPU, 26GB RAM)

storage:
  database: gp3 SSD with 3000 IOPS
  application: gp3 SSD
  backups: S3 Standard-IA

networking:
  vpc: Multi-AZ setup
  load_balancer: Application Load Balancer
  cdn: CloudFront
```

#### Azure
```yaml
compute:
  application: Standard_D8s_v4 (8 vCPU, 32GB RAM)
  database: Standard_D16s_v4 (16 vCPU, 64GB RAM)
  cache: Standard_D4s_v4 (4 vCPU, 16GB RAM)

storage:
  database: Premium SSD
  application: Standard SSD
  backups: Blob Storage Cool tier
```

#### Google Cloud
```yaml
compute:
  application: c2-standard-8 (8 vCPU, 32GB RAM)
  database: db-custom-8-32768 (8 vCPU, 32GB RAM)
  cache: n1-standard-4 (4 vCPU, 15GB RAM)

storage:
  database: SSD persistent disk
  application: Balanced persistent disk
  backups: Cloud Storage Nearline
```

## 🌍 Environment Setup

### Environment Configuration

#### Production Environment Variables
```bash
# Application Configuration
FLASK_ENV=production
FLASK_DEBUG=false
SECRET_KEY=your-production-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-key-here

# Database Configuration
DATABASE_URL=postgresql://username:password@db-host:5432/vendor_management
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=50
DATABASE_POOL_TIMEOUT=30

# Cache Configuration
REDIS_URL=redis://cache-host:6379/0
REDIS_PASSWORD=your-redis-password
REDIS_SSL=true

# Security Configuration
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
TRUSTED_HOSTS=yourdomain.com,app.yourdomain.com
SECURE_COOKIES=true
SESSION_COOKIE_SECURE=true

# AI/ML Configuration
AI_MODELS_PATH=/opt/vendor_management/models
AI_CONFIDENCE_THRESHOLD=0.75
AI_BATCH_SIZE=100
AI_PROCESSING_TIMEOUT=300

# Performance Configuration
WORKERS=4
WORKER_CLASS=gevent
WORKER_CONNECTIONS=1000
MAX_REQUESTS=10000
MAX_REQUESTS_JITTER=100

# Monitoring Configuration
METRICS_ENABLED=true
PROMETHEUS_PORT=9090
LOG_LEVEL=INFO
SENTRY_DSN=your-sentry-dsn-here

# External Integrations
SMTP_HOST=your-smtp-host
SMTP_PORT=587
SMTP_USERNAME=your-smtp-username
SMTP_PASSWORD=your-smtp-password
SMTP_USE_TLS=true

# File Storage
STORAGE_TYPE=s3
S3_BUCKET=your-vendor-management-bucket
S3_REGION=us-west-2
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
```

#### Environment-Specific Configurations

**Development**
```python
# config/development.py
class DevelopmentConfig:
    DEBUG = True
    TESTING = False
    DATABASE_URL = 'postgresql://localhost/vendor_management_dev'
    REDIS_URL = 'redis://localhost:6379/0'
    LOG_LEVEL = 'DEBUG'
    AI_MOCK_MODE = True
```

**Staging**
```python
# config/staging.py
class StagingConfig:
    DEBUG = False
    TESTING = True
    DATABASE_URL = os.environ.get('STAGING_DATABASE_URL')
    REDIS_URL = os.environ.get('STAGING_REDIS_URL')
    LOG_LEVEL = 'INFO'
    AI_MOCK_MODE = False
```

**Production**
```python
# config/production.py
class ProductionConfig:
    DEBUG = False
    TESTING = False
    DATABASE_URL = os.environ.get('DATABASE_URL')
    REDIS_URL = os.environ.get('REDIS_URL')
    LOG_LEVEL = 'WARNING'
    MONITORING_ENABLED = True
    ENCRYPTION_ENABLED = True
```

## 🐳 Docker Deployment

### Production Docker Setup

#### Dockerfile
```dockerfile
# Multi-stage build for production
FROM python:3.12-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.12-slim

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create directories
RUN mkdir -p /app/logs /app/static /app/models
RUN chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser . /app/
WORKDIR /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# Expose port
EXPOSE 5000

# Start application
CMD ["gunicorn", "--config", "gunicorn.conf.py", "app:app"]
```

#### Docker Compose Production
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  web:
    build:
      context: .
      dockerfile: Dockerfile
    image: vendor-management:latest
    restart: unless-stopped
    environment:
      - FLASK_ENV=production
      - DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD}@db:5432/vendor_management
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    networks:
      - vendor-network
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.vendor-management.rule=Host(`vendor.yourdomain.com`)"
      - "traefik.http.routers.vendor-management.tls=true"
      - "traefik.http.routers.vendor-management.tls.certresolver=letsencrypt"

  db:
    image: postgres:14-alpine
    restart: unless-stopped
    environment:
      - POSTGRES_DB=vendor_management
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database_schema.sql:/docker-entrypoint-initdb.d/01-schema.sql
    networks:
      - vendor-network
    command: >
      postgres
      -c shared_preload_libraries=pg_stat_statements
      -c pg_stat_statements.track=all
      -c max_connections=200
      -c shared_buffers=256MB
      -c effective_cache_size=1GB

  redis:
    image: redis:6-alpine
    restart: unless-stopped
    command: >
      redis-server
      --requirepass ${REDIS_PASSWORD}
      --maxmemory 512mb
      --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    networks:
      - vendor-network

  nginx:
    image: nginx:alpine
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
      - static_files:/var/www/static
    depends_on:
      - web
    networks:
      - vendor-network

  worker:
    build:
      context: .
      dockerfile: Dockerfile
    image: vendor-management:latest
    restart: unless-stopped
    command: celery -A app.celery worker --loglevel=info
    environment:
      - FLASK_ENV=production
      - DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD}@db:5432/vendor_management
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    networks:
      - vendor-network

  scheduler:
    build:
      context: .
      dockerfile: Dockerfile
    image: vendor-management:latest
    restart: unless-stopped
    command: celery -A app.celery beat --loglevel=info
    environment:
      - FLASK_ENV=production
      - DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD}@db:5432/vendor_management
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    networks:
      - vendor-network

volumes:
  postgres_data:
    driver: local
  redis_data:
    driver: local
  static_files:
    driver: local

networks:
  vendor-network:
    driver: bridge
```

#### Nginx Configuration
```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream vendor_management {
        server web:5000;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=login:10m rate=1r/s;

    server {
        listen 80;
        server_name vendor.yourdomain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name vendor.yourdomain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
        ssl_prefer_server_ciphers off;

        # Security headers
        add_header X-Content-Type-Options nosniff;
        add_header X-Frame-Options DENY;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

        # Static files
        location /static/ {
            alias /var/www/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }

        # API endpoints with rate limiting
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://vendor_management;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Login endpoint with stricter rate limiting
        location /api/v1/auth/login {
            limit_req zone=login burst=3 nodelay;
            proxy_pass http://vendor_management;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Main application
        location / {
            proxy_pass http://vendor_management;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket support
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
    }
}
```

#### Deployment Commands
```bash
# Build and deploy
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d

# Scale web workers
docker-compose -f docker-compose.prod.yml up -d --scale web=3

# View logs
docker-compose -f docker-compose.prod.yml logs -f web

# Database migration
docker-compose -f docker-compose.prod.yml exec web flask db upgrade

# Health check
curl -f https://vendor.yourdomain.com/health
```

## ☸️ Kubernetes Deployment

### Kubernetes Manifests

#### Namespace
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: vendor-management
  labels:
    name: vendor-management
```

#### ConfigMap
```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: vendor-management-config
  namespace: vendor-management
data:
  FLASK_ENV: "production"
  LOG_LEVEL: "INFO"
  AI_CONFIDENCE_THRESHOLD: "0.75"
  DATABASE_POOL_SIZE: "20"
  REDIS_DB: "0"
```

#### Secrets
```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: vendor-management-secrets
  namespace: vendor-management
type: Opaque
data:
  SECRET_KEY: <base64-encoded-secret>
  JWT_SECRET_KEY: <base64-encoded-jwt-secret>
  DATABASE_URL: <base64-encoded-database-url>
  REDIS_PASSWORD: <base64-encoded-redis-password>
```

#### Deployment
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vendor-management-web
  namespace: vendor-management
  labels:
    app: vendor-management
    component: web
spec:
  replicas: 3
  selector:
    matchLabels:
      app: vendor-management
      component: web
  template:
    metadata:
      labels:
        app: vendor-management
        component: web
    spec:
      containers:
      - name: web
        image: vendor-management:latest
        ports:
        - containerPort: 5000
        env:
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: vendor-management-secrets
              key: SECRET_KEY
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: vendor-management-secrets
              key: DATABASE_URL
        envFrom:
        - configMapRef:
            name: vendor-management-config
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 2
            memory: 4Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 5000
          initialDelaySeconds: 10
          periodSeconds: 5
        volumeMounts:
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: logs
        emptyDir: {}
      imagePullSecrets:
      - name: registry-secret
```

#### Service
```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: vendor-management-service
  namespace: vendor-management
spec:
  selector:
    app: vendor-management
    component: web
  ports:
  - protocol: TCP
    port: 80
    targetPort: 5000
  type: ClusterIP
```

#### Ingress
```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: vendor-management-ingress
  namespace: vendor-management
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - vendor.yourdomain.com
    secretName: vendor-management-tls
  rules:
  - host: vendor.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: vendor-management-service
            port:
              number: 80
```

#### HorizontalPodAutoscaler
```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vendor-management-hpa
  namespace: vendor-management
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vendor-management-web
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
```

#### Database (PostgreSQL)
```yaml
# k8s/postgres.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: vendor-management
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
          value: vendor_management
        - name: POSTGRES_USER
          value: postgres
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: vendor-management-secrets
              key: POSTGRES_PASSWORD
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            cpu: 1
            memory: 2Gi
          limits:
            cpu: 4
            memory: 8Gi
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

#### Deployment Commands
```bash
# Apply all manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n vendor-management

# Scale deployment
kubectl scale deployment vendor-management-web --replicas=5 -n vendor-management

# View logs
kubectl logs -f deployment/vendor-management-web -n vendor-management

# Port forward for testing
kubectl port-forward service/vendor-management-service 8080:80 -n vendor-management
```

## 🗄️ Database Setup

### PostgreSQL Production Configuration

#### Database Server Setup
```bash
# Install PostgreSQL 14
sudo apt update
sudo apt install postgresql-14 postgresql-14-contrib

# Configure PostgreSQL
sudo -u postgres psql
```

#### Production Configuration
```sql
-- postgresql.conf settings
max_connections = 200
shared_buffers = 1GB
effective_cache_size = 4GB
maintenance_work_mem = 256MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
work_mem = 4MB

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
```

#### Database Creation and Setup
```bash
# Create production database
sudo -u postgres createdb vendor_management_prod

# Create database user
sudo -u postgres psql -c "CREATE USER vm_user WITH PASSWORD 'secure_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE vendor_management_prod TO vm_user;"

# Import schema
psql -U vm_user -d vendor_management_prod -f database_schema.sql

# Verify installation
psql -U vm_user -d vendor_management_prod -c "\dt vm_*"
```

#### Database Backup Configuration
```bash
# Create backup script
cat << 'EOF' > /opt/scripts/backup-db.sh
#!/bin/bash
BACKUP_DIR="/opt/backups/postgres"
DATE=$(date +%Y%m%d_%H%M%S)
DB_NAME="vendor_management_prod"

# Create backup directory
mkdir -p $BACKUP_DIR

# Perform backup
pg_dump -U vm_user -h localhost $DB_NAME | gzip > $BACKUP_DIR/vm_backup_$DATE.sql.gz

# Keep only last 30 days of backups
find $BACKUP_DIR -name "vm_backup_*.sql.gz" -mtime +30 -delete

# Upload to S3 (optional)
aws s3 cp $BACKUP_DIR/vm_backup_$DATE.sql.gz s3://your-backup-bucket/database/
EOF

chmod +x /opt/scripts/backup-db.sh

# Schedule daily backups
echo "0 2 * * * /opt/scripts/backup-db.sh" | sudo crontab -
```

### Redis Production Setup

#### Redis Configuration
```bash
# Install Redis
sudo apt install redis-server

# Configure Redis
sudo nano /etc/redis/redis.conf
```

```conf
# redis.conf production settings
bind 127.0.0.1 ::1
port 6379
requirepass your-secure-redis-password
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

```bash
# Start Redis
sudo systemctl enable redis-server
sudo systemctl start redis-server
```

## 🔒 Security Configuration

### SSL/TLS Setup

#### Let's Encrypt Certificate
```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Generate certificate
sudo certbot --nginx -d vendor.yourdomain.com

# Auto-renewal
echo "0 12 * * * /usr/bin/certbot renew --quiet" | sudo crontab -
```

#### Custom Certificate
```bash
# Generate private key
openssl genrsa -out vendor-management.key 2048

# Generate certificate signing request
openssl req -new -key vendor-management.key -out vendor-management.csr

# Generate self-signed certificate (for testing)
openssl x509 -req -days 365 -in vendor-management.csr -signkey vendor-management.key -out vendor-management.crt
```

### Firewall Configuration
```bash
# Configure UFW firewall
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw --force enable

# Application-specific rules
sudo ufw allow from 10.0.0.0/8 to any port 5432  # PostgreSQL
sudo ufw allow from 10.0.0.0/8 to any port 6379  # Redis
```

### Security Hardening
```bash
# Disable root login
sudo sed -i 's/PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config

# Configure fail2ban
sudo apt install fail2ban
sudo cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local

# System updates
sudo apt update && sudo apt upgrade -y
sudo apt install unattended-upgrades
```

## 📊 Monitoring & Logging

### Prometheus Monitoring Setup

#### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'vendor-management'
    static_configs:
      - targets: ['localhost:5000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'postgres'
    static_configs:
      - targets: ['localhost:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['localhost:9121']

  - job_name: 'nginx'
    static_configs:
      - targets: ['localhost:9113']
```

#### Grafana Dashboard
```json
{
  "dashboard": {
    "title": "Vendor Management Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(flask_http_request_total[5m])"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph", 
        "targets": [
          {
            "expr": "histogram_quantile(0.95, flask_http_request_duration_seconds_bucket)"
          }
        ]
      },
      {
        "title": "Database Connections",
        "type": "graph",
        "targets": [
          {
            "expr": "pg_stat_database_numbackends"
          }
        ]
      }
    ]
  }
}
```

### Logging Configuration

#### Application Logging
```python
# logging_config.py
import logging
import logging.handlers
import os

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
        'json': {
            'format': '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'detailed'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/app/logs/vendor_management.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'json'
        },
        'syslog': {
            'class': 'logging.handlers.SysLogHandler',
            'address': '/dev/log',
            'formatter': 'detailed'
        }
    },
    'loggers': {
        'vendor_management': {
            'handlers': ['console', 'file', 'syslog'],
            'level': 'INFO',
            'propagate': False
        }
    }
}
```

#### ELK Stack Integration
```yaml
# docker-compose.logging.yml
version: '3.8'

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.15.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data

  logstash:
    image: docker.elastic.co/logstash/logstash:7.15.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:7.15.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    depends_on:
      - elasticsearch

volumes:
  elasticsearch_data:
```

## ⚡ Performance Optimization

### Application Performance

#### Gunicorn Configuration
```python
# gunicorn.conf.py
bind = "0.0.0.0:5000"
workers = 4
worker_class = "gevent"
worker_connections = 1000
max_requests = 10000
max_requests_jitter = 100
timeout = 30
keepalive = 60
preload_app = True

# Logging
accesslog = "/app/logs/access.log"
errorlog = "/app/logs/error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "vendor-management"

# Security
limit_request_line = 4094
limit_request_fields = 100
```

#### Database Optimization
```sql
-- Performance tuning queries
-- Check slow queries
SELECT query, mean_time, calls, total_time
FROM pg_stat_statements
ORDER BY total_time DESC
LIMIT 10;

-- Index usage analysis
SELECT schemaname, tablename, attname, n_distinct, correlation
FROM pg_stats
WHERE tablename LIKE 'vm_%'
ORDER BY n_distinct DESC;

-- Connection monitoring
SELECT count(*), state
FROM pg_stat_activity
GROUP BY state;
```

#### Caching Strategy
```python
# caching.py
import redis
from functools import wraps

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_result(timeout=3600):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            redis_client.setex(cache_key, timeout, json.dumps(result))
            return result
        return wrapper
    return decorator
```

### Infrastructure Optimization

#### Load Balancer Configuration
```nginx
# upstream configuration with health checks
upstream vendor_management_backend {
    least_conn;
    server web1:5000 max_fails=3 fail_timeout=30s;
    server web2:5000 max_fails=3 fail_timeout=30s;
    server web3:5000 max_fails=3 fail_timeout=30s;
    keepalive 32;
}

# Connection pooling
proxy_http_version 1.1;
proxy_set_header Connection "";
proxy_buffering on;
proxy_buffer_size 8k;
proxy_buffers 16 8k;
```

#### CDN Configuration
```yaml
# CloudFront distribution
Resources:
  VendorManagementCDN:
    Type: AWS::CloudFront::Distribution
    Properties:
      DistributionConfig:
        Origins:
          - DomainName: vendor.yourdomain.com
            Id: VendorManagementOrigin
            CustomOriginConfig:
              HTTPPort: 443
              OriginProtocolPolicy: https-only
        DefaultCacheBehavior:
          TargetOriginId: VendorManagementOrigin
          ViewerProtocolPolicy: redirect-to-https
          Compress: true
          CachePolicyId: 4135ea2d-6df8-44a3-9df3-4b5a84be39ad
        CacheBehaviors:
          - PathPattern: "/api/*"
            TargetOriginId: VendorManagementOrigin
            ViewerProtocolPolicy: https-only
            CachePolicyId: 4135ea2d-6df8-44a3-9df3-4b5a84be39ad
            TTL: 0
          - PathPattern: "/static/*"
            TargetOriginId: VendorManagementOrigin
            ViewerProtocolPolicy: https-only
            CachePolicyId: 658327ea-f89d-4fab-a63d-7e88639e58f6
            TTL: 31536000
```

## 💾 Backup & Recovery

### Automated Backup Strategy

#### Database Backup Script
```bash
#!/bin/bash
# backup-database.sh

set -e

# Configuration
DB_NAME="vendor_management_prod"
DB_USER="vm_user"
BACKUP_DIR="/opt/backups/postgres"
S3_BUCKET="your-backup-bucket"
RETENTION_DAYS=30

# Create backup directory
mkdir -p $BACKUP_DIR

# Generate timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="vm_backup_${TIMESTAMP}.sql.gz"

# Perform database backup
echo "Starting database backup..."
pg_dump -U $DB_USER -h localhost $DB_NAME | gzip > $BACKUP_DIR/$BACKUP_FILE

# Verify backup
if [ $? -eq 0 ]; then
    echo "Database backup completed: $BACKUP_FILE"
    
    # Upload to S3
    aws s3 cp $BACKUP_DIR/$BACKUP_FILE s3://$S3_BUCKET/database/
    
    # Clean old backups
    find $BACKUP_DIR -name "vm_backup_*.sql.gz" -mtime +$RETENTION_DAYS -delete
    
    echo "Backup uploaded and old backups cleaned"
else
    echo "Database backup failed!"
    exit 1
fi
```

#### Application Data Backup
```bash
#!/bin/bash
# backup-application.sh

# Backup application files
tar -czf /opt/backups/app_backup_$(date +%Y%m%d).tar.gz \
    /app \
    /etc/nginx/sites-available \
    /etc/systemd/system/vendor-management.service

# Backup Redis data
redis-cli --rdb /opt/backups/redis_backup_$(date +%Y%m%d).rdb

# Upload to S3
aws s3 sync /opt/backups/ s3://your-backup-bucket/application/
```

### Disaster Recovery Plan

#### Recovery Procedures
```bash
# 1. Database Recovery
# Stop application
sudo systemctl stop vendor-management

# Restore database
gunzip -c vm_backup_20250129_120000.sql.gz | psql -U vm_user -d vendor_management_prod

# 2. Application Recovery
# Restore application files
tar -xzf app_backup_20250129.tar.gz -C /

# Restore Redis data
sudo systemctl stop redis
cp redis_backup_20250129.rdb /var/lib/redis/dump.rdb
sudo chown redis:redis /var/lib/redis/dump.rdb
sudo systemctl start redis

# 3. Restart services
sudo systemctl start vendor-management
sudo systemctl start nginx

# 4. Verify recovery
curl -f https://vendor.yourdomain.com/health
```

#### Backup Verification
```bash
# Test backup integrity
gunzip -t vm_backup_20250129_120000.sql.gz

# Test restore in isolated environment
docker run --rm -d --name test-postgres postgres:14
sleep 10
gunzip -c vm_backup_20250129_120000.sql.gz | docker exec -i test-postgres psql -U postgres
docker exec test-postgres psql -U postgres -c "\dt vm_*"
docker stop test-postgres
```

## 🔄 Maintenance & Updates

### Rolling Updates

#### Application Updates
```bash
# 1. Build new image
docker build -t vendor-management:v1.1.0 .

# 2. Update one instance at a time
docker-compose -f docker-compose.prod.yml up -d --scale web=2
docker-compose -f docker-compose.prod.yml stop web_1
docker-compose -f docker-compose.prod.yml rm web_1
docker-compose -f docker-compose.prod.yml up -d --scale web=3

# 3. Verify health
curl -f https://vendor.yourdomain.com/health

# 4. Repeat for other instances
```

#### Database Migrations
```bash
# 1. Backup database before migration
/opt/scripts/backup-db.sh

# 2. Run migrations
docker-compose -f docker-compose.prod.yml exec web flask db upgrade

# 3. Verify migration
docker-compose -f docker-compose.prod.yml exec web flask db current
```

### Health Checks

#### Application Health Check
```python
# health.py
from flask import Blueprint, jsonify
import redis
import psycopg2

health_bp = Blueprint('health', __name__)

@health_bp.route('/health')
def health_check():
    """Comprehensive health check endpoint"""
    
    status = {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'checks': {}
    }
    
    # Database check
    try:
        db_conn = psycopg2.connect(DATABASE_URL)
        db_conn.cursor().execute('SELECT 1')
        db_conn.close()
        status['checks']['database'] = 'healthy'
    except Exception as e:
        status['checks']['database'] = f'unhealthy: {str(e)}'
        status['status'] = 'unhealthy'
    
    # Redis check
    try:
        redis_client = redis.from_url(REDIS_URL)
        redis_client.ping()
        status['checks']['redis'] = 'healthy'
    except Exception as e:
        status['checks']['redis'] = f'unhealthy: {str(e)}'
        status['status'] = 'unhealthy'
    
    # AI service check
    try:
        # Check AI service availability
        status['checks']['ai_service'] = 'healthy'
    except Exception as e:
        status['checks']['ai_service'] = f'degraded: {str(e)}'
    
    return jsonify(status), 200 if status['status'] == 'healthy' else 503
```

#### Monitoring Script
```bash
#!/bin/bash
# monitor.sh

# Health check
HEALTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" https://vendor.yourdomain.com/health)

if [ $HEALTH_STATUS -ne 200 ]; then
    echo "Health check failed with status: $HEALTH_STATUS"
    # Send alert
    curl -X POST -H 'Content-type: application/json' \
        --data '{"text":"Vendor Management health check failed!"}' \
        $SLACK_WEBHOOK_URL
fi

# Check disk space
DISK_USAGE=$(df / | awk 'NR==2{print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 80 ]; then
    echo "Disk usage is ${DISK_USAGE}%"
    # Send alert
fi

# Check memory usage
MEMORY_USAGE=$(free | awk 'NR==2{printf "%.0f", $3*100/$2}')
if [ $MEMORY_USAGE -gt 85 ]; then
    echo "Memory usage is ${MEMORY_USAGE}%"
    # Send alert
fi
```

### System Maintenance

#### Regular Maintenance Tasks
```bash
# Weekly maintenance script
#!/bin/bash

# 1. System updates
sudo apt update && sudo apt upgrade -y

# 2. Docker cleanup
docker system prune -f
docker volume prune -f

# 3. Log rotation
sudo logrotate -f /etc/logrotate.conf

# 4. Database maintenance
docker-compose exec db psql -U postgres -d vendor_management -c "VACUUM ANALYZE;"

# 5. SSL certificate renewal
sudo certbot renew --quiet

# 6. Backup verification
/opt/scripts/verify-backups.sh

echo "Maintenance completed at $(date)"
```

#### Security Updates
```bash
# Security patch deployment
#!/bin/bash

# 1. Check for security updates
sudo apt list --upgradable | grep -i security

# 2. Apply security patches
sudo unattended-upgrade -d

# 3. Restart services if needed
sudo systemctl daemon-reload

# 4. Verify system status
systemctl status vendor-management
systemctl status nginx
systemctl status postgresql
```

---

## 🎯 Deployment Checklist

### Pre-Deployment
- [ ] Infrastructure provisioned and configured
- [ ] SSL certificates obtained and installed
- [ ] Database created and schema deployed
- [ ] Environment variables configured
- [ ] Security hardening completed
- [ ] Monitoring and logging configured
- [ ] Backup systems tested

### Deployment
- [ ] Application built and deployed
- [ ] Database migrations executed
- [ ] Health checks passing
- [ ] Load balancer configured
- [ ] DNS records updated
- [ ] CDN configured (if applicable)

### Post-Deployment
- [ ] Smoke tests completed
- [ ] Performance benchmarks verified
- [ ] Monitoring alerts configured
- [ ] Documentation updated
- [ ] Team notified of deployment
- [ ] Rollback plan verified

### Production Readiness
- [ ] Load testing completed
- [ ] Security scan passed
- [ ] Disaster recovery tested
- [ ] Support runbooks updated
- [ ] Monitoring dashboards configured
- [ ] SLA agreements in place

---

*This completes the comprehensive deployment guide. For specific cloud provider deployment guides, refer to the cloud-specific documentation sections.* 🚀