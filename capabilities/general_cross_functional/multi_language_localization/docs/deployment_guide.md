# Multi-language Localization Deployment Guide

Complete deployment guide for the APG Multi-language Localization capability across different environments including Docker, Kubernetes, cloud platforms, and production best practices.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development-setup)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Cloud Platform Deployment](#cloud-platform-deployment)
6. [Production Configuration](#production-configuration)
7. [Monitoring & Observability](#monitoring--observability)
8. [Security Configuration](#security-configuration)
9. [Performance Optimization](#performance-optimization)
10. [Backup & Recovery](#backup--recovery)
11. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

**Minimum Requirements**:
- CPU: 2 cores
- RAM: 4GB
- Storage: 20GB SSD
- Network: 100Mbps

**Recommended for Production**:
- CPU: 4+ cores
- RAM: 8GB+
- Storage: 100GB+ SSD
- Network: 1Gbps+

### Software Dependencies

- **Python**: 3.11+
- **PostgreSQL**: 14+
- **Redis**: 6.0+
- **Docker**: 20.10+
- **Kubernetes**: 1.25+ (for K8s deployment)

### External Services

- **Database**: PostgreSQL with async support
- **Cache**: Redis for translation caching
- **Translation APIs**: Google Translate, Azure Translator (optional)
- **File Storage**: S3-compatible storage for translation files
- **Monitoring**: Prometheus, Grafana (recommended)

## Local Development Setup

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/datacraft/apg.git
cd apg/capabilities/general_cross_functional/multi_language_localization

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. Database Setup

```bash
# Start PostgreSQL (via Docker)
docker run -d \
  --name localization-db \
  -e POSTGRES_DB=localization \
  -e POSTGRES_USER=localization_user \
  -e POSTGRES_PASSWORD=secure_password \
  -p 5432:5432 \
  postgres:14

# Run migrations
alembic upgrade head
```

### 3. Redis Setup

```bash
# Start Redis (via Docker)
docker run -d \
  --name localization-redis \
  -p 6379:6379 \
  redis:6-alpine
```

### 4. Configuration

Create `.env` file:
```bash
# Database
DATABASE_URL=postgresql+asyncpg://localization_user:secure_password@localhost:5432/localization

# Redis
REDIS_URL=redis://localhost:6379/0

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true

# Translation Services
GOOGLE_TRANSLATE_API_KEY=your_google_api_key
AZURE_TRANSLATOR_KEY=your_azure_key

# Security
SECRET_KEY=your_super_secret_key_here
JWT_SECRET_KEY=your_jwt_secret_key
```

### 5. Start Development Server

```bash
# Start the API server
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# Start background workers (separate terminal)
celery -A service.tasks worker --loglevel=info

# Run tests
pytest tests/ -v --cov=multi_language_localization
```

## Docker Deployment

### 1. Dockerfile

```dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        postgresql-client \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. Docker Compose

```yaml
version: '3.8'

services:
  localization-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+asyncpg://localization_user:secure_password@db:5432/localization
      - REDIS_URL=redis://redis:6379/0
      - SECRET_KEY=your_super_secret_key_here
    depends_on:
      - db
      - redis
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped

  localization-worker:
    build: .
    command: celery -A service.tasks worker --loglevel=info
    environment:
      - DATABASE_URL=postgresql+asyncpg://localization_user:secure_password@db:5432/localization
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    restart: unless-stopped

  db:
    image: postgres:14
    environment:
      - POSTGRES_DB=localization
      - POSTGRES_USER=localization_user
      - POSTGRES_PASSWORD=secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped

  redis:
    image: redis:6-alpine
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/ssl/certs
    depends_on:
      - localization-api
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

### 3. Build and Deploy

```bash
# Build the image
docker build -t datacraft/apg-localization:latest .

# Start services
docker-compose up -d

# Check logs
docker-compose logs -f localization-api

# Run migrations
docker-compose exec localization-api alembic upgrade head

# Scale workers
docker-compose up -d --scale localization-worker=3
```

## Kubernetes Deployment

### 1. Namespace and ConfigMap

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: localization
---
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: localization-config
  namespace: localization
data:
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
  REDIS_URL: "redis://localization-redis:6379/0"
  ENABLE_AUTO_TRANSLATION: "true"
  DEFAULT_LOCALE: "en-US"
  LOG_LEVEL: "INFO"
```

### 2. Secrets

```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: localization-secrets
  namespace: localization
type: Opaque
data:
  database-url: cG9zdGdyZXNxbCthc3luY3BnOi8vbG9jYWxpemF0aW9uX3VzZXI6c2VjdXJlX3Bhc3N3b3JkQGxvY2FsaXphdGlvbi1kYjo1NDMyL2xvY2FsaXphdGlvbg==
  secret-key: eW91cl9zdXBlcl9zZWNyZXRfa2V5X2hlcmU=
  jwt-secret: eW91cl9qd3Rfc2VjcmV0X2tleQ==
  google-api-key: eW91cl9nb29nbGVfYXBpX2tleQ==
  azure-translator-key: eW91cl9henVyZV9rZXk=
```

### 3. PostgreSQL Deployment

```yaml
# postgres.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: localization-db
  namespace: localization
spec:
  serviceName: localization-db
  replicas: 1
  selector:
    matchLabels:
      app: localization-db
  template:
    metadata:
      labels:
        app: localization-db
    spec:
      containers:
      - name: postgres
        image: postgres:14
        env:
        - name: POSTGRES_DB
          value: localization
        - name: POSTGRES_USER
          value: localization_user
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-data
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
  volumeClaimTemplates:
  - metadata:
      name: postgres-data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 20Gi
---
apiVersion: v1
kind: Service
metadata:
  name: localization-db
  namespace: localization
spec:
  selector:
    app: localization-db
  ports:
  - port: 5432
    targetPort: 5432
  clusterIP: None
```

### 4. Redis Deployment

```yaml
# redis.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: localization-redis
  namespace: localization
spec:
  replicas: 1
  selector:
    matchLabels:
      app: localization-redis
  template:
    metadata:
      labels:
        app: localization-redis
    spec:
      containers:
      - name: redis
        image: redis:6-alpine
        ports:
        - containerPort: 6379
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        volumeMounts:
        - name: redis-data
          mountPath: /data
      volumes:
      - name: redis-data
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: localization-redis
  namespace: localization
spec:
  selector:
    app: localization-redis
  ports:
  - port: 6379
    targetPort: 6379
```

### 5. Application Deployment

```yaml
# app-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: localization-api
  namespace: localization
spec:
  replicas: 3
  selector:
    matchLabels:
      app: localization-api
  template:
    metadata:
      labels:
        app: localization-api
    spec:
      containers:
      - name: localization-api
        image: datacraft/apg-localization:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: localization-secrets
              key: database-url
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: localization-secrets
              key: secret-key
        envFrom:
        - configMapRef:
            name: localization-config
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: localization-api
  namespace: localization
spec:
  selector:
    app: localization-api
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP
```

### 6. Ingress Configuration

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: localization-ingress
  namespace: localization
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - localization.example.com
    secretName: localization-tls
  rules:
  - host: localization.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: localization-api
            port:
              number: 8000
```

### 7. Horizontal Pod Autoscaler

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: localization-api-hpa
  namespace: localization
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: localization-api
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

### 8. Deploy to Kubernetes

```bash
# Apply configurations
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f secrets.yaml
kubectl apply -f postgres.yaml
kubectl apply -f redis.yaml
kubectl apply -f app-deployment.yaml
kubectl apply -f ingress.yaml
kubectl apply -f hpa.yaml

# Wait for deployments
kubectl wait --for=condition=available --timeout=300s deployment/localization-api -n localization

# Run migrations
kubectl exec -it deployment/localization-api -n localization -- alembic upgrade head

# Check status
kubectl get pods -n localization
kubectl logs -f deployment/localization-api -n localization
```

## Cloud Platform Deployment

### AWS EKS Deployment

```bash
# Create EKS cluster
eksctl create cluster \
  --name localization-cluster \
  --region us-west-2 \
  --nodegroup-name localization-nodes \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 10 \
  --node-type t3.medium

# Install AWS Load Balancer Controller
kubectl apply -k "github.com/aws/eks-charts/stable/aws-load-balancer-controller//crds?ref=master"

# Deploy application
kubectl apply -f k8s/
```

### Google GKE Deployment

```bash
# Create GKE cluster
gcloud container clusters create localization-cluster \
  --zone us-central1-a \
  --num-nodes 3 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 10 \
  --machine-type e2-medium

# Get credentials
gcloud container clusters get-credentials localization-cluster --zone us-central1-a

# Deploy application
kubectl apply -f k8s/
```

### Azure AKS Deployment

```bash
# Create resource group
az group create --name localization-rg --location eastus

# Create AKS cluster
az aks create \
  --resource-group localization-rg \
  --name localization-cluster \
  --node-count 3 \
  --enable-addons monitoring \
  --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group localization-rg --name localization-cluster

# Deploy application
kubectl apply -f k8s/
```

## Production Configuration

### 1. Environment Variables

```bash
# Production environment file
# Database
DATABASE_URL=postgresql+asyncpg://prod_user:secure_pass@prod-db-cluster:5432/localization
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Redis
REDIS_URL=redis://prod-redis-cluster:6379/0
REDIS_POOL_SIZE=50

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
WORKERS=4
DEBUG=false
LOG_LEVEL=INFO

# Security
SECRET_KEY=production_secret_key_here
JWT_SECRET_KEY=production_jwt_secret
CORS_ORIGINS=["https://app.example.com", "https://admin.example.com"]

# Translation Services
GOOGLE_TRANSLATE_API_KEY=prod_google_key
AZURE_TRANSLATOR_KEY=prod_azure_key
ENABLE_AUTO_TRANSLATION=true

# Performance
CACHE_TTL=3600
ENABLE_COMPRESSION=true
MAX_REQUEST_SIZE=10485760

# Monitoring
SENTRY_DSN=https://your-sentry-dsn
PROMETHEUS_ENABLED=true
METRICS_PORT=9090
```

### 2. Gunicorn Configuration

```python
# gunicorn.conf.py
bind = "0.0.0.0:8000"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
timeout = 30
keepalive = 2
preload_app = True

# Logging
accesslog = "/app/logs/access.log"
errorlog = "/app/logs/error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'
```

### 3. Nginx Configuration

```nginx
# nginx.conf
upstream localization_backend {
    server localization-api:8000;
}

server {
    listen 80;
    server_name localization.example.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name localization.example.com;

    ssl_certificate /etc/ssl/certs/localization.crt;
    ssl_certificate_key /etc/ssl/private/localization.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;

    # Gzip compression
    gzip on;
    gzip_types
        application/json
        application/javascript
        text/css
        text/plain
        text/xml;

    location / {
        proxy_pass http://localization_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }

    location /health {
        proxy_pass http://localization_backend/health;
        access_log off;
    }

    location /metrics {
        proxy_pass http://localization_backend/metrics;
        allow 10.0.0.0/8;
        deny all;
    }
}
```

## Monitoring & Observability

### 1. Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "localization_rules.yml"

scrape_configs:
  - job_name: 'localization-api'
    static_configs:
      - targets: ['localization-api:9090']
    metrics_path: '/metrics'
    scrape_interval: 30s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### 2. Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Localization Service Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "Requests/sec"
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
        "title": "Translation Cache Hit Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "translation_cache_hit_rate",
            "legendFormat": "Cache Hit Rate"
          }
        ]
      }
    ]
  }
}
```

### 3. Application Metrics

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')

# Translation metrics
TRANSLATION_REQUESTS = Counter('translation_requests_total', 'Total translation requests', ['language'])
CACHE_HITS = Counter('translation_cache_hits_total', 'Translation cache hits')
CACHE_MISSES = Counter('translation_cache_misses_total', 'Translation cache misses')

# System metrics
ACTIVE_CONNECTIONS = Gauge('active_db_connections', 'Active database connections')
REDIS_MEMORY_USAGE = Gauge('redis_memory_usage_bytes', 'Redis memory usage')
```

### 4. Logging Configuration

```python
# logging_config.py
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
        'json': {
            'format': '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
        }
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'json',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'INFO',
            'formatter': 'json',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/app/logs/app.log',
            'maxBytes': 10485760,
            'backupCount': 5,
        }
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': False
        }
    }
}
```

## Security Configuration

### 1. SSL/TLS Configuration

```bash
# Generate SSL certificates (Let's Encrypt)
certbot certonly \
  --nginx \
  --email admin@example.com \
  --agree-tos \
  --domains localization.example.com

# Auto-renewal
echo "0 12 * * * /usr/bin/certbot renew --quiet" | crontab -
```

### 2. Network Security

```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: localization-network-policy
  namespace: localization
spec:
  podSelector:
    matchLabels:
      app: localization-api
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
    - podSelector:
        matchLabels:
          app: localization-db
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: localization-redis
    ports:
    - protocol: TCP
      port: 6379
```

### 3. Pod Security Policy

```yaml
# pod-security-policy.yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: localization-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
```

## Performance Optimization

### 1. Database Optimization

```sql
-- Database indexes for performance
CREATE INDEX CONCURRENTLY idx_translations_key_lang ON ml_translations(translation_key_id, language_id);
CREATE INDEX CONCURRENTLY idx_translations_status ON ml_translations(status) WHERE status = 'published';
CREATE INDEX CONCURRENTLY idx_translation_keys_namespace ON ml_translation_keys(namespace_id);
CREATE INDEX CONCURRENTLY idx_languages_code ON ml_languages(code) WHERE status = 'active';

-- Optimize PostgreSQL settings
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
SELECT pg_reload_conf();
```

### 2. Redis Optimization

```bash
# Redis configuration for production
echo "maxmemory 512mb" >> /etc/redis/redis.conf
echo "maxmemory-policy allkeys-lru" >> /etc/redis/redis.conf
echo "save 900 1" >> /etc/redis/redis.conf
echo "save 300 10" >> /etc/redis/redis.conf
echo "save 60 10000" >> /etc/redis/redis.conf
echo "tcp-keepalive 300" >> /etc/redis/redis.conf
echo "timeout 300" >> /etc/redis/redis.conf
```

### 3. Application Optimization

```python
# Connection pooling
DATABASE_SETTINGS = {
    "pool_size": 20,
    "max_overflow": 30,
    "pool_timeout": 30,
    "pool_recycle": 3600,
    "pool_pre_ping": True
}

# Redis connection pool
REDIS_SETTINGS = {
    "connection_pool_kwargs": {
        "max_connections": 50,
        "socket_timeout": 5,
        "socket_connect_timeout": 5,
        "retry_on_timeout": True
    }
}

# FastAPI optimizations
app = FastAPI(
    title="Localization API",
    version="1.0.0",
    docs_url="/docs" if DEBUG else None,
    redoc_url="/redoc" if DEBUG else None
)

# Add middleware for compression
app.add_middleware(GZipMiddleware, minimum_size=1000)
```

## Backup & Recovery

### 1. Database Backup

```bash
#!/bin/bash
# backup-db.sh

BACKUP_DIR="/backups/localization"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="localization_backup_${DATE}.sql"

# Create backup directory
mkdir -p $BACKUP_DIR

# Perform backup
pg_dump $DATABASE_URL > "$BACKUP_DIR/$BACKUP_FILE"

# Compress backup
gzip "$BACKUP_DIR/$BACKUP_FILE"

# Upload to S3 (optional)
aws s3 cp "$BACKUP_DIR/$BACKUP_FILE.gz" s3://your-backup-bucket/localization/

# Keep only last 7 days of backups
find $BACKUP_DIR -name "*.gz" -mtime +7 -delete

echo "Backup completed: $BACKUP_FILE.gz"
```

### 2. Automated Backup CronJob

```yaml
# backup-cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: localization-backup
  namespace: localization
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: postgres:14
            env:
            - name: PGPASSWORD
              valueFrom:
                secretKeyRef:
                  name: localization-secrets
                  key: db-password
            command:
            - /bin/bash
            - -c
            - |
              pg_dump -h localization-db -U localization_user localization > /backup/localization_$(date +%Y%m%d_%H%M%S).sql
              gzip /backup/localization_*.sql
            volumeMounts:
            - name: backup-storage
              mountPath: /backup
          volumes:
          - name: backup-storage
            persistentVolumeClaim:
              claimName: backup-pvc
          restartPolicy: OnFailure
```

### 3. Disaster Recovery Plan

```bash
#!/bin/bash
# disaster-recovery.sh

# 1. Restore database from backup
LATEST_BACKUP=$(ls -t /backups/localization/*.gz | head -1)
gunzip $LATEST_BACKUP
psql $DATABASE_URL < ${LATEST_BACKUP%.gz}

# 2. Clear Redis cache
redis-cli FLUSHALL

# 3. Restart application
kubectl rollout restart deployment/localization-api -n localization

# 4. Run health checks
kubectl wait --for=condition=available --timeout=300s deployment/localization-api -n localization

# 5. Verify functionality
curl -f https://localization.example.com/health
```

## Troubleshooting

### Common Issues

**1. Database Connection Issues**
```bash
# Check database connectivity
kubectl exec -it deployment/localization-api -n localization -- \
  psql $DATABASE_URL -c "SELECT 1;"

# Check connection pool status
kubectl logs deployment/localization-api -n localization | grep "pool"
```

**2. Redis Connection Issues**
```bash
# Test Redis connectivity
kubectl exec -it deployment/localization-api -n localization -- \
  redis-cli -u $REDIS_URL ping

# Check Redis memory usage
kubectl exec -it deployment/localization-redis -n localization -- \
  redis-cli info memory
```

**3. High Memory Usage**
```bash
# Check pod memory usage
kubectl top pods -n localization

# Get detailed resource usage
kubectl describe pod <pod-name> -n localization

# Scale up if needed
kubectl scale deployment localization-api --replicas=5 -n localization
```

**4. Slow API Responses**
```bash
# Check database queries
kubectl exec -it deployment/localization-db -n localization -- \
  psql -U localization_user -d localization -c \
  "SELECT query, calls, total_time FROM pg_stat_statements ORDER BY total_time DESC LIMIT 10;"

# Check cache hit rate
kubectl exec -it deployment/localization-redis -n localization -- \
  redis-cli info stats | grep keyspace
```

### Debugging Commands

```bash
# View application logs
kubectl logs -f deployment/localization-api -n localization

# Get pod details
kubectl describe pod <pod-name> -n localization

# Execute shell in pod
kubectl exec -it <pod-name> -n localization -- /bin/bash

# Port forward for local testing
kubectl port-forward svc/localization-api 8000:8000 -n localization

# Check events
kubectl get events -n localization --sort-by=.metadata.creationTimestamp

# Resource utilization
kubectl top nodes
kubectl top pods -n localization --sort-by=memory
```

### Performance Monitoring

```bash
# Monitor API response times
curl -w "@curl-format.txt" -o /dev/null -s "https://localization.example.com/api/v1/translate?key=test&language=en"

# Database performance
kubectl exec -it deployment/localization-db -n localization -- \
  psql -U localization_user -d localization -c \
  "SELECT * FROM pg_stat_activity WHERE state = 'active';"

# Check for deadlocks
kubectl exec -it deployment/localization-db -n localization -- \
  psql -U localization_user -d localization -c \
  "SELECT * FROM pg_stat_database_conflicts;"
```

---

**Company**: Datacraft  
**Website**: www.datacraft.co.ke  
**Contact**: nyimbi@gmail.com