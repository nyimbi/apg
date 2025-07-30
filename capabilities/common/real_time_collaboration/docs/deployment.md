# APG Real-Time Collaboration - Deployment Guide

Comprehensive deployment guide for production environments with high availability, scalability, and security.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Infrastructure Requirements](#infrastructure-requirements)
3. [Database Setup](#database-setup)
4. [Application Deployment](#application-deployment)
5. [WebSocket Configuration](#websocket-configuration)
6. [Third-Party Integration Setup](#third-party-integration-setup)
7. [Security Configuration](#security-configuration)
8. [Monitoring & Logging](#monitoring--logging)
9. [Performance Optimization](#performance-optimization)
10. [Disaster Recovery](#disaster-recovery)

## 🔧 Prerequisites

### System Requirements

#### Minimum Requirements (Development)
- **CPU**: 2 cores, 2.5GHz
- **RAM**: 8GB
- **Storage**: 50GB SSD
- **Network**: 100Mbps
- **OS**: Ubuntu 20.04+, CentOS 8+, or RHEL 8+

#### Recommended Requirements (Production)
- **CPU**: 8 cores, 3.0GHz
- **RAM**: 32GB
- **Storage**: 500GB NVMe SSD
- **Network**: 1Gbps with redundancy
- **OS**: Ubuntu 22.04 LTS or RHEL 9

#### High Availability Requirements
- **Load Balancer**: 2x instances (active/passive)
- **Application Servers**: 3+ instances (active/active)
- **Database**: PostgreSQL cluster with streaming replication
- **Cache**: Redis cluster with failover
- **Storage**: Distributed storage with redundancy

### Software Dependencies

```bash
# Python 3.12+
python3 --version

# PostgreSQL 15+
psql --version

# Redis 7+
redis-server --version

# Node.js 18+ (for frontend builds)
node --version

# Docker & Docker Compose (optional)
docker --version
docker-compose --version
```

## 🏗️ Infrastructure Requirements

### Network Architecture

```
Internet
    |
[Load Balancer] (HAProxy/Nginx)
    |
[Application Tier] (3+ instances)
    |
[Database Tier] (PostgreSQL Cluster)
    |
[Cache Tier] (Redis Cluster)
    |
[Storage Tier] (Distributed Storage)
```

### Port Requirements

| Service | Port | Protocol | Purpose |
|---------|------|----------|---------|
| Application | 8000 | HTTP/HTTPS | Main API |
| WebSocket | 8765 | WS/WSS | Real-time communication |
| PostgreSQL | 5432 | TCP | Database |
| Redis | 6379 | TCP | Cache/Sessions |
| Monitoring | 9090 | HTTP | Prometheus |
| Health Check | 8080 | HTTP | Load balancer health |

### Security Groups/Firewall Rules

```bash
# Application tier
iptables -A INPUT -p tcp --dport 8000 -s 10.0.0.0/8 -j ACCEPT
iptables -A INPUT -p tcp --dport 8765 -s 10.0.0.0/8 -j ACCEPT

# Database tier
iptables -A INPUT -p tcp --dport 5432 -s 10.0.1.0/24 -j ACCEPT

# Cache tier
iptables -A INPUT -p tcp --dport 6379 -s 10.0.1.0/24 -j ACCEPT
```

## 🗄️ Database Setup

### PostgreSQL Cluster Setup

#### Primary Database Configuration

```sql
-- Create database and user
CREATE DATABASE apg_rtc;
CREATE USER rtc_user WITH ENCRYPTED PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE apg_rtc TO rtc_user;

-- Enable required extensions
\c apg_rtc;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
```

#### postgresql.conf (Primary)

```ini
# Connection settings
listen_addresses = '*'
port = 5432
max_connections = 200

# Memory settings
shared_buffers = 8GB
effective_cache_size = 24GB
work_mem = 256MB
maintenance_work_mem = 2GB

# Replication settings
wal_level = replica
max_wal_senders = 3
max_replication_slots = 3
wal_keep_size = 1GB

# Performance settings
checkpoint_completion_target = 0.9
wal_buffers = 64MB
random_page_cost = 1.1
effective_io_concurrency = 200

# Logging
log_destination = 'stderr'
logging_collector = on
log_directory = '/var/log/postgresql'
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
log_min_duration_statement = 1000
log_checkpoints = on
log_connections = on
log_disconnections = on
```

#### pg_hba.conf

```
# TYPE  DATABASE        USER            ADDRESS                 METHOD
local   all             postgres                                peer
local   all             all                                     md5
host    all             rtc_user        10.0.0.0/8              md5
host    replication     replicator      10.0.2.0/24             md5
```

#### Streaming Replication Setup

```bash
# On primary server
sudo -u postgres createuser --replication -P replicator

# On replica servers
sudo -u postgres pg_basebackup -h primary-server -D /var/lib/postgresql/data -U replicator -P -v -R -W
```

### Database Schema Migration

```bash
# Install Python dependencies
pip install alembic psycopg2-binary

# Initialize Alembic
alembic init alembic

# Generate initial migration
alembic revision --autogenerate -m "Initial RTC schema"

# Apply migrations
alembic upgrade head
```

### Database Performance Tuning

```sql
-- Create indexes for performance
CREATE INDEX CONCURRENTLY idx_rtc_sessions_tenant_active 
ON rtc_sessions(tenant_id, is_active) WHERE is_active = true;

CREATE INDEX CONCURRENTLY idx_rtc_participants_session_user 
ON rtc_participants(session_id, user_id);

CREATE INDEX CONCURRENTLY idx_rtc_messages_page_timestamp 
ON rtc_messages(page_url, created_at DESC);

CREATE INDEX CONCURRENTLY idx_rtc_page_collaboration_url_tenant 
ON rtc_page_collaboration(page_url, tenant_id) WHERE is_active = true;

-- Partitioning for large tables
CREATE TABLE rtc_activities_2024 PARTITION OF rtc_activities 
FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

-- Vacuum and analyze
VACUUM ANALYZE;
```

## 🚀 Application Deployment

### Docker Deployment

#### Dockerfile

```dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    postgresql-client \
    redis-tools \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 rtcuser && chown -R rtcuser:rtcuser /app
USER rtcuser

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/rtc/health || exit 1

# Expose ports
EXPOSE 8000 8765

# Start application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "app:app"]
```

#### docker-compose.yml

```yaml
version: '3.8'

services:
  rtc-app:
    build: .
    ports:
      - "8000:8000"
      - "8765:8765"
    environment:
      - DATABASE_URL=postgresql://rtc_user:secure_password@db:5432/apg_rtc
      - REDIS_URL=redis://redis:6379/0
      - SECRET_KEY=${SECRET_KEY}
      - APG_TENANT_ID=${APG_TENANT_ID}
    depends_on:
      - db
      - redis
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=apg_rtc
      - POSTGRES_USER=rtc_user
      - POSTGRES_PASSWORD=secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./postgresql.conf:/etc/postgresql/postgresql.conf
    ports:
      - "5432:5432"
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    command: redis-server --appendonly yes --maxmemory 2gb --maxmemory-policy allkeys-lru

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - rtc-app
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

### Kubernetes Deployment

#### Namespace

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: apg-rtc
```

#### ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: rtc-config
  namespace: apg-rtc
data:
  DATABASE_URL: "postgresql://rtc_user:secure_password@postgres-service:5432/apg_rtc"
  REDIS_URL: "redis://redis-service:6379/0"
  LOG_LEVEL: "INFO"
  WEBSOCKET_PORT: "8765"
```

#### Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rtc-app
  namespace: apg-rtc
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rtc-app
  template:
    metadata:
      labels:
        app: rtc-app
    spec:
      containers:
      - name: rtc-app
        image: apg/rtc:latest
        ports:
        - containerPort: 8000
        - containerPort: 8765
        envFrom:
        - configMapRef:
            name: rtc-config
        - secretRef:
            name: rtc-secrets
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /api/v1/rtc/health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /api/v1/rtc/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

#### Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: rtc-service
  namespace: apg-rtc
spec:
  selector:
    app: rtc-app
  ports:
  - name: http
    port: 80
    targetPort: 8000
  - name: websocket
    port: 8765
    targetPort: 8765
  type: ClusterIP
```

#### Ingress

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rtc-ingress
  namespace: apg-rtc
  annotations:
    nginx.ingress.kubernetes.io/websocket-services: "rtc-service"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "3600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "3600"
spec:
  tls:
  - hosts:
    - rtc.apg.com
    secretName: rtc-tls
  rules:
  - host: rtc.apg.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: rtc-service
            port:
              number: 80
      - path: /ws
        pathType: Prefix
        backend:
          service:
            name: rtc-service
            port:
              number: 8765
```

## 🔌 WebSocket Configuration

### Load Balancer Configuration (Nginx)

```nginx
upstream rtc_websocket {
    least_conn;
    server app1.internal:8765;
    server app2.internal:8765;
    server app3.internal:8765;
}

upstream rtc_app {
    least_conn;
    server app1.internal:8000;
    server app2.internal:8000;
    server app3.internal:8000;
}

server {
    listen 443 ssl http2;
    server_name rtc.apg.com;

    ssl_certificate /etc/ssl/certs/rtc.apg.com.crt;
    ssl_certificate_key /etc/ssl/private/rtc.apg.com.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;

    # WebSocket location
    location /ws/ {
        proxy_pass http://rtc_websocket;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
        proxy_connect_timeout 60s;
    }

    # API location
    location / {
        proxy_pass http://rtc_app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        proxy_connect_timeout 60s;
    }
}
```

### Redis Configuration for WebSocket Scaling

```redis
# redis.conf
bind 0.0.0.0
port 6379
protected-mode yes
requirepass secure_redis_password

# Memory optimization
maxmemory 8gb
maxmemory-policy allkeys-lru

# Persistence
save 900 1
save 300 10
save 60 10000

# Pub/Sub for WebSocket messaging
client-output-buffer-limit pubsub 32mb 8mb 60

# Cluster configuration (if using Redis Cluster)
cluster-enabled yes
cluster-config-file nodes.conf
cluster-node-timeout 5000
```

## 🔗 Third-Party Integration Setup

### Microsoft Teams Integration

#### Azure App Registration

```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login to Azure
az login

# Create app registration
az ad app create \
  --display-name "APG Real-Time Collaboration" \
  --sign-in-audience "AzureADMultipleOrgs" \
  --web-redirect-uris "https://rtc.apg.com/auth/teams/callback"

# Get application ID and create client secret
az ad app credential reset --id <app-id> --append
```

#### Required Permissions

```json
{
  "requiredResourceAccess": [
    {
      "resourceAppId": "00000003-0000-0000-c000-000000000000",
      "resourceAccess": [
        {
          "id": "e1fe6dd8-ba31-4d61-89e7-88639da4683d",
          "type": "Scope"
        },
        {
          "id": "ba47897c-39ec-4d83-8086-ee8256fa737d",
          "type": "Role"
        }
      ]
    }
  ]
}
```

### Zoom Integration

#### Zoom App Creation

```bash
# Environment variables
export ZOOM_API_KEY="your_zoom_api_key"
export ZOOM_API_SECRET="your_zoom_api_secret"
export ZOOM_WEBHOOK_SECRET="your_webhook_secret"

# Test Zoom API connection
curl -X POST "https://api.zoom.us/v2/users/me/meetings" \
  -H "Authorization: Bearer $ZOOM_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Test Meeting",
    "type": 1,
    "duration": 30
  }'
```

### Google Meet Integration

#### Google Cloud Setup

```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash

# Authenticate
gcloud auth login

# Enable APIs
gcloud services enable calendar-json.googleapis.com
gcloud services enable drive.googleapis.com

# Create service account
gcloud iam service-accounts create apg-rtc \
  --display-name="APG RTC Service Account"

# Create and download key
gcloud iam service-accounts keys create credentials.json \
  --iam-account=apg-rtc@your-project.iam.gserviceaccount.com
```

## 🔒 Security Configuration

### Environment Variables

```bash
# .env.production
SECRET_KEY=super_secure_random_string_64_chars_minimum
DATABASE_URL=postgresql://rtc_user:secure_password@db-cluster:5432/apg_rtc
REDIS_URL=redis://:redis_password@redis-cluster:6379/0

# APG Integration
APG_AUTH_SERVICE_URL=https://auth.apg.com
APG_AI_SERVICE_URL=https://ai.apg.com
APG_NOTIFICATION_SERVICE_URL=https://notifications.apg.com

# Third-party Integration
TEAMS_CLIENT_ID=your_teams_client_id
TEAMS_CLIENT_SECRET=your_teams_client_secret
ZOOM_API_KEY=your_zoom_api_key
ZOOM_API_SECRET=your_zoom_api_secret
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret

# Security
CORS_ORIGINS=https://app.apg.com,https://admin.apg.com
ALLOWED_HOSTS=rtc.apg.com,app.apg.com
SSL_REQUIRED=true
HSTS_MAX_AGE=31536000
```

### SSL/TLS Configuration

```bash
# Generate CSR
openssl req -new -newkey rsa:4096 -nodes -keyout rtc.apg.com.key -out rtc.apg.com.csr

# After receiving certificate from CA
# Configure SSL in Nginx/Load Balancer
# Enable HSTS, CSRF protection, and secure headers
```

### Security Headers

```nginx
# Security headers
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
add_header X-Content-Type-Options nosniff always;
add_header X-Frame-Options DENY always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'" always;
```

## 📊 Monitoring & Logging

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'rtc-app'
    static_configs:
      - targets: ['app1:8000', 'app2:8000', 'app3:8000']
    metrics_path: /metrics
    scrape_interval: 30s

  - job_name: 'postgresql'
    static_configs:
      - targets: ['db1:9187', 'db2:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis1:9121', 'redis2:9121']
```

### Grafana Dashboards

```json
{
  "dashboard": {
    "title": "APG RTC Monitoring",
    "panels": [
      {
        "title": "Active WebSocket Connections",
        "type": "graph",
        "targets": [
          {
            "expr": "rtc_websocket_connections_total"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(rtc_request_duration_seconds_bucket[5m]))"
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
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'format': '{levelname} {asctime} {module} {message}',
            'style': '{',
        },
        'json': {
            'format': '{"level": "%(levelname)s", "timestamp": "%(asctime)s", "module": "%(module)s", "message": "%(message)s"}',
        }
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/var/log/rtc/app.log',
            'maxBytes': 50*1024*1024,  # 50MB
            'backupCount': 10,
            'formatter': 'json',
        },
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'detailed',
        }
    },
    'loggers': {
        'rtc': {
            'handlers': ['file', 'console'],
            'level': 'INFO',
            'propagate': False,
        }
    }
}
```

## ⚡ Performance Optimization

### Application Performance

```python
# performance_config.py
import asyncio
import uvloop

# Use uvloop for better async performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Connection pooling
DATABASE_POOL_SIZE = 20
DATABASE_MAX_OVERFLOW = 30
DATABASE_POOL_TIMEOUT = 30

# Redis connection pooling
REDIS_POOL_SIZE = 50
REDIS_POOL_TIMEOUT = 5

# WebSocket settings
WEBSOCKET_MAX_CONNECTIONS = 10000
WEBSOCKET_PING_INTERVAL = 30
WEBSOCKET_PING_TIMEOUT = 10
```

### Database Optimization

```sql
-- Optimize PostgreSQL settings
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
ALTER SYSTEM SET track_activity_query_size = 2048;
ALTER SYSTEM SET pg_stat_statements.track = 'all';

-- Create materialized views for analytics
CREATE MATERIALIZED VIEW collaboration_stats AS
SELECT 
    date_trunc('hour', created_at) as hour,
    COUNT(*) as session_count,
    AVG(duration_minutes) as avg_duration
FROM rtc_sessions 
WHERE created_at >= NOW() - INTERVAL '7 days'
GROUP BY date_trunc('hour', created_at);

-- Refresh materialized view hourly
CREATE OR REPLACE FUNCTION refresh_collaboration_stats()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY collaboration_stats;
END;
$$ LANGUAGE plpgsql;

SELECT cron.schedule('refresh-stats', '0 * * * *', 'SELECT refresh_collaboration_stats();');
```

### Caching Strategy

```python
# caching.py
import redis
from functools import wraps

redis_client = redis.Redis(host='redis-cluster', port=6379, db=0)

def cache_result(expiry=300):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # Call function and cache result
            result = await func(*args, **kwargs)
            redis_client.setex(cache_key, expiry, json.dumps(result))
            return result
        return wrapper
    return decorator

# Usage
@cache_result(expiry=60)
async def get_page_presence(page_url: str):
    # Expensive database query
    pass
```

## 🔄 Disaster Recovery

### Backup Strategy

```bash
#!/bin/bash
# backup.sh

# Database backup
pg_dump -h $DB_HOST -U $DB_USER -d apg_rtc \
  --format=custom \
  --compress=9 \
  --file="/backups/rtc_$(date +%Y%m%d_%H%M%S).dump"

# Redis backup
redis-cli --rdb /backups/redis_$(date +%Y%m%d_%H%M%S).rdb

# Upload to S3
aws s3 sync /backups/ s3://apg-rtc-backups/

# Cleanup old backups (keep 30 days)
find /backups/ -name "*.dump" -mtime +30 -delete
find /backups/ -name "*.rdb" -mtime +30 -delete
```

### Restore Procedures

```bash
#!/bin/bash
# restore.sh

# Stop application
kubectl scale deployment rtc-app --replicas=0

# Restore database
pg_restore -h $DB_HOST -U $DB_USER -d apg_rtc_restored \
  --clean --if-exists \
  /backups/rtc_latest.dump

# Restore Redis
redis-cli --rdb /backups/redis_latest.rdb

# Start application
kubectl scale deployment rtc-app --replicas=3
```

### High Availability Setup

```yaml
# ha-setup.yml
apiVersion: v1
kind: Service
metadata:
  name: rtc-service-ha
spec:
  type: LoadBalancer
  selector:
    app: rtc-app
  sessionAffinity: ClientIP
  ports:
  - port: 80
    targetPort: 8000
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rtc-app-ha
spec:
  replicas: 5
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2
      maxUnavailable: 1
  template:
    spec:
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values:
                - rtc-app
            topologyKey: kubernetes.io/hostname
```

## 📝 Post-Deployment Checklist

### Smoke Tests

```bash
#!/bin/bash
# smoke_tests.sh

# Health check
curl -f https://rtc.apg.com/api/v1/rtc/health

# WebSocket connection test
wscat -c wss://rtc.apg.com/ws/test/test

# Database connectivity
psql $DATABASE_URL -c "SELECT COUNT(*) FROM rtc_sessions;"

# Redis connectivity
redis-cli -u $REDIS_URL ping

# Third-party integrations
curl -f https://rtc.apg.com/api/v1/rtc/integrations/teams/test
curl -f https://rtc.apg.com/api/v1/rtc/integrations/zoom/test
curl -f https://rtc.apg.com/api/v1/rtc/integrations/google-meet/test
```

### Performance Validation

```bash
# Load testing with Artillery
artillery run load-test.yml

# WebSocket load testing
artillery run websocket-test.yml

# Database performance check
pgbench -c 10 -j 2 -t 1000 apg_rtc
```

### Security Validation

```bash
# SSL/TLS check
ssllabs-scan rtc.apg.com

# Security headers check
curl -I https://rtc.apg.com

# Vulnerability scan
nmap -sV rtc.apg.com
```

---

## 📞 Support

- **Deployment Issues**: [DevOps team](mailto:devops@apg.com)
- **Performance Questions**: [Performance engineering](mailto:performance@apg.com)
- **Security Concerns**: [Security team](mailto:security@apg.com)
- **Emergency Support**: [24/7 support hotline](tel:+1-800-APG-HELP)

---

**© 2025 Datacraft | Contact: nyimbi@gmail.com | Website: www.datacraft.co.ke**

*Production-ready deployment guide for APG Real-Time Collaboration*