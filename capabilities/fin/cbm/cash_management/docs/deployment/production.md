# APG Cash Management - Production Deployment Guide

**Enterprise Production Deployment for World-Class Cash Management**

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero | APG Platform Architect

---

## 🚀 Production Deployment Overview

This guide provides comprehensive instructions for deploying the APG Cash Management System in production environments. The system is designed for enterprise-scale deployments with high availability, security, and performance requirements.

### 📋 Deployment Checklist

- [ ] Infrastructure provisioning complete
- [ ] Database cluster configured with replication
- [ ] Redis cluster deployed and tested
- [ ] Load balancers configured
- [ ] SSL certificates installed
- [ ] Security configurations applied
- [ ] Monitoring and logging systems active
- [ ] Backup and disaster recovery tested
- [ ] Performance testing completed
- [ ] Security penetration testing passed

---

## 🏗️ Infrastructure Requirements

### Minimum Production Requirements

| Component | Specification | Recommended |
|-----------|--------------|-------------|
| **CPU** | 8 cores | 16+ cores |
| **Memory** | 32 GB RAM | 64+ GB RAM |
| **Storage** | 1 TB SSD | 2+ TB NVMe SSD |
| **Network** | 1 Gbps | 10+ Gbps |
| **OS** | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS |

### High Availability Architecture

```mermaid
graph TB
    subgraph "Load Balancer Tier"
        LB1[HAProxy 1]
        LB2[HAProxy 2]
    end
    
    subgraph "Application Tier"
        APP1[APG App 1<br/>Primary]
        APP2[APG App 2<br/>Secondary]
        APP3[APG App 3<br/>Tertiary]
    end
    
    subgraph "Database Tier"
        DB_PRIMARY[PostgreSQL Primary]
        DB_REPLICA1[PostgreSQL Replica 1]
        DB_REPLICA2[PostgreSQL Replica 2]
    end
    
    subgraph "Cache Tier"
        REDIS_MASTER[Redis Master]
        REDIS_SLAVE1[Redis Slave 1]
        REDIS_SLAVE2[Redis Slave 2]
    end
    
    subgraph "External Services"
        MONITOR[Monitoring<br/>Prometheus/Grafana]
        LOGS[Logging<br/>ELK Stack]
        BACKUP[Backup<br/>Storage]
    end
    
    LB1 --> APP1
    LB1 --> APP2
    LB2 --> APP2
    LB2 --> APP3
    
    APP1 --> DB_PRIMARY
    APP2 --> DB_REPLICA1
    APP3 --> DB_REPLICA2
    
    APP1 --> REDIS_MASTER
    APP2 --> REDIS_SLAVE1
    APP3 --> REDIS_SLAVE2
    
    DB_PRIMARY --> DB_REPLICA1
    DB_PRIMARY --> DB_REPLICA2
    REDIS_MASTER --> REDIS_SLAVE1
    REDIS_MASTER --> REDIS_SLAVE2
    
    APP1 --> MONITOR
    APP2 --> MONITOR
    APP3 --> MONITOR
    
    APP1 --> LOGS
    APP2 --> LOGS
    APP3 --> LOGS
    
    DB_PRIMARY --> BACKUP
```

---

## 🐳 Docker Production Deployment

### 1. Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  # APG Cash Management Application
  apg-cash-management:
    image: datacraft/apg-cash-management:1.0.0
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
        window: 120s
      update_config:
        parallelism: 1
        delay: 10s
        failure_action: rollback
        order: start-first
    environment:
      - APG_ENVIRONMENT=production
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - SECRET_KEY=${SECRET_KEY}
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
      - ENCRYPTION_KEY=${ENCRYPTION_KEY}
      - NEW_RELIC_LICENSE_KEY=${NEW_RELIC_LICENSE_KEY}
    ports:
      - "8000:8000"
    volumes:
      - /var/log/apg-cash:/app/logs:rw
      - /etc/ssl/certs:/etc/ssl/certs:ro
    networks:
      - apg-network
    depends_on:
      - postgres-primary
      - redis-master
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  # PostgreSQL Primary Database
  postgres-primary:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=${POSTGRES_DB}
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_REPLICATION_USER=${POSTGRES_REPLICATION_USER}
      - POSTGRES_REPLICATION_PASSWORD=${POSTGRES_REPLICATION_PASSWORD}
    volumes:
      - postgres-primary-data:/var/lib/postgresql/data
      - ./config/postgres/postgresql.conf:/etc/postgresql/postgresql.conf:ro
      - ./config/postgres/pg_hba.conf:/etc/postgresql/pg_hba.conf:ro
      - ./scripts/postgres/setup-replication.sh:/docker-entrypoint-initdb.d/setup-replication.sh:ro
    ports:
      - "5432:5432"
    networks:
      - apg-network
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 4G
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
      interval: 30s
      timeout: 10s
      retries: 5

  # PostgreSQL Read Replica
  postgres-replica:
    image: postgres:15-alpine
    environment:
      - PGUSER=${POSTGRES_REPLICATION_USER}
      - POSTGRES_PASSWORD=${POSTGRES_REPLICATION_PASSWORD}
      - POSTGRES_MASTER_SERVICE=postgres-primary
    volumes:
      - postgres-replica-data:/var/lib/postgresql/data
      - ./scripts/postgres/setup-replica.sh:/docker-entrypoint-initdb.d/setup-replica.sh:ro
    ports:
      - "5433:5432"
    networks:
      - apg-network
    depends_on:
      - postgres-primary
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G

  # Redis Master
  redis-master:
    image: redis:7-alpine
    command: redis-server /etc/redis/redis.conf
    volumes:
      - redis-master-data:/data
      - ./config/redis/redis-master.conf:/etc/redis/redis.conf:ro
    ports:
      - "6379:6379"
    networks:
      - apg-network
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Redis Slave
  redis-slave:
    image: redis:7-alpine
    command: redis-server /etc/redis/redis.conf
    volumes:
      - redis-slave-data:/data
      - ./config/redis/redis-slave.conf:/etc/redis/redis.conf:ro
    ports:
      - "6380:6379"
    networks:
      - apg-network
    depends_on:
      - redis-master
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G

  # HAProxy Load Balancer
  haproxy:
    image: haproxy:2.8-alpine
    volumes:
      - ./config/haproxy/haproxy.cfg:/usr/local/etc/haproxy/haproxy.cfg:ro
      - /etc/ssl/certs:/etc/ssl/certs:ro
    ports:
      - "80:80"
      - "443:443"
      - "8404:8404"  # Stats interface
    networks:
      - apg-network
    depends_on:
      - apg-cash-management
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 256M

  # Prometheus Monitoring
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./config/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    ports:
      - "9090:9090"
    networks:
      - apg-network
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
      - '--storage.tsdb.retention.time=90d'
      - '--web.enable-lifecycle'

  # Grafana Dashboards
  grafana:
    image: grafana/grafana:latest
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD}
      - GF_INSTALL_PLUGINS=grafana-clock-panel,grafana-simple-json-datasource
    volumes:
      - grafana-data:/var/lib/grafana
      - ./config/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./config/grafana/datasources:/etc/grafana/provisioning/datasources:ro
    ports:
      - "3000:3000"
    networks:
      - apg-network
    depends_on:
      - prometheus

volumes:
  postgres-primary-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /data/postgres-primary
  postgres-replica-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /data/postgres-replica
  redis-master-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /data/redis-master
  redis-slave-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /data/redis-slave
  prometheus-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /data/prometheus
  grafana-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /data/grafana

networks:
  apg-network:
    driver: overlay
    attachable: true
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### 2. Production Environment Variables

```bash
# .env.production
# APG Cash Management Production Configuration

# Environment
APG_ENVIRONMENT=production
APG_LOG_LEVEL=INFO
APG_DEBUG=false

# Database Configuration
DATABASE_URL=postgresql://apg_user:${POSTGRES_PASSWORD}@postgres-primary:5432/apg_cash_production
DATABASE_POOL_SIZE=50
DATABASE_MAX_OVERFLOW=100
DATABASE_POOL_TIMEOUT=30
DATABASE_POOL_RECYCLE=3600

# Database Replication
DATABASE_READ_URL=postgresql://apg_user:${POSTGRES_PASSWORD}@postgres-replica:5432/apg_cash_production

# Redis Configuration
REDIS_URL=redis://redis-master:6379/0
REDIS_SENTINEL_URLS=redis://redis-sentinel-1:26379,redis://redis-sentinel-2:26379,redis://redis-sentinel-3:26379
REDIS_MASTER_NAME=apg-redis-master
REDIS_POOL_SIZE=100
REDIS_SOCKET_TIMEOUT=30
REDIS_SOCKET_CONNECT_TIMEOUT=30

# Security Configuration
SECRET_KEY=${SECRET_KEY}
JWT_SECRET_KEY=${JWT_SECRET_KEY}
JWT_ALGORITHM=HS256
JWT_EXPIRATION_DELTA=3600
ENCRYPTION_KEY=${ENCRYPTION_KEY}

# SSL/TLS Configuration
SSL_CERT_PATH=/etc/ssl/certs/apg-cash-management.crt
SSL_KEY_PATH=/etc/ssl/private/apg-cash-management.key
SSL_CA_PATH=/etc/ssl/certs/ca-certificates.crt

# Bank Integration
BANK_API_TIMEOUT=30
BANK_SYNC_FREQUENCY=300  # 5 minutes
BANK_RETRY_ATTEMPTS=3
BANK_RATE_LIMIT_REQUESTS=1000
BANK_RATE_LIMIT_WINDOW=3600

# AI/ML Configuration
ML_MODEL_CACHE_SIZE=2000
ML_TRAINING_SCHEDULE="0 2 * * 0"  # Weekly at 2 AM Sunday
ML_PREDICTION_BATCH_SIZE=5000
ML_MODEL_STORAGE_PATH=/data/ml-models
ML_FEATURE_STORE_URL=${ML_FEATURE_STORE_URL}

# Risk Analytics
RISK_VAR_CONFIDENCE_LEVEL=0.95
RISK_STRESS_TEST_FREQUENCY=daily
RISK_ALERT_THRESHOLD=0.05
RISK_MONTE_CARLO_SIMULATIONS=100000

# Performance Configuration
UVICORN_WORKERS=8
UVICORN_MAX_REQUESTS=10000
UVICORN_KEEPALIVE=300
UVICORN_TIMEOUT_KEEP_ALIVE=5

# Monitoring & Observability
NEW_RELIC_LICENSE_KEY=${NEW_RELIC_LICENSE_KEY}
NEW_RELIC_APP_NAME=APG-Cash-Management-Production
PROMETHEUS_METRICS_PORT=8001
GRAFANA_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD}

# Logging Configuration
LOG_FORMAT=json
LOG_DESTINATION=file
LOG_FILE_PATH=/app/logs/apg-cash-management.log
LOG_MAX_SIZE=100MB
LOG_BACKUP_COUNT=10
LOG_ROTATION=daily

# Backup Configuration
BACKUP_SCHEDULE="0 1 * * *"  # Daily at 1 AM
BACKUP_STORAGE_URL=${BACKUP_STORAGE_URL}
BACKUP_RETENTION_DAYS=90
BACKUP_ENCRYPTION_KEY=${BACKUP_ENCRYPTION_KEY}

# Email Configuration
SMTP_HOST=${SMTP_HOST}
SMTP_PORT=587
SMTP_USERNAME=${SMTP_USERNAME}
SMTP_PASSWORD=${SMTP_PASSWORD}
SMTP_USE_TLS=true
EMAIL_FROM=noreply@datacraft.co.ke

# Webhook Configuration
WEBHOOK_RETRY_ATTEMPTS=3
WEBHOOK_TIMEOUT=30
WEBHOOK_SIGNATURE_SECRET=${WEBHOOK_SIGNATURE_SECRET}

# Rate Limiting
RATE_LIMIT_STORAGE_URL=${REDIS_URL}
RATE_LIMIT_STRATEGY=sliding_window
RATE_LIMIT_DEFAULT_LIMIT=1000
RATE_LIMIT_DEFAULT_WINDOW=3600

# Feature Flags
FEATURE_ADVANCED_ML_MODELS=true
FEATURE_REAL_TIME_OPTIMIZATION=true
FEATURE_ADVANCED_RISK_ANALYTICS=true
FEATURE_BLOCKCHAIN_INTEGRATION=false
FEATURE_MOBILE_API=true

# External Service URLs
MARKET_DATA_API_URL=${MARKET_DATA_API_URL}
REGULATORY_API_URL=${REGULATORY_API_URL}
PAYMENT_GATEWAY_URL=${PAYMENT_GATEWAY_URL}
NOTIFICATION_SERVICE_URL=${NOTIFICATION_SERVICE_URL}

# Compliance Configuration
AUDIT_LOG_RETENTION_YEARS=7
DATA_RETENTION_POLICY_DAYS=2555  # 7 years
PII_ENCRYPTION_REQUIRED=true
GDPR_COMPLIANCE_ENABLED=true
SOX_COMPLIANCE_ENABLED=true

# Container Configuration
CONTAINER_MEMORY_LIMIT=4G
CONTAINER_CPU_LIMIT=2.0
CONTAINER_RESTART_POLICY=on-failure
CONTAINER_HEALTH_CHECK_INTERVAL=30s
```

---

## 🔧 Configuration Files

### 1. HAProxy Configuration

```bash
# config/haproxy/haproxy.cfg
global
    daemon
    maxconn 4096
    log stdout local0
    
    # SSL Configuration
    ssl-default-bind-ciphers ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384
    ssl-default-bind-options ssl-min-ver TLSv1.2 no-tls-tickets
    
    # Performance tuning
    tune.ssl.default-dh-param 2048
    tune.bufsize 32768
    tune.maxrewrite 1024

defaults
    mode http
    timeout connect 10s
    timeout client 30s
    timeout server 30s
    timeout http-request 10s
    timeout http-keep-alive 2s
    timeout check 5s
    
    option httplog
    option dontlognull
    option http-server-close
    option forwardfor except 127.0.0.0/8
    option redispatch
    
    retries 3
    balance roundrobin

# Frontend configuration
frontend apg_frontend
    bind *:80
    bind *:443 ssl crt /etc/ssl/certs/apg-cash-management.pem
    
    # Redirect HTTP to HTTPS
    redirect scheme https if !{ ssl_fc }
    
    # Security headers
    http-response set-header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload"
    http-response set-header X-Frame-Options "SAMEORIGIN"
    http-response set-header X-Content-Type-Options "nosniff"
    http-response set-header X-XSS-Protection "1; mode=block"
    http-response set-header Referrer-Policy "strict-origin-when-cross-origin"
    
    # Rate limiting
    stick-table type ip size 100k expire 30s store http_req_rate(10s)
    http-request track-sc0 src
    http-request reject if { sc_http_req_rate(0) gt 20 }
    
    # Health check endpoint
    acl is_health_check path_beg /health
    http-request use-service prometheus-exporter if is_health_check
    
    # API routing
    acl is_api path_beg /api/
    use_backend apg_api_backend if is_api
    
    # Web interface routing
    default_backend apg_web_backend

# API Backend
backend apg_api_backend
    balance roundrobin
    option httpchk GET /health
    
    server apg-api-1 apg-cash-management-1:8000 check inter 5s rise 2 fall 3
    server apg-api-2 apg-cash-management-2:8000 check inter 5s rise 2 fall 3
    server apg-api-3 apg-cash-management-3:8000 check inter 5s rise 2 fall 3

# Web Backend
backend apg_web_backend
    balance roundrobin
    option httpchk GET /health
    
    server apg-web-1 apg-cash-management-1:8000 check inter 5s rise 2 fall 3
    server apg-web-2 apg-cash-management-2:8000 check inter 5s rise 2 fall 3
    server apg-web-3 apg-cash-management-3:8000 check inter 5s rise 2 fall 3

# Statistics interface
listen stats
    bind *:8404
    stats enable
    stats uri /stats
    stats refresh 30s
    stats admin if TRUE
    stats auth admin:${HAPROXY_STATS_PASSWORD}
```

### 2. PostgreSQL Configuration

```bash
# config/postgres/postgresql.conf
# PostgreSQL Configuration for APG Cash Management

# Connection Settings
listen_addresses = '*'
port = 5432
max_connections = 200
superuser_reserved_connections = 3

# Memory Settings
shared_buffers = 2GB
effective_cache_size = 6GB
work_mem = 64MB
maintenance_work_mem = 512MB
dynamic_shared_memory_type = posix

# Write Ahead Logging (WAL)
wal_level = replica
max_wal_size = 4GB
min_wal_size = 512MB
wal_buffers = 64MB
checkpoint_completion_target = 0.9
checkpoint_timeout = 15min

# Replication
max_wal_senders = 10
max_replication_slots = 10
hot_standby = on
hot_standby_feedback = on

# Query Planner
random_page_cost = 1.1
effective_io_concurrency = 200
default_statistics_target = 100

# Logging
log_destination = 'csvlog'
logging_collector = on
log_directory = '/var/log/postgresql'
log_filename = 'postgresql-%Y-%m-%d.log'
log_file_mode = 0644
log_rotation_age = 1d
log_rotation_size = 100MB
log_min_duration_statement = 1000ms
log_connections = on
log_disconnections = on
log_lock_waits = on
log_temp_files = 0

# Performance Monitoring
shared_preload_libraries = 'pg_stat_statements'
track_activities = on
track_counts = on
track_io_timing = on
track_functions = all

# Security
ssl = on
ssl_cert_file = '/etc/ssl/certs/server.crt'
ssl_key_file = '/etc/ssl/private/server.key'
password_encryption = scram-sha-256

# Autovacuum
autovacuum = on
autovacuum_max_workers = 6
autovacuum_naptime = 30s
autovacuum_vacuum_threshold = 500
autovacuum_analyze_threshold = 250
```

### 3. Redis Configuration

```bash
# config/redis/redis-master.conf
# Redis Master Configuration for APG Cash Management

# Network
bind 0.0.0.0
port 6379
tcp-backlog 511
timeout 300
tcp-keepalive 300

# General
daemonize no
supervised systemd
pidfile /var/run/redis.pid
loglevel notice
logfile ""

# Snapshotting
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir /data

# Replication
replica-serve-stale-data yes
replica-read-only yes
repl-diskless-sync no
repl-diskless-sync-delay 5
repl-ping-replica-period 10
repl-timeout 60
repl-disable-tcp-nodelay no
repl-backlog-size 16mb
repl-backlog-ttl 3600

# Security
requirepass ${REDIS_PASSWORD}
masterauth ${REDIS_PASSWORD}

# Memory Management
maxmemory 1.5gb
maxmemory-policy allkeys-lru
maxmemory-samples 5

# Lazy Freeing
lazyfree-lazy-eviction yes
lazyfree-lazy-expire yes
lazyfree-lazy-server-del yes
replica-lazy-flush yes

# Threaded I/O
io-threads 4
io-threads-do-reads yes

# Performance
hash-max-ziplist-entries 512
hash-max-ziplist-value 64
list-max-ziplist-size -2
list-compress-depth 0
set-max-intset-entries 512
zset-max-ziplist-entries 128
zset-max-ziplist-value 64
hll-sparse-max-bytes 3000
stream-node-max-bytes 4096
stream-node-max-entries 100

# Active Rehashing
activerehashing yes

# Client Output Buffer Limits
client-output-buffer-limit normal 0 0 0
client-output-buffer-limit replica 256mb 64mb 60
client-output-buffer-limit pubsub 32mb 8mb 60

# Slow Log
slowlog-log-slower-than 10000
slowlog-max-len 128

# Latency Monitoring
latency-monitor-threshold 100

# Event Notification
notify-keyspace-events "Ex"

# Advanced Config
tcp-keepalive 300
databases 16
always-show-logo yes
```

---

## 🚀 Deployment Steps

### 1. Pre-Deployment Preparation

```bash
#!/bin/bash
# scripts/pre-deployment.sh

set -euo pipefail

echo "🚀 APG Cash Management - Pre-Deployment Setup"
echo "==============================================="

# Create directory structure
echo "📁 Creating directory structure..."
sudo mkdir -p /data/{postgres-primary,postgres-replica,redis-master,redis-slave,prometheus,grafana}
sudo mkdir -p /var/log/apg-cash
sudo mkdir -p /etc/ssl/{certs,private}
sudo mkdir -p /opt/apg-cash-management/{config,scripts,backups}

# Set permissions
echo "🔒 Setting permissions..."
sudo chown -R 999:999 /data/postgres-*
sudo chown -R 999:999 /data/redis-*
sudo chown -R 472:472 /data/grafana
sudo chown -R 65534:65534 /data/prometheus
sudo chown -R $(whoami):$(whoami) /var/log/apg-cash
sudo chmod 755 /data/*
sudo chmod 700 /etc/ssl/private

# Install Docker and Docker Compose
echo "🐳 Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $(whoami)
    rm get-docker.sh
fi

if ! command -v docker-compose &> /dev/null; then
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Generate SSL certificates
echo "🔐 Generating SSL certificates..."
if [[ ! -f /etc/ssl/certs/apg-cash-management.crt ]]; then
    sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout /etc/ssl/private/apg-cash-management.key \
        -out /etc/ssl/certs/apg-cash-management.crt \
        -subj "/C=KE/ST=Nairobi/L=Nairobi/O=Datacraft/OU=APG/CN=apg-cash-management.datacraft.co.ke"
    
    # Create PEM file for HAProxy
    sudo cat /etc/ssl/certs/apg-cash-management.crt /etc/ssl/private/apg-cash-management.key > /etc/ssl/certs/apg-cash-management.pem
fi

# Setup firewall
echo "🛡️ Configuring firewall..."
sudo ufw --force enable
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 8404/tcp  # HAProxy stats
sudo ufw allow 9090/tcp  # Prometheus
sudo ufw allow 3000/tcp  # Grafana

# Install monitoring tools
echo "📊 Installing monitoring tools..."
if ! command -v htop &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y htop iotop nethogs
fi

echo "✅ Pre-deployment setup completed successfully!"
echo "📝 Next steps:"
echo "   1. Configure environment variables in .env.production"
echo "   2. Review and customize configuration files"
echo "   3. Run deployment script"
```

### 2. Production Deployment Script

```bash
#!/bin/bash
# scripts/deploy-production.sh

set -euo pipefail

echo "🚀 APG Cash Management - Production Deployment"
echo "=============================================="

# Configuration
PROJECT_NAME="apg-cash-management"
ENVIRONMENT="production"
BACKUP_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Load environment variables
if [[ -f .env.production ]]; then
    source .env.production
else
    echo "❌ Error: .env.production file not found"
    exit 1
fi

# Validate required environment variables
required_vars=(
    "DATABASE_URL"
    "REDIS_URL"
    "SECRET_KEY"
    "JWT_SECRET_KEY"
    "ENCRYPTION_KEY"
)

for var in "${required_vars[@]}"; do
    if [[ -z "${!var:-}" ]]; then
        echo "❌ Error: Required environment variable $var is not set"
        exit 1
    fi
done

echo "✅ Environment validation passed"

# Create backup of current deployment (if exists)
if docker-compose -f docker-compose.prod.yml ps | grep -q "Up"; then
    echo "💾 Creating backup of current deployment..."
    docker-compose -f docker-compose.prod.yml exec -T postgres-primary pg_dump -U $POSTGRES_USER $POSTGRES_DB > "backups/pre-deployment-backup-${BACKUP_TIMESTAMP}.sql"
    echo "✅ Backup created: backups/pre-deployment-backup-${BACKUP_TIMESTAMP}.sql"
fi

# Pull latest images
echo "📥 Pulling latest Docker images..."
docker-compose -f docker-compose.prod.yml pull

# Stop existing services gracefully
echo "🛑 Stopping existing services..."
docker-compose -f docker-compose.prod.yml down --timeout 30

# Start database services first
echo "🗄️ Starting database services..."
docker-compose -f docker-compose.prod.yml up -d postgres-primary postgres-replica redis-master redis-slave

# Wait for databases to be ready
echo "⏳ Waiting for databases to be ready..."
timeout 60 bash -c 'until docker-compose -f docker-compose.prod.yml exec postgres-primary pg_isready -U $POSTGRES_USER; do sleep 2; done'
timeout 60 bash -c 'until docker-compose -f docker-compose.prod.yml exec redis-master redis-cli ping | grep -q PONG; do sleep 2; done'

echo "✅ Databases are ready"

# Run database migrations
echo "🔄 Running database migrations..."
docker-compose -f docker-compose.prod.yml run --rm apg-cash-management python manage.py migrate

# Start application services
echo "🚀 Starting application services..."
docker-compose -f docker-compose.prod.yml up -d apg-cash-management

# Wait for application to be ready
echo "⏳ Waiting for application to be ready..."
timeout 120 bash -c 'until curl -f http://localhost:8000/health > /dev/null 2>&1; do sleep 5; done'

echo "✅ Application is ready"

# Start remaining services
echo "📊 Starting monitoring and load balancer..."
docker-compose -f docker-compose.prod.yml up -d haproxy prometheus grafana

# Health check
echo "🏥 Performing health checks..."
sleep 10

# Check application health
if curl -f http://localhost/health > /dev/null 2>&1; then
    echo "✅ Application health check passed"
else
    echo "❌ Application health check failed"
    exit 1
fi

# Check database connectivity
if docker-compose -f docker-compose.prod.yml exec postgres-primary pg_isready -U $POSTGRES_USER > /dev/null 2>&1; then
    echo "✅ Database health check passed"
else
    echo "❌ Database health check failed"
    exit 1
fi

# Check Redis connectivity
if docker-compose -f docker-compose.prod.yml exec redis-master redis-cli ping | grep -q PONG; then
    echo "✅ Redis health check passed"
else
    echo "❌ Redis health check failed"
    exit 1
fi

# Display deployment information
echo ""
echo "🎉 Deployment completed successfully!"
echo "========================================"
echo "📊 Service Status:"
docker-compose -f docker-compose.prod.yml ps

echo ""
echo "🔗 Service URLs:"
echo "   • Application: https://localhost"
echo "   • API Documentation: https://localhost/docs"
echo "   • HAProxy Stats: http://localhost:8404/stats"
echo "   • Prometheus: http://localhost:9090"
echo "   • Grafana: http://localhost:3000"
echo ""
echo "📝 Next steps:"
echo "   1. Configure SSL certificates for production domain"
echo "   2. Setup DNS records"
echo "   3. Configure monitoring alerts"
echo "   4. Setup backup automation"
echo "   5. Perform load testing"
echo ""
echo "🔒 Security reminders:"
echo "   • Change default passwords"
echo "   • Configure firewall rules"
echo "   • Enable audit logging"
echo "   • Setup intrusion detection"
```

### 3. Post-Deployment Validation

```bash
#!/bin/bash
# scripts/post-deployment-validation.sh

set -euo pipefail

echo "🔍 APG Cash Management - Post-Deployment Validation"
echo "=================================================="

# Test endpoints
endpoints=(
    "http://localhost/health"
    "http://localhost/api/v1/accounts"
    "http://localhost/api/v1/cash-flows"
    "http://localhost/api/v1/forecasting/performance"
)

echo "🌐 Testing application endpoints..."
for endpoint in "${endpoints[@]}"; do
    if curl -f -H "Authorization: Bearer $TEST_JWT_TOKEN" "$endpoint" > /dev/null 2>&1; then
        echo "✅ $endpoint - OK"
    else
        echo "❌ $endpoint - FAILED"
    fi
done

# Performance tests
echo "⚡ Running performance tests..."
ab -n 100 -c 10 http://localhost/health > /tmp/ab_results.txt 2>&1
requests_per_sec=$(grep "Requests per second" /tmp/ab_results.txt | awk '{print $4}')
echo "📊 Performance: $requests_per_sec requests/second"

# Database performance
echo "🗄️ Testing database performance..."
db_response_time=$(docker-compose -f docker-compose.prod.yml exec postgres-primary psql -U $POSTGRES_USER -d $POSTGRES_DB -c "SELECT NOW();" | grep -E "[0-9]+ ms" | awk '{print $2}')
echo "📊 Database response time: ${db_response_time:-"< 1ms"}"

# Security validation
echo "🔒 Security validation..."
ssl_check=$(curl -I https://localhost 2>&1 | grep -c "SSL")
if [[ $ssl_check -gt 0 ]]; then
    echo "✅ SSL/TLS encryption enabled"
else
    echo "❌ SSL/TLS encryption not working"
fi

echo "✅ Post-deployment validation completed!"
```

---

## 📊 Monitoring & Alerting

### Prometheus Configuration

```yaml
# config/prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alerts.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'apg-cash-management'
    static_configs:
      - targets: ['apg-cash-management:8001']
    metrics_path: /metrics
    scrape_interval: 10s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'haproxy'
    static_configs:
      - targets: ['haproxy:8404']
    metrics_path: /stats/prometheus

  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']
```

### Alert Rules

```yaml
# config/prometheus/alerts.yml
groups:
  - name: apg-cash-management
    rules:
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"

      - alert: DatabaseConnectionsHigh
        expr: pg_stat_activity_count > 180
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Database connections are high"

      - alert: RedisMemoryUsageHigh
        expr: redis_memory_used_bytes / redis_memory_max_bytes > 0.9
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Redis memory usage is critical"
```

---

## 🔄 Backup & Recovery

### Automated Backup Script

```bash
#!/bin/bash
# scripts/backup.sh

BACKUP_DIR="/opt/apg-cash-management/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

# Database backup
docker-compose -f docker-compose.prod.yml exec -T postgres-primary pg_dump -U $POSTGRES_USER $POSTGRES_DB | gzip > "$BACKUP_DIR/db_backup_$TIMESTAMP.sql.gz"

# Redis backup
docker-compose -f docker-compose.prod.yml exec -T redis-master redis-cli --rdb /tmp/dump.rdb BGSAVE
docker cp $(docker-compose -f docker-compose.prod.yml ps -q redis-master):/tmp/dump.rdb "$BACKUP_DIR/redis_backup_$TIMESTAMP.rdb"

# Application logs backup
tar -czf "$BACKUP_DIR/logs_backup_$TIMESTAMP.tar.gz" /var/log/apg-cash

# Clean old backups
find "$BACKUP_DIR" -name "*.gz" -type f -mtime +$RETENTION_DAYS -delete
find "$BACKUP_DIR" -name "*.rdb" -type f -mtime +$RETENTION_DAYS -delete

echo "Backup completed: $TIMESTAMP"
```

---

## 🛡️ Security Hardening

### Security Checklist

- [ ] SSL/TLS certificates configured
- [ ] Firewall rules implemented
- [ ] Database encryption at rest enabled
- [ ] Redis authentication configured
- [ ] Application secrets properly managed
- [ ] Rate limiting configured
- [ ] Security headers implemented
- [ ] Audit logging enabled
- [ ] Regular security updates scheduled
- [ ] Intrusion detection system configured

### Security Configuration Script

```bash
#!/bin/bash
# scripts/security-hardening.sh

echo "🛡️ Applying security hardening..."

# Disable root SSH access
sudo sed -i 's/#PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
sudo systemctl reload sshd

# Install fail2ban
sudo apt-get install -y fail2ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban

# Setup automatic security updates
sudo apt-get install -y unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades

# Configure audit logging
sudo apt-get install -y auditd
sudo systemctl enable auditd
sudo systemctl start auditd

echo "✅ Security hardening completed"
```

---

## 📈 Performance Optimization

### System Optimization

```bash
#!/bin/bash
# scripts/optimize-performance.sh

echo "⚡ Optimizing system performance..."

# Kernel parameters
cat >> /etc/sysctl.conf << EOF
# Network optimization
net.core.somaxconn = 65535
net.core.netdev_max_backlog = 5000
net.ipv4.tcp_max_syn_backlog = 65535
net.ipv4.tcp_keepalive_time = 600
net.ipv4.tcp_keepalive_intvl = 60
net.ipv4.tcp_keepalive_probes = 3

# Memory optimization
vm.swappiness = 10
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5

# File system optimization
fs.file-max = 2097152
EOF

sysctl -p

# Increase file limits
cat >> /etc/security/limits.conf << EOF
* soft nofile 65535
* hard nofile 65535
* soft nproc 65535
* hard nproc 65535
EOF

echo "✅ Performance optimization completed"
```

---

**Next Review Date:** April 27, 2025

*© 2025 Datacraft. All rights reserved.*