# Registry (regy) - Deployment Guide

**APG Registry Capability - Production Deployment & Operations**

Version: 1.0.0  
Author: APG Platform Team  
Copyright: © 2025 Datacraft  
Website: www.datacraft.co.ke

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Production Deployment](#production-deployment)
5. [High Availability Setup](#high-availability-setup)
6. [Security Configuration](#security-configuration)
7. [Monitoring & Observability](#monitoring--observability)
8. [Performance Tuning](#performance-tuning)
9. [Backup & Recovery](#backup--recovery)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 4 cores
- Memory: 8GB RAM
- Storage: 100GB SSD
- Network: 1Gbps

**Recommended for Production:**
- CPU: 8+ cores
- Memory: 16GB+ RAM
- Storage: 500GB+ SSD with high IOPS
- Network: 10Gbps with redundancy

### Software Dependencies

**Core Dependencies:**
```bash
# Python Runtime
Python >= 3.9
pip >= 21.0

# Database
PostgreSQL >= 13.0
Redis >= 6.0 (optional, for caching)

# APG Platform
apg-auth >= 1.0.0
apg-conf >= 1.0.0
apg-moni >= 1.0.0
apg-audl >= 1.0.0
```

**Optional Dependencies:**
```bash
# ML/AI Features
scikit-learn >= 1.0.0
tensorflow >= 2.8.0
prometheus-client >= 0.14.0

# High Availability
consul >= 1.12.0
etcd >= 3.5.0
```

### Network Requirements

**Inbound Ports:**
- 5000: HTTP API
- 5001: WebSocket (real-time updates)
- 5432: PostgreSQL (if co-located)
- 6379: Redis (if co-located)

**Outbound Ports:**
- 443: HTTPS (external dependencies)
- 5432: PostgreSQL (external)
- 6379: Redis (external)
- 8500: Consul (if used)
- 2379: etcd (if used)

---

## Installation

### APG Platform Installation

1. **Install APG Core:**
```bash
# Install APG platform
curl -sSL https://install.datacraft.co.ke/apg | bash

# Verify installation
apg version
apg status
```

2. **Initialize APG Database:**
```bash
apg db init --database apg_registry
apg db migrate --capability regy
```

3. **Install Registry Capability:**
```bash
# Install from APG registry
apg capability install regy

# Or install from source
cd /path/to/apg/capabilities/common/regy
apg capability install --local .
```

### Manual Installation

1. **Clone Repository:**
```bash
git clone https://github.com/datacraft/apg.git
cd apg/capabilities/common/regy
```

2. **Setup Python Environment:**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

3. **Database Setup:**
```bash
# Create database
createdb apg_registry

# Run migrations
alembic upgrade head
```

4. **Initialize Registry:**
```bash
python -m apg.capabilities.common.regy.service --init
```

### Docker Installation

1. **Using Docker Compose:**
```yaml
# docker-compose.yml
version: '3.8'

services:
  apg-registry:
    image: datacraft/apg-registry:1.0.0
    ports:
      - "5000:5000"
      - "5001:5001"
    environment:
      - APG_DATABASE_URL=postgresql://user:pass@db:5432/apg_registry
      - APG_REDIS_URL=redis://redis:6379/0
      - APG_TENANT_ID=default
      - APG_LOG_LEVEL=INFO
    depends_on:
      - db
      - redis
    volumes:
      - ./config:/app/config
    restart: unless-stopped

  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=apg_registry
      - POSTGRES_USER=apg_user
      - POSTGRES_PASSWORD=secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:6-alpine
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

2. **Start Services:**
```bash
docker-compose up -d
```

### Kubernetes Installation

1. **Namespace Setup:**
```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: apg-registry
  labels:
    name: apg-registry
```

2. **ConfigMap:**
```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: apg-registry-config
  namespace: apg-registry
data:
  config.yaml: |
    database:
      url: postgresql://user:pass@postgres:5432/apg_registry
      max_connections: 100
      timeout_seconds: 30
    
    cache:
      enabled: true
      redis_url: redis://redis:6379/0
      ttl_seconds: 300
    
    ml_features:
      enabled: true
      model_path: /app/models
      prediction_interval_minutes: 5
    
    security:
      authentication_required: true
      encryption_enabled: true
      rate_limiting:
        requests_per_minute: 1000
        burst_size: 100
```

3. **Deployment:**
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: apg-registry
  namespace: apg-registry
spec:
  replicas: 3
  selector:
    matchLabels:
      app: apg-registry
  template:
    metadata:
      labels:
        app: apg-registry
    spec:
      containers:
      - name: apg-registry
        image: datacraft/apg-registry:1.0.0
        ports:
        - containerPort: 5000
        - containerPort: 5001
        env:
        - name: APG_CONFIG_FILE
          value: /app/config/config.yaml
        volumeMounts:
        - name: config
          mountPath: /app/config
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /api/regy/v1/status
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/regy/v1/ready
            port: 5000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: config
        configMap:
          name: apg-registry-config
```

4. **Service:**
```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: apg-registry-service
  namespace: apg-registry
spec:
  selector:
    app: apg-registry
  ports:
  - name: http
    port: 5000
    targetPort: 5000
  - name: websocket
    port: 5001
    targetPort: 5001
  type: ClusterIP
```

5. **Deploy to Kubernetes:**
```bash
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

---

## Configuration

### Core Configuration

Create `/etc/apg/registry/config.yaml`:

```yaml
# Core Settings
tenant_id: "production"
service_name: "apg-registry"
log_level: "INFO"
debug_mode: false

# Database Configuration
database:
  url: "postgresql://apg_user:secure_pass@localhost:5432/apg_registry"
  pool_size: 20
  max_connections: 100
  connection_timeout: 30
  query_timeout: 60
  ssl_mode: "require"

# Cache Configuration
cache:
  enabled: true
  backend: "redis"
  redis_url: "redis://localhost:6379/0"
  ttl_seconds: 300
  max_entries: 10000

# Security Configuration
security:
  authentication_required: true
  authorization_enabled: true
  encryption_enabled: true
  
  # JWT Configuration
  jwt:
    secret_key: "${JWT_SECRET_KEY}"
    algorithm: "HS256"
    expiration_hours: 24
  
  # Rate Limiting
  rate_limiting:
    enabled: true
    requests_per_minute: 1000
    burst_size: 100
    per_ip_limit: 100

# ML/AI Features
ml_features:
  enabled: true
  model_path: "/var/lib/apg/registry/models"
  
  # Intelligent Discovery
  intelligent_ranking:
    enabled: true
    model_version: "v2.1"
    update_interval_hours: 6
  
  # Predictive Analytics
  predictive_scaling:
    enabled: true
    prediction_window_hours: 24
    confidence_threshold: 0.8
  
  # Anomaly Detection
  anomaly_detection:
    enabled: true
    sensitivity: "medium"
    alert_threshold: 0.75
    auto_remediation: false

# Health Monitoring
health_monitoring:
  enabled: true
  default_interval_seconds: 30
  max_concurrent_checks: 100
  retry_attempts: 3
  
  # Adaptive Monitoring
  adaptive_intervals:
    enabled: true
    min_interval_seconds: 10
    max_interval_seconds: 300
    adjustment_factor: 1.5

# Circuit Breaker
circuit_breaker:
  enabled: true
  default_failure_threshold: 5
  default_success_threshold: 3
  default_timeout_seconds: 60
  
  # ML-Enhanced Features
  adaptive_thresholds: true
  pattern_recognition: true
  intelligent_recovery: true

# Performance Settings
performance:
  # Connection Pooling
  connection_pool:
    initial_size: 10
    max_size: 100
    idle_timeout_seconds: 300
  
  # Query Optimization
  query_optimization:
    enabled: true
    index_hints: true
    query_cache_size: 1000
  
  # Async Settings
  async_workers: 10
  max_concurrent_requests: 1000

# Integration Settings
integrations:
  # APG Platform Integration
  apg:
    auth_service_url: "http://apg-auth:5000"
    config_service_url: "http://apg-config:5000"
    monitoring_service_url: "http://apg-monitoring:5000"
    audit_service_url: "http://apg-audit:5000"
  
  # External Integrations
  external:
    consul:
      enabled: false
      url: "http://consul:8500"
    
    etcd:
      enabled: false
      endpoints: ["http://etcd:2379"]
    
    prometheus:
      enabled: true
      port: 9090
      metrics_path: "/metrics"

# Logging Configuration
logging:
  version: 1
  disable_existing_loggers: false
  
  formatters:
    detailed:
      format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
  handlers:
    console:
      class: logging.StreamHandler
      level: INFO
      formatter: detailed
      stream: ext://sys.stdout
    
    file:
      class: logging.handlers.RotatingFileHandler
      level: DEBUG
      formatter: detailed
      filename: /var/log/apg/registry.log
      maxBytes: 10485760  # 10MB
      backupCount: 5
  
  loggers:
    apg.registry:
      level: DEBUG
      handlers: [console, file]
      propagate: false
    
    apg.registry.ml:
      level: INFO
      handlers: [console, file]
      propagate: false
  
  root:
    level: INFO
    handlers: [console]
```

### Environment Variables

Set the following environment variables:

```bash
# Core Configuration
export APG_TENANT_ID="production"
export APG_CONFIG_FILE="/etc/apg/registry/config.yaml"
export APG_LOG_LEVEL="INFO"

# Database
export APG_DATABASE_URL="postgresql://user:pass@localhost:5432/apg_registry"
export APG_DATABASE_SSL_MODE="require"

# Cache
export APG_REDIS_URL="redis://localhost:6379/0"
export APG_CACHE_TTL="300"

# Security
export JWT_SECRET_KEY="your-super-secure-secret-key"
export APG_ENCRYPTION_KEY="your-encryption-key"

# ML Features
export APG_ML_ENABLED="true"
export APG_ML_MODEL_PATH="/var/lib/apg/registry/models"

# Performance
export APG_MAX_WORKERS="10"
export APG_MAX_CONNECTIONS="100"
```

---

## Production Deployment

### Pre-deployment Checklist

1. **Infrastructure Verification:**
```bash
# Check system resources
free -h
df -h
nproc

# Verify network connectivity
nc -zv postgres-host 5432
nc -zv redis-host 6379
```

2. **Database Setup:**
```bash
# Create production database
createdb apg_registry_prod

# Run migrations
export APG_DATABASE_URL="postgresql://user:pass@host:5432/apg_registry_prod"
alembic upgrade head

# Verify schema
psql $APG_DATABASE_URL -c "\dt"
```

3. **Security Hardening:**
```bash
# Generate secure keys
openssl rand -hex 32  # JWT secret
openssl rand -hex 32  # Encryption key

# Set proper file permissions
chmod 600 /etc/apg/registry/config.yaml
chown apg:apg /etc/apg/registry/config.yaml
```

### Deployment Steps

1. **Install Application:**
```bash
# Create application user
useradd -r -s /bin/false apg-registry

# Install application
cp -r /path/to/apg/capabilities/common/regy /opt/apg-registry/
chown -R apg-registry:apg-registry /opt/apg-registry/

# Install Python dependencies
cd /opt/apg-registry/
pip install -r requirements.txt
```

2. **Create Systemd Service:**
```ini
# /etc/systemd/system/apg-registry.service
[Unit]
Description=APG Registry Service
After=network.target postgresql.service redis.service
Wants=postgresql.service redis.service

[Service]
Type=exec
User=apg-registry
Group=apg-registry
WorkingDirectory=/opt/apg-registry
Environment=APG_CONFIG_FILE=/etc/apg/registry/config.yaml
ExecStart=/opt/apg-registry/venv/bin/python -m apg.capabilities.common.regy.api
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=apg-registry

# Security settings
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/var/log/apg /var/lib/apg

[Install]
WantedBy=multi-user.target
```

3. **Start and Enable Service:**
```bash
systemctl daemon-reload
systemctl enable apg-registry
systemctl start apg-registry
systemctl status apg-registry
```

4. **Verify Deployment:**
```bash
# Check service status
curl http://localhost:5000/api/regy/v1/status

# Check readiness
curl http://localhost:5000/api/regy/v1/ready

# Verify database connectivity
curl http://localhost:5000/api/regy/v1/metrics/registry/statistics
```

### Load Balancer Configuration

**Nginx Configuration:**
```nginx
# /etc/nginx/sites-available/apg-registry
upstream apg_registry {
    least_conn;
    server registry-1.internal:5000 weight=1 max_fails=3 fail_timeout=30s;
    server registry-2.internal:5000 weight=1 max_fails=3 fail_timeout=30s;
    server registry-3.internal:5000 weight=1 max_fails=3 fail_timeout=30s;
}

upstream apg_registry_ws {
    ip_hash;  # Sticky sessions for WebSocket
    server registry-1.internal:5001;
    server registry-2.internal:5001;
    server registry-3.internal:5001;
}

server {
    listen 80;
    listen 443 ssl http2;
    server_name registry.yourcompany.com;
    
    # SSL Configuration
    ssl_certificate /etc/ssl/certs/registry.crt;
    ssl_certificate_key /etc/ssl/private/registry.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE+AESGCM:ECDHE+AES256:ECDHE+AES128:!aNULL:!MD5:!DSS;
    
    # Security Headers
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
    
    # API Endpoints
    location /api/regy/ {
        proxy_pass http://apg_registry;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # Buffer settings
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }
    
    # WebSocket Endpoints
    location /ws/ {
        proxy_pass http://apg_registry_ws;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket timeouts
        proxy_connect_timeout 7d;
        proxy_send_timeout 7d;
        proxy_read_timeout 7d;
    }
    
    # Health Check
    location /health {
        proxy_pass http://apg_registry/api/regy/v1/status;
        access_log off;
    }
}
```

---

## High Availability Setup

### Multi-Region Deployment

1. **Primary Region (us-east-1):**
```yaml
# primary-region.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: apg-registry-primary
  namespace: apg-registry
  labels:
    region: us-east-1
    role: primary
spec:
  replicas: 3
  selector:
    matchLabels:
      app: apg-registry
      region: us-east-1
  template:
    metadata:
      labels:
        app: apg-registry
        region: us-east-1
    spec:
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchLabels:
                app: apg-registry
            topologyKey: kubernetes.io/hostname
      containers:
      - name: apg-registry
        image: datacraft/apg-registry:1.0.0
        env:
        - name: APG_REGION
          value: "us-east-1"
        - name: APG_ROLE
          value: "primary"
        - name: APG_REPLICATION_MODE
          value: "master"
```

2. **Secondary Region (us-west-2):**
```yaml
# secondary-region.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: apg-registry-secondary
  namespace: apg-registry
  labels:
    region: us-west-2
    role: secondary
spec:
  replicas: 2
  selector:
    matchLabels:
      app: apg-registry
      region: us-west-2
  template:
    metadata:
      labels:
        app: apg-registry
        region: us-west-2
    spec:
      containers:
      - name: apg-registry
        image: datacraft/apg-registry:1.0.0
        env:
        - name: APG_REGION
          value: "us-west-2"
        - name: APG_ROLE
          value: "secondary"
        - name: APG_REPLICATION_MODE
          value: "replica"
        - name: APG_PRIMARY_ENDPOINT
          value: "https://registry-primary.us-east-1.internal"
```

### Database High Availability

1. **PostgreSQL Streaming Replication:**
```ini
# postgresql.conf (Primary)
listen_addresses = '*'
wal_level = replica
max_wal_senders = 3
max_replication_slots = 3
synchronous_commit = on
synchronous_standby_names = 'standby1,standby2'

# Enable archiving
archive_mode = on
archive_command = 'cp %p /var/lib/postgresql/archive/%f'
```

```ini
# recovery.conf (Standby)
standby_mode = 'on'
primary_conninfo = 'host=primary-db port=5432 user=replicator'
trigger_file = '/tmp/postgresql.trigger'
```

2. **Redis Sentinel Configuration:**
```conf
# sentinel.conf
port 26379
sentinel monitor apg-registry-master 192.168.1.10 6379 2
sentinel auth-pass apg-registry-master yourpassword
sentinel down-after-milliseconds apg-registry-master 5000
sentinel failover-timeout apg-registry-master 10000
sentinel parallel-syncs apg-registry-master 1
```

### Service Mesh Integration

**Istio Configuration:**
```yaml
# service-mesh.yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: apg-registry-routing
spec:
  hosts:
  - registry.yourcompany.com
  http:
  - match:
    - headers:
        region:
          exact: us-east-1
    route:
    - destination:
        host: apg-registry-service
        subset: us-east-1
      weight: 100
  - match:
    - headers:
        region:
          exact: us-west-2
    route:
    - destination:
        host: apg-registry-service
        subset: us-west-2
      weight: 100
  - route:
    - destination:
        host: apg-registry-service
        subset: us-east-1
      weight: 70
    - destination:
        host: apg-registry-service
        subset: us-west-2
      weight: 30
    fault:
      delay:
        percentage:
          value: 0.1
        fixedDelay: 5s

---
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: apg-registry-destinations
spec:
  host: apg-registry-service
  trafficPolicy:
    outlierDetection:
      consecutiveErrors: 3
      interval: 30s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
    loadBalancer:
      consistentHash:
        httpHeaderName: "X-Tenant-ID"
  subsets:
  - name: us-east-1
    labels:
      region: us-east-1
  - name: us-west-2
    labels:
      region: us-west-2
```

---

## Security Configuration

### Authentication & Authorization

1. **JWT Configuration:**
```yaml
# Enhanced JWT settings
security:
  jwt:
    secret_key: "${JWT_SECRET_KEY}"
    algorithm: "RS256"  # Use RSA for production
    public_key_file: "/etc/apg/registry/jwt-public.pem"
    private_key_file: "/etc/apg/registry/jwt-private.pem"
    expiration_hours: 8  # Shorter expiration
    refresh_token_enabled: true
    refresh_token_expiration_days: 7
    
  # Multi-factor Authentication
  mfa:
    enabled: true
    totp_issuer: "APG Registry"
    backup_codes: 10
```

2. **RBAC Policies:**
```yaml
# rbac-policies.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: apg-registry-rbac
data:
  policies.yaml: |
    roles:
      - name: "registry-admin"
        permissions:
          - "registry:*"
          - "health:*"
          - "metrics:*"
          - "events:*"
      
      - name: "service-manager"
        permissions:
          - "registry:register_service"
          - "registry:update_service"
          - "registry:deregister_service"
          - "registry:list_services"
          - "registry:get_service"
          - "health:view_health"
      
      - name: "service-reader"
        permissions:
          - "registry:discover_services"
          - "registry:list_services" 
          - "registry:get_service"
          - "health:view_health"
      
      - name: "monitoring-user"
        permissions:
          - "health:*"
          - "metrics:*"
          - "events:view_events"
    
    users:
      - username: "admin@yourcompany.com"
        roles: ["registry-admin"]
      - username: "devops@yourcompany.com"  
        roles: ["service-manager", "monitoring-user"]
      - username: "developer@yourcompany.com"
        roles: ["service-reader"]
```

### Network Security

1. **Firewall Rules:**
```bash
# iptables rules
iptables -A INPUT -p tcp --dport 5000 -s 10.0.0.0/8 -j ACCEPT
iptables -A INPUT -p tcp --dport 5001 -s 10.0.0.0/8 -j ACCEPT
iptables -A INPUT -p tcp --dport 5000 -j DROP
iptables -A INPUT -p tcp --dport 5001 -j DROP
```

2. **Network Policies (Kubernetes):**
```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: apg-registry-network-policy
  namespace: apg-registry
spec:
  podSelector:
    matchLabels:
      app: apg-registry
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: apg-platform
    - podSelector:
        matchLabels:
          apg-component: "true"
    ports:
    - protocol: TCP
      port: 5000
    - protocol: TCP
      port: 5001
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: apg-database
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - namespaceSelector:
        matchLabels:
          name: apg-cache
    ports:
    - protocol: TCP
      port: 6379
```

### Data Encryption

1. **Database Encryption:**
```sql
-- Enable transparent data encryption
ALTER SYSTEM SET ssl = on;
ALTER SYSTEM SET ssl_cert_file = '/etc/ssl/certs/postgresql.crt';
ALTER SYSTEM SET ssl_key_file = '/etc/ssl/private/postgresql.key';

-- Column-level encryption for sensitive data
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Encrypt service metadata
ALTER TABLE services ADD COLUMN encrypted_metadata BYTEA;
UPDATE services SET encrypted_metadata = pgp_sym_encrypt(metadata::text, 'encryption_key');
```

2. **Application-level Encryption:**
```python
# config.yaml
security:
  encryption:
    enabled: true
    algorithm: "AES-256-GCM"
    key_rotation_days: 90
    
    # Fields to encrypt
    encrypted_fields:
      - "service_metadata"
      - "instance_details"
      - "health_check_credentials"
      - "circuit_breaker_config"
```

---

## Monitoring & Observability

### Prometheus Metrics

1. **Metrics Configuration:**
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  
scrape_configs:
- job_name: 'apg-registry'
  static_configs:
  - targets: 
    - 'registry-1.internal:9090'
    - 'registry-2.internal:9090'  
    - 'registry-3.internal:9090'
  metrics_path: '/metrics'
  scrape_interval: 10s
  
- job_name: 'apg-registry-health'
  static_configs:
  - targets:
    - 'registry-1.internal:5000'
    - 'registry-2.internal:5000'
    - 'registry-3.internal:5000'
  metrics_path: '/api/regy/v1/metrics/registry'
  scrape_interval: 30s
```

2. **Custom Metrics:**
```python
# Custom metrics exported
registry_services_total
registry_services_healthy
registry_services_degraded  
registry_services_unhealthy
registry_discovery_requests_total
registry_discovery_duration_seconds
registry_health_checks_total
registry_circuit_breaker_state
registry_ml_predictions_total
registry_cache_hit_ratio
```

### Grafana Dashboards

1. **Service Overview Dashboard:**
```json
{
  "dashboard": {
    "title": "APG Registry - Service Overview",
    "panels": [
      {
        "title": "Total Services",
        "type": "stat",
        "targets": [
          {
            "expr": "registry_services_total",
            "legendFormat": "Total Services"
          }
        ]
      },
      {
        "title": "Service Health Distribution",
        "type": "piechart", 
        "targets": [
          {
            "expr": "registry_services_healthy",
            "legendFormat": "Healthy"
          },
          {
            "expr": "registry_services_degraded",
            "legendFormat": "Degraded"
          },
          {
            "expr": "registry_services_unhealthy", 
            "legendFormat": "Unhealthy"
          }
        ]
      },
      {
        "title": "Discovery Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(registry_discovery_requests_total[5m])",
            "legendFormat": "Requests/sec"
          },
          {
            "expr": "histogram_quantile(0.95, registry_discovery_duration_seconds_bucket)",
            "legendFormat": "95th percentile latency"
          }
        ]
      }
    ]
  }
}
```

### Log Aggregation

1. **Fluentd Configuration:**
```conf
<source>
  @type tail
  path /var/log/apg/registry.log
  pos_file /var/log/fluentd/apg-registry.log.pos
  tag apg.registry
  format json
</source>

<filter apg.registry>
  @type parser
  key_name message
  reserve_data true
  <parse>
    @type json
  </parse>
</filter>

<match apg.registry>
  @type elasticsearch
  host elasticsearch.internal
  port 9200
  index_name apg-registry
  type_name _doc
  include_timestamp true
  
  <buffer>
    @type file
    path /var/log/fluentd/buffer/apg-registry
    flush_mode interval
    flush_interval 10s
    chunk_limit_size 10MB
  </buffer>
</match>
```

### Alert Rules

1. **Prometheus Alert Rules:**
```yaml
# alerts.yml
groups:
- name: apg-registry
  rules:
  - alert: RegistryHighErrorRate
    expr: rate(registry_errors_total[5m]) > 0.1
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Registry error rate is high"
      description: "Error rate has exceeded 10% for 5 minutes"
  
  - alert: RegistryServiceDown
    expr: up{job="apg-registry"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Registry service is down"
      description: "Registry service {{ $labels.instance }} is unreachable"
  
  - alert: RegistryHighLatency
    expr: histogram_quantile(0.95, registry_discovery_duration_seconds_bucket) > 0.5
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "Registry discovery latency is high"
      description: "95th percentile latency is {{ $value }}s"
  
  - alert: RegistryUnhealthyServices
    expr: registry_services_unhealthy > 5
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Many services are unhealthy"
      description: "{{ $value }} services are in unhealthy state"
```

---

## Performance Tuning

### Database Optimization

1. **PostgreSQL Tuning:**
```sql
-- postgresql.conf optimizations
shared_buffers = '4GB'                # 25% of total RAM
effective_cache_size = '12GB'         # 75% of total RAM
work_mem = '256MB'                    # Per query memory
maintenance_work_mem = '1GB'          # Maintenance operations
checkpoint_completion_target = 0.7    # Spread checkpoints
wal_buffers = '64MB'                  # WAL buffer size
random_page_cost = 1.1               # SSD optimization

-- Connection pooling
max_connections = 200
shared_preload_libraries = 'pg_stat_statements'

-- Query optimization
enable_partitionwise_join = on
enable_partitionwise_aggregate = on
```

2. **Indexing Strategy:**
```sql
-- Core indexes for performance
CREATE INDEX CONCURRENTLY idx_services_tenant_name ON services(tenant_id, name);
CREATE INDEX CONCURRENTLY idx_services_type_env ON services(service_type, environment);
CREATE INDEX CONCURRENTLY idx_services_namespace ON services(namespace);
CREATE INDEX CONCURRENTLY idx_services_tags ON services USING GIN(tags);
CREATE INDEX CONCURRENTLY idx_services_status ON services(status) WHERE status != 'healthy';

-- Health monitoring indexes
CREATE INDEX CONCURRENTLY idx_health_service_instance ON service_health(service_id, instance_id);
CREATE INDEX CONCURRENTLY idx_health_timestamp ON service_health(last_updated);

-- Metrics indexes
CREATE INDEX CONCURRENTLY idx_metrics_service_time ON service_metrics(service_id, timestamp);
CREATE INDEX CONCURRENTLY idx_metrics_time_type ON service_metrics(timestamp, metric_type);

-- Events indexes  
CREATE INDEX CONCURRENTLY idx_events_service_time ON service_events(service_id, timestamp);
CREATE INDEX CONCURRENTLY idx_events_type_severity ON service_events(event_type, severity);
```

### Application Performance

1. **Caching Strategy:**
```yaml
# Enhanced caching configuration
cache:
  enabled: true
  backend: "redis_cluster"
  
  # Redis cluster configuration
  redis_cluster:
    nodes:
      - "redis-1.internal:6379"
      - "redis-2.internal:6379" 
      - "redis-3.internal:6379"
    max_connections: 100
    retry_on_timeout: true
    
  # Cache policies
  policies:
    service_discovery:
      ttl_seconds: 300
      max_entries: 10000
      eviction_policy: "lru"
    
    service_health:
      ttl_seconds: 60
      max_entries: 50000
      eviction_policy: "ttl"
    
    registry_stats:
      ttl_seconds: 60
      max_entries: 1000
      eviction_policy: "lru"

# Connection pooling
connection_pool:
  database:
    min_connections: 10
    max_connections: 100
    connection_lifetime_minutes: 30
    
  redis:
    min_connections: 5
    max_connections: 50
    connection_lifetime_minutes: 15
```

2. **Async Optimization:**
```python
# async-config.yaml
async_settings:
  # Worker configuration
  workers: 16                          # Number of async workers
  max_concurrent_requests: 1000        # Concurrent request limit
  request_timeout_seconds: 30          # Request timeout
  
  # Event loop settings
  event_loop_policy: "uvloop"          # High-performance event loop
  
  # Connection limits
  tcp_connector:
    limit: 100                         # Total connection pool size
    limit_per_host: 30                 # Per-host connection limit
    keepalive_timeout: 30              # Keep-alive timeout
    
  # Background tasks
  background_tasks:
    health_monitoring:
      interval_seconds: 30
      batch_size: 100
      
    metrics_collection:
      interval_seconds: 60
      batch_size: 50
      
    cache_cleanup:
      interval_seconds: 300
      batch_size: 1000
```

### Load Testing

1. **Performance Benchmarking:**
```bash
# Install load testing tools
pip install locust

# Run load tests
locust -f load_tests.py --host=http://registry.internal:5000
```

```python
# load_tests.py
from locust import HttpUser, task, between

class RegistryUser(HttpUser):
    wait_time = between(0.5, 2.0)
    
    def on_start(self):
        self.client.headers.update({
            'X-Tenant-ID': 'load-test',
            'Content-Type': 'application/json'
        })
    
    @task(3)
    def discover_services(self):
        self.client.post("/api/regy/v1/discovery/search", json={
            "service_type": "microservice",
            "environment": "production",
            "limit": 25
        })
    
    @task(2)
    def list_services(self):
        self.client.get("/api/regy/v1/services?limit=50")
    
    @task(1)
    def health_check(self):
        self.client.get("/api/regy/v1/health")
    
    @task(1)
    def registry_stats(self):
        self.client.get("/api/regy/v1/metrics/registry/statistics")
```

2. **Performance Targets:**
```yaml
# performance-targets.yaml
targets:
  throughput:
    service_discovery: "1000 req/s"     # Peak discovery load
    service_registration: "100 req/s"   # Registration throughput  
    health_checks: "500 req/s"          # Health monitoring load
    
  latency:
    service_discovery_p95: "50ms"       # 95th percentile latency
    service_registration_p95: "100ms"   # Registration latency
    health_check_p95: "25ms"            # Health check latency
    
  availability:
    uptime: "99.95%"                    # Service availability
    error_rate: "<0.1%"                 # Error rate threshold
    
  scalability:
    max_services: 100000                # Maximum registered services
    max_concurrent_users: 10000         # Concurrent user limit
    max_requests_per_second: 10000      # Peak RPS
```

---

## Backup & Recovery

### Database Backup

1. **Automated Backup Script:**
```bash
#!/bin/bash
# /usr/local/bin/apg-registry-backup.sh

set -euo pipefail

# Configuration
DB_HOST="localhost"
DB_NAME="apg_registry"
DB_USER="apg_backup"
BACKUP_DIR="/var/backups/apg-registry"
RETENTION_DAYS=30

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Generate backup filename
BACKUP_FILE="$BACKUP_DIR/apg-registry-$(date +%Y%m%d_%H%M%S).sql.gz"

# Perform backup
pg_dump -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" \
    --verbose --clean --if-exists --no-owner --no-privileges \
    | gzip > "$BACKUP_FILE"

# Verify backup
if [ -s "$BACKUP_FILE" ]; then
    echo "Backup completed successfully: $BACKUP_FILE"
else
    echo "Backup failed: $BACKUP_FILE is empty"
    exit 1
fi

# Clean old backups
find "$BACKUP_DIR" -name "apg-registry-*.sql.gz" -mtime +$RETENTION_DAYS -delete

# Upload to S3 (optional)
if command -v aws &> /dev/null; then
    aws s3 cp "$BACKUP_FILE" "s3://your-backup-bucket/apg-registry/"
fi

echo "Backup process completed"
```

2. **Cron Configuration:**
```bash
# crontab -e
# Daily backup at 2 AM
0 2 * * * /usr/local/bin/apg-registry-backup.sh >> /var/log/apg-registry-backup.log 2>&1

# Weekly full backup at Sunday 1 AM
0 1 * * 0 /usr/local/bin/apg-registry-full-backup.sh >> /var/log/apg-registry-backup.log 2>&1
```

### Disaster Recovery

1. **Recovery Procedures:**
```bash
#!/bin/bash
# /usr/local/bin/apg-registry-restore.sh

set -euo pipefail

BACKUP_FILE="$1"
DB_HOST="localhost"
DB_NAME="apg_registry"
DB_USER="postgres"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

# Stop registry service
systemctl stop apg-registry

# Drop and recreate database
dropdb -h "$DB_HOST" -U "$DB_USER" "$DB_NAME" || true
createdb -h "$DB_HOST" -U "$DB_USER" "$DB_NAME"

# Restore from backup
if [[ "$BACKUP_FILE" == *.gz ]]; then
    gunzip -c "$BACKUP_FILE" | psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME"
else
    psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" < "$BACKUP_FILE"
fi

# Run migrations (if needed)
cd /opt/apg-registry
alembic upgrade head

# Start registry service
systemctl start apg-registry

# Verify restoration
sleep 10
curl -f http://localhost:5000/api/regy/v1/status

echo "Database restoration completed successfully"
```

2. **Point-in-Time Recovery:**
```bash
# Enable WAL archiving in postgresql.conf
wal_level = replica
archive_mode = on
archive_command = 'test ! -f /var/lib/postgresql/archive/%f && cp %p /var/lib/postgresql/archive/%f'
archive_timeout = 300

# Recovery script
#!/bin/bash
# Point-in-time recovery to specific timestamp
RECOVERY_TIME="2025-01-15 14:30:00"

# Stop service
systemctl stop apg-registry

# Restore base backup
pg_basebackup -h primary-db -D /var/lib/postgresql/recovery -U replicator -v -P -W

# Configure recovery
cat > /var/lib/postgresql/recovery/recovery.conf << EOF
restore_command = 'cp /var/lib/postgresql/archive/%f %p'
recovery_target_time = '$RECOVERY_TIME'
recovery_target_action = 'promote'
EOF

# Start PostgreSQL in recovery mode
sudo -u postgres /usr/lib/postgresql/13/bin/postgres -D /var/lib/postgresql/recovery
```

### Configuration Backup

1. **Configuration Sync:**
```bash
#!/bin/bash
# /usr/local/bin/apg-registry-config-backup.sh

CONFIG_DIRS=(
    "/etc/apg/registry"
    "/opt/apg-registry/config"
    "/etc/systemd/system/apg-registry.service"
)

BACKUP_DIR="/var/backups/apg-registry-config"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p "$BACKUP_DIR"

# Backup configurations
tar -czf "$BACKUP_DIR/config-$DATE.tar.gz" "${CONFIG_DIRS[@]}"

# Sync to remote location
rsync -av "$BACKUP_DIR/" backup-server:/backups/apg-registry-config/
```

---

## Troubleshooting

### Common Issues

#### High Memory Usage

**Problem**: Registry consuming excessive memory

**Diagnosis:**
```bash
# Check memory usage
ps aux | grep apg-registry
free -h

# Check for memory leaks
valgrind --tool=memcheck python -m apg.capabilities.common.regy.api

# Monitor memory over time
while true; do
    echo "$(date): $(ps -o pid,vsz,rss,comm -p $(pgrep apg-registry))"
    sleep 60
done
```

**Solutions:**
1. Tune cache settings
2. Reduce worker count
3. Enable garbage collection tuning
4. Check for circular references

#### Database Connection Issues

**Problem**: Connection pool exhaustion

**Diagnosis:**
```sql
-- Check active connections
SELECT count(*) as active_connections 
FROM pg_stat_activity 
WHERE datname = 'apg_registry';

-- Check connection states
SELECT state, count(*) 
FROM pg_stat_activity 
WHERE datname = 'apg_registry' 
GROUP BY state;

-- Check for blocked queries
SELECT pid, now() - pg_stat_activity.query_start AS duration, query 
FROM pg_stat_activity 
WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes';
```

**Solutions:**
```yaml
# Optimize connection pooling
database:
  pool_size: 20
  max_overflow: 30
  pool_timeout: 30
  pool_recycle: 3600
  pool_pre_ping: true
```

#### Service Discovery Performance

**Problem**: Slow discovery queries

**Diagnosis:**
```sql
-- Enable query logging
ALTER SYSTEM SET log_statement = 'all';
ALTER SYSTEM SET log_min_duration_statement = 100;

-- Check slow queries
SELECT query, mean_time, calls, total_time 
FROM pg_stat_statements 
WHERE mean_time > 100 
ORDER BY mean_time DESC;

-- Analyze query plans
EXPLAIN ANALYZE SELECT * FROM services WHERE service_type = 'microservice';
```

**Solutions:**
1. Add missing indexes
2. Optimize query filters
3. Enable intelligent ranking carefully
4. Use database query cache

### Debug Mode

**Enable Debug Logging:**
```yaml
# config.yaml
logging:
  level: DEBUG
  
debug:
  enabled: true
  
  # Debug features
  query_logging: true
  performance_tracking: true
  memory_profiling: true
  
  # Debug endpoints
  debug_endpoints: true
  health_verbose: true
```

**Debug Endpoints:**
```bash
# Internal state
curl http://localhost:5000/debug/state

# Performance metrics
curl http://localhost:5000/debug/performance

# Memory usage
curl http://localhost:5000/debug/memory

# Active connections
curl http://localhost:5000/debug/connections
```

### Log Analysis

**Key Log Patterns:**
```bash
# Error patterns
grep -E "(ERROR|CRITICAL)" /var/log/apg/registry.log

# Performance issues
grep -E "(slow|timeout|exceeded)" /var/log/apg/registry.log

# Database issues
grep -E "(connection.*failed|deadlock|timeout)" /var/log/apg/registry.log

# Memory issues
grep -E "(memory|oom|OutOfMemory)" /var/log/apg/registry.log
```

### Support Information

When contacting support, include:

1. **System Information:**
```bash
# System specs
uname -a
lscpu
free -h
df -h

# APG version
apg version

# Service status
systemctl status apg-registry
```

2. **Logs:**
```bash
# Recent logs
journalctl -u apg-registry --since "1 hour ago"

# Application logs
tail -n 1000 /var/log/apg/registry.log
```

3. **Configuration:**
```bash
# Sanitized configuration (remove secrets)
grep -v -E "(password|secret|key)" /etc/apg/registry/config.yaml
```

---

*This deployment guide covers comprehensive production deployment of the APG Registry capability. For additional support and advanced configurations, contact the APG Platform Team at support@datacraft.co.ke.*