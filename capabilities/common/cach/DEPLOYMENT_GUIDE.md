# APG Cache Management - Production Deployment Guide

## 🚀 Production Deployment Checklist

### Pre-Deployment Requirements

#### System Requirements
- **CPU**: 4+ cores (8+ recommended for high load)
- **Memory**: 8GB+ RAM (16GB+ recommended)
- **Storage**: 100GB+ SSD storage
- **Network**: 1Gbps+ network connection
- **Python**: 3.11+ with async support

#### Dependencies
```bash
# Core APG dependencies
- auth (Authentication & Authorization)
- audl (Audit & Logging)
- mten (Multi-tenancy)
- moni (Monitoring)
- conf (Configuration Management)

# Optional dependencies
- aicr (AI & Core Reasoning)
- pred (Predictive Analytics)
- anom (Anomaly Detection)
- agnt (Agent Framework)
```

### 1. Environment Setup

#### Production Environment Variables
```bash
# Core Configuration
export CACHE_SIZE_MB=4096
export CACHE_MAX_ENTRIES=5000000
export CACHE_DEFAULT_TTL=3600
export CACHE_CLEANUP_INTERVAL=300

# AI Features
export AI_OPTIMIZATION_ENABLED=true
export PREDICTIVE_CACHING_ENABLED=true
export INTELLIGENT_WARMING_ENABLED=true
export OPTIMIZATION_INTERVAL=600

# Security
export CACHE_SECURITY_LEVEL=HIGH
export ENCRYPTION_ENABLED=true
export QUANTUM_SECURITY_ENABLED=true
export SECURE_DELETION_ENABLED=true

# Multi-Tier Configuration
export MULTI_TIER_ENABLED=true
export L1_SIZE_MB=1024
export L2_SIZE_MB=2048
export L3_SIZE_MB=8192
export EDGE_SIZE_MB=512

# Redis Configuration (for L2 tier)
export REDIS_URL=redis://redis-cluster:6379
export REDIS_PASSWORD=your_secure_password
export REDIS_CLUSTER_ENABLED=true

# Monitoring
export MONITORING_ENABLED=true
export METRICS_EXPORT_INTERVAL=60
export HEALTH_CHECK_INTERVAL=30
export PERFORMANCE_HISTORY_RETENTION_DAYS=30

# Logging
export LOG_LEVEL=INFO
export LOG_FORMAT=json
export AUDIT_LOGGING_ENABLED=true
```

### 2. Infrastructure Setup

#### Docker Configuration
```dockerfile
# Dockerfile.production
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 apgcache && chown -R apgcache:apgcache /app
USER apgcache

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/api/cache/health')"

# Expose port
EXPOSE 8080

# Start the service
CMD ["python", "-m", "capabilities.common.cach.service"]
```

#### Docker Compose Configuration
```yaml
# docker-compose.production.yml
version: '3.8'

services:
  apg-cache:
    build:
      context: .
      dockerfile: Dockerfile.production
    ports:
      - "8080:8080"
    environment:
      - CACHE_SIZE_MB=4096
      - REDIS_URL=redis://redis:6379
      - AI_OPTIMIZATION_ENABLED=true
    volumes:
      - cache-data:/app/data
      - ./logs:/app/logs
    depends_on:
      - redis
      - monitoring
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
        reservations:
          memory: 4G
          cpus: '2'

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes --maxmemory 2gb --maxmemory-policy allkeys-lru
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
      - apg-cache
    restart: unless-stopped

volumes:
  cache-data:
  redis-data:
```

### 3. Kubernetes Deployment

#### Namespace Configuration
```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: apg-cache
  labels:
    name: apg-cache
    tier: data-layer
```

#### ConfigMap
```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: apg-cache-config
  namespace: apg-cache
data:
  CACHE_SIZE_MB: "4096"
  AI_OPTIMIZATION_ENABLED: "true"
  PREDICTIVE_CACHING_ENABLED: "true"
  MULTI_TIER_ENABLED: "true"
  MONITORING_ENABLED: "true"
  LOG_LEVEL: "INFO"
```

#### Deployment
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: apg-cache
  namespace: apg-cache
  labels:
    app: apg-cache
spec:
  replicas: 3
  selector:
    matchLabels:
      app: apg-cache
  template:
    metadata:
      labels:
        app: apg-cache
    spec:
      containers:
      - name: apg-cache
        image: apg-cache:1.0.0
        ports:
        - containerPort: 8080
        envFrom:
        - configMapRef:
            name: apg-cache-config
        - secretRef:
            name: apg-cache-secrets
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
        livenessProbe:
          httpGet:
            path: /api/cache/health
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /api/cache/health
            port: 8080
          initialDelaySeconds: 15
          periodSeconds: 10
        volumeMounts:
        - name: cache-data
          mountPath: /app/data
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: cache-data
        persistentVolumeClaim:
          claimName: apg-cache-pvc
      - name: logs
        emptyDir: {}
```

#### Service
```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: apg-cache-service
  namespace: apg-cache
  labels:
    app: apg-cache
spec:
  selector:
    app: apg-cache
  ports:
  - name: http
    port: 80
    targetPort: 8080
  - name: metrics
    port: 9090
    targetPort: 9090
  type: ClusterIP
```

### 4. Load Balancer Configuration

#### Nginx Configuration
```nginx
# nginx.conf
upstream apg_cache_backend {
    least_conn;
    server apg-cache-1:8080 max_fails=3 fail_timeout=30s;
    server apg-cache-2:8080 max_fails=3 fail_timeout=30s;
    server apg-cache-3:8080 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name cache.apg.example.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name cache.apg.example.com;

    # SSL Configuration
    ssl_certificate /etc/nginx/ssl/cache.apg.example.com.crt;
    ssl_certificate_key /etc/nginx/ssl/cache.apg.example.com.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;

    # Cache API
    location /api/cache/ {
        proxy_pass http://apg_cache_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeout settings
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
        
        # Health check
        proxy_next_upstream error timeout http_502 http_503 http_504;
    }

    # Dashboard
    location /cache/ {
        proxy_pass http://apg_cache_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Health check endpoint
    location /health {
        proxy_pass http://apg_cache_backend/api/cache/health;
        access_log off;
    }
}
```

### 5. Monitoring Setup

#### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "apg_cache_rules.yml"

scrape_configs:
  - job_name: 'apg-cache'
    static_configs:
      - targets: ['apg-cache:9090']
    scrape_interval: 30s
    metrics_path: /metrics
    
alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

#### Grafana Dashboard
```json
{
  "dashboard": {
    "id": null,
    "title": "APG Cache Management",
    "panels": [
      {
        "title": "Hit Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "apg_cache_hit_rate"
          }
        ]
      },
      {
        "title": "Operations per Second",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(apg_cache_operations_total[5m])"
          }
        ]
      },
      {
        "title": "Latency Percentiles",
        "type": "graph",
        "targets": [
          {
            "expr": "apg_cache_latency_p50"
          },
          {
            "expr": "apg_cache_latency_p95"
          },
          {
            "expr": "apg_cache_latency_p99"
          }
        ]
      }
    ]
  }
}
```

### 6. Security Hardening

#### SSL/TLS Configuration
```bash
# Generate SSL certificate
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout cache.apg.example.com.key \
  -out cache.apg.example.com.crt \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=cache.apg.example.com"

# Set appropriate permissions
chmod 600 cache.apg.example.com.key
chmod 644 cache.apg.example.com.crt
```

#### Firewall Configuration
```bash
# Allow HTTP/HTTPS
ufw allow 80/tcp
ufw allow 443/tcp

# Allow cache service port (internal only)
ufw allow from 10.0.0.0/8 to any port 8080

# Allow Redis (internal only)
ufw allow from 10.0.0.0/8 to any port 6379

# Allow monitoring (internal only)
ufw allow from 10.0.0.0/8 to any port 9090
```

#### Secrets Management
```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: apg-cache-secrets
  namespace: apg-cache
type: Opaque
stringData:
  REDIS_PASSWORD: "your_secure_redis_password"
  ENCRYPTION_KEY: "your_32_byte_encryption_key"
  JWT_SECRET: "your_jwt_signing_secret"
  DATABASE_URL: "postgresql://user:pass@db:5432/apg_cache"
```

### 7. Performance Tuning

#### Operating System Tuning
```bash
# Network tuning
echo 'net.core.somaxconn = 65535' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_max_syn_backlog = 65535' >> /etc/sysctl.conf
echo 'net.core.netdev_max_backlog = 5000' >> /etc/sysctl.conf

# Memory tuning
echo 'vm.swappiness = 10' >> /etc/sysctl.conf
echo 'vm.dirty_ratio = 15' >> /etc/sysctl.conf
echo 'vm.dirty_background_ratio = 5' >> /etc/sysctl.conf

# File descriptor limits
echo '* soft nofile 65535' >> /etc/security/limits.conf
echo '* hard nofile 65535' >> /etc/security/limits.conf

# Apply changes
sysctl -p
```

#### Redis Tuning (for L2 tier)
```bash
# redis.conf optimizations
maxmemory 4gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
tcp-keepalive 300
timeout 0
tcp-backlog 511
```

### 8. Backup and Recovery

#### Backup Strategy
```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/apg-cache"
DATE=$(date +"%Y%m%d_%H%M%S")

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup cache configuration
kubectl get configmaps -n apg-cache -o yaml > $BACKUP_DIR/configmaps_$DATE.yaml
kubectl get secrets -n apg-cache -o yaml > $BACKUP_DIR/secrets_$DATE.yaml

# Backup cache data (if persistent)
tar -czf $BACKUP_DIR/cache_data_$DATE.tar.gz /app/data

# Backup Redis data
redis-cli --rdb $BACKUP_DIR/redis_dump_$DATE.rdb

# Remove old backups (keep last 7 days)
find $BACKUP_DIR -name "*.yaml" -mtime +7 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
find $BACKUP_DIR -name "*.rdb" -mtime +7 -delete

echo "Backup completed: $DATE"
```

#### Recovery Procedure
```bash
#!/bin/bash
# recovery.sh

BACKUP_DIR="/backups/apg-cache"
RESTORE_DATE="$1"

if [ -z "$RESTORE_DATE" ]; then
    echo "Usage: $0 <YYYYMMDD_HHMMSS>"
    exit 1
fi

# Restore configuration
kubectl apply -f $BACKUP_DIR/configmaps_$RESTORE_DATE.yaml
kubectl apply -f $BACKUP_DIR/secrets_$RESTORE_DATE.yaml

# Restore cache data
tar -xzf $BACKUP_DIR/cache_data_$RESTORE_DATE.tar.gz -C /

# Restore Redis data
redis-cli --pipe < $BACKUP_DIR/redis_dump_$RESTORE_DATE.rdb

# Restart services
kubectl rollout restart deployment/apg-cache -n apg-cache

echo "Recovery completed for backup: $RESTORE_DATE"
```

### 9. Monitoring and Alerting

#### Health Checks
```python
# health_check.py
import asyncio
import aiohttp
import sys

async def check_cache_health():
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8080/api/cache/health') as resp:
                if resp.status == 200:
                    health_data = await resp.json()
                    if health_data.get('healthy'):
                        print("✅ Cache service is healthy")
                        return True
                    else:
                        print("❌ Cache service reports unhealthy")
                        return False
                else:
                    print(f"❌ Health check failed with status: {resp.status}")
                    return False
    except Exception as e:
        print(f"❌ Health check failed with error: {e}")
        return False

if __name__ == "__main__":
    healthy = asyncio.run(check_cache_health())
    sys.exit(0 if healthy else 1)
```

#### Alert Rules
```yaml
# apg_cache_rules.yml
groups:
- name: apg_cache_alerts
  rules:
  - alert: CacheHighLatency
    expr: apg_cache_latency_p95 > 100
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Cache latency is high"
      description: "P95 latency is {{ $value }}ms"

  - alert: CacheLowHitRate
    expr: apg_cache_hit_rate < 0.8
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "Cache hit rate is low"
      description: "Hit rate is {{ $value }}"

  - alert: CacheServiceDown
    expr: up{job="apg-cache"} == 0
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "Cache service is down"
      description: "Cache service has been down for more than 2 minutes"
```

### 10. Deployment Automation

#### CI/CD Pipeline
```yaml
# .github/workflows/deploy.yml
name: Deploy APG Cache Management

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install uv
        uv sync
    
    - name: Run tests
      run: |
        uv run pytest tests/ci/ -v
        uv run pyright
    
    - name: Security scan
      run: |
        uv run bandit -r capabilities/common/cach/

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: |
        docker build -t apg-cache:${{ github.sha }} -f Dockerfile.production .
    
    - name: Push to registry
      run: |
        docker push apg-cache:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - name: Deploy to Kubernetes
      run: |
        kubectl set image deployment/apg-cache apg-cache=apg-cache:${{ github.sha }} -n apg-cache
        kubectl rollout status deployment/apg-cache -n apg-cache
```

### 11. Post-Deployment Verification

#### Smoke Tests
```bash
#!/bin/bash
# smoke_test.sh

API_URL="https://cache.apg.example.com"

echo "🧪 Running smoke tests..."

# Test 1: Health check
echo "1. Health check..."
response=$(curl -s -o /dev/null -w "%{http_code}" $API_URL/api/cache/health)
if [ $response -eq 200 ]; then
    echo "✅ Health check passed"
else
    echo "❌ Health check failed (HTTP $response)"
    exit 1
fi

# Test 2: Basic cache operations
echo "2. Basic cache operations..."
curl -X POST $API_URL/api/cache/set \
  -H "Content-Type: application/json" \
  -d '{"key":"test:smoke","value":"smoke_test_value"}'

value=$(curl -s $API_URL/api/cache/get/test:smoke | jq -r '.value')
if [ "$value" = "smoke_test_value" ]; then
    echo "✅ Basic cache operations passed"
else
    echo "❌ Basic cache operations failed"
    exit 1
fi

# Test 3: Performance check
echo "3. Performance check..."
start_time=$(date +%s%N)
curl -s $API_URL/api/cache/stats > /dev/null
end_time=$(date +%s%N)
duration=$((($end_time - $start_time) / 1000000))

if [ $duration -lt 1000 ]; then
    echo "✅ Performance check passed ($duration ms)"
else
    echo "❌ Performance check failed ($duration ms)"
    exit 1
fi

echo "🎉 All smoke tests passed!"
```

### 12. Troubleshooting Guide

#### Common Issues

**Issue**: High memory usage
```bash
# Check memory statistics
kubectl top pods -n apg-cache

# Check cache memory usage
curl https://cache.apg.example.com/api/cache/stats | jq '.memory_usage_mb'

# Trigger cache cleanup
curl -X POST https://cache.apg.example.com/api/cache/cleanup
```

**Issue**: Low hit rate
```bash
# Get AI insights
curl https://cache.apg.example.com/api/cache/ai-insights | jq '.optimization_opportunities'

# Check access patterns
curl https://cache.apg.example.com/api/cache/analytics | jq '.access_patterns'

# Trigger optimization
curl -X POST https://cache.apg.example.com/api/cache/optimize
```

**Issue**: High latency
```bash
# Check tier distribution
curl https://cache.apg.example.com/api/cache/hierarchy-stats

# Check Redis connectivity (L2 tier)
redis-cli -h redis-cluster ping

# Check network latency
ping cache.apg.example.com
```

### 13. Scaling Considerations

#### Horizontal Scaling
```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: apg-cache-hpa
  namespace: apg-cache
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: apg-cache
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
```

#### Vertical Scaling
```bash
# Increase resource limits
kubectl patch deployment apg-cache -n apg-cache -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [
          {
            "name": "apg-cache",
            "resources": {
              "limits": {
                "memory": "16Gi",
                "cpu": "8000m"
              },
              "requests": {
                "memory": "8Gi",
                "cpu": "4000m"
              }
            }
          }
        ]
      }
    }
  }
}'
```

## ✅ Production Readiness Checklist

- [ ] **Environment Setup**: All environment variables configured
- [ ] **Infrastructure**: Docker/Kubernetes configuration deployed
- [ ] **Security**: SSL certificates installed and firewall configured
- [ ] **Monitoring**: Prometheus, Grafana, and alerting configured
- [ ] **Backup**: Backup strategy implemented and tested
- [ ] **Load Testing**: Performance validated under expected load
- [ ] **Health Checks**: Health checks and readiness probes configured
- [ ] **Documentation**: Runbooks and troubleshooting guides created
- [ ] **CI/CD**: Automated deployment pipeline configured
- [ ] **Smoke Tests**: Post-deployment verification automated

## 📞 Support

For production deployment support:
- **Documentation**: Internal deployment guides
- **Monitoring**: Production monitoring dashboards
- **Alerts**: Critical issue alerting configured
- **Support Team**: 24/7 production support available

---

**APG Cache Management Production Deployment** - *Enterprise-grade deployment for mission-critical applications.*