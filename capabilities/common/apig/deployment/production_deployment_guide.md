# APIG Production Deployment Guide

**Version:** 1.0.0  
**Date:** August 11, 2025  
**Status:** Production Ready ✅  
**Team:** APG Platform Team  

## Quick Start

### Prerequisites
- Python 3.11+ with async support
- APG Platform services (auth, monitoring, config, AI orchestration, message queue, audit)
- Ollama server for AI features (optional - fallback available)
- Redis for intelligent caching (optional - in-memory fallback)

### Installation
```bash
# Clone repository
git clone <repository-url>
cd apig

# Install dependencies
pip install -r requirements.txt

# Verify installation
python tests/test_minimal_integration.py
```

### Basic Configuration
```python
# config/production.yaml
apig:
  environment: production
  listen_port: 8080
  max_connections: 10000
  
apg_services:
  auth_service_url: "https://auth.apg.datacraft.co.ke"
  monitoring_service_url: "https://monitoring.apg.datacraft.co.ke"
  config_service_url: "https://config.apg.datacraft.co.ke"
  api_key: "${APG_API_KEY}"
  
ollama:
  base_url: "http://localhost:11434"
  default_model: "llama3.2:latest"
  timeout: 60
  
redis:
  url: "redis://localhost:6379"
  max_connections: 100
```

### Start Service
```bash
python -m service_production --config config/production.yaml
```

## Architecture Overview

### Core Components
```
┌─────────────────────────────────────┐
│           APIG Gateway              │
├─────────────────────────────────────┤
│  Control Plane (Natural Language)  │
├─────────────────────────────────────┤
│     Edge Engine (AI Processing)    │
├─────────────────────────────────────┤
│   WASM Runtime | Ollama Client     │
├─────────────────────────────────────┤
│        APG Service Clients         │
└─────────────────────────────────────┘
```

### Data Flow
1. **HTTP Request** → APIG Gateway
2. **Security Analysis** → Threat detection
3. **Cache Check** → Intelligent caching layer
4. **Policy Evaluation** → Rate limiting, auth
5. **WASM Processing** → Edge computing
6. **Upstream Routing** → Load balancing
7. **Response Processing** → Transform & cache

## Production Configuration

### Environment Variables
```bash
# APG Platform
export APG_API_KEY="your-production-api-key"
export APG_TENANT_ID="your-tenant-id"
export APG_AUTH_SERVICE_URL="https://auth.apg.datacraft.co.ke"
export APG_MONITORING_SERVICE_URL="https://monitoring.apg.datacraft.co.ke"
export APG_CONFIG_SERVICE_URL="https://config.apg.datacraft.co.ke"

# AI Features
export OLLAMA_URL="http://ollama-server:11434"
export OLLAMA_MODEL="llama3.2:latest"
export OLLAMA_TIMEOUT="60"

# Caching
export REDIS_URL="redis://redis-cluster:6379"
export CACHE_TTL_SECONDS="300"

# Performance
export MAX_CONNECTIONS="10000"
export WORKER_PROCESSES="4"
export CONNECTION_TIMEOUT="30000"
```

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

EXPOSE 8080
CMD ["python", "-m", "service_production"]
```

### Docker Compose
```yaml
version: '3.8'
services:
  apig:
    build: .
    ports:
      - "8080:8080"
    environment:
      - APG_API_KEY=${APG_API_KEY}
      - APG_TENANT_ID=${APG_TENANT_ID}
      - OLLAMA_URL=http://ollama:11434
      - REDIS_URL=redis://redis:6379
    depends_on:
      - ollama
      - redis
      
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
      
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  ollama_data:
  redis_data:
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: apig-gateway
spec:
  replicas: 3
  selector:
    matchLabels:
      app: apig-gateway
  template:
    metadata:
      labels:
        app: apig-gateway
    spec:
      containers:
      - name: apig
        image: apig:latest
        ports:
        - containerPort: 8080
        env:
        - name: APG_API_KEY
          valueFrom:
            secretKeyRef:
              name: apig-secrets
              key: api-key
        - name: OLLAMA_URL
          value: "http://ollama-service:11434"
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: apig-service
spec:
  selector:
    app: apig-gateway
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
```

## Security Configuration

### TLS/SSL Setup
```python
# config/tls.yaml
tls:
  enabled: true
  certificate_path: "/etc/ssl/certs/apig.crt"
  private_key_path: "/etc/ssl/private/apig.key"
  min_version: "TLSv1.2"
  cipher_suites:
    - "TLS_AES_256_GCM_SHA384"
    - "TLS_CHACHA20_POLY1305_SHA256"
```

### Security Policies
```python
# config/security.yaml
security:
  threat_detection:
    enabled: true
    confidence_threshold: 0.8
    block_high_threats: true
    rate_limit_medium_threats: true
    
  ip_filtering:
    enabled: true
    whitelist: []
    blacklist: ["192.168.0.0/16"]
    
  rate_limiting:
    global_limit: 10000  # requests per minute
    per_ip_limit: 1000   # requests per minute
    burst_size: 100
```

### Authentication Setup
```python
# config/auth.yaml  
authentication:
  methods:
    - jwt
    - api_key
    - oauth2
    
  jwt:
    secret_key: "${JWT_SECRET_KEY}"
    algorithm: "RS256"
    issuer: "apg-platform"
    
  api_key:
    header: "X-API-Key"
    required_for_admin: true
```

## Monitoring & Observability

### Health Endpoints
- `GET /health` - Service health status
- `GET /ready` - Readiness probe
- `GET /metrics` - Prometheus metrics
- `GET /status` - Detailed system status

### Metrics Collection
```python
# Prometheus metrics
apig_requests_total{method, status, route}
apig_request_duration_seconds{method, route}
apig_cache_hits_total{cache_type}
apig_cache_misses_total{cache_type}
apig_threats_detected_total{threat_level}
apig_policies_applied_total{policy_type}
apig_wasm_executions_total{status}
apig_ollama_requests_total{model, status}
```

### Logging Configuration
```python
# config/logging.yaml
logging:
  level: INFO
  format: json
  outputs:
    - console
    - file: /var/log/apig/gateway.log
    - syslog
    
  structured_logging:
    enabled: true
    fields:
      - timestamp
      - level
      - component
      - tenant_id
      - request_id
      - duration_ms
```

## Performance Tuning

### Connection Pool Settings
```python
# config/performance.yaml
performance:
  connection_pools:
    apg_services:
      max_connections: 100
      max_keepalive_connections: 20
      keepalive_expiry: 300
      
    ollama:
      max_connections: 10
      max_keepalive_connections: 5
      keepalive_expiry: 60
      
  timeouts:
    request: 30000      # 30 seconds
    connection: 5000    # 5 seconds  
    keepalive: 75000    # 75 seconds
```

### Caching Strategy
```python
# config/caching.yaml
caching:
  intelligent_cache:
    enabled: true
    max_memory_mb: 1024
    ttl_seconds: 300
    
  redis:
    enabled: true
    cluster_nodes:
      - "redis-1:6379"
      - "redis-2:6379"
      - "redis-3:6379"
    max_connections: 100
    
  cache_policies:
    - pattern: "/api/users/*"
      ttl: 60
      cache_responses: [200]
      
    - pattern: "/api/static/*"
      ttl: 3600
      cache_responses: [200, 304]
```

### WASM Configuration
```python
# config/wasm.yaml
wasm:
  runtime:
    max_modules: 100
    module_cache_size: 50
    execution_timeout_ms: 5000
    memory_limit_mb: 64
    
  security:
    fuel_limit: 1000000
    allowed_imports:
      - "env.console_log"
      - "env.http_request"
    
  modules:
    - name: "rate_limiter"
      path: "/opt/apig/wasm/rate_limiter.wasm"
      enabled: true
      
    - name: "auth_validator"
      path: "/opt/apig/wasm/auth_validator.wasm"
      enabled: true
```

## Scaling Configuration

### Horizontal Scaling
```yaml
# kubernetes/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: apig-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: apig-gateway
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

### Load Balancing
```nginx
# nginx/apig.conf
upstream apig_backend {
    least_conn;
    server apig-1:8080 max_fails=3 fail_timeout=30s;
    server apig-2:8080 max_fails=3 fail_timeout=30s;
    server apig-3:8080 max_fails=3 fail_timeout=30s;
}

server {
    listen 443 ssl http2;
    server_name api.datacraft.co.ke;
    
    ssl_certificate /etc/ssl/certs/datacraft.crt;
    ssl_certificate_key /etc/ssl/private/datacraft.key;
    
    location / {
        proxy_pass http://apig_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        proxy_connect_timeout 5s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
        
        proxy_buffering on;
        proxy_buffer_size 16k;
        proxy_buffers 8 16k;
    }
}
```

## Troubleshooting

### Common Issues

#### High Memory Usage
```bash
# Check memory usage
kubectl top pods -l app=apig-gateway

# Adjust memory limits
kubectl patch deployment apig-gateway -p '{"spec":{"template":{"spec":{"containers":[{"name":"apig","resources":{"limits":{"memory":"4Gi"}}}]}}}}'
```

#### Slow Response Times
```bash
# Check cache hit rate
curl http://apig-service/metrics | grep cache_hits

# Tune cache settings
kubectl edit configmap apig-config
```

#### AI Features Not Working
```bash
# Check Ollama connectivity
kubectl exec -it apig-pod -- curl http://ollama-service:11434/api/version

# Check model availability
kubectl exec -it ollama-pod -- ollama list
```

### Performance Monitoring
```python
# Monitor key metrics
import asyncio
from service_production import ProductionAPGIntelligentGatewayService

async def monitor_performance():
    service = ProductionAPGIntelligentGatewayService("prod-tenant")
    await service.initialize()
    
    metrics = await service.get_performance_metrics()
    print(f"Requests/sec: {metrics.requests_per_second}")
    print(f"Cache hit rate: {metrics.cache_hit_rate:.2%}")
    print(f"Average response time: {metrics.avg_response_time_ms}ms")
    print(f"Active connections: {metrics.active_connections}")

asyncio.run(monitor_performance())
```

### Health Checks
```bash
# Service health
curl http://apig-service/health

# Detailed status
curl http://apig-service/status | jq

# Component status
curl http://apig-service/status/components | jq
```

## Maintenance

### Updates & Patches
```bash
# Rolling update
kubectl set image deployment/apig-gateway apig=apig:v1.1.0

# Check rollout status  
kubectl rollout status deployment/apig-gateway

# Rollback if needed
kubectl rollout undo deployment/apig-gateway
```

### Backup & Recovery
```bash
# Backup configuration
kubectl get configmap apig-config -o yaml > apig-config-backup.yaml

# Backup policies
curl http://apig-service/api/policies > policies-backup.json

# Recovery
kubectl apply -f apig-config-backup.yaml
curl -X POST http://apig-service/api/policies -d @policies-backup.json
```

### Log Management
```bash
# View logs
kubectl logs -f deployment/apig-gateway

# Log rotation setup
logrotate -f /etc/logrotate.d/apig

# Centralized logging
kubectl apply -f logging/fluentd-apig.yaml
```

## Support

### Documentation
- API Reference: `/docs/api/`
- Architecture Guide: `/docs/architecture/`
- Performance Tuning: `/docs/performance/`

### Monitoring Dashboards
- Grafana: `https://monitoring.datacraft.co.ke/grafana`
- Prometheus: `https://monitoring.datacraft.co.ke/prometheus`
- APG Platform: `https://platform.apg.datacraft.co.ke`

### Contact
- **Email:** support@datacraft.co.ke
- **Slack:** #apig-support
- **Documentation:** https://docs.apg.datacraft.co.ke/apig

---

**Deployment Guide Version:** 1.0.0  
**Last Updated:** August 11, 2025  
**Status:** Production Ready ✅