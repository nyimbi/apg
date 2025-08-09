# Troubleshooting Guide

This guide provides solutions to common issues encountered when deploying and operating the Integration API Management capability.

## Common Issues

### 1. Service Startup Problems

#### Database Connection Failed

**Symptoms:**
- Service fails to start
- Error: "Could not connect to database"
- Connection timeout errors

**Diagnosis:**
```bash
# Check database connectivity
kubectl exec -it api-management-xxx -n integration-api-management -- nc -zv postgres-service 5432

# Check database credentials
kubectl get secret database-secret -n integration-api-management -o yaml

# Check database status
kubectl exec -it postgres-xxx -n integration-api-management -- pg_isready -U postgres
```

**Solutions:**

1. **Verify Database Configuration:**
```bash
# Check database service
kubectl get svc postgres-service -n integration-api-management

# Check database pods
kubectl get pods -l app=postgres -n integration-api-management

# Check database logs
kubectl logs postgres-xxx -n integration-api-management
```

2. **Fix Connection String:**
```yaml
# Correct database secret format
apiVersion: v1
kind: Secret
metadata:
  name: database-secret
data:
  connection-string: cG9zdGdyZXNxbDovL3VzZXI6cGFzc0Bob3N0OjU0MzIvZGI=
```

3. **Network Policies:**
```yaml
# Ensure network policy allows database access
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-database-access
spec:
  podSelector:
    matchLabels:
      app: api-management
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
```

#### Redis Connection Issues

**Symptoms:**
- Cache operations fail
- Rate limiting not working
- Session management errors

**Diagnosis:**
```bash
# Test Redis connectivity
kubectl exec -it api-management-xxx -n integration-api-management -- redis-cli -h redis-service ping

# Check Redis service
kubectl get svc redis-service -n integration-api-management

# Check Redis logs
kubectl logs redis-xxx -n integration-api-management
```

**Solutions:**

1. **Verify Redis Configuration:**
```toml
[redis]
host = "redis-service"
port = 6379
database = 0
password = "${REDIS_PASSWORD}"
ssl = false
```

2. **Check Redis Authentication:**
```bash
# Test with password
kubectl exec -it redis-xxx -n integration-api-management -- redis-cli -a "your_password" ping
```

3. **Resource Limits:**
```yaml
# Increase Redis memory if needed
resources:
  limits:
    memory: "2Gi"
  requests:
    memory: "1Gi"
```

### 2. Performance Issues

#### High Response Times

**Symptoms:**
- API responses taking >1 second
- Gateway timeouts
- High CPU usage

**Diagnosis:**
```bash
# Check resource usage
kubectl top pods -n integration-api-management

# Check metrics
curl http://api-management-service:8080/metrics | grep response_time

# Check logs for slow queries
kubectl logs api-management-xxx -n integration-api-management | grep "slow query"
```

**Solutions:**

1. **Increase Resources:**
```yaml
resources:
  requests:
    cpu: "2"
    memory: "4Gi"
  limits:
    cpu: "4"
    memory: "8Gi"
```

2. **Database Optimization:**
```sql
-- Add missing indexes
CREATE INDEX CONCURRENTLY idx_usage_records_tenant_timestamp 
ON am_usage_records (tenant_id, timestamp);

-- Analyze query performance
EXPLAIN ANALYZE SELECT * FROM am_apis WHERE tenant_id = 'xxx';
```

3. **Cache Configuration:**
```toml
[cache]
enabled = true
ttl_seconds = 300
max_size = 10000

[database]
pool_size = 20
max_overflow = 30
```

#### Memory Leaks

**Symptoms:**
- Memory usage continuously increasing
- Out of memory errors
- Pod restarts

**Diagnosis:**
```bash
# Monitor memory usage over time
kubectl top pod api-management-xxx -n integration-api-management --containers

# Check for memory leaks in logs
kubectl logs api-management-xxx -n integration-api-management | grep -i "memory\|oom"

# Generate memory dump (if Python profiling enabled)
kubectl exec -it api-management-xxx -n integration-api-management -- python -c "
import tracemalloc
tracemalloc.start()
# ... run some operations
current, peak = tracemalloc.get_traced_memory()
print(f'Current memory usage: {current / 1024 / 1024:.1f} MB')
print(f'Peak memory usage: {peak / 1024 / 1024:.1f} MB')
"
```

**Solutions:**

1. **Configure Memory Limits:**
```yaml
resources:
  limits:
    memory: "4Gi"
  requests:
    memory: "2Gi"
```

2. **Fix Connection Pool Leaks:**
```python
# Ensure proper connection cleanup
class DatabaseManager:
    async def __aenter__(self):
        self.session = self.Session()
        return self.session
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()
```

3. **Garbage Collection Tuning:**
```python
import gc
import os

# Force garbage collection periodically
if os.getenv('FORCE_GC', 'false').lower() == 'true':
    gc.collect()
```

### 3. Authentication and Authorization Issues

#### API Key Validation Failures

**Symptoms:**
- Valid API keys being rejected
- 401 Unauthorized errors
- Intermittent authentication failures

**Diagnosis:**
```bash
# Check API key in database
kubectl exec -it postgres-xxx -n integration-api-management -- psql -U postgres -d integration_api_management -c "
SELECT key_id, key_prefix, active, expires_at 
FROM am_api_keys 
WHERE key_prefix = 'ak_live_1234';
"

# Test authentication endpoint
curl -X GET "https://gateway.yourcompany.com/api/v1/test" \
  -H "X-API-Key: ak_live_1234567890abcdef1234567890abcdef" \
  -v
```

**Solutions:**

1. **Check Key Expiration:**
```sql
-- Update expired keys
UPDATE am_api_keys 
SET expires_at = '2026-12-31 23:59:59+00'
WHERE key_id = 'key_123' AND expires_at < NOW();
```

2. **Verify Tenant Context:**
```python
# Ensure tenant isolation is working
async def validate_api_key(api_key: str, tenant_id: str):
    key_hash = hash_api_key(api_key)
    key_record = await session.query(AMAPIKey).filter_by(
        key_hash=key_hash,
        active=True
    ).join(AMConsumer).filter_by(
        tenant_id=tenant_id  # Ensure tenant match
    ).first()
    return key_record
```

3. **Cache Invalidation:**
```bash
# Clear authentication cache
kubectl exec -it redis-xxx -n integration-api-management -- redis-cli FLUSHDB
```

#### Rate Limiting Issues

**Symptoms:**
- Rate limits not being enforced
- Requests blocked incorrectly
- Rate limit counters not resetting

**Diagnosis:**
```bash
# Check rate limit configuration
kubectl exec -it api-management-xxx -n integration-api-management -- python -c "
from integration_api_management.models import AMPolicy
# Query rate limiting policies
"

# Check Redis rate limit keys
kubectl exec -it redis-xxx -n integration-api-management -- redis-cli KEYS "rate_limit:*"
```

**Solutions:**

1. **Fix Rate Limit Logic:**
```python
# Correct sliding window implementation
async def check_rate_limit(key: str, limit: int, window: int) -> bool:
    now = time.time()
    pipe = redis.pipeline()
    
    # Remove expired entries
    pipe.zremrangebyscore(key, 0, now - window)
    
    # Count current requests
    pipe.zcard(key)
    
    # Add current request
    pipe.zadd(key, {str(now): now})
    pipe.expire(key, window)
    
    results = await pipe.execute()
    current_count = results[1]
    
    return current_count < limit
```

2. **Verify Policy Configuration:**
```json
{
  "policy_type": "rate_limiting",
  "config": {
    "requests_per_minute": 1000,
    "burst_size": 100,
    "key_extraction": "consumer_id",
    "window_type": "sliding"
  }
}
```

### 4. Gateway and Routing Issues

#### 404 Route Not Found

**Symptoms:**
- APIs return 404 errors
- Routes not being matched
- Gateway cannot find upstream services

**Diagnosis:**
```bash
# Check API registration
kubectl exec -it api-management-xxx -n integration-api-management -- python -c "
from integration_api_management.service import APILifecycleService
service = APILifecycleService()
apis = await service.get_apis_by_tenant('your_tenant_id')
for api in apis:
    print(f'{api.api_name}: {api.base_path} -> {api.upstream_url}')
"

# Test upstream connectivity
kubectl exec -it api-management-xxx -n integration-api-management -- curl -v http://upstream-service:8000/health
```

**Solutions:**

1. **Verify API Configuration:**
```python
# Ensure API is active and properly configured
api_config = {
    "api_name": "user_api",
    "base_path": "/api/users/v1",  # Must start with /
    "upstream_url": "http://user-service:8000",
    "status": "active"  # Must be active
}
```

2. **Check Route Matching:**
```python
# Debug route matching logic
def match_route(request_path: str, base_path: str) -> bool:
    # Remove trailing slashes for comparison
    request_path = request_path.rstrip('/')
    base_path = base_path.rstrip('/')
    
    return request_path.startswith(base_path)
```

3. **Update Route Cache:**
```bash
# Clear route cache
kubectl exec -it redis-xxx -n integration-api-management -- redis-cli DEL "route_cache:*"
```

#### Upstream Service Errors

**Symptoms:**
- 502 Bad Gateway errors
- Upstream timeouts
- Connection refused errors

**Diagnosis:**
```bash
# Check upstream service health
kubectl get pods -l app=upstream-service
kubectl logs upstream-service-xxx

# Test direct connectivity
kubectl exec -it api-management-xxx -n integration-api-management -- nc -zv upstream-service 8000

# Check service discovery
kubectl get endpoints upstream-service
```

**Solutions:**

1. **Health Check Configuration:**
```yaml
# Add proper health checks
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 5
```

2. **Circuit Breaker Configuration:**
```python
circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=30,
    expected_exception=UpstreamError
)
```

3. **Retry Logic:**
```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def call_upstream(url: str, data: dict):
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data) as response:
            return await response.json()
```

### 5. Database Issues

#### Slow Query Performance

**Symptoms:**
- Database queries taking >100ms
- High database CPU usage
- Lock contention

**Diagnosis:**
```sql
-- Check slow queries
SELECT query, mean_time, calls
FROM pg_stat_statements
WHERE mean_time > 100
ORDER BY mean_time DESC;

-- Check locks
SELECT * FROM pg_locks WHERE NOT granted;

-- Check database size
SELECT schemaname, tablename, 
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

**Solutions:**

1. **Add Missing Indexes:**
```sql
-- Common indexes for performance
CREATE INDEX CONCURRENTLY idx_usage_records_tenant_time 
ON am_usage_records (tenant_id, timestamp DESC);

CREATE INDEX CONCURRENTLY idx_api_keys_consumer_active 
ON am_api_keys (consumer_id) WHERE active = true;

CREATE INDEX CONCURRENTLY idx_apis_tenant_status 
ON am_apis (tenant_id, status);
```

2. **Optimize Queries:**
```python
# Use pagination for large result sets
async def get_usage_records(tenant_id: str, limit: int = 100, offset: int = 0):
    return await session.query(AMUsageRecord).filter_by(
        tenant_id=tenant_id
    ).order_by(
        AMUsageRecord.timestamp.desc()
    ).limit(limit).offset(offset).all()
```

3. **Connection Pool Tuning:**
```toml
[database]
pool_size = 20
max_overflow = 30
pool_timeout = 30
pool_recycle = 3600
```

#### Database Migration Issues

**Symptoms:**
- Migration failures
- Schema inconsistencies
- Data corruption

**Diagnosis:**
```bash
# Check migration status
kubectl exec -it api-management-xxx -n integration-api-management -- python -m alembic current

# Check migration history
kubectl exec -it api-management-xxx -n integration-api-management -- python -m alembic history

# Validate schema
kubectl exec -it postgres-xxx -n integration-api-management -- pg_dump --schema-only -U postgres integration_api_management
```

**Solutions:**

1. **Manual Migration Fix:**
```bash
# Mark specific migration as applied
kubectl exec -it api-management-xxx -n integration-api-management -- python -m alembic stamp head

# Run specific migration
kubectl exec -it api-management-xxx -n integration-api-management -- python -m alembic upgrade +1
```

2. **Backup Before Migration:**
```bash
# Create backup
kubectl exec -it postgres-xxx -n integration-api-management -- pg_dump -U postgres integration_api_management > backup.sql

# Restore if needed
kubectl exec -i postgres-xxx -n integration-api-management -- psql -U postgres integration_api_management < backup.sql
```

### 6. Monitoring and Alerting Issues

#### Missing Metrics

**Symptoms:**
- Prometheus not scraping metrics
- Missing dashboards in Grafana
- No alerts being triggered

**Diagnosis:**
```bash
# Check metrics endpoint
curl http://api-management-service:8080/metrics

# Check Prometheus targets
kubectl port-forward svc/prometheus 9090:9090
# Navigate to http://localhost:9090/targets

# Check ServiceMonitor
kubectl get servicemonitor -n integration-api-management
```

**Solutions:**

1. **Fix ServiceMonitor:**
```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: api-management-metrics
  namespace: integration-api-management
  labels:
    app: api-management
spec:
  selector:
    matchLabels:
      app: api-management
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
```

2. **Enable Metrics in Code:**
```python
from prometheus_client import Counter, Histogram, generate_latest

# Define metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'status'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'Request duration')

# Export metrics endpoint
async def metrics_handler(request):
    return web.Response(
        text=generate_latest(),
        content_type='text/plain'
    )
```

#### Alert Configuration Issues

**Symptoms:**
- Alerts not firing when they should
- Too many false positive alerts
- Missing alert notifications

**Diagnosis:**
```yaml
# Check PrometheusRule
kubectl get prometheusrule -n integration-api-management -o yaml

# Check AlertManager config
kubectl get secret alertmanager-main -n monitoring -o yaml
```

**Solutions:**

1. **Fix Alert Rules:**
```yaml
groups:
- name: api-management
  rules:
  - alert: HighErrorRate
    expr: rate(api_requests_total{status=~"5.."}[5m]) / rate(api_requests_total[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "API error rate is high"
      description: "Error rate is {{ $value | humanizePercentage }} for the last 5 minutes"
```

2. **Configure Notifications:**
```yaml
# AlertManager configuration
route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'
receivers:
- name: 'web.hook'
  slack_configs:
  - api_url: 'YOUR_SLACK_WEBHOOK_URL'
    channel: '#alerts'
    title: 'API Management Alert'
```

## Debugging Commands

### Log Analysis

```bash
# Get recent logs
kubectl logs -f api-management-xxx -n integration-api-management --since=1h

# Search for errors
kubectl logs api-management-xxx -n integration-api-management | grep -i error

# Filter by request ID
kubectl logs api-management-xxx -n integration-api-management | grep "req_123456"

# Get logs from all pods
kubectl logs -l app=api-management -n integration-api-management --since=30m
```

### Database Debugging

```bash
# Connect to database
kubectl exec -it postgres-xxx -n integration-api-management -- psql -U postgres integration_api_management

# Check table sizes
SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) 
FROM pg_tables WHERE schemaname = 'public';

# Check active connections
SELECT count(*) FROM pg_stat_activity WHERE state = 'active';

# Check locks
SELECT * FROM pg_locks WHERE NOT granted;
```

### Redis Debugging

```bash
# Connect to Redis
kubectl exec -it redis-xxx -n integration-api-management -- redis-cli

# Check memory usage
kubectl exec -it redis-xxx -n integration-api-management -- redis-cli INFO memory

# List all keys
kubectl exec -it redis-xxx -n integration-api-management -- redis-cli KEYS "*"

# Check rate limit keys
kubectl exec -it redis-xxx -n integration-api-management -- redis-cli KEYS "rate_limit:*"
```

### Network Debugging

```bash
# Test connectivity between pods
kubectl exec -it api-management-xxx -n integration-api-management -- nc -zv postgres-service 5432

# Check DNS resolution
kubectl exec -it api-management-xxx -n integration-api-management -- nslookup postgres-service

# Check network policies
kubectl get networkpolicy -n integration-api-management

# Port forward for testing
kubectl port-forward svc/api-management-service 8080:8080 -n integration-api-management
```

## Emergency Procedures

### Service Recovery

1. **Rolling Restart:**
```bash
kubectl rollout restart deployment/api-management -n integration-api-management
kubectl rollout status deployment/api-management -n integration-api-management
```

2. **Scale Down/Up:**
```bash
kubectl scale deployment api-management --replicas=0 -n integration-api-management
kubectl scale deployment api-management --replicas=3 -n integration-api-management
```

3. **Emergency Rollback:**
```bash
kubectl rollout undo deployment/api-management -n integration-api-management
kubectl rollout history deployment/api-management -n integration-api-management
```

### Database Recovery

1. **Connection Pool Reset:**
```bash
kubectl exec -it api-management-xxx -n integration-api-management -- python -c "
from integration_api_management.database import engine
engine.dispose()
"
```

2. **Database Restart:**
```bash
kubectl delete pod postgres-xxx -n integration-api-management
kubectl wait --for=condition=ready pod -l app=postgres -n integration-api-management
```

### Cache Recovery

1. **Clear All Cache:**
```bash
kubectl exec -it redis-xxx -n integration-api-management -- redis-cli FLUSHALL
```

2. **Clear Specific Cache:**
```bash
kubectl exec -it redis-xxx -n integration-api-management -- redis-cli DEL "cache:api:*"
```

## Performance Tuning

### Application Tuning

```toml
# Optimized configuration
[gateway]
workers = 4
max_connections = 10000
keep_alive_timeout = 75

[database]
pool_size = 20
max_overflow = 30
pool_timeout = 30

[redis]
connection_pool_size = 20
```

### Database Tuning

```sql
-- PostgreSQL optimization
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '12GB';
ALTER SYSTEM SET maintenance_work_mem = '1GB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
SELECT pg_reload_conf();
```

### Kubernetes Resource Optimization

```yaml
resources:
  requests:
    cpu: "2"
    memory: "4Gi"
  limits:
    cpu: "4"
    memory: "8Gi"

# HPA configuration
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-management-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-management
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Contact Support

When contacting support, please include:

1. **Environment Information:**
   - Kubernetes version
   - Application version
   - Configuration files

2. **Error Details:**
   - Error messages
   - Stack traces
   - Request IDs

3. **Diagnostic Information:**
   - Logs from last 30 minutes
   - Resource usage metrics
   - Network connectivity tests

**Support Channels:**
- **Emergency**: [emergency@datacraft.co.ke](mailto:emergency@datacraft.co.ke)
- **Technical Support**: [support@datacraft.co.ke](mailto:support@datacraft.co.ke)
- **Documentation**: [https://docs.datacraft.co.ke](https://docs.datacraft.co.ke)
- **Status Page**: [https://status.datacraft.co.ke](https://status.datacraft.co.ke)