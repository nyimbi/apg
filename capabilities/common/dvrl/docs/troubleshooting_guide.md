# APG Data Virtualization (DVRL) Troubleshooting Guide

## Table of Contents
1. [Quick Diagnostics](#quick-diagnostics)
2. [Common Issues and Solutions](#common-issues-and-solutions)
3. [Performance Issues](#performance-issues)
4. [Data Source Connectivity](#data-source-connectivity)
5. [Query Execution Problems](#query-execution-problems)
6. [APG Integration Issues](#apg-integration-issues)
7. [Security and Authentication](#security-and-authentication)
8. [Caching Problems](#caching-problems)
9. [Monitoring and Alerting](#monitoring-and-alerting)
10. [Advanced Diagnostics](#advanced-diagnostics)
11. [Recovery Procedures](#recovery-procedures)
12. [Getting Support](#getting-support)

## Quick Diagnostics

### Health Check Commands
```bash
# Quick system health check
curl -k https://dvrl.apg.yourcompany.com/health | jq

# Check pod status
kubectl get pods -n apg-dvrl -o wide

# View recent logs
kubectl logs -n apg-dvrl deployment/dvrl-deployment --tail=100
```

### Component Status Check
```bash
# Check APG platform integration
curl -H "Authorization: Bearer ${APG_TOKEN}" \
  ${APG_BASE_URL}/api/v1/capabilities/status | jq '.dvrl'

# Database connectivity
python3 -c "
import asyncio, asyncpg, os
async def check():
    try:
        conn = await asyncpg.connect(os.environ['DATABASE_URL'])
        print('✓ Database: Connected')
        await conn.close()
    except Exception as e:
        print(f'✗ Database: {e}')
asyncio.run(check())
"

# Cache connectivity  
redis-cli -h ${REDIS_HOST} -p ${REDIS_PORT} ping
```

### Performance Quick Check
```bash
# Query performance metrics
curl -H "Authorization: Bearer ${APG_TOKEN}" \
  https://dvrl.apg.yourcompany.com/api/v1/metrics | jq '.query_performance'

# Resource usage
kubectl top pods -n apg-dvrl
```

## Common Issues and Solutions

### Issue: DVRL Service Not Starting

**Symptoms:**
- Pods in CrashLoopBackOff status
- "Connection refused" errors in logs
- Service unavailable responses

**Diagnostic Commands:**
```bash
# Check pod events
kubectl describe pod -n apg-dvrl -l app=dvrl

# Check resource limits
kubectl get pods -n apg-dvrl -o jsonpath='{.items[*].spec.containers[*].resources}'

# View detailed logs
kubectl logs -n apg-dvrl -l app=dvrl --previous
```

**Common Causes and Solutions:**

1. **Insufficient Resources**
   ```yaml
   # Fix: Increase resource limits in deployment
   resources:
     requests:
       memory: "4Gi"
       cpu: "2"
     limits:
       memory: "8Gi" 
       cpu: "4"
   ```

2. **Database Connection Issues**
   ```bash
   # Fix: Verify database credentials
   kubectl get secret dvrl-secrets -n apg-dvrl -o yaml
   
   # Test connection manually
   psql -h ${DB_HOST} -U ${DB_USER} -d ${DB_DATABASE}
   ```

3. **Missing Environment Variables**
   ```bash
   # Fix: Check required environment variables
   kubectl describe deployment dvrl-deployment -n apg-dvrl | grep -A 20 Environment
   ```

### Issue: Authentication Failures

**Symptoms:**
- "401 Unauthorized" responses
- "Token validation failed" in logs
- Users unable to access DVRL interface

**Diagnostic Commands:**
```bash
# Verify token structure
python3 -c "
import jwt, json, sys
token = sys.argv[1]
try:
    decoded = jwt.decode(token, options={'verify_signature': False})
    print(json.dumps(decoded, indent=2))
except Exception as e:
    print(f'Token error: {e}')
" "${APG_TOKEN}"

# Test APG auth service
curl -v ${APG_BASE_URL}/api/v1/auth/validate \
  -H "Authorization: Bearer ${APG_TOKEN}"
```

**Solutions:**

1. **Token Expiry**
   ```bash
   # Generate new token
   APG_TOKEN=$(curl -X POST ${APG_BASE_URL}/api/v1/auth/token \
     -H "Content-Type: application/json" \
     -d '{"username":"${USERNAME}", "password":"${PASSWORD}"}' | jq -r '.access_token')
   ```

2. **APG Auth Service Issues**
   ```bash
   # Check APG auth service status
   kubectl get pods -n apg-platform -l app=auth-service
   
   # Restart auth service if needed
   kubectl rollout restart deployment/auth-service -n apg-platform
   ```

### Issue: Data Source Registration Failures

**Symptoms:**
- "Unable to connect to data source" errors
- Registration API returns 500 errors
- Schema discovery fails

**Diagnostic Commands:**
```bash
# List data sources and their status
curl -H "Authorization: Bearer ${APG_TOKEN}" \
  https://dvrl.apg.yourcompany.com/api/v1/data-sources | jq '.data_sources[] | {name, status, error_message}'

# Check connector logs
kubectl logs -n apg-dvrl deployment/dvrl-deployment | grep -i connector
```

**Solutions:**

1. **Network Connectivity Issues**
   ```bash
   # Test connectivity from DVRL pod
   kubectl exec -n apg-dvrl deployment/dvrl-deployment -- \
     nc -zv ${DB_HOST} ${DB_PORT}
   
   # Check firewall rules
   telnet ${DB_HOST} ${DB_PORT}
   ```

2. **Credential Issues**
   ```bash
   # Test credentials manually
   mysql -h ${MYSQL_HOST} -P ${MYSQL_PORT} -u ${MYSQL_USER} -p${MYSQL_PASSWORD}
   
   # PostgreSQL test
   psql -h ${PG_HOST} -p ${PG_PORT} -U ${PG_USER} -d ${PG_DATABASE}
   ```

3. **SSL/TLS Configuration**
   ```bash
   # Test SSL connection
   openssl s_client -connect ${DB_HOST}:${DB_PORT} -servername ${DB_HOST}
   
   # Verify certificates
   curl -v --cacert ca.pem https://${DB_HOST}:${DB_PORT}
   ```

## Performance Issues

### Issue: Slow Query Execution

**Symptoms:**
- Queries taking longer than expected
- Timeout errors
- High resource usage during queries

**Diagnostic Commands:**
```bash
# Check query performance metrics
curl -H "Authorization: Bearer ${APG_TOKEN}" \
  https://dvrl.apg.yourcompany.com/api/v1/metrics | jq '.query_performance'

# Monitor resource usage
kubectl top pods -n apg-dvrl --containers

# View slow query logs
kubectl logs -n apg-dvrl deployment/dvrl-deployment | grep "execution_time_ms" | sort -k5 -nr
```

**Solutions:**

1. **Optimize Query Plans**
   ```sql
   -- Check execution plan
   EXPLAIN (ANALYZE, BUFFERS) 
   SELECT c.name, COUNT(o.id) 
   FROM customers c 
   LEFT JOIN orders o ON c.id = o.customer_id 
   GROUP BY c.id, c.name;
   ```

2. **Increase Connection Pool Size**
   ```yaml
   # Update configuration
   database:
     connection_pool_size: 50
     max_connections: 100
   ```

3. **Enable Query Caching**
   ```json
   {
     "sql": "SELECT * FROM products",
     "options": {
       "cache_strategy": "aggressive",
       "cache_ttl_seconds": 3600
     }
   }
   ```

### Issue: Memory Issues

**Symptoms:**
- OutOfMemory errors in logs
- Pods being killed by OOMKiller
- Performance degradation over time

**Diagnostic Commands:**
```bash
# Check memory usage
kubectl top pods -n apg-dvrl --sort-by=memory

# View memory-related events
kubectl get events -n apg-dvrl --sort-by='.lastTimestamp' | grep -i memory

# Check memory limits
kubectl describe pods -n apg-dvrl -l app=dvrl | grep -A 5 Limits
```

**Solutions:**

1. **Increase Memory Limits**
   ```yaml
   resources:
     limits:
       memory: "16Gi"  # Increase from current value
     requests:
       memory: "8Gi"   # Set appropriate request
   ```

2. **Optimize Caching Configuration**
   ```yaml
   cache:
     memory:
       max_size_mb: 2048  # Reduce if necessary
       eviction_policy: "lru"
   query_engine:
     max_result_rows: 100000  # Limit result set size
   ```

3. **Enable Memory Monitoring**
   ```yaml
   monitoring:
     memory_alerts:
       enabled: true
       warning_threshold: "80%"
       critical_threshold: "90%"
   ```

## Data Source Connectivity

### Issue: Connection Pool Exhaustion

**Symptoms:**
- "Connection pool exhausted" errors
- New connections failing
- Existing queries hanging

**Diagnostic Commands:**
```bash
# Check connection pool status
curl -H "Authorization: Bearer ${APG_TOKEN}" \
  https://dvrl.apg.yourcompany.com/api/v1/connector-details | jq '.connection_pools'

# Monitor database connections
# PostgreSQL
psql -h ${DB_HOST} -U ${DB_USER} -d postgres -c "SELECT count(*) FROM pg_stat_activity WHERE usename = '${DB_USER}';"

# MySQL  
mysql -h ${DB_HOST} -u ${DB_USER} -p -e "SHOW PROCESSLIST;" | grep ${DB_USER} | wc -l
```

**Solutions:**

1. **Increase Pool Size**
   ```yaml
   connectors:
     postgresql:
       pool_config:
         min_size: 10
         max_size: 50
         max_overflow: 20
   ```

2. **Optimize Connection Timeouts**
   ```yaml
   connectors:
     postgresql:
       connection_params:
         command_timeout: 30
         server_settings:
           tcp_keepalives_idle: "600"
           tcp_keepalives_interval: "30"
   ```

3. **Connection Cleanup**
   ```bash
   # Force connection pool reset
   curl -X POST -H "Authorization: Bearer ${APG_TOKEN}" \
     https://dvrl.apg.yourcompany.com/api/v1/admin/reset-connections
   ```

### Issue: Data Source Health Check Failures

**Symptoms:**
- Data sources showing as "unhealthy"
- Intermittent connection failures
- Health check timeouts

**Diagnostic Commands:**
```bash
# Check health status
curl -H "Authorization: Bearer ${APG_TOKEN}" \
  https://dvrl.apg.yourcompany.com/api/v1/data-sources/health | jq

# Test individual data source
curl -X POST -H "Authorization: Bearer ${APG_TOKEN}" \
  https://dvrl.apg.yourcompany.com/api/v1/data-sources/${DS_ID}/health-check
```

**Solutions:**

1. **Adjust Health Check Parameters**
   ```yaml
   health_checks:
     enabled: true
     interval_seconds: 60
     timeout_seconds: 10
     failure_threshold: 3
     success_threshold: 1
   ```

2. **Custom Health Check Queries**
   ```yaml
   data_sources:
     postgresql_prod:
       health_check_query: "SELECT 1"
     mysql_warehouse:
       health_check_query: "SELECT COUNT(*) FROM information_schema.tables LIMIT 1"
   ```

## Query Execution Problems

### Issue: SQL Parsing Errors

**Symptoms:**
- "Invalid SQL syntax" errors
- Queries failing to parse
- Unexpected parsing behavior

**Diagnostic Commands:**
```bash
# Test query parsing
curl -X POST -H "Authorization: Bearer ${APG_TOKEN}" \
  https://dvrl.apg.yourcompany.com/api/v1/queries/parse \
  -d '{"sql": "SELECT * FROM orders WHERE invalid_syntax"}'

# View parser logs
kubectl logs -n apg-dvrl deployment/dvrl-deployment | grep -i "parse_error"
```

**Solutions:**

1. **SQL Dialect Compatibility**
   ```sql
   -- PostgreSQL specific
   SELECT date_trunc('day', created_at) FROM orders;
   
   -- Generic SQL alternative
   SELECT DATE(created_at) FROM orders;
   ```

2. **Query Rewriting**
   ```python
   # Enable automatic query rewriting
   query_options = {
       "enable_rewriting": True,
       "target_dialect": "standard_sql"
   }
   ```

3. **Federation Syntax**
   ```sql
   -- Use explicit data source references
   SELECT o.*, c.name
   FROM postgres_db.orders o
   JOIN mysql_db.customers c ON o.customer_id = c.id;
   ```

### Issue: Join Performance Problems

**Symptoms:**
- Cross-database joins taking too long
- Memory issues during join operations
- Join operations failing

**Diagnostic Commands:**
```bash
# Analyze join execution plan
curl -X POST -H "Authorization: Bearer ${APG_TOKEN}" \
  https://dvrl.apg.yourcompany.com/api/v1/queries/explain \
  -d '{"sql": "SELECT * FROM table1 t1 JOIN table2 t2 ON t1.id = t2.id"}'

# Check join statistics
kubectl logs -n apg-dvrl deployment/dvrl-deployment | grep -i "join_execution"
```

**Solutions:**

1. **Optimize Join Order**
   ```sql
   -- Filter early to reduce data transfer
   SELECT o.*, c.name
   FROM (SELECT * FROM orders WHERE created_at >= '2024-01-01') o
   JOIN customers c ON o.customer_id = c.id;
   ```

2. **Use Broadcast Joins**
   ```json
   {
     "sql": "SELECT * FROM large_table l JOIN small_table s ON l.id = s.id",
     "options": {
       "join_strategy": "broadcast",
       "broadcast_threshold": 10000
     }
   }
   ```

3. **Create Virtual Tables**
   ```json
   {
     "name": "customer_orders_view",
     "federation_query": {
       "sql": "SELECT c.*, o.total FROM customers c LEFT JOIN orders o ON c.id = o.customer_id"
     },
     "materialization_strategy": "incremental"
   }
   ```

## APG Integration Issues

### Issue: APG Service Communication Failures

**Symptoms:**
- "Service unavailable" errors for APG capabilities
- Metadata sync failures
- Cache service errors

**Diagnostic Commands:**
```bash
# Test APG service endpoints
for service in auth meta cach moni; do
  echo "Testing ${service}:"
  curl -s -o /dev/null -w "%{http_code} %{time_total}s\n" \
    -H "Authorization: Bearer ${APG_TOKEN}" \
    ${APG_BASE_URL}/api/v1/${service}/health
done

# Check network policies
kubectl get networkpolicies -n apg-platform
```

**Solutions:**

1. **Service Discovery Issues**
   ```bash
   # Check DNS resolution
   kubectl exec -n apg-dvrl deployment/dvrl-deployment -- \
     nslookup auth-service.apg-platform.svc.cluster.local
   
   # Update service endpoints
   kubectl get endpoints -n apg-platform
   ```

2. **APG Service Restart**
   ```bash
   # Restart specific APG service
   kubectl rollout restart deployment/meta-service -n apg-platform
   
   # Check service status
   kubectl get pods -n apg-platform -l app=meta-service
   ```

### Issue: Metadata Synchronization Problems

**Symptoms:**
- Schema information not updating
- Missing table metadata
- Lineage tracking failures

**Diagnostic Commands:**
```bash
# Check metadata sync status
curl -H "Authorization: Bearer ${APG_TOKEN}" \
  https://dvrl.apg.yourcompany.com/api/v1/admin/metadata-sync-status

# View sync logs
kubectl logs -n apg-dvrl deployment/dvrl-deployment | grep -i metadata_sync
```

**Solutions:**

1. **Force Metadata Refresh**
   ```bash
   # Trigger manual sync
   curl -X POST -H "Authorization: Bearer ${APG_TOKEN}" \
     https://dvrl.apg.yourcompany.com/api/v1/admin/refresh-metadata
   ```

2. **Check Meta Service Configuration**
   ```yaml
   meta:
     endpoint: "${APG_BASE_URL}/api/v1/meta"
     sync_interval: 300  # 5 minutes
     batch_size: 100
     timeout_seconds: 30
   ```

## Security and Authentication

### Issue: Permission Denied Errors

**Symptoms:**
- "403 Forbidden" responses
- "Insufficient permissions" errors
- Users unable to access data sources

**Diagnostic Commands:**
```bash
# Check user permissions
curl -H "Authorization: Bearer ${APG_TOKEN}" \
  ${APG_BASE_URL}/api/v1/auth/user-permissions | jq '.permissions[] | select(.resource | startswith("dvrl"))'

# Verify RBAC configuration
kubectl get clusterrolebinding | grep dvrl
```

**Solutions:**

1. **Role Assignment**
   ```bash
   # Assign DVRL roles to user
   curl -X POST -H "Authorization: Bearer ${ADMIN_TOKEN}" \
     ${APG_BASE_URL}/api/v1/auth/users/${USER_ID}/roles \
     -d '{"roles": ["dvrl_analyst", "dvrl_user"]}'
   ```

2. **Data Source Permissions**
   ```json
   {
     "data_source_permissions": {
       "production_db": {
         "read": ["data_analysts", "managers"],
         "write": ["data_engineers"],
         "admin": ["database_admins"]
       }
     }
   }
   ```

### Issue: SSL/TLS Certificate Problems

**Symptoms:**
- Certificate validation errors
- "SSL handshake failed" messages
- Insecure connection warnings

**Diagnostic Commands:**
```bash
# Check certificate validity
openssl s_client -connect dvrl.apg.yourcompany.com:443 -servername dvrl.apg.yourcompany.com

# Verify certificate in Kubernetes
kubectl describe secret dvrl-tls-secret -n apg-dvrl
```

**Solutions:**

1. **Certificate Renewal**
   ```bash
   # Generate new certificate
   openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
     -keyout new-dvrl.key -out new-dvrl.crt \
     -subj "/CN=dvrl.apg.yourcompany.com"
   
   # Update Kubernetes secret
   kubectl create secret tls dvrl-tls-secret -n apg-dvrl \
     --cert=new-dvrl.crt --key=new-dvrl.key --dry-run=client -o yaml | kubectl apply -f -
   ```

2. **Certificate Chain Issues**
   ```bash
   # Include intermediate certificates
   cat dvrl.crt intermediate.crt root.crt > dvrl-chain.crt
   ```

## Caching Problems

### Issue: Low Cache Hit Rate

**Symptoms:**
- Cache hit ratio below 50%
- Repeated identical queries not cached
- Poor query performance despite caching

**Diagnostic Commands:**
```bash
# Check cache statistics
curl -H "Authorization: Bearer ${APG_TOKEN}" \
  https://dvrl.apg.yourcompany.com/api/v1/metrics | jq '.cache_performance'

# View cache contents
redis-cli -h ${REDIS_HOST} -p ${REDIS_PORT} --scan --pattern "dvrl:*" | head -10
```

**Solutions:**

1. **Cache Configuration Tuning**
   ```yaml
   cache:
     memory:
       max_size_mb: 4096
       ttl_seconds: 7200
     intelligent_caching:
       enabled: true
       similarity_threshold: 0.8
   ```

2. **Query Normalization**
   ```python
   # Enable query normalization for better cache hits
   query_options = {
       "normalize_queries": True,
       "cache_key_strategy": "semantic"
   }
   ```

### Issue: Cache Memory Issues

**Symptoms:**
- Cache evictions happening frequently
- Memory warnings in Redis logs
- Cache performance degradation

**Diagnostic Commands:**
```bash
# Check Redis memory usage
redis-cli -h ${REDIS_HOST} -p ${REDIS_PORT} INFO memory

# Monitor cache evictions
redis-cli -h ${REDIS_HOST} -p ${REDIS_PORT} INFO stats | grep evicted
```

**Solutions:**

1. **Increase Cache Memory**
   ```bash
   # Update Redis configuration
   redis-cli -h ${REDIS_HOST} -p ${REDIS_PORT} CONFIG SET maxmemory 8gb
   ```

2. **Optimize Eviction Policy**
   ```bash
   # Use LRU eviction for query results
   redis-cli -h ${REDIS_HOST} -p ${REDIS_PORT} CONFIG SET maxmemory-policy allkeys-lru
   ```

## Monitoring and Alerting

### Issue: Missing Metrics Data

**Symptoms:**
- Monitoring dashboard showing no data
- Metrics endpoint returning errors
- Alerting not triggering

**Diagnostic Commands:**
```bash
# Test metrics endpoint
curl -H "Authorization: Bearer ${APG_TOKEN}" \
  https://dvrl.apg.yourcompany.com/api/v1/metrics

# Check Prometheus scraping
kubectl logs -n monitoring prometheus-server-0 | grep dvrl
```

**Solutions:**

1. **Enable Metrics Collection**
   ```yaml
   monitoring:
     metrics:
       enabled: true
       port: 9090
       path: "/metrics"
       format: "prometheus"
   ```

2. **ServiceMonitor Configuration**
   ```yaml
   apiVersion: monitoring.coreos.com/v1
   kind: ServiceMonitor
   metadata:
     name: dvrl-metrics
     namespace: apg-dvrl
   spec:
     selector:
       matchLabels:
         app: dvrl
     endpoints:
     - port: metrics
       path: /metrics
   ```

## Advanced Diagnostics

### Debug Mode Activation

```bash
# Enable debug logging
kubectl set env deployment/dvrl-deployment -n apg-dvrl LOG_LEVEL=DEBUG

# Enable query tracing
curl -X POST -H "Authorization: Bearer ${APG_TOKEN}" \
  https://dvrl.apg.yourcompany.com/api/v1/admin/debug/enable-tracing

# Enable performance profiling
kubectl port-forward -n apg-dvrl deployment/dvrl-deployment 6060:6060
curl http://localhost:6060/debug/pprof/
```

### Database Query Analysis

```sql
-- PostgreSQL: Check slow queries
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;

-- Check connection count
SELECT count(*) as active_connections 
FROM pg_stat_activity 
WHERE state = 'active';

-- Check lock contention
SELECT blocked_locks.pid AS blocked_pid,
       blocked_activity.usename AS blocked_user,
       blocking_locks.pid AS blocking_pid,
       blocking_activity.usename AS blocking_user,
       blocked_activity.query AS blocked_statement
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks ON blocking_locks.locktype = blocked_locks.locktype
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.GRANTED;
```

### Network Analysis

```bash
# Test network latency to data sources
for host in postgres-db mysql-db mongodb-cluster; do
  echo "Testing $host:"
  kubectl exec -n apg-dvrl deployment/dvrl-deployment -- \
    ping -c 3 $host
done

# Check network policies
kubectl get networkpolicies -n apg-dvrl -o yaml

# Monitor network traffic
kubectl exec -n apg-dvrl deployment/dvrl-deployment -- \
  netstat -an | grep :5432
```

## Recovery Procedures

### Emergency Restart Procedure

```bash
# 1. Save current state
kubectl get all -n apg-dvrl -o yaml > dvrl-backup.yaml

# 2. Graceful restart
kubectl rollout restart deployment/dvrl-deployment -n apg-dvrl

# 3. Wait for rollout completion
kubectl rollout status deployment/dvrl-deployment -n apg-dvrl --timeout=300s

# 4. Verify health
curl -k https://dvrl.apg.yourcompany.com/health
```

### Database Recovery

```bash
# 1. Backup current database
pg_dump -h ${DB_HOST} -U ${DB_USER} dvrl_metadata > dvrl-emergency-backup.sql

# 2. Reset connection pools
curl -X POST -H "Authorization: Bearer ${APG_TOKEN}" \
  https://dvrl.apg.yourcompany.com/api/v1/admin/reset-connections

# 3. If needed, restore from backup
# psql -h ${DB_HOST} -U ${DB_USER} dvrl_metadata < dvrl-backup.sql
```

### Cache Recovery

```bash
# 1. Clear problematic cache entries
redis-cli -h ${REDIS_HOST} -p ${REDIS_PORT} FLUSHALL

# 2. Restart cache service
kubectl rollout restart deployment/redis -n apg-platform

# 3. Warm up cache with common queries
curl -X POST -H "Authorization: Bearer ${APG_TOKEN}" \
  https://dvrl.apg.yourcompany.com/api/v1/admin/warmup-cache
```

## Getting Support

### Information to Gather

When contacting support, please provide:

```bash
# System information
kubectl version
kubectl get pods -n apg-dvrl -o wide

# Application logs (last 100 lines)
kubectl logs -n apg-dvrl deployment/dvrl-deployment --tail=100

# Configuration
kubectl get configmap dvrl-config -n apg-dvrl -o yaml

# Resource usage
kubectl top pods -n apg-dvrl --containers

# Recent events
kubectl get events -n apg-dvrl --sort-by='.lastTimestamp' | tail -20
```

### Support Channels

- **Emergency**: APG Platform On-Call (for production issues)
- **Standard**: APG Support Portal (https://support.apg.yourcompany.com)
- **Community**: APG User Forum
- **Documentation**: https://docs.apg.yourcompany.com/dvrl

### Escalation Criteria

Escalate immediately for:
- Production system down (>5 minutes)
- Data corruption or loss
- Security incidents
- Critical performance degradation (>50% slower)

---

**Document Version**: 1.0  
**Last Updated**: 2025-01-11  
**Author**: APG Platform Team