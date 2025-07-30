# APG RAG Operations Manual

> **Complete operations and maintenance guide for production environments**

## 📋 Operations Overview

This manual provides comprehensive guidance for operating and maintaining the APG RAG capability in production environments. It covers monitoring, troubleshooting, maintenance, scaling, and disaster recovery procedures.

## 🚀 System Operations

### Daily Operations Checklist

#### Morning Health Check (9:00 AM)

```bash
#!/bin/bash
# daily_health_check.sh

echo "=== APG RAG Daily Health Check - $(date) ==="

# 1. Check service health
echo "1. Checking service health..."
curl -s http://localhost/api/v1/rag/health | jq '.data.service_status'

# 2. Check database connectivity
echo "2. Checking database connectivity..."
psql $DATABASE_URL -c "SELECT 'Database OK' as status;" 2>/dev/null || echo "Database FAILED"

# 3. Check Ollama service
echo "3. Checking Ollama service..."
curl -s $OLLAMA_BASE_URL/api/tags | jq '.models | length'

# 4. Check Redis cache
echo "4. Checking Redis cache..."
redis-cli ping 2>/dev/null || echo "Redis FAILED"

# 5. Check disk usage
echo "5. Checking disk usage..."
df -h | grep -E "(data|logs|backups)"

# 6. Check memory usage
echo "6. Checking memory usage..."
free -h

# 7. Check active connections
echo "7. Checking active connections..."
kubectl get pods -n apg-rag -o wide

# 8. Check recent errors
echo "8. Checking recent errors (last hour)..."
kubectl logs -n apg-rag --since=1h --selector=app=rag-service | grep -i error | tail -10

echo "=== Health Check Complete ==="
```

#### Evening Metrics Review (6:00 PM)

```bash
#!/bin/bash
# evening_metrics_review.sh

echo "=== APG RAG Evening Metrics Review - $(date) ==="

# 1. Query volume and performance
echo "1. Query Performance Metrics:"
curl -s "http://prometheus:9090/api/v1/query?query=sum(rate(rag_queries_total[24h]))" | \
  jq '.data.result[0].value[1]' | \
  awk '{printf "Total queries today: %.0f\n", $1 * 86400}'

# 2. Average response time
echo "2. Average Response Time:"
curl -s "http://prometheus:9090/api/v1/query?query=avg(rag_response_time_ms)" | \
  jq '.data.result[0].value[1]' | \
  awk '{printf "Average response time: %.2f ms\n", $1}'

# 3. Error rate
echo "3. Error Rate:"
curl -s "http://prometheus:9090/api/v1/query?query=sum(rate(rag_errors_total[24h]))" | \
  jq '.data.result[0].value[1]' | \
  awk '{printf "Error rate: %.2f%%\n", $1 * 100}'

# 4. Cache hit rate
echo "4. Cache Performance:"
curl -s "http://prometheus:9090/api/v1/query?query=rag_cache_hit_rate" | \
  jq '.data.result[0].value[1]' | \
  awk '{printf "Cache hit rate: %.2f%%\n", $1 * 100}'

# 5. Resource utilization
echo "5. Resource Utilization:"
kubectl top pods -n apg-rag --containers

echo "=== Metrics Review Complete ==="
```

### Weekly Maintenance Tasks

#### Sunday Maintenance Window (2:00 AM - 4:00 AM)

```bash
#!/bin/bash
# weekly_maintenance.sh

echo "=== APG RAG Weekly Maintenance - $(date) ==="

# 1. Database maintenance
echo "1. Running database maintenance..."
psql $DATABASE_URL -c "
  -- Vacuum and analyze tables
  VACUUM ANALYZE apg_rag_documents;
  VACUUM ANALYZE apg_rag_document_chunks; 
  VACUUM ANALYZE apg_rag_conversations;
  
  -- Reindex vector indexes
  REINDEX INDEX CONCURRENTLY apg_rag_document_chunks_embedding_idx;
  
  -- Update statistics
  ANALYZE;
"

# 2. Log rotation and cleanup
echo "2. Rotating logs..."
find /app/logs -name "*.log" -mtime +7 -delete
find /app/logs -name "*.log.[0-9]*" -mtime +30 -delete

# 3. Cache warmup
echo "3. Warming cache..."
python scripts/cache_warmup.py

# 4. Security scan
echo "4. Running security scan..."
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image apg-rag:latest

# 5. Backup verification
echo "5. Verifying backups..."
python scripts/verify_backups.py

# 6. Performance baseline
echo "6. Running performance baseline..."
python scripts/performance_baseline.py

echo "=== Weekly Maintenance Complete ==="
```

## 📊 Monitoring and Alerting

### Key Performance Indicators

#### Service Level Indicators (SLIs)

```yaml
# monitoring/sli_config.yaml
slis:
  availability:
    description: "Service uptime and accessibility"
    threshold: 99.9%
    measurement: "Percentage of successful health checks"
    
  latency:
    description: "Query response time"
    threshold: 2000  # milliseconds
    measurement: "95th percentile response time"
    
  throughput:
    description: "Query processing rate"
    threshold: 100   # queries per second
    measurement: "Queries processed per second"
    
  error_rate:
    description: "Failed request percentage"
    threshold: 0.1%  # 0.1% error rate
    measurement: "Percentage of failed requests"
    
  quality:
    description: "Response quality score"
    threshold: 0.85  # 85% quality score
    measurement: "Average user satisfaction rating"
```

#### Alert Rules

```yaml
# monitoring/alert_rules.yaml
groups:
  - name: rag_service_alerts
    rules:
      - alert: RAGServiceDown
        expr: up{job="rag-service"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "RAG service is down"
          description: "RAG service has been down for more than 1 minute"
          runbook_url: "https://docs.company.com/runbooks/rag-service-down"
          
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rag_response_time_seconds) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          description: "95th percentile response time is {{ $value }}s"
          
      - alert: HighErrorRate
        expr: rate(rag_errors_total[5m]) > 0.01
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }}"
          
      - alert: DatabaseConnectionsHigh
        expr: pg_stat_activity_count / pg_settings_max_connections > 0.8
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Database connections usage high"
          description: "{{ $value | humanizePercentage }} of connections in use"
          
      - alert: OllamaServiceDown
        expr: up{job="ollama"} == 0
        for: 30s
        labels:
          severity: critical
        annotations:
          summary: "Ollama service is down"
          description: "Ollama AI service is unavailable"
```

### Monitoring Dashboards

#### Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "title": "APG RAG Operations Dashboard",
    "panels": [
      {
        "title": "Service Health",
        "type": "stat",
        "targets": [
          {
            "expr": "up{job=\"rag-service\"}",
            "legendFormat": "Service Status"
          }
        ]
      },
      {
        "title": "Query Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rag_response_time_seconds)",
            "legendFormat": "95th Percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rag_response_time_seconds)",
            "legendFormat": "Median"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(rag_errors_total[5m])",
            "legendFormat": "Error Rate"
          }
        ]
      },
      {
        "title": "Database Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "pg_stat_database_tup_fetched",
            "legendFormat": "Rows Fetched"
          },
          {
            "expr": "pg_stat_database_tup_inserted",
            "legendFormat": "Rows Inserted"
          }
        ]
      }
    ]
  }
}
```

## 🛠️ Troubleshooting Guide

### Common Issues and Solutions

#### 1. Service Not Responding

**Symptoms:**
- Health check endpoint returns 502/503
- Application logs show connection errors
- High response times or timeouts

**Diagnosis:**
```bash
# Check service status
kubectl get pods -n apg-rag
kubectl describe pod rag-service-xxx -n apg-rag

# Check service logs
kubectl logs -f rag-service-xxx -n apg-rag --tail=100

# Check resource usage
kubectl top pod rag-service-xxx -n apg-rag --containers

# Test connectivity
kubectl exec -it rag-service-xxx -n apg-rag -- curl localhost:5000/api/v1/rag/health
```

**Solutions:**
```bash
# Restart service
kubectl rollout restart deployment/rag-service -n apg-rag

# Scale up if resource constrained
kubectl scale deployment rag-service --replicas=5 -n apg-rag

# Check and fix configuration
kubectl edit configmap rag-config -n apg-rag
```

#### 2. Database Connection Issues

**Symptoms:**
- "Connection refused" errors in logs
- Database timeouts
- Connection pool exhaustion

**Diagnosis:**
```bash
# Check PostgreSQL status
kubectl get pods -n apg-rag | grep postgres
kubectl logs postgres-cluster-1 -n apg-rag

# Check connection count
psql $DATABASE_URL -c "
  SELECT count(*) as active_connections,
         setting as max_connections
  FROM pg_stat_activity, pg_settings 
  WHERE pg_settings.name = 'max_connections';
"

# Check slow queries
psql $DATABASE_URL -c "
  SELECT query, state, query_start, now() - query_start AS duration
  FROM pg_stat_activity 
  WHERE state != 'idle' 
  ORDER BY duration DESC 
  LIMIT 10;
"
```

**Solutions:**
```bash
# Kill long-running queries
psql $DATABASE_URL -c "
  SELECT pg_terminate_backend(pid) 
  FROM pg_stat_activity 
  WHERE state != 'idle' 
  AND now() - query_start > interval '5 minutes';
"

# Increase connection pool size
kubectl patch deployment rag-service -n apg-rag -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [
          {
            "name": "rag-service",
            "env": [
              {
                "name": "DATABASE_POOL_SIZE",
                "value": "30"
              }
            ]
          }
        ]
      }
    }
  }
}'

# Scale database if needed
kubectl patch cluster postgres-cluster -n apg-rag --type='json' -p='
[
  {
    "op": "replace",
    "path": "/spec/instances",
    "value": 5
  }
]'
```

#### 3. Ollama Service Issues

**Symptoms:**
- Embedding generation failures
- Model loading errors
- GPU memory issues

**Diagnosis:**
```bash
# Check Ollama service
kubectl get pods -n apg-rag | grep ollama
kubectl logs ollama-xxx -n apg-rag

# Check available models
kubectl exec -it ollama-xxx -n apg-rag -- ollama list

# Check GPU usage
kubectl exec -it ollama-xxx -n apg-rag -- nvidia-smi

# Test model endpoint
curl -X POST http://ollama:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "bge-m3", "prompt": "test"}'
```

**Solutions:**
```bash
# Restart Ollama service
kubectl rollout restart deployment/ollama -n apg-rag

# Reload models
kubectl exec -it ollama-xxx -n apg-rag -- ollama pull bge-m3
kubectl exec -it ollama-xxx -n apg-rag -- ollama pull qwen3

# Scale Ollama pods for load balancing
kubectl scale deployment ollama --replicas=3 -n apg-rag

# Clear GPU memory
kubectl exec -it ollama-xxx -n apg-rag -- nvidia-smi --gpu-reset
```

#### 4. High Memory Usage

**Symptoms:**
- OOMKilled pod restarts
- Slow query performance
- Cache eviction warnings

**Diagnosis:**
```bash
# Check memory usage
kubectl top pods -n apg-rag --containers
kubectl describe pod rag-service-xxx -n apg-rag | grep -A 5 Limits

# Check memory-intensive processes
kubectl exec -it rag-service-xxx -n apg-rag -- ps aux --sort=-%mem | head -10

# Check cache sizes
kubectl exec -it rag-service-xxx -n apg-rag -- \
  python -c "
import psutil
print(f'Memory usage: {psutil.virtual_memory().percent}%')
print(f'Available: {psutil.virtual_memory().available // 1024 // 1024}MB')
"
```

**Solutions:**
```bash
# Increase memory limits
kubectl patch deployment rag-service -n apg-rag -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [
          {
            "name": "rag-service",
            "resources": {
              "limits": {
                "memory": "8Gi"
              },
              "requests": {
                "memory": "4Gi"
              }
            }
          }
        ]
      }
    }
  }
}'

# Reduce cache sizes
kubectl set env deployment/rag-service -n apg-rag \
  VECTOR_CACHE_SIZE=50000 \
  MAX_CONCURRENT_OPERATIONS=50

# Enable horizontal scaling
kubectl apply -f - <<EOF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-service-hpa
  namespace: apg-rag
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-service
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 75
EOF
```

### Performance Optimization

#### Database Performance Tuning

```sql
-- Optimize vector search performance
ALTER SYSTEM SET max_parallel_workers_per_gather = 4;
ALTER SYSTEM SET max_parallel_workers = 8;
ALTER SYSTEM SET shared_buffers = '2GB';
ALTER SYSTEM SET effective_cache_size = '6GB';
ALTER SYSTEM SET maintenance_work_mem = '1GB';

-- Optimize pgvector indexes
SET ivfflat.probes = 10;  -- Adjust based on accuracy vs speed needs

-- Monitor slow queries
SELECT 
  query,
  calls,
  total_time,
  mean_time,
  rows
FROM pg_stat_statements 
WHERE mean_time > 1000  -- Queries taking more than 1 second
ORDER BY mean_time DESC 
LIMIT 10;
```

#### Application Performance Tuning

```python
# config/production_optimized.py
PERFORMANCE_CONFIG = {
    # Connection pooling
    "database_pool_size": 30,
    "database_max_overflow": 50,
    "database_pool_timeout": 30,
    
    # Caching
    "vector_cache_ttl": 3600,  # 1 hour
    "query_cache_ttl": 1800,   # 30 minutes
    "document_cache_ttl": 7200, # 2 hours
    
    # Processing
    "max_concurrent_operations": 100,
    "chunk_batch_size": 1000,
    "embedding_batch_size": 50,
    
    # Ollama optimization
    "ollama_connection_pool": 20,
    "ollama_timeout": 30,
    "ollama_max_retries": 2,
}
```

## 🔄 Backup and Recovery

### Backup Strategy

#### Automated Daily Backups

```bash
#!/bin/bash
# backup_daily.sh

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/app/backups/daily"
RETENTION_DAYS=30

echo "=== Starting Daily Backup - $BACKUP_DATE ==="

# 1. PostgreSQL backup
echo "1. Backing up PostgreSQL..."
pg_dump $DATABASE_URL | gzip > "$BACKUP_DIR/postgres_$BACKUP_DATE.sql.gz"

# 2. Document storage backup
echo "2. Backing up document storage..."
tar -czf "$BACKUP_DIR/documents_$BACKUP_DATE.tar.gz" /app/data/documents/

# 3. Configuration backup
echo "3. Backing up configuration..."
kubectl get configmaps -n apg-rag -o yaml > "$BACKUP_DIR/config_$BACKUP_DATE.yaml"
kubectl get secrets -n apg-rag -o yaml > "$BACKUP_DIR/secrets_$BACKUP_DATE.yaml"

# 4. Clean old backups
echo "4. Cleaning old backups..."
find $BACKUP_DIR -name "*.gz" -mtime +$RETENTION_DAYS -delete
find $BACKUP_DIR -name "*.yaml" -mtime +$RETENTION_DAYS -delete

# 5. Verify backup integrity
echo "5. Verifying backup integrity..."
gzip -t "$BACKUP_DIR/postgres_$BACKUP_DATE.sql.gz" && echo "PostgreSQL backup OK"
tar -tzf "$BACKUP_DIR/documents_$BACKUP_DATE.tar.gz" >/dev/null && echo "Documents backup OK"

# 6. Upload to remote storage (optional)
if [ "$BACKUP_REMOTE_ENABLED" = "true" ]; then
    echo "6. Uploading to remote storage..."
    aws s3 cp "$BACKUP_DIR/postgres_$BACKUP_DATE.sql.gz" s3://$BACKUP_BUCKET/daily/
    aws s3 cp "$BACKUP_DIR/documents_$BACKUP_DATE.tar.gz" s3://$BACKUP_BUCKET/daily/
fi

echo "=== Daily Backup Complete ==="
```

#### Weekly Full System Backup

```bash
#!/bin/bash
# backup_weekly.sh

BACKUP_DATE=$(date +%Y%m%d)
BACKUP_DIR="/app/backups/weekly"

echo "=== Starting Weekly Full Backup - $BACKUP_DATE ==="

# 1. Full system backup with Velero
echo "1. Creating full system backup..."
velero backup create apg-rag-weekly-$BACKUP_DATE \
  --include-namespaces apg-rag \
  --storage-location default \
  --ttl 2160h  # 90 days

# 2. Database cluster backup
echo "2. Creating database cluster backup..."
kubectl exec -n apg-rag postgres-cluster-1 -- \
  pg_basebackup -D /backup/weekly_$BACKUP_DATE -Ft -z -P

# 3. Export monitoring data
echo "3. Exporting monitoring data..."
curl -G 'http://prometheus:9090/api/v1/export' \
  --data-urlencode 'start='$(date -d '7 days ago' +%s) \
  --data-urlencode 'end='$(date +%s) \
  | gzip > "$BACKUP_DIR/metrics_$BACKUP_DATE.json.gz"

echo "=== Weekly Full Backup Complete ==="
```

### Disaster Recovery Procedures

#### Complete System Recovery

```bash
#!/bin/bash
# disaster_recovery.sh

RECOVERY_DATE=$1
if [ -z "$RECOVERY_DATE" ]; then
    echo "Usage: $0 <backup_date>"
    exit 1
fi

echo "=== Starting Disaster Recovery for $RECOVERY_DATE ==="

# 1. Restore Kubernetes resources
echo "1. Restoring Kubernetes resources..."
velero restore create apg-rag-restore-$(date +%Y%m%d) \
  --from-backup apg-rag-weekly-$RECOVERY_DATE

# 2. Wait for pods to be ready
echo "2. Waiting for pods to be ready..."
kubectl wait --for=condition=ready pod -l app=postgres -n apg-rag --timeout=300s
kubectl wait --for=condition=ready pod -l app=rag-service -n apg-rag --timeout=300s

# 3. Restore database
echo "3. Restoring database..."
gunzip -c "/app/backups/daily/postgres_${RECOVERY_DATE}*.sql.gz" | \
  psql $DATABASE_URL

# 4. Restore document storage
echo "4. Restoring document storage..."
tar -xzf "/app/backups/daily/documents_${RECOVERY_DATE}*.tar.gz" -C /

# 5. Verify system health
echo "5. Verifying system health..."
sleep 30  # Allow services to start
curl -f http://localhost/api/v1/rag/health || {
    echo "Health check failed"
    exit 1
}

# 6. Run data integrity checks
echo "6. Running data integrity checks..."
python scripts/verify_data_integrity.py

echo "=== Disaster Recovery Complete ==="
```

#### Point-in-Time Recovery

```bash
#!/bin/bash
# point_in_time_recovery.sh

TARGET_TIME=$1
if [ -z "$TARGET_TIME" ]; then
    echo "Usage: $0 <target_time_iso8601>"
    echo "Example: $0 '2025-01-29T14:30:00Z'"
    exit 1
fi

echo "=== Point-in-Time Recovery to $TARGET_TIME ==="

# 1. Stop applications
echo "1. Stopping applications..."
kubectl scale deployment rag-service --replicas=0 -n apg-rag

# 2. Perform PITR
echo "2. Performing point-in-time recovery..."
kubectl exec -n apg-rag postgres-cluster-1 -- \
  pg_ctl stop -D /var/lib/postgresql/data
  
# Restore from base backup and apply WAL files up to target time
kubectl exec -n apg-rag postgres-cluster-1 -- \
  pg_basebackup -D /var/lib/postgresql/data-recovery \
  --target-time="$TARGET_TIME"

# 3. Start database
echo "3. Starting database..."
kubectl exec -n apg-rag postgres-cluster-1 -- \
  pg_ctl start -D /var/lib/postgresql/data-recovery

# 4. Restart applications
echo "4. Restarting applications..."
kubectl scale deployment rag-service --replicas=3 -n apg-rag

echo "=== Point-in-Time Recovery Complete ==="
```

## 📈 Scaling Operations

### Horizontal Scaling

#### Auto-scaling Configuration

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-service-hpa
  namespace: apg-rag
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-service
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
  - type: Pods
    pods:
      metric:
        name: rag_active_requests
      target:
        type: AverageValue
        averageValue: "100"
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

#### Manual Scaling Commands

```bash
# Scale RAG service
kubectl scale deployment rag-service --replicas=10 -n apg-rag

# Scale Ollama service
kubectl scale deployment ollama --replicas=5 -n apg-rag

# Scale database cluster
kubectl patch cluster postgres-cluster -n apg-rag --type='json' -p='
[
  {
    "op": "replace",
    "path": "/spec/instances", 
    "value": 5
  }
]'

# Scale Redis cluster
kubectl scale statefulset redis --replicas=3 -n apg-rag
```

### Vertical Scaling

#### Resource Optimization

```yaml
# k8s/resource_limits.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-service
  namespace: apg-rag
spec:
  template:
    spec:
      containers:
      - name: rag-service
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "16Gi"
            cpu: "8"
        env:
        - name: MAX_CONCURRENT_OPERATIONS
          value: "200"
        - name: DATABASE_POOL_SIZE
          value: "50"
        - name: VECTOR_CACHE_SIZE
          value: "200000"
```

## 🔒 Security Operations

### Security Monitoring

#### Security Audit Script

```bash
#!/bin/bash
# security_audit.sh

echo "=== APG RAG Security Audit - $(date) ==="

# 1. Check for vulnerabilities
echo "1. Scanning for vulnerabilities..."
trivy image apg-rag:latest --format json > security_scan.json

# 2. Check SSL certificates
echo "2. Checking SSL certificates..."
echo | openssl s_client -connect localhost:443 2>/dev/null | \
  openssl x509 -noout -dates

# 3. Audit database permissions
echo "3. Auditing database permissions..."
psql $DATABASE_URL -c "
  SELECT rolname, rolsuper, rolcreaterole, rolcreatedb 
  FROM pg_roles 
  WHERE rolname NOT LIKE 'pg_%';
"

# 4. Check network policies
echo "4. Checking network policies..."
kubectl get networkpolicies -n apg-rag

# 5. Audit access logs
echo "5. Analyzing access logs..."
tail -1000 /app/logs/access.log | \
  awk '{print $1}' | sort | uniq -c | sort -nr | head -10

# 6. Check for security updates
echo "6. Checking for security updates..."
kubectl get pods -n apg-rag -o jsonpath='{.items[*].spec.containers[*].image}' | \
  tr ' ' '\n' | sort -u

echo "=== Security Audit Complete ==="
```

### Compliance Operations

#### GDPR Compliance Check

```python
# scripts/gdpr_compliance_check.py
import asyncio
import asyncpg
from datetime import datetime, timedelta

async def check_gdpr_compliance():
    """Check GDPR compliance status."""
    conn = await asyncpg.connect(DATABASE_URL)
    
    # Check data retention
    old_data = await conn.fetch("""
        SELECT table_name, count(*) as old_records
        FROM (
            SELECT 'apg_rag_audit_logs' as table_name, count(*) 
            FROM apg_rag_audit_logs 
            WHERE created_at < NOW() - INTERVAL '7 years'
            
            UNION ALL
            
            SELECT 'apg_rag_conversations' as table_name, count(*)
            FROM apg_rag_conversations 
            WHERE created_at < NOW() - INTERVAL '2 years'
        ) as old_data_check;
    """)
    
    # Check encryption status
    encrypted_fields = await conn.fetch("""
        SELECT column_name, data_type
        FROM information_schema.columns 
        WHERE table_name LIKE 'apg_rag_%' 
        AND column_name IN ('content', 'query_text', 'response_text');
    """)
    
    # Generate compliance report
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "data_retention": old_data,
        "encryption_status": encrypted_fields,
        "compliance_status": "COMPLIANT" if not old_data else "REQUIRES_CLEANUP"
    }
    
    print(f"GDPR Compliance Report: {report}")
    await conn.close()
    return report

if __name__ == "__main__":
    asyncio.run(check_gdpr_compliance())
```

## 📞 Incident Response

### Incident Response Playbook

#### Severity Levels

| Level | Description | Response Time | Escalation |
|-------|-------------|---------------|------------|
| P0 | Service completely down | 15 minutes | Immediate |
| P1 | Severe performance degradation | 30 minutes | 1 hour |
| P2 | Moderate issues affecting some users | 2 hours | 4 hours |
| P3 | Minor issues or planned maintenance | 24 hours | 48 hours |

#### P0 Incident Response

```bash
#!/bin/bash
# p0_incident_response.sh

INCIDENT_ID=$1
if [ -z "$INCIDENT_ID" ]; then
    echo "Usage: $0 <incident_id>"
    exit 1
fi

echo "=== P0 Incident Response: $INCIDENT_ID ==="

# 1. Immediate assessment
echo "1. Performing immediate assessment..."
kubectl get pods -n apg-rag
curl -f http://localhost/api/v1/rag/health || echo "Service DOWN"

# 2. Activate war room
echo "2. Activating war room..."
# Send alerts to on-call team
curl -X POST "$SLACK_WEBHOOK_URL" -d '{
  "text": "🚨 P0 INCIDENT: '"$INCIDENT_ID"' - APG RAG Service Down",
  "channel": "#incidents",
  "username": "APG-RAG-Bot"
}'

# 3. Immediate mitigation
echo "3. Attempting immediate mitigation..."
# Restart services
kubectl rollout restart deployment/rag-service -n apg-rag

# Scale up if needed
kubectl scale deployment rag-service --replicas=5 -n apg-rag

# 4. Collect diagnostics
echo "4. Collecting diagnostics..."
mkdir -p /tmp/incident_$INCIDENT_ID
kubectl logs -n apg-rag --selector=app=rag-service --since=1h > \
  /tmp/incident_$INCIDENT_ID/service_logs.txt
kubectl describe pods -n apg-rag > \
  /tmp/incident_$INCIDENT_ID/pod_status.txt

# 5. Monitor recovery
echo "5. Monitoring recovery..."
for i in {1..10}; do
    if curl -f http://localhost/api/v1/rag/health; then
        echo "✅ Service recovered after $((i*30)) seconds"
        break
    fi
    sleep 30
done

echo "=== P0 Incident Response Complete ==="
```

This comprehensive operations manual provides the foundation for maintaining the APG RAG capability in production environments with high availability, security, and performance standards.