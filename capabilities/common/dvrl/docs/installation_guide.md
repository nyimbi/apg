# APG Data Virtualization (DVRL) Installation Guide

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [APG Platform Setup](#apg-platform-setup)
3. [DVRL Capability Installation](#dvrl-capability-installation)
4. [Configuration](#configuration)
5. [Database Setup](#database-setup)
6. [Environment Variables](#environment-variables)
7. [APG Integration Configuration](#apg-integration-configuration)
8. [Security Configuration](#security-configuration)
9. [Performance Tuning](#performance-tuning)
10. [Verification](#verification)
11. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements
- **Operating System**: Linux (RHEL 8+, Ubuntu 20.04+) or Docker-compatible environment
- **Python**: 3.11 or higher
- **Memory**: Minimum 8GB RAM (16GB+ recommended for production)
- **CPU**: 4+ cores (8+ cores recommended for production)
- **Storage**: 100GB+ available disk space
- **Network**: High-speed network connectivity to data sources

### Required Software
- **Docker**: Version 24.0 or higher
- **Docker Compose**: Version 2.0 or higher
- **PostgreSQL**: 14+ (for APG platform metadata)
- **Redis**: 6.2+ (for caching and session management)

### APG Platform Dependencies
Ensure the following APG capabilities are installed and configured:
- **auth_rbac**: Authentication and authorization
- **meta**: Metadata management and schema registry
- **cach**: Intelligent caching framework
- **moni**: Monitoring and observability
- **audit_compliance**: Audit logging and compliance

## APG Platform Setup

### 1. Verify APG Platform Installation
```bash
# Check APG platform status
kubectl get pods -n apg-platform

# Verify core capabilities
curl -H "Authorization: Bearer ${APG_TOKEN}" \
  ${APG_BASE_URL}/api/v1/capabilities/status
```

### 2. Tenant Configuration
Ensure your organization's tenant is properly configured:

```bash
# Create tenant (if not exists)
kubectl apply -f - <<EOF
apiVersion: apg.platform/v1
kind: Tenant
metadata:
  name: ${TENANT_ID}
spec:
  displayName: "${TENANT_DISPLAY_NAME}"
  capabilities:
    - auth_rbac
    - meta
    - cach
    - dvrl
  resources:
    memory: "16Gi"
    cpu: "8"
    storage: "500Gi"
EOF
```

### 3. Service Account Setup
Create a dedicated service account for DVRL:

```bash
# Create service account
kubectl create serviceaccount dvrl-service-account -n apg-platform

# Apply RBAC permissions
kubectl apply -f - <<EOF
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: dvrl-cluster-role
rules:
- apiGroups: [""]
  resources: ["secrets", "configmaps", "services"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments", "statefulsets"]
  verbs: ["get", "list", "watch", "create", "update", "patch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: dvrl-cluster-role-binding
subjects:
- kind: ServiceAccount
  name: dvrl-service-account
  namespace: apg-platform
roleRef:
  kind: ClusterRole
  name: dvrl-cluster-role
  apiGroup: rbac.authorization.k8s.io
EOF
```

## DVRL Capability Installation

### 1. Download DVRL Package
```bash
# Download from APG repository
wget ${APG_RELEASES_URL}/dvrl-v1.0.0.tar.gz

# Extract package
tar -xzf dvrl-v1.0.0.tar.gz
cd dvrl-v1.0.0
```

### 2. Container Image Setup
```bash
# Build DVRL container image
docker build -t apg-dvrl:v1.0.0 .

# Tag for registry
docker tag apg-dvrl:v1.0.0 ${REGISTRY_URL}/apg-dvrl:v1.0.0

# Push to registry
docker push ${REGISTRY_URL}/apg-dvrl:v1.0.0
```

### 3. Kubernetes Deployment
```bash
# Create namespace (if not exists)
kubectl create namespace apg-dvrl

# Create secrets
kubectl create secret generic dvrl-secrets -n apg-dvrl \
  --from-literal=database-password="${DB_PASSWORD}" \
  --from-literal=redis-password="${REDIS_PASSWORD}" \
  --from-literal=jwt-secret="${JWT_SECRET}"

# Apply deployment
kubectl apply -f kubernetes/
```

### 4. Helm Installation (Alternative)
```bash
# Add APG Helm repository
helm repo add apg https://charts.apg.platform
helm repo update

# Install DVRL capability
helm install apg-dvrl apg/dvrl \
  --namespace apg-dvrl \
  --create-namespace \
  --set image.repository=${REGISTRY_URL}/apg-dvrl \
  --set image.tag=v1.0.0 \
  --set config.tenant_id=${TENANT_ID} \
  --set config.apg_base_url=${APG_BASE_URL} \
  --values values-production.yaml
```

## Configuration

### 1. Core Configuration
Create the main configuration file:

```yaml
# config/dvrl-config.yaml
tenant_id: "${TENANT_ID}"
environment: "production"

apg_platform:
  base_url: "${APG_BASE_URL}"
  auth_endpoint: "${APG_BASE_URL}/api/v1/auth"
  capabilities:
    meta:
      enabled: true
      endpoint: "${APG_BASE_URL}/api/v1/meta"
    cach:
      enabled: true
      endpoint: "${APG_BASE_URL}/api/v1/cach"
    auth_rbac:
      enabled: true
      endpoint: "${APG_BASE_URL}/api/v1/auth"

database:
  host: "postgresql.apg-platform.svc.cluster.local"
  port: 5432
  database: "dvrl_metadata"
  username: "dvrl_user"
  password: "${DB_PASSWORD}"
  pool_size: 20
  max_connections: 100

cache:
  redis:
    host: "redis.apg-platform.svc.cluster.local"
    port: 6379
    password: "${REDIS_PASSWORD}"
    db: 0
  memory:
    max_size_mb: 1024
    ttl_seconds: 3600

query_engine:
  max_concurrent_queries: 100
  default_timeout_seconds: 300
  max_result_rows: 1000000
  streaming:
    enabled: true
    buffer_size_mb: 50
    batch_size: 1000

security:
  jwt_secret: "${JWT_SECRET}"
  encryption_key: "${ENCRYPTION_KEY}"
  ssl:
    enabled: true
    cert_file: "/etc/ssl/certs/dvrl.crt"
    key_file: "/etc/ssl/private/dvrl.key"

logging:
  level: "INFO"
  format: "json"
  output: "stdout"
  audit:
    enabled: true
    retention_days: 90

monitoring:
  metrics:
    enabled: true
    port: 9090
    path: "/metrics"
  health_check:
    enabled: true
    port: 8080
    path: "/health"
```

### 2. Data Source Connectors Configuration
```yaml
# config/connectors-config.yaml
connectors:
  postgresql:
    driver: "asyncpg"
    connection_params:
      ssl: "require"
      command_timeout: 60
    pool_config:
      min_size: 5
      max_size: 20
      max_inactive_connection_lifetime: 300
      
  mysql:
    driver: "aiomysql"
    connection_params:
      charset: "utf8mb4"
      autocommit: true
    pool_config:
      min_size: 5
      max_size: 15
      
  mongodb:
    driver: "motor"
    connection_params:
      retryWrites: true
      serverSelectionTimeoutMS: 5000
    pool_config:
      max_pool_size: 100
      min_pool_size: 10
      
  oracle:
    driver: "cx_oracle"
    connection_params:
      encoding: "UTF-8"
      threaded: true
    pool_config:
      min: 5
      max: 20
      increment: 1
```

## Database Setup

### 1. PostgreSQL Metadata Database
```sql
-- Create database and user
CREATE DATABASE dvrl_metadata;
CREATE USER dvrl_user WITH ENCRYPTED PASSWORD '${DB_PASSWORD}';
GRANT ALL PRIVILEGES ON DATABASE dvrl_metadata TO dvrl_user;

-- Connect to dvrl_metadata database
\c dvrl_metadata

-- Create schema
CREATE SCHEMA IF NOT EXISTS dvrl;
ALTER USER dvrl_user SET search_path = dvrl, public;
GRANT ALL PRIVILEGES ON SCHEMA dvrl TO dvrl_user;

-- Create tables (will be automated by migration scripts)
```

### 2. Database Migration
```bash
# Run database migrations
python -m dvrl.migrations upgrade

# Verify tables created
python -c "
import asyncio
from dvrl.database import get_database_connection

async def verify():
    async with get_database_connection() as conn:
        tables = await conn.fetch('SELECT tablename FROM pg_tables WHERE schemaname = \'dvrl\'')
        print(f'Created {len(tables)} tables: {[t[\"tablename\"] for t in tables]}')

asyncio.run(verify())
"
```

### 3. Initial Data Setup
```bash
# Create default configuration
python -m dvrl.setup --tenant-id=${TENANT_ID} --admin-user=${ADMIN_EMAIL}

# Load sample data source configurations (optional)
python -m dvrl.samples.load_demo_data
```

## Environment Variables

### Required Environment Variables
```bash
# APG Platform Configuration
export TENANT_ID="your-organization-tenant"
export APG_BASE_URL="https://api.apg.yourcompany.com"
export APG_ACCESS_TOKEN="your-service-account-token"

# Database Configuration
export DB_HOST="postgresql.apg-platform.svc.cluster.local"
export DB_PORT="5432"
export DB_NAME="dvrl_metadata"
export DB_USER="dvrl_user"
export DB_PASSWORD="secure-database-password"

# Cache Configuration
export REDIS_HOST="redis.apg-platform.svc.cluster.local"
export REDIS_PORT="6379"
export REDIS_PASSWORD="secure-redis-password"

# Security Configuration
export JWT_SECRET="secure-jwt-secret-key"
export ENCRYPTION_KEY="secure-encryption-key-32-chars"

# Performance Configuration
export DVRL_MAX_CONCURRENT_QUERIES="100"
export DVRL_DEFAULT_TIMEOUT="300"
export DVRL_CACHE_SIZE_MB="1024"

# Monitoring Configuration
export DVRL_LOG_LEVEL="INFO"
export DVRL_METRICS_ENABLED="true"
export DVRL_HEALTH_CHECK_ENABLED="true"
```

### Optional Environment Variables
```bash
# Development/Testing
export DVRL_DEBUG="false"
export DVRL_MOCK_DATA_SOURCES="false"
export DVRL_SKIP_AUTH="false"

# Performance Tuning
export DVRL_CONNECTION_POOL_SIZE="20"
export DVRL_QUERY_CACHE_TTL="3600"
export DVRL_STREAMING_BUFFER_SIZE="50"

# Feature Flags
export DVRL_ENABLE_NLP="true"
export DVRL_ENABLE_STREAMING="true"
export DVRL_ENABLE_TRANSACTIONS="true"
```

## APG Integration Configuration

### 1. Auth RBAC Integration
```yaml
# config/auth-integration.yaml
auth_rbac:
  endpoint: "${APG_BASE_URL}/api/v1/auth"
  token_validation:
    enabled: true
    cache_ttl: 300
    refresh_threshold: 60
  role_mapping:
    dvrl_admin:
      permissions:
        - "dvrl:admin"
        - "dvrl:read"
        - "dvrl:write"
        - "dvrl:execute"
    dvrl_analyst:
      permissions:
        - "dvrl:read"
        - "dvrl:execute"
    dvrl_viewer:
      permissions:
        - "dvrl:read"
```

### 2. Metadata Service Integration
```yaml
# config/meta-integration.yaml
meta:
  endpoint: "${APG_BASE_URL}/api/v1/meta"
  schema_sync:
    enabled: true
    interval_seconds: 300
    batch_size: 100
  lineage_tracking:
    enabled: true
    track_queries: true
    track_transformations: true
```

### 3. Caching Service Integration
```yaml
# config/cach-integration.yaml
cach:
  endpoint: "${APG_BASE_URL}/api/v1/cach"
  cache_levels:
    memory:
      enabled: true
      max_size_mb: 1024
      ttl_seconds: 3600
    distributed:
      enabled: true
      cluster_nodes: 3
      replication_factor: 2
  intelligent_caching:
    enabled: true
    ml_model: "cache_prediction_v1"
    confidence_threshold: 0.8
```

## Security Configuration

### 1. SSL/TLS Configuration
```bash
# Generate SSL certificates (or use existing ones)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout dvrl.key \
  -out dvrl.crt \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=dvrl.apg.yourcompany.com"

# Create Kubernetes secret
kubectl create secret tls dvrl-tls-secret -n apg-dvrl \
  --cert=dvrl.crt \
  --key=dvrl.key
```

### 2. Data Encryption Configuration
```yaml
# config/encryption.yaml
encryption:
  data_at_rest:
    enabled: true
    algorithm: "AES-256-GCM"
    key_rotation: true
    rotation_interval_days: 90
  data_in_transit:
    enabled: true
    min_tls_version: "1.2"
    cipher_suites:
      - "ECDHE-RSA-AES128-GCM-SHA256"
      - "ECDHE-RSA-AES256-GCM-SHA384"
  connection_strings:
    encrypt_passwords: true
    mask_in_logs: true
```

### 3. Access Control Configuration
```yaml
# config/access-control.yaml
access_control:
  row_level_security:
    enabled: true
    default_policy: "deny"
    tenant_isolation: true
  column_level_security:
    enabled: true
    sensitive_columns:
      - "email"
      - "ssn"
      - "credit_card"
      - "phone"
  dynamic_data_masking:
    enabled: true
    masking_rules:
      email: "hash_domain"
      ssn: "mask_middle"
      credit_card: "mask_middle"
```

## Performance Tuning

### 1. Database Optimization
```sql
-- PostgreSQL optimization
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '12GB';
ALTER SYSTEM SET work_mem = '256MB';
ALTER SYSTEM SET maintenance_work_mem = '1GB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET random_page_cost = 1.1;

-- Restart PostgreSQL to apply changes
SELECT pg_reload_conf();
```

### 2. Redis Cache Optimization
```bash
# Redis configuration
redis-cli CONFIG SET maxmemory 4gb
redis-cli CONFIG SET maxmemory-policy allkeys-lru
redis-cli CONFIG SET save "900 1 300 10 60 10000"
redis-cli CONFIG SET stop-writes-on-bgsave-error no
redis-cli CONFIG SET rdbcompression yes
```

### 3. Application Performance Tuning
```yaml
# config/performance.yaml
performance:
  connection_pools:
    postgresql:
      min_size: 10
      max_size: 50
      max_overflow: 20
      pool_pre_ping: true
    redis:
      max_connections: 100
      retry_on_timeout: true
      socket_keepalive: true
  query_optimization:
    cost_based_optimization: true
    statistics_collection: true
    plan_caching: true
    parallel_execution: true
  caching:
    query_cache:
      enabled: true
      max_entries: 10000
      ttl_seconds: 3600
    result_cache:
      enabled: true
      max_size_mb: 2048
      compression: true
```

### 4. Kubernetes Resource Limits
```yaml
# kubernetes/deployment.yaml (performance sections)
resources:
  requests:
    memory: "8Gi"
    cpu: "4"
    ephemeral-storage: "20Gi"
  limits:
    memory: "16Gi"
    cpu: "8"
    ephemeral-storage: "50Gi"

# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: dvrl-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: dvrl-deployment
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

## Verification

### 1. Health Check Verification
```bash
# Check pod status
kubectl get pods -n apg-dvrl

# Check health endpoint
curl -k https://dvrl.apg.yourcompany.com/health

# Expected response:
{
  "status": "healthy",
  "version": "1.0.0",
  "components": {
    "database": {"status": "healthy"},
    "cache": {"status": "healthy"},
    "apg_integration": {"status": "healthy"}
  }
}
```

### 2. API Endpoint Verification
```bash
# Test authentication
curl -H "Authorization: Bearer ${APG_TOKEN}" \
  https://dvrl.apg.yourcompany.com/api/v1/data-sources

# Test basic query execution
curl -X POST https://dvrl.apg.yourcompany.com/api/v1/queries/sql \
  -H "Authorization: Bearer ${APG_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "sql": "SELECT 1 as test_value",
    "options": {"cache_strategy": "disabled"}
  }'
```

### 3. Integration Tests
```bash
# Run integration test suite
python -m pytest tests/integration/ -v

# Run APG integration tests
python -m pytest tests/apg_integration/ -v

# Run performance tests
python -m pytest tests/performance/ -v --benchmark-only
```

### 4. Load Testing
```bash
# Install load testing tools
pip install locust

# Run load tests
locust -f tests/load/dvrl_load_test.py \
  --host=https://dvrl.apg.yourcompany.com \
  --users=50 \
  --spawn-rate=5 \
  --run-time=300s
```

## Troubleshooting

### Common Installation Issues

#### 1. Database Connection Issues
```bash
# Test database connectivity
python -c "
import asyncio
import asyncpg

async def test_db():
    try:
        conn = await asyncpg.connect(
            host='${DB_HOST}',
            port=${DB_PORT},
            user='${DB_USER}',
            password='${DB_PASSWORD}',
            database='${DB_NAME}'
        )
        print('Database connection successful')
        await conn.close()
    except Exception as e:
        print(f'Database connection failed: {e}')

asyncio.run(test_db())
"
```

#### 2. APG Integration Issues
```bash
# Test APG connectivity
curl -v -H "Authorization: Bearer ${APG_TOKEN}" \
  ${APG_BASE_URL}/api/v1/capabilities/status

# Check APG token validity
python -c "
import jwt
import json
token = '${APG_TOKEN}'
try:
    decoded = jwt.decode(token, options={'verify_signature': False})
    print('Token payload:', json.dumps(decoded, indent=2))
except Exception as e:
    print('Token decode error:', e)
"
```

#### 3. Memory and Performance Issues
```bash
# Check memory usage
kubectl top pods -n apg-dvrl

# Check logs for memory issues
kubectl logs -n apg-dvrl deployment/dvrl-deployment | grep -i "memory\|oom"

# Monitor query performance
curl -H "Authorization: Bearer ${APG_TOKEN}" \
  https://dvrl.apg.yourcompany.com/api/v1/metrics
```

#### 4. SSL Certificate Issues
```bash
# Verify certificate
openssl x509 -in dvrl.crt -text -noout

# Test SSL connectivity
openssl s_client -connect dvrl.apg.yourcompany.com:443 -servername dvrl.apg.yourcompany.com

# Check certificate in Kubernetes
kubectl describe secret dvrl-tls-secret -n apg-dvrl
```

### Log Analysis
```bash
# View application logs
kubectl logs -n apg-dvrl deployment/dvrl-deployment -f

# Filter for errors
kubectl logs -n apg-dvrl deployment/dvrl-deployment | grep ERROR

# Export logs for analysis
kubectl logs -n apg-dvrl deployment/dvrl-deployment --since=1h > dvrl-logs.txt
```

### Recovery Procedures

#### 1. Database Recovery
```bash
# Backup current state
pg_dump -h ${DB_HOST} -U ${DB_USER} dvrl_metadata > dvrl_backup.sql

# Restore from backup (if needed)
psql -h ${DB_HOST} -U ${DB_USER} dvrl_metadata < dvrl_backup.sql

# Reset database (last resort)
python -m dvrl.database.reset --confirm
python -m dvrl.migrations upgrade
```

#### 2. Cache Recovery
```bash
# Clear Redis cache
redis-cli -h ${REDIS_HOST} -p ${REDIS_PORT} FLUSHALL

# Restart cache service
kubectl rollout restart deployment/redis -n apg-platform
```

#### 3. Application Recovery
```bash
# Rolling restart
kubectl rollout restart deployment/dvrl-deployment -n apg-dvrl

# Force pod recreation
kubectl delete pods -l app=dvrl -n apg-dvrl
```

---

For additional support, contact the APG Platform Support team or refer to the [Troubleshooting Guide](troubleshooting_guide.md).

**Document Version**: 1.0  
**Last Updated**: 2025-01-11  
**Author**: APG Platform Team