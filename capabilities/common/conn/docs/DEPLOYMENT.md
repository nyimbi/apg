# APG Connection Management - Deployment Guide

**Author:** Nyimbi Odero
**Company:** Datacraft
**Copyright:** © 2025

This guide provides comprehensive instructions for deploying the APG Connection Management capability in production environments.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Environment Configuration](#environment-configuration)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Monitoring & Observability](#monitoring--observability)
- [Security Configuration](#security-configuration)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)
- [Maintenance](#maintenance)

## Overview

The APG Connection Management capability is designed for high-availability production deployments with:

- **Horizontal scaling** with load balancing
- **Database clustering** with PostgreSQL
- **Caching layer** with Redis
- **Monitoring** with Prometheus and Grafana
- **Tracing** with OpenTelemetry
- **Security** with RBAC, encryption, and secrets management
- **Auto-scaling** based on CPU and memory metrics

### Architecture Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   API Gateway   │    │   Monitoring    │
│   (Nginx/HAProxy)│    │   (Optional)    │    │  (Prometheus)   │
└─────────┬───────┘    └─────────┬───────┘    └─────────────────┘
          │                      │
    ┌─────┴─────┐           ┌────┴────┐
    │    APG    │           │   APG   │
    │   Conn    │  ◄──────► │  Conn   │  (Multiple Instances)
    │ Instance  │           │Instance │
    └─────┬─────┘           └────┬────┘
          │                      │
    ┌─────┴─────────────────────┴─────┐
    │        Shared Services          │
    ├─────────────┬───────────────────┤
    │ PostgreSQL  │      Redis        │
    │ (Clustered) │    (Caching)      │
    └─────────────┴───────────────────┘
```

## Prerequisites

### System Requirements

**Minimum Production Requirements:**
- **CPU:** 4 cores per instance
- **Memory:** 8GB RAM per instance
- **Storage:** 100GB SSD for application + database
- **Network:** 1Gbps bandwidth
- **OS:** Linux (Ubuntu 20.04+, RHEL 8+, or equivalent)

**Recommended Production Setup:**
- **CPU:** 8+ cores per instance
- **Memory:** 16GB+ RAM per instance
- **Storage:** 500GB+ NVMe SSD
- **Network:** 10Gbps bandwidth
- **Load Balancer:** Dedicated hardware or cloud load balancer

### Software Dependencies

- **Container Runtime:** Docker 20.10+ or containerd 1.6+
- **Orchestration:** Kubernetes 1.24+ (for K8s deployment)
- **Database:** PostgreSQL 13+
- **Cache:** Redis 6.0+
- **Monitoring:** Prometheus 2.35+, Grafana 9.0+
- **Proxy:** Nginx 1.20+ or HAProxy 2.4+

### Network Requirements

**Inbound Ports:**
- `80/tcp` - HTTP (redirects to HTTPS)
- `443/tcp` - HTTPS
- `8000/tcp` - Application port (internal)
- `9090/tcp` - Prometheus (monitoring network)
- `3000/tcp` - Grafana (monitoring network)

**Outbound Ports:**
- `443/tcp` - HTTPS for external APIs
- `587/tcp` - SMTP for notifications
- `53/tcp` - DNS resolution

## Environment Configuration

### Required Environment Variables

Create a `.env` file or configure environment variables:

```bash
# Application Configuration
APG_ENV=production
APG_LOG_LEVEL=INFO
APG_WORKERS=4
APG_HOST=0.0.0.0
APG_PORT=8000

# Database Configuration
APG_DB_HOST=your-postgres-host
APG_DB_PORT=5432
APG_DB_NAME=apg
APG_DB_USER=apg
APG_DB_PASSWORD=your-secure-password
APG_DB_SSL_MODE=require

# Cache Configuration
APG_REDIS_HOST=your-redis-host
APG_REDIS_PORT=6379
APG_REDIS_PASSWORD=your-redis-password
APG_REDIS_DB=0

# Security Configuration
APG_JWT_SECRET=your-256-bit-secret-key
APG_ENCRYPTION_KEY=your-base64-encoded-key
APG_ALLOWED_HOSTS=your-domain.com,api.your-domain.com
APG_CORS_ORIGINS=https://your-frontend.com

# Monitoring Configuration
APG_ENABLE_METRICS=true
APG_ENABLE_TRACING=true
APG_OTEL_ENDPOINT=http://otel-collector:4317
APG_OTEL_SERVICE_NAME=apg-connection-management

# External Services
APG_SMTP_HOST=smtp.your-domain.com
APG_SMTP_PORT=587
APG_SMTP_USER=your-smtp-user
APG_SMTP_PASSWORD=your-smtp-password
APG_NOTIFICATION_EMAIL=admin@your-domain.com

# Feature Flags
APG_ENABLE_API_DOCS=true
APG_ENABLE_ADMIN_UI=true
APG_ENABLE_HEALTH_CHECKS=true
APG_ENABLE_RATE_LIMITING=true
```

### Security Secrets Management

**For Docker Compose:**
Create secret files in `deployment/docker/secrets/`:

```bash
# Create secrets directory
mkdir -p deployment/docker/secrets

# Database password
echo "your-secure-db-password" > deployment/docker/secrets/db_password.txt

# JWT secret (generate with: openssl rand -hex 32)
echo "your-jwt-secret-key" > deployment/docker/secrets/jwt_secret.txt

# Encryption key (generate with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
echo "your-encryption-key" > deployment/docker/secrets/encryption_key.txt

# Grafana admin password
echo "your-grafana-password" > deployment/docker/secrets/grafana_password.txt

# Set proper permissions
chmod 600 deployment/docker/secrets/*.txt
```

**For Kubernetes:**
```bash
# Create namespace
kubectl create namespace apg-conn

# Create secrets
kubectl create secret generic apg-conn-secrets \
  --from-literal=db_password="your-secure-db-password" \
  --from-literal=jwt_secret="your-jwt-secret-key" \
  --from-literal=encryption_key="your-encryption-key" \
  -n apg-conn
```

## Docker Deployment

### Quick Start with Docker Compose

1. **Clone and prepare the deployment:**
```bash
cd capabilities/common/conn/deployment/docker
cp docker-compose.yml docker-compose.prod.yml
```

2. **Configure environment:**
```bash
# Edit environment variables in docker-compose.prod.yml
vim docker-compose.prod.yml
```

3. **Create secrets:**
```bash
./create-secrets.sh
```

4. **Deploy the stack:**
```bash
docker-compose -f docker-compose.prod.yml up -d
```

5. **Verify deployment:**
```bash
# Check all services are running
docker-compose -f docker-compose.prod.yml ps

# Check application health
curl http://localhost:8000/monitoring/api/health

# View logs
docker-compose -f docker-compose.prod.yml logs -f apg-conn
```

### Production Docker Deployment

**1. Build Production Image:**
```bash
# Build the image
docker build -t datacraft/apg-connection-mgmt:1.0.0 -f deployment/docker/Dockerfile .

# Tag for registry
docker tag datacraft/apg-connection-mgmt:1.0.0 your-registry.com/apg-connection-mgmt:1.0.0

# Push to registry
docker push your-registry.com/apg-connection-mgmt:1.0.0
```

**2. Database Setup:**
```bash
# Start PostgreSQL
docker run -d \
  --name apg-postgres \
  --network apg-network \
  -e POSTGRES_DB=apg \
  -e POSTGRES_USER=apg \
  -e POSTGRES_PASSWORD_FILE=/run/secrets/db_password \
  -v postgres-data:/var/lib/postgresql/data \
  -v ./init-db.sql:/docker-entrypoint-initdb.d/init-db.sql:ro \
  --secret db_password \
  postgres:15-alpine

# Wait for database to be ready
docker exec apg-postgres pg_isready -U apg -d apg
```

**3. Application Deployment:**
```bash
# Deploy application
docker run -d \
  --name apg-conn-app \
  --network apg-network \
  -p 8000:8000 \
  -e APG_DB_HOST=apg-postgres \
  -e APG_DB_PASSWORD_FILE=/run/secrets/db_password \
  -v apg-logs:/app/logs \
  -v apg-data:/app/data \
  --secret db_password \
  --secret jwt_secret \
  --secret encryption_key \
  --health-cmd="/healthcheck.sh" \
  --health-interval=30s \
  --health-timeout=10s \
  --health-retries=3 \
  your-registry.com/apg-connection-mgmt:1.0.0
```

**4. Load Balancer Setup:**
```bash
# Deploy Nginx
docker run -d \
  --name apg-nginx \
  --network apg-network \
  -p 80:80 \
  -p 443:443 \
  -v ./nginx.conf:/etc/nginx/nginx.conf:ro \
  -v ./ssl:/etc/nginx/ssl:ro \
  nginx:1.24-alpine
```

## Kubernetes Deployment

### Prerequisites

1. **Kubernetes cluster** (1.24+) with:
   - StorageClass configured
   - Ingress controller (nginx recommended)
   - Cert-manager for TLS (optional but recommended)
   - Monitoring stack (Prometheus Operator)

2. **kubectl** configured for your cluster

3. **Container registry** access

### Deployment Steps

**1. Prepare the cluster:**
```bash
# Create namespace
kubectl apply -f deployment/kubernetes/namespace.yaml

# Apply RBAC
kubectl apply -f deployment/kubernetes/rbac.yaml

# Create secrets
kubectl apply -f deployment/kubernetes/secrets.yaml
```

**2. Deploy dependencies:**
```bash
# Deploy PostgreSQL (or use cloud-managed database)
kubectl apply -f deployment/kubernetes/postgres.yaml

# Deploy Redis (or use cloud-managed cache)
kubectl apply -f deployment/kubernetes/redis.yaml

# Wait for databases to be ready
kubectl wait --for=condition=ready pod -l app=postgres -n apg-conn --timeout=300s
kubectl wait --for=condition=ready pod -l app=redis -n apg-conn --timeout=300s
```

**3. Deploy the application:**
```bash
# Apply all Kubernetes manifests
kubectl apply -f deployment/kubernetes/

# Wait for deployment to be ready
kubectl wait --for=condition=available deployment/apg-conn-deployment -n apg-conn --timeout=600s

# Check pods are running
kubectl get pods -n apg-conn
```

**4. Configure ingress:**
```bash
# Update ingress hostname in deployment.yaml
sed -i 's/conn.apg.yourdomain.com/your-actual-domain.com/g' deployment/kubernetes/deployment.yaml

# Apply ingress configuration
kubectl apply -f deployment/kubernetes/ingress.yaml

# Get ingress IP
kubectl get ingress -n apg-conn
```

**5. Verify deployment:**
```bash
# Check all resources
kubectl get all -n apg-conn

# Check application health
kubectl port-forward svc/apg-conn-service 8000:8000 -n apg-conn &
curl http://localhost:8000/monitoring/api/health

# View logs
kubectl logs -f deployment/apg-conn-deployment -n apg-conn
```

### Scaling and Updates

**Horizontal Scaling:**
```bash
# Manual scaling
kubectl scale deployment apg-conn-deployment --replicas=5 -n apg-conn

# Auto-scaling is configured via HPA in deployment.yaml
kubectl get hpa -n apg-conn
```

**Rolling Updates:**
```bash
# Update image
kubectl set image deployment/apg-conn-deployment apg-conn=your-registry.com/apg-connection-mgmt:1.1.0 -n apg-conn

# Check rollout status
kubectl rollout status deployment/apg-conn-deployment -n apg-conn

# Rollback if needed
kubectl rollout undo deployment/apg-conn-deployment -n apg-conn
```

## Monitoring & Observability

### Prometheus Configuration

The deployment includes Prometheus monitoring with:

- **Application metrics** via `/monitoring/api/metrics/prometheus`
- **System metrics** via node-exporter
- **Container metrics** via cAdvisor
- **Custom business metrics** from the application

**Key Metrics to Monitor:**
```prometheus
# Application Performance
apg_request_duration_seconds
apg_requests_total
apg_active_connections
apg_data_processed_bytes
apg_errors_total

# System Metrics
container_cpu_usage_seconds_total
container_memory_usage_bytes
container_network_transmit_bytes_total

# Database Metrics
postgresql_connections_total
postgresql_queries_total
redis_connected_clients
```

### Grafana Dashboards

Pre-configured dashboards are included for:

1. **Application Overview** - Request rates, response times, error rates
2. **Connection Management** - Active connections, flow executions, data throughput
3. **System Resources** - CPU, memory, disk, network usage
4. **Database Performance** - Query performance, connection pools
5. **Business Metrics** - Data lineage updates, capability compositions

**Access Grafana:**
```bash
# Get Grafana admin password
kubectl get secret grafana-admin -n apg-conn -o jsonpath="{.data.admin-password}" | base64 -d

# Port forward to access UI
kubectl port-forward svc/grafana 3000:3000 -n apg-conn
# Open http://localhost:3000
```

### Alerting Rules

Critical alerts are configured for:

- **High error rate** (>5% for 5 minutes)
- **High response time** (>2s p95 for 5 minutes)
- **Database connectivity** (connection failures)
- **Memory usage** (>90% for 10 minutes)
- **Disk space** (>85% used)
- **Pod crashes** (restart loops)

### Distributed Tracing

OpenTelemetry tracing provides:

- **Request tracing** across service boundaries
- **Database query tracing** with performance metrics
- **External API call tracing**
- **Custom span creation** for business operations

**View traces:**
```bash
# Access Jaeger UI (if deployed)
kubectl port-forward svc/jaeger-query 16686:16686 -n monitoring
# Open http://localhost:16686
```

## Security Configuration

### SSL/TLS Setup

**1. Generate certificates:**
```bash
# Using Let's Encrypt with cert-manager (Kubernetes)
kubectl apply -f https://github.com/jetstack/cert-manager/releases/download/v1.11.0/cert-manager.yaml

# Create ClusterIssuer
kubectl apply -f deployment/kubernetes/cert-manager.yaml
```

**2. Configure TLS termination:**
- **Docker:** Configure nginx with SSL certificates
- **Kubernetes:** Use cert-manager with ingress annotations

### Access Control

**1. Role-Based Access Control (RBAC):**
```bash
# Users and roles are managed through the application
# Default roles: viewer, operator, admin
# Create admin user:
kubectl exec -it pod/apg-conn-deployment-xxx -n apg-conn -- python -c "
from security import auth_manager
admin_user = auth_manager.create_user(
    username='admin',
    email='admin@your-domain.com',
    password='secure-admin-password',
    tenant_id='system',
    roles=['admin'],
    is_admin=True
)
print(f'Created admin user: {admin_user.username}')
"
```

**2. Network Policies:**
```bash
# Apply network policies (Kubernetes)
kubectl apply -f deployment/kubernetes/network-policy.yaml
```

**3. Secrets Encryption:**
All sensitive data is encrypted at rest using:
- **Database:** PostgreSQL native encryption
- **Application:** Custom encryption for connection credentials
- **Kubernetes:** etcd encryption at rest

### Security Scanning

**1. Container scanning:**
```bash
# Scan Docker image for vulnerabilities
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image datacraft/apg-connection-mgmt:1.0.0
```

**2. Dependency scanning:**
```bash
# Scan Python dependencies
pip-audit --desc --format=json
```

## Performance Tuning

### Database Optimization

**PostgreSQL Configuration:**
```sql
-- Connection pooling
max_connections = 200
shared_buffers = 1GB
effective_cache_size = 3GB
work_mem = 16MB
maintenance_work_mem = 256MB

-- Write performance
wal_buffers = 32MB
checkpoint_completion_target = 0.7
max_wal_size = 2GB

-- Query optimization
default_statistics_target = 1000
random_page_cost = 1.1
effective_io_concurrency = 200
```

**Connection Pooling:**
```bash
# Use PgBouncer for connection pooling
docker run -d \
  --name pgbouncer \
  --network apg-network \
  -p 5432:5432 \
  -e DATABASE_URL=postgres://apg:password@postgres:5432/apg \
  -e POOL_MODE=transaction \
  -e MAX_CLIENT_CONN=200 \
  -e DEFAULT_POOL_SIZE=25 \
  pgbouncer/pgbouncer:latest
```

### Redis Configuration

```bash
# Redis optimization for caching
maxmemory 2gb
maxmemory-policy allkeys-lru
tcp-keepalive 60
timeout 300

# Persistence configuration
save 900 1
save 300 10
save 60 10000
appendonly yes
appendfsync everysec
```

### Application Tuning

**Gunicorn Configuration:**
```python
# gunicorn.conf.py
bind = "0.0.0.0:8000"
workers = 4  # 2 * CPU cores
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
preload_app = True
timeout = 120
keepalive = 2
```

**Memory Management:**
```bash
# Set memory limits
export MALLOC_ARENA_MAX=2
export PYTHONMALLOC=pymalloc_debug  # Development only
```

### Load Testing

```bash
# Use k6 for load testing
k6 run deployment/testing/load-test.js

# Or use Apache Bench
ab -n 10000 -c 100 http://your-domain.com/monitoring/api/health
```

## Troubleshooting

### Common Issues

**1. Application won't start:**
```bash
# Check logs
docker logs apg-conn-app
kubectl logs -f deployment/apg-conn-deployment -n apg-conn

# Common causes:
# - Database connection failed
# - Missing environment variables
# - Invalid secrets
# - Port conflicts
```

**2. Database connection errors:**
```bash
# Test database connectivity
psql -h $APG_DB_HOST -p $APG_DB_PORT -U $APG_DB_USER -d $APG_DB_NAME -c "SELECT version();"

# Check network connectivity
telnet $APG_DB_HOST $APG_DB_PORT

# Verify credentials
echo $APG_DB_PASSWORD
```

**3. High memory usage:**
```bash
# Monitor memory usage
docker stats apg-conn-app
kubectl top pods -n apg-conn

# Check for memory leaks
python -m memory_profiler app.py
```

**4. SSL/TLS issues:**
```bash
# Test SSL certificate
openssl s_client -connect your-domain.com:443 -servername your-domain.com

# Check certificate expiry
openssl x509 -in /path/to/cert.pem -noout -dates
```

### Debug Mode

**Enable debug logging:**
```bash
# Set environment variable
export APG_LOG_LEVEL=DEBUG

# Or in Kubernetes
kubectl set env deployment/apg-conn-deployment APG_LOG_LEVEL=DEBUG -n apg-conn
```

**Access debug endpoints:**
```bash
# Health check with details
curl http://localhost:8000/monitoring/api/health

# Metrics endpoint
curl http://localhost:8000/monitoring/api/metrics

# Application info
curl http://localhost:8000/monitoring/api/info
```

### Performance Debugging

**1. Slow queries:**
```sql
-- Enable slow query logging in PostgreSQL
ALTER SYSTEM SET log_min_duration_statement = 1000;  -- Log queries > 1s
SELECT pg_reload_conf();

-- View slow queries
SELECT query, mean_time, calls FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;
```

**2. Memory profiling:**
```bash
# Profile memory usage
python -m memory_profiler -T 0.1 app.py

# Heap profiling
python -m heapprofile app.py
```

**3. CPU profiling:**
```bash
# Profile CPU usage
python -m cProfile -o profile.stats app.py
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"
```

## Maintenance

### Backup and Recovery

**1. Database backup:**
```bash
# Create backup
pg_dump -h $APG_DB_HOST -U $APG_DB_USER -d $APG_DB_NAME > backup_$(date +%Y%m%d_%H%M%S).sql

# Automated backup script
#!/bin/bash
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump -h $APG_DB_HOST -U $APG_DB_USER -d $APG_DB_NAME | gzip > $BACKUP_DIR/apg_backup_$DATE.sql.gz

# Keep only last 30 days of backups
find $BACKUP_DIR -name "apg_backup_*.sql.gz" -mtime +30 -delete
```

**2. Application data backup:**
```bash
# Backup persistent volumes (Kubernetes)
kubectl get pv

# Create volume snapshot
kubectl apply -f - <<EOF
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshot
metadata:
  name: apg-data-snapshot-$(date +%Y%m%d)
  namespace: apg-conn
spec:
  source:
    persistentVolumeClaimName: apg-conn-data-pvc
EOF
```

**3. Disaster recovery:**
```bash
# Restore from backup
psql -h $APG_DB_HOST -U $APG_DB_USER -d $APG_DB_NAME < backup_20250101_120000.sql

# Restore persistent volume from snapshot
kubectl apply -f deployment/kubernetes/restore-pvc.yaml
```

### Updates and Patches

**1. Application updates:**
```bash
# Rolling update (Kubernetes)
kubectl set image deployment/apg-conn-deployment apg-conn=datacraft/apg-connection-mgmt:1.1.0 -n apg-conn

# Docker Compose update
docker-compose pull
docker-compose up -d
```

**2. Database migrations:**
```bash
# Run migrations
kubectl exec -it pod/apg-conn-deployment-xxx -n apg-conn -- alembic upgrade head

# Rollback migration
kubectl exec -it pod/apg-conn-deployment-xxx -n apg-conn -- alembic downgrade -1
```

**3. Security updates:**
```bash
# Update base images
docker build --no-cache -t datacraft/apg-connection-mgmt:1.0.1 .

# Scan for vulnerabilities
trivy image datacraft/apg-connection-mgmt:1.0.1
```

### Monitoring and Alerting

**1. Health monitoring:**
```bash
# Set up health check monitoring
curl -f http://localhost:8000/monitoring/api/health || exit 1

# Kubernetes liveness/readiness probes are configured automatically
```

**2. Log rotation:**
```bash
# Configure logrotate
cat > /etc/logrotate.d/apg-conn <<EOF
/app/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    create 0644 apg apg
    postrotate
        docker kill -s USR1 apg-conn-app
    endscript
}
EOF
```

**3. Performance monitoring:**
```bash
# Monitor key metrics
watch -n 5 'curl -s http://localhost:8000/monitoring/api/metrics | grep -E "(requests_total|response_time|active_connections)"'
```

### Scaling Operations

**1. Vertical scaling:**
```bash
# Increase resources (Kubernetes)
kubectl patch deployment apg-conn-deployment -n apg-conn -p '{"spec":{"template":{"spec":{"containers":[{"name":"apg-conn","resources":{"requests":{"memory":"1Gi","cpu":"500m"},"limits":{"memory":"2Gi","cpu":"2000m"}}}]}}}}'
```

**2. Horizontal scaling:**
```bash
# Scale replicas
kubectl scale deployment apg-conn-deployment --replicas=6 -n apg-conn

# Verify auto-scaling
kubectl get hpa -n apg-conn -w
```

**3. Database scaling:**
```bash
# Scale PostgreSQL (depends on your setup)
# - Read replicas for read scaling
# - Connection pooling for connection scaling
# - Partitioning for data scaling
```

---

## Support and Documentation

For additional support and documentation:

- **Internal Documentation:** [APG Platform Docs](https://docs.internal.datacraft.co.ke)
- **API Documentation:** Available at `/docs` when deployed
- **Monitoring Dashboards:** Available in Grafana
- **Support:** Contact the APG Platform Team

---

**Deployment Guide Version:** 1.0.0
**Last Updated:** January 2025
**Next Review:** April 2025