# APG IMEX Capability - Production Deployment Guide

## Overview

This directory contains production-ready deployment configurations for the APG Import/Export (IMEX) capability. The deployment supports Docker, Kubernetes, and traditional server setups with comprehensive monitoring, security, and scalability features.

## 🚀 Quick Start

### Docker Deployment

```bash
# 1. Generate production configuration
python deployment/production_config.py

# 2. Set environment variables
export DB_PASSWORD="your_secure_password"
export REDIS_PASSWORD="your_redis_password"
export APG_SECRET_KEY="your_secret_key"

# 3. Start services
docker-compose up -d

# 4. Check health
curl http://localhost:8000/health
```

### Kubernetes Deployment

```bash
# 1. Create namespace
kubectl create namespace apg-imex

# 2. Create secrets
kubectl create secret generic apg-secrets \
  --from-literal=db-host=your-db-host \
  --from-literal=db-user=your-db-user \
  --from-literal=db-password=your-db-password \
  -n apg-imex

# 3. Apply manifests
kubectl apply -f deployment/kubernetes.yml -n apg-imex

# 4. Check deployment
kubectl get pods -n apg-imex
```

## 📁 Directory Structure

```
deployment/
├── README.md                 # This file
├── production_config.py      # Configuration management
├── Dockerfile               # Production container image
├── requirements.txt         # Python dependencies
├── wsgi.py                 # WSGI application entry point
├── test_deployment.py      # Deployment tests
├── docker-compose.yml      # Docker Compose configuration (generated)
├── kubernetes.yml          # Kubernetes manifests (generated)
├── nginx.conf             # Nginx configuration (generated)
└── production_config.json # Production settings (generated)
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `APG_ENVIRONMENT` | Deployment environment | `production` | Yes |
| `DB_HOST` | PostgreSQL host | `localhost` | Yes |
| `DB_PORT` | PostgreSQL port | `5432` | No |
| `DB_NAME` | Database name | `apg_imex` | Yes |
| `DB_USER` | Database user | `apg` | Yes |
| `DB_PASSWORD` | Database password | - | Yes |
| `REDIS_HOST` | Redis host | `localhost` | No |
| `REDIS_PORT` | Redis port | `6379` | No |
| `REDIS_PASSWORD` | Redis password | - | No |
| `OLLAMA_HOST` | Ollama AI service host | `localhost` | No |
| `OLLAMA_PORT` | Ollama AI service port | `11434` | No |
| `LOG_LEVEL` | Logging level | `INFO` | No |
| `WORKER_PROCESSES` | Number of workers | `4` | No |
| `WORKER_THREADS` | Threads per worker | `2` | No |

### Security Configuration

```bash
# Generate secure keys
python -c "
from deployment.production_config import generate_secure_keys
keys = generate_secure_keys()
print('APG_SECRET_KEY=' + keys['secret_key'])
print('APG_JWT_SECRET=' + keys['jwt_secret_key'])
print('APG_ENCRYPTION_KEY=' + keys['encryption_key'])
"
```

### Database Setup

```sql
-- Create database and user
CREATE DATABASE apg_imex;
CREATE USER apg_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE apg_imex TO apg_user;

-- Create extensions
\c apg_imex
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
```

## 🐳 Docker Deployment

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 4GB RAM minimum
- 20GB disk space

### Configuration

1. **Generate configuration files:**
   ```bash
   python -c "
   from deployment.production_config import create_production_config, save_deployment_configs
   config = create_production_config('production')
   save_deployment_configs(config, './deployment')
   "
   ```

2. **Set environment variables:**
   ```bash
   # Create .env file
   cat > .env << EOF
   APG_ENVIRONMENT=production
   DB_PASSWORD=your_secure_password
   REDIS_PASSWORD=your_redis_password
   APG_SECRET_KEY=your_generated_secret_key
   APG_JWT_SECRET=your_generated_jwt_secret
   EOF
   ```

3. **Deploy services:**
   ```bash
   docker-compose --env-file .env up -d
   ```

### Monitoring

Access monitoring dashboards:

- **Application**: http://localhost:8000
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Metrics**: http://localhost:9090/metrics

## ☸️ Kubernetes Deployment

### Prerequisites

- Kubernetes 1.20+
- kubectl configured
- Ingress controller (nginx)
- Cert-manager for TLS
- Storage class `fast-ssd`

### Configuration

1. **Create namespace:**
   ```bash
   kubectl create namespace apg-imex
   ```

2. **Create secrets:**
   ```bash
   kubectl create secret generic apg-secrets \
     --from-literal=db-host=postgres.example.com \
     --from-literal=db-user=apg_user \
     --from-literal=db-password=secure_password \
     --from-literal=secret-key=your_secret_key \
     --from-literal=jwt-secret=your_jwt_secret \
     -n apg-imex
   ```

3. **Deploy PostgreSQL (if needed):**
   ```bash
   helm install postgres bitnami/postgresql \
     --set auth.postgresPassword=secure_password \
     --set auth.database=apg_imex \
     --set primary.persistence.size=100Gi \
     -n apg-imex
   ```

4. **Deploy IMEX application:**
   ```bash
   kubectl apply -f deployment/kubernetes.yml -n apg-imex
   ```

### Scaling

```bash
# Scale application
kubectl scale deployment apg-imex --replicas=5 -n apg-imex

# Horizontal Pod Autoscaling
kubectl autoscale deployment apg-imex \
  --cpu-percent=70 \
  --min=3 \
  --max=10 \
  -n apg-imex
```

## 🔒 Security

### SSL/TLS Configuration

#### Let's Encrypt (Automatic)
```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Create ClusterIssuer
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: nyimbi@gmail.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

#### Manual Certificate
```bash
# Generate self-signed certificate (development only)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout tls.key -out tls.crt \
  -subj "/CN=imex.apg.datacraft.co.ke"

kubectl create secret tls apg-imex-tls \
  --key=tls.key --cert=tls.crt -n apg-imex
```

### Network Policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: apg-imex-network-policy
  namespace: apg-imex
spec:
  podSelector:
    matchLabels:
      app: apg-imex
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
```

## 📊 Monitoring & Observability

### Health Checks

| Endpoint | Purpose | Response |
|----------|---------|-----------|
| `/health` | Liveness probe | Service health status |
| `/ready` | Readiness probe | Service readiness |
| `/metrics` | Prometheus metrics | Application metrics |
| `/info` | Application info | Version and features |

### Metrics

Key metrics collected:

- **System**: CPU, memory, disk, network usage
- **Application**: Request latency, throughput, errors
- **Business**: Jobs processed, data volume, success rate
- **Security**: Authentication attempts, rate limits

### Logging

Structured JSON logging with fields:

```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "level": "INFO",
  "logger": "imex.service",
  "message": "Job completed successfully",
  "job_id": "job-123",
  "tenant_id": "tenant-456",
  "duration_ms": 1500,
  "records_processed": 10000
}
```

### Alerting

Configure alerts for:

- **Critical**: Service down, database disconnected
- **Warning**: High error rate, resource exhaustion
- **Info**: Job completion, performance degradation

## 🔧 Maintenance

### Database Migrations

```bash
# Run migrations in container
docker exec -it apg-imex-app python -c "
import asyncio
from database import DatabaseManager, DatabaseConfig
# Run migration logic
"

# Or in Kubernetes
kubectl exec -it deployment/apg-imex -n apg-imex -- \
  python -c "import asyncio; ..."
```

### Backup & Recovery

#### Database Backup
```bash
# PostgreSQL backup
pg_dump -h $DB_HOST -U $DB_USER apg_imex > backup_$(date +%Y%m%d).sql

# Kubernetes backup
kubectl exec postgres-0 -n apg-imex -- \
  pg_dump -U postgres apg_imex > backup.sql
```

#### Application Data Backup
```bash
# Backup upload files
tar -czf uploads_backup_$(date +%Y%m%d).tar.gz /opt/apg/uploads/

# Kubernetes persistent volume backup
kubectl cp apg-imex/apg-imex-pod:/opt/apg/uploads ./uploads_backup/
```

### Log Rotation

```bash
# Configure logrotate
cat > /etc/logrotate.d/apg-imex << EOF
/opt/apg/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0644 apg apg
    postrotate
        systemctl reload apg-imex
    endscript
}
EOF
```

## ⚡ Performance Tuning

### Database Optimization

```sql
-- PostgreSQL configuration
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.7;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
SELECT pg_reload_conf();
```

### Application Tuning

```bash
# Environment variables for performance
export WORKER_PROCESSES=8
export WORKER_THREADS=4
export BATCH_SIZE=5000
export MEMORY_LIMIT_MB=4096
export PARALLEL_JOBS=8
```

### Resource Limits

#### Kubernetes Resources
```yaml
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

#### Docker Resources
```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
    reservations:
      cpus: '0.5'
      memory: 1G
```

## 🚨 Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check logs
docker-compose logs apg-imex
kubectl logs deployment/apg-imex -n apg-imex

# Check configuration
python -c "
from deployment.production_config import create_production_config
config = create_production_config()
print(config.model_dump_json(indent=2))
"
```

#### Database Connection Issues
```bash
# Test database connection
python -c "
import asyncio
import asyncpg
async def test():
    conn = await asyncpg.connect(
        host='localhost', port=5432,
        database='apg_imex', user='apg_user', password='password'
    )
    print('Connected successfully')
    await conn.close()
asyncio.run(test())
"
```

#### Performance Issues
```bash
# Check system resources
curl http://localhost:8000/metrics | grep -E "(cpu|memory|disk)"

# Check job performance
curl http://localhost:8000/api/v1/secure/imex/jobs | jq '.data.jobs[].status'
```

### Health Check Debug
```bash
# Detailed health check
curl -s http://localhost:8000/health | jq '.'

# Component status
curl -s http://localhost:8000/info | jq '.features'
```

## 📞 Support

- **Documentation**: Internal wiki
- **Issues**: GitHub Issues
- **Email**: nyimbi@gmail.com
- **Website**: www.datacraft.co.ke

## 📄 License

© 2025 Datacraft. All rights reserved.