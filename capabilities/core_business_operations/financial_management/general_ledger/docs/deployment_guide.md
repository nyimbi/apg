# APG General Ledger - Deployment Guide
**Revolutionary AI-powered General Ledger System**  
© 2025 Datacraft. All rights reserved.

## 🚀 Quick Start Deployment

### Prerequisites
- Docker 20.10+
- Docker Compose 2.0+
- Kubernetes 1.24+ (for production)
- kubectl configured (for K8s deployment)

### 🐳 Docker Compose Deployment (Recommended for Development/Testing)

**1. Clone and Setup**
```bash
git clone https://github.com/datacraft/apg-general-ledger.git
cd apg-general-ledger
```

**2. Configure Environment**
```bash
# Copy example environment file
cp .env.example .env

# Edit configuration
nano .env
```

**Required Environment Variables:**
```bash
# Database Configuration
DB_NAME=apg_gl
DB_USER=gl_user
DB_PASSWORD=your_secure_password

# Application Secrets
SECRET_KEY=your_super_secret_key_here
JWT_SECRET=your_jwt_secret_key_here

# APG Platform Integration
APG_PLATFORM_URL=https://platform.company.com
APG_API_KEY=your-apg-api-key

# AI Configuration
OPENAI_API_KEY=your-openai-api-key
AI_MODEL=gpt-4
AI_CONFIDENCE_THRESHOLD=0.8

# Environment
ENVIRONMENT=production
LOG_LEVEL=INFO

# Monitoring
GRAFANA_PASSWORD=your_grafana_password
```

**3. Deploy Services**
```bash
# Deploy all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f gl-service
```

**4. Verify Deployment**
```bash
# Health check
curl http://localhost/health

# API check
curl http://localhost/api/v1/accounts

# Access dashboards
open http://localhost        # Application
open http://localhost:3000   # Grafana (admin/admin)
open http://localhost:5601   # Kibana
```

## ☸️ Kubernetes Deployment (Production)

### 1. Prepare Kubernetes Cluster

**Create Namespace**
```bash
kubectl create namespace apg-general-ledger
```

**Configure Secrets**
```bash
# Database secret
kubectl create secret generic postgres-secret \
  --from-literal=password=your_secure_db_password \
  -n apg-general-ledger

# Application secrets
kubectl create secret generic gl-secrets \
  --from-literal=database-url="postgresql://gl_user:password@postgres:5432/apg_gl" \
  --from-literal=redis-url="redis://redis:6379" \
  --from-literal=secret-key="your-super-secret-key" \
  --from-literal=jwt-secret="your-jwt-secret" \
  --from-literal=openai-api-key="your-openai-api-key" \
  --from-literal=apg-api-key="your-apg-api-key" \
  -n apg-general-ledger
```

### 2. Deploy Infrastructure

**Deploy PostgreSQL**
```bash
kubectl apply -f k8s/postgres.yaml -n apg-general-ledger
```

**Deploy Redis**
```bash
kubectl apply -f k8s/redis.yaml -n apg-general-ledger
```

**Deploy Application**
```bash
kubectl apply -f k8s/deployment.yaml -n apg-general-ledger
```

### 3. Configure Ingress (Optional)

**NGINX Ingress**
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: gl-ingress
  namespace: apg-general-ledger
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - gl.yourcompany.com
    secretName: gl-tls
  rules:
  - host: gl.yourcompany.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: general-ledger
            port:
              number: 80
```

### 4. Verify Kubernetes Deployment

**Check Pod Status**
```bash
kubectl get pods -n apg-general-ledger
kubectl logs -f deployment/general-ledger -n apg-general-ledger
```

**Port Forward for Testing**
```bash
kubectl port-forward svc/general-ledger 8080:80 -n apg-general-ledger
curl http://localhost:8080/health
```

## 🔧 Advanced Configuration

### SSL/TLS Configuration

**Generate SSL Certificates**
```bash
# For development (self-signed)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/server.key \
  -out nginx/ssl/server.crt \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"

# For production (Let's Encrypt)
certbot certonly --webroot -w /var/www/html -d yourdomain.com
```

**NGINX SSL Configuration**
```nginx
server {
    listen 443 ssl http2;
    server_name yourdomain.com;
    
    ssl_certificate /etc/nginx/ssl/server.crt;
    ssl_certificate_key /etc/nginx/ssl/server.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    
    location / {
        proxy_pass http://gl-service:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Database Optimization

**PostgreSQL Configuration**
```bash
# Create optimized postgresql.conf
cat > postgres/postgresql.conf << EOF
# Memory Configuration
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB

# Checkpoint Configuration
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100

# Connection Configuration
max_connections = 200
EOF
```

**Database Tuning Script**
```sql
-- Create indexes for performance
CREATE INDEX CONCURRENTLY idx_gl_entries_tenant_date 
ON gl_journal_entries(tenant_id, entry_date);

CREATE INDEX CONCURRENTLY idx_gl_lines_account_date 
ON gl_journal_entry_lines(account_id, created_at);

-- Analyze tables for query optimization
ANALYZE gl_journal_entries;
ANALYZE gl_journal_entry_lines;
ANALYZE gl_accounts;
```

### Monitoring Setup

**Prometheus Configuration**
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'gl-service'
    static_configs:
      - targets: ['gl-service:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
```

**Grafana Dashboards**
```bash
# Import pre-built dashboards
curl -X POST \
  http://admin:admin@localhost:3000/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @monitoring/grafana/gl-dashboard.json
```

### Backup Configuration

**Automated Backup Script**
```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
DB_HOST="database"
DB_NAME="apg_gl"
DB_USER="gl_user"

# Database backup
pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME \
  | gzip > $BACKUP_DIR/gl_backup_$DATE.sql.gz

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -name "gl_backup_*.sql.gz" -mtime +30 -delete

echo "Backup completed: gl_backup_$DATE.sql.gz"
```

**Backup Cron Job**
```bash
# Add to crontab
0 2 * * * /app/scripts/backup.sh >> /var/log/backup.log 2>&1
```

## 🔍 Troubleshooting

### Common Issues

**1. Service Won't Start**
```bash
# Check logs
docker-compose logs gl-service

# Common causes:
# - Database connection issues
# - Missing environment variables
# - Port conflicts
```

**2. Database Connection Errors**
```bash
# Test database connectivity
docker-compose exec gl-service python -c "
from models import db
print('Database connection:', db.engine.execute('SELECT 1').scalar())
"
```

**3. AI Features Not Working**
```bash
# Check OpenAI API key
curl -H "Authorization: Bearer YOUR_API_KEY" \
  https://api.openai.com/v1/models

# Verify AI service logs
docker-compose logs gl-service | grep -i "ai\|openai"
```

**4. Performance Issues**
```bash
# Check resource usage
docker stats

# Monitor database performance
docker-compose exec database psql -U gl_user -d apg_gl -c "
SELECT query, calls, total_time, mean_time 
FROM pg_stat_statements 
ORDER BY total_time DESC LIMIT 10;
"
```

### Health Check Endpoints

**System Health**
```bash
curl http://localhost/health
# Returns: {"status": "healthy", "checks": {...}}
```

**Database Health**
```bash
curl http://localhost/health/database
# Returns: {"status": "healthy", "response_time_ms": 10}
```

**AI Service Health**
```bash
curl http://localhost/health/ai
# Returns: {"status": "healthy", "model": "gpt-4"}
```

## 🔄 Updates and Maintenance

### Rolling Updates

**Docker Compose Update**
```bash
# Pull latest images
docker-compose pull

# Rolling update with zero downtime
docker-compose up -d --no-deps gl-service
```

**Kubernetes Rolling Update**
```bash
# Update image
kubectl set image deployment/general-ledger \
  general-ledger=apg/general-ledger:v1.1.0 \
  -n apg-general-ledger

# Monitor rollout
kubectl rollout status deployment/general-ledger -n apg-general-ledger

# Rollback if needed
kubectl rollout undo deployment/general-ledger -n apg-general-ledger
```

### Database Migrations

**Run Migrations**
```bash
# Docker Compose
docker-compose exec gl-service python scripts/migrate.py

# Kubernetes
kubectl run migration-job \
  --image=apg/general-ledger:latest \
  --restart=Never \
  --rm -i \
  --namespace=apg-general-ledger \
  -- python scripts/migrate.py
```

### Maintenance Tasks

**Daily Tasks**
```bash
# Log rotation
logrotate /etc/logrotate.d/gl-service

# Cache cleanup
redis-cli FLUSHDB

# Database maintenance
VACUUM ANALYZE;
```

**Weekly Tasks**
```bash
# Database backup verification
pg_restore --list backup.sql.gz | head -10

# Performance tuning
pg_stat_reset();
```

## 📊 Performance Tuning

### Application Tuning

**Gunicorn Configuration**
```python
# gunicorn.conf.py
bind = "0.0.0.0:8000"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
preload_app = True
timeout = 30
keepalive = 2
```

**Redis Optimization**
```bash
# redis.conf
maxmemory 512mb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

### Database Tuning

**Connection Pooling**
```python
# Database connection pool
SQLALCHEMY_ENGINE_OPTIONS = {
    'pool_size': 20,
    'max_overflow': 30,
    'pool_timeout': 30,
    'pool_recycle': 3600,
    'pool_pre_ping': True
}
```

**Query Optimization**
```sql
-- Add missing indexes
CREATE INDEX CONCURRENTLY idx_performance_1 
ON gl_journal_entries(tenant_id, status, entry_date);

CREATE INDEX CONCURRENTLY idx_performance_2 
ON gl_journal_entry_lines(account_id, journal_entry_id);
```

## 🛡️ Security Hardening

### Container Security

**Security Scanning**
```bash
# Scan for vulnerabilities
docker scan apg/general-ledger:latest

# Use security-hardened base images
FROM python:3.11-slim-bullseye
```

**Runtime Security**
```bash
# Run as non-root user
USER 1000:1000

# Read-only root filesystem
docker run --read-only --tmpfs /tmp apg/general-ledger
```

### Network Security

**Firewall Rules**
```bash
# Allow only necessary ports
ufw allow 80/tcp
ufw allow 443/tcp
ufw deny 5432/tcp  # Don't expose database directly
```

**Network Policies (Kubernetes)**
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: gl-network-policy
spec:
  podSelector:
    matchLabels:
      app: general-ledger
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
```

---

## 🎉 Deployment Complete!

Your revolutionary General Ledger system is now deployed and ready to delight users with its 10x better experience!

**Next Steps:**
1. Configure user accounts and permissions
2. Import chart of accounts
3. Train users on revolutionary features
4. Monitor system performance
5. Scale as needed

**Support:** nyimbi@gmail.com | www.datacraft.co.ke