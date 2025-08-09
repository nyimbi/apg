# Deployment Guide

This comprehensive guide covers deploying the Integration API Management capability across different environments, from development to production.

## Overview

The Integration API Management capability is designed for cloud-native deployment with support for:

- **Container orchestration** (Kubernetes, Docker Swarm)
- **Cloud platforms** (AWS, Azure, GCP)
- **Auto-scaling** and load balancing
- **High availability** and disaster recovery
- **Multi-region** deployments

## Prerequisites

### System Requirements

- **CPU**: Minimum 2 cores, recommended 4+ cores per instance
- **Memory**: Minimum 4GB RAM, recommended 8GB+ per instance
- **Storage**: 20GB for application, additional storage for logs and data
- **Network**: 1Gbps network interface

### Dependencies

- **PostgreSQL**: 15+ (for primary data storage)
- **Redis**: 7+ (for caching and session management)
- **APG Platform Core**: Latest version
- **Load Balancer**: nginx, HAProxy, or cloud load balancer

### Software Requirements

- **Python**: 3.11+
- **Docker**: 20.10+ (for containerized deployment)
- **Kubernetes**: 1.25+ (for orchestrated deployment)

## Deployment Options

### 1. Docker Compose (Development/Testing)

Perfect for local development and testing environments.

#### docker-compose.yml

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: integration_api_management
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  api-management:
    build: .
    environment:
      - DATABASE_URL=postgresql://postgres:secure_password@postgres:5432/integration_api_management
      - REDIS_URL=redis://redis:6379/0
      - APG_ENVIRONMENT=development
      - LOG_LEVEL=DEBUG
    ports:
      - "8080:8080"
      - "8081:8081"  # Gateway port
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    depends_on:
      - api-management
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

#### Quick Start

```bash
# Clone repository
git clone https://github.com/datacraft/apg-capabilities
cd apg-capabilities/general_cross_functional/integration_api_management

# Configure environment
cp config/settings.example.toml config/settings.toml
# Edit settings.toml with your configuration

# Start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api-management

# Initialize database
docker-compose exec api-management python -m alembic upgrade head

# Create admin user
docker-compose exec api-management python scripts/create_admin.py
```

### 2. Kubernetes Deployment (Production)

Recommended for production environments requiring high availability and scalability.

#### Namespace Configuration

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: integration-api-management
  labels:
    name: integration-api-management
    environment: production
```

#### ConfigMap

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: api-management-config
  namespace: integration-api-management
data:
  settings.toml: |
    [environment]
    name = "production"
    debug = false
    
    [database]
    engine = "postgresql"
    host = "postgres-service"
    port = 5432
    database = "integration_api_management"
    
    [redis]
    host = "redis-service"
    port = 6379
    database = 0
    
    [gateway]
    host = "0.0.0.0"
    port = 8081
    workers = 4
    
    [monitoring]
    log_level = "INFO"
    metrics_enabled = true
    tracing_enabled = true
```

#### Database Deployment

```yaml
# postgres.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: integration-api-management
spec:
  serviceName: postgres-service
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15
        env:
        - name: POSTGRES_DB
          value: integration_api_management
        - name: POSTGRES_USER
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: username
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 20Gi

---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: integration-api-management
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
  type: ClusterIP
```

#### Redis Deployment

```yaml
# redis.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: integration-api-management
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        command: ["redis-server", "--appendonly", "yes"]
        ports:
        - containerPort: 6379
        volumeMounts:
        - name: redis-storage
          mountPath: /data
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
      volumes:
      - name: redis-storage
        persistentVolumeClaim:
          claimName: redis-pvc

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: redis-pvc
  namespace: integration-api-management
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi

---
apiVersion: v1
kind: Service
metadata:
  name: redis-service
  namespace: integration-api-management
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
  type: ClusterIP
```

#### API Management Deployment

```yaml
# api-management.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-management
  namespace: integration-api-management
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api-management
  template:
    metadata:
      labels:
        app: api-management
    spec:
      containers:
      - name: api-management
        image: datacraft/integration-api-management:1.0.0
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-secret
              key: connection-string
        - name: REDIS_URL
          value: "redis://redis-service:6379/0"
        - name: APG_ENVIRONMENT
          value: "production"
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 8081
          name: gateway
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
      volumes:
      - name: config-volume
        configMap:
          name: api-management-config

---
apiVersion: v1
kind: Service
metadata:
  name: api-management-service
  namespace: integration-api-management
spec:
  selector:
    app: api-management
  ports:
  - name: http
    port: 8080
    targetPort: 8080
  - name: gateway
    port: 8081
    targetPort: 8081
  type: ClusterIP

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-management-hpa
  namespace: integration-api-management
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-management
  minReplicas: 3
  maxReplicas: 10
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

#### Ingress Configuration

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: api-management-ingress
  namespace: integration-api-management
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - api-management.yourcompany.com
    - gateway.yourcompany.com
    secretName: api-management-tls
  rules:
  - host: api-management.yourcompany.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: api-management-service
            port:
              number: 8080
  - host: gateway.yourcompany.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: api-management-service
            port:
              number: 8081
```

#### Secrets Management

```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: postgres-secret
  namespace: integration-api-management
type: Opaque
data:
  username: cG9zdGdyZXM=  # base64 encoded 'postgres'
  password: <base64-encoded-password>

---
apiVersion: v1
kind: Secret
metadata:
  name: database-secret
  namespace: integration-api-management
type: Opaque
data:
  connection-string: <base64-encoded-connection-string>

---
apiVersion: v1
kind: Secret
metadata:
  name: jwt-secret
  namespace: integration-api-management
type: Opaque
data:
  jwt-secret-key: <base64-encoded-jwt-secret>
```

### 3. Cloud Platform Deployments

#### AWS EKS Deployment

```yaml
# aws-eks-values.yaml for Helm chart
replicaCount: 3

image:
  repository: your-account.dkr.ecr.region.amazonaws.com/integration-api-management
  tag: 1.0.0
  pullPolicy: IfNotPresent

service:
  type: LoadBalancer
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: nlb
    service.beta.kubernetes.io/aws-load-balancer-backend-protocol: http

ingress:
  enabled: true
  annotations:
    kubernetes.io/ingress.class: alb
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
    cert-manager.io/cluster-issuer: letsencrypt-prod

database:
  external: true
  host: your-rds-endpoint.region.rds.amazonaws.com
  port: 5432
  ssl: true

redis:
  external: true
  host: your-elasticache-endpoint.cache.amazonaws.com
  port: 6379

monitoring:
  enabled: true
  prometheus: true
  cloudwatch: true

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

resources:
  requests:
    cpu: 2
    memory: 4Gi
  limits:
    cpu: 4
    memory: 8Gi

nodeSelector:
  node-type: api-management
```

#### Azure AKS Deployment

```yaml
# azure-aks-values.yaml
replicaCount: 3

image:
  repository: youracr.azurecr.io/integration-api-management
  tag: 1.0.0

service:
  type: LoadBalancer
  annotations:
    service.beta.kubernetes.io/azure-load-balancer-resource-group: your-resource-group

ingress:
  enabled: true
  annotations:
    kubernetes.io/ingress.class: azure/application-gateway
    appgw.ingress.kubernetes.io/ssl-redirect: "true"

database:
  external: true
  host: your-postgres-server.postgres.database.azure.com
  ssl: true

redis:
  external: true
  host: your-redis-cache.redis.cache.windows.net
  ssl: true

monitoring:
  enabled: true
  azureMonitor: true
```

#### Google GKE Deployment

```yaml
# gcp-gke-values.yaml
replicaCount: 3

image:
  repository: gcr.io/your-project/integration-api-management
  tag: 1.0.0

service:
  type: LoadBalancer
  annotations:
    cloud.google.com/load-balancer-type: External

ingress:
  enabled: true
  annotations:
    kubernetes.io/ingress.class: gce
    kubernetes.io/ingress.global-static-ip-name: api-management-ip
    networking.gke.io/managed-certificates: api-management-ssl

database:
  external: true
  host: your-cloud-sql-ip
  ssl: true

redis:
  external: true
  host: your-memorystore-ip

monitoring:
  enabled: true
  stackdriver: true
```

## Configuration Management

### Environment-Specific Configurations

#### Development Environment

```toml
# config/development.toml
[environment]
name = "development"
debug = true

[database]
engine = "postgresql"
host = "localhost"
port = 5432
database = "integration_api_management_dev"
username = "postgres"
password = "postgres"
pool_size = 5
echo = true

[redis]
host = "localhost"
port = 6379
database = 1

[gateway]
host = "127.0.0.1"
port = 8081
workers = 1

[monitoring]
log_level = "DEBUG"
metrics_enabled = true
```

#### Staging Environment

```toml
# config/staging.toml
[environment]
name = "staging"
debug = false

[database]
engine = "postgresql"
host = "staging-postgres.internal"
port = 5432
database = "integration_api_management"
pool_size = 10
ssl_mode = "require"

[redis]
host = "staging-redis.internal"
port = 6379
database = 0
ssl = true

[gateway]
host = "0.0.0.0"
port = 8081
workers = 2

[monitoring]
log_level = "INFO"
metrics_enabled = true
tracing_enabled = true
```

#### Production Environment

```toml
# config/production.toml
[environment]
name = "production"
debug = false

[database]
engine = "postgresql"
host = "${DATABASE_HOST}"
port = 5432
database = "${DATABASE_NAME}"
username = "${DATABASE_USER}"
password = "${DATABASE_PASSWORD}"
pool_size = 20
ssl_mode = "require"
ssl_cert = "/etc/ssl/certs/client-cert.pem"
ssl_key = "/etc/ssl/private/client-key.pem"
ssl_ca = "/etc/ssl/certs/ca-cert.pem"

[redis]
host = "${REDIS_HOST}"
port = 6379
database = 0
password = "${REDIS_PASSWORD}"
ssl = true
ssl_cert_reqs = "required"

[gateway]
host = "0.0.0.0"
port = 8081
workers = 4
max_connections = 10000

[security]
jwt_secret_key = "${JWT_SECRET_KEY}"
encryption_key = "${ENCRYPTION_KEY}"
allowed_origins = [
    "https://yourcompany.com",
    "https://*.yourcompany.com"
]

[monitoring]
log_level = "WARNING"
metrics_enabled = true
tracing_enabled = true
alerting_enabled = true
```

## Security Configuration

### TLS/SSL Setup

#### Generate Certificates

```bash
# Self-signed certificates for development
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Let's Encrypt certificates for production
certbot certonly --webroot -w /var/www/html -d api-management.yourcompany.com

# Kubernetes cert-manager
kubectl apply -f https://github.com/jetstack/cert-manager/releases/download/v1.11.0/cert-manager.yaml
```

#### nginx SSL Configuration

```nginx
# nginx/nginx.conf
server {
    listen 80;
    server_name api-management.yourcompany.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api-management.yourcompany.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;

    location / {
        proxy_pass http://api-management:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Network Security

#### Firewall Rules

```bash
# Allow HTTP/HTTPS traffic
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Allow internal communication
sudo ufw allow from 10.0.0.0/8 to any port 5432  # PostgreSQL
sudo ufw allow from 10.0.0.0/8 to any port 6379  # Redis

# Allow application ports
sudo ufw allow from 10.0.0.0/8 to any port 8080  # API Management
sudo ufw allow from 10.0.0.0/8 to any port 8081  # Gateway
```

#### Kubernetes Network Policies

```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-management-network-policy
  namespace: integration-api-management
spec:
  podSelector:
    matchLabels:
      app: api-management
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
      port: 8080
    - protocol: TCP
      port: 8081
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
```

## Monitoring and Observability

### Prometheus Monitoring

```yaml
# monitoring.yaml
apiVersion: v1
kind: ServiceMonitor
metadata:
  name: api-management-metrics
  namespace: integration-api-management
spec:
  selector:
    matchLabels:
      app: api-management
  endpoints:
  - port: http
    path: /metrics
    interval: 30s

---
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: api-management-alerts
  namespace: integration-api-management
spec:
  groups:
  - name: api-management
    rules:
    - alert: HighErrorRate
      expr: rate(api_requests_total{status=~"5.."}[5m]) > 0.1
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "High error rate detected"
        description: "Error rate is above 10% for 5 minutes"
    
    - alert: HighLatency
      expr: histogram_quantile(0.95, rate(api_request_duration_seconds_bucket[5m])) > 2
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "High latency detected"
        description: "95th percentile latency is above 2 seconds"
```

### Logging Configuration

```yaml
# logging.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluent-bit-config
  namespace: integration-api-management
data:
  fluent-bit.conf: |
    [INPUT]
        Name tail
        Path /var/log/containers/*api-management*.log
        Tag api-management.*
        Parser docker
        
    [FILTER]
        Name kubernetes
        Match api-management.*
        
    [OUTPUT]
        Name forward
        Match api-management.*
        Host elasticsearch.logging.svc.cluster.local
        Port 9200
```

## Backup and Disaster Recovery

### Database Backup

```bash
#!/bin/bash
# backup-database.sh

BACKUP_DIR="/backup/database"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="${BACKUP_DIR}/api_management_backup_${TIMESTAMP}.sql"

# Create backup directory
mkdir -p ${BACKUP_DIR}

# Perform backup
pg_dump -h postgres-service -U postgres -d integration_api_management > ${BACKUP_FILE}

# Compress backup
gzip ${BACKUP_FILE}

# Clean old backups (keep last 7 days)
find ${BACKUP_DIR} -name "*.sql.gz" -mtime +7 -delete

echo "Backup completed: ${BACKUP_FILE}.gz"
```

### Kubernetes Backup (Velero)

```yaml
# backup-schedule.yaml
apiVersion: velero.io/v1
kind: Schedule
metadata:
  name: api-management-backup
  namespace: velero
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  template:
    includedNamespaces:
    - integration-api-management
    storageLocation: aws-s3-backup
    ttl: 720h  # 30 days
```

## Performance Tuning

### PostgreSQL Optimization

```sql
-- postgresql.conf optimizations
shared_buffers = 4GB
effective_cache_size = 12GB
maintenance_work_mem = 1GB
checkpoint_completion_target = 0.9
wal_buffers = 64MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
work_mem = 64MB
min_wal_size = 2GB
max_wal_size = 4GB
```

### Redis Optimization

```conf
# redis.conf optimizations
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
tcp-keepalive 300
timeout 0
```

### Application Tuning

```toml
# Performance configuration
[gateway]
workers = 4
max_connections = 10000
keep_alive_timeout = 75
client_timeout = 60

[database]
pool_size = 20
max_overflow = 30
pool_timeout = 30
pool_recycle = 3600

[redis]
connection_pool_size = 20
socket_timeout = 5
socket_connect_timeout = 5
```

## Troubleshooting

### Common Issues

#### Connection Refused

```bash
# Check service status
kubectl get pods -n integration-api-management
kubectl describe pod api-management-xxx -n integration-api-management

# Check logs
kubectl logs api-management-xxx -n integration-api-management

# Test connectivity
kubectl exec -it api-management-xxx -n integration-api-management -- nc -zv postgres-service 5432
```

#### High Memory Usage

```bash
# Monitor memory usage
kubectl top pods -n integration-api-management

# Adjust resource limits
kubectl patch deployment api-management -n integration-api-management -p '{"spec":{"template":{"spec":{"containers":[{"name":"api-management","resources":{"limits":{"memory":"8Gi"}}}]}}}}'
```

#### Database Connection Issues

```bash
# Test database connection
kubectl exec -it postgres-xxx -n integration-api-management -- psql -U postgres -d integration_api_management -c "SELECT 1;"

# Check connection pool
kubectl exec -it api-management-xxx -n integration-api-management -- python -c "
from sqlalchemy import create_engine
engine = create_engine('postgresql://user:pass@host:5432/db')
print(engine.pool.status())
"
```

## Maintenance

### Rolling Updates

```bash
# Update deployment
kubectl set image deployment/api-management api-management=datacraft/integration-api-management:1.1.0 -n integration-api-management

# Monitor rollout
kubectl rollout status deployment/api-management -n integration-api-management

# Rollback if needed
kubectl rollout undo deployment/api-management -n integration-api-management
```

### Database Migrations

```bash
# Run migrations
kubectl exec -it api-management-xxx -n integration-api-management -- python -m alembic upgrade head

# Check migration status
kubectl exec -it api-management-xxx -n integration-api-management -- python -m alembic current
```

### Scaling

```bash
# Manual scaling
kubectl scale deployment api-management --replicas=5 -n integration-api-management

# Update HPA
kubectl patch hpa api-management-hpa -n integration-api-management -p '{"spec":{"maxReplicas":15}}'
```

## Support

For deployment assistance:

- **Documentation**: [https://docs.datacraft.co.ke/apg/deployment](https://docs.datacraft.co.ke/apg/deployment)
- **Support Team**: [devops@datacraft.co.ke](mailto:devops@datacraft.co.ke)
- **Emergency**: [emergency@datacraft.co.ke](mailto:emergency@datacraft.co.ke)
- **Status Page**: [https://status.datacraft.co.ke](https://status.datacraft.co.ke)