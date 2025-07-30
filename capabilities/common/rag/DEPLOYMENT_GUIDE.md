# APG RAG Production Deployment Guide

> **Complete guide for deploying APG RAG in production environments**

## 🚀 Quick Start Deployment

### Prerequisites
- Docker 20.10+ with Docker Compose
- PostgreSQL 15+ with pgvector and pgai extensions
- Ollama service with GPU support (optional but recommended)
- Minimum 8GB RAM, 4 CPU cores, 50GB storage

### 1. Clone and Configure

```bash
# Clone the repository
git clone <your-repo-url>
cd apg/capabilities/common/rag

# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

### 2. Essential Configuration

Update these critical variables in `.env`:

```bash
# Database
DATABASE_URL=postgresql://apg_user:your_secure_password@postgres:5432/apg_rag
POSTGRES_PASSWORD=your_secure_password_2025

# Security
ENCRYPTION_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")

# Tenant
TENANT_ID=your-company-tenant-id

# Ollama
OLLAMA_BASE_URL=http://ollama:11434
```

### 3. Deploy with Docker Compose

```bash
# Create data directories
mkdir -p data/{postgres,ollama,redis,rag,logs,backups}

# Deploy all services
docker-compose up -d

# Check service health
docker-compose ps
docker-compose logs -f rag-service
```

### 4. Initialize the System

```bash
# Wait for all services to be healthy
./scripts/wait-for-services.sh

# Run database migrations
docker-compose exec rag-service python -m alembic upgrade head

# Load initial models in Ollama
docker-compose exec ollama ollama pull bge-m3
docker-compose exec ollama ollama pull qwen3
docker-compose exec ollama ollama pull deepseek-r1

# Create first knowledge base
curl -X POST http://localhost/api/v1/rag/knowledge-bases \
  -H "Content-Type: application/json" \
  -d '{"name": "Getting Started", "description": "Initial knowledge base"}'
```

### 5. Verify Deployment

```bash
# Check system health
curl http://localhost/api/v1/rag/health

# Access monitoring dashboards
# Grafana: http://localhost:3000 (admin/admin_password_2025)
# Prometheus: http://localhost:9091
```

## 🏢 Enterprise Production Deployment

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer                            │
│                   (Nginx/CloudFlare)                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                 Kubernetes Cluster                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ RAG Service │  │ RAG Service │  │ RAG Service │        │
│  │   (Pod 1)   │  │   (Pod 2)   │  │   (Pod N)   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ PostgreSQL  │  │   Ollama    │  │    Redis    │        │
│  │ (Primary +  │  │  (GPU Pool) │  │  (Cluster)  │        │
│  │  Replica)   │  │             │  │             │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### Kubernetes Deployment

#### 1. Create Kubernetes Manifests

```bash
# Generate Kubernetes configurations
mkdir -p k8s

# Create namespace
cat > k8s/namespace.yaml << EOF
apiVersion: v1
kind: Namespace
metadata:
  name: apg-rag
  labels:
    name: apg-rag
EOF

# Create secrets
kubectl create secret generic apg-rag-secrets \
  --from-env-file=.env \
  --namespace=apg-rag
```

#### 2. Deploy PostgreSQL with High Availability

```yaml
# k8s/postgres-cluster.yaml
apiVersion: postgresql.cnpg.io/v1
kind: Cluster
metadata:
  name: postgres-cluster
  namespace: apg-rag
spec:
  instances: 3
  
  postgresql:
    parameters:
      max_connections: "200"
      shared_buffers: "256MB"
      effective_cache_size: "1GB"
      
  storage:
    size: 1Ti
    storageClass: fast-ssd
    
  monitoring:
    enabled: true
    
  backup:
    barmanObjectStore:
      s3Credentials:
        accessKeyId:
          name: backup-credentials
          key: ACCESS_KEY_ID
        secretAccessKey:
          name: backup-credentials  
          key: SECRET_ACCESS_KEY
      wal:
        retention: "7d"
      data:
        retention: "30d"
```

#### 3. Deploy Ollama GPU Pool

```yaml
# k8s/ollama-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ollama
  namespace: apg-rag
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ollama
  template:
    metadata:
      labels:
        app: ollama
    spec:
      containers:
      - name: ollama
        image: ollama/ollama:latest
        resources:
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: 1
        ports:
        - containerPort: 11434
        volumeMounts:
        - name: ollama-data
          mountPath: /root/.ollama
      volumes:
      - name: ollama-data
        persistentVolumeClaim:
          claimName: ollama-pvc
      nodeSelector:
        accelerator: nvidia-tesla-v100
```

#### 4. Deploy RAG Service with Auto-scaling

```yaml
# k8s/rag-service.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-service
  namespace: apg-rag
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-service
  template:
    metadata:
      labels:
        app: rag-service
    spec:
      containers:
      - name: rag-service
        image: apg-rag:1.0.0
        envFrom:
        - secretRef:
            name: apg-rag-secrets
        resources:
          limits:
            memory: "4Gi"
            cpu: "2"
          requests:
            memory: "2Gi"
            cpu: "1"
        ports:
        - containerPort: 5000
        - containerPort: 9090
        livenessProbe:
          httpGet:
            path: /api/v1/rag/health
            port: 5000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /api/v1/rag/health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10

---
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
```

### 5. Deploy with Helm

```bash
# Create Helm chart
helm create apg-rag

# Deploy with custom values
helm install apg-rag ./apg-rag \
  --namespace apg-rag \
  --create-namespace \
  --values values-production.yaml
```

## 🔒 Security Hardening

### 1. Network Security

```yaml
# k8s/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: apg-rag-network-policy
  namespace: apg-rag
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    - podSelector:
        matchLabels:
          app: rag-service
    ports:
    - protocol: TCP
      port: 5000
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
          app: ollama
    ports:
    - protocol: TCP
      port: 11434
```

### 2. Pod Security Standards

```yaml
# k8s/pod-security-policy.yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: apg-rag-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
```

### 3. SSL/TLS Configuration

```bash
# Generate certificates with cert-manager
kubectl apply -f https://github.com/jetstack/cert-manager/releases/download/v1.11.0/cert-manager.yaml

# Create certificate issuer
cat > k8s/cert-issuer.yaml << EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@your-company.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

## 📊 Monitoring & Observability

### 1. Prometheus Configuration

```yaml
# k8s/prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: apg-rag
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
    
    rule_files:
      - "rules/*.yml"
    
    scrape_configs:
      - job_name: 'rag-service'
        static_configs:
          - targets: ['rag-service:9090']
        metrics_path: /metrics
        scrape_interval: 15s
      
      - job_name: 'postgres'
        static_configs:
          - targets: ['postgres-exporter:9187']
      
      - job_name: 'redis'
        static_configs:
          - targets: ['redis-exporter:9121']
```

### 2. Grafana Dashboards

```bash
# Import APG RAG dashboard
curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @grafana/apg-rag-dashboard.json
```

### 3. Alerting Rules

```yaml
# k8s/alerting-rules.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-rules
  namespace: apg-rag
data:
  rules.yml: |
    groups:
    - name: rag-service
      rules:
      - alert: RAGServiceDown
        expr: up{job="rag-service"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "RAG service is down"
          
      - alert: HighResponseTime
        expr: rag_response_time_ms > 5000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          
      - alert: DatabaseConnectionsHigh
        expr: postgres_connections_active / postgres_connections_max > 0.8
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Database connections usage high"
```

## 🔄 CI/CD Pipeline

### 1. GitHub Actions Workflow

```yaml
# .github/workflows/deploy.yml
name: Deploy APG RAG

on:
  push:
    branches: [main]
    tags: ['v*']

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Run tests
      run: |
        docker-compose -f docker-compose.test.yml up --abort-on-container-exit
        
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: |
          your-registry/apg-rag:latest
          your-registry/apg-rag:${{ github.sha }}
          
  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - name: Deploy to production
      run: |
        helm upgrade --install apg-rag ./helm/apg-rag \
          --namespace apg-rag \
          --set image.tag=${{ github.sha }} \
          --values values-production.yaml
```

## 🔧 Maintenance & Operations

### 1. Backup Procedures

```bash
# Database backup
kubectl exec -n apg-rag postgres-cluster-1 -- pg_dump apg_rag | \
  gzip > backup-$(date +%Y%m%d).sql.gz

# Full system backup with Velero
velero backup create apg-rag-backup-$(date +%Y%m%d) \
  --include-namespaces apg-rag
```

### 2. Scaling Operations

```bash
# Scale RAG service
kubectl scale deployment rag-service --replicas=10 -n apg-rag

# Scale Ollama pool
kubectl scale deployment ollama --replicas=5 -n apg-rag

# Scale database (with CNPG)
kubectl patch cluster postgres-cluster -n apg-rag \
  --type='json' -p='[{"op": "replace", "path": "/spec/instances", "value": 5}]'
```

### 3. Troubleshooting Commands

```bash
# Check service health
kubectl get pods -n apg-rag
kubectl describe pod rag-service-xxx -n apg-rag

# View logs
kubectl logs -f deployment/rag-service -n apg-rag

# Debug networking
kubectl exec -it rag-service-xxx -n apg-rag -- nslookup postgres-cluster

# Check resource usage
kubectl top pods -n apg-rag
kubectl top nodes
```

## 📈 Performance Optimization

### 1. Database Tuning

```sql
-- PostgreSQL optimization
ALTER SYSTEM SET shared_buffers = '2GB';
ALTER SYSTEM SET effective_cache_size = '6GB';
ALTER SYSTEM SET maintenance_work_mem = '512MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;

-- Vector-specific optimization
ALTER SYSTEM SET max_parallel_workers_per_gather = 4;
ALTER SYSTEM SET max_parallel_workers = 8;
SELECT pg_reload_conf();
```

### 2. Application Tuning

```python
# config/production.py
PERFORMANCE_CONFIG = {
    "max_concurrent_operations": 200,
    "database_pool_size": 50,
    "vector_cache_size": 100000,
    "ollama_connection_pool": 20,
    "chunk_batch_size": 1000,
    "embedding_batch_size": 100,
}
```

### 3. Resource Allocation

```yaml
# Optimized resource allocation
resources:
  requests:
    memory: "4Gi"
    cpu: "2"
  limits:
    memory: "8Gi" 
    cpu: "4"

# JVM tuning for better memory management  
env:
- name: JAVA_OPTS
  value: "-Xmx6g -Xms2g -XX:+UseG1GC"
```

## 🚨 Disaster Recovery

### 1. Backup Strategy

- **Database**: Point-in-time recovery with 30-day retention
- **Documents**: Replicated to 3 geographic regions  
- **Configuration**: Version controlled and automated
- **Monitoring Data**: 90-day retention with compression

### 2. Recovery Procedures

```bash
# Database recovery
pg_restore --clean --if-exists -d apg_rag backup-20250129.sql

# Application recovery
helm rollback apg-rag 1 -n apg-rag

# Full disaster recovery
velero restore create apg-rag-restore-$(date +%Y%m%d) \
  --from-backup apg-rag-backup-20250129
```

This deployment guide provides comprehensive instructions for production deployment, from simple Docker Compose setups to enterprise Kubernetes clusters with full observability, security, and disaster recovery capabilities.

The APG RAG capability is now ready for any scale of deployment! 🚀