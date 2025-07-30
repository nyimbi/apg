# Computer Vision & Visual Intelligence - Deployment Guide

**Version:** 1.0.0  
**Target Audience:** DevOps Engineers, System Administrators, Platform Engineers  
**Prerequisites:** Kubernetes 1.28+, Docker, Helm 3.0+  

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Installation Methods](#installation-methods)
4. [Configuration](#configuration)
5. [Database Setup](#database-setup)
6. [Storage Configuration](#storage-configuration)
7. [Security Configuration](#security-configuration)
8. [Monitoring and Observability](#monitoring-and-observability)
9. [Performance Tuning](#performance-tuning)
10. [Troubleshooting](#troubleshooting)
11. [Maintenance](#maintenance)
12. [Upgrade Procedures](#upgrade-procedures)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Load Balancer / Ingress                    │
└─────────────────┬───────────────────────────────────────────────────┘
                  │
┌─────────────────┴─────────────────────────────────────────────────┐
│                    Computer Vision API Pods                      │
│  ┌─────────────┬─────────────┬─────────────┬─────────────────────┐ │
│  │   API Pod 1 │   API Pod 2 │   API Pod 3 │   Worker Pods       │ │
│  │   (FastAPI) │   (FastAPI) │   (FastAPI) │   (AI Processing)   │ │
│  └─────────────┴─────────────┴─────────────┴─────────────────────┘ │
└───────────────────────────────────────────────────────────────────┘
                  │
┌─────────────────┴─────────────────────────────────────────────────┐
│                         Data Layer                               │
│  ┌─────────────┬─────────────┬─────────────┬─────────────────────┐ │
│  │ PostgreSQL  │   Redis     │ Object      │   Model Storage     │ │
│  │  Database   │   Cache     │ Storage     │   (AI Models)       │ │
│  └─────────────┴─────────────┴─────────────┴─────────────────────┘ │
└───────────────────────────────────────────────────────────────────┘
                  │
┌─────────────────┴─────────────────────────────────────────────────┐
│                    Monitoring & Logging                          │
│  ┌─────────────┬─────────────┬─────────────┬─────────────────────┐ │
│  │ Prometheus  │   Grafana   │ Jaeger      │   ELK Stack         │ │
│  │  Metrics    │ Dashboard   │ Tracing     │   Logging           │ │
│  └─────────────┴─────────────┴─────────────┴─────────────────────┘ │
└───────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### System Requirements

**Minimum Requirements:**
- **Kubernetes:** v1.28+ with RBAC enabled
- **Nodes:** 3 worker nodes minimum
- **CPU:** 4 cores per node
- **Memory:** 16GB RAM per node
- **Storage:** 100GB SSD per node
- **Network:** 1Gbps internal connectivity

**Recommended Requirements:**
- **Kubernetes:** v1.29+ with auto-scaling
- **Nodes:** 5+ worker nodes with GPU support
- **CPU:** 8 cores per node
- **Memory:** 32GB RAM per node
- **Storage:** 500GB NVMe SSD per node
- **Network:** 10Gbps internal, dedicated network for ML workloads

### Software Dependencies

```bash
# Required tools
kubectl >= v1.28
helm >= v3.0
docker >= v24.0

# Optional but recommended
kustomize >= v5.0
istio >= v1.18 (for service mesh)
cert-manager >= v1.12 (for SSL certificates)
```

### External Services

**Database:**
- PostgreSQL 14+ with replication
- Minimum 4 CPU cores, 8GB RAM
- 100GB storage with backup capability

**Cache:**
- Redis 7+ cluster setup
- Minimum 2 CPU cores, 4GB RAM
- Persistent storage for durability

**Object Storage:**
- S3-compatible storage (AWS S3, MinIO, etc.)
- Minimum 500GB capacity
- Cross-region replication recommended

---

## Installation Methods

### Method 1: Helm Chart Installation (Recommended)

1. **Add Helm Repository**
```bash
helm repo add datacraft https://charts.datacraft.co.ke
helm repo update
```

2. **Create Namespace**
```bash
kubectl create namespace computer-vision
```

3. **Install with Default Values**
```bash
helm install computer-vision datacraft/computer-vision \
  --namespace computer-vision \
  --version 1.0.0
```

4. **Custom Installation**
```bash
# Create custom values file
cat > values-production.yaml << EOF
# Production configuration
replicaCount: 3
image:
  tag: "1.0.0"
  pullPolicy: Always

resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 500m
    memory: 2Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70

postgresql:
  enabled: false
  external:
    host: postgres.database.svc.cluster.local
    port: 5432
    database: computer_vision
    username: cv_user

redis:
  enabled: false
  external:
    host: redis.cache.svc.cluster.local
    port: 6379

storage:
  type: s3
  s3:
    bucket: computer-vision-storage
    region: us-west-2
    endpoint: https://s3.us-west-2.amazonaws.com

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
  hosts:
    - host: cv.your-domain.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: cv-tls
      hosts:
        - cv.your-domain.com

monitoring:
  enabled: true
  serviceMonitor:
    enabled: true
  prometheusRule:
    enabled: true
EOF

# Install with custom values
helm install computer-vision datacraft/computer-vision \
  --namespace computer-vision \
  --values values-production.yaml
```

### Method 2: Kustomize Deployment

1. **Clone Configuration Repository**
```bash
git clone https://github.com/datacraft/apg-computer-vision-k8s.git
cd apg-computer-vision-k8s
```

2. **Customize Configuration**
```bash
# Edit kustomization.yaml for your environment
vim overlays/production/kustomization.yaml
```

3. **Deploy**
```bash
kubectl apply -k overlays/production/
```

### Method 3: Operator Installation

1. **Install Computer Vision Operator**
```bash
kubectl apply -f https://github.com/datacraft/cv-operator/releases/download/v1.0.0/operator.yaml
```

2. **Create Computer Vision Custom Resource**
```yaml
# cv-instance.yaml
apiVersion: cv.datacraft.co.ke/v1alpha1
kind: ComputerVision
metadata:
  name: production-cv
  namespace: computer-vision
spec:
  version: "1.0.0"
  replicas: 3
  resources:
    limits:
      cpu: "2000m"
      memory: "4Gi"
    requests:
      cpu: "500m"
      memory: "2Gi"
  database:
    type: external
    host: postgres.database.svc.cluster.local
    port: 5432
    database: computer_vision
  cache:
    type: external
    host: redis.cache.svc.cluster.local
    port: 6379
  storage:
    type: s3
    bucket: computer-vision-storage
    region: us-west-2
  monitoring:
    enabled: true
    metrics: true
    tracing: true
```

```bash
kubectl apply -f cv-instance.yaml
```

---

## Configuration

### Environment Variables

**Core Configuration:**
```bash
# Database
CV_DATABASE_URL="postgresql://user:pass@host:port/db"
CV_DATABASE_POOL_SIZE="20"
CV_DATABASE_POOL_OVERFLOW="0"
CV_DATABASE_POOL_TIMEOUT="30"

# Cache
CV_REDIS_URL="redis://host:port/0"
CV_REDIS_CLUSTER_NODES="redis1:6379,redis2:6379,redis3:6379"
CV_CACHE_TTL="3600"
CV_CACHE_MAX_MEMORY="1gb"

# Storage
CV_STORAGE_BACKEND="s3"
CV_S3_BUCKET="computer-vision-files"
CV_S3_REGION="us-west-2"
CV_S3_ACCESS_KEY_ID=""
CV_S3_SECRET_ACCESS_KEY=""
CV_S3_ENDPOINT_URL=""

# AI Models
CV_MODEL_PATH="/models"
CV_MODEL_CACHE_SIZE="10gb"
CV_YOLO_MODEL="yolov8n.pt"
CV_OCR_LANGUAGES="eng,fra,deu,spa"
CV_TESSERACT_DATA_PATH="/usr/share/tesseract-ocr/4.00/tessdata"

# Performance
CV_MAX_CONCURRENT_JOBS="50"
CV_MAX_FILE_SIZE_MB="50"
CV_JOB_TIMEOUT_MINUTES="30"
CV_WORKER_PROCESSES="4"
CV_WORKER_THREADS="8"

# Security
CV_JWT_SECRET_KEY=""
CV_JWT_ALGORITHM="HS256"
CV_JWT_EXPIRATION_HOURS="24"
CV_ENCRYPTION_KEY=""
CV_ALLOWED_ORIGINS="https://your-domain.com"

# Monitoring
CV_METRICS_ENABLED="true"
CV_METRICS_PORT="9090"
CV_TRACING_ENABLED="true"
CV_JAEGER_ENDPOINT="http://jaeger:14268/api/traces"
CV_LOG_LEVEL="INFO"
CV_LOG_FORMAT="json"
```

### Kubernetes ConfigMap

```yaml
# config-map.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: computer-vision-config
  namespace: computer-vision
data:
  CV_MAX_CONCURRENT_JOBS: "50"
  CV_MAX_FILE_SIZE_MB: "50"
  CV_JOB_TIMEOUT_MINUTES: "30"
  CV_WORKER_PROCESSES: "4"
  CV_WORKER_THREADS: "8"
  CV_CACHE_TTL: "3600"
  CV_MODEL_PATH: "/models"
  CV_OCR_LANGUAGES: "eng,fra,deu,spa"
  CV_METRICS_ENABLED: "true"
  CV_METRICS_PORT: "9090"
  CV_TRACING_ENABLED: "true"
  CV_LOG_LEVEL: "INFO"
  CV_LOG_FORMAT: "json"
```

### Kubernetes Secrets

```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: computer-vision-secrets
  namespace: computer-vision
type: Opaque
data:
  CV_DATABASE_URL: cG9zdGdyZXNxbDovL3VzZXI6cGFzc0Bob3N0OjU0MzIvZGI=
  CV_REDIS_URL: cmVkaXM6Ly9ob3N0OjYzNzkvMA==
  CV_S3_ACCESS_KEY_ID: QUtJQUlPU0ZPRE5ON0VYQU1QTEU=
  CV_S3_SECRET_ACCESS_KEY: d0pBbFJYVXRuRkVNSS9LN01ERU5HL2JQeFJmaUNZRVhBTVBMRUtFWQ==
  CV_JWT_SECRET_KEY: c2VjcmV0LWp3dC1rZXktZm9yLXByb2R1Y3Rpb24=
  CV_ENCRYPTION_KEY: ZW5jcnlwdGlvbi1rZXktZm9yLXNlbnNpdGl2ZS1kYXRh
```

---

## Database Setup

### PostgreSQL Configuration

1. **Install PostgreSQL Operator**
```bash
kubectl apply -f https://github.com/postgres-operator/postgres-operator/releases/download/v1.8.0/postgres-operator.yaml
```

2. **Create PostgreSQL Cluster**
```yaml
# postgresql-cluster.yaml
apiVersion: postgresql.cnpg.io/v1
kind: Cluster
metadata:
  name: computer-vision-db
  namespace: computer-vision
spec:
  instances: 3
  
  postgresql:
    parameters:
      max_connections: "200"
      shared_buffers: "256MB"
      effective_cache_size: "1GB"
      maintenance_work_mem: "64MB"
      checkpoint_completion_target: "0.9"
      wal_buffers: "16MB"
      default_statistics_target: "100"
      random_page_cost: "1.1"
      effective_io_concurrency: "200"
  
  bootstrap:
    initdb:
      database: computer_vision
      owner: cv_user
      secret:
        name: computer-vision-db-credentials
  
  storage:
    size: 100Gi
    storageClass: fast-ssd
  
  monitoring:
    enabled: true
  
  backup:
    retentionPolicy: "30d"
    barmanObjectStore:
      destinationPath: "s3://backups/postgresql"
      s3Credentials:
        accessKeyId:
          name: s3-credentials
          key: ACCESS_KEY_ID
        secretAccessKey:
          name: s3-credentials
          key: SECRET_ACCESS_KEY
      wal:
        retention: "5d"
      data:
        retention: "30d"
```

3. **Apply Database Schema**
```bash
# Wait for database to be ready
kubectl wait --for=condition=Ready cluster/computer-vision-db --timeout=300s

# Apply schema migrations
kubectl exec -it computer-vision-db-1 -- psql -U cv_user -d computer_vision -f /migrations/001_initial_schema.sql
```

### Database Migration Scripts

```sql
-- 001_initial_schema.sql
-- Computer Vision Database Schema

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create schema for multi-tenant support
CREATE SCHEMA IF NOT EXISTS cv_tenant_template;

-- Processing jobs table
CREATE TABLE cv_tenant_template.processing_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_name VARCHAR(255) NOT NULL,
    processing_type VARCHAR(50) NOT NULL,
    content_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    progress_percentage FLOAT DEFAULT 0.0,
    input_file_path TEXT NOT NULL,
    output_file_path TEXT,
    processing_parameters JSONB,
    results JSONB,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    priority INTEGER DEFAULT 5,
    tenant_id VARCHAR(100) NOT NULL,
    created_by VARCHAR(100) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Image processing table
CREATE TABLE cv_tenant_template.image_processing (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID REFERENCES cv_tenant_template.processing_jobs(id) ON DELETE CASCADE,
    original_filename VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    file_hash VARCHAR(128) NOT NULL,
    image_dimensions JSONB NOT NULL,
    image_format VARCHAR(20) NOT NULL,
    color_mode VARCHAR(20) NOT NULL,
    processing_type VARCHAR(50) NOT NULL,
    confidence_score FLOAT,
    processing_duration_ms INTEGER,
    processing_model VARCHAR(100),
    model_version VARCHAR(20),
    tenant_id VARCHAR(100) NOT NULL,
    created_by VARCHAR(100) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Object detection results
CREATE TABLE cv_tenant_template.object_detections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID REFERENCES cv_tenant_template.processing_jobs(id) ON DELETE CASCADE,
    image_id UUID REFERENCES cv_tenant_template.image_processing(id) ON DELETE CASCADE,
    detection_model VARCHAR(100) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    detected_objects JSONB NOT NULL,
    total_objects INTEGER NOT NULL,
    detection_confidence FLOAT NOT NULL,
    inference_time_ms INTEGER NOT NULL,
    preprocessing_time_ms INTEGER,
    postprocessing_time_ms INTEGER,
    image_resolution JSONB NOT NULL,
    tenant_id VARCHAR(100) NOT NULL,
    created_by VARCHAR(100) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_processing_jobs_tenant_status ON cv_tenant_template.processing_jobs(tenant_id, status);
CREATE INDEX idx_processing_jobs_created_at ON cv_tenant_template.processing_jobs(created_at DESC);
CREATE INDEX idx_image_processing_job_id ON cv_tenant_template.image_processing(job_id);
CREATE INDEX idx_object_detections_job_id ON cv_tenant_template.object_detections(job_id);

-- Function to create tenant schema
CREATE OR REPLACE FUNCTION create_tenant_schema(tenant_name TEXT)
RETURNS VOID AS $$
BEGIN
    EXECUTE format('CREATE SCHEMA IF NOT EXISTS cv_tenant_%I', tenant_name);
    EXECUTE format('CREATE TABLE cv_tenant_%I.processing_jobs (LIKE cv_tenant_template.processing_jobs INCLUDING ALL)', tenant_name);
    EXECUTE format('CREATE TABLE cv_tenant_%I.image_processing (LIKE cv_tenant_template.image_processing INCLUDING ALL)', tenant_name);
    EXECUTE format('CREATE TABLE cv_tenant_%I.object_detections (LIKE cv_tenant_template.object_detections INCLUDING ALL)', tenant_name);
    
    -- Add foreign key constraints
    EXECUTE format('ALTER TABLE cv_tenant_%I.image_processing ADD CONSTRAINT fk_job_id FOREIGN KEY (job_id) REFERENCES cv_tenant_%I.processing_jobs(id) ON DELETE CASCADE', tenant_name, tenant_name);
    EXECUTE format('ALTER TABLE cv_tenant_%I.object_detections ADD CONSTRAINT fk_job_id FOREIGN KEY (job_id) REFERENCES cv_tenant_%I.processing_jobs(id) ON DELETE CASCADE', tenant_name, tenant_name);
    EXECUTE format('ALTER TABLE cv_tenant_%I.object_detections ADD CONSTRAINT fk_image_id FOREIGN KEY (image_id) REFERENCES cv_tenant_%I.image_processing(id) ON DELETE CASCADE', tenant_name, tenant_name);
    
    -- Create indexes
    EXECUTE format('CREATE INDEX idx_processing_jobs_tenant_status_%I ON cv_tenant_%I.processing_jobs(tenant_id, status)', tenant_name, tenant_name);
    EXECUTE format('CREATE INDEX idx_processing_jobs_created_at_%I ON cv_tenant_%I.processing_jobs(created_at DESC)', tenant_name, tenant_name);
END;
$$ LANGUAGE plpgsql;
```

---

## Storage Configuration

### S3-Compatible Storage

1. **AWS S3 Configuration**
```yaml
# s3-storage-config.yaml
apiVersion: v1
kind: Secret
metadata:
  name: s3-credentials
  namespace: computer-vision
type: Opaque
data:
  ACCESS_KEY_ID: QUtJQUlPU0ZPRE5ON0VYQU1QTEU=
  SECRET_ACCESS_KEY: d0pBbFJYVXRuRkVNSS9LN01ERU5HL2JQeFJmaUNZRVhBTVBMRUtFWQ==
  BUCKET_NAME: Y29tcHV0ZXItdmlzaW9uLXN0b3JhZ2U=
  REGION: dXMtd2VzdC0y
  ENDPOINT_URL: aHR0cHM6Ly9zMy51cy13ZXN0LTIuYW1hem9uYXdzLmNvbQ==
```

2. **MinIO On-Premises Storage**
```yaml
# minio-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: minio
  namespace: computer-vision
spec:
  replicas: 1
  selector:
    matchLabels:
      app: minio
  template:
    metadata:
      labels:
        app: minio
    spec:
      containers:
      - name: minio
        image: minio/minio:RELEASE.2024-01-01T16-36-33Z
        args:
        - server
        - /data
        - --console-address
        - ":9001"
        env:
        - name: MINIO_ROOT_USER
          value: "minioadmin"
        - name: MINIO_ROOT_PASSWORD
          value: "minioadmin123"
        ports:
        - containerPort: 9000
        - containerPort: 9001
        volumeMounts:
        - name: data
          mountPath: /data
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: minio-pvc
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: minio-pvc
  namespace: computer-vision
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 500Gi
  storageClassName: fast-ssd
```

### Persistent Volumes for AI Models

```yaml
# model-storage-pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ai-models-pvc
  namespace: computer-vision
spec:
  accessModes:
    - ReadOnlyMany
  resources:
    requests:
      storage: 50Gi
  storageClassName: fast-ssd
---
# Model download init container
apiVersion: batch/v1
kind: Job
metadata:
  name: download-models
  namespace: computer-vision
spec:
  template:
    spec:
      initContainers:
      - name: model-downloader
        image: datacraft/model-downloader:1.0.0
        command:
        - /bin/bash
        - -c
        - |
          # Download YOLO models
          wget -O /models/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
          wget -O /models/yolov8s.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
          
          # Download Vision Transformer models
          python -c "
          from transformers import ViTImageProcessor, ViTForImageClassification
          processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
          model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
          processor.save_pretrained('/models/vit-base-patch16-224')
          model.save_pretrained('/models/vit-base-patch16-224')
          "
          
          # Download Tesseract language data
          mkdir -p /models/tesseract
          wget -O /models/tesseract/eng.traineddata https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata
          wget -O /models/tesseract/fra.traineddata https://github.com/tesseract-ocr/tessdata/raw/main/fra.traineddata
        volumeMounts:
        - name: models
          mountPath: /models
      containers:
      - name: placeholder
        image: busybox
        command: ['sleep', '30']
      restartPolicy: OnFailure
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: ai-models-pvc
```

---

## Security Configuration

### RBAC Configuration

```yaml
# rbac.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: computer-vision
  namespace: computer-vision
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: computer-vision
  name: computer-vision-role
rules:
- apiGroups: [""]
  resources: ["pods", "configmaps", "secrets"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: computer-vision-binding
  namespace: computer-vision
subjects:
- kind: ServiceAccount
  name: computer-vision
  namespace: computer-vision
roleRef:
  kind: Role
  name: computer-vision-role
  apiGroup: rbac.authorization.k8s.io
```

### Network Policies

```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: computer-vision-network-policy
  namespace: computer-vision
spec:
  podSelector:
    matchLabels:
      app: computer-vision
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
          app: computer-vision
    ports:
    - protocol: TCP
      port: 8000
    - protocol: TCP
      port: 9090
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: database
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - namespaceSelector:
        matchLabels:
          name: cache
    ports:
    - protocol: TCP
      port: 6379
  - to: []
    ports:
    - protocol: TCP
      port: 443
    - protocol: TCP
      port: 80
```

### Pod Security Standards

```yaml
# pod-security-policy.yaml
apiVersion: v1
kind: Pod
metadata:
  name: computer-vision-api
  namespace: computer-vision
spec:
  serviceAccountName: computer-vision
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    runAsGroup: 1000
    fsGroup: 1000
    seccompProfile:
      type: RuntimeDefault
  containers:
  - name: api
    image: datacraft/computer-vision:1.0.0
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL
    resources:
      limits:
        memory: "4Gi"
        cpu: "2000m"
      requests:
        memory: "2Gi"
        cpu: "500m"
    volumeMounts:
    - name: tmp
      mountPath: /tmp
    - name: app-tmp
      mountPath: /app/tmp
  volumes:
  - name: tmp
    emptyDir: {}
  - name: app-tmp
    emptyDir: {}
```

---

## Monitoring and Observability

### Prometheus Configuration

```yaml
# prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-cv-config
  namespace: monitoring
data:
  computer-vision.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
    
    rule_files:
      - "computer_vision_rules.yml"
    
    scrape_configs:
      - job_name: 'computer-vision-api'
        kubernetes_sd_configs:
        - role: pod
          namespaces:
            names:
            - computer-vision
        relabel_configs:
        - source_labels: [__meta_kubernetes_pod_label_app]
          action: keep
          regex: computer-vision
        - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
          action: keep
          regex: true
        - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_port]
          action: replace
          target_label: __address__
          regex: ([^:]+)(?::\d+)?;(\d+)
          replacement: $1:$2
        metrics_path: /metrics
        scrape_interval: 30s
```

### Grafana Dashboards

```json
{
  "dashboard": {
    "id": null,
    "title": "Computer Vision - Performance Dashboard",
    "tags": ["computer-vision", "ai", "performance"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Processing Jobs Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(cv_processing_jobs_total[5m])",
            "legendFormat": "Jobs/sec"
          }
        ],
        "yAxes": [
          {
            "label": "Jobs per second"
          }
        ]
      },
      {
        "id": 2,
        "title": "Average Processing Time",
        "type": "graph",
        "targets": [
          {
            "expr": "avg(cv_processing_duration_seconds)",
            "legendFormat": "Avg Duration"
          }
        ]
      },
      {
        "id": 3,
        "title": "Success Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "rate(cv_processing_jobs_success_total[5m]) / rate(cv_processing_jobs_total[5m]) * 100",
            "legendFormat": "Success Rate %"
          }
        ],
        "format": "percent"
      },
      {
        "id": 4,
        "title": "Active Jobs",
        "type": "singlestat",  
        "targets": [
          {
            "expr": "cv_active_jobs",
            "legendFormat": "Active Jobs"
          }
        ]
      },
      {
        "id": 5,
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "container_memory_usage_bytes{pod=~\"computer-vision.*\"}",
            "legendFormat": "{{ pod }}"
          }
        ]
      },
      {
        "id": 6,
        "title": "CPU Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(container_cpu_usage_seconds_total{pod=~\"computer-vision.*\"}[5m]) * 100",
            "legendFormat": "{{ pod }}"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
```

### Alerting Rules

```yaml
# alerting-rules.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: computer-vision-alerts
  namespace: computer-vision
spec:
  groups:
  - name: computer-vision.rules
    rules:
    - alert: ComputerVisionHighErrorRate
      expr: rate(cv_processing_jobs_error_total[5m]) / rate(cv_processing_jobs_total[5m]) > 0.1
      for: 5m
      labels:
        severity: warning
        service: computer-vision
      annotations:
        summary: "High error rate in Computer Vision processing"
        description: "Error rate is {{ $value | humanizePercentage }} for the last 5 minutes"
    
    - alert: ComputerVisionHighLatency
      expr: histogram_quantile(0.95, rate(cv_processing_duration_seconds_bucket[5m])) > 10
      for: 5m
      labels:
        severity: warning
        service: computer-vision
      annotations:
        summary: "High processing latency in Computer Vision"
        description: "95th percentile latency is {{ $value }}s"
    
    - alert: ComputerVisionServiceDown
      expr: up{job="computer-vision-api"} == 0
      for: 1m
      labels:
        severity: critical
        service: computer-vision
      annotations:
        summary: "Computer Vision service is down"
        description: "Computer Vision API is not responding"
    
    - alert: ComputerVisionHighMemoryUsage
      expr: container_memory_usage_bytes{pod=~"computer-vision.*"} / container_spec_memory_limit_bytes > 0.9
      for: 5m
      labels:
        severity: warning
        service: computer-vision
      annotations:
        summary: "High memory usage in Computer Vision pod"
        description: "Memory usage is {{ $value | humanizePercentage }} in pod {{ $labels.pod }}"
```

---

## Performance Tuning

### Resource Optimization

```yaml
# optimized-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: computer-vision-api
  namespace: computer-vision
spec:
  replicas: 3
  selector:
    matchLabels:
      app: computer-vision-api
  template:
    metadata:
      labels:
        app: computer-vision-api
    spec:
      nodeSelector:
        node-type: cpu-optimized
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - computer-vision-api
              topologyKey: kubernetes.io/hostname
      containers:
      - name: api
        image: datacraft/computer-vision:1.0.0
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        env:
        - name: CV_WORKER_PROCESSES
          value: "4"
        - name: CV_WORKER_THREADS
          value: "8"
        - name: CV_MAX_CONCURRENT_JOBS
          value: "20"
        - name: PYTHONUNBUFFERED
          value: "1"
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

### Horizontal Pod Autoscaler

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: computer-vision-hpa
  namespace: computer-vision
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: computer-vision-api
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
        name: cv_active_jobs
      target:
        type: AverageValue
        averageValue: "10"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

### GPU Support (Optional)

```yaml
# gpu-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: computer-vision-gpu
  namespace: computer-vision
spec:
  replicas: 2
  selector:
    matchLabels:
      app: computer-vision-gpu
  template:
    metadata:
      labels:
        app: computer-vision-gpu
    spec:
      nodeSelector:
        accelerator: nvidia-tesla-v100
      containers:
      - name: api
        image: datacraft/computer-vision:1.0.0-gpu
        resources:
          requests:
            memory: "4Gi"
            cpu: "1000m"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            cpu: "4000m"
            nvidia.com/gpu: 1
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        - name: CV_USE_GPU
          value: "true"
```

---

## Troubleshooting

### Common Issues

**Issue: Pods stuck in Pending state**
```bash
# Check node resources
kubectl describe nodes

# Check pod events
kubectl describe pod -n computer-vision <pod-name>

# Check resource quotas
kubectl describe resourcequota -n computer-vision
```

**Issue: High memory usage**
```bash
# Check memory usage
kubectl top pods -n computer-vision

# Check for memory leaks
kubectl exec -it -n computer-vision <pod-name> -- ps aux --sort=-%mem

# Adjust memory limits
kubectl patch deployment computer-vision-api -n computer-vision -p '{"spec":{"template":{"spec":{"containers":[{"name":"api","resources":{"limits":{"memory":"6Gi"}}}]}}}}'
```

**Issue: Database connection failures**
```bash
# Test database connectivity
kubectl exec -it -n computer-vision <pod-name> -- pg_isready -h postgres.database.svc.cluster.local -p 5432

# Check database logs
kubectl logs -n database postgres-0

# Verify credentials
kubectl get secret computer-vision-secrets -n computer-vision -o yaml
```

### Debugging Commands

```bash
# Check all resources
kubectl get all -n computer-vision

# View logs
kubectl logs -f deployment/computer-vision-api -n computer-vision

# Execute commands in pod
kubectl exec -it -n computer-vision <pod-name> -- /bin/bash

# Port forward for debugging
kubectl port-forward -n computer-vision svc/computer-vision-api 8000:8000

# Check resource usage
kubectl top pods -n computer-vision
kubectl top nodes

# View events
kubectl get events -n computer-vision --sort-by='.lastTimestamp'
```

---

## Maintenance

### Regular Maintenance Tasks

**Daily:**
- Monitor system health and alerts
- Check processing job success rates
- Review resource utilization
- Verify backup completion

**Weekly:**
- Update security patches
- Review and rotate logs
- Check disk space usage
- Analyze performance trends

**Monthly:**
- Update AI models if available
- Review and update resource limits
- Conduct security scans
- Test disaster recovery procedures

### Backup Procedures

```bash
# Database backup
kubectl exec -n computer-vision computer-vision-db-1 -- pg_dump -U cv_user computer_vision > backup-$(date +%Y%m%d).sql

# Model backup
kubectl cp computer-vision/<pod-name>:/models ./models-backup-$(date +%Y%m%d)

# Configuration backup
kubectl get all,configmap,secret -n computer-vision -o yaml > cv-config-backup-$(date +%Y%m%d).yaml
```

### Log Management

```bash
# Configure log rotation
kubectl create configmap logrotate-config -n computer-vision --from-file=logrotate.conf

# Clean old logs
kubectl exec -n computer-vision <pod-name> -- find /var/log -name "*.log" -mtime +7 -delete
```

---

## Upgrade Procedures

### Rolling Update

```bash
# Update to new version
helm upgrade computer-vision datacraft/computer-vision \
  --namespace computer-vision \
  --version 1.1.0 \
  --values values-production.yaml

# Monitor rollout
kubectl rollout status deployment/computer-vision-api -n computer-vision

# Verify deployment
kubectl get pods -n computer-vision
curl -f http://cv.your-domain.com/health
```

### Blue-Green Deployment

```bash
# Deploy green environment
helm install computer-vision-green datacraft/computer-vision \
  --namespace computer-vision-green \
  --version 1.1.0 \
  --values values-production.yaml

# Test green environment
kubectl port-forward -n computer-vision-green svc/computer-vision-api 8001:8000

# Switch traffic
kubectl patch ingress computer-vision-ingress -n computer-vision -p '{"spec":{"rules":[{"host":"cv.your-domain.com","http":{"paths":[{"path":"/","pathType":"Prefix","backend":{"service":{"name":"computer-vision-green","port":{"number":8000}}}}]}}]}}'

# Cleanup blue environment
helm uninstall computer-vision -n computer-vision
```

### Rollback Procedures

```bash
# Rollback to previous version
helm rollback computer-vision -n computer-vision

# Verify rollback
kubectl rollout status deployment/computer-vision-api -n computer-vision

# Check application health
curl -f http://cv.your-domain.com/health
```

---

## Support and Documentation

For deployment support:

**Documentation:** https://docs.datacraft.co.ke/computer-vision/deployment  
**Support Email:** devops-support@datacraft.co.ke  
**Community Forum:** https://community.datacraft.co.ke/deployment  
**GitHub Issues:** https://github.com/datacraft/apg-computer-vision/issues  

---

*This deployment guide is regularly updated. For the latest version, check the documentation portal.*

**Last Updated:** January 27, 2025  
**Version:** 1.0.0  
**© 2025 Datacraft. All rights reserved.**