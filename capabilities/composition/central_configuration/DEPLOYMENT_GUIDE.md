# APG Central Configuration - Production Deployment Guide

## 🚀 Complete Deployment Documentation

**Version**: 1.0.0  
**Date**: January 31, 2025  
**Status**: Production Ready  

---

## 📋 **Pre-Deployment Checklist**

### **Infrastructure Requirements**
- [ ] **Kubernetes Cluster** (v1.24+) or Docker Swarm
- [ ] **PostgreSQL Database** (v13+) with 500GB+ storage
- [ ] **Redis Cache** (v6+) with clustering support
- [ ] **Load Balancer** with SSL termination
- [ ] **Storage** - 1TB+ for backups and logs
- [ ] **Network** - Private VPC with security groups

### **Security Prerequisites**
- [ ] **SSL Certificates** for HTTPS endpoints
- [ ] **Encryption Keys** generated and stored in HSM/Vault
- [ ] **Service Accounts** with minimal required permissions
- [ ] **Firewall Rules** configured for required ports only
- [ ] **Secrets Management** system deployed (Kubernetes Secrets/Vault)
- [ ] **Audit Logging** destination configured

### **Monitoring Setup**
- [ ] **Prometheus** deployed with persistent storage
- [ ] **Grafana** configured with dashboards
- [ ] **AlertManager** with notification channels
- [ ] **Log Aggregation** (ELK/Fluentd) configured
- [ ] **Health Check** endpoints accessible
- [ ] **Performance Monitoring** tools deployed

---

## 🏗️ **Deployment Architecture**

```
┌─────────────────────────────────────────────────────────┐
│                    Load Balancer                        │
│                   (nginx/HAProxy)                       │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────┴───────────────────────────────────────┐
│                 Kubernetes Cluster                      │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   API Pods   │  │   Web Pods   │  │  Worker Pods │  │
│  │   (3 replicas)│  │  (2 replicas)│  │  (2 replicas)│  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  PostgreSQL  │  │    Redis     │  │  Prometheus  │  │
│  │  (Primary +  │  │  (Cluster)   │  │   Grafana    │  │
│  │   Replica)   │  │              │  │              │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                  │
┌─────────────────┴───────────────────────────────────────┐
│              External Integrations                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  │  Slack   │ │  Teams   │ │   JIRA   │ │ Datadog  │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 **Quick Start Deployment**

### **Option 1: Docker Compose (Development/Testing)**

```bash
# Clone the repository
git clone <repository-url>
cd capabilities/composition/central_configuration

# Start all services
docker-compose up -d

# Verify deployment
docker-compose ps
curl http://localhost:8080/health

# Access the interface
open http://localhost:8080
```

### **Option 2: Kubernetes (Production)**

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml

# Wait for deployment
kubectl rollout status deployment/central-config-api -n central-config

# Verify deployment
kubectl get pods -n central-config
kubectl get services -n central-config

# Access the interface
kubectl port-forward svc/central-config-service 8080:80 -n central-config
open http://localhost:8080
```

### **Option 3: Automated Deployment Script**

```bash
# Use the automated deployment script
python deploy.py \
  --environment prod \
  --platform kubernetes \
  --image-tag central-config:latest \
  --replicas 3 \
  --namespace central-config

# Monitor deployment progress
tail -f deployment.log
```

---

## ⚙️ **Detailed Configuration**

### **Environment Variables**

```bash
# Core Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false

# Database Configuration
DATABASE_URL=postgresql://username:password@postgres:5432/central_config
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Redis Configuration
REDIS_URL=redis://redis-cluster:6379/0
REDIS_CLUSTER_NODES=redis-1:6379,redis-2:6379,redis-3:6379

# Security Configuration
ENCRYPTION_KEY_ID=central-config-master-key
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-here
SESSION_TIMEOUT=3600

# AI Configuration
OLLAMA_HOST=http://ollama:11434
OLLAMA_MODELS=llama3.2:3b,codellama:7b,nomic-embed-text

# Monitoring Configuration
PROMETHEUS_URL=http://prometheus:9090
GRAFANA_URL=http://grafana:3000
ENABLE_METRICS=true
METRICS_PORT=9000

# Integration Configuration
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
TEAMS_WEBHOOK_URL=https://outlook.office.com/webhook/...
JIRA_BASE_URL=https://your-company.atlassian.net
DATADOG_API_KEY=your-datadog-api-key
```

### **Kubernetes Configuration**

#### **ConfigMap**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: central-config-config
  namespace: central-config
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  ENABLE_METRICS: "true"
  METRICS_PORT: "9000"
  OLLAMA_HOST: "http://ollama:11434"
  PROMETHEUS_URL: "http://prometheus:9090"
```

#### **Secrets**
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: central-config-secrets
  namespace: central-config
type: Opaque
data:
  DATABASE_URL: <base64-encoded-database-url>
  REDIS_URL: <base64-encoded-redis-url>
  SECRET_KEY: <base64-encoded-secret-key>
  JWT_SECRET_KEY: <base64-encoded-jwt-secret>
  ENCRYPTION_KEY: <base64-encoded-encryption-key>
```

### **Resource Requirements**

#### **Minimum Resources (Development)**
```yaml
resources:
  requests:
    cpu: 500m
    memory: 1Gi
  limits:
    cpu: 1000m
    memory: 2Gi
```

#### **Production Resources**
```yaml
resources:
  requests:
    cpu: 2000m
    memory: 4Gi
  limits:
    cpu: 4000m
    memory: 8Gi
```

---

## 🔐 **Security Configuration**

### **SSL/TLS Setup**

```bash
# Generate SSL certificates (or use Let's Encrypt)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout tls.key -out tls.crt \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=central-config.yourdomain.com"

# Create Kubernetes TLS secret
kubectl create secret tls central-config-tls \
  --cert=tls.crt --key=tls.key -n central-config
```

### **Encryption Keys Setup**

```bash
# Generate encryption keys
python -c "
import secrets
import base64
key = secrets.token_bytes(32)
print('ENCRYPTION_KEY=' + base64.b64encode(key).decode())
"

# Create encryption key secret
kubectl create secret generic central-config-encryption \
  --from-literal=ENCRYPTION_KEY=$ENCRYPTION_KEY \
  -n central-config
```

### **RBAC Configuration**

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: central-config
  name: central-config-role
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps", "secrets"]
  verbs: ["get", "list", "watch", "create", "update", "patch"]
- apiGroups: ["apps"]
  resources: ["deployments"]
  verbs: ["get", "list", "watch", "create", "update", "patch"]
```

---

## 📊 **Monitoring Setup**

### **Prometheus Configuration**

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'central-config'
    static_configs:
      - targets: ['central-config-service:9000']
    metrics_path: /metrics
    scrape_interval: 10s

  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis-exporter'
    static_configs:
      - targets: ['redis-exporter:9121']
```

### **Grafana Dashboard Import**

```bash
# Import pre-built dashboard
curl -X POST \
  http://admin:admin@grafana:3000/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @monitoring/grafana-dashboard.json
```

### **Alert Rules**

```yaml
# alerts.yml
groups:
  - name: central-config-alerts
    rules:
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.5
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: High response time detected

      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: High error rate detected
```

---

## 🔄 **Database Setup**

### **PostgreSQL Configuration**

```sql
-- Create database and user
CREATE DATABASE central_config;
CREATE USER central_config_user WITH ENCRYPTED PASSWORD 'your-password';
GRANT ALL PRIVILEGES ON DATABASE central_config TO central_config_user;

-- Enable required extensions
\c central_config;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
```

### **Database Migration**

```bash
# Run database migrations
python scripts/migrate.py --upgrade

# Initialize sample data (optional)
python scripts/init_db.py --sample-data
```

### **Redis Configuration**

```bash
# Redis cluster configuration
redis-cli --cluster create \
  redis-1:6379 redis-2:6379 redis-3:6379 \
  redis-4:6379 redis-5:6379 redis-6:6379 \
  --cluster-replicas 1
```

---

## 🔌 **Integration Setup**

### **Slack Integration**

```bash
# Create Slack app and get webhook URL
# Configure webhook in environment variables
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."

# Test integration
curl -X POST $SLACK_WEBHOOK_URL \
  -H 'Content-Type: application/json' \
  -d '{"text": "Central Configuration deployment completed!"}'
```

### **Microsoft Teams Integration**

```bash
# Create Teams webhook connector
# Configure webhook in environment variables
export TEAMS_WEBHOOK_URL="https://outlook.office.com/webhook/..."

# Test integration
curl -X POST $TEAMS_WEBHOOK_URL \
  -H 'Content-Type: application/json' \
  -d '{
    "@type": "MessageCard",
    "@context": "https://schema.org/extensions",
    "summary": "Deployment Complete",
    "themeColor": "00FF00",
    "sections": [{
      "activityTitle": "Central Configuration",
      "activitySubtitle": "Deployment completed successfully"
    }]
  }'
```

### **JIRA Integration**

```bash
# Configure JIRA connection
export JIRA_BASE_URL="https://your-company.atlassian.net"
export JIRA_USERNAME="your-username"
export JIRA_API_TOKEN="your-api-token"
export JIRA_PROJECT_KEY="CC"

# Test JIRA integration
python -c "
from integrations.enterprise_connectors import JiraConnector
connector = JiraConnector()
# Test connection
"
```

---

## 🚀 **Deployment Strategies**

### **Blue-Green Deployment**

```bash
# Deploy to green environment
kubectl apply -f k8s/ --namespace=central-config-green

# Verify green deployment
kubectl get pods -n central-config-green

# Switch traffic to green
kubectl patch service central-config-service \
  --patch '{"spec":{"selector":{"version":"green"}}}' \
  -n central-config

# Clean up blue environment
kubectl delete namespace central-config-blue
```

### **Canary Deployment**

```bash
# Deploy canary version (10% traffic)
kubectl apply -f k8s/canary-deployment.yaml

# Monitor canary metrics
kubectl get pods -l version=canary -n central-config

# Promote canary to full deployment
kubectl scale deployment central-config-canary --replicas=3
kubectl scale deployment central-config-stable --replicas=0
```

### **Rolling Update**

```bash
# Update image tag
kubectl set image deployment/central-config-api \
  central-config=central-config:v1.1.0 \
  -n central-config

# Monitor rollout
kubectl rollout status deployment/central-config-api -n central-config

# Rollback if needed
kubectl rollout undo deployment/central-config-api -n central-config
```

---

## 🔍 **Health Checks and Verification**

### **Application Health Checks**

```bash
# Basic health check
curl http://central-config-service/health

# Detailed system status
curl http://central-config-service/health/detailed

# Database connectivity
curl http://central-config-service/health/database

# AI engine status
curl http://central-config-service/health/ai

# Integration status
curl http://central-config-service/health/integrations
```

### **Performance Verification**

```bash
# Load testing with Apache Bench
ab -n 1000 -c 10 http://central-config-service/api/v1/configurations

# Response time testing
curl -w "@curl-format.txt" -o /dev/null -s http://central-config-service/health

# Memory and CPU usage
kubectl top pods -n central-config
```

### **Security Verification**

```bash
# SSL certificate verification
openssl s_client -connect central-config.yourdomain.com:443 -servername central-config.yourdomain.com

# Port scanning
nmap -p 80,443,8080 central-config.yourdomain.com

# Vulnerability scanning (example with trivy)
trivy image central-config:latest
```

---

## 🚨 **Troubleshooting Guide**

### **Common Issues**

#### **Database Connection Issues**
```bash
# Check database connectivity
kubectl exec -it deploy/central-config-api -n central-config -- \
  psql $DATABASE_URL -c "SELECT 1"

# Check database logs
kubectl logs -l app=postgresql -n central-config

# Verify database secrets
kubectl get secret central-config-secrets -o yaml -n central-config
```

#### **Redis Connection Issues**
```bash
# Test Redis connectivity
kubectl exec -it deploy/central-config-api -n central-config -- \
  redis-cli -h redis ping

# Check Redis cluster status
kubectl exec -it redis-0 -n central-config -- \
  redis-cli cluster info
```

#### **Pod Startup Issues**
```bash
# Check pod events
kubectl describe pod <pod-name> -n central-config

# View pod logs
kubectl logs <pod-name> -n central-config --previous

# Check resource constraints
kubectl top pod <pod-name> -n central-config
```

#### **Performance Issues**
```bash
# Check resource usage
kubectl top pods -n central-config

# Monitor response times
curl -w "%{time_total}\n" -o /dev/null -s http://central-config-service/health

# Check database performance
kubectl exec -it postgres-0 -n central-config -- \
  psql -c "SELECT * FROM pg_stat_activity;"
```

### **Log Analysis**

```bash
# Application logs
kubectl logs -f deployment/central-config-api -n central-config

# Filtered error logs
kubectl logs deployment/central-config-api -n central-config | grep ERROR

# Aggregated logs (if using ELK)
curl -X GET "elasticsearch:9200/central-config-logs/_search" -H 'Content-Type: application/json' -d'
{
  "query": {
    "range": {
      "@timestamp": {
        "gte": "now-1h"
      }
    }
  }
}'
```

---

## 📈 **Performance Tuning**

### **Database Optimization**

```sql
-- PostgreSQL tuning
ALTER SYSTEM SET shared_buffers = '512MB';
ALTER SYSTEM SET effective_cache_size = '2GB';
ALTER SYSTEM SET work_mem = '16MB';
ALTER SYSTEM SET maintenance_work_mem = '256MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.7;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;

-- Reload configuration
SELECT pg_reload_conf();
```

### **Redis Optimization**

```bash
# Redis configuration
redis-cli CONFIG SET maxmemory 1gb
redis-cli CONFIG SET maxmemory-policy allkeys-lru
redis-cli CONFIG SET save "900 1 300 10 60 10000"
```

### **Application Tuning**

```yaml
# Kubernetes resource optimization
resources:
  requests:
    cpu: 2000m
    memory: 4Gi
  limits:
    cpu: 4000m
    memory: 8Gi

# JVM tuning for Python applications
env:
- name: PYTHONUNBUFFERED
  value: "1"
- name: PYTHON_MALLOC_STATS
  value: "1"
```

---

## 🔄 **Backup and Recovery**

### **Database Backup**

```bash
# Automated backup script
#!/bin/bash
BACKUP_DIR="/backups/$(date +%Y-%m-%d)"
mkdir -p $BACKUP_DIR

# Create database backup
pg_dump $DATABASE_URL > $BACKUP_DIR/central_config_$(date +%H%M%S).sql

# Compress backup
gzip $BACKUP_DIR/central_config_*.sql

# Upload to S3 (optional)
aws s3 sync $BACKUP_DIR s3://central-config-backups/$(date +%Y-%m-%d)/
```

### **Configuration Backup**

```bash
# Backup Kubernetes configurations
kubectl get all -n central-config -o yaml > central-config-backup.yaml

# Backup secrets
kubectl get secrets -n central-config -o yaml > central-config-secrets-backup.yaml
```

### **Disaster Recovery**

```bash
# Complete system restore procedure
# 1. Restore database
psql $DATABASE_URL < backup/central_config_backup.sql

# 2. Restore Kubernetes resources
kubectl apply -f central-config-backup.yaml

# 3. Verify restoration
kubectl get pods -n central-config
curl http://central-config-service/health
```

---

## 📞 **Support and Maintenance**

### **Regular Maintenance Tasks**

```bash
# Weekly maintenance script
#!/bin/bash

# Update database statistics
kubectl exec -it postgres-0 -n central-config -- \
  psql -c "ANALYZE;"

# Clear old logs
kubectl exec -it deployment/central-config-api -n central-config -- \
  find /var/log -name "*.log" -mtime +7 -delete

# Restart services if needed
kubectl rollout restart deployment/central-config-api -n central-config

# Update security patches
kubectl set image deployment/central-config-api \
  central-config=central-config:latest-security-patch
```

### **Monitoring Dashboard URLs**

- **Application Dashboard**: http://central-config.yourdomain.com
- **Grafana Monitoring**: http://grafana.yourdomain.com/d/central-config
- **Prometheus Metrics**: http://prometheus.yourdomain.com/graph
- **Kubernetes Dashboard**: http://k8s-dashboard.yourdomain.com

### **Emergency Contacts**

- **Primary Support**: support@datacraft.co.ke
- **Emergency Hotline**: +254-XXX-XXXX
- **Slack Channel**: #central-config-support
- **Documentation**: https://docs.central-config.yourdomain.com

---

## ✅ **Post-Deployment Checklist**

- [ ] All pods running and healthy
- [ ] Database connections working
- [ ] Redis cluster operational
- [ ] SSL certificates valid
- [ ] Health checks passing
- [ ] Monitoring dashboards accessible
- [ ] Alert rules configured
- [ ] Backup systems operational
- [ ] Integration tests passing
- [ ] Performance benchmarks met
- [ ] Security scans completed
- [ ] Documentation updated
- [ ] Team trained on operations
- [ ] Runbooks created
- [ ] Support contacts configured

---

*© 2025 Datacraft. All rights reserved.*  
*For technical support, contact: nyimbi@gmail.com*  
*Website: www.datacraft.co.ke*