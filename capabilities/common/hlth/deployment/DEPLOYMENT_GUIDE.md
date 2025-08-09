# APG System Health Management (HLTH) - Deployment Guide

Copyright © 2025 Datacraft - www.datacraft.co.ke  
Author: Nyimbi Odero <nyimbi@gmail.com>

## Overview

This guide provides comprehensive instructions for deploying the APG System Health Management (HLTH) capability to production Kubernetes environments. The HLTH system provides revolutionary health management with ML-powered capabilities, autonomous remediation, and enterprise-grade features.

## Prerequisites

### System Requirements

- **Kubernetes**: 1.24+ with RBAC enabled
- **CPU**: Minimum 4 cores (8+ cores recommended)
- **Memory**: Minimum 8GB RAM (16GB+ recommended)
- **Storage**: Minimum 200GB (with high-performance SSD recommended)
- **Network**: Load balancer support for ingress

### Required Tools

- `kubectl` (1.24+)
- `helm` (3.8+) - optional but recommended
- `docker` - for custom image builds
- `aws` CLI - if using S3 for backups

### Access Requirements

- Kubernetes cluster admin access
- Container registry access (for custom images)
- DNS management capability
- SSL certificate management

## Pre-Deployment Checklist

### 1. Cluster Preparation

```bash
# Verify cluster access
kubectl cluster-info

# Check node resources
kubectl top nodes

# Verify storage classes
kubectl get storageclass

# Check ingress controller
kubectl get pods -n ingress-nginx
```

### 2. Storage Configuration

The deployment requires three storage classes:
- `fast-ssd`: High-performance SSD storage for databases
- `nfs-storage`: Shared storage for ML models
- Default storage class for temporary data

```bash
# Example storage class for fast SSD (adjust for your environment)
cat <<EOF | kubectl apply -f -
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/aws-ebs  # Adjust for your cloud provider
parameters:
  type: gp3
  fsType: ext4
allowVolumeExpansion: true
reclaimPolicy: Retain
EOF
```

### 3. DNS and TLS Preparation

Update the following domains in deployment files:
- `hlth.your-domain.com` - Main application
- `api.hlth.your-domain.com` - API endpoint
- `grafana.hlth.your-domain.com` - Monitoring dashboard
- `prometheus.hlth.your-domain.com` - Metrics
- `jaeger.hlth.your-domain.com` - Tracing

## Deployment Process

### Method 1: Automated Deployment (Recommended)

Use the automated deployment script:

```bash
cd deployment/scripts

# Dry run to verify configuration
./deploy.sh --dry-run

# Production deployment
./deploy.sh --environment production

# Check deployment status
./health-check.sh --verbose
```

### Method 2: Manual Step-by-Step Deployment

#### Step 1: Create Namespace and Resources

```bash
# Create namespace
kubectl apply -f kubernetes/namespace.yaml

# Deploy secrets (update values first!)
kubectl apply -f kubernetes/secrets.yaml

# Deploy ConfigMaps
kubectl apply -f kubernetes/configmap.yaml
```

#### Step 2: Deploy Database Layer

```bash
# Deploy PostgreSQL and Redis
kubectl apply -f kubernetes/database.yaml

# Wait for databases to be ready
kubectl wait --for=condition=ready pod -l app=postgres -n hlth --timeout=300s
kubectl wait --for=condition=ready pod -l app=redis -n hlth --timeout=300s
```

#### Step 3: Deploy Core Services

```bash
# Deploy HLTH services
kubectl apply -f kubernetes/services.yaml

# Wait for services to be ready
kubectl wait --for=condition=available deployment -l app.kubernetes.io/name=apg-hlth -n hlth --timeout=600s
```

#### Step 4: Deploy Monitoring Stack

```bash
# Deploy monitoring
kubectl apply -f kubernetes/monitoring.yaml

# Wait for monitoring services
kubectl wait --for=condition=available deployment -l app.kubernetes.io/part-of=monitoring -n hlth --timeout=300s
```

#### Step 5: Configure Ingress

```bash
# Deploy ingress (update domain names first!)
kubectl apply -f kubernetes/ingress.yaml
```

## Configuration

### Security Configuration

#### 1. Update Secrets

**Critical**: Change all default secret values before deployment:

```bash
# Generate new passwords
DB_PASSWORD=$(openssl rand -base64 32)
REDIS_PASSWORD=$(openssl rand -base64 16)
SECRET_KEY=$(openssl rand -base64 64)
JWT_SECRET=$(openssl rand -base64 48)

# Update secrets
kubectl create secret generic hlth-secrets \
  --from-literal=db-password="$DB_PASSWORD" \
  --from-literal=redis-password="$REDIS_PASSWORD" \
  --from-literal=secret-key="$SECRET_KEY" \
  --from-literal=jwt-secret="$JWT_SECRET" \
  -n hlth --dry-run=client -o yaml | kubectl apply -f -
```

#### 2. Configure TLS Certificates

Using cert-manager (recommended):

```bash
# Install cert-manager if not already installed
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Create ClusterIssuer
cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@your-domain.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

### Application Configuration

#### 1. Environment-Specific Settings

Update `kubernetes/configmap.yaml` for your environment:

```yaml
data:
  # Production settings
  log_level: "INFO"
  workers: "8"
  processing_batch_size: "500"
  
  # ML Configuration
  ml_enabled: "true"
  training_schedule: "0 2 * * *"  # Daily at 2 AM
  
  # Monitoring
  metrics_enabled: "true"
  tracing_enabled: "true"
  
  # Enterprise Features
  multi_tenancy_enabled: "true"
  compliance_enabled: "true"
```

#### 2. Resource Limits

Adjust resource limits based on your cluster capacity:

```yaml
resources:
  requests:
    memory: "1Gi"     # Adjust based on usage
    cpu: "500m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

## Post-Deployment Tasks

### 1. Verify Deployment

```bash
# Run comprehensive health checks
./deployment/scripts/health-check.sh --verbose

# Check all pods are running
kubectl get pods -n hlth

# Check services
kubectl get services -n hlth

# Check ingress
kubectl get ingress -n hlth
```

### 2. Configure DNS

Point your domains to the ingress controller's external IP:

```bash
# Get ingress IP
kubectl get ingress -n hlth hlth-ingress -o jsonpath='{.status.loadBalancer.ingress[0].ip}'

# Create DNS records (example for AWS Route 53)
aws route53 change-resource-record-sets --hosted-zone-id ZXXXXXXXXXXXXX --change-batch '{
  "Changes": [{
    "Action": "CREATE",
    "ResourceRecordSet": {
      "Name": "hlth.your-domain.com",
      "Type": "A",
      "TTL": 300,
      "ResourceRecords": [{"Value": "YOUR_INGRESS_IP"}]
    }
  }]
}'
```

### 3. Test API Endpoints

```bash
# Test health endpoint
curl -k https://hlth.your-domain.com/health

# Test API endpoint
curl -k https://hlth.your-domain.com/api/v1/health

# Test authentication (after setting up first user)
curl -k -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  https://hlth.your-domain.com/api/v1/components
```

### 4. Configure Monitoring

Access monitoring dashboards:
- **Grafana**: `https://grafana.hlth.your-domain.com`
- **Prometheus**: `https://prometheus.hlth.your-domain.com`
- **Jaeger**: `https://jaeger.hlth.your-domain.com`

Default Grafana credentials:
- Username: `admin`
- Password: Check `grafana-admin-secret`

### 5. Set Up Alerting

Configure alerting channels in Prometheus/Grafana:

```yaml
# Example Slack notification
- name: 'slack-alerts'
  slack_configs:
  - api_url: 'YOUR_SLACK_WEBHOOK_URL'
    channel: '#alerts'
    title: 'HLTH Alert'
    text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
```

## Backup and Recovery

### 1. Automated Backups

Set up regular backups:

```bash
# Create backup manually
./deployment/scripts/backup.sh --s3-bucket your-backup-bucket

# Set up automated backups (add to crontab)
0 2 * * * /path/to/deployment/scripts/backup.sh --s3-bucket your-backup-bucket
```

### 2. Disaster Recovery

Recovery process:

```bash
# 1. Restore database from backup
kubectl exec -n hlth deployment/postgres -- psql -U hlth hlth < backup.sql

# 2. Restore Redis data
kubectl cp backup.rdb hlth/redis-pod:/data/dump.rdb

# 3. Restart services
kubectl rollout restart deployment -n hlth
```

## Maintenance

### 1. Regular Health Checks

```bash
# Weekly health check
./deployment/scripts/health-check.sh --verbose > weekly-health-$(date +%Y%m%d).log
```

### 2. Updates and Upgrades

```bash
# Update application images
kubectl set image deployment/hlth-api-gateway api-gateway=datacraft/apg-hlth-api-gateway:1.1.0 -n hlth

# Rolling update
kubectl rollout restart deployment/hlth-api-gateway -n hlth
kubectl rollout status deployment/hlth-api-gateway -n hlth
```

### 3. Scaling

```bash
# Scale services based on load
kubectl scale deployment hlth-api-gateway --replicas=5 -n hlth
kubectl scale deployment hlth-health-service --replicas=3 -n hlth
```

### 4. Log Management

```bash
# View logs
kubectl logs -n hlth deployment/hlth-api-gateway -c api-gateway --tail=100

# Stream logs
kubectl logs -n hlth -f deployment/hlth-api-gateway -c api-gateway
```

## Troubleshooting

### Common Issues

#### 1. Pod Startup Issues

```bash
# Check pod status
kubectl describe pod -n hlth <pod-name>

# Check events
kubectl get events -n hlth --sort-by='.lastTimestamp'

# Check logs
kubectl logs -n hlth <pod-name> --previous
```

#### 2. Database Connection Issues

```bash
# Test database connectivity
kubectl exec -n hlth deployment/postgres -- pg_isready -U hlth -d hlth

# Check database logs
kubectl logs -n hlth deployment/postgres
```

#### 3. Ingress/Networking Issues

```bash
# Check ingress controller
kubectl get pods -n ingress-nginx

# Test internal service connectivity
kubectl exec -n hlth deployment/hlth-api-gateway -- curl http://postgres-service:5432
```

#### 4. Performance Issues

```bash
# Check resource usage
kubectl top pods -n hlth
kubectl top nodes

# Check HPA status
kubectl get hpa -n hlth
```

### Recovery Procedures

#### 1. Service Recovery

```bash
# Restart individual service
kubectl rollout restart deployment/hlth-api-gateway -n hlth

# Force pod recreation
kubectl delete pod -n hlth -l app=hlth-api-gateway
```

#### 2. Database Recovery

```bash
# Restart database
kubectl rollout restart statefulset/postgres -n hlth

# Restore from backup if needed
kubectl exec -n hlth postgres-0 -- psql -U hlth hlth < /backup/database.sql
```

## Security Best Practices

### 1. Access Control

- Use RBAC for fine-grained access control
- Implement network policies
- Regular security audits

### 2. Secret Management

- Rotate secrets regularly
- Use external secret management (AWS Secrets Manager, HashiCorp Vault)
- Never store secrets in code

### 3. Network Security

- Implement ingress filtering
- Use TLS for all communications
- Regular vulnerability scanning

### 4. Compliance

- Enable audit logging
- Implement data retention policies
- Regular compliance audits

## Support and Contact

For deployment issues or questions:

- **Documentation**: Check the comprehensive docs in the `docs/` directory
- **Issues**: Create issues in the project repository
- **Email**: Support available at nyimbi@gmail.com
- **Website**: Visit www.datacraft.co.ke

## License

Copyright © 2025 Datacraft. All rights reserved.

This deployment guide is part of the APG System Health Management capability.