# APG Event Streaming Bus - Deployment Guide

This guide provides comprehensive instructions for deploying the APG Event Streaming Bus in various environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Local Development](#local-development)
4. [Docker Deployment](#docker-deployment)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [Production Deployment](#production-deployment)
7. [Monitoring Setup](#monitoring-setup)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Tools

- **Docker** 24.0+ with Docker Compose
- **Kubernetes** 1.28+ (for production)
- **kubectl** 1.28+
- **Helm** 3.12+ (optional)
- **AWS CLI** 2.0+ (for AWS deployments)
- **Python** 3.11+
- **Git** 2.40+

### Required Services

- **PostgreSQL** 15+
- **Redis** 7.0+
- **Apache Kafka** 3.0+
- **Prometheus** (for monitoring)
- **Grafana** (for visualization)

## Environment Setup

### Environment Variables

Create environment-specific configuration files:

```bash
# Development (.env.dev)
ENV=development
LOG_LEVEL=DEBUG
DATABASE_URL=postgresql://esb_user:esb_password@localhost:5432/apg_esb_dev
REDIS_URL=redis://localhost:6379/0
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# Staging (.env.staging)
ENV=staging
LOG_LEVEL=INFO
DATABASE_URL=postgresql://esb_user:esb_password@postgres:5432/apg_esb_staging
REDIS_URL=redis://redis:6379/0
KAFKA_BOOTSTRAP_SERVERS=kafka:9092

# Production (.env.prod)
ENV=production
LOG_LEVEL=INFO
DATABASE_URL=postgresql://esb_user:esb_password@postgres:5432/apg_esb_prod
REDIS_URL=redis://redis:6379/0
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
```

### Security Configuration

```bash
# Generate secure secrets
JWT_SECRET_KEY=$(openssl rand -base64 32)
ENCRYPTION_KEY=$(openssl rand -base64 32)
DATABASE_PASSWORD=$(openssl rand -base64 16)
REDIS_PASSWORD=$(openssl rand -base64 16)
```

## Local Development

### Quick Start

```bash
# Clone repository
git clone <repository-url>
cd event_streaming_bus

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Start services
docker-compose up -d postgres redis kafka

# Initialize database
python -m alembic upgrade head

# Run application
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8080
```

### Development Services

Start all development services:

```bash
# Start complete development environment
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f event-streaming-bus
```

## Docker Deployment

### Single Container

```bash
# Build image
docker build -t apg-event-streaming-bus:latest .

# Run container
docker run -d \
  --name apg-esb \
  -p 8080:8080 \
  -p 9090:9090 \
  -e DATABASE_URL=postgresql://... \
  -e REDIS_URL=redis://... \
  -e KAFKA_BOOTSTRAP_SERVERS=kafka:9092 \
  apg-event-streaming-bus:latest
```

### Docker Compose

```bash
# Production deployment
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Check deployment
docker-compose ps
docker-compose logs -f
```

### Multi-stage Build

```dockerfile
# Build for specific target
docker build --target production -t apg-esb:prod .
docker build --target development -t apg-esb:dev .
```

## Kubernetes Deployment

### Namespace Setup

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Verify namespace
kubectl get namespace apg-event-streaming-bus
```

### Configuration

```bash
# Apply configuration
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml

# Verify configuration
kubectl get configmap -n apg-event-streaming-bus
kubectl get secret -n apg-event-streaming-bus
```

### Application Deployment

```bash
# Deploy application
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml

# Check deployment status
kubectl get pods -n apg-event-streaming-bus
kubectl get services -n apg-event-streaming-bus
kubectl get ingress -n apg-event-streaming-bus
```

### Database Setup

```bash
# Deploy PostgreSQL (if not using external)
kubectl apply -f k8s/postgres.yaml

# Deploy Redis
kubectl apply -f k8s/redis.yaml

# Deploy Kafka
kubectl apply -f k8s/kafka.yaml
```

### Scaling

```bash
# Scale API pods
kubectl scale deployment event-streaming-bus --replicas=5 -n apg-event-streaming-bus

# Scale worker pods
kubectl scale deployment event-streaming-bus-worker --replicas=3 -n apg-event-streaming-bus

# Auto-scaling
kubectl apply -f k8s/hpa.yaml
```

## Production Deployment

### Automated Deployment

Use the provided deployment script:

```bash
# Deploy to staging
./scripts/deploy.sh --environment staging

# Deploy to production
./scripts/deploy.sh --environment production --image-tag v1.0.0

# Dry run
./scripts/deploy.sh --environment production --dry-run

# Force deployment
./scripts/deploy.sh --environment production --force
```

### Manual Deployment Steps

1. **Pre-deployment Checks**
   ```bash
   # Check cluster status
   kubectl cluster-info
   kubectl get nodes
   
   # Check current deployment
   kubectl get deployment event-streaming-bus -n apg-event-streaming-bus
   ```

2. **Backup Current State**
   ```bash
   # Backup configurations
   kubectl get all -n apg-event-streaming-bus -o yaml > backup-$(date +%Y%m%d).yaml
   ```

3. **Deploy New Version**
   ```bash
   # Update image
   kubectl set image deployment/event-streaming-bus \
     event-streaming-bus=ghcr.io/datacraft/apg-event-streaming-bus:v1.0.0 \
     -n apg-event-streaming-bus
   
   # Wait for rollout
   kubectl rollout status deployment/event-streaming-bus -n apg-event-streaming-bus
   ```

4. **Verify Deployment**
   ```bash
   # Check pod status
   kubectl get pods -n apg-event-streaming-bus
   
   # Run health check
   kubectl run health-check --rm -i --restart=Never --image=curlimages/curl:latest -- \
     curl -f http://event-streaming-bus.apg-event-streaming-bus:8080/health
   ```

### Blue-Green Deployment

```bash
# Deploy green version
kubectl apply -f k8s/deployment-green.yaml

# Wait for green deployment
kubectl rollout status deployment/event-streaming-bus-green -n apg-event-streaming-bus

# Switch traffic
kubectl patch service event-streaming-bus \
  -p '{"spec":{"selector":{"version":"green"}}}' \
  -n apg-event-streaming-bus

# Remove blue deployment
kubectl delete deployment event-streaming-bus-blue -n apg-event-streaming-bus
```

### Rollback Deployment

```bash
# Rollback to previous version
kubectl rollout undo deployment/event-streaming-bus -n apg-event-streaming-bus

# Rollback to specific revision
kubectl rollout undo deployment/event-streaming-bus --to-revision=2 -n apg-event-streaming-bus

# Check rollout history
kubectl rollout history deployment/event-streaming-bus -n apg-event-streaming-bus
```

## Monitoring Setup

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'event-streaming-bus'
    static_configs:
      - targets: ['event-streaming-bus:9090']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### Grafana Dashboards

Import pre-built dashboards:

```bash
# Copy dashboards
kubectl create configmap grafana-dashboards \
  --from-file=dashboards/ \
  -n monitoring

# Update Grafana deployment
kubectl apply -f monitoring/grafana.yaml
```

### Alerting Rules

```yaml
# alerts.yml
groups:
  - name: esb-alerts
    rules:
      - alert: ESBHighErrorRate
        expr: rate(esb_errors_total[5m]) > 0.05
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate in Event Streaming Bus"
```

### Log Aggregation

```bash
# Deploy Elasticsearch and Kibana
kubectl apply -f monitoring/elasticsearch.yaml
kubectl apply -f monitoring/kibana.yaml

# Configure log forwarding
kubectl apply -f monitoring/fluentd.yaml
```

## Troubleshooting

### Common Issues

#### 1. Pod Startup Issues

```bash
# Check pod logs
kubectl logs -f deployment/event-streaming-bus -n apg-event-streaming-bus

# Describe pod for events
kubectl describe pods -l app.kubernetes.io/name=event-streaming-bus -n apg-event-streaming-bus

# Check resource usage
kubectl top pods -n apg-event-streaming-bus
```

#### 2. Database Connection Issues

```bash
# Test database connectivity
kubectl run db-test --rm -i --restart=Never --image=postgres:15 -- \
  psql postgresql://esb_user:esb_password@postgres:5432/apg_esb -c "SELECT 1"

# Check database logs
kubectl logs -f postgres-0 -n apg-event-streaming-bus
```

#### 3. Kafka Connection Issues

```bash
# Test Kafka connectivity
kubectl run kafka-test --rm -i --restart=Never --image=confluentinc/cp-kafka:7.4.0 -- \
  kafka-topics --bootstrap-server kafka:9092 --list

# Check Kafka logs
kubectl logs -f kafka-0 -n apg-event-streaming-bus
```

#### 4. Memory Issues

```bash
# Check memory usage
kubectl top pods -n apg-event-streaming-bus

# Increase memory limits
kubectl patch deployment event-streaming-bus \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"event-streaming-bus","resources":{"limits":{"memory":"4Gi"}}}]}}}}' \
  -n apg-event-streaming-bus
```

### Debug Commands

```bash
# Get all resources
kubectl get all -n apg-event-streaming-bus

# Check events
kubectl get events -n apg-event-streaming-bus --sort-by='.lastTimestamp'

# Port forward for local access
kubectl port-forward service/event-streaming-bus 8080:8080 -n apg-event-streaming-bus

# Execute shell in pod
kubectl exec -it deployment/event-streaming-bus -n apg-event-streaming-bus -- /bin/bash

# Check configuration
kubectl get configmap esb-config -o yaml -n apg-event-streaming-bus
```

### Performance Tuning

```bash
# Horizontal Pod Autoscaler
kubectl apply -f k8s/hpa.yaml

# Vertical Pod Autoscaler
kubectl apply -f k8s/vpa.yaml

# Resource quotas
kubectl apply -f k8s/resource-quota.yaml

# Network policies
kubectl apply -f k8s/network-policy.yaml
```

### Backup and Recovery

```bash
# Backup database
kubectl exec -n apg-event-streaming-bus postgres-0 -- \
  pg_dump -U esb_user apg_esb > backup-$(date +%Y%m%d).sql

# Backup Kubernetes resources
kubectl get all -n apg-event-streaming-bus -o yaml > k8s-backup-$(date +%Y%m%d).yaml

# Restore database
kubectl exec -i -n apg-event-streaming-bus postgres-0 -- \
  psql -U esb_user apg_esb < backup-20250126.sql
```

## Security Considerations

### TLS Configuration

```bash
# Generate TLS certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout tls.key -out tls.crt \
  -subj "/CN=api.event-streaming-bus.datacraft.co.ke"

# Create TLS secret
kubectl create secret tls esb-tls-secret \
  --cert=tls.crt --key=tls.key \
  -n apg-event-streaming-bus
```

### Network Security

```bash
# Apply network policies
kubectl apply -f k8s/network-policy.yaml

# Check network policies
kubectl get networkpolicy -n apg-event-streaming-bus
```

### RBAC Configuration

```bash
# Apply RBAC
kubectl apply -f k8s/rbac.yaml

# Check service accounts
kubectl get serviceaccount -n apg-event-streaming-bus
```

## Maintenance

### Regular Tasks

```bash
# Update dependencies
pip-compile requirements.in
pip-compile requirements-dev.in

# Security updates
safety check
bandit -r .

# Performance monitoring
kubectl top pods -n apg-event-streaming-bus
kubectl top nodes
```

### Cleanup

```bash
# Clean up old deployments
kubectl delete replicaset -l app.kubernetes.io/name=event-streaming-bus -n apg-event-streaming-bus

# Clean up old images
docker system prune -a

# Clean up old logs
kubectl logs --tail=0 deployment/event-streaming-bus -n apg-event-streaming-bus
```

For more information, see the [Operations Guide](operations.md) and [Troubleshooting Guide](troubleshooting.md).