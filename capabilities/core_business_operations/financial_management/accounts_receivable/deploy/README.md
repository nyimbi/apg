# APG Accounts Receivable - Deployment Guide

Complete deployment configuration and documentation for the APG Accounts Receivable capability.

## Overview

This directory contains comprehensive deployment configurations for the APG Accounts Receivable capability, supporting multiple deployment scenarios:

- **Development**: Local development with Docker Compose
- **Staging**: Kubernetes staging environment
- **Production**: Enterprise Kubernetes production deployment
- **Testing**: Automated CI/CD pipeline integration

## Deployment Architecture

### Container Strategy

```
📦 APG AR Container
├── 🖥️  API Service (FastAPI + uvicorn)
├── 👷 Worker Service (Background task processing)
├── ⏰ Scheduler Service (Cron-like scheduled tasks)
└── 🏥 Health Check Service (Comprehensive monitoring)
```

### Infrastructure Components

- **Application Layer**: API, Worker, Scheduler services
- **Data Layer**: PostgreSQL database with multi-tenant support
- **Cache Layer**: Redis for session management and task queues
- **Load Balancer**: NGINX with SSL termination
- **Monitoring**: Prometheus + Grafana stack
- **Security**: TLS encryption, secrets management, network policies

## Quick Start

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- Kubernetes 1.25+ (for K8s deployment)
- kubectl configured
- Helm 3.0+ (optional)

### Local Development Deployment

```bash
# Clone the repository
git clone https://github.com/datacraft/apg-platform.git
cd apg-platform/capabilities/core_financials/accounts_receivable

# Start development environment
cd deploy
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f ar-api

# Access services
# API: http://localhost:8000
# Grafana: http://localhost:3000 (admin/admin123)
# Prometheus: http://localhost:9090
```

### Production Kubernetes Deployment

```bash
# Prepare deployment directory
cd deploy/kubernetes

# Create namespace
kubectl apply -f namespace.yaml

# Deploy configuration and secrets
kubectl apply -f configmap.yaml
kubectl apply -f secrets.yaml

# Deploy services
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml

# Monitor deployment
kubectl get pods -n apg-ar -w
kubectl logs -f deployment/apg-ar-api -n apg-ar
```

## Configuration

### Environment Variables

#### Required Configuration

```bash
# Environment
APG_ENVIRONMENT=production

# Database
DATABASE_URL=postgresql://user:password@host:port/database

# Cache
REDIS_URL=redis://host:port/db

# Security
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-key
ENCRYPTION_KEY=your-encryption-key

# APG Platform Integration
APG_PLATFORM_URL=https://platform.apg.company.com
APG_API_KEY=your-apg-api-key
```

#### Optional Configuration

```bash
# Performance
APG_AR_WORKERS=4
APG_AR_LOG_LEVEL=INFO
WORKER_CONCURRENCY=4

# AI Services
FEDERATED_LEARNING_URL=https://ai.apg.platform/federated-learning
AI_ORCHESTRATION_URL=https://ai.apg.platform/orchestration
TIME_SERIES_ANALYTICS_URL=https://analytics.apg.platform/time-series

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
```

### Resource Requirements

#### Minimum Requirements (Development)

| Component | CPU | Memory | Storage |
|-----------|-----|--------|---------|
| API Service | 250m | 512Mi | - |
| Worker Service | 125m | 256Mi | - |
| Scheduler | 50m | 128Mi | - |
| PostgreSQL | 250m | 512Mi | 10Gi |
| Redis | 100m | 128Mi | 1Gi |

#### Production Requirements

| Component | CPU | Memory | Storage |
|-----------|-----|--------|---------|
| API Service (3 replicas) | 500m | 1Gi | - |
| Worker Service (2 replicas) | 250m | 512Mi | - |
| Scheduler (1 replica) | 100m | 256Mi | - |
| PostgreSQL | 1000m | 2Gi | 100Gi |
| Redis | 250m | 256Mi | 10Gi |

## Security

### Secrets Management

The deployment uses Kubernetes secrets for sensitive data. In production, integrate with external secret management:

```bash
# Example: Using sealed-secrets
kubeseal --format yaml < secrets.yaml > sealed-secrets.yaml

# Example: Using external-secrets with HashiCorp Vault
kubectl apply -f external-secrets/vault-secret-store.yaml
kubectl apply -f external-secrets/vault-secret.yaml
```

### Network Security

- **TLS Encryption**: All traffic encrypted with TLS 1.3
- **Network Policies**: Strict ingress/egress rules
- **Service Mesh**: Optional Istio integration
- **Firewall Rules**: Port restrictions and IP whitelisting

### Container Security

- **Non-root User**: Containers run as non-root user (UID 1000)
- **Read-only Filesystem**: Root filesystem is read-only
- **Capability Dropping**: All capabilities dropped
- **Security Scanning**: Trivy scanning in CI/CD

## Monitoring and Observability

### Health Checks

The deployment includes comprehensive health checking:

```bash
# Container health check
docker exec apg-ar-api python /opt/apg/healthcheck.py

# Kubernetes health checks
kubectl get pods -n apg-ar
kubectl describe pod <pod-name> -n apg-ar
```

### Metrics and Monitoring

- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization dashboards
- **Custom Metrics**: Business-specific AR metrics
- **Log Aggregation**: Centralized logging with structured JSON

### Grafana Dashboards

Pre-configured dashboards include:

1. **APG AR Overview**: High-level system metrics
2. **API Performance**: Request latency and throughput
3. **Database Performance**: Query performance and connections
4. **Business Metrics**: AR-specific KPIs
5. **Infrastructure**: Resource utilization

## Scaling

### Horizontal Scaling

```bash
# Scale API services
kubectl scale deployment apg-ar-api --replicas=5 -n apg-ar

# Scale worker services
kubectl scale deployment apg-ar-worker --replicas=4 -n apg-ar
```

### Vertical Scaling

Update resource requests/limits in `deployment.yaml`:

```yaml
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
```

### Auto-scaling

```bash
# Enable Horizontal Pod Autoscaler
kubectl apply -f hpa.yaml

# Monitor auto-scaling
kubectl get hpa -n apg-ar
```

## Backup and Disaster Recovery

### Database Backup

```bash
# Manual backup
kubectl exec -it postgres-pod -n apg-ar -- pg_dump -U ar_user apg_ar > backup.sql

# Scheduled backups using CronJob
kubectl apply -f backup/postgres-backup-cronjob.yaml
```

### Configuration Backup

```bash
# Backup Kubernetes resources
kubectl get all,configmap,secret -n apg-ar -o yaml > apg-ar-backup.yaml
```

### Disaster Recovery

1. **Database Recovery**: Restore from latest backup
2. **Configuration Recovery**: Reapply Kubernetes manifests
3. **Service Recovery**: Rolling restart of services
4. **Validation**: Health checks and smoke tests

## CI/CD Integration

### GitHub Actions

The deployment includes a comprehensive CI/CD pipeline:

```bash
# Trigger deployment
git push origin main  # Triggers staging deployment
git tag v1.0.0 && git push origin v1.0.0  # Triggers production deployment
```

### Pipeline Stages

1. **Code Quality**: Linting, formatting, type checking
2. **Security Scanning**: Dependency and container scanning
3. **Testing**: Unit, integration, and performance tests
4. **Building**: Container image building and pushing
5. **Deployment**: Automated deployment to environments
6. **Validation**: Post-deployment health checks

## Troubleshooting

### Common Issues

#### Pod Startup Issues

```bash
# Check pod status
kubectl get pods -n apg-ar

# View pod events
kubectl describe pod <pod-name> -n apg-ar

# Check logs
kubectl logs <pod-name> -n apg-ar
```

#### Database Connection Issues

```bash
# Test database connectivity
kubectl exec -it apg-ar-api-pod -n apg-ar -- python -c "
import psycopg2
conn = psycopg2.connect('postgresql://user:pass@host:port/db')
print('Database connected successfully')
"
```

#### Performance Issues

```bash
# Check resource usage
kubectl top pods -n apg-ar

# View metrics
curl http://localhost:9090/api/v1/query?query=rate(http_requests_total[5m])
```

### Debugging Commands

```bash
# Get comprehensive system status
kubectl get all -n apg-ar

# Check resource quotas
kubectl describe resourcequota -n apg-ar

# View recent events
kubectl get events -n apg-ar --sort-by=.metadata.creationTimestamp

# Debug networking
kubectl exec -it apg-ar-api-pod -n apg-ar -- nslookup postgres-service
```

## Performance Tuning

### Database Optimization

```sql
-- Example PostgreSQL optimizations
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
SELECT pg_reload_conf();
```

### Application Optimization

- **Connection Pooling**: Configure optimal pool sizes
- **Async Operations**: Maximize async/await usage
- **Caching**: Implement Redis caching strategies
- **Batch Processing**: Optimize bulk operations

### Container Optimization

- **Multi-stage Builds**: Minimize image size
- **Layer Caching**: Optimize Docker layer caching
- **Resource Limits**: Right-size resource allocation
- **Init Containers**: Use init containers for setup

## Maintenance

### Regular Maintenance Tasks

#### Daily
- Monitor system health and alerts
- Review error logs
- Check backup completion

#### Weekly
- Update security patches
- Review performance metrics
- Database maintenance (VACUUM, ANALYZE)

#### Monthly
- Review capacity planning
- Update monitoring dashboards
- Security audit review

### Upgrade Procedure

1. **Backup**: Create full system backup
2. **Test**: Deploy to staging environment
3. **Validate**: Run comprehensive tests
4. **Deploy**: Rolling update to production
5. **Monitor**: Watch metrics and logs
6. **Rollback**: Have rollback plan ready

## Support and Documentation

### Additional Resources

- [APG Platform Documentation](https://docs.apg.platform)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Docker Documentation](https://docs.docker.com/)
- [Prometheus Documentation](https://prometheus.io/docs/)

### Getting Help

- **Email**: support@datacraft.co.ke
- **Slack**: #apg-ar-support
- **Issues**: GitHub Issues
- **Documentation**: Internal wiki

---

*For additional deployment support and customization, contact the APG Platform team or refer to the enterprise support documentation.*