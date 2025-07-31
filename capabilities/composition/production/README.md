# APG Composition Production Deployment

Complete production deployment configuration for the APG Composition capability with enterprise-grade infrastructure, monitoring, and security.

## Architecture Overview

The production deployment provides:

- **High Availability**: Multi-AZ deployment with auto-scaling
- **Security**: Network isolation, encryption at rest and in transit, RBAC
- **Monitoring**: Comprehensive metrics, logging, and alerting
- **Backup & Recovery**: Automated backups with point-in-time recovery
- **Performance**: Optimized for enterprise workloads with caching and CDN

## Infrastructure Components

### Core Services
- **EKS Cluster**: Kubernetes orchestration with auto-scaling nodes
- **RDS PostgreSQL**: Primary database with read replica
- **ElastiCache Redis**: Distributed caching and session management
- **Application Load Balancer**: Traffic distribution and SSL termination

### Monitoring & Observability
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization and dashboards
- **ELK Stack**: Centralized logging and analysis
- **CloudWatch**: AWS native monitoring integration

### Security
- **WAF**: Web Application Firewall protection
- **Secrets Manager**: Secure credential management
- **Network ACLs**: Layer-4 network security
- **Security Groups**: Application-level firewall rules

## Deployment Options

### 1. Docker Compose (Development/Staging)

```bash
# Set environment variables
export DB_PASSWORD=$(openssl rand -base64 32)
export REDIS_PASSWORD=$(openssl rand -base64 32)
export JWT_SECRET=$(openssl rand -base64 64)
export SECRET_KEY=$(openssl rand -base64 32)

# Deploy the stack
docker-compose -f docker-compose.prod.yml up -d

# Verify deployment
docker-compose -f docker-compose.prod.yml ps
```

### 2. Kubernetes Deployment

```bash
# Create namespace
kubectl create namespace apg-composition

# Create secrets
kubectl create secret generic composition-secrets \
  --from-literal=database-url="postgresql://..." \
  --from-literal=redis-url="redis://..." \
  --from-literal=jwt-secret="$(openssl rand -base64 64)" \
  --from-literal=secret-key="$(openssl rand -base64 32)" \
  -n apg-composition

# Deploy application
kubectl apply -f kubernetes/ -n apg-composition

# Verify deployment
kubectl get pods -n apg-composition
kubectl get services -n apg-composition
```

### 3. Terraform Infrastructure Provisioning

```bash
# Initialize Terraform
cd terraform/
terraform init

# Plan deployment
terraform plan -var-file="production.tfvars"

# Apply infrastructure
terraform apply -var-file="production.tfvars"

# Get outputs
terraform output
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `ENV` | Environment (production) | Yes |
| `DATABASE_URL` | PostgreSQL connection string | Yes |
| `REDIS_URL` | Redis connection string | Yes |
| `SECRET_KEY` | Application secret key | Yes |
| `JWT_SECRET` | JWT signing secret | Yes |
| `APG_INTEGRATION_TOKEN` | APG platform integration token | Yes |
| `SENTRY_DSN` | Error tracking DSN | No |
| `LOG_LEVEL` | Logging level (INFO) | No |

### Resource Requirements

#### Minimum Requirements
- **CPU**: 2 vCPUs per instance
- **Memory**: 4GB RAM per instance  
- **Storage**: 100GB for database
- **Network**: 1Gbps bandwidth

#### Recommended Production
- **CPU**: 4 vCPUs per instance
- **Memory**: 8GB RAM per instance
- **Storage**: 500GB for database with auto-scaling
- **Network**: 10Gbps bandwidth
- **Replicas**: 3+ instances for high availability

## Monitoring & Alerting

### Metrics Collection

Key metrics monitored:
- **Application**: Request rate, response time, error rate
- **Database**: Connection count, query performance, replication lag
- **Cache**: Hit rate, memory usage, eviction rate
- **Infrastructure**: CPU, memory, disk, network utilization

### Dashboards

Access monitoring dashboards:
- **Grafana**: `https://grafana.apg.datacraft.co.ke`
- **Prometheus**: `https://prometheus.apg.datacraft.co.ke`
- **Kibana**: `https://kibana.apg.datacraft.co.ke`

### Alerting Rules

Critical alerts configured for:
- Application downtime (> 1 minute)
- High error rate (> 5%)
- Database connection failures
- Memory usage > 85%
- Disk usage > 90%

## Security Configuration

### Network Security
- Private subnets for application and database tiers
- Public subnets only for load balancers
- Network ACLs and Security Groups for traffic control
- VPC Flow Logs for network monitoring

### Data Security
- Encryption at rest for all data stores
- Encryption in transit with TLS 1.3
- Secrets stored in AWS Secrets Manager
- Database credentials rotated automatically

### Access Control
- RBAC for Kubernetes cluster access
- IAM roles with least privilege principle
- Service accounts for application components
- Multi-factor authentication required

## Backup & Recovery

### Automated Backups
- **Database**: Daily snapshots with 30-day retention
- **Configuration**: Version-controlled infrastructure code
- **Application Data**: S3 cross-region replication

### Disaster Recovery
- **RTO**: 15 minutes (Recovery Time Objective)
- **RPO**: 5 minutes (Recovery Point Objective)
- **Multi-AZ**: Automatic failover for database
- **Cross-Region**: Backup replication for compliance

### Recovery Procedures

1. **Database Recovery**:
   ```bash
   # Restore from point-in-time
   aws rds restore-db-instance-to-point-in-time \
     --source-db-instance-identifier composition-db \
     --target-db-instance-identifier composition-db-restored \
     --restore-time 2024-01-01T12:00:00.000Z
   ```

2. **Application Recovery**:
   ```bash
   # Rollback deployment
   kubectl rollout undo deployment/composition-api -n apg-composition
   
   # Scale up replicas
   kubectl scale deployment composition-api --replicas=5 -n apg-composition
   ```

## Performance Optimization

### Scaling Configuration

- **Horizontal Pod Autoscaler**: Scale 3-20 pods based on CPU/memory
- **Vertical Pod Autoscaler**: Automatic resource right-sizing
- **Cluster Autoscaler**: Node scaling based on demand
- **Database Scaling**: Read replicas for read-heavy workloads

### Caching Strategy
- **Application Cache**: Redis for session and API response caching
- **Database Cache**: PostgreSQL query result caching
- **CDN**: CloudFront for static asset delivery
- **DNS Cache**: Route 53 resolver for DNS optimization

## Troubleshooting

### Common Issues

1. **Database Connection Errors**
   ```bash
   # Check database status
   kubectl exec -it <pod-name> -- psql $DATABASE_URL -c "SELECT 1;"
   
   # Verify secrets
   kubectl get secret composition-secrets -o yaml
   ```

2. **High Memory Usage**
   ```bash
   # Check pod resource usage
   kubectl top pods -n apg-composition
   
   # Scale up resources
   kubectl patch deployment composition-api -p '{"spec":{"template":{"spec":{"containers":[{"name":"composition-api","resources":{"limits":{"memory":"2Gi"}}}]}}}}'
   ```

3. **Cache Connection Issues**
   ```bash
   # Test Redis connectivity
   kubectl exec -it <pod-name> -- redis-cli -u $REDIS_URL ping
   
   # Check cache metrics
   kubectl port-forward svc/composition-api-service 8001:8001
   curl http://localhost:8001/metrics | grep redis
   ```

### Debugging Tools

- **Logs**: `kubectl logs -f deployment/composition-api -n apg-composition`
- **Shell**: `kubectl exec -it <pod-name> -n apg-composition -- /bin/bash`
- **Metrics**: `curl http://pod-ip:8001/metrics`
- **Health**: `curl http://pod-ip:8000/health`

## Maintenance

### Regular Tasks

- **Weekly**: Review monitoring dashboards and alerts
- **Monthly**: Update dependencies and security patches
- **Quarterly**: Performance review and capacity planning
- **Annually**: Disaster recovery testing

### Update Procedures

1. **Rolling Updates**:
   ```bash
   # Update image
   kubectl set image deployment/composition-api composition-api=apg/composition:v2.1.0 -n apg-composition
   
   # Monitor rollout
   kubectl rollout status deployment/composition-api -n apg-composition
   ```

2. **Blue-Green Deployment**:
   ```bash
   # Deploy to staging environment
   kubectl apply -f kubernetes/ -n apg-composition-staging
   
   # Validate deployment
   # Switch traffic
   # Retire old version
   ```

## Support & Escalation

### Contact Information
- **Platform Team**: platform-team@datacraft.co.ke
- **On-Call**: +254-XXX-XXXX-XXX
- **Slack**: #apg-composition-support

### SLA Commitments
- **Uptime**: 99.9% availability
- **Response Time**: < 200ms (95th percentile)
- **Support**: 24/7 monitoring with 15-minute response time

---

© 2025 Datacraft. All rights reserved.
APG Composition Production Deployment Guide