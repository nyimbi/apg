# Production Deployment Guide

## Overview

This guide covers deploying the APG Billing System to production environments with high availability, security, and scalability considerations.

## Pre-Deployment Checklist

### Infrastructure Requirements

#### Minimum Production Requirements
- **Compute**: 4 vCPUs, 16GB RAM
- **Database**: PostgreSQL 13+ with 100GB SSD storage
- **Cache**: Redis 6+ with 4GB memory
- **Load Balancer**: Application load balancer with SSL termination
- **Storage**: 200GB for logs and backups

#### Recommended Production Setup
- **Compute**: 8 vCPUs, 32GB RAM (auto-scaling group)
- **Database**: PostgreSQL 15 with read replicas, 500GB SSD
- **Cache**: Redis cluster with high availability
- **CDN**: CloudFront or similar for static assets
- **Monitoring**: DataDog, New Relic, or Prometheus stack

### Security Requirements

#### SSL/TLS Configuration
```bash
# Ensure SSL certificates are configured
SSL_CERTIFICATE_PATH=/path/to/ssl/cert.pem
SSL_PRIVATE_KEY_PATH=/path/to/ssl/private.key
SSL_CERTIFICATE_CHAIN_PATH=/path/to/ssl/chain.pem

# Force HTTPS redirect
FORCE_HTTPS=true
HSTS_MAX_AGE=31536000
```

#### Environment Variables Security
```bash
# Use secure secret management
DATABASE_URL="postgresql://user:$(vault kv get -field=password secret/db)@host:5432/db"
STRIPE_SECRET_KEY="$(vault kv get -field=stripe_secret secret/payment_processors)"
ENCRYPTION_KEY="$(vault kv get -field=encryption_key secret/apg_billing)"
```

#### Network Security
- VPC with private subnets for application and database
- Security groups restricting access to necessary ports only
- WAF (Web Application Firewall) for API protection
- DDoS protection enabled

## Deployment Methods

### Method 1: Docker Deployment

#### Production Dockerfile
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd --create-home --shell /bin/bash app

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set ownership and permissions
RUN chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/billing/health || exit 1

# Expose port
EXPOSE 5000

# Start application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "60", "service:app"]
```

#### Production Docker Compose
```yaml
version: '3.8'

services:
  apg-billing:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - STRIPE_SECRET_KEY=${STRIPE_SECRET_KEY}
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/billing/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "5"

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=${POSTGRES_DB}
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups
    restart: unless-stopped
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    restart: unless-stopped
    ports:
      - "6379:6379"

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - apg-billing
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

### Method 2: Kubernetes Deployment

#### Namespace and Resources
```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: apg-billing
---
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: apg-billing-config
  namespace: apg-billing
data:
  FLASK_ENV: "production"
  LOG_LEVEL: "INFO"
  DEFAULT_CURRENCY: "USD"
  REDIS_URL: "redis://redis-service:6379/0"
---
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: apg-billing-secrets
  namespace: apg-billing
type: Opaque
data:
  database-url: <base64-encoded-database-url>
  stripe-secret-key: <base64-encoded-stripe-key>
  encryption-key: <base64-encoded-encryption-key>
```

#### Application Deployment
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: apg-billing
  namespace: apg-billing
spec:
  replicas: 3
  selector:
    matchLabels:
      app: apg-billing
  template:
    metadata:
      labels:
        app: apg-billing
    spec:
      containers:
      - name: apg-billing
        image: datacraft/apg-billing:latest
        ports:
        - containerPort: 5000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: apg-billing-secrets
              key: database-url
        - name: STRIPE_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: apg-billing-secrets
              key: stripe-secret-key
        envFrom:
        - configMapRef:
            name: apg-billing-config
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /billing/health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /billing/health
            port: 5000
          initialDelaySeconds: 5
          periodSeconds: 5
```

#### Services and Ingress
```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: apg-billing-service
  namespace: apg-billing
spec:
  selector:
    app: apg-billing
  ports:
  - protocol: TCP
    port: 80
    targetPort: 5000
  type: ClusterIP
---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: apg-billing-ingress
  namespace: apg-billing
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - billing.yourdomain.com
    secretName: apg-billing-tls
  rules:
  - host: billing.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: apg-billing-service
            port:
              number: 80
```

### Method 3: AWS ECS Deployment

#### Task Definition
```json
{
  "family": "apg-billing",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "apg-billing",
      "image": "your-account.dkr.ecr.region.amazonaws.com/apg-billing:latest",
      "portMappings": [
        {
          "containerPort": 5000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "FLASK_ENV",
          "value": "production"
        }
      ],
      "secrets": [
        {
          "name": "DATABASE_URL",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:apg-billing/database-url"
        },
        {
          "name": "STRIPE_SECRET_KEY", 
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:apg-billing/stripe-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/apg-billing",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": [
          "CMD-SHELL",
          "curl -f http://localhost:5000/billing/health || exit 1"
        ],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

#### ECS Service Configuration
```json
{
  "serviceName": "apg-billing-service",
  "cluster": "apg-billing-cluster",
  "taskDefinition": "apg-billing:1",
  "desiredCount": 3,
  "launchType": "FARGATE",
  "networkConfiguration": {
    "awsvpcConfiguration": {
      "subnets": [
        "subnet-12345678",
        "subnet-87654321"
      ],
      "securityGroups": [
        "sg-apg-billing"
      ],
      "assignPublicIp": "DISABLED"
    }
  },
  "loadBalancers": [
    {
      "targetGroupArn": "arn:aws:elasticloadbalancing:region:account:targetgroup/apg-billing",
      "containerName": "apg-billing",
      "containerPort": 5000
    }
  ],
  "serviceRegistries": [
    {
      "registryArn": "arn:aws:servicediscovery:region:account:service/srv-apg-billing"
    }
  ]
}
```

## Database Setup

### PostgreSQL Production Configuration

#### Master Database Setup
```sql
-- Create database and user
CREATE DATABASE apg_billing_prod;
CREATE USER apg_billing_user WITH PASSWORD 'secure_random_password';
GRANT ALL PRIVILEGES ON DATABASE apg_billing_prod TO apg_billing_user;

-- Connect to the database
\c apg_billing_prod;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Grant permissions on schema
GRANT ALL ON SCHEMA public TO apg_billing_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO apg_billing_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO apg_billing_user;
```

#### PostgreSQL Configuration (postgresql.conf)
```ini
# Connection settings
max_connections = 200
shared_buffers = 4GB
effective_cache_size = 12GB
work_mem = 256MB
maintenance_work_mem = 1GB

# Write ahead log
wal_level = replica
max_wal_size = 4GB
min_wal_size = 1GB
checkpoint_completion_target = 0.9

# Replication settings
max_replication_slots = 10
max_wal_senders = 10

# Logging
log_destination = 'stderr'
logging_collector = on
log_directory = 'log'
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
log_rotation_age = 1d
log_min_duration_statement = 1000
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '
```

#### Read Replica Setup
```sql
-- On master, create replication user
CREATE USER replicator REPLICATION LOGIN CONNECTION LIMIT 5 ENCRYPTED PASSWORD 'replica_password';

-- Configure pg_hba.conf for replication
host replication replicator replica_ip/32 md5
```

### Database Migration Strategy

#### Production Migration Process
```bash
# 1. Backup current database
pg_dump -h production-db -U apg_billing_user -d apg_billing_prod > backup_$(date +%Y%m%d_%H%M%S).sql

# 2. Test migrations on staging
flask db upgrade --directory migrations_staging

# 3. Run migrations on production during maintenance window
flask db upgrade --directory migrations_production

# 4. Verify migration success
python -c "
from service import get_billing_service
service = get_billing_service()
print('Migration successful:', service.get_service_status()['services']['database'])
"
```

## Load Balancer Configuration

### NGINX Configuration
```nginx
upstream apg_billing {
    least_conn;
    server apg-billing-1:5000 max_fails=3 fail_timeout=30s;
    server apg-billing-2:5000 max_fails=3 fail_timeout=30s;
    server apg-billing-3:5000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name billing.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name billing.yourdomain.com;

    # SSL Configuration
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/private.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;

    # Proxy configuration
    location / {
        proxy_pass http://apg_billing;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # Health check
        proxy_next_upstream error timeout invalid_header http_500 http_502 http_503;
    }

    # Health check endpoint
    location /billing/health {
        proxy_pass http://apg_billing;
        access_log off;
    }

    # Static files (if any)
    location /static/ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

### AWS Application Load Balancer
```json
{
  "LoadBalancerName": "apg-billing-alb",
  "Scheme": "internet-facing",
  "Type": "application",
  "IpAddressType": "ipv4",
  "SecurityGroups": ["sg-alb"],
  "Subnets": ["subnet-12345", "subnet-67890"],
  "Tags": [
    {
      "Key": "Name",
      "Value": "APG Billing ALB"
    }
  ]
}
```

## Monitoring and Logging Setup

### Centralized Logging with ELK Stack

#### Logstash Configuration
```ruby
input {
  beats {
    port => 5044
  }
}

filter {
  if [fields][service] == "apg-billing" {
    json {
      source => "message"
    }
    
    date {
      match => [ "timestamp", "ISO8601" ]
    }
    
    if [level] == "ERROR" or [level] == "CRITICAL" {
      mutate {
        add_tag => [ "alert" ]
      }
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "apg-billing-%{+YYYY.MM.dd}"
  }
}
```

#### Filebeat Configuration
```yaml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /app/logs/*.log
  fields:
    service: apg-billing
  fields_under_root: true

output.logstash:
  hosts: ["logstash:5044"]

logging.level: info
logging.to_files: true
logging.files:
  path: /var/log/filebeat
  name: filebeat
  keepfiles: 7
  permissions: 0644
```

### Application Metrics

#### Prometheus Metrics Endpoint
```python
from prometheus_client import make_wsgi_app, Gauge, Counter, Histogram
from werkzeug.middleware.dispatcher import DispatcherMiddleware

# Create metrics
active_customers = Gauge('apg_billing_active_customers', 'Number of active customers')
payment_counter = Counter('apg_billing_payments_total', 'Total payments processed', ['status'])
response_time = Histogram('apg_billing_request_duration_seconds', 'Request duration')

# Add metrics endpoint to Flask app
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})
```

## Security Hardening

### Application Security

#### Environment Variable Security
```bash
# Use a secrets management system
export DATABASE_URL=$(aws secretsmanager get-secret-value --secret-id apg-billing/database --query SecretString --output text)
export STRIPE_SECRET_KEY=$(aws secretsmanager get-secret-value --secret-id apg-billing/stripe --query SecretString --output text)
```

#### API Security
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Rate limiting
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["1000 per hour", "100 per minute"]
)

# API key authentication
@app.before_request
def authenticate_request():
    if request.endpoint and request.endpoint.startswith('api.'):
        api_key = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not validate_api_key(api_key):
            abort(401)
```

### Infrastructure Security

#### Security Groups (AWS)
```json
{
  "GroupName": "apg-billing-app",
  "Description": "Security group for APG Billing application",
  "VpcId": "vpc-12345678",
  "SecurityGroupRules": [
    {
      "IpPermissions": [
        {
          "IpProtocol": "tcp",
          "FromPort": 5000,
          "ToPort": 5000,
          "UserIdGroupPairs": [
            {
              "GroupId": "sg-alb",
              "Description": "Allow ALB access"
            }
          ]
        },
        {
          "IpProtocol": "tcp",
          "FromPort": 22,
          "ToPort": 22,
          "IpRanges": [
            {
              "CidrIp": "10.0.0.0/8",
              "Description": "SSH from VPC"
            }
          ]
        }
      ]
    }
  ]
}
```

## Backup and Disaster Recovery

### Database Backup Strategy

#### Automated Backups
```bash
#!/bin/bash
# backup_database.sh

BACKUP_DIR="/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="apg_billing_backup_${TIMESTAMP}.sql"

# Create backup
pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME > "$BACKUP_DIR/$BACKUP_FILE"

# Compress backup
gzip "$BACKUP_DIR/$BACKUP_FILE"

# Upload to S3
aws s3 cp "$BACKUP_DIR/${BACKUP_FILE}.gz" s3://apg-billing-backups/

# Clean up old local backups (keep last 7 days)
find $BACKUP_DIR -name "*.gz" -mtime +7 -delete

# Clean up old S3 backups (keep last 30 days)
aws s3 ls s3://apg-billing-backups/ | while read -r line; do
    create_date=$(echo $line | awk '{print $1" "$2}')
    create_date=$(date -d "$create_date" +%s)
    older_than=$(date -d "30 days ago" +%s)
    if [[ $create_date -lt $older_than ]]; then
        file_name=$(echo $line | awk '{print $4}')
        if [[ $file_name != "" ]]; then
            aws s3 rm s3://apg-billing-backups/$file_name
        fi
    fi
done
```

#### Point-in-Time Recovery Setup
```sql
-- Enable continuous archiving
archive_mode = on
archive_command = 'aws s3 cp %p s3://apg-billing-wal-archive/%f'
archive_timeout = 60

-- Configure for PITR
max_wal_senders = 3
wal_keep_segments = 32
```

### Application Backup
```bash
#!/bin/bash
# backup_application.sh

# Backup configuration
tar -czf "/backups/apg_billing_config_$(date +%Y%m%d).tar.gz" \
    /app/.env \
    /app/config/ \
    /app/ssl/

# Backup logs
tar -czf "/backups/apg_billing_logs_$(date +%Y%m%d).tar.gz" \
    /app/logs/

# Upload to S3
aws s3 sync /backups/ s3://apg-billing-backups/application/
```

## Deployment Automation

### CI/CD Pipeline (GitHub Actions)
```yaml
name: Deploy APG Billing

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest
    - name: Run tests
      run: pytest tests/ -v

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v3
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v1
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-west-2

    - name: Build and push Docker image
      run: |
        aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ECR_REGISTRY
        docker build -t apg-billing:$GITHUB_SHA .
        docker tag apg-billing:$GITHUB_SHA $ECR_REGISTRY/apg-billing:$GITHUB_SHA
        docker tag apg-billing:$GITHUB_SHA $ECR_REGISTRY/apg-billing:latest
        docker push $ECR_REGISTRY/apg-billing:$GITHUB_SHA
        docker push $ECR_REGISTRY/apg-billing:latest

    - name: Deploy to ECS
      run: |
        aws ecs update-service --cluster apg-billing-cluster --service apg-billing-service --force-new-deployment
```

### Blue-Green Deployment Script
```bash
#!/bin/bash
# blue_green_deploy.sh

set -e

# Configuration
CLUSTER_NAME="apg-billing-cluster"
SERVICE_NAME="apg-billing-service"
NEW_IMAGE="$1"

if [ -z "$NEW_IMAGE" ]; then
    echo "Usage: $0 <new_image_uri>"
    exit 1
fi

echo "Starting blue-green deployment..."

# Get current task definition
CURRENT_TASK_DEF=$(aws ecs describe-services --cluster $CLUSTER_NAME --services $SERVICE_NAME --query 'services[0].taskDefinition' --output text)

# Create new task definition with new image
NEW_TASK_DEF=$(aws ecs describe-task-definition --task-definition $CURRENT_TASK_DEF --query 'taskDefinition')
NEW_TASK_DEF=$(echo $NEW_TASK_DEF | jq --arg IMAGE "$NEW_IMAGE" '.containerDefinitions[0].image = $IMAGE | del(.taskDefinitionArn) | del(.revision) | del(.status) | del(.requiresAttributes) | del(.placementConstraints) | del(.compatibilities) | del(.registeredAt) | del(.registeredBy)')

# Register new task definition
NEW_TASK_DEF_ARN=$(aws ecs register-task-definition --cli-input-json "$NEW_TASK_DEF" --query 'taskDefinition.taskDefinitionArn' --output text)

echo "Created new task definition: $NEW_TASK_DEF_ARN"

# Update service with new task definition
aws ecs update-service --cluster $CLUSTER_NAME --service $SERVICE_NAME --task-definition $NEW_TASK_DEF_ARN

echo "Waiting for deployment to complete..."

# Wait for deployment to complete
aws ecs wait services-stable --cluster $CLUSTER_NAME --services $SERVICE_NAME

echo "Deployment completed successfully!"

# Health check
HEALTH_CHECK_URL="https://billing.yourdomain.com/billing/health"
for i in {1..30}; do
    if curl -f $HEALTH_CHECK_URL > /dev/null 2>&1; then
        echo "Health check passed!"
        break
    fi
    echo "Health check attempt $i failed, retrying in 10 seconds..."
    sleep 10
    
    if [ $i -eq 30 ]; then
        echo "Health check failed after 30 attempts, rolling back..."
        aws ecs update-service --cluster $CLUSTER_NAME --service $SERVICE_NAME --task-definition $CURRENT_TASK_DEF
        exit 1
    fi
done

echo "Blue-green deployment completed successfully!"
```

## Post-Deployment Verification

### Health Check Script
```bash
#!/bin/bash
# verify_deployment.sh

BASE_URL="https://billing.yourdomain.com"

echo "Verifying APG Billing deployment..."

# Health check
echo "Checking health endpoint..."
curl -f "$BASE_URL/billing/health" || exit 1

# API endpoints check
echo "Checking API endpoints..."
curl -f "$BASE_URL/api/v1/billing/plans" -H "Authorization: Bearer $API_KEY" || exit 1

# Database connectivity
echo "Checking database connectivity..."
python -c "
from service import get_billing_service
service = get_billing_service()
status = service.get_service_status()
assert status['services']['database'] == 'connected', 'Database not connected'
print('Database: OK')
"

# Background tasks
echo "Checking background tasks..."
python -c "
from service import get_billing_service
service = get_billing_service()
status = service.get_service_status()
for task, status in status['background_tasks'].items():
    assert status == 'running', f'Task {task} not running'
    print(f'{task}: OK')
"

echo "All checks passed! Deployment verified."
```

---

© 2025 Datacraft. All rights reserved.