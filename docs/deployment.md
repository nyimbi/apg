# APG Deployment Guide

Complete guide for deploying APG in various environments, from development to production at scale.

## 🚀 Deployment Options

### 1. Development Deployment
- **Target**: Local development and testing
- **Resources**: Minimal (laptop/desktop)
- **Features**: Full functionality, hot reload, debug mode

### 2. Staging Deployment
- **Target**: Pre-production testing
- **Resources**: Medium (1-2 servers)
- **Features**: Production-like environment, testing data

### 3. Production Deployment
- **Target**: Live production environment
- **Resources**: Scalable (multiple servers, load balancing)
- **Features**: High availability, monitoring, backup

### 4. Cloud Deployment
- **Target**: AWS, GCP, Azure, or other cloud providers
- **Resources**: Auto-scaling, managed services
- **Features**: Global distribution, disaster recovery

## 🏗️ Infrastructure Requirements

### Minimum Production Requirements

**Hardware**:
- 4 CPU cores (8+ recommended)
- 16GB RAM (32GB+ recommended)
- 100GB SSD storage (500GB+ recommended)
- 1Gbps network connection

**Software**:
- Ubuntu 20.04+ or CentOS 8+ (Docker alternative available)
- Python 3.11+
- PostgreSQL 14+
- Redis 7+
- nginx 1.20+

### Recommended Production Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Load Balancer                          │
│                    (nginx/HAProxy)                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
    ┌─────────────────┼─────────────────┐
    │                 │                 │
┌───▼───┐        ┌───▼───┐        ┌───▼───┐
│ APG   │        │ APG   │        │ APG   │
│ App 1 │        │ App 2 │        │ App 3 │
└───┬───┘        └───┬───┘        └───┬───┘
    │                │                │
    └─────────────────┼─────────────────┘
                      │
    ┌─────────────────┼─────────────────┐
    │                 │                 │
┌───▼───┐        ┌───▼───┐        ┌───▼───┐
│ Redis │        │PostgreSQL      │ File  │
│Cluster│        │ Primary        │Storage│
└───────┘        └───┬───┘        └───────┘
                     │
                ┌───▼───┐
                │PostgreSQL
                │Replica │
                └───────┘
```

## 🐳 Docker Deployment

### Development with Docker Compose

1. **Clone Repository**:
```bash
git clone <repository-url>
cd apg
```

2. **Development Environment**:
```bash
# Copy environment file
cp .env.example .env.dev

# Start development stack
docker-compose -f docker-compose.dev.yml up -d

# Check status
docker-compose -f docker-compose.dev.yml ps
```

3. **Development Configuration** (`.env.dev`):
```env
# Application
FLASK_ENV=development
DEBUG=True
SECRET_KEY=dev-secret-key-change-in-production

# Database
DATABASE_URL=postgresql://apg_user:apg_password@postgres:5432/apg_dev
REDIS_URL=redis://redis:6379/0

# APG Configuration
APG_DATA_DIR=/app/data
APG_LOGS_DIR=/app/logs

# Optional Features
WEB3_PROVIDER_URL=https://mainnet.infura.io/v3/YOUR_PROJECT_ID
PYTORCH_DEVICE=cpu
```

4. **Development Services**:
```yaml
# docker-compose.dev.yml
version: '3.8'
services:
  postgres:
    image: postgres:14
    environment:
      POSTGRES_DB: apg_dev
      POSTGRES_USER: apg_user
      POSTGRES_PASSWORD: apg_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  apg:
    build:
      context: .
      dockerfile: docker/Dockerfile.dev
    environment:
      - DATABASE_URL=postgresql://apg_user:apg_password@postgres:5432/apg_dev
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - .:/app
      - apg_data:/app/data
    ports:
      - "5000:5000"
    depends_on:
      - postgres
      - redis
    command: python cli.py run --debug --host 0.0.0.0

volumes:
  postgres_data:
  apg_data:
```

### Production Docker Deployment

1. **Production Environment** (`.env.prod`):
```env
# Application
FLASK_ENV=production
DEBUG=False
SECRET_KEY=your-very-secure-secret-key-here

# Database with connection pooling
DATABASE_URL=postgresql://apg_user:strong_password@postgres:5432/apg_prod?pool_size=20&max_overflow=30

# Redis cluster
REDIS_URL=redis://redis-cluster:6379/0

# Security
SECURE_SSL_REDIRECT=True
SESSION_COOKIE_SECURE=True
CSRF_COOKIE_SECURE=True

# Performance
WEB_CONCURRENCY=8
MAX_WORKERS=16
WORKER_TIMEOUT=300

# Monitoring
PROMETHEUS_ENABLED=True
GRAFANA_ENABLED=True

# Optional: External services
S3_BUCKET=your-apg-bucket
CDN_URL=https://cdn.yourdomain.com
```

2. **Production Docker Compose**:
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
      - static_files:/app/static
    depends_on:
      - apg

  apg:
    build:
      context: .
      dockerfile: docker/Dockerfile.prod
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
    volumes:
      - static_files:/app/static
      - upload_files:/app/uploads
    depends_on:
      - postgres
      - redis
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
          cpus: '1.0'

  postgres:
    image: postgres:14
    environment:
      POSTGRES_DB: apg_prod
      POSTGRES_USER: apg_user
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./postgres/postgresql.conf:/etc/postgresql/postgresql.conf
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --maxmemory 512mb
    volumes:
      - redis_data:/data
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '0.5'

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  postgres_data:
  redis_data:
  static_files:
  upload_files:
  prometheus_data:
  grafana_data:
```

## ☁️ Cloud Deployment

### AWS Deployment

#### Using Amazon ECS (Elastic Container Service)

1. **Create ECS Cluster**:
```bash
# Install AWS CLI and configure
aws configure

# Create ECS cluster
aws ecs create-cluster --cluster-name apg-prod

# Create task definition
aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json
```

2. **ECS Task Definition** (`ecs-task-definition.json`):
```json
{
  "family": "apg-prod",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "apg-app",
      "image": "your-account.dkr.ecr.region.amazonaws.com/apg:latest",
      "portMappings": [
        {
          "containerPort": 5000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "DATABASE_URL",
          "value": "postgresql://user:pass@rds-endpoint:5432/apg_prod"
        },
        {
          "name": "REDIS_URL", 
          "value": "redis://elasticache-endpoint:6379/0"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/apg-prod",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

3. **Infrastructure as Code with Terraform**:
```hcl
# main.tf
provider "aws" {
  region = "us-west-2"
}

# VPC and networking
resource "aws_vpc" "apg_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "apg-vpc"
  }
}

# RDS instance
resource "aws_db_instance" "apg_database" {
  identifier     = "apg-prod-db"
  engine         = "postgres"
  engine_version = "14.9"
  instance_class = "db.t3.medium"
  
  allocated_storage     = 100
  max_allocated_storage = 1000
  storage_type          = "gp2"
  storage_encrypted     = true
  
  db_name  = "apg_prod"
  username = "apg_user"
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.apg.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = false
  final_snapshot_identifier = "apg-prod-final-snapshot"
  
  tags = {
    Name = "apg-prod-database"
  }
}

# ElastiCache Redis
resource "aws_elasticache_subnet_group" "apg" {
  name       = "apg-cache-subnet"
  subnet_ids = aws_subnet.private[*].id
}

resource "aws_elasticache_replication_group" "apg" {
  replication_group_id       = "apg-redis"
  description                = "Redis cluster for APG"
  
  node_type            = "cache.t3.micro"
  port                 = 6379
  parameter_group_name = "default.redis7"
  
  num_cache_clusters = 2
  
  subnet_group_name  = aws_elasticache_subnet_group.apg.name
  security_group_ids = [aws_security_group.redis.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  
  tags = {
    Name = "apg-redis"
  }
}

# Application Load Balancer
resource "aws_lb" "apg" {
  name               = "apg-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets           = aws_subnet.public[*].id

  enable_deletion_protection = true

  tags = {
    Name = "apg-alb"
  }
}

# ECS Service
resource "aws_ecs_service" "apg" {
  name            = "apg-service"
  cluster         = aws_ecs_cluster.apg.id
  task_definition = aws_ecs_task_definition.apg.arn
  desired_count   = 3
  launch_type     = "FARGATE"

  network_configuration {
    security_groups  = [aws_security_group.ecs_tasks.id]
    subnets         = aws_subnet.private[*].id
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.apg.arn
    container_name   = "apg-app"
    container_port   = 5000
  }

  depends_on = [aws_lb_listener.apg]
}
```

#### Deployment Script:
```bash
#!/bin/bash
# deploy-aws.sh

set -e

# Build and push Docker image
docker build -t apg:latest -f docker/Dockerfile.prod .
docker tag apg:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/apg:latest

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Push image
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/apg:latest

# Deploy infrastructure
cd terraform/
terraform init
terraform plan -var="db_password=$DB_PASSWORD"
terraform apply -auto-approve -var="db_password=$DB_PASSWORD"

# Update ECS service
aws ecs update-service --cluster apg-prod --service apg-service --force-new-deployment

echo "Deployment completed successfully!"
```

### Google Cloud Platform (GCP) Deployment

1. **Using Google Kubernetes Engine (GKE)**:
```bash
# Create GKE cluster
gcloud container clusters create apg-cluster \
  --num-nodes=3 \
  --machine-type=e2-standard-4 \
  --enable-autoscaling \
  --min-nodes=1 \
  --max-nodes=10 \
  --zone=us-central1-a

# Get credentials
gcloud container clusters get-credentials apg-cluster --zone=us-central1-a
```

2. **Kubernetes Deployment** (`k8s/deployment.yaml`):
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: apg-deployment
  labels:
    app: apg
spec:
  replicas: 3
  selector:
    matchLabels:
      app: apg
  template:
    metadata:
      labels:
        app: apg
    spec:
      containers:
      - name: apg
        image: gcr.io/your-project/apg:latest
        ports:
        - containerPort: 5000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: apg-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: apg-secrets
              key: redis-url
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        readinessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 60
          periodSeconds: 30
---
apiVersion: v1
kind: Service
metadata:
  name: apg-service
spec:
  selector:
    app: apg
  ports:
  - port: 80
    targetPort: 5000
  type: LoadBalancer
```

### Azure Deployment

1. **Using Azure Container Instances**:
```bash
# Create resource group
az group create --name apg-rg --location eastus

# Create container registry
az acr create --resource-group apg-rg --name apgregistry --sku Basic

# Build and push image
az acr build --registry apgregistry --image apg:latest .

# Deploy container
az container create \
  --resource-group apg-rg \
  --name apg-container \
  --image apgregistry.azurecr.io/apg:latest \
  --cpu 2 \
  --memory 4 \
  --registry-login-server apgregistry.azurecr.io \
  --registry-username apgregistry \
  --registry-password <password> \
  --dns-name-label apg-prod \
  --ports 5000 \
  --environment-variables \
    DATABASE_URL='<database-url>' \
    REDIS_URL='<redis-url>'
```

## 🔧 Configuration Management

### Environment-Specific Configuration

**Development** (`.env.development`):
```env
FLASK_ENV=development
DEBUG=True
SECRET_KEY=dev-secret-key
DATABASE_URL=postgresql://localhost:5432/apg_dev
REDIS_URL=redis://localhost:6379/0
APG_LOG_LEVEL=DEBUG
ENABLE_PROFILING=True
```

**Staging** (`.env.staging`):
```env
FLASK_ENV=staging
DEBUG=False
SECRET_KEY=${STAGING_SECRET_KEY}
DATABASE_URL=${STAGING_DATABASE_URL}
REDIS_URL=${STAGING_REDIS_URL}
APG_LOG_LEVEL=INFO
ENABLE_PROFILING=False
SSL_REQUIRED=True
```

**Production** (`.env.production`):
```env
FLASK_ENV=production
DEBUG=False
SECRET_KEY=${PROD_SECRET_KEY}
DATABASE_URL=${PROD_DATABASE_URL}
REDIS_URL=${PROD_REDIS_URL}
APG_LOG_LEVEL=WARNING
ENABLE_PROFILING=False
SSL_REQUIRED=True
SESSION_COOKIE_SECURE=True
CSRF_COOKIE_SECURE=True
```

### Configuration Validation

```python
# config_validator.py
def validate_production_config():
    """Validate production configuration"""
    required_vars = [
        'SECRET_KEY',
        'DATABASE_URL', 
        'REDIS_URL'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ConfigurationError(f"Missing required environment variables: {missing_vars}")
    
    # Validate database connection
    try:
        engine = create_engine(os.getenv('DATABASE_URL'))
        engine.connect()
    except Exception as e:
        raise ConfigurationError(f"Database connection failed: {e}")
    
    # Validate Redis connection
    try:
        redis_client = redis.from_url(os.getenv('REDIS_URL'))
        redis_client.ping()
    except Exception as e:
        raise ConfigurationError(f"Redis connection failed: {e}")
    
    print("✅ Configuration validation passed")
```

## 📊 Monitoring & Logging

### Application Monitoring

1. **Prometheus Configuration** (`prometheus/prometheus.yml`):
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'apg-app'
    static_configs:
      - targets: ['apg:5000']
    metrics_path: /metrics
    scrape_interval: 30s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

2. **Grafana Dashboard Configuration**:
```json
{
  "dashboard": {
    "title": "APG System Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(flask_http_request_total[5m])",
            "legendFormat": "{{method}} {{status}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph", 
        "targets": [
          {
            "expr": "flask_http_request_duration_seconds",
            "legendFormat": "{{quantile}}"
          }
        ]
      },
      {
        "title": "Active Workflows",
        "type": "stat",
        "targets": [
          {
            "expr": "apg_active_workflows_total"
          }
        ]
      }
    ]
  }
}
```

### Logging Configuration

```python
# logging_config.py
import structlog
import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": structlog.stdlib.ProcessorFormatter,
            "processor": structlog.dev.ConsoleRenderer(colors=False),
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "json",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "/app/logs/apg.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "json",
        },
    },
    "loggers": {
        "apg": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": False,
        },
    },
    "root": {
        "level": "WARNING",
        "handlers": ["console"],
    },
}

def configure_logging():
    logging.config.dictConfig(LOGGING_CONFIG)
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
```

## 🔐 Security Hardening

### SSL/TLS Configuration

1. **nginx SSL Configuration**:
```nginx
server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /etc/nginx/ssl/yourdomain.com.crt;
    ssl_certificate_key /etc/nginx/ssl/yourdomain.com.key;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;

    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";

    location / {
        proxy_pass http://apg-backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

2. **Firewall Configuration**:
```bash
# Ubuntu UFW
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable

# Application-specific rules
sudo ufw allow from 10.0.0.0/8 to any port 5432  # PostgreSQL
sudo ufw allow from 10.0.0.0/8 to any port 6379  # Redis
```

### Database Security

```sql
-- PostgreSQL security configuration
-- Create dedicated user with limited privileges
CREATE USER apg_app WITH PASSWORD 'strong_password';

-- Grant only necessary privileges
GRANT CONNECT ON DATABASE apg_prod TO apg_app;
GRANT USAGE ON SCHEMA public TO apg_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO apg_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO apg_app;

-- Enable SSL
ALTER SYSTEM SET ssl = on;
ALTER SYSTEM SET ssl_cert_file = 'server.crt';
ALTER SYSTEM SET ssl_key_file = 'server.key';

-- Configure pg_hba.conf for SSL-only connections
-- hostssl apg_prod apg_app 0.0.0.0/0 md5
```

## 🚨 Disaster Recovery

### Backup Strategy

1. **Database Backups**:
```bash
#!/bin/bash
# backup-database.sh

BACKUP_DIR="/backups/postgresql"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="apg_prod_${DATE}.sql"

# Create backup
pg_dump -h $DB_HOST -U $DB_USER -d apg_prod -f ${BACKUP_DIR}/${BACKUP_FILE}

# Compress backup
gzip ${BACKUP_DIR}/${BACKUP_FILE}

# Upload to S3 (optional)
aws s3 cp ${BACKUP_DIR}/${BACKUP_FILE}.gz s3://your-backup-bucket/postgresql/

# Cleanup old backups (keep last 30 days)
find ${BACKUP_DIR} -name "*.sql.gz" -mtime +30 -delete

echo "Database backup completed: ${BACKUP_FILE}.gz"
```

2. **Redis Backups**:
```bash
#!/bin/bash
# backup-redis.sh

BACKUP_DIR="/backups/redis"
DATE=$(date +%Y%m%d_%H%M%S)

# Create Redis backup
redis-cli --rdb ${BACKUP_DIR}/redis_${DATE}.rdb

# Compress and upload
gzip ${BACKUP_DIR}/redis_${DATE}.rdb
aws s3 cp ${BACKUP_DIR}/redis_${DATE}.rdb.gz s3://your-backup-bucket/redis/

echo "Redis backup completed: redis_${DATE}.rdb.gz"
```

3. **Automated Backup Cron**:
```bash
# Add to crontab
0 2 * * * /opt/apg/scripts/backup-database.sh
30 2 * * * /opt/apg/scripts/backup-redis.sh
0 3 * * 0 /opt/apg/scripts/backup-files.sh  # Weekly file backup
```

### Recovery Procedures

1. **Database Recovery**:
```bash
#!/bin/bash
# restore-database.sh

BACKUP_FILE=$1
if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

# Stop application
docker-compose stop apg

# Download backup if from S3
aws s3 cp s3://your-backup-bucket/postgresql/${BACKUP_FILE} /tmp/

# Decompress
gunzip /tmp/${BACKUP_FILE}

# Restore database
psql -h $DB_HOST -U $DB_USER -d apg_prod < /tmp/${BACKUP_FILE%.gz}

# Start application
docker-compose start apg

echo "Database restored from: $BACKUP_FILE"
```

2. **Complete System Recovery**:
```bash
#!/bin/bash
# disaster-recovery.sh

echo "Starting disaster recovery process..."

# 1. Provision new infrastructure
terraform apply -auto-approve

# 2. Restore database
./restore-database.sh apg_prod_20250115_020000.sql.gz

# 3. Restore Redis
./restore-redis.sh redis_20250115_023000.rdb.gz

# 4. Deploy application
docker-compose -f docker-compose.prod.yml up -d

# 5. Verify system health
./health-check.sh

echo "Disaster recovery completed!"
```

## 📋 Deployment Checklist

### Pre-Deployment
- [ ] Environment variables configured
- [ ] Database migrations ready
- [ ] SSL certificates obtained
- [ ] Backup systems configured
- [ ] Monitoring setup complete
- [ ] Security hardening applied
- [ ] Load testing performed
- [ ] Documentation updated

### During Deployment
- [ ] Application deployed
- [ ] Database migrations executed
- [ ] Health checks passing
- [ ] Monitoring alerts configured
- [ ] SSL certificates installed
- [ ] DNS records updated
- [ ] Load balancer configured

### Post-Deployment
- [ ] System health verified
- [ ] Performance benchmarks met
- [ ] Security scan completed
- [ ] Backup verification
- [ ] Rollback plan tested
- [ ] Team notified
- [ ] Documentation updated
- [ ] Monitoring dashboard configured

## 🔧 Troubleshooting

### Common Issues

1. **Database Connection Issues**:
```bash
# Check database connectivity
pg_isready -h $DB_HOST -p 5432 -U $DB_USER

# Check connection limits
psql -h $DB_HOST -U $DB_USER -c "SELECT count(*) FROM pg_stat_activity;"
```

2. **Redis Connection Issues**:
```bash
# Test Redis connectivity
redis-cli -h $REDIS_HOST -p 6379 ping

# Check Redis memory usage
redis-cli -h $REDIS_HOST -p 6379 info memory
```

3. **Application Performance Issues**:
```bash
# Check application logs
docker-compose logs -f apg

# Monitor resource usage
htop
iotop
```

### Emergency Procedures

1. **Rollback Deployment**:
```bash
# Quick rollback using Docker
docker-compose -f docker-compose.prod.yml down
docker tag apg:previous apg:latest
docker-compose -f docker-compose.prod.yml up -d
```

2. **Scale Application**:
```bash
# Scale up application instances
docker-compose -f docker-compose.prod.yml up -d --scale apg=5

# For Kubernetes
kubectl scale deployment apg-deployment --replicas=5
```

---

*Next: [Monitoring & Operations Guide](./monitoring.md) →*