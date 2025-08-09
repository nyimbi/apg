# APG Key Management - Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the APG Key Management capability in various environments, from development to enterprise production deployments.

## Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 2 cores, 2.4 GHz
- RAM: 4 GB
- Storage: 20 GB SSD
- Network: 100 Mbps
- OS: Linux (Ubuntu 20.04+, RHEL 8+, CentOS 8+)

**Recommended Production Requirements:**
- CPU: 8 cores, 3.0 GHz
- RAM: 16 GB
- Storage: 100 GB SSD (with RAID 1)
- Network: 1 Gbps
- OS: Linux (Ubuntu 22.04 LTS, RHEL 9)

### Software Dependencies

- Python 3.11 or higher
- PostgreSQL 14 or higher
- Redis 6.2 or higher
- Docker (optional, for containerized deployment)
- Kubernetes (optional, for orchestrated deployment)

### Hardware Security Module (Optional)

For enhanced security in production:
- Thales Luna HSM 7
- SafeNet ProtectServer
- AWS CloudHSM
- Azure Dedicated HSM

## Quick Start Deployment

### Development Environment

```bash
# Clone the repository
git clone https://github.com/datacraft/apg.git
cd apg/capabilities/common/keym

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export KEYM_DATABASE_URL="postgresql://keym:password@localhost/keym_dev"
export KEYM_CACHE_URL="redis://localhost:6379/0"
export KEYM_ENCRYPTION_KEY="$(openssl rand -hex 32)"

# Initialize database
python -m keym.database.init

# Run development server
python -m keym.app --debug
```

The service will be available at `http://localhost:8080`

### Docker Deployment

```bash
# Build the image
docker build -t datacraft/apg-keym:latest .

# Run with Docker Compose
docker-compose up -d
```

### Docker Compose Configuration

```yaml
version: '3.8'

services:
  keym-app:
    image: datacraft/apg-keym:latest
    ports:
      - "8080:8080"
    environment:
      - KEYM_DATABASE_URL=postgresql://keym:${KEYM_DB_PASSWORD}@keym-db:5432/keym
      - KEYM_CACHE_URL=redis://keym-cache:6379/0
      - KEYM_ENCRYPTION_KEY=${KEYM_ENCRYPTION_KEY}
      - APG_TENANT_ID=${APG_TENANT_ID}
    depends_on:
      - keym-db
      - keym-cache
    networks:
      - keym-network
    volumes:
      - keym-data:/app/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  keym-db:
    image: postgres:15
    environment:
      - POSTGRES_DB=keym
      - POSTGRES_USER=keym
      - POSTGRES_PASSWORD=${KEYM_DB_PASSWORD}
    volumes:
      - keym-db-data:/var/lib/postgresql/data
    networks:
      - keym-network
    restart: unless-stopped

  keym-cache:
    image: redis:7-alpine
    command: redis-server --appendonly yes --requirepass ${KEYM_CACHE_PASSWORD}
    volumes:
      - keym-cache-data:/data
    networks:
      - keym-network
    restart: unless-stopped

volumes:
  keym-data:
  keym-db-data:
  keym-cache-data:

networks:
  keym-network:
    driver: bridge
```

## Production Deployment

### Environment Configuration

Create `.env` file:

```bash
# Database Configuration
KEYM_DATABASE_URL=postgresql://keym_user:SECURE_PASSWORD@db-host:5432/keym_prod
KEYM_DATABASE_POOL_SIZE=50
KEYM_DATABASE_MAX_OVERFLOW=100

# Cache Configuration
KEYM_CACHE_URL=redis://:CACHE_PASSWORD@cache-host:6379/0
KEYM_CACHE_TTL=3600

# Security Configuration
KEYM_ENCRYPTION_KEY=64_character_hex_key_generated_securely
KEYM_JWT_SECRET=jwt_signing_secret_key
KEYM_API_KEY_SALT=api_key_salt_for_hashing

# HSM Configuration (Optional)
KEYM_HSM_ENABLED=true
KEYM_HSM_LIBRARY_PATH=/usr/lib/pkcs11/libCryptoki2_64.so
KEYM_HSM_SLOT_ID=0
KEYM_HSM_PIN=hsm_pin

# Multi-Cloud Configuration
KEYM_AWS_ACCESS_KEY_ID=your_aws_access_key
KEYM_AWS_SECRET_ACCESS_KEY=your_aws_secret_key
KEYM_AZURE_TENANT_ID=your_azure_tenant_id
KEYM_AZURE_CLIENT_ID=your_azure_client_id
KEYM_AZURE_CLIENT_SECRET=your_azure_client_secret

# Monitoring Configuration
KEYM_METRICS_ENABLED=true
KEYM_PROMETHEUS_PORT=9090
KEYM_LOG_LEVEL=INFO

# Performance Configuration
KEYM_MAX_CONCURRENT_OPERATIONS=1000
KEYM_OPERATION_TIMEOUT=30
KEYM_REQUEST_TIMEOUT=60

# APG Integration
APG_TENANT_ID=your_enterprise_tenant_id
APG_REGISTRY_URL=http://apg-registry:8080
APG_EVENT_BUS_URL=http://apg-eventbus:8080
```

### Database Setup

#### PostgreSQL Installation and Configuration

```bash
# Install PostgreSQL (Ubuntu/Debian)
sudo apt update
sudo apt install postgresql postgresql-contrib

# Create database and user
sudo -u postgres createuser --createdb --pwprompt keym_user
sudo -u postgres createdb --owner=keym_user keym_prod

# Configure PostgreSQL for performance
sudo nano /etc/postgresql/15/main/postgresql.conf
```

**PostgreSQL Performance Configuration:**

```conf
# Memory
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB

# Connections
max_connections = 200
max_prepared_transactions = 100

# Write-ahead logging
wal_buffers = 16MB
checkpoint_completion_target = 0.9
wal_writer_delay = 200ms

# Query planner
random_page_cost = 1.1
effective_io_concurrency = 200
```

#### Database Migration

```bash
# Run database migrations
python -m alembic upgrade head

# Initialize seed data
python -m keym.database.seed --environment=production
```

### Application Server Configuration

#### Systemd Service Configuration

Create `/etc/systemd/system/keym.service`:

```ini
[Unit]
Description=APG Key Management Service
After=network.target postgresql.service redis.service
Wants=postgresql.service redis.service

[Service]
Type=simple
User=keym
Group=keym
WorkingDirectory=/opt/keym
Environment=KEYM_CONFIG_FILE=/etc/keym/config.yaml
ExecStart=/opt/keym/venv/bin/python -m keym.app --config /etc/keym/config.yaml
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=keym

# Security settings
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/opt/keym/data /var/log/keym

[Install]
WantedBy=multi-user.target
```

#### Application Configuration

Create `/etc/keym/config.yaml`:

```yaml
# APG Key Management Configuration
app:
  name: "APG Key Management"
  version: "1.0.0"
  environment: "production"
  debug: false
  
database:
  url: "${KEYM_DATABASE_URL}"
  pool_size: 50
  max_overflow: 100
  pool_timeout: 30
  pool_recycle: 3600
  echo: false

cache:
  url: "${KEYM_CACHE_URL}"
  default_timeout: 3600
  key_prefix: "keym:"
  serialization: "pickle"

security:
  encryption_key: "${KEYM_ENCRYPTION_KEY}"
  jwt_secret: "${KEYM_JWT_SECRET}"
  api_key_salt: "${KEYM_API_KEY_SALT}"
  password_hash_algorithm: "bcrypt"
  password_hash_rounds: 12
  
  # Rate limiting
  rate_limits:
    default: "1000/hour"
    crypto_operations: "10000/hour"
    batch_operations: "100/hour"

hsm:
  enabled: true
  library_path: "/usr/lib/pkcs11/libCryptoki2_64.so"
  slot_id: 0
  pin: "${KEYM_HSM_PIN}"
  
  # HSM connection pool
  pool_size: 10
  max_sessions: 50
  session_timeout: 300

multi_cloud:
  enabled: true
  
  aws:
    access_key_id: "${KEYM_AWS_ACCESS_KEY_ID}"
    secret_access_key: "${KEYM_AWS_SECRET_ACCESS_KEY}"
    region: "us-east-1"
    kms_key_spec: "AES_256"
  
  azure:
    tenant_id: "${KEYM_AZURE_TENANT_ID}"
    client_id: "${KEYM_AZURE_CLIENT_ID}"
    client_secret: "${KEYM_AZURE_CLIENT_SECRET}"
    key_vault_url: "https://your-vault.vault.azure.net/"
  
  gcp:
    project_id: "your-gcp-project"
    location: "global"
    key_ring: "keym-keyring"
    credentials_file: "/etc/keym/gcp-credentials.json"

logging:
  version: 1
  disable_existing_loggers: false
  
  formatters:
    default:
      format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    json:
      format: '{"timestamp": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'
  
  handlers:
    console:
      class: logging.StreamHandler
      formatter: default
      stream: ext://sys.stdout
    
    file:
      class: logging.handlers.RotatingFileHandler
      formatter: json
      filename: /var/log/keym/app.log
      maxBytes: 10485760
      backupCount: 10
    
    audit:
      class: logging.handlers.RotatingFileHandler
      formatter: json
      filename: /var/log/keym/audit.log
      maxBytes: 10485760
      backupCount: 50
  
  loggers:
    keym:
      level: INFO
      handlers: [console, file]
    
    keym.audit:
      level: INFO
      handlers: [audit]
      propagate: false

monitoring:
  metrics_enabled: true
  prometheus_port: 9090
  health_check_port: 8081
  
  alerts:
    error_rate_threshold: 0.01
    latency_threshold_ms: 1000
    memory_usage_threshold: 0.8
    disk_usage_threshold: 0.9

performance:
  max_concurrent_operations: 1000
  operation_timeout: 30
  request_timeout: 60
  connection_timeout: 10
  
  # Background tasks
  background_tasks:
    key_rotation_check: "0 2 * * *"  # Daily at 2 AM
    cleanup_expired_keys: "0 3 * * *"  # Daily at 3 AM
    audit_log_rotation: "0 4 * * 0"  # Weekly on Sunday at 4 AM

apg_integration:
  tenant_id: "${APG_TENANT_ID}"
  registry_url: "${APG_REGISTRY_URL}"
  event_bus_url: "${APG_EVENT_BUS_URL}"
  
  # Capability registration
  capability:
    name: "keym"
    version: "1.0.0"
    description: "Enterprise Key Management System"
    endpoints: ["/keym/api/v1"]
    dependencies: ["auth_rbac", "notification"]
```

### Load Balancer Configuration

#### Nginx Configuration

Create `/etc/nginx/sites-available/keym`:

```nginx
upstream keym_backend {
    least_conn;
    server 127.0.0.1:8080 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8081 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8082 max_fails=3 fail_timeout=30s;
    
    # Health check
    keepalive 32;
}

server {
    listen 443 ssl http2;
    server_name keym.your-domain.com;
    
    # SSL Configuration
    ssl_certificate /etc/ssl/certs/keym.your-domain.com.crt;
    ssl_certificate_key /etc/ssl/private/keym.your-domain.com.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=keym_api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=keym_crypto:10m rate=100r/s;
    
    location /keym/api/v1/crypto/ {
        limit_req zone=keym_crypto burst=20 nodelay;
        proxy_pass http://keym_backend;
        include proxy_params;
    }
    
    location /keym/api/v1/ {
        limit_req zone=keym_api burst=5 nodelay;
        proxy_pass http://keym_backend;
        include proxy_params;
    }
    
    location /keym/health {
        access_log off;
        proxy_pass http://keym_backend;
        include proxy_params;
    }
    
    location /keym/metrics {
        allow 10.0.0.0/8;
        allow 192.168.0.0/16;
        deny all;
        proxy_pass http://keym_backend;
        include proxy_params;
    }
    
    # WebSocket support for real-time features
    location /keym/ws {
        proxy_pass http://keym_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name keym.your-domain.com;
    return 301 https://$server_name$request_uri;
}
```

### Kubernetes Deployment

#### Namespace and ConfigMap

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: apg-keym
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: keym-config
  namespace: apg-keym
data:
  config.yaml: |
    app:
      name: "APG Key Management"
      environment: "production"
    database:
      url: "postgresql://keym:$(KEYM_DB_PASSWORD)@keym-postgresql:5432/keym"
      pool_size: 20
    cache:
      url: "redis://:$(KEYM_CACHE_PASSWORD)@keym-redis:6379/0"
    # ... rest of configuration
```

#### Secrets

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: keym-secrets
  namespace: apg-keym
type: Opaque
data:
  database-password: <base64-encoded-password>
  cache-password: <base64-encoded-password>
  encryption-key: <base64-encoded-encryption-key>
  jwt-secret: <base64-encoded-jwt-secret>
  hsm-pin: <base64-encoded-hsm-pin>
```

#### Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: keym-app
  namespace: apg-keym
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  selector:
    matchLabels:
      app: keym-app
  template:
    metadata:
      labels:
        app: keym-app
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: keym-service-account
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: keym
        image: datacraft/apg-keym:1.0.0
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: KEYM_DATABASE_URL
          value: "postgresql://keym:$(KEYM_DB_PASSWORD)@keym-postgresql:5432/keym"
        - name: KEYM_DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: keym-secrets
              key: database-password
        - name: KEYM_CACHE_URL
          value: "redis://:$(KEYM_CACHE_PASSWORD)@keym-redis:6379/0"
        - name: KEYM_CACHE_PASSWORD
          valueFrom:
            secretKeyRef:
              name: keym-secrets
              key: cache-password
        - name: KEYM_ENCRYPTION_KEY
          valueFrom:
            secretKeyRef:
              name: keym-secrets
              key: encryption-key
        volumeMounts:
        - name: config
          mountPath: /etc/keym/config.yaml
          subPath: config.yaml
        - name: data
          mountPath: /app/data
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          successThreshold: 1
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          successThreshold: 1
          failureThreshold: 3
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
      volumes:
      - name: config
        configMap:
          name: keym-config
      - name: data
        persistentVolumeClaim:
          claimName: keym-data-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: keym-service
  namespace: apg-keym
spec:
  selector:
    app: keym-app
  ports:
  - name: http
    port: 8080
    targetPort: 8080
  - name: metrics
    port: 9090
    targetPort: 9090
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: keym-ingress
  namespace: apg-keym
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/use-regex: "true"
    nginx.ingress.kubernetes.io/rate-limit-rpm: "600"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - keym.your-domain.com
    secretName: keym-tls
  rules:
  - host: keym.your-domain.com
    http:
      paths:
      - path: /keym
        pathType: Prefix
        backend:
          service:
            name: keym-service
            port:
              number: 8080
```

## Monitoring and Observability

### Prometheus Configuration

```yaml
global:
  scrape_interval: 15s

scrape_configs:
- job_name: 'keym'
  static_configs:
  - targets: ['localhost:9090']
  metrics_path: /metrics
  scrape_interval: 30s
  
  # Custom labels
  relabel_configs:
  - source_labels: [__address__]
    target_label: instance
    replacement: 'keym-production'
```

### Grafana Dashboard

Import the pre-built Grafana dashboard:

```json
{
  "dashboard": {
    "id": null,
    "title": "APG Key Management",
    "tags": ["apg", "keym", "security"],
    "panels": [
      {
        "title": "Key Operations Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(keym_operations_total[5m])",
            "legendFormat": "{{operation}}"
          }
        ]
      },
      {
        "title": "Active Keys",
        "type": "stat",
        "targets": [
          {
            "expr": "keym_keys_total{status=\"active\"}"
          }
        ]
      },
      {
        "title": "HSM Status",
        "type": "stat",
        "targets": [
          {
            "expr": "keym_hsm_status"
          }
        ]
      }
    ]
  }
}
```

### Log Aggregation

#### ELK Stack Configuration

**Logstash Configuration:**

```ruby
input {
  file {
    path => "/var/log/keym/*.log"
    start_position => "beginning"
    codec => json
  }
}

filter {
  if [name] == "keym.audit" {
    mutate {
      add_tag => ["audit"]
    }
  }
  
  date {
    match => [ "timestamp", "ISO8601" ]
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "keym-%{+YYYY.MM.dd}"
  }
}
```

## Backup and Disaster Recovery

### Database Backup

```bash
#!/bin/bash
# backup_keym_db.sh

BACKUP_DIR="/backup/keym"
DATE=$(date +%Y%m%d_%H%M%S)
DB_NAME="keym_prod"
DB_USER="keym_backup"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Perform backup
pg_dump -h localhost -U "$DB_USER" -d "$DB_NAME" \
        -f "$BACKUP_DIR/keym_backup_$DATE.sql" \
        --verbose --clean --no-owner --no-privileges

# Compress backup
gzip "$BACKUP_DIR/keym_backup_$DATE.sql"

# Keep only last 30 days of backups
find "$BACKUP_DIR" -name "keym_backup_*.sql.gz" -mtime +30 -delete

# Upload to cloud storage (optional)
aws s3 cp "$BACKUP_DIR/keym_backup_$DATE.sql.gz" \
          "s3://your-backup-bucket/keym/db/"
```

### HSM Key Backup

```bash
#!/bin/bash
# backup_hsm_keys.sh

HSM_BACKUP_DIR="/secure/backup/hsm"
DATE=$(date +%Y%m%d_%H%M%S)

# Create secure backup directory
mkdir -p -m 700 "$HSM_BACKUP_DIR"

# Export HSM key metadata (not actual key material)
python -m keym.hsm.backup --export-metadata \
                          --output "$HSM_BACKUP_DIR/hsm_metadata_$DATE.json"

# Encrypt backup
gpg --symmetric --cipher-algo AES256 \
    --output "$HSM_BACKUP_DIR/hsm_metadata_$DATE.json.gpg" \
    "$HSM_BACKUP_DIR/hsm_metadata_$DATE.json"

# Remove unencrypted file
rm "$HSM_BACKUP_DIR/hsm_metadata_$DATE.json"
```

### Disaster Recovery Procedure

1. **Immediate Response**
   ```bash
   # Stop affected services
   sudo systemctl stop keym
   
   # Assess damage
   python -m keym.tools.health_check --full
   ```

2. **Database Recovery**
   ```bash
   # Restore from latest backup
   gunzip keym_backup_YYYYMMDD_HHMMSS.sql.gz
   psql -h localhost -U keym_user -d keym_prod < keym_backup_YYYYMMDD_HHMMSS.sql
   ```

3. **HSM Recovery**
   ```bash
   # Reinitialize HSM connection
   python -m keym.hsm.initialize --reset
   
   # Restore key metadata
   gpg --decrypt hsm_metadata_YYYYMMDD_HHMMSS.json.gpg | \
   python -m keym.hsm.restore --from-metadata
   ```

4. **Service Restart**
   ```bash
   # Restart services
   sudo systemctl start keym
   
   # Verify functionality
   python -m keym.tools.verify_installation
   ```

## Security Hardening

### System Security

```bash
# Create dedicated user
sudo useradd -r -s /bin/false -d /opt/keym keym

# Set proper permissions
sudo chown -R keym:keym /opt/keym
sudo chmod 750 /opt/keym
sudo chmod 640 /etc/keym/config.yaml

# Secure log files
sudo mkdir -p /var/log/keym
sudo chown keym:keym /var/log/keym
sudo chmod 750 /var/log/keym
```

### Network Security

```bash
# Firewall rules (iptables)
# Allow HTTPS traffic
iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# Allow application port (internal only)
iptables -A INPUT -p tcp --dport 8080 -s 10.0.0.0/8 -j ACCEPT

# Allow monitoring (internal only)
iptables -A INPUT -p tcp --dport 9090 -s 10.0.0.0/8 -j ACCEPT

# Default deny
iptables -A INPUT -j DROP
```

### SSL/TLS Configuration

```bash
# Generate strong SSL certificate
openssl req -x509 -nodes -days 365 -newkey rsa:4096 \
        -keyout /etc/ssl/private/keym.key \
        -out /etc/ssl/certs/keym.crt \
        -config /etc/ssl/keym.conf

# Set proper permissions
chmod 400 /etc/ssl/private/keym.key
chmod 644 /etc/ssl/certs/keym.crt
```

## Troubleshooting

### Common Issues

#### Database Connection Issues

```bash
# Check database connectivity
pg_isready -h localhost -p 5432 -U keym_user

# Check connection pool
python -c "
from keym.service import KeyManagementService
service = KeyManagementService()
print(service.check_database_health())
"
```

#### HSM Issues

```bash
# Check HSM connectivity
python -m keym.hsm.test_connection

# Check HSM status
python -c "
from keym.hsm_integration import HSMManager
hsm = HSMManager()
print(hsm.get_all_hsm_status())
"
```

#### Performance Issues

```bash
# Check system resources
htop
iostat -x 1
free -h

# Check application metrics
curl http://localhost:9090/metrics | grep keym_

# Check database performance
sudo -u postgres psql -c "
SELECT query, calls, total_time, mean_time 
FROM pg_stat_statements 
ORDER BY total_time DESC 
LIMIT 10;
"
```

### Log Analysis

```bash
# Check application logs
tail -f /var/log/keym/app.log

# Check audit logs
tail -f /var/log/keym/audit.log

# Search for errors
grep -i error /var/log/keym/*.log

# Analyze performance
grep "duration" /var/log/keym/app.log | awk '{print $NF}' | sort -n
```

## Maintenance

### Regular Maintenance Tasks

```bash
#!/bin/bash
# keym_maintenance.sh

# Daily tasks
if [[ $(date +%H) -eq 2 ]]; then
    # Database maintenance
    sudo -u postgres psql -d keym_prod -c "VACUUM ANALYZE;"
    
    # Log rotation
    /usr/sbin/logrotate /etc/logrotate.d/keym
    
    # Clean temporary files
    find /tmp -name "keym_*" -mtime +1 -delete
fi

# Weekly tasks
if [[ $(date +%u) -eq 7 ]] && [[ $(date +%H) -eq 3 ]]; then
    # Full database vacuum
    sudo -u postgres psql -d keym_prod -c "VACUUM FULL;"
    
    # Update statistics
    sudo -u postgres psql -d keym_prod -c "ANALYZE;"
    
    # Archive old audit logs
    find /var/log/keym -name "audit.log.*" -mtime +90 -delete
fi

# Monthly tasks
if [[ $(date +%d) -eq 1 ]] && [[ $(date +%H) -eq 4 ]]; then
    # Security scan
    python -m keym.tools.security_scan
    
    # Performance analysis
    python -m keym.tools.performance_report
    
    # Certificate expiry check
    python -m keym.tools.cert_check
fi
```

---

This deployment guide provides comprehensive instructions for deploying APG Key Management in production environments. For additional support, refer to the troubleshooting section or contact Datacraft support.

**Contact Information**
- Website: www.datacraft.co.ke
- Email: nyimbi@gmail.com
- Copyright: © 2025 Datacraft