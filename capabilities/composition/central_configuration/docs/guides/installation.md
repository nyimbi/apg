# Installation Guide

## Prerequisites

### System Requirements

- **Python**: 3.11 or higher
- **Memory**: Minimum 4GB RAM, recommended 8GB+
- **Storage**: 10GB+ available disk space
- **Network**: Outbound internet access for package installation

### External Dependencies

#### Required Services
- **PostgreSQL**: 13+ (primary database)
- **Redis**: 6+ (caching and pub/sub)

#### Optional Services (for advanced features)
- **Apache Kafka**: 2.8+ (real-time synchronization)
- **MQTT Broker**: Eclipse Mosquitto or equivalent
- **Kubernetes**: 1.20+ (for container deployment)

## Installation Methods

### Method 1: Python Package Installation

#### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 2. Install Dependencies

```bash
# Core dependencies
pip install -r requirements-prod.txt

# Development dependencies (optional)
pip install -r requirements-dev.txt
```

#### 3. Install APG Central Configuration

```bash
pip install -e .
```

### Method 2: Docker Installation

#### 1. Pull Docker Images

```bash
docker pull datacraft/apg-central-config:latest
docker pull postgres:15
docker pull redis:7
```

#### 2. Run with Docker Compose

```bash
# Copy example configuration
cp .env.example .env

# Edit configuration
nano .env

# Start services
docker-compose up -d
```

### Method 3: Kubernetes Deployment

#### 1. Create Namespace

```bash
kubectl create namespace apg-central-config
```

#### 2. Apply Kubernetes Manifests

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
```

## Configuration

### Environment Variables

Create a `.env` file with the following configuration:

```bash
# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/central_config
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_MAX_CONNECTIONS=100

# Security Configuration
ENCRYPTION_ENABLED=true
JWT_SECRET_KEY=your-secret-key-here
QUANTUM_CRYPTO_ENABLED=true

# Real-time Synchronization
REALTIME_SYNC_ENABLED=true
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
MQTT_BROKER_HOST=localhost
MQTT_BROKER_PORT=1883

# Enterprise Integrations
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
ZENDESK_SUBDOMAIN=yourcompany
ZENDESK_EMAIL=admin@yourcompany.com
ZENDESK_API_TOKEN=your-zendesk-token

# Monitoring and Logging
LOG_LEVEL=INFO
METRICS_ENABLED=true
PROMETHEUS_PORT=9090

# Performance Tuning
MAX_WORKERS=4
CACHE_TTL=3600
BATCH_SIZE=100
```

### Database Setup

#### 1. Create Database

```sql
CREATE DATABASE central_config;
CREATE USER config_user WITH ENCRYPTED PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE central_config TO config_user;
```

#### 2. Run Migrations

```bash
# Initialize database schema
python -m capabilities.composition.central_configuration.service init-db

# Run migrations
alembic upgrade head
```

### Security Setup

#### 1. Generate Encryption Keys

```bash
# Generate quantum-resistant keys
python -m capabilities.composition.central_configuration.security_engine generate-keys

# Output will show key locations:
# Kyber768 keys: /etc/apg/keys/kyber768_*
# Dilithium3 keys: /etc/apg/keys/dilithium3_*
```

#### 2. Configure OAuth2 Providers

```bash
# Configure OAuth2 providers
python -m capabilities.composition.central_configuration.security_engine configure-oauth2 \
  --provider google \
  --client-id your-client-id \
  --client-secret your-client-secret
```

## Verification

### 1. Health Check

```bash
# Check service health
curl http://localhost:8080/health
```

Expected response:
```json
{
  "status": "healthy",
  "components": {
    "database": "healthy",
    "redis": "healthy",
    "encryption": "healthy",
    "realtime_sync": "healthy"
  },
  "version": "1.0.0",
  "timestamp": "2025-01-01T00:00:00Z"
}
```

### 2. Test Configuration API

```bash
# Set a test configuration
curl -X POST http://localhost:8080/api/v1/config \
  -H "Content-Type: application/json" \
  -d '{"key": "test.example", "value": "hello world"}'

# Get the configuration
curl http://localhost:8080/api/v1/config/test.example
```

### 3. Test Real-time Synchronization

```bash
# Connect to WebSocket endpoint
wscat -c ws://localhost:8080/ws/config

# You should see connection confirmation and sync events
```

## Production Deployment

### High Availability Setup

#### 1. Multi-Region Deployment

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  central-config-primary:
    image: datacraft/apg-central-config:latest
    environment:
      - NODE_ROLE=primary
      - REGION=us-east-1
    
  central-config-replica:
    image: datacraft/apg-central-config:latest
    environment:
      - NODE_ROLE=replica
      - REGION=us-west-2
      - PRIMARY_NODE=central-config-primary
```

#### 2. Load Balancer Configuration

```nginx
upstream central_config {
    server central-config-1:8080;
    server central-config-2:8080;
    server central-config-3:8080;
}

server {
    listen 80;
    server_name config.yourcompany.com;
    
    location / {
        proxy_pass http://central_config;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /ws/ {
        proxy_pass http://central_config;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### Security Hardening

#### 1. Enable TLS

```bash
# Generate TLS certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout /etc/ssl/private/central-config.key \
  -out /etc/ssl/certs/central-config.crt

# Update configuration
export TLS_ENABLED=true
export TLS_CERT_PATH=/etc/ssl/certs/central-config.crt
export TLS_KEY_PATH=/etc/ssl/private/central-config.key
```

#### 2. Configure Firewall

```bash
# Allow only necessary ports
ufw allow 8080/tcp  # API port
ufw allow 8443/tcp  # HTTPS port
ufw allow 9090/tcp  # Metrics port
ufw enable
```

### Monitoring Setup

#### 1. Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'central-config'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: /metrics
```

#### 2. Grafana Dashboards

```bash
# Import pre-built dashboard
curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @monitoring/grafana-dashboard.json
```

## Troubleshooting

### Common Issues

#### Connection Errors

```bash
# Check service status
systemctl status apg-central-config

# Check logs
journalctl -u apg-central-config -f

# Test database connection
python -c "
import asyncpg
import asyncio
async def test():
    conn = await asyncpg.connect('postgresql://user:pass@localhost:5432/db')
    print('Database connection successful')
    await conn.close()
asyncio.run(test())
"
```

#### Performance Issues

```bash
# Check resource usage
htop
iostat -x 1

# Monitor database performance
sudo -u postgres psql -d central_config -c "
SELECT query, calls, total_time, mean_time 
FROM pg_stat_statements 
ORDER BY total_time DESC 
LIMIT 10;
"
```

#### Memory Issues

```bash
# Check memory usage
free -h

# Monitor Python memory usage
python -m memory_profiler your_script.py

# Adjust worker settings
export MAX_WORKERS=2
export CACHE_SIZE=1000000  # 1MB cache
```

### Support

For additional help:
- Check the [Troubleshooting Guide](../troubleshooting/common-issues.md)
- Review logs at `/var/log/apg-central-config/`
- Contact support: nyimbi@gmail.com

## Next Steps

After successful installation:
1. Read the [User Guide](user-guide.md)
2. Configure [Enterprise Integrations](../integrations/enterprise-connectors.md)
3. Set up [Security](../security/authentication.md)
4. Enable [Real-time Synchronization](../advanced/realtime-sync.md)