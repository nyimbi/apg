# APG Payment Gateway - Production Deployment Guide

A comprehensive guide for deploying the APG Payment Gateway to production environments with high availability, security, and scalability.

## Prerequisites

### System Requirements
- **OS**: Ubuntu 20.04 LTS or later / CentOS 8 / RHEL 8
- **CPU**: Minimum 4 cores (8+ recommended for production)
- **RAM**: Minimum 8GB (16GB+ recommended for production)
- **Storage**: 100GB+ SSD storage
- **Network**: Stable internet connection with static IP

### Required Software
- Docker 20.10+
- Docker Compose 2.0+
- Git
- SSL certificates (Let's Encrypt or commercial)
- Domain name with DNS access

## Quick Start Deployment

### 1. Clone Repository
```bash
git clone https://github.com/your-org/apg-payment-gateway.git
cd apg-payment-gateway/capabilities/common/payment_gateway
```

### 2. Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit environment variables
nano .env
```

### 3. SSL Certificate Setup
```bash
# Create SSL directory
mkdir -p nginx/ssl

# Option 1: Let's Encrypt (recommended)
sudo apt install certbot
sudo certbot certonly --standalone -d your-domain.com

# Copy certificates
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem nginx/ssl/
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem nginx/ssl/

# Option 2: Self-signed (development only)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/privkey.pem \
  -out nginx/ssl/fullchain.pem
```

### 4. Deploy Services
```bash
# Build and start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f payment-gateway
```

## Detailed Configuration

### Environment Variables

#### Critical Security Settings
```bash
# Strong encryption keys
JWT_SECRET=$(openssl rand -base64 32)
ENCRYPTION_KEY=$(openssl rand -base64 32)
POSTGRES_PASSWORD=$(openssl rand -base64 16)
GRAFANA_PASSWORD=$(openssl rand -base64 16)

# Production environment
APP_ENV=production
DEBUG=false
LOG_LEVEL=info
```

#### Payment Provider Configuration
```bash
# Stripe Production
STRIPE_SECRET_KEY=sk_live_your_live_key
STRIPE_PUBLISHABLE_KEY=pk_live_your_live_key
STRIPE_WEBHOOK_SECRET=whsec_your_webhook_secret

# Adyen Production
ADYEN_API_KEY=your_live_api_key
ADYEN_MERCHANT_ACCOUNT=your_merchant_account
ADYEN_ENVIRONMENT=live

# Flutterwave Production
FLUTTERWAVE_SECRET_KEY=FLWSECK-your_live_key
FLUTTERWAVE_ENVIRONMENT=live

# M-Pesa Production
MPESA_ENVIRONMENT=live
MPESA_BUSINESS_SHORTCODE=your_business_shortcode

# Update callback URLs to your domain
MPESA_CALLBACK_URL=https://your-domain.com/webhooks/mpesa
PESAPAL_IPN_URL=https://your-domain.com/webhooks/pesapal
DPO_CALLBACK_URL=https://your-domain.com/webhooks/dpo
```

### Nginx Configuration

Create `nginx/nginx.conf`:
```nginx
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 1024;
    use epoll;
    multi_accept on;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    # Logging
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';
    access_log /var/log/nginx/access.log main;

    # Performance
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 10240;
    gzip_proxied expired no-cache no-store private must-revalidate;
    gzip_types text/plain text/css text/xml text/javascript 
               application/x-javascript application/xml+rss application/json;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=webhooks:10m rate=100r/s;

    # Upstream backend
    upstream payment_gateway {
        server payment-gateway:8080;
        keepalive 32;
    }

    # HTTP to HTTPS redirect
    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$server_name$request_uri;
    }

    # HTTPS server
    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        # SSL configuration
        ssl_certificate /etc/nginx/ssl/fullchain.pem;
        ssl_certificate_key /etc/nginx/ssl/privkey.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
        ssl_prefer_server_ciphers off;
        ssl_session_cache shared:SSL:10m;
        ssl_session_timeout 10m;

        # API endpoints
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://payment_gateway;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
        }

        # Webhook endpoints
        location /webhooks/ {
            limit_req zone=webhooks burst=50 nodelay;
            proxy_pass http://payment_gateway;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Health check
        location /health {
            proxy_pass http://payment_gateway;
            access_log off;
        }

        # Static files (if any)
        location /static/ {
            alias /app/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }
}
```

### Database Initialization

Create `sql/init.sql`:
```sql
-- APG Payment Gateway Database Schema
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create application user
CREATE USER payment_app WITH ENCRYPTED PASSWORD 'secure_app_password';

-- Create database
CREATE DATABASE payment_gateway_prod OWNER payment_user;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE payment_gateway_prod TO payment_user;
GRANT CONNECT ON DATABASE payment_gateway_prod TO payment_app;

-- Switch to the application database
\c payment_gateway_prod;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS transactions;
CREATE SCHEMA IF NOT EXISTS providers;
CREATE SCHEMA IF NOT EXISTS webhooks;
CREATE SCHEMA IF NOT EXISTS audit;

-- Grant schema permissions
GRANT USAGE ON SCHEMA transactions TO payment_app;
GRANT USAGE ON SCHEMA providers TO payment_app;
GRANT USAGE ON SCHEMA webhooks TO payment_app;
GRANT USAGE ON SCHEMA audit TO payment_app;

-- Create tables will be handled by application migrations
```

## High Availability Setup

### Load Balancer Configuration

For multiple instances, update `docker-compose.yml`:
```yaml
services:
  payment-gateway-1:
    build: .
    container_name: apg-payment-gateway-1
    # ... same configuration
    
  payment-gateway-2:
    build: .
    container_name: apg-payment-gateway-2
    # ... same configuration

  nginx:
    # Update upstream in nginx.conf:
    # upstream payment_gateway {
    #     server payment-gateway-1:8080;
    #     server payment-gateway-2:8080;
    # }
```

### Database Clustering

For PostgreSQL high availability:
```yaml
services:
  postgres-primary:
    image: postgres:15
    environment:
      - POSTGRES_REPLICATION_MODE=master
      - POSTGRES_REPLICATION_USER=replicator
      - POSTGRES_REPLICATION_PASSWORD=replication_pass

  postgres-replica:
    image: postgres:15
    environment:
      - POSTGRES_REPLICATION_MODE=slave
      - POSTGRES_MASTER_HOST=postgres-primary
      - POSTGRES_REPLICATION_USER=replicator
      - POSTGRES_REPLICATION_PASSWORD=replication_pass
```

### Redis Clustering

```yaml
services:
  redis-master:
    image: redis:7-alpine
    command: redis-server --appendonly yes

  redis-replica:
    image: redis:7-alpine
    command: redis-server --appendonly yes --slaveof redis-master 6379
```

## Monitoring Setup

### Prometheus Configuration

Create `monitoring/prometheus.yml`:
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'payment-gateway'
    static_configs:
      - targets: ['payment-gateway:8080']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx:80']
```

### Grafana Dashboards

Create monitoring dashboards in `monitoring/grafana/dashboards/`:

1. **Payment Gateway Overview**
2. **Transaction Metrics**
3. **Provider Performance**
4. **System Resources**
5. **Error Rates & Alerts**

### Alerting Rules

Create `monitoring/prometheus/alerts.yml`:
```yaml
groups:
  - name: payment_gateway_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: High error rate detected

      - alert: DatabaseConnectionFailure
        expr: up{job="postgres"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: Database connection failure
```

## Security Hardening

### Firewall Configuration
```bash
# Ubuntu/Debian
sudo ufw allow 22/tcp      # SSH
sudo ufw allow 80/tcp      # HTTP
sudo ufw allow 443/tcp     # HTTPS
sudo ufw allow 3000/tcp    # Grafana (restrict to admin IPs)
sudo ufw allow 9090/tcp    # Prometheus (restrict to admin IPs)
sudo ufw enable

# CentOS/RHEL
sudo firewall-cmd --permanent --add-service=ssh
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --reload
```

### Container Security
```bash
# Scan images for vulnerabilities
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image apg-payment-gateway:latest

# Run containers with security options
docker-compose down
docker-compose up -d --security-opt no-new-privileges:true
```

### Secrets Management
```bash
# Use Docker secrets for sensitive data
echo "your_secret_key" | docker secret create jwt_secret -
echo "your_db_password" | docker secret create db_password -

# Update docker-compose.yml to use secrets
secrets:
  jwt_secret:
    external: true
  db_password:
    external: true
```

## Backup & Recovery

### Database Backup
```bash
# Create backup script
cat > backup_database.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/backups/postgres"
DATE=$(date +"%Y%m%d_%H%M%S")
mkdir -p $BACKUP_DIR

docker exec apg-postgres pg_dump -U payment_user payment_gateway > \
  $BACKUP_DIR/payment_gateway_$DATE.sql

# Compress and encrypt
gzip $BACKUP_DIR/payment_gateway_$DATE.sql
gpg --symmetric --cipher-algo AES256 $BACKUP_DIR/payment_gateway_$DATE.sql.gz

# Clean old backups (keep 30 days)
find $BACKUP_DIR -name "*.sql.gz.gpg" -mtime +30 -delete
EOF

chmod +x backup_database.sh

# Add to crontab
echo "0 2 * * * /path/to/backup_database.sh" | crontab -
```

### Volume Backup
```bash
# Backup Docker volumes
docker run --rm -v apg_postgres_data:/data -v $(pwd)/backups:/backup \
  alpine tar czf /backup/postgres_data_$(date +%Y%m%d).tar.gz -C /data .
```

## Performance Optimization

### Application Tuning
```yaml
# Update docker-compose.yml for production
services:
  payment-gateway:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
    environment:
      - GUNICORN_WORKERS=8
      - GUNICORN_MAX_REQUESTS=1000
      - GUNICORN_TIMEOUT=30
```

### Database Tuning
```sql
-- PostgreSQL configuration tuning
ALTER SYSTEM SET shared_buffers = '1GB';
ALTER SYSTEM SET effective_cache_size = '3GB';
ALTER SYSTEM SET maintenance_work_mem = '256MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
SELECT pg_reload_conf();
```

### Redis Optimization
```bash
# Redis configuration
echo "maxmemory 2gb" >> redis.conf
echo "maxmemory-policy allkeys-lru" >> redis.conf
echo "save 900 1" >> redis.conf
echo "save 300 10" >> redis.conf
echo "save 60 10000" >> redis.conf
```

## Troubleshooting

### Common Issues

1. **Service Won't Start**
```bash
# Check logs
docker-compose logs payment-gateway

# Check resource usage
docker stats

# Verify environment variables
docker-compose config
```

2. **Database Connection Issues**
```bash
# Test database connectivity
docker exec -it apg-postgres psql -U payment_user -d payment_gateway -c "SELECT 1;"

# Check database logs
docker-compose logs postgres
```

3. **SSL Certificate Issues**
```bash
# Test SSL certificate
openssl s_client -connect your-domain.com:443 -servername your-domain.com

# Renew Let's Encrypt certificate
sudo certbot renew
```

4. **High Memory Usage**
```bash
# Monitor memory usage
docker exec payment-gateway ps aux --sort=-%mem | head

# Check for memory leaks
docker exec payment-gateway cat /proc/meminfo
```

### Performance Monitoring

```bash
# Monitor system resources
htop

# Check Docker resource usage
docker system df
docker system prune -f

# Monitor network connections
ss -tuln

# Check disk I/O
iotop
```

## Deployment Checklist

### Pre-deployment
- [ ] Environment variables configured
- [ ] SSL certificates installed
- [ ] Database schema migrated
- [ ] Backup procedures tested
- [ ] Monitoring configured
- [ ] Security hardening applied
- [ ] Load testing completed

### Post-deployment
- [ ] Health checks passing
- [ ] All services running
- [ ] Logs are clean
- [ ] Monitoring alerts configured
- [ ] Backup jobs scheduled
- [ ] Documentation updated
- [ ] Team notified

### Provider Verification
- [ ] Stripe webhooks configured
- [ ] Adyen notifications working
- [ ] Flutterwave callbacks tested
- [ ] M-Pesa endpoints verified
- [ ] DPO integration confirmed
- [ ] Pesapal IPN configured

## Support & Maintenance

### Regular Maintenance Tasks
- **Daily**: Check service health, review error logs
- **Weekly**: Verify backups, update security patches
- **Monthly**: Review performance metrics, update dependencies
- **Quarterly**: Security audit, disaster recovery test

### Contact Information
- **Technical Support**: nyimbi@gmail.com
- **Website**: www.datacraft.co.ke
- **Documentation**: /docs/
- **Issue Tracker**: GitHub Issues

© 2025 Datacraft. All rights reserved.