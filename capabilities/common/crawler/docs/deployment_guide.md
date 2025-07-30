# APG Crawler Capability - Deployment Guide

**Version:** 2.0.0  
**Author:** Datacraft  
**Copyright:** © 2025 Datacraft  
**Email:** nyimbi@gmail.com  

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Database Setup](#database-setup)
5. [Environment Setup](#environment-setup)
6. [Production Deployment](#production-deployment)
7. [Docker Deployment](#docker-deployment)
8. [Kubernetes Deployment](#kubernetes-deployment)
9. [Monitoring and Logging](#monitoring-and-logging)
10. [Performance Tuning](#performance-tuning)
11. [Security Hardening](#security-hardening)
12. [Backup and Recovery](#backup-and-recovery)
13. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

**Minimum Requirements:**
- Python 3.12+
- PostgreSQL 14+
- Redis 6+ (for caching and queuing)
- 4GB RAM
- 2 CPU cores
- 20GB storage

**Recommended for Production:**
- Python 3.12+
- PostgreSQL 15+
- Redis 7+
- 16GB RAM
- 8 CPU cores
- 100GB SSD storage
- Load balancer (nginx/Apache)

### Dependencies

**Core Python Packages:**
```
asyncio>=3.4.3
aiohttp>=3.9.0
httpx>=0.27.0
beautifulsoup4>=4.12.0
lxml>=5.0.0
playwright>=1.40.0
selenium>=4.15.0
cloudscraper>=1.2.71
```

**AI/ML Packages:**
```
spacy>=3.7.0
nltk>=3.8.1
textblob>=0.17.1
langdetect>=1.0.9
trafilatura>=1.6.0
newspaper3k>=0.2.8
readability-lxml>=0.8.1
```

**Database and Infrastructure:**
```
sqlalchemy>=2.0.0
asyncpg>=0.29.0
alembic>=1.13.0
redis>=5.0.0
celery>=5.3.0
```

**APG Integration:**
```
flask>=3.0.0
flask-appbuilder>=4.4.0
pydantic>=2.5.0
```

## Installation

### Development Installation

1. **Clone the Repository:**
```bash
git clone <apg-repository>
cd apg/capabilities/common/crawler
```

2. **Create Virtual Environment:**
```bash
python3.12 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install Additional ML Models:**
```bash
# Install spaCy English model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

5. **Install Browser Dependencies:**
```bash
# Install Playwright browsers
playwright install chromium
playwright install firefox

# For Selenium (optional)
# Download ChromeDriver and place in PATH
```

### Production Installation

1. **System Packages (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev
sudo apt install -y postgresql-15 postgresql-contrib
sudo apt install -y redis-server
sudo apt install -y nginx
sudo apt install -y supervisor
```

2. **Python Environment:**
```bash
# Create production user
sudo useradd -m -s /bin/bash apgcrawler
sudo su - apgcrawler

# Setup environment
python3.12 -m venv /opt/apg/crawler/venv
source /opt/apg/crawler/venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r /opt/apg/crawler/requirements.txt
```

3. **Browser Setup for Production:**
```bash
# Install system dependencies for headless browsers
sudo apt install -y chromium-browser firefox-esr
sudo apt install -y xvfb  # Virtual display

# Install Playwright
playwright install --with-deps chromium
```

## Configuration

### Configuration Files

Create configuration files for different environments:

**`config/development.py`:**
```python
import os

class DevelopmentConfig:
    # Database
    DATABASE_URL = "postgresql://user:pass@localhost/apg_crawler_dev"
    
    # Redis
    REDIS_URL = "redis://localhost:6379/0"
    
    # Crawler Settings
    MAX_CONCURRENT_REQUESTS = 5
    MAX_SESSIONS = 10
    DEFAULT_TIMEOUT = 30
    
    # Content Processing
    ENABLE_CONTENT_INTELLIGENCE = True
    ENABLE_RAG_PROCESSING = True
    ENABLE_GRAPHRAG_PROCESSING = True
    
    # Security
    RATE_LIMIT_PER_DOMAIN = 1.0  # requests per second
    RESPECT_ROBOTS_TXT = True
    
    # Logging
    LOG_LEVEL = "DEBUG"
    LOG_FILE = "/tmp/apg_crawler.log"
```

**`config/production.py`:**
```python
import os

class ProductionConfig:
    # Database (use environment variables)
    DATABASE_URL = os.environ.get('DATABASE_URL')
    
    # Redis
    REDIS_URL = os.environ.get('REDIS_URL')
    
    # Crawler Settings
    MAX_CONCURRENT_REQUESTS = 20
    MAX_SESSIONS = 50
    DEFAULT_TIMEOUT = 45
    
    # Content Processing
    ENABLE_CONTENT_INTELLIGENCE = True
    ENABLE_RAG_PROCESSING = True
    ENABLE_GRAPHRAG_PROCESSING = True
    
    # Security
    RATE_LIMIT_PER_DOMAIN = 0.5  # More conservative in production
    RESPECT_ROBOTS_TXT = True
    USER_AGENT_ROTATION = True
    
    # Performance
    CONNECTION_POOL_SIZE = 50
    CONNECTION_POOL_MAX_OVERFLOW = 20
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FILE = "/var/log/apg/crawler.log"
    
    # Monitoring
    METRICS_ENABLED = True
    HEALTH_CHECK_PORT = 8080
```

### Environment Variables

Create `.env` file for sensitive configuration:

```bash
# Database
DATABASE_URL=postgresql://apgcrawler:secure_password@localhost/apg_crawler_prod
DATABASE_POOL_SIZE=20

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=redis_password

# Security
SECRET_KEY=your-secret-key-here
ENCRYPTION_KEY=your-encryption-key

# APG Integration
APG_API_URL=https://api.apg.datacraft.co.ke
APG_API_KEY=your-apg-api-key
APG_TENANT_ID=your-tenant-id

# External Services
OPENAI_API_KEY=your-openai-key  # For embeddings
ANTHROPIC_API_KEY=your-anthropic-key

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_URL=https://grafana.datacraft.co.ke
```

## Database Setup

### PostgreSQL Installation and Configuration

1. **Install PostgreSQL:**
```bash
# Ubuntu/Debian
sudo apt install postgresql-15 postgresql-contrib

# CentOS/RHEL
sudo yum install postgresql15-server postgresql15-contrib
```

2. **Configure PostgreSQL:**
```bash
# Switch to postgres user
sudo su - postgres

# Create database and user
createdb apg_crawler_prod
createuser apgcrawler
psql -c "ALTER USER apgcrawler WITH PASSWORD 'secure_password';"
psql -c "GRANT ALL PRIVILEGES ON DATABASE apg_crawler_prod TO apgcrawler;"
```

3. **Enable Required Extensions:**
```sql
-- Connect to database
psql apg_crawler_prod

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "vector";  -- For RAG embeddings

-- Exit
\q
```

4. **Run Database Migrations:**
```bash
# Navigate to crawler directory
cd /opt/apg/crawler

# Run schema creation
psql -U apgcrawler -d apg_crawler_prod -f database_schema.sql

# Verify tables created
psql -U apgcrawler -d apg_crawler_prod -c "\dt"
```

### Database Performance Tuning

**`postgresql.conf` optimizations:**
```conf
# Memory settings
shared_buffers = 4GB                    # 25% of total RAM
effective_cache_size = 12GB             # 75% of total RAM
work_mem = 64MB                         # For complex queries
maintenance_work_mem = 1GB              # For maintenance operations

# Connection settings
max_connections = 200
shared_preload_libraries = 'pg_stat_statements'

# Checkpoint settings
checkpoint_completion_target = 0.9
wal_buffers = 16MB
checkpoint_timeout = 10min

# Query planning
random_page_cost = 1.1                  # For SSD storage
effective_io_concurrency = 200          # For SSD

# Logging
log_min_duration_statement = 1000       # Log slow queries
log_checkpoints = on
log_connections = on
log_disconnections = on
```

## Environment Setup

### Service User Setup

1. **Create Service User:**
```bash
sudo useradd -r -s /bin/false apgcrawler
sudo mkdir -p /opt/apg/crawler
sudo mkdir -p /var/log/apg
sudo mkdir -p /var/run/apg
sudo chown -R apgcrawler:apgcrawler /opt/apg/crawler
sudo chown -R apgcrawler:apgcrawler /var/log/apg
sudo chown -R apgcrawler:apgcrawler /var/run/apg
```

2. **Setup Application Files:**
```bash
# Copy application files
sudo cp -r capabilities/common/crawler/* /opt/apg/crawler/
sudo chown -R apgcrawler:apgcrawler /opt/apg/crawler

# Set permissions
sudo chmod +x /opt/apg/crawler/*.py
sudo chmod 600 /opt/apg/crawler/.env
```

### Systemd Service Configuration

**`/etc/systemd/system/apg-crawler.service`:**
```ini
[Unit]
Description=APG Crawler Service
After=network.target postgresql.service redis.service
Wants=postgresql.service redis.service

[Service]
Type=forking
User=apgcrawler
Group=apgcrawler
WorkingDirectory=/opt/apg/crawler
Environment=PATH=/opt/apg/crawler/venv/bin
EnvironmentFile=/opt/apg/crawler/.env
ExecStart=/opt/apg/crawler/venv/bin/python -m gunicorn app:app \
    --bind 127.0.0.1:8000 \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --pid /var/run/apg/crawler.pid \
    --daemon
ExecReload=/bin/kill -s HUP $MAINPID
ExecStop=/bin/kill -s QUIT $MAINPID
PIDFile=/var/run/apg/crawler.pid
TimeoutStopSec=30
KillMode=mixed
PrivateTmp=true
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
```

**Enable and start service:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable apg-crawler
sudo systemctl start apg-crawler
sudo systemctl status apg-crawler
```

## Production Deployment

### Nginx Configuration

**`/etc/nginx/sites-available/apg-crawler`:**
```nginx
upstream apg_crawler {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;  # Additional worker if needed
    keepalive 32;
}

server {
    listen 80;
    server_name crawler.apg.datacraft.co.ke;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name crawler.apg.datacraft.co.ke;
    
    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/crawler.apg.datacraft.co.ke/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/crawler.apg.datacraft.co.ke/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;
    
    # Security headers
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=crawler_api:10m rate=10r/s;
    
    # Logging
    access_log /var/log/nginx/apg_crawler_access.log;
    error_log /var/log/nginx/apg_crawler_error.log;
    
    # Main application
    location / {
        limit_req zone=crawler_api burst=20 nodelay;
        
        proxy_pass http://apg_crawler;
        proxy_set_header Host $http_host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_redirect off;
        
        # Timeouts for long crawling operations
        proxy_connect_timeout 30s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
        
        # Buffer settings
        proxy_buffering on;
        proxy_buffer_size 128k;
        proxy_buffers 4 256k;
    }
    
    # Health check endpoint
    location /health {
        proxy_pass http://apg_crawler;
        access_log off;
    }
    
    # Static files (if any)
    location /static {
        alias /opt/apg/crawler/static;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

**Enable nginx configuration:**
```bash
sudo ln -s /etc/nginx/sites-available/apg-crawler /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### SSL Certificate Setup

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d crawler.apg.datacraft.co.ke

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### Process Management with Supervisor

**`/etc/supervisor/conf.d/apg-crawler.conf`:**
```ini
[program:apg-crawler-web]
command=/opt/apg/crawler/venv/bin/gunicorn app:app --bind 127.0.0.1:8000 --workers 4
directory=/opt/apg/crawler
user=apgcrawler
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/apg/crawler-web.log
environment=PATH="/opt/apg/crawler/venv/bin"

[program:apg-crawler-worker]
command=/opt/apg/crawler/venv/bin/celery -A crawler.celery worker --loglevel=info --concurrency=8
directory=/opt/apg/crawler
user=apgcrawler
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/apg/crawler-worker.log
environment=PATH="/opt/apg/crawler/venv/bin"

[program:apg-crawler-scheduler]
command=/opt/apg/crawler/venv/bin/celery -A crawler.celery beat --loglevel=info
directory=/opt/apg/crawler
user=apgcrawler
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/apg/crawler-scheduler.log
environment=PATH="/opt/apg/crawler/venv/bin"

[group:apg-crawler]
programs=apg-crawler-web,apg-crawler-worker,apg-crawler-scheduler
```

**Start supervisor services:**
```bash
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start apg-crawler:*
```

## Docker Deployment

### Dockerfile

**`Dockerfile`:**
```dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    redis-tools \
    chromium \
    firefox-esr \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -r -s /bin/false apgcrawler

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install ML models
RUN python -m spacy download en_core_web_sm
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Install browsers for Playwright
RUN playwright install chromium --with-deps

# Copy application code
COPY . .

# Change ownership to app user
RUN chown -R apgcrawler:apgcrawler /app

# Switch to app user
USER apgcrawler

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Start command
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000", "--workers", "4"]
```

### Docker Compose

**`docker-compose.yml`:**
```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: apg_crawler
      POSTGRES_USER: apgcrawler
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database_schema.sql:/docker-entrypoint-initdb.d/schema.sql
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U apgcrawler"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5

  crawler:
    build: .
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    environment:
      DATABASE_URL: postgresql://apgcrawler:secure_password@postgres/apg_crawler
      REDIS_URL: redis://redis:6379/0
    ports:
      - "8000:8000"
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped

  worker:
    build: .
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    environment:
      DATABASE_URL: postgresql://apgcrawler:secure_password@postgres/apg_crawler
      REDIS_URL: redis://redis:6379/0
    command: celery -A crawler.celery worker --loglevel=info --concurrency=8
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    depends_on:
      - crawler
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

**Deploy with Docker Compose:**
```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f crawler

# Scale workers
docker-compose up -d --scale worker=4
```

## Kubernetes Deployment

### Kubernetes Manifests

**`k8s/namespace.yaml`:**
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: apg-crawler
```

**`k8s/configmap.yaml`:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: apg-crawler-config
  namespace: apg-crawler
data:
  config.py: |
    MAX_CONCURRENT_REQUESTS = 20
    MAX_SESSIONS = 50
    DEFAULT_TIMEOUT = 45
    ENABLE_CONTENT_INTELLIGENCE = True
    RATE_LIMIT_PER_DOMAIN = 0.5
```

**`k8s/secret.yaml`:**
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: apg-crawler-secrets
  namespace: apg-crawler
type: Opaque
data:
  database-url: <base64-encoded-database-url>
  redis-url: <base64-encoded-redis-url>
  secret-key: <base64-encoded-secret-key>
```

**`k8s/deployment.yaml`:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: apg-crawler
  namespace: apg-crawler
spec:
  replicas: 3
  selector:
    matchLabels:
      app: apg-crawler
  template:
    metadata:
      labels:
        app: apg-crawler
    spec:
      containers:
      - name: crawler
        image: apg-crawler:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: apg-crawler-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: apg-crawler-secrets
              key: redis-url
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

**`k8s/service.yaml`:**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: apg-crawler-service
  namespace: apg-crawler
spec:
  selector:
    app: apg-crawler
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```

**Deploy to Kubernetes:**
```bash
kubectl apply -f k8s/
kubectl get pods -n apg-crawler
kubectl logs -f deployment/apg-crawler -n apg-crawler
```

## Monitoring and Logging

### Prometheus Metrics

**`metrics.py`:**
```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Metrics
crawl_requests_total = Counter('crawler_requests_total', 'Total crawl requests', ['status', 'strategy'])
crawl_duration_seconds = Histogram('crawler_duration_seconds', 'Crawl duration')
active_crawls = Gauge('crawler_active_crawls', 'Number of active crawls')
content_intelligence_requests = Counter('crawler_intelligence_requests_total', 'Content intelligence requests')

def record_crawl_metrics(result, strategy, duration):
    status = 'success' if result.success else 'failure'
    crawl_requests_total.labels(status=status, strategy=strategy).inc()
    crawl_duration_seconds.observe(duration)

# Start metrics server
start_http_server(9090)
```

### Structured Logging

**`logging_config.py`:**
```python
import logging
import json
from datetime import datetime

class StructuredFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        return datetime.fromtimestamp(record.created).isoformat()
    
    def format(self, record):
        log_obj = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)
        
        # Add custom fields if present
        if hasattr(record, 'url'):
            log_obj['url'] = record.url
        if hasattr(record, 'tenant_id'):
            log_obj['tenant_id'] = record.tenant_id
        if hasattr(record, 'duration'):
            log_obj['duration'] = record.duration
            
        return json.dumps(log_obj)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler('/var/log/apg/crawler.json'),
        logging.StreamHandler()
    ]
)

for handler in logging.getLogger().handlers:
    handler.setFormatter(StructuredFormatter())
```

### Log Aggregation with ELK Stack

**`logstash.conf`:**
```ruby
input {
  file {
    path => "/var/log/apg/crawler.json"
    codec => "json"
    type => "apg-crawler"
  }
}

filter {
  if [type] == "apg-crawler" {
    date {
      match => [ "timestamp", "ISO8601" ]
    }
    
    if [level] == "ERROR" {
      mutate {
        add_tag => [ "error" ]
      }
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "apg-crawler-%{+YYYY.MM.dd}"
  }
}
```

## Performance Tuning

### Database Optimization

**Connection Pooling:**
```python
from sqlalchemy.pool import QueuePool

engine = create_async_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600
)
```

**Query Optimization:**
```sql
-- Create indexes for common queries
CREATE INDEX CONCURRENTLY idx_cr_data_records_tenant_created 
ON cr_data_records (tenant_id, created_at DESC);

CREATE INDEX CONCURRENTLY idx_cr_data_records_fingerprint 
ON cr_data_records USING hash (content_fingerprint);

CREATE INDEX CONCURRENTLY idx_cr_rag_chunks_tenant_embedding 
ON cr_rag_chunks (tenant_id) INCLUDE (vector_embedding);

-- Analyze table statistics
ANALYZE cr_data_records;
ANALYZE cr_rag_chunks;
```

### Application Optimization

**Async Connection Pooling:**
```python
import aiohttp
from aiohttp_retry import RetryClient, ExponentialRetry

# Global session with connection pooling
connector = aiohttp.TCPConnector(
    limit=100,                    # Total connection pool size
    limit_per_host=20,           # Per-host connection limit
    ttl_dns_cache=300,           # DNS cache TTL
    use_dns_cache=True,
    keepalive_timeout=30,
    enable_cleanup_closed=True
)

retry_options = ExponentialRetry(attempts=3)
session = RetryClient(
    connector=connector,
    retry_options=retry_options,
    timeout=aiohttp.ClientTimeout(total=60)
)
```

**Memory Management:**
```python
import gc
import psutil

async def monitor_memory():
    """Monitor and manage memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    if memory_info.rss > 2 * 1024 * 1024 * 1024:  # 2GB
        logging.warning(f"High memory usage: {memory_info.rss / 1024 / 1024:.1f}MB")
        gc.collect()  # Force garbage collection
```

### Caching Strategy

**Redis Caching:**
```python
import redis.asyncio as redis
import pickle
import hashlib

class ContentCache:
    def __init__(self, redis_url):
        self.redis = redis.from_url(redis_url)
    
    async def get_cached_content(self, url: str, config_hash: str):
        key = f"content:{hashlib.md5(f'{url}:{config_hash}'.encode()).hexdigest()}"
        cached = await self.redis.get(key)
        if cached:
            return pickle.loads(cached)
        return None
    
    async def cache_content(self, url: str, config_hash: str, result, ttl=3600):
        key = f"content:{hashlib.md5(f'{url}:{config_hash}'.encode()).hexdigest()}"
        await self.redis.setex(key, ttl, pickle.dumps(result))
```

## Security Hardening

### Network Security

**Firewall Configuration (ufw):**
```bash
# Allow SSH
sudo ufw allow 22/tcp

# Allow HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Allow PostgreSQL (only from application server)
sudo ufw allow from 10.0.1.0/24 to any port 5432

# Allow Redis (only from application server)
sudo ufw allow from 10.0.1.0/24 to any port 6379

# Enable firewall
sudo ufw --force enable
```

### Application Security

**Input Validation:**
```python
from urllib.parse import urlparse
import validators

def validate_crawl_url(url: str) -> bool:
    """Validate URL for security"""
    if not validators.url(url):
        return False
    
    parsed = urlparse(url)
    
    # Block private networks
    private_networks = ['127.', '10.', '192.168.', '172.16.']
    if any(parsed.hostname.startswith(net) for net in private_networks):
        return False
    
    # Block non-standard ports
    if parsed.port and parsed.port not in [80, 443, 8080, 8443]:
        return False
    
    return True
```

**Rate Limiting:**
```python
from functools import wraps
import time
from collections import defaultdict

rate_limits = defaultdict(list)

def rate_limit(max_requests=100, window=3600):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            client_ip = get_client_ip()  # Extract from request
            now = time.time()
            
            # Clean old requests
            rate_limits[client_ip] = [
                req_time for req_time in rate_limits[client_ip]
                if now - req_time < window
            ]
            
            # Check rate limit
            if len(rate_limits[client_ip]) >= max_requests:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            rate_limits[client_ip].append(now)
            return await func(*args, **kwargs)
        return wrapper
    return decorator
```

### SSL/TLS Configuration

**Strong SSL Configuration:**
```nginx
# Strong SSL configuration
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
ssl_prefer_server_ciphers off;
ssl_session_cache shared:SSL:10m;
ssl_session_timeout 10m;

# HSTS
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;

# OCSP Stapling
ssl_stapling on;
ssl_stapling_verify on;
ssl_trusted_certificate /etc/letsencrypt/live/crawler.apg.datacraft.co.ke/chain.pem;
```

## Backup and Recovery

### Database Backup

**Automated Backup Script:**
```bash
#!/bin/bash
# /opt/apg/scripts/backup_database.sh

BACKUP_DIR="/opt/apg/backups"
DATE=$(date +%Y%m%d_%H%M%S)
DB_NAME="apg_crawler_prod"
DB_USER="apgcrawler"

# Create backup directory
mkdir -p $BACKUP_DIR

# Create database backup
pg_dump -U $DB_USER -h localhost $DB_NAME | gzip > $BACKUP_DIR/crawler_backup_$DATE.sql.gz

# Keep only last 7 days of backups
find $BACKUP_DIR -name "crawler_backup_*.sql.gz" -mtime +7 -delete

# Log backup
echo "$(date): Database backup completed - crawler_backup_$DATE.sql.gz" >> /var/log/apg/backup.log
```

**Cron job for automated backups:**
```bash
# Edit crontab for apgcrawler user
sudo crontab -u apgcrawler -e

# Add daily backup at 2 AM
0 2 * * * /opt/apg/scripts/backup_database.sh
```

### Application Data Backup

**Content and Configuration Backup:**
```bash
#!/bin/bash
# Backup application data

BACKUP_DIR="/opt/apg/backups/app"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup configuration
tar -czf $BACKUP_DIR/config_$DATE.tar.gz /opt/apg/crawler/config/

# Backup logs (last 30 days)
find /var/log/apg -name "*.log" -mtime -30 | tar -czf $BACKUP_DIR/logs_$DATE.tar.gz -T -

# Cleanup old backups
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
```

### Disaster Recovery Plan

**Recovery Procedure:**
```bash
# 1. Restore database
gunzip -c /opt/apg/backups/crawler_backup_YYYYMMDD_HHMMSS.sql.gz | psql -U apgcrawler apg_crawler_prod

# 2. Restore application files
tar -xzf /opt/apg/backups/app/config_YYYYMMDD_HHMMSS.tar.gz -C /

# 3. Restart services
sudo systemctl restart apg-crawler
sudo systemctl restart nginx

# 4. Verify functionality
curl -f http://localhost:8000/health
```

## Troubleshooting

### Common Issues

#### 1. Browser Dependencies Issues

**Problem:** Playwright or Selenium browsers not working
```bash
# Symptoms
playwright._impl._api_types.Error: Executable doesn't exist

# Solution
playwright install chromium --with-deps
sudo apt install -y chromium-browser xvfb
```

#### 2. Memory Issues

**Problem:** High memory usage and crashes
```bash
# Symptoms
MemoryError or OOMKilled in container logs

# Solutions
# 1. Increase memory limits
# 2. Tune garbage collection
export PYTHONMALLOC=malloc
export MALLOC_TRIM_THRESHOLD_=100000

# 3. Add memory monitoring
import gc
import psutil
```

#### 3. Database Connection Issues

**Problem:** Connection pool exhaustion
```bash
# Symptoms
sqlalchemy.exc.TimeoutError: QueuePool limit of size 20 overflow 30 reached

# Solution
# Increase pool size or optimize queries
engine = create_async_engine(
    DATABASE_URL,
    pool_size=50,
    max_overflow=100
)
```

#### 4. Rate Limiting Issues

**Problem:** Too many requests being blocked
```bash
# Symptoms
HTTP 429 errors or blocked by target sites

# Solution
# Adjust rate limiting
RATE_LIMIT_PER_DOMAIN = 0.2  # Slower crawling
USE_PROXY_ROTATION = True
```

### Debugging Tools

**Health Check Endpoint:**
```python
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    checks = {}
    
    # Database check
    try:
        await db_service.health_check()
        checks['database'] = 'healthy'
    except Exception as e:
        checks['database'] = f'unhealthy: {str(e)}'
    
    # Redis check
    try:
        await redis_client.ping()
        checks['redis'] = 'healthy'
    except Exception as e:
        checks['redis'] = f'unhealthy: {str(e)}'
    
    # Browser check
    try:
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch()
        await browser.close()
        await playwright.stop()
        checks['browser'] = 'healthy'
    except Exception as e:
        checks['browser'] = f'unhealthy: {str(e)}'
    
    overall_healthy = all('healthy' in status for status in checks.values())
    
    return {
        'status': 'healthy' if overall_healthy else 'unhealthy',
        'checks': checks,
        'timestamp': datetime.utcnow().isoformat()
    }
```

**Debug Mode Configuration:**
```python
# Enable debug logging
logging.getLogger('crawler').setLevel(logging.DEBUG)
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

# Add request tracing
import uuid

async def trace_request(request):
    trace_id = str(uuid.uuid4())
    request.state.trace_id = trace_id
    logger.info(f"Request started", extra={'trace_id': trace_id, 'url': request.url})
```

### Log Analysis

**Common Log Patterns:**
```bash
# Find errors in logs
grep "ERROR" /var/log/apg/crawler.log | tail -20

# Monitor crawl success rates
grep "SUCCESS\|FAILED" /var/log/apg/crawler.log | tail -100 | awk '{print $3}' | sort | uniq -c

# Check memory usage patterns
grep "memory" /var/log/apg/crawler.log | tail -50

# Monitor response times
grep "response_time" /var/log/apg/crawler.log | awk '{print $NF}' | sort -n
```

### Performance Analysis

**Database Query Analysis:**
```sql
-- Find slow queries
SELECT query, calls, total_time, mean_time
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;

-- Check connection usage
SELECT count(*) as connections, state 
FROM pg_stat_activity 
GROUP BY state;

-- Index usage analysis
SELECT schemaname, tablename, attname, n_distinct, correlation 
FROM pg_stats 
WHERE tablename LIKE 'cr_%';
```

### Support and Maintenance

**Regular Maintenance Tasks:**
```bash
# Weekly maintenance script
#!/bin/bash

# 1. Database maintenance
psql -U apgcrawler apg_crawler_prod -c "VACUUM ANALYZE;"

# 2. Log rotation
logrotate /etc/logrotate.d/apg-crawler

# 3. Clear old cache entries
redis-cli --scan --pattern "content:*" | head -1000 | xargs redis-cli del

# 4. Update browser drivers
playwright install chromium

# 5. Check disk usage
df -h
du -sh /var/log/apg/
du -sh /opt/apg/crawler/
```

**Monitoring Checklist:**
- [ ] Application health endpoint responding
- [ ] Database connections within limits
- [ ] Memory usage under 80%
- [ ] Disk usage under 80%
- [ ] Error rate under 5%
- [ ] Response times under 30s average
- [ ] SSL certificates valid (>30 days)
- [ ] Backups completed successfully

---

## Quick Deployment Checklist

### Pre-Deployment
- [ ] System requirements met
- [ ] Dependencies installed
- [ ] Database configured
- [ ] SSL certificates obtained
- [ ] Firewall configured
- [ ] Monitoring setup

### Deployment
- [ ] Application deployed
- [ ] Services started
- [ ] Health checks passing
- [ ] Load balancer configured
- [ ] DNS configured
- [ ] Backups scheduled

### Post-Deployment
- [ ] Smoke tests passed
- [ ] Performance tests passed
- [ ] Security scan passed
- [ ] Documentation updated
- [ ] Team trained
- [ ] Monitoring alerts configured

**For additional support:** nyimbi@gmail.com