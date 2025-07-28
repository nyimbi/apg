# Geographical Location Services - Deployment Guide

**Version:** 2.0.0  
**Author:** Nyimbi Odero <nyimbi@gmail.com>  
**Company:** Datacraft  
**Website:** www.datacraft.co.ke  

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Infrastructure Requirements](#infrastructure-requirements)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Database Setup](#database-setup)
6. [Configuration Management](#configuration-management)
7. [Environment Setup](#environment-setup)
8. [Security Configuration](#security-configuration)
9. [Performance Tuning](#performance-tuning)
10. [Monitoring & Logging](#monitoring--logging)
11. [Backup & Recovery](#backup--recovery)
12. [Troubleshooting](#troubleshooting)
13. [Integration Testing](#integration-testing)
14. [Production Checklist](#production-checklist)

---

## ✅ Prerequisites

### System Requirements

#### Minimum Hardware
- **CPU**: 4 cores, 2.4 GHz
- **RAM**: 8 GB
- **Storage**: 100 GB SSD
- **Network**: 1 Gbps

#### Recommended Hardware (Production)
- **CPU**: 16 cores, 3.0 GHz
- **RAM**: 32 GB
- **Storage**: 500 GB NVMe SSD
- **Network**: 10 Gbps

#### Software Dependencies
- **OS**: Ubuntu 20.04+ / CentOS 8+ / RHEL 8+
- **Python**: 3.11+
- **PostgreSQL**: 15+ with PostGIS 3.3+
- **Redis**: 7.0+
- **Docker**: 24.0+
- **Kubernetes**: 1.28+ (for K8s deployment)

### External Services

#### Required APIs
- **Geocoding Provider**: Google Maps API, Mapbox, or OpenCage
- **Weather Service**: OpenWeatherMap or AccuWeather (optional)
- **Authentication**: JWT token provider
- **Message Queue**: Redis Streams or Apache Kafka

#### Network Requirements
- **Outbound HTTPS**: Access to geocoding APIs
- **Inbound HTTP/HTTPS**: API access (ports 80/443)
- **Database Access**: PostgreSQL (port 5432)
- **Cache Access**: Redis (port 6379)
- **WebSocket**: Real-time streaming (port 8080)

---

## 🏗️ Infrastructure Requirements

### Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Load Balancer  │    │   API Gateway   │    │  Rate Limiter   │
│    (HAProxy)    │    │    (Kong)       │    │    (Redis)      │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
┌─────────────────────────────────┼─────────────────────────────────┐
│                      Application Tier                            │
├─────────────────┬───────────────┬───────────────┬─────────────────┤
│   GLS Service   │   GLS Service │   GLS Service │   GLS Service   │
│   (Instance 1)  │   (Instance 2)│   (Instance 3)│   (Instance 4)  │
└─────────────────┴───────────────┴───────────────┴─────────────────┘
                                 │
┌─────────────────────────────────┼─────────────────────────────────┐
│                        Data Tier                                  │
├─────────────────┬───────────────┬───────────────┬─────────────────┤
│   PostgreSQL    │     Redis     │  ML Models    │   File Storage  │
│   (Primary +    │   (Cluster)   │   (TensorFlow │     (S3/Minio)  │
│    Replicas)    │               │    Serving)   │                 │
└─────────────────┴───────────────┴───────────────┴─────────────────┘
```

### Resource Allocation

#### Production Environment (Per Instance)
```yaml
resources:
  requests:
    memory: "4Gi"
    cpu: "2000m"
  limits:
    memory: "8Gi"
    cpu: "4000m"
```

#### Staging Environment (Per Instance)
```yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

#### Development Environment (Per Instance)
```yaml
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
```

---

## 🐳 Docker Deployment

### Dockerfile

```dockerfile
# Multi-stage build for production optimization
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libgeos-dev \
    libproj-dev \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    libgeos-c1v5 \
    libproj22 \
    libgdal28 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash gls

# Set work directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Change ownership to non-root user
RUN chown -R gls:gls /app

# Switch to non-root user
USER gls

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/geographical-location/health || exit 1

# Expose port
EXPOSE 8000

# Start command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  gls-api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://gls_user:gls_password@postgres:5432/gls_db
      - REDIS_URL=redis://redis:6379/0
      - API_KEY_GOOGLE_MAPS=${GOOGLE_MAPS_API_KEY}
      - JWT_SECRET=${JWT_SECRET}
      - LOG_LEVEL=INFO
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 4G
          cpus: '2'
        reservations:
          memory: 2G
          cpus: '1'

  postgres:
    image: postgis/postgis:15-3.3
    environment:
      - POSTGRES_DB=gls_db
      - POSTGRES_USER=gls_user
      - POSTGRES_PASSWORD=gls_password
      - POSTGRES_INITDB_ARGS=--encoding=UTF-8 --lc-collate=C --lc-ctype=C
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-scripts:/docker-entrypoint-initdb.d
    ports:
      - "5432:5432"
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U gls_user -d gls_db"]
      interval: 10s
      timeout: 5s
      retries: 5
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
        reservations:
          memory: 4G
          cpus: '2'

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --maxmemory 2gb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1'
        reservations:
          memory: 1G
          cpus: '0.5'

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - gls-api
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  default:
    driver: bridge
```

### Environment Configuration

```bash
# .env file
DATABASE_URL=postgresql://gls_user:gls_password@localhost:5432/gls_db
REDIS_URL=redis://localhost:6379/0

# API Keys
GOOGLE_MAPS_API_KEY=your_google_maps_api_key
MAPBOX_API_KEY=your_mapbox_api_key
OPENCAGE_API_KEY=your_opencage_api_key

# Authentication
JWT_SECRET=your_jwt_secret_key
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# Application
API_PREFIX=/api/v1/geographical-location
LOG_LEVEL=INFO
DEBUG=false
CORS_ORIGINS=["http://localhost:3000","https://yourdomain.com"]

# Performance
MAX_CONCURRENT_REQUESTS=1000
REQUEST_TIMEOUT_SECONDS=30
BATCH_SIZE_LIMIT=1000
CACHE_TTL_SECONDS=3600

# ML Models
MODEL_CACHE_SIZE=5
TENSORFLOW_SERVING_URL=http://tf-serving:8501
PYTORCH_MODEL_PATH=/app/models

# Monitoring
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=8001
JAEGER_ENABLED=true
JAEGER_ENDPOINT=http://jaeger:14268/api/traces
```

---

## ☸️ Kubernetes Deployment

### Namespace

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: gls
  labels:
    name: geographical-location-services
```

### ConfigMap

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: gls-config
  namespace: gls
data:
  API_PREFIX: "/api/v1/geographical-location"
  LOG_LEVEL: "INFO"
  DEBUG: "false"
  MAX_CONCURRENT_REQUESTS: "1000"
  REQUEST_TIMEOUT_SECONDS: "30"
  BATCH_SIZE_LIMIT: "1000"
  CACHE_TTL_SECONDS: "3600"
  PROMETHEUS_ENABLED: "true"
  PROMETHEUS_PORT: "8001"
  CORS_ORIGINS: '["https://yourdomain.com"]'
```

### Secrets

```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: gls-secrets
  namespace: gls
type: Opaque
data:
  DATABASE_URL: cG9zdGdyZXNxbDovL2dscy51c2VyOmdscy5wYXNzd29yZEBwb3N0Z3Jlcy1zZXJ2aWNlOjU0MzIvZ2xzX2Ri
  REDIS_URL: cmVkaXM6Ly9yZWRpcy1zZXJ2aWNlOjYzNzkvMA==
  JWT_SECRET: eW91cl9qd3Rfc2VjcmV0X2tleQ==
  GOOGLE_MAPS_API_KEY: eW91cl9nb29nbGVfbWFwc19hcGlfa2V5
  MAPBOX_API_KEY: eW91cl9tYXBib3hfYXBpX2tleQ==
```

### Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gls-api
  namespace: gls
  labels:
    app: gls-api
spec:
  replicas: 4
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: gls-api
  template:
    metadata:
      labels:
        app: gls-api
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8001"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: gls-api
        image: datacraft/gls-api:2.0.0
        ports:
        - containerPort: 8000
        - containerPort: 8001  # Prometheus metrics
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: gls-secrets
              key: DATABASE_URL
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: gls-secrets
              key: REDIS_URL
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: gls-secrets
              key: JWT_SECRET
        - name: GOOGLE_MAPS_API_KEY
          valueFrom:
            secretKeyRef:
              name: gls-secrets
              key: GOOGLE_MAPS_API_KEY
        envFrom:
        - configMapRef:
            name: gls-config
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /api/v1/geographical-location/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /api/v1/geographical-location/health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
        volumeMounts:
        - name: logs
          mountPath: /app/logs
        - name: models
          mountPath: /app/models
      volumes:
      - name: logs
        emptyDir: {}
      - name: models
        persistentVolumeClaim:
          claimName: gls-models-pvc
      nodeSelector:
        node-type: compute
      tolerations:
      - key: "compute"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"
```

### Service

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: gls-api-service
  namespace: gls
  labels:
    app: gls-api
spec:
  selector:
    app: gls-api
  ports:
  - name: http
    port: 80
    targetPort: 8000
    protocol: TCP
  - name: metrics
    port: 8001
    targetPort: 8001
    protocol: TCP
  type: ClusterIP
```

### Ingress

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: gls-api-ingress
  namespace: gls
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "1000"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
spec:
  tls:
  - hosts:
    - api.datacraft.co.ke
    secretName: gls-api-tls
  rules:
  - host: api.datacraft.co.ke
    http:
      paths:
      - path: /api/v1/geographical-location
        pathType: Prefix
        backend:
          service:
            name: gls-api-service
            port:
              number: 80
```

### Horizontal Pod Autoscaler

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gls-api-hpa
  namespace: gls
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gls-api
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
```

---

## 🗄️ Database Setup

### PostgreSQL with PostGIS

```sql
-- init-db.sql
-- Create database and user
CREATE DATABASE gls_db;
CREATE USER gls_user WITH ENCRYPTED PASSWORD 'gls_password';
GRANT ALL PRIVILEGES ON DATABASE gls_db TO gls_user;

-- Connect to gls_db
\c gls_db;

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Grant permissions
GRANT ALL ON SCHEMA public TO gls_user;
GRANT ALL ON ALL TABLES IN SCHEMA public TO gls_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO gls_user;
GRANT ALL ON ALL FUNCTIONS IN SCHEMA public TO gls_user;

-- Create custom functions for H3 (placeholder)
CREATE OR REPLACE FUNCTION generate_h3_indices_sql(lat DECIMAL, lng DECIMAL)
RETURNS JSONB AS $$
DECLARE
    result JSONB := '{}';
    base_hash BIGINT;
    resolution_hash BIGINT;
    h3_index TEXT;
    resolution INT;
BEGIN
    -- Simple deterministic H3-like index generation
    base_hash := abs(hashtext(lat::TEXT || ',' || lng::TEXT));
    
    FOR resolution IN 0..10 LOOP
        resolution_hash := (base_hash >> (resolution * 2)) & 9223372036854775807;
        h3_index := '8' || to_hex(resolution) || to_hex(resolution_hash % 1152921504606846976);
        result := result || jsonb_build_object(resolution::TEXT, h3_index);
    END LOOP;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Create geohash function
CREATE OR REPLACE FUNCTION calculate_geohash(lat DECIMAL, lng DECIMAL, precision INT DEFAULT 8)
RETURNS TEXT AS $$
DECLARE
    base32 TEXT := '0123456789bcdefghjkmnpqrstuvwxyz';
    lat_range DECIMAL[] := ARRAY[-90.0, 90.0];
    lng_range DECIMAL[] := ARRAY[-180.0, 180.0];
    geohash TEXT := '';
    bits INTEGER := 0;
    bit_count INTEGER := 0;
    even_bit BOOLEAN := TRUE;
    mid DECIMAL;
BEGIN
    WHILE length(geohash) < precision LOOP
        IF even_bit THEN
            mid := (lng_range[1] + lng_range[2]) / 2;
            IF lng >= mid THEN
                bits := (bits << 1) | 1;
                lng_range[1] := mid;
            ELSE
                bits := bits << 1;
                lng_range[2] := mid;
            END IF;
        ELSE
            mid := (lat_range[1] + lat_range[2]) / 2;
            IF lat >= mid THEN
                bits := (bits << 1) | 1;
                lat_range[1] := mid;
            ELSE
                bits := bits << 1;
                lat_range[2] := mid;
            END IF;
        END IF;
        
        even_bit := NOT even_bit;
        bit_count := bit_count + 1;
        
        IF bit_count = 5 THEN
            geohash := geohash || substr(base32, (bits % 32) + 1, 1);
            bits := 0;
            bit_count := 0;
        END IF;
    END LOOP;
    
    RETURN geohash;
END;
$$ LANGUAGE plpgsql IMMUTABLE;
```

### Database Migration Scripts

```python
# migrations/001_initial_schema.py
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

def upgrade():
    # Create coordinates table
    op.create_table(
        'coordinates',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('entity_id', sa.String(255), nullable=False),
        sa.Column('entity_type', sa.String(50), nullable=False),
        sa.Column('latitude', sa.Numeric(10, 8), nullable=False),
        sa.Column('longitude', sa.Numeric(11, 8), nullable=False),
        sa.Column('altitude', sa.Numeric(8, 2)),
        sa.Column('h3_indices', postgresql.JSONB(), nullable=False, server_default='{}'),
        sa.Column('geohash', sa.String(12)),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('accuracy_meters', sa.Numeric(8, 2)),
        sa.Column('speed_kmh', sa.Numeric(8, 2)),
        sa.Column('bearing_degrees', sa.Numeric(5, 2)),
        sa.Column('metadata', postgresql.JSONB(), server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
        sa.Column('tenant_id', sa.String(255), nullable=False),
    )
    
    # Add check constraints
    op.create_check_constraint(
        'coordinates_latitude_check',
        'coordinates',
        'latitude >= -90 AND latitude <= 90'
    )
    op.create_check_constraint(
        'coordinates_longitude_check', 
        'coordinates',
        'longitude >= -180 AND longitude <= 180'
    )
    
    # Add computed column for primary H3 index
    op.execute("ALTER TABLE coordinates ADD COLUMN primary_h3_index VARCHAR(15) GENERATED ALWAYS AS (h3_indices->>'4') STORED")
    
    # Add PostGIS geometry column
    op.execute("ALTER TABLE coordinates ADD COLUMN geom GEOMETRY(POINT, 4326) GENERATED ALWAYS AS (ST_Point(longitude, latitude)) STORED")
    
    # Create indices
    op.create_index('idx_coordinates_geom', 'coordinates', ['geom'], postgresql_using='gist')
    op.create_index('idx_coordinates_entity_time', 'coordinates', ['entity_id', 'timestamp'])
    op.create_index('idx_coordinates_h3_city', 'coordinates', ['primary_h3_index'])
    op.create_index('idx_coordinates_timestamp', 'coordinates', ['timestamp'])
    op.create_index('idx_coordinates_tenant', 'coordinates', ['tenant_id'])

def downgrade():
    op.drop_table('coordinates')
```

### Connection Pool Configuration

```python
# database.py
import asyncpg
from asyncpg import Pool
from typing import Optional
import logging

class DatabaseManager:
    """Async database connection manager with pooling."""
    
    def __init__(self):
        self.pool: Optional[Pool] = None
        self.logger = logging.getLogger(__name__)
    
    async def create_pool(
        self,
        database_url: str,
        min_connections: int = 10,
        max_connections: int = 50,
        command_timeout: int = 60
    ):
        """Create connection pool."""
        
        self.pool = await asyncpg.create_pool(
            database_url,
            min_size=min_connections,
            max_size=max_connections,
            command_timeout=command_timeout,
            server_settings={
                'application_name': 'gls-api',
                'timezone': 'UTC',
            }
        )
        
        self.logger.info(f"Database pool created: {min_connections}-{max_connections} connections")
    
    async def close_pool(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            self.logger.info("Database pool closed")
    
    async def execute(self, query: str, *args):
        """Execute a query."""
        async with self.pool.acquire() as connection:
            return await connection.execute(query, *args)
    
    async def fetch_all(self, query: str, *args):
        """Fetch all rows."""
        async with self.pool.acquire() as connection:
            return await connection.fetch(query, *args)
    
    async def fetch_one(self, query: str, *args):
        """Fetch one row."""
        async with self.pool.acquire() as connection:
            return await connection.fetchrow(query, *args)
    
    async def fetch_val(self, query: str, *args):
        """Fetch single value."""
        async with self.pool.acquire() as connection:
            return await connection.fetchval(query, *args)
    
    async def executemany(self, query: str, args_list):
        """Execute query with multiple parameter sets."""
        async with self.pool.acquire() as connection:
            return await connection.executemany(query, args_list)
    
    async def transaction(self):
        """Get transaction context manager."""
        return self.pool.acquire()

# Usage
db = DatabaseManager()

async def startup():
    await db.create_pool(
        database_url=settings.DATABASE_URL,
        min_connections=10,
        max_connections=50
    )

async def shutdown():
    await db.close_pool()
```

---

## ⚙️ Configuration Management

### Application Settings

```python
# settings.py
from pydantic import BaseSettings, Field, validator
from typing import List, Optional
import os

class Settings(BaseSettings):
    """Application configuration settings."""
    
    # Database
    DATABASE_URL: str = Field(..., env="DATABASE_URL")
    DATABASE_POOL_MIN: int = Field(10, env="DATABASE_POOL_MIN")
    DATABASE_POOL_MAX: int = Field(50, env="DATABASE_POOL_MAX")
    
    # Redis
    REDIS_URL: str = Field(..., env="REDIS_URL")
    REDIS_MAX_CONNECTIONS: int = Field(100, env="REDIS_MAX_CONNECTIONS")
    
    # API Configuration
    API_PREFIX: str = Field("/api/v1/geographical-location", env="API_PREFIX")
    API_VERSION: str = Field("2.0.0", env="API_VERSION")
    DEBUG: bool = Field(False, env="DEBUG")
    
    # Security
    JWT_SECRET: str = Field(..., env="JWT_SECRET")
    JWT_ALGORITHM: str = Field("HS256", env="JWT_ALGORITHM")
    JWT_EXPIRATION_HOURS: int = Field(24, env="JWT_EXPIRATION_HOURS")
    
    # CORS
    CORS_ORIGINS: List[str] = Field(["*"], env="CORS_ORIGINS")
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = Field(1000, env="RATE_LIMIT_REQUESTS")
    RATE_LIMIT_WINDOW: int = Field(3600, env="RATE_LIMIT_WINDOW")
    
    # External APIs
    GOOGLE_MAPS_API_KEY: Optional[str] = Field(None, env="GOOGLE_MAPS_API_KEY")
    MAPBOX_API_KEY: Optional[str] = Field(None, env="MAPBOX_API_KEY")
    OPENCAGE_API_KEY: Optional[str] = Field(None, env="OPENCAGE_API_KEY")
    
    # Performance
    MAX_CONCURRENT_REQUESTS: int = Field(1000, env="MAX_CONCURRENT_REQUESTS")
    REQUEST_TIMEOUT_SECONDS: int = Field(30, env="REQUEST_TIMEOUT_SECONDS")
    BATCH_SIZE_LIMIT: int = Field(1000, env="BATCH_SIZE_LIMIT")
    
    # Caching
    CACHE_TTL_SECONDS: int = Field(3600, env="CACHE_TTL_SECONDS")
    CACHE_MAX_KEYS: int = Field(100000, env="CACHE_MAX_KEYS")
    
    # ML Models
    MODEL_CACHE_SIZE: int = Field(5, env="MODEL_CACHE_SIZE")
    TENSORFLOW_SERVING_URL: Optional[str] = Field(None, env="TENSORFLOW_SERVING_URL")
    PYTORCH_MODEL_PATH: str = Field("/app/models", env="PYTORCH_MODEL_PATH")
    
    # Monitoring
    PROMETHEUS_ENABLED: bool = Field(True, env="PROMETHEUS_ENABLED")
    PROMETHEUS_PORT: int = Field(8001, env="PROMETHEUS_PORT")
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")
    
    # WebSocket
    WEBSOCKET_MAX_CONNECTIONS: int = Field(1000, env="WEBSOCKET_MAX_CONNECTIONS")
    WEBSOCKET_HEARTBEAT_INTERVAL: int = Field(30, env="WEBSOCKET_HEARTBEAT_INTERVAL")
    
    @validator('CORS_ORIGINS')
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    @validator('LOG_LEVEL')
    def validate_log_level(cls, v):
        allowed_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in allowed_levels:
            raise ValueError(f'LOG_LEVEL must be one of: {allowed_levels}')
        return v.upper()
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()
```

### Environment-Specific Configurations

```bash
# environments/development.env
DEBUG=true
LOG_LEVEL=DEBUG
DATABASE_URL=postgresql://gls_user:gls_password@localhost:5432/gls_dev
REDIS_URL=redis://localhost:6379/0
CORS_ORIGINS=["http://localhost:3000","http://localhost:8080"]
RATE_LIMIT_REQUESTS=10000
PROMETHEUS_ENABLED=false
```

```bash
# environments/staging.env
DEBUG=false
LOG_LEVEL=INFO
DATABASE_URL=postgresql://gls_user:gls_password@postgres-staging:5432/gls_staging
REDIS_URL=redis://redis-staging:6379/0
CORS_ORIGINS=["https://staging.datacraft.co.ke"]
RATE_LIMIT_REQUESTS=5000
PROMETHEUS_ENABLED=true
```

```bash
# environments/production.env
DEBUG=false
LOG_LEVEL=WARNING
DATABASE_URL=postgresql://gls_user:gls_password@postgres-prod:5432/gls_prod
REDIS_URL=redis://redis-prod:6379/0
CORS_ORIGINS=["https://api.datacraft.co.ke","https://app.datacraft.co.ke"]
RATE_LIMIT_REQUESTS=1000
PROMETHEUS_ENABLED=true
MAX_CONCURRENT_REQUESTS=2000
REQUEST_TIMEOUT_SECONDS=60
```

---

## 🔒 Security Configuration

### JWT Authentication

```python
# auth.py
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timedelta
from typing import Optional

security = HTTPBearer()

class JWTManager:
    """JWT token management."""
    
    def __init__(self, secret: str, algorithm: str = "HS256"):
        self.secret = secret
        self.algorithm = algorithm
    
    def create_token(self, user_id: str, tenant_id: str, scopes: List[str]) -> str:
        """Create JWT token."""
        payload = {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "scopes": scopes,
            "exp": datetime.utcnow() + timedelta(hours=24),
            "iat": datetime.utcnow(),
            "iss": "gls-api"
        }
        
        return jwt.encode(payload, self.secret, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> dict:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")

jwt_manager = JWTManager(settings.JWT_SECRET, settings.JWT_ALGORITHM)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Extract current user from JWT token."""
    return jwt_manager.verify_token(credentials.credentials)

async def require_scope(required_scope: str):
    """Dependency to require specific scope."""
    def scope_checker(user: dict = Depends(get_current_user)):
        if required_scope not in user.get("scopes", []):
            raise HTTPException(
                status_code=403, 
                detail=f"Insufficient permissions. Required scope: {required_scope}"
            )
        return user
    return scope_checker
```

### Rate Limiting

```python
# rate_limiter.py
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
import asyncio
import time
from collections import defaultdict
from typing import Dict, Tuple
import redis.asyncio as redis

class RateLimiter:
    """Redis-based rate limiter."""
    
    def __init__(self, redis_url: str):
        self.redis_client = redis.from_url(redis_url)
        self.local_cache: Dict[str, Tuple[int, float]] = {}
        self.cleanup_interval = 300  # 5 minutes
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_local_cache())
    
    async def check_rate_limit(
        self, 
        key: str, 
        limit: int, 
        window: int
    ) -> Tuple[bool, Dict[str, int]]:
        """Check if request is within rate limit."""
        
        current_time = int(time.time())
        window_start = current_time - window
        
        # Use Redis sliding window
        pipe = self.redis_client.pipeline()
        
        # Remove old entries
        pipe.zremrangebyscore(key, 0, window_start)
        
        # Count current requests
        pipe.zcard(key)
        
        # Add current request
        pipe.zadd(key, {str(current_time): current_time})
        
        # Set expiry
        pipe.expire(key, window)
        
        results = await pipe.execute()
        current_requests = results[1]
        
        # Check limit
        allowed = current_requests < limit
        
        # Rate limit headers
        headers = {
            "X-RateLimit-Limit": limit,
            "X-RateLimit-Remaining": max(0, limit - current_requests - 1),
            "X-RateLimit-Reset": current_time + window
        }
        
        return allowed, headers
    
    async def rate_limit_middleware(self, request: Request, call_next):
        """Rate limiting middleware."""
        
        # Get client identifier
        client_ip = request.client.host
        api_key = request.headers.get("X-API-Key")
        identifier = api_key if api_key else client_ip
        
        # Determine rate limits based on endpoint
        endpoint = request.url.path
        if endpoint.startswith("/api/v1/geographical-location/streaming"):
            limit, window = 10, 60  # 10 connections per minute
        elif endpoint.startswith("/api/v1/geographical-location/batch"):
            limit, window = 50, 3600  # 50 batch requests per hour
        else:
            limit, window = 1000, 3600  # 1000 requests per hour
        
        # Check rate limit
        allowed, headers = await self.check_rate_limit(
            f"rate_limit:{identifier}:{endpoint}", 
            limit, 
            window
        )
        
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Limit: {limit} per {window} seconds"
                },
                headers=headers
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        for header, value in headers.items():
            response.headers[header] = str(value)
        
        return response
    
    async def _cleanup_local_cache(self):
        """Cleanup expired local cache entries."""
        while True:
            current_time = time.time()
            expired_keys = [
                key for key, (_, timestamp) in self.local_cache.items()
                if current_time - timestamp > 300  # 5 minutes
            ]
            
            for key in expired_keys:
                del self.local_cache[key]
            
            await asyncio.sleep(self.cleanup_interval)
```

### Input Validation & Sanitization

```python
# validators.py
from pydantic import validator, BaseModel
from typing import Any, Dict
import re
import html

class SecurityValidators:
    """Security validation utilities."""
    
    @staticmethod
    def sanitize_string(value: str) -> str:
        """Sanitize string input."""
        if not isinstance(value, str):
            return value
        
        # HTML escape
        value = html.escape(value)
        
        # Remove potential SQL injection patterns
        dangerous_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
            r"(--|/\*|\*/|;|'|\")",
            r"(\b(script|javascript|vbscript|onload|onerror)\b)"
        ]
        
        for pattern in dangerous_patterns:
            value = re.sub(pattern, "", value, flags=re.IGNORECASE)
        
        return value.strip()
    
    @staticmethod
    def validate_coordinate(lat: float, lng: float) -> bool:
        """Validate coordinate bounds."""
        return -90 <= lat <= 90 and -180 <= lng <= 180
    
    @staticmethod
    def validate_h3_index(h3_index: str) -> bool:
        """Validate H3 index format."""
        if not h3_index or len(h3_index) != 15:
            return False
        
        # H3 index should start with '8' and contain only hex characters
        pattern = r"^8[0-9a-fA-F]{14}$"
        return bool(re.match(pattern, h3_index))
    
    @staticmethod
    def sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize metadata dictionary."""
        if not isinstance(metadata, dict):
            return {}
        
        sanitized = {}
        for key, value in metadata.items():
            # Sanitize key
            clean_key = SecurityValidators.sanitize_string(str(key))[:100]
            
            # Sanitize value
            if isinstance(value, str):
                clean_value = SecurityValidators.sanitize_string(value)[:1000]
            elif isinstance(value, (int, float, bool)):
                clean_value = value
            elif isinstance(value, dict):
                clean_value = SecurityValidators.sanitize_metadata(value)
            else:
                clean_value = SecurityValidators.sanitize_string(str(value))[:1000]
            
            sanitized[clean_key] = clean_value
        
        return sanitized

# Apply to models
class SecureGLSCoordinate(BaseModel):
    """Secure coordinate model with validation."""
    
    latitude: float
    longitude: float
    entity_id: str
    metadata: Dict[str, Any] = {}
    
    @validator('latitude', 'longitude')
    def validate_coordinates(cls, v, values, field):
        if field.name == 'latitude':
            if not -90 <= v <= 90:
                raise ValueError('Latitude must be between -90 and 90')
        elif field.name == 'longitude':
            if not -180 <= v <= 180:
                raise ValueError('Longitude must be between -180 and 180')
        return v
    
    @validator('entity_id')
    def sanitize_entity_id(cls, v):
        return SecurityValidators.sanitize_string(v)[:255]
    
    @validator('metadata')
    def sanitize_metadata_field(cls, v):
        return SecurityValidators.sanitize_metadata(v)
```

---

## 🚀 Performance Tuning

### Application Optimization

```python
# performance.py
import asyncio
from functools import wraps, lru_cache
from typing import Dict, Any, Callable
import time
import psutil
import gc

class PerformanceMonitor:
    """Performance monitoring and optimization."""
    
    def __init__(self):
        self.request_times: Dict[str, list] = {}
        self.memory_usage: list = []
        self.cpu_usage: list = []
    
    def timing_decorator(self, endpoint_name: str):
        """Decorator to measure endpoint execution time."""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    execution_time = time.time() - start_time
                    
                    if endpoint_name not in self.request_times:
                        self.request_times[endpoint_name] = []
                    
                    self.request_times[endpoint_name].append(execution_time)
                    
                    # Keep only last 1000 measurements
                    if len(self.request_times[endpoint_name]) > 1000:
                        self.request_times[endpoint_name] = self.request_times[endpoint_name][-1000:]
            
            return wrapper
        return decorator
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        stats = {
            "endpoints": {},
            "system": {
                "memory_usage_mb": psutil.virtual_memory().used / 1024 / 1024,
                "memory_percent": psutil.virtual_memory().percent,
                "cpu_percent": psutil.cpu_percent(),
                "disk_usage_percent": psutil.disk_usage('/').percent
            }
        }
        
        for endpoint, times in self.request_times.items():
            if times:
                stats["endpoints"][endpoint] = {
                    "avg_response_time": sum(times) / len(times),
                    "min_response_time": min(times),
                    "max_response_time": max(times),
                    "total_requests": len(times),
                    "p95_response_time": sorted(times)[int(len(times) * 0.95)]
                }
        
        return stats
    
    async def cleanup_memory(self):
        """Periodic memory cleanup."""
        while True:
            # Force garbage collection
            gc.collect()
            
            # Log memory usage
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 80:
                # Emergency cleanup
                gc.collect()
                
            await asyncio.sleep(300)  # Every 5 minutes

# Caching utilities
class CacheManager:
    """Application-level caching."""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.local_cache = {}
        self.local_cache_size = 1000
    
    @lru_cache(maxsize=1000)
    def cache_h3_calculation(self, lat: float, lng: float) -> str:
        """Cache H3 calculations locally."""
        return generate_h3_indices(lat, lng)
    
    async def cached_geocoding(self, address: str, provider: str) -> Dict[str, Any]:
        """Cache geocoding results."""
        cache_key = f"geocode:{provider}:{hash(address)}"
        
        # Try Redis cache first
        cached_result = await self.redis.get(cache_key)
        if cached_result:
            return json.loads(cached_result)
        
        # Try local cache
        if cache_key in self.local_cache:
            return self.local_cache[cache_key]
        
        # Not cached - would perform actual geocoding here
        result = await self._perform_geocoding(address, provider)
        
        # Cache in Redis (24 hours)
        await self.redis.setex(cache_key, 86400, json.dumps(result))
        
        # Cache locally (limit size)
        if len(self.local_cache) >= self.local_cache_size:
            # Remove oldest entries
            oldest_keys = list(self.local_cache.keys())[:100]
            for key in oldest_keys:
                del self.local_cache[key]
        
        self.local_cache[cache_key] = result
        return result
    
    async def _perform_geocoding(self, address: str, provider: str) -> Dict[str, Any]:
        """Actual geocoding implementation."""
        # Implementation would go here
        pass

# Connection pooling
class OptimizedConnectionPool:
    """Optimized database connection pool."""
    
    def __init__(self):
        self.pools = {}
        self.connection_stats = {}
    
    async def create_optimized_pool(self, database_url: str):
        """Create optimized connection pool."""
        
        # Calculate optimal pool size based on system resources
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Formula: (cpu_count * 2) + effective_spindle_count
        # For SSD, effective_spindle_count ≈ cpu_count
        min_connections = max(5, cpu_count)
        max_connections = min(50, cpu_count * 4)
        
        # Adjust based on available memory
        if memory_gb < 8:
            max_connections = min(max_connections, 20)
        elif memory_gb < 16:
            max_connections = min(max_connections, 30)
        
        pool = await asyncpg.create_pool(
            database_url,
            min_size=min_connections,
            max_size=max_connections,
            command_timeout=60,
            server_settings={
                'application_name': 'gls-api-optimized',
                'shared_preload_libraries': 'pg_stat_statements',
                'track_counts': 'on',
                'track_functions': 'all'
            }
        )
        
        return pool
```

### Database Performance

```sql
-- performance_tuning.sql

-- Analyze and update table statistics
ANALYZE coordinates;
ANALYZE trajectories;
ANALYZE hotspots;
ANALYZE predictions;
ANALYZE anomalies;

-- Create partial indices for common queries
CREATE INDEX CONCURRENTLY idx_coordinates_recent 
ON coordinates(entity_id, timestamp DESC) 
WHERE timestamp > NOW() - INTERVAL '7 days';

CREATE INDEX CONCURRENTLY idx_coordinates_active_entities
ON coordinates(entity_id, primary_h3_index, timestamp DESC)
WHERE timestamp > NOW() - INTERVAL '24 hours';

-- Create functional indices
CREATE INDEX CONCURRENTLY idx_coordinates_date_trunc_day
ON coordinates(entity_id, date_trunc('day', timestamp));

CREATE INDEX CONCURRENTLY idx_coordinates_speed_filter
ON coordinates(entity_id, timestamp DESC)
WHERE speed_kmh IS NOT NULL AND speed_kmh > 0;

-- Optimize common query patterns
CREATE MATERIALIZED VIEW coordinates_hourly_summary AS
SELECT 
    entity_id,
    entity_type,
    date_trunc('hour', timestamp) as hour_bucket,
    primary_h3_index,
    COUNT(*) as point_count,
    AVG(speed_kmh) as avg_speed,
    MAX(speed_kmh) as max_speed,
    ST_Centroid(ST_Collect(geom)) as centroid_geom
FROM coordinates
WHERE timestamp > NOW() - INTERVAL '7 days'
GROUP BY entity_id, entity_type, date_trunc('hour', timestamp), primary_h3_index;

CREATE INDEX idx_hourly_summary_entity_hour ON coordinates_hourly_summary(entity_id, hour_bucket DESC);
CREATE INDEX idx_hourly_summary_h3 ON coordinates_hourly_summary(primary_h3_index);
CREATE INDEX idx_hourly_summary_geom ON coordinates_hourly_summary USING GIST(centroid_geom);

-- Automatic statistics collection
CREATE EXTENSION IF NOT EXISTS pg_cron;

-- Update statistics daily
SELECT cron.schedule('analyze-tables', '0 2 * * *', 'ANALYZE;');

-- Refresh materialized views
SELECT cron.schedule('refresh-hourly-summary', '0 */1 * * *', 'REFRESH MATERIALIZED VIEW CONCURRENTLY coordinates_hourly_summary;');

-- Connection optimization
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '12GB';
ALTER SYSTEM SET maintenance_work_mem = '1GB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '64MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;  -- For SSD
ALTER SYSTEM SET effective_io_concurrency = 200;  -- For SSD

-- PostGIS optimization
ALTER SYSTEM SET max_locks_per_transaction = 256;
SELECT pg_reload_conf();
```

---

## 📊 Monitoring & Logging

### Prometheus Metrics

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import psutil

# Define metrics
REQUEST_COUNT = Counter(
    'gls_requests_total',
    'Total GLS requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'gls_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

ACTIVE_CONNECTIONS = Gauge(
    'gls_active_connections',
    'Active database connections'
)

CACHE_HITS = Counter(
    'gls_cache_hits_total',
    'Cache hits',
    ['cache_type']
)

CACHE_MISSES = Counter(
    'gls_cache_misses_total', 
    'Cache misses',
    ['cache_type']
)

GEOCODING_REQUESTS = Counter(
    'gls_geocoding_requests_total',
    'Geocoding requests',
    ['provider', 'status']
)

H3_CALCULATIONS = Counter(
    'gls_h3_calculations_total',
    'H3 index calculations',
    ['resolution']
)

TRAJECTORY_ANALYSES = Counter(
    'gls_trajectory_analyses_total',
    'Trajectory analyses performed',
    ['pattern_detected']
)

HOTSPOT_DETECTIONS = Counter(
    'gls_hotspot_detections_total',
    'Hotspot detections performed',
    ['algorithm']
)

MEMORY_USAGE = Gauge(
    'gls_memory_usage_bytes',
    'Memory usage in bytes'
)

CPU_USAGE = Gauge(
    'gls_cpu_usage_percent',
    'CPU usage percentage'
)

WEBSOCKET_CONNECTIONS = Gauge(
    'gls_websocket_connections_active',
    'Active WebSocket connections'
)

class MetricsCollector:
    """Collect and expose metrics."""
    
    def __init__(self, port: int = 8001):
        self.port = port
        start_http_server(port)
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record request metrics."""
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
        REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_cache_operation(self, cache_type: str, hit: bool):
        """Record cache operation."""
        if hit:
            CACHE_HITS.labels(cache_type=cache_type).inc()
        else:
            CACHE_MISSES.labels(cache_type=cache_type).inc()
    
    def update_system_metrics(self):
        """Update system resource metrics."""
        MEMORY_USAGE.set(psutil.virtual_memory().used)
        CPU_USAGE.set(psutil.cpu_percent())
    
    async def collect_metrics_periodically(self):
        """Collect metrics in background."""
        while True:
            self.update_system_metrics()
            await asyncio.sleep(30)

# Middleware for automatic metrics collection
async def metrics_middleware(request: Request, call_next):
    """Metrics collection middleware."""
    start_time = time.time()
    method = request.method
    endpoint = request.url.path
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    status_code = response.status_code
    
    # Record metrics
    metrics_collector.record_request(method, endpoint, status_code, duration)
    
    return response

metrics_collector = MetricsCollector()
```

### Structured Logging

```python
# logging_config.py
import logging
import json
from datetime import datetime
from typing import Dict, Any
import traceback

class StructuredLogger:
    """Structured JSON logger for cloud environments."""
    
    def __init__(self, name: str, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level))
        
        # Create structured formatter
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(handler)
    
    def info(self, message: str, **kwargs):
        """Log info message with structured data."""
        self.logger.info(message, extra={"structured_data": kwargs})
    
    def warning(self, message: str, **kwargs):
        """Log warning message with structured data."""
        self.logger.warning(message, extra={"structured_data": kwargs})
    
    def error(self, message: str, error: Exception = None, **kwargs):
        """Log error message with structured data."""
        if error:
            kwargs["error_type"] = type(error).__name__
            kwargs["error_message"] = str(error)
            kwargs["traceback"] = traceback.format_exc()
        
        self.logger.error(message, extra={"structured_data": kwargs})
    
    def debug(self, message: str, **kwargs):
        """Log debug message with structured data."""
        self.logger.debug(message, extra={"structured_data": kwargs})

class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add structured data if present
        if hasattr(record, 'structured_data'):
            log_entry.update(record.structured_data)
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)

# Usage
logger = StructuredLogger("gls-api", settings.LOG_LEVEL)

# Request logging middleware
async def logging_middleware(request: Request, call_next):
    """Request logging middleware."""
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]
    
    # Log request
    logger.info(
        "Request started",
        request_id=request_id,
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host,
        user_agent=request.headers.get("user-agent", "")
    )
    
    response = await call_next(request)
    
    # Log response
    duration = time.time() - start_time
    logger.info(
        "Request completed",
        request_id=request_id,
        status_code=response.status_code,
        duration_seconds=duration
    )
    
    return response
```

### Health Checks

```python
# health.py
import asyncio
import psutil
from typing import Dict, Any
from datetime import datetime, timedelta

class HealthChecker:
    """Comprehensive health checking system."""
    
    def __init__(self, db_pool, redis_client):
        self.db_pool = db_pool
        self.redis_client = redis_client
        self.last_check = {}
        self.check_interval = 30  # seconds
    
    async def comprehensive_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {}
        }
        
        # Database health
        db_health = await self._check_database()
        health_status["checks"]["database"] = db_health
        
        # Redis health
        redis_health = await self._check_redis()
        health_status["checks"]["redis"] = redis_health
        
        # System resources
        system_health = await self._check_system_resources()
        health_status["checks"]["system"] = system_health
        
        # External APIs
        api_health = await self._check_external_apis()
        health_status["checks"]["external_apis"] = api_health
        
        # Overall status
        if any(check["status"] != "healthy" for check in health_status["checks"].values()):
            health_status["status"] = "unhealthy"
        
        return health_status
    
    async def _check_database(self) -> Dict[str, Any]:
        """Check database connectivity and performance."""
        try:
            start_time = time.time()
            
            # Test connection
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                
            # Test spatial functions
            async with self.db_pool.acquire() as conn:
                spatial_test = await conn.fetchval(
                    "SELECT ST_Distance(ST_Point(0, 0), ST_Point(1, 1))"
                )
            
            response_time = time.time() - start_time
            
            # Check pool status
            pool_stats = {
                "total_connections": self.db_pool.get_size(),
                "available_connections": self.db_pool.get_idle_size(),
                "max_connections": self.db_pool.get_max_size()
            }
            
            return {
                "status": "healthy",
                "response_time_ms": round(response_time * 1000, 2),
                "pool_stats": pool_stats,
                "spatial_functions": "working" if spatial_test else "error"
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "response_time_ms": None
            }
    
    async def _check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity and performance."""
        try:
            start_time = time.time()
            
            # Test basic operations
            await self.redis_client.set("health_check", "ok", ex=60)
            result = await self.redis_client.get("health_check")
            await self.redis_client.delete("health_check")
            
            response_time = time.time() - start_time
            
            # Get Redis info
            info = await self.redis_client.info()
            
            return {
                "status": "healthy",
                "response_time_ms": round(response_time * 1000, 2),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_mb": round(info.get("used_memory", 0) / 1024 / 1024, 2),
                "test_result": result.decode() if result else None
            }
            
        except Exception as e:
            return {
                "status": "unhealthy", 
                "error": str(e),
                "response_time_ms": None
            }
    
    async def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource utilization."""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Determine health based on thresholds
            status = "healthy"
            warnings = []
            
            if memory.percent > 90:
                status = "unhealthy"
                warnings.append("High memory usage")
            elif memory.percent > 80:
                warnings.append("Elevated memory usage")
            
            if disk.percent > 95:
                status = "unhealthy"
                warnings.append("Disk space critical")
            elif disk.percent > 85:
                warnings.append("Disk space low")
            
            if cpu_percent > 95:
                status = "unhealthy"
                warnings.append("High CPU usage")
            elif cpu_percent > 80:
                warnings.append("Elevated CPU usage")
            
            return {
                "status": status,
                "warnings": warnings,
                "memory": {
                    "used_percent": memory.percent,
                    "available_gb": round(memory.available / 1024**3, 2)
                },
                "disk": {
                    "used_percent": disk.percent,
                    "free_gb": round(disk.free / 1024**3, 2)
                },
                "cpu": {
                    "usage_percent": cpu_percent,
                    "core_count": psutil.cpu_count()
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def _check_external_apis(self) -> Dict[str, Any]:
        """Check external API connectivity."""
        api_checks = {}
        
        # Test geocoding APIs
        if settings.GOOGLE_MAPS_API_KEY:
            google_health = await self._test_geocoding_api("google")
            api_checks["google_maps"] = google_health
        
        if settings.MAPBOX_API_KEY:
            mapbox_health = await self._test_geocoding_api("mapbox")
            api_checks["mapbox"] = mapbox_health
        
        overall_status = "healthy"
        if any(check["status"] != "healthy" for check in api_checks.values()):
            overall_status = "degraded"  # External APIs are not critical
        
        return {
            "status": overall_status,
            "apis": api_checks
        }
    
    async def _test_geocoding_api(self, provider: str) -> Dict[str, Any]:
        """Test geocoding API health."""
        try:
            start_time = time.time()
            
            # Simple test geocode (this would use actual provider)
            # For demo, we'll simulate the test
            await asyncio.sleep(0.1)  # Simulate API call
            
            response_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "response_time_ms": round(response_time * 1000, 2),
                "provider": provider
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "provider": provider
            }

# Usage
health_checker = HealthChecker(db.pool, redis_client)

@router.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check endpoint."""
    return await health_checker.comprehensive_health_check()
```

---

## 💾 Backup & Recovery

### Database Backup Strategy

```bash
#!/bin/bash
# backup_database.sh

set -e

# Configuration
DB_HOST=${DB_HOST:-"localhost"}
DB_PORT=${DB_PORT:-"5432"}
DB_NAME=${DB_NAME:-"gls_db"}
DB_USER=${DB_USER:-"gls_user"}
BACKUP_DIR=${BACKUP_DIR:-"/backups"}
RETENTION_DAYS=${RETENTION_DAYS:-"30"}
S3_BUCKET=${S3_BUCKET:-"gls-backups"}

# Create backup directory
mkdir -p $BACKUP_DIR

# Generate backup filename
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="gls_backup_${TIMESTAMP}.sql"
BACKUP_PATH="${BACKUP_DIR}/${BACKUP_FILE}"

echo "Starting backup of database ${DB_NAME}..."

# Create database backup
pg_dump \
    --host=$DB_HOST \
    --port=$DB_PORT \
    --username=$DB_USER \
    --dbname=$DB_NAME \
    --verbose \
    --format=custom \
    --no-owner \
    --no-privileges \
    --file=$BACKUP_PATH

# Compress backup
gzip $BACKUP_PATH
BACKUP_PATH="${BACKUP_PATH}.gz"

echo "Backup created: $BACKUP_PATH"

# Upload to S3 if configured
if [ ! -z "$S3_BUCKET" ]; then
    echo "Uploading backup to S3..."
    aws s3 cp $BACKUP_PATH s3://$S3_BUCKET/database/$(basename $BACKUP_PATH)
    echo "Backup uploaded to S3"
fi

# Clean up old backups
echo "Cleaning up old backups..."
find $BACKUP_DIR -name "gls_backup_*.sql.gz" -mtime +$RETENTION_DAYS -delete

echo "Backup completed successfully"
```

### Disaster Recovery Procedures

```bash
#!/bin/bash
# restore_database.sh

set -e

# Configuration
DB_HOST=${DB_HOST:-"localhost"}
DB_PORT=${DB_PORT:-"5432"}
DB_NAME=${DB_NAME:-"gls_db"}
DB_USER=${DB_USER:-"gls_user"}
BACKUP_FILE=$1

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

echo "Starting database restore from $BACKUP_FILE..."

# Check if backup file exists
if [ ! -f "$BACKUP_FILE" ]; then
    echo "Backup file not found: $BACKUP_FILE"
    exit 1
fi

# Create new database (drop existing if needed)
echo "Preparing database for restore..."
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -c "DROP DATABASE IF EXISTS ${DB_NAME}_restore;"
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -c "CREATE DATABASE ${DB_NAME}_restore;"

# Restore database
echo "Restoring database..."
if [[ $BACKUP_FILE == *.gz ]]; then
    gunzip -c $BACKUP_FILE | pg_restore \
        --host=$DB_HOST \
        --port=$DB_PORT \
        --username=$DB_USER \
        --dbname=${DB_NAME}_restore \
        --verbose \
        --no-owner \
        --no-privileges
else
    pg_restore \
        --host=$DB_HOST \
        --port=$DB_PORT \
        --username=$DB_USER \
        --dbname=${DB_NAME}_restore \
        --verbose \
        --no-owner \
        --no-privileges \
        $BACKUP_FILE
fi

echo "Database restored to ${DB_NAME}_restore"
echo "To switch to restored database:"
echo "1. Stop application"
echo "2. Rename current database: ALTER DATABASE $DB_NAME RENAME TO ${DB_NAME}_old;"
echo "3. Rename restored database: ALTER DATABASE ${DB_NAME}_restore RENAME TO $DB_NAME;"
echo "4. Start application"
```

### Redis Backup

```bash
#!/bin/bash
# backup_redis.sh

set -e

REDIS_HOST=${REDIS_HOST:-"localhost"}
REDIS_PORT=${REDIS_PORT:-"6379"}
BACKUP_DIR=${BACKUP_DIR:-"/backups/redis"}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p $BACKUP_DIR

echo "Creating Redis backup..."

# Create Redis backup
redis-cli -h $REDIS_HOST -p $REDIS_PORT --rdb "${BACKUP_DIR}/redis_backup_${TIMESTAMP}.rdb"

# Compress backup
gzip "${BACKUP_DIR}/redis_backup_${TIMESTAMP}.rdb"

echo "Redis backup completed: ${BACKUP_DIR}/redis_backup_${TIMESTAMP}.rdb.gz"
```

---

## 🔧 Troubleshooting

### Common Issues & Solutions

#### 1. High Memory Usage

```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head -10

# Check Python memory usage
python3 -c "
import psutil
process = psutil.Process()
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB')
print(f'Memory percent: {process.memory_percent():.2f}%')
"

# Solutions:
# 1. Reduce batch sizes
# 2. Implement memory limits
# 3. Add garbage collection
# 4. Optimize queries
```

#### 2. Database Connection Pool Exhaustion

```python
# Monitor connection pool
async def check_pool_status():
    pool_size = db.pool.get_size()
    idle_connections = db.pool.get_idle_size() 
    max_connections = db.pool.get_max_size()
    
    print(f"Pool size: {pool_size}/{max_connections}")
    print(f"Idle connections: {idle_connections}")
    print(f"Active connections: {pool_size - idle_connections}")
    
    if idle_connections == 0:
        print("WARNING: Connection pool exhausted!")

# Solutions:
# 1. Increase pool size
# 2. Reduce connection timeout
# 3. Fix connection leaks
# 4. Optimize query performance
```

#### 3. Redis Memory Issues

```bash
# Check Redis memory usage
redis-cli info memory

# Check for large keys
redis-cli --bigkeys

# Solutions:
# 1. Set maxmemory policy
# 2. Use appropriate data structures
# 3. Set TTL on keys
# 4. Monitor key sizes
```

#### 4. Slow Query Performance

```sql
-- Enable query logging
ALTER SYSTEM SET log_statement = 'all';
ALTER SYSTEM SET log_min_duration_statement = 1000; -- Log queries > 1 second

-- Check slow queries
SELECT query, mean_time, calls, total_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;

-- Check missing indices
SELECT schemaname, tablename, attname, n_distinct, correlation
FROM pg_stats
WHERE schemaname = 'public'
AND n_distinct > 100
AND correlation < 0.1;
```

### Diagnostic Tools

```python
# diagnostics.py
import psutil
import asyncio
import time
from typing import Dict, Any

class DiagnosticTool:
    """System diagnostic utilities."""
    
    async def full_system_diagnostic(self) -> Dict[str, Any]:
        """Run comprehensive system diagnostic."""
        
        diagnostics = {
            "timestamp": time.time(),
            "system_info": self._get_system_info(),
            "process_info": self._get_process_info(),
            "network_info": await self._get_network_info(),
            "database_info": await self._get_database_info(),
            "redis_info": await self._get_redis_info(),
            "performance_metrics": await self._get_performance_metrics()
        }
        
        return diagnostics
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "cpu_count": psutil.cpu_count(logical=False),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "memory_total_gb": round(psutil.virtual_memory().total / 1024**3, 2),
            "disk_total_gb": round(psutil.disk_usage('/').total / 1024**3, 2),
            "boot_time": psutil.boot_time(),
            "platform": psutil.os.name
        }
    
    def _get_process_info(self) -> Dict[str, Any]:
        """Get current process information."""
        process = psutil.Process()
        
        return {
            "pid": process.pid,
            "memory_mb": round(process.memory_info().rss / 1024 / 1024, 2),
            "memory_percent": round(process.memory_percent(), 2),
            "cpu_percent": round(process.cpu_percent(), 2),
            "num_threads": process.num_threads(),
            "open_files": len(process.open_files()),
            "connections": len(process.connections()),
            "create_time": process.create_time()
        }
    
    async def _get_network_info(self) -> Dict[str, Any]:
        """Get network information."""
        stats = psutil.net_io_counters()
        
        return {
            "bytes_sent": stats.bytes_sent,
            "bytes_recv": stats.bytes_recv,
            "packets_sent": stats.packets_sent,
            "packets_recv": stats.packets_recv,
            "errors_in": stats.errin,
            "errors_out": stats.errout,
            "drops_in": stats.dropin,
            "drops_out": stats.dropout
        }
    
    async def _get_database_info(self) -> Dict[str, Any]:
        """Get database diagnostic information."""
        try:
            async with db.pool.acquire() as conn:
                # Connection info
                conn_info = {
                    "server_version": conn.get_server_version(),
                    "pool_size": db.pool.get_size(),
                    "pool_idle": db.pool.get_idle_size(),
                    "pool_max": db.pool.get_max_size()
                }
                
                # Database stats
                db_stats = await conn.fetchrow("""
                    SELECT 
                        pg_database_size(current_database()) as db_size,
                        (SELECT count(*) FROM coordinates) as coordinates_count,
                        (SELECT count(*) FROM trajectories) as trajectories_count,
                        (SELECT count(*) FROM hotspots) as hotspots_count
                """)
                
                conn_info.update(dict(db_stats))
                return conn_info
                
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_redis_info(self) -> Dict[str, Any]:
        """Get Redis diagnostic information."""
        try:
            info = await redis_client.info()
            
            return {
                "redis_version": info.get("redis_version"),
                "connected_clients": info.get("connected_clients"),
                "used_memory": info.get("used_memory"),
                "used_memory_peak": info.get("used_memory_peak"),
                "keyspace_hits": info.get("keyspace_hits"),
                "keyspace_misses": info.get("keyspace_misses"),
                "total_commands_processed": info.get("total_commands_processed")
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            "cpu_times": psutil.cpu_times()._asdict(),
            "memory_stats": psutil.virtual_memory()._asdict(),
            "disk_io": psutil.disk_io_counters()._asdict(),
            "network_io": psutil.net_io_counters()._asdict()
        }

# Usage
diagnostic_tool = DiagnosticTool()

@router.get("/diagnostics")
async def get_diagnostics():
    """Get system diagnostics."""
    return await diagnostic_tool.full_system_diagnostic()
```

---

## ✅ Production Checklist

### Pre-Deployment Checklist

```markdown
## Infrastructure
- [ ] Hardware resources meet requirements
- [ ] Network connectivity tested
- [ ] SSL certificates installed and valid
- [ ] Load balancer configured
- [ ] Database replication set up
- [ ] Backup systems configured

## Security
- [ ] JWT secrets configured
- [ ] API keys secured
- [ ] Database credentials secured
- [ ] CORS origins configured
- [ ] Rate limiting enabled
- [ ] Input validation implemented
- [ ] SQL injection protection verified

## Configuration
- [ ] Environment variables set
- [ ] Database migrations applied
- [ ] Redis configuration optimized
- [ ] Logging level set appropriately
- [ ] Monitoring enabled
- [ ] Health checks configured

## Performance
- [ ] Connection pools optimized
- [ ] Caching strategy implemented
- [ ] Database indices created
- [ ] Query performance tested
- [ ] Load testing completed
- [ ] Memory usage validated

## Monitoring
- [ ] Prometheus metrics collecting
- [ ] Grafana dashboards created
- [ ] Log aggregation configured
- [ ] Alert rules defined
- [ ] Error tracking set up
- [ ] Health checks automated

## Testing
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] Load tests completed
- [ ] Security tests passed
- [ ] API documentation updated
- [ ] Disaster recovery tested
```

### Post-Deployment Validation

```python
# post_deployment_validation.py
import asyncio
import aiohttp
import time
from typing import Dict, Any, List

class DeploymentValidator:
    """Validate deployment after going live."""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    async def run_validation_suite(self) -> Dict[str, Any]:
        """Run complete validation suite."""
        
        results = {
            "timestamp": time.time(),
            "overall_status": "unknown",
            "tests": {}
        }
        
        tests = [
            ("health_check", self._test_health_endpoint),
            ("basic_geocoding", self._test_geocoding),
            ("fuzzy_search", self._test_fuzzy_search),
            ("trajectory_analysis", self._test_trajectory_analysis),
            ("websocket_connection", self._test_websocket),
            ("performance", self._test_performance),
            ("error_handling", self._test_error_handling)
        ]
        
        passed_tests = 0
        
        async with aiohttp.ClientSession() as session:
            for test_name, test_func in tests:
                try:
                    test_result = await test_func(session)
                    results["tests"][test_name] = test_result
                    
                    if test_result.get("status") == "passed":
                        passed_tests += 1
                        
                except Exception as e:
                    results["tests"][test_name] = {
                        "status": "failed",
                        "error": str(e)
                    }
        
        # Determine overall status
        if passed_tests == len(tests):
            results["overall_status"] = "passed"
        elif passed_tests >= len(tests) * 0.8:  # 80% pass rate
            results["overall_status"] = "warning"
        else:
            results["overall_status"] = "failed"
        
        results["pass_rate"] = passed_tests / len(tests)
        
        return results
    
    async def _test_health_endpoint(self, session) -> Dict[str, Any]:
        """Test health endpoint."""
        start_time = time.time()
        
        async with session.get(f"{self.base_url}/health") as response:
            response_time = time.time() - start_time
            
            if response.status == 200:
                data = await response.json()
                
                return {
                    "status": "passed",
                    "response_time_ms": round(response_time * 1000, 2),
                    "health_status": data.get("status"),
                    "services": data.get("services", {})
                }
            else:
                return {
                    "status": "failed",
                    "response_code": response.status,
                    "response_time_ms": round(response_time * 1000, 2)
                }
    
    async def _test_geocoding(self, session) -> Dict[str, Any]:
        """Test basic geocoding functionality."""
        payload = {
            "address": {
                "street": "350 5th Ave",
                "city": "New York",
                "state": "NY",
                "country": "US"
            }
        }
        
        start_time = time.time()
        
        async with session.post(
            f"{self.base_url}/geocode",
            json=payload,
            headers=self.headers
        ) as response:
            response_time = time.time() - start_time
            
            if response.status == 200:
                data = await response.json()
                
                # Validate response structure
                if (data.get("success") and 
                    "data" in data and 
                    "address" in data["data"]):
                    
                    return {
                        "status": "passed",
                        "response_time_ms": round(response_time * 1000, 2),
                        "geocoded": True
                    }
                else:
                    return {
                        "status": "failed",
                        "error": "Invalid response structure",
                        "response_time_ms": round(response_time * 1000, 2)
                    }
            else:
                return {
                    "status": "failed", 
                    "response_code": response.status,
                    "response_time_ms": round(response_time * 1000, 2)
                }
    
    async def _test_performance(self, session) -> Dict[str, Any]:
        """Test performance under load."""
        concurrent_requests = 10
        total_requests = 50
        
        payload = {
            "query_text": "New York",
            "fuzzy_match_type": "jaro_winkler",
            "confidence_threshold": 0.7
        }
        
        async def single_request():
            start_time = time.time()
            async with session.post(
                f"{self.base_url}/fuzzy-search",
                json=payload,
                headers=self.headers
            ) as response:
                response_time = time.time() - start_time
                return response.status, response_time
        
        # Run concurrent requests
        start_time = time.time()
        
        tasks = []
        for _ in range(total_requests):
            task = asyncio.create_task(single_request())
            tasks.append(task)
            
            # Control concurrency
            if len(tasks) >= concurrent_requests:
                completed_tasks = await asyncio.gather(*tasks)
                tasks = []
        
        # Wait for remaining tasks
        if tasks:
            remaining_tasks = await asyncio.gather(*tasks)
            completed_tasks.extend(remaining_tasks)
        
        total_time = time.time() - start_time
        
        # Analyze results
        successful_requests = sum(1 for status, _ in completed_tasks if status == 200)
        response_times = [rt for _, rt in completed_tasks]
        
        avg_response_time = sum(response_times) / len(response_times)
        p95_response_time = sorted(response_times)[int(len(response_times) * 0.95)]
        
        return {
            "status": "passed" if successful_requests >= total_requests * 0.95 else "failed",
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "success_rate": successful_requests / total_requests,
            "avg_response_time_ms": round(avg_response_time * 1000, 2),
            "p95_response_time_ms": round(p95_response_time * 1000, 2),
            "requests_per_second": round(total_requests / total_time, 2)
        }

# Usage
async def validate_deployment():
    validator = DeploymentValidator(
        base_url="https://api.datacraft.co.ke/api/v1/geographical-location",
        api_key="your_api_key"
    )
    
    results = await validator.run_validation_suite()
    
    print(f"Deployment validation: {results['overall_status']}")
    print(f"Pass rate: {results['pass_rate']:.2%}")
    
    for test_name, test_result in results["tests"].items():
        status = test_result.get("status", "unknown")
        print(f"  {test_name}: {status}")
    
    return results
```

---

*© 2025 Datacraft. All rights reserved. This deployment guide is part of the APG Platform documentation.*