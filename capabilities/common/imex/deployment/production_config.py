"""
APG Import/Export (IMEX) Production Deployment Configuration

Purpose: Production-grade configuration management for enterprise deployment
         with comprehensive environment handling and security settings.
Dependencies: pydantic, python-dotenv, cryptography
Usage Context: Production deployment configuration for IMEX capability

This module provides:
- Production environment configuration management
- Database connection pooling and optimization settings
- Security configuration for enterprise environments
- Performance tuning parameters
- Monitoring and logging configuration
- Docker and Kubernetes deployment settings
"""

import os
import secrets
from datetime import timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict, validator
try:
    from cryptography.fernet import Fernet
except ImportError:
    from security import Fernet

class DeploymentEnvironment(str, Enum):
    """Deployment environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class DatabaseConfig(BaseModel):
    """Production database configuration"""
    host: str = Field(..., description="Database host")
    port: int = Field(5432, description="Database port")
    database: str = Field(..., description="Database name")
    user: str = Field(..., description="Database user")
    password: str = Field(..., description="Database password")
    ssl_mode: str = Field("require", description="SSL mode")
    pool_size: int = Field(20, description="Connection pool size")
    max_overflow: int = Field(30, description="Max pool overflow")
    pool_timeout: int = Field(30, description="Pool timeout seconds")
    pool_recycle: int = Field(3600, description="Pool recycle seconds")
    statement_timeout: int = Field(30000, description="Statement timeout ms")

    model_config = ConfigDict(extra='forbid')

class SecurityConfig(BaseModel):
    """Production security configuration"""
    secret_key: str = Field(..., description="Application secret key")
    jwt_secret_key: str = Field(..., description="JWT secret key")
    encryption_key: str = Field(..., description="Data encryption key")
    password_salt: str = Field(..., description="Password salt")
    api_key_prefix: str = Field("apg_", description="API key prefix")
    session_timeout: int = Field(3600, description="Session timeout seconds")
    max_login_attempts: int = Field(5, description="Max failed login attempts")
    lockout_duration: int = Field(900, description="Account lockout duration seconds")
    require_mfa: bool = Field(True, description="Require multi-factor auth")
    allowed_origins: List[str] = Field(default_factory=list, description="CORS allowed origins")
    rate_limit_per_hour: int = Field(1000, description="API rate limit per hour")
    audit_retention_days: int = Field(365, description="Audit log retention days")

    model_config = ConfigDict(extra='forbid')

class RedisConfig(BaseModel):
    """Redis configuration for caching and sessions"""
    host: str = Field("localhost", description="Redis host")
    port: int = Field(6379, description="Redis port")
    database: int = Field(0, description="Redis database number")
    password: Optional[str] = Field(None, description="Redis password")
    ssl: bool = Field(False, description="Use SSL")
    pool_size: int = Field(10, description="Connection pool size")
    timeout: int = Field(5, description="Connection timeout seconds")

    model_config = ConfigDict(extra='forbid')

class MonitoringConfig(BaseModel):
    """Monitoring and observability configuration"""
    enable_metrics: bool = Field(True, description="Enable metrics collection")
    metrics_port: int = Field(9090, description="Metrics server port")
    log_level: str = Field("INFO", description="Logging level")
    log_format: str = Field("json", description="Log format (json|text)")
    structured_logging: bool = Field(True, description="Enable structured logging")
    trace_sampling_rate: float = Field(0.1, description="Distributed tracing sample rate")
    health_check_interval: int = Field(30, description="Health check interval seconds")
    performance_monitoring: bool = Field(True, description="Enable performance monitoring")
    alert_webhook_url: Optional[str] = Field(None, description="Alert webhook URL")

    model_config = ConfigDict(extra='forbid')

class AIConfig(BaseModel):
    """AI and ML service configuration"""
    ollama_host: str = Field("localhost", description="Ollama server host")
    ollama_port: int = Field(11434, description="Ollama server port")
    default_model: str = Field("llama3.1:8b", description="Default LLM model")
    fallback_models: List[str] = Field(default_factory=lambda: ["llama3:8b", "mistral:7b"], description="Fallback models")
    max_tokens: int = Field(2048, description="Maximum tokens per request")
    temperature: float = Field(0.1, description="Model temperature")
    timeout_seconds: int = Field(30, description="Request timeout")
    cache_enabled: bool = Field(True, description="Enable response caching")
    cache_ttl_seconds: int = Field(3600, description="Cache TTL")

    model_config = ConfigDict(extra='forbid')

class WorkerConfig(BaseModel):
    """Worker process configuration"""
    worker_processes: int = Field(4, description="Number of worker processes")
    worker_threads: int = Field(2, description="Threads per worker")
    worker_timeout: int = Field(300, description="Worker timeout seconds")
    max_requests: int = Field(1000, description="Max requests per worker")
    max_requests_jitter: int = Field(100, description="Max requests jitter")
    preload_app: bool = Field(True, description="Preload application")

    model_config = ConfigDict(extra='forbid')

class ProductionConfig(BaseModel):
    """Complete production configuration"""
    environment: DeploymentEnvironment = Field(..., description="Deployment environment")
    debug: bool = Field(False, description="Debug mode")
    testing: bool = Field(False, description="Testing mode")

    # Core configurations
    database: DatabaseConfig = Field(..., description="Database configuration")
    security: SecurityConfig = Field(..., description="Security configuration")
    redis: RedisConfig = Field(..., description="Redis configuration")
    monitoring: MonitoringConfig = Field(..., description="Monitoring configuration")
    ai: AIConfig = Field(..., description="AI configuration")
    worker: WorkerConfig = Field(..., description="Worker configuration")

    # Application settings
    app_name: str = Field("APG-IMEX", description="Application name")
    app_version: str = Field("1.0.0", description="Application version")
    api_prefix: str = Field("/api/v1", description="API URL prefix")
    max_content_length: int = Field(100 * 1024 * 1024, description="Max content length bytes")  # 100MB

    # File handling
    upload_folder: str = Field("/opt/apg/uploads", description="Upload folder path")
    temp_folder: str = Field("/tmp/apg", description="Temporary folder path")
    max_file_size: int = Field(1024 * 1024 * 1024, description="Max file size bytes")  # 1GB
    allowed_extensions: List[str] = Field(
        default_factory=lambda: ['.csv', '.xlsx', '.json', '.xml', '.parquet'],
        description="Allowed file extensions"
    )

    # Performance settings
    batch_size: int = Field(1000, description="Default batch size")
    parallel_jobs: int = Field(4, description="Parallel job execution limit")
    memory_limit_mb: int = Field(2048, description="Memory limit per job MB")

    model_config = ConfigDict(extra='forbid')

    @validator('upload_folder', 'temp_folder')
    def validate_paths(cls, v):
        """Ensure paths exist"""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v

def generate_secure_keys() -> Dict[str, str]:
    """Generate secure keys for production deployment"""
    return {
        'secret_key': secrets.token_urlsafe(64),
        'jwt_secret_key': secrets.token_urlsafe(64),
        'encryption_key': Fernet.generate_key().decode(),
        'password_salt': secrets.token_urlsafe(32)
    }

def create_production_config(
    environment: str = "production",
    database_url: Optional[str] = None,
    redis_url: Optional[str] = None
) -> ProductionConfig:
    """
    Create production configuration from environment variables.

    Args:
        environment: Deployment environment
        database_url: Database connection URL
        redis_url: Redis connection URL

    Returns:
        ProductionConfig: Complete production configuration
    """

    # Generate or load secure keys
    keys = generate_secure_keys()
    if os.getenv('APG_SECRET_KEY'):
        keys['secret_key'] = os.getenv('APG_SECRET_KEY')
    if os.getenv('APG_JWT_SECRET'):
        keys['jwt_secret_key'] = os.getenv('APG_JWT_SECRET')
    if os.getenv('APG_ENCRYPTION_KEY'):
        keys['encryption_key'] = os.getenv('APG_ENCRYPTION_KEY')

    # Database configuration
    db_config = DatabaseConfig(
        host=os.getenv('DB_HOST', 'localhost'),
        port=int(os.getenv('DB_PORT', '5432')),
        database=os.getenv('DB_NAME', 'apg_imex'),
        user=os.getenv('DB_USER', 'apg'),
        password=os.getenv('DB_PASSWORD', ''),
        ssl_mode=os.getenv('DB_SSL_MODE', 'require'),
        pool_size=int(os.getenv('DB_POOL_SIZE', '20')),
        max_overflow=int(os.getenv('DB_MAX_OVERFLOW', '30'))
    )

    # Security configuration
    security_config = SecurityConfig(
        secret_key=keys['secret_key'],
        jwt_secret_key=keys['jwt_secret_key'],
        encryption_key=keys['encryption_key'],
        password_salt=keys['password_salt'],
        require_mfa=environment == 'production',
        allowed_origins=os.getenv('CORS_ORIGINS', '').split(',') if os.getenv('CORS_ORIGINS') else [],
        rate_limit_per_hour=int(os.getenv('RATE_LIMIT_PER_HOUR', '1000')),
        audit_retention_days=int(os.getenv('AUDIT_RETENTION_DAYS', '365'))
    )

    # Redis configuration
    redis_config = RedisConfig(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', '6379')),
        database=int(os.getenv('REDIS_DB', '0')),
        password=os.getenv('REDIS_PASSWORD'),
        ssl=os.getenv('REDIS_SSL', 'false').lower() == 'true'
    )

    # Monitoring configuration
    monitoring_config = MonitoringConfig(
        log_level=os.getenv('LOG_LEVEL', 'INFO'),
        metrics_port=int(os.getenv('METRICS_PORT', '9090')),
        alert_webhook_url=os.getenv('ALERT_WEBHOOK_URL'),
        trace_sampling_rate=float(os.getenv('TRACE_SAMPLING_RATE', '0.1'))
    )

    # AI configuration
    ai_config = AIConfig(
        ollama_host=os.getenv('OLLAMA_HOST', 'localhost'),
        ollama_port=int(os.getenv('OLLAMA_PORT', '11434')),
        default_model=os.getenv('OLLAMA_MODEL', 'llama3.1:8b'),
        timeout_seconds=int(os.getenv('AI_TIMEOUT', '30'))
    )

    # Worker configuration
    worker_config = WorkerConfig(
        worker_processes=int(os.getenv('WORKER_PROCESSES', '4')),
        worker_threads=int(os.getenv('WORKER_THREADS', '2')),
        worker_timeout=int(os.getenv('WORKER_TIMEOUT', '300'))
    )

    return ProductionConfig(
        environment=DeploymentEnvironment(environment),
        debug=environment != 'production',
        database=db_config,
        security=security_config,
        redis=redis_config,
        monitoring=monitoring_config,
        ai=ai_config,
        worker=worker_config,
        upload_folder=os.getenv('UPLOAD_FOLDER', '/opt/apg/uploads'),
        temp_folder=os.getenv('TEMP_FOLDER', '/tmp/apg'),
        max_file_size=int(os.getenv('MAX_FILE_SIZE', str(1024 * 1024 * 1024))),  # 1GB
        batch_size=int(os.getenv('BATCH_SIZE', '1000')),
        parallel_jobs=int(os.getenv('PARALLEL_JOBS', '4'))
    )

def create_docker_compose_config(config: ProductionConfig) -> str:
    """
    Generate docker-compose.yml configuration.

    Args:
        config: Production configuration

    Returns:
        str: Docker Compose YAML content
    """

    return f"""version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: {config.database.database}
      POSTGRES_USER: {config.database.user}
      POSTGRES_PASSWORD: {config.database.password}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "{config.database.port}:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U {config.database.user}"]
      interval: 30s
      timeout: 10s
      retries: 5

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass {config.redis.password or 'redis_password'}
    ports:
      - "{config.redis.port}:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 5

  ollama:
    image: ollama/ollama:latest
    ports:
      - "{config.ai.ollama_port}:11434"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - OLLAMA_MODELS={config.ai.default_model}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/version"]
      interval: 30s
      timeout: 10s
      retries: 5

  apg-imex:
    build: .
    environment:
      - APG_ENVIRONMENT={config.environment.value}
      - DB_HOST=postgres
      - DB_PORT={config.database.port}
      - DB_NAME={config.database.database}
      - DB_USER={config.database.user}
      - DB_PASSWORD={config.database.password}
      - REDIS_HOST=redis
      - REDIS_PORT={config.redis.port}
      - REDIS_PASSWORD={config.redis.password or 'redis_password'}
      - OLLAMA_HOST=ollama
      - OLLAMA_PORT={config.ai.ollama_port}
      - LOG_LEVEL={config.monitoring.log_level}
      - WORKER_PROCESSES={config.worker.worker_processes}
    ports:
      - "8000:8000"
      - "{config.monitoring.metrics_port}:{config.monitoring.metrics_port}"
    volumes:
      - {config.upload_folder}:/opt/apg/uploads
      - {config.temp_folder}:/tmp/apg
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      ollama:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 5
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
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources

volumes:
  postgres_data:
  redis_data:
  ollama_data:
  prometheus_data:
  grafana_data:

networks:
  default:
    driver: bridge
"""

def create_kubernetes_deployment(config: ProductionConfig) -> str:
    """
    Generate Kubernetes deployment configuration.

    Args:
        config: Production configuration

    Returns:
        str: Kubernetes YAML content
    """

    return f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: apg-imex
  labels:
    app: apg-imex
    version: {config.app_version}
spec:
  replicas: 3
  selector:
    matchLabels:
      app: apg-imex
  template:
    metadata:
      labels:
        app: apg-imex
    spec:
      containers:
      - name: apg-imex
        image: datacraft/apg-imex:{config.app_version}
        ports:
        - containerPort: 8000
        - containerPort: {config.monitoring.metrics_port}
        env:
        - name: APG_ENVIRONMENT
          value: "{config.environment.value}"
        - name: DB_HOST
          valueFrom:
            secretKeyRef:
              name: apg-secrets
              key: db-host
        - name: DB_USER
          valueFrom:
            secretKeyRef:
              name: apg-secrets
              key: db-user
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: apg-secrets
              key: db-password
        - name: REDIS_HOST
          value: "redis-service"
        - name: OLLAMA_HOST
          value: "ollama-service"
        - name: LOG_LEVEL
          value: "{config.monitoring.log_level}"
        - name: WORKER_PROCESSES
          value: "{config.worker.worker_processes}"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "{config.memory_limit_mb}Mi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: upload-storage
          mountPath: {config.upload_folder}
        - name: temp-storage
          mountPath: {config.temp_folder}
      volumes:
      - name: upload-storage
        persistentVolumeClaim:
          claimName: apg-upload-pvc
      - name: temp-storage
        emptyDir: {{}}

---
apiVersion: v1
kind: Service
metadata:
  name: apg-imex-service
spec:
  selector:
    app: apg-imex
  ports:
  - name: http
    port: 80
    targetPort: 8000
  - name: metrics
    port: {config.monitoring.metrics_port}
    targetPort: {config.monitoring.metrics_port}
  type: ClusterIP

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: apg-imex-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - imex.apg.datacraft.co.ke
    secretName: apg-imex-tls
  rules:
  - host: imex.apg.datacraft.co.ke
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: apg-imex-service
            port:
              number: 80

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: apg-upload-pvc
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd
"""

def create_nginx_config(config: ProductionConfig) -> str:
    """
    Generate Nginx configuration for production deployment.

    Args:
        config: Production configuration

    Returns:
        str: Nginx configuration content
    """

    return f"""upstream apg_imex {{
    server 127.0.0.1:8000;
    # Add more servers for load balancing
    # server 127.0.0.1:8001;
    # server 127.0.0.1:8002;
}}

server {{
    listen 80;
    server_name imex.apg.datacraft.co.ke;
    return 301 https://$server_name$request_uri;
}}

server {{
    listen 443 ssl http2;
    server_name imex.apg.datacraft.co.ke;

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/imex.apg.datacraft.co.ke/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/imex.apg.datacraft.co.ke/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # File Upload Settings
    client_max_body_size {config.max_file_size // (1024 * 1024)}M;
    client_body_timeout 300s;
    client_header_timeout 300s;

    # Compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/json
        application/javascript
        application/xml+rss
        application/atom+xml
        image/svg+xml;

    # Static Files
    location /static/ {{
        alias /opt/apg/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }}

    # API Routes
    location {config.api_prefix}/ {{
        proxy_pass http://apg_imex;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }}

    # WebSocket Support
    location /ws/ {{
        proxy_pass http://apg_imex;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }}

    # Metrics (Protected)
    location /metrics {{
        proxy_pass http://apg_imex;
        allow 127.0.0.1;
        allow 10.0.0.0/8;
        deny all;
    }}

    # Main Application
    location / {{
        proxy_pass http://apg_imex;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }}

    # Health Check
    location /health {{
        access_log off;
        proxy_pass http://apg_imex;
    }}
}}
"""

def save_deployment_configs(config: ProductionConfig, output_dir: str = "./deployment"):
    """
    Save all deployment configurations to files.

    Args:
        config: Production configuration
        output_dir: Output directory for configuration files
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save configurations
    configs = {
        'docker-compose.yml': create_docker_compose_config(config),
        'kubernetes.yml': create_kubernetes_deployment(config),
        'nginx.conf': create_nginx_config(config),
        'production_config.json': config.model_dump_json(indent=2)
    }

    for filename, content in configs.items():
        config_file = output_path / filename
        config_file.write_text(content)
        print(f"✓ Created: {config_file}")

# Production deployment registry
deployment_registry = {
    'config': ProductionConfig,
    'create_config': create_production_config,
    'docker_compose': create_docker_compose_config,
    'kubernetes': create_kubernetes_deployment,
    'nginx': create_nginx_config,
    'save_configs': save_deployment_configs,
    'generate_keys': generate_secure_keys
}

__all__ = [
    'ProductionConfig',
    'DeploymentEnvironment',
    'DatabaseConfig',
    'SecurityConfig',
    'create_production_config',
    'create_docker_compose_config',
    'create_kubernetes_deployment',
    'create_nginx_config',
    'save_deployment_configs',
    'generate_secure_keys',
    'deployment_registry'
]
