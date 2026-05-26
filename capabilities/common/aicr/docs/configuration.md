# AICR Configuration Guide

**Version:** 1.0.0
**Author:** Nyimbi Odero <nyimbi@gmail.com>
**Copyright:** © 2025 Datacraft

## Table of Contents

1. [Configuration Overview](#configuration-overview)
2. [Environment Variables](#environment-variables)
3. [Configuration Files](#configuration-files)
4. [Service Configuration](#service-configuration)
5. [Security Configuration](#security-configuration)
6. [Performance Configuration](#performance-configuration)
7. [Monitoring Configuration](#monitoring-configuration)
8. [Advanced Configuration](#advanced-configuration)
9. [Validation and Testing](#validation-and-testing)

## Configuration Overview

AICR supports multiple configuration methods to accommodate different deployment scenarios:

1. **Environment Variables**: For containerized deployments and secrets
2. **Configuration Files**: For complex, structured configuration
3. **Runtime Configuration**: For dynamic configuration via API
4. **APG Integration**: Configuration through the APG platform

### Configuration Priority

Configuration values are resolved in the following priority order (highest to lowest):

1. Runtime API configuration
2. Environment variables
3. Configuration files
4. APG platform defaults
5. AICR built-in defaults

## Environment Variables

### Core Configuration

```bash
# Service Identity
AICR_SERVICE_ID=aicr-production-001
AICR_SERVICE_NAME="AI Core Framework"
AICR_VERSION=1.0.0

# Network Configuration
AICR_HOST=0.0.0.0                    # Bind address
AICR_PORT=8080                       # Main service port
AICR_WORKERS=4                       # Number of worker processes
AICR_MAX_CONNECTIONS=1000            # Maximum concurrent connections

# API Configuration
AICR_API_PREFIX=/api/v1              # API URL prefix
AICR_CORS_ORIGINS=*                  # CORS allowed origins
AICR_REQUEST_TIMEOUT=300             # Request timeout in seconds
AICR_MAX_REQUEST_SIZE=100MB          # Maximum request size
```

### Database Configuration

```bash
# Primary Database (PostgreSQL)
DATABASE_URL=postgresql://user:password@localhost:5432/aicr_db
DB_POOL_SIZE=10                      # Connection pool size
DB_MAX_OVERFLOW=20                   # Pool overflow limit
DB_POOL_TIMEOUT=30                   # Pool checkout timeout
DB_POOL_RECYCLE=3600                 # Connection recycle time

# Time Series Database (Optional - InfluxDB)
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=your_influxdb_token
INFLUXDB_ORG=datacraft
INFLUXDB_BUCKET=aicr_metrics

# Cache Database (Redis)
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=your_redis_password
REDIS_MAX_CONNECTIONS=50
```

### Storage Configuration

```bash
# Model Storage
MODEL_STORAGE_TYPE=filesystem          # filesystem, s3, gcs, azure
MODEL_STORAGE_PATH=/opt/aicr/models   # Local filesystem path
MODEL_STORAGE_BUCKET=aicr-models      # Cloud storage bucket
MODEL_MAX_SIZE=10GB                   # Maximum model size

# Temporary Storage
TEMP_STORAGE_PATH=/tmp/aicr
TEMP_CLEANUP_INTERVAL=3600            # Cleanup interval in seconds
TEMP_MAX_AGE=86400                    # Max age for temp files

# Cloud Storage (S3/MinIO)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-west-2
AWS_ENDPOINT_URL=http://localhost:9000  # For MinIO

# Azure Storage
AZURE_STORAGE_ACCOUNT=your_account
AZURE_STORAGE_KEY=your_key
AZURE_CONTAINER_NAME=aicr-models

# Google Cloud Storage
GCS_CREDENTIALS_PATH=/path/to/credentials.json
GCS_PROJECT_ID=your_project_id
```

### Security Configuration

```bash
# Authentication
ENABLE_AUTHENTICATION=true
JWT_SECRET_KEY=your_jwt_secret_key_256_bits
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24
JWT_REFRESH_DAYS=7

# Authorization
ENABLE_AUTHORIZATION=true
RBAC_ENABLED=true
DEFAULT_USER_ROLE=user
ADMIN_USERS=admin@datacraft.co.ke,root@datacraft.co.ke

# Encryption
ENCRYPTION_KEY=your_32_byte_encryption_key_here
ENCRYPTION_ALGORITHM=AES-256-GCM
ENABLE_FIELD_ENCRYPTION=true

# Quantum Security
ENABLE_QUANTUM_SECURITY=false
QUANTUM_KEY_SIZE=256
QUANTUM_ALGORITHM=CRYSTALS-Kyber

# SSL/TLS
SSL_ENABLED=true
SSL_CERT_PATH=/path/to/certificate.pem
SSL_KEY_PATH=/path/to/private_key.pem
SSL_CA_PATH=/path/to/ca_bundle.pem
TLS_MIN_VERSION=1.2
```

### Performance Configuration

```bash
# Inference Configuration
MAX_CONCURRENT_INFERENCES=100
INFERENCE_TIMEOUT_SECONDS=300
BATCH_SIZE_LIMIT=1000
ENABLE_MODEL_CACHING=true
MODEL_CACHE_SIZE=5GB

# Auto-scaling
ENABLE_AUTO_SCALING=true
MIN_INSTANCES=2
MAX_INSTANCES=20
TARGET_CPU_UTILIZATION=70
SCALE_UP_THRESHOLD=80
SCALE_DOWN_THRESHOLD=30
SCALING_COOLDOWN=300

# Memory Management
MAX_MEMORY_USAGE=80                  # Percentage
MEMORY_CLEANUP_INTERVAL=300
ENABLE_MEMORY_MONITORING=true

# GPU Configuration
ENABLE_GPU=true
GPU_MEMORY_FRACTION=0.8
GPU_ALLOW_GROWTH=true
```

### Monitoring Configuration

```bash
# Metrics
METRICS_ENABLED=true
PROMETHEUS_PORT=9090
METRICS_INTERVAL=30                  # Collection interval in seconds
METRICS_RETENTION_DAYS=30

# Logging
LOG_LEVEL=INFO                       # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT=json                      # json, text
LOG_FILE=/var/log/aicr/aicr.log
LOG_ROTATION_SIZE=100MB
LOG_ROTATION_COUNT=10

# Alerting
ENABLE_ALERTING=true
ALERT_MANAGER_URL=http://localhost:9093
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USERNAME=alerts@datacraft.co.ke
EMAIL_PASSWORD=your_email_password

# Health Checks
HEALTH_CHECK_INTERVAL=30
HEALTH_CHECK_TIMEOUT=10
HEALTH_CHECK_RETRIES=3
```

### Feature Flags

```bash
# Experimental Features
ENABLE_NEUROMORPHIC_ENGINE=false
ENABLE_QUANTUM_COMPUTING=false
ENABLE_FEDERATED_LEARNING=true
ENABLE_EDGE_AI=true
ENABLE_MODEL_MARKETPLACE=true

# API Features
ENABLE_WEBSOCKET_API=true
ENABLE_GRAPHQL_API=false
ENABLE_GRPC_API=false

# Development Features
ENABLE_DEBUG_MODE=false
ENABLE_PROFILING=false
ENABLE_API_DOCS=true
ENABLE_SWAGGER_UI=true
```

## Configuration Files

### Main Configuration File

Create `config/aicr_config.yaml`:

```yaml
# AICR Configuration File
service:
  id: "aicr-production-001"
  name: "AI Core Framework"
  version: "1.0.0"
  environment: "production"

network:
  host: "0.0.0.0"
  port: 8080
  workers: 4
  max_connections: 1000

api:
  prefix: "/api/v1"
  cors_origins: ["https://app.datacraft.co.ke"]
  request_timeout: 300
  max_request_size: "100MB"
  rate_limiting:
    enabled: true
    requests_per_minute: 1000
    burst_size: 100

database:
  url: "postgresql://aicr_user:password@localhost:5432/aicr_db"
  pool_size: 10
  max_overflow: 20
  pool_timeout: 30
  pool_recycle: 3600

  # Time series database for metrics
  influxdb:
    url: "http://localhost:8086"
    token: "${INFLUXDB_TOKEN}"
    org: "datacraft"
    bucket: "aicr_metrics"

  # Cache database
  redis:
    url: "redis://localhost:6379/0"
    password: "${REDIS_PASSWORD}"
    max_connections: 50

storage:
  models:
    type: "filesystem"  # filesystem, s3, gcs, azure
    path: "/opt/aicr/models"
    max_size: "10GB"

  temporary:
    path: "/tmp/aicr"
    cleanup_interval: 3600
    max_age: 86400

  # Cloud storage configuration
  cloud:
    provider: "s3"  # s3, gcs, azure
    bucket: "aicr-models"
    region: "us-west-2"
    credentials:
      access_key: "${AWS_ACCESS_KEY_ID}"
      secret_key: "${AWS_SECRET_ACCESS_KEY}"

security:
  authentication:
    enabled: true
    jwt:
      secret_key: "${JWT_SECRET_KEY}"
      algorithm: "HS256"
      expiration_hours: 24
      refresh_days: 7

  authorization:
    enabled: true
    rbac_enabled: true
    default_role: "user"
    admin_users:
      - "admin@datacraft.co.ke"
      - "root@datacraft.co.ke"

  encryption:
    key: "${ENCRYPTION_KEY}"
    algorithm: "AES-256-GCM"
    field_encryption: true

  quantum:
    enabled: false
    key_size: 256
    algorithm: "CRYSTALS-Kyber"

  ssl:
    enabled: true
    cert_path: "/etc/ssl/certs/aicr.pem"
    key_path: "/etc/ssl/private/aicr.key"
    ca_path: "/etc/ssl/ca_bundle.pem"
    min_version: "1.2"

performance:
  inference:
    max_concurrent: 100
    timeout_seconds: 300
    batch_size_limit: 1000
    model_caching: true
    cache_size: "5GB"

  auto_scaling:
    enabled: true
    min_instances: 2
    max_instances: 20
    target_cpu: 70
    scale_up_threshold: 80
    scale_down_threshold: 30
    cooldown: 300

  memory:
    max_usage_percent: 80
    cleanup_interval: 300
    monitoring: true

  gpu:
    enabled: true
    memory_fraction: 0.8
    allow_growth: true

monitoring:
  metrics:
    enabled: true
    prometheus_port: 9090
    interval: 30
    retention_days: 30

  logging:
    level: "INFO"
    format: "json"
    file: "/var/log/aicr/aicr.log"
    rotation:
      size: "100MB"
      count: 10

  alerting:
    enabled: true
    alert_manager_url: "http://localhost:9093"
    slack_webhook: "${SLACK_WEBHOOK_URL}"
    email:
      smtp_server: "smtp.gmail.com"
      smtp_port: 587
      username: "alerts@datacraft.co.ke"
      password: "${EMAIL_PASSWORD}"

  health_checks:
    interval: 30
    timeout: 10
    retries: 3

features:
  # Core features
  neuromorphic_engine: false
  quantum_computing: false
  federated_learning: true
  edge_ai: true
  model_marketplace: true

  # API features
  websocket_api: true
  graphql_api: false
  grpc_api: false

  # Development features
  debug_mode: false
  profiling: false
  api_docs: true
  swagger_ui: true

# Component-specific configuration
components:
  inference_engine:
    frameworks:
      - pytorch
      - tensorflow
      - onnx
    optimization:
      enabled: true
      precision: "fp16"
      batch_optimization: true

  distributed_computing:
    cluster:
      min_nodes: 2
      max_nodes: 10
      node_selection: "resource_based"
    load_balancing:
      algorithm: "weighted_round_robin"
      health_check_interval: 30

  model_marketplace:
    curation:
      enabled: true
      quality_threshold: 0.8
      security_scanning: true
    recommendation:
      algorithm: "collaborative_filtering"
      update_interval: 3600

# Environment-specific overrides
environments:
  development:
    logging:
      level: "DEBUG"
    features:
      debug_mode: true
      api_docs: true
    performance:
      auto_scaling:
        enabled: false

  staging:
    performance:
      auto_scaling:
        min_instances: 1
        max_instances: 5

  production:
    security:
      ssl:
        enabled: true
      quantum:
        enabled: true
    monitoring:
      alerting:
        enabled: true
```

### Service-Specific Configuration

Create component-specific configuration files:

#### Inference Engine Configuration (`config/inference_config.yaml`)

```yaml
inference_engine:
  frameworks:
    pytorch:
      enabled: true
      device: "auto"  # auto, cpu, cuda, mps
      optimization:
        torch_script: true
        tensorrt: false
        openvino: false

    tensorflow:
      enabled: true
      device: "auto"
      optimization:
        xla: true
        tensorrt: false
        tflite: false

    onnx:
      enabled: true
      providers:
        - "CPUExecutionProvider"
        - "CUDAExecutionProvider"
      optimization:
        graph_optimization: true
        quantization: false

  batching:
    enabled: true
    max_batch_size: 32
    timeout_ms: 100
    padding_strategy: "longest"

  caching:
    enabled: true
    max_size: "2GB"
    ttl_seconds: 3600
    eviction_policy: "LRU"
```

#### Security Configuration (`config/security_config.yaml`)

```yaml
security:
  policies:
    password_policy:
      min_length: 12
      require_uppercase: true
      require_lowercase: true
      require_digits: true
      require_special: true
      max_age_days: 90

    session_policy:
      max_duration: 86400  # 24 hours
      idle_timeout: 3600   # 1 hour
      max_concurrent: 5

    api_policy:
      rate_limiting:
        default: 1000      # requests per hour
        burst: 100
        by_user: 10000
        by_ip: 5000

      request_validation:
        max_size: "100MB"
        allowed_types:
          - "application/json"
          - "multipart/form-data"
          - "application/octet-stream"

  compliance:
    gdpr:
      enabled: true
      data_retention_days: 365
      anonymization: true

    hipaa:
      enabled: false
      encryption_at_rest: true
      audit_logging: true

    sox:
      enabled: false
      data_integrity: true
      access_controls: true
```

## Service Configuration

### APG Integration Configuration

```python
# config/apg_integration.py
from apg.composition import CapabilityConfiguration

aicr_capability_config = CapabilityConfiguration(
    name="aicr",
    version="1.0.0",
    interfaces=[
        "ai.inference",
        "ai.training",
        "ai.management",
        "ai.monitoring"
    ],
    dependencies=[
        "security",
        "monitoring",
        "storage",
        "networking"
    ],
    resources={
        "cpu": "2-16",
        "memory": "4Gi-32Gi",
        "storage": "100Gi-1Ti",
        "gpu": "0-4"
    },
    scaling={
        "min_replicas": 1,
        "max_replicas": 10,
        "target_cpu": 70,
        "target_memory": 80
    },
    health_check={
        "path": "/health",
        "interval": 30,
        "timeout": 10,
        "retries": 3
    },
    configuration={
        "auto_start": True,
        "graceful_shutdown": True,
        "startup_timeout": 300
    }
)
```

### Runtime Configuration

```python
# Dynamic configuration via service API
async def update_runtime_config():
    from aicr.service import AICoreService

    service = AICoreService()

    # Update performance settings
    await service.update_configuration({
        "performance.max_concurrent_inferences": 200,
        "performance.batch_size_limit": 1500,
        "monitoring.metrics_interval": 15
    })

    # Update feature flags
    await service.toggle_feature("experimental.quantum_inference", True)

    # Update model caching
    await service.configure_caching({
        "enabled": True,
        "size": "10GB",
        "ttl": 7200,
        "compression": True
    })
```

## Security Configuration

### Authentication Configuration

```yaml
authentication:
  providers:
    local:
      enabled: true
      password_hashing: "bcrypt"
      rounds: 12

    oauth2:
      enabled: true
      providers:
        google:
          client_id: "${GOOGLE_CLIENT_ID}"
          client_secret: "${GOOGLE_CLIENT_SECRET}"
          scopes: ["openid", "email", "profile"]

        azure:
          tenant_id: "${AZURE_TENANT_ID}"
          client_id: "${AZURE_CLIENT_ID}"
          client_secret: "${AZURE_CLIENT_SECRET}"

    ldap:
      enabled: false
      server: "ldap://ldap.company.com"
      base_dn: "ou=users,dc=company,dc=com"
      user_filter: "(uid={username})"

    saml:
      enabled: false
      idp_url: "https://idp.company.com/saml"
      certificate_path: "/etc/ssl/saml/certificate.pem"

jwt:
  issuer: "aicr.datacraft.co.ke"
  audience: "aicr-api"
  algorithm: "RS256"
  public_key_path: "/etc/ssl/jwt/public.pem"
  private_key_path: "/etc/ssl/jwt/private.pem"
  expiration: 3600
  refresh_expiration: 604800

  claims:
    include_user_info: true
    include_permissions: true
    include_roles: true
    custom_claims:
      tenant_id: true
      organization: true
```

### Authorization Configuration

```yaml
authorization:
  rbac:
    enabled: true

    roles:
      admin:
        description: "Full system access"
        permissions: ["*"]

      model_manager:
        description: "Model management access"
        permissions:
          - "model:create"
          - "model:read"
          - "model:update"
          - "model:delete"
          - "model:deploy"
          - "model:undeploy"

      data_scientist:
        description: "Model development access"
        permissions:
          - "model:read"
          - "model:create"
          - "inference:execute"
          - "pipeline:create"
          - "pipeline:execute"

      operator:
        description: "Operations access"
        permissions:
          - "model:read"
          - "deployment:read"
          - "monitoring:read"
          - "health:read"

      user:
        description: "Basic inference access"
        permissions:
          - "inference:execute"
          - "model:read"

    policies:
      resource_ownership:
        enabled: true
        owner_permissions: ["read", "update", "delete"]
        organization_permissions: ["read"]

      data_classification:
        enabled: true
        classifications:
          public: ["*"]
          internal: ["employee", "contractor"]
          confidential: ["admin", "authorized_user"]
          restricted: ["admin"]
```

## Performance Configuration

### Auto-Scaling Configuration

```yaml
auto_scaling:
  horizontal:
    enabled: true
    min_replicas: 2
    max_replicas: 50

    metrics:
      cpu:
        target_utilization: 70
        scale_up_threshold: 80
        scale_down_threshold: 30

      memory:
        target_utilization: 75
        scale_up_threshold: 85
        scale_down_threshold: 40

      custom:
        inference_queue_length:
          target_value: 10
          scale_up_threshold: 20
          scale_down_threshold: 5

        requests_per_second:
          target_value: 1000
          scale_up_threshold: 1500
          scale_down_threshold: 500

    behavior:
      scale_up:
        stabilization_window: 60
        policies:
          - type: "Percent"
            value: 100
            period: 60
          - type: "Pods"
            value: 5
            period: 60

      scale_down:
        stabilization_window: 300
        policies:
          - type: "Percent"
            value: 50
            period: 300
          - type: "Pods"
            value: 2
            period: 300

  vertical:
    enabled: true
    update_mode: "Auto"

    resources:
      cpu:
        min: "100m"
        max: "16"
        target_utilization: 70

      memory:
        min: "512Mi"
        max: "32Gi"
        target_utilization: 75
```

### Caching Configuration

```yaml
caching:
  model_cache:
    enabled: true
    type: "memory"  # memory, redis, memcached
    size: "5GB"
    ttl: 3600
    eviction_policy: "LRU"
    compression: true

  inference_cache:
    enabled: true
    type: "redis"
    ttl: 1800
    max_entries: 100000
    key_prefix: "aicr:inference:"

  metadata_cache:
    enabled: true
    type: "memory"
    size: "1GB"
    ttl: 300
    refresh_ahead: true
```

## Monitoring Configuration

### Metrics Configuration

```yaml
metrics:
  collection:
    enabled: true
    interval: 30
    batch_size: 1000
    compression: true

  exporters:
    prometheus:
      enabled: true
      port: 9090
      path: "/metrics"
      labels:
        service: "aicr"
        version: "1.0.0"
        environment: "${ENVIRONMENT}"

    influxdb:
      enabled: true
      url: "${INFLUXDB_URL}"
      token: "${INFLUXDB_TOKEN}"
      org: "datacraft"
      bucket: "aicr_metrics"
      precision: "s"

    datadog:
      enabled: false
      api_key: "${DATADOG_API_KEY}"
      tags:
        - "service:aicr"
        - "env:production"

  retention:
    raw_data: "7d"
    aggregated_5m: "30d"
    aggregated_1h: "365d"
    aggregated_1d: "5y"
```

### Alerting Configuration

```yaml
alerting:
  rules:
    high_error_rate:
      condition: "error_rate > 5"
      duration: "5m"
      severity: "warning"
      message: "High error rate detected: {{ $value }}%"

    high_latency:
      condition: "p95_latency > 1000"
      duration: "3m"
      severity: "critical"
      message: "High inference latency: {{ $value }}ms"

    low_availability:
      condition: "availability < 99"
      duration: "1m"
      severity: "critical"
      message: "Service availability below threshold: {{ $value }}%"

    disk_space_low:
      condition: "disk_usage > 85"
      duration: "10m"
      severity: "warning"
      message: "Disk usage high: {{ $value }}%"

  channels:
    email:
      enabled: true
      to: ["alerts@datacraft.co.ke", "oncall@datacraft.co.ke"]
      smtp:
        server: "smtp.gmail.com"
        port: 587
        username: "${EMAIL_USERNAME}"
        password: "${EMAIL_PASSWORD}"

    slack:
      enabled: true
      webhook_url: "${SLACK_WEBHOOK_URL}"
      channel: "#aicr-alerts"
      username: "AICR Monitor"

    pagerduty:
      enabled: false
      integration_key: "${PAGERDUTY_INTEGRATION_KEY}"
      service_key: "${PAGERDUTY_SERVICE_KEY}"
```

## Advanced Configuration

### Multi-Tenant Configuration

```yaml
multi_tenancy:
  enabled: true
  isolation_level: "namespace"  # namespace, database, cluster

  tenant_detection:
    method: "header"  # header, subdomain, path, jwt_claim
    header_name: "X-Tenant-ID"
    jwt_claim: "tenant_id"

  resource_quotas:
    default:
      cpu: "4"
      memory: "8Gi"
      storage: "100Gi"
      models: 50
      requests_per_hour: 10000

    premium:
      cpu: "16"
      memory: "32Gi"
      storage: "1Ti"
      models: 500
      requests_per_hour: 100000

  billing:
    enabled: true
    metrics:
      - "inference_requests"
      - "compute_hours"
      - "storage_gb_hours"
      - "data_transfer_gb"
```

### Disaster Recovery Configuration

```yaml
disaster_recovery:
  backup:
    enabled: true
    schedule: "0 2 * * *"  # Daily at 2 AM
    retention: "30d"
    compression: true
    encryption: true

    destinations:
      - type: "s3"
        bucket: "aicr-backups"
        prefix: "daily/"
      - type: "gcs"
        bucket: "aicr-backups-secondary"
        prefix: "daily/"

  replication:
    enabled: true
    mode: "async"  # sync, async

    targets:
      - region: "us-east-1"
        priority: 1
        lag_threshold: "5m"
      - region: "eu-west-1"
        priority: 2
        lag_threshold: "15m"

  failover:
    enabled: true
    auto_failover: false
    health_check_interval: "30s"
    health_check_timeout: "10s"
    max_failure_count: 3
```

## Validation and Testing

### Configuration Validation

```python
# validate_config.py
from aicr.config import ConfigValidator

async def validate_configuration():
    validator = ConfigValidator()

    # Validate configuration file
    result = await validator.validate_file("config/aicr_config.yaml")

    if result.is_valid:
        print("✅ Configuration is valid")
        print(f"Validated {len(result.sections)} sections")
    else:
        print("❌ Configuration validation failed")
        for error in result.errors:
            print(f"  - {error.section}: {error.message}")

    # Test database connectivity
    db_result = await validator.test_database_connection()
    print(f"Database: {'✅' if db_result.success else '❌'}")

    # Test storage access
    storage_result = await validator.test_storage_access()
    print(f"Storage: {'✅' if storage_result.success else '❌'}")

    # Test external services
    external_result = await validator.test_external_services()
    print(f"External Services: {'✅' if external_result.success else '❌'}")

# Run validation
import asyncio
asyncio.run(validate_configuration())
```

### Configuration Testing

```bash
# Test configuration with dry-run
python -m aicr.service --config-file config/aicr_config.yaml --dry-run

# Validate environment variables
python -m aicr.config.validate --env

# Test specific components
python -m aicr.config.test --component database
python -m aicr.config.test --component storage
python -m aicr.config.test --component security

# Performance test with configuration
python -m aicr.performance.test --config config/aicr_config.yaml
```

## Configuration Management Best Practices

### 1. **Secret Management**

```bash
# Use external secret management
export JWT_SECRET_KEY=$(kubectl get secret aicr-secrets -o jsonpath='{.data.jwt-secret}' | base64 -d)
export DATABASE_PASSWORD=$(aws secretsmanager get-secret-value --secret-id aicr/db/password --query SecretString --output text)
```

### 2. **Environment-Specific Configuration**

```bash
# Use configuration overlays
cp config/base.yaml config/production.yaml
# Edit production-specific settings
yq eval-all 'select(fileIndex == 0) * select(fileIndex == 1)' config/base.yaml config/production.yaml > config/final.yaml
```

### 3. **Configuration Versioning**

```bash
# Version configuration files
git tag config-v1.0.0
git push origin config-v1.0.0

# Deploy specific configuration version
kubectl apply -f k8s/configmap-v1.0.0.yaml
```

### 4. **Configuration Monitoring**

```python
# Monitor configuration changes
from aicr.config import ConfigMonitor

monitor = ConfigMonitor()
monitor.watch_file("config/aicr_config.yaml", callback=reload_configuration)
monitor.watch_env_vars(["DATABASE_URL", "JWT_SECRET_KEY"], callback=restart_service)
```

---

**Next Steps:**
- Review [Security Guide](guides/security_guide.md) for security configuration details
- Check [Performance Tuning](guides/performance_tuning.md) for optimization
- Explore [Deployment Documentation](deployment/) for production setup