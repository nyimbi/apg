# AICR Production Deployment Guide

**Version:** 1.0.0
**Author:** Nyimbi Odero <nyimbi@gmail.com>
**Copyright:** © 2025 Datacraft

## Table of Contents

1. [Production Architecture](#production-architecture)
2. [Infrastructure Requirements](#infrastructure-requirements)
3. [Security Hardening](#security-hardening)
4. [High Availability Setup](#high-availability-setup)
5. [Performance Optimization](#performance-optimization)
6. [Monitoring and Alerting](#monitoring-and-alerting)
7. [Backup and Disaster Recovery](#backup-and-disaster-recovery)
8. [Operational Procedures](#operational-procedures)

## Production Architecture

### Reference Architecture

```
                                 ┌─────────────────────────────────────┐
                                 │            Load Balancer            │
                                 │         (HAProxy/nginx)             │
                                 └─────────────────┬───────────────────┘
                                                   │
                    ┌──────────────────────────────┼──────────────────────────────┐
                    │                              │                              │
        ┌─────────────────────┐        ┌─────────────────────┐        ┌─────────────────────┐
        │   AICR Instance 1   │        │   AICR Instance 2   │        │   AICR Instance 3   │
        │   (Primary AZ)      │        │   (Secondary AZ)    │        │   (Tertiary AZ)     │
        └─────────────────────┘        └─────────────────────┘        └─────────────────────┘
                    │                              │                              │
        ┌─────────────────────────────────────────────────────────────────────────────────────┐
        │                           Shared Services Layer                                     │
        ├─────────────────┬─────────────────┬─────────────────┬─────────────────────────────┤
        │   PostgreSQL    │   Redis Cluster │   Storage       │   Monitoring Stack          │
        │   (Primary +    │   (3 nodes)     │   (MinIO/S3)    │   (Prometheus + Grafana)    │
        │   Standby)      │                 │                 │                             │
        └─────────────────┴─────────────────┴─────────────────┴─────────────────────────────┘
```

### Components Overview

| Component | Purpose | Redundancy | Scaling |
|-----------|---------|------------|---------|
| **Load Balancer** | Traffic distribution, SSL termination | Active-Passive | Horizontal |
| **AICR Instances** | Core AI processing services | Multi-AZ deployment | Auto-scaling |
| **PostgreSQL** | Metadata and configuration storage | Master-Slave replication | Read replicas |
| **Redis Cluster** | Caching and session management | 3-node cluster with failover | Vertical + Horizontal |
| **Object Storage** | Model artifacts and data | Multi-region replication | Unlimited |
| **Monitoring** | Observability and alerting | Clustered deployment | Horizontal |

## Infrastructure Requirements

### Hardware Specifications

#### Production Environment

```yaml
# Minimum Production Setup (3-node cluster)
nodes:
  control_plane:
    count: 3
    cpu: 8 cores
    memory: 32 GB
    storage: 500 GB SSD
    network: 10 Gbps

  worker_nodes:
    count: 6
    cpu: 16 cores
    memory: 64 GB
    storage: 1 TB NVMe SSD
    gpu: 2x NVIDIA A100 (optional)
    network: 25 Gbps

  storage_nodes:
    count: 3
    cpu: 4 cores
    memory: 16 GB
    storage: 10 TB (RAID 10)
    network: 10 Gbps

# Database Servers
database:
  primary:
    cpu: 16 cores
    memory: 128 GB
    storage: 2 TB NVMe SSD (RAID 1)
    network: 10 Gbps

  replica:
    cpu: 8 cores
    memory: 64 GB
    storage: 2 TB SSD
    network: 10 Gbps

# Load Balancers
load_balancer:
  count: 2
  cpu: 4 cores
  memory: 8 GB
  storage: 100 GB SSD
  network: 10 Gbps
```

#### High-Performance Setup

```yaml
# High-Performance Production Setup
nodes:
  control_plane:
    count: 5
    cpu: 16 cores
    memory: 64 GB
    storage: 1 TB NVMe SSD
    network: 25 Gbps

  inference_nodes:
    count: 10
    cpu: 32 cores
    memory: 128 GB
    storage: 2 TB NVMe SSD
    gpu: 4x NVIDIA H100
    network: 100 Gbps (InfiniBand)

  training_nodes:
    count: 8
    cpu: 64 cores
    memory: 512 GB
    storage: 4 TB NVMe SSD
    gpu: 8x NVIDIA H100
    network: 100 Gbps (InfiniBand)
```

### Network Architecture

```yaml
# Network Configuration
network:
  vpc_cidr: "10.0.0.0/16"

  subnets:
    public:
      - "10.0.1.0/24"  # AZ-1 (Load Balancers)
      - "10.0.2.0/24"  # AZ-2 (Load Balancers)
      - "10.0.3.0/24"  # AZ-3 (Load Balancers)

    private:
      - "10.0.10.0/24"  # AZ-1 (AICR Services)
      - "10.0.11.0/24"  # AZ-2 (AICR Services)
      - "10.0.12.0/24"  # AZ-3 (AICR Services)

    database:
      - "10.0.20.0/24"  # AZ-1 (Database)
      - "10.0.21.0/24"  # AZ-2 (Database)
      - "10.0.22.0/24"  # AZ-3 (Database)

  security_groups:
    load_balancer:
      ingress:
        - port: 443, protocol: tcp, source: "0.0.0.0/0"
        - port: 80, protocol: tcp, source: "0.0.0.0/0"

    aicr_services:
      ingress:
        - port: 8080, protocol: tcp, source: load_balancer_sg
        - port: 9090, protocol: tcp, source: monitoring_sg
        - port: 22, protocol: tcp, source: bastion_sg

    database:
      ingress:
        - port: 5432, protocol: tcp, source: aicr_services_sg
        - port: 6379, protocol: tcp, source: aicr_services_sg
```

## Security Hardening

### System-Level Security

```bash
#!/bin/bash
# Security hardening script for AICR production deployment

# 1. Update system packages
apt update && apt upgrade -y

# 2. Configure firewall
ufw --force reset
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 8080/tcp  # AICR service
ufw allow 9090/tcp  # Prometheus metrics
ufw --force enable

# 3. Disable unused services
systemctl disable bluetooth
systemctl disable cups
systemctl disable avahi-daemon

# 4. Secure SSH configuration
cat > /etc/ssh/sshd_config.d/security.conf << EOF
# Disable root login
PermitRootLogin no

# Use key-based authentication only
PasswordAuthentication no
PubkeyAuthentication yes

# Limit login attempts
MaxAuthTries 3
MaxSessions 3

# Disable unused features
X11Forwarding no
AllowTcpForwarding no
AllowAgentForwarding no

# Session timeout
ClientAliveInterval 300
ClientAliveCountMax 2
EOF

systemctl restart sshd

# 5. Configure fail2ban
apt install -y fail2ban
cat > /etc/fail2ban/jail.local << EOF
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3

[sshd]
enabled = true
port = ssh
logpath = /var/log/auth.log

[aicr-api]
enabled = true
port = 8080
logpath = /var/log/aicr/access.log
maxretry = 10
EOF

systemctl enable fail2ban
systemctl start fail2ban

# 6. Configure audit logging
apt install -y auditd
cat > /etc/audit/rules.d/aicr.rules << EOF
# Monitor AICR files
-w /opt/aicr/ -p wa -k aicr_files
-w /etc/aicr/ -p wa -k aicr_config

# Monitor system calls
-a always,exit -F arch=b64 -S execve -k process_exec
-a always,exit -F arch=b64 -S connect -k network_connect

# Monitor privileged commands
-w /usr/bin/sudo -p x -k privileged_commands
EOF

augenrules --load
systemctl restart auditd
```

### Application Security

```yaml
# AICR Security Configuration
security:
  authentication:
    jwt:
      algorithm: "RS256"
      key_rotation_hours: 24
      max_token_age_hours: 8

    oauth2:
      enabled: true
      providers:
        - name: "company_sso"
          issuer: "https://sso.company.com"
          client_id: "${OAUTH2_CLIENT_ID}"
          client_secret: "${OAUTH2_CLIENT_SECRET}"

    multi_factor:
      enabled: true
      required_for_admin: true
      totp_issuer: "AICR Production"

  authorization:
    rbac:
      enabled: true
      policy_engine: "opa"  # Open Policy Agent

    policies:
      - name: "admin_access"
        subjects: ["role:admin"]
        resources: ["*"]
        actions: ["*"]

      - name: "model_operator"
        subjects: ["role:model_operator"]
        resources: ["model:*", "deployment:*"]
        actions: ["read", "create", "update"]

      - name: "inference_user"
        subjects: ["role:user"]
        resources: ["inference:*"]
        actions: ["execute"]

  encryption:
    data_at_rest:
      algorithm: "AES-256-GCM"
      key_rotation_days: 90

    data_in_transit:
      tls_version: "1.3"
      cipher_suites:
        - "TLS_AES_256_GCM_SHA384"
        - "TLS_CHACHA20_POLY1305_SHA256"

    field_encryption:
      enabled: true
      fields: ["model_weights", "user_data", "api_keys"]

  compliance:
    gdpr:
      enabled: true
      data_retention_days: 730
      anonymization_enabled: true

    hipaa:
      enabled: false  # Enable if handling healthcare data

    sox:
      enabled: true
      audit_logging: true
      data_integrity_checks: true
```

### Network Security

```yaml
# Network Security Configuration
network_security:
  ssl_termination:
    certificate_authority: "letsencrypt"
    auto_renewal: true

  waf:
    enabled: true
    rules:
      - name: "sql_injection"
        pattern: "(?i)(union|select|insert|delete|drop|create|alter)"
        action: "block"

      - name: "xss_protection"
        pattern: "(?i)(<script|javascript:|onload=|onerror=)"
        action: "block"

      - name: "rate_limiting"
        requests_per_minute: 1000
        burst_size: 100
        action: "throttle"

  ddos_protection:
    enabled: true
    threshold_requests_per_second: 10000
    mitigation_action: "challenge"

  ip_filtering:
    whitelist:
      - "10.0.0.0/8"      # Internal networks
      - "172.16.0.0/12"   # Private networks
      - "192.168.0.0/16"  # Local networks

    blacklist:
      - "192.0.2.0/24"    # Test network
      - "198.51.100.0/24" # Documentation network
```

## High Availability Setup

### Database High Availability

```yaml
# PostgreSQL HA Configuration
postgresql_ha:
  primary:
    host: "aicr-db-primary.internal"
    port: 5432
    config:
      max_connections: 200
      shared_buffers: "32GB"
      effective_cache_size: "96GB"
      checkpoint_completion_target: 0.9
      wal_buffers: "16MB"
      default_statistics_target: 100

  standby:
    host: "aicr-db-standby.internal"
    port: 5432
    replication_mode: "streaming"
    synchronous_commit: "on"

  failover:
    automatic: true
    timeout_seconds: 30
    promotion_trigger_file: "/tmp/postgresql.trigger"

  backup:
    method: "continuous_archiving"
    archive_command: "aws s3 cp %p s3://aicr-db-backups/wal/%f"
    retention_days: 30

# Redis Cluster Configuration
redis_cluster:
  nodes:
    - host: "aicr-redis-1.internal"
      port: 6379
      role: "master"
    - host: "aicr-redis-2.internal"
      port: 6379
      role: "master"
    - host: "aicr-redis-3.internal"
      port: 6379
      role: "master"
    - host: "aicr-redis-4.internal"
      port: 6379
      role: "slave"
    - host: "aicr-redis-5.internal"
      port: 6379
      role: "slave"
    - host: "aicr-redis-6.internal"
      port: 6379
      role: "slave"

  sentinel:
    enabled: true
    quorum: 2
    down_after_milliseconds: 30000
    failover_timeout: 180000
```

### Load Balancer Configuration

```nginx
# HAProxy Configuration for AICR
global
    daemon
    chroot /var/lib/haproxy
    stats socket /run/haproxy/admin.sock mode 660 level admin
    stats timeout 30s
    user haproxy
    group haproxy

    # SSL/TLS Configuration
    ssl-default-bind-ciphers ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384
    ssl-default-bind-options ssl-min-ver TLSv1.2 no-sslv3

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms
    option httplog
    option dontlognull
    option redispatch
    retries 3

    # Health check configuration
    option httpchk GET /health HTTP/1.1\r\nHost:\ api.datacraft.co.ke

frontend aicr_frontend
    bind *:443 ssl crt /etc/ssl/certs/aicr.pem
    bind *:80
    redirect scheme https if !{ ssl_fc }

    # Security headers
    http-response set-header Strict-Transport-Security "max-age=31536000; includeSubDomains"
    http-response set-header X-Frame-Options "DENY"
    http-response set-header X-Content-Type-Options "nosniff"
    http-response set-header X-XSS-Protection "1; mode=block"

    # Rate limiting
    stick-table type ip size 100k expire 30s store http_req_rate(10s)
    http-request track-sc0 src
    http-request deny if { sc_http_req_rate(0) gt 100 }

    default_backend aicr_backend

backend aicr_backend
    balance roundrobin

    # Health checks
    option httpchk GET /health
    http-check expect status 200

    # Servers
    server aicr-1 10.0.10.10:8080 check inter 2000ms rise 2 fall 3
    server aicr-2 10.0.11.10:8080 check inter 2000ms rise 2 fall 3
    server aicr-3 10.0.12.10:8080 check inter 2000ms rise 2 fall 3

    # Backup server
    server aicr-backup 10.0.10.11:8080 check backup

# Statistics interface
listen stats
    bind *:8404
    stats enable
    stats uri /stats
    stats refresh 30s
    stats admin if TRUE
```

### Auto-Scaling Configuration

```yaml
# Kubernetes HPA Configuration
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: aicr-hpa
  namespace: aicr-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: aicr-service

  minReplicas: 3
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

    - type: Pods
      pods:
        metric:
          name: inference_queue_length
        target:
          type: AverageValue
          averageValue: "10"

    - type: Pods
      pods:
        metric:
          name: response_time_p95
        target:
          type: AverageValue
          averageValue: "100m"  # 100ms

  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Percent
          value: 100
          periodSeconds: 60
        - type: Pods
          value: 5
          periodSeconds: 60

    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 50
          periodSeconds: 300
        - type: Pods
          value: 2
          periodSeconds: 300
```

## Performance Optimization

### JVM and Python Optimization

```bash
# AICR Service Performance Tuning
export AICR_OPTS="
  # Memory settings
  -Xms8g
  -Xmx32g
  -XX:NewRatio=3
  -XX:SurvivorRatio=3
  -XX:MaxMetaspaceSize=512m

  # Garbage collection
  -XX:+UseG1GC
  -XX:MaxGCPauseMillis=200
  -XX:ParallelGCThreads=20
  -XX:G1HeapRegionSize=16m

  # Performance monitoring
  -XX:+PrintGC
  -XX:+PrintGCDetails
  -XX:+PrintGCTimeStamps
  -Xloggc:/var/log/aicr/gc.log

  # JIT optimization
  -XX:+TieredCompilation
  -XX:TieredStopAtLevel=4
  -XX:CompileThreshold=10000
"

# Python optimization
export PYTHONPATH="/opt/aicr/lib"
export PYTHONOPTIMIZE=2
export PYTHONDONTWRITEBYTECODE=1

# CUDA optimization (if using GPU)
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_CACHE_MAXSIZE=2147483648  # 2GB

# PyTorch optimization
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_SHOW_CPP_STACKTRACES=1
```

### Database Performance Tuning

```sql
-- PostgreSQL Performance Configuration
-- /etc/postgresql/15/main/postgresql.conf

-- Memory settings
shared_buffers = '32GB'                    -- 25% of system RAM
effective_cache_size = '96GB'              -- 75% of system RAM
work_mem = '256MB'
maintenance_work_mem = '2GB'
huge_pages = 'try'

-- Checkpoint settings
checkpoint_completion_target = 0.9
checkpoint_timeout = '15min'
max_wal_size = '16GB'
min_wal_size = '2GB'

-- Connection settings
max_connections = 200
superuser_reserved_connections = 3

-- Logging settings
log_statement = 'mod'
log_min_duration_statement = 1000         -- Log slow queries > 1s
log_checkpoints = on
log_connections = on
log_disconnections = on
log_lock_waits = on

-- Query planner settings
default_statistics_target = 100
constraint_exclusion = partition
cursor_tuple_fraction = 0.1

-- Write ahead logging
wal_buffers = '16MB'
wal_writer_delay = '200ms'
commit_delay = 0
commit_siblings = 5

-- Background writer
bgwriter_delay = '200ms'
bgwriter_lru_maxpages = 100
bgwriter_lru_multiplier = 2.0

-- Vacuum settings
autovacuum = on
autovacuum_naptime = '1min'
autovacuum_vacuum_threshold = 50
autovacuum_analyze_threshold = 50
autovacuum_vacuum_scale_factor = 0.2
autovacuum_analyze_scale_factor = 0.1
```

### Model Serving Optimization

```yaml
# Model serving optimization configuration
model_serving:
  optimization:
    # Model compilation
    compile_models: true
    optimization_level: "O3"
    target_architecture: "native"

    # Batch processing
    dynamic_batching:
      enabled: true
      max_batch_size: 64
      batch_timeout_ms: 50
      preferred_batch_sizes: [1, 2, 4, 8, 16, 32, 64]

    # Memory optimization
    model_caching:
      enabled: true
      cache_size: "8GB"
      cache_policy: "LRU"
      preload_models: ["sentiment_v2", "classification_v3"]

    # GPU optimization
    gpu_settings:
      memory_fraction: 0.9
      allow_growth: true
      force_gpu_compatible: true
      enable_mixed_precision: true

    # Threading
    inference_threads: 8
    inter_op_parallelism: 4
    intra_op_parallelism: 16

  # Resource allocation
  resource_limits:
    cpu: "16"
    memory: "32Gi"
    gpu: "2"

  resource_requests:
    cpu: "8"
    memory: "16Gi"
    gpu: "1"
```

## Monitoring and Alerting

### Prometheus Configuration

```yaml
# Prometheus configuration for AICR monitoring
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "/etc/prometheus/rules/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager-1:9093
          - alertmanager-2:9093

scrape_configs:
  # AICR service metrics
  - job_name: 'aicr'
    static_configs:
      - targets:
          - 'aicr-1:9090'
          - 'aicr-2:9090'
          - 'aicr-3:9090'
    metrics_path: '/metrics'
    scrape_interval: 15s

  # System metrics
  - job_name: 'node'
    static_configs:
      - targets:
          - 'aicr-1:9100'
          - 'aicr-2:9100'
          - 'aicr-3:9100'

  # Database metrics
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  # Redis metrics
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  # Load balancer metrics
  - job_name: 'haproxy'
    static_configs:
      - targets: ['haproxy-exporter:9101']
```

### Alert Rules

```yaml
# AICR Alerting Rules
groups:
  - name: aicr.rules
    rules:
      # Service availability
      - alert: AICRServiceDown
        expr: up{job="aicr"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "AICR service is down"
          description: "AICR service {{ $labels.instance }} has been down for more than 1 minute"

      # High error rate
      - alert: HighErrorRate
        expr: rate(aicr_http_requests_total{status=~"5.."}[5m]) / rate(aicr_http_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} for the last 5 minutes"

      # High latency
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(aicr_inference_duration_seconds_bucket[5m])) > 1.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High inference latency"
          description: "95th percentile latency is {{ $value }}s"

      # Memory usage
      - alert: HighMemoryUsage
        expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.9
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is above 90% for 10 minutes"

      # Disk space
      - alert: DiskSpaceLow
        expr: (node_filesystem_avail_bytes / node_filesystem_size_bytes) < 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Low disk space"
          description: "Disk space is below 10% on {{ $labels.device }}"

      # Database connection
      - alert: DatabaseConnectionHigh
        expr: pg_stat_activity_count > 180
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High database connections"
          description: "Database has {{ $value }} active connections"

  - name: model.rules
    rules:
      # Model deployment failure
      - alert: ModelDeploymentFailed
        expr: aicr_model_deployment_status{status="failed"} > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Model deployment failed"
          description: "Model {{ $labels.model_id }} deployment failed"

      # Model inference queue
      - alert: ModelInferenceQueueHigh
        expr: aicr_inference_queue_length > 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High inference queue length"
          description: "Inference queue length is {{ $value }} for model {{ $labels.model_id }}"
```

### Grafana Dashboards

```json
{
  "dashboard": {
    "title": "AICR Production Overview",
    "panels": [
      {
        "title": "Service Health",
        "type": "stat",
        "targets": [
          {
            "expr": "up{job=\"aicr\"}",
            "legendFormat": "{{ instance }}"
          }
        ]
      },
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(aicr_http_requests_total[5m])",
            "legendFormat": "{{ method }} {{ status }}"
          }
        ]
      },
      {
        "title": "Inference Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(aicr_inference_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(aicr_inference_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          }
        ]
      },
      {
        "title": "Model Performance",
        "type": "table",
        "targets": [
          {
            "expr": "aicr_model_accuracy",
            "legendFormat": "{{ model_id }}"
          }
        ]
      }
    ]
  }
}
```

## Backup and Disaster Recovery

### Backup Strategy

```bash
#!/bin/bash
# AICR Backup Script

BACKUP_DIR="/opt/backups/aicr"
S3_BUCKET="s3://aicr-production-backups"
RETENTION_DAYS=30

# Database backup
echo "Starting database backup..."
pg_dump -h localhost -U aicr_user -d aicr_production | gzip > "$BACKUP_DIR/db_$(date +%Y%m%d_%H%M%S).sql.gz"

# Model artifacts backup
echo "Starting model artifacts backup..."
rsync -av /opt/aicr/models/ "$BACKUP_DIR/models/"

# Configuration backup
echo "Starting configuration backup..."
tar -czf "$BACKUP_DIR/config_$(date +%Y%m%d_%H%M%S).tar.gz" /etc/aicr/

# Upload to S3
echo "Uploading to S3..."
aws s3 sync "$BACKUP_DIR/" "$S3_BUCKET/$(date +%Y/%m/%d)/"

# Cleanup old backups
echo "Cleaning up old backups..."
find "$BACKUP_DIR" -name "*.gz" -mtime +$RETENTION_DAYS -delete
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +$RETENTION_DAYS -delete

echo "Backup completed successfully"
```

### Disaster Recovery Plan

```yaml
# Disaster Recovery Configuration
disaster_recovery:
  rto: "2 hours"      # Recovery Time Objective
  rpo: "15 minutes"   # Recovery Point Objective

  backup_strategy:
    database:
      method: "continuous_archiving"
      frequency: "15 minutes"
      retention: "30 days"

    models:
      method: "incremental_sync"
      frequency: "1 hour"
      retention: "90 days"

    configuration:
      method: "git_based"
      frequency: "on_change"
      retention: "indefinite"

  recovery_procedures:
    database_recovery:
      - "Stop AICR services"
      - "Restore database from backup"
      - "Replay WAL files to target time"
      - "Update connection strings"
      - "Start AICR services"
      - "Verify data integrity"

    full_site_recovery:
      - "Provision infrastructure"
      - "Deploy AICR from infrastructure as code"
      - "Restore database"
      - "Restore model artifacts"
      - "Update DNS records"
      - "Validate all services"

  testing:
    frequency: "quarterly"
    scope: "full_recovery"
    documentation_required: true
```

## Operational Procedures

### Deployment Procedures

```bash
#!/bin/bash
# Production Deployment Script

set -euo pipefail

ENVIRONMENT="production"
NAMESPACE="aicr-production"
IMAGE_TAG="${1:-latest}"

echo "Starting AICR production deployment..."

# Pre-deployment checks
echo "Running pre-deployment checks..."
kubectl get nodes --show-labels
kubectl get pods -n $NAMESPACE
helm list -n $NAMESPACE

# Database migration (if needed)
echo "Checking for database migrations..."
kubectl run migration-check --image=aicr:$IMAGE_TAG --rm -i --restart=Never -- python manage.py migrate --check

# Rolling deployment
echo "Starting rolling deployment..."
helm upgrade aicr ./helm/aicr \
  --namespace $NAMESPACE \
  --set image.tag=$IMAGE_TAG \
  --set environment=$ENVIRONMENT \
  --wait \
  --timeout=600s

# Post-deployment verification
echo "Running post-deployment verification..."
kubectl rollout status deployment/aicr-service -n $NAMESPACE
kubectl get pods -n $NAMESPACE -l app=aicr

# Health checks
echo "Running health checks..."
for i in {1..30}; do
  if curl -sf http://aicr-service/health; then
    echo "Health check passed"
    break
  fi
  echo "Waiting for service to be ready... ($i/30)"
  sleep 10
done

# Smoke tests
echo "Running smoke tests..."
kubectl run smoke-test --image=aicr:$IMAGE_TAG --rm -i --restart=Never -- python -m pytest tests/smoke/

echo "Deployment completed successfully!"
```

### Maintenance Procedures

```bash
#!/bin/bash
# Maintenance procedures for AICR

# Rolling restart of services
rolling_restart() {
  echo "Performing rolling restart..."
  kubectl rollout restart deployment/aicr-service -n aicr-production
  kubectl rollout status deployment/aicr-service -n aicr-production
}

# Update SSL certificates
update_ssl_certificates() {
  echo "Updating SSL certificates..."
  certbot renew --quiet
  kubectl create secret tls aicr-tls \
    --cert=/etc/letsencrypt/live/api.datacraft.co.ke/fullchain.pem \
    --key=/etc/letsencrypt/live/api.datacraft.co.ke/privkey.pem \
    --dry-run=client -o yaml | kubectl apply -f -
}

# Database maintenance
database_maintenance() {
  echo "Running database maintenance..."
  kubectl exec -it postgres-primary -- psql -U postgres -d aicr_production -c "VACUUM ANALYZE;"
  kubectl exec -it postgres-primary -- psql -U postgres -d aicr_production -c "REINDEX DATABASE aicr_production;"
}

# Log rotation
rotate_logs() {
  echo "Rotating logs..."
  kubectl exec -it deployment/aicr-service -- logrotate /etc/logrotate.d/aicr
}

# Performance optimization
optimize_performance() {
  echo "Running performance optimization..."
  # Clear model cache
  curl -X POST http://aicr-service/admin/cache/clear

  # Restart inference engines
  curl -X POST http://aicr-service/admin/inference/restart

  # Garbage collection
  kubectl exec -it deployment/aicr-service -- python -c "import gc; gc.collect()"
}

# Security updates
security_updates() {
  echo "Applying security updates..."
  kubectl set image deployment/aicr-service aicr=aicr:latest-security
  kubectl rollout status deployment/aicr-service
}
```

---

**Next Steps:**
- [Docker Deployment](docker_deployment.md) - Containerized deployment guide
- [Kubernetes Deployment](kubernetes_deployment.md) - Kubernetes orchestration
- [Monitoring Guide](../guides/monitoring_guide.md) - Comprehensive monitoring setup
- [Troubleshooting](../troubleshooting.md) - Common issues and solutions