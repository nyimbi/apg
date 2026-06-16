"""
Production Configuration Management for AICR

This module provides comprehensive production configuration management including:
- Environment-specific configuration management
- Kubernetes deployment configuration
- Docker containerization settings
- Load balancer and ingress configuration
- Auto-scaling and resource management
- Health checks and monitoring configuration
- Security and compliance settings
- Database and cache configuration
- Logging and telemetry setup

Author: Nyimbi Odero <nyimbi@gmail.com>
Copyright: © 2025 Datacraft
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from uuid_extensions import uuid7str


class Environment(str, Enum):
	"""Deployment environments."""
	DEVELOPMENT = "development"
	STAGING = "staging"
	PRODUCTION = "production"
	DISASTER_RECOVERY = "disaster_recovery"


class ScalingPolicy(str, Enum):
	"""Auto-scaling policies."""
	CPU_BASED = "cpu_based"
	MEMORY_BASED = "memory_based"
	REQUEST_BASED = "request_based"
	CUSTOM_METRIC = "custom_metric"


class DatabaseEngine(str, Enum):
	"""Database engines."""
	POSTGRESQL = "postgresql"
	MYSQL = "mysql"
	ORACLE = "oracle"
	SQL_SERVER = "sql_server"


class CacheEngine(str, Enum):
	"""Cache engines."""
	REDIS = "redis"
	MEMCACHED = "memcached"
	HAZELCAST = "hazelcast"


class LoadBalancerType(str, Enum):
	"""Load balancer types."""
	NGINX = "nginx"
	HAPROXY = "haproxy"
	AWS_ALB = "aws_alb"
	AZURE_LB = "azure_lb"
	GCP_LB = "gcp_lb"


class ResourceLimits(BaseModel):
	"""Resource limits configuration."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	cpu: str = "2"
	memory: str = "4Gi"
	gpu: Optional[str] = None
	storage: str = "20Gi"
	max_connections: int = 1000
	max_file_descriptors: int = 65536


class ResourceRequests(BaseModel):
	"""Resource requests configuration."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	cpu: str = "1"
	memory: str = "2Gi"
	gpu: Optional[str] = None
	storage: str = "10Gi"


class AutoScalingConfig(BaseModel):
	"""Auto-scaling configuration."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	enabled: bool = True
	min_replicas: int = 3
	max_replicas: int = 20
	target_cpu_utilization: int = 70
	target_memory_utilization: int = 80
	scale_up_threshold: int = 85
	scale_down_threshold: int = 30
	scale_up_cooldown_seconds: int = 300
	scale_down_cooldown_seconds: int = 600
	custom_metrics: List[Dict[str, Any]] = Field(default_factory=list)


class HealthCheckConfig(BaseModel):
	"""Health check configuration."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	enabled: bool = True
	path: str = "/health"
	port: int = 8080
	interval_seconds: int = 30
	timeout_seconds: int = 10
	healthy_threshold: int = 2
	unhealthy_threshold: int = 3
	initial_delay_seconds: int = 30
	failure_threshold: int = 3
	success_threshold: int = 1


class DatabaseConfig(BaseModel):
	"""Database configuration."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	engine: DatabaseEngine = DatabaseEngine.POSTGRESQL
	host: str
	port: int = 5432
	database: str = "aicr_production"
	username: str
	password: str
	pool_size: int = 20
	max_overflow: int = 10
	pool_timeout: int = 30
	pool_recycle: int = 3600
	ssl_mode: str = "require"
	ssl_cert_path: Optional[str] = None
	backup_enabled: bool = True
	backup_retention_days: int = 30
	read_replicas: List[str] = Field(default_factory=list)


class CacheConfig(BaseModel):
	"""Cache configuration."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	engine: CacheEngine = CacheEngine.REDIS
	host: str
	port: int = 6379
	password: Optional[str] = None
	database: int = 0
	ssl_enabled: bool = True
	connection_pool_size: int = 50
	socket_timeout: int = 30
	socket_connect_timeout: int = 30
	retry_on_timeout: bool = True
	cluster_mode: bool = False
	cluster_nodes: List[str] = Field(default_factory=list)
	sentinel_hosts: List[str] = Field(default_factory=list)
	max_memory_policy: str = "allkeys-lru"


class SecurityConfig(BaseModel):
	"""Security configuration."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	tls_enabled: bool = True
	tls_cert_path: str = "/etc/ssl/certs/aicr.crt"
	tls_key_path: str = "/etc/ssl/private/aicr.key"
	ca_cert_path: Optional[str] = None
	cipher_suites: List[str] = Field(default_factory=lambda: [
		"TLS_AES_256_GCM_SHA384",
		"TLS_CHACHA20_POLY1305_SHA256",
		"TLS_AES_128_GCM_SHA256"
	])
	min_tls_version: str = "1.2"
	hsts_enabled: bool = True
	hsts_max_age: int = 31536000
	cors_enabled: bool = True
	cors_origins: List[str] = Field(default_factory=lambda: ["https://*.datacraft.co.ke"])
	rate_limiting_enabled: bool = True
	rate_limit_requests_per_minute: int = 1000
	rate_limit_burst_size: int = 100
	api_key_required: bool = True
	jwt_secret_key: str
	jwt_algorithm: str = "HS256"
	jwt_expiration_hours: int = 24
	password_policy: Dict[str, Any] = Field(default_factory=lambda: {
		"min_length": 12,
		"require_uppercase": True,
		"require_lowercase": True,
		"require_numbers": True,
		"require_special_chars": True
	})


class LoggingConfig(BaseModel):
	"""Logging configuration."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	level: str = "INFO"
	format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
	file_path: str = "/var/log/aicr/aicr.log"
	max_file_size_mb: int = 100
	backup_count: int = 10
	json_format: bool = True
	structured_logging: bool = True
	correlation_id_enabled: bool = True
	sensitive_data_masking: bool = True
	audit_logging_enabled: bool = True
	audit_log_path: str = "/var/log/aicr/audit.log"
	performance_logging_enabled: bool = True
	error_tracking_enabled: bool = True
	log_sampling_rate: float = 1.0


class MonitoringConfig(BaseModel):
	"""Monitoring configuration."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	metrics_enabled: bool = True
	metrics_port: int = 9090
	metrics_path: str = "/metrics"
	prometheus_enabled: bool = True
	grafana_enabled: bool = True
	alerting_enabled: bool = True
	alert_manager_url: str
	notification_channels: List[str] = Field(default_factory=lambda: ["email", "slack"])
	custom_metrics_enabled: bool = True
	tracing_enabled: bool = True
	tracing_sample_rate: float = 0.1
	profiling_enabled: bool = False
	health_check_interval_seconds: int = 30


class LoadBalancerConfig(BaseModel):
	"""Load balancer configuration."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	type: LoadBalancerType = LoadBalancerType.NGINX
	external_ip: Optional[str] = None
	port: int = 443
	ssl_termination: bool = True
	session_affinity: bool = False
	algorithm: str = "round_robin"
	health_check_enabled: bool = True
	timeout_seconds: int = 60
	max_connections: int = 10000
	connection_timeout: int = 5
	client_timeout: int = 50
	server_timeout: int = 50
	retry_attempts: int = 3
	circuit_breaker_enabled: bool = True
	rate_limiting_enabled: bool = True
	compression_enabled: bool = True
	caching_enabled: bool = True


class NetworkConfig(BaseModel):
	"""Network configuration."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	vpc_cidr: str = "10.0.0.0/16"
	public_subnets: List[str] = Field(default_factory=lambda: [
		"10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"
	])
	private_subnets: List[str] = Field(default_factory=lambda: [
		"10.0.10.0/24", "10.0.11.0/24", "10.0.12.0/24"
	])
	database_subnets: List[str] = Field(default_factory=lambda: [
		"10.0.20.0/24", "10.0.21.0/24", "10.0.22.0/24"
	])
	availability_zones: List[str] = Field(default_factory=lambda: [
		"us-west-2a", "us-west-2b", "us-west-2c"
	])
	nat_gateway_enabled: bool = True
	internet_gateway_enabled: bool = True
	vpc_peering_enabled: bool = False
	private_dns_enabled: bool = True


class BackupConfig(BaseModel):
	"""Backup configuration."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	enabled: bool = True
	schedule: str = "0 2 * * *"  # Daily at 2 AM
	retention_days: int = 30
	encryption_enabled: bool = True
	compression_enabled: bool = True
	backup_location: str = "s3://aicr-production-backups"
	cross_region_replication: bool = True
	point_in_time_recovery: bool = True
	backup_verification: bool = True
	automated_restore_testing: bool = True


class DisasterRecoveryConfig(BaseModel):
	"""Disaster recovery configuration."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	enabled: bool = True
	secondary_region: str = "us-east-1"
	rto_minutes: int = 120  # Recovery Time Objective
	rpo_minutes: int = 15   # Recovery Point Objective
	automated_failover: bool = True
	cross_region_backup: bool = True
	data_replication_enabled: bool = True
	warm_standby: bool = True
	dns_failover_enabled: bool = True
	monitoring_enabled: bool = True
	testing_schedule: str = "0 0 1 * *"  # Monthly


class PerformanceConfig(BaseModel):
	"""Performance configuration."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	model_caching_enabled: bool = True
	model_cache_size_gb: int = 8
	model_preloading_enabled: bool = True
	batch_processing_enabled: bool = True
	max_batch_size: int = 64
	batch_timeout_ms: int = 50
	connection_pooling_enabled: bool = True
	async_processing_enabled: bool = True
	gpu_acceleration_enabled: bool = True
	mixed_precision_enabled: bool = True
	model_optimization_enabled: bool = True
	jit_compilation_enabled: bool = True
	memory_optimization_enabled: bool = True


class ProductionConfig(BaseModel):
	"""Complete production configuration."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	config_id: str = Field(default_factory=uuid7str)
	environment: Environment = Environment.PRODUCTION
	region: str = "us-west-2"
	availability_zone: str = "us-west-2a"
	cluster_name: str = "aicr-production"
	namespace: str = "aicr-production"

	# Core configuration
	resource_limits: ResourceLimits = Field(default_factory=ResourceLimits)
	resource_requests: ResourceRequests = Field(default_factory=ResourceRequests)
	auto_scaling: AutoScalingConfig = Field(default_factory=AutoScalingConfig)
	health_check: HealthCheckConfig = Field(default_factory=HealthCheckConfig)

	# Infrastructure configuration
	database: DatabaseConfig
	cache: CacheConfig
	security: SecurityConfig
	logging: LoggingConfig = Field(default_factory=LoggingConfig)
	monitoring: MonitoringConfig
	load_balancer: LoadBalancerConfig = Field(default_factory=LoadBalancerConfig)
	network: NetworkConfig = Field(default_factory=NetworkConfig)

	# Operational configuration
	backup: BackupConfig = Field(default_factory=BackupConfig)
	disaster_recovery: DisasterRecoveryConfig = Field(default_factory=DisasterRecoveryConfig)
	performance: PerformanceConfig = Field(default_factory=PerformanceConfig)

	# Application-specific configuration
	application: Dict[str, Any] = Field(default_factory=dict)
	feature_flags: Dict[str, bool] = Field(default_factory=dict)
	integrations: Dict[str, Any] = Field(default_factory=dict)

	# Metadata
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	version: str = "1.0.0"
	created_by: str = "system"
	tags: Dict[str, str] = Field(default_factory=dict)


class ConfigurationManager:
	"""Production configuration manager."""

	def __init__(self, config_dir: str = "/etc/aicr"):
		self.config_dir = Path(config_dir)
		self.logger = logging.getLogger(f"{__name__}.ConfigurationManager")
		self._config_cache: Dict[str, ProductionConfig] = {}

	def create_production_config(
		self,
		environment: Environment,
		database_config: DatabaseConfig,
		monitoring_config: MonitoringConfig,
		security_config: SecurityConfig,
		**kwargs
	) -> ProductionConfig:
		"""Create production configuration."""
		try:
			config = ProductionConfig(
				environment=environment,
				database=database_config,
				monitoring=monitoring_config,
				security=security_config,
				**kwargs
			)

			# Apply environment-specific defaults
			self._apply_environment_defaults(config)

			# Validate configuration
			self._validate_config(config)

			self.logger.info(f"Production configuration created for environment: {environment}")
			return config

		except Exception as e:
			self.logger.error(f"Failed to create production config: {e}")
			raise

	def save_config(self, config: ProductionConfig, filename: Optional[str] = None) -> Path:
		"""Save configuration to file."""
		try:
			if not filename:
				filename = f"aicr-{config.environment.value}-{config.version}.yaml"

			config_path = self.config_dir / filename
			config_path.parent.mkdir(parents=True, exist_ok=True)

			# Convert to dictionary and save as YAML
			config_dict = config.model_dump()

			with open(config_path, 'w') as f:
				yaml.dump(config_dict, f, default_flow_style=False, indent=2)

			# Cache the configuration
			self._config_cache[str(config_path)] = config

			self.logger.info(f"Configuration saved to: {config_path}")
			return config_path

		except Exception as e:
			self.logger.error(f"Failed to save configuration: {e}")
			raise

	def load_config(self, config_path: Union[str, Path]) -> ProductionConfig:
		"""Load configuration from file."""
		try:
			config_path = Path(config_path)

			# Check cache first
			if str(config_path) in self._config_cache:
				return self._config_cache[str(config_path)]

			if not config_path.exists():
				raise FileNotFoundError(f"Configuration file not found: {config_path}")

			with open(config_path, 'r') as f:
				config_dict = yaml.safe_load(f)

			config = ProductionConfig(**config_dict)

			# Cache the configuration
			self._config_cache[str(config_path)] = config

			self.logger.info(f"Configuration loaded from: {config_path}")
			return config

		except Exception as e:
			self.logger.error(f"Failed to load configuration: {e}")
			raise

	def generate_kubernetes_manifests(self, config: ProductionConfig) -> Dict[str, str]:
		"""Generate Kubernetes manifests from configuration."""
		try:
			manifests = {}

			# Namespace
			manifests['namespace.yaml'] = self._generate_namespace_manifest(config)

			# ConfigMap
			manifests['configmap.yaml'] = self._generate_configmap_manifest(config)

			# Secret
			manifests['secret.yaml'] = self._generate_secret_manifest(config)

			# Deployment
			manifests['deployment.yaml'] = self._generate_deployment_manifest(config)

			# Service
			manifests['service.yaml'] = self._generate_service_manifest(config)

			# Ingress
			manifests['ingress.yaml'] = self._generate_ingress_manifest(config)

			# HorizontalPodAutoscaler
			manifests['hpa.yaml'] = self._generate_hpa_manifest(config)

			# PersistentVolumeClaim
			manifests['pvc.yaml'] = self._generate_pvc_manifest(config)

			# ServiceMonitor (for Prometheus)
			manifests['servicemonitor.yaml'] = self._generate_servicemonitor_manifest(config)

			self.logger.info("Kubernetes manifests generated successfully")
			return manifests

		except Exception as e:
			self.logger.error(f"Failed to generate Kubernetes manifests: {e}")
			raise

	def generate_docker_compose(self, config: ProductionConfig) -> str:
		"""Generate Docker Compose configuration."""
		try:
			compose_config = {
				'version': '3.8',
				'services': {
					'aicr': {
						'image': 'datacraft/aicr:latest',
						'container_name': 'aicr-production',
						'restart': 'unless-stopped',
						'ports': [f'{config.load_balancer.port}:8080'],
						'environment': self._generate_environment_variables(config),
						'volumes': [
							'/var/log/aicr:/var/log/aicr',
							'/etc/ssl/aicr:/etc/ssl/aicr:ro',
							'/data/aicr:/data/aicr'
						],
						'networks': ['aicr-network'],
						'healthcheck': {
							'test': [
								'CMD',
								'curl',
								'-f',
								f'http://localhost:8080{config.health_check.path}'
							],
							'interval': f'{config.health_check.interval_seconds}s',
							'timeout': f'{config.health_check.timeout_seconds}s',
							'retries': config.health_check.unhealthy_threshold,
							'start_period': f'{config.health_check.initial_delay_seconds}s'
						},
						'deploy': {
							'resources': {
								'limits': {
									'cpus': config.resource_limits.cpu,
									'memory': config.resource_limits.memory
								},
								'reservations': {
									'cpus': config.resource_requests.cpu,
									'memory': config.resource_requests.memory
								}
							},
							'replicas': config.auto_scaling.min_replicas
						}
					},
					'postgres': {
						'image': 'postgres:15-alpine',
						'container_name': 'aicr-postgres',
						'restart': 'unless-stopped',
						'environment': {
							'POSTGRES_DB': config.database.database,
							'POSTGRES_USER': config.database.username,
							'POSTGRES_PASSWORD': config.database.password
						},
						'volumes': [
							'postgres_data:/var/lib/postgresql/data',
							'./init.sql:/docker-entrypoint-initdb.d/init.sql'
						],
						'networks': ['aicr-network'],
						'ports': [f'{config.database.port}:5432']
					},
					'redis': {
						'image': 'redis:7-alpine',
						'container_name': 'aicr-redis',
						'restart': 'unless-stopped',
						'command': [
							'redis-server',
							'--requirepass',
							config.cache.password or 'redis_password',
							'--maxmemory',
							'2gb',
							'--maxmemory-policy',
							config.cache.max_memory_policy
						],
						'volumes': ['redis_data:/data'],
						'networks': ['aicr-network'],
						'ports': [f'{config.cache.port}:6379']
					},
					'nginx': {
						'image': 'nginx:alpine',
						'container_name': 'aicr-nginx',
						'restart': 'unless-stopped',
						'ports': ['80:80', '443:443'],
						'volumes': [
							'./nginx.conf:/etc/nginx/nginx.conf:ro',
							'/etc/ssl/aicr:/etc/ssl/aicr:ro'
						],
						'networks': ['aicr-network'],
						'depends_on': ['aicr']
					}
				},
				'networks': {
					'aicr-network': {
						'driver': 'bridge'
					}
				},
				'volumes': {
					'postgres_data': {},
					'redis_data': {}
				}
			}

			return yaml.dump(compose_config, default_flow_style=False, indent=2)

		except Exception as e:
			self.logger.error(f"Failed to generate Docker Compose: {e}")
			raise

	def generate_nginx_config(self, config: ProductionConfig) -> str:
		"""Generate Nginx configuration."""
		try:
			nginx_config = f"""
events {{
    worker_connections 1024;
}}

http {{
    upstream aicr_backend {{
        server aicr:8080;
        keepalive 32;
    }}

    server {{
        listen 80;
        server_name api.datacraft.co.ke;
        return 301 https://$server_name$request_uri;
    }}

    server {{
        listen 443 ssl http2;
        server_name api.datacraft.co.ke;

        # SSL Configuration
        ssl_certificate {config.security.tls_cert_path};
        ssl_certificate_key {config.security.tls_key_path};
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;
        ssl_session_cache shared:SSL:10m;
        ssl_session_timeout 10m;

        # Security Headers
        add_header Strict-Transport-Security "max-age={config.security.hsts_max_age}; includeSubDomains" always;
        add_header X-Frame-Options DENY always;
        add_header X-Content-Type-Options nosniff always;
        add_header X-XSS-Protection "1; mode=block" always;
        add_header Referrer-Policy "strict-origin-when-cross-origin" always;

        # Rate Limiting
        limit_req_zone $binary_remote_addr zone=api:10m rate={config.security.rate_limit_requests_per_minute}r/m;
        limit_req zone=api burst={config.security.rate_limit_burst_size} nodelay;

        # Health Check
        location {config.health_check.path} {{
            proxy_pass http://aicr_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            access_log off;
        }}

        # API Routes
        location /api/ {{
            proxy_pass http://aicr_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_connect_timeout {config.load_balancer.connection_timeout}s;
            proxy_send_timeout {config.load_balancer.timeout_seconds}s;
            proxy_read_timeout {config.load_balancer.timeout_seconds}s;
            proxy_buffering off;
        }}

        # WebSocket Support
        location /ws/ {{
            proxy_pass http://aicr_backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }}

        # Metrics (Prometheus)
        location {config.monitoring.metrics_path} {{
            proxy_pass http://aicr_backend;
            proxy_set_header Host $host;
            allow 10.0.0.0/8;
            allow 172.16.0.0/12;
            allow 192.168.0.0/16;
            deny all;
        }}
    }}
}}
"""

			return nginx_config.strip()

		except Exception as e:
			self.logger.error(f"Failed to generate Nginx config: {e}")
			raise

	def _apply_environment_defaults(self, config: ProductionConfig) -> None:
		"""Apply environment-specific defaults."""
		if config.environment == Environment.PRODUCTION:
			# Production-specific settings
			config.auto_scaling.min_replicas = max(config.auto_scaling.min_replicas, 3)
			config.resource_limits.cpu = "4"
			config.resource_limits.memory = "8Gi"
			config.logging.level = "INFO"
			config.monitoring.tracing_sample_rate = 0.1
			config.backup.enabled = True
			config.disaster_recovery.enabled = True

		elif config.environment == Environment.STAGING:
			# Staging-specific settings
			config.auto_scaling.min_replicas = 2
			config.resource_limits.cpu = "2"
			config.resource_limits.memory = "4Gi"
			config.logging.level = "DEBUG"
			config.monitoring.tracing_sample_rate = 0.5
			config.backup.enabled = True
			config.disaster_recovery.enabled = False

		elif config.environment == Environment.DEVELOPMENT:
			# Development-specific settings
			config.auto_scaling.min_replicas = 1
			config.resource_limits.cpu = "1"
			config.resource_limits.memory = "2Gi"
			config.logging.level = "DEBUG"
			config.monitoring.tracing_sample_rate = 1.0
			config.backup.enabled = False
			config.disaster_recovery.enabled = False

	def _validate_config(self, config: ProductionConfig) -> None:
		"""Validate configuration."""
		# Validate resource constraints
		if config.auto_scaling.min_replicas > config.auto_scaling.max_replicas:
			raise ValueError("min_replicas cannot be greater than max_replicas")

		# Validate network configuration
		if not config.network.public_subnets:
			raise ValueError("At least one public subnet is required")

		# Validate security configuration
		if config.security.tls_enabled and not config.security.tls_cert_path:
			raise ValueError("TLS certificate path is required when TLS is enabled")

		# Validate database configuration
		if not config.database.host:
			raise ValueError("Database host is required")

		# Validate monitoring configuration
		if config.monitoring.alerting_enabled and not config.monitoring.alert_manager_url:
			raise ValueError("Alert manager URL is required when alerting is enabled")

	def _generate_namespace_manifest(self, config: ProductionConfig) -> str:
		"""Generate Kubernetes namespace manifest."""
		return f"""apiVersion: v1
kind: Namespace
metadata:
  name: {config.namespace}
  labels:
    app: aicr
    environment: {config.environment.value}
    version: {config.version}
"""

	def _generate_configmap_manifest(self, config: ProductionConfig) -> str:
		"""Generate Kubernetes ConfigMap manifest."""
		config_data = {
			'ENVIRONMENT': config.environment.value,
			'LOG_LEVEL': config.logging.level,
			'METRICS_PORT': str(config.monitoring.metrics_port),
			'HEALTH_CHECK_PATH': config.health_check.path,
			'DATABASE_HOST': config.database.host,
			'DATABASE_PORT': str(config.database.port),
			'DATABASE_NAME': config.database.database,
			'CACHE_HOST': config.cache.host,
			'CACHE_PORT': str(config.cache.port),
			'CACHE_DATABASE': str(config.cache.database)
		}

		config_yaml = yaml.dump(config_data, default_flow_style=False, indent=2)

		return f"""apiVersion: v1
kind: ConfigMap
metadata:
  name: aicr-config
  namespace: {config.namespace}
  labels:
    app: aicr
    environment: {config.environment.value}
data:
{config_yaml}"""

	def _generate_secret_manifest(self, config: ProductionConfig) -> str:
		"""Generate Kubernetes Secret manifest."""
		import base64

		secret_data = {
			'DATABASE_PASSWORD': base64.b64encode(config.database.password.encode()).decode(),
			'JWT_SECRET_KEY': base64.b64encode(config.security.jwt_secret_key.encode()).decode()
		}

		if config.cache.password:
			secret_data['CACHE_PASSWORD'] = base64.b64encode(config.cache.password.encode()).decode()

		secret_yaml = yaml.dump(secret_data, default_flow_style=False, indent=2)

		return f"""apiVersion: v1
kind: Secret
metadata:
  name: aicr-secrets
  namespace: {config.namespace}
  labels:
    app: aicr
    environment: {config.environment.value}
type: Opaque
data:
{secret_yaml}"""

	def _generate_deployment_manifest(self, config: ProductionConfig) -> str:
		"""Generate Kubernetes Deployment manifest."""
		return f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: aicr
  namespace: {config.namespace}
  labels:
    app: aicr
    environment: {config.environment.value}
spec:
  replicas: {config.auto_scaling.min_replicas}
  selector:
    matchLabels:
      app: aicr
  template:
    metadata:
      labels:
        app: aicr
        environment: {config.environment.value}
    spec:
      containers:
      - name: aicr
        image: datacraft/aicr:latest
        ports:
        - containerPort: 8080
          name: http
        - containerPort: {config.monitoring.metrics_port}
          name: metrics
        envFrom:
        - configMapRef:
            name: aicr-config
        - secretRef:
            name: aicr-secrets
        resources:
          limits:
            cpu: {config.resource_limits.cpu}
            memory: {config.resource_limits.memory}
          requests:
            cpu: {config.resource_requests.cpu}
            memory: {config.resource_requests.memory}
        livenessProbe:
          httpGet:
            path: {config.health_check.path}
            port: 8080
          initialDelaySeconds: {config.health_check.initial_delay_seconds}
          periodSeconds: {config.health_check.interval_seconds}
          timeoutSeconds: {config.health_check.timeout_seconds}
          failureThreshold: {config.health_check.failure_threshold}
        readinessProbe:
          httpGet:
            path: {config.health_check.path}
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: {config.health_check.timeout_seconds}
          successThreshold: {config.health_check.success_threshold}
        volumeMounts:
        - name: logs
          mountPath: /var/log/aicr
        - name: ssl-certs
          mountPath: /etc/ssl/aicr
          readOnly: true
      volumes:
      - name: logs
        emptyDir: {{}}
      - name: ssl-certs
        secret:
          secretName: aicr-ssl-certs
"""

	def _generate_service_manifest(self, config: ProductionConfig) -> str:
		"""Generate Kubernetes Service manifest."""
		return f"""apiVersion: v1
kind: Service
metadata:
  name: aicr-service
  namespace: {config.namespace}
  labels:
    app: aicr
    environment: {config.environment.value}
spec:
  selector:
    app: aicr
  ports:
  - name: http
    port: 80
    targetPort: 8080
    protocol: TCP
  - name: metrics
    port: {config.monitoring.metrics_port}
    targetPort: {config.monitoring.metrics_port}
    protocol: TCP
  type: ClusterIP
"""

	def _generate_ingress_manifest(self, config: ProductionConfig) -> str:
		"""Generate Kubernetes Ingress manifest."""
		return f"""apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: aicr-ingress
  namespace: {config.namespace}
  labels:
    app: aicr
    environment: {config.environment.value}
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    nginx.ingress.kubernetes.io/backend-protocol: "HTTP"
    nginx.ingress.kubernetes.io/rate-limit: "{config.security.rate_limit_requests_per_minute}"
    nginx.ingress.kubernetes.io/rate-limit-rps: "{config.security.rate_limit_requests_per_minute // 60}"
spec:
  tls:
  - hosts:
    - api.datacraft.co.ke
    secretName: aicr-tls
  rules:
  - host: api.datacraft.co.ke
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: aicr-service
            port:
              number: 80
"""

	def _generate_hpa_manifest(self, config: ProductionConfig) -> str:
		"""Generate Kubernetes HorizontalPodAutoscaler manifest."""
		return f"""apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: aicr-hpa
  namespace: {config.namespace}
  labels:
    app: aicr
    environment: {config.environment.value}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: aicr
  minReplicas: {config.auto_scaling.min_replicas}
  maxReplicas: {config.auto_scaling.max_replicas}
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: {config.auto_scaling.target_cpu_utilization}
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: {config.auto_scaling.target_memory_utilization}
  behavior:
    scaleUp:
      stabilizationWindowSeconds: {config.auto_scaling.scale_up_cooldown_seconds}
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: {config.auto_scaling.scale_down_cooldown_seconds}
      policies:
      - type: Percent
        value: 50
        periodSeconds: 300
"""

	def _generate_pvc_manifest(self, config: ProductionConfig) -> str:
		"""Generate Kubernetes PersistentVolumeClaim manifest."""
		return f"""apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: aicr-storage
  namespace: {config.namespace}
  labels:
    app: aicr
    environment: {config.environment.value}
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: {config.resource_requests.storage}
  storageClassName: gp2
"""

	def _generate_servicemonitor_manifest(self, config: ProductionConfig) -> str:
		"""Generate Kubernetes ServiceMonitor manifest for Prometheus."""
		return f"""apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: aicr-metrics
  namespace: {config.namespace}
  labels:
    app: aicr
    environment: {config.environment.value}
spec:
  selector:
    matchLabels:
      app: aicr
  endpoints:
  - port: metrics
    path: {config.monitoring.metrics_path}
    interval: 30s
    scrapeTimeout: 10s
"""

	def _generate_environment_variables(self, config: ProductionConfig) -> Dict[str, str]:
		"""Generate environment variables for containers."""
		return {
			'ENVIRONMENT': config.environment.value,
			'LOG_LEVEL': config.logging.level,
			'DATABASE_HOST': config.database.host,
			'DATABASE_PORT': str(config.database.port),
			'DATABASE_NAME': config.database.database,
			'DATABASE_USERNAME': config.database.username,
			'CACHE_HOST': config.cache.host,
			'CACHE_PORT': str(config.cache.port),
			'CACHE_DATABASE': str(config.cache.database),
			'METRICS_PORT': str(config.monitoring.metrics_port),
			'HEALTH_CHECK_PATH': config.health_check.path,
			'TLS_ENABLED': str(config.security.tls_enabled).lower(),
			'CORS_ENABLED': str(config.security.cors_enabled).lower(),
			'RATE_LIMITING_ENABLED': str(config.security.rate_limiting_enabled).lower()
		}


# Example usage
def create_production_configuration():
	"""Create a complete production configuration."""

	# Database configuration
	database_config = DatabaseConfig(
		engine=DatabaseEngine.POSTGRESQL,
		host="aicr-postgres.c9x8d7h6k5l4.us-west-2.rds.amazonaws.com",
		port=5432,
		database="aicr_production",
		username="aicr_user",
		password="secure_db_password_2025",
		pool_size=20,
		ssl_mode="require"
	)

	# Cache configuration
	cache_config = CacheConfig(
		engine=CacheEngine.REDIS,
		host="aicr-redis-cluster.abc123.cache.amazonaws.com",
		port=6379,
		password="secure_redis_password_2025",
		ssl_enabled=True,
		cluster_mode=True
	)

	# Security configuration
	security_config = SecurityConfig(
		tls_enabled=True,
		tls_cert_path="/etc/ssl/certs/aicr.crt",
		tls_key_path="/etc/ssl/private/aicr.key",
		jwt_secret_key="ultra_secure_jwt_secret_key_2025",
		rate_limiting_enabled=True,
		rate_limit_requests_per_minute=1000
	)

	# Monitoring configuration
	monitoring_config = MonitoringConfig(
		metrics_enabled=True,
		prometheus_enabled=True,
		grafana_enabled=True,
		alerting_enabled=True,
		alert_manager_url="https://alertmanager.datacraft.co.ke",
		tracing_enabled=True
	)

	# Create configuration manager
	config_manager = ConfigurationManager()

	# Create production configuration
	prod_config = config_manager.create_production_config(
		environment=Environment.PRODUCTION,
		database_config=database_config,
		monitoring_config=monitoring_config,
		security_config=security_config,
		cache=cache_config,
		region="us-west-2",
		cluster_name="aicr-production-cluster"
	)

	# Save configuration
	config_path = config_manager.save_config(prod_config)
	print(f"Production configuration saved to: {config_path}")

	# Generate Kubernetes manifests
	manifests = config_manager.generate_kubernetes_manifests(prod_config)

	# Save manifests to files
	manifests_dir = Path("./k8s-manifests")
	manifests_dir.mkdir(exist_ok=True)

	for filename, content in manifests.items():
		manifest_path = manifests_dir / filename
		with open(manifest_path, 'w') as f:
			f.write(content)
		print(f"Kubernetes manifest saved: {manifest_path}")

	# Generate Docker Compose
	docker_compose = config_manager.generate_docker_compose(prod_config)
	with open("docker-compose.production.yml", 'w') as f:
		f.write(docker_compose)
	print("Docker Compose configuration saved: docker-compose.production.yml")

	# Generate Nginx configuration
	nginx_config = config_manager.generate_nginx_config(prod_config)
	with open("nginx.conf", 'w') as f:
		f.write(nginx_config)
	print("Nginx configuration saved: nginx.conf")

	return prod_config


if __name__ == "__main__":
	create_production_configuration()