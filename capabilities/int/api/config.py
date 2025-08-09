"""
APG Integration API Management - Configuration Management

Configuration management system for API gateway settings, deployment configurations,
and runtime parameters with environment-specific overrides.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import os
import json
import yaml
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, validator
from pydantic.env_settings import BaseSettings

# =============================================================================
# Configuration Enums and Types
# =============================================================================

class Environment(str, Enum):
	"""Deployment environments."""
	DEVELOPMENT = "development"
	TESTING = "testing" 
	STAGING = "staging"
	PRODUCTION = "production"

class LogLevel(str, Enum):
	"""Logging levels."""
	DEBUG = "DEBUG"
	INFO = "INFO"
	WARNING = "WARNING"
	ERROR = "ERROR"
	CRITICAL = "CRITICAL"

class DatabaseEngine(str, Enum):
	"""Database engine types."""
	POSTGRESQL = "postgresql"
	MYSQL = "mysql"
	SQLITE = "sqlite"

class CacheEngine(str, Enum):
	"""Cache engine types."""
	REDIS = "redis"
	MEMCACHED = "memcached"
	MEMORY = "memory"

# =============================================================================
# Configuration Models
# =============================================================================

class DatabaseConfig(BaseModel):
	"""Database configuration."""
	
	engine: DatabaseEngine = DatabaseEngine.POSTGRESQL
	host: str = "localhost"
	port: int = 5432
	database: str = "apg_api_management"
	username: str = "postgres"
	password: str = "password"
	pool_size: int = 20
	max_overflow: int = 30
	pool_timeout: int = 30
	pool_recycle: int = 3600
	echo: bool = False
	
	@property
	def url(self) -> str:
		"""Get database URL."""
		return f"{self.engine.value}://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

class RedisConfig(BaseModel):
	"""Redis configuration."""
	
	host: str = "localhost"
	port: int = 6379
	database: int = 0
	password: Optional[str] = None
	ssl: bool = False
	ssl_cert_reqs: Optional[str] = None
	max_connections: int = 100
	retry_on_timeout: bool = True
	socket_timeout: int = 5
	socket_connect_timeout: int = 5
	
	@property
	def url(self) -> str:
		"""Get Redis URL."""
		scheme = "rediss" if self.ssl else "redis"
		auth = f":{self.password}@" if self.password else ""
		return f"{scheme}://{auth}{self.host}:{self.port}/{self.database}"

class GatewayConfig(BaseModel):
	"""Gateway runtime configuration."""
	
	host: str = "0.0.0.0"
	port: int = 8080
	workers: int = 4
	max_requests_per_worker: int = 10000
	worker_timeout: int = 300
	keepalive: int = 2
	
	# Performance settings
	max_concurrent_requests: int = 10000
	request_timeout_ms: int = 30000
	upstream_timeout_ms: int = 30000
	connect_timeout_ms: int = 5000
	
	# Rate limiting
	global_rate_limit_per_minute: int = 100000
	default_api_rate_limit_per_minute: int = 1000
	burst_multiplier: float = 1.5
	
	# Circuit breaker
	circuit_breaker_failure_threshold: int = 5
	circuit_breaker_timeout_seconds: int = 60
	circuit_breaker_half_open_max_requests: int = 10
	
	# Load balancing
	upstream_health_check_interval: int = 30
	upstream_health_check_timeout: int = 5
	upstream_max_retries: int = 3
	
	# Caching
	cache_enabled: bool = True
	cache_default_ttl: int = 300
	cache_max_size_mb: int = 512
	
	# Security
	cors_enabled: bool = True
	cors_allowed_origins: List[str] = field(default_factory=lambda: ["*"])
	cors_allowed_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"])
	cors_allowed_headers: List[str] = field(default_factory=lambda: ["*"])
	
	# Monitoring
	metrics_enabled: bool = True
	health_check_enabled: bool = True
	tracing_enabled: bool = False
	
	@validator('workers')
	def validate_workers(cls, v):
		if v < 1:
			raise ValueError('Workers must be at least 1')
		return v

class SecurityConfig(BaseModel):
	"""Security configuration."""
	
	# JWT settings
	jwt_secret_key: str = "your-secret-key-change-in-production"
	jwt_algorithm: str = "HS256"
	jwt_expiration_hours: int = 24
	jwt_refresh_expiration_days: int = 30
	
	# API Key settings
	api_key_length: int = 32
	api_key_prefix_length: int = 8
	api_key_hash_algorithm: str = "sha256"
	
	# OAuth2 settings
	oauth2_enabled: bool = False
	oauth2_provider_url: Optional[str] = None
	oauth2_client_id: Optional[str] = None
	oauth2_client_secret: Optional[str] = None
	oauth2_scope: str = "read write"
	
	# TLS settings
	tls_enabled: bool = False
	tls_cert_file: Optional[str] = None
	tls_key_file: Optional[str] = None
	tls_ca_file: Optional[str] = None
	
	# Security headers
	security_headers_enabled: bool = True
	hsts_enabled: bool = True
	hsts_max_age: int = 31536000
	content_security_policy: str = "default-src 'self'"
	
	@validator('jwt_secret_key')
	def validate_jwt_secret(cls, v):
		if len(v) < 32:
			raise ValueError('JWT secret key must be at least 32 characters')
		return v

class MonitoringConfig(BaseModel):
	"""Monitoring and observability configuration."""
	
	# Metrics
	metrics_retention_days: int = 30
	metrics_aggregation_interval: int = 60
	
	# Health checks
	health_check_interval: int = 30
	health_check_timeout: int = 10
	health_check_failure_threshold: int = 3
	
	# Alerting
	alerting_enabled: bool = True
	alert_email_enabled: bool = False
	alert_email_smtp_host: Optional[str] = None
	alert_email_smtp_port: int = 587
	alert_email_username: Optional[str] = None
	alert_email_password: Optional[str] = None
	alert_email_recipients: List[str] = field(default_factory=list)
	
	# Slack alerts
	alert_slack_enabled: bool = False
	alert_slack_webhook_url: Optional[str] = None
	alert_slack_channel: str = "#alerts"
	
	# Logging
	log_level: LogLevel = LogLevel.INFO
	log_format: str = "json"
	log_file: Optional[str] = None
	log_max_size_mb: int = 100
	log_backup_count: int = 5
	
	# Tracing
	tracing_enabled: bool = False
	tracing_endpoint: Optional[str] = None
	tracing_sample_rate: float = 0.1

class APIManagementSettings(BaseSettings):
	"""Main API management settings."""
	
	# Environment
	environment: Environment = Environment.DEVELOPMENT
	debug: bool = False
	testing: bool = False
	
	# Application
	app_name: str = "APG Integration API Management"
	app_version: str = "1.0.0"
	app_description: str = "Comprehensive API gateway and management platform"
	
	# Database
	database: DatabaseConfig = DatabaseConfig()
	
	# Redis
	redis: RedisConfig = RedisConfig()
	
	# Gateway
	gateway: GatewayConfig = GatewayConfig()
	
	# Security
	security: SecurityConfig = SecurityConfig()
	
	# Monitoring
	monitoring: MonitoringConfig = MonitoringConfig()
	
	# Multi-tenancy
	multi_tenant_enabled: bool = True
	default_tenant_id: str = "default"
	
	# Feature flags
	features: Dict[str, bool] = field(default_factory=lambda: {
		"analytics_enabled": True,
		"developer_portal_enabled": True,
		"policy_enforcement_enabled": True,
		"caching_enabled": True,
		"rate_limiting_enabled": True,
		"circuit_breaker_enabled": True,
		"load_balancing_enabled": True,
		"health_monitoring_enabled": True
	})
	
	class Config:
		env_file = ".env"
		env_nested_delimiter = "__"
		case_sensitive = False

# =============================================================================
# Configuration Manager
# =============================================================================

class ConfigurationManager:
	"""Manages configuration loading, validation, and environment overrides."""
	
	def __init__(self, config_dir: str = "config", environment: Optional[Environment] = None):
		self.config_dir = Path(config_dir)
		self.environment = environment or Environment.DEVELOPMENT
		self.settings = None
		
	def load_configuration(self) -> APIManagementSettings:
		"""Load configuration with environment-specific overrides."""
		
		# Load base configuration
		base_config = self._load_base_config()
		
		# Load environment-specific overrides
		env_overrides = self._load_environment_overrides()
		
		# Merge configurations
		merged_config = self._merge_configs(base_config, env_overrides)
		
		# Load from environment variables and validate
		self.settings = APIManagementSettings(**merged_config)
		
		# Apply post-load validations
		self._validate_configuration()
		
		return self.settings
	
	def _load_base_config(self) -> Dict[str, Any]:
		"""Load base configuration from file."""
		
		config_file = self.config_dir / "base.yaml"
		if config_file.exists():
			with open(config_file, 'r') as f:
				return yaml.safe_load(f) or {}
		
		return {}
	
	def _load_environment_overrides(self) -> Dict[str, Any]:
		"""Load environment-specific configuration overrides."""
		
		env_config_file = self.config_dir / f"{self.environment.value}.yaml"
		if env_config_file.exists():
			with open(env_config_file, 'r') as f:
				return yaml.safe_load(f) or {}
		
		return {}
	
	def _merge_configs(self, base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
		"""Recursively merge configuration dictionaries."""
		
		result = base.copy()
		
		for key, value in overrides.items():
			if key in result and isinstance(result[key], dict) and isinstance(value, dict):
				result[key] = self._merge_configs(result[key], value)
			else:
				result[key] = value
		
		return result
	
	def _validate_configuration(self):
		"""Perform additional configuration validation."""
		
		if not self.settings:
			raise ValueError("Configuration not loaded")
		
		# Production environment validations
		if self.settings.environment == Environment.PRODUCTION:
			self._validate_production_config()
		
		# Database connectivity validation
		self._validate_database_config()
		
		# Redis connectivity validation
		self._validate_redis_config()
		
		# Security configuration validation
		self._validate_security_config()
	
	def _validate_production_config(self):
		"""Validate production-specific configuration requirements."""
		
		# Ensure debug is disabled
		if self.settings.debug:
			raise ValueError("Debug mode must be disabled in production")
		
		# Ensure strong JWT secret
		if self.settings.security.jwt_secret_key == "your-secret-key-change-in-production":
			raise ValueError("JWT secret key must be changed for production")
		
		# Ensure TLS is enabled
		if not self.settings.security.tls_enabled:
			print("WARNING: TLS is not enabled in production environment")
		
		# Ensure monitoring is enabled
		if not self.settings.monitoring.alerting_enabled:
			print("WARNING: Alerting is not enabled in production environment")
	
	def _validate_database_config(self):
		"""Validate database configuration."""
		
		# Check required database settings
		db_config = self.settings.database
		
		if not db_config.host:
			raise ValueError("Database host is required")
		
		if not db_config.database:
			raise ValueError("Database name is required")
		
		if not db_config.username:
			raise ValueError("Database username is required")
	
	def _validate_redis_config(self):
		"""Validate Redis configuration."""
		
		redis_config = self.settings.redis
		
		if not redis_config.host:
			raise ValueError("Redis host is required")
		
		if redis_config.port <= 0 or redis_config.port > 65535:
			raise ValueError("Redis port must be between 1 and 65535")
	
	def _validate_security_config(self):
		"""Validate security configuration."""
		
		security_config = self.settings.security
		
		# Validate JWT settings
		if len(security_config.jwt_secret_key) < 32:
			raise ValueError("JWT secret key must be at least 32 characters")
		
		# Validate OAuth2 settings if enabled
		if security_config.oauth2_enabled:
			if not security_config.oauth2_provider_url:
				raise ValueError("OAuth2 provider URL is required when OAuth2 is enabled")
			
			if not security_config.oauth2_client_id:
				raise ValueError("OAuth2 client ID is required when OAuth2 is enabled")
	
	def get_feature_flag(self, feature_name: str) -> bool:
		"""Get feature flag value."""
		
		if not self.settings:
			return False
		
		return self.settings.features.get(feature_name, False)
	
	def set_feature_flag(self, feature_name: str, enabled: bool):
		"""Set feature flag value."""
		
		if not self.settings:
			raise ValueError("Configuration not loaded")
		
		self.settings.features[feature_name] = enabled
	
	def export_config(self, output_file: str, format: str = "yaml"):
		"""Export current configuration to file."""
		
		if not self.settings:
			raise ValueError("Configuration not loaded")
		
		config_dict = self.settings.dict()
		
		output_path = Path(output_file)
		
		if format.lower() == "yaml":
			with open(output_path, 'w') as f:
				yaml.dump(config_dict, f, default_flow_style=False, indent=2)
		elif format.lower() == "json":
			with open(output_path, 'w') as f:
				json.dump(config_dict, f, indent=2, default=str)
		else:
			raise ValueError("Unsupported format. Use 'yaml' or 'json'")
	
	def reload_configuration(self):
		"""Reload configuration from files."""
		
		self.load_configuration()

# =============================================================================
# Configuration Factory
# =============================================================================

def create_configuration(environment: Environment = None, 
						config_dir: str = "config") -> APIManagementSettings:
	"""Factory function to create configuration."""
	
	# Determine environment from environment variable if not provided
	if environment is None:
		env_name = os.getenv("ENVIRONMENT", "development").lower()
		environment = Environment(env_name)
	
	# Create configuration manager
	config_manager = ConfigurationManager(config_dir=config_dir, environment=environment)
	
	# Load and return configuration
	return config_manager.load_configuration()

def get_database_url(config: APIManagementSettings) -> str:
	"""Get database URL from configuration."""
	return config.database.url

def get_redis_url(config: APIManagementSettings) -> str:
	"""Get Redis URL from configuration."""
	return config.redis.url

# =============================================================================
# Configuration Templates
# =============================================================================

def generate_config_templates(output_dir: str = "config"):
	"""Generate configuration template files."""
	
	output_path = Path(output_dir)
	output_path.mkdir(exist_ok=True)
	
	# Base configuration template
	base_config = {
		"app_name": "APG Integration API Management",
		"app_version": "1.0.0",
		"multi_tenant_enabled": True,
		"database": {
			"engine": "postgresql",
			"host": "localhost",
			"port": 5432,
			"database": "apg_api_management",
			"pool_size": 20
		},
		"redis": {
			"host": "localhost",
			"port": 6379,
			"database": 0,
			"max_connections": 100
		},
		"gateway": {
			"host": "0.0.0.0",
			"port": 8080,
			"workers": 4,
			"max_concurrent_requests": 10000,
			"global_rate_limit_per_minute": 100000
		},
		"security": {
			"jwt_algorithm": "HS256",
			"jwt_expiration_hours": 24,
			"api_key_length": 32,
			"security_headers_enabled": True
		},
		"monitoring": {
			"metrics_retention_days": 30,
			"health_check_interval": 30,
			"log_level": "INFO",
			"alerting_enabled": True
		}
	}
	
	# Write base configuration
	with open(output_path / "base.yaml", 'w') as f:
		yaml.dump(base_config, f, default_flow_style=False, indent=2)
	
	# Development environment
	dev_config = {
		"environment": "development",
		"debug": True,
		"database": {
			"echo": True
		},
		"security": {
			"jwt_secret_key": "development-secret-key-32-chars-min",
			"tls_enabled": False
		},
		"monitoring": {
			"log_level": "DEBUG",
			"tracing_enabled": True
		}
	}
	
	with open(output_path / "development.yaml", 'w') as f:
		yaml.dump(dev_config, f, default_flow_style=False, indent=2)
	
	# Production environment
	prod_config = {
		"environment": "production",
		"debug": False,
		"database": {
			"pool_size": 50,
			"max_overflow": 100
		},
		"gateway": {
			"workers": 8,
			"max_concurrent_requests": 50000
		},
		"security": {
			"tls_enabled": True,
			"security_headers_enabled": True,
			"hsts_enabled": True
		},
		"monitoring": {
			"alerting_enabled": True,
			"alert_email_enabled": True,
			"tracing_enabled": True
		}
	}
	
	with open(output_path / "production.yaml", 'w') as f:
		yaml.dump(prod_config, f, default_flow_style=False, indent=2)
	
	print(f"Configuration templates generated in {output_dir}/")

# =============================================================================
# Export Configuration Components
# =============================================================================

__all__ = [
	# Enums
	'Environment',
	'LogLevel',
	'DatabaseEngine',
	'CacheEngine',
	
	# Configuration Models
	'DatabaseConfig',
	'RedisConfig',
	'GatewayConfig',
	'SecurityConfig',
	'MonitoringConfig',
	'APIManagementSettings',
	
	# Configuration Manager
	'ConfigurationManager',
	
	# Factory Functions
	'create_configuration',
	'get_database_url',
	'get_redis_url',
	'generate_config_templates'
]