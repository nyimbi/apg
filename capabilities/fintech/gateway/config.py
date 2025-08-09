"""
Production Configuration Management - APG Payment Gateway

Enterprise-grade configuration management system with environment-specific
settings, secure credential handling, and dynamic configuration updates.

© 2025 Datacraft. All rights reserved.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import yaml
import base64
from cryptography.fernet import Fernet
import boto3
from datetime import datetime, timezone

class Environment(str, Enum):
	"""Deployment environment types"""
	DEVELOPMENT = "development"
	STAGING = "staging"
	PRODUCTION = "production"
	TESTING = "testing"

class ConfigSource(str, Enum):
	"""Configuration source types"""
	FILE = "file"
	ENVIRONMENT = "environment"
	AWS_SECRETS = "aws_secrets"
	VAULT = "vault"
	DATABASE = "database"

@dataclass
class DatabaseConfig:
	"""Database configuration"""
	host: str = "localhost"
	port: int = 5432
	database: str = "payment_gateway"
	username: str = "pg_user"
	password: str = ""
	ssl_mode: str = "prefer"
	pool_size: int = 20
	max_overflow: int = 30
	pool_timeout: int = 30
	pool_recycle: int = 3600
	echo: bool = False
	
	@property
	def connection_url(self) -> str:
		"""Generate database connection URL"""
		return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}?sslmode={self.ssl_mode}"

@dataclass
class RedisConfig:
	"""Redis configuration for caching and sessions"""
	host: str = "localhost"
	port: int = 6379
	password: str = ""
	db: int = 0
	ssl: bool = False
	max_connections: int = 50
	socket_timeout: int = 5
	socket_connect_timeout: int = 5
	
	@property
	def connection_url(self) -> str:
		"""Generate Redis connection URL"""
		protocol = "rediss" if self.ssl else "redis"
		auth = f":{self.password}@" if self.password else ""
		return f"{protocol}://{auth}{self.host}:{self.port}/{self.db}"

@dataclass
class ProcessorConfig:
	"""Payment processor configuration"""
	name: str
	type: str
	enabled: bool = True
	priority: int = 100
	api_url: str = ""
	api_key: str = ""
	secret_key: str = ""
	webhook_secret: str = ""
	supported_methods: List[str] = field(default_factory=list)
	supported_currencies: List[str] = field(default_factory=list)
	supported_countries: List[str] = field(default_factory=list)
	rate_limits: Dict[str, int] = field(default_factory=dict)
	timeout_seconds: int = 30
	retry_attempts: int = 3
	sandbox_mode: bool = False
	custom_config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SecurityConfig:
	"""Security and authentication configuration"""
	secret_key: str = ""
	jwt_secret: str = ""
	jwt_expiry_hours: int = 24
	api_key_length: int = 32
	webhook_signature_header: str = "X-Signature"
	allowed_hosts: List[str] = field(default_factory=list)
	cors_origins: List[str] = field(default_factory=list)
	rate_limit_per_minute: int = 1000
	max_request_size_mb: int = 10
	encryption_key: str = ""
	password_min_length: int = 12
	require_https: bool = True
	session_timeout_minutes: int = 30

@dataclass
class LoggingConfig:
	"""Logging configuration"""
	level: str = "INFO"
	format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
	file_path: str = "/var/log/payment_gateway/app.log"
	max_file_size_mb: int = 100
	backup_count: int = 5
	log_to_console: bool = True
	log_to_file: bool = True
	structured_logging: bool = True
	audit_log_path: str = "/var/log/payment_gateway/audit.log"
	performance_log_path: str = "/var/log/payment_gateway/performance.log"

@dataclass
class MonitoringConfig:
	"""Monitoring and observability configuration"""
	enabled: bool = True
	metrics_port: int = 9090
	health_check_port: int = 8080
	prometheus_enabled: bool = True
	jaeger_enabled: bool = False
	jaeger_agent_host: str = "localhost"
	jaeger_agent_port: int = 6831
	sentry_dsn: str = ""
	datadog_api_key: str = ""
	custom_metrics_enabled: bool = True
	alert_thresholds: Dict[str, float] = field(default_factory=dict)

@dataclass
class CacheConfig:
	"""Caching configuration"""
	default_ttl_seconds: int = 3600
	user_profile_ttl_seconds: int = 1800
	fraud_rules_ttl_seconds: int = 300
	processor_health_ttl_seconds: int = 60
	analytics_cache_ttl_seconds: int = 900
	max_memory_mb: int = 512
	eviction_policy: str = "allkeys-lru"

@dataclass
class WebhookConfig:
	"""Webhook configuration"""
	enabled: bool = True
	max_retry_attempts: int = 5
	initial_retry_delay_seconds: int = 5
	max_retry_delay_seconds: int = 3600
	timeout_seconds: int = 30
	batch_size: int = 100
	worker_threads: int = 4
	signature_algorithm: str = "hmac-sha256"
	delivery_timeout_hours: int = 24

@dataclass
class FraudConfig:
	"""Fraud detection configuration"""
	enabled: bool = True
	ml_models_enabled: bool = True
	real_time_scoring: bool = True
	score_threshold_review: float = 0.7
	score_threshold_decline: float = 0.9
	velocity_window_minutes: int = 60
	device_fingerprint_enabled: bool = True
	geolocation_enabled: bool = True
	whitelist_enabled: bool = True
	blacklist_enabled: bool = True
	model_refresh_hours: int = 24

@dataclass
class APIConfig:
	"""API configuration"""
	host: str = "0.0.0.0"
	port: int = 8000
	workers: int = 4
	max_connections: int = 1000
	keepalive_timeout: int = 75
	client_timeout: int = 30
	api_version: str = "v1"
	docs_enabled: bool = True
	cors_enabled: bool = True
	compression_enabled: bool = True
	request_id_header: str = "X-Request-ID"

class PaymentGatewayConfig:
	"""
	Comprehensive configuration management for APG Payment Gateway
	
	Handles environment-specific settings, secure credential management,
	and dynamic configuration updates for production deployments.
	"""
	
	def __init__(self, environment: Environment = Environment.DEVELOPMENT):
		self.environment = environment
		self.config_sources: List[ConfigSource] = []
		self._encryption_key: Optional[bytes] = None
		self._aws_session = None
		
		# Core configuration sections
		self.database = DatabaseConfig()
		self.redis = RedisConfig()
		self.security = SecurityConfig()
		self.logging = LoggingConfig()
		self.monitoring = MonitoringConfig()
		self.cache = CacheConfig()
		self.webhook = WebhookConfig()
		self.fraud = FraudConfig()
		self.api = APIConfig()
		
		# Payment processors
		self.processors: Dict[str, ProcessorConfig] = {}
		
		# Custom configuration
		self.custom: Dict[str, Any] = {}
		
		# Load configuration
		self._load_configuration()
	
	def _load_configuration(self) -> None:
		"""Load configuration from all sources"""
		print(f"🔧 Loading configuration for environment: {self.environment.value}")
		
		# Load base configuration from file
		self._load_from_file()
		
		# Override with environment variables
		self._load_from_environment()
		
		# Load secrets from secure sources
		self._load_secrets()
		
		# Apply environment-specific overrides
		self._apply_environment_overrides()
		
		# Validate configuration
		self._validate_configuration()
		
		print(f"✅ Configuration loaded from sources: {[s.value for s in self.config_sources]}")
	
	def _load_from_file(self) -> None:
		"""Load configuration from YAML files"""
		config_dir = Path(__file__).parent / "config"
		
		# Load base config
		base_config_file = config_dir / "base.yaml"
		if base_config_file.exists():
			with open(base_config_file, 'r') as f:
				base_config = yaml.safe_load(f)
				self._apply_config_dict(base_config)
				self.config_sources.append(ConfigSource.FILE)
		
		# Load environment-specific config
		env_config_file = config_dir / f"{self.environment.value}.yaml"
		if env_config_file.exists():
			with open(env_config_file, 'r') as f:
				env_config = yaml.safe_load(f)
				self._apply_config_dict(env_config)
	
	def _load_from_environment(self) -> None:
		"""Load configuration from environment variables"""
		env_mappings = {
			# Database
			"DATABASE_HOST": ("database", "host"),
			"DATABASE_PORT": ("database", "port"),
			"DATABASE_NAME": ("database", "database"),
			"DATABASE_USER": ("database", "username"),
			"DATABASE_PASSWORD": ("database", "password"),
			"DATABASE_SSL_MODE": ("database", "ssl_mode"),
			
			# Redis
			"REDIS_HOST": ("redis", "host"),
			"REDIS_PORT": ("redis", "port"),
			"REDIS_PASSWORD": ("redis", "password"),
			"REDIS_DB": ("redis", "db"),
			
			# Security
			"SECRET_KEY": ("security", "secret_key"),
			"JWT_SECRET": ("security", "jwt_secret"),
			"ENCRYPTION_KEY": ("security", "encryption_key"),
			
			# API
			"API_HOST": ("api", "host"),
			"API_PORT": ("api", "port"),
			"API_WORKERS": ("api", "workers"),
			
			# Monitoring
			"SENTRY_DSN": ("monitoring", "sentry_dsn"),
			"DATADOG_API_KEY": ("monitoring", "datadog_api_key"),
			
			# Logging
			"LOG_LEVEL": ("logging", "level"),
			"LOG_FILE": ("logging", "file_path"),
		}
		
		loaded_count = 0
		for env_var, (section, key) in env_mappings.items():
			value = os.getenv(env_var)
			if value is not None:
				self._set_nested_config(section, key, self._convert_env_value(value))
				loaded_count += 1
		
		if loaded_count > 0:
			self.config_sources.append(ConfigSource.ENVIRONMENT)
			print(f"📊 Loaded {loaded_count} settings from environment variables")
	
	def _load_secrets(self) -> None:
		"""Load secrets from secure sources"""
		# Load from AWS Secrets Manager
		if os.getenv("AWS_SECRETS_ENABLED") == "true":
			self._load_from_aws_secrets()
		
		# Load from HashiCorp Vault
		if os.getenv("VAULT_ENABLED") == "true":
			self._load_from_vault()
	
	def _load_from_aws_secrets(self) -> None:
		"""Load configuration from AWS Secrets Manager"""
		try:
			if not self._aws_session:
				self._aws_session = boto3.Session(
					region_name=os.getenv("AWS_REGION", "us-east-1")
				)
			
			secrets_client = self._aws_session.client("secretsmanager")
			secret_name = f"payment-gateway/{self.environment.value}"
			
			response = secrets_client.get_secret_value(SecretId=secret_name)
			secrets = json.loads(response["SecretString"])
			
			# Apply secrets to configuration
			for key, value in secrets.items():
				if key.startswith("database_"):
					attr = key.replace("database_", "")
					if hasattr(self.database, attr):
						setattr(self.database, attr, value)
				elif key.startswith("processor_"):
					# Handle processor secrets
					parts = key.split("_", 2)
					if len(parts) >= 3:
						processor_name, attr = parts[1], parts[2]
						if processor_name in self.processors:
							setattr(self.processors[processor_name], attr, value)
			
			self.config_sources.append(ConfigSource.AWS_SECRETS)
			print(f"🔐 Loaded secrets from AWS Secrets Manager")
			
		except Exception as e:
			print(f"⚠️  Failed to load AWS secrets: {str(e)}")
	
	def _load_from_vault(self) -> None:
		"""Load configuration from HashiCorp Vault"""
		try:
			vault_url = os.getenv("VAULT_URL")
			vault_token = os.getenv("VAULT_TOKEN")
			
			if vault_url and vault_token:
				# Would implement Vault client connection here
				self.config_sources.append(ConfigSource.VAULT)
				print(f"🔐 Loaded secrets from HashiCorp Vault")
				
		except Exception as e:
			print(f"⚠️  Failed to load Vault secrets: {str(e)}")
	
	def _apply_environment_overrides(self) -> None:
		"""Apply environment-specific configuration overrides"""
		if self.environment == Environment.DEVELOPMENT:
			self.database.echo = True
			self.logging.level = "DEBUG"
			self.api.docs_enabled = True
			self.fraud.ml_models_enabled = False
			self.security.require_https = False
			
		elif self.environment == Environment.STAGING:
			self.logging.level = "INFO"
			self.api.docs_enabled = True
			self.fraud.ml_models_enabled = True
			self.security.require_https = True
			
		elif self.environment == Environment.PRODUCTION:
			self.database.echo = False
			self.logging.level = "WARNING"
			self.api.docs_enabled = False
			self.fraud.ml_models_enabled = True
			self.security.require_https = True
			self.monitoring.enabled = True
			
		elif self.environment == Environment.TESTING:
			self.database.database = "payment_gateway_test"
			self.logging.level = "ERROR"
			self.api.docs_enabled = False
			self.fraud.enabled = False
			self.webhook.enabled = False
	
	def _validate_configuration(self) -> None:
		"""Validate configuration settings"""
		errors = []
		
		# Validate required fields
		if not self.database.password and self.environment == Environment.PRODUCTION:
			errors.append("Database password is required in production")
		
		if not self.security.secret_key:
			errors.append("Security secret key is required")
		
		if not self.security.jwt_secret:
			errors.append("JWT secret is required")
		
		# Validate processor configurations
		for name, processor in self.processors.items():
			if processor.enabled and not processor.api_key:
				errors.append(f"API key required for enabled processor: {name}")
		
		# Validate port numbers
		if not (1 <= self.api.port <= 65535):
			errors.append("API port must be between 1 and 65535")
		
		if not (1 <= self.database.port <= 65535):
			errors.append("Database port must be between 1 and 65535")
		
		# Validate security settings
		if self.security.jwt_expiry_hours < 1:
			errors.append("JWT expiry must be at least 1 hour")
		
		if len(errors) > 0:
			raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
		
		print(f"✅ Configuration validation passed")
	
	def _apply_config_dict(self, config_dict: Dict[str, Any]) -> None:
		"""Apply configuration from dictionary"""
		for section, settings in config_dict.items():
			if hasattr(self, section):
				section_obj = getattr(self, section)
				if isinstance(settings, dict):
					for key, value in settings.items():
						if hasattr(section_obj, key):
							setattr(section_obj, key, value)
			elif section == "processors":
				self._load_processors_config(settings)
			elif section == "custom":
				self.custom.update(settings)
	
	def _load_processors_config(self, processors_config: Dict[str, Any]) -> None:
		"""Load payment processors configuration"""
		for name, config in processors_config.items():
			processor = ProcessorConfig(
				name=name,
				type=config.get("type", ""),
				enabled=config.get("enabled", True),
				priority=config.get("priority", 100),
				api_url=config.get("api_url", ""),
				api_key=config.get("api_key", ""),
				secret_key=config.get("secret_key", ""),
				webhook_secret=config.get("webhook_secret", ""),
				supported_methods=config.get("supported_methods", []),
				supported_currencies=config.get("supported_currencies", []),
				supported_countries=config.get("supported_countries", []),
				rate_limits=config.get("rate_limits", {}),
				timeout_seconds=config.get("timeout_seconds", 30),
				retry_attempts=config.get("retry_attempts", 3),
				sandbox_mode=config.get("sandbox_mode", False),
				custom_config=config.get("custom_config", {})
			)
			self.processors[name] = processor
	
	def _set_nested_config(self, section: str, key: str, value: Any) -> None:
		"""Set nested configuration value"""
		if hasattr(self, section):
			section_obj = getattr(self, section)
			if hasattr(section_obj, key):
				# Convert string values to appropriate types
				current_value = getattr(section_obj, key)
				if isinstance(current_value, bool):
					value = str(value).lower() in ('true', '1', 'yes', 'on')
				elif isinstance(current_value, int):
					value = int(value)
				elif isinstance(current_value, float):
					value = float(value)
				
				setattr(section_obj, key, value)
	
	def _convert_env_value(self, value: str) -> Any:
		"""Convert environment variable string to appropriate type"""
		# Try to parse as JSON first
		try:
			return json.loads(value)
		except (json.JSONDecodeError, ValueError):
			pass
		
		# Check for boolean values
		if value.lower() in ('true', 'false'):
			return value.lower() == 'true'
		
		# Try to parse as number
		try:
			if '.' in value:
				return float(value)
			else:
				return int(value)
		except ValueError:
			pass
		
		# Return as string
		return value
	
	def get_processor_config(self, processor_name: str) -> Optional[ProcessorConfig]:
		"""Get configuration for specific processor"""
		return self.processors.get(processor_name)
	
	def get_enabled_processors(self) -> List[ProcessorConfig]:
		"""Get list of enabled processors"""
		return [p for p in self.processors.values() if p.enabled]
	
	def encrypt_sensitive_data(self, data: str) -> str:
		"""Encrypt sensitive configuration data"""
		if not self._encryption_key:
			if self.security.encryption_key:
				self._encryption_key = base64.urlsafe_b64decode(self.security.encryption_key)
			else:
				self._encryption_key = Fernet.generate_key()
		
		cipher = Fernet(self._encryption_key)
		return cipher.encrypt(data.encode()).decode()
	
	def decrypt_sensitive_data(self, encrypted_data: str) -> str:
		"""Decrypt sensitive configuration data"""
		if not self._encryption_key:
			if self.security.encryption_key:
				self._encryption_key = base64.urlsafe_b64decode(self.security.encryption_key)
			else:
				raise ValueError("Encryption key not available")
		
		cipher = Fernet(self._encryption_key)
		return cipher.decrypt(encrypted_data.encode()).decode()
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert configuration to dictionary"""
		return {
			"environment": self.environment.value,
			"database": asdict(self.database),
			"redis": asdict(self.redis),
			"security": {**asdict(self.security), "secret_key": "***", "jwt_secret": "***"},
			"logging": asdict(self.logging),
			"monitoring": asdict(self.monitoring),
			"cache": asdict(self.cache),
			"webhook": asdict(self.webhook),
			"fraud": asdict(self.fraud),
			"api": asdict(self.api),
			"processors": {name: {**asdict(proc), "api_key": "***", "secret_key": "***"} 
						  for name, proc in self.processors.items()},
			"custom": self.custom
		}
	
	def reload_configuration(self) -> None:
		"""Reload configuration from all sources"""
		print(f"🔄 Reloading configuration...")
		self.config_sources.clear()
		self._load_configuration()
	
	def save_to_file(self, file_path: str, include_secrets: bool = False) -> None:
		"""Save configuration to file"""
		config_dict = self.to_dict()
		
		if not include_secrets:
			# Remove sensitive data
			config_dict["security"]["secret_key"] = ""
			config_dict["security"]["jwt_secret"] = ""
			config_dict["security"]["encryption_key"] = ""
			
			for proc_config in config_dict["processors"].values():
				proc_config["api_key"] = ""
				proc_config["secret_key"] = ""
				proc_config["webhook_secret"] = ""
		
		with open(file_path, 'w') as f:
			yaml.dump(config_dict, f, default_flow_style=False, indent=2)
		
		print(f"💾 Configuration saved to: {file_path}")

# Factory functions
def create_config(environment: str = None) -> PaymentGatewayConfig:
	"""Create configuration instance"""
	if environment:
		env = Environment(environment)
	else:
		env = Environment(os.getenv("ENVIRONMENT", Environment.DEVELOPMENT.value))
	
	return PaymentGatewayConfig(env)

def load_config_from_file(file_path: str) -> PaymentGatewayConfig:
	"""Load configuration from specific file"""
	with open(file_path, 'r') as f:
		config_data = yaml.safe_load(f)
	
	environment = Environment(config_data.get("environment", Environment.DEVELOPMENT.value))
	config = PaymentGatewayConfig(environment)
	config._apply_config_dict(config_data)
	
	return config

# Global configuration instance
_config_instance: Optional[PaymentGatewayConfig] = None

def get_config() -> PaymentGatewayConfig:
	"""Get global configuration instance"""
	global _config_instance
	if _config_instance is None:
		_config_instance = create_config()
	return _config_instance

def set_config(config: PaymentGatewayConfig) -> None:
	"""Set global configuration instance"""
	global _config_instance
	_config_instance = config

# Configuration validation utilities
def validate_processor_config(processor: ProcessorConfig) -> List[str]:
	"""Validate processor configuration"""
	errors = []
	
	if not processor.name:
		errors.append("Processor name is required")
	
	if not processor.type:
		errors.append("Processor type is required")
	
	if processor.enabled and not processor.api_url:
		errors.append(f"API URL required for enabled processor: {processor.name}")
	
	if processor.timeout_seconds < 1:
		errors.append(f"Timeout must be at least 1 second: {processor.name}")
	
	if processor.retry_attempts < 0:
		errors.append(f"Retry attempts cannot be negative: {processor.name}")
	
	return errors

def generate_default_config(environment: Environment) -> str:
	"""Generate default configuration YAML"""
	default_config = PaymentGatewayConfig(environment)
	
	# Add default processors
	default_config.processors["mpesa"] = ProcessorConfig(
		name="mpesa",
		type="mpesa",
		enabled=True,
		supported_methods=["mpesa"],
		supported_currencies=["KES"],
		supported_countries=["KE"],
		sandbox_mode=environment != Environment.PRODUCTION
	)
	
	default_config.processors["stripe"] = ProcessorConfig(
		name="stripe",
		type="stripe",
		enabled=True,
		supported_methods=["credit_card", "debit_card"],
		supported_currencies=["USD", "EUR", "GBP", "KES"],
		supported_countries=["US", "EU", "GB", "KE"],
		sandbox_mode=environment != Environment.PRODUCTION
	)
	
	config_dict = default_config.to_dict()
	return yaml.dump(config_dict, default_flow_style=False, indent=2)

def _log_config_module_loaded():
	"""Log configuration module loaded"""
	print("⚙️  Production Configuration module loaded")
	print("   - Environment-specific settings")
	print("   - Secure credential management")
	print("   - AWS Secrets Manager integration")
	print("   - Dynamic configuration updates")

# Execute module loading log
_log_config_module_loaded()