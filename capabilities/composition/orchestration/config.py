#!/usr/bin/env python3
"""
APG Workflow Orchestration Configuration Management

Configuration validation, default settings, and environment management.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import yaml

from pydantic import BaseModel, Field, ConfigDict, validator
from pydantic_settings import BaseSettings

from apg.framework.base_service import APGBaseService
from apg.framework.config import APGConfig
from apg.framework.security import APGSecurity


logger = logging.getLogger(__name__)


class Environment(str, Enum):
	"""Deployment environments."""
	DEVELOPMENT = "development"
	TESTING = "testing"
	STAGING = "staging"
	PRODUCTION = "production"


class DatabaseType(str, Enum):
	"""Supported database types."""
	POSTGRESQL = "postgresql"
	MYSQL = "mysql"
	SQLITE = "sqlite"


class ExecutorType(str, Enum):
	"""Workflow executor types."""
	LOCAL = "local"
	CELERY = "celery"
	KUBERNETES = "kubernetes"
	AWS_BATCH = "aws_batch"
	AZURE_BATCH = "azure_batch"


class LogLevel(str, Enum):
	"""Logging levels."""
	DEBUG = "DEBUG"
	INFO = "INFO"
	WARNING = "WARNING"
	ERROR = "ERROR"
	CRITICAL = "CRITICAL"


@dataclass
class DatabaseConfig:
	"""Database configuration."""
	type: DatabaseType = DatabaseType.POSTGRESQL
	host: str = "localhost"
	port: int = 5432
	database: str = "apg_workflow"
	username: str = "apg_user"
	password: str = ""
	ssl_mode: str = "prefer"
	pool_size: int = 20
	max_overflow: int = 30
	pool_timeout: int = 30
	pool_recycle: int = 3600
	echo_sql: bool = False
	
	def get_url(self) -> str:
		"""Get database connection URL."""
		if self.type == DatabaseType.SQLITE:
			return f"sqlite:///{self.database}"
		
		auth = f"{self.username}:{self.password}@" if self.password else f"{self.username}@"
		return f"{self.type.value}://{auth}{self.host}:{self.port}/{self.database}"


@dataclass
class RedisConfig:
	"""Redis configuration for caching and queuing."""
	host: str = "localhost"
	port: int = 6379
	database: int = 0
	password: Optional[str] = None
	ssl: bool = False
	max_connections: int = 50
	socket_timeout: int = 30
	
	def get_url(self) -> str:
		"""Get Redis connection URL."""
		protocol = "rediss" if self.ssl else "redis"
		auth = f":{self.password}@" if self.password else ""
		return f"{protocol}://{auth}{self.host}:{self.port}/{self.database}"


@dataclass
class ExecutorConfig:
	"""Workflow executor configuration."""
	type: ExecutorType = ExecutorType.LOCAL
	max_workers: int = 4
	max_concurrent_workflows: int = 100
	task_timeout: int = 3600
	cleanup_interval: int = 300
	
	# Celery-specific configuration
	broker_url: Optional[str] = None
	result_backend: Optional[str] = None
	
	# Kubernetes-specific configuration
	namespace: str = "default"
	image: str = "apg-workflow-executor:latest"
	cpu_request: str = "100m"
	cpu_limit: str = "1000m"
	memory_request: str = "128Mi"
	memory_limit: str = "512Mi"
	
	# Cloud-specific configuration
	aws_region: Optional[str] = None
	azure_resource_group: Optional[str] = None


@dataclass
class SecurityConfig:
	"""Security configuration."""
	encryption_key: Optional[str] = None
	jwt_secret: Optional[str] = None
	jwt_expiry_hours: int = 24
	api_rate_limit: int = 1000
	enable_cors: bool = True
	cors_origins: List[str] = field(default_factory=lambda: ["*"])
	enable_csrf: bool = True
	session_timeout: int = 3600
	max_login_attempts: int = 5
	lockout_duration: int = 900


@dataclass
class MonitoringConfig:
	"""Monitoring and observability configuration."""
	enable_metrics: bool = True
	enable_tracing: bool = True
	enable_logging: bool = True
	
	# Metrics configuration
	metrics_port: int = 9090
	metrics_path: str = "/metrics"
	
	# Tracing configuration
	jaeger_endpoint: Optional[str] = None
	sampling_rate: float = 0.1
	
	# Logging configuration
	log_level: LogLevel = LogLevel.INFO
	log_format: str = "json"
	log_file: Optional[str] = None


@dataclass
class IntegrationConfig:
	"""External integration configuration."""
	enable_webhooks: bool = True
	webhook_timeout: int = 30
	webhook_retry_attempts: int = 3
	
	# API Gateway configuration
	api_gateway_url: Optional[str] = None
	api_gateway_key: Optional[str] = None
	
	# Message queue configuration
	message_queue_type: str = "redis"
	message_queue_url: Optional[str] = None
	
	# External service timeouts
	external_service_timeout: int = 60
	max_concurrent_requests: int = 100


@dataclass
class WorkflowConfig:
	"""Workflow-specific configuration."""
	default_timeout: int = 3600
	max_retries: int = 3
	retry_delay: int = 60
	enable_checkpoints: bool = True
	checkpoint_interval: int = 300
	enable_compensation: bool = True
	max_workflow_duration: int = 86400  # 24 hours
	cleanup_completed_after: int = 604800  # 7 days
	cleanup_failed_after: int = 2592000  # 30 days


class WorkflowOrchestrationConfig(BaseSettings):
	"""Main configuration class using Pydantic settings."""
	
	model_config = ConfigDict(
		env_prefix="WO_",
		env_file=".env",
		env_file_encoding="utf-8",
		case_sensitive=False,
		validate_assignment=True,
		extra="forbid"
	)
	
	# Environment settings
	environment: Environment = Environment.DEVELOPMENT
	debug: bool = False
	
	# Service settings
	service_name: str = "workflow-orchestration"
	service_version: str = "1.0.0"
	service_port: int = 8080
	service_host: str = "0.0.0.0"
	
	# APG Integration
	apg_tenant_id: str = "default"
	apg_service_id: str = "workflow_orchestration"
	
	# Component configurations
	database: DatabaseConfig = field(default_factory=DatabaseConfig)
	redis: RedisConfig = field(default_factory=RedisConfig)
	executor: ExecutorConfig = field(default_factory=ExecutorConfig)
	security: SecurityConfig = field(default_factory=SecurityConfig)
	monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
	integration: IntegrationConfig = field(default_factory=IntegrationConfig)
	workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
	
	@validator('environment')
	def validate_environment(cls, v):
		"""Validate environment setting."""
		if isinstance(v, str):
			return Environment(v.lower())
		return v
	
	@validator('database')
	def validate_database_config(cls, v):
		"""Validate database configuration."""
		if isinstance(v, dict):
			return DatabaseConfig(**v)
		return v
	
	@validator('redis')
	def validate_redis_config(cls, v):
		"""Validate Redis configuration."""
		if isinstance(v, dict):
			return RedisConfig(**v)
		return v
	
	def is_production(self) -> bool:
		"""Check if running in production."""
		return self.environment == Environment.PRODUCTION
	
	def is_development(self) -> bool:
		"""Check if running in development."""
		return self.environment == Environment.DEVELOPMENT
	
	def get_database_url(self) -> str:
		"""Get database connection URL."""
		return self.database.get_url()
	
	def get_redis_url(self) -> str:
		"""Get Redis connection URL."""
		return self.redis.get_url()
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert configuration to dictionary."""
		return {
			"environment": self.environment.value,
			"debug": self.debug,
			"service_name": self.service_name,
			"service_version": self.service_version,
			"service_port": self.service_port,
			"service_host": self.service_host,
			"apg_tenant_id": self.apg_tenant_id,
			"apg_service_id": self.apg_service_id,
			"database": asdict(self.database),
			"redis": asdict(self.redis),
			"executor": asdict(self.executor),
			"security": asdict(self.security),
			"monitoring": asdict(self.monitoring),
			"integration": asdict(self.integration),
			"workflow": asdict(self.workflow)
		}


class ConfigurationManager(APGBaseService):
	"""Configuration management service."""
	
	def __init__(self):
		super().__init__()
		self.config: Optional[WorkflowOrchestrationConfig] = None
		self.security = APGSecurity()
		self._config_file_paths = [
			Path("config/workflow_orchestration.yaml"),
			Path("config/workflow_orchestration.yml"),
			Path("config/workflow_orchestration.json"),
			Path("workflow_orchestration.yaml"),
			Path("workflow_orchestration.yml"),
			Path("workflow_orchestration.json")
		]
	
	async def start(self):
		"""Start configuration manager."""
		await super().start()
		await self.load_configuration()
		await self.validate_configuration()
		logger.info("Configuration manager started")
	
	async def load_configuration(self) -> WorkflowOrchestrationConfig:
		"""Load configuration from multiple sources."""
		config_data = {}
		
		# Load from file if exists
		config_file_data = await self._load_from_file()
		if config_file_data:
			config_data.update(config_file_data)
		
		# Load from environment variables
		env_data = await self._load_from_environment()
		config_data.update(env_data)
		
		# Load from APG configuration system
		apg_data = await self._load_from_apg()
		if apg_data:
			config_data.update(apg_data)
		
		# Create configuration instance
		self.config = WorkflowOrchestrationConfig(**config_data)
		
		# Decrypt sensitive values
		await self._decrypt_sensitive_values()
		
		return self.config
	
	async def _load_from_file(self) -> Optional[Dict[str, Any]]:
		"""Load configuration from file."""
		for config_path in self._config_file_paths:
			if config_path.exists():
				try:
					with open(config_path, 'r') as f:
						if config_path.suffix in ['.yaml', '.yml']:
							data = yaml.safe_load(f)
						else:
							data = json.load(f)
					
					logger.info(f"Loaded configuration from {config_path}")
					return data
				except Exception as e:
					logger.warning(f"Failed to load config from {config_path}: {e}")
		
		return None
	
	async def _load_from_environment(self) -> Dict[str, Any]:
		"""Load configuration from environment variables."""
		env_data = {}
		
		# Map environment variables to configuration structure
		env_mappings = {
			'WO_ENVIRONMENT': 'environment',
			'WO_DEBUG': 'debug',
			'WO_SERVICE_PORT': 'service_port',
			'WO_DATABASE_HOST': 'database.host',
			'WO_DATABASE_PORT': 'database.port',
			'WO_DATABASE_NAME': 'database.database',
			'WO_DATABASE_USER': 'database.username',
			'WO_DATABASE_PASSWORD': 'database.password',
			'WO_REDIS_HOST': 'redis.host',
			'WO_REDIS_PORT': 'redis.port',
			'WO_REDIS_PASSWORD': 'redis.password',
		}
		
		for env_var, config_path in env_mappings.items():
			value = os.getenv(env_var)
			if value is not None:
				# Convert string values to appropriate types
				if env_var.endswith('_PORT'):
					value = int(value)
				elif env_var.endswith('_DEBUG'):
					value = value.lower() in ('true', '1', 'yes', 'on')
				
				# Set nested configuration values
				self._set_nested_value(env_data, config_path, value)
		
		return env_data
	
	async def _load_from_apg(self) -> Optional[Dict[str, Any]]:
		"""Load configuration from APG configuration system."""
		try:
			apg_config = APGConfig()
			config_data = await apg_config.get_service_config(
				service_id="workflow_orchestration",
				version="1.0.0"
			)
			
			if config_data:
				logger.info("Loaded configuration from APG config system")
				return config_data
		except Exception as e:
			logger.warning(f"Failed to load config from APG: {e}")
		
		return None
	
	async def _decrypt_sensitive_values(self):
		"""Decrypt sensitive configuration values."""
		if not self.config:
			return
		
		sensitive_fields = [
			'database.password',
			'redis.password',
			'security.encryption_key',
			'security.jwt_secret',
			'integration.api_gateway_key'
		]
		
		for field_path in sensitive_fields:
			value = self._get_nested_value(self.config.to_dict(), field_path)
			if value and isinstance(value, str) and value.startswith('encrypted:'):
				try:
					decrypted = await self.security.decrypt(value[10:])  # Remove 'encrypted:' prefix
					self._set_nested_value(self.config, field_path, decrypted)
				except Exception as e:
					logger.warning(f"Failed to decrypt {field_path}: {e}")
	
	async def validate_configuration(self) -> bool:
		"""Validate the loaded configuration."""
		if not self.config:
			raise ValueError("Configuration not loaded")
		
		validation_errors = []
		
		# Validate database configuration
		try:
			db_url = self.config.get_database_url()
			if not db_url:
				validation_errors.append("Invalid database configuration")
		except Exception as e:
			validation_errors.append(f"Database config error: {e}")
		
		# Validate Redis configuration
		try:
			redis_url = self.config.get_redis_url()
			if not redis_url:
				validation_errors.append("Invalid Redis configuration")
		except Exception as e:
			validation_errors.append(f"Redis config error: {e}")
		
		# Validate security configuration
		if self.config.is_production():
			if not self.config.security.encryption_key:
				validation_errors.append("Encryption key required in production")
			if not self.config.security.jwt_secret:
				validation_errors.append("JWT secret required in production")
		
		# Validate executor configuration
		if self.config.executor.type == ExecutorType.CELERY:
			if not self.config.executor.broker_url:
				validation_errors.append("Celery broker URL required")
		
		if validation_errors:
			error_msg = "Configuration validation failed:\n" + "\n".join(validation_errors)
			logger.error(error_msg)
			raise ValueError(error_msg)
		
		logger.info("Configuration validation passed")
		return True
	
	async def save_configuration(self, config: WorkflowOrchestrationConfig, 
								 file_path: Optional[Path] = None):
		"""Save configuration to file."""
		if not file_path:
			file_path = Path("config/workflow_orchestration.yaml")
		
		# Create config directory if it doesn't exist
		file_path.parent.mkdir(parents=True, exist_ok=True)
		
		# Convert to dictionary and encrypt sensitive values
		config_dict = config.to_dict()
		await self._encrypt_sensitive_values(config_dict)
		
		# Save to file
		with open(file_path, 'w') as f:
			yaml.safe_dump(config_dict, f, default_flow_style=False, indent=2)
		
		logger.info(f"Configuration saved to {file_path}")
	
	async def _encrypt_sensitive_values(self, config_dict: Dict[str, Any]):
		"""Encrypt sensitive values in configuration dictionary."""
		sensitive_fields = [
			'database.password',
			'redis.password',
			'security.encryption_key',
			'security.jwt_secret',
			'integration.api_gateway_key'
		]
		
		for field_path in sensitive_fields:
			value = self._get_nested_value(config_dict, field_path)
			if value and isinstance(value, str) and not value.startswith('encrypted:'):
				try:
					encrypted = await self.security.encrypt(value)
					self._set_nested_value(config_dict, field_path, f'encrypted:{encrypted}')
				except Exception as e:
					logger.warning(f"Failed to encrypt {field_path}: {e}")
	
	def get_config(self) -> WorkflowOrchestrationConfig:
		"""Get current configuration."""
		if not self.config:
			raise ValueError("Configuration not loaded")
		return self.config
	
	def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
		"""Get nested value from dictionary using dot notation."""
		keys = path.split('.')
		current = data
		
		for key in keys:
			if isinstance(current, dict) and key in current:
				current = current[key]
			else:
				return None
		
		return current
	
	def _set_nested_value(self, data: Union[Dict[str, Any], object], path: str, value: Any):
		"""Set nested value in dictionary or object using dot notation."""
		keys = path.split('.')
		current = data
		
		for key in keys[:-1]:
			if isinstance(current, dict):
				if key not in current:
					current[key] = {}
				current = current[key]
			else:
				# Handle object attributes
				if not hasattr(current, key):
					setattr(current, key, {})
				current = getattr(current, key)
		
		# Set the final value
		final_key = keys[-1]
		if isinstance(current, dict):
			current[final_key] = value
		else:
			setattr(current, final_key, value)


# Global configuration manager instance
config_manager = ConfigurationManager()


async def get_config() -> WorkflowOrchestrationConfig:
	"""Get the global configuration."""
	if not config_manager.config:
		await config_manager.start()
	return config_manager.get_config()


async def validate_config_file(file_path: Path) -> bool:
	"""Validate a configuration file."""
	try:
		temp_manager = ConfigurationManager()
		temp_manager._config_file_paths = [file_path]
		await temp_manager.load_configuration()
		await temp_manager.validate_configuration()
		return True
	except Exception as e:
		logger.error(f"Configuration file validation failed: {e}")
		return False