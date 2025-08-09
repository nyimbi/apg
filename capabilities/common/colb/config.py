"""
Configuration Management for APG Real-Time Collaboration

Handles environment-specific configuration with defaults and validation.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DatabaseConfig:
	"""Database configuration"""
	url: str = field(default_factory=lambda: os.getenv(
		'DATABASE_URL', 
		'postgresql://rtc_user:rtc_password@localhost:5432/apg_rtc'
	))
	pool_size: int = field(default_factory=lambda: int(os.getenv('DB_POOL_SIZE', '20')))
	max_overflow: int = field(default_factory=lambda: int(os.getenv('DB_MAX_OVERFLOW', '30')))
	pool_timeout: int = field(default_factory=lambda: int(os.getenv('DB_POOL_TIMEOUT', '30')))
	echo: bool = field(default_factory=lambda: os.getenv('DB_ECHO', 'false').lower() == 'true')


@dataclass
class RedisConfig:
	"""Redis configuration"""
	url: str = field(default_factory=lambda: os.getenv(
		'REDIS_URL', 
		'redis://localhost:6379/0'
	))
	pool_size: int = field(default_factory=lambda: int(os.getenv('REDIS_POOL_SIZE', '50')))
	pool_timeout: int = field(default_factory=lambda: int(os.getenv('REDIS_POOL_TIMEOUT', '5')))
	password: Optional[str] = field(default_factory=lambda: os.getenv('REDIS_PASSWORD'))


@dataclass
class WebSocketConfig:
	"""WebSocket configuration"""
	host: str = field(default_factory=lambda: os.getenv('WEBSOCKET_HOST', '0.0.0.0'))
	port: int = field(default_factory=lambda: int(os.getenv('WEBSOCKET_PORT', '8765')))
	max_connections: int = field(default_factory=lambda: int(os.getenv('WS_MAX_CONNECTIONS', '10000')))
	ping_interval: int = field(default_factory=lambda: int(os.getenv('WS_PING_INTERVAL', '30')))
	ping_timeout: int = field(default_factory=lambda: int(os.getenv('WS_PING_TIMEOUT', '10')))
	heartbeat_interval: int = field(default_factory=lambda: int(os.getenv('WS_HEARTBEAT_INTERVAL', '30')))


@dataclass
class APIConfig:
	"""API configuration"""
	host: str = field(default_factory=lambda: os.getenv('API_HOST', '0.0.0.0'))
	port: int = field(default_factory=lambda: int(os.getenv('API_PORT', '8000')))
	workers: int = field(default_factory=lambda: int(os.getenv('API_WORKERS', '4')))
	debug: bool = field(default_factory=lambda: os.getenv('API_DEBUG', 'false').lower() == 'true')
	reload: bool = field(default_factory=lambda: os.getenv('API_RELOAD', 'false').lower() == 'true')


@dataclass
class SecurityConfig:
	"""Security configuration"""
	secret_key: str = field(default_factory=lambda: os.getenv(
		'SECRET_KEY', 
		'dev-secret-key-change-in-production-to-secure-random-string'
	))
	cors_origins: list[str] = field(default_factory=lambda: 
		os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://localhost:8080').split(',')
	)
	allowed_hosts: list[str] = field(default_factory=lambda:
		os.getenv('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')
	)
	ssl_required: bool = field(default_factory=lambda: os.getenv('SSL_REQUIRED', 'false').lower() == 'true')
	hsts_max_age: int = field(default_factory=lambda: int(os.getenv('HSTS_MAX_AGE', '31536000')))


@dataclass
class APGIntegrationConfig:
	"""APG service integration configuration"""
	auth_service_url: str = field(default_factory=lambda: os.getenv(
		'APG_AUTH_SERVICE_URL', 
		'http://localhost:8001'
	))
	ai_service_url: str = field(default_factory=lambda: os.getenv(
		'APG_AI_SERVICE_URL', 
		'http://localhost:8002'
	))
	notification_service_url: str = field(default_factory=lambda: os.getenv(
		'APG_NOTIFICATION_SERVICE_URL', 
		'http://localhost:8003'
	))
	tenant_id: str = field(default_factory=lambda: os.getenv('APG_TENANT_ID', 'default'))
	api_timeout: int = field(default_factory=lambda: int(os.getenv('APG_API_TIMEOUT', '30')))


@dataclass
class ThirdPartyIntegrationConfig:
	"""Third-party platform integration configuration"""
	
	# Microsoft Teams
	teams_enabled: bool = field(default_factory=lambda: os.getenv('TEAMS_ENABLED', 'false').lower() == 'true')
	teams_client_id: Optional[str] = field(default_factory=lambda: os.getenv('TEAMS_CLIENT_ID'))
	teams_client_secret: Optional[str] = field(default_factory=lambda: os.getenv('TEAMS_CLIENT_SECRET'))
	teams_tenant_id: Optional[str] = field(default_factory=lambda: os.getenv('TEAMS_TENANT_ID'))
	
	# Zoom
	zoom_enabled: bool = field(default_factory=lambda: os.getenv('ZOOM_ENABLED', 'false').lower() == 'true')
	zoom_api_key: Optional[str] = field(default_factory=lambda: os.getenv('ZOOM_API_KEY'))
	zoom_api_secret: Optional[str] = field(default_factory=lambda: os.getenv('ZOOM_API_SECRET'))
	zoom_account_id: Optional[str] = field(default_factory=lambda: os.getenv('ZOOM_ACCOUNT_ID'))
	
	# Google Meet
	google_meet_enabled: bool = field(default_factory=lambda: os.getenv('GOOGLE_MEET_ENABLED', 'false').lower() == 'true')
	google_client_id: Optional[str] = field(default_factory=lambda: os.getenv('GOOGLE_CLIENT_ID'))
	google_client_secret: Optional[str] = field(default_factory=lambda: os.getenv('GOOGLE_CLIENT_SECRET'))
	google_workspace_domain: Optional[str] = field(default_factory=lambda: os.getenv('GOOGLE_WORKSPACE_DOMAIN'))


@dataclass
class LoggingConfig:
	"""Logging configuration"""
	level: str = field(default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO'))
	format: str = field(default_factory=lambda: os.getenv(
		'LOG_FORMAT', 
		'%(asctime)s - %(name)s - %(levelname)s - %(message)s'
	))
	file_path: Optional[str] = field(default_factory=lambda: os.getenv('LOG_FILE_PATH'))
	max_bytes: int = field(default_factory=lambda: int(os.getenv('LOG_MAX_BYTES', str(50 * 1024 * 1024))))
	backup_count: int = field(default_factory=lambda: int(os.getenv('LOG_BACKUP_COUNT', '10')))


@dataclass
class PerformanceConfig:
	"""Performance and scaling configuration"""
	max_concurrent_sessions: int = field(default_factory=lambda: int(os.getenv('MAX_CONCURRENT_SESSIONS', '1000')))
	max_participants_per_session: int = field(default_factory=lambda: int(os.getenv('MAX_PARTICIPANTS_PER_SESSION', '100')))
	max_video_calls: int = field(default_factory=lambda: int(os.getenv('MAX_VIDEO_CALLS', '50')))
	message_rate_limit: int = field(default_factory=lambda: int(os.getenv('MESSAGE_RATE_LIMIT', '1000')))
	recording_size_limit_gb: int = field(default_factory=lambda: int(os.getenv('RECORDING_SIZE_LIMIT_GB', '10')))
	session_timeout_minutes: int = field(default_factory=lambda: int(os.getenv('SESSION_TIMEOUT_MINUTES', '480')))


@dataclass
class AppConfig:
	"""Main application configuration"""
	environment: str = field(default_factory=lambda: os.getenv('ENVIRONMENT', 'development'))
	debug: bool = field(default_factory=lambda: os.getenv('DEBUG', 'false').lower() == 'true')
	testing: bool = field(default_factory=lambda: os.getenv('TESTING', 'false').lower() == 'true')
	
	# Sub-configurations
	database: DatabaseConfig = field(default_factory=DatabaseConfig)
	redis: RedisConfig = field(default_factory=RedisConfig)
	websocket: WebSocketConfig = field(default_factory=WebSocketConfig)
	api: APIConfig = field(default_factory=APIConfig)
	security: SecurityConfig = field(default_factory=SecurityConfig)
	apg_integration: APGIntegrationConfig = field(default_factory=APGIntegrationConfig)
	third_party: ThirdPartyIntegrationConfig = field(default_factory=ThirdPartyIntegrationConfig)
	logging: LoggingConfig = field(default_factory=LoggingConfig)
	performance: PerformanceConfig = field(default_factory=PerformanceConfig)
	
	def __post_init__(self):
		"""Validate configuration after initialization"""
		self._validate_config()
	
	def _validate_config(self):
		"""Validate configuration values"""
		errors = []
		
		# Validate required fields in production
		if self.environment == 'production':
			if self.security.secret_key == 'dev-secret-key-change-in-production-to-secure-random-string':
				errors.append("SECRET_KEY must be set to a secure random string in production")
			
			if not self.security.ssl_required:
				errors.append("SSL_REQUIRED should be true in production")
		
		# Validate numeric limits
		if self.performance.max_concurrent_sessions <= 0:
			errors.append("MAX_CONCURRENT_SESSIONS must be greater than 0")
		
		if self.performance.max_participants_per_session <= 0:
			errors.append("MAX_PARTICIPANTS_PER_SESSION must be greater than 0")
		
		# Validate third-party integration configs
		if self.third_party.teams_enabled:
			if not all([self.third_party.teams_client_id, self.third_party.teams_client_secret]):
				errors.append("Teams integration enabled but missing client_id or client_secret")
		
		if self.third_party.zoom_enabled:
			if not all([self.third_party.zoom_api_key, self.third_party.zoom_api_secret]):
				errors.append("Zoom integration enabled but missing api_key or api_secret")
		
		if self.third_party.google_meet_enabled:
			if not all([self.third_party.google_client_id, self.third_party.google_client_secret]):
				errors.append("Google Meet integration enabled but missing client_id or client_secret")
		
		if errors:
			raise ValueError("Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors))
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert configuration to dictionary"""
		return {
			'environment': self.environment,
			'debug': self.debug,
			'testing': self.testing,
			'database': {
				'url': self.database.url,
				'pool_size': self.database.pool_size,
				'max_overflow': self.database.max_overflow,
				'pool_timeout': self.database.pool_timeout,
				'echo': self.database.echo
			},
			'redis': {
				'url': self.redis.url,
				'pool_size': self.redis.pool_size,
				'pool_timeout': self.redis.pool_timeout
			},
			'websocket': {
				'host': self.websocket.host,
				'port': self.websocket.port,
				'max_connections': self.websocket.max_connections,
				'ping_interval': self.websocket.ping_interval,
				'ping_timeout': self.websocket.ping_timeout
			},
			'api': {
				'host': self.api.host,
				'port': self.api.port,
				'workers': self.api.workers,
				'debug': self.api.debug,
				'reload': self.api.reload
			},
			'performance': {
				'max_concurrent_sessions': self.performance.max_concurrent_sessions,
				'max_participants_per_session': self.performance.max_participants_per_session,
				'max_video_calls': self.performance.max_video_calls,
				'message_rate_limit': self.performance.message_rate_limit,
				'session_timeout_minutes': self.performance.session_timeout_minutes
			}
		}


# Global configuration instance
config = AppConfig()


def load_config_from_file(config_path: str) -> AppConfig:
	"""Load configuration from file"""
	config_file = Path(config_path)
	
	if not config_file.exists():
		raise FileNotFoundError(f"Configuration file not found: {config_path}")
	
	# Load environment variables from file if it's a .env file
	if config_file.suffix == '.env':
		_load_env_file(config_file)
	
	# Return new config instance with updated environment variables
	return AppConfig()


def _load_env_file(env_file: Path):
	"""Load environment variables from .env file"""
	with open(env_file, 'r') as f:
		for line in f:
			line = line.strip()
			if line and not line.startswith('#') and '=' in line:
				key, value = line.split('=', 1)
				os.environ[key.strip()] = value.strip()


def get_config() -> AppConfig:
	"""Get the global configuration instance"""
	return config


# Configuration validation utility
def validate_config(config_instance: AppConfig = None) -> list[str]:
	"""Validate configuration and return list of errors"""
	if config_instance is None:
		config_instance = config
	
	errors = []
	
	try:
		config_instance._validate_config()
	except ValueError as e:
		errors = str(e).split('\n')[1:]  # Skip first line "Configuration validation failed:"
		errors = [error.lstrip('- ') for error in errors if error.strip()]
	
	return errors


# Environment-specific configuration presets
def get_development_config() -> AppConfig:
	"""Get development environment configuration"""
	os.environ.update({
		'ENVIRONMENT': 'development',
		'DEBUG': 'true',
		'API_RELOAD': 'true',
		'DB_ECHO': 'true',
		'LOG_LEVEL': 'DEBUG'
	})
	return AppConfig()


def get_production_config() -> AppConfig:
	"""Get production environment configuration"""
	os.environ.update({
		'ENVIRONMENT': 'production',
		'DEBUG': 'false',
		'API_RELOAD': 'false',
		'DB_ECHO': 'false',
		'SSL_REQUIRED': 'true',
		'LOG_LEVEL': 'INFO'
	})
	return AppConfig()


def get_testing_config() -> AppConfig:
	"""Get testing environment configuration"""
	os.environ.update({
		'ENVIRONMENT': 'testing',
		'TESTING': 'true',
		'DATABASE_URL': 'postgresql://rtc_user:rtc_password@localhost:5432/apg_rtc_test',
		'REDIS_URL': 'redis://localhost:6379/1',
		'LOG_LEVEL': 'WARNING'
	})
	return AppConfig()