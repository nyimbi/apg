"""
Application settings and configuration management

© 2025 Datacraft. All rights reserved.
"""

import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List
from functools import lru_cache

try:
	from pydantic_settings import BaseSettings
	from pydantic import Field, field_validator
except ImportError:
	# Fallback for older pydantic versions
	from pydantic import BaseSettings, Field
	def field_validator(field_name):
		def decorator(func):
			func.__validator_field__ = field_name
			return func
		return decorator

from .environment import Environment, get_environment, get_config_file_path


class DatabaseConfig(BaseSettings):
	"""Database configuration"""
	
	# SQLite configuration (for offline storage)
	sqlite_path: str = Field(default="~/.apg_workflow_mobile/offline.db")
	sqlite_timeout: float = Field(default=30.0, ge=1.0)
	sqlite_check_same_thread: bool = False
	
	# Cache database
	cache_sqlite_path: str = Field(default="~/.apg_workflow_mobile/cache.db")
	cache_ttl: int = Field(default=3600, ge=60)  # 1 hour
	
	# Connection pool settings
	pool_size: int = Field(default=10, ge=1, le=100)
	max_overflow: int = Field(default=20, ge=0)
	pool_timeout: int = Field(default=30, ge=1)
	
	@field_validator('sqlite_path', 'cache_sqlite_path')
	@classmethod
	def expand_user_path(cls, v):
		return str(Path(v).expanduser())


class APIConfig(BaseSettings):
	"""API configuration"""
	
	# Base URL configuration
	base_url: str = Field(...)
	version: str = Field(default="v1")
	
	# Request configuration
	timeout: float = Field(default=30.0, ge=1.0, le=300.0)
	retry_attempts: int = Field(default=3, ge=1, le=10)
	retry_delay: float = Field(default=1.0, ge=0.1, le=60.0)
	retry_backoff_multiplier: float = Field(default=2.0, ge=1.0, le=10.0)
	
	# Authentication
	api_key: Optional[str] = None
	client_id: Optional[str] = None
	client_secret: Optional[str] = None
	
	# Rate limiting
	rate_limit_requests: int = Field(default=100, ge=1)
	rate_limit_window: int = Field(default=60, ge=1)  # seconds
	
	@field_validator('base_url')
	@classmethod
	def validate_base_url(cls, v):
		if not v.startswith(('http://', 'https://')):
			raise ValueError('base_url must start with http:// or https://')
		return v.rstrip('/')


class SecurityConfig(BaseSettings):
	"""Security configuration"""
	
	# Encryption
	encryption_algorithm: str = Field(default="Fernet")
	encryption_key: Optional[str] = None
	
	# JWT configuration
	jwt_secret: Optional[str] = None
	jwt_algorithm: str = Field(default="HS256")
	jwt_expiry: int = Field(default=3600, ge=300)  # 1 hour
	jwt_refresh_threshold: int = Field(default=300, ge=60)  # 5 minutes
	
	# Keyring
	keyring_service: str = Field(default="co.ke.datacraft.apg-workflow-mobile")
	
	# Biometric settings
	biometric_timeout: int = Field(default=30, ge=5, le=300)
	biometric_max_attempts: int = Field(default=3, ge=1, le=10)
	
	# Password policy
	password_min_length: int = Field(default=8, ge=6)
	password_require_uppercase: bool = True
	password_require_lowercase: bool = True
	password_require_numbers: bool = True
	password_require_symbols: bool = False


class SyncConfig(BaseSettings):
	"""Synchronization configuration"""
	
	# Sync intervals
	sync_interval: int = Field(default=300, ge=30)  # 5 minutes
	sync_batch_size: int = Field(default=50, ge=1, le=1000)
	sync_timeout: int = Field(default=60, ge=10)  # 1 minute
	
	# Retry configuration
	max_sync_retries: int = Field(default=3, ge=1, le=10)
	sync_retry_delay: int = Field(default=5, ge=1, le=60)  # seconds
	
	# Conflict resolution
	conflict_resolution_strategy: str = Field(default="server_wins")  # server_wins, client_wins, manual
	
	# Background sync
	background_sync_enabled: bool = True
	background_sync_wifi_only: bool = False


class FileConfig(BaseSettings):
	"""File handling configuration"""
	
	# File size limits
	max_file_size: int = Field(default=100 * 1024 * 1024, ge=1024)  # 100MB
	max_total_storage: int = Field(default=1024 * 1024 * 1024, ge=1024 * 1024)  # 1GB
	
	# Supported file types
	supported_file_types: List[str] = Field(default=[
		".txt", ".pdf", ".doc", ".docx", ".xls", ".xlsx", 
		".jpg", ".jpeg", ".png", ".gif", ".mp4", ".mp3",
		".zip", ".rar", ".csv", ".json", ".xml"
	])
	
	# Upload configuration
	upload_chunk_size: int = Field(default=1024 * 1024, ge=1024)  # 1MB
	upload_timeout: int = Field(default=300, ge=30)  # 5 minutes
	download_timeout: int = Field(default=600, ge=60)  # 10 minutes
	
	# Cleanup settings
	temp_file_cleanup_interval: int = Field(default=3600, ge=300)  # 1 hour
	temp_file_max_age: int = Field(default=86400, ge=3600)  # 24 hours


class UIConfig(BaseSettings):
	"""UI configuration"""
	
	# Theme
	theme_mode: str = Field(default="auto")  # light, dark, auto
	primary_color: str = Field(default="#1976D2")
	secondary_color: str = Field(default="#FFC107")
	
	# Animation
	animation_duration: int = Field(default=300, ge=0, le=2000)  # milliseconds
	toast_duration: int = Field(default=3000, ge=1000, le=10000)  # milliseconds
	
	# Pagination
	default_page_size: int = Field(default=20, ge=5, le=100)
	max_page_size: int = Field(default=100, ge=10, le=1000)
	
	# Refresh
	pull_to_refresh_enabled: bool = True
	auto_refresh_interval: int = Field(default=60000, ge=5000)  # 1 minute in ms


class LoggingConfig(BaseSettings):
	"""Logging configuration"""
	
	# Log levels
	log_level: str = Field(default="INFO")
	console_log_level: str = Field(default="INFO")
	file_log_level: str = Field(default="DEBUG")
	
	# File logging
	log_file_path: str = Field(default="~/.apg_workflow_mobile/logs/app.log")
	log_max_size: int = Field(default=10 * 1024 * 1024, ge=1024)  # 10MB
	log_backup_count: int = Field(default=5, ge=1, le=50)
	
	# Log format
	log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
	log_date_format: str = Field(default="%Y-%m-%d %H:%M:%S")
	
	# Features
	colored_logs: bool = True
	structured_logging: bool = False  # JSON format
	
	@field_validator('log_file_path')
	@classmethod
	def expand_log_path(cls, v):
		return str(Path(v).expanduser())


class FeatureFlags(BaseSettings):
	"""Feature flags configuration"""
	
	# Authentication features
	biometric_auth_enabled: bool = True
	two_factor_auth_enabled: bool = False
	social_login_enabled: bool = False
	
	# Core features
	offline_mode_enabled: bool = True
	real_time_sync_enabled: bool = True
	push_notifications_enabled: bool = True
	
	# File features
	file_upload_enabled: bool = True
	file_preview_enabled: bool = True
	
	# Advanced features
	voice_commands_enabled: bool = False
	analytics_enabled: bool = True
	crash_reporting_enabled: bool = True
	
	# Debug features
	debug_mode_enabled: bool = False
	mock_api_enabled: bool = False
	verbose_logging_enabled: bool = False


class Settings(BaseSettings):
	"""Main application settings"""
	
	# Application info
	app_name: str = Field(default="APG Workflow Manager")
	app_version: str = Field(default="1.0.0")
	company_name: str = Field(default="Datacraft")
	company_url: str = Field(default="https://www.datacraft.co.ke")
	
	# Environment
	environment: Environment = Field(default_factory=get_environment)
	debug: bool = Field(default=False)
	
	# Configuration sections
	database: DatabaseConfig = Field(default_factory=DatabaseConfig)
	api: APIConfig
	security: SecurityConfig = Field(default_factory=SecurityConfig)
	sync: SyncConfig = Field(default_factory=SyncConfig)
	files: FileConfig = Field(default_factory=FileConfig)
	ui: UIConfig = Field(default_factory=UIConfig)
	logging: LoggingConfig = Field(default_factory=LoggingConfig)
	features: FeatureFlags = Field(default_factory=FeatureFlags)
	
	class Config:
		env_prefix = "APG_"
		env_nested_delimiter = "__"
		case_sensitive = False
		
	@classmethod
	def load_from_file(cls, config_file: Optional[str] = None) -> "Settings":
		"""Load settings from YAML configuration file"""
		if not config_file:
			config_file = get_config_file_path()
		
		if config_file and os.path.exists(config_file):
			with open(config_file, 'r') as f:
				config_data = yaml.safe_load(f)
			
			# Override with environment variables
			env_overrides = {}
			for key, value in os.environ.items():
				if key.startswith("APG_"):
					env_key = key[4:].lower()  # Remove APG_ prefix
					env_overrides[env_key] = value
			
			# Merge configuration
			if config_data:
				config_data.update(env_overrides)
				return cls(**config_data)
		
		# Fallback to environment variables only
		return cls()
	
	@property
	def is_development(self) -> bool:
		"""Check if running in development environment"""
		return self.environment == Environment.DEVELOPMENT
	
	@property
	def is_production(self) -> bool:
		"""Check if running in production environment"""
		return self.environment == Environment.PRODUCTION
	
	@property
	def is_testing(self) -> bool:
		"""Check if running in testing environment"""
		return self.environment == Environment.TESTING


@lru_cache()
def get_settings() -> Settings:
	"""Get cached application settings"""
	return Settings.load_from_file()


def reload_settings() -> Settings:
	"""Reload settings (clears cache)"""
	get_settings.cache_clear()
	return get_settings()