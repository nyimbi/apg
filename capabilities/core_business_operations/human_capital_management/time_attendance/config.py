"""
Time & Attendance Capability Configuration

Comprehensive configuration management for the revolutionary APG Time & Attendance
capability with multi-tenant support, AI model settings, and integration parameters.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import os
import json
from datetime import timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal
from pydantic import BaseModel, Field, ConfigDict, validator


class EnvironmentType(str, Enum):
	"""Environment types"""
	DEVELOPMENT = "development"
	TESTING = "testing"
	STAGING = "staging"
	PRODUCTION = "production"


class TimeTrackingMode(str, Enum):
	"""Time tracking modes"""
	BASIC = "basic"
	ADVANCED = "advanced"
	AI_POWERED = "ai_powered"
	BIOMETRIC_ENABLED = "biometric_enabled"


@dataclass
class DatabaseConfig:
	"""Database configuration"""
	host: str = os.getenv("TA_DB_HOST", "localhost")
	port: int = int(os.getenv("TA_DB_PORT", "5432"))
	database: str = os.getenv("TA_DB_NAME", "apg_time_attendance")
	username: str = os.getenv("TA_DB_USER", "postgres")
	password: str = os.getenv("TA_DB_PASSWORD", "")
	pool_size: int = int(os.getenv("TA_DB_POOL_SIZE", "10"))
	max_overflow: int = int(os.getenv("TA_DB_MAX_OVERFLOW", "20"))
	pool_timeout: int = int(os.getenv("TA_DB_POOL_TIMEOUT", "30"))
	echo: bool = os.getenv("TA_DB_ECHO", "false").lower() == "true"
	
	@property
	def connection_string(self) -> str:
		"""Generate PostgreSQL connection string"""
		return f"postgresql+asyncpg://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class RedisConfig:
	"""Redis configuration for caching and sessions"""
	host: str = os.getenv("TA_REDIS_HOST", "localhost")
	port: int = int(os.getenv("TA_REDIS_PORT", "6379"))
	database: int = int(os.getenv("TA_REDIS_DB", "0"))
	password: Optional[str] = os.getenv("TA_REDIS_PASSWORD")
	max_connections: int = int(os.getenv("TA_REDIS_MAX_CONNECTIONS", "50"))
	socket_timeout: float = float(os.getenv("TA_REDIS_SOCKET_TIMEOUT", "5.0"))
	health_check_interval: int = int(os.getenv("TA_REDIS_HEALTH_CHECK", "30"))
	
	@property
	def connection_string(self) -> str:
		"""Generate Redis connection string"""
		auth = f":{self.password}@" if self.password else ""
		return f"redis://{auth}{self.host}:{self.port}/{self.database}"


@dataclass
class AIModelConfig:
	"""AI/ML model configuration"""
	fraud_detection_model: str = os.getenv("TA_FRAUD_MODEL", "ta_fraud_detector_v1")
	prediction_model: str = os.getenv("TA_PREDICTION_MODEL", "ta_workforce_predictor_v1")
	optimization_model: str = os.getenv("TA_OPTIMIZATION_MODEL", "ta_schedule_optimizer_v1")
	behavioral_model: str = os.getenv("TA_BEHAVIORAL_MODEL", "ta_behavior_analyzer_v1")
	
	# Model thresholds
	fraud_detection_threshold: float = float(os.getenv("TA_FRAUD_THRESHOLD", "0.8"))
	anomaly_detection_threshold: float = float(os.getenv("TA_ANOMALY_THRESHOLD", "0.7"))
	prediction_confidence_threshold: float = float(os.getenv("TA_PREDICTION_THRESHOLD", "0.75"))
	
	# Model update intervals
	model_retrain_interval_days: int = int(os.getenv("TA_MODEL_RETRAIN_DAYS", "30"))
	model_evaluation_interval_days: int = int(os.getenv("TA_MODEL_EVAL_DAYS", "7"))
	
	# Performance targets
	target_accuracy: float = float(os.getenv("TA_TARGET_ACCURACY", "0.998"))
	target_processing_time_ms: int = int(os.getenv("TA_TARGET_PROCESSING_MS", "200"))


@dataclass
class BiometricConfig:
	"""Biometric authentication configuration"""
	enabled: bool = os.getenv("TA_BIOMETRIC_ENABLED", "true").lower() == "true"
	facial_recognition_enabled: bool = os.getenv("TA_FACIAL_ENABLED", "true").lower() == "true"
	fingerprint_enabled: bool = os.getenv("TA_FINGERPRINT_ENABLED", "false").lower() == "true"
	liveness_detection_enabled: bool = os.getenv("TA_LIVENESS_ENABLED", "true").lower() == "true"
	
	# Quality thresholds
	minimum_template_quality: float = float(os.getenv("TA_MIN_TEMPLATE_QUALITY", "0.8"))
	minimum_match_score: float = float(os.getenv("TA_MIN_MATCH_SCORE", "0.85"))
	minimum_liveness_score: float = float(os.getenv("TA_MIN_LIVENESS_SCORE", "0.9"))
	
	# Data retention
	template_retention_days: int = int(os.getenv("TA_TEMPLATE_RETENTION_DAYS", "30"))
	session_retention_days: int = int(os.getenv("TA_SESSION_RETENTION_DAYS", "7"))
	
	# Integration
	computer_vision_capability_url: str = os.getenv(
		"TA_CV_CAPABILITY_URL", 
		"http://localhost:5000/api/computer_vision"
	)


@dataclass
class LocationConfig:
	"""Location and geofencing configuration"""
	geofencing_enabled: bool = os.getenv("TA_GEOFENCING_ENABLED", "false").lower() == "true"
	gps_accuracy_threshold_meters: float = float(os.getenv("TA_GPS_ACCURACY_THRESHOLD", "50.0"))
	location_tolerance_meters: float = float(os.getenv("TA_LOCATION_TOLERANCE", "100.0"))
	
	# Default office locations (can be overridden per tenant)
	default_office_locations: List[Dict[str, Any]] = field(default_factory=lambda: [
		{
			"name": "Main Office",
			"latitude": 0.0,
			"longitude": 0.0,
			"radius_meters": 100.0,
			"enabled": False
		}
	])


@dataclass
class ComplianceConfig:
	"""Compliance and regulatory configuration"""
	flsa_compliance_enabled: bool = os.getenv("TA_FLSA_ENABLED", "true").lower() == "true"
	gdpr_compliance_enabled: bool = os.getenv("TA_GDPR_ENABLED", "true").lower() == "true"
	ccpa_compliance_enabled: bool = os.getenv("TA_CCPA_ENABLED", "false").lower() == "true"
	
	# Overtime calculations
	overtime_threshold_hours: float = float(os.getenv("TA_OVERTIME_THRESHOLD", "40.0"))
	daily_overtime_threshold_hours: float = float(os.getenv("TA_DAILY_OVERTIME_THRESHOLD", "8.0"))
	double_time_threshold_hours: float = float(os.getenv("TA_DOUBLE_TIME_THRESHOLD", "12.0"))
	
	# Break requirements
	minimum_break_minutes: int = int(os.getenv("TA_MIN_BREAK_MINUTES", "30"))
	break_auto_deduction: bool = os.getenv("TA_BREAK_AUTO_DEDUCTION", "true").lower() == "true"
	
	# Audit and retention
	audit_retention_years: int = int(os.getenv("TA_AUDIT_RETENTION_YEARS", "7"))
	compliance_report_frequency_days: int = int(os.getenv("TA_COMPLIANCE_REPORT_DAYS", "1"))


@dataclass
class NotificationConfig:
	"""Notification and alert configuration"""
	enabled: bool = os.getenv("TA_NOTIFICATIONS_ENABLED", "true").lower() == "true"
	
	# Notification channels
	email_enabled: bool = os.getenv("TA_EMAIL_ENABLED", "true").lower() == "true"
	sms_enabled: bool = os.getenv("TA_SMS_ENABLED", "false").lower() == "true"
	push_enabled: bool = os.getenv("TA_PUSH_ENABLED", "true").lower() == "true"
	in_app_enabled: bool = os.getenv("TA_IN_APP_ENABLED", "true").lower() == "true"
	
	# Notification engine integration
	notification_engine_url: str = os.getenv(
		"TA_NOTIFICATION_ENGINE_URL",
		"http://localhost:5000/api/notification_engine"
	)
	
	# Alert thresholds
	fraud_alert_threshold: float = float(os.getenv("TA_FRAUD_ALERT_THRESHOLD", "0.8"))
	overtime_alert_threshold_hours: float = float(os.getenv("TA_OVERTIME_ALERT_THRESHOLD", "10.0"))
	absence_alert_enabled: bool = os.getenv("TA_ABSENCE_ALERT_ENABLED", "true").lower() == "true"


@dataclass
class WorkflowConfig:
	"""Workflow and approval configuration"""
	approval_workflows_enabled: bool = os.getenv("TA_APPROVAL_WORKFLOWS_ENABLED", "true").lower() == "true"
	auto_approval_enabled: bool = os.getenv("TA_AUTO_APPROVAL_ENABLED", "true").lower() == "true"
	
	# Approval thresholds
	auto_approval_threshold_hours: float = float(os.getenv("TA_AUTO_APPROVAL_THRESHOLD", "8.5"))
	manager_approval_required_hours: float = float(os.getenv("TA_MANAGER_APPROVAL_THRESHOLD", "10.0"))
	
	# Escalation
	escalation_timeout_hours: int = int(os.getenv("TA_ESCALATION_TIMEOUT_HOURS", "24"))
	max_escalation_levels: int = int(os.getenv("TA_MAX_ESCALATION_LEVELS", "3"))
	
	# Workflow BPM integration
	workflow_bpm_url: str = os.getenv(
		"TA_WORKFLOW_BPM_URL",
		"http://localhost:5000/api/workflow_bpm"
	)


@dataclass
class PerformanceConfig:
	"""Performance and optimization configuration"""
	# Response time targets
	target_response_time_ms: int = int(os.getenv("TA_TARGET_RESPONSE_MS", "200"))
	target_availability_percent: float = float(os.getenv("TA_TARGET_AVAILABILITY", "99.99"))
	
	# Caching
	cache_enabled: bool = os.getenv("TA_CACHE_ENABLED", "true").lower() == "true"
	cache_ttl_seconds: int = int(os.getenv("TA_CACHE_TTL_SECONDS", "300"))
	
	# Rate limiting
	rate_limit_enabled: bool = os.getenv("TA_RATE_LIMIT_ENABLED", "true").lower() == "true"
	rate_limit_requests_per_minute: int = int(os.getenv("TA_RATE_LIMIT_RPM", "1000"))
	
	# Connection pooling
	max_concurrent_connections: int = int(os.getenv("TA_MAX_CONNECTIONS", "100"))
	connection_timeout_seconds: int = int(os.getenv("TA_CONNECTION_TIMEOUT", "30"))


@dataclass
class IntegrationConfig:
	"""APG capability integration configuration"""
	# Employee Data Management
	employee_data_mgmt_url: str = os.getenv(
		"TA_EDM_URL",
		"http://localhost:5000/api/human_capital_management/employee_data_management"
	)
	
	# Payroll
	payroll_url: str = os.getenv(
		"TA_PAYROLL_URL",
		"http://localhost:5000/api/human_capital_management/payroll"
	)
	
	# Auth RBAC
	auth_rbac_url: str = os.getenv(
		"TA_AUTH_RBAC_URL",
		"http://localhost:5000/api/auth_rbac"
	)
	
	# Computer Vision
	computer_vision_url: str = os.getenv(
		"TA_COMPUTER_VISION_URL",
		"http://localhost:5000/api/computer_vision"
	)
	
	# IoT Management
	iot_management_url: str = os.getenv(
		"TA_IOT_MANAGEMENT_URL",
		"http://localhost:5000/api/iot_management"
	)
	
	# Data Analytics
	data_analytics_url: str = os.getenv(
		"TA_DATA_ANALYTICS_URL",
		"http://localhost:5000/api/data_analytics"
	)
	
	# Integration timeouts
	integration_timeout_seconds: int = int(os.getenv("TA_INTEGRATION_TIMEOUT", "30"))
	integration_retry_attempts: int = int(os.getenv("TA_INTEGRATION_RETRIES", "3"))
	integration_retry_backoff_seconds: float = float(os.getenv("TA_INTEGRATION_BACKOFF", "1.0"))


@dataclass
class SecurityConfig:
	"""Security configuration"""
	# Encryption
	encryption_key: str = os.getenv("TA_ENCRYPTION_KEY", "change-me-in-production")
	biometric_encryption_enabled: bool = os.getenv("TA_BIOMETRIC_ENCRYPTION", "true").lower() == "true"
	
	# Session management
	session_timeout_minutes: int = int(os.getenv("TA_SESSION_TIMEOUT_MINUTES", "60"))
	max_failed_attempts: int = int(os.getenv("TA_MAX_FAILED_ATTEMPTS", "5"))
	lockout_duration_minutes: int = int(os.getenv("TA_LOCKOUT_DURATION_MINUTES", "15"))
	
	# CORS
	cors_enabled: bool = os.getenv("TA_CORS_ENABLED", "true").lower() == "true"
	cors_origins: List[str] = os.getenv("TA_CORS_ORIGINS", "*").split(",")
	
	# JWT
	jwt_secret_key: str = os.getenv("TA_JWT_SECRET", "change-me-in-production")
	jwt_expiration_hours: int = int(os.getenv("TA_JWT_EXPIRATION_HOURS", "24"))


class TimeAttendanceConfig:
	"""
	Main configuration class for Time & Attendance capability
	
	Centralized configuration management with environment-based overrides,
	validation, and integration with APG ecosystem capabilities.
	"""
	
	def __init__(self, environment: EnvironmentType = EnvironmentType.DEVELOPMENT):
		self.environment = environment
		self.tracking_mode = TimeTrackingMode(
			os.getenv("TA_TRACKING_MODE", TimeTrackingMode.AI_POWERED.value)
		)
		
		# Core configurations
		self.database = DatabaseConfig()
		self.redis = RedisConfig()
		self.ai_models = AIModelConfig()
		self.biometric = BiometricConfig()
		self.location = LocationConfig()
		self.compliance = ComplianceConfig()
		self.notifications = NotificationConfig()
		self.workflow = WorkflowConfig()
		self.performance = PerformanceConfig()
		self.integration = IntegrationConfig()
		self.security = SecurityConfig()
		
		# Feature flags
		self.features = self._load_feature_flags()
		
		# Validate configuration
		self._validate_config()
	
	def _load_feature_flags(self) -> Dict[str, bool]:
		"""Load feature flags from environment or defaults"""
		return {
			"ai_fraud_detection": os.getenv("TA_FEATURE_AI_FRAUD", "true").lower() == "true",
			"biometric_authentication": os.getenv("TA_FEATURE_BIOMETRIC", "true").lower() == "true",
			"predictive_analytics": os.getenv("TA_FEATURE_PREDICTIVE", "true").lower() == "true",
			"real_time_processing": os.getenv("TA_FEATURE_REALTIME", "true").lower() == "true",
			"geofencing": os.getenv("TA_FEATURE_GEOFENCING", "false").lower() == "true",
			"voice_commands": os.getenv("TA_FEATURE_VOICE", "false").lower() == "true",
			"mobile_app": os.getenv("TA_FEATURE_MOBILE", "true").lower() == "true",
			"overtime_calculation": os.getenv("TA_FEATURE_OVERTIME", "true").lower() == "true",
			"leave_management": os.getenv("TA_FEATURE_LEAVE", "true").lower() == "true",
			"schedule_optimization": os.getenv("TA_FEATURE_SCHEDULE_OPT", "true").lower() == "true",
			"compliance_automation": os.getenv("TA_FEATURE_COMPLIANCE", "true").lower() == "true",
			"advanced_reporting": os.getenv("TA_FEATURE_REPORTING", "true").lower() == "true",
		}
	
	def _validate_config(self) -> None:
		"""Validate configuration settings"""
		errors = []
		
		# Database validation
		if not self.database.host:
			errors.append("Database host is required")
		
		if not self.database.database:
			errors.append("Database name is required")
		
		# Security validation
		if self.environment == EnvironmentType.PRODUCTION:
			if self.security.encryption_key == "change-me-in-production":
				errors.append("Production encryption key must be changed")
			
			if self.security.jwt_secret_key == "change-me-in-production":
				errors.append("Production JWT secret must be changed")
		
		# AI model validation
		if self.features["ai_fraud_detection"]:
			if not (0.0 <= self.ai_models.fraud_detection_threshold <= 1.0):
				errors.append("Fraud detection threshold must be between 0.0 and 1.0")
		
		# Biometric validation
		if self.features["biometric_authentication"]:
			if not (0.0 <= self.biometric.minimum_match_score <= 1.0):
				errors.append("Biometric match score threshold must be between 0.0 and 1.0")
		
		if errors:
			raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
	
	def is_feature_enabled(self, feature_name: str) -> bool:
		"""Check if a feature is enabled"""
		return self.features.get(feature_name, False)
	
	def get_capability_url(self, capability_name: str) -> Optional[str]:
		"""Get URL for APG capability integration"""
		capability_urls = {
			"employee_data_management": self.integration.employee_data_mgmt_url,
			"payroll": self.integration.payroll_url,
			"auth_rbac": self.integration.auth_rbac_url,
			"computer_vision": self.integration.computer_vision_url,
			"iot_management": self.integration.iot_management_url,
			"data_analytics": self.integration.data_analytics_url,
			"notification_engine": self.notifications.notification_engine_url,
			"workflow_bpm": self.workflow.workflow_bpm_url,
		}
		return capability_urls.get(capability_name)
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert configuration to dictionary"""
		return {
			"environment": self.environment.value,
			"tracking_mode": self.tracking_mode.value,
			"features": self.features,
			"database": {
				"host": self.database.host,
				"port": self.database.port,
				"database": self.database.database,
				"pool_size": self.database.pool_size,
			},
			"performance": {
				"target_response_time_ms": self.performance.target_response_time_ms,
				"target_availability_percent": self.performance.target_availability_percent,
				"cache_enabled": self.performance.cache_enabled,
			},
			"ai_models": {
				"fraud_detection_threshold": self.ai_models.fraud_detection_threshold,
				"target_accuracy": self.ai_models.target_accuracy,
			},
			"compliance": {
				"flsa_enabled": self.compliance.flsa_compliance_enabled,
				"gdpr_enabled": self.compliance.gdpr_compliance_enabled,
				"overtime_threshold": self.compliance.overtime_threshold_hours,
			}
		}
	
	@classmethod
	def from_file(cls, config_file_path: str) -> 'TimeAttendanceConfig':
		"""Load configuration from JSON file"""
		with open(config_file_path, 'r') as f:
			config_data = json.load(f)
		
		# Set environment variables from config file
		for key, value in config_data.items():
			if isinstance(value, dict):
				for sub_key, sub_value in value.items():
					env_key = f"TA_{key.upper()}_{sub_key.upper()}"
					os.environ[env_key] = str(sub_value)
			else:
				env_key = f"TA_{key.upper()}"
				os.environ[env_key] = str(value)
		
		environment = EnvironmentType(config_data.get("environment", "development"))
		return cls(environment)


# Global configuration instance
config = TimeAttendanceConfig()


# Utility functions
def get_config() -> TimeAttendanceConfig:
	"""Get global configuration instance"""
	return config


def update_config(updates: Dict[str, Any]) -> None:
	"""Update configuration with new values"""
	global config
	
	# Apply updates to environment variables
	for key, value in updates.items():
		env_key = f"TA_{key.upper()}"
		os.environ[env_key] = str(value)
	
	# Recreate configuration instance
	config = TimeAttendanceConfig(config.environment)


def validate_tenant_config(tenant_id: str, tenant_config: Dict[str, Any]) -> List[str]:
	"""Validate tenant-specific configuration"""
	errors = []
	
	# Required tenant settings
	required_fields = ["company_name", "timezone", "business_hours"]
	for field in required_fields:
		if field not in tenant_config:
			errors.append(f"Required field '{field}' missing for tenant {tenant_id}")
	
	# Business hours validation
	if "business_hours" in tenant_config:
		business_hours = tenant_config["business_hours"]
		if not isinstance(business_hours, dict):
			errors.append("Business hours must be a dictionary")
		else:
			required_days = ["monday", "tuesday", "wednesday", "thursday", "friday"]
			for day in required_days:
				if day not in business_hours:
					errors.append(f"Business hours missing for {day}")
	
	# Overtime rules validation
	if "overtime_rules" in tenant_config:
		overtime_rules = tenant_config["overtime_rules"]
		if "threshold_hours" in overtime_rules:
			if not isinstance(overtime_rules["threshold_hours"], (int, float)) or overtime_rules["threshold_hours"] <= 0:
				errors.append("Overtime threshold hours must be a positive number")
	
	return errors


# Export configuration classes and functions
__all__ = [
	"TimeAttendanceConfig",
	"EnvironmentType", 
	"TimeTrackingMode",
	"DatabaseConfig",
	"RedisConfig", 
	"AIModelConfig",
	"BiometricConfig",
	"LocationConfig",
	"ComplianceConfig",
	"NotificationConfig",
	"WorkflowConfig",
	"PerformanceConfig",
	"IntegrationConfig",
	"SecurityConfig",
	"get_config",
	"update_config",
	"validate_tenant_config"
]