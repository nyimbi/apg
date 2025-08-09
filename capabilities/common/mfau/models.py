"""
APG Multi-Factor Authentication (MFA) - Data Models

Core Pydantic v2 models with APG-compatible patterns, comprehensive validation,
and multi-tenancy support for revolutionary MFA capability.

Copyright © 2025 Datacraft
Author: Nyimbi Odero <nyimbi@gmail.com>
Website: www.datacraft.co.ke
"""

from typing import Optional, Any, Annotated
from datetime import datetime, timedelta
from enum import Enum
from uuid_extensions import uuid7str
from pydantic import BaseModel, Field, AfterValidator, ConfigDict
import json


def _log_model_validation(cls_name: str, field_name: str, value: Any) -> str:
	"""Log model validation for debugging"""
	return f"[MFA] Validating {cls_name}.{field_name}: {type(value).__name__}"


def validate_mfa_strength(value: str) -> str:
	"""Validate MFA method strength"""
	if not value or len(value) < 3:
		raise ValueError("MFA method must have valid identifier")
	return value


def validate_risk_score(value: float) -> float:
	"""Validate risk score is between 0 and 1"""
	if not 0.0 <= value <= 1.0:
		raise ValueError("Risk score must be between 0.0 and 1.0")
	return value


def validate_confidence_score(value: float) -> float:
	"""Validate confidence score is between 0 and 1"""
	if not 0.0 <= value <= 1.0:
		raise ValueError("Confidence score must be between 0.0 and 1.0")
	return value


def validate_tenant_id(value: str) -> str:
	"""Validate tenant ID format"""
	if not value or len(value) < 5:
		raise ValueError("Tenant ID must be at least 5 characters")
	return value


def validate_user_id(value: str) -> str:
	"""Validate user ID format"""
	if not value or len(value) < 3:
		raise ValueError("User ID must be at least 3 characters")
	return value


class APGBase(BaseModel):
	"""Base model with APG-compatible configuration"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True,
		str_strip_whitespace=True,
		use_enum_values=True
	)
	
	id: str = Field(default_factory=uuid7str, description="Unique identifier")
	tenant_id: Annotated[str, AfterValidator(validate_tenant_id)] = Field(
		description="Multi-tenant identifier"
	)
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	created_by: Annotated[str, AfterValidator(validate_user_id)] = Field(
		description="User who created this record"
	)
	updated_by: Annotated[str, AfterValidator(validate_user_id)] = Field(
		description="User who last updated this record"
	)


class MFAMethodType(str, Enum):
	"""Multi-factor authentication method types"""
	BIOMETRIC_FACE = "biometric_face"
	BIOMETRIC_VOICE = "biometric_voice"
	BIOMETRIC_BEHAVIORAL = "biometric_behavioral"
	BIOMETRIC_MULTI_MODAL = "biometric_multi_modal"
	TOKEN_TOTP = "token_totp"
	TOKEN_HOTP = "token_hotp"
	TOKEN_HARDWARE = "token_hardware"
	SMS = "sms"
	EMAIL = "email"
	PUSH_NOTIFICATION = "push_notification"
	BACKUP_CODES = "backup_codes"
	DELEGATION_TOKEN = "delegation_token"


class TrustLevel(str, Enum):
	"""Device and method trust levels"""
	UNKNOWN = "unknown"
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	VERIFIED = "verified"


class RiskLevel(str, Enum):
	"""Risk assessment levels"""
	MINIMAL = "minimal"
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	CRITICAL = "critical"


class AuthenticationStatus(str, Enum):
	"""Authentication status values"""
	PENDING = "pending"
	IN_PROGRESS = "in_progress"
	SUCCESS = "success"
	FAILED = "failed"
	EXPIRED = "expired"
	REVOKED = "revoked"


class DeviceBinding(APGBase):
	"""Device binding information for MFA methods"""
	device_id: str = Field(description="Unique device identifier")
	device_type: str = Field(description="Type of device (mobile, desktop, etc.)")
	device_name: str = Field(description="Human-readable device name")
	device_fingerprint: str = Field(description="Cryptographic device fingerprint")
	trust_level: TrustLevel = Field(default=TrustLevel.UNKNOWN, description="Device trust level")
	last_seen: datetime = Field(default_factory=datetime.utcnow, description="Last device activity")
	is_active: bool = Field(default=True, description="Whether device binding is active")
	
	# Device security attributes
	is_rooted_jailbroken: Optional[bool] = Field(default=None, description="Device security status")
	os_version: Optional[str] = Field(default=None, description="Operating system version")
	app_version: Optional[str] = Field(default=None, description="Application version")
	location_data: Optional[dict[str, Any]] = Field(default=None, description="Device location data")


class BiometricTemplate(APGBase):
	"""Biometric template storage with privacy protection"""
	biometric_type: str = Field(description="Type of biometric (face, voice, behavioral)")
	template_data: str = Field(description="Encrypted biometric template")
	template_version: str = Field(description="Template format version")
	quality_score: Annotated[float, AfterValidator(validate_confidence_score)] = Field(
		description="Template quality score (0.0-1.0)"
	)
	enrollment_date: datetime = Field(default_factory=datetime.utcnow, description="Enrollment timestamp")
	last_verified: Optional[datetime] = Field(default=None, description="Last successful verification")
	verification_count: int = Field(default=0, description="Number of successful verifications")
	
	# Privacy and security
	is_encrypted: bool = Field(default=True, description="Whether template is encrypted")
	encryption_method: str = Field(default="aes256", description="Encryption method used")
	can_be_reconstructed: bool = Field(default=False, description="Whether template can reconstruct original")


class MFAMethod(APGBase):
	"""Multi-factor authentication method configuration"""
	user_id: Annotated[str, AfterValidator(validate_user_id)] = Field(
		description="User who owns this MFA method"
	)
	method_type: MFAMethodType = Field(description="Type of MFA method")
	method_name: str = Field(description="Human-readable method name")
	is_primary: bool = Field(default=False, description="Whether this is the primary method")
	is_active: bool = Field(default=True, description="Whether method is active")
	
	# Method-specific configuration
	method_config: dict[str, Any] = Field(default_factory=dict, description="Method-specific settings")
	trust_level: TrustLevel = Field(default=TrustLevel.MEDIUM, description="Method trust level")
	
	# Associated data
	device_binding: Optional[DeviceBinding] = Field(default=None, description="Associated device")
	biometric_template: Optional[BiometricTemplate] = Field(default=None, description="Biometric data")
	backup_codes: Optional[list[str]] = Field(default=None, description="Backup recovery codes")
	
	# Usage statistics
	total_uses: int = Field(default=0, description="Total number of uses")
	success_rate: Annotated[float, AfterValidator(validate_confidence_score)] = Field(
		default=0.0, description="Success rate (0.0-1.0)"
	)
	last_used: Optional[datetime] = Field(default=None, description="Last usage timestamp")
	consecutive_failures: int = Field(default=0, description="Consecutive failure count")
	
	# Security settings
	max_failures_before_lockout: int = Field(default=5, description="Max failures before lockout")
	lockout_duration_minutes: int = Field(default=30, description="Lockout duration in minutes")
	requires_device_binding: bool = Field(default=True, description="Whether device binding is required")


class RiskFactor(BaseModel):
	"""Individual risk factor in risk assessment"""
	model_config = ConfigDict(extra='forbid')
	
	factor_type: str = Field(description="Type of risk factor")
	factor_name: str = Field(description="Human-readable factor name") 
	risk_score: Annotated[float, AfterValidator(validate_risk_score)] = Field(
		description="Risk contribution (0.0-1.0)"
	)
	confidence: Annotated[float, AfterValidator(validate_confidence_score)] = Field(
		description="Confidence in this factor (0.0-1.0)"
	)
	evidence: dict[str, Any] = Field(default_factory=dict, description="Supporting evidence")
	mitigation_suggestions: list[str] = Field(default_factory=list, description="Suggested mitigations")


class RiskAssessment(APGBase):
	"""Comprehensive risk assessment for authentication"""
	user_id: Annotated[str, AfterValidator(validate_user_id)] = Field(
		description="User being assessed"
	)
	session_id: str = Field(description="Authentication session identifier")
	
	# Overall risk scoring
	overall_risk_score: Annotated[float, AfterValidator(validate_risk_score)] = Field(
		description="Overall risk score (0.0-1.0)"
	)
	risk_level: RiskLevel = Field(description="Categorical risk level")
	confidence_level: Annotated[float, AfterValidator(validate_confidence_score)] = Field(
		description="Confidence in assessment (0.0-1.0)"
	)
	
	# Risk factors
	risk_factors: list[RiskFactor] = Field(default_factory=list, description="Individual risk factors")
	
	# Contextual information
	device_context: dict[str, Any] = Field(default_factory=dict, description="Device context data")
	location_context: dict[str, Any] = Field(default_factory=dict, description="Location context data")
	behavioral_context: dict[str, Any] = Field(default_factory=dict, description="Behavioral context data")
	temporal_context: dict[str, Any] = Field(default_factory=dict, description="Time-based context data")
	
	# Recommendations
	recommended_auth_methods: list[MFAMethodType] = Field(
		default_factory=list, description="Recommended authentication methods"
	)
	recommended_actions: list[str] = Field(default_factory=list, description="Recommended security actions")
	
	# AI/ML metadata
	model_version: str = Field(description="Risk assessment model version")
	processing_time_ms: int = Field(description="Assessment processing time in milliseconds")


class AuthEvent(APGBase):
	"""Authentication event record for audit and analytics"""
	user_id: Annotated[str, AfterValidator(validate_user_id)] = Field(
		description="User performing authentication"
	)
	session_id: str = Field(description="Authentication session identifier")
	event_type: str = Field(description="Type of authentication event")
	
	# Authentication details
	method_used: Optional[MFAMethodType] = Field(default=None, description="MFA method used")
	method_id: Optional[str] = Field(default=None, description="Specific method instance ID")
	status: AuthenticationStatus = Field(description="Authentication result status")
	
	# Risk and trust
	risk_score: Optional[float] = Field(default=None, description="Risk score at time of auth")
	trust_score: Optional[float] = Field(default=None, description="Trust score achieved")
	
	# Context information
	device_info: dict[str, Any] = Field(default_factory=dict, description="Device information")
	location_info: dict[str, Any] = Field(default_factory=dict, description="Location information")
	network_info: dict[str, Any] = Field(default_factory=dict, description="Network information")
	
	# Timing and performance
	duration_ms: Optional[int] = Field(default=None, description="Authentication duration in milliseconds")
	retry_count: int = Field(default=0, description="Number of retries")
	
	# Error information
	error_code: Optional[str] = Field(default=None, description="Error code if failed")
	error_message: Optional[str] = Field(default=None, description="Error message if failed")
	
	# APG integration
	audit_trail_id: Optional[str] = Field(default=None, description="Audit compliance trail ID")
	notification_sent: bool = Field(default=False, description="Whether notification was sent")


class RecoveryMethod(APGBase):
	"""Account recovery method configuration"""
	user_id: Annotated[str, AfterValidator(validate_user_id)] = Field(
		description="User who owns this recovery method"
	)
	recovery_type: str = Field(description="Type of recovery method")
	recovery_name: str = Field(description="Human-readable recovery method name")
	is_active: bool = Field(default=True, description="Whether recovery method is active")
	
	# Recovery-specific configuration
	recovery_config: dict[str, Any] = Field(default_factory=dict, description="Method-specific settings")
	verification_required: bool = Field(default=True, description="Whether verification is required")
	
	# Security settings
	max_uses_per_day: int = Field(default=3, description="Maximum uses per day")
	uses_today: int = Field(default=0, description="Uses today count")
	last_used: Optional[datetime] = Field(default=None, description="Last usage timestamp")
	
	# Emergency settings
	is_emergency_method: bool = Field(default=False, description="Whether this is an emergency method")
	requires_admin_approval: bool = Field(default=False, description="Whether admin approval is required")


class AuthToken(APGBase):
	"""Authentication token management"""
	user_id: Annotated[str, AfterValidator(validate_user_id)] = Field(
		description="User who owns this token"
	)
	token_type: str = Field(description="Type of authentication token")
	token_value: str = Field(description="Encrypted token value")
	
	# Token lifecycle
	issued_at: datetime = Field(default_factory=datetime.utcnow, description="Token issuance time")
	expires_at: datetime = Field(description="Token expiration time")
	is_active: bool = Field(default=True, description="Whether token is active")
	is_single_use: bool = Field(default=False, description="Whether token is single-use")
	
	# Usage tracking
	used_count: int = Field(default=0, description="Number of times used")
	max_uses: Optional[int] = Field(default=None, description="Maximum allowed uses")
	last_used: Optional[datetime] = Field(default=None, description="Last usage timestamp")
	
	# Security context
	device_binding_required: bool = Field(default=True, description="Whether device binding is required")
	allowed_devices: list[str] = Field(default_factory=list, description="Allowed device IDs")
	ip_restrictions: list[str] = Field(default_factory=list, description="Allowed IP addresses")
	
	# Delegation (for collaborative authentication)
	is_delegation_token: bool = Field(default=False, description="Whether this is a delegation token")
	delegated_by: Optional[str] = Field(default=None, description="User who delegated access")
	delegation_scope: list[str] = Field(default_factory=list, description="Scope of delegated access")


class MFAUserProfile(APGBase):
	"""Comprehensive MFA user profile with AI-powered insights"""
	user_id: Annotated[str, AfterValidator(validate_user_id)] = Field(
		description="Primary user identifier"
	)
	
	# Authentication methods
	authentication_methods: list[MFAMethod] = Field(
		default_factory=list, description="Configured MFA methods"
	)
	primary_method_id: Optional[str] = Field(default=None, description="Primary MFA method ID")
	
	# Risk and trust profile
	base_risk_score: Annotated[float, AfterValidator(validate_risk_score)] = Field(
		default=0.5, description="Base risk score for user (0.0-1.0)"
	)
	trust_score: Annotated[float, AfterValidator(validate_confidence_score)] = Field(
		default=0.0, description="Current trust score (0.0-1.0)"
	)
	behavioral_baseline: dict[str, Any] = Field(
		default_factory=dict, description="Behavioral pattern baseline"
	)
	
	# Device and location trust
	trusted_devices: list[str] = Field(default_factory=list, description="Trusted device IDs")
	trusted_locations: list[dict[str, Any]] = Field(
		default_factory=list, description="Trusted location patterns"
	)
	device_trust_scores: dict[str, float] = Field(
		default_factory=dict, description="Per-device trust scores"
	)
	
	# Authentication history and analytics
	authentication_history: list[str] = Field(
		default_factory=list, description="Recent authentication event IDs"
	)
	total_authentications: int = Field(default=0, description="Total authentication count")
	successful_authentications: int = Field(default=0, description="Successful authentication count")
	failed_authentications: int = Field(default=0, description="Failed authentication count")
	last_successful_auth: Optional[datetime] = Field(
		default=None, description="Last successful authentication"
	)
	
	# Recovery configuration
	recovery_methods: list[RecoveryMethod] = Field(
		default_factory=list, description="Configured recovery methods"
	)
	recovery_questions: Optional[dict[str, str]] = Field(
		default=None, description="Security questions and encrypted answers"
	)
	
	# Security settings
	lockout_until: Optional[datetime] = Field(default=None, description="Account lockout expiration")
	security_notifications_enabled: bool = Field(
		default=True, description="Whether security notifications are enabled"
	)
	adaptive_auth_enabled: bool = Field(
		default=True, description="Whether adaptive authentication is enabled"
	)
	
	# Compliance and privacy
	consent_given: bool = Field(default=False, description="Whether user has given consent")
	data_retention_days: int = Field(default=365, description="Data retention period in days")
	privacy_settings: dict[str, Any] = Field(
		default_factory=dict, description="User privacy preferences"
	)
	
	# AI/ML insights
	ml_insights: dict[str, Any] = Field(default_factory=dict, description="ML-generated user insights")
	risk_patterns: list[str] = Field(default_factory=list, description="Identified risk patterns")
	recommendation_engine_data: dict[str, Any] = Field(
		default_factory=dict, description="Data for recommendation engine"
	)


# Model validation functions
def validate_mfa_user_profile(profile: MFAUserProfile) -> None:
	"""Validate MFA user profile consistency"""
	# Ensure primary method exists in methods list
	if profile.primary_method_id:
		method_ids = [method.id for method in profile.authentication_methods]
		if profile.primary_method_id not in method_ids:
			raise ValueError("Primary method ID must reference an existing authentication method")
	
	# Validate trust scores are consistent
	if profile.trust_score > 1.0 or profile.trust_score < 0.0:
		raise ValueError("Trust score must be between 0.0 and 1.0")


def validate_auth_event_consistency(event: AuthEvent) -> None:
	"""Validate authentication event consistency"""
	# Ensure method_id is provided when method_used is specified
	if event.method_used and not event.method_id:
		raise ValueError("Method ID must be provided when method_used is specified")
	
	# Validate timing consistency
	if event.duration_ms and event.duration_ms < 0:
		raise ValueError("Duration cannot be negative")


# Export all models
__all__ = [
	"APGBase",
	"MFAMethodType",
	"TrustLevel", 
	"RiskLevel",
	"AuthenticationStatus",
	"DeviceBinding",
	"BiometricTemplate",
	"MFAMethod",
	"RiskFactor",
	"RiskAssessment",
	"AuthEvent", 
	"RecoveryMethod",
	"AuthToken",
	"MFAUserProfile",
	"validate_mfa_user_profile",
	"validate_auth_event_consistency"
]