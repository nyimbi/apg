"""
Enhanced Authentication Models and Data Structures

Revolutionary authentication models supporting behavioral biometrics, 
quantum-resistant cryptography, privacy-preserving analytics, and 
identity graph intelligence.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from typing import Dict, Any, Optional, List, Set, Union, Tuple, Literal
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json
from pathlib import Path
from uuid_extensions import uuid7str
from pydantic import BaseModel, Field, ConfigDict, EmailStr, AfterValidator
from annotated_types import Annotated
import hashlib
import secrets

from.behavioral_auth import BehavioralBaseline, BehavioralEventType
from .contextual_risk import AuthContext, RiskAssessment

# Enhanced Enums for Revolutionary Features
class BiometricType(str, Enum):
	"""Types of biometric authentication supported"""
	FACE = "face"
	FINGERPRINT = "fingerprint"
	VOICE = "voice"
	IRIS = "iris"
	PALM = "palm"
	BEHAVIORAL = "behavioral"
	KEYSTROKE = "keystroke"
	GAIT = "gait"
	HEARTRATE = "heartrate"

class PrivacyLevel(str, Enum):
	"""Privacy levels for user data"""
	PUBLIC = "public"
	INTERNAL = "internal"
	CONFIDENTIAL = "confidential"
	RESTRICTED = "restricted"
	TOP_SECRET = "top_secret"

class ConsentType(str, Enum):
	"""Types of user consent"""
	DATA_PROCESSING = "data_processing"
	BEHAVIORAL_TRACKING = "behavioral_tracking"
	BIOMETRIC_STORAGE = "biometric_storage"
	LOCATION_TRACKING = "location_tracking"
	ANALYTICS = "analytics"
	MARKETING = "marketing"
	RESEARCH = "research"

class CryptographicAlgorithm(str, Enum):
	"""Supported cryptographic algorithms"""
	# Classical algorithms
	RSA = "rsa"
	ECDSA = "ecdsa"
	AES = "aes"
	
	# Post-quantum algorithms
	CRYSTALS_KYBER = "crystals_kyber"
	CRYSTALS_DILITHIUM = "crystals_dilithium"
	FALCON = "falcon"
	SPHINCS_PLUS = "sphincs_plus"
	
	# Hybrid algorithms
	RSA_KYBER = "rsa_kyber"
	ECDSA_DILITHIUM = "ecdsa_dilithium"

class IdentityRelationType(str, Enum):
	"""Types of identity relationships in graph"""
	SAME_DEVICE = "same_device"
	SAME_LOCATION = "same_location"
	SAME_NETWORK = "same_network"
	SIMILAR_BEHAVIOR = "similar_behavior"
	TEMPORAL_CORRELATION = "temporal_correlation"
	FAMILY_MEMBER = "family_member"
	COLLEAGUE = "colleague"
	SUSPICIOUS = "suspicious"

# Validation Functions
def validate_biometric_hash(v: str) -> str:
	"""Validate biometric hash format"""
	if not v or len(v) < 32:
		raise ValueError("Biometric hash must be at least 32 characters")
	return v

def validate_encryption_key(v: bytes) -> bytes:
	"""Validate encryption key length"""
	if not v or len(v) < 16:
		raise ValueError("Encryption key must be at least 16 bytes")
	return v

def validate_confidence_score(v: float) -> float:
	"""Validate confidence score range"""
	if not 0.0 <= v <= 1.0:
		raise ValueError("Confidence score must be between 0.0 and 1.0")
	return v

# Enhanced Biometric Models
class BiometricTemplate(BaseModel):
	"""Enhanced biometric template with privacy protection"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Template identifier")
	user_id: str = Field(..., description="User identifier")
	biometric_type: BiometricType = Field(..., description="Type of biometric")
	
	# Privacy-protected template data
	template_hash: Annotated[str, AfterValidator(validate_biometric_hash)] = Field(
		..., description="Privacy-protected biometric hash"
	)
	encryption_key_id: str = Field(..., description="Key used for template encryption")
	algorithm_version: str = Field(..., description="Algorithm version used")
	
	# Template metadata
	quality_score: Annotated[float, AfterValidator(validate_confidence_score)] = Field(
		..., description="Template quality score"
	)
	confidence_level: Annotated[float, AfterValidator(validate_confidence_score)] = Field(
		..., description="Confidence in template accuracy"
	)
	
	# Enrollment data
	enrolled_at: datetime = Field(default_factory=datetime.utcnow, description="Enrollment timestamp")
	enrolled_by: Optional[str] = Field(default=None, description="Enrolling user/system")
	enrollment_device: Optional[str] = Field(default=None, description="Enrollment device")
	
	# Usage tracking
	last_used_at: Optional[datetime] = Field(default=None, description="Last authentication use")
	use_count: int = Field(default=0, description="Number of times used")
	success_count: int = Field(default=0, description="Successful authentications")
	
	# Validity
	expires_at: Optional[datetime] = Field(default=None, description="Template expiration")
	is_active: bool = Field(default=True, description="Template active status")
	
	# Privacy and compliance
	consent_given: bool = Field(default=False, description="User consent for biometric storage")
	retention_policy_id: Optional[str] = Field(default=None, description="Data retention policy")
	can_be_shared: bool = Field(default=False, description="Can be shared for analytics")
	
	def is_valid(self) -> bool:
		"""Check if biometric template is valid for use"""
		if not self.is_active or not self.consent_given:
			return False
		if self.expires_at and self.expires_at <= datetime.utcnow():
			return False
		return True
	
	def update_usage(self, success: bool = True):
		"""Update usage statistics"""
		self.use_count += 1
		if success:
			self.success_count += 1
		self.last_used_at = datetime.utcnow()
	
	def get_success_rate(self) -> float:
		"""Get authentication success rate"""
		if self.use_count == 0:
			return 0.0
		return self.success_count / self.use_count

class QuantumKey(BaseModel):
	"""Quantum-resistant cryptographic key"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Key identifier")
	user_id: str = Field(..., description="Owner user identifier")
	algorithm: CryptographicAlgorithm = Field(..., description="Cryptographic algorithm")
	
	# Key data (encrypted at rest)
	public_key: bytes = Field(..., description="Public key data")
	private_key_encrypted: bytes = Field(..., description="Encrypted private key")
	key_derivation_salt: bytes = Field(..., description="Key derivation salt")
	
	# Key metadata
	key_size: int = Field(..., description="Key size in bits")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Key creation timestamp")
	expires_at: Optional[datetime] = Field(default=None, description="Key expiration")
	
	# Security parameters
	security_level: int = Field(default=128, description="Security level in bits")
	is_hybrid: bool = Field(default=False, description="Hybrid classical/quantum-safe key")
	classical_backup: Optional[str] = Field(default=None, description="Classical algorithm backup")
	
	# Usage tracking
	last_used_at: Optional[datetime] = Field(default=None, description="Last key usage")
	use_count: int = Field(default=0, description="Number of times used")
	
	# Key management
	is_revoked: bool = Field(default=False, description="Key revocation status")
	revoked_at: Optional[datetime] = Field(default=None, description="Revocation timestamp")
	revoked_reason: Optional[str] = Field(default=None, description="Revocation reason")
	
	def is_valid(self) -> bool:
		"""Check if key is valid for use"""
		if self.is_revoked:
			return False
		if self.expires_at and self.expires_at <= datetime.utcnow():
			return False
		return True
	
	def revoke(self, reason: str = "Manual revocation"):
		"""Revoke the key"""
		self.is_revoked = True
		self.revoked_at = datetime.utcnow()
		self.revoked_reason = reason

class PrivacyPreferences(BaseModel):
	"""User privacy preferences and consent management"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Preferences identifier")
	user_id: str = Field(..., description="User identifier")
	
	# Consent settings
	consent_records: Dict[ConsentType, bool] = Field(
		default_factory=dict, description="Consent status by type"
	)
	consent_timestamps: Dict[ConsentType, datetime] = Field(
		default_factory=dict, description="When consent was given/withdrawn"
	)
	
	# Data processing preferences
	allow_behavioral_tracking: bool = Field(default=False, description="Allow behavioral analysis")
	allow_biometric_storage: bool = Field(default=False, description="Allow biometric template storage")
	allow_location_tracking: bool = Field(default=False, description="Allow location-based risk analysis")
	allow_analytics: bool = Field(default=False, description="Allow usage analytics")
	
	# Privacy levels
	data_privacy_level: PrivacyLevel = Field(default=PrivacyLevel.CONFIDENTIAL, description="Data privacy level")
	sharing_allowed: bool = Field(default=False, description="Allow data sharing for research")
	retention_period_days: int = Field(default=365, description="Data retention period in days")
	
	# Right to be forgotten
	deletion_requested: bool = Field(default=False, description="User requested data deletion")
	deletion_request_date: Optional[datetime] = Field(default=None, description="When deletion was requested")
	deletion_completed_date: Optional[datetime] = Field(default=None, description="When deletion was completed")
	
	# Notification preferences
	notify_on_unusual_activity: bool = Field(default=True, description="Notify on unusual login activity")
	notify_on_policy_changes: bool = Field(default=True, description="Notify on privacy policy changes")
	notification_channels: List[str] = Field(default_factory=lambda: ["email"], description="Notification channels")
	
	# Compliance
	gdpr_compliant: bool = Field(default=True, description="GDPR compliance status")
	ccpa_compliant: bool = Field(default=True, description="CCPA compliance status")
	last_review_date: Optional[datetime] = Field(default=None, description="Last privacy review date")
	
	def give_consent(self, consent_type: ConsentType, granted: bool = True):
		"""Record user consent"""
		self.consent_records[consent_type] = granted
		self.consent_timestamps[consent_type] = datetime.utcnow()
		
		# Update related preferences
		if consent_type == ConsentType.BEHAVIORAL_TRACKING:
			self.allow_behavioral_tracking = granted
		elif consent_type == ConsentType.BIOMETRIC_STORAGE:
			self.allow_biometric_storage = granted
		elif consent_type == ConsentType.LOCATION_TRACKING:
			self.allow_location_tracking = granted
		elif consent_type == ConsentType.ANALYTICS:
			self.allow_analytics = granted
	
	def has_consent(self, consent_type: ConsentType) -> bool:
		"""Check if user has given consent for specific type"""
		return self.consent_records.get(consent_type, False)
	
	def request_deletion(self):
		"""Request data deletion (right to be forgotten)"""
		self.deletion_requested = True
		self.deletion_request_date = datetime.utcnow()

class ConsentRecord(BaseModel):
	"""Individual consent record with audit trail"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Record identifier")
	user_id: str = Field(..., description="User identifier")
	consent_type: ConsentType = Field(..., description="Type of consent")
	
	# Consent details
	granted: bool = Field(..., description="Whether consent was granted")
	granted_at: datetime = Field(default_factory=datetime.utcnow, description="When consent was given/withdrawn")
	granted_by: str = Field(..., description="Who granted consent (user/guardian/admin)")
	
	# Legal basis
	legal_basis: str = Field(..., description="Legal basis for processing (GDPR Article 6)")
	purpose: str = Field(..., description="Purpose of data processing")
	data_categories: List[str] = Field(default_factory=list, description="Categories of data involved")
	
	# Consent mechanism
	consent_method: str = Field(..., description="How consent was obtained (web form, API, etc.)")
	ip_address: Optional[str] = Field(default=None, description="IP address when consent given")
	user_agent: Optional[str] = Field(default=None, description="User agent string")
	
	# Consent proof
	consent_proof: Optional[str] = Field(default=None, description="Digital proof of consent")
	consent_hash: Optional[str] = Field(default=None, description="Hash of consent data for integrity")
	
	# Validity
	expires_at: Optional[datetime] = Field(default=None, description="Consent expiration")
	withdrawn_at: Optional[datetime] = Field(default=None, description="When consent was withdrawn")
	withdrawn_reason: Optional[str] = Field(default=None, description="Reason for withdrawal")
	
	def is_valid(self) -> bool:
		"""Check if consent is still valid"""
		if not self.granted:
			return False
		if self.withdrawn_at:
			return False
		if self.expires_at and self.expires_at <= datetime.utcnow():
			return False
		return True
	
	def withdraw(self, reason: str = "User requested"):
		"""Withdraw consent"""
		self.granted = False
		self.withdrawn_at = datetime.utcnow()
		self.withdrawn_reason = reason

class IdentityGraphNode(BaseModel):
	"""Node in the identity graph representing an entity"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Node identifier")
	entity_type: str = Field(..., description="Type of entity (user, device, location, etc.)")
	entity_id: str = Field(..., description="Entity identifier")
	
	# Node attributes
	attributes: Dict[str, Any] = Field(default_factory=dict, description="Entity attributes")
	labels: Set[str] = Field(default_factory=set, description="Node labels/tags")
	
	# Graph metadata
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Node creation time")
	last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update time")
	
	# Risk scoring
	risk_score: float = Field(default=0.0, description="Entity risk score", ge=0.0, le=1.0)
	reputation_score: float = Field(default=0.5, description="Entity reputation", ge=0.0, le=1.0)
	
	# Activity tracking
	activity_count: int = Field(default=0, description="Number of activities involving this entity")
	last_activity_at: Optional[datetime] = Field(default=None, description="Last activity timestamp")

class IdentityGraphEdge(BaseModel):
	"""Edge in the identity graph representing a relationship"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Edge identifier")
	source_node_id: str = Field(..., description="Source node ID")
	target_node_id: str = Field(..., description="Target node ID")
	relationship_type: IdentityRelationType = Field(..., description="Type of relationship")
	
	# Relationship strength and confidence
	strength: float = Field(default=1.0, description="Relationship strength", ge=0.0, le=1.0)
	confidence: Annotated[float, AfterValidator(validate_confidence_score)] = Field(
		default=1.0, description="Confidence in relationship"
	)
	
	# Relationship metadata
	first_observed: datetime = Field(default_factory=datetime.utcnow, description="First observation time")
	last_observed: datetime = Field(default_factory=datetime.utcnow, description="Last observation time")
	observation_count: int = Field(default=1, description="Number of times observed")
	
	# Evidence
	evidence: List[Dict[str, Any]] = Field(default_factory=list, description="Evidence for relationship")
	supporting_data: Dict[str, Any] = Field(default_factory=dict, description="Supporting data")
	
	# Temporal aspects
	is_temporal: bool = Field(default=False, description="Is this a temporal relationship")
	duration_seconds: Optional[int] = Field(default=None, description="Relationship duration")
	
	def update_observation(self):
		"""Update observation tracking"""
		self.last_observed = datetime.utcnow()
		self.observation_count += 1
	
	def add_evidence(self, evidence_data: Dict[str, Any]):
		"""Add evidence for the relationship"""
		self.evidence.append({
			**evidence_data,
			"timestamp": datetime.utcnow().isoformat()
		})

class EnhancedUser(BaseModel):
	"""Enhanced User model with revolutionary authentication features"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	# Core identity (inherited from base User model)
	id: str = Field(default_factory=uuid7str, description="Unique user identifier")
	email: EmailStr = Field(..., description="User email address")
	username: Optional[str] = Field(default=None, description="Username (optional)")
	password_hash: str = Field(..., description="Hashed password")
	salt: str = Field(default_factory=lambda: secrets.token_hex(32), description="Password salt")
	
	# Profile information
	first_name: Optional[str] = Field(default=None, description="First name")
	last_name: Optional[str] = Field(default=None, description="Last name")
	display_name: Optional[str] = Field(default=None, description="Display name")
	avatar_url: Optional[str] = Field(default=None, description="Avatar image URL")
	
	# Revolutionary authentication data
	behavioral_baseline: Optional[BehavioralBaseline] = Field(
		default=None, description="Behavioral authentication baseline"
	)
	biometric_templates: List[BiometricTemplate] = Field(
		default_factory=list, description="Biometric templates"
	)
	quantum_keys: List[QuantumKey] = Field(
		default_factory=list, description="Quantum-resistant keys"
	)
	
	# Identity graph
	graph_node_id: Optional[str] = Field(default=None, description="Identity graph node ID")
	identity_connections: Set[str] = Field(
		default_factory=set, description="Connected identity node IDs"
	)
	identity_risk_score: float = Field(default=0.0, description="Identity graph risk score")
	
	# Privacy and consent
	privacy_preferences: Optional[PrivacyPreferences] = Field(
		default=None, description="Privacy preferences and consent"
	)
	consent_records: List[ConsentRecord] = Field(
		default_factory=list, description="Detailed consent audit trail"
	)
	
	# Advanced security features
	zero_knowledge_enabled: bool = Field(default=False, description="Zero-knowledge auth enabled")
	neuromorphic_enabled: bool = Field(default=False, description="Neuromorphic processing enabled")
	adaptive_policy_enabled: bool = Field(default=True, description="Adaptive policy learning enabled")
	
	# Security settings (enhanced)
	mfa_enabled: bool = Field(default=False, description="Multi-factor authentication enabled")
	mfa_methods: Set[str] = Field(default_factory=set, description="Enabled MFA methods")
	biometric_auth_enabled: bool = Field(default=False, description="Biometric authentication enabled")
	behavioral_auth_enabled: bool = Field(default=False, description="Behavioral authentication enabled")
	
	# Risk assessment history
	risk_assessments: List[str] = Field(
		default_factory=list, description="Recent risk assessment IDs"
	)
	last_risk_assessment: Optional[RiskAssessment] = Field(
		default=None, description="Most recent risk assessment"
	)
	
	# Account security
	security_score: float = Field(default=0.5, description="Overall security score", ge=0.0, le=1.0)
	trust_score: float = Field(default=0.5, description="Trust score based on behavior", ge=0.0, le=1.0)
	anomaly_count: int = Field(default=0, description="Number of detected anomalies")
	last_anomaly_at: Optional[datetime] = Field(default=None, description="Last anomaly detection")
	
	# Enhanced metadata
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	last_login_at: Optional[datetime] = Field(default=None, description="Last successful login")
	last_failed_login_at: Optional[datetime] = Field(default=None, description="Last failed login")
	
	# Multi-tenancy
	tenant_id: Optional[str] = Field(default=None, description="Primary tenant ID")
	tenant_memberships: Set[str] = Field(default_factory=set, description="All tenant memberships")
	
	# Status and lifecycle
	status: str = Field(default="active", description="Account status")
	email_verified_at: Optional[datetime] = Field(default=None, description="Email verification timestamp")
	phone_verified_at: Optional[datetime] = Field(default=None, description="Phone verification timestamp")
	
	# Account protection
	failed_login_attempts: int = Field(default=0, description="Failed login attempt count")
	locked_until: Optional[datetime] = Field(default=None, description="Account lock expiration")
	
	# Preferences
	timezone: str = Field(default="UTC", description="User timezone")
	language: str = Field(default="en", description="Preferred language")
	preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")
	
	def __init__(self, **data):
		super().__init__(**data)
		# Initialize privacy preferences if not provided
		if not self.privacy_preferences:
			self.privacy_preferences = PrivacyPreferences(user_id=self.id)
	
	def update_timestamp(self):
		"""Update the modified timestamp"""
		self.updated_at = datetime.utcnow()
	
	def is_active(self) -> bool:
		"""Check if user is active"""
		return self.status == "active"
	
	def is_locked(self) -> bool:
		"""Check if user account is locked"""
		if self.status == "locked":
			return True
		if self.locked_until and self.locked_until > datetime.utcnow():
			return True
		return False
	
	def has_biometric_enrolled(self, biometric_type: Optional[BiometricType] = None) -> bool:
		"""Check if user has biometric template enrolled"""
		if not self.biometric_templates:
			return False
		
		if biometric_type:
			return any(
				template.biometric_type == biometric_type and template.is_valid()
				for template in self.biometric_templates
			)
		
		return any(template.is_valid() for template in self.biometric_templates)
	
	def get_active_biometric_types(self) -> Set[BiometricType]:
		"""Get set of active biometric types"""
		return {
			template.biometric_type 
			for template in self.biometric_templates 
			if template.is_valid()
		}
	
	def has_quantum_key(self, algorithm: Optional[CryptographicAlgorithm] = None) -> bool:
		"""Check if user has quantum-resistant key"""
		if not self.quantum_keys:
			return False
		
		if algorithm:
			return any(
				key.algorithm == algorithm and key.is_valid()
				for key in self.quantum_keys
			)
		
		return any(key.is_valid() for key in self.quantum_keys)
	
	def get_security_capabilities(self) -> Dict[str, bool]:
		"""Get user's security capabilities"""
		return {
			"behavioral_auth": self.behavioral_auth_enabled and self.behavioral_baseline is not None,
			"biometric_auth": self.biometric_auth_enabled and self.has_biometric_enrolled(),
			"quantum_resistant": self.has_quantum_key(),
			"zero_knowledge": self.zero_knowledge_enabled,
			"neuromorphic": self.neuromorphic_enabled,
			"adaptive_policy": self.adaptive_policy_enabled,
			"mfa": self.mfa_enabled
		}
	
	def calculate_security_score(self) -> float:
		"""Calculate overall security score based on enabled features"""
		capabilities = self.get_security_capabilities()
		
		# Weight different security features
		weights = {
			"behavioral_auth": 0.2,
			"biometric_auth": 0.25,
			"quantum_resistant": 0.15,
			"zero_knowledge": 0.1,
			"neuromorphic": 0.1,
			"adaptive_policy": 0.1,
			"mfa": 0.1
		}
		
		score = sum(
			weights[capability] * (1.0 if enabled else 0.0)
			for capability, enabled in capabilities.items()
		)
		
		# Add baseline security
		baseline_score = 0.3 if not self.is_locked() else 0.0
		
		self.security_score = min(1.0, baseline_score + score)
		return self.security_score
	
	def add_biometric_template(self, template: BiometricTemplate):
		"""Add biometric template to user"""
		assert template.user_id == self.id, "Template user ID must match user ID"
		self.biometric_templates.append(template)
		self.update_timestamp()
	
	def add_quantum_key(self, key: QuantumKey):
		"""Add quantum-resistant key to user"""
		assert key.user_id == self.id, "Key user ID must match user ID"
		self.quantum_keys.append(key)
		self.update_timestamp()
	
	def record_anomaly(self):
		"""Record security anomaly detection"""
		self.anomaly_count += 1
		self.last_anomaly_at = datetime.utcnow()
		self.update_timestamp()
	
	def update_trust_score(self, adjustment: float):
		"""Update trust score based on behavior"""
		self.trust_score = max(0.0, min(1.0, self.trust_score + adjustment))
		self.update_timestamp()
	
	def has_consent(self, consent_type: ConsentType) -> bool:
		"""Check if user has given specific consent"""
		if not self.privacy_preferences:
			return False
		return self.privacy_preferences.has_consent(consent_type)
	
	def give_consent(self, consent_type: ConsentType, granted: bool = True):
		"""Give or withdraw consent"""
		if not self.privacy_preferences:
			self.privacy_preferences = PrivacyPreferences(user_id=self.id)
		
		self.privacy_preferences.give_consent(consent_type, granted)
		
		# Create audit record
		consent_record = ConsentRecord(
			user_id=self.id,
			consent_type=consent_type,
			granted=granted,
			granted_by=self.id,
			legal_basis="Consent (GDPR Art. 6(1)(a))",
			purpose=f"Enable {consent_type.value} functionality",
			consent_method="user_interface"
		)
		self.consent_records.append(consent_record)
		self.update_timestamp()
	
	def request_data_deletion(self):
		"""Request data deletion (GDPR right to be forgotten)"""
		if not self.privacy_preferences:
			self.privacy_preferences = PrivacyPreferences(user_id=self.id)
		
		self.privacy_preferences.request_deletion()
		self.update_timestamp()
	
	def get_data_retention_date(self) -> datetime:
		"""Get date when user data should be deleted"""
		if not self.privacy_preferences:
			# Default retention: 2 years from last activity
			base_date = self.last_login_at or self.created_at
			return base_date + timedelta(days=730)
		
		if self.privacy_preferences.deletion_requested:
			# Grace period for deletion request
			return self.privacy_preferences.deletion_request_date + timedelta(days=30)
		
		# Use user's preferred retention period
		base_date = self.last_login_at or self.created_at
		return base_date + timedelta(days=self.privacy_preferences.retention_period_days)
	
	def should_be_deleted(self) -> bool:
		"""Check if user data should be deleted based on retention policy"""
		return datetime.utcnow() >= self.get_data_retention_date()

class EnhancedSession(BaseModel):
	"""Enhanced session model with behavioral monitoring and adaptive security"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	# Core session data
	id: str = Field(default_factory=uuid7str, description="Session identifier")
	user_id: str = Field(..., description="User identifier")
	tenant_id: Optional[str] = Field(default=None, description="Session tenant context")
	
	# Session lifecycle
	status: str = Field(default="active", description="Session status")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Session creation")
	expires_at: datetime = Field(..., description="Session expiration")
	last_activity_at: datetime = Field(default_factory=datetime.utcnow, description="Last activity")
	
	# Authentication context
	initial_auth_methods: List[str] = Field(default_factory=list, description="Initial auth methods used")
	step_up_auth_required: bool = Field(default=False, description="Step-up auth required")
	step_up_completed: bool = Field(default=False, description="Step-up auth completed")
	
	# Behavioral monitoring
	behavioral_confidence: float = Field(default=1.0, description="Current behavioral confidence")
	behavioral_anomalies: List[Dict[str, Any]] = Field(
		default_factory=list, description="Detected behavioral anomalies"
	)
	continuous_monitoring_enabled: bool = Field(default=True, description="Continuous behavior monitoring")
	
	# Risk assessment
	initial_risk_assessment: Optional[RiskAssessment] = Field(
		default=None, description="Risk assessment at session creation"
	)
	current_risk_level: str = Field(default="low", description="Current risk level")
	risk_score_history: List[Dict[str, Any]] = Field(
		default_factory=list, description="Risk score over time"
	)
	
	# Adaptive security
	adaptive_timeout_enabled: bool = Field(default=True, description="Adaptive session timeout")
	base_timeout_minutes: int = Field(default=60, description="Base session timeout")
	current_timeout_minutes: int = Field(default=60, description="Current adaptive timeout")
	
	# Session forensics
	access_patterns: List[Dict[str, Any]] = Field(
		default_factory=list, description="Access patterns during session"
	)
	security_events: List[Dict[str, Any]] = Field(
		default_factory=list, description="Security events during session"
	)
	
	# Client information
	ip_address: Optional[str] = Field(default=None, description="Client IP address")
	user_agent: Optional[str] = Field(default=None, description="Client user agent")
	device_fingerprint: Optional[str] = Field(default=None, description="Device fingerprint")
	
	# Tokens
	access_token: str = Field(..., description="JWT access token")
	refresh_token: str = Field(default_factory=lambda: secrets.token_urlsafe(32), description="Refresh token")
	
	def is_active(self) -> bool:
		"""Check if session is active"""
		return (self.status == "active" and 
				self.expires_at > datetime.utcnow())
	
	def is_expired(self) -> bool:
		"""Check if session is expired"""
		return self.expires_at <= datetime.utcnow()
	
	def refresh_activity(self):
		"""Update last activity timestamp and adjust timeout if adaptive"""
		self.last_activity_at = datetime.utcnow()
		
		if self.adaptive_timeout_enabled:
			self._adjust_adaptive_timeout()
	
	def _adjust_adaptive_timeout(self):
		"""Adjust session timeout based on current risk and behavior"""
		risk_multiplier = {
			"very_low": 1.5,
			"low": 1.2,
			"moderate": 1.0,
			"high": 0.7,
			"very_high": 0.5,
			"critical": 0.3
		}.get(self.current_risk_level, 1.0)
		
		# Behavioral confidence affects timeout
		behavior_multiplier = self.behavioral_confidence
		
		# Calculate new timeout
		new_timeout = int(self.base_timeout_minutes * risk_multiplier * behavior_multiplier)
		self.current_timeout_minutes = max(10, min(240, new_timeout))  # Clamp between 10 and 240 minutes
		
		# Update expiration
		self.expires_at = self.last_activity_at + timedelta(minutes=self.current_timeout_minutes)
	
	def add_security_event(self, event_type: str, details: Dict[str, Any]):
		"""Add security event to session"""
		event = {
			"type": event_type,
			"timestamp": datetime.utcnow().isoformat(),
			"details": details
		}
		self.security_events.append(event)
		
		# Keep only last 50 events
		self.security_events = self.security_events[-50:]
	
	def record_behavioral_anomaly(self, anomaly_data: Dict[str, Any]):
		"""Record behavioral anomaly during session"""
		anomaly = {
			"timestamp": datetime.utcnow().isoformat(),
			"confidence_before": self.behavioral_confidence,
			**anomaly_data
		}
		self.behavioral_anomalies.append(anomaly)
		
		# Update behavioral confidence
		severity = anomaly_data.get("severity", 0.5)
		confidence_reduction = severity * 0.3  # Reduce confidence based on severity
		self.behavioral_confidence = max(0.1, self.behavioral_confidence - confidence_reduction)
		
		# Check if step-up auth is needed
		if self.behavioral_confidence < 0.5 and not self.step_up_auth_required:
			self.step_up_auth_required = True
			self.add_security_event("step_up_auth_triggered", {
				"reason": "behavioral_anomaly",
				"confidence": self.behavioral_confidence
			})
	
	def complete_step_up_auth(self, auth_method: str):
		"""Complete step-up authentication"""
		self.step_up_completed = True
		self.step_up_auth_required = False
		self.behavioral_confidence = min(1.0, self.behavioral_confidence + 0.3)  # Restore some confidence
		
		self.add_security_event("step_up_auth_completed", {
			"method": auth_method,
			"new_confidence": self.behavioral_confidence
		})
	
	def update_risk_level(self, new_risk_level: str, risk_score: float):
		"""Update session risk level"""
		old_risk_level = self.current_risk_level
		self.current_risk_level = new_risk_level
		
		# Record risk score history
		risk_entry = {
			"timestamp": datetime.utcnow().isoformat(),
			"risk_level": new_risk_level,
			"risk_score": risk_score,
			"previous_level": old_risk_level
		}
		self.risk_score_history.append(risk_entry)
		
		# Keep only last 20 risk assessments
		self.risk_score_history = self.risk_score_history[-20:]
		
		# Adjust timeout based on new risk
		if self.adaptive_timeout_enabled:
			self._adjust_adaptive_timeout()
	
	def terminate(self, reason: str = "User logout"):
		"""Terminate session"""
		self.status = "terminated"
		self.expires_at = datetime.utcnow()
		self.add_security_event("session_terminated", {"reason": reason})
	
	def get_session_summary(self) -> Dict[str, Any]:
		"""Get comprehensive session summary"""
		duration = (datetime.utcnow() - self.created_at).total_seconds()
		
		return {
			"session_id": self.id,
			"duration_seconds": duration,
			"status": self.status,
			"risk_level": self.current_risk_level,
			"behavioral_confidence": self.behavioral_confidence,
			"anomaly_count": len(self.behavioral_anomalies),
			"security_event_count": len(self.security_events),
			"step_up_auth_required": self.step_up_auth_required,
			"step_up_auth_completed": self.step_up_completed,
			"adaptive_timeout_minutes": self.current_timeout_minutes
		}