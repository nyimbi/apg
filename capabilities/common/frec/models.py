"""
APG Facial Recognition - Data Models

Revolutionary facial recognition data models with contextual intelligence, emotion analysis,
collaborative verification, and privacy-first architecture.

Author: Datacraft (nyimbi@gmail.com)
Copyright: © 2025 Datacraft
"""

import asyncio
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Any, Optional
from uuid_extensions import uuid7str

from pydantic import BaseModel, Field, ConfigDict, AfterValidator
from sqlalchemy import Column, String, DateTime, Float, Integer, Boolean, Text, LargeBinary, JSON, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

# Base SQLAlchemy model
Base = declarative_base()

# Enums for facial recognition
class FaVerificationType(str, Enum):
	"""Facial verification types"""
	AUTHENTICATION = "authentication"
	IDENTIFICATION = "identification"
	ENROLLMENT = "enrollment"
	LIVENESS_CHECK = "liveness_check"
	EMOTION_ANALYSIS = "emotion_analysis"

class FaEmotionType(str, Enum):
	"""Emotion classification types"""
	HAPPY = "happy"
	SAD = "sad"
	ANGRY = "angry"
	FEARFUL = "fearful"
	SURPRISED = "surprised"
	DISGUSTED = "disgusted"
	NEUTRAL = "neutral"
	STRESSED = "stressed"
	CONFUSED = "confused"
	EXCITED = "excited"

class FaQualityLevel(str, Enum):
	"""Face image quality levels"""
	EXCELLENT = "excellent"
	GOOD = "good"
	ACCEPTABLE = "acceptable"
	POOR = "poor"
	REJECTED = "rejected"

class FaLivenessResult(str, Enum):
	"""Liveness detection results"""
	LIVE = "live"
	SPOOF = "spoof"
	UNCERTAIN = "uncertain"
	NOT_TESTED = "not_tested"

class FaProcessingStatus(str, Enum):
	"""Processing status for facial recognition tasks"""
	PENDING = "pending"
	PROCESSING = "processing"
	COMPLETED = "completed"
	FAILED = "failed"
	CANCELLED = "cancelled"

# SQLAlchemy Models

class FaUser(Base):
	"""User facial profile management with multi-tenant support"""
	__tablename__ = 'fa_users'
	__table_args__ = (
		Index('idx_fa_users_tenant_external', 'tenant_id', 'external_user_id'),
		Index('idx_fa_users_created', 'created_at'),
		Index('idx_fa_users_status', 'status'),
	)

	id = Column(String, primary_key=True, default=uuid7str)
	tenant_id = Column(String, nullable=False)
	external_user_id = Column(String, nullable=False)  # Link to APG auth_rbac user
	
	# User profile information
	full_name = Column(String(255))
	email = Column(String(255))
	department = Column(String(100))
	role = Column(String(100))
	
	# Facial recognition settings
	enrollment_status = Column(String(50), default='not_enrolled')
	consent_given = Column(Boolean, default=False)
	consent_date = Column(DateTime(timezone=True))
	
	# Privacy and compliance
	data_retention_days = Column(Integer, default=365)
	anonymize_analytics = Column(Boolean, default=True)
	cross_border_consent = Column(Boolean, default=False)
	
	# Audit fields
	created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
	updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
	created_by = Column(String)
	status = Column(String(50), default='active')
	
	# Metadata for business context
	metadata = Column(JSON, default=dict)

	# Relationships
	templates = relationship("FaTemplate", back_populates="user", cascade="all, delete-orphan")
	verifications = relationship("FaVerification", back_populates="user", cascade="all, delete-orphan")
	emotions = relationship("FaEmotion", back_populates="user", cascade="all, delete-orphan")

class FaTemplate(Base):
	"""Encrypted facial template storage with versioning"""
	__tablename__ = 'fa_templates'
	__table_args__ = (
		Index('idx_fa_templates_user', 'user_id'),
		Index('idx_fa_templates_quality', 'quality_score'),
		Index('idx_fa_templates_version', 'template_version'),
		Index('idx_fa_templates_created', 'created_at'),
	)

	id = Column(String, primary_key=True, default=uuid7str)
	user_id = Column(String, ForeignKey('fa_users.id'), nullable=False)
	
	# Template data (encrypted)
	template_data = Column(LargeBinary, nullable=False)  # Encrypted facial features
	template_version = Column(String(20), default='1.0.0')
	template_algorithm = Column(String(50), default='facenet')
	
	# Quality metrics
	quality_score = Column(Float, nullable=False)
	sharpness_score = Column(Float)
	brightness_score = Column(Float)
	contrast_score = Column(Float)
	
	# Facial characteristics
	landmark_points = Column(JSON)  # 68+ facial landmarks
	face_pose = Column(JSON)  # Pitch, yaw, roll angles
	face_dimensions = Column(JSON)  # Width, height, aspect ratio
	
	# Enrollment context
	enrollment_device = Column(String(100))
	enrollment_location = Column(String(255))
	lighting_conditions = Column(String(50))
	
	# Encryption metadata
	encryption_key_id = Column(String)
	encryption_algorithm = Column(String(50), default='AES-256-GCM')
	
	# Audit fields
	created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
	updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
	is_active = Column(Boolean, default=True)
	
	# Compliance
	retention_date = Column(DateTime(timezone=True))
	anonymized = Column(Boolean, default=False)
	
	# Metadata
	metadata = Column(JSON, default=dict)

	# Relationships
	user = relationship("FaUser", back_populates="templates")

class FaVerification(Base):
	"""Facial verification attempts and results with contextual intelligence"""
	__tablename__ = 'fa_verifications'
	__table_args__ = (
		Index('idx_fa_verifications_user', 'user_id'),
		Index('idx_fa_verifications_type', 'verification_type'),
		Index('idx_fa_verifications_status', 'status'),
		Index('idx_fa_verifications_created', 'created_at'),
		Index('idx_fa_verifications_confidence', 'confidence_score'),
	)

	id = Column(String, primary_key=True, default=uuid7str)
	user_id = Column(String, ForeignKey('fa_users.id'), nullable=False)
	
	# Verification details
	verification_type = Column(String(50), nullable=False)
	template_id = Column(String, ForeignKey('fa_templates.id'))
	
	# Results
	status = Column(String(50), nullable=False)
	confidence_score = Column(Float, nullable=False)
	similarity_score = Column(Float)
	processing_time_ms = Column(Integer)
	
	# Liveness detection
	liveness_score = Column(Float)
	liveness_result = Column(String(50))
	anti_spoofing_checks = Column(JSON)
	
	# Quality assessment
	input_quality_score = Column(Float)
	quality_issues = Column(JSON)
	
	# Business context
	business_context = Column(JSON, default=dict)
	risk_factors = Column(JSON, default=list)
	access_level_required = Column(String(50))
	
	# Device and environment
	device_info = Column(JSON)
	location_data = Column(JSON)
	network_info = Column(JSON)
	
	# Contextual intelligence
	behavior_pattern = Column(JSON)
	time_pattern_analysis = Column(JSON)
	anomaly_indicators = Column(JSON)
	
	# Audit fields
	created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
	verification_timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
	
	# Failure analysis
	failure_reason = Column(String(255))
	failure_details = Column(JSON)
	retry_count = Column(Integer, default=0)
	
	# Metadata
	metadata = Column(JSON, default=dict)

	# Relationships
	user = relationship("FaUser", back_populates="verifications")

class FaEmotion(Base):
	"""Real-time emotion analysis results with micro-expression detection"""
	__tablename__ = 'fa_emotions'
	__table_args__ = (
		Index('idx_fa_emotions_user', 'user_id'),
		Index('idx_fa_emotions_primary', 'primary_emotion'),
		Index('idx_fa_emotions_confidence', 'confidence_score'),
		Index('idx_fa_emotions_created', 'created_at'),
	)

	id = Column(String, primary_key=True, default=uuid7str)
	user_id = Column(String, ForeignKey('fa_users.id'), nullable=False)
	verification_id = Column(String, ForeignKey('fa_verifications.id'))
	
	# Primary emotion detection
	primary_emotion = Column(String(50), nullable=False)
	confidence_score = Column(Float, nullable=False)
	
	# Emotion distribution (7 basic emotions)
	emotion_scores = Column(JSON, nullable=False)  # {happy: 0.8, sad: 0.1, ...}
	
	# Micro-expression analysis (20 micro-expressions)
	micro_expressions = Column(JSON, default=dict)
	
	# Stress and wellness indicators
	stress_level = Column(Float)  # 0.0 to 1.0
	arousal_level = Column(Float)  # Low to high arousal
	valence_score = Column(Float)  # Negative to positive
	
	# Physiological indicators
	blink_rate = Column(Float)
	eye_movement_pattern = Column(JSON)
	facial_muscle_tension = Column(JSON)
	
	# Context analysis
	environmental_factors = Column(JSON)
	social_context = Column(JSON)
	temporal_context = Column(JSON)
	
	# Business applications
	engagement_score = Column(Float)
	attention_level = Column(Float)
	deception_indicators = Column(JSON)
	
	# Processing details
	processing_algorithm = Column(String(50), default='emotion_net')
	analysis_duration_ms = Column(Integer)
	
	# Audit fields
	created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
	analysis_timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
	
	# Privacy settings
	anonymized = Column(Boolean, default=False)
	consent_for_analysis = Column(Boolean, default=True)
	
	# Metadata
	metadata = Column(JSON, default=dict)

	# Relationships
	user = relationship("FaUser", back_populates="emotions")

class FaCollaboration(Base):
	"""Multi-expert collaborative verification sessions"""
	__tablename__ = 'fa_collaborations'
	__table_args__ = (
		Index('idx_fa_collaborations_verification', 'verification_id'),
		Index('idx_fa_collaborations_status', 'status'),
		Index('idx_fa_collaborations_created', 'created_at'),
		Index('idx_fa_collaborations_urgency', 'urgency_level'),
	)

	id = Column(String, primary_key=True, default=uuid7str)
	verification_id = Column(String, ForeignKey('fa_verifications.id'), nullable=False)
	
	# Session details
	session_name = Column(String(255))
	description = Column(Text)
	status = Column(String(50), default='pending')
	
	# Collaboration metadata
	case_complexity = Column(String(50))
	urgency_level = Column(String(50), default='medium')
	required_expertise = Column(JSON, default=list)
	
	# Expert participation
	invited_experts = Column(JSON, default=list)
	active_experts = Column(JSON, default=list)
	expert_decisions = Column(JSON, default=dict)
	
	# Consensus building
	consensus_threshold = Column(Float, default=0.75)
	current_consensus = Column(Float)
	consensus_achieved = Column(Boolean, default=False)
	final_decision = Column(String(50))
	
	# Session timeline
	started_at = Column(DateTime(timezone=True))
	ended_at = Column(DateTime(timezone=True))
	duration_minutes = Column(Integer)
	
	# Business context
	business_impact = Column(String(50))
	compliance_requirements = Column(JSON, default=list)
	escalation_path = Column(JSON)
	
	# Knowledge sharing
	learning_notes = Column(Text)
	best_practices = Column(JSON)
	decision_rationale = Column(Text)
	
	# Audit fields
	created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
	created_by = Column(String, nullable=False)
	updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
	
	# Metadata
	metadata = Column(JSON, default=dict)

class FaAuditLog(Base):
	"""Comprehensive audit logging for facial recognition activities"""
	__tablename__ = 'fa_audit_logs'
	__table_args__ = (
		Index('idx_fa_audit_tenant', 'tenant_id'),
		Index('idx_fa_audit_action', 'action_type'),
		Index('idx_fa_audit_user', 'user_id'),
		Index('idx_fa_audit_created', 'created_at'),
		Index('idx_fa_audit_resource', 'resource_type', 'resource_id'),
	)

	id = Column(String, primary_key=True, default=uuid7str)
	tenant_id = Column(String, nullable=False)
	
	# Action details
	action_type = Column(String(100), nullable=False)
	resource_type = Column(String(100), nullable=False)
	resource_id = Column(String)
	
	# User context
	user_id = Column(String)
	actor_id = Column(String, nullable=False)  # Who performed the action
	actor_type = Column(String(50), default='user')  # user, system, api
	
	# Action details
	action_description = Column(Text)
	action_result = Column(String(50))  # success, failure, partial
	
	# Data changes
	old_values = Column(JSON)
	new_values = Column(JSON)
	changed_fields = Column(JSON, default=list)
	
	# Context information
	ip_address = Column(String(45))
	user_agent = Column(String(500))
	device_info = Column(JSON)
	location_info = Column(JSON)
	
	# Business context
	business_justification = Column(Text)
	approval_chain = Column(JSON)
	compliance_flags = Column(JSON, default=list)
	
	# Technical details
	processing_time_ms = Column(Integer)
	api_endpoint = Column(String(255))
	request_id = Column(String)
	session_id = Column(String)
	
	# Error details (if applicable)
	error_code = Column(String(50))
	error_message = Column(Text)
	error_stack_trace = Column(Text)
	
	# Audit fields
	created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
	
	# Retention and compliance
	retention_date = Column(DateTime(timezone=True))
	compliance_reviewed = Column(Boolean, default=False)
	
	# Metadata
	metadata = Column(JSON, default=dict)

class FaSettings(Base):
	"""User and tenant configuration settings"""
	__tablename__ = 'fa_settings'
	__table_args__ = (
		Index('idx_fa_settings_tenant', 'tenant_id'),
		Index('idx_fa_settings_user', 'user_id'),
		Index('idx_fa_settings_category', 'setting_category'),
		Index('idx_fa_settings_key', 'setting_key'),
	)

	id = Column(String, primary_key=True, default=uuid7str)
	tenant_id = Column(String, nullable=False)
	user_id = Column(String)  # NULL for tenant-level settings
	
	# Setting identification
	setting_category = Column(String(100), nullable=False)
	setting_key = Column(String(100), nullable=False)
	setting_value = Column(JSON, nullable=False)
	
	# Setting metadata
	setting_type = Column(String(50), default='configuration')
	data_type = Column(String(50), default='json')
	is_encrypted = Column(Boolean, default=False)
	
	# Validation rules
	validation_schema = Column(JSON)
	allowed_values = Column(JSON)
	default_value = Column(JSON)
	
	# Access control
	required_permission = Column(String(100))
	modification_level = Column(String(50), default='user')  # user, admin, system
	
	# Audit fields
	created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
	updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
	created_by = Column(String)
	updated_by = Column(String)
	
	# Versioning
	version = Column(Integer, default=1)
	previous_value = Column(JSON)
	
	# Metadata
	metadata = Column(JSON, default=dict)

# Pydantic Models for API

def _validate_confidence_score(v: float) -> float:
	"""Validate confidence score is between 0 and 1"""
	assert 0.0 <= v <= 1.0, 'Confidence score must be between 0.0 and 1.0'
	return v

def _validate_quality_score(v: float) -> float:
	"""Validate quality score is between 0 and 1"""
	assert 0.0 <= v <= 1.0, 'Quality score must be between 0.0 and 1.0'
	return v

def _validate_emotion_scores(v: dict[str, float]) -> dict[str, float]:
	"""Validate emotion scores dictionary"""
	assert isinstance(v, dict), 'Emotion scores must be a dictionary'
	for emotion, score in v.items():
		assert 0.0 <= score <= 1.0, f'Emotion score for {emotion} must be between 0.0 and 1.0'
	return v

class FaUserCreate(BaseModel):
	"""Create facial user request model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	external_user_id: str = Field(..., description="External user ID from APG auth_rbac")
	full_name: str = Field(..., min_length=1, max_length=255, description="User full name")
	email: str = Field(..., description="User email address")
	department: str | None = Field(None, max_length=100, description="User department")
	role: str | None = Field(None, max_length=100, description="User role")
	consent_given: bool = Field(False, description="Privacy consent given")
	metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class FaUserUpdate(BaseModel):
	"""Update facial user request model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	full_name: str | None = Field(None, min_length=1, max_length=255)
	email: str | None = Field(None)
	department: str | None = Field(None, max_length=100)
	role: str | None = Field(None, max_length=100)
	consent_given: bool | None = Field(None)
	anonymize_analytics: bool | None = Field(None)
	cross_border_consent: bool | None = Field(None)
	metadata: dict[str, Any] | None = Field(None)

class FaUserResponse(BaseModel):
	"""Facial user response model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str
	tenant_id: str
	external_user_id: str
	full_name: str | None
	email: str | None
	department: str | None
	role: str | None
	enrollment_status: str
	consent_given: bool
	consent_date: datetime | None
	created_at: datetime
	updated_at: datetime
	status: str
	metadata: dict[str, Any]

class FaTemplateCreate(BaseModel):
	"""Create facial template request model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	user_id: str = Field(..., description="User ID")
	template_data: bytes = Field(..., description="Encrypted template data")
	quality_score: float = Field(..., description="Template quality score")
	template_algorithm: str = Field(default="facenet", description="Algorithm used")
	landmark_points: dict[str, Any] | None = Field(None, description="Facial landmarks")
	enrollment_device: str | None = Field(None, max_length=100)
	lighting_conditions: str | None = Field(None, max_length=50)
	metadata: dict[str, Any] = Field(default_factory=dict)

	quality_score: float = Field(..., validators=[AfterValidator(_validate_quality_score)])

class FaTemplateResponse(BaseModel):
	"""Facial template response model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str
	user_id: str
	template_version: str
	template_algorithm: str
	quality_score: float
	created_at: datetime
	is_active: bool
	metadata: dict[str, Any]

class FaVerificationRequest(BaseModel):
	"""Facial verification request model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	user_id: str = Field(..., description="User ID for verification")
	verification_type: FaVerificationType = Field(..., description="Type of verification")
	face_data: bytes = Field(..., description="Face image data for verification")
	business_context: dict[str, Any] = Field(default_factory=dict, description="Business context")
	require_liveness: bool = Field(default=True, description="Require liveness detection")
	require_emotion: bool = Field(default=False, description="Include emotion analysis")
	device_info: dict[str, Any] = Field(default_factory=dict, description="Device information")
	location_data: dict[str, Any] = Field(default_factory=dict, description="Location data")

class FaVerificationResponse(BaseModel):
	"""Facial verification response model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str
	user_id: str
	verification_type: FaVerificationType
	status: FaProcessingStatus
	confidence_score: float
	similarity_score: float | None
	processing_time_ms: int
	liveness_score: float | None
	liveness_result: FaLivenessResult | None
	input_quality_score: float | None
	business_context: dict[str, Any]
	verification_timestamp: datetime
	failure_reason: str | None
	metadata: dict[str, Any]

	confidence_score: float = Field(..., validators=[AfterValidator(_validate_confidence_score)])

class FaEmotionRequest(BaseModel):
	"""Emotion analysis request model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	user_id: str | None = Field(None, description="User ID (optional for anonymous analysis)")
	face_data: bytes = Field(..., description="Face image data for emotion analysis")
	include_micro_expressions: bool = Field(default=False, description="Include micro-expression analysis")
	include_stress_indicators: bool = Field(default=False, description="Include stress analysis")
	business_context: dict[str, Any] = Field(default_factory=dict, description="Business context")
	consent_for_analysis: bool = Field(default=True, description="Consent for emotion analysis")

class FaEmotionResponse(BaseModel):
	"""Emotion analysis response model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str
	user_id: str | None
	primary_emotion: FaEmotionType
	confidence_score: float
	emotion_scores: dict[str, float]
	micro_expressions: dict[str, float] | None
	stress_level: float | None
	arousal_level: float | None
	valence_score: float | None
	engagement_score: float | None
	attention_level: float | None
	analysis_timestamp: datetime
	metadata: dict[str, Any]

	confidence_score: float = Field(..., validators=[AfterValidator(_validate_confidence_score)])
	emotion_scores: dict[str, float] = Field(..., validators=[AfterValidator(_validate_emotion_scores)])

class FaCollaborationCreate(BaseModel):
	"""Create collaboration session request model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	verification_id: str = Field(..., description="Verification ID requiring collaboration")
	session_name: str = Field(..., min_length=1, max_length=255, description="Session name")
	description: str | None = Field(None, description="Session description")
	case_complexity: str = Field(default="medium", description="Case complexity level")
	urgency_level: str = Field(default="medium", description="Urgency level")
	required_expertise: list[str] = Field(default_factory=list, description="Required expert skills")
	invited_experts: list[str] = Field(default_factory=list, description="List of expert IDs")
	business_impact: str | None = Field(None, description="Business impact level")
	compliance_requirements: list[str] = Field(default_factory=list, description="Compliance requirements")

class FaCollaborationResponse(BaseModel):
	"""Collaboration session response model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str
	verification_id: str
	session_name: str
	description: str | None
	status: str
	case_complexity: str
	urgency_level: str
	required_expertise: list[str]
	invited_experts: list[str]
	active_experts: list[str]
	consensus_threshold: float
	current_consensus: float | None
	consensus_achieved: bool
	final_decision: str | None
	created_at: datetime
	started_at: datetime | None
	ended_at: datetime | None
	created_by: str
	metadata: dict[str, Any]

def _log_model_created(model_name: str, model_id: str) -> None:
	"""Log model creation for audit purposes"""
	print(f"Created {model_name} with ID: {model_id}")

def _log_model_updated(model_name: str, model_id: str) -> None:
	"""Log model update for audit purposes"""
	print(f"Updated {model_name} with ID: {model_id}")

def _log_model_deleted(model_name: str, model_id: str) -> None:
	"""Log model deletion for audit purposes"""
	print(f"Deleted {model_name} with ID: {model_id}")

async def create_database_tables(engine) -> bool:
	"""Create all facial recognition database tables"""
	try:
		assert engine is not None, "Database engine cannot be None"
		
		# Create all tables
		Base.metadata.create_all(bind=engine)
		
		_log_model_created("Database Tables", "all_facial_tables")
		return True
		
	except Exception as e:
		print(f"Failed to create database tables: {e}")
		return False

async def drop_database_tables(engine) -> bool:
	"""Drop all facial recognition database tables"""
	try:
		assert engine is not None, "Database engine cannot be None"
		
		# Drop all tables
		Base.metadata.drop_all(bind=engine)
		
		_log_model_deleted("Database Tables", "all_facial_tables")
		return True
		
	except Exception as e:
		print(f"Failed to drop database tables: {e}")
		return False

# Export models for use in other modules
__all__ = [
	# Enums
	'FaVerificationType', 'FaEmotionType', 'FaQualityLevel', 
	'FaLivenessResult', 'FaProcessingStatus',
	
	# SQLAlchemy Models
	'FaUser', 'FaTemplate', 'FaVerification', 'FaEmotion', 
	'FaCollaboration', 'FaAuditLog', 'FaSettings',
	
	# Pydantic Models
	'FaUserCreate', 'FaUserUpdate', 'FaUserResponse',
	'FaTemplateCreate', 'FaTemplateResponse',
	'FaVerificationRequest', 'FaVerificationResponse',
	'FaEmotionRequest', 'FaEmotionResponse',
	'FaCollaborationCreate', 'FaCollaborationResponse',
	
	# Utility Functions
	'create_database_tables', 'drop_database_tables',
	'Base'
]