"""
APG Access Control Integration Models

Revolutionary data models for world-class access control with APG integration patterns.
Implements all 10 revolutionary differentiators with async SQLAlchemy 2.0 and modern typing.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from sqlalchemy import (
	Column, String, Integer, Boolean, DateTime, Text, JSON, Float,
	ForeignKey, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.ext.declarative import declarative_base
from uuid_extensions import uuid7str
from pydantic import BaseModel, Field, ConfigDict, AfterValidator
from pydantic.functional_validators import field_validator

# APG Base Imports
from apg.base.models import APGBaseModel, APGAuditMixin, APGTenantMixin
from apg.base.validators import validate_tenant_id, validate_user_id

Base = declarative_base()

class ACSecurityPolicy(APGBaseModel, APGAuditMixin, APGTenantMixin):
	"""Revolutionary security policy model with multiverse simulation support."""
	__tablename__ = 'ac_security_policy'
	
	# Primary Identity
	policy_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid7str)
	policy_name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
	policy_version: Mapped[str] = mapped_column(String(20), default="1.0.0")
	
	# Policy Classification
	policy_type: Mapped[str] = mapped_column(
		String(50), nullable=False, index=True
	)  # authentication, authorization, compliance, behavioral
	security_level: Mapped[str] = mapped_column(
		String(20), default="standard", index=True
	)  # basic, standard, high, critical, quantum
	
	# Policy Definition
	description: Mapped[Optional[str]] = mapped_column(Text)
	policy_rules: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
	conditions: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
	exceptions: Mapped[List[str]] = mapped_column(JSON, default=list)
	
	# Revolutionary Features
	neuromorphic_patterns: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
	holographic_requirements: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
	quantum_encryption_required: Mapped[bool] = mapped_column(Boolean, default=False)
	ambient_context_aware: Mapped[bool] = mapped_column(Boolean, default=False)
	emotional_intelligence_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
	temporal_constraints: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
	
	# Multiverse Simulation
	simulation_tested: Mapped[bool] = mapped_column(Boolean, default=False)
	simulation_results: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
	parallel_universe_validated: Mapped[bool] = mapped_column(Boolean, default=False)
	
	# Policy Scope
	applies_to: Mapped[List[str]] = mapped_column(JSON, default=list)  # capabilities, users, roles
	resource_filters: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
	time_constraints: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
	
	# Status and Lifecycle
	is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
	effective_from: Mapped[Optional[datetime]] = mapped_column(DateTime)
	expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
	
	# Performance Metrics
	evaluation_count: Mapped[int] = mapped_column(Integer, default=0)
	average_evaluation_time_ms: Mapped[float] = mapped_column(Float, default=0.0)
	violation_count: Mapped[int] = mapped_column(Integer, default=0)
	
	__table_args__ = (
		Index('idx_policy_tenant_type', 'tenant_id', 'policy_type'),
		Index('idx_policy_active', 'is_active', 'effective_from', 'expires_at'),
		UniqueConstraint('tenant_id', 'policy_name', 'policy_version'),
	)

class ACNeuromorphicProfile(APGBaseModel, APGAuditMixin, APGTenantMixin):
	"""Neuromorphic authentication profile for brain-inspired security patterns."""
	__tablename__ = 'ac_neuromorphic_profile'
	
	# Primary Identity
	profile_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid7str)
	user_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
	
	# Neuromorphic Pattern Data
	pattern_type: Mapped[str] = mapped_column(
		String(50), nullable=False
	)  # keystroke, mouse, behavioral, biometric_fusion
	neural_signature: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
	spike_patterns: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
	synaptic_weights: Mapped[List[float]] = mapped_column(JSON, default=list)
	
	# Learning and Adaptation
	training_iterations: Mapped[int] = mapped_column(Integer, default=0)
	learning_rate: Mapped[float] = mapped_column(Float, default=0.001)
	adaptation_threshold: Mapped[float] = mapped_column(Float, default=0.85)
	last_training: Mapped[Optional[datetime]] = mapped_column(DateTime)
	
	# Accuracy and Performance
	accuracy_score: Mapped[float] = mapped_column(Float, default=0.0)
	false_positive_rate: Mapped[float] = mapped_column(Float, default=0.0)
	false_negative_rate: Mapped[float] = mapped_column(Float, default=0.0)
	confidence_level: Mapped[float] = mapped_column(Float, default=0.0)
	
	# Behavioral Model
	behavioral_model: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
	temporal_patterns: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
	contextual_adaptations: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
	
	# Security Features
	anomaly_detection_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
	continuous_learning: Mapped[bool] = mapped_column(Boolean, default=True)
	privacy_preserving: Mapped[bool] = mapped_column(Boolean, default=True)
	
	# Status
	is_active: Mapped[bool] = mapped_column(Boolean, default=True)
	calibration_required: Mapped[bool] = mapped_column(Boolean, default=False)
	
	__table_args__ = (
		Index('idx_neuromorphic_user', 'tenant_id', 'user_id'),
		Index('idx_neuromorphic_pattern', 'pattern_type', 'is_active'),
	)

class ACHolographicIdentity(APGBaseModel, APGAuditMixin, APGTenantMixin):
	"""Holographic identity verification with quantum-encrypted storage."""
	__tablename__ = 'ac_holographic_identity'
	
	# Primary Identity
	hologram_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid7str)
	user_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
	
	# Holographic Data
	hologram_data_hash: Mapped[str] = mapped_column(String(128), nullable=False)
	hologram_3d_signature: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
	dimensional_coordinates: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
	depth_map_data: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
	
	# Quality Metrics
	hologram_quality: Mapped[float] = mapped_column(Float, nullable=False)
	resolution_3d: Mapped[Dict[str, int]] = mapped_column(JSON, default=dict)
	capture_conditions: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
	lighting_analysis: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
	
	# Verification Data
	verification_accuracy: Mapped[float] = mapped_column(Float, default=0.0)
	last_verification: Mapped[Optional[datetime]] = mapped_column(DateTime)
	verification_history: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, default=list)
	anti_spoofing_checks: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
	
	# Quantum Security
	quantum_encrypted: Mapped[bool] = mapped_column(Boolean, default=True)
	quantum_key_id: Mapped[Optional[str]] = mapped_column(String(64))
	post_quantum_algorithm: Mapped[Optional[str]] = mapped_column(String(50))
	
	# Biometric Integration
	facial_landmarks: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
	iris_patterns: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
	voice_print_3d: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
	
	# Status and Lifecycle
	is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
	last_scan: Mapped[Optional[datetime]] = mapped_column(DateTime)
	expiry_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
	
	__table_args__ = (
		Index('idx_holographic_user', 'tenant_id', 'user_id'),
		Index('idx_holographic_verified', 'is_verified', 'last_scan'),
	)

class ACQuantumKey(APGBaseModel, APGAuditMixin, APGTenantMixin):
	"""Quantum cryptographic key management for post-quantum security."""
	__tablename__ = 'ac_quantum_key'
	
	# Primary Identity
	key_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid7str)
	key_name: Mapped[str] = mapped_column(String(200), nullable=False)
	
	# Quantum Key Properties
	algorithm_type: Mapped[str] = mapped_column(
		String(50), nullable=False
	)  # CRYSTALS-Kyber, CRYSTALS-Dilithium, FALCON, etc.
	key_size_bits: Mapped[int] = mapped_column(Integer, nullable=False)
	quantum_resistance_level: Mapped[str] = mapped_column(String(20), default="high")
	
	# Key Data (encrypted at rest)
	public_key_data: Mapped[str] = mapped_column(Text, nullable=False)
	key_usage_type: Mapped[str] = mapped_column(
		String(30), nullable=False
	)  # encryption, signing, key_exchange
	
	# Quantum Properties
	entanglement_state: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
	quantum_state_verification: Mapped[bool] = mapped_column(Boolean, default=False)
	decoherence_time: Mapped[Optional[float]] = mapped_column(Float)
	
	# Key Distribution
	qkd_enabled: Mapped[bool] = mapped_column(Boolean, default=False)  # Quantum Key Distribution
	distribution_history: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, default=list)
	shared_with_entities: Mapped[List[str]] = mapped_column(JSON, default=list)
	
	# Security and Lifecycle
	generation_source: Mapped[str] = mapped_column(String(50), default="quantum_rng")
	entropy_level: Mapped[float] = mapped_column(Float, default=0.0)
	compromise_indicator: Mapped[bool] = mapped_column(Boolean, default=False)
	
	# Usage Tracking
	usage_count: Mapped[int] = mapped_column(Integer, default=0)
	last_used: Mapped[Optional[datetime]] = mapped_column(DateTime)
	max_usage_limit: Mapped[Optional[int]] = mapped_column(Integer)
	
	# Status
	is_active: Mapped[bool] = mapped_column(Boolean, default=True)
	created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
	expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
	
	__table_args__ = (
		Index('idx_quantum_key_tenant_type', 'tenant_id', 'key_usage_type'),
		Index('idx_quantum_key_active', 'is_active', 'expires_at'),
	)

class ACThreatIntelligence(APGBaseModel, APGAuditMixin, APGTenantMixin):
	"""Predictive threat intelligence with AI-powered analysis."""
	__tablename__ = 'ac_threat_intelligence'
	
	# Primary Identity
	threat_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid7str)
	threat_signature: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
	
	# Threat Classification
	threat_type: Mapped[str] = mapped_column(
		String(50), nullable=False, index=True
	)  # brute_force, anomalous_behavior, credential_stuffing, etc.
	severity_level: Mapped[str] = mapped_column(
		String(20), nullable=False, index=True
	)  # low, medium, high, critical, quantum_level
	confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
	
	# AI-Powered Analysis
	ml_model_prediction: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
	behavioral_analysis: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
	pattern_recognition_results: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
	
	# Predictive Intelligence
	predicted_attack_vector: Mapped[Optional[str]] = mapped_column(String(100))
	probability_of_success: Mapped[float] = mapped_column(Float, default=0.0)
	estimated_impact: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
	recommended_countermeasures: Mapped[List[str]] = mapped_column(JSON, default=list)
	
	# Contextual Information
	source_ip_address: Mapped[Optional[str]] = mapped_column(String(45))
	user_agent_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
	geolocation_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
	device_fingerprint: Mapped[Optional[str]] = mapped_column(String(128))
	
	# Temporal Analysis
	first_detected: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
	last_activity: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
	attack_pattern_timeline: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, default=list)
	
	# Response and Mitigation
	is_mitigated: Mapped[bool] = mapped_column(Boolean, default=False)
	mitigation_actions: Mapped[List[str]] = mapped_column(JSON, default=list)
	automated_response_triggered: Mapped[bool] = mapped_column(Boolean, default=False)
	
	# Intelligence Sharing
	threat_feed_reported: Mapped[bool] = mapped_column(Boolean, default=False)
	shared_indicators: Mapped[List[str]] = mapped_column(JSON, default=list)
	
	__table_args__ = (
		Index('idx_threat_tenant_severity', 'tenant_id', 'severity_level'),
		Index('idx_threat_type_detected', 'threat_type', 'first_detected'),
		Index('idx_threat_mitigated', 'is_mitigated', 'last_activity'),
	)

class ACAmbientDevice(APGBaseModel, APGAuditMixin, APGTenantMixin):
	"""Ambient intelligence device for IoT-based security monitoring."""
	__tablename__ = 'ac_ambient_device'
	
	# Primary Identity
	device_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid7str)
	device_name: Mapped[str] = mapped_column(String(200), nullable=False)
	device_type: Mapped[str] = mapped_column(
		String(50), nullable=False
	)  # camera, microphone, motion_sensor, biometric_scanner, etc.
	
	# Device Properties
	manufacturer: Mapped[Optional[str]] = mapped_column(String(100))
	model: Mapped[Optional[str]] = mapped_column(String(100))
	firmware_version: Mapped[Optional[str]] = mapped_column(String(50))
	mac_address: Mapped[Optional[str]] = mapped_column(String(17))
	
	# Location and Context
	physical_location: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
	coverage_area: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
	environmental_context: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
	
	# Security Capabilities
	security_features: Mapped[List[str]] = mapped_column(JSON, default=list)
	data_encryption_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
	authentication_method: Mapped[str] = mapped_column(String(50), default="certificate")
	trust_level: Mapped[float] = mapped_column(Float, default=0.5)
	
	# Monitoring Configuration
	monitoring_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
	data_collection_frequency: Mapped[int] = mapped_column(Integer, default=60)  # seconds
	alert_thresholds: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
	
	# Ambient Intelligence
	behavior_learning_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
	pattern_recognition_active: Mapped[bool] = mapped_column(Boolean, default=True)
	anomaly_detection_sensitivity: Mapped[float] = mapped_column(Float, default=0.8)
	
	# Device Health
	last_heartbeat: Mapped[Optional[datetime]] = mapped_column(DateTime)
	battery_level: Mapped[Optional[float]] = mapped_column(Float)
	connection_status: Mapped[str] = mapped_column(String(20), default="online")
	maintenance_schedule: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
	
	# Status
	is_active: Mapped[bool] = mapped_column(Boolean, default=True)
	deployment_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
	
	__table_args__ = (
		Index('idx_ambient_device_tenant_type', 'tenant_id', 'device_type'),
		Index('idx_ambient_device_status', 'is_active', 'connection_status'),
	)

# Pydantic Models for API and Validation
class SecurityPolicySchema(BaseModel):
	"""Pydantic model for security policy validation."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	policy_name: str = Field(..., min_length=1, max_length=200)
	policy_type: str = Field(..., pattern="^(authentication|authorization|compliance|behavioral)$")
	security_level: str = Field(default="standard", pattern="^(basic|standard|high|critical|quantum)$")
	description: Optional[str] = None
	policy_rules: Dict[str, Any] = Field(default_factory=dict)
	conditions: Dict[str, Any] = Field(default_factory=dict)
	applies_to: List[str] = Field(default_factory=list)
	is_active: bool = True
	
	# Revolutionary features
	neuromorphic_patterns: Optional[Dict[str, Any]] = None
	holographic_requirements: Optional[Dict[str, Any]] = None
	quantum_encryption_required: bool = False
	ambient_context_aware: bool = False
	emotional_intelligence_enabled: bool = False

class NeuromorphicProfileSchema(BaseModel):
	"""Pydantic model for neuromorphic profile validation."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	user_id: str = Field(..., min_length=36, max_length=36)
	pattern_type: str = Field(..., pattern="^(keystroke|mouse|behavioral|biometric_fusion)$")
	neural_signature: Dict[str, Any] = Field(...)
	learning_rate: float = Field(default=0.001, ge=0.0001, le=0.1)
	adaptation_threshold: float = Field(default=0.85, ge=0.5, le=0.99)
	is_active: bool = True
	
	@field_validator('user_id')
	@classmethod
	def validate_user_id_format(cls, v: str) -> str:
		"""Validate user ID format."""
		return validate_user_id(v)

class HolographicIdentitySchema(BaseModel):
	"""Pydantic model for holographic identity validation."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	user_id: str = Field(..., min_length=36, max_length=36)
	hologram_3d_signature: Dict[str, Any] = Field(...)
	hologram_quality: float = Field(..., ge=0.0, le=1.0)
	quantum_encrypted: bool = True
	is_verified: bool = False
	
	@field_validator('hologram_quality')
	@classmethod
	def validate_quality_threshold(cls, v: float) -> float:
		"""Ensure hologram quality meets minimum threshold."""
		if v < 0.8:
			raise ValueError("Hologram quality must be at least 0.8 for security purposes")
		return v

# Export all models
__all__ = [
	"ACSecurityPolicy",
	"ACNeuromorphicProfile", 
	"ACHolographicIdentity",
	"ACQuantumKey",
	"ACThreatIntelligence",
	"ACAmbientDevice",
	"SecurityPolicySchema",
	"NeuromorphicProfileSchema",
	"HolographicIdentitySchema"
]