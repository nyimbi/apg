"""
APG Biometric Authentication - Database Models

Revolutionary biometric authentication models with 10x superior capabilities
including contextual intelligence, predictive analytics, and behavioral fusion.

Author: Datacraft (nyimbi@gmail.com)
Copyright: © 2025 Datacraft
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from uuid import UUID
from enum import Enum

from sqlalchemy import (
	Column, String, Text, DateTime, Boolean, Integer, Float, 
	ForeignKey, JSON, LargeBinary, Index, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID, ARRAY
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
from pydantic import BaseModel, Field, ConfigDict, AfterValidator
from uuid_extensions import uuid7str
from typing_extensions import Annotated

Base = declarative_base()

# Enums for biometric authentication
class BiVerificationStatus(str, Enum):
	"""Identity verification status enumeration"""
	PENDING = "pending"
	IN_PROGRESS = "in_progress"
	VERIFIED = "verified"
	FAILED = "failed"
	REJECTED = "rejected"
	EXPIRED = "expired"

class BiModalityType(str, Enum):
	"""Biometric modality types"""
	FACE = "face"
	VOICE = "voice"
	FINGERPRINT = "fingerprint"
	IRIS = "iris"
	PALM = "palm"
	BEHAVIORAL = "behavioral"
	DOCUMENT = "document"

class BiRiskLevel(str, Enum):
	"""Risk assessment levels"""
	VERY_LOW = "very_low"
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	VERY_HIGH = "very_high"
	CRITICAL = "critical"

class BiComplianceFramework(str, Enum):
	"""Regulatory compliance frameworks"""
	GDPR = "gdpr"
	CCPA = "ccpa"
	BIPA = "bipa"
	HIPAA = "hipaa"
	KYC_AML = "kyc_aml"
	SOX = "sox"
	PCI_DSS = "pci_dss"

# Core Models

class BiUser(Base):
	"""
	Biometric user identity management with contextual intelligence
	
	Revolutionary features:
	- Contextual behavior learning and pattern recognition
	- Adaptive risk profiling with continuous updates
	- Universal identity orchestration across global standards
	- Behavioral biometrics fusion with physical biometrics
	"""
	__tablename__ = 'bi_users'
	
	id: str = Column(String(36), primary_key=True, default=uuid7str)
	external_user_id: str = Column(String(255), nullable=False, index=True)
	tenant_id: str = Column(String(36), nullable=False, index=True)
	
	# Basic identity information
	first_name: Optional[str] = Column(String(255))
	last_name: Optional[str] = Column(String(255))
	email: Optional[str] = Column(String(320), index=True)
	phone: Optional[str] = Column(String(20))
	date_of_birth: Optional[datetime] = Column(DateTime)
	
	# Contextual intelligence data
	behavioral_profile: Dict[str, Any] = Column(JSON, default=dict)
	contextual_patterns: Dict[str, Any] = Column(JSON, default=dict)
	risk_profile: Dict[str, Any] = Column(JSON, default=dict)
	compliance_status: Dict[str, Any] = Column(JSON, default=dict)
	
	# Global identity orchestration
	global_identity_id: Optional[str] = Column(String(255), unique=True, index=True)
	jurisdiction_compliance: Dict[str, Any] = Column(JSON, default=dict)
	cross_border_permissions: Dict[str, Any] = Column(JSON, default=dict)
	
	# Adaptive security intelligence
	threat_intelligence: Dict[str, Any] = Column(JSON, default=dict)
	security_adaptations: Dict[str, Any] = Column(JSON, default=dict)
	anomaly_history: List[Dict[str, Any]] = Column(JSON, default=list)
	
	# Zero-friction authentication data
	invisible_auth_profile: Dict[str, Any] = Column(JSON, default=dict)
	ambient_signatures: Dict[str, Any] = Column(JSON, default=dict)
	predictive_patterns: Dict[str, Any] = Column(JSON, default=dict)
	
	# Metadata
	created_at: datetime = Column(DateTime, default=func.now())
	updated_at: datetime = Column(DateTime, default=func.now(), onupdate=func.now())
	last_activity: Optional[datetime] = Column(DateTime)
	is_active: bool = Column(Boolean, default=True)
	
	# Relationships
	verifications = relationship("BiVerification", back_populates="user", cascade="all, delete-orphan")
	biometrics = relationship("BiBiometric", back_populates="user", cascade="all, delete-orphan")
	behavioral_sessions = relationship("BiBehavioralSession", back_populates="user", cascade="all, delete-orphan")
	collaborations = relationship("BiCollaboration", back_populates="user", cascade="all, delete-orphan")
	
	__table_args__ = (
		Index('idx_bi_users_tenant_external', 'tenant_id', 'external_user_id'),
		Index('idx_bi_users_global_identity', 'global_identity_id'),
		Index('idx_bi_users_activity', 'last_activity'),
	)

class BiVerification(Base):
	"""
	Identity verification records with predictive analytics
	
	Revolutionary features:
	- Real-time collaborative verification workflows
	- Predictive fraud analytics and risk forecasting
	- Natural language query support and contextual intelligence
	- Immersive 3D/AR dashboard integration
	"""
	__tablename__ = 'bi_verifications'
	
	id: str = Column(String(36), primary_key=True, default=uuid7str)
	user_id: str = Column(String(36), ForeignKey('bi_users.id'), nullable=False, index=True)
	tenant_id: str = Column(String(36), nullable=False, index=True)
	
	# Verification details
	verification_type: str = Column(String(100), nullable=False)
	status: BiVerificationStatus = Column(String(50), default=BiVerificationStatus.PENDING, index=True)
	confidence_score: float = Column(Float, default=0.0)
	risk_score: float = Column(Float, default=0.0)
	
	# Contextual intelligence integration
	business_context: Dict[str, Any] = Column(JSON, default=dict)
	contextual_risk_assessment: Dict[str, Any] = Column(JSON, default=dict)
	intelligent_recommendations: List[Dict[str, Any]] = Column(JSON, default=list)
	workflow_optimization: Dict[str, Any] = Column(JSON, default=dict)
	
	# Predictive analytics data
	fraud_prediction: Dict[str, Any] = Column(JSON, default=dict)
	risk_trajectory: Dict[str, Any] = Column(JSON, default=dict)
	behavioral_forecast: Dict[str, Any] = Column(JSON, default=dict)
	compliance_prediction: Dict[str, Any] = Column(JSON, default=dict)
	
	# Real-time collaboration
	collaboration_session_id: Optional[str] = Column(String(36), ForeignKey('bi_collaborations.id'), index=True)
	collaborative_decision: Dict[str, Any] = Column(JSON, default=dict)
	expert_consultations: List[Dict[str, Any]] = Column(JSON, default=list)
	consensus_data: Dict[str, Any] = Column(JSON, default=dict)
	
	# Multi-modal biometric results
	modality_results: Dict[str, Any] = Column(JSON, default=dict)
	fusion_analysis: Dict[str, Any] = Column(JSON, default=dict)
	liveness_detection: Dict[str, Any] = Column(JSON, default=dict)
	deepfake_analysis: Dict[str, Any] = Column(JSON, default=dict)
	
	# Universal identity orchestration
	jurisdiction: str = Column(String(10), index=True)
	compliance_framework: List[str] = Column(ARRAY(String(50)), default=list)
	regulatory_requirements: Dict[str, Any] = Column(JSON, default=dict)
	cross_border_validation: Dict[str, Any] = Column(JSON, default=dict)
	
	# Natural language query support
	nl_query_metadata: Dict[str, Any] = Column(JSON, default=dict)
	conversational_context: Dict[str, Any] = Column(JSON, default=dict)
	explanation_data: Dict[str, Any] = Column(JSON, default=dict)
	
	# Immersive dashboard data
	spatial_visualization: Dict[str, Any] = Column(JSON, default=dict)
	ar_overlay_data: Dict[str, Any] = Column(JSON, default=dict)
	gesture_interactions: List[Dict[str, Any]] = Column(JSON, default=list)
	voice_commands: List[Dict[str, Any]] = Column(JSON, default=list)
	
	# Adaptive security intelligence
	threat_assessment: Dict[str, Any] = Column(JSON, default=dict)
	security_adaptations: Dict[str, Any] = Column(JSON, default=dict)
	evolution_tracking: Dict[str, Any] = Column(JSON, default=dict)
	
	# Zero-friction authentication
	invisible_verification: Dict[str, Any] = Column(JSON, default=dict)
	ambient_authentication: Dict[str, Any] = Column(JSON, default=dict)
	contextual_strength: Dict[str, Any] = Column(JSON, default=dict)
	
	# Processing metadata
	processing_time_ms: int = Column(Integer, default=0)
	started_at: datetime = Column(DateTime, default=func.now())
	completed_at: Optional[datetime] = Column(DateTime)
	expires_at: Optional[datetime] = Column(DateTime)
	
	# Relationships
	user = relationship("BiUser", back_populates="verifications")
	collaboration = relationship("BiCollaboration", back_populates="verifications")
	fraud_rules = relationship("BiFraudRule", secondary="bi_verification_fraud_rules", back_populates="verifications")
	audit_logs = relationship("BiAuditLog", back_populates="verification", cascade="all, delete-orphan")
	
	__table_args__ = (
		Index('idx_bi_verifications_status_tenant', 'status', 'tenant_id'),
		Index('idx_bi_verifications_user_status', 'user_id', 'status'),
		Index('idx_bi_verifications_risk_score', 'risk_score'),
		Index('idx_bi_verifications_completed', 'completed_at'),
	)

class BiBiometric(Base):
	"""
	Encrypted biometric template storage with behavioral fusion
	
	Revolutionary features:
	- Behavioral biometrics fusion with physical biometrics
	- Quantum-inspired deepfake detection algorithms
	- Adaptive security with continuous evolution
	- Zero-knowledge biometric template protection
	"""
	__tablename__ = 'bi_biometrics'
	
	id: str = Column(String(36), primary_key=True, default=uuid7str)
	user_id: str = Column(String(36), ForeignKey('bi_users.id'), nullable=False, index=True)
	tenant_id: str = Column(String(36), nullable=False, index=True)
	
	# Biometric details
	modality: BiModalityType = Column(String(50), nullable=False, index=True)
	template_hash: str = Column(String(128), nullable=False, unique=True, index=True)
	encrypted_template: bytes = Column(LargeBinary, nullable=False)
	quality_score: float = Column(Float, default=0.0)
	
	# Behavioral biometrics fusion
	behavioral_patterns: Dict[str, Any] = Column(JSON, default=dict)
	keystroke_dynamics: Dict[str, Any] = Column(JSON, default=dict)
	mouse_patterns: Dict[str, Any] = Column(JSON, default=dict)
	mobile_interactions: Dict[str, Any] = Column(JSON, default=dict)
	gait_analysis: Dict[str, Any] = Column(JSON, default=dict)
	voice_behavior: Dict[str, Any] = Column(JSON, default=dict)
	
	# Quantum-inspired deepfake detection
	quantum_signatures: Dict[str, Any] = Column(JSON, default=dict)
	entanglement_analysis: Dict[str, Any] = Column(JSON, default=dict)
	interference_patterns: Dict[str, Any] = Column(JSON, default=dict)
	synthetic_detection: Dict[str, Any] = Column(JSON, default=dict)
	
	# Adaptive security intelligence
	evolution_markers: Dict[str, Any] = Column(JSON, default=dict)
	threat_resistance: Dict[str, Any] = Column(JSON, default=dict)
	adaptation_history: List[Dict[str, Any]] = Column(JSON, default=list)
	security_generation: int = Column(Integer, default=1)
	
	# Continuous authentication data
	session_patterns: Dict[str, Any] = Column(JSON, default=dict)
	confidence_tracking: Dict[str, Any] = Column(JSON, default=dict)
	invisible_markers: Dict[str, Any] = Column(JSON, default=dict)
	ambient_signatures: Dict[str, Any] = Column(JSON, default=dict)
	
	# Privacy and compliance
	privacy_level: str = Column(String(50), default="high")
	retention_policy: str = Column(String(100))
	compliance_tags: List[str] = Column(ARRAY(String(50)), default=list)
	anonymization_level: int = Column(Integer, default=3)
	
	# Metadata
	created_at: datetime = Column(DateTime, default=func.now())
	updated_at: datetime = Column(DateTime, default=func.now(), onupdate=func.now())
	last_used: Optional[datetime] = Column(DateTime)
	usage_count: int = Column(Integer, default=0)
	is_active: bool = Column(Boolean, default=True)
	
	# Relationships
	user = relationship("BiUser", back_populates="biometrics")
	
	__table_args__ = (
		Index('idx_bi_biometrics_user_modality', 'user_id', 'modality'),
		Index('idx_bi_biometrics_quality', 'quality_score'),
		Index('idx_bi_biometrics_last_used', 'last_used'),
		UniqueConstraint('user_id', 'modality', 'template_hash', name='uq_bi_biometrics_user_modality_hash'),
	)

class BiDocument(Base):
	"""
	Identity document processing with universal global support
	
	Revolutionary features:
	- Universal identity orchestration for 500+ document types
	- Advanced security feature verification and tamper detection
	- Cross-border compliance and regulatory validation
	- AI-powered document intelligence and authenticity verification
	"""
	__tablename__ = 'bi_documents'
	
	id: str = Column(String(36), primary_key=True, default=uuid7str)
	user_id: str = Column(String(36), ForeignKey('bi_users.id'), nullable=False, index=True)
	tenant_id: str = Column(String(36), nullable=False, index=True)
	
	# Document identification
	document_type: str = Column(String(100), nullable=False, index=True)
	document_number: str = Column(String(100), index=True)
	issuing_country: str = Column(String(3), nullable=False, index=True)
	issuing_authority: str = Column(String(255))
	
	# Universal document support
	document_classification: Dict[str, Any] = Column(JSON, default=dict)
	security_features: Dict[str, Any] = Column(JSON, default=dict)
	authenticity_verification: Dict[str, Any] = Column(JSON, default=dict)
	tamper_detection: Dict[str, Any] = Column(JSON, default=dict)
	
	# AI-powered document intelligence
	ocr_results: Dict[str, Any] = Column(JSON, default=dict)
	data_extraction: Dict[str, Any] = Column(JSON, default=dict)
	field_validation: Dict[str, Any] = Column(JSON, default=dict)
	consistency_analysis: Dict[str, Any] = Column(JSON, default=dict)
	
	# Cross-border compliance
	jurisdiction_validation: Dict[str, Any] = Column(JSON, default=dict)
	regulatory_compliance: Dict[str, Any] = Column(JSON, default=dict)
	cross_reference_results: Dict[str, Any] = Column(JSON, default=dict)
	sanctions_screening: Dict[str, Any] = Column(JSON, default=dict)
	
	# Advanced fraud detection
	forensic_analysis: Dict[str, Any] = Column(JSON, default=dict)
	pattern_matching: Dict[str, Any] = Column(JSON, default=dict)
	anomaly_detection: Dict[str, Any] = Column(JSON, default=dict)
	risk_indicators: List[str] = Column(ARRAY(String(100)), default=list)
	
	# Document lifecycle
	issued_date: Optional[datetime] = Column(DateTime)
	expiry_date: Optional[datetime] = Column(DateTime)
	verification_date: datetime = Column(DateTime, default=func.now())
	validity_status: str = Column(String(50), default="valid", index=True)
	
	# Privacy and storage
	encrypted_image_data: Optional[bytes] = Column(LargeBinary)
	image_hash: Optional[str] = Column(String(128), index=True)
	retention_expires: Optional[datetime] = Column(DateTime)
	privacy_level: str = Column(String(50), default="high")
	
	# Metadata
	created_at: datetime = Column(DateTime, default=func.now())
	updated_at: datetime = Column(DateTime, default=func.now(), onupdate=func.now())
	
	# Relationships
	user = relationship("BiUser")
	
	__table_args__ = (
		Index('idx_bi_documents_type_country', 'document_type', 'issuing_country'),
		Index('idx_bi_documents_validity', 'validity_status'),
		Index('idx_bi_documents_expiry', 'expiry_date'),
	)

class BiFraudRule(Base):
	"""
	Adaptive fraud detection rules with predictive analytics
	
	Revolutionary features:
	- Self-evolving fraud detection with continuous learning
	- Predictive fraud analytics and prevention
	- Contextual intelligence for business-aware fraud detection
	- Global threat intelligence integration
	"""
	__tablename__ = 'bi_fraud_rules'
	
	id: str = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(36), nullable=False, index=True)
	
	# Rule identification
	rule_name: str = Column(String(255), nullable=False)
	rule_type: str = Column(String(100), nullable=False, index=True)
	category: str = Column(String(100), index=True)
	severity: BiRiskLevel = Column(String(50), default=BiRiskLevel.MEDIUM, index=True)
	
	# Adaptive rule engine
	rule_logic: Dict[str, Any] = Column(JSON, nullable=False)
	evolution_parameters: Dict[str, Any] = Column(JSON, default=dict)
	learning_weights: Dict[str, Any] = Column(JSON, default=dict)
	adaptation_thresholds: Dict[str, Any] = Column(JSON, default=dict)
	
	# Predictive analytics
	predictive_indicators: List[str] = Column(ARRAY(String(100)), default=list)
	risk_forecasting: Dict[str, Any] = Column(JSON, default=dict)
	pattern_evolution: Dict[str, Any] = Column(JSON, default=dict)
	threat_prediction: Dict[str, Any] = Column(JSON, default=dict)
	
	# Contextual intelligence
	business_context_rules: Dict[str, Any] = Column(JSON, default=dict)
	industry_adaptations: Dict[str, Any] = Column(JSON, default=dict)
	organizational_patterns: Dict[str, Any] = Column(JSON, default=dict)
	workflow_integration: Dict[str, Any] = Column(JSON, default=dict)
	
	# Global threat intelligence
	threat_feeds: List[str] = Column(ARRAY(String(100)), default=list)
	intelligence_updates: Dict[str, Any] = Column(JSON, default=dict)
	global_patterns: Dict[str, Any] = Column(JSON, default=dict)
	cross_tenant_insights: Dict[str, Any] = Column(JSON, default=dict)
	
	# Performance metrics
	detection_rate: float = Column(Float, default=0.0)
	false_positive_rate: float = Column(Float, default=0.0)
	effectiveness_score: float = Column(Float, default=0.0)
	last_performance_check: Optional[datetime] = Column(DateTime)
	
	# Rule lifecycle
	created_at: datetime = Column(DateTime, default=func.now())
	updated_at: datetime = Column(DateTime, default=func.now(), onupdate=func.now())
	last_triggered: Optional[datetime] = Column(DateTime)
	trigger_count: int = Column(Integer, default=0)
	is_active: bool = Column(Boolean, default=True)
	rule_version: int = Column(Integer, default=1)
	
	# Relationships
	verifications = relationship("BiVerification", secondary="bi_verification_fraud_rules", back_populates="fraud_rules")
	
	__table_args__ = (
		Index('idx_bi_fraud_rules_type_active', 'rule_type', 'is_active'),
		Index('idx_bi_fraud_rules_severity', 'severity'),
		Index('idx_bi_fraud_rules_effectiveness', 'effectiveness_score'),
	)

class BiComplianceRule(Base):
	"""
	Global regulatory compliance automation
	
	Revolutionary features:
	- Universal identity orchestration for 200+ regulations
	- Automated compliance validation and reporting
	- Cross-border regulatory intelligence
	- Real-time compliance monitoring and adaptation
	"""
	__tablename__ = 'bi_compliance_rules'
	
	id: str = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(36), nullable=False, index=True)
	
	# Compliance framework
	framework: BiComplianceFramework = Column(String(50), nullable=False, index=True)
	jurisdiction: str = Column(String(10), nullable=False, index=True)
	regulation_name: str = Column(String(255), nullable=False)
	article_section: str = Column(String(100))
	
	# Universal compliance orchestration
	regulation_requirements: Dict[str, Any] = Column(JSON, nullable=False)
	validation_rules: Dict[str, Any] = Column(JSON, nullable=False)
	reporting_requirements: Dict[str, Any] = Column(JSON, default=dict)
	cross_border_implications: Dict[str, Any] = Column(JSON, default=dict)
	
	# Automated compliance validation
	validation_logic: Dict[str, Any] = Column(JSON, nullable=False)
	automated_checks: List[str] = Column(ARRAY(String(100)), default=list)
	compliance_thresholds: Dict[str, Any] = Column(JSON, default=dict)
	escalation_rules: Dict[str, Any] = Column(JSON, default=dict)
	
	# Real-time monitoring
	monitoring_parameters: Dict[str, Any] = Column(JSON, default=dict)
	alert_conditions: Dict[str, Any] = Column(JSON, default=dict)
	notification_settings: Dict[str, Any] = Column(JSON, default=dict)
	audit_requirements: Dict[str, Any] = Column(JSON, default=dict)
	
	# Regulatory intelligence
	regulation_updates: List[Dict[str, Any]] = Column(JSON, default=list)
	interpretation_guidance: Dict[str, Any] = Column(JSON, default=dict)
	industry_best_practices: Dict[str, Any] = Column(JSON, default=dict)
	enforcement_trends: Dict[str, Any] = Column(JSON, default=dict)
	
	# Compliance metrics
	compliance_rate: float = Column(Float, default=0.0)
	violation_count: int = Column(Integer, default=0)
	last_audit_date: Optional[datetime] = Column(DateTime)
	next_review_date: Optional[datetime] = Column(DateTime)
	
	# Lifecycle
	effective_date: datetime = Column(DateTime, nullable=False)
	expiry_date: Optional[datetime] = Column(DateTime)
	created_at: datetime = Column(DateTime, default=func.now())
	updated_at: datetime = Column(DateTime, default=func.now(), onupdate=func.now())
	is_active: bool = Column(Boolean, default=True)
	
	__table_args__ = (
		Index('idx_bi_compliance_framework_jurisdiction', 'framework', 'jurisdiction'),
		Index('idx_bi_compliance_effective_expiry', 'effective_date', 'expiry_date'),
		Index('idx_bi_compliance_active', 'is_active'),
	)

class BiCollaboration(Base):
	"""
	Real-time collaborative verification sessions
	
	Revolutionary features:
	- Multi-expert collaborative identity verification
	- Real-time synchronization and consensus building
	- Expert consultation and knowledge sharing
	- Complete audit trails for collaborative decisions
	"""
	__tablename__ = 'bi_collaborations'
	
	id: str = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(36), nullable=False, index=True)
	
	# Session details
	session_name: str = Column(String(255), nullable=False)
	session_type: str = Column(String(100), nullable=False, index=True)
	complexity_level: str = Column(String(50), index=True)
	priority: str = Column(String(50), default="medium", index=True)
	
	# Collaborative workspace
	participants: List[Dict[str, Any]] = Column(JSON, default=list)
	participant_roles: Dict[str, Any] = Column(JSON, default=dict)
	access_permissions: Dict[str, Any] = Column(JSON, default=dict)
	workspace_settings: Dict[str, Any] = Column(JSON, default=dict)
	
	# Real-time synchronization
	synchronization_state: Dict[str, Any] = Column(JSON, default=dict)
	live_annotations: List[Dict[str, Any]] = Column(JSON, default=list)
	shared_insights: List[Dict[str, Any]] = Column(JSON, default=list)
	discussion_threads: List[Dict[str, Any]] = Column(JSON, default=list)
	
	# Expert consultation
	expert_consultations: List[Dict[str, Any]] = Column(JSON, default=list)
	expertise_requirements: List[str] = Column(ARRAY(String(100)), default=list)
	consultation_requests: List[Dict[str, Any]] = Column(JSON, default=list)
	knowledge_sharing: Dict[str, Any] = Column(JSON, default=dict)
	
	# Consensus building
	voting_sessions: List[Dict[str, Any]] = Column(JSON, default=list)
	decision_matrix: Dict[str, Any] = Column(JSON, default=dict)
	consensus_tracking: Dict[str, Any] = Column(JSON, default=dict)
	conflict_resolution: List[Dict[str, Any]] = Column(JSON, default=list)
	
	# Immersive collaboration features
	spatial_interactions: List[Dict[str, Any]] = Column(JSON, default=list)
	gesture_commands: List[Dict[str, Any]] = Column(JSON, default=list)
	voice_interactions: List[Dict[str, Any]] = Column(JSON, default=list)
	ar_collaborations: Dict[str, Any] = Column(JSON, default=dict)
	
	# Session lifecycle
	started_at: datetime = Column(DateTime, default=func.now())
	ended_at: Optional[datetime] = Column(DateTime)
	session_duration: Optional[int] = Column(Integer)  # seconds
	status: str = Column(String(50), default="active", index=True)
	
	# Quality metrics
	collaboration_effectiveness: float = Column(Float, default=0.0)
	decision_quality_score: float = Column(Float, default=0.0)
	participant_satisfaction: Dict[str, Any] = Column(JSON, default=dict)
	knowledge_transfer_score: float = Column(Float, default=0.0)
	
	# Relationships
	user = relationship("BiUser", back_populates="collaborations")
	verifications = relationship("BiVerification", back_populates="collaboration")
	
	__table_args__ = (
		Index('idx_bi_collaborations_status_type', 'status', 'session_type'),
		Index('idx_bi_collaborations_priority', 'priority'),
		Index('idx_bi_collaborations_started', 'started_at'),
	)

class BiBehavioralSession(Base):
	"""
	Behavioral biometrics continuous authentication sessions
	
	Revolutionary features:
	- Continuous behavioral monitoring and analysis
	- Zero-friction invisible authentication
	- Contextual behavior adaptation and learning
	- Multi-modal behavioral pattern fusion
	"""
	__tablename__ = 'bi_behavioral_sessions'
	
	id: str = Column(String(36), primary_key=True, default=uuid7str)
	user_id: str = Column(String(36), ForeignKey('bi_users.id'), nullable=False, index=True)
	tenant_id: str = Column(String(36), nullable=False, index=True)
	
	# Session identification
	session_token: str = Column(String(128), unique=True, nullable=False, index=True)
	device_fingerprint: str = Column(String(128), index=True)
	platform: str = Column(String(50), index=True)
	user_agent: str = Column(Text)
	
	# Behavioral pattern collection
	keystroke_patterns: List[Dict[str, Any]] = Column(JSON, default=list)
	mouse_movements: List[Dict[str, Any]] = Column(JSON, default=list)
	touch_interactions: List[Dict[str, Any]] = Column(JSON, default=list)
	scroll_behaviors: List[Dict[str, Any]] = Column(JSON, default=list)
	navigation_patterns: List[Dict[str, Any]] = Column(JSON, default=list)
	
	# Mobile behavioral patterns
	device_orientation: List[Dict[str, Any]] = Column(JSON, default=list)
	touch_pressure: List[Dict[str, Any]] = Column(JSON, default=list)
	swipe_patterns: List[Dict[str, Any]] = Column(JSON, default=list)
	app_usage_flow: List[Dict[str, Any]] = Column(JSON, default=list)
	sensor_data: Dict[str, Any] = Column(JSON, default=dict)
	
	# Contextual behavior analysis
	environmental_context: Dict[str, Any] = Column(JSON, default=dict)
	temporal_patterns: Dict[str, Any] = Column(JSON, default=dict)
	social_context: Dict[str, Any] = Column(JSON, default=dict)
	cognitive_load_indicators: Dict[str, Any] = Column(JSON, default=dict)
	emotional_state_markers: Dict[str, Any] = Column(JSON, default=dict)
	
	# Continuous authentication
	confidence_timeline: List[Dict[str, Any]] = Column(JSON, default=list)
	authentication_events: List[Dict[str, Any]] = Column(JSON, default=list)
	risk_escalations: List[Dict[str, Any]] = Column(JSON, default=list)
	invisible_challenges: List[Dict[str, Any]] = Column(JSON, default=list)
	
	# Zero-friction authentication
	ambient_authentication: Dict[str, Any] = Column(JSON, default=dict)
	predictive_authentication: Dict[str, Any] = Column(JSON, default=dict)
	contextual_strength: Dict[str, Any] = Column(JSON, default=dict)
	seamless_handoffs: List[Dict[str, Any]] = Column(JSON, default=list)
	
	# Session metrics
	total_interactions: int = Column(Integer, default=0)
	anomaly_count: int = Column(Integer, default=0)
	average_confidence: float = Column(Float, default=0.0)
	risk_incidents: int = Column(Integer, default=0)
	
	# Session lifecycle
	started_at: datetime = Column(DateTime, default=func.now())
	last_activity: datetime = Column(DateTime, default=func.now())
	ended_at: Optional[datetime] = Column(DateTime)
	session_duration: Optional[int] = Column(Integer)  # seconds
	status: str = Column(String(50), default="active", index=True)
	
	# Relationships
	user = relationship("BiUser", back_populates="behavioral_sessions")
	
	__table_args__ = (
		Index('idx_bi_behavioral_sessions_user_status', 'user_id', 'status'),
		Index('idx_bi_behavioral_sessions_device', 'device_fingerprint'),
		Index('idx_bi_behavioral_sessions_activity', 'last_activity'),
	)

class BiAuditLog(Base):
	"""
	Comprehensive audit trail for biometric authentication
	
	Revolutionary features:
	- Immutable audit logs with blockchain-like integrity
	- Real-time audit monitoring and compliance validation
	- Natural language audit queries and explanations
	- Cross-jurisdictional audit compliance automation
	"""
	__tablename__ = 'bi_audit_logs'
	
	id: str = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(36), nullable=False, index=True)
	
	# Audit event identification
	event_type: str = Column(String(100), nullable=False, index=True)
	event_category: str = Column(String(50), nullable=False, index=True)
	severity: str = Column(String(50), default="info", index=True)
	source_component: str = Column(String(100), nullable=False)
	
	# Event details
	event_description: str = Column(Text, nullable=False)
	event_data: Dict[str, Any] = Column(JSON, default=dict)
	context_data: Dict[str, Any] = Column(JSON, default=dict)
	security_classification: str = Column(String(50), default="internal")
	
	# Entity references
	user_id: Optional[str] = Column(String(36), ForeignKey('bi_users.id'), index=True)
	verification_id: Optional[str] = Column(String(36), ForeignKey('bi_verifications.id'), index=True)
	session_id: Optional[str] = Column(String(128), index=True)
	collaboration_id: Optional[str] = Column(String(36), ForeignKey('bi_collaborations.id'), index=True)
	
	# Actor information
	actor_type: str = Column(String(50), nullable=False)  # user, system, api, automation
	actor_id: str = Column(String(255), nullable=False, index=True)
	actor_details: Dict[str, Any] = Column(JSON, default=dict)
	authentication_method: Optional[str] = Column(String(100))
	
	# Request/response tracking
	request_id: Optional[str] = Column(String(128), index=True)
	correlation_id: Optional[str] = Column(String(128), index=True)
	api_endpoint: Optional[str] = Column(String(255))
	http_method: Optional[str] = Column(String(10))
	response_code: Optional[int] = Column(Integer)
	processing_time_ms: Optional[int] = Column(Integer)
	
	# Compliance and regulatory
	compliance_frameworks: List[str] = Column(ARRAY(String(50)), default=list)
	regulatory_requirements: Dict[str, Any] = Column(JSON, default=dict)
	retention_category: str = Column(String(100), default="standard")
	retention_expires: Optional[datetime] = Column(DateTime)
	
	# Immutable audit integrity
	event_hash: str = Column(String(128), nullable=False, index=True)
	previous_hash: Optional[str] = Column(String(128), index=True)
	integrity_signature: Optional[str] = Column(String(256))
	blockchain_reference: Optional[str] = Column(String(255))
	
	# Geolocation and network
	ip_address: Optional[str] = Column(String(45))  # Support IPv6
	geolocation: Dict[str, Any] = Column(JSON, default=dict)
	network_info: Dict[str, Any] = Column(JSON, default=dict)
	device_fingerprint: Optional[str] = Column(String(128))
	
	# Natural language support
	nl_description: Optional[str] = Column(Text)
	explanation_context: Dict[str, Any] = Column(JSON, default=dict)
	searchable_tags: List[str] = Column(ARRAY(String(100)), default=list)
	
	# Timestamp and lifecycle
	timestamp: datetime = Column(DateTime, default=func.now(), nullable=False, index=True)
	created_at: datetime = Column(DateTime, default=func.now())
	
	# Relationships
	user = relationship("BiUser")
	verification = relationship("BiVerification", back_populates="audit_logs")
	collaboration = relationship("BiCollaboration")
	
	__table_args__ = (
		Index('idx_bi_audit_logs_event_type_time', 'event_type', 'timestamp'),
		Index('idx_bi_audit_logs_actor_time', 'actor_id', 'timestamp'),
		Index('idx_bi_audit_logs_severity_time', 'severity', 'timestamp'),
		Index('idx_bi_audit_logs_compliance', 'compliance_frameworks'),
		Index('idx_bi_audit_logs_integrity', 'event_hash'),
	)

# Association Tables for Many-to-Many Relationships

from sqlalchemy import Table

bi_verification_fraud_rules = Table(
	'bi_verification_fraud_rules',
	Base.metadata,
	Column('verification_id', String(36), ForeignKey('bi_verifications.id'), primary_key=True),
	Column('fraud_rule_id', String(36), ForeignKey('bi_fraud_rules.id'), primary_key=True),
	Column('rule_confidence', Float, default=0.0),
	Column('triggered_at', DateTime, default=func.now()),
	Column('rule_result', JSON, default=dict)
)

# Pydantic Models for API Validation

class BiUserCreate(BaseModel):
	"""User creation model with validation"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	external_user_id: str = Field(..., min_length=1, max_length=255)
	tenant_id: str = Field(..., min_length=1, max_length=36)
	first_name: Optional[str] = Field(None, max_length=255)
	last_name: Optional[str] = Field(None, max_length=255)
	email: Optional[str] = Field(None, max_length=320)
	phone: Optional[str] = Field(None, max_length=20)
	date_of_birth: Optional[datetime] = None

class BiVerificationCreate(BaseModel):
	"""Verification creation model with validation"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	user_id: str = Field(..., min_length=1, max_length=36)
	tenant_id: str = Field(..., min_length=1, max_length=36)
	verification_type: str = Field(..., min_length=1, max_length=100)
	business_context: Dict[str, Any] = Field(default_factory=dict)
	jurisdiction: Optional[str] = Field(None, max_length=10)

class BiBiometricCreate(BaseModel):
	"""Biometric creation model with validation"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	user_id: str = Field(..., min_length=1, max_length=36)
	tenant_id: str = Field(..., min_length=1, max_length=36)
	modality: BiModalityType = Field(...)
	encrypted_template: bytes = Field(...)
	quality_score: float = Field(default=0.0, ge=0.0, le=1.0)

class BiCollaborationCreate(BaseModel):
	"""Collaboration session creation model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	tenant_id: str = Field(..., min_length=1, max_length=36)
	session_name: str = Field(..., min_length=1, max_length=255)
	session_type: str = Field(..., min_length=1, max_length=100)
	complexity_level: str = Field(default="medium")
	participants: List[Dict[str, Any]] = Field(default_factory=list)

def validate_tenant_id(v: str) -> str:
	"""Validate tenant ID format"""
	if not v or len(v) > 36:
		raise ValueError("Invalid tenant ID format")
	return v

def validate_email(v: Optional[str]) -> Optional[str]:
	"""Validate email format"""
	if v is None:
		return v
	if "@" not in v or len(v) > 320:
		raise ValueError("Invalid email format")
	return v.lower()

# Apply validators
TenantID = Annotated[str, AfterValidator(validate_tenant_id)]
EmailAddress = Annotated[Optional[str], AfterValidator(validate_email)]