#!/usr/bin/env python3
"""
APG Master Data Management (MDM) - Core Data Models
Advanced multi-tenant master data management with APG ecosystem integration

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Union
from enum import Enum
import json
import uuid

from pydantic import BaseModel, Field, ConfigDict, validator, root_validator
from pydantic.types import UUID4
from uuid_extensions import uuid7str
from sqlalchemy import Column, String, Text, DateTime, Float, Integer, Boolean, JSON, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB

# APG Base Model Integration
Base = declarative_base()


class EntityType(str, Enum):
	"""Master data entity types supported by APG MDM"""
	CUSTOMER = "customer"
	PRODUCT = "product" 
	SUPPLIER = "supplier"
	EMPLOYEE = "employee"
	ASSET = "asset"
	LOCATION = "location"
	ACCOUNT = "account"
	CONTRACT = "contract"
	ORGANIZATION = "organization"
	CUSTOM = "custom"


class EntityStatus(str, Enum):
	"""Entity lifecycle status"""
	ACTIVE = "active"
	INACTIVE = "inactive"
	PENDING = "pending"
	MERGED = "merged"
	DELETED = "deleted"
	ARCHIVED = "archived"


class DataQualityStatus(str, Enum):
	"""Data quality assessment status"""
	EXCELLENT = "excellent"		# 95-100%
	GOOD = "good"				# 80-94%
	FAIR = "fair"				# 60-79%
	POOR = "poor"				# 40-59%
	CRITICAL = "critical"		# 0-39%


class MatchConfidence(str, Enum):
	"""Duplicate matching confidence levels"""
	EXACT = "exact"				# 100%
	HIGH = "high"				# 90-99%
	MEDIUM = "medium"			# 70-89%
	LOW = "low"					# 50-69%
	UNCERTAIN = "uncertain"		# <50%


class SurvivorshipRule(str, Enum):
	"""Golden record survivorship strategies"""
	MOST_RECENT = "most_recent"
	MOST_COMPLETE = "most_complete"
	MOST_TRUSTED_SOURCE = "most_trusted_source"
	HIGHEST_QUALITY = "highest_quality"
	CUSTOM_RULES = "custom_rules"
	AI_DETERMINED = "ai_determined"


# Pydantic Models for API Integration

class MdEntityBase(BaseModel):
	"""Base entity model for API operations"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True,
		use_enum_values=True
	)
	
	entity_type: EntityType
	entity_name: str = Field(..., min_length=1, max_length=255)
	entity_description: Optional[str] = Field(None, max_length=1000)
	business_key: str = Field(..., min_length=1, max_length=100)
	source_system: str = Field(..., min_length=1, max_length=100)
	tenant_id: str = Field(..., min_length=1, max_length=36)
	status: EntityStatus = EntityStatus.ACTIVE
	attributes: Dict[str, Any] = Field(default_factory=dict)
	tags: List[str] = Field(default_factory=list)
	data_classification: str = Field("internal", max_length=50)
	
	@validator('business_key')
	def validate_business_key(cls, v):
		if not v or v.strip() == "":
			raise ValueError("Business key cannot be empty")
		return v.strip()
	
	@validator('attributes')
	def validate_attributes(cls, v):
		if v is None:
			return {}
		# Ensure attributes are serializable
		try:
			json.dumps(v)
		except (TypeError, ValueError):
			raise ValueError("Attributes must be JSON serializable")
		return v


class MdEntityCreate(MdEntityBase):
	"""Entity creation request model"""
	pass


class MdEntityUpdate(BaseModel):
	"""Entity update request model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	entity_name: Optional[str] = Field(None, min_length=1, max_length=255)
	entity_description: Optional[str] = Field(None, max_length=1000)
	status: Optional[EntityStatus] = None
	attributes: Optional[Dict[str, Any]] = None
	tags: Optional[List[str]] = None
	data_classification: Optional[str] = Field(None, max_length=50)


class MdDataQualityScore(BaseModel):
	"""Data quality assessment model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	entity_id: str
	tenant_id: str
	overall_score: float = Field(..., ge=0.0, le=100.0)
	completeness_score: float = Field(..., ge=0.0, le=100.0)
	accuracy_score: float = Field(..., ge=0.0, le=100.0)
	consistency_score: float = Field(..., ge=0.0, le=100.0)
	validity_score: float = Field(..., ge=0.0, le=100.0)
	uniqueness_score: float = Field(..., ge=0.0, le=100.0)
	timeliness_score: float = Field(..., ge=0.0, le=100.0)
	quality_status: DataQualityStatus
	quality_issues: List[Dict[str, Any]] = Field(default_factory=list)
	assessment_timestamp: datetime = Field(default_factory=datetime.utcnow)
	assessment_duration_ms: Optional[float] = None


class MdMatchCandidate(BaseModel):
	"""Duplicate matching candidate model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	candidate_id: str
	match_score: float = Field(..., ge=0.0, le=100.0)
	confidence: MatchConfidence
	matching_attributes: List[str] = Field(default_factory=list)
	similarity_details: Dict[str, float] = Field(default_factory=dict)
	recommended_action: str  # merge, review, ignore
	match_explanation: Optional[str] = None


class MdDuplicateDetectionResult(BaseModel):
	"""Duplicate detection result model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	entity_id: str
	tenant_id: str
	total_candidates: int = 0
	high_confidence_matches: int = 0
	medium_confidence_matches: int = 0
	low_confidence_matches: int = 0
	match_candidates: List[MdMatchCandidate] = Field(default_factory=list)
	detection_timestamp: datetime = Field(default_factory=datetime.utcnow)
	detection_duration_ms: Optional[float] = None
	algorithm_version: str = "1.0.0"


# SQLAlchemy Database Models

class MdEntity(Base):
	"""Master Data Entity - Core entity storage with multi-tenant isolation"""
	__tablename__ = 'md_entities'
	
	# Primary identifier using UUID7 for time-ordered UUIDs
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Entity identification
	entity_type = Column(String(50), nullable=False, index=True)
	entity_name = Column(String(255), nullable=False)
	entity_description = Column(Text)
	business_key = Column(String(100), nullable=False)
	source_system = Column(String(100), nullable=False)
	
	# Entity lifecycle
	status = Column(String(20), nullable=False, default='active', index=True)
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
	created_by = Column(String(100), nullable=False)
	updated_by = Column(String(100), nullable=False)
	
	# Data attributes and metadata
	attributes = Column(JSONB, nullable=False, default=dict)
	tags = Column(JSONB, nullable=False, default=list)
	data_classification = Column(String(50), default='internal')
	
	# Data quality and governance
	quality_score = Column(Float, default=0.0, index=True)
	last_quality_check = Column(DateTime)
	is_golden_record = Column(Boolean, default=False, index=True)
	golden_record_id = Column(String(36), ForeignKey('md_golden_records.id'))
	
	# APG Integration fields
	audit_trail_id = Column(String(36))  # Links to APG audit capability
	encryption_key_id = Column(String(36))  # Links to APG encryption capability
	
	# Relationships
	versions = relationship("MdEntityVersion", back_populates="entity", cascade="all, delete-orphan")
	cross_references = relationship("MdCrossReference", back_populates="entity", cascade="all, delete-orphan")
	quality_assessments = relationship("MdDataQualityAssessment", back_populates="entity", cascade="all, delete-orphan")
	
	# Indexes for performance
	__table_args__ = (
		Index('ix_md_entities_tenant_type', 'tenant_id', 'entity_type'),
		Index('ix_md_entities_business_key', 'tenant_id', 'business_key', 'source_system'),
		Index('ix_md_entities_quality', 'tenant_id', 'quality_score'),
		Index('ix_md_entities_golden', 'tenant_id', 'is_golden_record'),
	)


class MdEntityVersion(Base):
	"""Entity version history for complete audit trail and rollback capability"""
	__tablename__ = 'md_entity_versions'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	entity_id = Column(String(36), ForeignKey('md_entities.id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Version metadata
	version_number = Column(Integer, nullable=False)
	version_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
	version_type = Column(String(20), nullable=False)  # create, update, merge, split
	created_by = Column(String(100), nullable=False)
	change_description = Column(Text)
	
	# Snapshot of entity state at this version
	entity_snapshot = Column(JSONB, nullable=False)
	attributes_snapshot = Column(JSONB, nullable=False)
	quality_score_snapshot = Column(Float)
	
	# Change tracking
	changed_fields = Column(JSONB)  # List of fields that changed
	previous_values = Column(JSONB)  # Previous values for changed fields
	change_source = Column(String(100))  # Source system or user that made change
	
	# Relationships
	entity = relationship("MdEntity", back_populates="versions")
	
	__table_args__ = (
		Index('ix_md_entity_versions_entity_version', 'entity_id', 'version_number'),
		Index('ix_md_entity_versions_tenant_time', 'tenant_id', 'version_timestamp'),
	)


class MdGoldenRecord(Base):
	"""Golden Record - Authoritative master data with survivorship rules"""
	__tablename__ = 'md_golden_records'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Golden record identification
	entity_type = Column(String(50), nullable=False)
	golden_record_name = Column(String(255), nullable=False)
	business_key = Column(String(100), nullable=False)
	
	# Consolidated attributes from source entities
	consolidated_attributes = Column(JSONB, nullable=False, default=dict)
	source_entity_ids = Column(JSONB, nullable=False, default=list)  # List of contributing entity IDs
	
	# Survivorship configuration
	survivorship_rules = Column(JSONB, nullable=False, default=dict)
	consolidation_method = Column(String(50), default='ai_determined')
	
	# Quality and confidence
	overall_quality_score = Column(Float, default=0.0)
	consolidation_confidence = Column(Float, default=0.0)
	data_completeness = Column(Float, default=0.0)
	
	# Lifecycle
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
	last_consolidation = Column(DateTime, default=datetime.utcnow)
	created_by = Column(String(100), nullable=False)
	
	# Status and governance
	is_active = Column(Boolean, default=True)
	approval_status = Column(String(20), default='auto_approved')  # auto_approved, pending, approved, rejected
	approved_by = Column(String(100))
	approved_at = Column(DateTime)
	
	__table_args__ = (
		Index('ix_md_golden_records_tenant_type', 'tenant_id', 'entity_type'),
		Index('ix_md_golden_records_quality', 'tenant_id', 'overall_quality_score'),
		Index('ix_md_golden_records_business_key', 'tenant_id', 'business_key'),
	)


class MdCrossReference(Base):
	"""Cross-system identifier mappings for entity integration"""
	__tablename__ = 'md_cross_references'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	entity_id = Column(String(36), ForeignKey('md_entities.id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Cross-reference identification
	source_system = Column(String(100), nullable=False)
	source_entity_id = Column(String(255), nullable=False)
	source_entity_type = Column(String(100))
	
	# Reference quality and reliability
	confidence_score = Column(Float, default=100.0)  # Confidence in this mapping
	is_primary_reference = Column(Boolean, default=False)
	reference_quality = Column(String(20), default='high')
	
	# Lifecycle and maintenance
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
	last_verified = Column(DateTime, default=datetime.utcnow)
	created_by = Column(String(100), nullable=False)
	
	# Status
	is_active = Column(Boolean, default=True)
	
	# Relationships
	entity = relationship("MdEntity", back_populates="cross_references")
	
	__table_args__ = (
		Index('ix_md_cross_references_source', 'tenant_id', 'source_system', 'source_entity_id'),
		Index('ix_md_cross_references_entity', 'tenant_id', 'entity_id'),
	)


class MdDataQualityAssessment(Base):
	"""Data quality assessment results with detailed scoring"""
	__tablename__ = 'md_data_quality_assessments'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	entity_id = Column(String(36), ForeignKey('md_entities.id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Quality scoring (0-100 scale)
	overall_score = Column(Float, nullable=False, index=True)
	completeness_score = Column(Float, nullable=False)
	accuracy_score = Column(Float, nullable=False)
	consistency_score = Column(Float, nullable=False)
	validity_score = Column(Float, nullable=False)
	uniqueness_score = Column(Float, nullable=False)
	timeliness_score = Column(Float, nullable=False)
	
	# Assessment metadata
	quality_status = Column(String(20), nullable=False)  # excellent, good, fair, poor, critical
	quality_issues = Column(JSONB, default=list)  # List of identified issues
	assessment_algorithm = Column(String(50), default='ml_enhanced')
	algorithm_version = Column(String(20), default='1.0.0')
	
	# Performance metrics
	assessment_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
	assessment_duration_ms = Column(Float)  # Processing time in milliseconds
	
	# Quality improvement recommendations
	recommendations = Column(JSONB, default=list)
	priority_issues = Column(JSONB, default=list)
	auto_fix_suggestions = Column(JSONB, default=list)
	
	# Relationships
	entity = relationship("MdEntity", back_populates="quality_assessments")
	
	__table_args__ = (
		Index('ix_md_quality_assessments_score', 'tenant_id', 'overall_score'),
		Index('ix_md_quality_assessments_status', 'tenant_id', 'quality_status'),
		Index('ix_md_quality_assessments_time', 'tenant_id', 'assessment_timestamp'),
	)


class MdMatchRule(Base):
	"""Configurable matching rules for duplicate detection"""
	__tablename__ = 'md_match_rules'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Rule identification
	rule_name = Column(String(255), nullable=False)
	rule_description = Column(Text)
	entity_type = Column(String(50), nullable=False)
	
	# Rule configuration
	rule_config = Column(JSONB, nullable=False)  # Matching algorithm configuration
	matching_attributes = Column(JSONB, nullable=False)  # Fields to match on
	weight_config = Column(JSONB, default=dict)  # Attribute weights for scoring
	
	# Thresholds
	exact_match_threshold = Column(Float, default=100.0)
	high_confidence_threshold = Column(Float, default=90.0)
	medium_confidence_threshold = Column(Float, default=70.0)
	minimum_match_threshold = Column(Float, default=50.0)
	
	# Rule performance and tuning
	rule_version = Column(String(20), default='1.0.0')
	performance_stats = Column(JSONB, default=dict)  # Precision, recall, F1 score
	last_performance_review = Column(DateTime)
	
	# Lifecycle
	is_active = Column(Boolean, default=True)
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
	created_by = Column(String(100), nullable=False)
	
	__table_args__ = (
		Index('ix_md_match_rules_tenant_type', 'tenant_id', 'entity_type'),
		Index('ix_md_match_rules_active', 'tenant_id', 'is_active'),
	)


class MdSurvivorshipRule(Base):
	"""Survivorship rules for golden record creation"""
	__tablename__ = 'md_survivorship_rules'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Rule identification
	rule_name = Column(String(255), nullable=False)
	rule_description = Column(Text)
	entity_type = Column(String(50), nullable=False)
	
	# Survivorship configuration
	survivorship_strategy = Column(String(50), default='ai_determined')
	attribute_rules = Column(JSONB, nullable=False, default=dict)  # Per-attribute survivorship rules
	source_system_rankings = Column(JSONB, default=dict)  # Trust ranking for source systems
	
	# Rule logic
	rule_conditions = Column(JSONB, default=dict)  # Conditions for rule application
	fallback_strategy = Column(String(50), default='most_recent')
	conflict_resolution = Column(JSONB, default=dict)  # How to resolve conflicts
	
	# Performance and validation
	rule_effectiveness = Column(Float, default=0.0)  # Success rate of rule
	validation_results = Column(JSONB, default=dict)
	last_validation = Column(DateTime)
	
	# Lifecycle
	is_active = Column(Boolean, default=True)
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
	created_by = Column(String(100), nullable=False)
	
	__table_args__ = (
		Index('ix_md_survivorship_rules_tenant_type', 'tenant_id', 'entity_type'),
		Index('ix_md_survivorship_rules_active', 'tenant_id', 'is_active'),
	)


class MdAuditLog(Base):
	"""Comprehensive audit logging for all MDM operations"""
	__tablename__ = 'md_audit_logs'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Audit event identification
	event_type = Column(String(50), nullable=False, index=True)  # create, update, delete, merge, etc.
	entity_id = Column(String(36), index=True)  # Can be null for system-wide events
	entity_type = Column(String(50), index=True)
	
	# Event details
	event_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
	event_description = Column(Text)
	event_details = Column(JSONB, default=dict)
	
	# User and system context
	user_id = Column(String(100), nullable=False)
	user_name = Column(String(255))
	source_system = Column(String(100))
	client_ip = Column(String(45))  # IPv6 compatible
	user_agent = Column(String(500))
	
	# Data changes
	before_values = Column(JSONB)  # Previous state
	after_values = Column(JSONB)   # New state
	changed_fields = Column(JSONB, default=list)  # List of changed field names
	
	# Operation context
	operation_id = Column(String(36))  # Links related operations
	batch_id = Column(String(36))      # Groups batch operations
	api_endpoint = Column(String(255)) # API endpoint used
	request_id = Column(String(36))    # Request tracing ID
	
	# Data governance
	data_sensitivity = Column(String(20), default='internal')
	compliance_tags = Column(JSONB, default=list)
	retention_date = Column(DateTime)  # When this log can be purged
	
	__table_args__ = (
		Index('ix_md_audit_logs_entity_time', 'tenant_id', 'entity_id', 'event_timestamp'),
		Index('ix_md_audit_logs_user_time', 'tenant_id', 'user_id', 'event_timestamp'),
		Index('ix_md_audit_logs_event_type', 'tenant_id', 'event_type'),
	)


class MdDataLineage(Base):
	"""Data lineage tracking for complete provenance"""
	__tablename__ = 'md_data_lineage'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Lineage relationship
	source_entity_id = Column(String(36), nullable=False, index=True)
	target_entity_id = Column(String(36), nullable=False, index=True)
	relationship_type = Column(String(50), nullable=False)  # derived_from, merged_into, split_from, etc.
	
	# Transformation details
	transformation_type = Column(String(100), nullable=False)
	transformation_details = Column(JSONB, default=dict)
	transformation_rules = Column(JSONB, default=dict)
	quality_impact = Column(JSONB, default=dict)
	
	# Lineage metadata
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	created_by = Column(String(100), nullable=False)
	source_system = Column(String(100))
	confidence_score = Column(Float, default=100.0)
	
	# Validation and verification
	is_verified = Column(Boolean, default=False)
	verified_by = Column(String(100))
	verified_at = Column(DateTime)
	verification_method = Column(String(100))
	
	__table_args__ = (
		Index('ix_md_data_lineage_source', 'tenant_id', 'source_entity_id'),
		Index('ix_md_data_lineage_target', 'tenant_id', 'target_entity_id'),
		Index('ix_md_data_lineage_type', 'tenant_id', 'relationship_type'),
	)


# Export all models
__all__ = [
	# Enums
	'EntityType', 'EntityStatus', 'DataQualityStatus', 'MatchConfidence', 'SurvivorshipRule',
	
	# Pydantic Models
	'MdEntityBase', 'MdEntityCreate', 'MdEntityUpdate',
	'MdDataQualityScore', 'MdMatchCandidate', 'MdDuplicateDetectionResult',
	
	# SQLAlchemy Models
	'MdEntity', 'MdEntityVersion', 'MdGoldenRecord', 'MdCrossReference',
	'MdDataQualityAssessment', 'MdMatchRule', 'MdSurvivorshipRule', 
	'MdAuditLog', 'MdDataLineage',
	
	# Base
	'Base'
]