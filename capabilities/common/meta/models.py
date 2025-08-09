#!/usr/bin/env python3
"""
APG Metadata Management - Core Data Models
Advanced metadata management with AI-powered intelligence and APG ecosystem integration

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Union
from enum import Enum
import json
from uuid_extensions import uuid7str

from pydantic import BaseModel, Field, ConfigDict, validator, root_validator, AfterValidator
from pydantic.types import UUID4
from sqlalchemy import Column, String, Text, DateTime, Float, Integer, Boolean, JSON, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.schema import CheckConstraint

# APG Base Model Integration
Base = declarative_base()


class AssetType(str, Enum):
	"""Types of metadata assets supported"""
	TABLE = "table"
	VIEW = "view"
	FILE = "file"
	API = "api"
	MODEL = "model"
	DASHBOARD = "dashboard"
	REPORT = "report"
	NOTEBOOK = "notebook"
	WORKFLOW = "workflow"
	STREAM = "stream"
	TOPIC = "topic"
	QUEUE = "queue"
	CUSTOM = "custom"


class AssetStatus(str, Enum):
	"""Asset lifecycle status"""
	ACTIVE = "active"
	DEPRECATED = "deprecated"
	ARCHIVED = "archived"
	DELETED = "deleted"
	DRAFT = "draft"
	PENDING = "pending"


class LineageType(str, Enum):
	"""Types of lineage relationships"""
	DIRECT = "direct"				# Direct transformation
	DERIVED = "derived"				# Calculated/derived from source
	AGGREGATED = "aggregated"		# Aggregation of multiple sources
	JOINED = "joined"				# Result of join operation
	FILTERED = "filtered"			# Subset of source data
	UNION = "union"					# Union of multiple sources
	CUSTOM = "custom"				# Custom transformation logic


class ClassificationType(str, Enum):
	"""Data classification types"""
	PII = "pii"						# Personally Identifiable Information
	PHI = "phi"						# Protected Health Information
	FINANCIAL = "financial"		# Financial/payment data
	CONFIDENTIAL = "confidential"	# Business confidential
	INTERNAL = "internal"			# Internal use only
	PUBLIC = "public"				# Public data
	RESTRICTED = "restricted"		# Restricted access
	SENSITIVE = "sensitive"			# Sensitive but not regulated


class SourceSystemType(str, Enum):
	"""Types of source systems"""
	DATABASE = "database"
	FILE_SYSTEM = "file_system"
	API_SERVICE = "api_service"
	ML_PLATFORM = "ml_platform"
	BI_TOOL = "bi_tool"
	ETL_TOOL = "etl_tool"
	STREAMING = "streaming"
	CLOUD_STORAGE = "cloud_storage"
	APPLICATION = "application"
	CUSTOM = "custom"


class QualityDimension(str, Enum):
	"""Data quality dimensions"""
	COMPLETENESS = "completeness"
	ACCURACY = "accuracy"
	CONSISTENCY = "consistency"
	VALIDITY = "validity"
	UNIQUENESS = "uniqueness"
	TIMELINESS = "timeliness"
	CONFORMITY = "conformity"
	INTEGRITY = "integrity"


class GovernanceAction(str, Enum):
	"""Governance actions that can be taken"""
	APPROVE = "approve"
	REJECT = "reject"
	REVIEW = "review"
	ARCHIVE = "archive"
	ESCALATE = "escalate"
	REMEDIATE = "remediate"


# Pydantic Models for API Integration

class MetaAssetBase(BaseModel):
	"""Base metadata asset model for API operations"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True,
		use_enum_values=True
	)
	
	name: str = Field(..., min_length=1, max_length=255, description="Asset name")
	display_name: Optional[str] = Field(None, max_length=255, description="Human-readable display name")
	description: Optional[str] = Field(None, max_length=2000, description="Asset description")
	asset_type: AssetType = Field(..., description="Type of metadata asset")
	source_system: str = Field(..., min_length=1, max_length=100, description="Source system identifier")
	source_system_type: SourceSystemType = Field(..., description="Type of source system")
	tenant_id: str = Field(..., min_length=1, max_length=36, description="Tenant identifier")
	status: AssetStatus = AssetStatus.ACTIVE
	
	# Schema and structure information
	schema_info: Dict[str, Any] = Field(default_factory=dict, description="Schema definition")
	column_count: Optional[int] = Field(None, ge=0, description="Number of columns/fields")
	row_count: Optional[int] = Field(None, ge=0, description="Number of rows/records")
	size_bytes: Optional[int] = Field(None, ge=0, description="Size in bytes")
	
	# Business context
	business_domain: Optional[str] = Field(None, max_length=100, description="Business domain")
	business_owner: Optional[str] = Field(None, max_length=100, description="Business owner")
	technical_owner: Optional[str] = Field(None, max_length=100, description="Technical owner")
	data_steward: Optional[str] = Field(None, max_length=100, description="Data steward")
	
	# Classification and governance
	classifications: List[str] = Field(default_factory=list, description="Data classifications")
	tags: List[str] = Field(default_factory=list, description="Asset tags")
	custom_attributes: Dict[str, Any] = Field(default_factory=dict, description="Custom attributes")
	
	# Quality and usage
	quality_score: Optional[float] = Field(None, ge=0.0, le=100.0, description="Overall quality score")
	usage_score: Optional[float] = Field(None, ge=0.0, le=100.0, description="Usage popularity score")
	
	@validator('name')
	def validate_name(cls, v):
		if not v or v.strip() == "":
			raise ValueError("Asset name cannot be empty")
		return v.strip()
	
	@validator('schema_info')
	def validate_schema_info(cls, v):
		if v is None:
			return {}
		try:
			json.dumps(v)
		except (TypeError, ValueError):
			raise ValueError("Schema info must be JSON serializable")
		return v


class MetaAssetCreate(MetaAssetBase):
	"""Asset creation request model"""
	pass


class MetaAssetUpdate(BaseModel):
	"""Asset update request model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	name: Optional[str] = Field(None, min_length=1, max_length=255)
	display_name: Optional[str] = Field(None, max_length=255)
	description: Optional[str] = Field(None, max_length=2000)
	status: Optional[AssetStatus] = None
	business_domain: Optional[str] = Field(None, max_length=100)
	business_owner: Optional[str] = Field(None, max_length=100)
	technical_owner: Optional[str] = Field(None, max_length=100)
	data_steward: Optional[str] = Field(None, max_length=100)
	classifications: Optional[List[str]] = None
	tags: Optional[List[str]] = None
	custom_attributes: Optional[Dict[str, Any]] = None


class MetaLineageCreate(BaseModel):
	"""Lineage relationship creation model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	source_asset_id: str = Field(..., description="Source asset ID")
	target_asset_id: str = Field(..., description="Target asset ID")
	relationship_type: LineageType = Field(..., description="Type of lineage relationship")
	column_mappings: Dict[str, str] = Field(default_factory=dict, description="Column-level mappings")
	transformation_logic: Optional[str] = Field(None, max_length=5000, description="Transformation description")
	confidence_score: float = Field(0.0, ge=0.0, le=1.0, description="Confidence in relationship")
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class MetaClassificationCreate(BaseModel):
	"""Data classification creation model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	asset_id: str = Field(..., description="Asset ID")
	classification_type: ClassificationType = Field(..., description="Type of classification")
	column_name: Optional[str] = Field(None, max_length=255, description="Specific column if applicable")
	confidence_score: float = Field(0.0, ge=0.0, le=1.0, description="Classification confidence")
	classification_method: str = Field("manual", max_length=50, description="How was it classified")
	reviewer: Optional[str] = Field(None, max_length=100, description="Reviewer identifier")
	review_notes: Optional[str] = Field(None, max_length=1000, description="Review notes")


class MetaQualityAssessment(BaseModel):
	"""Data quality assessment model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	asset_id: str = Field(..., description="Asset ID")
	tenant_id: str = Field(..., description="Tenant ID")
	overall_score: float = Field(..., ge=0.0, le=100.0, description="Overall quality score")
	
	# Quality dimension scores
	completeness_score: float = Field(..., ge=0.0, le=100.0)
	accuracy_score: float = Field(..., ge=0.0, le=100.0)
	consistency_score: float = Field(..., ge=0.0, le=100.0)
	validity_score: float = Field(..., ge=0.0, le=100.0)
	uniqueness_score: float = Field(..., ge=0.0, le=100.0)
	timeliness_score: float = Field(..., ge=0.0, le=100.0)
	
	# Issues and recommendations
	quality_issues: List[Dict[str, Any]] = Field(default_factory=list)
	recommendations: List[str] = Field(default_factory=list)
	assessment_timestamp: datetime = Field(default_factory=datetime.utcnow)
	assessment_method: str = Field("automated", max_length=50)


# SQLAlchemy Database Models

class MetaAsset(Base):
	"""Metadata Asset - Core asset registry with multi-tenant isolation"""
	__tablename__ = 'meta_assets'
	
	# Primary identifier using UUID7 for time-ordered UUIDs
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Basic asset information
	name = Column(String(255), nullable=False)
	display_name = Column(String(255))
	description = Column(Text)
	asset_type = Column(String(50), nullable=False, index=True)
	source_system = Column(String(100), nullable=False, index=True)
	source_system_type = Column(String(50), nullable=False)
	status = Column(String(20), nullable=False, default='active', index=True)
	
	# External identifiers and references
	external_id = Column(String(255), index=True)
	source_uri = Column(String(1000))
	parent_asset_id = Column(String(36), ForeignKey('meta_assets.id'), index=True)
	
	# Schema and structure information
	schema_info = Column(JSONB)
	column_count = Column(Integer)
	row_count = Column(Integer)
	size_bytes = Column(Integer)
	schema_version = Column(String(50))
	schema_hash = Column(String(64), index=True)
	
	# Business context
	business_domain = Column(String(100), index=True)
	business_owner = Column(String(100))
	technical_owner = Column(String(100))
	data_steward = Column(String(100))
	criticality_level = Column(String(20), default='medium')
	
	# Classification and governance
	classifications = Column(JSONB, default=list)
	tags = Column(JSONB, default=list)
	custom_attributes = Column(JSONB, default=dict)
	governance_status = Column(String(50), default='unreviewed')
	
	# Quality and usage metrics
	quality_score = Column(Float, default=0.0)
	usage_score = Column(Float, default=0.0)
	popularity_rank = Column(Integer)
	last_accessed = Column(DateTime)
	access_count = Column(Integer, default=0)
	
	# Lineage summary (for performance)
	upstream_count = Column(Integer, default=0)
	downstream_count = Column(Integer, default=0)
	lineage_depth = Column(Integer, default=0)
	
	# AI/ML metadata
	contains_pii = Column(Boolean, default=False)
	contains_phi = Column(Boolean, default=False)
	ml_model_type = Column(String(50))
	ml_framework = Column(String(50))
	
	# Lifecycle and auditing
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
	created_by = Column(String(100))
	updated_by = Column(String(100))
	version = Column(Integer, default=1)
	is_deleted = Column(Boolean, default=False, index=True)
	deleted_at = Column(DateTime)
	
	# Relationships
	parent = relationship("MetaAsset", remote_side=[id], backref="children")
	lineage_source = relationship("MetaLineage", foreign_keys="MetaLineage.source_asset_id", backref="source_asset")
	lineage_target = relationship("MetaLineage", foreign_keys="MetaLineage.target_asset_id", backref="target_asset")
	classifications_rel = relationship("MetaClassification", backref="asset")
	quality_assessments = relationship("MetaQualityAssessment", backref="asset")
	
	# Indexes for performance
	__table_args__ = (
		Index('ix_meta_assets_tenant_type', 'tenant_id', 'asset_type'),
		Index('ix_meta_assets_tenant_status', 'tenant_id', 'status'),
		Index('ix_meta_assets_tenant_domain', 'tenant_id', 'business_domain'),
		Index('ix_meta_assets_quality_score', 'quality_score'),
		Index('ix_meta_assets_usage_score', 'usage_score'),
		Index('ix_meta_assets_created_at', 'created_at'),
		Index('ix_meta_assets_updated_at', 'updated_at'),
		CheckConstraint('quality_score >= 0 AND quality_score <= 100', name='check_quality_score_range'),
		CheckConstraint('usage_score >= 0 AND usage_score <= 100', name='check_usage_score_range'),
	)


class MetaAssetVersion(Base):
	"""Asset version history for tracking changes over time"""
	__tablename__ = 'meta_asset_versions'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	asset_id = Column(String(36), ForeignKey('meta_assets.id'), nullable=False, index=True)
	version_number = Column(Integer, nullable=False)
	
	# Snapshot of asset data at this version
	asset_data = Column(JSONB, nullable=False)
	schema_diff = Column(JSONB)
	change_summary = Column(Text)
	change_type = Column(String(50))  # schema, metadata, classification, etc.
	
	# Change tracking
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	created_by = Column(String(100))
	change_reason = Column(Text)
	
	# Relationships
	asset = relationship("MetaAsset", backref="versions")
	
	__table_args__ = (
		Index('ix_meta_asset_versions_asset_version', 'asset_id', 'version_number'),
		Index('ix_meta_asset_versions_tenant_created', 'tenant_id', 'created_at'),
	)


class MetaLineage(Base):
	"""Data lineage relationships between assets"""
	__tablename__ = 'meta_lineage'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	source_asset_id = Column(String(36), ForeignKey('meta_assets.id'), nullable=False, index=True)
	target_asset_id = Column(String(36), ForeignKey('meta_assets.id'), nullable=False, index=True)
	
	# Lineage details
	relationship_type = Column(String(50), nullable=False)
	column_mappings = Column(JSONB, default=dict)
	transformation_logic = Column(Text)
	transformation_type = Column(String(100))
	
	# Confidence and validation
	confidence_score = Column(Float, default=0.0)
	is_validated = Column(Boolean, default=False)
	validation_method = Column(String(50))
	
	# Processing and performance metadata
	processing_engine = Column(String(100))
	execution_frequency = Column(String(50))
	last_execution = Column(DateTime)
	execution_duration_ms = Column(Integer)
	
	# Additional metadata
	metadata = Column(JSONB, default=dict)
	tags = Column(JSONB, default=list)
	
	# Lifecycle
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	created_by = Column(String(100))
	is_active = Column(Boolean, default=True, index=True)
	
	__table_args__ = (
		Index('ix_meta_lineage_source_target', 'source_asset_id', 'target_asset_id'),
		Index('ix_meta_lineage_tenant_active', 'tenant_id', 'is_active'),
		Index('ix_meta_lineage_relationship_type', 'relationship_type'),
		CheckConstraint('confidence_score >= 0 AND confidence_score <= 1', name='check_confidence_range'),
	)


class MetaClassification(Base):
	"""Data classification and sensitivity labels"""
	__tablename__ = 'meta_classifications'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	asset_id = Column(String(36), ForeignKey('meta_assets.id'), nullable=False, index=True)
	
	# Classification details
	classification_type = Column(String(50), nullable=False, index=True)
	column_name = Column(String(255), index=True)  # null means entire asset
	classification_value = Column(String(100))
	sensitivity_level = Column(String(20), default='medium')
	
	# Confidence and review
	confidence_score = Column(Float, default=0.0)
	classification_method = Column(String(50), default='manual')
	auto_detected = Column(Boolean, default=False)
	
	# Review and approval workflow
	status = Column(String(20), default='pending')  # pending, approved, rejected
	reviewer = Column(String(100))
	reviewed_at = Column(DateTime)
	review_notes = Column(Text)
	
	# Pattern and rule information
	detection_pattern = Column(String(500))
	rule_id = Column(String(100))
	rule_version = Column(String(20))
	
	# Additional metadata
	metadata = Column(JSONB, default=dict)
	
	# Lifecycle
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	created_by = Column(String(100))
	
	__table_args__ = (
		Index('ix_meta_classifications_asset_type', 'asset_id', 'classification_type'),
		Index('ix_meta_classifications_tenant_type', 'tenant_id', 'classification_type'),
		Index('ix_meta_classifications_status', 'status'),
		CheckConstraint('confidence_score >= 0 AND confidence_score <= 1', name='check_classification_confidence_range'),
	)


class MetaQualityAssessment(Base):
	"""Data quality assessment results and metrics"""
	__tablename__ = 'meta_quality_assessments'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	asset_id = Column(String(36), ForeignKey('meta_assets.id'), nullable=False, index=True)
	
	# Overall quality metrics
	overall_score = Column(Float, nullable=False, index=True)
	grade = Column(String(2))  # A+, A, B+, B, C+, C, D+, D, F
	
	# Quality dimension scores
	completeness_score = Column(Float, default=0.0)
	accuracy_score = Column(Float, default=0.0)
	consistency_score = Column(Float, default=0.0)
	validity_score = Column(Float, default=0.0)
	uniqueness_score = Column(Float, default=0.0)
	timeliness_score = Column(Float, default=0.0)
	conformity_score = Column(Float, default=0.0)
	integrity_score = Column(Float, default=0.0)
	
	# Quality issues and findings
	quality_issues = Column(JSONB, default=list)
	issue_count = Column(Integer, default=0)
	critical_issue_count = Column(Integer, default=0)
	warning_count = Column(Integer, default=0)
	
	# Recommendations and actions
	recommendations = Column(JSONB, default=list)
	auto_fix_suggestions = Column(JSONB, default=list)
	remediation_priority = Column(String(20), default='medium')
	
	# Assessment metadata
	assessment_method = Column(String(50), default='automated')
	assessment_scope = Column(String(100))  # full, sample, incremental
	sample_size = Column(Integer)
	assessment_duration_ms = Column(Integer)
	
	# Rules and thresholds applied
	rules_applied = Column(JSONB, default=list)
	thresholds_used = Column(JSONB, default=dict)
	
	# Comparison with previous assessments
	previous_score = Column(Float)
	score_change = Column(Float)
	trend = Column(String(20))  # improving, declining, stable
	
	# Lifecycle
	assessment_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
	created_by = Column(String(100))
	
	__table_args__ = (
		Index('ix_meta_quality_assessments_asset_timestamp', 'asset_id', 'assessment_timestamp'),
		Index('ix_meta_quality_assessments_tenant_score', 'tenant_id', 'overall_score'),
		Index('ix_meta_quality_assessments_grade', 'grade'),
		CheckConstraint('overall_score >= 0 AND overall_score <= 100', name='check_overall_score_range'),
	)


class MetaGovernancePolicy(Base):
	"""Data governance policies and rules"""
	__tablename__ = 'meta_governance_policies'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Policy definition
	name = Column(String(255), nullable=False)
	description = Column(Text)
	policy_type = Column(String(50), nullable=False, index=True)
	category = Column(String(100), index=True)
	
	# Policy rules and conditions
	conditions = Column(JSONB, nullable=False)
	actions = Column(JSONB, nullable=False)
	severity = Column(String(20), default='medium')
	
	# Scope and applicability
	asset_types = Column(JSONB, default=list)
	business_domains = Column(JSONB, default=list)
	classifications = Column(JSONB, default=list)
	
	# Policy status and lifecycle
	status = Column(String(20), default='draft', index=True)  # draft, active, deprecated
	effective_date = Column(DateTime)
	expiry_date = Column(DateTime)
	version = Column(String(20), default='1.0')
	
	# Enforcement and monitoring
	enforcement_level = Column(String(20), default='warning')  # info, warning, error, blocking
	monitoring_enabled = Column(Boolean, default=True)
	notification_enabled = Column(Boolean, default=True)
	
	# Usage statistics
	violation_count = Column(Integer, default=0)
	last_violation = Column(DateTime)
	application_count = Column(Integer, default=0)
	
	# Lifecycle
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	created_by = Column(String(100))
	updated_by = Column(String(100))
	
	__table_args__ = (
		Index('ix_meta_governance_policies_tenant_type', 'tenant_id', 'policy_type'),
		Index('ix_meta_governance_policies_status', 'status'),
		Index('ix_meta_governance_policies_category', 'category'),
	)


class MetaUserActivity(Base):
	"""User activity tracking for collaboration and usage analytics"""
	__tablename__ = 'meta_user_activities'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	user_id = Column(String(100), nullable=False, index=True)
	asset_id = Column(String(36), ForeignKey('meta_assets.id'), nullable=False, index=True)
	
	# Activity details
	activity_type = Column(String(50), nullable=False, index=True)  # view, search, download, edit, etc.
	activity_details = Column(JSONB, default=dict)
	session_id = Column(String(100))
	
	# Context information
	source_ip = Column(String(45))
	user_agent = Column(Text)
	referrer = Column(String(500))
	
	# Metrics
	duration_ms = Column(Integer)
	bytes_transferred = Column(Integer)
	
	# Timestamp
	timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
	
	__table_args__ = (
		Index('ix_meta_user_activities_user_timestamp', 'user_id', 'timestamp'),
		Index('ix_meta_user_activities_asset_timestamp', 'asset_id', 'timestamp'),
		Index('ix_meta_user_activities_tenant_activity', 'tenant_id', 'activity_type'),
	)


class MetaComment(Base):
	"""User comments and annotations on metadata assets"""
	__tablename__ = 'meta_comments'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	asset_id = Column(String(36), ForeignKey('meta_assets.id'), nullable=False, index=True)
	parent_comment_id = Column(String(36), ForeignKey('meta_comments.id'), index=True)
	
	# Comment content
	content = Column(Text, nullable=False)
	comment_type = Column(String(50), default='general')  # general, issue, suggestion, question
	
	# User information
	author = Column(String(100), nullable=False)
	author_role = Column(String(50))
	
	# Status and moderation
	status = Column(String(20), default='active')  # active, hidden, deleted, flagged
	is_pinned = Column(Boolean, default=False)
	
	# Engagement metrics
	likes_count = Column(Integer, default=0)
	replies_count = Column(Integer, default=0)
	
	# Lifecycle
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	deleted_at = Column(DateTime)
	
	# Relationships
	asset = relationship("MetaAsset", backref="comments")
	parent = relationship("MetaComment", remote_side=[id], backref="replies")
	
	__table_args__ = (
		Index('ix_meta_comments_asset_created', 'asset_id', 'created_at'),
		Index('ix_meta_comments_tenant_status', 'tenant_id', 'status'),
		Index('ix_meta_comments_author', 'author'),
	)


class MetaBookmark(Base):
	"""User bookmarks for saved/favorite metadata assets"""
	__tablename__ = 'meta_bookmarks'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	user_id = Column(String(100), nullable=False, index=True)
	asset_id = Column(String(36), ForeignKey('meta_assets.id'), nullable=False, index=True)
	
	# Bookmark details
	name = Column(String(255))
	notes = Column(Text)
	folder = Column(String(100))
	tags = Column(JSONB, default=list)
	
	# Lifecycle
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	
	__table_args__ = (
		Index('ix_meta_bookmarks_user_created', 'user_id', 'created_at'),
		Index('ix_meta_bookmarks_tenant_user', 'tenant_id', 'user_id'),
	)


class MetaSearchHistory(Base):
	"""Search history and analytics for improving search experience"""
	__tablename__ = 'meta_search_history'
	
	id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	user_id = Column(String(100), index=True)
	session_id = Column(String(100))
	
	# Search details
	query = Column(Text, nullable=False)
	query_type = Column(String(50), default='text')  # text, faceted, semantic
	filters_applied = Column(JSONB, default=dict)
	
	# Results and performance
	results_count = Column(Integer, default=0)
	response_time_ms = Column(Integer)
	clicked_results = Column(JSONB, default=list)
	clicked_position = Column(Integer)
	
	# Search context
	source_page = Column(String(200))
	search_context = Column(String(100))  # browse, recommendation, direct
	
	# Timestamp
	timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
	
	__table_args__ = (
		Index('ix_meta_search_history_user_timestamp', 'user_id', 'timestamp'),
		Index('ix_meta_search_history_tenant_timestamp', 'tenant_id', 'timestamp'),
	)