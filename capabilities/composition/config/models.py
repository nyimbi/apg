"""
APG Central Configuration - Comprehensive Data Models

Revolutionary configuration management with AI-powered optimization,
multi-cloud abstraction, and zero-trust security.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass

from sqlalchemy import (
	Column, String, Text, DateTime, Boolean, Integer, Float, 
	ForeignKey, JSON, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pydantic import BaseModel, Field, validator
from pydantic.config import ConfigDict
from uuid_extensions import uuid7str

Base = declarative_base()


class ConfigurationStatus(str, Enum):
	"""Configuration status enumeration."""
	DRAFT = "draft"
	ACTIVE = "active"
	DEPRECATED = "deprecated"
	ARCHIVED = "archived"


class EnvironmentType(str, Enum):
	"""Environment type enumeration."""
	DEVELOPMENT = "development"
	STAGING = "staging"
	PRODUCTION = "production"
	TEST = "test"
	SANDBOX = "sandbox"


class SecurityLevel(str, Enum):
	"""Security level enumeration."""
	PUBLIC = "public"
	INTERNAL = "internal"
	CONFIDENTIAL = "confidential"
	RESTRICTED = "restricted"
	TOP_SECRET = "top_secret"


class ChangeAction(str, Enum):
	"""Configuration change action enumeration."""
	CREATE = "create"
	UPDATE = "update"
	DELETE = "delete"
	RESTORE = "restore"
	MERGE = "merge"


class RecommendationType(str, Enum):
	"""AI recommendation type enumeration."""
	PERFORMANCE = "performance"
	SECURITY = "security"
	COST = "cost"
	RELIABILITY = "reliability"
	COMPLIANCE = "compliance"


# ==================== Core Configuration Models ====================

class CCConfiguration(Base):
	"""Main configuration entity with hierarchical structure."""
	
	__tablename__ = "cc_configurations"
	
	id: str = Column(String(26), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(26), nullable=False, index=True)
	workspace_id: str = Column(String(26), ForeignKey("cc_workspaces.id"), nullable=False)
	parent_id: Optional[str] = Column(String(26), ForeignKey("cc_configurations.id"), nullable=True)
	
	# Basic properties
	name: str = Column(String(255), nullable=False)
	description: Optional[str] = Column(Text)
	key_path: str = Column(String(1000), nullable=False)  # Hierarchical path like /app/database/redis
	
	# Configuration data
	value: Dict[str, Any] = Column(JSONB, nullable=False, default=dict)
	schema_definition: Optional[Dict[str, Any]] = Column(JSONB)
	default_value: Optional[Dict[str, Any]] = Column(JSONB)
	
	# Metadata
	tags: List[str] = Column(JSONB, default=list)
	metadata: Dict[str, Any] = Column(JSONB, default=dict)
	
	# Status and lifecycle
	status: ConfigurationStatus = Column(String(20), nullable=False, default=ConfigurationStatus.DRAFT)
	version: str = Column(String(50), nullable=False, default="1.0.0")
	security_level: SecurityLevel = Column(String(20), nullable=False, default=SecurityLevel.INTERNAL)
	
	# Timestamps
	created_at: datetime = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
	expires_at: Optional[datetime] = Column(DateTime(timezone=True))
	
	# Relationships
	workspace = relationship("CCWorkspace", back_populates="configurations")
	versions = relationship("CCConfigurationVersion", back_populates="configuration", cascade="all, delete-orphan")
	children = relationship("CCConfiguration", backref="parent", remote_side=[id])
	usage_metrics = relationship("CCUsageMetrics", back_populates="configuration", cascade="all, delete-orphan")
	recommendations = relationship("CCRecommendation", back_populates="configuration", cascade="all, delete-orphan")
	
	# Indexes
	__table_args__ = (
		Index("ix_cc_config_tenant_workspace", "tenant_id", "workspace_id"),
		Index("ix_cc_config_key_path", "key_path"),
		Index("ix_cc_config_status", "status"),
		Index("ix_cc_config_updated", "updated_at"),
		UniqueConstraint("tenant_id", "workspace_id", "key_path", name="uq_cc_config_path"),
	)

	@validates('key_path')
	def validate_key_path(self, key, key_path):
		"""Validate key path format."""
		if not key_path.startswith('/'):
			raise ValueError("Key path must start with /")
		if '//' in key_path:
			raise ValueError("Key path cannot contain empty segments")
		return key_path


class CCConfigurationVersion(Base):
	"""Configuration version control with complete change history."""
	
	__tablename__ = "cc_configuration_versions"
	
	id: str = Column(String(26), primary_key=True, default=uuid7str)
	configuration_id: str = Column(String(26), ForeignKey("cc_configurations.id"), nullable=False)
	
	# Version information
	version: str = Column(String(50), nullable=False)
	previous_version: Optional[str] = Column(String(50))
	change_action: ChangeAction = Column(String(20), nullable=False)
	
	# Change data
	value_before: Optional[Dict[str, Any]] = Column(JSONB)
	value_after: Dict[str, Any] = Column(JSONB, nullable=False)
	diff: Dict[str, Any] = Column(JSONB)  # Structured diff
	
	# Change metadata
	change_summary: Optional[str] = Column(Text)
	change_reason: Optional[str] = Column(Text)
	tags: List[str] = Column(JSONB, default=list)
	
	# Author and approval
	created_by: str = Column(String(26), ForeignKey("cc_users.id"), nullable=False)
	approved_by: Optional[str] = Column(String(26), ForeignKey("cc_users.id"))
	approved_at: Optional[datetime] = Column(DateTime(timezone=True))
	
	# Timestamps
	created_at: datetime = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
	deployed_at: Optional[datetime] = Column(DateTime(timezone=True))
	
	# Relationships
	configuration = relationship("CCConfiguration", back_populates="versions")
	created_user = relationship("CCUser", foreign_keys=[created_by])
	approved_user = relationship("CCUser", foreign_keys=[approved_by])
	
	# Indexes
	__table_args__ = (
		Index("ix_cc_version_config", "configuration_id"),
		Index("ix_cc_version_created", "created_at"),
		UniqueConstraint("configuration_id", "version", name="uq_cc_version"),
	)


class CCTemplate(Base):
	"""Reusable configuration templates with intelligent variables."""
	
	__tablename__ = "cc_templates"
	
	id: str = Column(String(26), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(26), nullable=False, index=True)
	workspace_id: str = Column(String(26), ForeignKey("cc_workspaces.id"), nullable=False)
	
	# Template properties
	name: str = Column(String(255), nullable=False)
	description: Optional[str] = Column(Text)
	category: str = Column(String(100), nullable=False)
	
	# Template data
	template_data: Dict[str, Any] = Column(JSONB, nullable=False)
	variables: Dict[str, Any] = Column(JSONB, default=dict)  # Variable definitions
	schema_definition: Optional[Dict[str, Any]] = Column(JSONB)
	
	# Metadata
	tags: List[str] = Column(JSONB, default=list)
	metadata: Dict[str, Any] = Column(JSONB, default=dict)
	
	# Usage and stats
	usage_count: int = Column(Integer, default=0)
	rating: float = Column(Float, default=0.0)
	
	# Status
	is_public: bool = Column(Boolean, default=False)
	is_verified: bool = Column(Boolean, default=False)
	
	# Timestamps
	created_at: datetime = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
	
	# Relationships
	workspace = relationship("CCWorkspace", back_populates="templates")
	
	# Indexes
	__table_args__ = (
		Index("ix_cc_template_tenant_workspace", "tenant_id", "workspace_id"),
		Index("ix_cc_template_category", "category"),
		Index("ix_cc_template_public", "is_public"),
		Index("ix_cc_template_usage", "usage_count"),
	)


class CCEnvironment(Base):
	"""Environment management with inheritance and overrides."""
	
	__tablename__ = "cc_environments"
	
	id: str = Column(String(26), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(26), nullable=False, index=True)
	workspace_id: str = Column(String(26), ForeignKey("cc_workspaces.id"), nullable=False)
	parent_id: Optional[str] = Column(String(26), ForeignKey("cc_environments.id"))
	
	# Environment properties
	name: str = Column(String(100), nullable=False)
	environment_type: EnvironmentType = Column(String(20), nullable=False)
	description: Optional[str] = Column(Text)
	
	# Configuration overrides
	configuration_overrides: Dict[str, Any] = Column(JSONB, default=dict)
	variable_overrides: Dict[str, Any] = Column(JSONB, default=dict)
	
	# Environment settings
	auto_deploy: bool = Column(Boolean, default=False)
	requires_approval: bool = Column(Boolean, default=True)
	
	# Status
	is_active: bool = Column(Boolean, default=True)
	
	# Timestamps
	created_at: datetime = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
	
	# Relationships
	workspace = relationship("CCWorkspace", back_populates="environments")
	children = relationship("CCEnvironment", backref="parent", remote_side=[id])
	
	# Indexes
	__table_args__ = (
		Index("ix_cc_env_tenant_workspace", "tenant_id", "workspace_id"),
		Index("ix_cc_env_type", "environment_type"),
		UniqueConstraint("tenant_id", "workspace_id", "name", name="uq_cc_env_name"),
	)


class CCWorkspace(Base):
	"""Team collaboration and access control boundaries."""
	
	__tablename__ = "cc_workspaces"
	
	id: str = Column(String(26), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(26), nullable=False, index=True)
	
	# Workspace properties
	name: str = Column(String(100), nullable=False)
	description: Optional[str] = Column(Text)
	slug: str = Column(String(100), nullable=False)
	
	# Settings
	default_environment_id: Optional[str] = Column(String(26))
	settings: Dict[str, Any] = Column(JSONB, default=dict)
	
	# Status
	is_active: bool = Column(Boolean, default=True)
	
	# Timestamps
	created_at: datetime = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
	
	# Relationships
	configurations = relationship("CCConfiguration", back_populates="workspace", cascade="all, delete-orphan")
	templates = relationship("CCTemplate", back_populates="workspace", cascade="all, delete-orphan")
	environments = relationship("CCEnvironment", back_populates="workspace", cascade="all, delete-orphan")
	team_memberships = relationship("CCTeamMembership", back_populates="workspace", cascade="all, delete-orphan")
	
	# Indexes
	__table_args__ = (
		Index("ix_cc_workspace_tenant", "tenant_id"),
		UniqueConstraint("tenant_id", "slug", name="uq_cc_workspace_slug"),
	)


# ==================== Security & Access Models ====================

class CCUser(Base):
	"""User management with advanced role-based permissions."""
	
	__tablename__ = "cc_users"
	
	id: str = Column(String(26), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(26), nullable=False, index=True)
	
	# User properties
	email: str = Column(String(255), nullable=False)
	name: str = Column(String(255), nullable=False)
	username: str = Column(String(100), nullable=False)
	
	# Authentication
	password_hash: Optional[str] = Column(String(255))
	api_key_hash: Optional[str] = Column(String(255))
	
	# Profile
	avatar_url: Optional[str] = Column(String(500))
	timezone: str = Column(String(50), default="UTC")
	preferences: Dict[str, Any] = Column(JSONB, default=dict)
	
	# Status
	is_active: bool = Column(Boolean, default=True)
	is_verified: bool = Column(Boolean, default=False)
	last_login_at: Optional[datetime] = Column(DateTime(timezone=True))
	
	# Timestamps
	created_at: datetime = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
	
	# Relationships
	team_memberships = relationship("CCTeamMembership", back_populates="user", cascade="all, delete-orphan")
	access_controls = relationship("CCAccessControl", back_populates="user", cascade="all, delete-orphan")
	
	# Indexes
	__table_args__ = (
		Index("ix_cc_user_tenant", "tenant_id"),
		Index("ix_cc_user_email", "email"),
		UniqueConstraint("tenant_id", "email", name="uq_cc_user_email"),
		UniqueConstraint("tenant_id", "username", name="uq_cc_user_username"),
	)


class CCTeam(Base):
	"""Team-based access control with hierarchical permissions."""
	
	__tablename__ = "cc_teams"
	
	id: str = Column(String(26), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(26), nullable=False, index=True)
	parent_id: Optional[str] = Column(String(26), ForeignKey("cc_teams.id"))
	
	# Team properties
	name: str = Column(String(100), nullable=False)
	description: Optional[str] = Column(Text)
	
	# Permissions
	permissions: List[str] = Column(JSONB, default=list)
	
	# Status
	is_active: bool = Column(Boolean, default=True)
	
	# Timestamps
	created_at: datetime = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
	
	# Relationships
	children = relationship("CCTeam", backref="parent", remote_side=[id])
	memberships = relationship("CCTeamMembership", back_populates="team", cascade="all, delete-orphan")
	
	# Indexes
	__table_args__ = (
		Index("ix_cc_team_tenant", "tenant_id"),
		UniqueConstraint("tenant_id", "name", name="uq_cc_team_name"),
	)


class CCTeamMembership(Base):
	"""Team membership with role-based permissions."""
	
	__tablename__ = "cc_team_memberships"
	
	id: str = Column(String(26), primary_key=True, default=uuid7str)
	team_id: str = Column(String(26), ForeignKey("cc_teams.id"), nullable=False)
	user_id: str = Column(String(26), ForeignKey("cc_users.id"), nullable=False)
	workspace_id: str = Column(String(26), ForeignKey("cc_workspaces.id"), nullable=False)
	
	# Membership properties
	role: str = Column(String(50), nullable=False, default="member")
	permissions: List[str] = Column(JSONB, default=list)
	
	# Status
	is_active: bool = Column(Boolean, default=True)
	
	# Timestamps
	created_at: datetime = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
	
	# Relationships
	team = relationship("CCTeam", back_populates="memberships")
	user = relationship("CCUser", back_populates="team_memberships")
	workspace = relationship("CCWorkspace", back_populates="team_memberships")
	
	# Indexes
	__table_args__ = (
		Index("ix_cc_membership_team", "team_id"),
		Index("ix_cc_membership_user", "user_id"),
		Index("ix_cc_membership_workspace", "workspace_id"),
		UniqueConstraint("team_id", "user_id", "workspace_id", name="uq_cc_membership"),
	)


class CCAccessControl(Base):
	"""Fine-grained access control with resource-level permissions."""
	
	__tablename__ = "cc_access_controls"
	
	id: str = Column(String(26), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(26), nullable=False, index=True)
	user_id: str = Column(String(26), ForeignKey("cc_users.id"), nullable=False)
	
	# Resource identification
	resource_type: str = Column(String(50), nullable=False)  # configuration, template, environment, etc.
	resource_id: str = Column(String(26), nullable=False)
	resource_path: Optional[str] = Column(String(1000))  # For hierarchical resources
	
	# Permissions
	permissions: List[str] = Column(JSONB, nullable=False)  # read, write, delete, admin
	conditions: Optional[Dict[str, Any]] = Column(JSONB)  # Conditional access rules
	
	# Status and expiration
	is_active: bool = Column(Boolean, default=True)
	expires_at: Optional[datetime] = Column(DateTime(timezone=True))
	
	# Timestamps
	created_at: datetime = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
	
	# Relationships
	user = relationship("CCUser", back_populates="access_controls")
	
	# Indexes
	__table_args__ = (
		Index("ix_cc_access_tenant_user", "tenant_id", "user_id"),
		Index("ix_cc_access_resource", "resource_type", "resource_id"),
		Index("ix_cc_access_expires", "expires_at"),
	)


class CCSecretStore(Base):
	"""Encrypted secrets storage with automatic rotation."""
	
	__tablename__ = "cc_secret_stores"
	
	id: str = Column(String(26), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(26), nullable=False, index=True)
	workspace_id: str = Column(String(26), ForeignKey("cc_workspaces.id"), nullable=False)
	
	# Secret identification
	name: str = Column(String(255), nullable=False)
	key_path: str = Column(String(1000), nullable=False)
	
	# Encrypted data
	encrypted_value: str = Column(Text, nullable=False)
	encryption_key_id: str = Column(String(100), nullable=False)
	encryption_algorithm: str = Column(String(50), nullable=False, default="AES-256-GCM")
	
	# Metadata
	description: Optional[str] = Column(Text)
	tags: List[str] = Column(JSONB, default=list)
	
	# Rotation settings
	auto_rotate: bool = Column(Boolean, default=False)
	rotation_interval_days: Optional[int] = Column(Integer)
	last_rotated_at: Optional[datetime] = Column(DateTime(timezone=True))
	next_rotation_at: Optional[datetime] = Column(DateTime(timezone=True))
	
	# Access tracking
	last_accessed_at: Optional[datetime] = Column(DateTime(timezone=True))
	access_count: int = Column(Integer, default=0)
	
	# Status
	is_active: bool = Column(Boolean, default=True)
	
	# Timestamps
	created_at: datetime = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
	expires_at: Optional[datetime] = Column(DateTime(timezone=True))
	
	# Indexes
	__table_args__ = (
		Index("ix_cc_secret_tenant_workspace", "tenant_id", "workspace_id"),
		Index("ix_cc_secret_key_path", "key_path"),
		Index("ix_cc_secret_rotation", "next_rotation_at"),
		UniqueConstraint("tenant_id", "workspace_id", "key_path", name="uq_cc_secret_path"),
	)


class CCAuditLog(Base):
	"""Immutable audit trail with comprehensive event tracking."""
	
	__tablename__ = "cc_audit_logs"
	
	id: str = Column(String(26), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(26), nullable=False, index=True)
	
	# Event identification
	event_type: str = Column(String(100), nullable=False)
	event_action: str = Column(String(100), nullable=False)
	resource_type: str = Column(String(50), nullable=False)
	resource_id: str = Column(String(26), nullable=False)
	
	# Actor information
	user_id: Optional[str] = Column(String(26))
	user_email: Optional[str] = Column(String(255))
	user_agent: Optional[str] = Column(Text)
	source_ip: Optional[str] = Column(String(45))
	
	# Event data
	event_data: Dict[str, Any] = Column(JSONB)
	changes: Optional[Dict[str, Any]] = Column(JSONB)  # Before/after data
	
	# Context
	session_id: Optional[str] = Column(String(100))
	request_id: Optional[str] = Column(String(100))
	correlation_id: Optional[str] = Column(String(100))
	
	# Compliance
	compliance_frameworks: List[str] = Column(JSONB, default=list)
	retention_until: Optional[datetime] = Column(DateTime(timezone=True))
	
	# Timestamp (immutable)
	created_at: datetime = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
	
	# Indexes
	__table_args__ = (
		Index("ix_cc_audit_tenant_created", "tenant_id", "created_at"),
		Index("ix_cc_audit_resource", "resource_type", "resource_id"),
		Index("ix_cc_audit_user", "user_id"),
		Index("ix_cc_audit_event", "event_type", "event_action"),
		Index("ix_cc_audit_retention", "retention_until"),
	)


# ==================== AI & Analytics Models ====================

class CCUsageMetrics(Base):
	"""Configuration usage patterns and performance metrics."""
	
	__tablename__ = "cc_usage_metrics"
	
	id: str = Column(String(26), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(26), nullable=False, index=True)
	configuration_id: str = Column(String(26), ForeignKey("cc_configurations.id"), nullable=False)
	
	# Metrics data
	access_count: int = Column(Integer, default=0)
	read_count: int = Column(Integer, default=0)
	write_count: int = Column(Integer, default=0)
	error_count: int = Column(Integer, default=0)
	
	# Performance metrics
	avg_response_time_ms: float = Column(Float, default=0.0)
	p95_response_time_ms: float = Column(Float, default=0.0)
	max_response_time_ms: float = Column(Float, default=0.0)
	
	# Usage patterns
	peak_access_hour: Optional[int] = Column(Integer)
	peak_access_day: Optional[int] = Column(Integer)
	usage_trend: Optional[str] = Column(String(20))  # increasing, decreasing, stable
	
	# Geographic distribution
	access_regions: List[str] = Column(JSONB, default=list)
	primary_region: Optional[str] = Column(String(50))
	
	# Time period
	period_start: datetime = Column(DateTime(timezone=True), nullable=False)
	period_end: datetime = Column(DateTime(timezone=True), nullable=False)
	
	# Timestamps
	created_at: datetime = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
	
	# Relationships
	configuration = relationship("CCConfiguration", back_populates="usage_metrics")
	
	# Indexes
	__table_args__ = (
		Index("ix_cc_metrics_tenant_config", "tenant_id", "configuration_id"),
		Index("ix_cc_metrics_period", "period_start", "period_end"),
		Index("ix_cc_metrics_access", "access_count"),
	)


class CCRecommendation(Base):
	"""AI-generated optimization recommendations."""
	
	__tablename__ = "cc_recommendations"
	
	id: str = Column(String(26), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(26), nullable=False, index=True)
	configuration_id: str = Column(String(26), ForeignKey("cc_configurations.id"), nullable=False)
	
	# Recommendation details
	recommendation_type: RecommendationType = Column(String(20), nullable=False)
	title: str = Column(String(255), nullable=False)
	description: Text = Column(Text, nullable=False)
	
	# AI analysis
	ai_model: str = Column(String(100), nullable=False)
	confidence_score: float = Column(Float, nullable=False)
	impact_score: float = Column(Float, nullable=False)
	priority: int = Column(Integer, nullable=False, default=3)  # 1=high, 3=low
	
	# Recommendation data
	current_config: Dict[str, Any] = Column(JSONB)
	recommended_config: Dict[str, Any] = Column(JSONB)
	expected_benefits: Dict[str, Any] = Column(JSONB)
	implementation_steps: List[str] = Column(JSONB, default=list)
	
	# Status
	status: str = Column(String(20), default="pending")  # pending, accepted, rejected, implemented
	
	# User interaction
	accepted_at: Optional[datetime] = Column(DateTime(timezone=True))
	accepted_by: Optional[str] = Column(String(26))
	rejection_reason: Optional[str] = Column(Text)
	
	# Timestamps
	created_at: datetime = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
	expires_at: Optional[datetime] = Column(DateTime(timezone=True))
	
	# Relationships
	configuration = relationship("CCConfiguration", back_populates="recommendations")
	
	# Indexes
	__table_args__ = (
		Index("ix_cc_rec_tenant_config", "tenant_id", "configuration_id"),
		Index("ix_cc_rec_type", "recommendation_type"),
		Index("ix_cc_rec_priority", "priority"),
		Index("ix_cc_rec_status", "status"),
		Index("ix_cc_rec_expires", "expires_at"),
	)


class CCAnomaly(Base):
	"""Detected anomalies with impact analysis."""
	
	__tablename__ = "cc_anomalies"
	
	id: str = Column(String(26), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(26), nullable=False, index=True)
	configuration_id: Optional[str] = Column(String(26), ForeignKey("cc_configurations.id"))
	
	# Anomaly details
	anomaly_type: str = Column(String(100), nullable=False)
	severity: str = Column(String(20), nullable=False)  # low, medium, high, critical
	title: str = Column(String(255), nullable=False)
	description: Text = Column(Text, nullable=False)
	
	# Detection data
	detected_at: datetime = Column(DateTime(timezone=True), nullable=False)
	detection_model: str = Column(String(100), nullable=False)
	confidence_score: float = Column(Float, nullable=False)
	
	# Anomaly data
	baseline_value: Optional[Dict[str, Any]] = Column(JSONB)
	anomalous_value: Dict[str, Any] = Column(JSONB)
	deviation_score: float = Column(Float, nullable=False)
	
	# Impact analysis
	affected_resources: List[str] = Column(JSONB, default=list)
	potential_impact: Dict[str, Any] = Column(JSONB)
	recommended_actions: List[str] = Column(JSONB, default=list)
	
	# Resolution
	status: str = Column(String(20), default="open")  # open, investigating, resolved, false_positive
	resolved_at: Optional[datetime] = Column(DateTime(timezone=True))
	resolved_by: Optional[str] = Column(String(26))
	resolution_notes: Optional[Text] = Column(Text)
	
	# Auto-remediation
	auto_remediated: bool = Column(Boolean, default=False)
	remediation_action: Optional[str] = Column(String(255))
	
	# Timestamps
	created_at: datetime = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
	
	# Indexes
	__table_args__ = (
		Index("ix_cc_anomaly_tenant", "tenant_id"),
		Index("ix_cc_anomaly_detected", "detected_at"),
		Index("ix_cc_anomaly_severity", "severity"),
		Index("ix_cc_anomaly_status", "status"),
	)


# ==================== Pydantic Models for API ====================

class ConfigurationCreate(BaseModel):
	"""Pydantic model for configuration creation."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	name: str = Field(..., min_length=1, max_length=255)
	description: Optional[str] = Field(None, max_length=2000)
	key_path: str = Field(..., regex=r'^/[\w\-/.]+$')
	value: Dict[str, Any] = Field(...)
	schema_definition: Optional[Dict[str, Any]] = None
	default_value: Optional[Dict[str, Any]] = None
	tags: List[str] = Field(default_factory=list)
	metadata: Dict[str, Any] = Field(default_factory=dict)
	security_level: SecurityLevel = SecurityLevel.INTERNAL
	expires_at: Optional[datetime] = None


class ConfigurationUpdate(BaseModel):
	"""Pydantic model for configuration updates."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	name: Optional[str] = Field(None, min_length=1, max_length=255)
	description: Optional[str] = Field(None, max_length=2000)
	value: Optional[Dict[str, Any]] = None
	schema_definition: Optional[Dict[str, Any]] = None
	default_value: Optional[Dict[str, Any]] = None
	tags: Optional[List[str]] = None
	metadata: Optional[Dict[str, Any]] = None
	security_level: Optional[SecurityLevel] = None
	status: Optional[ConfigurationStatus] = None
	expires_at: Optional[datetime] = None


class ConfigurationResponse(BaseModel):
	"""Pydantic model for configuration responses."""
	model_config = ConfigDict(from_attributes=True)
	
	id: str
	tenant_id: str
	workspace_id: str
	parent_id: Optional[str]
	name: str
	description: Optional[str]
	key_path: str
	value: Dict[str, Any]
	schema_definition: Optional[Dict[str, Any]]
	default_value: Optional[Dict[str, Any]]
	tags: List[str]
	metadata: Dict[str, Any]
	status: ConfigurationStatus
	version: str
	security_level: SecurityLevel
	created_at: datetime
	updated_at: datetime
	expires_at: Optional[datetime]


class TemplateCreate(BaseModel):
	"""Pydantic model for template creation."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	name: str = Field(..., min_length=1, max_length=255)
	description: Optional[str] = Field(None, max_length=2000)
	category: str = Field(..., min_length=1, max_length=100)
	template_data: Dict[str, Any] = Field(...)
	variables: Dict[str, Any] = Field(default_factory=dict)
	schema_definition: Optional[Dict[str, Any]] = None
	tags: List[str] = Field(default_factory=list)
	metadata: Dict[str, Any] = Field(default_factory=dict)
	is_public: bool = False


class WorkspaceCreate(BaseModel):
	"""Pydantic model for workspace creation."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	name: str = Field(..., min_length=1, max_length=100)
	description: Optional[str] = Field(None, max_length=2000)
	slug: str = Field(..., regex=r'^[a-z0-9-]+$', min_length=1, max_length=100)
	settings: Dict[str, Any] = Field(default_factory=dict)


class UserCreate(BaseModel):
	"""Pydantic model for user creation."""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
	name: str = Field(..., min_length=1, max_length=255)
	username: str = Field(..., regex=r'^[a-zA-Z0-9_-]+$', min_length=1, max_length=100)
	password: Optional[str] = Field(None, min_length=8)
	timezone: str = Field(default="UTC")
	preferences: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class ConfigurationDiff:
	"""Configuration difference representation."""
	added: Dict[str, Any]
	removed: Dict[str, Any]
	changed: Dict[str, Dict[str, Any]]  # key -> {old: value, new: value}


@dataclass
class AIInsight:
	"""AI-generated insight representation."""
	insight_type: str
	title: str
	description: str
	confidence: float
	impact: str
	recommendations: List[str]
	data: Dict[str, Any]