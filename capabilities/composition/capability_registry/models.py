"""
APG Capability Registry - Data Models

Comprehensive data models for capability metadata, dependencies, compositions,
and orchestration within APG's multi-tenant architecture.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union, Tuple
from enum import Enum
import json

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, Float, JSON, ForeignKey, Index, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str

# APG Base Model Integration
Base = declarative_base()

# =============================================================================
# Enums and Configuration
# =============================================================================

class CRCapabilityStatus(str, Enum):
	"""Capability lifecycle status."""
	DISCOVERED = "discovered"
	REGISTERED = "registered"
	VALIDATED = "validated"
	ACTIVE = "active"
	DEPRECATED = "deprecated"
	RETIRED = "retired"

class CRDependencyType(str, Enum):
	"""Types of capability dependencies."""
	REQUIRED = "required"
	OPTIONAL = "optional"
	RECOMMENDED = "recommended"
	CONFLICTING = "conflicting"
	ENHANCING = "enhancing"

class CRCompositionType(str, Enum):
	"""Types of capability compositions."""
	ERP_ENTERPRISE = "erp_enterprise"
	INDUSTRY_VERTICAL = "industry_vertical"
	DEPARTMENTAL = "departmental"
	MICROSERVICE = "microservice"
	HYBRID = "hybrid"
	CUSTOM = "custom"

class CRVersionConstraint(str, Enum):
	"""Version constraint types."""
	EXACT = "exact"
	MINIMUM = "minimum"
	MAXIMUM = "maximum"
	RANGE = "range"
	COMPATIBLE = "compatible"
	LATEST = "latest"

class CRValidationStatus(str, Enum):
	"""Composition validation status."""
	PENDING = "pending"
	VALIDATING = "validating"
	VALID = "valid"
	INVALID = "invalid"
	WARNING = "warning"
	ERROR = "error"

# =============================================================================
# Core Registry Models
# =============================================================================

class CRCapability(Base):
	"""Central capability metadata registry with APG integration."""
	__tablename__ = 'cr_capabilities'
	__table_args__ = (
		Index('idx_cr_capability_tenant', 'tenant_id'),
		Index('idx_cr_capability_code', 'capability_code'),
		Index('idx_cr_capability_status', 'status'),
		Index('idx_cr_capability_category', 'category'),
		Index('idx_cr_capability_search', 'name', 'description'),
		UniqueConstraint('tenant_id', 'capability_code', name='uq_tenant_capability'),
	)
	
	# Primary identification
	capability_id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	capability_code = Column(String(100), nullable=False, index=True)
	capability_name = Column(String(255), nullable=False)
	
	# Capability metadata
	description = Column(Text)
	version = Column(String(50), nullable=False)
	category = Column(String(100), nullable=False)
	subcategory = Column(String(100))
	priority = Column(Integer, default=1)
	status = Column(String(50), default=CRCapabilityStatus.DISCOVERED, nullable=False)
	
	# APG Integration metadata
	composition_keywords = Column(JSON, default=list)
	provides_services = Column(JSON, default=list)
	data_models = Column(JSON, default=list)
	api_endpoints = Column(JSON, default=list)
	
	# Feature flags
	multi_tenant = Column(Boolean, default=True, nullable=False)
	audit_enabled = Column(Boolean, default=True, nullable=False)
	security_integration = Column(Boolean, default=True, nullable=False)
	performance_optimized = Column(Boolean, default=False)
	ai_enhanced = Column(Boolean, default=False)
	
	# Business metadata
	target_users = Column(JSON, default=list)
	business_value = Column(Text)
	use_cases = Column(JSON, default=list)
	industry_focus = Column(JSON, default=list)
	
	# Technical metadata
	file_path = Column(String(500))
	module_path = Column(String(500))
	documentation_path = Column(String(500))
	repository_url = Column(String(500))
	
	# Performance and analytics
	complexity_score = Column(Float, default=1.0)
	quality_score = Column(Float, default=0.0)
	popularity_score = Column(Float, default=0.0)
	usage_count = Column(Integer, default=0)
	
	# APG Audit fields
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	created_by = Column(String(36), nullable=False)
	updated_by = Column(String(36))
	
	# Additional metadata
	metadata = Column(JSON, default=dict)
	
	# Relationships
	dependencies = relationship("CRDependency", foreign_keys="CRDependency.capability_id", back_populates="capability")
	dependent_on = relationship("CRDependency", foreign_keys="CRDependency.depends_on_id", back_populates="depends_on_capability")
	compositions = relationship("CRCompositionCapability", back_populates="capability")
	versions = relationship("CRVersion", back_populates="capability", order_by="CRVersion.version_number.desc()")

class CRDependency(Base):
	"""Capability dependency relationships with version constraints."""
	__tablename__ = 'cr_dependencies'
	__table_args__ = (
		Index('idx_cr_dependency_capability', 'capability_id'),
		Index('idx_cr_dependency_depends_on', 'depends_on_id'),
		Index('idx_cr_dependency_type', 'dependency_type'),
		UniqueConstraint('capability_id', 'depends_on_id', name='uq_capability_dependency'),
	)
	
	# Primary identification
	dependency_id = Column(String(36), primary_key=True, default=uuid7str)
	capability_id = Column(String(36), ForeignKey('cr_capabilities.capability_id'), nullable=False)
	depends_on_id = Column(String(36), ForeignKey('cr_capabilities.capability_id'), nullable=False)
	
	# Dependency metadata
	dependency_type = Column(String(50), default=CRDependencyType.REQUIRED, nullable=False)
	version_constraint = Column(String(50), default=CRVersionConstraint.LATEST)
	version_min = Column(String(50))
	version_max = Column(String(50))
	version_exact = Column(String(50))
	
	# Dependency configuration
	load_priority = Column(Integer, default=1)
	initialization_order = Column(Integer, default=1)
	optional_features = Column(JSON, default=list)
	
	# Validation and conflict resolution
	conflict_resolution = Column(String(100))
	alternative_capabilities = Column(JSON, default=list)
	fallback_strategy = Column(String(100))
	
	# APG Audit fields
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	created_by = Column(String(36), nullable=False)
	
	# Additional metadata
	metadata = Column(JSON, default=dict)
	
	# Relationships
	capability = relationship("CRCapability", foreign_keys=[capability_id], back_populates="dependencies")
	depends_on_capability = relationship("CRCapability", foreign_keys=[depends_on_id], back_populates="dependent_on")

class CRComposition(Base):
	"""Saved capability compositions and templates."""
	__tablename__ = 'cr_compositions'
	__table_args__ = (
		Index('idx_cr_composition_tenant', 'tenant_id'),
		Index('idx_cr_composition_type', 'composition_type'),
		Index('idx_cr_composition_status', 'validation_status'),
		Index('idx_cr_composition_created', 'created_at'),
	)
	
	# Primary identification
	composition_id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	name = Column(String(255), nullable=False)
	description = Column(Text)
	
	# Composition metadata
	composition_type = Column(String(50), default=CRCompositionType.CUSTOM, nullable=False)
	version = Column(String(50), default="1.0.0", nullable=False)
	industry_template = Column(String(100))
	deployment_strategy = Column(String(100))
	
	# Validation and status
	validation_status = Column(String(50), default=CRValidationStatus.PENDING, nullable=False)
	validation_results = Column(JSON, default=dict)
	validation_errors = Column(JSON, default=list)
	validation_warnings = Column(JSON, default=list)
	
	# Composition configuration
	configuration = Column(JSON, default=dict)
	environment_settings = Column(JSON, default=dict)
	deployment_config = Column(JSON, default=dict)
	
	# Performance and analytics
	estimated_complexity = Column(Float, default=1.0)
	estimated_cost = Column(Float, default=0.0)
	estimated_deployment_time = Column(String(50))
	performance_metrics = Column(JSON, default=dict)
	
	# Business metadata
	business_requirements = Column(JSON, default=list)
	compliance_requirements = Column(JSON, default=list)
	target_users = Column(JSON, default=list)
	
	# Sharing and collaboration
	is_template = Column(Boolean, default=False)
	is_public = Column(Boolean, default=False)
	shared_with_tenants = Column(JSON, default=list)
	
	# APG Audit fields
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	created_by = Column(String(36), nullable=False)
	updated_by = Column(String(36))
	
	# Additional metadata
	metadata = Column(JSON, default=dict)
	
	# Relationships
	capabilities = relationship("CRCompositionCapability", back_populates="composition", cascade="all, delete-orphan")

class CRCompositionCapability(Base):
	"""Many-to-many relationship between compositions and capabilities."""
	__tablename__ = 'cr_composition_capabilities'
	__table_args__ = (
		Index('idx_cr_comp_cap_composition', 'composition_id'),
		Index('idx_cr_comp_cap_capability', 'capability_id'),
		UniqueConstraint('composition_id', 'capability_id', name='uq_composition_capability'),
	)
	
	# Primary identification
	comp_cap_id = Column(String(36), primary_key=True, default=uuid7str)
	composition_id = Column(String(36), ForeignKey('cr_compositions.composition_id'), nullable=False)
	capability_id = Column(String(36), ForeignKey('cr_capabilities.capability_id'), nullable=False)
	
	# Configuration
	version_constraint = Column(String(50), default=CRVersionConstraint.LATEST)
	required = Column(Boolean, default=True, nullable=False)
	load_order = Column(Integer, default=1)
	configuration = Column(JSON, default=dict)
	
	# APG Audit fields
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	created_by = Column(String(36), nullable=False)
	
	# Relationships
	composition = relationship("CRComposition", back_populates="capabilities")
	capability = relationship("CRCapability", back_populates="compositions")

class CRVersion(Base):
	"""Capability version tracking and compatibility matrices."""
	__tablename__ = 'cr_versions'
	__table_args__ = (
		Index('idx_cr_version_capability', 'capability_id'),
		Index('idx_cr_version_number', 'version_number'),
		Index('idx_cr_version_released', 'release_date'),
		UniqueConstraint('capability_id', 'version_number', name='uq_capability_version'),
	)
	
	# Primary identification
	version_id = Column(String(36), primary_key=True, default=uuid7str)
	capability_id = Column(String(36), ForeignKey('cr_capabilities.capability_id'), nullable=False)
	version_number = Column(String(50), nullable=False)
	
	# Version metadata
	major_version = Column(Integer, nullable=False)
	minor_version = Column(Integer, nullable=False)
	patch_version = Column(Integer, nullable=False)
	pre_release = Column(String(50))
	build_metadata = Column(String(100))
	
	# Release information
	release_date = Column(DateTime, nullable=False)
	release_notes = Column(Text)
	breaking_changes = Column(JSON, default=list)
	deprecations = Column(JSON, default=list)
	new_features = Column(JSON, default=list)
	
	# Compatibility information
	compatible_versions = Column(JSON, default=list)
	incompatible_versions = Column(JSON, default=list)
	migration_path = Column(JSON, default=dict)
	upgrade_instructions = Column(Text)
	
	# API compatibility
	api_changes = Column(JSON, default=dict)
	backward_compatible = Column(Boolean, default=True)
	forward_compatible = Column(Boolean, default=False)
	
	# Quality and validation
	quality_score = Column(Float, default=0.0)
	test_coverage = Column(Float, default=0.0)
	documentation_score = Column(Float, default=0.0)
	security_audit_passed = Column(Boolean, default=False)
	
	# Lifecycle status
	status = Column(String(50), default="active")
	end_of_life_date = Column(DateTime)
	support_level = Column(String(50), default="full")
	
	# APG Audit fields
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	created_by = Column(String(36), nullable=False)
	
	# Additional metadata
	metadata = Column(JSON, default=dict)
	
	# Relationships
	capability = relationship("CRCapability", back_populates="versions")

class CRMetadata(Base):
	"""Extended metadata for capabilities with business and technical attributes."""
	__tablename__ = 'cr_metadata'
	__table_args__ = (
		Index('idx_cr_metadata_capability', 'capability_id'),
		Index('idx_cr_metadata_type', 'metadata_type'),
	)
	
	# Primary identification
	metadata_id = Column(String(36), primary_key=True, default=uuid7str)
	capability_id = Column(String(36), ForeignKey('cr_capabilities.capability_id'), nullable=False)
	metadata_type = Column(String(100), nullable=False)
	
	# Metadata content
	metadata_key = Column(String(255), nullable=False)
	metadata_value = Column(Text)
	metadata_json = Column(JSON)
	
	# Metadata properties
	is_searchable = Column(Boolean, default=True)
	is_public = Column(Boolean, default=True)
	data_type = Column(String(50), default="string")
	
	# APG Audit fields
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	created_by = Column(String(36), nullable=False)
	updated_by = Column(String(36))

class CRRegistry(Base):
	"""Central registry configuration and tenant settings."""
	__tablename__ = 'cr_registry'
	__table_args__ = (
		Index('idx_cr_registry_tenant', 'tenant_id'),
	)
	
	# Primary identification
	registry_id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	name = Column(String(255), nullable=False)
	description = Column(Text)
	
	# Registry configuration
	auto_discovery_enabled = Column(Boolean, default=True)
	auto_validation_enabled = Column(Boolean, default=True)
	marketplace_integration = Column(Boolean, default=True)
	ai_recommendations = Column(Boolean, default=True)
	
	# Discovery settings
	discovery_paths = Column(JSON, default=list)
	excluded_paths = Column(JSON, default=list)
	scan_frequency_hours = Column(Integer, default=24)
	last_scan_date = Column(DateTime)
	
	# Validation settings
	validation_rules = Column(JSON, default=dict)
	quality_thresholds = Column(JSON, default=dict)
	compliance_requirements = Column(JSON, default=list)
	
	# Performance settings
	cache_ttl_seconds = Column(Integer, default=3600)
	max_composition_size = Column(Integer, default=50)
	max_dependency_depth = Column(Integer, default=10)
	
	# APG Audit fields
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	created_by = Column(String(36), nullable=False)
	updated_by = Column(String(36))
	
	# Additional metadata
	metadata = Column(JSON, default=dict)

# =============================================================================
# Analytics and Monitoring Models
# =============================================================================

class CRUsageAnalytics(Base):
	"""Capability usage analytics and metrics."""
	__tablename__ = 'cr_usage_analytics'
	__table_args__ = (
		Index('idx_cr_usage_capability', 'capability_id'),
		Index('idx_cr_usage_date', 'usage_date'),
		Index('idx_cr_usage_tenant', 'tenant_id'),
	)
	
	# Primary identification
	usage_id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), nullable=False, index=True)
	capability_id = Column(String(36), ForeignKey('cr_capabilities.capability_id'), nullable=False)
	
	# Usage metrics
	usage_date = Column(DateTime, nullable=False, index=True)
	usage_count = Column(Integer, default=1)
	composition_count = Column(Integer, default=0)
	deployment_count = Column(Integer, default=0)
	error_count = Column(Integer, default=0)
	
	# Performance metrics
	avg_response_time_ms = Column(Float, default=0.0)
	avg_memory_usage_mb = Column(Float, default=0.0)
	avg_cpu_usage_pct = Column(Float, default=0.0)
	
	# User interaction metrics
	unique_users = Column(Integer, default=0)
	total_sessions = Column(Integer, default=0)
	avg_session_duration = Column(Float, default=0.0)
	
	# Additional metadata
	metadata = Column(JSON, default=dict)

class CRHealthMetrics(Base):
	"""Capability health and performance metrics."""
	__tablename__ = 'cr_health_metrics'
	__table_args__ = (
		Index('idx_cr_health_capability', 'capability_id'),
		Index('idx_cr_health_timestamp', 'timestamp'),
	)
	
	# Primary identification
	metric_id = Column(String(36), primary_key=True, default=uuid7str)
	capability_id = Column(String(36), ForeignKey('cr_capabilities.capability_id'), nullable=False)
	timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
	
	# Health metrics
	health_score = Column(Float, default=1.0)
	availability_pct = Column(Float, default=100.0)
	performance_score = Column(Float, default=1.0)
	error_rate_pct = Column(Float, default=0.0)
	
	# Dependency health
	dependency_health_score = Column(Float, default=1.0)
	missing_dependencies = Column(Integer, default=0)
	conflicting_dependencies = Column(Integer, default=0)
	
	# Quality metrics
	documentation_completeness = Column(Float, default=0.0)
	test_coverage_pct = Column(Float, default=0.0)
	code_quality_score = Column(Float, default=0.0)
	security_score = Column(Float, default=0.0)
	
	# Additional metadata
	metadata = Column(JSON, default=dict)

# =============================================================================
# Model Exports
# =============================================================================

__all__ = [
	# Enums
	"CRCapabilityStatus",
	"CRDependencyType",
	"CRCompositionType",
	"CRVersionConstraint", 
	"CRValidationStatus",
	
	# Core Models
	"CRCapability",
	"CRDependency",
	"CRComposition",
	"CRCompositionCapability",
	"CRVersion",
	"CRMetadata",
	"CRRegistry",
	
	# Analytics Models
	"CRUsageAnalytics",
	"CRHealthMetrics",
	
	# Database Base
	"Base"
]