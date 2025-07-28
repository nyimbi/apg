#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APG Sustainability & ESG Management - Data Models

Comprehensive SQLAlchemy models for ESG management with AI-enhanced tracking,
multi-tenant architecture, and real-time analytics integration.

Copyright © 2025 Datacraft - All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional, Union
from uuid import uuid4
from sqlalchemy import (
	Column, String, Integer, Boolean, DateTime, Text, Float, JSON, 
	ForeignKey, Index, UniqueConstraint, CheckConstraint, Numeric, Enum as SQLEnum
)
from sqlalchemy.orm import relationship, validates
from sqlalchemy.ext.hybrid import hybrid_property
from flask_appbuilder import Model
from pydantic import BaseModel, Field, ConfigDict, AfterValidator
from pydantic.types import StringConstraints
from typing_extensions import Annotated
from enum import Enum
import json

from uuid_extensions import uuid7str

class ESGFrameworkType(str, Enum):
	"""Supported ESG reporting frameworks"""
	GRI = "gri"
	SASB = "sasb"
	TCFD = "tcfd"
	CSRD = "csrd"
	CDP = "cdp"
	UN_GLOBAL_COMPACT = "un_global_compact"
	INTEGRATED_REPORTING = "integrated_reporting"
	EU_TAXONOMY = "eu_taxonomy"
	CUSTOM = "custom"

class ESGMetricType(str, Enum):
	"""Types of ESG metrics"""
	ENVIRONMENTAL = "environmental"
	SOCIAL = "social"
	GOVERNANCE = "governance"

class ESGMetricUnit(str, Enum):
	"""Units of measurement for ESG metrics"""
	TONNES_CO2 = "tonnes_co2"
	KWH = "kwh"
	LITERS = "liters"
	PERCENTAGE = "percentage"
	COUNT = "count"
	USD = "usd"
	HOURS = "hours"
	DAYS = "days"
	SCORE = "score"
	RATIO = "ratio"

class ESGTargetStatus(str, Enum):
	"""Status of ESG targets"""
	DRAFT = "draft"
	ACTIVE = "active"
	ON_TRACK = "on_track"
	AT_RISK = "at_risk"
	BEHIND = "behind"
	ACHIEVED = "achieved"
	PAUSED = "paused"
	CANCELLED = "cancelled"

class ESGReportStatus(str, Enum):
	"""Status of ESG reports"""
	DRAFT = "draft"
	IN_REVIEW = "in_review"
	APPROVED = "approved"
	PUBLISHED = "published"
	ARCHIVED = "archived"

class ESGInitiativeStatus(str, Enum):
	"""Status of ESG initiatives"""
	PLANNING = "planning"
	ACTIVE = "active"
	ON_HOLD = "on_hold"
	COMPLETED = "completed"
	CANCELLED = "cancelled"

class ESGRiskLevel(str, Enum):
	"""ESG risk levels"""
	VERY_LOW = "very_low"
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	VERY_HIGH = "very_high"
	CRITICAL = "critical"

# Base audit mixin for all ESG models
class ESGAuditMixin:
	"""Audit trail mixin for all ESG models"""
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
	created_by = Column(String(128), nullable=False)
	updated_by = Column(String(128), nullable=False)
	version = Column(Integer, default=1, nullable=False)
	is_deleted = Column(Boolean, default=False, nullable=False)
	deleted_at = Column(DateTime, nullable=True)
	deleted_by = Column(String(128), nullable=True)
	
	@validates('version')
	def _validate_version(self, key: str, value: int) -> int:
		"""Validate version is positive integer"""
		assert value > 0, "Version must be positive integer"
		return value

# Core ESG Models

class ESGTenant(Model, ESGAuditMixin):
	"""
	Multi-tenant organization model for ESG management with customizable
	frameworks, AI configuration, and stakeholder management.
	"""
	__tablename__ = 'esg_tenants'
	
	id: str = Column(String(36), primary_key=True, default=uuid7str)
	name: str = Column(String(255), nullable=False)
	slug: str = Column(String(128), unique=True, nullable=False, index=True)
	description: str = Column(Text, nullable=True)
	
	# Organization details
	industry: str = Column(String(128), nullable=True)
	headquarters_country: str = Column(String(3), nullable=True)  # ISO 3166-1 alpha-3
	employee_count: Optional[int] = Column(Integer, nullable=True)
	annual_revenue: Optional[Decimal] = Column(Numeric(18, 2), nullable=True)
	
	# ESG configuration
	esg_frameworks: List[str] = Column(JSON, default=list, nullable=False)
	sustainability_goals: Dict[str, Any] = Column(JSON, default=dict, nullable=False)
	
	# AI configuration
	ai_enabled: bool = Column(Boolean, default=True, nullable=False)
	ai_configuration: Dict[str, Any] = Column(JSON, default=dict, nullable=False)
	
	# Settings and preferences
	settings: Dict[str, Any] = Column(JSON, default=dict, nullable=False)
	timezone: str = Column(String(64), default='UTC', nullable=False)
	locale: str = Column(String(10), default='en_US', nullable=False)
	
	# Status and activation
	is_active: bool = Column(Boolean, default=True, nullable=False)
	subscription_tier: str = Column(String(32), default='standard', nullable=False)
	
	# Relationships
	metrics = relationship("ESGMetric", back_populates="tenant", cascade="all, delete-orphan")
	targets = relationship("ESGTarget", back_populates="tenant", cascade="all, delete-orphan")
	reports = relationship("ESGReport", back_populates="tenant", cascade="all, delete-orphan")
	stakeholders = relationship("ESGStakeholder", back_populates="tenant", cascade="all, delete-orphan")
	suppliers = relationship("ESGSupplier", back_populates="tenant", cascade="all, delete-orphan")
	initiatives = relationship("ESGInitiative", back_populates="tenant", cascade="all, delete-orphan")
	
	# Indexes for performance
	__table_args__ = (
		Index('idx_esg_tenant_slug', 'slug'),
		Index('idx_esg_tenant_industry', 'industry'),
		Index('idx_esg_tenant_active', 'is_active'),
		UniqueConstraint('slug', name='uq_esg_tenant_slug'),
		CheckConstraint('employee_count >= 0', name='ck_esg_tenant_employee_count'),
		CheckConstraint('annual_revenue >= 0', name='ck_esg_tenant_revenue')
	)
	
	def _log_tenant_status(self) -> str:
		"""Log tenant status for debugging"""
		return f"ESG Tenant {self.name} ({self.slug}): {self.subscription_tier} - Active: {self.is_active}"
	
	@validates('slug')
	def _validate_slug(self, key: str, value: str) -> str:
		"""Validate slug format"""
		assert value and len(value.strip()) > 0, "Slug cannot be empty"
		assert value.replace('-', '').replace('_', '').isalnum(), "Slug must be alphanumeric with hyphens/underscores"
		return value.lower().strip()
	
	@validates('esg_frameworks')
	def _validate_frameworks(self, key: str, value: List[str]) -> List[str]:
		"""Validate ESG frameworks list"""
		if not value:
			return ['gri']  # Default to GRI
		valid_frameworks = [f.value for f in ESGFrameworkType]
		for framework in value:
			assert framework in valid_frameworks, f"Invalid ESG framework: {framework}"
		return value

class ESGFramework(Model, ESGAuditMixin):
	"""
	ESG reporting framework definitions with metrics, indicators,
	and compliance requirements.
	"""
	__tablename__ = 'esg_frameworks'
	
	id: str = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(36), ForeignKey('esg_tenants.id'), nullable=False)
	
	name: str = Column(String(255), nullable=False)
	code: str = Column(String(32), nullable=False)
	framework_type: ESGFrameworkType = Column(SQLEnum(ESGFrameworkType), nullable=False)
	version: str = Column(String(32), nullable=False)
	
	# Framework details
	description: str = Column(Text, nullable=True)
	official_url: str = Column(String(512), nullable=True)
	effective_date: datetime = Column(DateTime, nullable=True)
	
	# Framework structure
	categories: List[Dict[str, Any]] = Column(JSON, default=list, nullable=False)
	standards: List[Dict[str, Any]] = Column(JSON, default=list, nullable=False)
	indicators: List[Dict[str, Any]] = Column(JSON, default=list, nullable=False)
	
	# Configuration
	is_mandatory: bool = Column(Boolean, default=False, nullable=False)
	is_active: bool = Column(Boolean, default=True, nullable=False)
	
	# Relationships
	tenant = relationship("ESGTenant", back_populates="frameworks")
	metrics = relationship("ESGMetric", back_populates="framework")
	
	__table_args__ = (
		Index('idx_esg_framework_tenant', 'tenant_id'),
		Index('idx_esg_framework_type', 'framework_type'),
		Index('idx_esg_framework_active', 'is_active'),
		UniqueConstraint('tenant_id', 'code', name='uq_esg_framework_tenant_code')
	)

class ESGMetric(Model, ESGAuditMixin):
	"""
	Core ESG metrics with AI-enhanced tracking, real-time data processing,
	and automated trend analysis.
	"""
	__tablename__ = 'esg_metrics'
	
	id: str = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(36), ForeignKey('esg_tenants.id'), nullable=False)
	framework_id: Optional[str] = Column(String(36), ForeignKey('esg_frameworks.id'), nullable=True)
	
	# Metric identification
	name: str = Column(String(255), nullable=False)
	code: str = Column(String(64), nullable=False)
	metric_type: ESGMetricType = Column(SQLEnum(ESGMetricType), nullable=False)
	category: str = Column(String(128), nullable=False)
	subcategory: str = Column(String(128), nullable=True)
	
	# Metric details
	description: str = Column(Text, nullable=True)
	calculation_method: str = Column(Text, nullable=True)
	data_sources: List[str] = Column(JSON, default=list, nullable=False)
	
	# Measurement
	unit: ESGMetricUnit = Column(SQLEnum(ESGMetricUnit), nullable=False)
	current_value: Optional[Decimal] = Column(Numeric(18, 6), nullable=True)
	target_value: Optional[Decimal] = Column(Numeric(18, 6), nullable=True)
	baseline_value: Optional[Decimal] = Column(Numeric(18, 6), nullable=True)
	
	# Time tracking
	measurement_period: str = Column(String(32), nullable=False, default='monthly')
	last_measured: Optional[datetime] = Column(DateTime, nullable=True)
	next_measurement: Optional[datetime] = Column(DateTime, nullable=True)
	
	# AI insights
	ai_predictions: Dict[str, Any] = Column(JSON, default=dict, nullable=False)
	trend_analysis: Dict[str, Any] = Column(JSON, default=dict, nullable=False)
	performance_insights: Dict[str, Any] = Column(JSON, default=dict, nullable=False)
	
	# Configuration
	is_kpi: bool = Column(Boolean, default=False, nullable=False)
	is_public: bool = Column(Boolean, default=False, nullable=False)
	is_automated: bool = Column(Boolean, default=False, nullable=False)
	automation_config: Dict[str, Any] = Column(JSON, default=dict, nullable=False)
	
	# Quality and validation
	data_quality_score: Optional[Decimal] = Column(Numeric(5, 2), nullable=True)
	validation_rules: List[Dict[str, Any]] = Column(JSON, default=list, nullable=False)
	
	# Relationships
	tenant = relationship("ESGTenant", back_populates="metrics")
	framework = relationship("ESGFramework", back_populates="metrics")
	measurements = relationship("ESGMeasurement", back_populates="metric", cascade="all, delete-orphan")
	targets = relationship("ESGTarget", back_populates="metric")
	
	__table_args__ = (
		Index('idx_esg_metric_tenant', 'tenant_id'),
		Index('idx_esg_metric_type', 'metric_type'),
		Index('idx_esg_metric_category', 'category'),
		Index('idx_esg_metric_kpi', 'is_kpi'),
		Index('idx_esg_metric_public', 'is_public'),
		UniqueConstraint('tenant_id', 'code', name='uq_esg_metric_tenant_code'),
		CheckConstraint('current_value >= 0', name='ck_esg_metric_current_value'),
		CheckConstraint('data_quality_score >= 0 AND data_quality_score <= 100', name='ck_esg_metric_quality_score')
	)
	
	def _log_metric_value(self) -> str:
		"""Log current metric value for monitoring"""
		return f"ESG Metric {self.name}: {self.current_value} {self.unit.value}"
	
	@validates('code')
	def _validate_code(self, key: str, value: str) -> str:
		"""Validate metric code format"""
		assert value and len(value.strip()) > 0, "Metric code cannot be empty"
		assert value.replace('_', '').isalnum(), "Metric code must be alphanumeric with underscores"
		return value.upper().strip()
	
	@hybrid_property
	def progress_to_target(self) -> Optional[Decimal]:
		"""Calculate progress towards target as percentage"""
		if not self.current_value or not self.target_value:
			return None
		if self.baseline_value:
			total_change = self.target_value - self.baseline_value
			current_change = self.current_value - self.baseline_value
			if total_change != 0:
				return (current_change / total_change) * 100
		return (self.current_value / self.target_value) * 100

class ESGMeasurement(Model, ESGAuditMixin):
	"""
	Time-series measurements for ESG metrics with automated data quality
	validation and anomaly detection.
	"""
	__tablename__ = 'esg_measurements'
	
	id: str = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(36), ForeignKey('esg_tenants.id'), nullable=False)
	metric_id: str = Column(String(36), ForeignKey('esg_metrics.id'), nullable=False)
	
	# Measurement data
	value: Decimal = Column(Numeric(18, 6), nullable=False)
	measurement_date: datetime = Column(DateTime, nullable=False)
	period_start: datetime = Column(DateTime, nullable=False)
	period_end: datetime = Column(DateTime, nullable=False)
	
	# Data context
	data_source: str = Column(String(128), nullable=False)
	collection_method: str = Column(String(64), nullable=False)
	
	# Quality and validation
	is_validated: bool = Column(Boolean, default=False, nullable=False)
	validation_score: Optional[Decimal] = Column(Numeric(5, 2), nullable=True)
	anomaly_score: Optional[Decimal] = Column(Numeric(5, 2), nullable=True)
	data_quality_flags: List[str] = Column(JSON, default=list, nullable=False)
	
	# Metadata
	metadata: Dict[str, Any] = Column(JSON, default=dict, nullable=False)
	notes: str = Column(Text, nullable=True)
	
	# Approval workflow
	is_approved: bool = Column(Boolean, default=False, nullable=False)
	approved_by: Optional[str] = Column(String(128), nullable=True)
	approved_at: Optional[datetime] = Column(DateTime, nullable=True)
	
	# Relationships
	tenant = relationship("ESGTenant")
	metric = relationship("ESGMetric", back_populates="measurements")
	
	__table_args__ = (
		Index('idx_esg_measurement_tenant', 'tenant_id'),
		Index('idx_esg_measurement_metric', 'metric_id'),
		Index('idx_esg_measurement_date', 'measurement_date'),
		Index('idx_esg_measurement_period', 'period_start', 'period_end'),
		Index('idx_esg_measurement_validated', 'is_validated'),
		Index('idx_esg_measurement_approved', 'is_approved'),
		CheckConstraint('value >= 0', name='ck_esg_measurement_value'),
		CheckConstraint('period_end >= period_start', name='ck_esg_measurement_period'),
		CheckConstraint('validation_score >= 0 AND validation_score <= 100', name='ck_esg_measurement_validation'),
		CheckConstraint('anomaly_score >= 0 AND anomaly_score <= 100', name='ck_esg_measurement_anomaly')
	)
	
	def _log_measurement_quality(self) -> str:
		"""Log measurement quality for monitoring"""
		return f"ESG Measurement {self.id}: Value={self.value}, Quality={self.validation_score}, Anomaly={self.anomaly_score}"

class ESGTarget(Model, ESGAuditMixin):
	"""
	Sustainability targets and goals with AI-powered progress tracking
	and achievement prediction.
	"""
	__tablename__ = 'esg_targets'
	
	id: str = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(36), ForeignKey('esg_tenants.id'), nullable=False)
	metric_id: str = Column(String(36), ForeignKey('esg_metrics.id'), nullable=False)
	
	# Target details
	name: str = Column(String(255), nullable=False)
	description: str = Column(Text, nullable=True)
	
	# Target values
	target_value: Decimal = Column(Numeric(18, 6), nullable=False)
	baseline_value: Optional[Decimal] = Column(Numeric(18, 6), nullable=True)
	current_progress: Optional[Decimal] = Column(Numeric(5, 2), nullable=True)  # Percentage
	
	# Timeline
	start_date: datetime = Column(DateTime, nullable=False)
	target_date: datetime = Column(DateTime, nullable=False)
	review_frequency: str = Column(String(32), default='quarterly', nullable=False)
	last_review_date: Optional[datetime] = Column(DateTime, nullable=True)
	next_review_date: Optional[datetime] = Column(DateTime, nullable=True)
	
	# Status and tracking
	status: ESGTargetStatus = Column(SQLEnum(ESGTargetStatus), default=ESGTargetStatus.ACTIVE, nullable=False)
	priority: str = Column(String(16), default='medium', nullable=False)
	
	# AI insights
	achievement_probability: Optional[Decimal] = Column(Numeric(5, 2), nullable=True)
	predicted_completion_date: Optional[datetime] = Column(DateTime, nullable=True)
	risk_factors: List[str] = Column(JSON, default=list, nullable=False)
	optimization_recommendations: List[Dict[str, Any]] = Column(JSON, default=list, nullable=False)
	
	# Ownership and accountability
	owner_id: str = Column(String(128), nullable=False)
	stakeholders: List[str] = Column(JSON, default=list, nullable=False)
	
	# Configuration
	is_public: bool = Column(Boolean, default=False, nullable=False)
	milestone_tracking: bool = Column(Boolean, default=True, nullable=False)
	automated_reporting: bool = Column(Boolean, default=True, nullable=False)
	
	# Relationships
	tenant = relationship("ESGTenant", back_populates="targets")
	metric = relationship("ESGMetric", back_populates="targets")
	milestones = relationship("ESGMilestone", back_populates="target", cascade="all, delete-orphan")
	
	__table_args__ = (
		Index('idx_esg_target_tenant', 'tenant_id'),
		Index('idx_esg_target_metric', 'metric_id'),
		Index('idx_esg_target_status', 'status'),
		Index('idx_esg_target_date', 'target_date'),
		Index('idx_esg_target_owner', 'owner_id'),
		Index('idx_esg_target_public', 'is_public'),
		CheckConstraint('target_date > start_date', name='ck_esg_target_dates'),
		CheckConstraint('current_progress >= 0 AND current_progress <= 100', name='ck_esg_target_progress'),
		CheckConstraint('achievement_probability >= 0 AND achievement_probability <= 100', name='ck_esg_target_probability')
	)
	
	def _log_target_progress(self) -> str:
		"""Log target progress for monitoring"""
		return f"ESG Target {self.name}: {self.current_progress}% complete, Status: {self.status.value}"

class ESGMilestone(Model, ESGAuditMixin):
	"""
	Target milestones with progress tracking and automated notifications.
	"""
	__tablename__ = 'esg_milestones'
	
	id: str = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(36), ForeignKey('esg_tenants.id'), nullable=False)
	target_id: str = Column(String(36), ForeignKey('esg_targets.id'), nullable=False)
	
	# Milestone details
	name: str = Column(String(255), nullable=False)
	description: str = Column(Text, nullable=True)
	
	# Progress tracking
	milestone_value: Decimal = Column(Numeric(18, 6), nullable=False)
	milestone_date: datetime = Column(DateTime, nullable=False)
	achieved_value: Optional[Decimal] = Column(Numeric(18, 6), nullable=True)
	achieved_date: Optional[datetime] = Column(DateTime, nullable=True)
	
	# Status
	is_achieved: bool = Column(Boolean, default=False, nullable=False)
	is_critical: bool = Column(Boolean, default=False, nullable=False)
	
	# Relationships
	tenant = relationship("ESGTenant")
	target = relationship("ESGTarget", back_populates="milestones")
	
	__table_args__ = (
		Index('idx_esg_milestone_tenant', 'tenant_id'),
		Index('idx_esg_milestone_target', 'target_id'),
		Index('idx_esg_milestone_date', 'milestone_date'),
		Index('idx_esg_milestone_achieved', 'is_achieved'),
		CheckConstraint('milestone_value >= 0', name='ck_esg_milestone_value')
	)

class ESGStakeholder(Model, ESGAuditMixin):
	"""
	Stakeholder management with engagement tracking, communication preferences,
	and AI-powered sentiment analysis.
	"""
	__tablename__ = 'esg_stakeholders'
	
	id: str = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(36), ForeignKey('esg_tenants.id'), nullable=False)
	
	# Stakeholder identity
	name: str = Column(String(255), nullable=False)
	organization: Optional[str] = Column(String(255), nullable=True)
	stakeholder_type: str = Column(String(64), nullable=False)  # investor, employee, customer, community, regulator
	email: Optional[str] = Column(String(255), nullable=True)
	phone: Optional[str] = Column(String(32), nullable=True)
	
	# Geographic and demographic info
	country: Optional[str] = Column(String(3), nullable=True)  # ISO 3166-1 alpha-3
	region: Optional[str] = Column(String(128), nullable=True)
	language_preference: str = Column(String(10), default='en_US', nullable=False)
	
	# Engagement preferences
	communication_preferences: Dict[str, Any] = Column(JSON, default=dict, nullable=False)
	esg_interests: List[str] = Column(JSON, default=list, nullable=False)
	engagement_frequency: str = Column(String(32), default='quarterly', nullable=False)
	
	# Engagement tracking
	engagement_score: Optional[Decimal] = Column(Numeric(5, 2), nullable=True)
	last_engagement: Optional[datetime] = Column(DateTime, nullable=True)
	next_engagement: Optional[datetime] = Column(DateTime, nullable=True)
	total_interactions: int = Column(Integer, default=0, nullable=False)
	
	# AI insights
	sentiment_score: Optional[Decimal] = Column(Numeric(5, 2), nullable=True)
	engagement_insights: Dict[str, Any] = Column(JSON, default=dict, nullable=False)
	influence_score: Optional[Decimal] = Column(Numeric(5, 2), nullable=True)
	
	# Access and permissions
	portal_access: bool = Column(Boolean, default=False, nullable=False)
	data_access_level: str = Column(String(32), default='public', nullable=False)
	
	# Status
	is_active: bool = Column(Boolean, default=True, nullable=False)
	
	# Relationships
	tenant = relationship("ESGTenant", back_populates="stakeholders")
	communications = relationship("ESGCommunication", back_populates="stakeholder", cascade="all, delete-orphan")
	
	__table_args__ = (
		Index('idx_esg_stakeholder_tenant', 'tenant_id'),
		Index('idx_esg_stakeholder_type', 'stakeholder_type'),
		Index('idx_esg_stakeholder_country', 'country'),
		Index('idx_esg_stakeholder_active', 'is_active'),
		Index('idx_esg_stakeholder_portal', 'portal_access'),
		CheckConstraint('engagement_score >= 0 AND engagement_score <= 100', name='ck_esg_stakeholder_engagement'),
		CheckConstraint('sentiment_score >= -100 AND sentiment_score <= 100', name='ck_esg_stakeholder_sentiment'),
		CheckConstraint('influence_score >= 0 AND influence_score <= 100', name='ck_esg_stakeholder_influence'),
		CheckConstraint('total_interactions >= 0', name='ck_esg_stakeholder_interactions')
	)
	
	def _log_stakeholder_engagement(self) -> str:
		"""Log stakeholder engagement for monitoring"""
		return f"ESG Stakeholder {self.name}: Engagement={self.engagement_score}, Sentiment={self.sentiment_score}"

class ESGCommunication(Model, ESGAuditMixin):
	"""
	Stakeholder communications with tracking, personalization, and effectiveness analysis.
	"""
	__tablename__ = 'esg_communications'
	
	id: str = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(36), ForeignKey('esg_tenants.id'), nullable=False)
	stakeholder_id: str = Column(String(36), ForeignKey('esg_stakeholders.id'), nullable=False)
	
	# Communication details
	subject: str = Column(String(255), nullable=False)
	content: str = Column(Text, nullable=False)
	communication_type: str = Column(String(32), nullable=False)  # email, report, meeting, survey
	channel: str = Column(String(32), nullable=False)
	
	# Timing
	sent_at: Optional[datetime] = Column(DateTime, nullable=True)
	delivered_at: Optional[datetime] = Column(DateTime, nullable=True)
	read_at: Optional[datetime] = Column(DateTime, nullable=True)
	responded_at: Optional[datetime] = Column(DateTime, nullable=True)
	
	# Effectiveness tracking
	engagement_metrics: Dict[str, Any] = Column(JSON, default=dict, nullable=False)
	response_sentiment: Optional[Decimal] = Column(Numeric(5, 2), nullable=True)
	effectiveness_score: Optional[Decimal] = Column(Numeric(5, 2), nullable=True)
	
	# Status
	status: str = Column(String(32), default='draft', nullable=False)
	
	# Relationships
	tenant = relationship("ESGTenant")
	stakeholder = relationship("ESGStakeholder", back_populates="communications")
	
	__table_args__ = (
		Index('idx_esg_communication_tenant', 'tenant_id'),
		Index('idx_esg_communication_stakeholder', 'stakeholder_id'),
		Index('idx_esg_communication_type', 'communication_type'),
		Index('idx_esg_communication_sent', 'sent_at'),
		Index('idx_esg_communication_status', 'status'),
		CheckConstraint('response_sentiment >= -100 AND response_sentiment <= 100', name='ck_esg_communication_sentiment'),
		CheckConstraint('effectiveness_score >= 0 AND effectiveness_score <= 100', name='ck_esg_communication_effectiveness')
	)

class ESGSupplier(Model, ESGAuditMixin):
	"""
	Supply chain sustainability tracking with AI-powered ESG scoring
	and collaborative improvement programs.
	"""
	__tablename__ = 'esg_suppliers'
	
	id: str = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(36), ForeignKey('esg_tenants.id'), nullable=False)
	
	# Supplier identity
	name: str = Column(String(255), nullable=False)
	legal_name: Optional[str] = Column(String(255), nullable=True)
	registration_number: Optional[str] = Column(String(64), nullable=True)
	
	# Contact information
	primary_contact: Optional[str] = Column(String(255), nullable=True)
	email: Optional[str] = Column(String(255), nullable=True)
	phone: Optional[str] = Column(String(32), nullable=True)
	
	# Location and industry
	country: str = Column(String(3), nullable=False)  # ISO 3166-1 alpha-3
	address: Optional[str] = Column(Text, nullable=True)
	industry_sector: str = Column(String(128), nullable=False)
	business_size: str = Column(String(32), nullable=True)  # small, medium, large, enterprise
	
	# Business relationship
	relationship_start: datetime = Column(DateTime, nullable=False)
	contract_value: Optional[Decimal] = Column(Numeric(18, 2), nullable=True)
	criticality_level: str = Column(String(16), default='medium', nullable=False)
	
	# ESG scoring
	overall_esg_score: Optional[Decimal] = Column(Numeric(5, 2), nullable=True)
	environmental_score: Optional[Decimal] = Column(Numeric(5, 2), nullable=True)
	social_score: Optional[Decimal] = Column(Numeric(5, 2), nullable=True)
	governance_score: Optional[Decimal] = Column(Numeric(5, 2), nullable=True)
	
	# Risk assessment
	risk_level: ESGRiskLevel = Column(SQLEnum(ESGRiskLevel), default=ESGRiskLevel.MEDIUM, nullable=False)
	risk_factors: List[str] = Column(JSON, default=list, nullable=False)
	last_assessment: Optional[datetime] = Column(DateTime, nullable=True)
	next_assessment: Optional[datetime] = Column(DateTime, nullable=True)
	
	# AI insights
	ai_risk_analysis: Dict[str, Any] = Column(JSON, default=dict, nullable=False)
	improvement_recommendations: List[Dict[str, Any]] = Column(JSON, default=list, nullable=False)
	performance_trends: Dict[str, Any] = Column(JSON, default=dict, nullable=False)
	
	# Collaboration and improvement
	improvement_program_participant: bool = Column(Boolean, default=False, nullable=False)
	sustainability_collaboration: Dict[str, Any] = Column(JSON, default=dict, nullable=False)
	
	# Certifications and compliance
	certifications: List[Dict[str, Any]] = Column(JSON, default=list, nullable=False)
	compliance_status: Dict[str, Any] = Column(JSON, default=dict, nullable=False)
	
	# Status
	is_active: bool = Column(Boolean, default=True, nullable=False)
	
	# Relationships
	tenant = relationship("ESGTenant", back_populates="suppliers")
	assessments = relationship("ESGSupplierAssessment", back_populates="supplier", cascade="all, delete-orphan")
	
	__table_args__ = (
		Index('idx_esg_supplier_tenant', 'tenant_id'),
		Index('idx_esg_supplier_country', 'country'),
		Index('idx_esg_supplier_industry', 'industry_sector'),
		Index('idx_esg_supplier_risk', 'risk_level'),
		Index('idx_esg_supplier_score', 'overall_esg_score'),
		Index('idx_esg_supplier_active', 'is_active'),
		CheckConstraint('overall_esg_score >= 0 AND overall_esg_score <= 100', name='ck_esg_supplier_overall_score'),
		CheckConstraint('environmental_score >= 0 AND environmental_score <= 100', name='ck_esg_supplier_env_score'),
		CheckConstraint('social_score >= 0 AND social_score <= 100', name='ck_esg_supplier_social_score'),
		CheckConstraint('governance_score >= 0 AND governance_score <= 100', name='ck_esg_supplier_gov_score'),
		CheckConstraint('contract_value >= 0', name='ck_esg_supplier_contract_value')
	)
	
	def _log_supplier_risk(self) -> str:
		"""Log supplier risk assessment for monitoring"""
		return f"ESG Supplier {self.name}: Score={self.overall_esg_score}, Risk={self.risk_level.value}"

class ESGSupplierAssessment(Model, ESGAuditMixin):
	"""
	Supplier ESG assessments with questionnaires, scoring, and improvement tracking.
	"""
	__tablename__ = 'esg_supplier_assessments'
	
	id: str = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(36), ForeignKey('esg_tenants.id'), nullable=False)
	supplier_id: str = Column(String(36), ForeignKey('esg_suppliers.id'), nullable=False)
	
	# Assessment details
	assessment_name: str = Column(String(255), nullable=False)
	assessment_type: str = Column(String(64), nullable=False)  # annual, onboarding, incident, audit
	assessment_date: datetime = Column(DateTime, nullable=False)
	assessor: str = Column(String(128), nullable=False)
	
	# Scoring
	questionnaire_responses: Dict[str, Any] = Column(JSON, default=dict, nullable=False)
	calculated_scores: Dict[str, Any] = Column(JSON, default=dict, nullable=False)
	manual_adjustments: Dict[str, Any] = Column(JSON, default=dict, nullable=False)
	
	# Results
	overall_score: Decimal = Column(Numeric(5, 2), nullable=False)
	grade: str = Column(String(8), nullable=False)  # A+, A, B+, B, C+, C, D, F
	risk_rating: ESGRiskLevel = Column(SQLEnum(ESGRiskLevel), nullable=False)
	
	# Findings and recommendations
	strengths: List[str] = Column(JSON, default=list, nullable=False)
	weaknesses: List[str] = Column(JSON, default=list, nullable=False)
	action_items: List[Dict[str, Any]] = Column(JSON, default=list, nullable=False)
	
	# Follow-up
	next_assessment_date: Optional[datetime] = Column(DateTime, nullable=True)
	improvement_plan: Dict[str, Any] = Column(JSON, default=dict, nullable=False)
	
	# Status
	status: str = Column(String(32), default='completed', nullable=False)
	is_approved: bool = Column(Boolean, default=False, nullable=False)
	
	# Relationships
	tenant = relationship("ESGTenant")
	supplier = relationship("ESGSupplier", back_populates="assessments")
	
	__table_args__ = (
		Index('idx_esg_supplier_assess_tenant', 'tenant_id'),
		Index('idx_esg_supplier_assess_supplier', 'supplier_id'),
		Index('idx_esg_supplier_assess_date', 'assessment_date'),
		Index('idx_esg_supplier_assess_score', 'overall_score'),
		Index('idx_esg_supplier_assess_risk', 'risk_rating'),
		CheckConstraint('overall_score >= 0 AND overall_score <= 100', name='ck_esg_supplier_assess_score')
	)

class ESGInitiative(Model, ESGAuditMixin):
	"""
	Sustainability initiatives and projects with progress tracking,
	impact measurement, and ROI analysis.
	"""
	__tablename__ = 'esg_initiatives'
	
	id: str = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(36), ForeignKey('esg_tenants.id'), nullable=False)
	
	# Initiative details
	name: str = Column(String(255), nullable=False)
	description: str = Column(Text, nullable=False)
	category: str = Column(String(128), nullable=False)
	initiative_type: str = Column(String(64), nullable=False)  # project, program, policy, investment
	
	# Objectives and scope
	objectives: List[str] = Column(JSON, default=list, nullable=False)
	target_metrics: List[str] = Column(JSON, default=list, nullable=False)  # Metric IDs
	expected_impact: Dict[str, Any] = Column(JSON, default=dict, nullable=False)
	
	# Timeline and budget
	start_date: datetime = Column(DateTime, nullable=False)
	planned_end_date: datetime = Column(DateTime, nullable=False)
	actual_end_date: Optional[datetime] = Column(DateTime, nullable=True)
	
	budget_allocated: Optional[Decimal] = Column(Numeric(18, 2), nullable=True)
	budget_spent: Optional[Decimal] = Column(Numeric(18, 2), nullable=True)
	budget_remaining: Optional[Decimal] = Column(Numeric(18, 2), nullable=True)
	
	# Progress tracking
	status: ESGInitiativeStatus = Column(SQLEnum(ESGInitiativeStatus), default=ESGInitiativeStatus.PLANNING, nullable=False)
	progress_percentage: Optional[Decimal] = Column(Numeric(5, 2), nullable=True)
	
	# Impact measurement
	measured_impact: Dict[str, Any] = Column(JSON, default=dict, nullable=False)
	impact_metrics: List[Dict[str, Any]] = Column(JSON, default=list, nullable=False)
	roi_calculation: Dict[str, Any] = Column(JSON, default=dict, nullable=False)
	
	# AI insights
	success_probability: Optional[Decimal] = Column(Numeric(5, 2), nullable=True)
	optimization_recommendations: List[Dict[str, Any]] = Column(JSON, default=list, nullable=False)
	risk_analysis: Dict[str, Any] = Column(JSON, default=dict, nullable=False)
	
	# Team and stakeholders
	project_manager: str = Column(String(128), nullable=False)
	team_members: List[str] = Column(JSON, default=list, nullable=False)
	stakeholders: List[str] = Column(JSON, default=list, nullable=False)
	
	# Configuration
	is_flagship: bool = Column(Boolean, default=False, nullable=False)
	is_public: bool = Column(Boolean, default=False, nullable=False)
	
	# Relationships
	tenant = relationship("ESGTenant", back_populates="initiatives")
	
	__table_args__ = (
		Index('idx_esg_initiative_tenant', 'tenant_id'),
		Index('idx_esg_initiative_category', 'category'),
		Index('idx_esg_initiative_status', 'status'),
		Index('idx_esg_initiative_manager', 'project_manager'),
		Index('idx_esg_initiative_flagship', 'is_flagship'),
		Index('idx_esg_initiative_public', 'is_public'),
		CheckConstraint('planned_end_date > start_date', name='ck_esg_initiative_dates'),
		CheckConstraint('progress_percentage >= 0 AND progress_percentage <= 100', name='ck_esg_initiative_progress'),
		CheckConstraint('success_probability >= 0 AND success_probability <= 100', name='ck_esg_initiative_probability'),
		CheckConstraint('budget_allocated >= 0', name='ck_esg_initiative_budget_allocated'),
		CheckConstraint('budget_spent >= 0', name='ck_esg_initiative_budget_spent')
	)
	
	def _log_initiative_progress(self) -> str:
		"""Log initiative progress for monitoring"""
		return f"ESG Initiative {self.name}: {self.progress_percentage}% complete, Status: {self.status.value}"

class ESGReport(Model, ESGAuditMixin):
	"""
	ESG reports with automated generation, regulatory compliance,
	and stakeholder distribution.
	"""
	__tablename__ = 'esg_reports'
	
	id: str = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(36), ForeignKey('esg_tenants.id'), nullable=False)
	
	# Report details
	name: str = Column(String(255), nullable=False)
	report_type: str = Column(String(64), nullable=False)  # annual, quarterly, ad_hoc, regulatory
	framework: ESGFrameworkType = Column(SQLEnum(ESGFrameworkType), nullable=False)
	
	# Reporting period
	period_start: datetime = Column(DateTime, nullable=False)
	period_end: datetime = Column(DateTime, nullable=False)
	reporting_year: int = Column(Integer, nullable=False)
	
	# Content
	executive_summary: str = Column(Text, nullable=True)
	content_sections: Dict[str, Any] = Column(JSON, default=dict, nullable=False)
	data_tables: Dict[str, Any] = Column(JSON, default=dict, nullable=False)
	charts_visualizations: Dict[str, Any] = Column(JSON, default=dict, nullable=False)
	
	# Automation and AI
	auto_generated: bool = Column(Boolean, default=False, nullable=False)
	ai_insights: Dict[str, Any] = Column(JSON, default=dict, nullable=False)
	generation_config: Dict[str, Any] = Column(JSON, default=dict, nullable=False)
	
	# Review and approval
	status: ESGReportStatus = Column(SQLEnum(ESGReportStatus), default=ESGReportStatus.DRAFT, nullable=False)
	reviewer: Optional[str] = Column(String(128), nullable=True)
	approver: Optional[str] = Column(String(128), nullable=True)
	approved_at: Optional[datetime] = Column(DateTime, nullable=True)
	
	# Publication
	published_at: Optional[datetime] = Column(DateTime, nullable=True)
	publication_url: Optional[str] = Column(String(512), nullable=True)
	distribution_list: List[str] = Column(JSON, default=list, nullable=False)
	
	# File management
	file_path: Optional[str] = Column(String(512), nullable=True)
	file_size: Optional[int] = Column(Integer, nullable=True)
	file_format: str = Column(String(16), default='pdf', nullable=False)
	
	# Metrics and analytics
	view_count: int = Column(Integer, default=0, nullable=False)
	download_count: int = Column(Integer, default=0, nullable=False)
	stakeholder_feedback: Dict[str, Any] = Column(JSON, default=dict, nullable=False)
	
	# Relationships
	tenant = relationship("ESGTenant", back_populates="reports")
	
	__table_args__ = (
		Index('idx_esg_report_tenant', 'tenant_id'),
		Index('idx_esg_report_type', 'report_type'),
		Index('idx_esg_report_framework', 'framework'),
		Index('idx_esg_report_year', 'reporting_year'),
		Index('idx_esg_report_status', 'status'),
		Index('idx_esg_report_published', 'published_at'),
		CheckConstraint('period_end > period_start', name='ck_esg_report_period'),
		CheckConstraint('reporting_year >= 2000', name='ck_esg_report_year'),
		CheckConstraint('file_size >= 0', name='ck_esg_report_file_size'),
		CheckConstraint('view_count >= 0', name='ck_esg_report_views'),
		CheckConstraint('download_count >= 0', name='ck_esg_report_downloads')
	)
	
	def _log_report_status(self) -> str:
		"""Log report status for monitoring"""
		return f"ESG Report {self.name}: Status={self.status.value}, Views={self.view_count}, Downloads={self.download_count}"

class ESGRisk(Model, ESGAuditMixin):
	"""
	ESG risks with AI-powered assessment, mitigation tracking,
	and predictive analysis.
	"""
	__tablename__ = 'esg_risks'
	
	id: str = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id: str = Column(String(36), ForeignKey('esg_tenants.id'), nullable=False)
	
	# Risk identification
	name: str = Column(String(255), nullable=False)
	description: str = Column(Text, nullable=False)
	risk_category: ESGMetricType = Column(SQLEnum(ESGMetricType), nullable=False)
	risk_subcategory: str = Column(String(128), nullable=True)
	
	# Risk assessment
	probability: Decimal = Column(Numeric(5, 2), nullable=False)  # 0-100%
	impact_severity: Decimal = Column(Numeric(5, 2), nullable=False)  # 0-100
	risk_score: Decimal = Column(Numeric(5, 2), nullable=False)  # Calculated
	risk_level: ESGRiskLevel = Column(SQLEnum(ESGRiskLevel), nullable=False)
	
	# Time horizon
	time_horizon: str = Column(String(32), nullable=False)  # short_term, medium_term, long_term
	emerging_risk: bool = Column(Boolean, default=False, nullable=False)
	
	# Business impact
	financial_impact: Optional[Decimal] = Column(Numeric(18, 2), nullable=True)
	operational_impact: Dict[str, Any] = Column(JSON, default=dict, nullable=False)
	reputational_impact: Dict[str, Any] = Column(JSON, default=dict, nullable=False)
	
	# Mitigation
	mitigation_strategies: List[Dict[str, Any]] = Column(JSON, default=list, nullable=False)
	mitigation_status: str = Column(String(32), default='planned', nullable=False)
	mitigation_effectiveness: Optional[Decimal] = Column(Numeric(5, 2), nullable=True)
	
	# AI insights
	ai_risk_analysis: Dict[str, Any] = Column(JSON, default=dict, nullable=False)
	trend_analysis: Dict[str, Any] = Column(JSON, default=dict, nullable=False)
	predictive_indicators: List[str] = Column(JSON, default=list, nullable=False)
	
	# Ownership and monitoring
	risk_owner: str = Column(String(128), nullable=False)
	last_review: Optional[datetime] = Column(DateTime, nullable=True)
	next_review: datetime = Column(DateTime, nullable=False)
	
	# Status
	is_active: bool = Column(Boolean, default=True, nullable=False)
	
	# Relationships
	tenant = relationship("ESGTenant")
	
	__table_args__ = (
		Index('idx_esg_risk_tenant', 'tenant_id'),
		Index('idx_esg_risk_category', 'risk_category'),
		Index('idx_esg_risk_level', 'risk_level'),
		Index('idx_esg_risk_score', 'risk_score'),
		Index('idx_esg_risk_owner', 'risk_owner'),
		Index('idx_esg_risk_active', 'is_active'),
		CheckConstraint('probability >= 0 AND probability <= 100', name='ck_esg_risk_probability'),
		CheckConstraint('impact_severity >= 0 AND impact_severity <= 100', name='ck_esg_risk_impact'),
		CheckConstraint('risk_score >= 0 AND risk_score <= 100', name='ck_esg_risk_score'),
		CheckConstraint('mitigation_effectiveness >= 0 AND mitigation_effectiveness <= 100', name='ck_esg_risk_mitigation'),
		CheckConstraint('financial_impact >= 0', name='ck_esg_risk_financial')
	)
	
	def _log_risk_level(self) -> str:
		"""Log risk level for monitoring"""
		return f"ESG Risk {self.name}: Level={self.risk_level.value}, Score={self.risk_score}"

# Pydantic Models for API Serialization (placed in views.py as per APG patterns)

class ESGTenantView(BaseModel):
	"""Pydantic model for ESG tenant API serialization"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str
	name: str
	slug: Annotated[str, StringConstraints(pattern=r'^[a-z0-9-_]+$')]
	description: Optional[str] = None
	industry: Optional[str] = None
	headquarters_country: Optional[str] = None
	employee_count: Optional[int] = Field(None, ge=0)
	annual_revenue: Optional[Decimal] = Field(None, ge=0)
	esg_frameworks: List[str] = Field(default_factory=list)
	ai_enabled: bool = True
	is_active: bool = True
	subscription_tier: str = 'standard'
	created_at: datetime
	updated_at: datetime

class ESGMetricView(BaseModel):
	"""Pydantic model for ESG metric API serialization"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str
	tenant_id: str
	name: str
	code: Annotated[str, StringConstraints(pattern=r'^[A-Z0-9_]+$')]
	metric_type: ESGMetricType
	category: str
	subcategory: Optional[str] = None
	description: Optional[str] = None
	unit: ESGMetricUnit
	current_value: Optional[Decimal] = Field(None, ge=0)
	target_value: Optional[Decimal] = Field(None, ge=0)
	baseline_value: Optional[Decimal] = Field(None, ge=0)
	is_kpi: bool = False
	is_public: bool = False
	is_automated: bool = False
	data_quality_score: Optional[Decimal] = Field(None, ge=0, le=100)
	created_at: datetime
	updated_at: datetime

class ESGTargetView(BaseModel):
	"""Pydantic model for ESG target API serialization"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str
	tenant_id: str
	metric_id: str
	name: str
	description: Optional[str] = None
	target_value: Decimal = Field(ge=0)
	baseline_value: Optional[Decimal] = Field(None, ge=0)
	current_progress: Optional[Decimal] = Field(None, ge=0, le=100)
	start_date: datetime
	target_date: datetime
	status: ESGTargetStatus
	priority: str = 'medium'
	achievement_probability: Optional[Decimal] = Field(None, ge=0, le=100)
	owner_id: str
	is_public: bool = False
	created_at: datetime
	updated_at: datetime
	
	@AfterValidator
	def validate_dates(cls, v):
		"""Validate target date is after start date"""
		if hasattr(v, 'target_date') and hasattr(v, 'start_date'):
			assert v.target_date > v.start_date, "Target date must be after start date"
		return v

class ESGStakeholderView(BaseModel):
	"""Pydantic model for ESG stakeholder API serialization"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str
	tenant_id: str
	name: str
	organization: Optional[str] = None
	stakeholder_type: str
	email: Optional[str] = None
	country: Optional[str] = None
	language_preference: str = 'en_US'
	engagement_score: Optional[Decimal] = Field(None, ge=0, le=100)
	sentiment_score: Optional[Decimal] = Field(None, ge=-100, le=100)
	influence_score: Optional[Decimal] = Field(None, ge=0, le=100)
	portal_access: bool = False
	is_active: bool = True
	created_at: datetime
	updated_at: datetime

class ESGSupplierView(BaseModel):
	"""Pydantic model for ESG supplier API serialization"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str
	tenant_id: str
	name: str
	country: str
	industry_sector: str
	relationship_start: datetime
	overall_esg_score: Optional[Decimal] = Field(None, ge=0, le=100)
	environmental_score: Optional[Decimal] = Field(None, ge=0, le=100)
	social_score: Optional[Decimal] = Field(None, ge=0, le=100)
	governance_score: Optional[Decimal] = Field(None, ge=0, le=100)
	risk_level: ESGRiskLevel
	criticality_level: str = 'medium'
	is_active: bool = True
	created_at: datetime
	updated_at: datetime

class ESGInitiativeView(BaseModel):
	"""Pydantic model for ESG initiative API serialization"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str
	tenant_id: str
	name: str
	description: str
	category: str
	initiative_type: str
	start_date: datetime
	planned_end_date: datetime
	actual_end_date: Optional[datetime] = None
	status: ESGInitiativeStatus
	progress_percentage: Optional[Decimal] = Field(None, ge=0, le=100)
	budget_allocated: Optional[Decimal] = Field(None, ge=0)
	budget_spent: Optional[Decimal] = Field(None, ge=0)
	success_probability: Optional[Decimal] = Field(None, ge=0, le=100)
	project_manager: str
	is_flagship: bool = False
	is_public: bool = False
	created_at: datetime
	updated_at: datetime
	
	@AfterValidator
	def validate_dates(cls, v):
		"""Validate planned end date is after start date"""
		if hasattr(v, 'planned_end_date') and hasattr(v, 'start_date'):
			assert v.planned_end_date > v.start_date, "Planned end date must be after start date"
		return v

class ESGReportView(BaseModel):
	"""Pydantic model for ESG report API serialization"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str
	tenant_id: str
	name: str
	report_type: str
	framework: ESGFrameworkType
	period_start: datetime
	period_end: datetime
	reporting_year: int = Field(ge=2000)
	status: ESGReportStatus
	auto_generated: bool = False
	published_at: Optional[datetime] = None
	view_count: int = Field(0, ge=0)
	download_count: int = Field(0, ge=0)
	file_format: str = 'pdf'
	created_at: datetime
	updated_at: datetime
	
	@AfterValidator
	def validate_period(cls, v):
		"""Validate period end is after period start"""
		if hasattr(v, 'period_end') and hasattr(v, 'period_start'):
			assert v.period_end > v.period_start, "Period end must be after period start"
		return v