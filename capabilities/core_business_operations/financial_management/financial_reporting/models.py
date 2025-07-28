"""
APG Financial Reporting Models - Revolutionary AI-Powered Financial Reporting Platform

Enhanced database models for the Financial Reporting capability with revolutionary AI features,
natural language processing, real-time collaboration, and predictive analytics.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero | APG Platform Architect
"""

import asyncio
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from decimal import Decimal
from enum import Enum
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, Date, DECIMAL, ForeignKey, UniqueConstraint, JSON, Index
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str
from pydantic import BaseModel, Field, ConfigDict, AfterValidator
from annotated_types import Annotated
import json

from ...auth_rbac.models import BaseMixin, AuditMixin, Model

# ==================================================================================
# REVOLUTIONARY AI-POWERED ENUMS AND TYPES
# ==================================================================================

class AIModelType(str, Enum):
	"""AI model types for financial reporting intelligence."""
	PREDICTIVE_ANALYTICS = "predictive_analytics"
	NATURAL_LANGUAGE = "natural_language"
	ANOMALY_DETECTION = "anomaly_detection"
	PATTERN_RECOGNITION = "pattern_recognition"
	SENTIMENT_ANALYSIS = "sentiment_analysis"
	FORECASTING = "forecasting"
	OPTIMIZATION = "optimization"

class CollaborationEventType(str, Enum):
	"""Real-time collaboration event types."""
	USER_JOINED = "user_joined"
	USER_LEFT = "user_left"
	CONTENT_EDITED = "content_edited"
	COMMENT_ADDED = "comment_added"
	APPROVAL_REQUESTED = "approval_requested"
	STATUS_CHANGED = "status_changed"
	DATA_UPDATED = "data_updated"

class ReportIntelligenceLevel(str, Enum):
	"""AI intelligence levels for report generation."""
	BASIC = "basic"
	ENHANCED = "enhanced"
	ADVANCED = "advanced"
	REVOLUTIONARY = "revolutionary"

class ConversationalIntentType(str, Enum):
	"""Natural language intent classification."""
	CREATE_REPORT = "create_report"
	ANALYZE_VARIANCE = "analyze_variance"
	GENERATE_INSIGHT = "generate_insight"
	EXPLAIN_METRIC = "explain_metric"
	FORECAST_TREND = "forecast_trend"
	COMPARE_PERIODS = "compare_periods"
	DRILL_DOWN = "drill_down"
	SUMMARIZE_DATA = "summarize_data"

class DataQualityScore(str, Enum):
	"""Data quality scoring levels."""
	EXCELLENT = "excellent"		# 95-100%
	GOOD = "good"				# 85-94%
	FAIR = "fair"				# 70-84%
	POOR = "poor"				# 50-69%
	CRITICAL = "critical"		# Below 50%

# ==================================================================================
# REVOLUTIONARY PYDANTIC MODELS FOR AI INTEGRATION
# ==================================================================================

class ConversationalRequest(BaseModel):
	"""Pydantic model for natural language report requests."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	request_id: str = Field(default_factory=uuid7str)
	user_query: str = Field(min_length=1, max_length=1000)
	intent_type: ConversationalIntentType
	confidence_score: float = Field(ge=0.0, le=1.0)
	extracted_entities: Dict[str, Any] = Field(default_factory=dict)
	context_data: Dict[str, Any] = Field(default_factory=dict)
	response_format: str = Field(default="interactive")
	timestamp: datetime = Field(default_factory=datetime.now)

class AIInsightGeneration(BaseModel):
	"""Pydantic model for AI-generated financial insights."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	insight_id: str = Field(default_factory=uuid7str)
	insight_type: str = Field(min_length=1)
	title: str = Field(min_length=1, max_length=200)
	description: str = Field(min_length=1)
	confidence_level: float = Field(ge=0.0, le=1.0)
	impact_score: float = Field(ge=0.0, le=10.0)
	recommended_actions: List[str] = Field(default_factory=list)
	supporting_data: Dict[str, Any] = Field(default_factory=dict)
	generated_at: datetime = Field(default_factory=datetime.now)

class PredictiveAnalyticsResult(BaseModel):
	"""Pydantic model for predictive analytics results."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	prediction_id: str = Field(default_factory=uuid7str)
	model_type: AIModelType
	prediction_value: float
	confidence_interval: Tuple[float, float]
	accuracy_score: float = Field(ge=0.0, le=1.0)
	prediction_horizon: int = Field(ge=1)  # Days
	contributing_factors: List[str] = Field(default_factory=list)
	risk_indicators: Dict[str, float] = Field(default_factory=dict)
	generated_at: datetime = Field(default_factory=datetime.now)

class CollaborativeSession(BaseModel):
	"""Pydantic model for real-time collaboration sessions."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	session_id: str = Field(default_factory=uuid7str)
	report_id: str
	active_users: List[str] = Field(default_factory=list)
	session_start: datetime = Field(default_factory=datetime.now)
	last_activity: datetime = Field(default_factory=datetime.now)
	collaboration_mode: str = Field(default="real_time")
	conflict_resolution: str = Field(default="intelligent")
	auto_save_enabled: bool = Field(default=True)

class SmartReportConfiguration(BaseModel):
	"""Pydantic model for intelligent report configuration."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	config_id: str = Field(default_factory=uuid7str)
	ai_enhancement_level: ReportIntelligenceLevel = Field(default=ReportIntelligenceLevel.ENHANCED)
	auto_narrative_generation: bool = Field(default=True)
	predictive_insights_enabled: bool = Field(default=True)
	adaptive_formatting: bool = Field(default=True)
	real_time_collaboration: bool = Field(default=True)
	natural_language_interface: bool = Field(default=True)
	voice_activation: bool = Field(default=False)
	mobile_optimization: bool = Field(default=True)
	accessibility_enhanced: bool = Field(default=True)

# ==================================================================================
# ENHANCED SQLALCHEMY MODELS WITH AI CAPABILITIES
# ==================================================================================

class CFRFReportTemplate(Model, AuditMixin, BaseMixin):
	"""
	Revolutionary AI-Powered Financial Report Template.
	
	Enhanced template system with adaptive intelligence, natural language generation,
	and collaborative features that learn and optimize based on usage patterns.
	"""
	__tablename__ = 'cf_fr_report_template'
	
	# Identity
	template_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Template Information
	template_code = Column(String(50), nullable=False, index=True)
	template_name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	
	# Template Classification
	statement_type = Column(String(50), nullable=False)  # balance_sheet, income_statement, cash_flow, equity_changes, custom
	category = Column(String(50), nullable=True)  # standard, regulatory, management, analytical
	format_type = Column(String(50), nullable=False)  # comparative, single_period, trend, variance
	
	# Revolutionary AI Enhancement Features
	ai_intelligence_level = Column(String(20), default='enhanced', index=True)  # basic, enhanced, advanced, revolutionary
	auto_narrative_generation = Column(Boolean, default=True)  # AI-generated explanations
	predictive_insights_enabled = Column(Boolean, default=True)  # ML-powered predictions
	adaptive_formatting = Column(Boolean, default=True)  # Self-optimizing layout
	natural_language_interface = Column(Boolean, default=True)  # Conversational creation
	voice_activation_enabled = Column(Boolean, default=False)  # Voice-driven reporting
	
	# Template Properties
	is_system = Column(Boolean, default=False)  # System templates cannot be deleted
	is_active = Column(Boolean, default=True)
	version = Column(String(20), default='1.0')
	
	# Enhanced Formatting Options
	currency_type = Column(String(20), default='single')  # single, multi, reporting
	show_percentages = Column(Boolean, default=False)
	show_variances = Column(Boolean, default=False)
	decimal_places = Column(Integer, default=2)
	thousands_separator = Column(Boolean, default=True)
	
	# Adaptive Layout Options
	page_orientation = Column(String(20), default='portrait')  # portrait, landscape
	font_size = Column(Integer, default=12)
	include_logo = Column(Boolean, default=True)
	include_header = Column(Boolean, default=True)
	include_footer = Column(Boolean, default=True)
	dynamic_layout_optimization = Column(Boolean, default=True)  # AI layout optimization
	responsive_design_enabled = Column(Boolean, default=True)  # Mobile-responsive
	accessibility_enhanced = Column(Boolean, default=True)  # WCAG 2.1 compliance
	
	# Intelligent Generation Options
	auto_generate = Column(Boolean, default=False)
	generation_frequency = Column(String(20), nullable=True)  # daily, weekly, monthly, quarterly, yearly
	last_generated = Column(DateTime, nullable=True)
	ai_optimization_score = Column(Float, default=0.0)  # AI learning effectiveness
	usage_pattern_data = Column(JSON, nullable=True)  # Learning from user behavior
	
	# Real-Time Collaboration Features
	real_time_collaboration = Column(Boolean, default=True)  # Multi-user editing
	collaborative_session_id = Column(String(36), nullable=True)  # Active session
	conflict_resolution_mode = Column(String(20), default='intelligent')  # AI conflict resolution
	version_control_enabled = Column(Boolean, default=True)  # Advanced versioning
	
	# Advanced AI Configuration
	ai_model_preferences = Column(JSON, nullable=True)  # AI model selection
	natural_language_prompts = Column(JSON, nullable=True)  # Custom AI prompts
	predictive_model_config = Column(JSON, nullable=True)  # ML model settings
	personalization_data = Column(JSON, nullable=True)  # User-specific adaptations
	
	# Performance and Analytics
	generation_performance_metrics = Column(JSON, nullable=True)  # Speed and accuracy
	user_satisfaction_score = Column(Float, default=0.0)  # User feedback
	usage_analytics = Column(JSON, nullable=True)  # Usage patterns and insights
	
	# Metadata
	configuration = Column(JSON, nullable=True)  # Additional configuration options
	
	# Relationships
	report_definitions = relationship("CFRFReportDefinition", back_populates="template")
	report_generations = relationship("CFRFReportGeneration", back_populates="template")
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'template_code', name='uq_template_code_tenant'),
	)
	
	def __repr__(self):
		return f"<CFRFReportTemplate {self.template_name}>"


class CFRFReportDefinition(Model, AuditMixin, BaseMixin):
	"""
	Report structure definitions.
	
	Defines the specific structure, sections, and formatting rules
	for each report template including line definitions and calculations.
	"""
	__tablename__ = 'cf_fr_report_definition'
	
	# Identity
	definition_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	template_id = Column(String(36), ForeignKey('cf_fr_report_template.template_id'), nullable=False)
	
	# Definition Information
	definition_name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	version = Column(String(20), default='1.0')
	
	# Structure Properties
	section_count = Column(Integer, default=1)
	line_count = Column(Integer, default=0)
	
	# Calculation Rules
	calculation_method = Column(String(50), default='standard')  # standard, consolidated, comparative
	balance_check = Column(Boolean, default=True)  # Verify balance sheet balances
	zero_suppression = Column(Boolean, default=False)  # Hide zero balances
	
	# Period Handling
	period_type = Column(String(20), default='monthly')  # daily, weekly, monthly, quarterly, yearly
	periods_to_show = Column(Integer, default=1)  # Number of periods to display
	comparative_periods = Column(Integer, default=0)  # Number of comparative periods
	
	# Consolidation Settings
	consolidation_required = Column(Boolean, default=False)
	elimination_entries = Column(Boolean, default=False)
	currency_translation = Column(Boolean, default=False)
	
	# Approval and Control
	requires_approval = Column(Boolean, default=True)
	approval_workflow = Column(String(100), nullable=True)
	
	# Metadata
	calculation_rules = Column(JSON, nullable=True)  # Complex calculation definitions
	formatting_rules = Column(JSON, nullable=True)  # Formatting specifications
	
	# Relationships
	template = relationship("CFRFReportTemplate", back_populates="report_definitions")
	report_lines = relationship("CFRFReportLine", back_populates="definition", cascade="all, delete-orphan")
	
	def __repr__(self):
		return f"<CFRFReportDefinition {self.definition_name}>"


class CFRFReportLine(Model, AuditMixin, BaseMixin):
	"""
	Individual report line definitions.
	
	Defines each line item in a financial report including account mappings,
	calculations, formatting, and display properties.
	"""
	__tablename__ = 'cf_fr_report_line'
	
	# Identity
	line_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	definition_id = Column(String(36), ForeignKey('cf_fr_report_definition.definition_id'), nullable=False)
	
	# Line Information
	line_code = Column(String(50), nullable=False, index=True)
	line_name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	
	# Hierarchy and Positioning
	parent_line_id = Column(String(36), ForeignKey('cf_fr_report_line.line_id'), nullable=True)
	level = Column(Integer, default=0)
	sort_order = Column(Integer, default=0)
	section_name = Column(String(100), nullable=True)
	
	# Line Type and Behavior
	line_type = Column(String(50), nullable=False)  # detail, subtotal, total, header, spacer
	data_source = Column(String(50), nullable=False)  # accounts, calculation, manual, reference
	calculation_method = Column(String(50), nullable=True)  # sum, average, formula, lookup
	
	# Account Mapping
	account_filter = Column(String(500), nullable=True)  # Account code patterns or ranges
	account_type_filter = Column(String(100), nullable=True)  # Account type restrictions
	include_children = Column(Boolean, default=True)  # Include child accounts
	
	# Calculation Properties
	formula = Column(Text, nullable=True)  # Calculation formula for computed lines
	sign_reversal = Column(Boolean, default=False)  # Reverse sign for display
	absolute_value = Column(Boolean, default=False)  # Show absolute value
	
	# Formatting Properties
	indent_level = Column(Integer, default=0)
	bold = Column(Boolean, default=False)
	italic = Column(Boolean, default=False)
	underline = Column(Boolean, default=False)
	font_size = Column(Integer, nullable=True)
	
	# Display Properties
	show_line = Column(Boolean, default=True)
	show_zero = Column(Boolean, default=True)
	print_line = Column(Boolean, default=True)
	
	# Notes and References
	note_reference = Column(String(10), nullable=True)  # Reference to financial statement notes
	disclosure_reference = Column(String(10), nullable=True)  # Reference to regulatory disclosures
	
	# Metadata
	line_attributes = Column(JSON, nullable=True)  # Additional line properties
	
	# Relationships
	definition = relationship("CFRFReportDefinition", back_populates="report_lines")
	parent_line = relationship("CFRFReportLine", remote_side=[line_id])
	child_lines = relationship("CFRFReportLine", back_populates="parent_line")
	
	def __repr__(self):
		return f"<CFRFReportLine {self.line_name}>"


class CFRFReportPeriod(Model, AuditMixin, BaseMixin):
	"""
	Reporting period definitions.
	
	Defines the time periods for which financial reports are generated,
	including fiscal years, quarters, months, and custom periods.
	"""
	__tablename__ = 'cf_fr_report_period'
	
	# Identity
	period_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Period Information
	period_code = Column(String(50), nullable=False, index=True)
	period_name = Column(String(100), nullable=False)
	description = Column(Text, nullable=True)
	
	# Period Type and Hierarchy
	period_type = Column(String(20), nullable=False)  # fiscal_year, quarter, month, week, day, custom
	fiscal_year = Column(Integer, nullable=False, index=True)
	period_number = Column(Integer, nullable=True)  # Quarter/Month number within fiscal year
	
	# Date Range
	start_date = Column(Date, nullable=False)
	end_date = Column(Date, nullable=False)
	days_in_period = Column(Integer, nullable=False)
	
	# Period Status
	is_current = Column(Boolean, default=False)
	is_closed = Column(Boolean, default=False)
	is_adjusting = Column(Boolean, default=False)  # Allows adjusting entries
	
	# Parent-Child Relationships
	parent_period_id = Column(String(36), ForeignKey('cf_fr_report_period.period_id'), nullable=True)
	
	# Metadata
	period_attributes = Column(JSON, nullable=True)
	
	# Relationships
	parent_period = relationship("CFRFReportPeriod", remote_side=[period_id])
	child_periods = relationship("CFRFReportPeriod", back_populates="parent_period")
	report_generations = relationship("CFRFReportGeneration", back_populates="period")
	financial_statements = relationship("CFRFFinancialStatement", back_populates="period")
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'period_code', name='uq_period_code_tenant'),
	)
	
	def __repr__(self):
		return f"<CFRFReportPeriod {self.period_name}>"


class CFRFReportGeneration(Model, AuditMixin, BaseMixin):
	"""
	Report generation history and status.
	
	Tracks the generation process for financial reports including
	status, parameters, and output information.
	"""
	__tablename__ = 'cf_fr_report_generation'
	
	# Identity
	generation_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	template_id = Column(String(36), ForeignKey('cf_fr_report_template.template_id'), nullable=False)
	period_id = Column(String(36), ForeignKey('cf_fr_report_period.period_id'), nullable=False)
	
	# Generation Information
	generation_name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	generation_type = Column(String(50), default='standard')  # standard, consolidation, restatement
	
	# Generation Status
	status = Column(String(50), default='pending')  # pending, running, completed, failed, cancelled
	progress_percentage = Column(Integer, default=0)
	start_time = Column(DateTime, nullable=True)
	end_time = Column(DateTime, nullable=True)
	duration_seconds = Column(Integer, nullable=True)
	
	# Generation Parameters
	as_of_date = Column(Date, nullable=False)
	include_adjustments = Column(Boolean, default=True)
	consolidation_level = Column(String(50), nullable=True)  # entity, division, corporate
	currency_code = Column(String(10), default='USD')
	
	# Output Information
	output_format = Column(String(20), default='pdf')  # pdf, excel, html, json, xml
	file_path = Column(String(500), nullable=True)
	file_size_bytes = Column(Integer, nullable=True)
	page_count = Column(Integer, nullable=True)
	
	# Quality Control
	balance_verified = Column(Boolean, default=False)
	variance_threshold = Column(DECIMAL(15, 4), nullable=True)
	warning_count = Column(Integer, default=0)
	error_count = Column(Integer, default=0)
	
	# Approval Status
	approval_status = Column(String(50), default='pending')  # pending, approved, rejected
	approved_by = Column(String(36), nullable=True)
	approved_date = Column(DateTime, nullable=True)
	approval_notes = Column(Text, nullable=True)
	
	# Distribution
	distribution_status = Column(String(50), default='pending')  # pending, sent, failed
	distribution_date = Column(DateTime, nullable=True)
	recipient_count = Column(Integer, default=0)
	
	# Metadata
	generation_log = Column(JSON, nullable=True)  # Detailed generation log
	parameters = Column(JSON, nullable=True)  # Generation parameters
	
	# Relationships
	template = relationship("CFRFReportTemplate", back_populates="report_generations")
	period = relationship("CFRFReportPeriod", back_populates="report_generations")
	financial_statements = relationship("CFRFFinancialStatement", back_populates="generation")
	
	def __repr__(self):
		return f"<CFRFReportGeneration {self.generation_name}>"


class CFRFFinancialStatement(Model, AuditMixin, BaseMixin):
	"""
	Generated financial statements.
	
	Stores the actual financial statement data generated from templates
	including line-by-line values and supporting information.
	"""
	__tablename__ = 'cf_fr_financial_statement'
	
	# Identity
	statement_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	generation_id = Column(String(36), ForeignKey('cf_fr_report_generation.generation_id'), nullable=False)
	period_id = Column(String(36), ForeignKey('cf_fr_report_period.period_id'), nullable=False)
	
	# Statement Information
	statement_name = Column(String(200), nullable=False)
	statement_type = Column(String(50), nullable=False)
	description = Column(Text, nullable=True)
	
	# Statement Properties
	as_of_date = Column(Date, nullable=False)
	currency_code = Column(String(10), default='USD')
	reporting_entity = Column(String(200), nullable=True)
	consolidation_level = Column(String(50), nullable=True)
	
	# Statement Status
	is_final = Column(Boolean, default=False)
	is_published = Column(Boolean, default=False)
	version = Column(String(20), default='1.0')
	
	# Financial Data Summary
	total_assets = Column(DECIMAL(20, 4), nullable=True)
	total_liabilities = Column(DECIMAL(20, 4), nullable=True)
	total_equity = Column(DECIMAL(20, 4), nullable=True)
	total_revenue = Column(DECIMAL(20, 4), nullable=True)
	net_income = Column(DECIMAL(20, 4), nullable=True)
	
	# Quality Metrics
	balance_difference = Column(DECIMAL(15, 4), nullable=True)  # For balance sheet verification
	variance_percentage = Column(DECIMAL(8, 4), nullable=True)
	data_completeness = Column(DECIMAL(5, 2), nullable=True)  # Percentage of complete data
	
	# Metadata
	statement_data = Column(JSON, nullable=False)  # Line-by-line statement data
	calculation_details = Column(JSON, nullable=True)  # Calculation breakdown
	notes_references = Column(JSON, nullable=True)  # References to notes
	
	# Relationships
	generation = relationship("CFRFReportGeneration", back_populates="financial_statements")
	period = relationship("CFRFReportPeriod", back_populates="financial_statements")
	notes = relationship("CFRFNotes", back_populates="statement")
	disclosures = relationship("CFRFDisclosure", back_populates="statement")
	
	def __repr__(self):
		return f"<CFRFFinancialStatement {self.statement_name}>"


class CFRFConsolidation(Model, AuditMixin, BaseMixin):
	"""
	Consolidation rules and processing.
	
	Defines consolidation rules for multi-entity reporting including
	ownership percentages, elimination entries, and currency translation.
	"""
	__tablename__ = 'cf_fr_consolidation'
	
	# Identity
	consolidation_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Consolidation Information
	consolidation_code = Column(String(50), nullable=False, index=True)
	consolidation_name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	
	# Consolidation Structure
	parent_entity = Column(String(100), nullable=False)
	subsidiary_entity = Column(String(100), nullable=False)
	consolidation_method = Column(String(50), nullable=False)  # full, proportional, equity
	
	# Ownership Information
	ownership_percentage = Column(DECIMAL(8, 4), nullable=False)
	voting_percentage = Column(DECIMAL(8, 4), nullable=True)
	acquisition_date = Column(Date, nullable=True)
	disposal_date = Column(Date, nullable=True)
	
	# Consolidation Rules
	eliminate_intercompany = Column(Boolean, default=True)
	currency_translation_method = Column(String(50), nullable=True)  # current_rate, temporal, hybrid
	functional_currency = Column(String(10), nullable=True)
	reporting_currency = Column(String(10), default='USD')
	
	# Elimination Settings
	elimination_accounts = Column(JSON, nullable=True)  # Account mapping for eliminations
	goodwill_account = Column(String(50), nullable=True)
	minority_interest_account = Column(String(50), nullable=True)
	
	# Effective Periods
	effective_from = Column(Date, nullable=False)
	effective_to = Column(Date, nullable=True)
	is_active = Column(Boolean, default=True)
	
	# Metadata
	consolidation_rules = Column(JSON, nullable=True)  # Detailed consolidation rules
	
	# Relationships
	analytical_reports = relationship("CFRFAnalyticalReport", back_populates="consolidation")
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'consolidation_code', name='uq_consolidation_code_tenant'),
	)
	
	def __repr__(self):
		return f"<CFRFConsolidation {self.consolidation_name}>"


class CFRFNotes(Model, AuditMixin, BaseMixin):
	"""
	Financial statement notes.
	
	Manages notes to financial statements including accounting policies,
	significant estimates, and detailed disclosures.
	"""
	__tablename__ = 'cf_fr_notes'
	
	# Identity
	note_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	statement_id = Column(String(36), ForeignKey('cf_fr_financial_statement.statement_id'), nullable=False)
	
	# Note Information
	note_number = Column(String(10), nullable=False)
	note_title = Column(String(200), nullable=False)
	note_category = Column(String(50), nullable=False)  # accounting_policy, estimate, disclosure, other
	
	# Note Content
	note_text = Column(Text, nullable=False)
	note_format = Column(String(20), default='text')  # text, html, markdown
	
	# Note Properties
	is_required = Column(Boolean, default=False)
	is_standard = Column(Boolean, default=False)
	sort_order = Column(Integer, default=0)
	
	# References
	referenced_accounts = Column(JSON, nullable=True)  # Account references
	referenced_lines = Column(JSON, nullable=True)  # Statement line references
	
	# Approval
	approval_status = Column(String(50), default='draft')  # draft, review, approved
	approved_by = Column(String(36), nullable=True)
	approved_date = Column(DateTime, nullable=True)
	
	# Metadata
	note_attributes = Column(JSON, nullable=True)
	
	# Relationships
	statement = relationship("CFRFFinancialStatement", back_populates="notes")
	
	def __repr__(self):
		return f"<CFRFNotes {self.note_title}>"


class CFRFDisclosure(Model, AuditMixin, BaseMixin):
	"""
	Regulatory disclosures.
	
	Manages regulatory and compliance disclosures required for
	financial reporting including risk disclosures and regulatory requirements.
	"""
	__tablename__ = 'cf_fr_disclosure'
	
	# Identity
	disclosure_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	statement_id = Column(String(36), ForeignKey('cf_fr_financial_statement.statement_id'), nullable=False)
	
	# Disclosure Information
	disclosure_code = Column(String(50), nullable=False, index=True)
	disclosure_title = Column(String(200), nullable=False)
	disclosure_type = Column(String(50), nullable=False)  # regulatory, risk, compliance, voluntary
	
	# Regulatory Information
	regulation_framework = Column(String(100), nullable=True)  # GAAP, IFRS, SOX, Basel, etc.
	regulation_section = Column(String(50), nullable=True)
	compliance_level = Column(String(50), nullable=True)  # required, recommended, optional
	
	# Disclosure Content
	disclosure_text = Column(Text, nullable=False)
	disclosure_format = Column(String(20), default='text')
	supporting_data = Column(JSON, nullable=True)
	
	# Risk Information
	risk_category = Column(String(50), nullable=True)  # credit, market, operational, liquidity
	risk_level = Column(String(20), nullable=True)  # low, medium, high, critical
	mitigation_measures = Column(Text, nullable=True)
	
	# Effective Period
	effective_from = Column(Date, nullable=False)
	effective_to = Column(Date, nullable=True)
	
	# Status
	disclosure_status = Column(String(50), default='draft')  # draft, review, approved, published
	review_frequency = Column(String(50), nullable=True)  # quarterly, annually, as_needed
	next_review_date = Column(Date, nullable=True)
	
	# Approval
	approved_by = Column(String(36), nullable=True)
	approved_date = Column(DateTime, nullable=True)
	approval_notes = Column(Text, nullable=True)
	
	# Metadata
	disclosure_attributes = Column(JSON, nullable=True)
	
	# Relationships
	statement = relationship("CFRFFinancialStatement", back_populates="disclosures")
	
	def __repr__(self):
		return f"<CFRFDisclosure {self.disclosure_title}>"


class CFRFAnalyticalReport(Model, AuditMixin, BaseMixin):
	"""
	Custom analytical reports.
	
	Manages custom analytical and management reports including
	variance analysis, trend analysis, and KPI reporting.
	"""
	__tablename__ = 'cf_fr_analytical_report'
	
	# Identity
	report_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	consolidation_id = Column(String(36), ForeignKey('cf_fr_consolidation.consolidation_id'), nullable=True)
	
	# Report Information
	report_code = Column(String(50), nullable=False, index=True)
	report_name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	
	# Report Type and Category
	report_type = Column(String(50), nullable=False)  # variance, trend, ratio, kpi, custom
	report_category = Column(String(50), nullable=False)  # management, operational, strategic, regulatory
	
	# Analysis Parameters
	analysis_type = Column(String(50), nullable=False)  # budget_vs_actual, period_comparison, trend_analysis
	analysis_periods = Column(Integer, default=12)  # Number of periods to analyze
	comparison_basis = Column(String(50), nullable=True)  # budget, prior_year, plan, forecast
	
	# Data Selection
	account_selection = Column(JSON, nullable=True)  # Account filters and criteria
	entity_selection = Column(JSON, nullable=True)  # Entity/division filters
	dimension_filters = Column(JSON, nullable=True)  # Cost center, department, etc.
	
	# Report Configuration
	chart_types = Column(JSON, nullable=True)  # Chart and visualization types
	key_metrics = Column(JSON, nullable=True)  # Key performance indicators
	threshold_values = Column(JSON, nullable=True)  # Alert thresholds
	
	# Scheduling
	is_scheduled = Column(Boolean, default=False)
	schedule_frequency = Column(String(50), nullable=True)  # daily, weekly, monthly, quarterly
	last_generated = Column(DateTime, nullable=True)
	next_generation = Column(DateTime, nullable=True)
	
	# Output Options
	output_formats = Column(JSON, nullable=True)  # Supported output formats
	default_format = Column(String(20), default='pdf')
	
	# Access Control
	is_public = Column(Boolean, default=False)
	restricted_access = Column(Boolean, default=False)
	access_groups = Column(JSON, nullable=True)  # User groups with access
	
	# Status
	is_active = Column(Boolean, default=True)
	approval_required = Column(Boolean, default=False)
	
	# Metadata
	report_configuration = Column(JSON, nullable=True)  # Detailed report configuration
	calculation_logic = Column(JSON, nullable=True)  # Custom calculation definitions
	
	# Relationships
	consolidation = relationship("CFRFConsolidation", back_populates="analytical_reports")
	distributions = relationship("CFRFReportDistribution", back_populates="analytical_report")
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'report_code', name='uq_analytical_report_code_tenant'),
	)
	
	def __repr__(self):
		return f"<CFRFAnalyticalReport {self.report_name}>"


class CFRFReportDistribution(Model, AuditMixin, BaseMixin):
	"""
	Report distribution management.
	
	Manages the distribution of financial reports to various stakeholders
	including email lists, file shares, and automated delivery systems.
	"""
	__tablename__ = 'cf_fr_report_distribution'
	
	# Identity
	distribution_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	analytical_report_id = Column(String(36), ForeignKey('cf_fr_analytical_report.report_id'), nullable=True)
	
	# Distribution Information
	distribution_name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	distribution_type = Column(String(50), nullable=False)  # email, file_share, api, portal
	
	# Distribution Lists
	email_recipients = Column(JSON, nullable=True)  # Email addresses and names
	distribution_groups = Column(JSON, nullable=True)  # Distribution group configurations
	external_recipients = Column(JSON, nullable=True)  # External stakeholder information
	
	# Delivery Configuration
	delivery_method = Column(String(50), nullable=False)  # email, ftp, sftp, api, portal
	delivery_format = Column(String(20), default='pdf')  # pdf, excel, html, json
	delivery_schedule = Column(String(50), nullable=True)  # Schedule for automatic delivery
	
	# Email Configuration
	email_subject_template = Column(String(500), nullable=True)
	email_body_template = Column(Text, nullable=True)
	include_attachments = Column(Boolean, default=True)
	compress_attachments = Column(Boolean, default=False)
	
	# File Share Configuration
	file_path = Column(String(500), nullable=True)
	file_naming_pattern = Column(String(200), nullable=True)
	retention_days = Column(Integer, nullable=True)
	
	# Security and Access
	encryption_required = Column(Boolean, default=False)
	password_protection = Column(Boolean, default=False)
	access_expiry_days = Column(Integer, nullable=True)
	
	# Distribution Status
	is_active = Column(Boolean, default=True)
	last_distribution = Column(DateTime, nullable=True)
	next_distribution = Column(DateTime, nullable=True)
	distribution_count = Column(Integer, default=0)
	success_count = Column(Integer, default=0)
	failure_count = Column(Integer, default=0)
	
	# Approval and Control
	requires_approval = Column(Boolean, default=False)
	approval_workflow = Column(String(100), nullable=True)
	
	# Metadata
	distribution_log = Column(JSON, nullable=True)  # Distribution history and logs
	configuration = Column(JSON, nullable=True)  # Additional configuration options
	
	# Relationships
	analytical_report = relationship("CFRFAnalyticalReport", back_populates="distributions")
	
	def __repr__(self):
		return f"<CFRFReportDistribution {self.distribution_name}>"


# ==================================================================================
# REVOLUTIONARY AI-POWERED MODELS FOR FINANCIAL INTELLIGENCE
# ==================================================================================

class CFRFConversationalInterface(Model, AuditMixin, BaseMixin):
	"""
	Natural Language Processing Interface for Financial Reporting.
	
	Enables conversational report creation and analysis using advanced NLP
	and AI understanding of financial terminology and context.
	"""
	__tablename__ = 'cf_fr_conversational_interface'
	
	# Identity
	conversation_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	user_id = Column(String(36), nullable=False, index=True)
	
	# Conversation Context
	session_id = Column(String(36), nullable=False, index=True)
	conversation_type = Column(String(50), default='report_creation')  # report_creation, analysis, explanation
	language_code = Column(String(10), default='en-US')
	
	# User Input
	user_query = Column(Text, nullable=False)
	query_timestamp = Column(DateTime, default=datetime.now)
	input_modality = Column(String(20), default='text')  # text, voice, gesture
	
	# AI Processing
	intent_classification = Column(String(50), nullable=False, index=True)
	confidence_score = Column(Float, default=0.0)  # 0.0 to 1.0
	extracted_entities = Column(JSON, nullable=True)  # Named entities and parameters
	context_understanding = Column(JSON, nullable=True)  # Contextual information
	
	# AI Response
	ai_response = Column(Text, nullable=True)
	response_type = Column(String(50), default='interactive')  # interactive, report, chart, insight
	generated_artifacts = Column(JSON, nullable=True)  # Reports, charts, analyses generated
	
	# Learning and Optimization
	user_feedback_score = Column(Float, nullable=True)  # User satisfaction rating
	resolution_success = Column(Boolean, default=True)  # Whether request was resolved
	learning_data = Column(JSON, nullable=True)  # Data for model improvement
	
	# Performance Metrics
	processing_time_ms = Column(Integer, nullable=True)
	model_version = Column(String(20), nullable=True)
	
	# Indexes for performance
	__table_args__ = (
		Index('idx_conversation_tenant_user', 'tenant_id', 'user_id'),
		Index('idx_conversation_session', 'session_id'),
		Index('idx_conversation_intent', 'intent_classification'),
	)
	
	def __repr__(self):
		return f"<CFRFConversationalInterface {self.conversation_id}>"


class CFRFAIInsightEngine(Model, AuditMixin, BaseMixin):
	"""
	AI-Powered Financial Insight Generation System.
	
	Automatically generates intelligent insights, explanations, and recommendations
	based on financial data patterns and industry knowledge.
	"""
	__tablename__ = 'cf_fr_ai_insight_engine'
	
	# Identity
	insight_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Insight Context
	source_report_id = Column(String(36), nullable=True, index=True)
	source_data_type = Column(String(50), nullable=False)  # financial_statement, variance_analysis, trend_analysis
	analysis_period = Column(String(50), nullable=True)
	
	# Insight Content
	insight_type = Column(String(50), nullable=False, index=True)  # variance_explanation, trend_alert, forecast_insight
	title = Column(String(200), nullable=False)
	description = Column(Text, nullable=False)
	narrative_explanation = Column(Text, nullable=True)  # AI-generated narrative
	
	# AI Assessment
	confidence_level = Column(Float, nullable=False)  # 0.0 to 1.0
	impact_score = Column(Float, nullable=False)  # 0.0 to 10.0 (business impact)
	urgency_level = Column(String(20), default='medium')  # low, medium, high, critical
	accuracy_validation = Column(Float, nullable=True)  # Post-validation accuracy
	
	# Actionable Intelligence
	recommended_actions = Column(JSON, nullable=True)  # List of recommended actions
	risk_indicators = Column(JSON, nullable=True)  # Associated risk factors
	opportunity_indicators = Column(JSON, nullable=True)  # Potential opportunities
	
	# Supporting Data
	supporting_metrics = Column(JSON, nullable=True)  # Key metrics and calculations
	data_sources = Column(JSON, nullable=True)  # Source data references
	related_insights = Column(JSON, nullable=True)  # Links to related insights
	
	# ML Model Information
	model_type = Column(String(50), nullable=False)  # predictive, anomaly_detection, pattern_recognition
	model_version = Column(String(20), nullable=True)
	training_data_period = Column(String(50), nullable=True)
	
	# User Interaction
	user_views = Column(Integer, default=0)
	user_actions_taken = Column(JSON, nullable=True)  # Actions users took based on insight
	feedback_ratings = Column(JSON, nullable=True)  # User feedback on insight quality
	
	# Status and Lifecycle
	insight_status = Column(String(20), default='active')  # active, validated, archived, superseded
	expiry_date = Column(DateTime, nullable=True)
	superseded_by = Column(String(36), nullable=True)  # ID of insight that replaces this one
	
	# Performance tracking
	generation_time_ms = Column(Integer, nullable=True)
	
	# Indexes for performance
	__table_args__ = (
		Index('idx_insight_tenant_type', 'tenant_id', 'insight_type'),
		Index('idx_insight_confidence', 'confidence_level'),
		Index('idx_insight_impact', 'impact_score'),
	)
	
	def __repr__(self):
		return f"<CFRFAIInsightEngine {self.title[:50]}>"


class CFRFPredictiveAnalytics(Model, AuditMixin, BaseMixin):
	"""
	Advanced Predictive Analytics for Financial Forecasting.
	
	Machine learning models for predicting financial trends, variances,
	and potential issues before they occur.
	"""
	__tablename__ = 'cf_fr_predictive_analytics'
	
	# Identity
	prediction_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Prediction Context
	prediction_type = Column(String(50), nullable=False, index=True)  # variance_forecast, revenue_prediction, expense_forecast
	target_metric = Column(String(100), nullable=False)  # What is being predicted
	prediction_horizon = Column(Integer, nullable=False)  # Days into the future
	base_period = Column(Date, nullable=False)  # Base date for prediction
	
	# Prediction Results
	predicted_value = Column(DECIMAL(20, 4), nullable=False)
	confidence_interval_lower = Column(DECIMAL(20, 4), nullable=True)
	confidence_interval_upper = Column(DECIMAL(20, 4), nullable=True)
	confidence_percentage = Column(Float, nullable=False)  # Confidence level (e.g., 95%)
	
	# Model Performance
	model_type = Column(String(50), nullable=False)  # linear_regression, random_forest, neural_network, ensemble
	model_accuracy_score = Column(Float, nullable=True)  # Historical accuracy
	feature_importance = Column(JSON, nullable=True)  # Most important factors
	training_data_points = Column(Integer, nullable=True)
	
	# Contributing Factors
	primary_drivers = Column(JSON, nullable=True)  # Main factors influencing prediction
	risk_factors = Column(JSON, nullable=True)  # Factors that could cause variance
	seasonal_adjustments = Column(JSON, nullable=True)  # Seasonal pattern adjustments
	external_factors = Column(JSON, nullable=True)  # External economic factors
	
	# Validation and Tracking
	actual_value = Column(DECIMAL(20, 4), nullable=True)  # Actual value when known
	prediction_error = Column(DECIMAL(15, 4), nullable=True)  # Difference from actual
	validation_date = Column(DateTime, nullable=True)  # When actual value was recorded
	
	# Alert Configuration
	variance_threshold = Column(DECIMAL(10, 4), nullable=True)  # Alert threshold
	alert_triggered = Column(Boolean, default=False)
	alert_recipients = Column(JSON, nullable=True)
	
	# Scenario Analysis
	best_case_scenario = Column(DECIMAL(20, 4), nullable=True)
	worst_case_scenario = Column(DECIMAL(20, 4), nullable=True)
	most_likely_scenario = Column(DECIMAL(20, 4), nullable=True)
	scenario_probabilities = Column(JSON, nullable=True)
	
	# Model Metadata
	model_training_date = Column(DateTime, nullable=True)
	model_retrain_frequency = Column(String(20), default='monthly')
	next_retrain_date = Column(DateTime, nullable=True)
	
	# Indexes for performance
	__table_args__ = (
		Index('idx_prediction_tenant_type', 'tenant_id', 'prediction_type'),
		Index('idx_prediction_horizon', 'prediction_horizon'),
		Index('idx_prediction_accuracy', 'model_accuracy_score'),
	)
	
	def __repr__(self):
		return f"<CFRFPredictiveAnalytics {self.target_metric} - {self.predicted_value}>"


class CFRFRealTimeCollaboration(Model, AuditMixin, BaseMixin):
	"""
	Real-Time Collaborative Financial Reporting System.
	
	Manages multi-user real-time collaboration with intelligent conflict resolution
	and synchronized editing capabilities.
	"""
	__tablename__ = 'cf_fr_real_time_collaboration'
	
	# Identity
	collaboration_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Session Information
	session_id = Column(String(36), nullable=False, index=True)
	report_id = Column(String(36), nullable=False, index=True)
	session_type = Column(String(50), default='collaborative_editing')  # editing, review, approval
	
	# Participants
	session_owner = Column(String(36), nullable=False)  # User who started session
	active_participants = Column(JSON, nullable=False)  # Currently active users
	total_participants = Column(JSON, nullable=True)  # All users who joined session
	max_concurrent_users = Column(Integer, default=0)
	
	# Session Timeline
	session_start = Column(DateTime, default=datetime.now)
	session_end = Column(DateTime, nullable=True)
	last_activity = Column(DateTime, default=datetime.now)
	session_duration_minutes = Column(Integer, nullable=True)
	
	# Collaboration Features
	real_time_sync_enabled = Column(Boolean, default=True)
	conflict_resolution_mode = Column(String(30), default='intelligent')  # manual, automatic, intelligent
	version_control_enabled = Column(Boolean, default=True)
	auto_save_interval = Column(Integer, default=30)  # Seconds
	
	# Activity Tracking
	edit_operations = Column(JSON, nullable=True)  # List of edit operations
	comment_threads = Column(JSON, nullable=True)  # Discussion threads
	approval_requests = Column(JSON, nullable=True)  # Approval workflow data
	
	# Conflict Management
	conflicts_detected = Column(Integer, default=0)
	conflicts_resolved = Column(Integer, default=0)
	conflict_resolution_log = Column(JSON, nullable=True)
	
	# AI Enhancement
	ai_suggestions_enabled = Column(Boolean, default=True)
	ai_suggestions_provided = Column(JSON, nullable=True)
	ai_suggestions_accepted = Column(Integer, default=0)
	
	# Performance Metrics
	sync_latency_ms = Column(Integer, nullable=True)  # Average sync latency
	operation_count = Column(Integer, default=0)  # Total operations performed
	bandwidth_usage_mb = Column(Float, nullable=True)
	
	# Quality Metrics
	user_satisfaction_scores = Column(JSON, nullable=True)  # User feedback
	productivity_metrics = Column(JSON, nullable=True)  # Efficiency measurements
	error_rate = Column(Float, default=0.0)  # Error percentage
	
	# Indexes for performance
	__table_args__ = (
		Index('idx_collaboration_tenant_session', 'tenant_id', 'session_id'),
		Index('idx_collaboration_report', 'report_id'),
		Index('idx_collaboration_active', 'last_activity'),
	)
	
	def __repr__(self):
		return f"<CFRFRealTimeCollaboration {self.session_id}>"


class CFRFDataQualityMonitor(Model, AuditMixin, BaseMixin):
	"""
	Intelligent Data Quality Monitoring and Validation System.
	
	Continuously monitors financial data quality with AI-powered anomaly detection
	and automated correction suggestions.
	"""
	__tablename__ = 'cf_fr_data_quality_monitor'
	
	# Identity
	monitor_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Monitoring Context
	data_source = Column(String(100), nullable=False, index=True)  # Source system or table
	monitoring_scope = Column(String(50), nullable=False)  # full_dataset, incremental, targeted
	check_frequency = Column(String(20), default='real_time')  # real_time, hourly, daily
	
	# Quality Assessment
	overall_quality_score = Column(Float, nullable=False, index=True)  # 0.0 to 100.0
	quality_trend = Column(String(20), default='stable')  # improving, stable, deteriorating
	last_assessment = Column(DateTime, default=datetime.now)
	
	# Quality Dimensions
	completeness_score = Column(Float, default=100.0)  # % of required fields populated
	accuracy_score = Column(Float, default=100.0)  # % of accurate values
	consistency_score = Column(Float, default=100.0)  # % of consistent values
	timeliness_score = Column(Float, default=100.0)  # % of timely data updates
	validity_score = Column(Float, default=100.0)  # % of valid format/range values
	
	# Issue Detection
	anomalies_detected = Column(Integer, default=0)
	critical_issues = Column(Integer, default=0)
	warning_issues = Column(Integer, default=0)
	info_issues = Column(Integer, default=0)
	
	# Issue Details
	detected_anomalies = Column(JSON, nullable=True)  # List of specific anomalies
	data_profiling_results = Column(JSON, nullable=True)  # Statistical profiles
	validation_rules_failed = Column(JSON, nullable=True)  # Failed validation rules
	
	# AI-Powered Enhancement
	ml_anomaly_detection = Column(Boolean, default=True)
	statistical_outlier_detection = Column(Boolean, default=True)
	pattern_recognition_enabled = Column(Boolean, default=True)
	ai_correction_suggestions = Column(JSON, nullable=True)
	
	# Auto-Correction
	auto_correction_enabled = Column(Boolean, default=False)
	corrections_applied = Column(Integer, default=0)
	correction_success_rate = Column(Float, default=0.0)
	correction_log = Column(JSON, nullable=True)
	
	# Performance Tracking
	monitoring_duration_ms = Column(Integer, nullable=True)
	records_processed = Column(Integer, nullable=True)
	processing_rate_per_second = Column(Float, nullable=True)
	
	# Alerting
	alert_thresholds = Column(JSON, nullable=True)  # Quality score thresholds
	alerts_triggered = Column(Integer, default=0)
	alert_recipients = Column(JSON, nullable=True)
	
	# Historical Tracking
	quality_history = Column(JSON, nullable=True)  # Historical quality scores
	improvement_suggestions = Column(JSON, nullable=True)  # Recommended improvements
	
	# Indexes for performance
	__table_args__ = (
		Index('idx_quality_tenant_source', 'tenant_id', 'data_source'),
		Index('idx_quality_score', 'overall_quality_score'),
		Index('idx_quality_assessment', 'last_assessment'),
	)
	
	def __repr__(self):
		return f"<CFRFDataQualityMonitor {self.data_source} - {self.overall_quality_score}%>"


class CFRFBlockchainAuditTrail(Model, AuditMixin, BaseMixin):
	"""
	Blockchain-Based Immutable Audit Trail for Financial Reporting.
	
	Provides cryptographic verification and immutable audit trails for
	regulatory compliance and forensic analysis.
	"""
	__tablename__ = 'cf_fr_blockchain_audit_trail'
	
	# Identity
	audit_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Blockchain Reference
	block_hash = Column(String(64), nullable=False, unique=True, index=True)
	transaction_hash = Column(String(64), nullable=False, index=True)
	block_number = Column(Integer, nullable=False, index=True)
	blockchain_network = Column(String(50), default='ethereum')
	
	# Audit Context
	audit_event_type = Column(String(50), nullable=False, index=True)  # report_creation, data_modification, approval
	source_entity_type = Column(String(50), nullable=False)  # report, statement, template
	source_entity_id = Column(String(36), nullable=False, index=True)
	
	# Event Details
	event_timestamp = Column(DateTime, default=datetime.now, index=True)
	user_id = Column(String(36), nullable=False)
	user_role = Column(String(50), nullable=True)
	event_description = Column(Text, nullable=False)
	
	# Data Integrity
	data_hash = Column(String(64), nullable=False)  # SHA-256 hash of the data
	previous_hash = Column(String(64), nullable=True)  # Previous record hash for chaining
	merkle_root = Column(String(64), nullable=True)  # Merkle tree root for batch verification
	
	# Cryptographic Verification
	digital_signature = Column(Text, nullable=True)  # Digital signature of the event
	public_key = Column(Text, nullable=True)  # Public key for signature verification
	certificate_authority = Column(String(100), nullable=True)
	
	# Compliance and Regulatory
	regulatory_framework = Column(String(50), nullable=True)  # SOX, GDPR, etc.
	compliance_status = Column(String(20), default='compliant')
	retention_period_years = Column(Integer, default=7)
	
	# Smart Contract Integration
	smart_contract_address = Column(String(42), nullable=True)  # Ethereum address
	smart_contract_function = Column(String(100), nullable=True)
	gas_used = Column(Integer, nullable=True)
	transaction_fee = Column(DECIMAL(18, 8), nullable=True)
	
	# Verification Status
	verification_status = Column(String(20), default='pending')  # pending, verified, failed
	verification_timestamp = Column(DateTime, nullable=True)
	verification_attempts = Column(Integer, default=0)
	
	# Performance Metrics
	blockchain_confirmation_time = Column(Integer, nullable=True)  # Seconds
	network_congestion_factor = Column(Float, nullable=True)
	
	# Forensic Analysis
	forensic_markers = Column(JSON, nullable=True)  # Markers for forensic investigation
	anomaly_indicators = Column(JSON, nullable=True)  # Unusual patterns detected
	
	# Indexes for performance
	__table_args__ = (
		Index('idx_audit_tenant_event', 'tenant_id', 'audit_event_type'),
		Index('idx_audit_entity', 'source_entity_type', 'source_entity_id'),
		Index('idx_audit_timestamp', 'event_timestamp'),
		Index('idx_audit_block', 'block_number'),
	)
	
	def __repr__(self):
		return f"<CFRFBlockchainAuditTrail {self.audit_event_type} - {self.block_hash[:8]}>"