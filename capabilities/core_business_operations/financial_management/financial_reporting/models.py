"""
Financial Reporting Models

Database models for the Financial Reporting sub-capability including report templates,
financial statements, consolidations, notes, and analytical reporting functionality.
"""

from datetime import datetime, date
from typing import Dict, List, Any, Optional
from decimal import Decimal
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, Date, DECIMAL, ForeignKey, UniqueConstraint, JSON
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str
import json

from ...auth_rbac.models import BaseMixin, AuditMixin, Model


class CFRFReportTemplate(Model, AuditMixin, BaseMixin):
	"""
	Financial report template definitions.
	
	Defines the structure and formatting rules for financial statements
	and other reports that can be generated repeatedly.
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
	
	# Template Properties
	is_system = Column(Boolean, default=False)  # System templates cannot be deleted
	is_active = Column(Boolean, default=True)
	version = Column(String(20), default='1.0')
	
	# Formatting Options
	currency_type = Column(String(20), default='single')  # single, multi, reporting
	show_percentages = Column(Boolean, default=False)
	show_variances = Column(Boolean, default=False)
	decimal_places = Column(Integer, default=2)
	thousands_separator = Column(Boolean, default=True)
	
	# Layout Options
	page_orientation = Column(String(20), default='portrait')  # portrait, landscape
	font_size = Column(Integer, default=12)
	include_logo = Column(Boolean, default=True)
	include_header = Column(Boolean, default=True)
	include_footer = Column(Boolean, default=True)
	
	# Generation Options
	auto_generate = Column(Boolean, default=False)
	generation_frequency = Column(String(20), nullable=True)  # daily, weekly, monthly, quarterly, yearly
	last_generated = Column(DateTime, nullable=True)
	
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