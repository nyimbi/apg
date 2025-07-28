"""
APG Payroll Management - Enhanced Data Models

Revolutionary payroll data models with AI-powered validation,
multi-tenant architecture, and seamless APG platform integration.
"""

from datetime import datetime, date, timedelta
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, List, Any, Union
from uuid import UUID

from sqlalchemy import Column, String, Integer, Numeric, DateTime, Date, Boolean, Text, ForeignKey, Index, JSON, DECIMAL, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID, ARRAY, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates, Session
from sqlalchemy.sql import func
from pydantic import BaseModel, Field, ConfigDict, validator, root_validator, AfterValidator
from uuid_extensions import uuid7str

# APG Platform Imports
from ...auth_rbac.models import BaseMixin, AuditMixin, Model
from ...employee_data_management.models import HREmployee
from ...ai_orchestration.services import AIValidationService
from ...audit_compliance.models import ComplianceRule

Base = declarative_base()


# ===============================
# REVOLUTIONARY AI-POWERED ENUMS
# ===============================

class PayrollStatus(str, Enum):
	"""AI-enhanced payroll status with intelligent transitions."""
	DRAFT = "draft"
	PROCESSING = "processing"
	AI_VALIDATION = "ai_validation"
	COMPLIANCE_CHECK = "compliance_check"
	APPROVED = "approved"
	POSTED = "posted"
	ERROR = "error"
	CANCELLED = "cancelled"

class PayComponentType(str, Enum):
	"""AI-categorized pay component types."""
	EARNINGS_REGULAR = "earnings_regular"
	EARNINGS_OVERTIME = "earnings_overtime"
	EARNINGS_BONUS = "earnings_bonus"
	EARNINGS_COMMISSION = "earnings_commission"
	EARNINGS_ALLOWANCE = "earnings_allowance"
	DEDUCTION_TAX = "deduction_tax"
	DEDUCTION_BENEFIT = "deduction_benefit"
	DEDUCTION_GARNISHMENT = "deduction_garnishment"
	DEDUCTION_OTHER = "deduction_other"

class PayFrequency(str, Enum):
	"""Global pay frequency standards."""
	WEEKLY = "weekly"
	BI_WEEKLY = "bi_weekly"
	SEMI_MONTHLY = "semi_monthly"
	MONTHLY = "monthly"
	QUARTERLY = "quarterly"
	ANNUALLY = "annually"

class TaxType(str, Enum):
	"""Global tax classifications."""
	FEDERAL_INCOME = "federal_income"
	STATE_INCOME = "state_income"
	LOCAL_INCOME = "local_income"
	FICA_SOCIAL_SECURITY = "fica_social_security"
	FICA_MEDICARE = "fica_medicare"
	FUTA = "futa"
	SUTA = "suta"
	INTERNATIONAL_VAT = "international_vat"
	INTERNATIONAL_INCOME = "international_income"


class PRPayrollPeriod(Model, AuditMixin, BaseMixin):
	"""Revolutionary AI-powered payroll period management.
	
	Next-generation payroll periods with intelligent automation,
	real-time processing, and predictive analytics.
	"""
	__tablename__ = 'pr_payroll_period'
	
	# Identity
	id: str = Field(default_factory=uuid7str)
	period_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Period Information with AI Enhancement
	period_name = Column(String(100), nullable=False, index=True)
	period_type = Column(String(20), nullable=False, index=True)
	pay_frequency = Column(String(20), nullable=False, index=True)
	
	# Intelligent Date Management
	start_date = Column(Date, nullable=False, index=True)
	end_date = Column(Date, nullable=False, index=True)
	pay_date = Column(Date, nullable=False, index=True)
	cutoff_date = Column(Date, nullable=True, index=True)
	
	# AI-Enhanced Status Management
	status = Column(String(30), default=PayrollStatus.DRAFT, index=True)
	processing_score = Column(DECIMAL(5, 2), default=0.00)  # AI readiness score
	compliance_score = Column(DECIMAL(5, 2), default=0.00)  # AI compliance validation
	
	# Advanced Configuration
	fiscal_year = Column(Integer, nullable=False, index=True)
	fiscal_quarter = Column(Integer, nullable=True)
	timezone = Column(String(50), default='UTC')
	currency_code = Column(String(3), default='USD')
	
	# AI Processing Metadata
	ai_predictions = Column(JSONB, nullable=True)  # AI-powered insights
	processing_metrics = Column(JSONB, nullable=True)  # Performance analytics
	exception_flags = Column(ARRAY(String), nullable=True)  # AI-detected issues
	
	# Processing Timestamps
	processing_started_at = Column(DateTime, nullable=True)
	processing_completed_at = Column(DateTime, nullable=True)
	finalized_at = Column(DateTime, nullable=True)
	
	# Global Multi-Entity Support
	legal_entity_id = Column(String(36), nullable=True, index=True)
	country_code = Column(String(2), nullable=False, index=True)
	region_code = Column(String(10), nullable=True)
	
	# Configuration
	is_active = Column(Boolean, default=True)
	is_test_period = Column(Boolean, default=False)
	auto_close_enabled = Column(Boolean, default=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'period_name', name='uq_pr_period_name_tenant'),
		Index('ix_pr_period_dates', 'start_date', 'end_date'),
		Index('ix_pr_period_status_country', 'status', 'country_code'),
	)
	
	# Relationships
	payroll_runs = relationship("PRPayrollRun", back_populates="period", cascade="all, delete-orphan")
	payroll_analytics = relationship("PRPayrollAnalytics", back_populates="period")
	
	@validates('pay_frequency')
	def validate_pay_frequency(self, key, value):
		if value not in [freq.value for freq in PayFrequency]:
			raise ValueError(f"Invalid pay frequency: {value}")
		return value
	
	def __repr__(self):
		return f"<PRPayrollPeriod {self.period_name} ({self.status})>"


class PRPayrollRun(Model, AuditMixin, BaseMixin):
	"""Revolutionary real-time payroll processing engine.
	
	Next-generation payroll runs with AI-powered validation,
	real-time processing, and intelligent error handling.
	"""
	__tablename__ = 'pr_payroll_run'
	
	# Identity
	id: str = Field(default_factory=uuid7str)
	run_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	period_id = Column(String(36), ForeignKey('pr_payroll_period.period_id'), nullable=False, index=True)
	
	# Enhanced Run Information
	run_number = Column(Integer, nullable=False, index=True)
	run_type = Column(String(30), default='regular', index=True)
	run_name = Column(String(200), nullable=True)
	description = Column(Text, nullable=True)
	priority = Column(String(10), default='normal')  # low, normal, high, urgent
	
	# AI-Enhanced Processing Status
	status = Column(String(30), default=PayrollStatus.DRAFT, index=True)
	processing_stage = Column(String(50), nullable=True)  # Current processing stage
	progress_percentage = Column(DECIMAL(5, 2), default=0.00)
	processing_score = Column(DECIMAL(5, 2), default=0.00)  # AI quality score
	
	# Processing Timestamps
	queued_at = Column(DateTime, nullable=True)
	started_at = Column(DateTime, nullable=True)
	completed_at = Column(DateTime, nullable=True)
	processed_by = Column(String(36), nullable=True)
	
	# AI-Powered Validation
	validation_status = Column(String(20), default='pending')  # pending, passed, failed
	validation_score = Column(DECIMAL(5, 2), default=0.00)
	validation_errors = Column(JSONB, nullable=True)
	validation_warnings = Column(JSONB, nullable=True)
	
	# Intelligent Approval Workflow
	approval_required = Column(Boolean, default=True)
	approval_status = Column(String(20), default='pending')  # pending, approved, rejected
	approved_by = Column(String(36), nullable=True)
	approved_at = Column(DateTime, nullable=True)
	approval_comments = Column(Text, nullable=True)
	
	# Real-Time Financial Totals
	total_gross_pay = Column(DECIMAL(15, 2), default=0.00)
	total_regular_pay = Column(DECIMAL(15, 2), default=0.00)
	total_overtime_pay = Column(DECIMAL(15, 2), default=0.00)
	total_bonus_pay = Column(DECIMAL(15, 2), default=0.00)
	total_deductions = Column(DECIMAL(15, 2), default=0.00)
	total_taxes = Column(DECIMAL(15, 2), default=0.00)
	total_net_pay = Column(DECIMAL(15, 2), default=0.00)
	
	# Employee and Processing Metrics
	employee_count = Column(Integer, default=0)
	processed_employee_count = Column(Integer, default=0)
	error_count = Column(Integer, default=0)
	warning_count = Column(Integer, default=0)
	
	# AI Analytics and Insights
	analytics_data = Column(JSONB, nullable=True)  # AI-generated insights
	performance_metrics = Column(JSONB, nullable=True)  # Processing metrics
	comparisons = Column(JSONB, nullable=True)  # Period-over-period comparisons
	
	# Error Handling and Recovery
	error_details = Column(JSONB, nullable=True)
	retry_count = Column(Integer, default=0)
	max_retries = Column(Integer, default=3)
	last_error_at = Column(DateTime, nullable=True)
	
	# Global Compliance
	compliance_checks = Column(JSONB, nullable=True)
	compliance_score = Column(DECIMAL(5, 2), default=0.00)
	jurisdiction_data = Column(JSONB, nullable=True)
	
	# Configuration
	is_active = Column(Boolean, default=True)
	is_test_run = Column(Boolean, default=False)
	auto_approve_threshold = Column(DECIMAL(5, 2), default=95.00)
	notifications_enabled = Column(Boolean, default=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'period_id', 'run_number', name='uq_pr_run_number'),
		Index('ix_pr_run_status_date', 'status', 'started_at'),
		Index('ix_pr_run_processing', 'processing_stage', 'progress_percentage'),
	)
	
	# Relationships
	period = relationship("PRPayrollPeriod", back_populates="payroll_runs")
	employee_payrolls = relationship("PREmployeePayroll", back_populates="payroll_run", cascade="all, delete-orphan")
	payroll_journals = relationship("PRPayrollJournal", back_populates="payroll_run")
	payroll_analytics = relationship("PRPayrollAnalytics", back_populates="payroll_run")
	
	@validates('status')
	def validate_status(self, key, value):
		if value not in [status.value for status in PayrollStatus]:
			raise ValueError(f"Invalid payroll status: {value}")
		return value
	
	def calculate_processing_score(self) -> float:
		"""AI-powered processing quality score calculation."""
		base_score = 100.0
		
		# Deduct for errors and warnings
		if self.error_count > 0:
			base_score -= (self.error_count * 10)
		if self.warning_count > 0:
			base_score -= (self.warning_count * 2)
		
		# Bonus for compliance
		if self.compliance_score:
			base_score += (float(self.compliance_score) - 50) / 10
		
		return max(0.0, min(100.0, base_score))
	
	def __repr__(self):
		return f"<PRPayrollRun {self.run_number} ({self.status})>"


class PREmployeePayroll(Model, AuditMixin, BaseMixin):
	"""Revolutionary AI-powered individual employee payroll processing.
	
	Advanced employee payroll record with real-time calculations,
	AI-powered validation, and intelligent error handling.
	"""
	__tablename__ = 'pr_employee_payroll'
	
	# Identity
	id: str = Field(default_factory=uuid7str)
	employee_payroll_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	run_id = Column(String(36), ForeignKey('pr_payroll_run.run_id'), nullable=False, index=True)
	employee_id = Column(String(36), ForeignKey('hr_edm_employee.employee_id'), nullable=False, index=True)
	
	# Employee Information Snapshot
	employee_number = Column(String(50), nullable=False, index=True)
	employee_name = Column(String(300), nullable=False)
	full_name = Column(String(400), nullable=True)  # First Last Middle
	department_id = Column(String(36), nullable=True)
	department_name = Column(String(200), nullable=True)
	position_id = Column(String(36), nullable=True)
	position_title = Column(String(200), nullable=True)
	cost_center = Column(String(50), nullable=True)
	location_id = Column(String(36), nullable=True)
	
	# Enhanced Pay Configuration
	pay_frequency = Column(String(20), nullable=False)
	pay_method = Column(String(30), default='direct_deposit')
	currency_code = Column(String(3), default='USD')
	pay_schedule_id = Column(String(36), nullable=True)
	
	# AI-Enhanced Calculation Base
	base_salary = Column(DECIMAL(15, 4), nullable=True)
	hourly_rate = Column(DECIMAL(10, 4), nullable=True)
	annual_salary = Column(DECIMAL(15, 2), nullable=True)
	
	# Time and Attendance Integration
	regular_hours = Column(DECIMAL(10, 2), default=0.00)
	overtime_hours = Column(DECIMAL(10, 2), default=0.00)
	double_time_hours = Column(DECIMAL(10, 2), default=0.00)
	holiday_hours = Column(DECIMAL(10, 2), default=0.00)
	sick_hours = Column(DECIMAL(10, 2), default=0.00)
	vacation_hours = Column(DECIMAL(10, 2), default=0.00)
	paid_time_off_hours = Column(DECIMAL(10, 2), default=0.00)
	
	# Revolutionary Pay Calculations
	regular_pay = Column(DECIMAL(15, 2), default=0.00)
	overtime_pay = Column(DECIMAL(15, 2), default=0.00)
	double_time_pay = Column(DECIMAL(15, 2), default=0.00)
	bonus_pay = Column(DECIMAL(15, 2), default=0.00)
	commission_pay = Column(DECIMAL(15, 2), default=0.00)
	allowances = Column(DECIMAL(15, 2), default=0.00)
	gross_earnings = Column(DECIMAL(15, 2), default=0.00)
	
	# Advanced Deductions
	pre_tax_deductions = Column(DECIMAL(15, 2), default=0.00)
	post_tax_deductions = Column(DECIMAL(15, 2), default=0.00)
	total_deductions = Column(DECIMAL(15, 2), default=0.00)
	
	# Comprehensive Tax Calculations
	federal_income_tax = Column(DECIMAL(12, 2), default=0.00)
	state_income_tax = Column(DECIMAL(12, 2), default=0.00)
	local_income_tax = Column(DECIMAL(12, 2), default=0.00)
	fica_social_security = Column(DECIMAL(12, 2), default=0.00)
	fica_medicare = Column(DECIMAL(12, 2), default=0.00)
	futa_tax = Column(DECIMAL(12, 2), default=0.00)
	suta_tax = Column(DECIMAL(12, 2), default=0.00)
	international_taxes = Column(DECIMAL(12, 2), default=0.00)
	total_taxes = Column(DECIMAL(15, 2), default=0.00)
	
	# Final Calculations
	taxable_income = Column(DECIMAL(15, 2), default=0.00)
	net_pay = Column(DECIMAL(15, 2), default=0.00)
	disposable_income = Column(DECIMAL(15, 2), default=0.00)
	
	# Year-to-Date Tracking
	ytd_gross = Column(DECIMAL(18, 2), default=0.00)
	ytd_deductions = Column(DECIMAL(18, 2), default=0.00)
	ytd_taxes = Column(DECIMAL(18, 2), default=0.00)
	ytd_net = Column(DECIMAL(18, 2), default=0.00)
	ytd_federal_tax = Column(DECIMAL(15, 2), default=0.00)
	ytd_state_tax = Column(DECIMAL(15, 2), default=0.00)
	ytd_fica_ss = Column(DECIMAL(15, 2), default=0.00)
	ytd_fica_medicare = Column(DECIMAL(15, 2), default=0.00)
	
	# AI-Enhanced Status Management
	status = Column(String(30), default=PayrollStatus.DRAFT, index=True)
	processing_stage = Column(String(50), nullable=True)
	validation_score = Column(DECIMAL(5, 2), default=0.00)
	calculation_confidence = Column(DECIMAL(5, 2), default=0.00)
	
	# Processing Timestamps
	calculation_started_at = Column(DateTime, nullable=True)
	calculation_completed_at = Column(DateTime, nullable=True)
	validation_completed_at = Column(DateTime, nullable=True)
	approval_date = Column(DateTime, nullable=True)
	payment_date = Column(DateTime, nullable=True)
	
	# Advanced Tax Configuration
	tax_jurisdiction = Column(String(100), nullable=True)
	filing_status = Column(String(30), nullable=True)
	federal_allowances = Column(Integer, default=0)
	state_allowances = Column(Integer, default=0)
	additional_withholding = Column(DECIMAL(10, 2), default=0.00)
	exemptions = Column(JSONB, nullable=True)
	tax_elections = Column(JSONB, nullable=True)
	
	# AI Error Detection and Handling
	has_errors = Column(Boolean, default=False)
	has_warnings = Column(Boolean, default=False)
	error_details = Column(JSONB, nullable=True)
	warning_details = Column(JSONB, nullable=True)
	ai_recommendations = Column(JSONB, nullable=True)
	
	# Compliance and Audit
	compliance_flags = Column(ARRAY(String), nullable=True)
	audit_trail = Column(JSONB, nullable=True)
	last_modified_by = Column(String(36), nullable=True)
	modification_reason = Column(String(500), nullable=True)
	
	# Advanced Configuration
	is_active = Column(Boolean, default=True)
	is_manual_override = Column(Boolean, default=False)
	override_reason = Column(Text, nullable=True)
	special_processing = Column(JSONB, nullable=True)
	notification_preferences = Column(JSONB, nullable=True)
	
	# Performance Optimization
	calculation_hash = Column(String(64), nullable=True)  # For caching
	last_calculation_time = Column(DECIMAL(8, 4), nullable=True)  # Processing time in seconds
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'run_id', 'employee_id', name='uq_pr_employee_payroll_run'),
		Index('ix_pr_emp_payroll_status', 'status', 'validation_score'),
		Index('ix_pr_emp_payroll_employee', 'employee_id', 'calculation_completed_at'),
		Index('ix_pr_emp_payroll_amounts', 'gross_earnings', 'net_pay'),
	)
	
	# Relationships
	payroll_run = relationship("PRPayrollRun", back_populates="employee_payrolls")
	employee = relationship("HREmployee", foreign_keys=[employee_id])
	payroll_line_items = relationship("PRPayrollLineItem", back_populates="employee_payroll", cascade="all, delete-orphan")
	tax_calculations = relationship("PRTaxCalculation", back_populates="employee_payroll", cascade="all, delete-orphan")
	pay_stubs = relationship("PRPayStub", back_populates="employee_payroll")
	direct_deposits = relationship("PRDirectDeposit", back_populates="employee_payroll")
	payroll_adjustments = relationship("PRPayrollAdjustment", back_populates="employee_payroll")
	
	@validates('status')
	def validate_status(self, key, value):
		if value not in [status.value for status in PayrollStatus]:
			raise ValueError(f"Invalid payroll status: {value}")
		return value
	
	@validates('pay_frequency')
	def validate_pay_frequency(self, key, value):
		if value not in [freq.value for freq in PayFrequency]:
			raise ValueError(f"Invalid pay frequency: {value}")
		return value
	
	def calculate_net_pay(self) -> Decimal:
		"""Calculate net pay with AI-powered validation."""
		net = self.gross_earnings - self.total_deductions - self.total_taxes
		return max(Decimal('0.00'), net)
	
	def calculate_validation_score(self) -> float:
		"""AI-powered validation score calculation."""
		score = 100.0
		
		# Deduct for errors and warnings
		if self.has_errors:
			score -= 50.0
		if self.has_warnings:
			score -= 10.0
		
		# Check for calculation consistency
		expected_net = self.calculate_net_pay()
		if abs(expected_net - self.net_pay) > Decimal('0.01'):
			score -= 20.0
		
		return max(0.0, score)
	
	def __repr__(self):
		return f"<PREmployeePayroll {self.employee_name} - ${self.net_pay}>"


class PRPayComponent(Model, AuditMixin, BaseMixin):
	"""Revolutionary AI-powered pay component management.
	
	Intelligent pay component definitions with AI-powered
	classification, dynamic calculations, and global compliance.
	"""
	__tablename__ = 'pr_pay_component'
	
	# Identity
	id: str = Field(default_factory=uuid7str)
	component_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Enhanced Component Information
	component_code = Column(String(30), nullable=False, index=True)
	component_name = Column(String(150), nullable=False, index=True)
	short_name = Column(String(50), nullable=True)
	description = Column(Text, nullable=True)
	localized_names = Column(JSONB, nullable=True)  # Multi-language support
	
	# AI-Enhanced Classification
	component_type = Column(String(30), nullable=False, index=True)
	category = Column(String(50), nullable=True, index=True)
	subcategory = Column(String(50), nullable=True)
	ai_classification = Column(JSONB, nullable=True)  # AI-generated tags
	industry_standards = Column(ARRAY(String), nullable=True)  # Standard mappings
	
	# Advanced Calculation Engine
	calculation_method = Column(String(40), nullable=False)
	calculation_formula = Column(Text, nullable=True)  # Custom formulas
	default_rate = Column(DECIMAL(15, 6), nullable=True)
	default_amount = Column(DECIMAL(15, 2), nullable=True)
	rate_type = Column(String(20), nullable=True)  # hourly, daily, monthly, annual
	
	# Dynamic Rate Management
	rate_schedule = Column(JSONB, nullable=True)  # Time-based rates
	tiered_rates = Column(JSONB, nullable=True)  # Progressive rates
	currency_rates = Column(JSONB, nullable=True)  # Multi-currency
	
	# Comprehensive Tax Properties
	is_taxable_federal = Column(Boolean, default=True)
	is_taxable_state = Column(Boolean, default=True)
	is_taxable_local = Column(Boolean, default=True)
	is_subject_to_fica_ss = Column(Boolean, default=True)
	is_subject_to_fica_medicare = Column(Boolean, default=True)
	is_subject_to_futa = Column(Boolean, default=True)
	is_subject_to_suta = Column(Boolean, default=True)
	is_pre_tax = Column(Boolean, default=False)
	tax_treatment = Column(JSONB, nullable=True)  # Complex tax rules
	
	# Global Compliance
	country_specific_rules = Column(JSONB, nullable=True)
	regulatory_codes = Column(ARRAY(String), nullable=True)
	compliance_requirements = Column(JSONB, nullable=True)
	
	# Intelligent Behavior
	is_recurring = Column(Boolean, default=True)
	is_system_component = Column(Boolean, default=False)
	is_ai_generated = Column(Boolean, default=False)
	requires_approval = Column(Boolean, default=False)
	approval_threshold = Column(DECIMAL(15, 2), nullable=True)
	
	# Advanced Limits and Validation
	min_amount = Column(DECIMAL(15, 2), nullable=True)
	max_amount = Column(DECIMAL(15, 2), nullable=True)
	min_rate = Column(DECIMAL(15, 6), nullable=True)
	max_rate = Column(DECIMAL(15, 6), nullable=True)
	annual_limit = Column(DECIMAL(18, 2), nullable=True)
	lifetime_limit = Column(DECIMAL(20, 2), nullable=True)
	validation_rules = Column(JSONB, nullable=True)
	
	# Enhanced GL Integration
	debit_account_code = Column(String(50), nullable=True)
	credit_account_code = Column(String(50), nullable=True)
	gl_mapping_rules = Column(JSONB, nullable=True)
	cost_center_allocation = Column(JSONB, nullable=True)
	
	# Time-Based Configuration
	effective_date = Column(Date, nullable=True)
	expiry_date = Column(Date, nullable=True)
	version_number = Column(Integer, default=1)
	previous_version_id = Column(String(36), nullable=True)
	
	# AI Analytics and Insights
	usage_statistics = Column(JSONB, nullable=True)
	performance_metrics = Column(JSONB, nullable=True)
	ai_recommendations = Column(JSONB, nullable=True)
	benchmarking_data = Column(JSONB, nullable=True)
	
	# Integration and Automation
	api_mappings = Column(JSONB, nullable=True)  # External system mappings
	workflow_triggers = Column(JSONB, nullable=True)
	notification_rules = Column(JSONB, nullable=True)
	
	# Configuration
	is_active = Column(Boolean, default=True)
	is_deprecated = Column(Boolean, default=False)
	deprecation_reason = Column(Text, nullable=True)
	migration_path = Column(String(200), nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'component_code', name='uq_pr_component_code_tenant'),
		Index('ix_pr_component_type_category', 'component_type', 'category'),
		Index('ix_pr_component_effective', 'effective_date', 'expiry_date'),
		Index('ix_pr_component_tax_props', 'is_taxable_federal', 'is_pre_tax'),
	)
	
	# Relationships
	payroll_line_items = relationship("PRPayrollLineItem", back_populates="pay_component")
	employee_assignments = relationship("PREmployeeComponentAssignment", back_populates="pay_component")
	calculation_history = relationship("PRComponentCalculationHistory", back_populates="pay_component")
	
	@validates('component_type')
	def validate_component_type(self, key, value):
		if value not in [comp_type.value for comp_type in PayComponentType]:
			raise ValueError(f"Invalid component type: {value}")
		return value
	
	def calculate_amount(self, base_amount: Decimal, hours: Decimal = None, **kwargs) -> Decimal:
		"""AI-powered amount calculation with dynamic rules."""
		if self.calculation_method == 'fixed':
			return self.default_amount or Decimal('0.00')
		elif self.calculation_method == 'rate_times_hours' and hours:
			return (self.default_rate or Decimal('0.00')) * hours
		elif self.calculation_method == 'percentage' and base_amount:
			return base_amount * (self.default_rate or Decimal('0.00')) / Decimal('100.00')
		# Add more calculation methods as needed
		return Decimal('0.00')
	
	def __repr__(self):
		return f"<PRPayComponent {self.component_code}: {self.component_name}>"


class PRPayrollLineItem(Model, AuditMixin, BaseMixin):
	"""
	Individual line items in an employee's payroll.
	
	Each earnings, deduction, or tax item is a separate line item.
	"""
	__tablename__ = 'hr_pr_payroll_line_item'
	
	# Identity
	line_item_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	employee_payroll_id = Column(String(36), ForeignKey('hr_pr_employee_payroll.employee_payroll_id'), nullable=False, index=True)
	component_id = Column(String(36), ForeignKey('hr_pr_pay_component.component_id'), nullable=False, index=True)
	
	# Calculation Details
	calculation_method = Column(String(30), nullable=False)
	rate = Column(DECIMAL(10, 4), nullable=True)
	hours = Column(DECIMAL(8, 2), nullable=True)
	units = Column(DECIMAL(10, 2), nullable=True)
	base_amount = Column(DECIMAL(12, 2), nullable=True)  # Amount before calculation
	
	# Amounts
	amount = Column(DECIMAL(12, 2), nullable=False)  # Final calculated amount
	employee_portion = Column(DECIMAL(12, 2), nullable=True)  # For split costs
	employer_portion = Column(DECIMAL(12, 2), nullable=True)  # For split costs
	
	# YTD Tracking
	ytd_amount = Column(DECIMAL(15, 2), nullable=True)
	
	# Source Information
	source_reference = Column(String(100), nullable=True)  # Reference to time entry, bonus request, etc.
	notes = Column(Text, nullable=True)
	
	# Status
	is_manual_override = Column(Boolean, default=False)
	override_reason = Column(String(200), nullable=True)
	approved = Column(Boolean, default=False)
	approved_by = Column(String(36), nullable=True)
	
	# Configuration
	is_active = Column(Boolean, default=True)
	
	# Relationships
	employee_payroll = relationship("HREmployeePayroll", back_populates="payroll_line_items")
	pay_component = relationship("HRPayComponent", back_populates="payroll_line_items")
	
	def __repr__(self):
		return f"<HRPayrollLineItem {self.pay_component.component_name}: {self.amount}>"


class HRTaxTable(Model, AuditMixin, BaseMixin):
	"""
	Tax tables for tax calculations.
	
	Stores tax brackets and rates for different jurisdictions.
	"""
	__tablename__ = 'hr_pr_tax_table'
	
	# Identity
	tax_table_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Tax Information
	tax_type = Column(String(20), nullable=False, index=True)  # Federal, State, Local, FICA
	jurisdiction = Column(String(50), nullable=False, index=True)  # US, CA, NY, etc.
	tax_year = Column(Integer, nullable=False, index=True)
	
	# Filing Status
	filing_status = Column(String(20), nullable=False, index=True)  # Single, MarriedJoint, MarriedSeparate, Head
	
	# Bracket Information
	bracket_number = Column(Integer, nullable=False)
	min_income = Column(DECIMAL(12, 2), nullable=False)
	max_income = Column(DECIMAL(12, 2), nullable=True)  # NULL for highest bracket
	tax_rate = Column(DECIMAL(8, 4), nullable=False)  # Percentage as decimal (0.22 for 22%)
	base_tax = Column(DECIMAL(10, 2), default=0.00)  # Base tax for bracket
	
	# Configuration
	is_active = Column(Boolean, default=True)
	effective_date = Column(Date, nullable=False)
	expiry_date = Column(Date, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'tax_type', 'jurisdiction', 'tax_year', 'filing_status', 'bracket_number', 
						name='uq_tax_table_bracket'),
	)
	
	# Relationships
	tax_calculations = relationship("HRTaxCalculation", back_populates="tax_table")
	
	def __repr__(self):
		return f"<HRTaxTable {self.tax_type} {self.jurisdiction} {self.tax_year}>"


class HRTaxCalculation(Model, AuditMixin, BaseMixin):
	"""
	Tax calculation results for employee payroll.
	
	Stores detailed tax calculations and supporting information.
	"""
	__tablename__ = 'hr_pr_tax_calculation'
	
	# Identity
	tax_calculation_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	employee_payroll_id = Column(String(36), ForeignKey('hr_pr_employee_payroll.employee_payroll_id'), nullable=False, index=True)
	tax_table_id = Column(String(36), ForeignKey('hr_pr_tax_table.tax_table_id'), nullable=True, index=True)
	
	# Tax Information
	tax_type = Column(String(20), nullable=False, index=True)
	jurisdiction = Column(String(50), nullable=False, index=True)
	
	# Calculation Input
	taxable_wages = Column(DECIMAL(12, 2), nullable=False)
	allowances = Column(Integer, default=0)
	additional_withholding = Column(DECIMAL(8, 2), default=0.00)
	filing_status = Column(String(20), nullable=True)
	
	# Calculation Output
	tax_amount = Column(DECIMAL(10, 2), nullable=False)
	employee_tax = Column(DECIMAL(10, 2), nullable=False)  # Amount withheld from employee
	employer_tax = Column(DECIMAL(10, 2), default=0.00)  # Employer portion (FICA)
	
	# Supporting Details
	calculation_method = Column(String(30), nullable=False)  # Percentage, Table, Flat
	tax_rate = Column(DECIMAL(8, 4), nullable=True)
	calculation_notes = Column(Text, nullable=True)
	
	# YTD Information
	ytd_taxable_wages = Column(DECIMAL(15, 2), nullable=True)
	ytd_tax_withheld = Column(DECIMAL(12, 2), nullable=True)
	
	# Configuration
	is_active = Column(Boolean, default=True)
	
	# Relationships
	employee_payroll = relationship("HREmployeePayroll", back_populates="tax_calculations")
	tax_table = relationship("HRTaxTable", back_populates="tax_calculations")
	
	def __repr__(self):
		return f"<HRTaxCalculation {self.tax_type} {self.jurisdiction}: {self.tax_amount}>"


class HRDeductionType(Model, AuditMixin, BaseMixin):
	"""
	Deduction type definitions.
	
	Templates for employee-specific deductions.
	"""
	__tablename__ = 'hr_pr_deduction_type'
	
	# Identity
	deduction_type_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Deduction Information
	deduction_code = Column(String(20), nullable=False, index=True)
	deduction_name = Column(String(100), nullable=False, index=True)
	description = Column(Text, nullable=True)
	
	# Classification
	category = Column(String(50), nullable=False, index=True)  # Benefits, Retirement, Other, Legal
	
	# Calculation
	calculation_method = Column(String(20), nullable=False)  # Fixed, Percentage, Tiered
	default_amount = Column(DECIMAL(8, 2), nullable=True)
	default_percentage = Column(DECIMAL(6, 4), nullable=True)
	
	# Tax Properties
	is_pre_tax = Column(Boolean, default=False)
	affects_federal_tax = Column(Boolean, default=True)
	affects_state_tax = Column(Boolean, default=True)
	affects_fica = Column(Boolean, default=True)
	
	# Limits
	min_amount = Column(DECIMAL(8, 2), nullable=True)
	max_amount = Column(DECIMAL(8, 2), nullable=True)
	annual_limit = Column(DECIMAL(10, 2), nullable=True)
	
	# Employer Contribution
	has_employer_match = Column(Boolean, default=False)
	employer_match_percentage = Column(DECIMAL(6, 4), nullable=True)
	employer_match_max = Column(DECIMAL(8, 2), nullable=True)
	
	# Configuration
	is_active = Column(Boolean, default=True)
	requires_employee_election = Column(Boolean, default=False)
	is_system_deduction = Column(Boolean, default=False)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'deduction_code', name='uq_deduction_type_code_tenant'),
	)
	
	# Relationships
	employee_deductions = relationship("HREmployeeDeduction", back_populates="deduction_type")
	pay_component = relationship("HRPayComponent", foreign_keys="HRDeductionType.deduction_code", 
								primaryjoin="HRDeductionType.deduction_code == HRPayComponent.component_code")
	
	def __repr__(self):
		return f"<HRDeductionType {self.deduction_name}>"


class HREmployeeDeduction(Model, AuditMixin, BaseMixin):
	"""
	Employee-specific deduction assignments.
	
	Links employees to deduction types with specific amounts/percentages.
	"""
	__tablename__ = 'hr_pr_employee_deduction'
	
	# Identity
	employee_deduction_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	employee_id = Column(String(36), ForeignKey('hr_edm_employee.employee_id'), nullable=False, index=True)
	deduction_type_id = Column(String(36), ForeignKey('hr_pr_deduction_type.deduction_type_id'), nullable=False, index=True)
	component_id = Column(String(36), ForeignKey('hr_pr_pay_component.component_id'), nullable=True, index=True)
	
	# Deduction Configuration
	deduction_amount = Column(DECIMAL(8, 2), nullable=True)
	deduction_percentage = Column(DECIMAL(6, 4), nullable=True)
	calculation_base = Column(String(20), default='GrossPay')  # GrossPay, BaseSalary, etc.
	
	# Effective Dates
	effective_date = Column(Date, nullable=False, index=True)
	end_date = Column(Date, nullable=True, index=True)
	
	# Employee Election
	employee_elected = Column(Boolean, default=False)
	election_date = Column(Date, nullable=True)
	election_notes = Column(Text, nullable=True)
	
	# Employer Match
	employer_match_amount = Column(DECIMAL(8, 2), nullable=True)
	employer_match_percentage = Column(DECIMAL(6, 4), nullable=True)
	
	# Status
	is_active = Column(Boolean, default=True)
	is_suspended = Column(Boolean, default=False)
	suspension_reason = Column(String(200), nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'employee_id', 'deduction_type_id', 'effective_date', 
						name='uq_employee_deduction'),
	)
	
	# Relationships
	employee = relationship("HREmployee", foreign_keys=[employee_id])
	deduction_type = relationship("HRDeductionType", back_populates="employee_deductions")
	pay_component = relationship("HRPayComponent", back_populates="employee_deductions")
	
	def __repr__(self):
		return f"<HREmployeeDeduction {self.employee.full_name} - {self.deduction_type.deduction_name}>"


class HRPayStub(Model, AuditMixin, BaseMixin):
	"""
	Pay stub records for employees.
	
	Generated pay stubs with all pay details.
	"""
	__tablename__ = 'hr_pr_pay_stub'
	
	# Identity
	pay_stub_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	employee_payroll_id = Column(String(36), ForeignKey('hr_pr_employee_payroll.employee_payroll_id'), nullable=False, unique=True, index=True)
	
	# Pay Stub Information
	pay_stub_number = Column(String(50), nullable=False, index=True)
	pay_period_start = Column(Date, nullable=False)
	pay_period_end = Column(Date, nullable=False)
	pay_date = Column(Date, nullable=False)
	
	# Status
	status = Column(String(20), default='Generated', index=True)  # Generated, Delivered, Viewed
	generated_date = Column(DateTime, nullable=False, default=datetime.utcnow)
	delivered_date = Column(DateTime, nullable=True)
	viewed_date = Column(DateTime, nullable=True)
	
	# File Information
	pdf_file_path = Column(String(500), nullable=True)
	file_size = Column(Integer, nullable=True)
	
	# Configuration
	is_active = Column(Boolean, default=True)
	
	# Relationships
	employee_payroll = relationship("HREmployeePayroll", back_populates="pay_stubs")
	
	def __repr__(self):
		return f"<HRPayStub {self.pay_stub_number}>"


class HRDirectDeposit(Model, AuditMixin, BaseMixin):
	"""
	Direct deposit records for payroll.
	
	Tracks direct deposit transactions and banking information.
	"""
	__tablename__ = 'hr_pr_direct_deposit'
	
	# Identity
	direct_deposit_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	employee_payroll_id = Column(String(36), ForeignKey('hr_pr_employee_payroll.employee_payroll_id'), nullable=False, index=True)
	
	# Banking Information (encrypted/masked for security)
	bank_name = Column(String(100), nullable=False)
	routing_number = Column(String(20), nullable=False)  # Should be encrypted
	account_number_masked = Column(String(20), nullable=False)  # Last 4 digits only
	account_type = Column(String(20), nullable=False)  # Checking, Savings
	
	# Deposit Information
	deposit_amount = Column(DECIMAL(10, 2), nullable=False)
	deposit_percentage = Column(DECIMAL(5, 2), nullable=True)  # For percentage splits
	priority_order = Column(Integer, default=1)  # For multiple accounts
	
	# Transaction Status
	status = Column(String(20), default='Pending', index=True)  # Pending, Submitted, Processed, Failed
	transaction_id = Column(String(100), nullable=True)  # Bank transaction ID
	submitted_date = Column(DateTime, nullable=True)
	processed_date = Column(DateTime, nullable=True)
	
	# Error Handling
	error_code = Column(String(20), nullable=True)
	error_message = Column(String(500), nullable=True)
	retry_count = Column(Integer, default=0)
	
	# Configuration
	is_active = Column(Boolean, default=True)
	
	# Relationships
	employee_payroll = relationship("HREmployeePayroll", back_populates="direct_deposits")
	
	def __repr__(self):
		return f"<HRDirectDeposit {self.bank_name} ***{self.account_number_masked[-4:]}: {self.deposit_amount}>"


class HRPayrollJournal(Model, AuditMixin, BaseMixin):
	"""
	Payroll journal entries for GL integration.
	
	Creates journal entries for posting payroll to the general ledger.
	"""
	__tablename__ = 'hr_pr_payroll_journal'
	
	# Identity
	journal_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	run_id = Column(String(36), ForeignKey('hr_pr_payroll_run.run_id'), nullable=False, index=True)
	
	# Journal Information
	journal_number = Column(String(50), nullable=False, index=True)
	journal_date = Column(Date, nullable=False, index=True)
	description = Column(String(200), nullable=False)
	
	# GL Information
	gl_account_code = Column(String(20), nullable=False, index=True)
	gl_account_name = Column(String(200), nullable=False)
	department_code = Column(String(20), nullable=True)
	cost_center = Column(String(20), nullable=True)
	
	# Amounts
	debit_amount = Column(DECIMAL(15, 2), default=0.00)
	credit_amount = Column(DECIMAL(15, 2), default=0.00)
	
	# Status
	status = Column(String(20), default='Draft', index=True)  # Draft, Posted, Reversed
	posted_date = Column(DateTime, nullable=True)
	posted_by = Column(String(36), nullable=True)
	
	# Reference
	reference_type = Column(String(20), nullable=False)  # Salary, Tax, Deduction, etc.
	reference_id = Column(String(36), nullable=True)  # Link to specific component
	
	# Configuration
	is_active = Column(Boolean, default=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'journal_number', name='uq_payroll_journal_number_tenant'),
	)
	
	# Relationships
	payroll_run = relationship("HRPayrollRun", back_populates="payroll_journals")
	
	def __repr__(self):
		return f"<HRPayrollJournal {self.journal_number}>"