"""
Budgeting & Forecasting Models

Database models for the Budgeting & Forecasting sub-capability including
budget creation, scenario planning, variance analysis, and forecasting.
"""

from datetime import datetime, date
from typing import Dict, List, Any, Optional
from decimal import Decimal
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, Date, DECIMAL, ForeignKey, UniqueConstraint, JSON
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str
import json

from ...auth_rbac.models import BaseMixin, AuditMixin, Model


class CFBFBudgetScenario(Model, AuditMixin, BaseMixin):
	"""
	Budget scenarios for different planning assumptions.
	
	Allows creation of multiple budget scenarios (base, optimistic, pessimistic)
	for comprehensive financial planning and risk analysis.
	"""
	__tablename__ = 'cf_bf_budget_scenario'
	
	# Identity
	scenario_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Scenario Information
	scenario_code = Column(String(20), nullable=False, index=True)
	scenario_name = Column(String(100), nullable=False)
	description = Column(Text, nullable=True)
	
	# Properties
	is_active = Column(Boolean, default=True)
	is_default = Column(Boolean, default=False)
	probability = Column(DECIMAL(5, 2), default=0.00)  # Probability percentage
	
	# Configuration
	assumptions = Column(JSON, nullable=True)  # Scenario assumptions
	parameters = Column(JSON, nullable=True)   # Calculation parameters
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'scenario_code', name='uq_scenario_code_tenant'),
	)
	
	# Relationships
	budgets = relationship("CFBFBudget", back_populates="scenario")
	forecasts = relationship("CFBFForecast", back_populates="scenario")
	
	def __repr__(self):
		return f"<CFBFBudgetScenario {self.scenario_code} - {self.scenario_name}>"


class CFBFTemplate(Model, AuditMixin, BaseMixin):
	"""
	Budget templates for standardized budget creation.
	
	Provides reusable templates with predefined accounts, allocations,
	and calculation formulas for consistent budgeting.
	"""
	__tablename__ = 'cf_bf_template'
	
	# Identity
	template_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Template Information
	template_code = Column(String(20), nullable=False, index=True)
	template_name = Column(String(100), nullable=False)
	description = Column(Text, nullable=True)
	category = Column(String(50), nullable=True)  # Revenue, Expense, etc.
	
	# Properties
	is_active = Column(Boolean, default=True)
	is_system = Column(Boolean, default=False)
	
	# Template Structure
	template_data = Column(JSON, nullable=True)    # Template structure
	default_accounts = Column(JSON, nullable=True) # Default GL accounts
	calculation_rules = Column(JSON, nullable=True) # Calculation formulas
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'template_code', name='uq_template_code_tenant'),
	)
	
	# Relationships
	budgets = relationship("CFBFBudget", back_populates="template")
	
	def __repr__(self):
		return f"<CFBFTemplate {self.template_code} - {self.template_name}>"


class CFBFDrivers(Model, AuditMixin, BaseMixin):
	"""
	Budget drivers and assumptions for driver-based budgeting.
	
	Key business metrics and assumptions that drive budget calculations
	like headcount, sales volume, inflation rates, etc.
	"""
	__tablename__ = 'cf_bf_drivers'
	
	# Identity
	driver_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Driver Information
	driver_code = Column(String(20), nullable=False, index=True)
	driver_name = Column(String(100), nullable=False)
	description = Column(Text, nullable=True)
	category = Column(String(50), nullable=True)  # Volume, Price, Rate, etc.
	
	# Properties
	data_type = Column(String(20), default='Numeric')  # Numeric, Percentage, Boolean
	unit_of_measure = Column(String(20), nullable=True)  # Units, %, Count, etc.
	is_active = Column(Boolean, default=True)
	
	# Values by Period
	base_value = Column(DECIMAL(15, 4), nullable=True)
	growth_rate = Column(DECIMAL(5, 2), default=0.00)  # Annual growth rate
	seasonal_factors = Column(JSON, nullable=True)     # Monthly seasonal adjustments
	
	# Historical Data
	historical_values = Column(JSON, nullable=True)    # Historical driver values
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'driver_code', name='uq_driver_code_tenant'),
	)
	
	# Relationships
	budget_lines = relationship("CFBFBudgetLine", back_populates="driver")
	forecast_lines = relationship("CFBFForecastLine", back_populates="driver")
	
	def __repr__(self):
		return f"<CFBFDrivers {self.driver_code} - {self.driver_name}>"
	
	def calculate_period_value(self, period: int, base_period: int = 1) -> Decimal:
		"""Calculate driver value for a specific period"""
		if not self.base_value:
			return Decimal('0.00')
		
		# Apply growth rate
		periods_growth = (period - base_period) / 12.0  # Assuming monthly periods
		growth_factor = (1 + self.growth_rate / 100) ** periods_growth
		
		period_value = self.base_value * Decimal(str(growth_factor))
		
		# Apply seasonal adjustment if available
		if self.seasonal_factors and isinstance(self.seasonal_factors, dict):
			month = ((period - 1) % 12) + 1
			seasonal_factor = self.seasonal_factors.get(str(month), 1.0)
			period_value *= Decimal(str(seasonal_factor))
		
		return period_value


class CFBFBudget(Model, AuditMixin, BaseMixin):
	"""
	Budget master/header records.
	
	Top-level budget entity containing budget metadata, status,
	and overall control information for budget management.
	"""
	__tablename__ = 'cf_bf_budget'
	
	# Identity
	budget_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Budget Information
	budget_number = Column(String(50), nullable=False, index=True)
	budget_name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	
	# Period Information
	fiscal_year = Column(Integer, nullable=False, index=True)
	budget_period = Column(String(20), default='Annual')  # Annual, Quarterly, Monthly
	start_date = Column(Date, nullable=False, index=True)
	end_date = Column(Date, nullable=False, index=True)
	
	# Relationships
	scenario_id = Column(String(36), ForeignKey('cf_bf_budget_scenario.scenario_id'), nullable=True, index=True)
	template_id = Column(String(36), ForeignKey('cf_bf_template.template_id'), nullable=True, index=True)
	
	# Status and Control
	status = Column(String(20), default='Draft', index=True)  # Draft, Submitted, Approved, Active, Locked
	version_number = Column(Integer, default=1)
	is_active = Column(Boolean, default=True)
	is_consolidated = Column(Boolean, default=False)
	
	# Approval Workflow
	requires_approval = Column(Boolean, default=True)
	approval_status = Column(String(20), default='Pending')  # Pending, Approved, Rejected
	submitted_date = Column(DateTime, nullable=True)
	submitted_by = Column(String(36), nullable=True)
	approved_date = Column(DateTime, nullable=True)
	approved_by = Column(String(36), nullable=True)
	
	# Totals
	total_revenue = Column(DECIMAL(15, 2), default=0.00)
	total_expenses = Column(DECIMAL(15, 2), default=0.00)
	net_income = Column(DECIMAL(15, 2), default=0.00)
	line_count = Column(Integer, default=0)
	
	# Configuration
	currency_code = Column(String(3), default='USD')
	consolidation_method = Column(String(20), nullable=True)
	
	# Notes and Comments
	notes = Column(Text, nullable=True)
	assumptions = Column(JSON, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'budget_number', name='uq_budget_number_tenant'),
	)
	
	# Relationships
	scenario = relationship("CFBFBudgetScenario", back_populates="budgets")
	template = relationship("CFBFTemplate", back_populates="budgets")
	lines = relationship("CFBFBudgetLine", back_populates="budget", cascade="all, delete-orphan")
	versions = relationship("CFBFBudgetVersion", back_populates="budget")
	approvals = relationship("CFBFApproval", back_populates="budget")
	variance_analyses = relationship("CFBFActualVsBudget", back_populates="budget")
	
	def __repr__(self):
		return f"<CFBFBudget {self.budget_number} - {self.budget_name}>"
	
	def calculate_totals(self):
		"""Recalculate budget totals from lines"""
		revenue_total = sum(line.total_amount for line in self.lines 
						   if line.account and line.account.account_type.type_code == 'R')
		expense_total = sum(line.total_amount for line in self.lines 
						   if line.account and line.account.account_type.type_code == 'X')
		
		self.total_revenue = revenue_total
		self.total_expenses = expense_total
		self.net_income = revenue_total - expense_total
		self.line_count = len(self.lines)
	
	def can_approve(self) -> bool:
		"""Check if budget can be approved"""
		return (
			self.status == 'Submitted' and
			self.requires_approval and
			self.approval_status == 'Pending'
		)
	
	def submit_for_approval(self, user_id: str):
		"""Submit budget for approval"""
		if self.status == 'Draft':
			self.status = 'Submitted'
			self.submitted_date = datetime.utcnow()
			self.submitted_by = user_id
			self.approval_status = 'Pending'
	
	def approve_budget(self, user_id: str):
		"""Approve the budget"""
		if self.can_approve():
			self.status = 'Approved'
			self.approval_status = 'Approved'
			self.approved_date = datetime.utcnow()
			self.approved_by = user_id


class CFBFBudgetLine(Model, AuditMixin, BaseMixin):
	"""
	Detailed budget line items.
	
	Individual budget line items with GL account mapping,
	period-by-period amounts, and driver-based calculations.
	"""
	__tablename__ = 'cf_bf_budget_line'
	
	# Identity
	budget_line_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	budget_id = Column(String(36), ForeignKey('cf_bf_budget.budget_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Line Information
	line_number = Column(Integer, nullable=False)
	description = Column(Text, nullable=True)
	
	# Account Information
	account_id = Column(String(36), ForeignKey('cf_gl_account.account_id'), nullable=False, index=True)
	
	# Driver-Based Budgeting
	driver_id = Column(String(36), ForeignKey('cf_bf_drivers.driver_id'), nullable=True, index=True)
	calculation_method = Column(String(20), default='Manual')  # Manual, Driver-Based, Formula
	calculation_formula = Column(Text, nullable=True)
	
	# Amounts
	annual_amount = Column(DECIMAL(15, 2), default=0.00)
	total_amount = Column(DECIMAL(15, 2), default=0.00)
	
	# Period Distribution
	period_amounts = Column(JSON, nullable=True)  # Monthly/quarterly amounts
	distribution_method = Column(String(20), default='Even')  # Even, Seasonal, Custom
	
	# Dimensions
	cost_center = Column(String(20), nullable=True)
	department = Column(String(20), nullable=True)
	project = Column(String(20), nullable=True)
	
	# Notes
	notes = Column(Text, nullable=True)
	assumptions = Column(Text, nullable=True)
	
	# Relationships
	budget = relationship("CFBFBudget", back_populates="lines")
	account = relationship("CFGLAccount")
	driver = relationship("CFBFDrivers", back_populates="budget_lines")
	allocations = relationship("CFBFAllocation", back_populates="budget_line")
	
	def __repr__(self):
		return f"<CFBFBudgetLine {self.line_number}: {self.account.account_code if self.account else 'No Account'}>"
	
	def calculate_driver_amount(self) -> Decimal:
		"""Calculate amount based on driver values"""
		if not self.driver or self.calculation_method != 'Driver-Based':
			return self.annual_amount
		
		# Simple calculation - can be enhanced with complex formulas
		driver_value = self.driver.base_value or Decimal('0.00')
		return driver_value * self.annual_amount
	
	def distribute_amount_by_periods(self, periods: int = 12) -> Dict[int, Decimal]:
		"""Distribute annual amount across periods"""
		if self.period_amounts:
			return {int(k): Decimal(str(v)) for k, v in self.period_amounts.items()}
		
		if self.distribution_method == 'Even':
			period_amount = self.total_amount / periods
			return {i+1: period_amount for i in range(periods)}
		
		# For seasonal or custom distribution, would need more complex logic
		return {i+1: Decimal('0.00') for i in range(periods)}


class CFBFBudgetVersion(Model, AuditMixin, BaseMixin):
	"""
	Budget versioning and revision control.
	
	Tracks budget versions, revisions, and changes throughout
	the budget cycle for audit trail and comparison.
	"""
	__tablename__ = 'cf_bf_budget_version'
	
	# Identity
	version_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	budget_id = Column(String(36), ForeignKey('cf_bf_budget.budget_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Version Information
	version_number = Column(Integer, nullable=False)
	version_name = Column(String(100), nullable=True)
	description = Column(Text, nullable=True)
	
	# Status
	is_current = Column(Boolean, default=False)
	is_baseline = Column(Boolean, default=False)
	
	# Version Data (snapshot)
	version_data = Column(JSON, nullable=True)  # Complete budget snapshot
	change_summary = Column(JSON, nullable=True)  # Summary of changes
	
	# Approval Information
	approval_status = Column(String(20), default='Pending')
	approved_date = Column(DateTime, nullable=True)
	approved_by = Column(String(36), nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('budget_id', 'version_number', name='uq_budget_version'),
	)
	
	# Relationships
	budget = relationship("CFBFBudget", back_populates="versions")
	
	def __repr__(self):
		return f"<CFBFBudgetVersion {self.budget.budget_number} v{self.version_number}>"


class CFBFForecast(Model, AuditMixin, BaseMixin):
	"""
	Forecast master/header records.
	
	Financial forecasts for predicting future performance
	based on trends, drivers, and business assumptions.
	"""
	__tablename__ = 'cf_bf_forecast'
	
	# Identity
	forecast_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Forecast Information
	forecast_number = Column(String(50), nullable=False, index=True)
	forecast_name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	
	# Forecast Type
	forecast_type = Column(String(20), default='Rolling')  # Rolling, Static, Budget-Based
	forecast_method = Column(String(20), default='Trend')  # Trend, Driver, Statistical
	
	# Period Information
	forecast_date = Column(Date, nullable=False, index=True)  # As of date
	start_date = Column(Date, nullable=False)
	end_date = Column(Date, nullable=False)
	periods_ahead = Column(Integer, default=12)  # Number of periods to forecast
	
	# Relationships
	scenario_id = Column(String(36), ForeignKey('cf_bf_budget_scenario.scenario_id'), nullable=True, index=True)
	base_budget_id = Column(String(36), ForeignKey('cf_bf_budget.budget_id'), nullable=True, index=True)
	
	# Status
	status = Column(String(20), default='Draft', index=True)  # Draft, Active, Archived
	confidence_level = Column(DECIMAL(5, 2), default=75.00)  # Forecast confidence %
	
	# Algorithm Configuration
	algorithm_type = Column(String(20), default='Linear')  # Linear, Exponential, Seasonal
	algorithm_parameters = Column(JSON, nullable=True)
	
	# Totals
	total_forecast_revenue = Column(DECIMAL(15, 2), default=0.00)
	total_forecast_expenses = Column(DECIMAL(15, 2), default=0.00)
	forecast_net_income = Column(DECIMAL(15, 2), default=0.00)
	
	# Generation Info
	generated_date = Column(DateTime, nullable=True)
	generated_by = Column(String(36), nullable=True)
	last_updated = Column(DateTime, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'forecast_number', name='uq_forecast_number_tenant'),
	)
	
	# Relationships
	scenario = relationship("CFBFBudgetScenario", back_populates="forecasts")
	base_budget = relationship("CFBFBudget")
	lines = relationship("CFBFForecastLine", back_populates="forecast", cascade="all, delete-orphan")
	
	def __repr__(self):
		return f"<CFBFForecast {self.forecast_number} - {self.forecast_name}>"
	
	def calculate_totals(self):
		"""Recalculate forecast totals from lines"""
		revenue_total = sum(line.forecast_amount for line in self.lines 
						   if line.account and line.account.account_type.type_code == 'R')
		expense_total = sum(line.forecast_amount for line in self.lines 
						   if line.account and line.account.account_type.type_code == 'X')
		
		self.total_forecast_revenue = revenue_total
		self.total_forecast_expenses = expense_total
		self.forecast_net_income = revenue_total - expense_total
	
	def generate_forecast(self, user_id: str):
		"""Generate forecast using selected algorithm"""
		self.status = 'Active'
		self.generated_date = datetime.utcnow()
		self.generated_by = user_id
		self.last_updated = datetime.utcnow()
		
		# Algorithm implementation would go here
		# This is a placeholder for the actual forecasting logic


class CFBFForecastLine(Model, AuditMixin, BaseMixin):
	"""
	Detailed forecast line items.
	
	Individual forecast line items with GL account mapping,
	statistical calculations, and trend analysis.
	"""
	__tablename__ = 'cf_bf_forecast_line'
	
	# Identity
	forecast_line_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	forecast_id = Column(String(36), ForeignKey('cf_bf_forecast.forecast_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Line Information
	line_number = Column(Integer, nullable=False)
	description = Column(Text, nullable=True)
	
	# Account Information
	account_id = Column(String(36), ForeignKey('cf_gl_account.account_id'), nullable=False, index=True)
	
	# Forecast Calculation
	driver_id = Column(String(36), ForeignKey('cf_bf_drivers.driver_id'), nullable=True, index=True)
	calculation_method = Column(String(20), default='Trend')  # Trend, Driver, Statistical, Manual
	
	# Historical Base
	historical_periods = Column(Integer, default=12)  # Periods of historical data used
	base_amount = Column(DECIMAL(15, 2), default=0.00)  # Starting amount
	trend_factor = Column(DECIMAL(10, 6), default=1.000000)  # Trend multiplier
	
	# Forecast Amount
	forecast_amount = Column(DECIMAL(15, 2), default=0.00)
	confidence_interval = Column(JSON, nullable=True)  # Upper/lower bounds
	
	# Period Breakdown
	period_forecasts = Column(JSON, nullable=True)  # Period-by-period forecasts
	
	# Statistical Measures
	r_squared = Column(DECIMAL(5, 4), nullable=True)  # Correlation coefficient
	std_error = Column(DECIMAL(15, 2), nullable=True)  # Standard error
	
	# Notes
	assumptions = Column(Text, nullable=True)
	methodology_notes = Column(Text, nullable=True)
	
	# Relationships
	forecast = relationship("CFBFForecast", back_populates="lines")
	account = relationship("CFGLAccount")
	driver = relationship("CFBFDrivers", back_populates="forecast_lines")
	
	def __repr__(self):
		return f"<CFBFForecastLine {self.line_number}: {self.account.account_code if self.account else 'No Account'}>"
	
	def calculate_trend_forecast(self, periods: int = 12) -> Dict[int, Decimal]:
		"""Calculate forecast using trend analysis"""
		if not self.base_amount or not self.trend_factor:
			return {i+1: Decimal('0.00') for i in range(periods)}
		
		forecasts = {}
		current_amount = self.base_amount
		
		for period in range(1, periods + 1):
			current_amount *= self.trend_factor
			forecasts[period] = current_amount
		
		return forecasts


class CFBFActualVsBudget(Model, AuditMixin, BaseMixin):
	"""
	Actual vs Budget variance analysis.
	
	Compares actual results to budget for variance analysis,
	performance measurement, and budget control.
	"""
	__tablename__ = 'cf_bf_actual_vs_budget'
	
	# Identity
	variance_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Reference Information
	budget_id = Column(String(36), ForeignKey('cf_bf_budget.budget_id'), nullable=False, index=True)
	account_id = Column(String(36), ForeignKey('cf_gl_account.account_id'), nullable=False, index=True)
	
	# Period Information
	analysis_date = Column(Date, nullable=False, index=True)
	fiscal_year = Column(Integer, nullable=False, index=True)
	period_number = Column(Integer, nullable=False, index=True)
	
	# Amounts
	budget_amount = Column(DECIMAL(15, 2), default=0.00)
	actual_amount = Column(DECIMAL(15, 2), default=0.00)
	variance_amount = Column(DECIMAL(15, 2), default=0.00)  # Actual - Budget
	variance_percent = Column(DECIMAL(10, 4), default=0.0000)
	
	# Year-to-Date
	ytd_budget_amount = Column(DECIMAL(15, 2), default=0.00)
	ytd_actual_amount = Column(DECIMAL(15, 2), default=0.00)
	ytd_variance_amount = Column(DECIMAL(15, 2), default=0.00)
	ytd_variance_percent = Column(DECIMAL(10, 4), default=0.0000)
	
	# Analysis
	is_favorable = Column(Boolean, nullable=True)  # True if favorable, False if unfavorable
	is_significant = Column(Boolean, default=False)  # Exceeds variance threshold
	variance_category = Column(String(20), nullable=True)  # Revenue, Expense, etc.
	
	# Alert Information
	alert_level = Column(String(20), nullable=True)  # Info, Warning, Critical
	alert_sent = Column(Boolean, default=False)
	alert_date = Column(DateTime, nullable=True)
	
	# Notes
	variance_explanation = Column(Text, nullable=True)
	corrective_action = Column(Text, nullable=True)
	
	# Generation Info
	generated_date = Column(DateTime, default=datetime.utcnow)
	generated_by = Column(String(36), nullable=False)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'budget_id', 'account_id', 'analysis_date', name='uq_variance_analysis'),
	)
	
	# Relationships
	budget = relationship("CFBFBudget", back_populates="variance_analyses")
	account = relationship("CFGLAccount")
	
	def __repr__(self):
		return f"<CFBFActualVsBudget {self.account.account_code if self.account else 'No Account'} - {self.analysis_date}>"
	
	def calculate_variance(self):
		"""Calculate variance amounts and percentages"""
		self.variance_amount = self.actual_amount - self.budget_amount
		self.ytd_variance_amount = self.ytd_actual_amount - self.ytd_budget_amount
		
		# Calculate percentages
		if self.budget_amount != 0:
			self.variance_percent = (self.variance_amount / self.budget_amount) * 100
		
		if self.ytd_budget_amount != 0:
			self.ytd_variance_percent = (self.ytd_variance_amount / self.ytd_budget_amount) * 100
		
		# Determine if favorable (depends on account type)
		if self.account:
			if self.account.account_type.type_code == 'R':  # Revenue
				self.is_favorable = self.variance_amount > 0
			elif self.account.account_type.type_code == 'X':  # Expense
				self.is_favorable = self.variance_amount < 0
		
		# Check significance thresholds (would be configurable)
		threshold_amount = Decimal('10000.00')
		threshold_percent = Decimal('10.00')
		
		self.is_significant = (
			abs(self.variance_amount) > threshold_amount or
			abs(self.variance_percent) > threshold_percent
		)
	
	def determine_alert_level(self):
		"""Determine appropriate alert level"""
		if not self.is_significant:
			self.alert_level = 'Info'
		elif abs(self.variance_percent) > 25:
			self.alert_level = 'Critical'
		else:
			self.alert_level = 'Warning'


class CFBFApproval(Model, AuditMixin, BaseMixin):
	"""
	Budget approval workflow tracking.
	
	Manages budget approval workflows with multiple approvers,
	approval hierarchy, and approval status tracking.
	"""
	__tablename__ = 'cf_bf_approval'
	
	# Identity
	approval_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Reference Information
	budget_id = Column(String(36), ForeignKey('cf_bf_budget.budget_id'), nullable=False, index=True)
	
	# Approval Information
	approval_level = Column(Integer, nullable=False)  # 1, 2, 3, etc.
	approver_id = Column(String(36), nullable=False, index=True)
	approver_role = Column(String(50), nullable=True)  # Manager, Director, CFO, etc.
	
	# Status
	status = Column(String(20), default='Pending', index=True)  # Pending, Approved, Rejected, Skipped
	required = Column(Boolean, default=True)
	
	# Approval Details
	approved_date = Column(DateTime, nullable=True)
	rejected_date = Column(DateTime, nullable=True)
	comments = Column(Text, nullable=True)
	conditions = Column(Text, nullable=True)  # Conditional approval notes
	
	# Workflow
	previous_approval_id = Column(String(36), nullable=True)  # Chain of approvals
	next_approval_id = Column(String(36), nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('budget_id', 'approval_level', 'approver_id', name='uq_budget_approval'),
	)
	
	# Relationships
	budget = relationship("CFBFBudget", back_populates="approvals")
	
	def __repr__(self):
		return f"<CFBFApproval Level {self.approval_level} - {self.status}>"
	
	def can_approve(self) -> bool:
		"""Check if this approval can be processed"""
		return self.status == 'Pending' and self.required
	
	def approve(self, comments: str = None):
		"""Approve this level"""
		if self.can_approve():
			self.status = 'Approved'
			self.approved_date = datetime.utcnow()
			if comments:
				self.comments = comments
	
	def reject(self, comments: str = None):
		"""Reject this approval"""
		if self.can_approve():
			self.status = 'Rejected'
			self.rejected_date = datetime.utcnow()
			if comments:
				self.comments = comments


class CFBFAllocation(Model, AuditMixin, BaseMixin):
	"""
	Budget allocation rules and calculations.
	
	Defines how budget amounts are allocated across cost centers,
	departments, projects, or other dimensions.
	"""
	__tablename__ = 'cf_bf_allocation'
	
	# Identity
	allocation_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Source Information
	budget_line_id = Column(String(36), ForeignKey('cf_bf_budget_line.budget_line_id'), nullable=False, index=True)
	
	# Allocation Target
	target_type = Column(String(20), nullable=False)  # CostCenter, Department, Project
	target_code = Column(String(20), nullable=False)
	target_name = Column(String(100), nullable=True)
	
	# Allocation Method
	allocation_method = Column(String(20), default='Percentage')  # Percentage, Amount, Driver, Formula
	allocation_basis = Column(String(50), nullable=True)  # Headcount, Revenue, Square Footage, etc.
	
	# Allocation Values
	allocation_percentage = Column(DECIMAL(10, 6), default=0.000000)
	allocation_amount = Column(DECIMAL(15, 2), default=0.00)
	driver_factor = Column(DECIMAL(15, 6), nullable=True)
	
	# Formula-Based Allocation
	allocation_formula = Column(Text, nullable=True)
	formula_variables = Column(JSON, nullable=True)
	
	# Status
	is_active = Column(Boolean, default=True)
	effective_date = Column(Date, nullable=True)
	expiration_date = Column(Date, nullable=True)
	
	# Notes
	description = Column(Text, nullable=True)
	
	# Relationships
	budget_line = relationship("CFBFBudgetLine", back_populates="allocations")
	
	def __repr__(self):
		return f"<CFBFAllocation {self.target_type}:{self.target_code} - {self.allocation_percentage}%>"
	
	def calculate_allocated_amount(self, base_amount: Decimal) -> Decimal:
		"""Calculate allocated amount based on method"""
		if self.allocation_method == 'Percentage':
			return base_amount * (self.allocation_percentage / 100)
		elif self.allocation_method == 'Amount':
			return self.allocation_amount
		elif self.allocation_method == 'Driver' and self.driver_factor:
			return base_amount * self.driver_factor
		else:
			# Formula-based or other methods would need implementation
			return Decimal('0.00')
	
	def validate_allocation(self) -> bool:
		"""Validate allocation parameters"""
		if self.allocation_method == 'Percentage':
			return 0 <= self.allocation_percentage <= 100
		elif self.allocation_method == 'Amount':
			return self.allocation_amount >= 0
		return True