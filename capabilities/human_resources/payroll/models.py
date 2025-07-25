"""
Payroll Models

Database models for payroll processing including pay periods, payroll runs,
employee payroll records, tax calculations, and deductions.
"""

from datetime import datetime, date
from typing import Dict, List, Any, Optional
from decimal import Decimal
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, Date, DECIMAL, ForeignKey, UniqueConstraint, Enum
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str
import json

from ...auth_rbac.models import BaseMixin, AuditMixin, Model


class HRPayrollPeriod(Model, AuditMixin, BaseMixin):
	"""
	Payroll periods defining pay cycles.
	
	Manages payroll periods with start/end dates, status, and configuration.
	"""
	__tablename__ = 'hr_pr_payroll_period'
	
	# Identity
	period_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Period Information
	period_name = Column(String(100), nullable=False, index=True)  # "2024-01", "Q1 2024", etc.
	period_type = Column(String(20), nullable=False, index=True)  # Weekly, Bi-Weekly, Monthly, Quarterly
	
	# Dates
	start_date = Column(Date, nullable=False, index=True)
	end_date = Column(Date, nullable=False, index=True)
	pay_date = Column(Date, nullable=False, index=True)  # When employees get paid
	
	# Status
	status = Column(String(20), default='Open', index=True)  # Open, Processing, Closed, Finalized
	
	# Configuration
	is_active = Column(Boolean, default=True)
	fiscal_year = Column(Integer, nullable=False, index=True)
	fiscal_quarter = Column(Integer, nullable=True)
	
	# Cutoff and Processing
	cutoff_date = Column(Date, nullable=True)  # Cutoff for timesheet submissions
	processing_date = Column(Date, nullable=True)  # When payroll was processed
	finalized_date = Column(Date, nullable=True)  # When payroll was finalized
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'period_name', name='uq_payroll_period_name_tenant'),
	)
	
	# Relationships
	payroll_runs = relationship("HRPayrollRun", back_populates="period")
	
	def __repr__(self):
		return f"<HRPayrollPeriod {self.period_name}>"


class HRPayrollRun(Model, AuditMixin, BaseMixin):
	"""
	Individual payroll processing runs within a period.
	
	Can have multiple runs per period for corrections, bonuses, etc.
	"""
	__tablename__ = 'hr_pr_payroll_run'
	
	# Identity
	run_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	period_id = Column(String(36), ForeignKey('hr_pr_payroll_period.period_id'), nullable=False, index=True)
	
	# Run Information
	run_number = Column(Integer, nullable=False, index=True)  # 1, 2, 3 for corrections
	run_type = Column(String(20), default='Regular', index=True)  # Regular, Correction, Bonus, Final
	description = Column(String(200), nullable=True)
	
	# Processing
	status = Column(String(20), default='Draft', index=True)  # Draft, Processing, Completed, Posted, Cancelled
	started_at = Column(DateTime, nullable=True)
	completed_at = Column(DateTime, nullable=True)
	processed_by = Column(String(36), nullable=True)  # Employee ID who processed
	
	# Approval
	approved = Column(Boolean, default=False)
	approved_by = Column(String(36), nullable=True)  # Employee ID who approved
	approved_at = Column(DateTime, nullable=True)
	
	# Totals
	total_gross_pay = Column(DECIMAL(15, 2), default=0.00)
	total_deductions = Column(DECIMAL(15, 2), default=0.00)
	total_taxes = Column(DECIMAL(15, 2), default=0.00)
	total_net_pay = Column(DECIMAL(15, 2), default=0.00)
	employee_count = Column(Integer, default=0)
	
	# Configuration
	is_active = Column(Boolean, default=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'period_id', 'run_number', name='uq_payroll_run_number'),
	)
	
	# Relationships
	period = relationship("HRPayrollPeriod", back_populates="payroll_runs")
	employee_payrolls = relationship("HREmployeePayroll", back_populates="payroll_run")
	payroll_journals = relationship("HRPayrollJournal", back_populates="payroll_run")
	
	def __repr__(self):
		return f"<HRPayrollRun {self.period.period_name} Run {self.run_number}>"


class HREmployeePayroll(Model, AuditMixin, BaseMixin):
	"""
	Individual employee payroll record for a specific run.
	
	Contains all pay components, deductions, and taxes for one employee.
	"""
	__tablename__ = 'hr_pr_employee_payroll'
	
	# Identity
	employee_payroll_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	run_id = Column(String(36), ForeignKey('hr_pr_payroll_run.run_id'), nullable=False, index=True)
	employee_id = Column(String(36), ForeignKey('hr_edm_employee.employee_id'), nullable=False, index=True)
	
	# Employee Information (snapshot at time of payroll)
	employee_number = Column(String(20), nullable=False, index=True)
	employee_name = Column(String(300), nullable=False)  
	department_name = Column(String(200), nullable=True)
	position_title = Column(String(200), nullable=True)
	
	# Pay Information
	pay_frequency = Column(String(20), nullable=False)  # Weekly, Bi-Weekly, Monthly
	pay_method = Column(String(20), default='DirectDeposit')  # DirectDeposit, Check, Cash
	currency_code = Column(String(3), default='USD')
	
	# Calculation Base
	base_salary = Column(DECIMAL(12, 2), nullable=True)
	hourly_rate = Column(DECIMAL(8, 2), nullable=True)
	hours_worked = Column(DECIMAL(8, 2), nullable=True)
	overtime_hours = Column(DECIMAL(8, 2), nullable=True)
	
	# Totals
	gross_earnings = Column(DECIMAL(12, 2), default=0.00)
	total_deductions = Column(DECIMAL(12, 2), default=0.00)
	total_taxes = Column(DECIMAL(12, 2), default=0.00)
	net_pay = Column(DECIMAL(12, 2), default=0.00)
	
	# YTD Totals
	ytd_gross = Column(DECIMAL(15, 2), default=0.00)
	ytd_deductions = Column(DECIMAL(15, 2), default=0.00)
	ytd_taxes = Column(DECIMAL(15, 2), default=0.00)
	ytd_net = Column(DECIMAL(15, 2), default=0.00)
	
	# Status
	status = Column(String(20), default='Draft', index=True)  # Draft, Calculated, Approved, Paid
	calculation_date = Column(DateTime, nullable=True)
	approval_date = Column(DateTime, nullable=True)
	payment_date = Column(DateTime, nullable=True)
	
	# Tax Information
	federal_allowances = Column(Integer, default=0)
	state_allowances = Column(Integer, default=0)
	marital_status = Column(String(20), nullable=True)
	additional_withholding = Column(DECIMAL(8, 2), default=0.00)
	
	# Configuration
	is_active = Column(Boolean, default=True)
	has_errors = Column(Boolean, default=False)
	error_message = Column(Text, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'run_id', 'employee_id', name='uq_employee_payroll_run'),
	)
	
	# Relationships
	payroll_run = relationship("HRPayrollRun", back_populates="employee_payrolls")
	employee = relationship("HREmployee", foreign_keys=[employee_id])
	payroll_line_items = relationship("HRPayrollLineItem", back_populates="employee_payroll")
	tax_calculations = relationship("HRTaxCalculation", back_populates="employee_payroll")
	pay_stubs = relationship("HRPayStub", back_populates="employee_payroll")
	direct_deposits = relationship("HRDirectDeposit", back_populates="employee_payroll")
	
	def __repr__(self):
		return f"<HREmployeePayroll {self.employee_name} - {self.payroll_run.period.period_name}>"


class HRPayComponent(Model, AuditMixin, BaseMixin):
	"""
	Pay component definitions (earnings, deductions, taxes).
	
	Defines all types of pay components that can be applied to payroll.
	"""
	__tablename__ = 'hr_pr_pay_component'
	
	# Identity
	component_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Component Information
	component_code = Column(String(20), nullable=False, index=True)
	component_name = Column(String(100), nullable=False, index=True)
	description = Column(Text, nullable=True)
	
	# Classification
	component_type = Column(String(20), nullable=False, index=True)  # Earnings, Deduction, Tax, Benefit
	category = Column(String(50), nullable=True, index=True)  # Regular, Overtime, Bonus, etc.
	
	# Calculation
	calculation_method = Column(String(30), nullable=False)  # Fixed, Hours_x_Rate, Percentage, Tax_Table
	default_rate = Column(DECIMAL(10, 4), nullable=True)
	default_amount = Column(DECIMAL(10, 2), nullable=True)
	
	# Tax Properties
	is_taxable = Column(Boolean, default=True)
	is_subject_to_fica = Column(Boolean, default=True)
	is_subject_to_futa = Column(Boolean, default=True)
	is_subject_to_suta = Column(Boolean, default=True)
	is_pre_tax = Column(Boolean, default=False)
	
	# Behavior
	is_recurring = Column(Boolean, default=True)
	is_system_component = Column(Boolean, default=False)  # System-defined components
	requires_approval = Column(Boolean, default=False)
	
	# Limits
	min_amount = Column(DECIMAL(10, 2), nullable=True)
	max_amount = Column(DECIMAL(10, 2), nullable=True)
	annual_limit = Column(DECIMAL(12, 2), nullable=True)
	
	# GL Account Integration
	debit_account = Column(String(50), nullable=True)  # GL account for debits
	credit_account = Column(String(50), nullable=True)  # GL account for credits
	
	# Configuration
	is_active = Column(Boolean, default=True)
	effective_date = Column(Date, nullable=True)
	expiry_date = Column(Date, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'component_code', name='uq_pay_component_code_tenant'),
	)
	
	# Relationships
	payroll_line_items = relationship("HRPayrollLineItem", back_populates="pay_component")
	employee_deductions = relationship("HREmployeeDeduction", back_populates="pay_component")
	
	def __repr__(self):
		return f"<HRPayComponent {self.component_name}>"


class HRPayrollLineItem(Model, AuditMixin, BaseMixin):
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