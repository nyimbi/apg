"""
APG Financial Management General Ledger - Enterprise Models

Comprehensive database models for enterprise-grade general ledger functionality including:
- Hierarchical chart of accounts with multi-dimensional analytics
- Advanced journal entry processing with approval workflows
- Multi-currency operations with real-time conversion
- Event sourcing for complete audit trails
- High-performance posting and reporting
- Regulatory compliance and controls

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from datetime import datetime, date
from typing import Dict, List, Any, Optional, Union
from decimal import Decimal
from enum import Enum as PyEnum
import json
import uuid
from dataclasses import dataclass

from sqlalchemy import (
	Column, String, Text, Integer, Float, Boolean, DateTime, Date, 
	DECIMAL, ForeignKey, UniqueConstraint, Index, CheckConstraint,
	JSON, ARRAY, Enum, event
)
from sqlalchemy.orm import relationship, validates, Session
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.sql import func
from pydantic import BaseModel, Field, validator, ConfigDict
from uuid_extensions import uuid7str

try:
	from ...auth_rbac.models import BaseMixin, AuditMixin, Model
except ImportError:
	# Fallback base classes for standalone operation
	from sqlalchemy.ext.declarative import declarative_base
	Base = declarative_base()
	
	class BaseMixin:
		pass
	
	class AuditMixin:
		created_at = Column(DateTime, default=func.now())
		updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
		created_by = Column(String(36))
		updated_by = Column(String(36))
	
	class Model(Base):
		__abstract__ = True
		id = Column(String(36), primary_key=True, default=uuid7str)


# =====================================
# ENUMS AND CONSTANTS
# =====================================

class AccountTypeEnum(PyEnum):
	"""Standard account types for financial classification"""
	ASSET = "ASSET"
	LIABILITY = "LIABILITY"
	EQUITY = "EQUITY"
	REVENUE = "REVENUE"
	EXPENSE = "EXPENSE"


class BalanceTypeEnum(PyEnum):
	"""Normal balance types for accounts"""
	DEBIT = "DEBIT"
	CREDIT = "CREDIT"


class PeriodStatusEnum(PyEnum):
	"""Accounting period status values"""
	FUTURE = "FUTURE"
	OPEN = "OPEN"
	SOFT_CLOSED = "SOFT_CLOSED"
	CLOSED = "CLOSED"
	LOCKED = "LOCKED"


class JournalStatusEnum(PyEnum):
	"""Journal entry status values"""
	DRAFT = "DRAFT"
	PENDING_APPROVAL = "PENDING_APPROVAL"
	APPROVED = "APPROVED"
	POSTED = "POSTED"
	REVERSED = "REVERSED"
	CANCELLED = "CANCELLED"


class JournalSourceEnum(PyEnum):
	"""Journal entry source systems"""
	MANUAL = "MANUAL"
	ACCOUNTS_PAYABLE = "ACCOUNTS_PAYABLE"
	ACCOUNTS_RECEIVABLE = "ACCOUNTS_RECEIVABLE"
	PAYROLL = "PAYROLL"
	INVENTORY = "INVENTORY"
	FIXED_ASSETS = "FIXED_ASSETS"
	BANK_FEEDS = "BANK_FEEDS"
	CONSOLIDATION = "CONSOLIDATION"
	TAX_PROVISION = "TAX_PROVISION"
	SYSTEM_GENERATED = "SYSTEM_GENERATED"


class CurrencyEnum(PyEnum):
	"""Supported currencies"""
	USD = "USD"
	EUR = "EUR"
	GBP = "GBP"
	JPY = "JPY"
	CAD = "CAD"
	AUD = "AUD"
	CHF = "CHF"
	CNY = "CNY"
	KES = "KES"


class ReportingFrameworkEnum(PyEnum):
	"""Financial reporting frameworks"""
	GAAP = "GAAP"
	IFRS = "IFRS"
	LOCAL_GAAP = "LOCAL_GAAP"
	TAX_BASIS = "TAX_BASIS"
	STATUTORY = "STATUTORY"


# =====================================
# PYDANTIC MODELS FOR VALIDATION
# =====================================

class GLAccountBase(BaseModel):
	"""Base validation model for GL accounts"""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	account_code: str = Field(min_length=1, max_length=20)
	account_name: str = Field(min_length=1, max_length=200)
	description: Optional[str] = Field(max_length=1000)
	account_type: AccountTypeEnum
	currency_code: str = Field(default="USD", min_length=3, max_length=3)
	
	@validator('account_code')
	def validate_account_code(cls, v):
		if not v.replace('-', '').replace('_', '').isalnum():
			raise ValueError('Account code must contain only letters, numbers, hyphens, and underscores')
		return v.upper()


class JournalEntryBase(BaseModel):
	"""Base validation model for journal entries"""
	model_config = ConfigDict(extra='forbid', validate_assignment=True)
	
	description: str = Field(min_length=1, max_length=500)
	entry_date: date
	posting_date: date
	reference: Optional[str] = Field(max_length=100)
	
	@validator('posting_date')
	def validate_posting_date(cls, v, values):
		if 'entry_date' in values and v < values['entry_date']:
			raise ValueError('Posting date cannot be before entry date')
		return v


# =====================================
# CORE DATABASE MODELS
# =====================================


class GLTenant(Model, AuditMixin, BaseMixin):
	"""
	Multi-tenant configuration for GL operations.
	
	Manages tenant-specific accounting configurations including
	fiscal year settings, base currency, and reporting frameworks.
	"""
	__tablename__ = 'gl_tenant'
	
	# Identity
	tenant_id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_code = Column(String(20), unique=True, nullable=False, index=True)
	tenant_name = Column(String(200), nullable=False)
	
	# Fiscal Configuration
	fiscal_year_start_month = Column(Integer, default=1)  # 1=January
	fiscal_year_end_month = Column(Integer, default=12)   # 12=December
	period_type = Column(String(20), default='MONTHLY')   # MONTHLY, QUARTERLY
	use_13_periods = Column(Boolean, default=False)
	
	# Currency Configuration
	base_currency = Column(Enum(CurrencyEnum), default=CurrencyEnum.USD, nullable=False)
	functional_currency = Column(Enum(CurrencyEnum), default=CurrencyEnum.USD, nullable=False)
	reporting_currencies = Column(JSON, default=list)  # List of additional reporting currencies
	
	# Reporting Configuration
	reporting_framework = Column(Enum(ReportingFrameworkEnum), default=ReportingFrameworkEnum.GAAP)
	consolidation_method = Column(String(20), default='FULL')  # FULL, PROPORTIONAL, EQUITY
	
	# Regional Settings
	country_code = Column(String(2), nullable=False, default='US')
	locale = Column(String(10), default='en_US')
	timezone = Column(String(50), default='UTC')
	
	# Compliance Settings
	sox_compliance = Column(Boolean, default=False)
	audit_trail_retention_years = Column(Integer, default=7)
	
	# Configuration Metadata
	configuration = Column(JSON, default=dict)
	
	# Relationships
	account_types = relationship("GLAccountType", back_populates="tenant")
	accounts = relationship("GLAccount", back_populates="tenant")
	periods = relationship("GLPeriod", back_populates="tenant")
	
	__table_args__ = (
		Index('ix_gl_tenant_code', 'tenant_code'),
		Index('ix_gl_tenant_currency', 'base_currency'),
	)
	
	def __repr__(self):
		return f"<GLTenant {self.tenant_code}: {self.tenant_name}>"


class GLAccountType(Model, AuditMixin, BaseMixin):
	"""
	Enhanced chart of accounts type classification.
	
	Provides comprehensive account type management with support for
	multiple reporting frameworks and regulatory requirements.
	"""
	__tablename__ = 'gl_account_type'
	
	# Identity
	type_id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), ForeignKey('gl_tenant.tenant_id'), nullable=False, index=True)
	
	# Type Information
	type_code = Column(Enum(AccountTypeEnum), nullable=False, index=True)
	type_name = Column(String(100), nullable=False)
	description = Column(Text)
	
	# Behavioral Properties
	normal_balance = Column(Enum(BalanceTypeEnum), nullable=False)
	is_balance_sheet = Column(Boolean, nullable=False)
	is_income_statement = Column(Boolean, nullable=False)
	is_cash_flow = Column(Boolean, default=False)
	
	# Financial Statement Classification
	balance_sheet_section = Column(String(50))  # Current Assets, Fixed Assets, etc.
	income_statement_section = Column(String(50))  # Operating Revenue, Operating Expense, etc.
	cash_flow_section = Column(String(50))  # Operating, Investing, Financing
	
	# Reporting Configuration
	reporting_sequence = Column(Integer, default=0)
	consolidation_rule = Column(String(20), default='SUM')  # SUM, AVERAGE, LAST
	elimination_required = Column(Boolean, default=False)
	
	# Compliance and Controls
	requires_approval = Column(Boolean, default=False)
	approval_limit = Column(DECIMAL(15, 2))
	supports_dimensions = Column(Boolean, default=True)
	
	# System Configuration
	is_system_type = Column(Boolean, default=False)
	is_active = Column(Boolean, default=True)
	sort_order = Column(Integer, default=0)
	
	# Validation Rules
	validation_rules = Column(JSON, default=dict)
	posting_restrictions = Column(JSON, default=dict)
	
	# Relationships
	tenant = relationship("GLTenant", back_populates="account_types")
	accounts = relationship("GLAccount", back_populates="account_type")
	
	__table_args__ = (
		UniqueConstraint('tenant_id', 'type_code', name='uq_account_type_tenant'),
		Index('ix_gl_account_type_tenant_code', 'tenant_id', 'type_code'),
		Index('ix_gl_account_type_balance_sheet', 'is_balance_sheet'),
	)
	
	def __repr__(self):
		return f"<GLAccountType {self.type_code.value}: {self.type_name}>"
	
	@property
	def is_debit_normal(self) -> bool:
		"""Check if account type has normal debit balance"""
		return self.normal_balance == BalanceTypeEnum.DEBIT
	
	def get_reporting_sign(self, account_balance: Decimal) -> Decimal:
		"""Get the reporting sign for the account balance"""
		if self.is_debit_normal:
			return account_balance
		else:
			return -account_balance


class GLAccount(Model, AuditMixin, BaseMixin):
	"""
	Enhanced General Ledger Chart of Accounts.
	
	Provides enterprise-grade account management with hierarchical structure,
	multi-currency support, dimensional analytics, and advanced reporting capabilities.
	"""
	__tablename__ = 'gl_account'
	
	# Identity
	account_id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), ForeignKey('gl_tenant.tenant_id'), nullable=False, index=True)
	
	# Account Information
	account_code = Column(String(20), nullable=False)
	account_name = Column(String(200), nullable=False)
	short_name = Column(String(50))  # Abbreviated name for reports
	description = Column(Text)
	external_code = Column(String(50))  # Integration with external systems
	
	# Classification and Hierarchy
	account_type_id = Column(String(36), ForeignKey('gl_account_type.type_id'), nullable=False)
	parent_account_id = Column(String(36), ForeignKey('gl_account.account_id'), index=True)
	account_group_id = Column(String(36), ForeignKey('gl_account_group.group_id'))
	
	# Hierarchy Management
	level = Column(Integer, default=0)
	path = Column(String(1000))  # Materialized path for efficient queries
	sort_code = Column(String(100))  # Custom sorting within hierarchy
	display_order = Column(Integer, default=0)
	
	# Account Properties
	is_active = Column(Boolean, default=True)
	is_header = Column(Boolean, default=False)
	is_system_account = Column(Boolean, default=False)
	allow_manual_posting = Column(Boolean, default=True)
	require_dimensions = Column(Boolean, default=False)
	
	# Currency and Multi-currency
	primary_currency = Column(Enum(CurrencyEnum), default=CurrencyEnum.USD)
	allow_multi_currency = Column(Boolean, default=False)
	revaluation_account_id = Column(String(36), ForeignKey('gl_account.account_id'))
	translation_method = Column(String(20), default='CURRENT_RATE')  # CURRENT_RATE, HISTORICAL_RATE
	
	# Balance Information (Base Currency)
	current_balance = Column(DECIMAL(18, 4), default=0.0000)
	ytd_balance = Column(DECIMAL(18, 4), default=0.0000)
	opening_balance = Column(DECIMAL(18, 4), default=0.0000)
	budget_balance = Column(DECIMAL(18, 4), default=0.0000)
	
	# Multi-currency Balances
	foreign_current_balance = Column(DECIMAL(18, 4), default=0.0000)
	foreign_ytd_balance = Column(DECIMAL(18, 4), default=0.0000)
	foreign_opening_balance = Column(DECIMAL(18, 4), default=0.0000)
	
	# Statistical Information
	transaction_count = Column(Integer, default=0)
	last_activity_date = Column(Date)
	largest_debit = Column(DECIMAL(18, 4), default=0.0000)
	largest_credit = Column(DECIMAL(18, 4), default=0.0000)
	
	# Dimensional Analytics Requirements
	cost_center_required = Column(Boolean, default=False)
	department_required = Column(Boolean, default=False)
	project_required = Column(Boolean, default=False)
	location_required = Column(Boolean, default=False)
	business_unit_required = Column(Boolean, default=False)
	custom_dimension1_required = Column(Boolean, default=False)
	custom_dimension2_required = Column(Boolean, default=False)
	custom_dimension3_required = Column(Boolean, default=False)
	
	# Tax and Compliance
	tax_category = Column(String(50))
	vat_code = Column(String(20))
	withholding_tax_code = Column(String(20))
	sales_tax_code = Column(String(20))
	
	# Workflow and Approval
	approval_required = Column(Boolean, default=False)
	approval_threshold = Column(DECIMAL(15, 2))
	workflow_template_id = Column(String(36))
	
	# Reporting Configuration
	consolidation_account_id = Column(String(36))  # Mapping for consolidation
	elimination_rules = Column(JSON, default=dict)
	reporting_overrides = Column(JSON, default=dict)
	
	# Integration and Automation
	auto_allocation_rules = Column(JSON, default=list)
	reconciliation_rules = Column(JSON, default=dict)
	posting_restrictions = Column(JSON, default=dict)
	
	# Audit and Control
	requires_supporting_docs = Column(Boolean, default=False)
	sensitive_account = Column(Boolean, default=False)
	fraud_monitoring = Column(Boolean, default=False)
	
	# Data Retention and Archival
	retain_detail_months = Column(Integer, default=84)  # 7 years default
	archive_after_months = Column(Integer, default=120)  # 10 years
	
	# Performance Optimization
	balance_cache_timestamp = Column(DateTime)
	balance_calculation_method = Column(String(20), default='INCREMENTAL')
	
	# Configuration and Metadata
	custom_attributes = Column(JSON, default=dict)
	validation_rules = Column(JSON, default=dict)
	business_rules = Column(JSON, default=dict)
	
	# Constraints and Indexes
	__table_args__ = (
		UniqueConstraint('tenant_id', 'account_code', name='uq_gl_account_code_tenant'),
		Index('ix_gl_account_tenant_type', 'tenant_id', 'account_type_id'),
		Index('ix_gl_account_tenant_active', 'tenant_id', 'is_active'),
		Index('ix_gl_account_path', 'path'),
		Index('ix_gl_account_parent', 'parent_account_id'),
		Index('ix_gl_account_currency', 'primary_currency'),
		Index('ix_gl_account_balance', 'current_balance'),
		CheckConstraint('level >= 0', name='ck_gl_account_level_positive'),
		CheckConstraint('display_order >= 0', name='ck_gl_account_display_order_positive'),
	)
	
	# Relationships
	tenant = relationship("GLTenant", back_populates="accounts")
	account_type = relationship("GLAccountType", back_populates="accounts")
	parent_account = relationship("GLAccount", remote_side=[account_id], backref="child_accounts")
	revaluation_account = relationship("GLAccount", remote_side=[account_id])
	account_group = relationship("GLAccountGroup", back_populates="accounts")
	
	# Transaction Relationships
	journal_lines = relationship("GLJournalLine", back_populates="account")
	postings = relationship("GLPosting", back_populates="account")
	balances = relationship("GLAccountBalance", back_populates="account")
	budgets = relationship("GLAccountBudget", back_populates="account")
	
	def __repr__(self):
		return f"<GLAccount {self.account_code}: {self.account_name}>"
	
	@validates('account_code')
	def validate_account_code(self, key, account_code):
		"""Validate account code format"""
		if not account_code or len(account_code.strip()) == 0:
			raise ValueError("Account code cannot be empty")
		return account_code.strip().upper()
	
	@validates('level')
	def validate_level(self, key, level):
		"""Validate hierarchy level"""
		if level < 0:
			raise ValueError("Account level cannot be negative")
		if level > 10:  # Reasonable maximum depth
			raise ValueError("Account hierarchy too deep (max 10 levels)")
		return level
	
	@hybrid_property
	def full_code(self):
		"""Get full hierarchical account code"""
		if self.parent_account:
			return f"{self.parent_account.full_code}.{self.account_code}"
		return self.account_code
	
	@hybrid_property
	def is_debit_normal(self):
		"""Check if account has normal debit balance"""
		return self.account_type.normal_balance == BalanceTypeEnum.DEBIT
	
	@hybrid_property
	def is_posting_account(self):
		"""Check if account allows posting"""
		return not self.is_header and self.allow_manual_posting and self.is_active
	
	def get_full_path(self) -> str:
		"""Get full account path for display"""
		if self.parent_account:
			return f"{self.parent_account.get_full_path()} > {self.account_name}"
		return self.account_name
	
	def calculate_balance(self, 
						 as_of_date: Optional[date] = None,
						 currency: Optional[str] = None,
						 include_children: bool = False) -> Decimal:
		"""
		Calculate account balance with advanced options.
		
		Args:
			as_of_date: Calculate balance as of specific date
			currency: Currency for balance calculation
			include_children: Include child account balances
		"""
		if as_of_date is None and not include_children:
			return self.current_balance
		
		# Complex balance calculation logic would go here
		# This is a simplified version
		balance = Decimal('0.00')
		
		# Calculate from postings if date specified
		if as_of_date:
			postings = [p for p in self.postings 
					   if p.posting_date <= as_of_date and p.is_posted]
			
			total_debits = sum(p.debit_amount for p in postings)
			total_credits = sum(p.credit_amount for p in postings)
			
			if self.is_debit_normal:
				balance = self.opening_balance + total_debits - total_credits
			else:
				balance = self.opening_balance + total_credits - total_debits
		else:
			balance = self.current_balance
		
		# Include child accounts if requested
		if include_children:
			for child in self.child_accounts:
				if child.is_active:
					balance += child.calculate_balance(as_of_date, currency, True)
		
		return balance
	
	def can_post_transaction(self, amount: Decimal, user_id: str = None) -> tuple[bool, str]:
		"""
		Check if a transaction can be posted to this account.
		
		Returns:
			tuple: (can_post: bool, reason: str)
		"""
		if not self.is_active:
			return False, "Account is inactive"
		
		if self.is_header:
			return False, "Cannot post to header accounts"
		
		if not self.allow_manual_posting:
			return False, "Manual posting not allowed"
		
		if self.approval_required and self.approval_threshold:
			if abs(amount) > self.approval_threshold:
				return False, "Amount exceeds approval threshold"
		
		return True, "OK"
	
	def update_balance(self, debit_amount: Decimal, credit_amount: Decimal):
		"""Update account balance after posting"""
		if self.is_debit_normal:
			self.current_balance += debit_amount - credit_amount
		else:
			self.current_balance += credit_amount - debit_amount
		
		self.transaction_count += 1
		self.last_activity_date = date.today()
		
		# Update largest amounts
		if debit_amount > self.largest_debit:
			self.largest_debit = debit_amount
		if credit_amount > self.largest_credit:
			self.largest_credit = credit_amount
	
	def build_path(self):
		"""Build materialized path for efficient hierarchy queries"""
		if self.parent_account:
			self.path = f"{self.parent_account.path}/{self.account_id}"
			self.level = self.parent_account.level + 1
		else:
			self.path = f"/{self.account_id}"
			self.level = 0


class GLAccountGroup(Model, AuditMixin, BaseMixin):
	"""
	Account groupings for enhanced reporting and organization.
	
	Provides logical groupings that cross the hierarchical boundaries,
	useful for management reporting and analysis.
	"""
	__tablename__ = 'gl_account_group'
	
	# Identity
	group_id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), ForeignKey('gl_tenant.tenant_id'), nullable=False, index=True)
	
	# Group Information
	group_code = Column(String(20), nullable=False)
	group_name = Column(String(200), nullable=False)
	description = Column(Text)
	
	# Configuration
	is_active = Column(Boolean, default=True)
	sort_order = Column(Integer, default=0)
	group_type = Column(String(50))  # MANAGEMENT, STATUTORY, TAX, etc.
	
	# Relationships
	accounts = relationship("GLAccount", back_populates="account_group")
	
	__table_args__ = (
		UniqueConstraint('tenant_id', 'group_code', name='uq_gl_account_group_code_tenant'),
		Index('ix_gl_account_group_tenant', 'tenant_id'),
	)
	
	def __repr__(self):
		return f"<GLAccountGroup {self.group_code}: {self.group_name}>"


class GLAccountBalance(Model, AuditMixin, BaseMixin):
	"""
	Historical account balances for performance optimization.
	
	Stores periodic snapshots of account balances to improve
	reporting performance and provide historical trending.
	"""
	__tablename__ = 'gl_account_balance'
	
	# Identity
	balance_id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), ForeignKey('gl_tenant.tenant_id'), nullable=False, index=True)
	account_id = Column(String(36), ForeignKey('gl_account.account_id'), nullable=False, index=True)
	period_id = Column(String(36), ForeignKey('gl_period.period_id'), nullable=False, index=True)
	
	# Balance Information
	as_of_date = Column(Date, nullable=False, index=True)
	opening_balance = Column(DECIMAL(18, 4), default=0.0000)
	period_debits = Column(DECIMAL(18, 4), default=0.0000)
	period_credits = Column(DECIMAL(18, 4), default=0.0000)
	ending_balance = Column(DECIMAL(18, 4), default=0.0000)
	
	# Multi-currency Information
	currency = Column(Enum(CurrencyEnum), default=CurrencyEnum.USD)
	exchange_rate = Column(DECIMAL(12, 6), default=1.000000)
	base_currency_balance = Column(DECIMAL(18, 4), default=0.0000)
	
	# Statistical Information
	transaction_count = Column(Integer, default=0)
	largest_debit = Column(DECIMAL(18, 4), default=0.0000)
	largest_credit = Column(DECIMAL(18, 4), default=0.0000)
	
	# Relationships
	account = relationship("GLAccount", back_populates="balances")
	period = relationship("GLPeriod", back_populates="account_balances")
	
	__table_args__ = (
		UniqueConstraint('account_id', 'period_id', 'currency', name='uq_gl_balance_account_period_currency'),
		Index('ix_gl_account_balance_date', 'as_of_date'),
		Index('ix_gl_account_balance_tenant_account', 'tenant_id', 'account_id'),
	)
	
	def __repr__(self):
		return f"<GLAccountBalance {self.account.account_code} {self.as_of_date}>"


class GLAccountBudget(Model, AuditMixin, BaseMixin):
	"""
	Budget amounts for accounts by period.
	
	Supports multiple budget scenarios and versions for comprehensive
	budget vs actual reporting and variance analysis.
	"""
	__tablename__ = 'gl_account_budget'
	
	# Identity
	budget_id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), ForeignKey('gl_tenant.tenant_id'), nullable=False, index=True)
	account_id = Column(String(36), ForeignKey('gl_account.account_id'), nullable=False, index=True)
	period_id = Column(String(36), ForeignKey('gl_period.period_id'), nullable=False, index=True)
	
	# Budget Information
	budget_scenario = Column(String(50), default='APPROVED')  # APPROVED, FORECAST, REVISED
	budget_version = Column(String(20), default='1.0')
	budget_amount = Column(DECIMAL(18, 4), default=0.0000)
	
	# Period Distribution
	monthly_amounts = Column(JSON, default=list)  # For quarterly budgets distributed monthly
	quarterly_amounts = Column(JSON, default=list)  # For annual budgets distributed quarterly
	
	# Metadata
	budget_notes = Column(Text)
	approval_status = Column(String(20), default='DRAFT')
	approved_by = Column(String(36))
	approved_date = Column(DateTime)
	
	# Relationships
	account = relationship("GLAccount", back_populates="budgets")
	period = relationship("GLPeriod", back_populates="account_budgets")
	
	__table_args__ = (
		UniqueConstraint('account_id', 'period_id', 'budget_scenario', 'budget_version', 
						name='uq_gl_budget_account_period_scenario_version'),
		Index('ix_gl_account_budget_tenant_account', 'tenant_id', 'account_id'),
		Index('ix_gl_account_budget_scenario', 'budget_scenario'),
	)
	
	def __repr__(self):
		return f"<GLAccountBudget {self.account.account_code} {self.budget_scenario}>"


class GLCurrencyRate(Model, AuditMixin, BaseMixin):
	"""
	Currency exchange rates for multi-currency operations.
	
	Maintains historical exchange rates with support for multiple
	rate types (spot, average, budget) and automated rate updates.
	"""
	__tablename__ = 'gl_currency_rate'
	
	# Identity
	rate_id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), ForeignKey('gl_tenant.tenant_id'), nullable=False, index=True)
	
	# Currency Information
	from_currency = Column(Enum(CurrencyEnum), nullable=False, index=True)
	to_currency = Column(Enum(CurrencyEnum), nullable=False, index=True)
	rate_type = Column(String(20), default='SPOT')  # SPOT, AVERAGE, BUDGET, CLOSING
	
	# Rate Information
	effective_date = Column(Date, nullable=False, index=True)
	expiry_date = Column(Date)
	exchange_rate = Column(DECIMAL(12, 6), nullable=False)
	inverse_rate = Column(DECIMAL(12, 6))
	
	# Source and Audit
	rate_source = Column(String(50))  # MANUAL, CENTRAL_BANK, REUTERS, etc.
	source_reference = Column(String(100))
	last_updated = Column(DateTime, default=func.now())
	
	# Status
	is_active = Column(Boolean, default=True)
	is_system_rate = Column(Boolean, default=False)
	
	__table_args__ = (
		UniqueConstraint('tenant_id', 'from_currency', 'to_currency', 'rate_type', 'effective_date',
						name='uq_gl_currency_rate_unique'),
		Index('ix_gl_currency_rate_currencies_date', 'from_currency', 'to_currency', 'effective_date'),
		Index('ix_gl_currency_rate_tenant_date', 'tenant_id', 'effective_date'),
	)
	
	def __repr__(self):
		return f"<GLCurrencyRate {self.from_currency.value}/{self.to_currency.value} {self.exchange_rate}>"


class GLPeriod(Model, AuditMixin, BaseMixin):
	"""
	Enhanced accounting periods for comprehensive period management.
	
	Supports flexible fiscal calendars, automated period operations,
	and comprehensive period-end procedures with audit trails.
	"""
	__tablename__ = 'gl_period'
	
	# Identity
	period_id = Column(String(36), primary_key=True, default=uuid7str)
	tenant_id = Column(String(36), ForeignKey('gl_tenant.tenant_id'), nullable=False, index=True)
	
	# Period Hierarchy
	fiscal_year = Column(Integer, nullable=False, index=True)
	fiscal_quarter = Column(Integer, nullable=False)  # 1-4
	period_number = Column(Integer, nullable=False, index=True)  # 1-13
	period_name = Column(String(50), nullable=False)
	period_type = Column(String(20), default='REGULAR')  # REGULAR, ADJUSTMENT, OPENING
	
	# Date Range
	start_date = Column(Date, nullable=False, index=True)
	end_date = Column(Date, nullable=False, index=True)
	
	# Status Management
	status = Column(Enum(PeriodStatusEnum), default=PeriodStatusEnum.FUTURE, index=True)
	is_adjustment_period = Column(Boolean, default=False)
	is_year_end_period = Column(Boolean, default=False)
	
	# Period Operations
	opened_date = Column(DateTime)
	opened_by = Column(String(36))
	closed_date = Column(DateTime)
	closed_by = Column(String(36))
	locked_date = Column(DateTime)
	locked_by = Column(String(36))
	
	# Closing Information
	closing_checklist = Column(JSON, default=list)
	closing_entries_count = Column(Integer, default=0)
	closing_notes = Column(Text)
	auto_close_enabled = Column(Boolean, default=False)
	auto_close_date = Column(Date)
	
	# Performance Metrics
	transaction_count = Column(Integer, default=0)
	posting_count = Column(Integer, default=0)
	journal_entry_count = Column(Integer, default=0)
	total_debits = Column(DECIMAL(18, 4), default=0.0000)
	total_credits = Column(DECIMAL(18, 4), default=0.0000)
	
	# Constraints and Indexes
	__table_args__ = (
		UniqueConstraint('tenant_id', 'fiscal_year', 'period_number', name='uq_gl_period_year_number_tenant'),
		Index('ix_gl_period_tenant_year', 'tenant_id', 'fiscal_year'),
		Index('ix_gl_period_tenant_status', 'tenant_id', 'status'),
		Index('ix_gl_period_dates', 'start_date', 'end_date'),
		CheckConstraint('fiscal_quarter >= 1 AND fiscal_quarter <= 4', name='ck_gl_period_quarter_valid'),
		CheckConstraint('period_number >= 1 AND period_number <= 13', name='ck_gl_period_number_valid'),
		CheckConstraint('start_date <= end_date', name='ck_gl_period_date_range_valid'),
	)
	
	# Relationships
	tenant = relationship("GLTenant", back_populates="periods")
	journal_entries = relationship("GLJournalEntry", back_populates="period")
	postings = relationship("GLPosting", back_populates="period")
	account_balances = relationship("GLAccountBalance", back_populates="period")
	account_budgets = relationship("GLAccountBudget", back_populates="period")
	
	def __repr__(self):
		return f"<GLPeriod {self.period_name} ({self.status.value})>"
	
	@validates('period_number')
	def validate_period_number(self, key, period_number):
		"""Validate period number is within valid range"""
		if not 1 <= period_number <= 13:
			raise ValueError("Period number must be between 1 and 13")
		return period_number
	
	@validates('fiscal_quarter')
	def validate_fiscal_quarter(self, key, fiscal_quarter):
		"""Validate fiscal quarter is within valid range"""
		if not 1 <= fiscal_quarter <= 4:
			raise ValueError("Fiscal quarter must be between 1 and 4")
		return fiscal_quarter
	
	@property
	def is_open(self) -> bool:
		"""Check if period is open for posting"""
		return self.status == PeriodStatusEnum.OPEN
	
	@property
	def is_closed(self) -> bool:
		"""Check if period is closed"""
		return self.status in [PeriodStatusEnum.CLOSED, PeriodStatusEnum.LOCKED]
	
	@property
	def can_post(self) -> bool:
		"""Check if period allows posting"""
		return self.status == PeriodStatusEnum.OPEN
	
	@property
	def can_reopen(self) -> bool:
		"""Check if period can be reopened"""
		return self.status == PeriodStatusEnum.SOFT_CLOSED
	
	def open_period(self, user_id: str) -> bool:
		"""Open the period for posting"""
		if self.status == PeriodStatusEnum.FUTURE:
			self.status = PeriodStatusEnum.OPEN
			self.opened_date = datetime.utcnow()
			self.opened_by = user_id
			return True
		return False
	
	def soft_close_period(self, user_id: str) -> bool:
		"""Soft close the period (can be reopened)"""
		if self.status == PeriodStatusEnum.OPEN:
			self.status = PeriodStatusEnum.SOFT_CLOSED
			self.closed_date = datetime.utcnow()
			self.closed_by = user_id
			return True
		return False
	
	def close_period(self, user_id: str, force: bool = False) -> bool:
		"""Hard close the period"""
		if self.status in [PeriodStatusEnum.OPEN, PeriodStatusEnum.SOFT_CLOSED] or force:
			self.status = PeriodStatusEnum.CLOSED
			self.closed_date = datetime.utcnow()
			self.closed_by = user_id
			return True
		return False
	
	def lock_period(self, user_id: str) -> bool:
		"""Lock the period (permanent)"""
		if self.status == PeriodStatusEnum.CLOSED:
			self.status = PeriodStatusEnum.LOCKED
			self.locked_date = datetime.utcnow()
			self.locked_by = user_id
			return True
		return False
	
	def get_period_range(self) -> tuple[date, date]:
		"""Get the period date range"""
		return self.start_date, self.end_date
	
	def is_date_in_period(self, check_date: date) -> bool:
		"""Check if a date falls within this period"""
		return self.start_date <= check_date <= self.end_date
	
	def update_statistics(self):
		"""Update period statistics from related transactions"""
		# This would be implemented to calculate statistics from related records
		# Simplified for now
		self.transaction_count = len(self.postings)
		self.journal_entry_count = len(self.journal_entries)
		
		if self.postings:
			self.total_debits = sum(p.debit_amount for p in self.postings)
			self.total_credits = sum(p.credit_amount for p in self.postings)


class CFGLJournalEntry(Model, AuditMixin, BaseMixin):
	"""
	Journal entries - headers for groups of journal lines.
	
	Each journal entry contains multiple lines that must balance
	(total debits = total credits).
	"""
	__tablename__ = 'cf_gl_journal_entry'
	
	# Identity
	journal_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Journal Information
	journal_number = Column(String(50), nullable=False, index=True)
	description = Column(Text, nullable=False)
	reference = Column(String(100), nullable=True)  # External reference
	
	# Dates
	entry_date = Column(Date, nullable=False, index=True)
	posting_date = Column(Date, nullable=False, index=True)
	
	# Period
	period_id = Column(String(36), ForeignKey('cf_gl_period.period_id'), nullable=False, index=True)
	
	# Status and Control
	status = Column(String(20), default='Draft', index=True)  # Draft, Posted, Reversed
	source = Column(String(50), default='Manual')  # Manual, AP, AR, Payroll, etc.
	batch_id = Column(String(36), nullable=True)  # Batch processing
	
	# Approval
	requires_approval = Column(Boolean, default=False)
	approved = Column(Boolean, default=False)
	approved_by = Column(String(36), nullable=True)
	approved_date = Column(DateTime, nullable=True)
	
	# Posting Information
	posted = Column(Boolean, default=False)
	posted_by = Column(String(36), nullable=True)
	posted_date = Column(DateTime, nullable=True)
	
	# Reversal Information  
	reversed = Column(Boolean, default=False)
	reversal_date = Column(DateTime, nullable=True)
	reversal_journal_id = Column(String(36), nullable=True)
	
	# Totals (for validation)
	total_debits = Column(DECIMAL(15, 2), default=0.00)
	total_credits = Column(DECIMAL(15, 2), default=0.00)
	line_count = Column(Integer, default=0)
	
	# Relationships
	period = relationship("CFGLPeriod", back_populates="journal_entries")
	lines = relationship("CFGLJournalLine", back_populates="journal_entry", cascade="all, delete-orphan")
	postings = relationship("CFGLPosting", back_populates="journal_entry")
	
	def __repr__(self):
		return f"<CFGLJournalEntry {self.journal_number}>"
	
	def validate_balance(self) -> bool:
		"""Validate that debits equal credits"""
		return abs(self.total_debits - self.total_credits) < 0.01
	
	def calculate_totals(self):
		"""Recalculate totals from lines"""
		self.total_debits = sum(line.debit_amount for line in self.lines)
		self.total_credits = sum(line.credit_amount for line in self.lines)
		self.line_count = len(self.lines)
	
	def can_post(self) -> bool:
		"""Check if journal entry can be posted"""
		return (
			self.status == 'Draft' and
			not self.posted and
			self.validate_balance() and
			self.period.can_post() and
			(not self.requires_approval or self.approved)
		)
	
	def post_entry(self, user_id: str):
		"""Post the journal entry"""
		if not self.can_post():
			raise ValueError("Journal entry cannot be posted")
		
		self.posted = True
		self.posted_by = user_id
		self.posted_date = datetime.utcnow()
		self.status = 'Posted'
		
		# Create postings for each line
		for line in self.lines:
			line.create_posting()


class CFGLJournalLine(Model, AuditMixin, BaseMixin):
	"""
	Individual journal entry lines.
	
	Each line contains a debit or credit to a specific GL account
	with optional dimensions like cost center or department.
	"""
	__tablename__ = 'cf_gl_journal_line'
	
	# Identity
	line_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	journal_id = Column(String(36), ForeignKey('cf_gl_journal_entry.journal_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Line Information
	line_number = Column(Integer, nullable=False)
	description = Column(Text, nullable=True)
	
	# Account and Amounts
	account_id = Column(String(36), ForeignKey('cf_gl_account.account_id'), nullable=False, index=True)
	debit_amount = Column(DECIMAL(15, 2), default=0.00)
	credit_amount = Column(DECIMAL(15, 2), default=0.00)
	
	# Dimensions (optional analytical attributes)
	cost_center = Column(String(20), nullable=True)
	department = Column(String(20), nullable=True)
	project = Column(String(20), nullable=True)
	employee_id = Column(String(36), nullable=True)
	
	# Reference Information
	reference_type = Column(String(50), nullable=True)  # Invoice, Receipt, etc.
	reference_id = Column(String(36), nullable=True)
	reference_number = Column(String(100), nullable=True)
	
	# Tax Information
	tax_code = Column(String(20), nullable=True)
	tax_amount = Column(DECIMAL(15, 2), default=0.00)
	
	# Relationships
	journal_entry = relationship("CFGLJournalEntry", back_populates="lines")
	account = relationship("CFGLAccount", back_populates="journal_lines")
	posting = relationship("CFGLPosting", uselist=False, back_populates="journal_line")
	
	def __repr__(self):
		return f"<CFGLJournalLine {self.line_number}: {self.account.account_code}>"
	
	def get_amount(self) -> Decimal:
		"""Get the line amount (debit or credit)"""
		return self.debit_amount if self.debit_amount > 0 else self.credit_amount
	
	def is_debit(self) -> bool:
		"""Check if line is a debit"""
		return self.debit_amount > 0
	
	def create_posting(self):
		"""Create GL posting when journal is posted"""
		if self.posting:
			return  # Already posted
		
		posting = CFGLPosting(
			tenant_id=self.tenant_id,
			account_id=self.account_id,
			journal_entry_id=self.journal_id,
			journal_line_id=self.line_id,
			period_id=self.journal_entry.period_id,
			posting_date=self.journal_entry.posting_date,
			debit_amount=self.debit_amount,
			credit_amount=self.credit_amount,
			description=self.description or self.journal_entry.description,
			reference=self.journal_entry.reference,
			is_posted=True,
			posted_by=self.journal_entry.posted_by,
			posted_date=self.journal_entry.posted_date
		)
		
		return posting


class CFGLPosting(Model, AuditMixin, BaseMixin):
	"""
	Posted GL transactions - the permanent record.
	
	Created from journal lines when journal entries are posted.
	Forms the basis for all financial reporting and account balances.
	"""
	__tablename__ = 'cf_gl_posting'
	
	# Identity
	posting_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Source Information
	journal_entry_id = Column(String(36), ForeignKey('cf_gl_journal_entry.journal_id'), nullable=False, index=True)
	journal_line_id = Column(String(36), ForeignKey('cf_gl_journal_line.line_id'), nullable=False, index=True)
	
	# Account and Period
	account_id = Column(String(36), ForeignKey('cf_gl_account.account_id'), nullable=False, index=True)
	period_id = Column(String(36), ForeignKey('cf_gl_period.period_id'), nullable=False, index=True)
	
	# Posting Details
	posting_date = Column(Date, nullable=False, index=True)
	debit_amount = Column(DECIMAL(15, 2), default=0.00)
	credit_amount = Column(DECIMAL(15, 2), default=0.00)
	
	# Description and Reference
	description = Column(Text, nullable=True)
	reference = Column(String(100), nullable=True)
	
	# Status
	is_posted = Column(Boolean, default=True)
	is_reversed = Column(Boolean, default=False)
	reversal_posting_id = Column(String(36), nullable=True)
	
	# Posting Audit
	posted_by = Column(String(36), nullable=False)
	posted_date = Column(DateTime, nullable=False)
	
	# Relationships
	journal_entry = relationship("CFGLJournalEntry", back_populates="postings")
	journal_line = relationship("CFGLJournalLine", back_populates="posting")
	account = relationship("CFGLAccount", back_populates="postings")
	period = relationship("CFGLPeriod", back_populates="postings")
	
	def __repr__(self):
		return f"<CFGLPosting {self.account.account_code}: {self.get_amount()}>"
	
	def get_amount(self) -> Decimal:
		"""Get the posting amount"""
		return self.debit_amount if self.debit_amount > 0 else self.credit_amount
	
	def is_debit(self) -> bool:
		"""Check if posting is a debit"""
		return self.debit_amount > 0


class CFGLTrialBalance(Model, AuditMixin, BaseMixin):
	"""
	Trial balance snapshots for performance and historical reporting.
	
	Stores account balances as of specific dates to avoid
	recalculating from all postings for each report.
	"""
	__tablename__ = 'cf_gl_trial_balance'
	
	# Identity
	trial_balance_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Trial Balance Information
	as_of_date = Column(Date, nullable=False, index=True)
	period_id = Column(String(36), ForeignKey('cf_gl_period.period_id'), nullable=False, index=True)
	
	# Account Balance
	account_id = Column(String(36), ForeignKey('cf_gl_account.account_id'), nullable=False, index=True)
	opening_balance = Column(DECIMAL(15, 2), default=0.00)
	period_debits = Column(DECIMAL(15, 2), default=0.00)
	period_credits = Column(DECIMAL(15, 2), default=0.00)
	ending_balance = Column(DECIMAL(15, 2), default=0.00)
	
	# Generation Info
	generated_date = Column(DateTime, default=datetime.utcnow)
	generated_by = Column(String(36), nullable=False)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'as_of_date', 'account_id', name='uq_trial_balance_date_account'),
	)
	
	# Relationships
	account = relationship("CFGLAccount")
	period = relationship("CFGLPeriod")
	
	def __repr__(self):
		return f"<CFGLTrialBalance {self.account.account_code} as of {self.as_of_date}>"