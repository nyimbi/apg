"""
Financial Management Models

Database models for comprehensive financial management and accounting system
including multi-currency support, automated workflows, and compliance features.
"""

from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, JSON, ForeignKey, Numeric
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str
from decimal import Decimal
import json

from ..auth_rbac.models import BaseMixin, AuditMixin, Model


def uuid7str():
	"""Generate UUID7 string for consistent ID generation"""
	from uuid_extensions import uuid7
	return str(uuid7())


class FMChartOfAccounts(Model, AuditMixin, BaseMixin):
	"""
	Chart of accounts for financial management system.
	
	Defines the hierarchical structure of financial accounts with
	multi-currency support and comprehensive categorization.
	"""
	__tablename__ = 'fm_chart_of_accounts'
	
	# Identity
	account_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Account Definition
	account_code = Column(String(50), nullable=False, index=True)  # Unique account code
	account_name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	
	# Account Classification
	account_type = Column(String(20), nullable=False, index=True)  # asset, liability, equity, revenue, expense
	account_subtype = Column(String(50), nullable=True, index=True)  # current_asset, fixed_asset, etc.
	normal_balance = Column(String(10), nullable=False)  # debit, credit
	
	# Hierarchy
	parent_account_id = Column(String(36), ForeignKey('fm_chart_of_accounts.account_id'), nullable=True, index=True)
	level = Column(Integer, default=0)  # Hierarchy level (0 = root)
	path = Column(String(500), nullable=True)  # Hierarchical path for queries
	
	# Currency and Localization
	base_currency = Column(String(3), default='USD')  # ISO currency code
	multi_currency_enabled = Column(Boolean, default=False)
	allowed_currencies = Column(JSON, default=list)  # List of allowed currencies
	
	# Account Properties
	is_active = Column(Boolean, default=True, index=True)
	is_system_account = Column(Boolean, default=False)  # System-managed account
	is_bank_account = Column(Boolean, default=False)
	is_cash_account = Column(Boolean, default=False)
	allow_manual_entries = Column(Boolean, default=True)
	
	# Tax and Compliance
	tax_code = Column(String(20), nullable=True)
	vat_rate = Column(Numeric(5, 4), nullable=True)  # VAT/tax rate as decimal
	tax_account_id = Column(String(36), nullable=True)  # Associated tax account
	
	# Organizational
	cost_center = Column(String(50), nullable=True)
	department = Column(String(100), nullable=True)
	profit_center = Column(String(50), nullable=True)
	
	# Financial Controls
	requires_approval = Column(Boolean, default=False)
	approval_limit = Column(Numeric(15, 2), nullable=True)
	budget_enabled = Column(Boolean, default=False)
	
	# Reporting and Analysis
	financial_statement_category = Column(String(50), nullable=True)
	report_order = Column(Integer, default=0)  # Display order in reports
	consolidation_account = Column(String(50), nullable=True)
	
	# Balance Information
	opening_balance = Column(Numeric(15, 2), default=0.00)
	current_balance = Column(Numeric(15, 2), default=0.00)
	ytd_balance = Column(Numeric(15, 2), default=0.00)
	last_transaction_date = Column(DateTime, nullable=True)
	
	# Relationships
	parent_account = relationship("FMChartOfAccounts", remote_side=[account_id], back_populates="child_accounts")
	child_accounts = relationship("FMChartOfAccounts", back_populates="parent_account", cascade="all, delete-orphan")
	journal_entries = relationship("FMJournalEntryLine", back_populates="account")
	budgets = relationship("FMBudget", back_populates="account", cascade="all, delete-orphan")
	
	def __repr__(self):
		return f"<FMChartOfAccounts {self.account_code}: {self.account_name}>"
	
	def get_full_account_name(self) -> str:
		"""Get hierarchical account name"""
		if self.parent_account:
			return f"{self.parent_account.get_full_account_name()} > {self.account_name}"
		return self.account_name
	
	def is_debit_account(self) -> bool:
		"""Check if account has normal debit balance"""
		return self.normal_balance == 'debit'
	
	def is_credit_account(self) -> bool:
		"""Check if account has normal credit balance"""
		return self.normal_balance == 'credit'
	
	def calculate_balance(self, as_of_date: datetime = None) -> Decimal:
		"""Calculate account balance as of specific date"""
		# This would query journal entries and sum debits/credits
		# Implementation would include proper currency conversion
		return Decimal(str(self.current_balance))
	
	def get_children_recursive(self) -> List['FMChartOfAccounts']:
		"""Get all descendant accounts"""
		children = list(self.child_accounts)
		for child in self.child_accounts:
			children.extend(child.get_children_recursive())
		return children
	
	def update_balance(self, debit_amount: Decimal = None, credit_amount: Decimal = None) -> None:
		"""Update account balance with new transaction amounts"""
		if debit_amount:
			if self.is_debit_account():
				self.current_balance += debit_amount
			else:
				self.current_balance -= debit_amount
		
		if credit_amount:
			if self.is_credit_account():
				self.current_balance += credit_amount
			else:
				self.current_balance -= credit_amount
		
		self.last_transaction_date = datetime.utcnow()


class FMJournalEntry(Model, AuditMixin, BaseMixin):
	"""
	Journal entries for financial transactions.
	
	Records financial transactions with comprehensive audit trails,
	multi-currency support, and approval workflows.
	"""
	__tablename__ = 'fm_journal_entry'
	
	# Identity
	entry_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Entry Information
	entry_number = Column(String(50), nullable=False, unique=True, index=True)
	entry_date = Column(DateTime, nullable=False, index=True)
	posting_date = Column(DateTime, nullable=True, index=True)
	
	# Transaction Details
	transaction_type = Column(String(50), nullable=False, index=True)  # journal_entry, invoice, payment, etc.
	reference_number = Column(String(100), nullable=True)
	description = Column(Text, nullable=False)
	memo = Column(Text, nullable=True)
	
	# Financial Information
	base_currency = Column(String(3), default='USD')
	total_debits = Column(Numeric(15, 2), nullable=False, default=0.00)
	total_credits = Column(Numeric(15, 2), nullable=False, default=0.00)
	
	# Status and Workflow
	status = Column(String(20), default='draft', index=True)  # draft, pending, approved, posted, reversed
	workflow_stage = Column(String(50), nullable=True)
	requires_approval = Column(Boolean, default=False)
	approved_by = Column(String(36), nullable=True)  # User ID
	approved_at = Column(DateTime, nullable=True)
	
	# Source Information
	source_document_type = Column(String(50), nullable=True)  # invoice, receipt, bank_statement, etc.
	source_document_id = Column(String(36), nullable=True)
	source_system = Column(String(100), nullable=True)
	
	# Period and Closing
	fiscal_period = Column(String(10), nullable=False, index=True)  # YYYY-MM format
	fiscal_year = Column(Integer, nullable=False, index=True)
	is_closing_entry = Column(Boolean, default=False)
	is_adjusting_entry = Column(Boolean, default=False)
	
	# Reversal Information
	is_reversed = Column(Boolean, default=False, index=True)
	reversed_by_entry_id = Column(String(36), nullable=True)
	reversal_reason = Column(Text, nullable=True)
	
	# Processing Information
	posted_by = Column(String(36), nullable=True, index=True)  # User ID
	batch_id = Column(String(36), nullable=True, index=True)  # For batch processing
	import_batch_id = Column(String(36), nullable=True)  # For imported entries
	
	# Additional Context
	cost_center = Column(String(50), nullable=True)
	department = Column(String(100), nullable=True)
	project_id = Column(String(36), nullable=True)
	tags = Column(JSON, default=list)
	
	# Compliance and Audit
	compliance_flags = Column(JSON, default=dict)  # Compliance-related flags
	audit_trail = Column(JSON, default=list)  # Detailed audit trail
	locked = Column(Boolean, default=False)  # Prevents modification
	lock_reason = Column(String(200), nullable=True)
	
	# Relationships
	journal_lines = relationship("FMJournalEntryLine", back_populates="journal_entry", cascade="all, delete-orphan")
	attachments = relationship("FMDocument", back_populates="journal_entry", cascade="all, delete-orphan")
	
	def __repr__(self):
		return f"<FMJournalEntry {self.entry_number} ({self.status})>"
	
	def is_balanced(self) -> bool:
		"""Check if journal entry is balanced (debits = credits)"""
		return abs(self.total_debits - self.total_credits) < Decimal('0.01')
	
	def can_be_posted(self) -> bool:
		"""Check if entry can be posted"""
		return (self.status in ['approved', 'pending'] and 
				self.is_balanced() and 
				not self.is_reversed and
				not self.locked)
	
	def calculate_totals(self) -> None:
		"""Recalculate debit and credit totals from lines"""
		self.total_debits = sum(line.debit_amount for line in self.journal_lines)
		self.total_credits = sum(line.credit_amount for line in self.journal_lines)
	
	def post_entry(self, posted_by_user_id: str) -> bool:
		"""Post the journal entry"""
		if not self.can_be_posted():
			return False
		
		self.status = 'posted'
		self.posting_date = datetime.utcnow()
		self.posted_by = posted_by_user_id
		
		# Update account balances
		for line in self.journal_lines:
			line.account.update_balance(line.debit_amount, line.credit_amount)
		
		# Add to audit trail
		self.audit_trail.append({
			'action': 'posted',
			'user_id': posted_by_user_id,
			'timestamp': datetime.utcnow().isoformat(),
			'details': 'Journal entry posted to general ledger'
		})
		
		return True
	
	def reverse_entry(self, reason: str, reversed_by_user_id: str) -> 'FMJournalEntry':
		"""Create reversal entry"""
		if self.status != 'posted':
			raise ValueError("Only posted entries can be reversed")
		
		# Create reversal entry
		reversal = FMJournalEntry(
			tenant_id=self.tenant_id,
			entry_number=f"REV-{self.entry_number}",
			entry_date=datetime.utcnow(),
			transaction_type='reversal',
			description=f"Reversal of {self.entry_number}: {reason}",
			base_currency=self.base_currency,
			fiscal_period=datetime.utcnow().strftime('%Y-%m'),
			fiscal_year=datetime.utcnow().year,
			status='posted',
			posting_date=datetime.utcnow(),
			posted_by=reversed_by_user_id
		)
		
		# Create reversal lines (swap debits and credits)
		for line in self.journal_lines:
			reversal_line = FMJournalEntryLine(
				account_id=line.account_id,
				debit_amount=line.credit_amount,
				credit_amount=line.debit_amount,
				description=f"Reversal: {line.description}",
				currency=line.currency,
				exchange_rate=line.exchange_rate
			)
			reversal.journal_lines.append(reversal_line)
		
		reversal.calculate_totals()
		
		# Update original entry
		self.is_reversed = True
		self.reversed_by_entry_id = reversal.entry_id
		self.reversal_reason = reason
		
		return reversal


class FMJournalEntryLine(Model, AuditMixin, BaseMixin):
	"""
	Individual lines within journal entries.
	
	Records individual debit/credit lines with account references,
	currency conversion, and detailed transaction context.
	"""
	__tablename__ = 'fm_journal_entry_line'
	
	# Identity
	line_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	entry_id = Column(String(36), ForeignKey('fm_journal_entry.entry_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Account Reference
	account_id = Column(String(36), ForeignKey('fm_chart_of_accounts.account_id'), nullable=False, index=True)
	
	# Financial Amounts
	debit_amount = Column(Numeric(15, 2), default=0.00)
	credit_amount = Column(Numeric(15, 2), default=0.00)
	
	# Currency Information
	currency = Column(String(3), default='USD')
	exchange_rate = Column(Numeric(10, 6), default=1.000000)
	base_currency_debit = Column(Numeric(15, 2), default=0.00)  # Amount in base currency
	base_currency_credit = Column(Numeric(15, 2), default=0.00)
	
	# Line Details
	description = Column(Text, nullable=False)
	reference = Column(String(100), nullable=True)
	line_number = Column(Integer, nullable=False)  # Line sequence within entry
	
	# Dimensional Analysis
	cost_center = Column(String(50), nullable=True)
	department = Column(String(100), nullable=True)
	project_id = Column(String(36), nullable=True)
	employee_id = Column(String(36), nullable=True)
	customer_id = Column(String(36), nullable=True)
	vendor_id = Column(String(36), nullable=True)
	
	# Tax Information
	tax_code = Column(String(20), nullable=True)
	tax_amount = Column(Numeric(15, 2), default=0.00)
	tax_rate = Column(Numeric(5, 4), nullable=True)
	is_tax_inclusive = Column(Boolean, default=False)
	
	# Reconciliation
	is_reconciled = Column(Boolean, default=False)
	reconciliation_id = Column(String(36), nullable=True)
	reconciliation_date = Column(DateTime, nullable=True)
	
	# Additional Context
	quantity = Column(Numeric(10, 4), nullable=True)  # For unit-based transactions
	unit_price = Column(Numeric(15, 4), nullable=True)
	unit_of_measure = Column(String(20), nullable=True)
	
	# Metadata
	tags = Column(JSON, default=list)
	custom_fields = Column(JSON, default=dict)
	
	# Relationships
	journal_entry = relationship("FMJournalEntry", back_populates="journal_lines")
	account = relationship("FMChartOfAccounts", back_populates="journal_entries")
	
	def __repr__(self):
		return f"<FMJournalEntryLine {self.account_id}: Dr {self.debit_amount} Cr {self.credit_amount}>"
	
	def get_net_amount(self) -> Decimal:
		"""Get net amount (debit - credit)"""
		return self.debit_amount - self.credit_amount
	
	def is_debit_line(self) -> bool:
		"""Check if this is a debit line"""
		return self.debit_amount > 0
	
	def is_credit_line(self) -> bool:
		"""Check if this is a credit line"""
		return self.credit_amount > 0
	
	def calculate_base_currency_amounts(self) -> None:
		"""Calculate amounts in base currency using exchange rate"""
		self.base_currency_debit = self.debit_amount * self.exchange_rate
		self.base_currency_credit = self.credit_amount * self.exchange_rate
	
	def calculate_tax_amount(self) -> None:
		"""Calculate tax amount based on tax rate"""
		if self.tax_rate:
			base_amount = max(self.debit_amount, self.credit_amount)
			if self.is_tax_inclusive:
				self.tax_amount = base_amount * self.tax_rate / (1 + self.tax_rate)
			else:
				self.tax_amount = base_amount * self.tax_rate


class FMBudget(Model, AuditMixin, BaseMixin):
	"""
	Budget management for financial planning and control.
	
	Stores budget allocations by account, period, and dimensions
	with variance analysis and approval workflows.
	"""
	__tablename__ = 'fm_budget'
	
	# Identity
	budget_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Budget Definition
	budget_name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	budget_type = Column(String(50), nullable=False, index=True)  # operating, capital, cash_flow, master
	
	# Account and Hierarchy
	account_id = Column(String(36), ForeignKey('fm_chart_of_accounts.account_id'), nullable=False, index=True)
	
	# Period Information
	fiscal_year = Column(Integer, nullable=False, index=True)
	budget_period = Column(String(20), nullable=False, index=True)  # annual, quarterly, monthly
	start_date = Column(DateTime, nullable=False)
	end_date = Column(DateTime, nullable=False)
	
	# Budget Amounts
	budgeted_amount = Column(Numeric(15, 2), nullable=False)
	currency = Column(String(3), default='USD')
	
	# Period Breakdown (for monthly/quarterly budgets)
	period_amounts = Column(JSON, default=dict)  # Amount by period
	
	# Dimensional Analysis
	cost_center = Column(String(50), nullable=True, index=True)
	department = Column(String(100), nullable=True, index=True)
	project_id = Column(String(36), nullable=True, index=True)
	
	# Status and Approval
	status = Column(String(20), default='draft', index=True)  # draft, submitted, approved, locked
	version = Column(Integer, default=1)
	approved_by = Column(String(36), nullable=True)
	approved_at = Column(DateTime, nullable=True)
	
	# Variance Analysis
	actual_amount = Column(Numeric(15, 2), default=0.00)
	variance_amount = Column(Numeric(15, 2), default=0.00)
	variance_percentage = Column(Numeric(5, 2), default=0.00)
	last_variance_calculation = Column(DateTime, nullable=True)
	
	# Forecasting
	forecast_amount = Column(Numeric(15, 2), nullable=True)
	forecast_confidence = Column(Numeric(3, 2), nullable=True)  # 0-1 confidence level
	forecast_method = Column(String(50), nullable=True)  # linear, seasonal, ml
	
	# Control and Alerts
	enable_budget_control = Column(Boolean, default=False)
	over_budget_allowed = Column(Boolean, default=False)
	alert_threshold_percentage = Column(Numeric(3, 2), default=0.90)  # Alert at 90%
	
	# Workflow
	created_by = Column(String(36), nullable=True, index=True)
	last_modified_by = Column(String(36), nullable=True)
	
	# Relationships
	account = relationship("FMChartOfAccounts", back_populates="budgets")
	
	def __repr__(self):
		return f"<FMBudget {self.budget_name} FY{self.fiscal_year}: {self.budgeted_amount}>"
	
	def calculate_variance(self) -> None:
		"""Calculate budget variance"""
		self.variance_amount = self.actual_amount - self.budgeted_amount
		
		if self.budgeted_amount != 0:
			self.variance_percentage = (self.variance_amount / self.budgeted_amount) * 100
		else:
			self.variance_percentage = 0
		
		self.last_variance_calculation = datetime.utcnow()
	
	def is_over_budget(self) -> bool:
		"""Check if actual spending exceeds budget"""
		return self.actual_amount > self.budgeted_amount
	
	def get_utilization_percentage(self) -> Decimal:
		"""Get budget utilization percentage"""
		if self.budgeted_amount == 0:
			return Decimal('0')
		return (self.actual_amount / self.budgeted_amount) * 100
	
	def should_alert(self) -> bool:
		"""Check if budget utilization should trigger alert"""
		if not self.enable_budget_control:
			return False
		
		utilization = self.get_utilization_percentage()
		return utilization >= (self.alert_threshold_percentage * 100)
	
	def get_remaining_budget(self) -> Decimal:
		"""Get remaining budget amount"""
		return self.budgeted_amount - self.actual_amount
	
	def update_actuals(self, actual_amount: Decimal) -> None:
		"""Update actual amounts and recalculate variance"""
		self.actual_amount = actual_amount
		self.calculate_variance()


class FMInvoice(Model, AuditMixin, BaseMixin):
	"""
	Customer invoices and billing management.
	
	Manages customer invoicing with comprehensive billing features,
	payment tracking, and integration with accounting.
	"""
	__tablename__ = 'fm_invoice'
	
	# Identity
	invoice_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Invoice Information
	invoice_number = Column(String(50), nullable=False, unique=True, index=True)
	invoice_date = Column(DateTime, nullable=False, index=True)
	due_date = Column(DateTime, nullable=False, index=True)
	
	# Customer Information
	customer_id = Column(String(36), nullable=False, index=True)
	customer_name = Column(String(200), nullable=False)
	billing_address = Column(JSON, default=dict)
	shipping_address = Column(JSON, default=dict)
	
	# Financial Information
	subtotal = Column(Numeric(15, 2), nullable=False, default=0.00)
	tax_amount = Column(Numeric(15, 2), default=0.00)
	discount_amount = Column(Numeric(15, 2), default=0.00)
	shipping_amount = Column(Numeric(15, 2), default=0.00)
	total_amount = Column(Numeric(15, 2), nullable=False)
	
	# Currency and Payments
	currency = Column(String(3), default='USD')
	exchange_rate = Column(Numeric(10, 6), default=1.000000)
	amount_paid = Column(Numeric(15, 2), default=0.00)
	amount_due = Column(Numeric(15, 2), nullable=False)
	
	# Status and Lifecycle
	status = Column(String(20), default='draft', index=True)  # draft, sent, paid, overdue, cancelled
	payment_status = Column(String(20), default='unpaid', index=True)  # unpaid, partial, paid, overpaid
	sent_date = Column(DateTime, nullable=True)
	
	# Terms and Conditions
	payment_terms = Column(String(50), nullable=True)  # net_30, net_15, due_on_receipt
	payment_method = Column(String(50), nullable=True)
	late_fee_rate = Column(Numeric(5, 4), nullable=True)
	discount_terms = Column(String(50), nullable=True)  # 2/10_net_30
	
	# Reference Information
	purchase_order = Column(String(100), nullable=True)
	project_id = Column(String(36), nullable=True)
	sales_order_id = Column(String(36), nullable=True)
	
	# Recurring Invoice Information
	is_recurring = Column(Boolean, default=False)
	recurring_frequency = Column(String(20), nullable=True)  # monthly, quarterly, annually
	next_invoice_date = Column(DateTime, nullable=True)
	parent_recurring_id = Column(String(36), nullable=True)
	
	# Communication
	notes = Column(Text, nullable=True)
	internal_notes = Column(Text, nullable=True)
	email_sent_count = Column(Integer, default=0)
	last_email_sent = Column(DateTime, nullable=True)
	
	# Aging Information
	days_outstanding = Column(Integer, default=0)
	aging_bucket = Column(String(20), nullable=True)  # current, 1-30, 31-60, 61-90, 90+
	
	# Relationships
	invoice_lines = relationship("FMInvoiceLine", back_populates="invoice", cascade="all, delete-orphan")
	payments = relationship("FMPayment", back_populates="invoice")
	
	def __repr__(self):
		return f"<FMInvoice {self.invoice_number}: {self.total_amount} ({self.status})>"
	
	def calculate_totals(self) -> None:
		"""Calculate invoice totals from line items"""
		self.subtotal = sum(line.line_total for line in self.invoice_lines)
		
		# Calculate total with tax, discount, and shipping
		self.total_amount = self.subtotal + self.tax_amount - self.discount_amount + self.shipping_amount
		self.amount_due = self.total_amount - self.amount_paid
	
	def is_overdue(self) -> bool:
		"""Check if invoice is overdue"""
		return (self.status != 'paid' and 
				self.due_date < datetime.utcnow() and
				self.amount_due > 0)
	
	def calculate_days_outstanding(self) -> None:
		"""Calculate days outstanding"""
		if self.status == 'paid':
			self.days_outstanding = 0
		else:
			self.days_outstanding = (datetime.utcnow() - self.invoice_date).days
		
		# Update aging bucket
		if self.days_outstanding <= 0:
			self.aging_bucket = 'current'
		elif self.days_outstanding <= 30:
			self.aging_bucket = '1-30'
		elif self.days_outstanding <= 60:
			self.aging_bucket = '31-60'
		elif self.days_outstanding <= 90:
			self.aging_bucket = '61-90'
		else:
			self.aging_bucket = '90+'
	
	def apply_payment(self, payment_amount: Decimal) -> None:
		"""Apply payment to invoice"""
		self.amount_paid += payment_amount
		self.amount_due = self.total_amount - self.amount_paid
		
		# Update payment status
		if self.amount_due <= 0:
			self.payment_status = 'paid'
			self.status = 'paid'
		elif self.amount_paid > 0:
			self.payment_status = 'partial'
		
		if self.amount_paid > self.total_amount:
			self.payment_status = 'overpaid'
	
	def calculate_late_fees(self) -> Decimal:
		"""Calculate late fees if applicable"""
		if not self.is_overdue() or not self.late_fee_rate:
			return Decimal('0.00')
		
		days_late = max(0, (datetime.utcnow() - self.due_date).days)
		return self.amount_due * self.late_fee_rate * (days_late / 30)  # Monthly rate


class FMInvoiceLine(Model, AuditMixin, BaseMixin):
	"""
	Individual line items within invoices.
	
	Records detailed line item information with product/service details,
	pricing, taxes, and accounting integration.
	"""
	__tablename__ = 'fm_invoice_line'
	
	# Identity
	line_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	invoice_id = Column(String(36), ForeignKey('fm_invoice.invoice_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Line Information
	line_number = Column(Integer, nullable=False)
	description = Column(Text, nullable=False)
	
	# Product/Service Details
	product_id = Column(String(36), nullable=True, index=True)
	product_code = Column(String(50), nullable=True)
	service_id = Column(String(36), nullable=True)
	
	# Quantity and Pricing
	quantity = Column(Numeric(10, 4), nullable=False, default=1.0000)
	unit_price = Column(Numeric(15, 4), nullable=False)
	unit_of_measure = Column(String(20), nullable=True)
	discount_percentage = Column(Numeric(5, 4), default=0.0000)
	discount_amount = Column(Numeric(15, 2), default=0.00)
	
	# Calculated Amounts
	line_subtotal = Column(Numeric(15, 2), nullable=False)  # quantity * unit_price
	line_discount = Column(Numeric(15, 2), default=0.00)
	line_total = Column(Numeric(15, 2), nullable=False)  # subtotal - discount
	
	# Tax Information
	tax_code = Column(String(20), nullable=True)
	tax_rate = Column(Numeric(5, 4), nullable=True)
	tax_amount = Column(Numeric(15, 2), default=0.00)
	is_tax_inclusive = Column(Boolean, default=False)
	
	# Accounting Integration
	revenue_account_id = Column(String(36), nullable=True)
	cost_account_id = Column(String(36), nullable=True)
	
	# Dimensional Analysis
	cost_center = Column(String(50), nullable=True)
	department = Column(String(100), nullable=True)
	project_id = Column(String(36), nullable=True)
	
	# Additional Details
	notes = Column(Text, nullable=True)
	custom_fields = Column(JSON, default=dict)
	
	# Relationships
	invoice = relationship("FMInvoice", back_populates="invoice_lines")
	
	def __repr__(self):
		return f"<FMInvoiceLine {self.description}: {self.quantity} x {self.unit_price}>"
	
	def calculate_amounts(self) -> None:
		"""Calculate line amounts"""
		self.line_subtotal = self.quantity * self.unit_price
		
		# Calculate discount
		if self.discount_percentage > 0:
			self.line_discount = self.line_subtotal * self.discount_percentage
		else:
			self.line_discount = self.discount_amount
		
		self.line_total = self.line_subtotal - self.line_discount
		
		# Calculate tax
		if self.tax_rate:
			if self.is_tax_inclusive:
				self.tax_amount = self.line_total * self.tax_rate / (1 + self.tax_rate)
			else:
				self.tax_amount = self.line_total * self.tax_rate
	
	def get_effective_unit_price(self) -> Decimal:
		"""Get unit price after discount"""
		if self.quantity == 0:
			return Decimal('0.00')
		return self.line_total / self.quantity


class FMPayment(Model, AuditMixin, BaseMixin):
	"""
	Customer payments and cash receipts.
	
	Records customer payments with comprehensive tracking,
	bank reconciliation, and accounting integration.
	"""
	__tablename__ = 'fm_payment'
	
	# Identity
	payment_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Payment Information
	payment_number = Column(String(50), nullable=False, unique=True, index=True)
	payment_date = Column(DateTime, nullable=False, index=True)
	
	# Customer and Invoice
	customer_id = Column(String(36), nullable=False, index=True)
	invoice_id = Column(String(36), ForeignKey('fm_invoice.invoice_id'), nullable=True, index=True)
	
	# Payment Details
	payment_amount = Column(Numeric(15, 2), nullable=False)
	currency = Column(String(3), default='USD')
	exchange_rate = Column(Numeric(10, 6), default=1.000000)
	base_currency_amount = Column(Numeric(15, 2), nullable=False)
	
	# Payment Method
	payment_method = Column(String(50), nullable=False, index=True)  # cash, check, credit_card, bank_transfer, etc.
	payment_reference = Column(String(100), nullable=True)  # Check number, transaction ID, etc.
	
	# Bank Information
	bank_account_id = Column(String(36), nullable=True, index=True)
	deposit_account_id = Column(String(36), nullable=True)  # GL account for deposit
	
	# Status and Processing
	status = Column(String(20), default='received', index=True)  # received, deposited, cleared, bounced
	is_deposited = Column(Boolean, default=False)
	deposit_date = Column(DateTime, nullable=True)
	is_cleared = Column(Boolean, default=False)
	cleared_date = Column(DateTime, nullable=True)
	
	# Reconciliation
	is_reconciled = Column(Boolean, default=False)
	reconciliation_id = Column(String(36), nullable=True)
	reconciliation_date = Column(DateTime, nullable=True)
	bank_statement_line_id = Column(String(36), nullable=True)
	
	# Fees and Charges
	processing_fee = Column(Numeric(15, 2), default=0.00)
	bank_charges = Column(Numeric(15, 2), default=0.00)
	net_amount = Column(Numeric(15, 2), nullable=False)  # Amount after fees
	
	# Additional Information
	notes = Column(Text, nullable=True)
	internal_notes = Column(Text, nullable=True)
	received_by = Column(String(36), nullable=True)  # User ID
	
	# Overpayment Handling
	overpayment_amount = Column(Numeric(15, 2), default=0.00)
	overpayment_action = Column(String(20), nullable=True)  # credit, refund, apply_to_next
	
	# Relationships
	invoice = relationship("FMInvoice", back_populates="payments")
	
	def __repr__(self):
		return f"<FMPayment {self.payment_number}: {self.payment_amount} ({self.payment_method})>"
	
	def calculate_net_amount(self) -> None:
		"""Calculate net amount after fees and charges"""
		self.net_amount = self.payment_amount - self.processing_fee - self.bank_charges
	
	def can_be_deposited(self) -> bool:
		"""Check if payment can be deposited"""
		return (self.status == 'received' and 
				not self.is_deposited and
				self.payment_method in ['check', 'cash', 'money_order'])
	
	def deposit_payment(self, deposit_date: datetime = None) -> None:
		"""Mark payment as deposited"""
		if not self.can_be_deposited():
			return
		
		self.is_deposited = True
		self.deposit_date = deposit_date or datetime.utcnow()
		self.status = 'deposited'
	
	def clear_payment(self, cleared_date: datetime = None) -> None:
		"""Mark payment as cleared by bank"""
		self.is_cleared = True
		self.cleared_date = cleared_date or datetime.utcnow()
		if self.status == 'deposited':
			self.status = 'cleared'
	
	def handle_overpayment(self) -> Decimal:
		"""Calculate and handle overpayment"""
		if self.invoice:
			if self.payment_amount > self.invoice.amount_due:
				self.overpayment_amount = self.payment_amount - self.invoice.amount_due
				return self.overpayment_amount
		return Decimal('0.00')


class FMFinancialReport(Model, AuditMixin, BaseMixin):
	"""
	Generated financial reports with caching and version control.
	
	Stores generated financial reports with parameters, data,
	and comprehensive audit trails for compliance.
	"""
	__tablename__ = 'fm_financial_report'
	
	# Identity
	report_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Report Definition
	report_name = Column(String(200), nullable=False)
	report_type = Column(String(50), nullable=False, index=True)  # balance_sheet, income_statement, etc.
	report_format = Column(String(20), default='json')  # json, pdf, excel, csv
	
	# Report Parameters
	parameters = Column(JSON, nullable=False)  # Report generation parameters
	date_range_start = Column(DateTime, nullable=True, index=True)
	date_range_end = Column(DateTime, nullable=True, index=True)
	fiscal_period = Column(String(20), nullable=True, index=True)
	
	# Filtering and Grouping
	account_filters = Column(JSON, default=dict)  # Account filtering criteria
	dimensional_filters = Column(JSON, default=dict)  # Cost center, department, etc.
	consolidation_level = Column(String(50), nullable=True)
	
	# Report Status
	status = Column(String(20), default='generating', index=True)  # generating, completed, failed, expired
	generated_at = Column(DateTime, nullable=True, index=True)
	generation_time_ms = Column(Float, nullable=True)
	
	# Report Data
	report_data = Column(JSON, nullable=True)  # Structured report data
	summary_metrics = Column(JSON, default=dict)  # Key summary metrics
	file_path = Column(String(1000), nullable=True)  # Path to generated file
	file_size_bytes = Column(Integer, nullable=True)
	
	# Version and Caching
	version = Column(Integer, default=1)
	is_cached = Column(Boolean, default=True)
	cache_expires_at = Column(DateTime, nullable=True)
	hash_key = Column(String(64), nullable=True, index=True)  # For cache invalidation
	
	# Audit and Compliance
	generated_by = Column(String(36), nullable=True, index=True)  # User ID
	data_source_info = Column(JSON, default=dict)  # Information about data sources
	compliance_flags = Column(JSON, default=dict)
	
	# Distribution
	recipients = Column(JSON, default=list)  # Report recipients
	distribution_status = Column(JSON, default=dict)  # Distribution tracking
	
	# Error Information
	error_message = Column(Text, nullable=True)
	warnings = Column(JSON, default=list)
	
	def __repr__(self):
		return f"<FMFinancialReport {self.report_name} ({self.report_type})>"
	
	def is_expired(self) -> bool:
		"""Check if cached report is expired"""
		return (self.cache_expires_at is not None and 
				datetime.utcnow() > self.cache_expires_at)
	
	def is_valid(self) -> bool:
		"""Check if report is valid and complete"""
		return (self.status == 'completed' and 
				self.report_data is not None and
				not self.is_expired())
	
	def get_generation_duration(self) -> Optional[float]:
		"""Get report generation duration in seconds"""
		return self.generation_time_ms / 1000 if self.generation_time_ms else None
	
	def calculate_file_size_formatted(self) -> str:
		"""Get formatted file size"""
		if not self.file_size_bytes:
			return "0 B"
		
		size = self.file_size_bytes
		if size < 1024:
			return f"{size} B"
		elif size < 1024 * 1024:
			return f"{size / 1024:.1f} KB"
		else:
			return f"{size / (1024 * 1024):.1f} MB"
	
	def set_cache_expiry(self, hours: int = 24) -> None:
		"""Set cache expiry time"""
		self.cache_expires_at = datetime.utcnow() + timedelta(hours=hours)
	
	def invalidate_cache(self) -> None:
		"""Invalidate cached report"""
		self.cache_expires_at = datetime.utcnow()
		self.is_cached = False


class FMDocument(Model, AuditMixin, BaseMixin):
	"""
	Document attachments for financial records.
	
	Manages document storage and metadata for invoices, receipts,
	contracts, and other financial documentation.
	"""
	__tablename__ = 'fm_document'
	
	# Identity
	document_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Document Information
	filename = Column(String(500), nullable=False)
	original_filename = Column(String(500), nullable=False)
	file_path = Column(String(1000), nullable=False)
	file_size_bytes = Column(Integer, nullable=False)
	mime_type = Column(String(100), nullable=False)
	file_hash = Column(String(64), nullable=False, index=True)  # SHA-256
	
	# Document Classification
	document_type = Column(String(50), nullable=False, index=True)  # receipt, invoice, contract, statement
	category = Column(String(100), nullable=True)
	tags = Column(JSON, default=list)
	
	# Financial Context
	journal_entry_id = Column(String(36), ForeignKey('fm_journal_entry.entry_id'), nullable=True, index=True)
	invoice_id = Column(String(36), nullable=True, index=True)
	payment_id = Column(String(36), nullable=True, index=True)
	
	# Document Metadata
	description = Column(Text, nullable=True)
	document_date = Column(DateTime, nullable=True)
	reference_number = Column(String(100), nullable=True)
	
	# Processing Status
	is_processed = Column(Boolean, default=False)
	ocr_completed = Column(Boolean, default=False)
	ocr_text = Column(Text, nullable=True)
	ocr_confidence = Column(Float, nullable=True)
	
	# Security and Access
	is_confidential = Column(Boolean, default=False)
	access_level = Column(String(20), default='internal')  # public, internal, confidential, restricted
	
	# Version Control
	version = Column(Integer, default=1)
	parent_document_id = Column(String(36), nullable=True)
	is_latest_version = Column(Boolean, default=True)
	
	# Storage Information
	storage_location = Column(String(100), nullable=True)  # local, s3, azure, etc.
	backup_status = Column(String(20), default='pending')  # pending, completed, failed
	
	# Relationships
	journal_entry = relationship("FMJournalEntry", back_populates="attachments")
	
	def __repr__(self):
		return f"<FMDocument {self.filename} ({self.document_type})>"
	
	def get_file_size_formatted(self) -> str:
		"""Get formatted file size"""
		size = self.file_size_bytes
		if size < 1024:
			return f"{size} B"
		elif size < 1024 * 1024:
			return f"{size / 1024:.1f} KB"
		else:
			return f"{size / (1024 * 1024):.1f} MB"
	
	def is_image(self) -> bool:
		"""Check if document is an image"""
		image_types = ['image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'image/tiff']
		return self.mime_type in image_types
	
	def is_pdf(self) -> bool:
		"""Check if document is PDF"""
		return self.mime_type == 'application/pdf'
	
	def needs_ocr(self) -> bool:
		"""Check if document needs OCR processing"""
		return not self.ocr_completed and (self.is_image() or self.is_pdf())
	
	def update_version(self, new_file_path: str, new_file_size: int, new_hash: str) -> None:
		"""Create new version of document"""
		self.version += 1
		self.file_path = new_file_path
		self.file_size_bytes = new_file_size
		self.file_hash = new_hash
		self.is_processed = False
		self.ocr_completed = False