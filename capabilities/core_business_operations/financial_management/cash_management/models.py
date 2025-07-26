"""
Cash Management Models

Database models for the Cash Management sub-capability including bank accounts,
transactions, reconciliations, cash forecasting, and liquidity management.
"""

from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional
from decimal import Decimal
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, Date, DECIMAL, ForeignKey, UniqueConstraint, Index
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str
import json

from ...auth_rbac.models import BaseMixin, AuditMixin, Model


class CFCMBankAccount(Model, AuditMixin, BaseMixin):
	"""
	Bank account master data.
	
	Manages bank account information including account details, GL integration,
	reconciliation settings, and account status.
	"""
	__tablename__ = 'cf_cm_bank_account'
	
	# Identity
	bank_account_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Account Information
	account_number = Column(String(50), nullable=False, index=True)
	account_name = Column(String(200), nullable=False, index=True)
	account_type = Column(String(50), nullable=False, index=True)  # CHECKING, SAVINGS, MONEY_MARKET, etc.
	
	# Bank Information
	bank_name = Column(String(200), nullable=False)
	bank_code = Column(String(20), nullable=True)  # SWIFT/BIC code
	routing_number = Column(String(20), nullable=True)
	branch_code = Column(String(20), nullable=True)
	branch_name = Column(String(200), nullable=True)
	
	# Address Information
	bank_address_line1 = Column(String(100), nullable=True)
	bank_address_line2 = Column(String(100), nullable=True)
	bank_city = Column(String(50), nullable=True)
	bank_state = Column(String(50), nullable=True)
	bank_postal_code = Column(String(20), nullable=True)
	bank_country = Column(String(50), nullable=True)
	
	# Account Configuration
	currency_code = Column(String(3), default='USD', nullable=False)
	is_active = Column(Boolean, default=True)
	is_primary = Column(Boolean, default=False)  # Primary operating account
	allow_overdraft = Column(Boolean, default=False)
	overdraft_limit = Column(DECIMAL(15, 2), default=0.00)
	
	# GL Integration
	gl_account_id = Column(String(36), nullable=False, index=True)  # Linked GL account
	gl_clearing_account_id = Column(String(36), nullable=True)  # For in-transit items
	
	# Reconciliation Settings
	requires_reconciliation = Column(Boolean, default=True)
	auto_reconciliation = Column(Boolean, default=True)
	reconciliation_tolerance = Column(DECIMAL(15, 2), default=5.00)
	last_reconciliation_date = Column(Date, nullable=True)
	last_statement_date = Column(Date, nullable=True)
	
	# Balance Information
	current_balance = Column(DECIMAL(15, 2), default=0.00)
	available_balance = Column(DECIMAL(15, 2), default=0.00)
	ledger_balance = Column(DECIMAL(15, 2), default=0.00)  # GL balance
	statement_balance = Column(DECIMAL(15, 2), default=0.00)  # Last bank statement
	
	# Interest and Fees
	interest_rate = Column(DECIMAL(6, 4), default=0.0000)
	monthly_fee = Column(DECIMAL(8, 2), default=0.00)
	minimum_balance = Column(DECIMAL(15, 2), default=0.00)
	
	# Online Banking
	online_banking_enabled = Column(Boolean, default=False)
	bank_connection_id = Column(String(100), nullable=True)  # For automated imports
	last_import_date = Column(DateTime, nullable=True)
	
	# Contact Information
	bank_contact_name = Column(String(100), nullable=True)
	bank_contact_phone = Column(String(50), nullable=True)
	bank_contact_email = Column(String(100), nullable=True)
	
	# Internal Notes
	notes = Column(Text, nullable=True)
	internal_notes = Column(Text, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'account_number', name='uq_bank_account_number'),
		Index('idx_bank_account_tenant_active', 'tenant_id', 'is_active'),
	)
	
	# Relationships
	transactions = relationship("CFCMBankTransaction", back_populates="bank_account")
	reconciliations = relationship("CFCMReconciliation", back_populates="bank_account")
	cash_positions = relationship("CFCMCashPosition", back_populates="bank_account")
	transfers_from = relationship("CFCMCashTransfer", foreign_keys="CFCMCashTransfer.from_account_id", back_populates="from_account")
	transfers_to = relationship("CFCMCashTransfer", foreign_keys="CFCMCashTransfer.to_account_id", back_populates="to_account")
	deposits = relationship("CFCMDeposit", back_populates="bank_account")
	checks = relationship("CFCMCheckRegister", back_populates="bank_account")
	investments = relationship("CFCMInvestment", back_populates="bank_account")
	
	def __repr__(self):
		return f"<CFCMBankAccount {self.account_number} - {self.account_name}>"
	
	def get_current_balance(self, include_pending: bool = False) -> Decimal:
		"""Get current account balance"""
		if include_pending:
			pending_debits = sum(
				t.amount for t in self.transactions 
				if t.transaction_type in ['WITHDRAWAL', 'TRANSFER_OUT', 'CHECK', 'ACH_OUT', 'WIRE_OUT'] 
				and t.status == 'Pending'
			)
			pending_credits = sum(
				t.amount for t in self.transactions 
				if t.transaction_type in ['DEPOSIT', 'TRANSFER_IN', 'ACH_IN', 'WIRE_IN'] 
				and t.status == 'Pending'
			)
			return self.current_balance - pending_debits + pending_credits
		
		return self.current_balance
	
	def get_available_balance(self) -> Decimal:
		"""Get available balance considering overdraft limit"""
		balance = self.get_current_balance(include_pending=True)
		if self.allow_overdraft:
			return balance + self.overdraft_limit
		return max(balance, Decimal('0.00'))
	
	def is_overdrawn(self) -> bool:
		"""Check if account is overdrawn"""
		return self.current_balance < 0
	
	def can_withdraw(self, amount: Decimal) -> bool:
		"""Check if amount can be withdrawn"""
		return self.get_available_balance() >= amount
	
	def update_balance(self, amount: Decimal, transaction_type: str):
		"""Update account balance based on transaction"""
		if transaction_type in ['DEPOSIT', 'TRANSFER_IN', 'ACH_IN', 'WIRE_IN', 'INTEREST']:
			self.current_balance += amount
		elif transaction_type in ['WITHDRAWAL', 'TRANSFER_OUT', 'CHECK', 'ACH_OUT', 'WIRE_OUT', 'FEE', 'NSF']:
			self.current_balance -= amount


class CFCMBankTransaction(Model, AuditMixin, BaseMixin):
	"""
	Bank transactions and statements.
	
	Records all bank account transactions including deposits, withdrawals,
	transfers, and automated clearing house (ACH) transactions.
	"""
	__tablename__ = 'cf_cm_bank_transaction'
	
	# Identity
	transaction_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Account Reference
	bank_account_id = Column(String(36), ForeignKey('cf_cm_bank_account.bank_account_id'), nullable=False, index=True)
	
	# Transaction Information
	transaction_number = Column(String(50), nullable=True, index=True)
	bank_reference = Column(String(100), nullable=True, index=True)  # Bank's transaction ID
	description = Column(Text, nullable=False)
	
	# Transaction Details
	transaction_date = Column(Date, nullable=False, index=True)
	value_date = Column(Date, nullable=True)  # When funds are available
	posting_date = Column(Date, nullable=True)  # When posted to account
	
	# Amount and Type
	amount = Column(DECIMAL(15, 2), nullable=False)
	transaction_type = Column(String(50), nullable=False, index=True)
	transaction_code = Column(String(20), nullable=True)  # Bank-specific code
	
	# Direction and Status
	is_debit = Column(Boolean, nullable=False)  # True for outflow, False for inflow
	status = Column(String(20), default='Posted', index=True)  # Posted, Pending, Cleared, Returned
	
	# Running Balance
	running_balance = Column(DECIMAL(15, 2), nullable=True)
	
	# Reconciliation
	is_reconciled = Column(Boolean, default=False, index=True)
	reconciliation_id = Column(String(36), ForeignKey('cf_cm_reconciliation.reconciliation_id'), nullable=True, index=True)
	reconciled_date = Column(Date, nullable=True)
	
	# Source Information
	source_type = Column(String(50), nullable=True)  # CHECK, ACH, WIRE, MANUAL, IMPORT
	source_id = Column(String(36), nullable=True)  # Reference to source document
	check_number = Column(String(20), nullable=True, index=True)
	
	# Counterparty Information
	counterparty_name = Column(String(200), nullable=True)
	counterparty_account = Column(String(50), nullable=True)
	counterparty_bank = Column(String(200), nullable=True)
	
	# GL Integration
	gl_posted = Column(Boolean, default=False)
	gl_journal_id = Column(String(36), nullable=True)
	gl_account_id = Column(String(36), nullable=True)  # Offset GL account
	
	# Currency and Exchange
	currency_code = Column(String(3), default='USD')
	exchange_rate = Column(DECIMAL(10, 6), default=1.000000)
	home_currency_amount = Column(DECIMAL(15, 2), nullable=True)
	
	# Import Information
	imported = Column(Boolean, default=False)
	import_date = Column(DateTime, nullable=True)
	import_batch_id = Column(String(36), nullable=True)
	
	# Additional Data
	memo = Column(Text, nullable=True)
	tags = Column(String(500), nullable=True)  # JSON array of tags
	
	# Constraints
	__table_args__ = (
		Index('idx_bank_transaction_date_account', 'bank_account_id', 'transaction_date'),
		Index('idx_bank_transaction_reconciled', 'bank_account_id', 'is_reconciled'),
	)
	
	# Relationships
	bank_account = relationship("CFCMBankAccount", back_populates="transactions")
	reconciliation = relationship("CFCMReconciliation", back_populates="bank_transactions")
	reconciliation_items = relationship("CFCMReconciliationItem", back_populates="bank_transaction")
	
	def __repr__(self):
		return f"<CFCMBankTransaction {self.transaction_date}: {self.description} - ${self.amount}>"
	
	def get_display_amount(self) -> str:
		"""Get formatted amount for display"""
		if self.is_debit:
			return f"-${abs(self.amount):,.2f}"
		else:
			return f"${self.amount:,.2f}"
	
	def can_reconcile(self) -> bool:
		"""Check if transaction can be reconciled"""
		return not self.is_reconciled and self.status == 'Posted'
	
	def reconcile(self, reconciliation_id: str):
		"""Mark transaction as reconciled"""
		if not self.can_reconcile():
			raise ValueError("Transaction cannot be reconciled")
		
		self.is_reconciled = True
		self.reconciliation_id = reconciliation_id
		self.reconciled_date = date.today()
	
	def post_to_gl(self, gl_account_id: str, journal_id: str):
		"""Post transaction to General Ledger"""
		if self.gl_posted:
			return
		
		self.gl_posted = True
		self.gl_journal_id = journal_id
		self.gl_account_id = gl_account_id


class CFCMReconciliation(Model, AuditMixin, BaseMixin):
	"""
	Bank reconciliation process.
	
	Manages the bank reconciliation workflow including statement processing,
	transaction matching, and variance analysis.
	"""
	__tablename__ = 'cf_cm_reconciliation'
	
	# Identity
	reconciliation_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Account Reference
	bank_account_id = Column(String(36), ForeignKey('cf_cm_bank_account.bank_account_id'), nullable=False, index=True)
	
	# Reconciliation Information
	reconciliation_number = Column(String(50), nullable=False, index=True)
	reconciliation_name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	
	# Statement Information
	statement_date = Column(Date, nullable=False, index=True)
	statement_number = Column(String(50), nullable=True)
	statement_beginning_balance = Column(DECIMAL(15, 2), nullable=False)
	statement_ending_balance = Column(DECIMAL(15, 2), nullable=False)
	
	# Period Information
	period_start_date = Column(Date, nullable=False)
	period_end_date = Column(Date, nullable=False)
	
	# Status and Workflow
	status = Column(String(20), default='Draft', index=True)  # Draft, In Progress, Completed, Approved
	reconciliation_type = Column(String(50), default='Manual')  # Manual, Automated, Assisted
	
	# Balance Information
	book_balance = Column(DECIMAL(15, 2), nullable=False)  # GL balance
	adjusted_book_balance = Column(DECIMAL(15, 2), nullable=False)
	adjusted_bank_balance = Column(DECIMAL(15, 2), nullable=False)
	variance_amount = Column(DECIMAL(15, 2), default=0.00)
	
	# Transaction Counts
	total_deposits = Column(Integer, default=0)
	total_withdrawals = Column(Integer, default=0)
	matched_transactions = Column(Integer, default=0)
	unmatched_bank_items = Column(Integer, default=0)
	unmatched_book_items = Column(Integer, default=0)
	
	# Amount Totals
	total_deposit_amount = Column(DECIMAL(15, 2), default=0.00)
	total_withdrawal_amount = Column(DECIMAL(15, 2), default=0.00)
	total_adjustments = Column(DECIMAL(15, 2), default=0.00)
	
	# Completion Information
	reconciled_by = Column(String(36), nullable=True)
	reconciled_date = Column(DateTime, nullable=True)
	approved_by = Column(String(36), nullable=True)
	approved_date = Column(DateTime, nullable=True)
	
	# Import Information
	statement_imported = Column(Boolean, default=False)
	import_file_path = Column(String(500), nullable=True)
	import_date = Column(DateTime, nullable=True)
	
	# Notes and Comments
	notes = Column(Text, nullable=True)
	approval_notes = Column(Text, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'reconciliation_number', name='uq_reconciliation_number'),
		Index('idx_reconciliation_account_date', 'bank_account_id', 'statement_date'),
	)
	
	# Relationships
	bank_account = relationship("CFCMBankAccount", back_populates="reconciliations")
	reconciliation_items = relationship("CFCMReconciliationItem", back_populates="reconciliation", cascade="all, delete-orphan")
	bank_transactions = relationship("CFCMBankTransaction", back_populates="reconciliation")
	
	def __repr__(self):
		return f"<CFCMReconciliation {self.reconciliation_number} - {self.statement_date}>"
	
	def calculate_variance(self) -> Decimal:
		"""Calculate reconciliation variance"""
		self.variance_amount = abs(self.adjusted_book_balance - self.adjusted_bank_balance)
		return self.variance_amount
	
	def is_balanced(self, tolerance: Optional[Decimal] = None) -> bool:
		"""Check if reconciliation is balanced within tolerance"""
		if tolerance is None:
			tolerance = self.bank_account.reconciliation_tolerance
		return self.calculate_variance() <= tolerance
	
	def can_complete(self) -> bool:
		"""Check if reconciliation can be completed"""
		return (
			self.status in ['Draft', 'In Progress'] and
			self.is_balanced() and
			self.unmatched_bank_items == 0 and
			self.unmatched_book_items == 0
		)
	
	def complete_reconciliation(self, user_id: str):
		"""Complete the reconciliation"""
		if not self.can_complete():
			raise ValueError("Reconciliation cannot be completed")
		
		self.status = 'Completed'
		self.reconciled_by = user_id
		self.reconciled_date = datetime.utcnow()
		
		# Update bank account last reconciliation date
		self.bank_account.last_reconciliation_date = self.statement_date
		self.bank_account.statement_balance = self.statement_ending_balance
	
	def approve_reconciliation(self, user_id: str, notes: Optional[str] = None):
		"""Approve the reconciliation"""
		if self.status != 'Completed':
			raise ValueError("Only completed reconciliations can be approved")
		
		self.status = 'Approved'
		self.approved_by = user_id
		self.approved_date = datetime.utcnow()
		if notes:
			self.approval_notes = notes


class CFCMReconciliationItem(Model, AuditMixin, BaseMixin):
	"""
	Individual reconciliation line items.
	
	Tracks matching between bank transactions and book transactions,
	adjustments, and outstanding items.
	"""
	__tablename__ = 'cf_cm_reconciliation_item'
	
	# Identity
	item_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	reconciliation_id = Column(String(36), ForeignKey('cf_cm_reconciliation.reconciliation_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Item Information
	line_number = Column(Integer, nullable=False)
	item_type = Column(String(50), nullable=False)  # MATCHED, BANK_ONLY, BOOK_ONLY, ADJUSTMENT
	description = Column(Text, nullable=False)
	
	# Transaction References
	bank_transaction_id = Column(String(36), ForeignKey('cf_cm_bank_transaction.transaction_id'), nullable=True, index=True)
	gl_transaction_id = Column(String(36), nullable=True)  # GL journal entry reference
	
	# Amount Information
	amount = Column(DECIMAL(15, 2), nullable=False)
	variance_amount = Column(DECIMAL(15, 2), default=0.00)
	is_debit = Column(Boolean, nullable=False)
	
	# Matching Information
	is_matched = Column(Boolean, default=False)
	match_confidence = Column(DECIMAL(5, 2), default=0.00)  # Confidence percentage
	match_method = Column(String(50), nullable=True)  # EXACT, AMOUNT, DATE, MANUAL
	
	# Outstanding Item Information
	is_outstanding = Column(Boolean, default=False)
	outstanding_days = Column(Integer, default=0)
	expected_clear_date = Column(Date, nullable=True)
	
	# Adjustment Information
	is_adjustment = Column(Boolean, default=False)
	adjustment_reason = Column(String(200), nullable=True)
	adjustment_type = Column(String(50), nullable=True)  # ERROR, FEE, INTEREST, etc.
	requires_journal_entry = Column(Boolean, default=False)
	
	# Status
	status = Column(String(20), default='Active')  # Active, Resolved, Void
	resolution_date = Column(Date, nullable=True)
	
	# Notes
	notes = Column(Text, nullable=True)
	
	# Relationships
	reconciliation = relationship("CFCMReconciliation", back_populates="reconciliation_items")
	bank_transaction = relationship("CFCMBankTransaction", back_populates="reconciliation_items")
	
	def __repr__(self):
		return f"<CFCMReconciliationItem {self.item_type}: ${self.amount}>"
	
	def is_variance_significant(self, threshold: Decimal = Decimal('5.00')) -> bool:
		"""Check if variance is significant"""
		return abs(self.variance_amount) > threshold
	
	def can_auto_match(self, confidence_threshold: Decimal = Decimal('95.00')) -> bool:
		"""Check if item can be auto-matched"""
		return (
			not self.is_matched and
			self.match_confidence >= confidence_threshold and
			self.item_type in ['BANK_ONLY', 'BOOK_ONLY']
		)
	
	def match_transaction(self, method: str, confidence: Decimal = Decimal('100.00')):
		"""Mark item as matched"""
		self.is_matched = True
		self.match_method = method
		self.match_confidence = confidence
		self.status = 'Resolved'


class CFCMCashForecast(Model, AuditMixin, BaseMixin):
	"""
	Cash flow forecasting.
	
	Manages cash flow projections based on historical data, budgets,
	and scheduled transactions to optimize liquidity management.
	"""
	__tablename__ = 'cf_cm_cash_forecast'
	
	# Identity
	forecast_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Forecast Information
	forecast_name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	forecast_type = Column(String(50), default='Rolling')  # Rolling, Fixed, Budget
	
	# Period Information
	forecast_date = Column(Date, nullable=False, index=True)
	forecast_horizon = Column(Integer, default=90)  # Days
	period_start_date = Column(Date, nullable=False)
	period_end_date = Column(Date, nullable=False)
	
	# Bank Account (optional - can be consolidated)
	bank_account_id = Column(String(36), ForeignKey('cf_cm_bank_account.bank_account_id'), nullable=True, index=True)
	
	# Forecast Category
	category_code = Column(String(50), nullable=False, index=True)
	category_name = Column(String(200), nullable=False)
	category_type = Column(String(20), nullable=False)  # INFLOW, OUTFLOW
	
	# Amount Information
	forecast_amount = Column(DECIMAL(15, 2), nullable=False)
	actual_amount = Column(DECIMAL(15, 2), default=0.00)
	variance_amount = Column(DECIMAL(15, 2), default=0.00)
	variance_percentage = Column(DECIMAL(5, 2), default=0.00)
	
	# Confidence and Method
	confidence_level = Column(DECIMAL(5, 2), default=50.00)  # Confidence percentage
	forecast_method = Column(String(50), nullable=False)  # HISTORICAL, BUDGET, SCHEDULE, ML
	data_source = Column(String(50), nullable=True)  # Source system
	
	# Recurrence (for recurring items)
	is_recurring = Column(Boolean, default=False)
	recurrence_pattern = Column(String(50), nullable=True)  # DAILY, WEEKLY, MONTHLY, etc.
	recurrence_frequency = Column(Integer, default=1)
	
	# Status and Tracking
	status = Column(String(20), default='Active', index=True)  # Active, Realized, Cancelled
	is_committed = Column(Boolean, default=False)  # Firm commitment vs estimate
	
	# Review Information
	last_updated_date = Column(DateTime, default=datetime.utcnow)
	next_review_date = Column(Date, nullable=True)
	updated_by = Column(String(36), nullable=True)
	
	# Model Information (for ML-based forecasts)
	model_version = Column(String(20), nullable=True)
	model_accuracy = Column(DECIMAL(5, 2), nullable=True)
	training_period_start = Column(Date, nullable=True)
	training_period_end = Column(Date, nullable=True)
	
	# Notes
	notes = Column(Text, nullable=True)
	assumptions = Column(Text, nullable=True)
	
	# Constraints
	__table_args__ = (
		Index('idx_cash_forecast_date_category', 'forecast_date', 'category_code'),
		Index('idx_cash_forecast_account_date', 'bank_account_id', 'forecast_date'),
	)
	
	# Relationships
	bank_account = relationship("CFCMBankAccount")
	
	def __repr__(self):
		return f"<CFCMCashForecast {self.category_name} - {self.forecast_date}: ${self.forecast_amount}>"
	
	def calculate_variance(self) -> tuple[Decimal, Decimal]:
		"""Calculate variance amount and percentage"""
		self.variance_amount = self.actual_amount - self.forecast_amount
		
		if self.forecast_amount != 0:
			self.variance_percentage = (self.variance_amount / self.forecast_amount) * 100
		else:
			self.variance_percentage = Decimal('0.00')
		
		return self.variance_amount, self.variance_percentage
	
	def update_actual(self, actual_amount: Decimal):
		"""Update actual amount and calculate variance"""
		self.actual_amount = actual_amount
		self.calculate_variance()
		self.status = 'Realized'
	
	def get_accuracy_rating(self) -> str:
		"""Get forecast accuracy rating"""
		if abs(self.variance_percentage) <= 5:
			return 'Excellent'
		elif abs(self.variance_percentage) <= 15:
			return 'Good'
		elif abs(self.variance_percentage) <= 25:
			return 'Fair'
		else:
			return 'Poor'
	
	def is_overdue(self) -> bool:
		"""Check if forecast period has passed"""
		return self.forecast_date < date.today() and self.status == 'Active'


class CFCMCashPosition(Model, AuditMixin, BaseMixin):
	"""
	Daily cash positions.
	
	Tracks daily cash balances across all accounts for liquidity monitoring
	and cash position reporting.
	"""
	__tablename__ = 'cf_cm_cash_position'
	
	# Identity
	position_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Position Information
	position_date = Column(Date, nullable=False, index=True)
	bank_account_id = Column(String(36), ForeignKey('cf_cm_bank_account.bank_account_id'), nullable=False, index=True)
	
	# Balance Information
	opening_balance = Column(DECIMAL(15, 2), nullable=False)
	closing_balance = Column(DECIMAL(15, 2), nullable=False)
	average_balance = Column(DECIMAL(15, 2), nullable=False)
	minimum_balance = Column(DECIMAL(15, 2), nullable=False)
	maximum_balance = Column(DECIMAL(15, 2), nullable=False)
	
	# Activity Summary
	total_inflows = Column(DECIMAL(15, 2), default=0.00)
	total_outflows = Column(DECIMAL(15, 2), default=0.00)
	net_change = Column(DECIMAL(15, 2), default=0.00)
	transaction_count = Column(Integer, default=0)
	
	# Cash Flow Categories
	operating_inflows = Column(DECIMAL(15, 2), default=0.00)
	operating_outflows = Column(DECIMAL(15, 2), default=0.00)
	investing_inflows = Column(DECIMAL(15, 2), default=0.00)
	investing_outflows = Column(DECIMAL(15, 2), default=0.00)
	financing_inflows = Column(DECIMAL(15, 2), default=0.00)
	financing_outflows = Column(DECIMAL(15, 2), default=0.00)
	
	# Interest and Fees
	interest_earned = Column(DECIMAL(15, 2), default=0.00)
	fees_charged = Column(DECIMAL(15, 2), default=0.00)
	
	# Currency Information
	currency_code = Column(String(3), default='USD')
	exchange_rate = Column(DECIMAL(10, 6), default=1.000000)
	home_currency_balance = Column(DECIMAL(15, 2), nullable=True)
	
	# Status and Validation
	is_reconciled = Column(Boolean, default=False)
	has_variances = Column(Boolean, default=False)
	variance_count = Column(Integer, default=0)
	
	# Generation Information
	generated_date = Column(DateTime, default=datetime.utcnow)
	generated_by = Column(String(36), nullable=True)
	data_source = Column(String(50), default='System')  # System, Manual, Import
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'position_date', 'bank_account_id', name='uq_cash_position_date_account'),
		Index('idx_cash_position_date', 'position_date'),
	)
	
	# Relationships
	bank_account = relationship("CFCMBankAccount", back_populates="cash_positions")
	
	def __repr__(self):
		return f"<CFCMCashPosition {self.bank_account.account_name} - {self.position_date}: ${self.closing_balance}>"
	
	def calculate_net_change(self) -> Decimal:
		"""Calculate net change for the day"""
		self.net_change = self.total_inflows - self.total_outflows
		return self.net_change
	
	def calculate_average_balance(self) -> Decimal:
		"""Calculate average balance (simplified - uses opening and closing)"""
		self.average_balance = (self.opening_balance + self.closing_balance) / 2
		return self.average_balance
	
	def get_liquidity_ratio(self) -> Decimal:
		"""Get liquidity ratio (inflows / outflows)"""
		if self.total_outflows == 0:
			return Decimal('999.99')  # High liquidity
		return self.total_inflows / self.total_outflows
	
	def is_cash_positive(self) -> bool:
		"""Check if position is cash positive"""
		return self.closing_balance > 0
	
	def exceeds_minimum_balance(self) -> bool:
		"""Check if position exceeds minimum balance requirement"""
		return self.minimum_balance >= self.bank_account.minimum_balance


class CFCMInvestment(Model, AuditMixin, BaseMixin):
	"""
	Short-term investments and securities.
	
	Manages short-term investment positions for excess cash optimization
	including money market funds, CDs, and treasury securities.
	"""
	__tablename__ = 'cf_cm_investment'
	
	# Identity
	investment_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Investment Information
	investment_number = Column(String(50), nullable=False, index=True)
	investment_name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	
	# Investment Type and Classification
	investment_type = Column(String(50), nullable=False)  # CD, MONEY_MARKET, TREASURY, BOND
	investment_category = Column(String(50), default='SHORT_TERM')  # SHORT_TERM, LONG_TERM
	risk_rating = Column(String(20), default='LOW')  # LOW, MEDIUM, HIGH
	
	# Account Information
	bank_account_id = Column(String(36), ForeignKey('cf_cm_bank_account.bank_account_id'), nullable=False, index=True)
	custodian_name = Column(String(200), nullable=True)
	account_number = Column(String(50), nullable=True)
	
	# Investment Details
	purchase_date = Column(Date, nullable=False, index=True)
	maturity_date = Column(Date, nullable=True, index=True)
	purchase_amount = Column(DECIMAL(15, 2), nullable=False)
	current_value = Column(DECIMAL(15, 2), nullable=False)
	face_value = Column(DECIMAL(15, 2), nullable=True)  # For bonds/CDs
	
	# Interest and Yield
	interest_rate = Column(DECIMAL(8, 4), nullable=False)  # Annual rate
	yield_to_maturity = Column(DECIMAL(8, 4), nullable=True)
	accrued_interest = Column(DECIMAL(15, 2), default=0.00)
	interest_payment_frequency = Column(String(20), default='MONTHLY')  # DAILY, MONTHLY, QUARTERLY, etc.
	last_interest_date = Column(Date, nullable=True)
	next_interest_date = Column(Date, nullable=True)
	
	# Status and Liquidity
	status = Column(String(20), default='Active', index=True)  # Active, Matured, Sold, Closed
	is_liquid = Column(Boolean, default=True)  # Can be sold before maturity
	liquidation_penalty = Column(DECIMAL(8, 4), default=0.0000)  # Early withdrawal penalty
	
	# Performance Tracking
	unrealized_gain_loss = Column(DECIMAL(15, 2), default=0.00)
	realized_gain_loss = Column(DECIMAL(15, 2), default=0.00)
	total_interest_earned = Column(DECIMAL(15, 2), default=0.00)
	total_return = Column(DECIMAL(15, 2), default=0.00)
	
	# Disposal Information
	disposal_date = Column(Date, nullable=True)
	disposal_amount = Column(DECIMAL(15, 2), nullable=True)
	disposal_method = Column(String(50), nullable=True)  # MATURITY, SALE, REDEMPTION
	
	# GL Integration
	gl_asset_account_id = Column(String(36), nullable=True)
	gl_income_account_id = Column(String(36), nullable=True)
	gl_gain_loss_account_id = Column(String(36), nullable=True)
	
	# Currency Information
	currency_code = Column(String(3), default='USD')
	exchange_rate = Column(DECIMAL(10, 6), default=1.000000)
	
	# Auto-rollover (for CDs and similar)
	auto_rollover = Column(Boolean, default=False)
	rollover_instructions = Column(Text, nullable=True)
	
	# Notes
	notes = Column(Text, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'investment_number', name='uq_investment_number'),
		Index('idx_investment_maturity', 'maturity_date', 'status'),
	)
	
	# Relationships
	bank_account = relationship("CFCMBankAccount", back_populates="investments")
	
	def __repr__(self):
		return f"<CFCMInvestment {self.investment_number} - {self.investment_name}: ${self.current_value}>"
	
	def calculate_days_to_maturity(self) -> Optional[int]:
		"""Calculate days until maturity"""
		if not self.maturity_date or self.status != 'Active':
			return None
		
		delta = self.maturity_date - date.today()
		return max(0, delta.days)
	
	def calculate_accrued_interest(self, as_of_date: Optional[date] = None) -> Decimal:
		"""Calculate accrued interest"""
		if as_of_date is None:
			as_of_date = date.today()
		
		if not self.last_interest_date:
			interest_start = self.purchase_date
		else:
			interest_start = self.last_interest_date
		
		days_elapsed = (as_of_date - interest_start).days
		daily_rate = self.interest_rate / 365 / 100
		
		self.accrued_interest = self.current_value * daily_rate * days_elapsed
		return self.accrued_interest
	
	def calculate_current_yield(self) -> Decimal:
		"""Calculate current yield percentage"""
		if self.current_value == 0:
			return Decimal('0.00')
		
		annual_interest = self.current_value * (self.interest_rate / 100)
		return (annual_interest / self.current_value) * 100
	
	def is_maturing_soon(self, days: int = 30) -> bool:
		"""Check if investment is maturing within specified days"""
		days_to_maturity = self.calculate_days_to_maturity()
		return days_to_maturity is not None and days_to_maturity <= days
	
	def can_liquidate(self) -> bool:
		"""Check if investment can be liquidated"""
		return self.status == 'Active' and self.is_liquid
	
	def calculate_liquidation_value(self) -> Decimal:
		"""Calculate net value after liquidation penalties"""
		penalty_amount = self.current_value * (self.liquidation_penalty / 100)
		return self.current_value - penalty_amount


class CFCMCurrencyRate(Model, AuditMixin, BaseMixin):
	"""
	Foreign exchange rates.
	
	Manages FX rates for multi-currency cash management including
	rate history, rate sources, and automatic rate updates.
	"""
	__tablename__ = 'cf_cm_currency_rate'
	
	# Identity
	rate_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Currency Information
	from_currency = Column(String(3), nullable=False, index=True)
	to_currency = Column(String(3), nullable=False, index=True)
	rate_date = Column(Date, nullable=False, index=True)
	
	# Rate Information
	exchange_rate = Column(DECIMAL(12, 8), nullable=False)  # High precision for FX
	inverse_rate = Column(DECIMAL(12, 8), nullable=False)  # Calculated inverse
	
	# Rate Type and Source
	rate_type = Column(String(50), default='SPOT')  # SPOT, FORWARD, AVERAGE
	rate_source = Column(String(100), nullable=False)  # Central Bank, Reuters, Bloomberg, etc.
	rate_provider = Column(String(100), nullable=True)  # Specific provider name
	
	# Validity and Status
	effective_time = Column(DateTime, nullable=True)  # When rate becomes effective
	expiry_time = Column(DateTime, nullable=True)  # When rate expires
	is_active = Column(Boolean, default=True)
	status = Column(String(20), default='Active')  # Active, Historical, Inactive
	
	# Rate Metadata
	bid_rate = Column(DECIMAL(12, 8), nullable=True)  # Bank buying rate
	ask_rate = Column(DECIMAL(12, 8), nullable=True)  # Bank selling rate
	mid_rate = Column(DECIMAL(12, 8), nullable=True)  # Mid-market rate
	spread = Column(DECIMAL(6, 4), nullable=True)  # Bid-ask spread
	
	# Update Information
	last_updated = Column(DateTime, default=datetime.utcnow)
	update_frequency = Column(String(20), default='DAILY')  # REAL_TIME, HOURLY, DAILY
	auto_update = Column(Boolean, default=True)
	
	# Variance Tracking
	previous_rate = Column(DECIMAL(12, 8), nullable=True)
	rate_change = Column(DECIMAL(12, 8), default=0.00000000)
	rate_change_percent = Column(DECIMAL(6, 4), default=0.0000)
	
	# Usage Tracking
	usage_count = Column(Integer, default=0)  # How many times used
	last_used_date = Column(DateTime, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'from_currency', 'to_currency', 'rate_date', 'rate_type', name='uq_currency_rate'),
		Index('idx_currency_rate_date', 'rate_date', 'is_active'),
		Index('idx_currency_rate_pair', 'from_currency', 'to_currency'),
	)
	
	def __repr__(self):
		return f"<CFCMCurrencyRate {self.from_currency}/{self.to_currency} - {self.rate_date}: {self.exchange_rate}>"
	
	def calculate_inverse_rate(self) -> Decimal:
		"""Calculate and store inverse rate"""
		if self.exchange_rate != 0:
			self.inverse_rate = Decimal('1.00000000') / self.exchange_rate
		else:
			self.inverse_rate = Decimal('0.00000000')
		return self.inverse_rate
	
	def calculate_rate_change(self) -> tuple[Decimal, Decimal]:
		"""Calculate rate change from previous rate"""
		if self.previous_rate and self.previous_rate != 0:
			self.rate_change = self.exchange_rate - self.previous_rate
			self.rate_change_percent = (self.rate_change / self.previous_rate) * 100
		else:
			self.rate_change = Decimal('0.00000000')
			self.rate_change_percent = Decimal('0.0000')
		
		return self.rate_change, self.rate_change_percent
	
	def convert_amount(self, amount: Decimal, reverse: bool = False) -> Decimal:
		"""Convert amount using this exchange rate"""
		self.usage_count += 1
		self.last_used_date = datetime.utcnow()
		
		if reverse:
			return amount * self.inverse_rate
		else:
			return amount * self.exchange_rate
	
	def is_current(self, tolerance_hours: int = 24) -> bool:
		"""Check if rate is current within tolerance"""
		if not self.last_updated:
			return False
		
		hours_old = (datetime.utcnow() - self.last_updated).total_seconds() / 3600
		return hours_old <= tolerance_hours
	
	def get_age_in_hours(self) -> float:
		"""Get age of rate in hours"""
		if not self.last_updated:
			return float('inf')
		
		return (datetime.utcnow() - self.last_updated).total_seconds() / 3600


class CFCMCashTransfer(Model, AuditMixin, BaseMixin):
	"""
	Inter-bank cash transfers.
	
	Manages transfers between bank accounts including approval workflow,
	transfer tracking, and automated GL posting.
	"""
	__tablename__ = 'cf_cm_cash_transfer'
	
	# Identity
	transfer_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Transfer Information
	transfer_number = Column(String(50), nullable=False, index=True)
	description = Column(Text, nullable=False)
	reference = Column(String(100), nullable=True)
	
	# Account Information
	from_account_id = Column(String(36), ForeignKey('cf_cm_bank_account.bank_account_id'), nullable=False, index=True)
	to_account_id = Column(String(36), ForeignKey('cf_cm_bank_account.bank_account_id'), nullable=False, index=True)
	
	# Transfer Details
	transfer_date = Column(Date, nullable=False, index=True)
	value_date = Column(Date, nullable=True)  # When funds are available
	requested_date = Column(Date, default=date.today)
	
	# Amount Information
	transfer_amount = Column(DECIMAL(15, 2), nullable=False)
	transfer_fee = Column(DECIMAL(15, 2), default=0.00)
	total_amount = Column(DECIMAL(15, 2), nullable=False)  # Amount + Fee
	
	# Transfer Method
	transfer_method = Column(String(50), nullable=False)  # WIRE, ACH, INTERNAL, CHECK
	transfer_type = Column(String(50), default='STANDARD')  # STANDARD, URGENT, SAME_DAY
	
	# Status and Workflow
	status = Column(String(20), default='Draft', index=True)  # Draft, Pending, Approved, Submitted, Completed, Failed, Cancelled
	
	# Approval Workflow
	requires_approval = Column(Boolean, default=True)
	approval_level = Column(Integer, default=1)
	approved = Column(Boolean, default=False)
	approved_by = Column(String(36), nullable=True)
	approved_date = Column(DateTime, nullable=True)
	
	# Execution Information
	submitted = Column(Boolean, default=False)
	submitted_by = Column(String(36), nullable=True)
	submitted_date = Column(DateTime, nullable=True)
	
	completed = Column(Boolean, default=False)
	completed_date = Column(DateTime, nullable=True)
	completion_reference = Column(String(100), nullable=True)  # Bank confirmation number
	
	# Currency Information
	currency_code = Column(String(3), default='USD')
	exchange_rate = Column(DECIMAL(10, 6), default=1.000000)
	fx_rate_id = Column(String(36), nullable=True)  # Reference to currency rate used
	
	# GL Integration
	gl_posted = Column(Boolean, default=False)
	gl_journal_id = Column(String(36), nullable=True)
	clearing_account_id = Column(String(36), nullable=True)  # For in-transit funds
	
	# Tracking Information
	tracking_number = Column(String(100), nullable=True)  # Bank tracking number
	confirmation_number = Column(String(100), nullable=True)
	
	# Error Handling
	failed = Column(Boolean, default=False)
	failure_reason = Column(Text, nullable=True)
	failure_date = Column(DateTime, nullable=True)
	retry_count = Column(Integer, default=0)
	
	# Cancellation
	cancelled = Column(Boolean, default=False)
	cancelled_by = Column(String(36), nullable=True)
	cancelled_date = Column(DateTime, nullable=True)
	cancellation_reason = Column(Text, nullable=True)
	
	# Notes
	notes = Column(Text, nullable=True)
	internal_notes = Column(Text, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'transfer_number', name='uq_transfer_number'),
		Index('idx_transfer_date_status', 'transfer_date', 'status'),
		Index('idx_transfer_accounts', 'from_account_id', 'to_account_id'),
	)
	
	# Relationships
	from_account = relationship("CFCMBankAccount", foreign_keys=[from_account_id], back_populates="transfers_from")
	to_account = relationship("CFCMBankAccount", foreign_keys=[to_account_id], back_populates="transfers_to")
	
	def __repr__(self):
		return f"<CFCMCashTransfer {self.transfer_number}: ${self.transfer_amount}>"
	
	def calculate_total_amount(self) -> Decimal:
		"""Calculate total amount including fees"""
		self.total_amount = self.transfer_amount + self.transfer_fee
		return self.total_amount
	
	def can_approve(self) -> bool:
		"""Check if transfer can be approved"""
		return (
			self.status in ['Draft', 'Pending'] and
			not self.approved and
			self.requires_approval and
			self.transfer_amount > 0 and
			self.from_account_id != self.to_account_id
		)
	
	def can_submit(self) -> bool:
		"""Check if transfer can be submitted"""
		return (
			self.status == 'Approved' and
			not self.submitted and
			(not self.requires_approval or self.approved) and
			self.from_account.can_withdraw(self.total_amount)
		)
	
	def can_cancel(self) -> bool:
		"""Check if transfer can be cancelled"""
		return (
			self.status in ['Draft', 'Pending', 'Approved'] and
			not self.submitted and
			not self.completed and
			not self.cancelled
		)
	
	def approve_transfer(self, user_id: str):
		"""Approve the transfer"""
		if not self.can_approve():
			raise ValueError("Transfer cannot be approved")
		
		self.approved = True
		self.approved_by = user_id
		self.approved_date = datetime.utcnow()
		self.status = 'Approved'
	
	def submit_transfer(self, user_id: str):
		"""Submit transfer for processing"""
		if not self.can_submit():
			raise ValueError("Transfer cannot be submitted")
		
		self.submitted = True
		self.submitted_by = user_id
		self.submitted_date = datetime.utcnow()
		self.status = 'Submitted'
		
		# Update account balances immediately for internal transfers
		if self.transfer_method == 'INTERNAL':
			self.from_account.update_balance(self.total_amount, 'TRANSFER_OUT')
			self.to_account.update_balance(self.transfer_amount, 'TRANSFER_IN')
			self.complete_transfer()
	
	def complete_transfer(self, confirmation_number: Optional[str] = None):
		"""Mark transfer as completed"""
		if self.status != 'Submitted':
			raise ValueError("Only submitted transfers can be completed")
		
		self.completed = True
		self.completed_date = datetime.utcnow()
		self.status = 'Completed'
		
		if confirmation_number:
			self.confirmation_number = confirmation_number
	
	def fail_transfer(self, reason: str):
		"""Mark transfer as failed"""
		self.failed = True
		self.failure_reason = reason
		self.failure_date = datetime.utcnow()
		self.status = 'Failed'
		self.retry_count += 1
	
	def cancel_transfer(self, user_id: str, reason: str):
		"""Cancel the transfer"""
		if not self.can_cancel():
			raise ValueError("Transfer cannot be cancelled")
		
		self.cancelled = True
		self.cancelled_by = user_id
		self.cancelled_date = datetime.utcnow()
		self.cancellation_reason = reason
		self.status = 'Cancelled'


class CFCMDeposit(Model, AuditMixin, BaseMixin):
	"""
	Deposit processing and lockbox management.
	
	Manages deposit slips, lockbox processing, and automated deposit
	allocation for efficient receivables processing.
	"""
	__tablename__ = 'cf_cm_deposit'
	
	# Identity
	deposit_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Deposit Information
	deposit_number = Column(String(50), nullable=False, index=True)
	deposit_slip_number = Column(String(50), nullable=True)
	description = Column(Text, nullable=True)
	
	# Account Information
	bank_account_id = Column(String(36), ForeignKey('cf_cm_bank_account.bank_account_id'), nullable=False, index=True)
	
	# Deposit Details
	deposit_date = Column(Date, nullable=False, index=True)
	deposit_time = Column(DateTime, nullable=True)
	value_date = Column(Date, nullable=True)  # When funds are available
	
	# Deposit Type and Method
	deposit_type = Column(String(50), nullable=False)  # CASH, CHECK, LOCKBOX, ELECTRONIC, WIRE
	deposit_method = Column(String(50), default='BRANCH')  # BRANCH, ATM, LOCKBOX, REMOTE, MOBILE
	
	# Amount Information
	cash_amount = Column(DECIMAL(15, 2), default=0.00)
	check_amount = Column(DECIMAL(15, 2), default=0.00)
	total_amount = Column(DECIMAL(15, 2), nullable=False)
	deposit_fee = Column(DECIMAL(15, 2), default=0.00)
	net_amount = Column(DECIMAL(15, 2), nullable=False)
	
	# Check Information
	check_count = Column(Integer, default=0)
	total_check_amount = Column(DECIMAL(15, 2), default=0.00)
	returned_check_count = Column(Integer, default=0)
	returned_check_amount = Column(DECIMAL(15, 2), default=0.00)
	
	# Status and Processing
	status = Column(String(20), default='Pending', index=True)  # Pending, Processed, Available, Held, Returned
	processing_status = Column(String(50), nullable=True)  # Detailed processing status
	
	# Hold Information
	hold_amount = Column(DECIMAL(15, 2), default=0.00)
	hold_reason = Column(String(200), nullable=True)
	hold_release_date = Column(Date, nullable=True)
	
	# Lockbox Information (if applicable)
	lockbox_number = Column(String(20), nullable=True)
	lockbox_batch = Column(String(50), nullable=True)
	remittance_count = Column(Integer, default=0)
	
	# Source Information
	depositor_name = Column(String(200), nullable=True)
	deposit_location = Column(String(100), nullable=True)
	branch_code = Column(String(20), nullable=True)
	
	# GL Integration
	gl_posted = Column(Boolean, default=False)
	gl_journal_id = Column(String(36), nullable=True)
	
	# Processing Information
	processed = Column(Boolean, default=False)
	processed_by = Column(String(36), nullable=True)
	processed_date = Column(DateTime, nullable=True)
	
	# Bank Confirmation
	bank_confirmed = Column(Boolean, default=False)
	bank_confirmation_number = Column(String(100), nullable=True)
	bank_confirmation_date = Column(DateTime, nullable=True)
	
	# Currency Information
	currency_code = Column(String(3), default='USD')
	exchange_rate = Column(DECIMAL(10, 6), default=1.000000)
	
	# Document Management
	document_count = Column(Integer, default=0)
	document_path = Column(String(500), nullable=True)
	
	# Notes
	notes = Column(Text, nullable=True)
	bank_notes = Column(Text, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'deposit_number', name='uq_deposit_number'),
		Index('idx_deposit_date_account', 'bank_account_id', 'deposit_date'),
		Index('idx_deposit_status', 'status', 'deposit_date'),
	)
	
	# Relationships
	bank_account = relationship("CFCMBankAccount", back_populates="deposits")
	
	def __repr__(self):
		return f"<CFCMDeposit {self.deposit_number}: ${self.total_amount}>"
	
	def calculate_net_amount(self) -> Decimal:
		"""Calculate net deposit amount after fees"""
		self.net_amount = self.total_amount - self.deposit_fee
		return self.net_amount
	
	def calculate_total_amount(self) -> Decimal:
		"""Calculate total deposit amount"""
		self.total_amount = self.cash_amount + self.check_amount
		return self.total_amount
	
	def can_process(self) -> bool:
		"""Check if deposit can be processed"""
		return (
			self.status == 'Pending' and
			not self.processed and
			self.total_amount > 0
		)
	
	def can_release_hold(self) -> bool:
		"""Check if deposit hold can be released"""
		return (
			self.status == 'Held' and
			self.hold_amount > 0 and
			(not self.hold_release_date or self.hold_release_date <= date.today())
		)
	
	def process_deposit(self, user_id: str):
		"""Process the deposit"""
		if not self.can_process():
			raise ValueError("Deposit cannot be processed")
		
		self.processed = True
		self.processed_by = user_id
		self.processed_date = datetime.utcnow()
		self.status = 'Processed'
		
		# Update bank account balance
		self.bank_account.update_balance(self.net_amount, 'DEPOSIT')
	
	def place_hold(self, hold_amount: Decimal, reason: str, release_date: Optional[date] = None):
		"""Place hold on deposit"""
		if hold_amount > self.total_amount:
			raise ValueError("Hold amount cannot exceed deposit amount")
		
		self.hold_amount = hold_amount
		self.hold_reason = reason
		self.hold_release_date = release_date
		self.status = 'Held'
	
	def release_hold(self):
		"""Release deposit hold"""
		if not self.can_release_hold():
			raise ValueError("Hold cannot be released")
		
		self.hold_amount = Decimal('0.00')
		self.hold_reason = None
		self.hold_release_date = None
		self.status = 'Available'
	
	def return_deposit(self, reason: str):
		"""Return deposit (for NSF checks, etc.)"""
		self.status = 'Returned'
		self.notes = f"Returned: {reason}"
		
		# Reverse bank account balance
		self.bank_account.update_balance(self.net_amount, 'WITHDRAWAL')


class CFCMCheckRegister(Model, AuditMixin, BaseMixin):
	"""
	Check register and void management.
	
	Tracks issued checks, void management, and stop payment orders
	for comprehensive check control and reconciliation.
	"""
	__tablename__ = 'cf_cm_check_register'
	
	# Identity
	check_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Check Information
	check_number = Column(String(20), nullable=False, index=True)
	bank_account_id = Column(String(36), ForeignKey('cf_cm_bank_account.bank_account_id'), nullable=False, index=True)
	
	# Check Details
	check_date = Column(Date, nullable=False, index=True)
	issue_date = Column(Date, default=date.today)
	payee_name = Column(String(200), nullable=False)
	check_amount = Column(DECIMAL(15, 2), nullable=False)
	
	# Check Status
	status = Column(String(20), default='Issued', index=True)  # Issued, Outstanding, Cleared, Voided, Stopped
	cleared_date = Column(Date, nullable=True)
	void_date = Column(Date, nullable=True)
	
	# Reference Information
	payment_id = Column(String(36), nullable=True)  # Link to AP payment
	invoice_number = Column(String(50), nullable=True)
	description = Column(Text, nullable=True)
	memo = Column(String(100), nullable=True)  # Memo line on check
	
	# Clearing Information
	is_cleared = Column(Boolean, default=False, index=True)
	cleared_amount = Column(DECIMAL(15, 2), nullable=True)  # Actual cleared amount
	bank_clearing_date = Column(Date, nullable=True)
	
	# Void Information
	is_voided = Column(Boolean, default=False, index=True)
	voided_by = Column(String(36), nullable=True)
	void_reason = Column(String(200), nullable=True)
	replacement_check_id = Column(String(36), nullable=True)  # If reissued
	
	# Stop Payment Information
	stop_payment = Column(Boolean, default=False)
	stop_payment_date = Column(Date, nullable=True)
	stop_payment_reason = Column(String(200), nullable=True)
	stop_payment_fee = Column(DECIMAL(8, 2), default=0.00)
	stop_payment_confirmed = Column(Boolean, default=False)
	
	# Outstanding Tracking
	days_outstanding = Column(Integer, default=0)
	is_stale_dated = Column(Boolean, default=False)  # Older than 6 months typically
	stale_date_threshold = Column(Date, nullable=True)
	
	# Reconciliation
	is_reconciled = Column(Boolean, default=False)
	reconciliation_id = Column(String(36), nullable=True)
	reconciled_date = Column(Date, nullable=True)
	
	# GL Integration
	gl_posted = Column(Boolean, default=False)
	gl_journal_id = Column(String(36), nullable=True)
	
	# Reissue Information
	is_reissue = Column(Boolean, default=False)
	original_check_id = Column(String(36), nullable=True)
	reissue_reason = Column(String(200), nullable=True)
	
	# Security Features
	security_features = Column(String(500), nullable=True)  # JSON of security features
	micr_line = Column(String(100), nullable=True)  # MICR encoding
	
	# Currency Information
	currency_code = Column(String(3), default='USD')
	
	# Notes
	notes = Column(Text, nullable=True)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'bank_account_id', 'check_number', name='uq_check_number_account'),
		Index('idx_check_status_date', 'status', 'check_date'),
		Index('idx_check_outstanding', 'is_cleared', 'days_outstanding'),
	)
	
	# Relationships
	bank_account = relationship("CFCMBankAccount", back_populates="checks")
	
	def __repr__(self):
		return f"<CFCMCheckRegister #{self.check_number}: {self.payee_name} - ${self.check_amount}>"
	
	def calculate_days_outstanding(self) -> int:
		"""Calculate days outstanding"""
		if self.is_cleared or self.is_voided:
			self.days_outstanding = 0
		else:
			self.days_outstanding = (date.today() - self.check_date).days
		
		return self.days_outstanding
	
	def is_stale_dated_check(self, stale_days: int = 180) -> bool:
		"""Check if check is stale dated"""
		self.calculate_days_outstanding()
		self.is_stale_dated = self.days_outstanding > stale_days and not self.is_cleared
		
		if self.is_stale_dated:
			self.stale_date_threshold = self.check_date + timedelta(days=stale_days)
		
		return self.is_stale_dated
	
	def can_void(self) -> bool:
		"""Check if check can be voided"""
		return (
			not self.is_voided and
			not self.is_cleared and
			not self.stop_payment and
			self.status in ['Issued', 'Outstanding']
		)
	
	def can_stop_payment(self) -> bool:
		"""Check if stop payment can be placed"""
		return (
			not self.is_cleared and
			not self.is_voided and
			not self.stop_payment and
			self.status in ['Issued', 'Outstanding']
		)
	
	def void_check(self, user_id: str, reason: str):
		"""Void the check"""
		if not self.can_void():
			raise ValueError("Check cannot be voided")
		
		self.is_voided = True
		self.voided_by = user_id
		self.void_date = date.today()
		self.void_reason = reason
		self.status = 'Voided'
		
		# Reverse bank account balance
		self.bank_account.update_balance(self.check_amount, 'DEPOSIT')  # Add back
	
	def place_stop_payment(self, reason: str, fee: Decimal = Decimal('0.00')):
		"""Place stop payment on check"""
		if not self.can_stop_payment():
			raise ValueError("Stop payment cannot be placed")
		
		self.stop_payment = True
		self.stop_payment_date = date.today()
		self.stop_payment_reason = reason
		self.stop_payment_fee = fee
		self.status = 'Stopped'
		
		# Charge stop payment fee if applicable
		if fee > 0:
			self.bank_account.update_balance(fee, 'FEE')
	
	def clear_check(self, cleared_amount: Optional[Decimal] = None, cleared_date: Optional[date] = None):
		"""Mark check as cleared"""
		if self.is_cleared:
			return
		
		self.is_cleared = True
		self.cleared_date = cleared_date or date.today()
		self.bank_clearing_date = self.cleared_date
		self.cleared_amount = cleared_amount or self.check_amount
		self.status = 'Cleared'
		
		# Handle amount variance
		if self.cleared_amount != self.check_amount:
			variance = self.check_amount - self.cleared_amount
			# This would typically create a variance transaction
	
	def reissue_check(self, new_check_number: str, reason: str) -> 'CFCMCheckRegister':
		"""Reissue check with new check number"""
		if not self.can_void():
			raise ValueError("Original check must be voidable to reissue")
		
		# Void original check
		self.void_check(self.created_by, f"Reissued as #{new_check_number}")
		
		# Create new check record
		new_check = CFCMCheckRegister(
			tenant_id=self.tenant_id,
			check_number=new_check_number,
			bank_account_id=self.bank_account_id,
			check_date=date.today(),
			payee_name=self.payee_name,
			check_amount=self.check_amount,
			payment_id=self.payment_id,
			invoice_number=self.invoice_number,
			description=self.description,
			memo=self.memo,
			is_reissue=True,
			original_check_id=self.check_id,
			reissue_reason=reason
		)
		
		# Update replacement reference
		self.replacement_check_id = new_check.check_id
		
		return new_check
	
	def get_age_category(self) -> str:
		"""Get check age category"""
		days = self.calculate_days_outstanding()
		
		if days <= 30:
			return 'Current'
		elif days <= 60:
			return '31-60 Days'
		elif days <= 90:
			return '61-90 Days'
		elif days <= 180:
			return '91-180 Days'
		else:
			return 'Over 180 Days'