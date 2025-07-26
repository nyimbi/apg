"""
General Ledger Models

Database models for the General Ledger sub-capability including chart of accounts,
journal entries, postings, and trial balance functionality.
"""

from datetime import datetime, date
from typing import Dict, List, Any, Optional
from decimal import Decimal
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, Date, DECIMAL, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str
import json

from ...auth_rbac.models import BaseMixin, AuditMixin, Model


class CFGLAccountType(Model, AuditMixin, BaseMixin):
	"""
	Chart of accounts type classification.
	
	Defines the major account types: Asset, Liability, Equity, Revenue, Expense
	"""
	__tablename__ = 'cf_gl_account_type'
	
	# Identity
	type_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Type Information
	type_code = Column(String(10), nullable=False, index=True)  # A, L, E, R, X
	type_name = Column(String(100), nullable=False)  # Asset, Liability, Equity, Revenue, Expense
	description = Column(Text, nullable=True)
	
	# Behavioral Properties
	normal_balance = Column(String(10), nullable=False)  # Debit or Credit
	is_balance_sheet = Column(Boolean, default=True)  # True for B/S, False for P&L
	sort_order = Column(Integer, default=0)
	
	# Relationships
	accounts = relationship("CFGLAccount", back_populates="account_type")
	
	def __repr__(self):
		return f"<CFGLAccountType {self.type_name}>"


class CFGLAccount(Model, AuditMixin, BaseMixin):
	"""
	Chart of Accounts - individual GL accounts.
	
	Hierarchical structure supporting parent-child relationships
	for account organization and reporting.
	"""
	__tablename__ = 'cf_gl_account'
	
	# Identity
	account_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Account Information
	account_code = Column(String(20), nullable=False, index=True)
	account_name = Column(String(200), nullable=False, index=True)
	description = Column(Text, nullable=True)
	
	# Classification
	account_type_id = Column(String(36), ForeignKey('cf_gl_account_type.type_id'), nullable=False, index=True)
	parent_account_id = Column(String(36), ForeignKey('cf_gl_account.account_id'), nullable=True, index=True)
	
	# Properties
	is_active = Column(Boolean, default=True)
	is_header = Column(Boolean, default=False)  # Header accounts for grouping
	is_system = Column(Boolean, default=False)  # System-managed accounts
	allow_posting = Column(Boolean, default=True)  # Can post transactions
	
	# Balance Information
	current_balance = Column(DECIMAL(15, 2), default=0.00)
	ytd_balance = Column(DECIMAL(15, 2), default=0.00)  # Year-to-date
	opening_balance = Column(DECIMAL(15, 2), default=0.00)
	
	# Configuration
	currency_code = Column(String(3), default='USD')
	tax_code = Column(String(20), nullable=True)
	cost_center_required = Column(Boolean, default=False)
	department_required = Column(Boolean, default=False)
	
	# Hierarchy
	level = Column(Integer, default=0)  # Hierarchy level (0 = root)
	path = Column(String(500), nullable=True)  # Full path for efficient queries
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'account_code', name='uq_account_code_tenant'),
	)
	
	# Relationships
	account_type = relationship("CFGLAccountType", back_populates="accounts")
	parent_account = relationship("CFGLAccount", remote_side=[account_id])
	child_accounts = relationship("CFGLAccount")
	journal_lines = relationship("CFGLJournalLine", back_populates="account")
	postings = relationship("CFGLPosting", back_populates="account")
	
	def __repr__(self):
		return f"<CFGLAccount {self.account_code} - {self.account_name}>"
	
	def get_full_path(self) -> str:
		"""Get full account path"""
		if self.parent_account:
			return f"{self.parent_account.get_full_path()} > {self.account_name}"
		return self.account_name
	
	def calculate_balance(self, as_of_date: Optional[date] = None) -> Decimal:
		"""Calculate account balance as of a specific date"""
		if as_of_date is None:
			return self.current_balance
		
		# Sum postings up to the date
		total_debits = sum(p.debit_amount for p in self.postings 
						  if p.posting_date <= as_of_date and p.is_posted)
		total_credits = sum(p.credit_amount for p in self.postings 
						   if p.posting_date <= as_of_date and p.is_posted)
		
		# Apply normal balance logic
		if self.account_type.normal_balance == 'Debit':
			return self.opening_balance + total_debits - total_credits
		else:
			return self.opening_balance + total_credits - total_debits
	
	def is_debit_balance(self) -> bool:
		"""Check if account has normal debit balance"""
		return self.account_type.normal_balance == 'Debit'


class CFGLPeriod(Model, AuditMixin, BaseMixin):
	"""
	Accounting periods for financial reporting and posting control.
	
	Manages fiscal year and period structure, posting status,
	and period-end closing procedures.
	"""
	__tablename__ = 'cf_gl_period'
	
	# Identity
	period_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Period Information
	fiscal_year = Column(Integer, nullable=False, index=True)
	period_number = Column(Integer, nullable=False, index=True)  # 1-12 for months
	period_name = Column(String(50), nullable=False)  # "January 2024"
	
	# Date Range
	start_date = Column(Date, nullable=False, index=True)
	end_date = Column(Date, nullable=False, index=True)
	
	# Status
	status = Column(String(20), default='Open', index=True)  # Open, Closed, Locked
	is_adjustment_period = Column(Boolean, default=False)  # 13th period for adjustments
	
	# Closing Information
	closed_date = Column(DateTime, nullable=True)
	closed_by = Column(String(36), nullable=True)
	closing_entries_count = Column(Integer, default=0)
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'fiscal_year', 'period_number', name='uq_period_year_tenant'),
	)
	
	# Relationships
	journal_entries = relationship("CFGLJournalEntry", back_populates="period")
	postings = relationship("CFGLPosting", back_populates="period")
	
	def __repr__(self):
		return f"<CFGLPeriod {self.period_name} ({self.status})>"
	
	def can_post(self) -> bool:
		"""Check if period allows posting"""
		return self.status == 'Open'
	
	def close_period(self, user_id: str):
		"""Close the accounting period"""
		if self.status == 'Open':
			self.status = 'Closed'
			self.closed_date = datetime.utcnow()
			self.closed_by = user_id


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