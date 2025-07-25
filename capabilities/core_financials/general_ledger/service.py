"""
General Ledger Service

Business logic for General Ledger operations including chart of accounts management,
journal entry processing, posting, and financial reporting.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, date
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func

from .models import (
	CFGLAccount, CFGLAccountType, CFGLPeriod, CFGLJournalEntry, 
	CFGLJournalLine, CFGLPosting, CFGLTrialBalance
)
from ...auth_rbac.models import db


class GeneralLedgerService:
	"""Service class for General Ledger operations"""
	
	def __init__(self, tenant_id: str):
		self.tenant_id = tenant_id
	
	# Chart of Accounts Management
	
	def create_account(self, account_data: Dict[str, Any]) -> CFGLAccount:
		"""Create a new GL account"""
		account = CFGLAccount(
			tenant_id=self.tenant_id,
			account_code=account_data['account_code'],
			account_name=account_data['account_name'],
			description=account_data.get('description'),
			account_type_id=account_data['account_type_id'],
			parent_account_id=account_data.get('parent_account_id'),
			is_active=account_data.get('is_active', True),
			is_header=account_data.get('is_header', False),
			allow_posting=account_data.get('allow_posting', True),
			currency_code=account_data.get('currency_code', 'USD'),
			tax_code=account_data.get('tax_code'),
			cost_center_required=account_data.get('cost_center_required', False),
			department_required=account_data.get('department_required', False)
		)
		
		# Calculate hierarchy level and path
		if account.parent_account_id:
			parent = self.get_account(account.parent_account_id)
			account.level = parent.level + 1
			account.path = f"{parent.path}/{account.account_code}" if parent.path else account.account_code
		else:
			account.level = 0
			account.path = account.account_code
		
		db.session.add(account)
		db.session.commit()
		return account
	
	def get_account(self, account_id: str) -> Optional[CFGLAccount]:
		"""Get account by ID"""
		return CFGLAccount.query.filter_by(
			tenant_id=self.tenant_id,
			account_id=account_id
		).first()
	
	def get_account_by_code(self, account_code: str) -> Optional[CFGLAccount]:
		"""Get account by code"""
		return CFGLAccount.query.filter_by(
			tenant_id=self.tenant_id,
			account_code=account_code
		).first()
	
	def get_chart_of_accounts(self, include_inactive: bool = False) -> List[CFGLAccount]:
		"""Get complete chart of accounts"""
		query = CFGLAccount.query.filter_by(tenant_id=self.tenant_id)
		
		if not include_inactive:
			query = query.filter_by(is_active=True)
		
		return query.order_by(CFGLAccount.account_code).all()
	
	def get_account_hierarchy(self, parent_id: Optional[str] = None) -> List[CFGLAccount]:
		"""Get accounts in hierarchical structure"""
		if parent_id:
			return CFGLAccount.query.filter_by(
				tenant_id=self.tenant_id,
				parent_account_id=parent_id,
				is_active=True
			).order_by(CFGLAccount.account_code).all()
		else:
			return CFGLAccount.query.filter_by(
				tenant_id=self.tenant_id,
				parent_account_id=None,
				is_active=True
			).order_by(CFGLAccount.account_code).all()
	
	# Period Management
	
	def create_period(self, period_data: Dict[str, Any]) -> CFGLPeriod:
		"""Create accounting period"""
		period = CFGLPeriod(
			tenant_id=self.tenant_id,
			fiscal_year=period_data['fiscal_year'],
			period_number=period_data['period_number'],
			period_name=period_data['period_name'],
			start_date=period_data['start_date'],
			end_date=period_data['end_date'],
			is_adjustment_period=period_data.get('is_adjustment_period', False)
		)
		
		db.session.add(period)
		db.session.commit()
		return period
	
	def get_current_period(self, as_of_date: Optional[date] = None) -> Optional[CFGLPeriod]:
		"""Get current accounting period"""
		if as_of_date is None:
			as_of_date = date.today()
		
		return CFGLPeriod.query.filter(
			CFGLPeriod.tenant_id == self.tenant_id,
			CFGLPeriod.start_date <= as_of_date,
			CFGLPeriod.end_date >= as_of_date
		).first()
	
	def get_open_periods(self) -> List[CFGLPeriod]:
		"""Get all open periods"""
		return CFGLPeriod.query.filter_by(
			tenant_id=self.tenant_id,
			status='Open'
		).order_by(CFGLPeriod.start_date).all()
	
	# Journal Entry Management
	
	def create_journal_entry(self, entry_data: Dict[str, Any]) -> CFGLJournalEntry:
		"""Create journal entry with lines"""
		# Get period
		period = self.get_current_period(entry_data['posting_date'])
		if not period or not period.can_post():
			raise ValueError("Cannot post to this period")
		
		# Create journal entry
		journal = CFGLJournalEntry(
			tenant_id=self.tenant_id,
			journal_number=entry_data.get('journal_number') or self._generate_journal_number(),
			description=entry_data['description'],
			reference=entry_data.get('reference'),
			entry_date=entry_data['entry_date'],
			posting_date=entry_data['posting_date'],
			period_id=period.period_id,
			source=entry_data.get('source', 'Manual'),
			requires_approval=entry_data.get('requires_approval', False)
		)
		
		db.session.add(journal)
		db.session.flush()  # Get journal ID
		
		# Create journal lines
		total_debits = Decimal('0.00')
		total_credits = Decimal('0.00')
		
		for i, line_data in enumerate(entry_data['lines'], 1):
			account = self.get_account(line_data['account_id'])
			if not account or not account.allow_posting:
				raise ValueError(f"Invalid account or account does not allow posting: {line_data['account_id']}")
			
			line = CFGLJournalLine(
				journal_id=journal.journal_id,
				tenant_id=self.tenant_id,
				line_number=i,
				description=line_data.get('description'),
				account_id=line_data['account_id'],
				debit_amount=Decimal(str(line_data.get('debit_amount', 0))),
				credit_amount=Decimal(str(line_data.get('credit_amount', 0))),
				cost_center=line_data.get('cost_center'),
				department=line_data.get('department'),
				project=line_data.get('project'),
				employee_id=line_data.get('employee_id'),
				reference_type=line_data.get('reference_type'),
				reference_id=line_data.get('reference_id'),
				reference_number=line_data.get('reference_number'),
				tax_code=line_data.get('tax_code'),
				tax_amount=Decimal(str(line_data.get('tax_amount', 0)))
			)
			
			total_debits += line.debit_amount
			total_credits += line.credit_amount
			
			db.session.add(line)
		
		# Update journal totals
		journal.total_debits = total_debits
		journal.total_credits = total_credits
		journal.line_count = len(entry_data['lines'])
		
		# Validate balance
		if not journal.validate_balance():
			raise ValueError("Journal entry does not balance")
		
		db.session.commit()
		return journal
	
	def post_journal_entry(self, journal_id: str, user_id: str) -> bool:
		"""Post journal entry"""
		journal = CFGLJournalEntry.query.filter_by(
			tenant_id=self.tenant_id,
			journal_id=journal_id
		).first()
		
		if not journal:
			raise ValueError("Journal entry not found")
		
		if not journal.can_post():
			raise ValueError("Journal entry cannot be posted")
		
		try:
			journal.post_entry(user_id)
			
			# Update account balances
			for line in journal.lines:
				account = line.account
				if line.debit_amount > 0:
					if account.is_debit_balance():
						account.current_balance += line.debit_amount
					else:
						account.current_balance -= line.debit_amount
				else:
					if account.is_debit_balance():
						account.current_balance -= line.credit_amount
					else:
						account.current_balance += line.credit_amount
			
			db.session.commit()
			return True
			
		except Exception as e:
			db.session.rollback()
			raise e
	
	def get_journal_entry(self, journal_id: str) -> Optional[CFGLJournalEntry]:
		"""Get journal entry by ID"""
		return CFGLJournalEntry.query.filter_by(
			tenant_id=self.tenant_id,
			journal_id=journal_id
		).first()
	
	def get_journal_entries(
		self, 
		status: Optional[str] = None,
		period_id: Optional[str] = None,
		account_id: Optional[str] = None,
		date_from: Optional[date] = None,
		date_to: Optional[date] = None,
		limit: int = 100
	) -> List[CFGLJournalEntry]:
		"""Get journal entries with filters"""
		query = CFGLJournalEntry.query.filter_by(tenant_id=self.tenant_id)
		
		if status:
			query = query.filter_by(status=status)
		if period_id:
			query = query.filter_by(period_id=period_id)
		if date_from:
			query = query.filter(CFGLJournalEntry.posting_date >= date_from)
		if date_to:
			query = query.filter(CFGLJournalEntry.posting_date <= date_to)
		if account_id:
			# Filter by journals that have lines for this account
			query = query.join(CFGLJournalLine).filter(CFGLJournalLine.account_id == account_id)
		
		return query.order_by(desc(CFGLJournalEntry.posting_date)).limit(limit).all()
	
	# Reporting
	
	def generate_trial_balance(
		self, 
		as_of_date: Optional[date] = None,
		account_type: Optional[str] = None
	) -> List[Dict[str, Any]]:
		"""Generate trial balance"""
		if as_of_date is None:
			as_of_date = date.today()
		
		# Get all accounts with their balances
		query = db.session.query(
			CFGLAccount,
			func.coalesce(
				func.sum(
					func.case(
						[(CFGLPosting.debit_amount > 0, CFGLPosting.debit_amount)],
						else_=0
					)
				), 0
			).label('total_debits'),
			func.coalesce(
				func.sum(
					func.case(
						[(CFGLPosting.credit_amount > 0, CFGLPosting.credit_amount)],
						else_=0
					)
				), 0
			).label('total_credits')
		).outerjoin(
			CFGLPosting, 
			and_(
				CFGLAccount.account_id == CFGLPosting.account_id,
				CFGLPosting.posting_date <= as_of_date,
				CFGLPosting.is_posted == True
			)
		).filter(
			CFGLAccount.tenant_id == self.tenant_id,
			CFGLAccount.is_active == True,
			CFGLAccount.allow_posting == True
		)
		
		if account_type:
			query = query.join(CFGLAccountType).filter(CFGLAccountType.type_code == account_type)
		
		query = query.group_by(CFGLAccount.account_id).order_by(CFGLAccount.account_code)
		
		trial_balance = []
		total_debits = Decimal('0.00')
		total_credits = Decimal('0.00')
		
		for account, debits, credits in query.all():
			# Calculate balance based on account type
			if account.is_debit_balance():
				balance = account.opening_balance + debits - credits
				debit_balance = balance if balance >= 0 else Decimal('0.00')
				credit_balance = abs(balance) if balance < 0 else Decimal('0.00')
			else:
				balance = account.opening_balance + credits - debits
				credit_balance = balance if balance >= 0 else Decimal('0.00')
				debit_balance = abs(balance) if balance < 0 else Decimal('0.00')
			
			trial_balance.append({
				'account_code': account.account_code,
				'account_name': account.account_name,
				'account_type': account.account_type.type_name,
				'debit_balance': debit_balance,
				'credit_balance': credit_balance
			})
			
			total_debits += debit_balance
			total_credits += credit_balance
		
		return {
			'as_of_date': as_of_date,
			'accounts': trial_balance,
			'total_debits': total_debits,
			'total_credits': total_credits,
			'balanced': abs(total_debits - total_credits) < 0.01
		}
	
	def get_account_ledger(
		self, 
		account_id: str,
		date_from: Optional[date] = None,
		date_to: Optional[date] = None,
		limit: int = 100
	) -> Dict[str, Any]:
		"""Get account ledger (detailed transactions)"""
		account = self.get_account(account_id)
		if not account:
			raise ValueError("Account not found")
		
		query = CFGLPosting.query.filter_by(
			tenant_id=self.tenant_id,
			account_id=account_id,
			is_posted=True
		)
		
		if date_from:
			query = query.filter(CFGLPosting.posting_date >= date_from)
		if date_to:
			query = query.filter(CFGLPosting.posting_date <= date_to)
		
		postings = query.order_by(CFGLPosting.posting_date, CFGLPosting.created_on).limit(limit).all()
		
		# Calculate running balance
		running_balance = account.opening_balance
		ledger_entries = []
		
		for posting in postings:
			if account.is_debit_balance():
				running_balance += posting.debit_amount - posting.credit_amount
			else:
				running_balance += posting.credit_amount - posting.debit_amount
			
			ledger_entries.append({
				'posting_date': posting.posting_date,
				'description': posting.description,
				'reference': posting.reference,
				'debit_amount': posting.debit_amount,
				'credit_amount': posting.credit_amount,
				'balance': running_balance
			})
		
		return {
			'account': {
				'code': account.account_code,
				'name': account.account_name,
				'type': account.account_type.type_name
			},
			'opening_balance': account.opening_balance,
			'closing_balance': running_balance,
			'entries': ledger_entries
		}
	
	# Utility Methods
	
	def _generate_journal_number(self) -> str:
		"""Generate unique journal number"""
		today = date.today()
		prefix = f"JE-{today.strftime('%Y%m')}"
		
		# Get next sequence number for the month
		last_journal = CFGLJournalEntry.query.filter(
			CFGLJournalEntry.tenant_id == self.tenant_id,
			CFGLJournalEntry.journal_number.like(f"{prefix}%")
		).order_by(desc(CFGLJournalEntry.journal_number)).first()
		
		if last_journal:
			try:
				last_seq = int(last_journal.journal_number.split('-')[-1])
				next_seq = last_seq + 1
			except (ValueError, IndexError):
				next_seq = 1
		else:
			next_seq = 1
		
		return f"{prefix}-{next_seq:04d}"
	
	def validate_account_code(self, account_code: str, exclude_id: Optional[str] = None) -> bool:
		"""Validate account code uniqueness"""
		query = CFGLAccount.query.filter_by(
			tenant_id=self.tenant_id,
			account_code=account_code
		)
		
		if exclude_id:
			query = query.filter(CFGLAccount.account_id != exclude_id)
		
		return query.first() is None
	
	def get_account_types(self) -> List[CFGLAccountType]:
		"""Get all account types"""
		return CFGLAccountType.query.filter_by(
			tenant_id=self.tenant_id
		).order_by(CFGLAccountType.sort_order, CFGLAccountType.type_name).all()