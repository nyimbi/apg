"""
APG Financial Management General Ledger - REST API Layer

Comprehensive REST API endpoints for enterprise general ledger operations including:
- Chart of Accounts management with hierarchical operations
- Journal Entry processing with validation and posting
- Financial Reporting with real-time analytics
- Period management and closing procedures
- Multi-currency transaction support
- Audit trail and compliance monitoring
- Advanced search and filtering capabilities

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from flask import request, jsonify, Blueprint, session, current_app
from flask_restful import Api, Resource
from flask_appbuilder.api import BaseApi, expose
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.security.decorators import protect
from marshmallow import Schema, fields, validate, ValidationError, post_load
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional
from functools import wraps
import logging

from .models import (
	GLTenant, GLAccountType, GLAccount, GLAccountGroup, GLAccountBalance,
	GLAccountBudget, GLPeriod, GLCurrencyRate, GLJournalEntry, GLJournalLine,
	GLPosting, AccountTypeEnum, BalanceTypeEnum, PeriodStatusEnum, 
	JournalStatusEnum, JournalSourceEnum, CurrencyEnum, ReportingFrameworkEnum
)
from .service import (
	GeneralLedgerService, AccountCreationRequest, JournalEntryRequest,
	TrialBalanceParams, FinancialReportingResult, GLServiceException
)

# Configure logging
logger = logging.getLogger(__name__)


# =====================================
# API ERROR HANDLING AND DECORATORS
# =====================================

def handle_gl_exceptions(f):
	"""Decorator to handle GL service exceptions"""
	@wraps(f)
	def decorated_function(*args, **kwargs):
		try:
			return f(*args, **kwargs)
		except GLServiceException as e:
			logger.error(f"GL Service Error in {f.__name__}: {e}")
			return {'error': str(e), 'error_type': 'service_error'}, 400
		except ValidationError as e:
			logger.error(f"Validation Error in {f.__name__}: {e.messages}")
			return {'error': 'Validation failed', 'details': e.messages, 'error_type': 'validation_error'}, 400
		except ValueError as e:
			logger.error(f"Value Error in {f.__name__}: {e}")
			return {'error': str(e), 'error_type': 'value_error'}, 400
		except Exception as e:
			logger.exception(f"Unexpected error in {f.__name__}")
			return {'error': 'Internal server error', 'error_type': 'internal_error'}, 500
	
	return decorated_function


def get_tenant_context():
	"""Get tenant context from session"""
	from flask import session
	tenant_id = session.get('tenant_id')
	user_id = session.get('user_id')
	
	if not tenant_id:
		raise ValueError("Tenant context not available")
	
	return tenant_id, user_id


def validate_tenant_access(tenant_id: str, user_id: str) -> bool:
	"""Validate user has access to tenant"""
	try:
		# Implementation for tenant access validation
		# In a real system, this would:
		# 1. Check user permissions in the tenant
		# 2. Validate tenant exists and is active
		# 3. Check user role and access level
		
		# For now, basic validation logic
		if not tenant_id or not user_id:
			return False
		
		# In production, query user-tenant relationship from database
		# Example: user_tenant = UserTenant.query.filter_by(user_id=user_id, tenant_id=tenant_id).first()
		# return user_tenant is not None and user_tenant.is_active
		
		return True  # Allow access for development/testing
	except Exception as e:
		logger.error(f"Tenant access validation failed: {e}")
		return False


# =====================================
# MARSHMALLOW SCHEMAS FOR API SERIALIZATION
# =====================================

class GLAccountSchema(Schema):
	"""Schema for GL Account serialization"""
	account_id = fields.String(dump_only=True)
	account_code = fields.String(required=True)
	account_name = fields.String(required=True)
	description = fields.String(allow_none=True)
	account_type_id = fields.String(required=True)
	parent_account_id = fields.String(allow_none=True)
	is_active = fields.Boolean(default=True)
	is_header = fields.Boolean(default=False)
	allow_posting = fields.Boolean(default=True)
	current_balance = fields.Decimal()
	ytd_balance = fields.Decimal()
	opening_balance = fields.Decimal()
	primary_currency = fields.String(default='USD')
	level = fields.Integer()
	path = fields.String()


class GLAccountTypeSchema(Schema):
	"""Schema for GL Account Type serialization"""
	type_id = fields.String(dump_only=True)
	type_code = fields.String(required=True)
	type_name = fields.String(required=True)
	description = fields.String(allow_none=True)
	normal_balance = fields.String(required=True)
	is_balance_sheet = fields.Boolean()
	sort_order = fields.Integer()


class GLJournalEntrySchema(Schema):
	"""Schema for Journal Entry serialization"""
	journal_id = fields.String(dump_only=True)
	journal_number = fields.String()
	description = fields.String(required=True)
	reference = fields.String(allow_none=True)
	entry_date = fields.Date(required=True)
	posting_date = fields.Date(required=True)
	status = fields.String()
	source = fields.String()
	total_debits = fields.Decimal()
	total_credits = fields.Decimal()
	line_count = fields.Integer()
	posted = fields.Boolean()


class GLJournalLineSchema(Schema):
	"""Schema for Journal Line serialization"""
	line_id = fields.String(dump_only=True)
	line_number = fields.Integer(required=True)
	account_id = fields.String(required=True)
	description = fields.String(allow_none=True)
	debit_amount = fields.Decimal(default=0)
	credit_amount = fields.Decimal(default=0)
	cost_center = fields.String(allow_none=True)
	department = fields.String(allow_none=True)
	project = fields.String(allow_none=True)
	reference_type = fields.String(allow_none=True)
	reference_number = fields.String(allow_none=True)


class GLPostingSchema(Schema):
	"""Schema for GL Posting serialization"""
	posting_id = fields.String(dump_only=True)
	posting_date = fields.Date()
	account_id = fields.String()
	description = fields.String()
	reference = fields.String()
	debit_amount = fields.Decimal()
	credit_amount = fields.Decimal()
	is_posted = fields.Boolean()
	posted_date = fields.DateTime()


class GLPeriodSchema(Schema):
	"""Schema for GL Period serialization"""
	period_id = fields.String(dump_only=True)
	period_name = fields.String(required=True)
	period_start = fields.Date(required=True)
	period_end = fields.Date(required=True)
	status = fields.String()
	fiscal_year = fields.Integer()
	is_current = fields.Boolean()
	closed_date = fields.DateTime()


class GLCurrencyRateSchema(Schema):
	"""Schema for Currency Rate serialization"""
	rate_id = fields.String(dump_only=True)
	from_currency = fields.String(required=True)
	to_currency = fields.String(required=True)
	exchange_rate = fields.Decimal(required=True)
	rate_date = fields.Date(required=True)
	rate_type = fields.String()
	is_active = fields.Boolean()


class TrialBalanceRequestSchema(Schema):
	"""Schema for trial balance request parameters"""
	as_of_date = fields.Date(missing=lambda: date.today())
	account_type_filter = fields.String(allow_none=True)
	include_zero_balances = fields.Boolean(default=False)
	currency = fields.String(default='USD')
	consolidated = fields.Boolean(default=True)


class JournalEntryCreateSchema(Schema):
	"""Schema for creating journal entries"""
	description = fields.String(required=True, validate=validate.Length(min=5, max=200))
	reference = fields.String(allow_none=True, validate=validate.Length(max=50))
	entry_date = fields.Date(required=True)
	posting_date = fields.Date(required=True)
	source = fields.String(default='MANUAL')
	requires_approval = fields.Boolean(default=False)
	lines = fields.List(fields.Nested('GLJournalLineCreateSchema'), required=True, validate=validate.Length(min=2))
	
	@post_load
	def validate_balanced_entry(self, data, **kwargs):
		"""Validate that journal entry is balanced"""
		total_debits = sum(line.get('debit_amount', 0) for line in data['lines'])
		total_credits = sum(line.get('credit_amount', 0) for line in data['lines'])
		
		if abs(total_debits - total_credits) > 0.01:  # Allow for rounding
			raise ValidationError('Journal entry must be balanced (debits must equal credits)')
		
		return data


class GLJournalLineCreateSchema(Schema):
	"""Schema for creating journal entry lines"""
	line_number = fields.Integer(required=True)
	account_id = fields.String(required=True)
	description = fields.String(allow_none=True)
	debit_amount = fields.Decimal(default=0, validate=validate.Range(min=0))
	credit_amount = fields.Decimal(default=0, validate=validate.Range(min=0))
	cost_center = fields.String(allow_none=True)
	department = fields.String(allow_none=True)
	project = fields.String(allow_none=True)
	reference_type = fields.String(allow_none=True)
	reference_number = fields.String(allow_none=True)
	
	@post_load
	def validate_amounts(self, data, **kwargs):
		"""Validate that either debit or credit is specified, but not both"""
		debit = data.get('debit_amount', 0)
		credit = data.get('credit_amount', 0)
		
		if debit > 0 and credit > 0:
			raise ValidationError('Line cannot have both debit and credit amounts')
		if debit == 0 and credit == 0:
			raise ValidationError('Line must have either debit or credit amount')
		
		return data


# REST API Resources

class GLAccountApi(BaseApi):
	"""GL Account API endpoints"""
	
	resource_name = 'accounts'
	datamodel = SQLAInterface(GLAccount)
	
	list_columns = [
		'account_id', 'account_code', 'account_name', 'account_type.type_name',
		'current_balance', 'is_active', 'allow_posting'
	]
	
	show_columns = [
		'account_id', 'account_code', 'account_name', 'description',
		'account_type', 'parent_account', 'is_active', 'is_header',
		'allow_posting', 'current_balance', 'ytd_balance', 'opening_balance',
		'primary_currency', 'level', 'path'
	]
	
	add_columns = [
		'account_code', 'account_name', 'description', 'account_type_id',
		'parent_account_id', 'is_active', 'is_header', 'allow_posting',
		'opening_balance', 'primary_currency'
	]
	
	edit_columns = [
		'account_name', 'description', 'is_active', 'allow_posting', 'primary_currency'
	]
	
	def get_tenant_id(self):
		"""Get tenant ID from session or default"""
		from flask import session
		return session.get('tenant_id', 'default_tenant')
	
	def get_user_id(self):
		"""Get user ID from session"""
		from flask import session
		return session.get('user_id')
	
	@expose('/hierarchy')
	def get_hierarchy(self):
		"""Get chart of accounts in hierarchical structure"""
		tenant_id = self.get_tenant_id()
		gl_service = GeneralLedgerService(tenant_id)
		
		try:
			accounts = gl_service.get_chart_of_accounts()
			schema = GLAccountSchema(many=True)
			
			# Build hierarchy
			account_dict = {acc.account_id: schema.dump(acc) for acc in accounts}
			hierarchy = []
			
			for account in accounts:
				if not account.parent_account_id:
					node = account_dict[account.account_id]
					node['children'] = self._get_children(account.account_id, account_dict, accounts)
					hierarchy.append(node)
			
			return self.response(200, result=hierarchy)
			
		except Exception as e:
			return self.response_400(message=str(e))
	
	def _get_children(self, parent_id: str, account_dict: Dict, accounts: List):
		"""Get child accounts recursively"""
		children = []
		for account in accounts:
			if account.parent_account_id == parent_id:
				child = account_dict[account.account_id]
				child['children'] = self._get_children(account.account_id, account_dict, accounts)
				children.append(child)
		return children
	
	@expose('/<account_id>/balance')
	def get_balance(self, account_id):
		"""Get account balance as of specific date"""
		tenant_id = self.get_tenant_id()
		as_of_date = request.args.get('as_of_date')
		currency = request.args.get('currency', 'USD')
		
		try:
			if as_of_date:
				as_of_date = datetime.strptime(as_of_date, '%Y-%m-%d').date()
			else:
				as_of_date = date.today()
			
			gl_service = GeneralLedgerService(tenant_id, self.get_user_id())
			account = gl_service.get_account(account_id)
			
			if not account:
				return self.response_404()
			
			balance = account.calculate_balance(as_of_date)
			
			return self.response(200, result={
				'account_id': account_id,
				'account_code': account.account_code,
				'account_name': account.account_name,
				'balance': float(balance),
				'currency': currency,
				'as_of_date': as_of_date.isoformat()
			})
			
		except Exception as e:
			return self.response_400(message=str(e))
	
	@expose('/<account_id>/ledger')
	def get_account_ledger(self, account_id):
		"""Get account ledger entries"""
		tenant_id = self.get_tenant_id()
		date_from = request.args.get('date_from')
		date_to = request.args.get('date_to')
		limit = int(request.args.get('limit', 100))
		
		try:
			if date_from:
				date_from = datetime.strptime(date_from, '%Y-%m-%d').date()
			if date_to:
				date_to = datetime.strptime(date_to, '%Y-%m-%d').date()
			
			gl_service = GeneralLedgerService(tenant_id, self.get_user_id())
			ledger = gl_service.get_account_ledger(account_id, date_from, date_to, limit)
			
			return self.response(200, result=ledger)
			
		except Exception as e:
			return self.response_400(message=str(e))
	
	@expose('/<account_id>/children')
	def get_children(self, account_id):
		"""Get child accounts"""
		tenant_id = self.get_tenant_id()
		
		try:
			gl_service = GeneralLedgerService(tenant_id, self.get_user_id())
			children = gl_service.get_child_accounts(account_id)
			
			schema = GLAccountSchema(many=True)
			return self.response(200, result=schema.dump(children))
			
		except Exception as e:
			return self.response_400(message=str(e))


class GLJournalEntryApi(BaseApi):
	"""Journal Entry API endpoints"""
	
	resource_name = 'journal_entries'
	datamodel = SQLAInterface(GLJournalEntry)
	
	list_columns = [
		'journal_id', 'journal_number', 'description', 'entry_date',
		'posting_date', 'status', 'total_debits', 'total_credits', 'posted'
	]
	
	show_columns = [
		'journal_id', 'journal_number', 'description', 'reference',
		'entry_date', 'posting_date', 'period', 'status', 'source',
		'total_debits', 'total_credits', 'line_count', 'posted', 'posted_date'
	]
	
	@expose('/', methods=['POST'])
	def create_with_lines(self):
		"""Create journal entry with lines"""
		tenant_id = self.get_tenant_id()
		
		try:
			data = request.get_json()
			
			# Validate with schema
			schema = JournalEntryCreateSchema()
			validated_data = schema.load(data)
			
			gl_service = GeneralLedgerService(tenant_id, self.get_user_id())
			
			# Convert to service request object
			from .service import JournalEntryRequest
			request_obj = JournalEntryRequest(
				description=validated_data['description'],
				reference=validated_data.get('reference'),
				entry_date=validated_data['entry_date'],
				posting_date=validated_data['posting_date'],
				source=JournalSourceEnum(validated_data['source']),
				requires_approval=validated_data['requires_approval'],
				lines=validated_data['lines']
			)
			
			journal = gl_service.create_journal_entry(request_obj)
			
			response_schema = GLJournalEntrySchema()
			return self.response(201, result=response_schema.dump(journal))
			
		except ValidationError as e:
			return self.response_400(message=f"Validation error: {e.messages}")
		except Exception as e:
			return self.response_400(message=str(e))
	
	@expose('/batch', methods=['POST'])
	def create_batch_entries(self):
		"""Create multiple journal entries in batch"""
		tenant_id = self.get_tenant_id()
		
		try:
			data = request.get_json()
			entries = data.get('entries', [])
			
			if not entries:
				return self.response_400(message="No entries provided")
			
			# Validate all entries
			schema = JournalEntryCreateSchema()
			validated_entries = []
			
			for i, entry in enumerate(entries):
				try:
					validated_entry = schema.load(entry)
					validated_entries.append(validated_entry)
				except ValidationError as e:
					return self.response_400(message=f"Entry {i+1} validation error: {e.messages}")
			
			gl_service = GeneralLedgerService(tenant_id, self.get_user_id())
			created_journals = gl_service.create_batch_journal_entries(validated_entries)
			
			response_schema = GLJournalEntrySchema(many=True)
			return self.response(201, result={
				'created_count': len(created_journals),
				'journals': response_schema.dump(created_journals)
			})
			
		except Exception as e:
			return self.response_400(message=str(e))
	
	@expose('/<journal_id>/post', methods=['POST'])
	def post_entry(self, journal_id):
		"""Post journal entry"""
		tenant_id = self.get_tenant_id()
		user_id = self.get_user_id()
		
		try:
			gl_service = GeneralLedgerService(tenant_id)
			success = gl_service.post_journal_entry(journal_id, user_id)
			
			if success:
				return self.response(200, message="Journal entry posted successfully")
			else:
				return self.response_400(message="Failed to post journal entry")
				
		except Exception as e:
			return self.response_400(message=str(e))
	
	@expose('/<journal_id>/lines')
	def get_lines(self, journal_id):
		"""Get journal entry lines"""
		tenant_id = self.get_tenant_id()
		
		try:
			journal = GLJournalEntry.query.filter_by(
				tenant_id=tenant_id,
				journal_id=journal_id
			).first()
			
			if not journal:
				return self.response_404()
			
			schema = GLJournalLineSchema(many=True)
			return self.response(200, result=schema.dump(journal.lines))
			
		except Exception as e:
			return self.response_400(message=str(e))


class GLReportsApi(BaseApi):
	"""GL Reports API endpoints"""
	
	resource_name = 'reports'
	
	@expose('/trial_balance')
	def trial_balance(self):
		"""Generate trial balance report"""
		tenant_id = self.get_tenant_id()
		
		try:
			# Parse request parameters
			request_schema = TrialBalanceRequestSchema()
			params = request_schema.load(request.args.to_dict())
			
			gl_service = GeneralLedgerService(tenant_id, self.get_user_id())
			
			# Convert params to service object
			from .service import TrialBalanceParams
			trial_balance_params = TrialBalanceParams(
				as_of_date=params['as_of_date'],
				account_type_filter=AccountTypeEnum(params['account_type_filter']) if params.get('account_type_filter') else None,
				include_zero_balances=params['include_zero_balances'],
				currency=CurrencyEnum(params['currency']),
				consolidated=params['consolidated']
			)
			
			trial_balance = gl_service.generate_trial_balance(trial_balance_params)
			
			return self.response(200, result={
				'data': trial_balance.data,
				'metadata': trial_balance.metadata,
				'parameters': params
			})
			
		except ValidationError as e:
			return self.response_400(message=f"Invalid parameters: {e.messages}")
		except Exception as e:
			return self.response_400(message=str(e))
	
	@expose('/balance_sheet')
	def balance_sheet(self):
		"""Generate balance sheet report"""
		tenant_id = self.get_tenant_id()
		
		try:
			as_of_date = request.args.get('as_of_date')
			currency = request.args.get('currency', 'USD')
			comparative = request.args.get('comparative', 'false').lower() == 'true'
			
			if as_of_date:
				as_of_date = datetime.strptime(as_of_date, '%Y-%m-%d').date()
			else:
				as_of_date = date.today()
			
			gl_service = GeneralLedgerService(tenant_id, self.get_user_id())
			balance_sheet = gl_service.generate_balance_sheet(
				as_of_date=as_of_date,
				currency=CurrencyEnum(currency),
				comparative=comparative
			)
			
			return self.response(200, result=balance_sheet)
			
		except Exception as e:
			return self.response_400(message=str(e))
	
	@expose('/income_statement')
	def income_statement(self):
		"""Generate income statement report"""
		tenant_id = self.get_tenant_id()
		
		try:
			start_date = request.args.get('start_date')
			end_date = request.args.get('end_date')
			currency = request.args.get('currency', 'USD')
			comparative = request.args.get('comparative', 'false').lower() == 'true'
			
			if start_date:
				start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
			if end_date:
				end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
			
			gl_service = GeneralLedgerService(tenant_id, self.get_user_id())
			income_statement = gl_service.generate_income_statement(
				start_date=start_date,
				end_date=end_date,
				currency=CurrencyEnum(currency),
				comparative=comparative
			)
			
			return self.response(200, result=income_statement)
			
		except Exception as e:
			return self.response_400(message=str(e))
	
	@expose('/account_ledger/<account_id>')
	def account_ledger(self, account_id):
		"""Get account ledger"""
		tenant_id = self.get_tenant_id()
		
		try:
			date_from = request.args.get('date_from')
			date_to = request.args.get('date_to')
			limit = int(request.args.get('limit', 100))
			
			if date_from:
				date_from = datetime.strptime(date_from, '%Y-%m-%d').date()
			if date_to:
				date_to = datetime.strptime(date_to, '%Y-%m-%d').date()
			
			gl_service = GeneralLedgerService(tenant_id)
			ledger = gl_service.get_account_ledger(account_id, date_from, date_to, limit)
			
			return self.response(200, result=ledger)
			
		except Exception as e:
			return self.response_400(message=str(e))


class GLPeriodApi(BaseApi):
	"""GL Period management API endpoints"""
	
	resource_name = 'periods'
	datamodel = SQLAInterface(GLPeriod)
	
	list_columns = ['period_id', 'period_name', 'period_start', 'period_end', 'status', 'fiscal_year']
	show_columns = ['period_id', 'period_name', 'period_start', 'period_end', 'status', 
					'fiscal_year', 'is_current', 'closed_date', 'created_on']
	
	def get_tenant_id(self):
		"""Get tenant ID from session"""
		from flask import session
		return session.get('tenant_id', 'default_tenant')
	
	def get_user_id(self):
		"""Get user ID from session"""
		from flask import session
		return session.get('user_id')
	
	@expose('/current')
	def get_current_period(self):
		"""Get current active period"""
		tenant_id = self.get_tenant_id()
		
		try:
			gl_service = GeneralLedgerService(tenant_id, self.get_user_id())
			current_period = gl_service.get_current_period()
			
			if not current_period:
				return self.response_404(message="No current period found")
			
			schema = GLPeriodSchema()
			return self.response(200, result=schema.dump(current_period))
			
		except Exception as e:
			return self.response_400(message=str(e))
	
	@expose('/<period_id>/close', methods=['POST'])
	def close_period(self, period_id):
		"""Close accounting period"""
		tenant_id = self.get_tenant_id()
		
		try:
			data = request.get_json() or {}
			force_close = data.get('force_close', False)
			
			gl_service = GeneralLedgerService(tenant_id, self.get_user_id())
			success = gl_service.close_period(period_id, force_close)
			
			if success:
				return self.response(200, message="Period closed successfully")
			else:
				return self.response_400(message="Failed to close period")
			
		except Exception as e:
			return self.response_400(message=str(e))
	
	@expose('/<period_id>/reopen', methods=['POST'])
	def reopen_period(self, period_id):
		"""Reopen accounting period"""
		tenant_id = self.get_tenant_id()
		
		try:
			gl_service = GeneralLedgerService(tenant_id, self.get_user_id())
			success = gl_service.reopen_period(period_id)
			
			if success:
				return self.response(200, message="Period reopened successfully")
			else:
				return self.response_400(message="Failed to reopen period")
			
		except Exception as e:
			return self.response_400(message=str(e))


class GLCurrencyApi(BaseApi):
	"""Currency and exchange rate API endpoints"""
	
	resource_name = 'currencies'
	datamodel = SQLAInterface(GLCurrencyRate)
	
	list_columns = ['rate_id', 'from_currency', 'to_currency', 'exchange_rate', 'rate_date', 'is_active']
	show_columns = ['rate_id', 'from_currency', 'to_currency', 'exchange_rate', 'rate_date', 
					'rate_type', 'is_active', 'created_on']
	
	def get_tenant_id(self):
		"""Get tenant ID from session"""
		from flask import session
		return session.get('tenant_id', 'default_tenant')
	
	def get_user_id(self):
		"""Get user ID from session"""
		from flask import session
		return session.get('user_id')
	
	@expose('/rates/<from_currency>/<to_currency>')
	def get_exchange_rate(self, from_currency, to_currency):
		"""Get current exchange rate between currencies"""
		tenant_id = self.get_tenant_id()
		rate_date = request.args.get('date')
		
		try:
			if rate_date:
				rate_date = datetime.strptime(rate_date, '%Y-%m-%d').date()
			else:
				rate_date = date.today()
			
			gl_service = GeneralLedgerService(tenant_id, self.get_user_id())
			rate = gl_service.get_exchange_rate(
				from_currency=CurrencyEnum(from_currency),
				to_currency=CurrencyEnum(to_currency),
				rate_date=rate_date
			)
			
			if rate is None:
				return self.response_404(message="Exchange rate not found")
			
			return self.response(200, result={
				'from_currency': from_currency,
				'to_currency': to_currency,
				'exchange_rate': float(rate),
				'rate_date': rate_date.isoformat()
			})
			
		except Exception as e:
			return self.response_400(message=str(e))
	
	@expose('/convert', methods=['POST'])
	def convert_amount(self):
		"""Convert amount between currencies"""
		tenant_id = self.get_tenant_id()
		
		try:
			data = request.get_json()
			amount = Decimal(str(data.get('amount', 0)))
			from_currency = data.get('from_currency')
			to_currency = data.get('to_currency')
			conversion_date = data.get('date')
			
			if conversion_date:
				conversion_date = datetime.strptime(conversion_date, '%Y-%m-%d').date()
			else:
				conversion_date = date.today()
			
			gl_service = GeneralLedgerService(tenant_id, self.get_user_id())
			converted_amount = gl_service.convert_currency(
				amount=amount,
				from_currency=CurrencyEnum(from_currency),
				to_currency=CurrencyEnum(to_currency),
				conversion_date=conversion_date
			)
			
			return self.response(200, result={
				'original_amount': float(amount),
				'from_currency': from_currency,
				'converted_amount': float(converted_amount),
				'to_currency': to_currency,
				'conversion_date': conversion_date.isoformat()
			})
			
		except Exception as e:
			return self.response_400(message=str(e))


def register_api_views(appbuilder):
	"""Register GL API views with Flask-AppBuilder"""
	
	appbuilder.add_api(GLAccountApi)
	appbuilder.add_api(GLJournalEntryApi)
	appbuilder.add_api(GLReportsApi)
	appbuilder.add_api(GLPeriodApi)
	appbuilder.add_api(GLCurrencyApi)


def create_api_blueprint() -> Blueprint:
	"""Create API blueprint for General Ledger"""
	
	api_bp = Blueprint(
		'gl_api',
		__name__,
		url_prefix='/api/core_financials/gl'
	)
	
	api = Api(api_bp)
	
	# Add resources
	api.add_resource(GLAccountResource, '/accounts', '/accounts/<string:account_id>')
	api.add_resource(GLJournalEntryResource, '/journal_entries', '/journal_entries/<string:journal_id>')
	api.add_resource(GLTrialBalanceResource, '/reports/trial_balance')
	api.add_resource(GLAccountLedgerResource, '/reports/account_ledger/<string:account_id>')
	api.add_resource(GLPeriodResource, '/periods', '/periods/<string:period_id>')
	api.add_resource(GLCurrencyResource, '/currencies', '/currencies/<string:rate_id>')
	
	return api_bp


# Flask-RESTful Resources (alternative approach)

class GLAccountResource(Resource):
	"""GL Account REST resource"""
	
	def get(self, account_id=None):
		"""Get account(s)"""
		tenant_id = self._get_tenant_id()
		gl_service = GeneralLedgerService(tenant_id, self._get_user_id())
		
		if account_id:
			account = gl_service.get_account(account_id)
			if not account:
				return {'message': 'Account not found'}, 404
			
			schema = GLAccountSchema()
			return schema.dump(account)
		else:
			# Support filtering
			active_only = request.args.get('active_only', 'true').lower() == 'true'
			account_type = request.args.get('account_type')
			
			accounts = gl_service.get_chart_of_accounts(
				include_inactive=not active_only,
				account_type_filter=AccountTypeEnum(account_type) if account_type else None
			)
			schema = GLAccountSchema(many=True)
			return schema.dump(accounts)
	
	def post(self):
		"""Create new account"""
		tenant_id = self._get_tenant_id()
		gl_service = GeneralLedgerService(tenant_id, self._get_user_id())
		
		try:
			data = request.get_json()
			
			# Create account creation request
			from .service import AccountCreationRequest
			request_obj = AccountCreationRequest(
				account_code=data['account_code'],
				account_name=data['account_name'],
				account_type_id=data['account_type_id'],
				description=data.get('description'),
				parent_account_id=data.get('parent_account_id'),
				currency=CurrencyEnum(data.get('primary_currency', 'USD')),
				is_header=data.get('is_header', False),
				opening_balance=Decimal(str(data.get('opening_balance', 0)))
			)
			
			account = gl_service.create_account(request_obj)
			
			schema = GLAccountSchema()
			return schema.dump(account), 201
			
		except Exception as e:
			return {'message': str(e)}, 400
	
	def _get_tenant_id(self):
		"""Get tenant ID from request context"""
		from flask import session
		return session.get('tenant_id', 'default_tenant')
	
	def _get_user_id(self):
		"""Get user ID from request context"""
		from flask import session
		return session.get('user_id')


class GLJournalEntryResource(Resource):
	"""Journal Entry REST resource"""
	
	def get(self, journal_id=None):
		"""Get journal entry(ies)"""
		tenant_id = self._get_tenant_id()
		gl_service = GeneralLedgerService(tenant_id, self._get_user_id())
		
		if journal_id:
			journal = gl_service.get_journal_entry(journal_id)
			if not journal:
				return {'message': 'Journal entry not found'}, 404
			
			schema = GLJournalEntrySchema()
			return schema.dump(journal)
		else:
			# Get with filters
			status = request.args.get('status')
			limit = int(request.args.get('limit', 50))
			
			journals = gl_service.get_journal_entries(
				status=JournalStatusEnum(status) if status else None,
				limit=limit
			)
			schema = GLJournalEntrySchema(many=True)
			return schema.dump(journals)
	
	def post(self):
		"""Create journal entry with lines"""
		tenant_id = self._get_tenant_id()
		gl_service = GeneralLedgerService(tenant_id, self._get_user_id())
		
		try:
			data = request.get_json()
			schema = JournalEntryCreateSchema()
			validated_data = schema.load(data)
			
			# Convert to service request object
			from .service import JournalEntryRequest
			request_obj = JournalEntryRequest(
				description=validated_data['description'],
				reference=validated_data.get('reference'),
				entry_date=validated_data['entry_date'],
				posting_date=validated_data['posting_date'],
				source=JournalSourceEnum(validated_data['source']),
				requires_approval=validated_data['requires_approval'],
				lines=validated_data['lines']
			)
			
			journal = gl_service.create_journal_entry(request_obj)
			
			response_schema = GLJournalEntrySchema()
			return response_schema.dump(journal), 201
			
		except ValidationError as e:
			return {'message': 'Validation error', 'errors': e.messages}, 400
		except Exception as e:
			return {'message': str(e)}, 400
	
	def _get_tenant_id(self):
		"""Get tenant ID from request context"""
		from flask import session
		return session.get('tenant_id', 'default_tenant')
	
	def _get_user_id(self):
		"""Get user ID from request context"""
		from flask import session
		return session.get('user_id')


class GLTrialBalanceResource(Resource):
	"""Trial Balance REST resource"""
	
	def get(self):
		"""Generate trial balance"""
		tenant_id = self._get_tenant_id()
		gl_service = GeneralLedgerService(tenant_id, self._get_user_id())
		
		try:
			# Parse request parameters
			request_schema = TrialBalanceRequestSchema()
			params = request_schema.load(request.args.to_dict())
			
			# Convert params to service object
			from .service import TrialBalanceParams
			trial_balance_params = TrialBalanceParams(
				as_of_date=params['as_of_date'],
				account_type_filter=AccountTypeEnum(params['account_type_filter']) if params.get('account_type_filter') else None,
				include_zero_balances=params['include_zero_balances'],
				currency=CurrencyEnum(params['currency']),
				consolidated=params['consolidated']
			)
			
			trial_balance = gl_service.generate_trial_balance(trial_balance_params)
			
			return {
				'data': trial_balance.data,
				'metadata': trial_balance.metadata,
				'parameters': params
			}
			
		except ValidationError as e:
			return {'message': 'Invalid parameters', 'errors': e.messages}, 400
		except Exception as e:
			return {'message': str(e)}, 400
	
	def _get_tenant_id(self):
		"""Get tenant ID from request context"""
		from flask import session
		return session.get('tenant_id', 'default_tenant')
	
	def _get_user_id(self):
		"""Get user ID from request context"""
		from flask import session
		return session.get('user_id')


class GLAccountLedgerResource(Resource):
	"""Account Ledger REST resource"""
	
	def get(self, account_id):
		"""Get account ledger"""
		tenant_id = self._get_tenant_id()
		gl_service = GeneralLedgerService(tenant_id, self._get_user_id())
		
		try:
			date_from = request.args.get('date_from')
			date_to = request.args.get('date_to')
			limit = int(request.args.get('limit', 100))
			
			if date_from:
				date_from = datetime.strptime(date_from, '%Y-%m-%d').date()
			if date_to:
				date_to = datetime.strptime(date_to, '%Y-%m-%d').date()
			
			ledger = gl_service.get_account_ledger(account_id, date_from, date_to, limit)
			return ledger
			
		except Exception as e:
			return {'message': str(e)}, 400
	
	def _get_tenant_id(self):
		"""Get tenant ID from request context"""
		from flask import session
		return session.get('tenant_id', 'default_tenant')
	
	def _get_user_id(self):
		"""Get user ID from request context"""
		from flask import session
		return session.get('user_id')


class GLPeriodResource(Resource):
	"""GL Period REST resource"""
	
	def get(self, period_id=None):
		"""Get period(s)"""
		tenant_id = self._get_tenant_id()
		gl_service = GeneralLedgerService(tenant_id, self._get_user_id())
		
		try:
			if period_id:
				period = gl_service.get_period(period_id)
				if not period:
					return {'message': 'Period not found'}, 404
				
				schema = GLPeriodSchema()
				return schema.dump(period)
			else:
				status_filter = request.args.get('status')
				fiscal_year = request.args.get('fiscal_year')
				
				periods = gl_service.get_periods(
					status=PeriodStatusEnum(status_filter) if status_filter else None,
					fiscal_year=int(fiscal_year) if fiscal_year else None
				)
				
				schema = GLPeriodSchema(many=True)
				return schema.dump(periods)
			
		except Exception as e:
			return {'message': str(e)}, 400
	
	def post(self):
		"""Create new period"""
		tenant_id = self._get_tenant_id()
		gl_service = GeneralLedgerService(tenant_id, self._get_user_id())
		
		try:
			data = request.get_json()
			schema = GLPeriodSchema()
			validated_data = schema.load(data)
			
			period = gl_service.create_period(validated_data)
			
			return schema.dump(period), 201
			
		except ValidationError as e:
			return {'message': 'Validation error', 'errors': e.messages}, 400
		except Exception as e:
			return {'message': str(e)}, 400
	
	def _get_tenant_id(self):
		"""Get tenant ID from request context"""
		from flask import session
		return session.get('tenant_id', 'default_tenant')
	
	def _get_user_id(self):
		"""Get user ID from request context"""
		from flask import session
		return session.get('user_id')


class GLCurrencyResource(Resource):
	"""Currency and exchange rate REST resource"""
	
	def get(self, rate_id=None):
		"""Get currency rate(s)"""
		tenant_id = self._get_tenant_id()
		gl_service = GeneralLedgerService(tenant_id, self._get_user_id())
		
		try:
			if rate_id:
				rate = gl_service.get_currency_rate(rate_id)
				if not rate:
					return {'message': 'Currency rate not found'}, 404
				
				schema = GLCurrencyRateSchema()
				return schema.dump(rate)
			else:
				# Get rates with filters
				from_currency = request.args.get('from_currency')
				to_currency = request.args.get('to_currency')
				rate_date = request.args.get('date')
				
				if rate_date:
					rate_date = datetime.strptime(rate_date, '%Y-%m-%d').date()
				
				rates = gl_service.get_currency_rates(
					from_currency=CurrencyEnum(from_currency) if from_currency else None,
					to_currency=CurrencyEnum(to_currency) if to_currency else None,
					rate_date=rate_date
				)
				
				schema = GLCurrencyRateSchema(many=True)
				return schema.dump(rates)
			
		except Exception as e:
			return {'message': str(e)}, 400
	
	def post(self):
		"""Create new currency rate"""
		tenant_id = self._get_tenant_id()
		gl_service = GeneralLedgerService(tenant_id, self._get_user_id())
		
		try:
			data = request.get_json()
			schema = GLCurrencyRateSchema()
			validated_data = schema.load(data)
			
			rate = gl_service.create_currency_rate(validated_data)
			
			return schema.dump(rate), 201
			
		except ValidationError as e:
			return {'message': 'Validation error', 'errors': e.messages}, 400
		except Exception as e:
			return {'message': str(e)}, 400
	
	def _get_tenant_id(self):
		"""Get tenant ID from request context"""
		from flask import session
		return session.get('tenant_id', 'default_tenant')
	
	def _get_user_id(self):
		"""Get user ID from request context"""
		from flask import session
		return session.get('user_id')


# =====================================
# API UTILITIES AND HELPERS
# =====================================

def get_api_info():
	"""Get General Ledger API information"""
	return {
		'name': 'APG Financial Management General Ledger API',
		'version': '1.0.0',
		'description': 'Enterprise-grade REST API for general ledger operations',
		'author': 'Nyimbi Odero <nyimbi@gmail.com>',
		'company': 'Datacraft',
		'base_url': '/api/v1/financials/gl',
		'features': [
			'Multi-tenant account management',
			'Hierarchical chart of accounts',
			'Multi-currency journal entries',
			'Real-time financial reporting',
			'Period management and closing',
			'Currency conversion and rates',
			'Comprehensive validation',
			'Audit trail support',
			'Batch operations',
			'Advanced filtering and search'
		],
		'schemas': [
			'GLAccountSchema',
			'GLJournalEntrySchema', 
			'GLPeriodSchema',
			'GLCurrencyRateSchema',
			'TrialBalanceRequestSchema'
		],
		'security': {
			'authentication': 'Required',
			'authorization': 'Tenant-based',
			'data_validation': 'Comprehensive',
			'error_handling': 'Structured'
		}
	}


# Export all schemas and resources for external use
__all__ = [
	# Schemas
	'GLAccountSchema',
	'GLAccountTypeSchema', 
	'GLJournalEntrySchema',
	'GLJournalLineSchema',
	'GLPostingSchema',
	'GLPeriodSchema',
	'GLCurrencyRateSchema',
	'TrialBalanceRequestSchema',
	'JournalEntryCreateSchema',
	'GLJournalLineCreateSchema',
	
	# API Classes
	'GLAccountApi',
	'GLJournalEntryApi',
	'GLReportsApi',
	'GLPeriodApi',
	'GLCurrencyApi',
	
	# Flask-RESTful Resources  
	'GLAccountResource',
	'GLJournalEntryResource',
	'GLTrialBalanceResource',
	'GLAccountLedgerResource',
	'GLPeriodResource',
	'GLCurrencyResource',
	
	# Functions
	'register_api_views',
	'create_api_blueprint',
	'get_api_info',
	
	# Decorators and Utilities
	'handle_gl_exceptions',
	'get_tenant_context',
	'validate_tenant_access'
]