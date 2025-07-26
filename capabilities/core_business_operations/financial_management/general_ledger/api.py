"""
General Ledger REST API

REST API endpoints for General Ledger functionality.
Provides programmatic access to GL operations.
"""

from flask import request, jsonify, Blueprint
from flask_restful import Api, Resource
from flask_appbuilder.api import BaseApi, expose
from flask_appbuilder.models.sqla.interface import SQLAInterface
from marshmallow import Schema, fields
from datetime import date, datetime
from typing import Dict, List, Any

from .models import (
	CFGLAccount, CFGLAccountType, CFGLPeriod, CFGLJournalEntry,
	CFGLJournalLine, CFGLPosting
)
from .service import GeneralLedgerService
from ...auth_rbac.models import db


# Marshmallow Schemas for API serialization

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
	currency_code = fields.String(default='USD')
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


# REST API Resources

class GLAccountApi(BaseApi):
	"""GL Account API endpoints"""
	
	resource_name = 'accounts'
	datamodel = SQLAInterface(CFGLAccount)
	
	list_columns = [
		'account_id', 'account_code', 'account_name', 'account_type.type_name',
		'current_balance', 'is_active', 'allow_posting'
	]
	
	show_columns = [
		'account_id', 'account_code', 'account_name', 'description',
		'account_type', 'parent_account', 'is_active', 'is_header',
		'allow_posting', 'current_balance', 'ytd_balance', 'opening_balance',
		'currency_code', 'level', 'path'
	]
	
	add_columns = [
		'account_code', 'account_name', 'description', 'account_type_id',
		'parent_account_id', 'is_active', 'is_header', 'allow_posting',
		'opening_balance', 'currency_code'
	]
	
	edit_columns = [
		'account_name', 'description', 'is_active', 'allow_posting', 'currency_code'
	]
	
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
		
		try:
			if as_of_date:
				as_of_date = datetime.strptime(as_of_date, '%Y-%m-%d').date()
			
			gl_service = GeneralLedgerService(tenant_id)
			account = gl_service.get_account(account_id)
			
			if not account:
				return self.response_404()
			
			balance = account.calculate_balance(as_of_date)
			
			return self.response(200, result={
				'account_id': account_id,
				'account_code': account.account_code,
				'account_name': account.account_name,
				'balance': float(balance),
				'as_of_date': as_of_date.isoformat() if as_of_date else date.today().isoformat()
			})
			
		except Exception as e:
			return self.response_400(message=str(e))


class GLJournalEntryApi(BaseApi):
	"""Journal Entry API endpoints"""
	
	resource_name = 'journal_entries'
	datamodel = SQLAInterface(CFGLJournalEntry)
	
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
			
			# Validate required fields
			required_fields = ['description', 'entry_date', 'posting_date', 'lines']
			for field in required_fields:
				if field not in data:
					return self.response_400(message=f"Missing required field: {field}")
			
			# Parse dates
			if isinstance(data['entry_date'], str):
				data['entry_date'] = datetime.strptime(data['entry_date'], '%Y-%m-%d').date()
			if isinstance(data['posting_date'], str):
				data['posting_date'] = datetime.strptime(data['posting_date'], '%Y-%m-%d').date()
			
			gl_service = GeneralLedgerService(tenant_id)
			journal = gl_service.create_journal_entry(data)
			
			schema = GLJournalEntrySchema()
			return self.response(201, result=schema.dump(journal))
			
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
			journal = CFGLJournalEntry.query.filter_by(
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
			as_of_date = request.args.get('as_of_date')
			account_type = request.args.get('account_type')
			
			if as_of_date:
				as_of_date = datetime.strptime(as_of_date, '%Y-%m-%d').date()
			
			gl_service = GeneralLedgerService(tenant_id)
			trial_balance = gl_service.generate_trial_balance(as_of_date, account_type)
			
			return self.response(200, result=trial_balance)
			
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


def register_api_views(appbuilder):
	"""Register GL API views with Flask-AppBuilder"""
	
	appbuilder.add_api(GLAccountApi)
	appbuilder.add_api(GLJournalEntryApi)
	appbuilder.add_api(GLReportsApi)


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
	
	return api_bp


# Flask-RESTful Resources (alternative approach)

class GLAccountResource(Resource):
	"""GL Account REST resource"""
	
	def get(self, account_id=None):
		"""Get account(s)"""
		tenant_id = self._get_tenant_id()
		gl_service = GeneralLedgerService(tenant_id)
		
		if account_id:
			account = gl_service.get_account(account_id)
			if not account:
				return {'message': 'Account not found'}, 404
			
			schema = GLAccountSchema()
			return schema.dump(account)
		else:
			accounts = gl_service.get_chart_of_accounts()
			schema = GLAccountSchema(many=True)
			return schema.dump(accounts)
	
	def post(self):
		"""Create new account"""
		tenant_id = self._get_tenant_id()
		gl_service = GeneralLedgerService(tenant_id)
		
		try:
			data = request.get_json()
			account = gl_service.create_account(data)
			
			schema = GLAccountSchema()
			return schema.dump(account), 201
			
		except Exception as e:
			return {'message': str(e)}, 400
	
	def _get_tenant_id(self):
		"""Get tenant ID from request context"""
		# TODO: Implement tenant resolution
		return "default_tenant"


class GLJournalEntryResource(Resource):
	"""Journal Entry REST resource"""
	
	def get(self, journal_id=None):
		"""Get journal entry(ies)"""
		tenant_id = self._get_tenant_id()
		gl_service = GeneralLedgerService(tenant_id)
		
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
			
			journals = gl_service.get_journal_entries(status=status, limit=limit)
			schema = GLJournalEntrySchema(many=True)
			return schema.dump(journals)
	
	def post(self):
		"""Create journal entry with lines"""
		tenant_id = self._get_tenant_id()
		gl_service = GeneralLedgerService(tenant_id)
		
		try:
			data = request.get_json()
			journal = gl_service.create_journal_entry(data)
			
			schema = GLJournalEntrySchema()
			return schema.dump(journal), 201
			
		except Exception as e:
			return {'message': str(e)}, 400
	
	def _get_tenant_id(self):
		"""Get tenant ID from request context"""
		# TODO: Implement tenant resolution
		return "default_tenant"


class GLTrialBalanceResource(Resource):
	"""Trial Balance REST resource"""
	
	def get(self):
		"""Generate trial balance"""
		tenant_id = self._get_tenant_id()
		gl_service = GeneralLedgerService(tenant_id)
		
		try:
			as_of_date = request.args.get('as_of_date')
			if as_of_date:
				as_of_date = datetime.strptime(as_of_date, '%Y-%m-%d').date()
			
			trial_balance = gl_service.generate_trial_balance(as_of_date)
			return trial_balance
			
		except Exception as e:
			return {'message': str(e)}, 400
	
	def _get_tenant_id(self):
		"""Get tenant ID from request context"""
		# TODO: Implement tenant resolution
		return "default_tenant"


class GLAccountLedgerResource(Resource):
	"""Account Ledger REST resource"""
	
	def get(self, account_id):
		"""Get account ledger"""
		tenant_id = self._get_tenant_id()
		gl_service = GeneralLedgerService(tenant_id)
		
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
		# TODO: Implement tenant resolution
		return "default_tenant"