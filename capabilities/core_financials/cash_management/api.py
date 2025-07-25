"""
Cash Management REST API

REST API endpoints for Cash Management functionality.
Provides programmatic access to cash management operations.
"""

from flask import request, jsonify, Blueprint
from flask_restful import Api, Resource
from flask_appbuilder.api import BaseApi, expose
from flask_appbuilder.models.sqla.interface import SQLAInterface
from marshmallow import Schema, fields, validate
from datetime import date, datetime, timedelta
from typing import Dict, List, Any
from decimal import Decimal

from .models import (
	CFCMBankAccount, CFCMBankTransaction, CFCMReconciliation, CFCMReconciliationItem,
	CFCMCashForecast, CFCMCashPosition, CFCMInvestment, CFCMCurrencyRate,
	CFCMCashTransfer, CFCMDeposit, CFCMCheckRegister
)
from .service import CashManagementService
from ...auth_rbac.models import db


# Marshmallow Schemas for API serialization

class BankAccountSchema(Schema):
	"""Schema for Bank Account serialization"""
	bank_account_id = fields.String(dump_only=True)
	account_number = fields.String(required=True)
	account_name = fields.String(required=True)
	account_type = fields.String(required=True, validate=validate.OneOf([
		'CHECKING', 'SAVINGS', 'MONEY_MARKET', 'INVESTMENT', 'PETTY_CASH', 'LOCKBOX'
	]))
	bank_name = fields.String(required=True)
	bank_code = fields.String(allow_none=True)
	routing_number = fields.String(allow_none=True)
	branch_name = fields.String(allow_none=True)
	currency_code = fields.String(default='USD')
	current_balance = fields.Decimal()
	available_balance = fields.Decimal()
	is_active = fields.Boolean(default=True)
	is_primary = fields.Boolean(default=False)
	requires_reconciliation = fields.Boolean(default=True)
	last_reconciliation_date = fields.Date(allow_none=True)
	interest_rate = fields.Decimal()
	minimum_balance = fields.Decimal()
	notes = fields.String(allow_none=True)


class BankTransactionSchema(Schema):
	"""Schema for Bank Transaction serialization"""
	transaction_id = fields.String(dump_only=True)
	bank_account_id = fields.String(required=True)
	transaction_date = fields.Date(required=True)
	description = fields.String(required=True)
	amount = fields.Decimal(required=True)
	is_debit = fields.Boolean(required=True)
	transaction_type = fields.String(required=True, validate=validate.OneOf([
		'DEPOSIT', 'WITHDRAWAL', 'TRANSFER_IN', 'TRANSFER_OUT', 'CHECK',
		'ACH_IN', 'ACH_OUT', 'WIRE_IN', 'WIRE_OUT', 'FEE', 'INTEREST', 'NSF'
	]))
	status = fields.String(validate=validate.OneOf(['Posted', 'Pending', 'Cleared', 'Returned']))
	is_reconciled = fields.Boolean()
	counterparty_name = fields.String(allow_none=True)
	check_number = fields.String(allow_none=True)
	bank_reference = fields.String(allow_none=True)
	memo = fields.String(allow_none=True)


class ReconciliationSchema(Schema):
	"""Schema for Bank Reconciliation serialization"""
	reconciliation_id = fields.String(dump_only=True)
	reconciliation_number = fields.String()
	reconciliation_name = fields.String(required=True)
	bank_account_id = fields.String(required=True)
	statement_date = fields.Date(required=True)
	statement_number = fields.String(allow_none=True)
	statement_beginning_balance = fields.Decimal(required=True)
	statement_ending_balance = fields.Decimal(required=True)
	book_balance = fields.Decimal()
	adjusted_book_balance = fields.Decimal()
	adjusted_bank_balance = fields.Decimal()
	variance_amount = fields.Decimal()
	status = fields.String(validate=validate.OneOf(['Draft', 'In Progress', 'Completed', 'Approved']))
	matched_transactions = fields.Integer()
	unmatched_bank_items = fields.Integer()
	unmatched_book_items = fields.Integer()
	reconciled_date = fields.DateTime(allow_none=True)
	notes = fields.String(allow_none=True)


class CashForecastSchema(Schema):
	"""Schema for Cash Forecast serialization"""
	forecast_id = fields.String(dump_only=True)
	forecast_name = fields.String(required=True)
	forecast_date = fields.Date(required=True)
	forecast_horizon = fields.Integer(default=90)
	category_code = fields.String(required=True)
	category_name = fields.String(required=True)
	category_type = fields.String(required=True, validate=validate.OneOf(['INFLOW', 'OUTFLOW']))
	forecast_amount = fields.Decimal(required=True)
	actual_amount = fields.Decimal()
	variance_amount = fields.Decimal()
	confidence_level = fields.Decimal(validate=validate.Range(0, 100))
	forecast_method = fields.String(validate=validate.OneOf(['HISTORICAL', 'BUDGET', 'SCHEDULE', 'ML']))
	status = fields.String(validate=validate.OneOf(['Active', 'Realized', 'Cancelled']))
	is_recurring = fields.Boolean()
	notes = fields.String(allow_none=True)


class CashPositionSchema(Schema):
	"""Schema for Cash Position serialization"""
	position_id = fields.String(dump_only=True)
	position_date = fields.Date(required=True)
	bank_account_id = fields.String(required=True)
	opening_balance = fields.Decimal()
	closing_balance = fields.Decimal()
	average_balance = fields.Decimal()
	total_inflows = fields.Decimal()
	total_outflows = fields.Decimal()
	net_change = fields.Decimal()
	transaction_count = fields.Integer()
	is_reconciled = fields.Boolean()


class InvestmentSchema(Schema):
	"""Schema for Investment serialization"""
	investment_id = fields.String(dump_only=True)
	investment_number = fields.String()
	investment_name = fields.String(required=True)
	investment_type = fields.String(required=True, validate=validate.OneOf([
		'CD', 'MONEY_MARKET', 'TREASURY', 'BOND'
	]))
	bank_account_id = fields.String(required=True)
	purchase_date = fields.Date(required=True)
	maturity_date = fields.Date(allow_none=True)
	purchase_amount = fields.Decimal(required=True)
	current_value = fields.Decimal()
	interest_rate = fields.Decimal()
	status = fields.String(validate=validate.OneOf(['Active', 'Matured', 'Sold', 'Closed']))
	is_liquid = fields.Boolean()
	auto_rollover = fields.Boolean()
	notes = fields.String(allow_none=True)


class CashTransferSchema(Schema):
	"""Schema for Cash Transfer serialization"""
	transfer_id = fields.String(dump_only=True)
	transfer_number = fields.String()
	description = fields.String(required=True)
	from_account_id = fields.String(required=True)
	to_account_id = fields.String(required=True)
	transfer_date = fields.Date(required=True)
	transfer_amount = fields.Decimal(required=True)
	transfer_fee = fields.Decimal()
	total_amount = fields.Decimal()
	transfer_method = fields.String(required=True, validate=validate.OneOf([
		'WIRE', 'ACH', 'INTERNAL', 'CHECK'
	]))
	status = fields.String(validate=validate.OneOf([
		'Draft', 'Pending', 'Approved', 'Submitted', 'Completed', 'Failed', 'Cancelled'
	]))
	approved = fields.Boolean()
	submitted = fields.Boolean()
	completed = fields.Boolean()
	notes = fields.String(allow_none=True)


class CheckRegisterSchema(Schema):
	"""Schema for Check Register serialization"""
	check_id = fields.String(dump_only=True)
	check_number = fields.String(required=True)
	bank_account_id = fields.String(required=True)
	check_date = fields.Date(required=True)
	payee_name = fields.String(required=True)
	check_amount = fields.Decimal(required=True)
	status = fields.String(validate=validate.OneOf([
		'Issued', 'Outstanding', 'Cleared', 'Voided', 'Stopped'
	]))
	is_cleared = fields.Boolean()
	is_voided = fields.Boolean()
	stop_payment = fields.Boolean()
	days_outstanding = fields.Integer()
	description = fields.String(allow_none=True)
	memo = fields.String(allow_none=True)


# REST API Resources

class BankAccountListAPI(Resource):
	"""Bank Account List API"""
	
	def get(self):
		"""Get list of bank accounts"""
		try:
			tenant_id = self._get_tenant_id()
			service = CashManagementService(db.session)
			
			# Get query parameters
			is_active = request.args.get('is_active', 'true').lower() == 'true'
			account_type = request.args.get('account_type')
			
			# Build query
			query = CFCMBankAccount.query.filter_by(tenant_id=tenant_id)
			
			if is_active:
				query = query.filter_by(is_active=True)
			
			if account_type:
				query = query.filter_by(account_type=account_type)
			
			accounts = query.all()
			
			# Serialize
			schema = BankAccountSchema(many=True)
			result = schema.dump(accounts)
			
			return {
				'success': True,
				'data': result,
				'count': len(result)
			}, 200
			
		except Exception as e:
			return {
				'success': False,
				'error': str(e)
			}, 500
	
	def post(self):
		"""Create new bank account"""
		try:
			tenant_id = self._get_tenant_id()
			service = CashManagementService(db.session)
			
			# Validate input
			schema = BankAccountSchema()
			data = schema.load(request.json)
			
			# Create account
			account = service.create_bank_account(tenant_id, data)
			
			# Serialize response
			result = schema.dump(account)
			
			return {
				'success': True,
				'data': result,
				'message': 'Bank account created successfully'
			}, 201
			
		except Exception as e:
			return {
				'success': False,
				'error': str(e)
			}, 400
	
	def _get_tenant_id(self):
		"""Get tenant ID from request context"""
		# This would integrate with your actual tenant resolution logic
		return request.headers.get('X-Tenant-ID', 'default_tenant')


class BankAccountAPI(Resource):
	"""Individual Bank Account API"""
	
	def get(self, account_id):
		"""Get bank account details"""
		try:
			tenant_id = self._get_tenant_id()
			
			account = CFCMBankAccount.query.filter_by(
				bank_account_id=account_id,
				tenant_id=tenant_id
			).first()
			
			if not account:
				return {
					'success': False,
					'error': 'Bank account not found'
				}, 404
			
			# Get additional data
			include_balance = request.args.get('include_balance', 'true').lower() == 'true'
			include_transactions = request.args.get('include_transactions', 'false').lower() == 'true'
			
			schema = BankAccountSchema()
			result = schema.dump(account)
			
			if include_balance:
				result['available_balance'] = float(account.get_available_balance())
				result['is_overdrawn'] = account.is_overdrawn()
			
			if include_transactions:
				service = CashManagementService(db.session)
				transactions = service.get_transaction_history(account_id, limit=10)
				result['recent_transactions'] = transactions
			
			return {
				'success': True,
				'data': result
			}, 200
			
		except Exception as e:
			return {
				'success': False,
				'error': str(e)
			}, 500
	
	def put(self, account_id):
		"""Update bank account"""
		try:
			tenant_id = self._get_tenant_id()
			
			account = CFCMBankAccount.query.filter_by(
				bank_account_id=account_id,
				tenant_id=tenant_id
			).first()
			
			if not account:
				return {
					'success': False,
					'error': 'Bank account not found'
				}, 404
			
			# Validate and update
			schema = BankAccountSchema(partial=True)
			data = schema.load(request.json)
			
			for key, value in data.items():
				if hasattr(account, key):
					setattr(account, key, value)
			
			db.session.commit()
			
			result = schema.dump(account)
			
			return {
				'success': True,
				'data': result,
				'message': 'Bank account updated successfully'
			}, 200
			
		except Exception as e:
			db.session.rollback()
			return {
				'success': False,
				'error': str(e)
			}, 400
	
	def _get_tenant_id(self):
		"""Get tenant ID from request context"""
		return request.headers.get('X-Tenant-ID', 'default_tenant')


class BankTransactionListAPI(Resource):
	"""Bank Transaction List API"""
	
	def get(self):
		"""Get list of bank transactions"""
		try:
			tenant_id = self._get_tenant_id()
			
			# Get query parameters
			account_id = request.args.get('account_id')
			start_date = request.args.get('start_date')
			end_date = request.args.get('end_date')
			transaction_type = request.args.get('transaction_type')
			is_reconciled = request.args.get('is_reconciled')
			limit = int(request.args.get('limit', 100))
			offset = int(request.args.get('offset', 0))
			
			# Build query
			query = CFCMBankTransaction.query.filter_by(tenant_id=tenant_id)
			
			if account_id:
				query = query.filter_by(bank_account_id=account_id)
			
			if start_date:
				query = query.filter(CFCMBankTransaction.transaction_date >= start_date)
			
			if end_date:
				query = query.filter(CFCMBankTransaction.transaction_date <= end_date)
			
			if transaction_type:
				query = query.filter_by(transaction_type=transaction_type)
			
			if is_reconciled is not None:
				reconciled = is_reconciled.lower() == 'true'
				query = query.filter_by(is_reconciled=reconciled)
			
			# Apply pagination
			total_count = query.count()
			transactions = query.order_by(CFCMBankTransaction.transaction_date.desc()).offset(offset).limit(limit).all()
			
			# Serialize
			schema = BankTransactionSchema(many=True)
			result = schema.dump(transactions)
			
			return {
				'success': True,
				'data': result,
				'pagination': {
					'total_count': total_count,
					'limit': limit,
					'offset': offset,
					'has_more': offset + limit < total_count
				}
			}, 200
			
		except Exception as e:
			return {
				'success': False,
				'error': str(e)
			}, 500
	
	def post(self):
		"""Create new bank transaction or import multiple transactions"""
		try:
			tenant_id = self._get_tenant_id()
			
			# Check if it's a bulk import
			if isinstance(request.json, list):
				return self._import_transactions(tenant_id, request.json)
			else:
				return self._create_transaction(tenant_id, request.json)
			
		except Exception as e:
			return {
				'success': False,
				'error': str(e)
			}, 400
	
	def _create_transaction(self, tenant_id, data):
		"""Create single transaction"""
		schema = BankTransactionSchema()
		validated_data = schema.load(data)
		
		transaction = CFCMBankTransaction(
			tenant_id=tenant_id,
			**validated_data
		)
		
		# Update account balance
		account = CFCMBankAccount.query.get(transaction.bank_account_id)
		if account:
			account.update_balance(transaction.amount, transaction.transaction_type)
		
		db.session.add(transaction)
		db.session.commit()
		
		result = schema.dump(transaction)
		
		return {
			'success': True,
			'data': result,
			'message': 'Transaction created successfully'
		}, 201
	
	def _import_transactions(self, tenant_id, transactions_data):
		"""Import multiple transactions"""
		service = CashManagementService(db.session)
		
		# Validate account_id is provided
		account_id = request.args.get('account_id')
		if not account_id:
			return {
				'success': False,
				'error': 'account_id parameter required for bulk import'
			}, 400
		
		# Import transactions
		result = service.import_bank_transactions(account_id, transactions_data)
		
		return {
			'success': True,
			'data': result,
			'message': f"Imported {result['imported_count']} transactions"
		}, 200
	
	def _get_tenant_id(self):
		"""Get tenant ID from request context"""
		return request.headers.get('X-Tenant-ID', 'default_tenant')


class ReconciliationListAPI(Resource):
	"""Bank Reconciliation List API"""
	
	def get(self):
		"""Get list of reconciliations"""
		try:
			tenant_id = self._get_tenant_id()
			
			# Get query parameters
			account_id = request.args.get('account_id')
			status = request.args.get('status')
			start_date = request.args.get('start_date')
			end_date = request.args.get('end_date')
			
			# Build query
			query = CFCMReconciliation.query.filter_by(tenant_id=tenant_id)
			
			if account_id:
				query = query.filter_by(bank_account_id=account_id)
			
			if status:
				query = query.filter_by(status=status)
			
			if start_date:
				query = query.filter(CFCMReconciliation.statement_date >= start_date)
			
			if end_date:
				query = query.filter(CFCMReconciliation.statement_date <= end_date)
			
			reconciliations = query.order_by(CFCMReconciliation.statement_date.desc()).all()
			
			# Serialize
			schema = ReconciliationSchema(many=True)
			result = schema.dump(reconciliations)
			
			return {
				'success': True,
				'data': result,
				'count': len(result)
			}, 200
			
		except Exception as e:
			return {
				'success': False,
				'error': str(e)
			}, 500
	
	def post(self):
		"""Create new reconciliation"""
		try:
			tenant_id = self._get_tenant_id()
			service = CashManagementService(db.session)
			
			# Validate input
			schema = ReconciliationSchema()
			data = schema.load(request.json)
			
			# Create reconciliation
			reconciliation = service.create_reconciliation(data['bank_account_id'], data)
			
			# Serialize response
			result = schema.dump(reconciliation)
			
			return {
				'success': True,
				'data': result,
				'message': 'Reconciliation created successfully'
			}, 201
			
		except Exception as e:
			return {
				'success': False,
				'error': str(e)
			}, 400
	
	def _get_tenant_id(self):
		"""Get tenant ID from request context"""
		return request.headers.get('X-Tenant-ID', 'default_tenant')


class ReconciliationAPI(Resource):
	"""Individual Reconciliation API"""
	
	def get(self, reconciliation_id):
		"""Get reconciliation details"""
		try:
			tenant_id = self._get_tenant_id()
			
			reconciliation = CFCMReconciliation.query.filter_by(
				reconciliation_id=reconciliation_id,
				tenant_id=tenant_id
			).first()
			
			if not reconciliation:
				return {
					'success': False,
					'error': 'Reconciliation not found'
				}, 404
			
			schema = ReconciliationSchema()
			result = schema.dump(reconciliation)
			
			# Include reconciliation items if requested
			include_items = request.args.get('include_items', 'false').lower() == 'true'
			if include_items:
				items = []
				for item in reconciliation.reconciliation_items:
					items.append({
						'item_id': item.item_id,
						'item_type': item.item_type,
						'description': item.description,
						'amount': float(item.amount),
						'is_matched': item.is_matched,
						'match_confidence': float(item.match_confidence) if item.match_confidence else None
					})
				result['reconciliation_items'] = items
			
			return {
				'success': True,
				'data': result
			}, 200
			
		except Exception as e:
			return {
				'success': False,
				'error': str(e)
			}, 500
	
	def post(self, reconciliation_id):
		"""Perform reconciliation actions"""
		try:
			tenant_id = self._get_tenant_id()
			service = CashManagementService(db.session)
			
			action = request.json.get('action')
			user_id = request.json.get('user_id', 'api_user')
			
			if action == 'auto_match':
				result = service.perform_auto_matching(reconciliation_id)
				return {
					'success': True,
					'data': result,
					'message': f"Auto-matched {result['matched_count']} transactions"
				}, 200
			
			elif action == 'complete':
				success = service.complete_reconciliation(reconciliation_id, user_id)
				if success:
					return {
						'success': True,
						'message': 'Reconciliation completed successfully'
					}, 200
				else:
					return {
						'success': False,
						'error': 'Reconciliation cannot be completed'
					}, 400
			
			else:
				return {
					'success': False,
					'error': 'Invalid action'
				}, 400
			
		except Exception as e:
			return {
				'success': False,
				'error': str(e)
			}, 500
	
	def _get_tenant_id(self):
		"""Get tenant ID from request context"""
		return request.headers.get('X-Tenant-ID', 'default_tenant')


class CashForecastListAPI(Resource):
	"""Cash Forecast List API"""
	
	def get(self):
		"""Get cash forecasts"""
		try:
			tenant_id = self._get_tenant_id()
			service = CashManagementService(db.session)
			
			# Get parameters
			horizon_days = int(request.args.get('horizon_days', 90))
			categories = request.args.getlist('categories')
			
			# Generate or get existing forecast
			if request.args.get('generate', 'false').lower() == 'true':
				forecast_config = {
					'horizon_days': horizon_days,
					'categories': categories,
					'method': request.args.get('method', 'HISTORICAL'),
					'replace_existing': True
				}
				
				result = service.generate_cash_forecast(tenant_id, forecast_config)
				
				return {
					'success': True,
					'data': result,
					'message': f"Generated {result['summary']['forecasts_created']} forecast items"
				}, 200
			
			else:
				# Get existing forecasts
				start_date = request.args.get('start_date', date.today().isoformat())
				end_date = request.args.get('end_date', 
					(date.today() + timedelta(days=horizon_days)).isoformat())
				
				query = CFCMCashForecast.query.filter_by(tenant_id=tenant_id)
				query = query.filter(CFCMCashForecast.forecast_date >= start_date)
				query = query.filter(CFCMCashForecast.forecast_date <= end_date)
				
				if categories:
					query = query.filter(CFCMCashForecast.category_code.in_(categories))
				
				forecasts = query.order_by(CFCMCashForecast.forecast_date).all()
				
				schema = CashForecastSchema(many=True)
				result = schema.dump(forecasts)
				
				return {
					'success': True,
					'data': result,
					'count': len(result)
				}, 200
			
		except Exception as e:
			return {
				'success': False,
				'error': str(e)
			}, 500
	
	def _get_tenant_id(self):
		"""Get tenant ID from request context"""
		return request.headers.get('X-Tenant-ID', 'default_tenant')


class CashPositionListAPI(Resource):
	"""Cash Position List API"""
	
	def get(self):
		"""Get cash positions"""
		try:
			tenant_id = self._get_tenant_id()
			service = CashManagementService(db.session)
			
			# Get parameters
			as_of_date = request.args.get('as_of_date')
			if as_of_date:
				as_of_date = datetime.strptime(as_of_date, '%Y-%m-%d').date()
			
			# Get summary or detailed positions
			if request.args.get('summary', 'false').lower() == 'true':
				summary = service.get_cash_position_summary(tenant_id, as_of_date)
				
				return {
					'success': True,
					'data': summary
				}, 200
			
			else:
				start_date = request.args.get('start_date')
				end_date = request.args.get('end_date')
				account_id = request.args.get('account_id')
				
				query = CFCMCashPosition.query.filter_by(tenant_id=tenant_id)
				
				if start_date:
					query = query.filter(CFCMCashPosition.position_date >= start_date)
				
				if end_date:
					query = query.filter(CFCMCashPosition.position_date <= end_date)
				
				if account_id:
					query = query.filter_by(bank_account_id=account_id)
				
				if as_of_date:
					query = query.filter_by(position_date=as_of_date)
				
				positions = query.order_by(CFCMCashPosition.position_date.desc()).all()
				
				schema = CashPositionSchema(many=True)
				result = schema.dump(positions)
				
				return {
					'success': True,
					'data': result,
					'count': len(result)
				}, 200
			
		except Exception as e:
			return {
				'success': False,
				'error': str(e)
			}, 500
	
	def _get_tenant_id(self):
		"""Get tenant ID from request context"""
		return request.headers.get('X-Tenant-ID', 'default_tenant')


class InvestmentListAPI(Resource):
	"""Investment List API"""
	
	def get(self):
		"""Get investments"""
		try:
			tenant_id = self._get_tenant_id()
			service = CashManagementService(db.session)
			
			# Check for maturing investments
			if request.args.get('maturing', 'false').lower() == 'true':
				days_ahead = int(request.args.get('days_ahead', 30))
				investments = service.get_maturing_investments(tenant_id, days_ahead)
				
				return {
					'success': True,
					'data': investments,
					'count': len(investments)
				}, 200
			
			else:
				# Get all investments
				status = request.args.get('status', 'Active')
				
				query = CFCMInvestment.query.filter_by(tenant_id=tenant_id)
				if status:
					query = query.filter_by(status=status)
				
				investments = query.order_by(CFCMInvestment.maturity_date).all()
				
				schema = InvestmentSchema(many=True)
				result = schema.dump(investments)
				
				return {
					'success': True,
					'data': result,
					'count': len(result)
				}, 200
			
		except Exception as e:
			return {
				'success': False,
				'error': str(e)
			}, 500
	
	def post(self):
		"""Create new investment"""
		try:
			tenant_id = self._get_tenant_id()
			service = CashManagementService(db.session)
			
			# Validate input
			schema = InvestmentSchema()
			data = schema.load(request.json)
			
			# Create investment
			investment = service.create_investment(tenant_id, data)
			
			# Serialize response
			result = schema.dump(investment)
			
			return {
				'success': True,
				'data': result,
				'message': 'Investment created successfully'
			}, 201
			
		except Exception as e:
			return {
				'success': False,
				'error': str(e)
			}, 400
	
	def _get_tenant_id(self):
		"""Get tenant ID from request context"""
		return request.headers.get('X-Tenant-ID', 'default_tenant')


class CashTransferListAPI(Resource):
	"""Cash Transfer List API"""
	
	def get(self):
		"""Get cash transfers"""
		try:
			tenant_id = self._get_tenant_id()
			
			# Get query parameters
			status = request.args.get('status')
			from_account_id = request.args.get('from_account_id')
			to_account_id = request.args.get('to_account_id')
			start_date = request.args.get('start_date')
			end_date = request.args.get('end_date')
			
			# Build query
			query = CFCMCashTransfer.query.filter_by(tenant_id=tenant_id)
			
			if status:
				query = query.filter_by(status=status)
			
			if from_account_id:
				query = query.filter_by(from_account_id=from_account_id)
			
			if to_account_id:
				query = query.filter_by(to_account_id=to_account_id)
			
			if start_date:
				query = query.filter(CFCMCashTransfer.transfer_date >= start_date)
			
			if end_date:
				query = query.filter(CFCMCashTransfer.transfer_date <= end_date)
			
			transfers = query.order_by(CFCMCashTransfer.transfer_date.desc()).all()
			
			# Serialize
			schema = CashTransferSchema(many=True)
			result = schema.dump(transfers)
			
			return {
				'success': True,
				'data': result,
				'count': len(result)
			}, 200
			
		except Exception as e:
			return {
				'success': False,
				'error': str(e)
			}, 500
	
	def post(self):
		"""Create new cash transfer"""
		try:
			tenant_id = self._get_tenant_id()
			service = CashManagementService(db.session)
			
			# Validate input
			schema = CashTransferSchema()
			data = schema.load(request.json)
			
			# Create transfer
			transfer = service.create_cash_transfer(tenant_id, data)
			
			# Serialize response
			result = schema.dump(transfer)
			
			return {
				'success': True,
				'data': result,
				'message': 'Cash transfer created successfully'
			}, 201
			
		except Exception as e:
			return {
				'success': False,
				'error': str(e)
			}, 400
	
	def _get_tenant_id(self):
		"""Get tenant ID from request context"""
		return request.headers.get('X-Tenant-ID', 'default_tenant')


class CheckRegisterListAPI(Resource):
	"""Check Register List API"""
	
	def get(self):
		"""Get check register entries"""
		try:
			tenant_id = self._get_tenant_id()
			
			# Get query parameters
			account_id = request.args.get('account_id')
			status = request.args.get('status')
			outstanding_only = request.args.get('outstanding_only', 'false').lower() == 'true'
			start_date = request.args.get('start_date')
			end_date = request.args.get('end_date')
			
			# Build query
			query = CFCMCheckRegister.query.filter_by(tenant_id=tenant_id)
			
			if account_id:
				query = query.filter_by(bank_account_id=account_id)
			
			if status:
				query = query.filter_by(status=status)
			
			if outstanding_only:
				query = query.filter_by(is_cleared=False, is_voided=False)
			
			if start_date:
				query = query.filter(CFCMCheckRegister.check_date >= start_date)
			
			if end_date:
				query = query.filter(CFCMCheckRegister.check_date <= end_date)
			
			checks = query.order_by(CFCMCheckRegister.check_date.desc()).all()
			
			# Calculate days outstanding for each check
			for check in checks:
				check.calculate_days_outstanding()
			
			# Serialize
			schema = CheckRegisterSchema(many=True)
			result = schema.dump(checks)
			
			return {
				'success': True,
				'data': result,
				'count': len(result)
			}, 200
			
		except Exception as e:
			return {
				'success': False,
				'error': str(e)
			}, 500
	
	def _get_tenant_id(self):
		"""Get tenant ID from request context"""
		return request.headers.get('X-Tenant-ID', 'default_tenant')


class CashFlowReportAPI(Resource):
	"""Cash Flow Report API"""
	
	def get(self):
		"""Generate cash flow report"""
		try:
			tenant_id = self._get_tenant_id()
			service = CashManagementService(db.session)
			
			# Get parameters
			start_date_str = request.args.get('start_date')
			end_date_str = request.args.get('end_date')
			account_ids = request.args.getlist('account_ids')
			
			# Default date range if not provided
			if not start_date_str:
				start_date = date.today() - timedelta(days=30)
			else:
				start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
			
			if not end_date_str:
				end_date = date.today()
			else:
				end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
			
			# Generate report
			report = service.generate_cash_flow_report(
				tenant_id, start_date, end_date, account_ids or None
			)
			
			return {
				'success': True,
				'data': report
			}, 200
			
		except Exception as e:
			return {
				'success': False,
				'error': str(e)
			}, 500
	
	def _get_tenant_id(self):
		"""Get tenant ID from request context"""
		return request.headers.get('X-Tenant-ID', 'default_tenant')


class DashboardAPI(Resource):
	"""Cash Management Dashboard API"""
	
	def get(self):
		"""Get dashboard data"""
		try:
			tenant_id = self._get_tenant_id()
			service = CashManagementService(db.session)
			
			# Get all dashboard data
			account_summary = service.get_bank_account_summary(tenant_id)
			cash_position = service.get_cash_position_summary(tenant_id)
			reconciliation_status = service.get_reconciliation_status(tenant_id)
			maturing_investments = service.get_maturing_investments(tenant_id, 30)
			
			# Get cash flow trend (last 7 days)
			end_date = date.today()
			start_date = end_date - timedelta(days=7)
			cash_flow_report = service.generate_cash_flow_report(tenant_id, start_date, end_date)
			
			dashboard_data = {
				'account_summary': account_summary,
				'cash_position': cash_position,
				'reconciliation_status': reconciliation_status,
				'maturing_investments': maturing_investments,
				'cash_flow_trend': cash_flow_report['daily_positions']
			}
			
			return {
				'success': True,
				'data': dashboard_data
			}, 200
			
		except Exception as e:
			return {
				'success': False,
				'error': str(e)
			}, 500
	
	def _get_tenant_id(self):
		"""Get tenant ID from request context"""
		return request.headers.get('X-Tenant-ID', 'default_tenant')