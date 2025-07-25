"""
Accounts Receivable REST API

REST API endpoints for Accounts Receivable functionality.
Provides programmatic access to AR operations including customer management,
invoice processing, payment processing, collections, and reporting.
"""

from flask import request, jsonify, Blueprint
from flask_restful import Api, Resource
from flask_appbuilder.api import BaseApi, expose
from flask_appbuilder.models.sqla.interface import SQLAInterface
from marshmallow import Schema, fields, ValidationError
from datetime import date, datetime
from typing import Dict, List, Any
from decimal import Decimal

from .models import (
	CFARCustomer, CFARInvoice, CFARInvoiceLine, CFARPayment, CFARPaymentLine,
	CFARCreditMemo, CFARCreditMemoLine, CFARStatement, CFARCollection,
	CFARAging, CFARTaxCode, CFARRecurringBilling
)
from .service import AccountsReceivableService
from ...auth_rbac.models import db


# Marshmallow Schemas for API serialization

class ARCustomerSchema(Schema):
	"""Schema for AR Customer serialization"""
	customer_id = fields.String(dump_only=True)
	customer_number = fields.String(required=True)
	customer_name = fields.String(required=True)
	customer_type = fields.String(default='RETAIL')
	contact_name = fields.String(allow_none=True)
	email = fields.Email(allow_none=True)
	phone = fields.String(allow_none=True)
	fax = fields.String(allow_none=True)
	website = fields.String(allow_none=True)
	billing_address_line1 = fields.String(allow_none=True)
	billing_address_line2 = fields.String(allow_none=True)
	billing_city = fields.String(allow_none=True)
	billing_state_province = fields.String(allow_none=True)
	billing_postal_code = fields.String(allow_none=True)
	billing_country = fields.String(allow_none=True)
	shipping_address_line1 = fields.String(allow_none=True)
	shipping_address_line2 = fields.String(allow_none=True)
	shipping_city = fields.String(allow_none=True)
	shipping_state_province = fields.String(allow_none=True)
	shipping_postal_code = fields.String(allow_none=True)
	shipping_country = fields.String(allow_none=True)
	payment_terms_code = fields.String(default='NET_30')
	payment_method = fields.String(default='CHECK')
	currency_code = fields.String(default='USD')
	credit_limit = fields.Decimal(default=0.00)
	credit_hold = fields.Boolean(default=False)
	credit_rating = fields.String(allow_none=True)
	tax_id = fields.String(allow_none=True)
	tax_exempt = fields.Boolean(default=False)
	tax_exempt_number = fields.String(allow_none=True)
	default_tax_code = fields.String(allow_none=True)
	is_active = fields.Boolean(default=True)
	allow_backorders = fields.Boolean(default=True)
	require_po = fields.Boolean(default=False)
	print_statements = fields.Boolean(default=True)
	send_dunning_letters = fields.Boolean(default=True)
	sales_rep_id = fields.String(allow_none=True)
	territory_id = fields.String(allow_none=True)
	price_level = fields.String(default='STANDARD')
	current_balance = fields.Decimal(dump_only=True)
	ytd_sales = fields.Decimal(dump_only=True)
	last_payment_date = fields.Date(dump_only=True)
	last_payment_amount = fields.Decimal(dump_only=True)
	notes = fields.String(allow_none=True)
	created_on = fields.DateTime(dump_only=True)
	updated_on = fields.DateTime(dump_only=True)


class ARInvoiceLineSchema(Schema):
	"""Schema for AR Invoice Line serialization"""
	line_id = fields.String(dump_only=True)
	line_number = fields.Integer(required=True)
	description = fields.String(allow_none=True)
	item_code = fields.String(allow_none=True)
	item_description = fields.String(allow_none=True)
	item_type = fields.String(default='PRODUCT')
	quantity = fields.Decimal(default=1.0000)
	unit_price = fields.Decimal(default=0.0000)
	line_amount = fields.Decimal(default=0.00)
	unit_cost = fields.Decimal(default=0.0000)
	gl_account_id = fields.String(required=True)
	tax_code = fields.String(allow_none=True)
	tax_rate = fields.Decimal(default=0.00)
	tax_amount = fields.Decimal(default=0.00)
	is_tax_inclusive = fields.Boolean(default=False)
	discount_percentage = fields.Decimal(default=0.00)
	discount_amount = fields.Decimal(default=0.00)
	commission_rate = fields.Decimal(default=0.00)
	commission_amount = fields.Decimal(default=0.00)
	cost_center = fields.String(allow_none=True)
	department = fields.String(allow_none=True)
	project = fields.String(allow_none=True)


class ARInvoiceSchema(Schema):
	"""Schema for AR Invoice serialization"""
	invoice_id = fields.String(dump_only=True)
	invoice_number = fields.String(required=True)
	customer_id = fields.String(required=True)
	customer_name = fields.String(dump_only=True)
	invoice_date = fields.Date(required=True)
	due_date = fields.Date(required=True)
	shipped_date = fields.Date(allow_none=True)
	description = fields.String(allow_none=True)
	sales_order_id = fields.String(allow_none=True)
	customer_po_number = fields.String(allow_none=True)
	status = fields.String(dump_only=True)
	subtotal_amount = fields.Decimal(dump_only=True)
	tax_amount = fields.Decimal(dump_only=True)
	discount_amount = fields.Decimal(default=0.00)
	freight_amount = fields.Decimal(default=0.00)
	misc_amount = fields.Decimal(default=0.00)
	total_amount = fields.Decimal(dump_only=True)
	payment_terms_code = fields.String(allow_none=True)
	discount_terms = fields.String(allow_none=True)
	currency_code = fields.String(default='USD')
	exchange_rate = fields.Decimal(default=1.000000)
	payment_status = fields.String(dump_only=True)
	paid_amount = fields.Decimal(dump_only=True)
	outstanding_amount = fields.Decimal(dump_only=True)
	collection_status = fields.String(dump_only=True)
	dunning_level = fields.Integer(dump_only=True)
	sales_rep_id = fields.String(allow_none=True)
	territory_id = fields.String(allow_none=True)
	commission_rate = fields.Decimal(default=0.00)
	commission_amount = fields.Decimal(default=0.00)
	posted = fields.Boolean(dump_only=True)
	posted_by = fields.String(dump_only=True)
	posted_date = fields.DateTime(dump_only=True)
	notes = fields.String(allow_none=True)
	lines = fields.Nested(ARInvoiceLineSchema, many=True, exclude=['invoice_id'])
	created_on = fields.DateTime(dump_only=True)
	updated_on = fields.DateTime(dump_only=True)


class ARPaymentLineSchema(Schema):
	"""Schema for AR Payment Line serialization"""
	payment_line_id = fields.String(dump_only=True)
	line_number = fields.Integer(required=True)
	invoice_id = fields.String(allow_none=True)
	credit_memo_id = fields.String(allow_none=True)
	original_amount = fields.Decimal(default=0.00)
	payment_amount = fields.Decimal(required=True)
	discount_taken = fields.Decimal(default=0.00)
	remaining_amount = fields.Decimal(dump_only=True)
	discount_available = fields.Decimal(default=0.00)
	discount_date = fields.Date(allow_none=True)
	writeoff_amount = fields.Decimal(default=0.00)
	writeoff_reason = fields.String(allow_none=True)
	notes = fields.String(allow_none=True)


class ARPaymentSchema(Schema):
	"""Schema for AR Payment serialization"""
	payment_id = fields.String(dump_only=True)
	payment_number = fields.String(required=True)
	customer_id = fields.String(required=True)
	customer_name = fields.String(dump_only=True)
	payment_date = fields.Date(required=True)
	payment_method = fields.String(required=True)
	check_number = fields.String(allow_none=True)
	reference_number = fields.String(allow_none=True)
	bank_account_id = fields.String(allow_none=True)
	status = fields.String(dump_only=True)
	payment_amount = fields.Decimal(required=True)
	discount_taken = fields.Decimal(dump_only=True)
	unapplied_amount = fields.Decimal(dump_only=True)
	total_amount = fields.Decimal(dump_only=True)
	currency_code = fields.String(default='USD')
	exchange_rate = fields.Decimal(default=1.000000)
	posted = fields.Boolean(dump_only=True)
	posted_by = fields.String(dump_only=True)
	posted_date = fields.DateTime(dump_only=True)
	cleared = fields.Boolean(dump_only=True)
	cleared_date = fields.Date(dump_only=True)
	returned = fields.Boolean(dump_only=True)
	return_date = fields.Date(dump_only=True)
	return_reason = fields.String(dump_only=True)
	nsf_fee = fields.Decimal(dump_only=True)
	notes = fields.String(allow_none=True)
	applications = fields.Nested(ARPaymentLineSchema, many=True, exclude=['payment_id'])
	created_on = fields.DateTime(dump_only=True)
	updated_on = fields.DateTime(dump_only=True)


class ARCreditMemoLineSchema(Schema):
	"""Schema for AR Credit Memo Line serialization"""
	line_id = fields.String(dump_only=True)
	line_number = fields.Integer(required=True)
	description = fields.String(allow_none=True)
	item_code = fields.String(allow_none=True)
	item_description = fields.String(allow_none=True)
	original_invoice_line_id = fields.String(allow_none=True)
	quantity = fields.Decimal(default=1.0000)
	unit_price = fields.Decimal(default=0.0000)
	line_amount = fields.Decimal(default=0.00)
	gl_account_id = fields.String(required=True)
	tax_code = fields.String(allow_none=True)
	tax_rate = fields.Decimal(default=0.00)
	tax_amount = fields.Decimal(default=0.00)
	is_tax_inclusive = fields.Boolean(default=False)
	cost_center = fields.String(allow_none=True)
	department = fields.String(allow_none=True)
	project = fields.String(allow_none=True)


class ARCreditMemoSchema(Schema):
	"""Schema for AR Credit Memo serialization"""
	credit_memo_id = fields.String(dump_only=True)
	credit_memo_number = fields.String(required=True)
	customer_id = fields.String(required=True)
	customer_name = fields.String(dump_only=True)
	credit_date = fields.Date(required=True)
	reference_invoice_id = fields.String(allow_none=True)
	reason_code = fields.String(allow_none=True)
	description = fields.String(allow_none=True)
	status = fields.String(dump_only=True)
	subtotal_amount = fields.Decimal(dump_only=True)
	tax_amount = fields.Decimal(dump_only=True)
	total_amount = fields.Decimal(dump_only=True)
	applied_amount = fields.Decimal(dump_only=True)
	unapplied_amount = fields.Decimal(dump_only=True)
	currency_code = fields.String(default='USD')
	exchange_rate = fields.Decimal(default=1.000000)
	posted = fields.Boolean(dump_only=True)
	posted_by = fields.String(dump_only=True)
	posted_date = fields.DateTime(dump_only=True)
	return_authorization = fields.String(allow_none=True)
	received_date = fields.Date(allow_none=True)
	notes = fields.String(allow_none=True)
	lines = fields.Nested(ARCreditMemoLineSchema, many=True, exclude=['credit_memo_id'])
	created_on = fields.DateTime(dump_only=True)
	updated_on = fields.DateTime(dump_only=True)


class ARStatementSchema(Schema):
	"""Schema for AR Statement serialization"""
	statement_id = fields.String(dump_only=True)
	statement_number = fields.String(dump_only=True)
	customer_id = fields.String(required=True)
	customer_name = fields.String(dump_only=True)
	statement_date = fields.Date(required=True)
	statement_period_start = fields.Date(required=True)
	statement_period_end = fields.Date(required=True)
	statement_type = fields.String(default='MONTHLY')
	beginning_balance = fields.Decimal(dump_only=True)
	charges = fields.Decimal(dump_only=True)
	payments = fields.Decimal(dump_only=True)
	adjustments = fields.Decimal(dump_only=True)
	ending_balance = fields.Decimal(dump_only=True)
	current_amount = fields.Decimal(dump_only=True)
	days_31_60 = fields.Decimal(dump_only=True)
	days_61_90 = fields.Decimal(dump_only=True)
	days_91_120 = fields.Decimal(dump_only=True)
	over_120_days = fields.Decimal(dump_only=True)
	status = fields.String(dump_only=True)
	delivery_method = fields.String(default='PRINT')
	email_address = fields.String(allow_none=True)
	template_name = fields.String(allow_none=True)
	include_remittance_slip = fields.Boolean(default=True)
	message = fields.String(allow_none=True)
	created_on = fields.DateTime(dump_only=True)


class ARCollectionSchema(Schema):
	"""Schema for AR Collection serialization"""
	collection_id = fields.String(dump_only=True)
	customer_id = fields.String(required=True)
	customer_name = fields.String(dump_only=True)
	collection_date = fields.Date(required=True)
	collection_type = fields.String(required=True)
	collector_id = fields.String(required=True)
	dunning_level = fields.Integer(default=1)
	days_past_due = fields.Integer(default=0)
	amount_past_due = fields.Decimal(default=0.00)
	subject = fields.String(required=True)
	notes = fields.String(allow_none=True)
	outcome = fields.String(allow_none=True)
	follow_up_date = fields.Date(allow_none=True)
	follow_up_required = fields.Boolean(default=False)
	promised_amount = fields.Decimal(default=0.00)
	promised_date = fields.Date(allow_none=True)
	promise_kept = fields.Boolean(allow_none=True)
	status = fields.String(default='Open')
	document_path = fields.String(allow_none=True)
	related_invoice_ids = fields.List(fields.String(), allow_none=True)
	created_on = fields.DateTime(dump_only=True)


class ARRecurringBillingSchema(Schema):
	"""Schema for AR Recurring Billing serialization"""
	recurring_billing_id = fields.String(dump_only=True)
	billing_name = fields.String(required=True)
	customer_id = fields.String(required=True)
	customer_name = fields.String(dump_only=True)
	description = fields.String(allow_none=True)
	frequency = fields.String(required=True)
	start_date = fields.Date(required=True)
	end_date = fields.Date(allow_none=True)
	next_billing_date = fields.Date(required=True)
	billing_amount = fields.Decimal(required=True)
	tax_code = fields.String(allow_none=True)
	payment_terms_code = fields.String(allow_none=True)
	gl_account_id = fields.String(required=True)
	invoice_template = fields.String(allow_none=True)
	invoice_description_template = fields.String(allow_none=True)
	is_active = fields.Boolean(default=True)
	is_paused = fields.Boolean(default=False)
	pause_start_date = fields.Date(allow_none=True)
	pause_end_date = fields.Date(allow_none=True)
	last_processed_date = fields.Date(dump_only=True)
	invoices_generated = fields.Integer(dump_only=True)
	auto_process = fields.Boolean(default=True)
	advance_days = fields.Integer(default=0)
	notes = fields.String(allow_none=True)
	created_on = fields.DateTime(dump_only=True)


class ARAgingSchema(Schema):
	"""Schema for AR Aging serialization"""
	aging_id = fields.String(dump_only=True)
	customer_id = fields.String(dump_only=True)
	customer_name = fields.String(dump_only=True)
	as_of_date = fields.Date(dump_only=True)
	current_amount = fields.Decimal(dump_only=True)
	days_31_60 = fields.Decimal(dump_only=True)
	days_61_90 = fields.Decimal(dump_only=True)
	days_91_120 = fields.Decimal(dump_only=True)
	over_120_days = fields.Decimal(dump_only=True)
	total_outstanding = fields.Decimal(dump_only=True)
	collection_status = fields.String(dump_only=True)
	dunning_level = fields.Integer(dump_only=True)
	last_collection_date = fields.Date(dump_only=True)
	generated_date = fields.DateTime(dump_only=True)


class ARTaxCodeSchema(Schema):
	"""Schema for AR Tax Code serialization"""
	tax_code_id = fields.String(dump_only=True)
	code = fields.String(required=True)
	name = fields.String(required=True)
	description = fields.String(allow_none=True)
	tax_rate = fields.Decimal(required=True)
	is_compound = fields.Boolean(default=False)
	gl_account_id = fields.String(allow_none=True)
	is_active = fields.Boolean(default=True)
	effective_date = fields.Date(allow_none=True)
	expiration_date = fields.Date(allow_none=True)
	jurisdiction = fields.String(allow_none=True)
	authority = fields.String(allow_none=True)
	created_on = fields.DateTime(dump_only=True)


# API Resources

class ARCustomerApi(Resource):
	"""Customer API endpoints"""
	
	def __init__(self):
		self.tenant_id = self._get_tenant_id()
		self.ar_service = AccountsReceivableService(self.tenant_id)
		self.schema = ARCustomerSchema()
	
	def get(self, customer_id=None):
		"""Get customer(s)"""
		if customer_id:
			customer = self.ar_service.get_customer(customer_id)
			if not customer:
				return {'error': 'Customer not found'}, 404
			return self.schema.dump(customer)
		else:
			# List customers with optional filtering
			include_inactive = request.args.get('include_inactive', 'false').lower() == 'true'
			customers = self.ar_service.get_customers(include_inactive)
			return self.schema.dump(customers, many=True)
	
	def post(self):
		"""Create new customer"""
		try:
			customer_data = self.schema.load(request.json)
			customer = self.ar_service.create_customer(customer_data)
			return self.schema.dump(customer), 201
		except ValidationError as e:
			return {'error': 'Validation error', 'details': e.messages}, 400
		except Exception as e:
			return {'error': str(e)}, 500
	
	def put(self, customer_id):
		"""Update customer"""
		customer = self.ar_service.get_customer(customer_id)
		if not customer:
			return {'error': 'Customer not found'}, 404
		
		try:
			customer_data = self.schema.load(request.json, partial=True)
			
			# Update customer fields
			for key, value in customer_data.items():
				if hasattr(customer, key):
					setattr(customer, key, value)
			
			db.session.commit()
			return self.schema.dump(customer)
		except ValidationError as e:
			return {'error': 'Validation error', 'details': e.messages}, 400
		except Exception as e:
			return {'error': str(e)}, 500
	
	def _get_customer_summary(self, customer_id):
		"""Get customer summary with balance and recent activity"""
		customer = self.ar_service.get_customer(customer_id)
		if not customer:
			return {'error': 'Customer not found'}, 404
		
		# Get recent invoices and payments
		recent_invoices = self.ar_service.get_invoices_by_customer(customer_id)[:10]
		outstanding_balance = customer.get_outstanding_balance()
		
		return {
			'customer': self.schema.dump(customer),
			'outstanding_balance': float(outstanding_balance),
			'recent_invoices_count': len(recent_invoices),
			'ytd_sales': float(customer.ytd_sales),
			'credit_available': float(customer.credit_limit - customer.current_balance) if customer.credit_limit > 0 else None,
			'is_over_limit': customer.is_over_credit_limit(),
			'can_ship': customer.can_ship()
		}
	
	def _place_customer_on_hold(self, customer_id):
		"""Place customer on credit hold"""
		data = request.get_json() or {}
		reason = data.get('reason', 'API hold request')
		user_id = data.get('user_id', 'api_user')
		
		try:
			self.ar_service.place_customer_on_hold(customer_id, reason, user_id)
			return {'message': 'Customer placed on hold'}, 200
		except Exception as e:
			return {'error': str(e)}, 500
	
	def _release_customer_hold(self, customer_id):
		"""Release customer from credit hold"""
		data = request.get_json() or {}
		user_id = data.get('user_id', 'api_user')
		
		try:
			self.ar_service.release_customer_hold(customer_id, user_id)
			return {'message': 'Customer hold released'}, 200
		except Exception as e:
			return {'error': str(e)}, 500
	
	def _get_tenant_id(self):
		"""Get tenant ID from request context"""
		return request.headers.get('X-Tenant-ID', 'default_tenant')


class ARInvoiceApi(Resource):
	"""Invoice API endpoints"""
	
	def __init__(self):
		self.tenant_id = self._get_tenant_id()
		self.ar_service = AccountsReceivableService(self.tenant_id)
		self.schema = ARInvoiceSchema()
	
	def get(self, invoice_id=None):
		"""Get invoice(s)"""
		if invoice_id:
			invoice = self.ar_service.get_invoice(invoice_id)
			if not invoice:
				return {'error': 'Invoice not found'}, 404
			return self.schema.dump(invoice)
		else:
			# List invoices with optional filtering
			customer_id = request.args.get('customer_id')
			status = request.args.get('status')
			
			if customer_id:
				invoices = self.ar_service.get_invoices_by_customer(customer_id)
			else:
				# Would need to implement get_all_invoices method
				return {'error': 'Customer ID required for invoice listing'}, 400
			
			if status:
				invoices = [inv for inv in invoices if inv.status == status]
			
			return self.schema.dump(invoices, many=True)
	
	def post(self):
		"""Create new invoice"""
		try:
			invoice_data = self.schema.load(request.json)
			invoice = self.ar_service.create_invoice(invoice_data)
			return self.schema.dump(invoice), 201
		except ValidationError as e:
			return {'error': 'Validation error', 'details': e.messages}, 400
		except Exception as e:
			return {'error': str(e)}, 500
	
	def put(self, invoice_id):
		"""Update invoice"""
		invoice = self.ar_service.get_invoice(invoice_id)
		if not invoice:
			return {'error': 'Invoice not found'}, 404
		
		if invoice.posted:
			return {'error': 'Cannot modify posted invoice'}, 400
		
		try:
			invoice_data = self.schema.load(request.json, partial=True)
			
			# Update invoice fields
			for key, value in invoice_data.items():
				if hasattr(invoice, key) and key not in ['lines']:  # Handle lines separately
					setattr(invoice, key, value)
			
			# Recalculate totals
			invoice.calculate_totals()
			db.session.commit()
			
			return self.schema.dump(invoice)
		except ValidationError as e:
			return {'error': 'Validation error', 'details': e.messages}, 400
		except Exception as e:
			return {'error': str(e)}, 500
	
	def _post_invoice(self, invoice_id):
		"""Post invoice to GL"""
		data = request.get_json() or {}
		user_id = data.get('user_id', 'api_user')
		
		try:
			success = self.ar_service.post_invoice(invoice_id, user_id)
			if success:
				return {'message': 'Invoice posted successfully'}, 200
			else:
				return {'error': 'Invoice cannot be posted'}, 400
		except Exception as e:
			return {'error': str(e)}, 500
	
	def _get_tenant_id(self):
		"""Get tenant ID from request context"""
		return request.headers.get('X-Tenant-ID', 'default_tenant')


class ARPaymentApi(Resource):
	"""Payment API endpoints"""
	
	def __init__(self):
		self.tenant_id = self._get_tenant_id()
		self.ar_service = AccountsReceivableService(self.tenant_id)
		self.schema = ARPaymentSchema()
	
	def get(self, payment_id=None):
		"""Get payment(s)"""
		if payment_id:
			payment = self.ar_service.get_payment(payment_id)
			if not payment:
				return {'error': 'Payment not found'}, 404
			return self.schema.dump(payment)
		else:
			# Would need to implement get_payments method
			return {'error': 'Payment listing not implemented'}, 501
	
	def post(self):
		"""Create new payment"""
		try:
			payment_data = self.schema.load(request.json)
			payment = self.ar_service.create_payment(payment_data)
			return self.schema.dump(payment), 201
		except ValidationError as e:
			return {'error': 'Validation error', 'details': e.messages}, 400
		except Exception as e:
			return {'error': str(e)}, 500
	
	def _post_payment(self, payment_id):
		"""Post payment to GL"""
		data = request.get_json() or {}
		user_id = data.get('user_id', 'api_user')
		
		try:
			success = self.ar_service.post_payment(payment_id, user_id)
			if success:
				return {'message': 'Payment posted successfully'}, 200
			else:
				return {'error': 'Payment cannot be posted'}, 400
		except Exception as e:
			return {'error': str(e)}, 500
	
	def _auto_apply_payment(self, payment_id):
		"""Auto-apply payment to invoices"""
		try:
			success = self.ar_service.auto_apply_payment(payment_id)
			if success:
				return {'message': 'Payment auto-applied successfully'}, 200
			else:
				return {'error': 'Payment cannot be auto-applied'}, 400
		except Exception as e:
			return {'error': str(e)}, 500
	
	def _get_tenant_id(self):
		"""Get tenant ID from request context"""
		return request.headers.get('X-Tenant-ID', 'default_tenant')


class ARCreditMemoApi(Resource):
	"""Credit Memo API endpoints"""
	
	def __init__(self):
		self.tenant_id = self._get_tenant_id()
		self.ar_service = AccountsReceivableService(self.tenant_id)
		self.schema = ARCreditMemoSchema()
	
	def get(self, credit_memo_id=None):
		"""Get credit memo(s)"""
		if credit_memo_id:
			credit_memo = self.ar_service.get_credit_memo(credit_memo_id)
			if not credit_memo:
				return {'error': 'Credit memo not found'}, 404
			return self.schema.dump(credit_memo)
		else:
			# Would need to implement get_credit_memos method
			return {'error': 'Credit memo listing not implemented'}, 501
	
	def post(self):
		"""Create new credit memo"""
		try:
			credit_memo_data = self.schema.load(request.json)
			credit_memo = self.ar_service.create_credit_memo(credit_memo_data)
			return self.schema.dump(credit_memo), 201
		except ValidationError as e:
			return {'error': 'Validation error', 'details': e.messages}, 400
		except Exception as e:
			return {'error': str(e)}, 500
	
	def _post_credit_memo(self, credit_memo_id):
		"""Post credit memo to GL"""
		data = request.get_json() or {}
		user_id = data.get('user_id', 'api_user')
		
		try:
			success = self.ar_service.post_credit_memo(credit_memo_id, user_id)
			if success:
				return {'message': 'Credit memo posted successfully'}, 200
			else:
				return {'error': 'Credit memo cannot be posted'}, 400
		except Exception as e:
			return {'error': str(e)}, 500
	
	def _get_tenant_id(self):
		"""Get tenant ID from request context"""
		return request.headers.get('X-Tenant-ID', 'default_tenant')


class ARStatementApi(Resource):
	"""Statement API endpoints"""
	
	def __init__(self):
		self.tenant_id = self._get_tenant_id()
		self.ar_service = AccountsReceivableService(self.tenant_id)
		self.schema = ARStatementSchema()
	
	def post(self):
		"""Generate customer statement"""
		try:
			data = request.get_json()
			customer_id = data.get('customer_id')
			statement_date = datetime.strptime(data.get('statement_date', str(date.today())), '%Y-%m-%d').date()
			user_id = data.get('user_id', 'api_user')
			
			if not customer_id:
				return {'error': 'customer_id is required'}, 400
			
			statement = self.ar_service.generate_statement(customer_id, statement_date, user_id)
			return self.schema.dump(statement), 201
		except Exception as e:
			return {'error': str(e)}, 500
	
	def _generate_batch_statements(self):
		"""Generate statements for multiple customers"""
		try:
			data = request.get_json()
			customer_ids = data.get('customer_ids', [])
			statement_date = datetime.strptime(data.get('statement_date', str(date.today())), '%Y-%m-%d').date()
			user_id = data.get('user_id', 'api_user')
			
			statements = []
			for customer_id in customer_ids:
				try:
					statement = self.ar_service.generate_statement(customer_id, statement_date, user_id)
					statements.append(statement)
				except Exception as e:
					print(f"Error generating statement for customer {customer_id}: {e}")
			
			return {
				'generated_count': len(statements),
				'statements': self.schema.dump(statements, many=True)
			}, 201
		except Exception as e:
			return {'error': str(e)}, 500
	
	def _get_tenant_id(self):
		"""Get tenant ID from request context"""
		return request.headers.get('X-Tenant-ID', 'default_tenant')


class ARCollectionApi(Resource):
	"""Collection API endpoints"""
	
	def __init__(self):
		self.tenant_id = self._get_tenant_id()
		self.ar_service = AccountsReceivableService(self.tenant_id)
		self.schema = ARCollectionSchema()
	
	def get(self, collection_id=None):
		"""Get collection activity(ies)"""
		if collection_id:
			# Would need to implement get_collection method
			return {'error': 'Collection retrieval not implemented'}, 501
		else:
			# Would need to implement get_collections method
			return {'error': 'Collection listing not implemented'}, 501
	
	def post(self):
		"""Create collection activity"""
		try:
			collection_data = self.schema.load(request.json)
			collection = self.ar_service.create_collection_activity(collection_data)
			return self.schema.dump(collection), 201
		except ValidationError as e:
			return {'error': 'Validation error', 'details': e.messages}, 400
		except Exception as e:
			return {'error': str(e)}, 500
	
	def _get_customers_for_collections(self):
		"""Get customers that need collection attention"""
		try:
			days_past_due = int(request.args.get('days_past_due', 30))
			customers = self.ar_service.get_customers_for_collections(days_past_due)
			
			customer_schema = ARCustomerSchema()
			return {
				'customers_count': len(customers),
				'customers': customer_schema.dump(customers, many=True)
			}
		except Exception as e:
			return {'error': str(e)}, 500
	
	def _generate_dunning_letters(self):
		"""Generate dunning letters for customers"""
		try:
			data = request.get_json()
			customer_ids = data.get('customer_ids', [])
			user_id = data.get('user_id', 'api_user')
			
			dunning_activities = self.ar_service.generate_dunning_letters(customer_ids, user_id)
			
			return {
				'generated_count': len(dunning_activities),
				'activities': self.schema.dump(dunning_activities, many=True)
			}, 201
		except Exception as e:
			return {'error': str(e)}, 500
	
	def _get_tenant_id(self):
		"""Get tenant ID from request context"""
		return request.headers.get('X-Tenant-ID', 'default_tenant')


class ARRecurringBillingApi(Resource):
	"""Recurring Billing API endpoints"""
	
	def __init__(self):
		self.tenant_id = self._get_tenant_id()
		self.ar_service = AccountsReceivableService(self.tenant_id)
		self.schema = ARRecurringBillingSchema()
	
	def post(self):
		"""Process recurring billing"""
		try:
			data = request.get_json() or {}
			as_of_date = datetime.strptime(data.get('as_of_date', str(date.today())), '%Y-%m-%d').date()
			user_id = data.get('user_id', 'api_user')
			
			generated_invoices = self.ar_service.process_recurring_billing(as_of_date, user_id)
			
			invoice_schema = ARInvoiceSchema()
			return {
				'generated_count': len(generated_invoices),
				'invoices': invoice_schema.dump(generated_invoices, many=True)
			}, 201
		except Exception as e:
			return {'error': str(e)}, 500
	
	def _get_tenant_id(self):
		"""Get tenant ID from request context"""
		return request.headers.get('X-Tenant-ID', 'default_tenant')


class ARAgingApi(Resource):
	"""Aging Report API endpoints"""
	
	def __init__(self):
		self.tenant_id = self._get_tenant_id()
		self.ar_service = AccountsReceivableService(self.tenant_id)
		self.schema = ARAgingSchema()
	
	def get(self):
		"""Get aging report"""
		try:
			as_of_date = request.args.get('as_of_date')
			if as_of_date:
				as_of_date = datetime.strptime(as_of_date, '%Y-%m-%d').date()
			else:
				as_of_date = date.today()
			
			user_id = request.args.get('user_id', 'api_user')
			aging_records = self.ar_service.generate_aging_report(as_of_date, user_id)
			
			# Calculate totals
			totals = {
				'current': sum(a.current_amount for a in aging_records),
				'31_60': sum(a.days_31_60 for a in aging_records),
				'61_90': sum(a.days_61_90 for a in aging_records),
				'91_120': sum(a.days_91_120 for a in aging_records),
				'over_120': sum(a.over_120_days for a in aging_records),
				'total': sum(a.total_outstanding for a in aging_records)
			}
			
			return {
				'as_of_date': as_of_date.isoformat(),
				'record_count': len(aging_records),
				'aging_records': self.schema.dump(aging_records, many=True),
				'totals': {k: float(v) for k, v in totals.items()}
			}
		except Exception as e:
			return {'error': str(e)}, 500
	
	def _get_aging_summary(self):
		"""Get aging summary statistics"""
		try:
			as_of_date = request.args.get('as_of_date')
			if as_of_date:
				as_of_date = datetime.strptime(as_of_date, '%Y-%m-%d').date()
			else:
				as_of_date = date.today()
			
			user_id = request.args.get('user_id', 'api_user')
			aging_records = self.ar_service.generate_aging_report(as_of_date, user_id)
			
			# Calculate summary statistics
			total_customers = len(aging_records)
			current_customers = len([a for a in aging_records if a.current_amount > 0 and a.total_outstanding == a.current_amount])
			past_due_customers = len([a for a in aging_records if a.total_outstanding > a.current_amount])
			collections_customers = len([a for a in aging_records if a.collection_status == 'Collections'])
			
			total_outstanding = sum(a.total_outstanding for a in aging_records)
			total_current = sum(a.current_amount for a in aging_records)
			total_past_due = total_outstanding - total_current
			
			return {
				'as_of_date': as_of_date.isoformat(),
				'customer_summary': {
					'total_customers': total_customers,
					'current_customers': current_customers,
					'past_due_customers': past_due_customers,
					'collections_customers': collections_customers
				},
				'amount_summary': {
					'total_outstanding': float(total_outstanding),
					'total_current': float(total_current),
					'total_past_due': float(total_past_due),
					'past_due_percentage': float(total_past_due / total_outstanding * 100) if total_outstanding > 0 else 0.0
				}
			}
		except Exception as e:
			return {'error': str(e)}, 500
	
	def _get_tenant_id(self):
		"""Get tenant ID from request context"""
		return request.headers.get('X-Tenant-ID', 'default_tenant')


class ARTaxCodeApi(Resource):
	"""Tax Code API endpoints"""
	
	def __init__(self):
		self.tenant_id = self._get_tenant_id()
		self.schema = ARTaxCodeSchema()
	
	def get(self, tax_code_id=None):
		"""Get tax code(s)"""
		if tax_code_id:
			tax_code = CFARTaxCode.query.filter_by(
				tenant_id=self.tenant_id,
				tax_code_id=tax_code_id
			).first()
			if not tax_code:
				return {'error': 'Tax code not found'}, 404
			return self.schema.dump(tax_code)
		else:
			tax_codes = CFARTaxCode.query.filter_by(tenant_id=self.tenant_id).all()
			return self.schema.dump(tax_codes, many=True)
	
	def post(self):
		"""Create new tax code"""
		try:
			tax_code_data = self.schema.load(request.json)
			tax_code_data['tenant_id'] = self.tenant_id
			
			tax_code = CFARTaxCode(**tax_code_data)
			db.session.add(tax_code)
			db.session.commit()
			
			return self.schema.dump(tax_code), 201
		except ValidationError as e:
			return {'error': 'Validation error', 'details': e.messages}, 400
		except Exception as e:
			return {'error': str(e)}, 500
	
	def _get_tenant_id(self):
		"""Get tenant ID from request context"""
		return request.headers.get('X-Tenant-ID', 'default_tenant')


class ARDashboardApi(Resource):
	"""Dashboard API endpoints"""
	
	def __init__(self):
		self.tenant_id = self._get_tenant_id()
		self.ar_service = AccountsReceivableService(self.tenant_id)
	
	def _get_dashboard_summary(self):
		"""Get dashboard summary data"""
		try:
			customers = self.ar_service.get_customers()
			
			total_customers = len(customers)
			active_customers = len([c for c in customers if c.is_active])
			customers_on_hold = len([c for c in customers if c.credit_hold])
			total_outstanding = sum(c.current_balance for c in customers)
			
			past_due_customers = self.ar_service.get_customers_for_collections()
			
			return {
				'customer_metrics': {
					'total_customers': total_customers,
					'active_customers': active_customers,
					'customers_on_hold': customers_on_hold,
					'past_due_customers': len(past_due_customers)
				},
				'financial_metrics': {
					'total_outstanding': float(total_outstanding),
					'average_balance': float(total_outstanding / total_customers) if total_customers > 0 else 0.0,
					'collection_required': len([c for c in past_due_customers if c.current_balance > 1000])
				}
			}
		except Exception as e:
			return {'error': str(e)}, 500
	
	def _get_cash_flow_projection(self):
		"""Get cash flow projection based on invoice due dates"""
		try:
			# This would calculate expected cash inflows
			# For now, return sample data
			return {
				'projections': [
					{'period': 'Next 30 Days', 'amount': 125000.00, 'invoice_count': 45},
					{'period': 'Next 60 Days', 'amount': 75000.00, 'invoice_count': 28},
					{'period': 'Next 90 Days', 'amount': 45000.00, 'invoice_count': 18},
					{'period': 'Beyond 90 Days', 'amount': 25000.00, 'invoice_count': 12}
				],
				'total_projected': 270000.00,
				'confidence_level': 'Medium'
			}
		except Exception as e:
			return {'error': str(e)}, 500
	
	def _get_tenant_id(self):
		"""Get tenant ID from request context"""
		return request.headers.get('X-Tenant-ID', 'default_tenant')


def create_api_blueprint() -> Blueprint:
	"""Create API blueprint for Accounts Receivable"""
	
	api_bp = Blueprint('ar_api', __name__, url_prefix='/api/ar')
	api = Api(api_bp)
	
	# Customer endpoints
	api.add_resource(ARCustomerApi, '/customers', '/customers/<string:customer_id>')
	api.add_resource(ARCustomerApi, '/customers/<string:customer_id>/summary', 
					endpoint='customer_summary', 
					resource_class_kwargs={'method': '_get_customer_summary'})
	api.add_resource(ARCustomerApi, '/customers/<string:customer_id>/hold', 
					endpoint='customer_hold', 
					resource_class_kwargs={'method': '_place_customer_on_hold'})
	api.add_resource(ARCustomerApi, '/customers/<string:customer_id>/release_hold', 
					endpoint='customer_release_hold', 
					resource_class_kwargs={'method': '_release_customer_hold'})
	
	# Invoice endpoints
	api.add_resource(ARInvoiceApi, '/invoices', '/invoices/<string:invoice_id>')
	api.add_resource(ARInvoiceApi, '/invoices/<string:invoice_id>/post', 
					endpoint='post_invoice', 
					resource_class_kwargs={'method': '_post_invoice'})
	
	# Payment endpoints
	api.add_resource(ARPaymentApi, '/payments', '/payments/<string:payment_id>')
	api.add_resource(ARPaymentApi, '/payments/<string:payment_id>/post', 
					endpoint='post_payment', 
					resource_class_kwargs={'method': '_post_payment'})
	api.add_resource(ARPaymentApi, '/payments/<string:payment_id>/auto_apply', 
					endpoint='auto_apply_payment', 
					resource_class_kwargs={'method': '_auto_apply_payment'})
	
	# Credit Memo endpoints
	api.add_resource(ARCreditMemoApi, '/credit_memos', '/credit_memos/<string:credit_memo_id>')
	api.add_resource(ARCreditMemoApi, '/credit_memos/<string:credit_memo_id>/post', 
					endpoint='post_credit_memo', 
					resource_class_kwargs={'method': '_post_credit_memo'})
	
	# Statement endpoints
	api.add_resource(ARStatementApi, '/statements')
	api.add_resource(ARStatementApi, '/statements/batch', 
					endpoint='batch_statements', 
					resource_class_kwargs={'method': '_generate_batch_statements'})
	
	# Collection endpoints
	api.add_resource(ARCollectionApi, '/collections', '/collections/<string:collection_id>')
	api.add_resource(ARCollectionApi, '/collections/customers_for_collections', 
					endpoint='customers_for_collections', 
					resource_class_kwargs={'method': '_get_customers_for_collections'})
	api.add_resource(ARCollectionApi, '/collections/dunning_letters', 
					endpoint='dunning_letters', 
					resource_class_kwargs={'method': '_generate_dunning_letters'})
	
	# Recurring Billing endpoints
	api.add_resource(ARRecurringBillingApi, '/recurring_billing/process')
	
	# Aging endpoints
	api.add_resource(ARAgingApi, '/aging')
	api.add_resource(ARAgingApi, '/aging/summary', 
					endpoint='aging_summary', 
					resource_class_kwargs={'method': '_get_aging_summary'})
	
	# Tax Code endpoints
	api.add_resource(ARTaxCodeApi, '/tax_codes', '/tax_codes/<string:tax_code_id>')
	
	# Dashboard endpoints
	api.add_resource(ARDashboardApi, '/dashboard/summary', 
					endpoint='dashboard_summary', 
					resource_class_kwargs={'method': '_get_dashboard_summary'})
	api.add_resource(ARDashboardApi, '/dashboard/cash_flow', 
					endpoint='dashboard_cash_flow', 
					resource_class_kwargs={'method': '_get_cash_flow_projection'})
	
	return api_bp