"""
Accounts Payable REST API

REST API endpoints for Accounts Payable functionality.
Provides programmatic access to AP operations including vendor management,
invoice processing, payment processing, and reporting.
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
	CFAPVendor, CFAPInvoice, CFAPInvoiceLine, CFAPPayment, CFAPPaymentLine,
	CFAPExpenseReport, CFAPExpenseLine, CFAPPurchaseOrder, CFAPTaxCode, CFAPAging
)
from .service import AccountsPayableService
from ...auth_rbac.models import db


# Marshmallow Schemas for API serialization

class APVendorSchema(Schema):
	"""Schema for AP Vendor serialization"""
	vendor_id = fields.String(dump_only=True)
	vendor_number = fields.String(required=True)
	vendor_name = fields.String(required=True)
	vendor_type = fields.String(default='SUPPLIER')
	contact_name = fields.String(allow_none=True)
	email = fields.Email(allow_none=True)
	phone = fields.String(allow_none=True)
	fax = fields.String(allow_none=True)
	website = fields.String(allow_none=True)
	address_line1 = fields.String(allow_none=True)
	address_line2 = fields.String(allow_none=True)
	city = fields.String(allow_none=True)
	state_province = fields.String(allow_none=True)
	postal_code = fields.String(allow_none=True)
	country = fields.String(allow_none=True)
	payment_terms_code = fields.String(default='NET_30')
	payment_method = fields.String(default='CHECK')
	currency_code = fields.String(default='USD')
	bank_name = fields.String(allow_none=True)
	bank_account_number = fields.String(allow_none=True)
	bank_routing_number = fields.String(allow_none=True)
	tax_id = fields.String(allow_none=True)
	tax_exempt = fields.Boolean(default=False)
	is_active = fields.Boolean(default=True)
	is_employee = fields.Boolean(default=False)
	credit_limit = fields.Decimal(default=0.00)
	require_po = fields.Boolean(default=False)
	hold_payment = fields.Boolean(default=False)
	is_1099_vendor = fields.Boolean(default=False)
	current_balance = fields.Decimal(dump_only=True)
	ytd_purchases = fields.Decimal(dump_only=True)
	notes = fields.String(allow_none=True)


class APInvoiceSchema(Schema):
	"""Schema for AP Invoice serialization"""
	invoice_id = fields.String(dump_only=True)
	invoice_number = fields.String(required=True)
	vendor_invoice_number = fields.String(required=True)
	vendor_id = fields.String(required=True)
	description = fields.String(allow_none=True)
	purchase_order_id = fields.String(allow_none=True)
	invoice_date = fields.Date(required=True)
	due_date = fields.Date(required=True)
	received_date = fields.Date(default=date.today)
	status = fields.String(default='Draft')
	subtotal_amount = fields.Decimal(default=0.00)
	tax_amount = fields.Decimal(default=0.00)
	discount_amount = fields.Decimal(default=0.00)
	freight_amount = fields.Decimal(default=0.00)
	misc_amount = fields.Decimal(default=0.00)
	total_amount = fields.Decimal(dump_only=True)
	payment_terms_code = fields.String(allow_none=True)
	currency_code = fields.String(default='USD')
	exchange_rate = fields.Decimal(default=1.000000)
	requires_approval = fields.Boolean(default=True)
	approved = fields.Boolean(dump_only=True)
	posted = fields.Boolean(dump_only=True)
	payment_status = fields.String(dump_only=True)
	outstanding_amount = fields.Decimal(dump_only=True)
	hold_flag = fields.Boolean(default=False)
	hold_reason = fields.String(allow_none=True)
	notes = fields.String(allow_none=True)


class APInvoiceLineSchema(Schema):
	"""Schema for AP Invoice Line serialization"""
	line_id = fields.String(dump_only=True)
	invoice_id = fields.String(required=True)
	line_number = fields.Integer(required=True)
	description = fields.String(allow_none=True)
	item_code = fields.String(allow_none=True)
	quantity = fields.Decimal(default=1.0000)
	unit_price = fields.Decimal(default=0.0000)
	line_amount = fields.Decimal(required=True)
	gl_account_id = fields.String(required=True)
	tax_code = fields.String(allow_none=True)
	tax_rate = fields.Decimal(default=0.00)
	tax_amount = fields.Decimal(dump_only=True)
	is_tax_inclusive = fields.Boolean(default=False)
	cost_center = fields.String(allow_none=True)
	department = fields.String(allow_none=True)
	project = fields.String(allow_none=True)
	is_asset = fields.Boolean(default=False)
	asset_id = fields.String(allow_none=True)


class APPaymentSchema(Schema):
	"""Schema for AP Payment serialization"""
	payment_id = fields.String(dump_only=True)
	payment_number = fields.String(required=True)
	vendor_id = fields.String(required=True)
	description = fields.String(allow_none=True)
	payment_date = fields.Date(required=True)
	payment_method = fields.String(required=True)
	check_number = fields.String(allow_none=True)
	bank_account_id = fields.String(allow_none=True)
	status = fields.String(default='Draft')
	payment_amount = fields.Decimal(dump_only=True)
	discount_taken = fields.Decimal(dump_only=True)
	total_amount = fields.Decimal(dump_only=True)
	currency_code = fields.String(default='USD')
	exchange_rate = fields.Decimal(default=1.000000)
	requires_approval = fields.Boolean(default=True)
	approved = fields.Boolean(dump_only=True)
	posted = fields.Boolean(dump_only=True)
	cleared = fields.Boolean(dump_only=True)
	voided = fields.Boolean(dump_only=True)
	notes = fields.String(allow_none=True)


class APPaymentLineSchema(Schema):
	"""Schema for AP Payment Line serialization"""
	payment_line_id = fields.String(dump_only=True)
	payment_id = fields.String(required=True)
	line_number = fields.Integer(required=True)
	invoice_id = fields.String(required=True)
	invoice_amount = fields.Decimal(dump_only=True)
	payment_amount = fields.Decimal(required=True)
	discount_taken = fields.Decimal(default=0.00)
	remaining_amount = fields.Decimal(dump_only=True)
	notes = fields.String(allow_none=True)


class APExpenseReportSchema(Schema):
	"""Schema for AP Expense Report serialization"""
	expense_report_id = fields.String(dump_only=True)
	report_number = fields.String(required=True)
	report_name = fields.String(required=True)
	employee_id = fields.String(required=True)
	employee_name = fields.String(required=True)
	department = fields.String(allow_none=True)
	vendor_id = fields.String(allow_none=True)
	report_date = fields.Date(required=True)
	period_start = fields.Date(required=True)
	period_end = fields.Date(required=True)
	status = fields.String(default='Draft')
	total_amount = fields.Decimal(dump_only=True)
	reimbursable_amount = fields.Decimal(dump_only=True)
	currency_code = fields.String(default='USD')
	requires_approval = fields.Boolean(default=True)
	approved = fields.Boolean(dump_only=True)
	rejected = fields.Boolean(dump_only=True)
	paid = fields.Boolean(dump_only=True)
	notes = fields.String(allow_none=True)


class APExpenseLineSchema(Schema):
	"""Schema for AP Expense Line serialization"""
	expense_line_id = fields.String(dump_only=True)
	expense_report_id = fields.String(required=True)
	line_number = fields.Integer(required=True)
	description = fields.String(required=True)
	expense_date = fields.Date(required=True)
	expense_category = fields.String(required=True)
	merchant_name = fields.String(allow_none=True)
	location = fields.String(allow_none=True)
	amount = fields.Decimal(required=True)
	currency_code = fields.String(default='USD')
	exchange_rate = fields.Decimal(default=1.000000)
	tax_code = fields.String(allow_none=True)
	tax_amount = fields.Decimal(default=0.00)
	gl_account_id = fields.String(required=True)
	is_reimbursable = fields.Boolean(default=True)
	reimbursement_rate = fields.Decimal(default=100.00)
	is_mileage = fields.Boolean(default=False)
	mileage_distance = fields.Decimal(default=0.00)
	mileage_rate = fields.Decimal(default=0.0000)
	has_receipt = fields.Boolean(default=False)
	is_personal = fields.Boolean(default=False)
	business_percentage = fields.Decimal(default=100.00)
	cost_center = fields.String(allow_none=True)
	project = fields.String(allow_none=True)
	notes = fields.String(allow_none=True)


class APAgingSchema(Schema):
	"""Schema for AP Aging serialization"""
	aging_id = fields.String(dump_only=True)
	as_of_date = fields.Date(required=True)
	vendor_id = fields.String(required=True)
	vendor_name = fields.String(dump_only=True)
	current_amount = fields.Decimal()
	days_31_60 = fields.Decimal()
	days_61_90 = fields.Decimal()
	days_91_120 = fields.Decimal()
	over_120_days = fields.Decimal()
	total_outstanding = fields.Decimal()
	generated_date = fields.DateTime(dump_only=True)


# REST API Resources

class APVendorApi(BaseApi):
	"""Vendor management API"""
	
	resource_name = 'vendors'
	datamodel = SQLAInterface(CFAPVendor)
	
	@expose('/', methods=['GET'])
	def get_list(self):
		"""Get list of vendors"""
		try:
			ap_service = AccountsPayableService(self.get_tenant_id())
			vendors = ap_service.get_vendors()
			
			schema = APVendorSchema(many=True)
			return jsonify({
				'success': True,
				'data': schema.dump(vendors),
				'count': len(vendors)
			})
		except Exception as e:
			return jsonify({'success': False, 'error': str(e)}), 500
	
	@expose('/<vendor_id>', methods=['GET'])
	def get_vendor(self, vendor_id):
		"""Get vendor by ID"""
		try:
			ap_service = AccountsPayableService(self.get_tenant_id())
			vendor = ap_service.get_vendor(vendor_id)
			
			if not vendor:
				return jsonify({'success': False, 'error': 'Vendor not found'}), 404
			
			schema = APVendorSchema()
			return jsonify({
				'success': True,
				'data': schema.dump(vendor)
			})
		except Exception as e:
			return jsonify({'success': False, 'error': str(e)}), 500
	
	@expose('/', methods=['POST'])
	def create_vendor(self):
		"""Create new vendor"""
		try:
			schema = APVendorSchema()
			vendor_data = schema.load(request.json)
			
			ap_service = AccountsPayableService(self.get_tenant_id())
			vendor = ap_service.create_vendor(vendor_data)
			
			return jsonify({
				'success': True,
				'data': schema.dump(vendor),
				'message': 'Vendor created successfully'
			}), 201
		except ValidationError as e:
			return jsonify({'success': False, 'errors': e.messages}), 400
		except Exception as e:
			return jsonify({'success': False, 'error': str(e)}), 500
	
	@expose('/<vendor_id>/summary', methods=['GET'])
	def get_vendor_summary(self, vendor_id):
		"""Get vendor summary with balances and activity"""
		try:
			ap_service = AccountsPayableService(self.get_tenant_id())
			vendor = ap_service.get_vendor(vendor_id)
			
			if not vendor:
				return jsonify({'success': False, 'error': 'Vendor not found'}), 404
			
			# Get recent activity
			invoices = ap_service.get_invoices(vendor_id=vendor_id)[:10]
			payments = ap_service.get_payments(vendor_id=vendor_id)[:10]
			
			# Calculate aging
			aging = ap_service._calculate_vendor_aging(vendor, date.today())
			
			return jsonify({
				'success': True,
				'data': {
					'vendor': APVendorSchema().dump(vendor),
					'recent_invoices': APInvoiceSchema(many=True).dump(invoices),
					'recent_payments': APPaymentSchema(many=True).dump(payments),
					'aging': APAgingSchema().dump(aging)
				}
			})
		except Exception as e:
			return jsonify({'success': False, 'error': str(e)}), 500


class APInvoiceApi(BaseApi):
	"""Invoice management API"""
	
	resource_name = 'invoices'
	datamodel = SQLAInterface(CFAPInvoice)
	
	@expose('/', methods=['GET'])
	def get_list(self):
		"""Get list of invoices"""
		try:
			status = request.args.get('status')
			vendor_id = request.args.get('vendor_id')
			limit = request.args.get('limit', 100, type=int)
			
			ap_service = AccountsPayableService(self.get_tenant_id())
			invoices = ap_service.get_invoices(status=status, vendor_id=vendor_id)
			
			if limit:
				invoices = invoices[:limit]
			
			schema = APInvoiceSchema(many=True)
			return jsonify({
				'success': True,
				'data': schema.dump(invoices),
				'count': len(invoices)
			})
		except Exception as e:
			return jsonify({'success': False, 'error': str(e)}), 500
	
	@expose('/<invoice_id>', methods=['GET'])
	def get_invoice(self, invoice_id):
		"""Get invoice by ID"""
		try:
			ap_service = AccountsPayableService(self.get_tenant_id())
			invoice = ap_service.get_invoice(invoice_id)
			
			if not invoice:
				return jsonify({'success': False, 'error': 'Invoice not found'}), 404
			
			schema = APInvoiceSchema()
			return jsonify({
				'success': True,
				'data': schema.dump(invoice)
			})
		except Exception as e:
			return jsonify({'success': False, 'error': str(e)}), 500
	
	@expose('/', methods=['POST'])
	def create_invoice(self):
		"""Create new invoice"""
		try:
			schema = APInvoiceSchema()
			invoice_data = schema.load(request.json)
			
			ap_service = AccountsPayableService(self.get_tenant_id())
			invoice = ap_service.create_invoice(invoice_data)
			
			return jsonify({
				'success': True,
				'data': schema.dump(invoice),
				'message': 'Invoice created successfully'
			}), 201
		except ValidationError as e:
			return jsonify({'success': False, 'errors': e.messages}), 400
		except Exception as e:
			return jsonify({'success': False, 'error': str(e)}), 500
	
	@expose('/<invoice_id>/approve', methods=['POST'])
	def approve_invoice(self, invoice_id):
		"""Approve invoice"""
		try:
			ap_service = AccountsPayableService(self.get_tenant_id())
			success = ap_service.approve_invoice(invoice_id, self.get_current_user_id())
			
			if success:
				return jsonify({
					'success': True,
					'message': 'Invoice approved successfully'
				})
			else:
				return jsonify({
					'success': False,
					'error': 'Invoice cannot be approved'
				}), 400
		except Exception as e:
			return jsonify({'success': False, 'error': str(e)}), 500
	
	@expose('/<invoice_id>/post', methods=['POST'])
	def post_invoice(self, invoice_id):
		"""Post invoice to GL"""
		try:
			ap_service = AccountsPayableService(self.get_tenant_id())
			success = ap_service.post_invoice(invoice_id, self.get_current_user_id())
			
			if success:
				return jsonify({
					'success': True,
					'message': 'Invoice posted to GL successfully'
				})
			else:
				return jsonify({
					'success': False,
					'error': 'Invoice cannot be posted'
				}), 400
		except Exception as e:
			return jsonify({'success': False, 'error': str(e)}), 500
	
	@expose('/<invoice_id>/lines', methods=['GET'])
	def get_invoice_lines(self, invoice_id):
		"""Get invoice lines"""
		try:
			ap_service = AccountsPayableService(self.get_tenant_id())
			invoice = ap_service.get_invoice(invoice_id)
			
			if not invoice:
				return jsonify({'success': False, 'error': 'Invoice not found'}), 404
			
			schema = APInvoiceLineSchema(many=True)
			return jsonify({
				'success': True,
				'data': schema.dump(invoice.lines)
			})
		except Exception as e:
			return jsonify({'success': False, 'error': str(e)}), 500
	
	@expose('/<invoice_id>/lines', methods=['POST'])
	def add_invoice_line(self, invoice_id):
		"""Add line to invoice"""
		try:
			schema = APInvoiceLineSchema()
			line_data = schema.load(request.json)
			line_data['invoice_id'] = invoice_id
			
			ap_service = AccountsPayableService(self.get_tenant_id())
			line = ap_service.add_invoice_line(invoice_id, line_data)
			
			return jsonify({
				'success': True,
				'data': schema.dump(line),
				'message': 'Invoice line added successfully'
			}), 201
		except ValidationError as e:
			return jsonify({'success': False, 'errors': e.messages}), 400
		except Exception as e:
			return jsonify({'success': False, 'error': str(e)}), 500


class APPaymentApi(BaseApi):
	"""Payment management API"""
	
	resource_name = 'payments'
	datamodel = SQLAInterface(CFAPPayment)
	
	@expose('/', methods=['GET'])
	def get_list(self):
		"""Get list of payments"""
		try:
			status = request.args.get('status')
			vendor_id = request.args.get('vendor_id')
			limit = request.args.get('limit', 100, type=int)
			
			ap_service = AccountsPayableService(self.get_tenant_id())
			payments = ap_service.get_payments(status=status, vendor_id=vendor_id)
			
			if limit:
				payments = payments[:limit]
			
			schema = APPaymentSchema(many=True)
			return jsonify({
				'success': True,
				'data': schema.dump(payments),
				'count': len(payments)
			})
		except Exception as e:
			return jsonify({'success': False, 'error': str(e)}), 500
	
	@expose('/<payment_id>', methods=['GET'])
	def get_payment(self, payment_id):
		"""Get payment by ID"""
		try:
			ap_service = AccountsPayableService(self.get_tenant_id())
			payment = ap_service.get_payment(payment_id)
			
			if not payment:
				return jsonify({'success': False, 'error': 'Payment not found'}), 404
			
			schema = APPaymentSchema()
			return jsonify({
				'success': True,
				'data': schema.dump(payment)
			})
		except Exception as e:
			return jsonify({'success': False, 'error': str(e)}), 500
	
	@expose('/', methods=['POST'])
	def create_payment(self):
		"""Create new payment"""
		try:
			schema = APPaymentSchema()
			payment_data = schema.load(request.json)
			
			ap_service = AccountsPayableService(self.get_tenant_id())
			payment = ap_service.create_payment(payment_data)
			
			return jsonify({
				'success': True,
				'data': schema.dump(payment),
				'message': 'Payment created successfully'
			}), 201
		except ValidationError as e:
			return jsonify({'success': False, 'errors': e.messages}), 400
		except Exception as e:
			return jsonify({'success': False, 'error': str(e)}), 500
	
	@expose('/<payment_id>/approve', methods=['POST'])
	def approve_payment(self, payment_id):
		"""Approve payment"""
		try:
			ap_service = AccountsPayableService(self.get_tenant_id())
			success = ap_service.approve_payment(payment_id, self.get_current_user_id())
			
			if success:
				return jsonify({
					'success': True,
					'message': 'Payment approved successfully'
				})
			else:
				return jsonify({
					'success': False,
					'error': 'Payment cannot be approved'
				}), 400
		except Exception as e:
			return jsonify({'success': False, 'error': str(e)}), 500
	
	@expose('/<payment_id>/post', methods=['POST'])
	def post_payment(self, payment_id):
		"""Post payment to GL"""
		try:
			ap_service = AccountsPayableService(self.get_tenant_id())
			success = ap_service.post_payment(payment_id, self.get_current_user_id())
			
			if success:
				return jsonify({
					'success': True,
					'message': 'Payment posted to GL successfully'
				})
			else:
				return jsonify({
					'success': False,
					'error': 'Payment cannot be posted'
				}), 400
		except Exception as e:
			return jsonify({'success': False, 'error': str(e)}), 500


class APExpenseReportApi(BaseApi):
	"""Expense report management API"""
	
	resource_name = 'expense_reports'
	datamodel = SQLAInterface(CFAPExpenseReport)
	
	@expose('/', methods=['GET'])
	def get_list(self):
		"""Get list of expense reports"""
		try:
			status = request.args.get('status')
			employee_id = request.args.get('employee_id')
			limit = request.args.get('limit', 100, type=int)
			
			ap_service = AccountsPayableService(self.get_tenant_id())
			reports = ap_service.get_expense_reports(status=status, employee_id=employee_id)
			
			if limit:
				reports = reports[:limit]
			
			schema = APExpenseReportSchema(many=True)
			return jsonify({
				'success': True,
				'data': schema.dump(reports),
				'count': len(reports)
			})
		except Exception as e:
			return jsonify({'success': False, 'error': str(e)}), 500
	
	@expose('/<report_id>', methods=['GET'])
	def get_expense_report(self, report_id):
		"""Get expense report by ID"""
		try:
			ap_service = AccountsPayableService(self.get_tenant_id())
			report = ap_service.get_expense_report(report_id)
			
			if not report:
				return jsonify({'success': False, 'error': 'Expense report not found'}), 404
			
			schema = APExpenseReportSchema()
			return jsonify({
				'success': True,
				'data': schema.dump(report)
			})
		except Exception as e:
			return jsonify({'success': False, 'error': str(e)}), 500
	
	@expose('/', methods=['POST'])
	def create_expense_report(self):
		"""Create new expense report"""
		try:
			schema = APExpenseReportSchema()
			report_data = schema.load(request.json)
			
			ap_service = AccountsPayableService(self.get_tenant_id())
			report = ap_service.create_expense_report(report_data)
			
			return jsonify({
				'success': True,
				'data': schema.dump(report),
				'message': 'Expense report created successfully'
			}), 201
		except ValidationError as e:
			return jsonify({'success': False, 'errors': e.messages}), 400
		except Exception as e:
			return jsonify({'success': False, 'error': str(e)}), 500
	
	@expose('/<report_id>/submit', methods=['POST'])
	def submit_expense_report(self, report_id):
		"""Submit expense report"""
		try:
			ap_service = AccountsPayableService(self.get_tenant_id())
			success = ap_service.submit_expense_report(report_id)
			
			if success:
				return jsonify({
					'success': True,
					'message': 'Expense report submitted successfully'
				})
			else:
				return jsonify({
					'success': False,
					'error': 'Expense report cannot be submitted'
				}), 400
		except Exception as e:
			return jsonify({'success': False, 'error': str(e)}), 500
	
	@expose('/<report_id>/approve', methods=['POST'])
	def approve_expense_report(self, report_id):
		"""Approve expense report"""
		try:
			ap_service = AccountsPayableService(self.get_tenant_id())
			success = ap_service.approve_expense_report(report_id, self.get_current_user_id())
			
			if success:
				return jsonify({
					'success': True,
					'message': 'Expense report approved successfully'
				})
			else:
				return jsonify({
					'success': False,
					'error': 'Expense report cannot be approved'
				}), 400
		except Exception as e:
			return jsonify({'success': False, 'error': str(e)}), 500


class APAgingApi(BaseApi):
	"""Aging analysis API"""
	
	resource_name = 'aging'
	
	@expose('/', methods=['GET'])
	def get_aging_report(self):
		"""Get AP aging analysis"""
		try:
			as_of_date_str = request.args.get('as_of_date')
			vendor_id = request.args.get('vendor_id')
			
			if as_of_date_str:
				try:
					as_of_date = datetime.strptime(as_of_date_str, '%Y-%m-%d').date()
				except ValueError:
					return jsonify({
						'success': False,
						'error': 'Invalid date format. Use YYYY-MM-DD'
					}), 400
			else:
				as_of_date = date.today()
			
			ap_service = AccountsPayableService(self.get_tenant_id())
			aging_records = ap_service.generate_aging_report(as_of_date, vendor_id)
			
			# Calculate summary
			summary = {
				'total_outstanding': float(sum(record.total_outstanding for record in aging_records)),
				'current': float(sum(record.current_amount for record in aging_records)),
				'days_31_60': float(sum(record.days_31_60 for record in aging_records)),
				'days_61_90': float(sum(record.days_61_90 for record in aging_records)),
				'days_91_120': float(sum(record.days_91_120 for record in aging_records)),
				'over_120': float(sum(record.over_120_days for record in aging_records)),
				'vendor_count': len(aging_records)
			}
			
			schema = APAgingSchema(many=True)
			return jsonify({
				'success': True,
				'data': schema.dump(aging_records),
				'summary': summary,
				'as_of_date': as_of_date.isoformat()
			})
		except Exception as e:
			return jsonify({'success': False, 'error': str(e)}), 500


class APDashboardApi(BaseApi):
	"""Dashboard API"""
	
	resource_name = 'dashboard'
	
	@expose('/summary', methods=['GET'])
	def get_summary(self):
		"""Get AP dashboard summary"""
		try:
			ap_service = AccountsPayableService(self.get_tenant_id())
			summary = ap_service.get_ap_summary()
			
			return jsonify({
				'success': True,
				'data': summary
			})
		except Exception as e:
			return jsonify({'success': False, 'error': str(e)}), 500
	
	@expose('/cash_requirements', methods=['GET'])
	def get_cash_requirements(self):
		"""Get cash flow requirements"""
		try:
			days = request.args.get('days', 30, type=int)
			ap_service = AccountsPayableService(self.get_tenant_id())
			cash_requirements = ap_service.get_cash_requirements(days)
			
			return jsonify({
				'success': True,
				'data': cash_requirements,
				'days_forward': days
			})
		except Exception as e:
			return jsonify({'success': False, 'error': str(e)}), 500


# API Blueprint setup
def create_api_blueprint():
	"""Create API blueprint with all endpoints"""
	api_bp = Blueprint('ap_api', __name__, url_prefix='/api/ap')
	api = Api(api_bp)
	
	# Register API resources
	api.add_resource(APVendorApi, '/vendors', '/vendors/<vendor_id>')
	api.add_resource(APInvoiceApi, '/invoices', '/invoices/<invoice_id>')
	api.add_resource(APPaymentApi, '/payments', '/payments/<payment_id>')
	api.add_resource(APExpenseReportApi, '/expense_reports', '/expense_reports/<report_id>')
	api.add_resource(APAgingApi, '/aging')
	api.add_resource(APDashboardApi, '/dashboard')
	
	return api_bp


# Utility functions for API responses
def format_api_response(success: bool, data: Any = None, message: str = None, 
					   error: str = None, status_code: int = 200) -> Tuple[Dict, int]:
	"""Format standardized API response"""
	response = {'success': success}
	
	if data is not None:
		response['data'] = data
	
	if message:
		response['message'] = message
	
	if error:
		response['error'] = error
	
	if success and status_code == 200 and data is not None:
		if isinstance(data, list):
			response['count'] = len(data)
	
	return response, status_code


def validate_date_parameter(date_str: str, param_name: str = 'date') -> date:
	"""Validate and parse date parameter"""
	if not date_str:
		return date.today()
	
	try:
		return datetime.strptime(date_str, '%Y-%m-%d').date()
	except ValueError:
		raise ValueError(f"Invalid {param_name} format. Use YYYY-MM-DD")


def paginate_results(query_results: List, page: int = 1, per_page: int = 50) -> Dict:
	"""Paginate query results"""
	total = len(query_results)
	start = (page - 1) * per_page
	end = start + per_page
	
	items = query_results[start:end]
	
	return {
		'items': items,
		'pagination': {
			'page': page,
			'per_page': per_page,
			'total': total,
			'pages': (total + per_page - 1) // per_page,
			'has_prev': page > 1,
			'has_next': end < total
		}
	}