"""
Accounts Receivable Views

Flask-AppBuilder views for Accounts Receivable functionality including
customer management, invoice processing, payment processing, collections, and reporting.
"""

from flask import flash, redirect, request, url_for, jsonify, render_template
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.charts.views import DirectByChartView
from flask_appbuilder.widgets import ListWidget, ShowWidget
from flask_appbuilder.actions import action
from wtforms import Form, StringField, SelectField, DecimalField, DateField, TextAreaField, BooleanField
from wtforms.validators import DataRequired, NumberRange, Email
from datetime import date, datetime, timedelta
from typing import Dict, List, Any

from .models import (
	CFARCustomer, CFARInvoice, CFARInvoiceLine, CFARPayment, CFARPaymentLine,
	CFARCreditMemo, CFARCreditMemoLine, CFARStatement, CFARCollection,
	CFARAging, CFARTaxCode, CFARRecurringBilling
)
from .service import AccountsReceivableService
from ...auth_rbac.models import db


class ARCustomerModelView(ModelView):
	"""Accounts Receivable Customer Management View"""
	
	datamodel = SQLAInterface(CFARCustomer)
	
	list_title = "Customers"
	show_title = "Customer Details"
	add_title = "Add Customer"
	edit_title = "Edit Customer"
	
	list_columns = [
		'customer_number', 'customer_name', 'customer_type', 'contact_name',
		'email', 'phone', 'payment_terms_code', 'current_balance', 'credit_limit',
		'credit_hold', 'is_active'
	]
	
	show_columns = [
		'customer_number', 'customer_name', 'customer_type', 'contact_name',
		'email', 'phone', 'fax', 'website',
		'billing_address_line1', 'billing_address_line2', 'billing_city', 
		'billing_state_province', 'billing_postal_code', 'billing_country',
		'shipping_address_line1', 'shipping_address_line2', 'shipping_city',
		'shipping_state_province', 'shipping_postal_code', 'shipping_country',
		'payment_terms_code', 'payment_method', 'currency_code',
		'credit_limit', 'credit_hold', 'credit_rating',
		'tax_id', 'tax_exempt', 'tax_exempt_number', 'default_tax_code',
		'is_active', 'allow_backorders', 'require_po', 'print_statements', 'send_dunning_letters',
		'sales_rep_id', 'territory_id', 'price_level',
		'current_balance', 'ytd_sales', 'last_payment_date', 'last_payment_amount',
		'notes', 'internal_notes',
		'created_on', 'updated_on'
	]
	
	add_columns = [
		'customer_number', 'customer_name', 'customer_type', 'contact_name',
		'email', 'phone', 'fax', 'website',
		'billing_address_line1', 'billing_address_line2', 'billing_city',
		'billing_state_province', 'billing_postal_code', 'billing_country',
		'shipping_address_line1', 'shipping_address_line2', 'shipping_city',
		'shipping_state_province', 'shipping_postal_code', 'shipping_country',
		'payment_terms_code', 'payment_method', 'currency_code',
		'credit_limit', 'credit_rating',
		'tax_id', 'tax_exempt', 'tax_exempt_number', 'default_tax_code',
		'is_active', 'allow_backorders', 'require_po', 'print_statements', 'send_dunning_letters',
		'sales_rep_id', 'territory_id', 'price_level',
		'notes'
	]
	
	edit_columns = [
		'customer_name', 'customer_type', 'contact_name',
		'email', 'phone', 'fax', 'website',
		'billing_address_line1', 'billing_address_line2', 'billing_city',
		'billing_state_province', 'billing_postal_code', 'billing_country',
		'shipping_address_line1', 'shipping_address_line2', 'shipping_city',
		'shipping_state_province', 'shipping_postal_code', 'shipping_country',
		'payment_terms_code', 'payment_method', 'currency_code',
		'credit_limit', 'credit_hold', 'credit_rating',
		'tax_id', 'tax_exempt', 'tax_exempt_number', 'default_tax_code',
		'is_active', 'allow_backorders', 'require_po', 'print_statements', 'send_dunning_letters',
		'sales_rep_id', 'territory_id', 'price_level',
		'notes', 'internal_notes'
	]
	
	search_columns = ['customer_number', 'customer_name', 'contact_name', 'email', 'tax_id']
	
	order_columns = ['customer_number', 'customer_name', 'customer_type', 'current_balance', 'credit_limit']
	
	base_order = ('customer_name', 'asc')
	
	formatters_columns = {
		'current_balance': lambda x: f"${x:,.2f}" if x else "$0.00",
		'ytd_sales': lambda x: f"${x:,.2f}" if x else "$0.00",
		'credit_limit': lambda x: f"${x:,.2f}" if x else "$0.00",
		'last_payment_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'email': lambda x: f'<a href="mailto:{x}">{x}</a>' if x else '',
		'credit_hold': lambda x: '<span class="label label-danger">On Hold</span>' if x else '<span class="label label-success">Active</span>',
	}
	
	def pre_add(self, item):
		"""Set tenant_id and generate customer number if needed"""
		item.tenant_id = self.get_tenant_id()
		
		if not item.customer_number:
			# Auto-generate customer number
			last_customer = CFARCustomer.query.filter_by(tenant_id=item.tenant_id)\
				.order_by(CFARCustomer.customer_number.desc()).first()
			
			if last_customer and last_customer.customer_number.isdigit():
				next_num = int(last_customer.customer_number) + 1
				item.customer_number = f"{next_num:06d}"
			else:
				item.customer_number = "000001"
	
	def pre_update(self, item):
		"""Update tenant_id"""
		item.tenant_id = self.get_tenant_id()
	
	@action("place_on_hold", "Place on Credit Hold", "Are you sure you want to place selected customers on credit hold?", "fa-ban")
	def place_on_hold(self, items):
		"""Place customers on credit hold"""
		if not isinstance(items, list):
			items = [items]
		
		ar_service = AccountsReceivableService(self.get_tenant_id())
		user_id = self.get_user_id()
		
		for item in items:
			ar_service.place_customer_on_hold(item.customer_id, "Manual hold via UI", user_id)
		
		flash(f"Placed {len(items)} customer(s) on credit hold", "success")
		return redirect(self.get_redirect())
	
	@action("release_hold", "Release Credit Hold", "Are you sure you want to release credit hold for selected customers?", "fa-check")
	def release_hold(self, items):
		"""Release customers from credit hold"""
		if not isinstance(items, list):
			items = [items]
		
		ar_service = AccountsReceivableService(self.get_tenant_id())
		user_id = self.get_user_id()
		
		for item in items:
			ar_service.release_customer_hold(item.customer_id, user_id)
		
		flash(f"Released credit hold for {len(items)} customer(s)", "success")
		return redirect(self.get_redirect())
	
	@expose('/customer_summary/<customer_id>')
	@has_access
	def customer_summary(self, customer_id):
		"""Show customer summary with balance, invoices, payments"""
		ar_service = AccountsReceivableService(self.get_tenant_id())
		customer = ar_service.get_customer(customer_id)
		
		if not customer:
			flash("Customer not found", "error")
			return redirect(self.get_redirect())
		
		# Get customer data
		recent_invoices = ar_service.get_invoices_by_customer(customer_id)[:10]
		outstanding_balance = customer.get_outstanding_balance()
		
		return self.render_template(
			'ar/customer_summary.html',
			customer=customer,
			recent_invoices=recent_invoices,
			outstanding_balance=outstanding_balance
		)
	
	def get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		# This would typically come from session or user context
		return "default_tenant"
	
	def get_user_id(self) -> str:
		"""Get current user ID"""
		# This would typically come from session
		return "default_user"


class ARInvoiceModelView(ModelView):
	"""Accounts Receivable Invoice Management View"""
	
	datamodel = SQLAInterface(CFARInvoice)
	
	list_title = "Customer Invoices"
	show_title = "Invoice Details"
	add_title = "Create Invoice"
	edit_title = "Edit Invoice"
	
	list_columns = [
		'invoice_number', 'customer.customer_name', 'invoice_date', 'due_date',
		'total_amount', 'outstanding_amount', 'status', 'payment_status'
	]
	
	show_columns = [
		'invoice_number', 'customer.customer_name', 'invoice_date', 'due_date',
		'description', 'sales_order_id', 'customer_po_number',
		'subtotal_amount', 'tax_amount', 'discount_amount', 'freight_amount', 'total_amount',
		'payment_status', 'paid_amount', 'outstanding_amount',
		'payment_terms_code', 'currency_code', 'status',
		'sales_rep_id', 'territory_id', 'commission_rate', 'commission_amount',
		'posted', 'posted_by', 'posted_date',
		'collection_status', 'dunning_level', 'last_dunning_date',
		'notes', 'internal_notes',
		'created_on', 'updated_on'
	]
	
	add_columns = [
		'customer', 'invoice_date', 'due_date', 'description',
		'sales_order_id', 'customer_po_number',
		'payment_terms_code', 'currency_code',
		'sales_rep_id', 'territory_id',
		'freight_amount', 'misc_amount', 'discount_amount',
		'notes'
	]
	
	edit_columns = [
		'customer', 'invoice_date', 'due_date', 'description',
		'sales_order_id', 'customer_po_number',
		'payment_terms_code', 'currency_code',
		'sales_rep_id', 'territory_id',
		'freight_amount', 'misc_amount', 'discount_amount',
		'hold_flag', 'hold_reason',
		'notes', 'internal_notes'
	]
	
	search_columns = ['invoice_number', 'customer.customer_name', 'customer_po_number', 'description']
	
	order_columns = ['invoice_number', 'invoice_date', 'due_date', 'total_amount', 'outstanding_amount']
	
	base_order = ('invoice_date', 'desc')
	
	formatters_columns = {
		'total_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'outstanding_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'paid_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'subtotal_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'tax_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'status': lambda x: f'<span class="label label-{self._get_status_class(x)}">{x}</span>',
		'payment_status': lambda x: f'<span class="label label-{self._get_payment_status_class(x)}">{x}</span>',
	}
	
	def _get_status_class(self, status):
		"""Get CSS class for status label"""
		status_classes = {
			'Draft': 'warning',
			'Posted': 'success',
			'Paid': 'info',
			'Cancelled': 'danger'
		}
		return status_classes.get(status, 'default')
	
	def _get_payment_status_class(self, status):
		"""Get CSS class for payment status label"""
		status_classes = {
			'Unpaid': 'danger',
			'Partial': 'warning',
			'Paid': 'success'
		}
		return status_classes.get(status, 'default')
	
	def pre_add(self, item):
		"""Set tenant_id and generate invoice number"""
		item.tenant_id = self.get_tenant_id()
		
		if not item.invoice_number:
			ar_service = AccountsReceivableService(self.get_tenant_id())
			item.invoice_number = ar_service._generate_invoice_number()
		
		if not item.due_date and item.customer and item.invoice_date:
			ar_service = AccountsReceivableService(self.get_tenant_id())
			payment_terms = item.customer.payment_terms_code or 'NET_30'
			item.due_date = ar_service._calculate_due_date(item.invoice_date, payment_terms)
	
	def pre_update(self, item):
		"""Update tenant_id"""
		item.tenant_id = self.get_tenant_id()
	
	@action("post_invoice", "Post to GL", "Are you sure you want to post selected invoices to General Ledger?", "fa-check")
	def post_invoice(self, items):
		"""Post invoices to General Ledger"""
		if not isinstance(items, list):
			items = [items]
		
		ar_service = AccountsReceivableService(self.get_tenant_id())
		user_id = self.get_user_id()
		
		posted_count = 0
		for item in items:
			if ar_service.post_invoice(item.invoice_id, user_id):
				posted_count += 1
		
		flash(f"Posted {posted_count} of {len(items)} invoice(s) to GL", "success")
		return redirect(self.get_redirect())
	
	@action("hold_invoice", "Place on Hold", "Are you sure you want to place selected invoices on hold?", "fa-pause")
	def hold_invoice(self, items):
		"""Place invoices on hold"""
		if not isinstance(items, list):
			items = [items]
		
		for item in items:
			item.hold_flag = True
			item.hold_reason = "Manual hold via UI"
		
		db.session.commit()
		flash(f"Placed {len(items)} invoice(s) on hold", "success")
		return redirect(self.get_redirect())
	
	@expose('/invoice_lines/<invoice_id>')
	@has_access
	def invoice_lines(self, invoice_id):
		"""Show invoice lines"""
		ar_service = AccountsReceivableService(self.get_tenant_id())
		invoice = ar_service.get_invoice(invoice_id)
		
		if not invoice:
			flash("Invoice not found", "error")
			return redirect(self.get_redirect())
		
		return self.render_template(
			'ar/invoice_lines.html',
			invoice=invoice,
			lines=invoice.lines
		)
	
	def get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"
	
	def get_user_id(self) -> str:
		"""Get current user ID"""
		return "default_user"


class ARPaymentModelView(ModelView):
	"""Accounts Receivable Payment Management View"""
	
	datamodel = SQLAInterface(CFARPayment)
	
	list_title = "Customer Payments"
	show_title = "Payment Details"
	add_title = "Record Payment"
	edit_title = "Edit Payment"
	
	list_columns = [
		'payment_number', 'customer.customer_name', 'payment_date',
		'payment_method', 'payment_amount', 'unapplied_amount', 'status'
	]
	
	show_columns = [
		'payment_number', 'customer.customer_name', 'payment_date',
		'payment_method', 'check_number', 'reference_number',
		'payment_amount', 'discount_taken', 'unapplied_amount', 'total_amount',
		'currency_code', 'exchange_rate', 'status',
		'posted', 'posted_by', 'posted_date',
		'cleared', 'cleared_date', 'bank_statement_date',
		'returned', 'return_date', 'return_reason', 'nsf_fee',
		'lockbox_batch_id', 'lockbox_processed',
		'notes',
		'created_on', 'updated_on'
	]
	
	add_columns = [
		'customer', 'payment_date', 'payment_method',
		'payment_amount', 'check_number', 'reference_number',
		'bank_account_id', 'currency_code', 'exchange_rate',
		'notes'
	]
	
	edit_columns = [
		'customer', 'payment_date', 'payment_method',
		'payment_amount', 'check_number', 'reference_number',
		'bank_account_id', 'currency_code', 'exchange_rate',
		'notes'
	]
	
	search_columns = ['payment_number', 'customer.customer_name', 'check_number', 'reference_number']
	
	order_columns = ['payment_number', 'payment_date', 'payment_amount']
	
	base_order = ('payment_date', 'desc')
	
	formatters_columns = {
		'payment_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'unapplied_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'total_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'discount_taken': lambda x: f"${x:,.2f}" if x else "$0.00",
		'nsf_fee': lambda x: f"${x:,.2f}" if x else "$0.00",
		'status': lambda x: f'<span class="label label-{self._get_status_class(x)}">{x}</span>',
	}
	
	def _get_status_class(self, status):
		"""Get CSS class for status label"""
		status_classes = {
			'Draft': 'warning',
			'Posted': 'success',
			'Cleared': 'info',
			'Returned': 'danger',
			'Voided': 'default'
		}
		return status_classes.get(status, 'default')
	
	def pre_add(self, item):
		"""Set tenant_id and generate payment number"""
		item.tenant_id = self.get_tenant_id()
		
		if not item.payment_number:
			ar_service = AccountsReceivableService(self.get_tenant_id())
			item.payment_number = ar_service._generate_payment_number()
	
	def pre_update(self, item):
		"""Update tenant_id"""
		item.tenant_id = self.get_tenant_id()
	
	@action("post_payment", "Post to GL", "Are you sure you want to post selected payments to General Ledger?", "fa-check")
	def post_payment(self, items):
		"""Post payments to General Ledger"""
		if not isinstance(items, list):
			items = [items]
		
		ar_service = AccountsReceivableService(self.get_tenant_id())
		user_id = self.get_user_id()
		
		posted_count = 0
		for item in items:
			if ar_service.post_payment(item.payment_id, user_id):
				posted_count += 1
		
		flash(f"Posted {posted_count} of {len(items)} payment(s) to GL", "success")
		return redirect(self.get_redirect())
	
	@action("auto_apply", "Auto Apply", "Auto-apply selected payments to oldest invoices?", "fa-magic")
	def auto_apply(self, items):
		"""Auto-apply payments to invoices"""
		if not isinstance(items, list):
			items = [items]
		
		ar_service = AccountsReceivableService(self.get_tenant_id())
		
		applied_count = 0
		for item in items:
			if ar_service.auto_apply_payment(item.payment_id):
				applied_count += 1
		
		flash(f"Auto-applied {applied_count} of {len(items)} payment(s)", "success")
		return redirect(self.get_redirect())
	
	@action("void_payment", "Void Payment", "Are you sure you want to void selected payments?", "fa-times")
	def void_payment(self, items):
		"""Void payments"""
		if not isinstance(items, list):
			items = [items]
		
		user_id = self.get_user_id()
		
		voided_count = 0
		for item in items:
			if item.can_void():
				item.void_payment(user_id, "Manual void via UI")
				voided_count += 1
		
		db.session.commit()
		flash(f"Voided {voided_count} of {len(items)} payment(s)", "success")
		return redirect(self.get_redirect())
	
	@expose('/payment_lines/<payment_id>')
	@has_access
	def payment_lines(self, payment_id):
		"""Show payment application lines"""
		ar_service = AccountsReceivableService(self.get_tenant_id())
		payment = ar_service.get_payment(payment_id)
		
		if not payment:
			flash("Payment not found", "error")
			return redirect(self.get_redirect())
		
		return self.render_template(
			'ar/payment_lines.html',
			payment=payment,
			lines=payment.payment_lines
		)
	
	def get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"
	
	def get_user_id(self) -> str:
		"""Get current user ID"""
		return "default_user"


class ARCreditMemoModelView(ModelView):
	"""Accounts Receivable Credit Memo Management View"""
	
	datamodel = SQLAInterface(CFARCreditMemo)
	
	list_title = "Credit Memos"
	show_title = "Credit Memo Details"
	add_title = "Create Credit Memo"
	edit_title = "Edit Credit Memo"
	
	list_columns = [
		'credit_memo_number', 'customer.customer_name', 'credit_date',
		'reason_code', 'total_amount', 'unapplied_amount', 'status'
	]
	
	show_columns = [
		'credit_memo_number', 'customer.customer_name', 'credit_date',
		'reference_invoice.invoice_number', 'reason_code', 'description',
		'subtotal_amount', 'tax_amount', 'total_amount',
		'applied_amount', 'unapplied_amount',
		'currency_code', 'exchange_rate', 'status',
		'return_authorization', 'received_date',
		'posted', 'posted_by', 'posted_date',
		'notes', 'internal_notes',
		'created_on', 'updated_on'
	]
	
	add_columns = [
		'customer', 'credit_date', 'reference_invoice',
		'reason_code', 'description',
		'return_authorization', 'currency_code',
		'notes'
	]
	
	edit_columns = [
		'customer', 'credit_date', 'reference_invoice',
		'reason_code', 'description',
		'return_authorization', 'received_date',
		'currency_code', 'exchange_rate',
		'notes', 'internal_notes'
	]
	
	search_columns = ['credit_memo_number', 'customer.customer_name', 'return_authorization', 'description']
	
	order_columns = ['credit_memo_number', 'credit_date', 'total_amount']
	
	base_order = ('credit_date', 'desc')
	
	formatters_columns = {
		'total_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'unapplied_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'applied_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'subtotal_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'tax_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'status': lambda x: f'<span class="label label-{self._get_status_class(x)}">{x}</span>',
	}
	
	def _get_status_class(self, status):
		"""Get CSS class for status label"""
		status_classes = {
			'Draft': 'warning',
			'Posted': 'success',
			'Applied': 'info'
		}
		return status_classes.get(status, 'default')
	
	def pre_add(self, item):
		"""Set tenant_id and generate credit memo number"""
		item.tenant_id = self.get_tenant_id()
		
		if not item.credit_memo_number:
			ar_service = AccountsReceivableService(self.get_tenant_id())
			item.credit_memo_number = ar_service._generate_credit_memo_number()
	
	def pre_update(self, item):
		"""Update tenant_id"""
		item.tenant_id = self.get_tenant_id()
	
	@action("post_credit_memo", "Post to GL", "Are you sure you want to post selected credit memos to General Ledger?", "fa-check")
	def post_credit_memo(self, items):
		"""Post credit memos to General Ledger"""
		if not isinstance(items, list):
			items = [items]
		
		ar_service = AccountsReceivableService(self.get_tenant_id())
		user_id = self.get_user_id()
		
		posted_count = 0
		for item in items:
			if ar_service.post_credit_memo(item.credit_memo_id, user_id):
				posted_count += 1
		
		flash(f"Posted {posted_count} of {len(items)} credit memo(s) to GL", "success")
		return redirect(self.get_redirect())
	
	def get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"
	
	def get_user_id(self) -> str:
		"""Get current user ID"""
		return "default_user"


class ARStatementModelView(ModelView):
	"""Accounts Receivable Statement Management View"""
	
	datamodel = SQLAInterface(CFARStatement)
	
	list_title = "Customer Statements"
	show_title = "Statement Details"
	add_title = "Generate Statement"
	edit_title = "Edit Statement"
	
	list_columns = [
		'statement_number', 'customer.customer_name', 'statement_date',
		'statement_type', 'ending_balance', 'status', 'delivery_method'
	]
	
	show_columns = [
		'statement_number', 'customer.customer_name', 'statement_date',
		'statement_period_start', 'statement_period_end', 'statement_type',
		'beginning_balance', 'charges', 'payments', 'adjustments', 'ending_balance',
		'current_amount', 'days_31_60', 'days_61_90', 'days_91_120', 'over_120_days',
		'status', 'delivery_method', 'email_address',
		'printed_date', 'emailed_date', 'pdf_generated',
		'template_name', 'include_remittance_slip',
		'currency_code', 'message',
		'created_on', 'updated_on'
	]
	
	add_columns = [
		'customer', 'statement_date', 'statement_type',
		'delivery_method', 'email_address',
		'template_name', 'include_remittance_slip',
		'message'
	]
	
	edit_columns = [
		'customer', 'statement_date', 'statement_type',
		'delivery_method', 'email_address',
		'template_name', 'include_remittance_slip',
		'message'
	]
	
	search_columns = ['statement_number', 'customer.customer_name']
	
	order_columns = ['statement_number', 'statement_date', 'ending_balance']
	
	base_order = ('statement_date', 'desc')
	
	formatters_columns = {
		'ending_balance': lambda x: f"${x:,.2f}" if x else "$0.00",
		'beginning_balance': lambda x: f"${x:,.2f}" if x else "$0.00",
		'charges': lambda x: f"${x:,.2f}" if x else "$0.00",
		'payments': lambda x: f"${x:,.2f}" if x else "$0.00",
		'adjustments': lambda x: f"${x:,.2f}" if x else "$0.00",
		'current_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'days_31_60': lambda x: f"${x:,.2f}" if x else "$0.00",
		'days_61_90': lambda x: f"${x:,.2f}" if x else "$0.00",
		'days_91_120': lambda x: f"${x:,.2f}" if x else "$0.00",
		'over_120_days': lambda x: f"${x:,.2f}" if x else "$0.00",
		'status': lambda x: f'<span class="label label-{self._get_status_class(x)}">{x}</span>',
	}
	
	def _get_status_class(self, status):
		"""Get CSS class for status label"""
		status_classes = {
			'Draft': 'warning',
			'Generated': 'info',
			'Printed': 'success',
			'Emailed': 'success',
			'Delivered': 'success'
		}
		return status_classes.get(status, 'default')
	
	def pre_add(self, item):
		"""Set tenant_id and generate statement number"""
		item.tenant_id = self.get_tenant_id()
		
		if not item.statement_number:
			ar_service = AccountsReceivableService(self.get_tenant_id())
			item.statement_number = ar_service._generate_statement_number()
	
	def pre_update(self, item):
		"""Update tenant_id"""
		item.tenant_id = self.get_tenant_id()
	
	@action("generate_statements", "Generate", "Generate selected statements?", "fa-file-alt")
	def generate_statements(self, items):
		"""Generate statements"""
		if not isinstance(items, list):
			items = [items]
		
		user_id = self.get_user_id()
		
		generated_count = 0
		for item in items:
			if item.status == 'Draft':
				item.generate_statement(user_id)
				generated_count += 1
		
		db.session.commit()
		flash(f"Generated {generated_count} of {len(items)} statement(s)", "success")
		return redirect(self.get_redirect())
	
	def get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"
	
	def get_user_id(self) -> str:
		"""Get current user ID"""
		return "default_user"


class ARCollectionModelView(ModelView):
	"""Accounts Receivable Collection Management View"""
	
	datamodel = SQLAInterface(CFARCollection)
	
	list_title = "Collection Activities"
	show_title = "Collection Details"
	add_title = "Add Collection Activity"
	edit_title = "Edit Collection Activity"
	
	list_columns = [
		'customer.customer_name', 'collection_date', 'collection_type',
		'dunning_level', 'days_past_due', 'amount_past_due', 'outcome', 'status'
	]
	
	show_columns = [
		'customer.customer_name', 'collection_date', 'collection_type',
		'collector_id', 'dunning_level', 'days_past_due', 'amount_past_due',
		'subject', 'notes', 'outcome',
		'follow_up_date', 'follow_up_required',
		'promised_amount', 'promised_date', 'promise_kept',
		'status', 'document_path',
		'created_on', 'updated_on'
	]
	
	add_columns = [
		'customer', 'collection_date', 'collection_type',
		'collector_id', 'dunning_level', 'days_past_due', 'amount_past_due',
		'subject', 'notes', 'outcome',
		'follow_up_date', 'follow_up_required',
		'promised_amount', 'promised_date'
	]
	
	edit_columns = [
		'customer', 'collection_date', 'collection_type',
		'collector_id', 'dunning_level', 'days_past_due', 'amount_past_due',
		'subject', 'notes', 'outcome',
		'follow_up_date', 'follow_up_required',
		'promised_amount', 'promised_date', 'promise_kept',
		'status'
	]
	
	search_columns = ['customer.customer_name', 'subject', 'notes', 'outcome']
	
	order_columns = ['collection_date', 'dunning_level', 'amount_past_due']
	
	base_order = ('collection_date', 'desc')
	
	formatters_columns = {
		'amount_past_due': lambda x: f"${x:,.2f}" if x else "$0.00",
		'promised_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'collection_type': lambda x: f'<span class="label label-info">{x}</span>',
		'status': lambda x: f'<span class="label label-{self._get_status_class(x)}">{x}</span>',
		'promise_kept': lambda x: '<span class="label label-success">Yes</span>' if x is True else '<span class="label label-danger">No</span>' if x is False else '-',
	}
	
	def _get_status_class(self, status):
		"""Get CSS class for status label"""
		status_classes = {
			'Open': 'warning',
			'Closed': 'success',
			'Follow_up_Required': 'danger'
		}
		return status_classes.get(status, 'default')
	
	def pre_add(self, item):
		"""Set tenant_id"""
		item.tenant_id = self.get_tenant_id()
	
	def pre_update(self, item):
		"""Update tenant_id"""
		item.tenant_id = self.get_tenant_id()
	
	@action("mark_promise_kept", "Mark Promise Kept", "Mark promise to pay as kept for selected activities?", "fa-check")
	def mark_promise_kept(self, items):
		"""Mark promise to pay as kept"""
		if not isinstance(items, list):
			items = [items]
		
		for item in items:
			item.mark_promise_kept(True)
		
		db.session.commit()
		flash(f"Marked {len(items)} promise(s) as kept", "success")
		return redirect(self.get_redirect())
	
	@action("mark_promise_broken", "Mark Promise Broken", "Mark promise to pay as broken for selected activities?", "fa-times")
	def mark_promise_broken(self, items):
		"""Mark promise to pay as broken"""
		if not isinstance(items, list):
			items = [items]
		
		for item in items:
			item.mark_promise_kept(False)
		
		db.session.commit()
		flash(f"Marked {len(items)} promise(s) as broken", "success")
		return redirect(self.get_redirect())
	
	def get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class ARRecurringBillingModelView(ModelView):
	"""Accounts Receivable Recurring Billing Management View"""
	
	datamodel = SQLAInterface(CFARRecurringBilling)
	
	list_title = "Recurring Billing"
	show_title = "Recurring Billing Details"
	add_title = "Setup Recurring Billing"
	edit_title = "Edit Recurring Billing"
	
	list_columns = [
		'billing_name', 'customer.customer_name', 'frequency',
		'billing_amount', 'next_billing_date', 'is_active', 'is_paused'
	]
	
	show_columns = [
		'billing_name', 'customer.customer_name', 'description',
		'frequency', 'start_date', 'end_date', 'next_billing_date',
		'billing_amount', 'tax_code', 'payment_terms_code',
		'gl_account_id', 'invoice_template', 'invoice_description_template',
		'is_active', 'is_paused', 'pause_start_date', 'pause_end_date',
		'last_processed_date', 'invoices_generated',
		'auto_process', 'advance_days',
		'notes',
		'created_on', 'updated_on'
	]
	
	add_columns = [
		'billing_name', 'customer', 'description',
		'frequency', 'start_date', 'end_date',
		'billing_amount', 'tax_code', 'payment_terms_code',
		'gl_account_id', 'invoice_template', 'invoice_description_template',
		'auto_process', 'advance_days',
		'notes'
	]
	
	edit_columns = [
		'billing_name', 'customer', 'description',
		'frequency', 'start_date', 'end_date', 'next_billing_date',
		'billing_amount', 'tax_code', 'payment_terms_code',
		'gl_account_id', 'invoice_template', 'invoice_description_template',
		'is_active', 'is_paused', 'pause_start_date', 'pause_end_date',
		'auto_process', 'advance_days',
		'notes'
	]
	
	search_columns = ['billing_name', 'customer.customer_name', 'description']
	
	order_columns = ['billing_name', 'next_billing_date', 'billing_amount']
	
	base_order = ('next_billing_date', 'asc')
	
	formatters_columns = {
		'billing_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'is_active': lambda x: '<span class="label label-success">Active</span>' if x else '<span class="label label-default">Inactive</span>',
		'is_paused': lambda x: '<span class="label label-warning">Paused</span>' if x else '<span class="label label-success">Running</span>',
	}
	
	def pre_add(self, item):
		"""Set tenant_id and next billing date"""
		item.tenant_id = self.get_tenant_id()
		
		if not item.next_billing_date and item.start_date:
			item.next_billing_date = item.start_date
	
	def pre_update(self, item):
		"""Update tenant_id"""
		item.tenant_id = self.get_tenant_id()
	
	@action("pause_billing", "Pause Billing", "Pause billing for selected items?", "fa-pause")
	def pause_billing(self, items):
		"""Pause recurring billing"""
		if not isinstance(items, list):
			items = [items]
		
		for item in items:
			item.pause_billing(date.today())
		
		db.session.commit()
		flash(f"Paused billing for {len(items)} item(s)", "success")
		return redirect(self.get_redirect())
	
	@action("resume_billing", "Resume Billing", "Resume billing for selected items?", "fa-play")
	def resume_billing(self, items):
		"""Resume recurring billing"""
		if not isinstance(items, list):
			items = [items]
		
		for item in items:
			item.resume_billing()
		
		db.session.commit()
		flash(f"Resumed billing for {len(items)} item(s)", "success")
		return redirect(self.get_redirect())
	
	@action("process_billing", "Process Now", "Process billing for selected items now?", "fa-cog")
	def process_billing(self, items):
		"""Process recurring billing now"""
		if not isinstance(items, list):
			items = [items]
		
		ar_service = AccountsReceivableService(self.get_tenant_id())
		user_id = self.get_user_id()
		
		generated_count = 0
		for item in items:
			if item.is_ready_for_billing():
				invoice = ar_service._generate_recurring_invoice(item, user_id)
				if invoice:
					generated_count += 1
		
		flash(f"Generated {generated_count} invoice(s) from recurring billing", "success")
		return redirect(self.get_redirect())
	
	def get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"
	
	def get_user_id(self) -> str:
		"""Get current user ID"""
		return "default_user"


class ARTaxCodeModelView(ModelView):
	"""Accounts Receivable Tax Code Management View"""
	
	datamodel = SQLAInterface(CFARTaxCode)
	
	list_title = "AR Tax Codes"
	show_title = "Tax Code Details"
	add_title = "Add Tax Code"
	edit_title = "Edit Tax Code"
	
	list_columns = [
		'code', 'name', 'tax_rate', 'jurisdiction', 'is_active'
	]
	
	show_columns = [
		'code', 'name', 'description', 'tax_rate', 'is_compound',
		'gl_account_id', 'is_active', 'effective_date', 'expiration_date',
		'jurisdiction', 'authority',
		'created_on', 'updated_on'
	]
	
	add_columns = [
		'code', 'name', 'description', 'tax_rate', 'is_compound',
		'gl_account_id', 'is_active', 'effective_date', 'expiration_date',
		'jurisdiction', 'authority'
	]
	
	edit_columns = [
		'name', 'description', 'tax_rate', 'is_compound',
		'gl_account_id', 'is_active', 'effective_date', 'expiration_date',
		'jurisdiction', 'authority'
	]
	
	search_columns = ['code', 'name', 'jurisdiction', 'authority']
	
	order_columns = ['code', 'name', 'tax_rate']
	
	base_order = ('code', 'asc')
	
	formatters_columns = {
		'tax_rate': lambda x: f"{x}%" if x else "0%",
		'is_active': lambda x: '<span class="label label-success">Active</span>' if x else '<span class="label label-default">Inactive</span>',
	}
	
	def pre_add(self, item):
		"""Set tenant_id"""
		item.tenant_id = self.get_tenant_id()
	
	def pre_update(self, item):
		"""Update tenant_id"""
		item.tenant_id = self.get_tenant_id()
	
	def get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class ARAgingView(BaseView):
	"""Accounts Receivable Aging Report View"""
	
	route_base = "/ar_aging"
	
	@expose('/')
	@has_access
	def index(self):
		"""Main aging report view"""
		ar_service = AccountsReceivableService(self.get_tenant_id())
		
		# Get aging data
		as_of_date = request.args.get('as_of_date')
		if as_of_date:
			as_of_date = datetime.strptime(as_of_date, '%Y-%m-%d').date()
		else:
			as_of_date = date.today()
		
		aging_records = ar_service.generate_aging_report(as_of_date, self.get_user_id())
		
		# Calculate totals
		total_current = sum(a.current_amount for a in aging_records)
		total_31_60 = sum(a.days_31_60 for a in aging_records)
		total_61_90 = sum(a.days_61_90 for a in aging_records)
		total_91_120 = sum(a.days_91_120 for a in aging_records)
		total_over_120 = sum(a.over_120_days for a in aging_records)
		total_outstanding = sum(a.total_outstanding for a in aging_records)
		
		return self.render_template(
			'ar/aging_report.html',
			aging_records=aging_records,
			as_of_date=as_of_date,
			totals={
				'current': total_current,
				'31_60': total_31_60,
				'61_90': total_61_90,
				'91_120': total_91_120,
				'over_120': total_over_120,
				'total': total_outstanding
			}
		)
	
	@expose('/export')
	@has_access
	def export(self):
		"""Export aging report to CSV"""
		# Implementation would export aging data to CSV
		flash("Aging report export functionality to be implemented", "info")
		return redirect(url_for('ARAgingView.index'))
	
	def get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"
	
	def get_user_id(self) -> str:
		"""Get current user ID"""
		return "default_user"


class ARDashboardView(BaseView):
	"""Accounts Receivable Dashboard View"""
	
	route_base = "/ar_dashboard"
	
	@expose('/')
	@has_access
	def index(self):
		"""Main AR dashboard"""
		ar_service = AccountsReceivableService(self.get_tenant_id())
		
		# Get summary data
		summary_data = self._get_ar_summary(ar_service)
		
		return self.render_template(
			'ar/dashboard.html',
			summary=summary_data
		)
	
	def _get_ar_summary(self, ar_service: AccountsReceivableService) -> Dict[str, Any]:
		"""Get AR summary data for dashboard"""
		# This would calculate key AR metrics
		customers = ar_service.get_customers()
		
		total_customers = len(customers)
		active_customers = len([c for c in customers if c.is_active])
		customers_on_hold = len([c for c in customers if c.credit_hold])
		
		# Get outstanding balances
		total_outstanding = sum(c.current_balance for c in customers)
		
		# Get past due customers
		past_due_customers = ar_service.get_customers_for_collections()
		
		return {
			'total_customers': total_customers,
			'active_customers': active_customers,
			'customers_on_hold': customers_on_hold,
			'total_outstanding': total_outstanding,
			'past_due_customers': len(past_due_customers),
			'collection_required': len([c for c in past_due_customers if c.current_balance > 1000])  # Arbitrary threshold
		}
	
	@expose('/api/summary')
	@has_access
	def api_summary(self):
		"""API endpoint for dashboard summary data"""
		ar_service = AccountsReceivableService(self.get_tenant_id())
		summary_data = self._get_ar_summary(ar_service)
		
		return jsonify(summary_data)
	
	@expose('/api/cash_flow')
	@has_access
	def api_cash_flow(self):
		"""API endpoint for cash flow projections"""
		# This would calculate expected cash inflows based on invoice due dates
		ar_service = AccountsReceivableService(self.get_tenant_id())
		
		# Simplified cash flow projection
		cash_flow_data = {
			'next_30_days': 50000.00,
			'next_60_days': 25000.00,
			'next_90_days': 15000.00,
			'beyond_90_days': 10000.00
		}
		
		return jsonify(cash_flow_data)
	
	def get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"