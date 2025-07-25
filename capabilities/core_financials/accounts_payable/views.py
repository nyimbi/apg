"""
Accounts Payable Views

Flask-AppBuilder views for Accounts Payable functionality including
vendor management, invoice processing, payment processing, and reporting.
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
	CFAPVendor, CFAPInvoice, CFAPInvoiceLine, CFAPPayment, CFAPPaymentLine,
	CFAPExpenseReport, CFAPExpenseLine, CFAPPurchaseOrder, CFAPTaxCode, CFAPAging
)
from .service import AccountsPayableService
from ...auth_rbac.models import db


class APVendorModelView(ModelView):
	"""Accounts Payable Vendor Management View"""
	
	datamodel = SQLAInterface(CFAPVendor)
	
	list_title = "Vendors"
	show_title = "Vendor Details"
	add_title = "Add Vendor"
	edit_title = "Edit Vendor"
	
	list_columns = [
		'vendor_number', 'vendor_name', 'vendor_type', 'contact_name',
		'email', 'phone', 'payment_terms_code', 'current_balance', 'is_active'
	]
	
	show_columns = [
		'vendor_number', 'vendor_name', 'vendor_type', 'contact_name',
		'email', 'phone', 'fax', 'website',
		'address_line1', 'address_line2', 'city', 'state_province', 'postal_code', 'country',
		'payment_terms_code', 'payment_method', 'currency_code',
		'bank_name', 'bank_account_number', 'bank_routing_number',
		'tax_id', 'tax_exempt', 'tax_code',
		'is_active', 'is_employee', 'credit_limit', 'require_po', 'hold_payment',
		'is_1099_vendor', 'form_1099_type',
		'current_balance', 'ytd_purchases',
		'notes', 'internal_notes',
		'created_on', 'updated_on'
	]
	
	add_columns = [
		'vendor_number', 'vendor_name', 'vendor_type', 'contact_name',
		'email', 'phone', 'fax', 'website',
		'address_line1', 'address_line2', 'city', 'state_province', 'postal_code', 'country',
		'payment_terms_code', 'payment_method', 'currency_code',
		'bank_name', 'bank_account_number', 'bank_routing_number',
		'tax_id', 'tax_exempt', 'tax_code',
		'is_active', 'is_employee', 'credit_limit', 'require_po',
		'is_1099_vendor', 'form_1099_type',
		'notes'
	]
	
	edit_columns = [
		'vendor_name', 'vendor_type', 'contact_name',
		'email', 'phone', 'fax', 'website',
		'address_line1', 'address_line2', 'city', 'state_province', 'postal_code', 'country',
		'payment_terms_code', 'payment_method', 'currency_code',
		'bank_name', 'bank_account_number', 'bank_routing_number',
		'tax_id', 'tax_exempt', 'tax_code',
		'is_active', 'is_employee', 'credit_limit', 'require_po', 'hold_payment',
		'is_1099_vendor', 'form_1099_type',
		'notes', 'internal_notes'
	]
	
	search_columns = ['vendor_number', 'vendor_name', 'contact_name', 'email', 'tax_id']
	
	order_columns = ['vendor_number', 'vendor_name', 'vendor_type', 'current_balance']
	
	base_order = ('vendor_name', 'asc')
	
	formatters_columns = {
		'current_balance': lambda x: f"${x:,.2f}" if x else "$0.00",
		'ytd_purchases': lambda x: f"${x:,.2f}" if x else "$0.00",
		'credit_limit': lambda x: f"${x:,.2f}" if x else "$0.00",
		'email': lambda x: f'<a href="mailto:{x}">{x}</a>' if x else '',
	}
	
	def pre_add(self, item):
		"""Set tenant_id and generate vendor number if needed"""
		item.tenant_id = self.get_tenant_id()
		
		if not item.vendor_number:
			# Auto-generate vendor number
			last_vendor = CFAPVendor.query.filter_by(tenant_id=item.tenant_id)\
				.order_by(CFAPVendor.vendor_number.desc()).first()
			
			if last_vendor and last_vendor.vendor_number.isdigit():
				next_num = int(last_vendor.vendor_number) + 1
				item.vendor_number = f"{next_num:06d}"
			else:
				item.vendor_number = "000001"
	
	def pre_update(self, item):
		"""Update vendor balance on save"""
		ap_service = AccountsPayableService(item.tenant_id)
		ap_service.update_vendor_balance(item.vendor_id)
	
	@action("hold_payment", "Hold Payment", "Hold payment for selected vendors", "fa-pause")
	def hold_payment(self, items):
		"""Hold payment for selected vendors"""
		count = 0
		for vendor in items:
			vendor.hold_payment = True
			count += 1
		
		db.session.commit()
		flash(f"Payment held for {count} vendor(s)", "success")
		return redirect(self.get_redirect())
	
	@action("release_hold", "Release Hold", "Release payment hold for selected vendors", "fa-play")
	def release_hold(self, items):
		"""Release payment hold for selected vendors"""
		count = 0
		for vendor in items:
			vendor.hold_payment = False
			count += 1
		
		db.session.commit()
		flash(f"Payment hold released for {count} vendor(s)", "success")
		return redirect(self.get_redirect())
	
	@expose('/vendor_summary/<vendor_id>')
	@has_access
	def vendor_summary(self, vendor_id):
		"""Display vendor summary with invoices and payments"""
		vendor = CFAPVendor.query.get_or_404(vendor_id)
		ap_service = AccountsPayableService(vendor.tenant_id)
		
		# Get recent invoices
		invoices = ap_service.get_invoices(vendor_id=vendor_id)[:10]
		
		# Get recent payments
		payments = ap_service.get_payments(vendor_id=vendor_id)[:10]
		
		# Calculate aging
		aging = ap_service._calculate_vendor_aging(vendor, date.today())
		
		return self.render_template(
			'ap/vendor_summary.html',
			vendor=vendor,
			invoices=invoices,
			payments=payments,
			aging=aging
		)


class APInvoiceModelView(ModelView):
	"""Accounts Payable Invoice Management View"""
	
	datamodel = SQLAInterface(CFAPInvoice)
	
	list_title = "Vendor Invoices"
	show_title = "Invoice Details"
	add_title = "Add Invoice"
	edit_title = "Edit Invoice"
	
	list_columns = [
		'invoice_number', 'vendor.vendor_name', 'vendor_invoice_number',
		'invoice_date', 'due_date', 'status', 'total_amount',
		'outstanding_amount', 'payment_status'
	]
	
	show_columns = [
		'invoice_number', 'vendor_invoice_number', 'vendor',
		'description', 'purchase_order',
		'invoice_date', 'due_date', 'received_date',
		'status', 'workflow_status',
		'subtotal_amount', 'tax_amount', 'discount_amount',
		'freight_amount', 'misc_amount', 'total_amount',
		'payment_terms_code', 'currency_code', 'exchange_rate',
		'requires_approval', 'approved', 'approved_by', 'approved_date',
		'posted', 'posted_by', 'posted_date',
		'payment_status', 'paid_amount', 'outstanding_amount',
		'hold_flag', 'hold_reason', 'recurring_flag',
		'document_path', 'document_count',
		'notes', 'internal_notes',
		'created_on', 'updated_on'
	]
	
	add_columns = [
		'invoice_number', 'vendor_invoice_number', 'vendor',
		'description', 'purchase_order',
		'invoice_date', 'due_date',
		'subtotal_amount', 'tax_amount', 'discount_amount',
		'freight_amount', 'misc_amount',
		'payment_terms_code', 'currency_code',
		'requires_approval', 'hold_flag', 'hold_reason',
		'document_path', 'notes'
	]
	
	edit_columns = [
		'vendor_invoice_number', 'description',
		'invoice_date', 'due_date',
		'subtotal_amount', 'tax_amount', 'discount_amount',
		'freight_amount', 'misc_amount',
		'payment_terms_code', 'currency_code',
		'hold_flag', 'hold_reason',
		'document_path', 'notes', 'internal_notes'
	]
	
	search_columns = [
		'invoice_number', 'vendor_invoice_number', 'vendor.vendor_name',
		'description', 'status'
	]
	
	order_columns = [
		'invoice_number', 'invoice_date', 'due_date', 'total_amount',
		'outstanding_amount', 'status'
	]
	
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
			'Draft': 'default',
			'Pending': 'warning',
			'Approved': 'info',
			'Posted': 'success',
			'Paid': 'success',
			'Cancelled': 'danger'
		}
		return status_classes.get(status, 'default')
	
	def _get_payment_status_class(self, payment_status):
		"""Get CSS class for payment status label"""
		payment_classes = {
			'Unpaid': 'warning',
			'Partial': 'info',
			'Paid': 'success'
		}
		return payment_classes.get(payment_status, 'default')
	
	def pre_add(self, item):
		"""Set tenant_id and calculate totals"""
		item.tenant_id = self.get_tenant_id()
		
		if not item.invoice_number:
			# Auto-generate invoice number
			today = date.today()
			prefix = f"INV{today.year}{today.month:02d}"
			
			last_invoice = CFAPInvoice.query.filter(
				CFAPInvoice.tenant_id == item.tenant_id,
				CFAPInvoice.invoice_number.like(f"{prefix}%")
			).order_by(CFAPInvoice.invoice_number.desc()).first()
			
			if last_invoice:
				try:
					last_num = int(last_invoice.invoice_number[-4:])
					next_num = last_num + 1
				except ValueError:
					next_num = 1
			else:
				next_num = 1
			
			item.invoice_number = f"{prefix}{next_num:04d}"
		
		# Calculate total
		item.total_amount = (item.subtotal_amount + item.tax_amount + 
							 item.freight_amount + item.misc_amount - 
							 item.discount_amount)
		item.outstanding_amount = item.total_amount
	
	def pre_update(self, item):
		"""Recalculate totals on update"""
		item.calculate_totals()
	
	@action("approve_invoice", "Approve Invoice", "Approve selected invoices", "fa-check")
	def approve_invoice(self, items):
		"""Approve selected invoices"""
		count = 0
		ap_service = AccountsPayableService(self.get_tenant_id())
		
		for invoice in items:
			if ap_service.approve_invoice(invoice.invoice_id, self.get_current_user_id()):
				count += 1
		
		flash(f"Approved {count} invoice(s)", "success")
		return redirect(self.get_redirect())
	
	@action("post_invoice", "Post to GL", "Post selected invoices to General Ledger", "fa-share")
	def post_invoice(self, items):
		"""Post selected invoices to GL"""
		count = 0
		ap_service = AccountsPayableService(self.get_tenant_id())
		
		for invoice in items:
			try:
				if ap_service.post_invoice(invoice.invoice_id, self.get_current_user_id()):
					count += 1
			except Exception as e:
				flash(f"Error posting invoice {invoice.invoice_number}: {str(e)}", "danger")
		
		flash(f"Posted {count} invoice(s) to GL", "success")
		return redirect(self.get_redirect())
	
	@action("hold_invoice", "Hold Invoice", "Put selected invoices on hold", "fa-pause")
	def hold_invoice(self, items):
		"""Put selected invoices on hold"""
		count = 0
		for invoice in items:
			invoice.hold_flag = True
			invoice.hold_reason = "Manual hold"
			count += 1
		
		db.session.commit()
		flash(f"Put {count} invoice(s) on hold", "success")
		return redirect(self.get_redirect())
	
	@expose('/invoice_lines/<invoice_id>')
	@has_access
	def invoice_lines(self, invoice_id):
		"""Display invoice lines"""
		invoice = CFAPInvoice.query.get_or_404(invoice_id)
		return self.render_template('ap/invoice_lines.html', invoice=invoice)


class APPaymentModelView(ModelView):
	"""Accounts Payable Payment Management View"""
	
	datamodel = SQLAInterface(CFAPPayment)
	
	list_title = "Vendor Payments"
	show_title = "Payment Details"
	add_title = "Add Payment"
	edit_title = "Edit Payment"
	
	list_columns = [
		'payment_number', 'vendor.vendor_name', 'payment_date',
		'payment_method', 'check_number', 'status',
		'total_amount', 'posted', 'cleared'
	]
	
	show_columns = [
		'payment_number', 'vendor', 'description',
		'payment_date', 'payment_method', 'check_number', 'bank_account_id',
		'status', 'payment_amount', 'discount_taken', 'total_amount',
		'currency_code', 'exchange_rate',
		'requires_approval', 'approved', 'approved_by', 'approved_date',
		'posted', 'posted_by', 'posted_date',
		'cleared', 'cleared_date', 'bank_statement_date',
		'voided', 'void_date', 'void_reason',
		'document_path', 'notes',
		'created_on', 'updated_on'
	]
	
	add_columns = [
		'payment_number', 'vendor', 'description',
		'payment_date', 'payment_method', 'check_number', 'bank_account_id',
		'currency_code', 'requires_approval',
		'document_path', 'notes'
	]
	
	edit_columns = [
		'description', 'payment_date', 'payment_method', 'check_number',
		'bank_account_id', 'currency_code',
		'document_path', 'notes'
	]
	
	search_columns = [
		'payment_number', 'vendor.vendor_name', 'check_number',
		'description', 'status'
	]
	
	order_columns = [
		'payment_number', 'payment_date', 'total_amount', 'status'
	]
	
	base_order = ('payment_date', 'desc')
	
	formatters_columns = {
		'total_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'payment_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'discount_taken': lambda x: f"${x:,.2f}" if x else "$0.00",
		'status': lambda x: f'<span class="label label-{self._get_payment_status_class(x)}">{x}</span>',
		'posted': lambda x: '<i class="fa fa-check text-success"></i>' if x else '<i class="fa fa-times text-danger"></i>',
		'cleared': lambda x: '<i class="fa fa-check text-success"></i>' if x else '<i class="fa fa-times text-danger"></i>',
	}
	
	def _get_payment_status_class(self, status):
		"""Get CSS class for payment status label"""
		status_classes = {
			'Draft': 'default',
			'Pending': 'warning',
			'Approved': 'info',
			'Posted': 'success',
			'Cleared': 'success',
			'Voided': 'danger'
		}
		return status_classes.get(status, 'default')
	
	def pre_add(self, item):
		"""Set tenant_id and auto-generate payment number"""
		item.tenant_id = self.get_tenant_id()
		
		if not item.payment_number:
			# Auto-generate payment number
			today = date.today()
			prefix = f"PAY{today.year}{today.month:02d}"
			
			last_payment = CFAPPayment.query.filter(
				CFAPPayment.tenant_id == item.tenant_id,
				CFAPPayment.payment_number.like(f"{prefix}%")
			).order_by(CFAPPayment.payment_number.desc()).first()
			
			if last_payment:
				try:
					last_num = int(last_payment.payment_number[-4:])
					next_num = last_num + 1
				except ValueError:
					next_num = 1
			else:
				next_num = 1
			
			item.payment_number = f"{prefix}{next_num:04d}"
	
	def pre_update(self, item):
		"""Recalculate totals on update"""
		item.calculate_totals()
	
	@action("approve_payment", "Approve Payment", "Approve selected payments", "fa-check")
	def approve_payment(self, items):
		"""Approve selected payments"""
		count = 0
		ap_service = AccountsPayableService(self.get_tenant_id())
		
		for payment in items:
			if ap_service.approve_payment(payment.payment_id, self.get_current_user_id()):
				count += 1
		
		flash(f"Approved {count} payment(s)", "success")
		return redirect(self.get_redirect())
	
	@action("post_payment", "Post to GL", "Post selected payments to General Ledger", "fa-share")
	def post_payment(self, items):
		"""Post selected payments to GL"""
		count = 0
		ap_service = AccountsPayableService(self.get_tenant_id())
		
		for payment in items:
			try:
				if ap_service.post_payment(payment.payment_id, self.get_current_user_id()):
					count += 1
			except Exception as e:
				flash(f"Error posting payment {payment.payment_number}: {str(e)}", "danger")
		
		flash(f"Posted {count} payment(s) to GL", "success")
		return redirect(self.get_redirect())
	
	@action("void_payment", "Void Payment", "Void selected payments", "fa-ban")
	def void_payment(self, items):
		"""Void selected payments"""
		count = 0
		for payment in items:
			if payment.can_void():
				payment.void_payment(self.get_current_user_id(), "Manual void")
				count += 1
		
		db.session.commit()
		flash(f"Voided {count} payment(s)", "success")
		return redirect(self.get_redirect())
	
	@expose('/payment_lines/<payment_id>')
	@has_access
	def payment_lines(self, payment_id):
		"""Display payment allocation lines"""
		payment = CFAPPayment.query.get_or_404(payment_id)
		return self.render_template('ap/payment_lines.html', payment=payment)


class APExpenseReportModelView(ModelView):
	"""Expense Report Management View"""
	
	datamodel = SQLAInterface(CFAPExpenseReport)
	
	list_title = "Expense Reports"
	show_title = "Expense Report Details"
	add_title = "Add Expense Report"
	edit_title = "Edit Expense Report"
	
	list_columns = [
		'report_number', 'report_name', 'employee_name', 'department',
		'report_date', 'status', 'total_amount', 'reimbursable_amount'
	]
	
	show_columns = [
		'report_number', 'report_name', 'employee_name', 'department', 'vendor',
		'report_date', 'period_start', 'period_end', 'submitted_date',
		'status', 'total_amount', 'reimbursable_amount', 'non_reimbursable_amount',
		'paid_amount', 'currency_code',
		'requires_approval', 'approved', 'approved_by', 'approved_date',
		'rejected', 'rejected_by', 'rejected_date', 'rejection_reason',
		'paid', 'payment_id', 'paid_date',
		'receipt_count', 'document_path',
		'notes', 'manager_notes',
		'created_on', 'updated_on'
	]
	
	add_columns = [
		'report_number', 'report_name', 'employee_name', 'department',
		'report_date', 'period_start', 'period_end',
		'currency_code', 'requires_approval',
		'document_path', 'notes'
	]
	
	edit_columns = [
		'report_name', 'department',
		'period_start', 'period_end',
		'currency_code', 'document_path', 'notes', 'manager_notes'
	]
	
	search_columns = [
		'report_number', 'report_name', 'employee_name', 'department', 'status'
	]
	
	order_columns = [
		'report_number', 'report_date', 'total_amount', 'status'
	]
	
	base_order = ('report_date', 'desc')
	
	formatters_columns = {
		'total_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'reimbursable_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'non_reimbursable_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'paid_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'status': lambda x: f'<span class="label label-{self._get_expense_status_class(x)}">{x}</span>',
	}
	
	def _get_expense_status_class(self, status):
		"""Get CSS class for expense status label"""
		status_classes = {
			'Draft': 'default',
			'Submitted': 'warning',
			'Approved': 'success',
			'Paid': 'success',
			'Rejected': 'danger'
		}
		return status_classes.get(status, 'default')
	
	def pre_add(self, item):
		"""Set tenant_id and generate report number"""
		item.tenant_id = self.get_tenant_id()
		
		if not item.report_number:
			# Auto-generate expense report number
			today = date.today()
			prefix = f"EXP{today.year}{today.month:02d}"
			
			last_report = CFAPExpenseReport.query.filter(
				CFAPExpenseReport.tenant_id == item.tenant_id,
				CFAPExpenseReport.report_number.like(f"{prefix}%")
			).order_by(CFAPExpenseReport.report_number.desc()).first()
			
			if last_report:
				try:
					last_num = int(last_report.report_number[-4:])
					next_num = last_num + 1
				except ValueError:
					next_num = 1
			else:
				next_num = 1
			
			item.report_number = f"{prefix}{next_num:04d}"
	
	def pre_update(self, item):
		"""Recalculate totals on update"""
		item.calculate_totals()
	
	@action("submit_report", "Submit Report", "Submit selected expense reports", "fa-paper-plane")
	def submit_report(self, items):
		"""Submit selected expense reports"""
		count = 0
		ap_service = AccountsPayableService(self.get_tenant_id())
		
		for report in items:
			if ap_service.submit_expense_report(report.expense_report_id):
				count += 1
		
		flash(f"Submitted {count} expense report(s)", "success")
		return redirect(self.get_redirect())
	
	@action("approve_report", "Approve Report", "Approve selected expense reports", "fa-check")
	def approve_report(self, items):
		"""Approve selected expense reports"""
		count = 0
		ap_service = AccountsPayableService(self.get_tenant_id())
		
		for report in items:
			if ap_service.approve_expense_report(report.expense_report_id, self.get_current_user_id()):
				count += 1
		
		flash(f"Approved {count} expense report(s)", "success")
		return redirect(self.get_redirect())
	
	@expose('/expense_lines/<report_id>')
	@has_access
	def expense_lines(self, report_id):
		"""Display expense report lines"""
		report = CFAPExpenseReport.query.get_or_404(report_id)
		return self.render_template('ap/expense_lines.html', report=report)


class APAgingView(BaseView):
	"""Accounts Payable Aging Analysis View"""
	
	default_view = 'index'
	
	@expose('/')
	@has_access
	def index(self):
		"""Display AP aging analysis"""
		as_of_date = request.args.get('as_of_date')
		if as_of_date:
			try:
				as_of_date = datetime.strptime(as_of_date, '%Y-%m-%d').date()
			except ValueError:
				as_of_date = date.today()
		else:
			as_of_date = date.today()
		
		ap_service = AccountsPayableService(self.get_tenant_id())
		aging_records = ap_service.generate_aging_report(as_of_date)
		
		# Calculate summary totals
		summary = {
			'total_outstanding': sum(record.total_outstanding for record in aging_records),
			'current': sum(record.current_amount for record in aging_records),
			'days_31_60': sum(record.days_31_60 for record in aging_records),
			'days_61_90': sum(record.days_61_90 for record in aging_records),
			'days_91_120': sum(record.days_91_120 for record in aging_records),
			'over_120': sum(record.over_120_days for record in aging_records),
			'vendor_count': len(aging_records)
		}
		
		return self.render_template(
			'ap/aging_report.html',
			aging_records=aging_records,
			summary=summary,
			as_of_date=as_of_date
		)
	
	@expose('/export')
	@has_access
	def export(self):
		"""Export aging report to CSV"""
		as_of_date = request.args.get('as_of_date')
		if as_of_date:
			try:
				as_of_date = datetime.strptime(as_of_date, '%Y-%m-%d').date()
			except ValueError:
				as_of_date = date.today()
		else:
			as_of_date = date.today()
		
		ap_service = AccountsPayableService(self.get_tenant_id())
		aging_records = ap_service.generate_aging_report(as_of_date)
		
		# Generate CSV content
		import csv
		import io
		
		output = io.StringIO()
		writer = csv.writer(output)
		
		# Header
		writer.writerow([
			'Vendor', 'Current', '31-60 Days', '61-90 Days',
			'91-120 Days', 'Over 120 Days', 'Total Outstanding'
		])
		
		# Data rows
		for record in aging_records:
			writer.writerow([
				record.vendor.vendor_name,
				float(record.current_amount),
				float(record.days_31_60),
				float(record.days_61_90),
				float(record.days_91_120),
				float(record.over_120_days),
				float(record.total_outstanding)
			])
		
		output.seek(0)
		
		from flask import Response
		return Response(
			output.getvalue(),
			mimetype='text/csv',
			headers={
				'Content-Disposition': f'attachment; filename=ap_aging_{as_of_date}.csv'
			}
		)


class APDashboardView(BaseView):
	"""Accounts Payable Dashboard View"""
	
	default_view = 'index'
	
	@expose('/')
	@has_access
	def index(self):
		"""Display AP dashboard"""
		ap_service = AccountsPayableService(self.get_tenant_id())
		
		# Get summary statistics
		summary = ap_service.get_ap_summary()
		
		# Get cash requirements
		cash_requirements = ap_service.get_cash_requirements(30)
		
		# Get aging summary
		aging_records = ap_service.generate_aging_report()
		aging_summary = {
			'total_outstanding': sum(record.total_outstanding for record in aging_records),
			'current': sum(record.current_amount for record in aging_records),
			'overdue': sum(
				record.days_31_60 + record.days_61_90 + 
				record.days_91_120 + record.over_120_days 
				for record in aging_records
			)
		}
		
		# Get recent activity
		recent_invoices = ap_service.get_invoices()[:5]
		recent_payments = ap_service.get_payments()[:5]
		
		return self.render_template(
			'ap/dashboard.html',
			summary=summary,
			cash_requirements=cash_requirements[:10],  # Top 10
			aging_summary=aging_summary,
			recent_invoices=recent_invoices,
			recent_payments=recent_payments
		)
	
	@expose('/api/summary')
	@has_access
	def api_summary(self):
		"""API endpoint for dashboard summary data"""
		ap_service = AccountsPayableService(self.get_tenant_id())
		summary = ap_service.get_ap_summary()
		return jsonify(summary)
	
	@expose('/api/cash_flow')
	@has_access
	def api_cash_flow(self):
		"""API endpoint for cash flow requirements"""
		days = request.args.get('days', 30, type=int)
		ap_service = AccountsPayableService(self.get_tenant_id())
		cash_requirements = ap_service.get_cash_requirements(days)
		return jsonify(cash_requirements)


class APTaxCodeModelView(ModelView):
	"""Tax Code Management View"""
	
	datamodel = SQLAInterface(CFAPTaxCode)
	
	list_title = "Tax Codes"
	show_title = "Tax Code Details"
	add_title = "Add Tax Code"
	edit_title = "Edit Tax Code"
	
	list_columns = [
		'code', 'name', 'tax_rate', 'is_active', 'is_recoverable'
	]
	
	show_columns = [
		'code', 'name', 'description', 'tax_rate', 'is_compound',
		'gl_account_id', 'is_active', 'is_recoverable',
		'created_on', 'updated_on'
	]
	
	add_columns = [
		'code', 'name', 'description', 'tax_rate', 'is_compound',
		'gl_account_id', 'is_active', 'is_recoverable'
	]
	
	edit_columns = [
		'name', 'description', 'tax_rate', 'is_compound',
		'gl_account_id', 'is_active', 'is_recoverable'
	]
	
	search_columns = ['code', 'name', 'description']
	
	order_columns = ['code', 'name', 'tax_rate']
	
	base_order = ('code', 'asc')
	
	formatters_columns = {
		'tax_rate': lambda x: f"{x:.2f}%" if x else "0.00%",
		'is_active': lambda x: '<i class="fa fa-check text-success"></i>' if x else '<i class="fa fa-times text-danger"></i>',
		'is_recoverable': lambda x: '<i class="fa fa-check text-success"></i>' if x else '<i class="fa fa-times text-danger"></i>',
	}
	
	def pre_add(self, item):
		"""Set tenant_id"""
		item.tenant_id = self.get_tenant_id()


class APPurchaseOrderModelView(ModelView):
	"""Purchase Order Management View"""
	
	datamodel = SQLAInterface(CFAPPurchaseOrder)
	
	list_title = "Purchase Orders"
	show_title = "Purchase Order Details"
	add_title = "Add Purchase Order"
	edit_title = "Edit Purchase Order"
	
	list_columns = [
		'po_number', 'vendor.vendor_name', 'po_date', 'required_date',
		'status', 'po_amount', 'received_amount', 'invoiced_amount'
	]
	
	show_columns = [
		'po_number', 'vendor', 'description',
		'po_date', 'required_date', 'status',
		'po_amount', 'received_amount', 'invoiced_amount',
		'currency_code', 'approved', 'approved_by', 'approved_date',
		'created_on', 'updated_on'
	]
	
	add_columns = [
		'po_number', 'vendor', 'description',
		'po_date', 'required_date', 'po_amount', 'currency_code'
	]
	
	edit_columns = [
		'description', 'required_date', 'po_amount', 'currency_code'
	]
	
	search_columns = ['po_number', 'vendor.vendor_name', 'description', 'status']
	
	order_columns = ['po_number', 'po_date', 'po_amount', 'status']
	
	base_order = ('po_date', 'desc')
	
	formatters_columns = {
		'po_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'received_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'invoiced_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'status': lambda x: f'<span class="label label-{self._get_po_status_class(x)}">{x}</span>',
		'approved': lambda x: '<i class="fa fa-check text-success"></i>' if x else '<i class="fa fa-times text-danger"></i>',
	}
	
	def _get_po_status_class(self, status):
		"""Get CSS class for PO status label"""
		status_classes = {
			'Open': 'info',
			'Received': 'warning',
			'Closed': 'success',
			'Cancelled': 'danger'
		}
		return status_classes.get(status, 'default')
	
	def pre_add(self, item):
		"""Set tenant_id and generate PO number"""
		item.tenant_id = self.get_tenant_id()
		
		if not item.po_number:
			# Auto-generate PO number
			today = date.today()
			prefix = f"PO{today.year}{today.month:02d}"
			
			last_po = CFAPPurchaseOrder.query.filter(
				CFAPPurchaseOrder.tenant_id == item.tenant_id,
				CFAPPurchaseOrder.po_number.like(f"{prefix}%")
			).order_by(CFAPPurchaseOrder.po_number.desc()).first()
			
			if last_po:
				try:
					last_num = int(last_po.po_number[-4:])
					next_num = last_num + 1
				except ValueError:
					next_num = 1
			else:
				next_num = 1
			
			item.po_number = f"{prefix}{next_num:04d}"