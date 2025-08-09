"""
APG Billing Flask-AppBuilder Views

Comprehensive billing management interface with customer portal,
admin console, and analytics dashboards.

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

from flask import flash, redirect, request, url_for
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.charts.views import DirectByChartView
from flask_appbuilder.widgets import ListWidget, ShowWidget
from wtforms import Form, StringField, DecimalField, SelectField, TextAreaField, DateTimeField
from wtforms.validators import DataRequired, Email, NumberRange

from .models import (
	BLCustomer, BLPlan, BLSubscription, BLUsage, BLInvoice, BLPayment,
	SubscriptionStatus, InvoiceStatus, PaymentStatus, BillingCurrency, BillingPeriod, PricingModel
)
from .service import get_billing_service


class BillingCustomerModelView(ModelView):
	"""Customer management view"""
	datamodel = SQLAInterface(BLCustomer)
	
	# List view configuration
	list_columns = ['name', 'email', 'company', 'currency', 'active', 'created_at']
	list_title = "Billing Customers"
	
	# Show view configuration
	show_columns = [
		'id', 'name', 'email', 'company', 'phone', 'currency',
		'billing_address', 'tax_info', 'payment_terms', 'active',
		'credit_limit', 'created_at', 'updated_at'
	]
	show_title = "Customer Details"
	
	# Edit view configuration
	edit_columns = [
		'name', 'email', 'company', 'phone', 'currency',
		'billing_address', 'tax_info', 'payment_terms', 'active', 'credit_limit'
	]
	edit_title = "Edit Customer"
	
	# Add view configuration
	add_columns = [
		'name', 'email', 'company', 'phone', 'currency',
		'billing_address', 'tax_info', 'payment_terms', 'credit_limit'
	]
	add_title = "Add Customer"
	
	# Search and filters
	search_columns = ['name', 'email', 'company']
	base_filters = [['active', lambda: True, lambda: True]]
	
	# Permissions
	base_permissions = ['can_list', 'can_show', 'can_add', 'can_edit', 'can_delete']


class BillingPlanModelView(ModelView):
	"""Billing plan management view"""
	datamodel = SQLAInterface(BLPlan)
	
	# List view configuration
	list_columns = ['name', 'pricing_model', 'base_price', 'currency', 'billing_period', 'active']
	list_title = "Billing Plans"
	
	# Show view configuration
	show_columns = [
		'id', 'name', 'description', 'pricing_model', 'base_price', 'currency',
		'billing_period', 'usage_charges', 'included_usage', 'features', 'limits',
		'trial_period_days', 'active', 'version', 'created_at', 'updated_at'
	]
	show_title = "Plan Details"
	
	# Edit view configuration
	edit_columns = [
		'name', 'description', 'pricing_model', 'base_price', 'currency',
		'billing_period', 'usage_charges', 'included_usage', 'features', 'limits',
		'trial_period_days', 'active'
	]
	edit_title = "Edit Plan"
	
	# Add view configuration
	add_columns = [
		'name', 'description', 'pricing_model', 'base_price', 'currency',
		'billing_period', 'usage_charges', 'included_usage', 'features', 'limits',
		'trial_period_days'
	]
	add_title = "Add Plan"
	
	# Search and filters
	search_columns = ['name', 'description']
	base_filters = [['active', lambda: True, lambda: True]]
	
	# Permissions
	base_permissions = ['can_list', 'can_show', 'can_add', 'can_edit', 'can_delete']


class BillingSubscriptionModelView(ModelView):
	"""Subscription management view"""
	datamodel = SQLAInterface(BLSubscription)
	
	# List view configuration
	list_columns = [
		'customer_id', 'plan_id', 'status', 'current_period_start',
		'current_period_end', 'trial_end', 'created_at'
	]
	list_title = "Subscriptions"
	
	# Show view configuration
	show_columns = [
		'id', 'customer_id', 'plan_id', 'status', 'current_period_start',
		'current_period_end', 'trial_start', 'trial_end', 'price_override',
		'currency_override', 'applied_discounts', 'cancel_at_period_end',
		'cancelled_at', 'cancellation_reason', 'created_at', 'updated_at'
	]
	show_title = "Subscription Details"
	
	# Edit view configuration
	edit_columns = [
		'status', 'price_override', 'currency_override', 'applied_discounts',
		'cancel_at_period_end', 'cancellation_reason'
	]
	edit_title = "Edit Subscription"
	
	# Search and filters
	search_columns = ['customer_id', 'plan_id']
	base_filters = [['status', lambda: SubscriptionStatus.ACTIVE, lambda: SubscriptionStatus.ACTIVE]]
	
	# Custom actions
	@expose('/cancel/<subscription_id>')
	@has_access
	def cancel_subscription(self, subscription_id):
		"""Cancel subscription"""
		billing_service = get_billing_service()
		try:
			user_id = self.appbuilder.sm.user.username  # Get current user
			subscription = billing_service.cancel_subscription(
				user_id=user_id,
				subscription_id=subscription_id,
				cancel_at_period_end=True,
				reason="Admin cancellation"
			)
			flash(f"Subscription {subscription_id} cancelled successfully", "success")
		except Exception as e:
			flash(f"Error cancelling subscription: {e}", "error")
		
		return redirect(url_for('BillingSubscriptionModelView.list'))
	
	# Permissions
	base_permissions = ['can_list', 'can_show', 'can_edit']


class BillingInvoiceModelView(ModelView):
	"""Invoice management view"""
	datamodel = SQLAInterface(BLInvoice)
	
	# List view configuration
	list_columns = [
		'invoice_number', 'customer_id', 'status', 'total', 'currency',
		'invoice_date', 'due_date', 'amount_due'
	]
	list_title = "Invoices"
	
	# Show view configuration
	show_columns = [
		'id', 'invoice_number', 'customer_id', 'subscription_id', 'status',
		'subtotal', 'tax_amount', 'discount_amount', 'total', 'amount_paid',
		'amount_due', 'currency', 'period_start', 'period_end',
		'invoice_date', 'due_date', 'paid_at', 'line_items',
		'created_at', 'updated_at'
	]
	show_title = "Invoice Details"
	
	# Edit view configuration
	edit_columns = ['status', 'due_date']
	edit_title = "Edit Invoice"
	
	# Search and filters
	search_columns = ['invoice_number', 'customer_id']
	base_filters = [['status', lambda: InvoiceStatus.PENDING, lambda: InvoiceStatus.PENDING]]
	
	# Custom actions
	@expose('/send/<invoice_id>')
	@has_access
	def send_invoice(self, invoice_id):
		"""Send invoice to customer"""
		try:
			# Get billing service and invoice
			from .service import get_billing_service
			from .email_services import get_email_service_manager
			
			billing_service = get_billing_service()
			invoice = billing_service.invoices.get(invoice_id)
			
			if not invoice:
				flash(f"Invoice {invoice_id} not found", "error")
				return redirect(url_for('BillingInvoiceModelView.list'))
			
			# Get customer for email address
			customer = billing_service.customers.get(invoice.customer_id)
			if not customer or not customer.email:
				flash(f"Customer email not found for invoice {invoice_id}", "error")
				return redirect(url_for('BillingInvoiceModelView.list'))
			
			# Get email service
			email_manager = get_email_service_manager()
			billing_email_service = email_manager.get_billing_email_manager()
			
			# Prepare invoice data for email
			invoice_data = {
				'customer_name': customer.name,
				'invoice_number': invoice.invoice_number,
				'amount_due': str(invoice.amount_due),
				'currency': invoice.currency.value if hasattr(invoice.currency, 'value') else str(invoice.currency),
				'due_date': invoice.due_date.strftime('%Y-%m-%d') if invoice.due_date else 'N/A',
				'invoice_url': f'/billing/invoices/{invoice_id}',
				'company_name': 'APG Billing System'
			}
			
			# Send invoice email
			result = billing_email_service.send_invoice_email(
				customer.email, invoice_data
			)
			
			if result.get('success'):
				# Update invoice status to sent
				invoice.status = InvoiceStatus.SENT
				invoice.sent_at = datetime.utcnow()
				flash(f"Invoice {invoice.invoice_number} sent successfully to {customer.email}", "success")
			else:
				flash(f"Failed to send invoice: {result.get('error', 'Unknown error')}", "error")
			
		except Exception as e:
			flash(f"Error sending invoice: {e}", "error")
		
		return redirect(url_for('BillingInvoiceModelView.list'))
	
	@expose('/mark_paid/<invoice_id>')
	@has_access
	def mark_paid(self, invoice_id):
		"""Mark invoice as paid"""
		try:
			# Get billing service and invoice
			from .service import get_billing_service
			from .models import InvoiceStatus, PaymentStatus
			
			billing_service = get_billing_service()
			invoice = billing_service.invoices.get(invoice_id)
			
			if not invoice:
				flash(f"Invoice {invoice_id} not found", "error")
				return redirect(url_for('BillingInvoiceModelView.list'))
			
			# Update invoice status
			invoice.status = InvoiceStatus.PAID
			invoice.paid_at = datetime.utcnow()
			invoice.amount_due = Decimal('0')
			
			# Create a manual payment record
			payment_data = {
				'customer_id': invoice.customer_id,
				'invoice_id': invoice.id,
				'amount': invoice.amount,
				'currency': invoice.currency,
				'payment_method': 'manual',
				'status': PaymentStatus.COMPLETED,
				'notes': 'Manually marked as paid via admin interface'
			}
			
			# Create payment record
			payment = billing_service.create_payment(payment_data)
			
			if payment:
				flash(f"Invoice {invoice.invoice_number} marked as paid successfully", "success")
			else:
				flash(f"Invoice status updated but payment record creation failed", "warning")
			
		except Exception as e:
			flash(f"Error marking invoice as paid: {e}", "error")
		
		return redirect(url_for('BillingInvoiceModelView.list'))
	
	# Permissions
	base_permissions = ['can_list', 'can_show', 'can_edit']


class BillingPaymentModelView(ModelView):
	"""Payment management view"""
	datamodel = SQLAInterface(BLPayment)
	
	# List view configuration
	list_columns = [
		'customer_id', 'invoice_id', 'amount', 'currency', 'status',
		'payment_method_type', 'processed_at'
	]
	list_title = "Payments"
	
	# Show view configuration
	show_columns = [
		'id', 'customer_id', 'invoice_id', 'external_id', 'status',
		'amount', 'currency', 'fee_amount', 'net_amount',
		'payment_method_type', 'payment_processor', 'processed_at',
		'settled_at', 'failure_reason', 'refunded', 'refund_amount',
		'disputed', 'risk_score', 'created_at', 'updated_at'
	]
	show_title = "Payment Details"
	
	# Search and filters
	search_columns = ['customer_id', 'invoice_id', 'external_id']
	base_filters = [['status', lambda: PaymentStatus.SUCCEEDED, lambda: PaymentStatus.SUCCEEDED]]
	
	# Custom actions
	@expose('/refund/<payment_id>')
	@has_access
	def refund_payment(self, payment_id):
		"""Refund payment"""
		try:
			# Get billing service and payment
			from .service import get_billing_service
			from .dispute_resolution import get_dispute_resolution_system
			from .models import PaymentStatus
			
			billing_service = get_billing_service()
			payment = billing_service.payments.get(payment_id)
			
			if not payment:
				flash(f"Payment {payment_id} not found", "error")
				return redirect(url_for('BillingPaymentModelView.list'))
			
			if payment.status != PaymentStatus.COMPLETED:
				flash(f"Payment {payment_id} cannot be refunded - status: {payment.status.value}", "error")
				return redirect(url_for('BillingPaymentModelView.list'))
			
			# Get dispute resolution system for refund processing
			dispute_system = get_dispute_resolution_system()
			
			# Create refund dispute data
			refund_data = {
				'dispute_type': 'refund_request',
				'customer_id': payment.customer_id,
				'payment_id': payment.id,
				'amount': payment.amount,
				'currency': payment.currency,
				'reason': 'Manual refund requested via admin interface',
				'requested_by': 'admin_user',  # In production, get from session
				'priority': 'normal'
			}
			
			# Create dispute for tracking
			dispute = await dispute_system.create_dispute(refund_data)
			
			if dispute:
				# Process the refund immediately
				refund_result = await dispute_system._process_stripe_refund(
					payment, payment.amount, dispute
				)
				
				if refund_result.get('success'):
					# Update payment status
					payment.status = PaymentStatus.REFUNDED
					payment.refunded_at = datetime.utcnow()
					
					# Update related invoice if exists
					if hasattr(payment, 'invoice_id') and payment.invoice_id:
						invoice = billing_service.invoices.get(payment.invoice_id)
						if invoice:
							invoice.amount_due += payment.amount
							if invoice.amount_due >= invoice.amount:
								invoice.status = InvoiceStatus.OUTSTANDING
					
					flash(f"Payment {payment_id} refunded successfully: ${payment.amount}", "success")
				else:
					flash(f"Refund failed: {refund_result.get('error', 'Unknown error')}", "error")
			else:
				flash(f"Failed to create refund dispute for payment {payment_id}", "error")
				
		except Exception as e:
			flash(f"Error refunding payment: {e}", "error")
		
		return redirect(url_for('BillingPaymentModelView.list'))
	
	# Permissions
	base_permissions = ['can_list', 'can_show']


class BillingUsageModelView(ModelView):
	"""Usage tracking view"""
	datamodel = SQLAInterface(BLUsage)
	
	# List view configuration
	list_columns = [
		'subscription_id', 'metric_name', 'quantity', 'unit',
		'timestamp', 'processed'
	]
	list_title = "Usage Records"
	
	# Show view configuration
	show_columns = [
		'id', 'subscription_id', 'customer_id', 'metric_name', 'quantity',
		'unit', 'timestamp', 'billing_period_start', 'billing_period_end',
		'unit_price', 'total_amount', 'currency', 'source_system',
		'processed', 'processed_at', 'created_at'
	]
	show_title = "Usage Details"
	
	# Search and filters
	search_columns = ['subscription_id', 'customer_id', 'metric_name']
	base_filters = [['processed', lambda: False, lambda: False]]
	
	# Permissions
	base_permissions = ['can_list', 'can_show']


class BillingDashboardView(BaseView):
	"""Billing dashboard with analytics and KPIs"""
	
	default_view = 'dashboard'
	
	@expose('/')
	@has_access
	def dashboard(self):
		"""Main billing dashboard"""
		billing_service = get_billing_service()
		
		try:
			# Get current user
			user_id = self.appbuilder.sm.user.username
			
			# Get billing analytics for last 30 days
			analytics = billing_service.get_billing_analytics(user_id)
			
			# Get service status
			status = billing_service.get_service_status()
			
			return self.render_template(
				'billing/dashboard.html',
				analytics=analytics,
				status=status,
				title="Billing Dashboard"
			)
			
		except Exception as e:
			flash(f"Error loading dashboard: {e}", "error")
			return self.render_template(
				'billing/dashboard.html',
				analytics={},
				status={},
				title="Billing Dashboard"
			)


class BillingReportsView(BaseView):
	"""Billing reports and analytics"""
	
	default_view = 'revenue_report'
	
	@expose('/revenue')
	@has_access
	def revenue_report(self):
		"""Revenue analytics report"""
		billing_service = get_billing_service()
		
		try:
			user_id = self.appbuilder.sm.user.username
			
			# Get revenue analytics
			analytics = billing_service.get_billing_analytics(user_id)
			
			return self.render_template(
				'billing/revenue_report.html',
				analytics=analytics,
				title="Revenue Report"
			)
			
		except Exception as e:
			flash(f"Error loading revenue report: {e}", "error")
			return self.render_template(
				'billing/revenue_report.html',
				analytics={},
				title="Revenue Report"
			)
	
	@expose('/customers')
	@has_access
	def customer_report(self):
		"""Customer analytics report"""
		billing_service = get_billing_service()
		
		try:
			user_id = self.appbuilder.sm.user.username
			
			# Get customer list and analytics
			customers = billing_service.list_customers(user_id)
			
			return self.render_template(
				'billing/customer_report.html',
				customers=customers,
				title="Customer Report"
			)
			
		except Exception as e:
			flash(f"Error loading customer report: {e}", "error")
			return self.render_template(
				'billing/customer_report.html',
				customers=[],
				title="Customer Report"
			)
	
	@expose('/subscriptions')
	@has_access
	def subscription_report(self):
		"""Subscription analytics report"""
		return self.render_template(
			'billing/subscription_report.html',
			title="Subscription Report"
		)


class BillingCustomerPortalView(BaseView):
	"""Customer self-service portal"""
	
	default_view = 'portal'
	
	@expose('/')
	@has_access
	def portal(self):
		"""Customer billing portal"""
		billing_service = get_billing_service()
		
		try:
			user_id = self.appbuilder.sm.user.username
			
			# Get customer's subscriptions and invoices
			# In real implementation, map user to customer
			customer_id = user_id  # Simplified mapping
			
			# Get customer data
			customer = billing_service.get_customer(user_id, customer_id)
			
			return self.render_template(
				'billing/customer_portal.html',
				customer=customer,
				title="Billing Portal"
			)
			
		except Exception as e:
			flash(f"Error loading portal: {e}", "error")
			return self.render_template(
				'billing/customer_portal.html',
				customer=None,
				title="Billing Portal"
			)
	
	@expose('/invoices')
	@has_access
	def customer_invoices(self):
		"""Customer invoice history"""
		return self.render_template(
			'billing/customer_invoices.html',
			title="Invoice History"
		)
	
	@expose('/usage')
	@has_access
	def customer_usage(self):
		"""Customer usage dashboard"""
		return self.render_template(
			'billing/customer_usage.html',
			title="Usage Dashboard"
		)


class RevenueByMonthChartView(DirectByChartView):
	"""Revenue by month chart"""
	chart_title = "Revenue by Month"
	chart_type = "LineChart"
	direct_columns = {
		'month': ('Invoice', 'invoice_date'),
		'revenue': ('Invoice', 'total')
	}
	base_order = ('month', 'asc')


class SubscriptionStatusChartView(DirectByChartView):
	"""Subscription status distribution chart"""
	chart_title = "Subscription Status Distribution"
	chart_type = "PieChart"
	direct_columns = {
		'status': ('Subscription', 'status'),
		'count': ('Subscription', 'id')
	}
	group_by_columns = ['status']


# Forms for custom operations

class CreateSubscriptionForm(Form):
	"""Form for creating subscriptions"""
	customer_id = StringField('Customer ID', validators=[DataRequired()])
	plan_id = StringField('Plan ID', validators=[DataRequired()])
	trial_period_days = StringField('Trial Period (days)')
	payment_method_id = StringField('Payment Method ID')


class SubmitUsageForm(Form):
	"""Form for submitting usage data"""
	subscription_id = StringField('Subscription ID', validators=[DataRequired()])
	metric_name = StringField('Metric Name', validators=[DataRequired()])
	quantity = DecimalField('Quantity', validators=[DataRequired(), NumberRange(min=0)])


class GenerateInvoiceForm(Form):
	"""Form for generating invoices"""
	subscription_id = StringField('Subscription ID', validators=[DataRequired()])
	billing_period_start = DateTimeField('Period Start', validators=[DataRequired()])
	billing_period_end = DateTimeField('Period End', validators=[DataRequired()])
	include_usage = SelectField('Include Usage', choices=[('true', 'Yes'), ('false', 'No')], default='true')
	auto_finalize = SelectField('Auto Finalize', choices=[('true', 'Yes'), ('false', 'No')], default='false')


# Export view classes for blueprint registration
__all__ = [
	'BillingCustomerModelView',
	'BillingPlanModelView', 
	'BillingSubscriptionModelView',
	'BillingInvoiceModelView',
	'BillingPaymentModelView',
	'BillingUsageModelView',
	'BillingDashboardView',
	'BillingReportsView',
	'BillingCustomerPortalView',
	'RevenueByMonthChartView',
	'SubscriptionStatusChartView'
]