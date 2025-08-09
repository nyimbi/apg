"""
APG Billing REST API

Comprehensive REST API endpoints for billing operations including
subscription management, usage tracking, invoice generation, and payments.

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

from flask import Blueprint, request, jsonify, current_app
from flask_restx import Api, Resource, fields, Namespace
from flask_restx.reqparse import RequestParser

from .service import get_billing_service, BillingError, SubscriptionError, UsageError, InvoiceError, PaymentError
from .models import (
	CreateSubscriptionRequest, UsageSubmissionRequest, InvoiceGenerationRequest,
	SubscriptionStatus, InvoiceStatus, PaymentStatus, BillingCurrency
)


# Create API blueprint
billing_api_bp = Blueprint('billing_api', __name__, url_prefix='/api/v1/billing')
api = Api(
	billing_api_bp,
	doc='/docs/',
	title='APG Billing API',
	version='1.0',
	description='Comprehensive billing and subscription management API',
	contact='nyimbi@gmail.com',
	contact_url='https://www.datacraft.co.ke'
)

# Create namespaces
customers_ns = Namespace('customers', description='Customer management operations')
plans_ns = Namespace('plans', description='Billing plan operations')
subscriptions_ns = Namespace('subscriptions', description='Subscription management')
usage_ns = Namespace('usage', description='Usage tracking operations')
invoices_ns = Namespace('invoices', description='Invoice management')
payments_ns = Namespace('payments', description='Payment processing')
analytics_ns = Namespace('analytics', description='Billing analytics and reporting')

api.add_namespace(customers_ns, path='/customers')
api.add_namespace(plans_ns, path='/plans')
api.add_namespace(subscriptions_ns, path='/subscriptions')
api.add_namespace(usage_ns, path='/usage')
api.add_namespace(invoices_ns, path='/invoices')
api.add_namespace(payments_ns, path='/payments')
api.add_namespace(analytics_ns, path='/analytics')


# Define API models for documentation
customer_model = api.model('Customer', {
	'id': fields.String(required=True, description='Customer ID'),
	'name': fields.String(required=True, description='Customer name'),
	'email': fields.String(required=True, description='Customer email'),
	'company': fields.String(description='Company name'),
	'phone': fields.String(description='Phone number'),
	'currency': fields.String(description='Default currency'),
	'active': fields.Boolean(description='Customer active status'),
	'created_at': fields.DateTime(description='Creation timestamp'),
	'updated_at': fields.DateTime(description='Last update timestamp')
})

plan_model = api.model('Plan', {
	'id': fields.String(required=True, description='Plan ID'),
	'name': fields.String(required=True, description='Plan name'),
	'description': fields.String(description='Plan description'),
	'pricing_model': fields.String(description='Pricing model'),
	'base_price': fields.Float(description='Base price'),
	'currency': fields.String(description='Plan currency'),
	'billing_period': fields.String(description='Billing period'),
	'features': fields.List(fields.String, description='Plan features'),
	'trial_period_days': fields.Integer(description='Trial period in days'),
	'active': fields.Boolean(description='Plan active status')
})

subscription_model = api.model('Subscription', {
	'id': fields.String(required=True, description='Subscription ID'),
	'customer_id': fields.String(required=True, description='Customer ID'),
	'plan_id': fields.String(required=True, description='Plan ID'),
	'status': fields.String(description='Subscription status'),
	'current_period_start': fields.DateTime(description='Current period start'),
	'current_period_end': fields.DateTime(description='Current period end'),
	'trial_start': fields.DateTime(description='Trial start'),
	'trial_end': fields.DateTime(description='Trial end'),
	'cancel_at_period_end': fields.Boolean(description='Cancel at period end'),
	'created_at': fields.DateTime(description='Creation timestamp')
})

usage_model = api.model('Usage', {
	'id': fields.String(required=True, description='Usage record ID'),
	'subscription_id': fields.String(required=True, description='Subscription ID'),
	'metric_name': fields.String(required=True, description='Usage metric name'),
	'quantity': fields.Float(required=True, description='Usage quantity'),
	'unit': fields.String(description='Usage unit'),
	'timestamp': fields.DateTime(description='Usage timestamp'),
	'processed': fields.Boolean(description='Processing status')
})

invoice_model = api.model('Invoice', {
	'id': fields.String(required=True, description='Invoice ID'),
	'invoice_number': fields.String(required=True, description='Invoice number'),
	'customer_id': fields.String(required=True, description='Customer ID'),
	'subscription_id': fields.String(description='Subscription ID'),
	'status': fields.String(description='Invoice status'),
	'total': fields.Float(description='Invoice total'),
	'amount_due': fields.Float(description='Amount due'),
	'currency': fields.String(description='Invoice currency'),
	'invoice_date': fields.DateTime(description='Invoice date'),
	'due_date': fields.DateTime(description='Due date'),
	'paid_at': fields.DateTime(description='Payment date')
})

payment_model = api.model('Payment', {
	'id': fields.String(required=True, description='Payment ID'),
	'customer_id': fields.String(required=True, description='Customer ID'),
	'invoice_id': fields.String(description='Invoice ID'),
	'amount': fields.Float(required=True, description='Payment amount'),
	'currency': fields.String(description='Payment currency'),
	'status': fields.String(description='Payment status'),
	'payment_method_type': fields.String(description='Payment method'),
	'processed_at': fields.DateTime(description='Processing timestamp')
})


# Helper functions
def get_current_user_id() -> str:
	"""Get current user ID from request context"""
	# In a real application, this would extract user ID from JWT token or session
	return request.headers.get('X-User-ID', 'api-user')


def handle_billing_error(func):
	"""Decorator to handle billing service errors"""
	def wrapper(*args, **kwargs):
		try:
			return func(*args, **kwargs)
		except BillingError as e:
			return {'error': str(e), 'error_code': e.error_code}, 400
		except SubscriptionError as e:
			return {'error': str(e), 'error_code': e.error_code}, 400
		except UsageError as e:
			return {'error': str(e), 'error_code': e.error_code}, 400
		except InvoiceError as e:
			return {'error': str(e), 'error_code': e.error_code}, 400
		except PaymentError as e:
			return {'error': str(e), 'error_code': e.error_code}, 400
		except Exception as e:
			current_app.logger.error(f"Unexpected error: {e}")
			return {'error': 'Internal server error'}, 500
	return wrapper


# Customer API endpoints
@customers_ns.route('/')
class CustomerListAPI(Resource):
	@customers_ns.doc('list_customers')
	@customers_ns.marshal_list_with(customer_model)
	@handle_billing_error
	async def get(self):
		"""List all customers"""
		billing_service = get_billing_service()
		user_id = get_current_user_id()
		
		# Parse query parameters
		active_only = request.args.get('active', 'true').lower() == 'true'
		currency = request.args.get('currency')
		
		filters = {}
		if active_only:
			filters['active'] = True
		if currency:
			filters['currency'] = currency
		
		customers = await billing_service.list_customers(user_id, filters)
		return [customer.model_dump() for customer in customers]
	
	@customers_ns.doc('create_customer')
	@customers_ns.expect(customer_model)
	@customers_ns.marshal_with(customer_model, code=201)
	@handle_billing_error
	async def post(self):
		"""Create a new customer"""
		billing_service = get_billing_service()
		user_id = get_current_user_id()
		
		customer_data = request.get_json()
		customer = await billing_service.create_customer(user_id, customer_data)
		
		return customer.model_dump(), 201


@customers_ns.route('/<string:customer_id>')
class CustomerAPI(Resource):
	@customers_ns.doc('get_customer')
	@customers_ns.marshal_with(customer_model)
	@handle_billing_error
	async def get(self, customer_id):
		"""Get customer by ID"""
		billing_service = get_billing_service()
		user_id = get_current_user_id()
		
		customer = await billing_service.get_customer(user_id, customer_id)
		if not customer:
			return {'error': 'Customer not found'}, 404
		
		return customer.model_dump()


# Plan API endpoints
@plans_ns.route('/')
class PlanListAPI(Resource):
	@plans_ns.doc('list_plans')
	@plans_ns.marshal_list_with(plan_model)
	@handle_billing_error
	async def get(self):
		"""List all billing plans"""
		billing_service = get_billing_service()
		user_id = get_current_user_id()
		
		# In a real implementation, this would call list_plans method
		plans = list(billing_service.plans.values())
		return [plan.model_dump() for plan in plans]
	
	@plans_ns.doc('create_plan')
	@plans_ns.expect(plan_model)
	@plans_ns.marshal_with(plan_model, code=201)
	@handle_billing_error
	async def post(self):
		"""Create a new billing plan"""
		billing_service = get_billing_service()
		user_id = get_current_user_id()
		
		plan_data = request.get_json()
		plan = await billing_service.create_plan(user_id, plan_data)
		
		return plan.model_dump(), 201


@plans_ns.route('/<string:plan_id>')
class PlanAPI(Resource):
	@plans_ns.doc('get_plan')
	@plans_ns.marshal_with(plan_model)
	@handle_billing_error
	async def get(self, plan_id):
		"""Get plan by ID"""
		billing_service = get_billing_service()
		user_id = get_current_user_id()
		
		plan = await billing_service.get_plan(user_id, plan_id)
		if not plan:
			return {'error': 'Plan not found'}, 404
		
		return plan.model_dump()


# Subscription API endpoints
@subscriptions_ns.route('/')
class SubscriptionListAPI(Resource):
	@subscriptions_ns.doc('list_subscriptions')
	@subscriptions_ns.marshal_list_with(subscription_model)
	@handle_billing_error
	async def get(self):
		"""List subscriptions"""
		billing_service = get_billing_service()
		user_id = get_current_user_id()
		
		# Parse query parameters
		customer_id = request.args.get('customer_id')
		status = request.args.get('status')
		
		# Filter subscriptions
		subscriptions = []
		for subscription in billing_service.subscriptions.values():
			if customer_id and subscription.customer_id != customer_id:
				continue
			if status and subscription.status.value != status:
				continue
			subscriptions.append(subscription)
		
		return [sub.model_dump() for sub in subscriptions]
	
	@subscriptions_ns.doc('create_subscription')
	@subscriptions_ns.expect(subscription_model)
	@subscriptions_ns.marshal_with(subscription_model, code=201)
	@handle_billing_error
	async def post(self):
		"""Create a new subscription"""
		billing_service = get_billing_service()
		user_id = get_current_user_id()
		
		data = request.get_json()
		request_obj = CreateSubscriptionRequest(**data)
		
		subscription = await billing_service.create_subscription(user_id, request_obj)
		return subscription.model_dump(), 201


@subscriptions_ns.route('/<string:subscription_id>')
class SubscriptionAPI(Resource):
	@subscriptions_ns.doc('get_subscription')
	@subscriptions_ns.marshal_with(subscription_model)
	@handle_billing_error
	async def get(self, subscription_id):
		"""Get subscription by ID"""
		billing_service = get_billing_service()
		user_id = get_current_user_id()
		
		subscription = await billing_service.get_subscription(user_id, subscription_id)
		if not subscription:
			return {'error': 'Subscription not found'}, 404
		
		return subscription.model_dump()
	
	@subscriptions_ns.doc('update_subscription')
	@subscriptions_ns.marshal_with(subscription_model)
	@handle_billing_error
	async def put(self, subscription_id):
		"""Update subscription"""
		billing_service = get_billing_service()
		user_id = get_current_user_id()
		
		updates = request.get_json()
		subscription = await billing_service.update_subscription(user_id, subscription_id, updates)
		
		return subscription.model_dump()


@subscriptions_ns.route('/<string:subscription_id>/cancel')
class SubscriptionCancelAPI(Resource):
	@subscriptions_ns.doc('cancel_subscription')
	@subscriptions_ns.marshal_with(subscription_model)
	@handle_billing_error
	async def post(self, subscription_id):
		"""Cancel subscription"""
		billing_service = get_billing_service()
		user_id = get_current_user_id()
		
		data = request.get_json() or {}
		cancel_at_period_end = data.get('cancel_at_period_end', True)
		reason = data.get('reason', 'API cancellation')
		
		subscription = await billing_service.cancel_subscription(
			user_id=user_id,
			subscription_id=subscription_id,
			cancel_at_period_end=cancel_at_period_end,
			reason=reason
		)
		
		return subscription.model_dump()


# Usage API endpoints
@usage_ns.route('/')
class UsageAPI(Resource):
	@usage_ns.doc('submit_usage')
	@usage_ns.expect(usage_model)
	@usage_ns.marshal_with(usage_model, code=201)
	@handle_billing_error
	async def post(self):
		"""Submit usage data"""
		billing_service = get_billing_service()
		user_id = get_current_user_id()
		
		data = request.get_json()
		request_obj = UsageSubmissionRequest(**data)
		
		usage = await billing_service.submit_usage(user_id, request_obj)
		return usage.model_dump(), 201


@usage_ns.route('/<string:subscription_id>/summary')
class UsageSummaryAPI(Resource):
	@usage_ns.doc('get_usage_summary')
	@handle_billing_error
	async def get(self, subscription_id):
		"""Get usage summary for subscription"""
		billing_service = get_billing_service()
		user_id = get_current_user_id()
		
		# Parse date parameters
		period_start_str = request.args.get('period_start')
		period_end_str = request.args.get('period_end')
		
		period_start = None
		period_end = None
		
		if period_start_str:
			period_start = datetime.fromisoformat(period_start_str)
		if period_end_str:
			period_end = datetime.fromisoformat(period_end_str)
		
		summary = await billing_service.get_usage_summary(
			user_id, subscription_id, period_start, period_end
		)
		
		return summary


# Invoice API endpoints
@invoices_ns.route('/')
class InvoiceListAPI(Resource):
	@invoices_ns.doc('list_invoices')
	@invoices_ns.marshal_list_with(invoice_model)
	@handle_billing_error
	async def get(self):
		"""List invoices"""
		billing_service = get_billing_service()
		user_id = get_current_user_id()
		
		# Parse query parameters
		customer_id = request.args.get('customer_id')
		status = request.args.get('status')
		
		# Filter invoices
		invoices = []
		for invoice in billing_service.invoices.values():
			if customer_id and invoice.customer_id != customer_id:
				continue
			if status and invoice.status.value != status:
				continue
			invoices.append(invoice)
		
		return [inv.model_dump() for inv in invoices]
	
	@invoices_ns.doc('generate_invoice')
	@invoices_ns.marshal_with(invoice_model, code=201)
	@handle_billing_error
	async def post(self):
		"""Generate invoice"""
		billing_service = get_billing_service()
		user_id = get_current_user_id()
		
		data = request.get_json()
		
		# Convert date strings to datetime objects
		if 'billing_period_start' in data:
			data['billing_period_start'] = datetime.fromisoformat(data['billing_period_start'])
		if 'billing_period_end' in data:
			data['billing_period_end'] = datetime.fromisoformat(data['billing_period_end'])
		
		request_obj = InvoiceGenerationRequest(**data)
		
		invoice = await billing_service.generate_invoice(user_id, request_obj)
		return invoice.model_dump(), 201


@invoices_ns.route('/<string:invoice_id>')
class InvoiceAPI(Resource):
	@invoices_ns.doc('get_invoice')
	@invoices_ns.marshal_with(invoice_model)
	@handle_billing_error
	async def get(self, invoice_id):
		"""Get invoice by ID"""
		billing_service = get_billing_service()
		user_id = get_current_user_id()
		
		invoice = await billing_service.get_invoice(user_id, invoice_id)
		if not invoice:
			return {'error': 'Invoice not found'}, 404
		
		return invoice.model_dump()


# Payment API endpoints
@payments_ns.route('/')
class PaymentListAPI(Resource):
	@payments_ns.doc('list_payments')
	@payments_ns.marshal_list_with(payment_model)
	@handle_billing_error
	async def get(self):
		"""List payments"""
		billing_service = get_billing_service()
		user_id = get_current_user_id()
		
		# Parse query parameters
		customer_id = request.args.get('customer_id')
		status = request.args.get('status')
		
		# Filter payments
		payments = []
		for payment in billing_service.payments.values():
			if customer_id and payment.customer_id != customer_id:
				continue
			if status and payment.status.value != status:
				continue
			payments.append(payment)
		
		return [pay.model_dump() for pay in payments]
	
	@payments_ns.doc('process_payment')
	@payments_ns.expect(payment_model)
	@payments_ns.marshal_with(payment_model, code=201)
	@handle_billing_error
	async def post(self):
		"""Process payment"""
		billing_service = get_billing_service()
		user_id = get_current_user_id()
		
		payment_data = request.get_json()
		payment = await billing_service.process_payment(user_id, payment_data)
		
		return payment.model_dump(), 201


@payments_ns.route('/<string:payment_id>')
class PaymentAPI(Resource):
	@payments_ns.doc('get_payment')
	@payments_ns.marshal_with(payment_model)
	@handle_billing_error
	async def get(self, payment_id):
		"""Get payment by ID"""
		billing_service = get_billing_service()
		
		payment = billing_service.payments.get(payment_id)
		if not payment:
			return {'error': 'Payment not found'}, 404
		
		return payment.model_dump()


# Analytics API endpoints
@analytics_ns.route('/billing')
class BillingAnalyticsAPI(Resource):
	@analytics_ns.doc('get_billing_analytics')
	@handle_billing_error
	async def get(self):
		"""Get billing analytics"""
		billing_service = get_billing_service()
		user_id = get_current_user_id()
		
		# Parse date parameters
		period_start_str = request.args.get('period_start')
		period_end_str = request.args.get('period_end')
		
		period_start = None
		period_end = None
		
		if period_start_str:
			period_start = datetime.fromisoformat(period_start_str)
		if period_end_str:
			period_end = datetime.fromisoformat(period_end_str)
		
		analytics = await billing_service.get_billing_analytics(
			user_id=user_id,
			period_start=period_start,
			period_end=period_end
		)
		
		return analytics


@analytics_ns.route('/revenue')
class RevenueAnalyticsAPI(Resource):
	@analytics_ns.doc('get_revenue_analytics')
	@handle_billing_error
	async def get(self):
		"""Get revenue analytics"""
		billing_service = get_billing_service()
		user_id = get_current_user_id()
		
		analytics = await billing_service.get_billing_analytics(user_id)
		
		# Extract revenue-specific metrics
		revenue_analytics = {
			'total_revenue': analytics['metrics']['total_revenue'],
			'revenue_by_day': analytics['revenue_by_day'],
			'currency_breakdown': analytics['currency_breakdown'],
			'payment_success_rate': analytics['metrics']['payment_success_rate'],
			'period_start': analytics['period_start'],
			'period_end': analytics['period_end']
		}
		
		return revenue_analytics


@analytics_ns.route('/customers')
class CustomerAnalyticsAPI(Resource):
	@analytics_ns.doc('get_customer_analytics')
	@handle_billing_error
	async def get(self):
		"""Get customer analytics"""
		billing_service = get_billing_service()
		user_id = get_current_user_id()
		
		analytics = await billing_service.get_billing_analytics(user_id)
		
		# Extract customer-specific metrics
		customer_analytics = {
			'total_customers': analytics['metrics']['total_customers'],
			'active_subscriptions': analytics['metrics']['active_subscriptions'],
			'subscription_distribution': {
				'active': sum(1 for s in billing_service.subscriptions.values() if s.status == SubscriptionStatus.ACTIVE),
				'trial': sum(1 for s in billing_service.subscriptions.values() if s.status == SubscriptionStatus.TRIAL),
				'cancelled': sum(1 for s in billing_service.subscriptions.values() if s.status == SubscriptionStatus.CANCELLED)
			},
			'period_start': analytics['period_start'],
			'period_end': analytics['period_end']
		}
		
		return customer_analytics


# Service status endpoint
@api.route('/status')
class ServiceStatusAPI(Resource):
	@api.doc('get_service_status')
	@handle_billing_error
	async def get(self):
		"""Get billing service status"""
		billing_service = get_billing_service()
		status = await billing_service.get_service_status()
		return status


# Error handlers
@api.errorhandler(BillingError)
def handle_billing_error(error):
	return {'error': str(error), 'error_code': getattr(error, 'error_code', 'billing_error')}, 400


@api.errorhandler(ValueError)
def handle_value_error(error):
	return {'error': str(error)}, 400


@api.errorhandler(Exception)
def handle_generic_error(error):
	current_app.logger.error(f"Unexpected API error: {error}")
	return {'error': 'Internal server error'}, 500


# Export the blueprint
__all__ = ['billing_api_bp']