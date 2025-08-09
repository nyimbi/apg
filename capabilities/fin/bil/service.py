"""
APG Billing Service

Comprehensive billing service with subscription management, usage tracking,
invoice generation, and revenue optimization using AI-powered intelligence.

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from uuid_extensions import uuid7str

from .models import (
	BLCustomer, BLPlan, BLSubscription, BLUsage, BLInvoice, BLPayment,
	BLPricingRule, BLTax, BLDiscount, BLRevenue,
	SubscriptionStatus, InvoiceStatus, PaymentStatus, BillingCurrency,
	CreateSubscriptionRequest, UsageSubmissionRequest, InvoiceGenerationRequest
)
from .payment_processors import get_payment_processor_manager
from .tax_services import get_tax_service_manager
from .email_services import get_email_service_manager
from .pricing_engine import get_pricing_engine
from .revenue_recognition import get_revenue_recognition_engine
from .webhook_system import get_webhook_system, emit_payment_succeeded, emit_payment_failed, emit_invoice_paid
from .cac_analytics import get_cac_analytics_engine
from .ml_churn_prediction import get_churn_prediction_engine
from .dunning_management import get_dunning_management_system
from .audit_compliance import get_audit_compliance_system, AuditEventType


class BillingError(Exception):
	"""Base billing error"""
	
	def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
		super().__init__(message)
		self.error_code = error_code
		self.details = details or {}
		self.timestamp = datetime.utcnow()


class SubscriptionError(BillingError):
	"""Subscription-related errors"""
	pass


class UsageError(BillingError):
	"""Usage tracking errors"""
	pass


class InvoiceError(BillingError):
	"""Invoice processing errors"""
	pass


class PaymentError(BillingError):
	"""Payment processing errors"""
	pass


class BillingService:
	"""
	Core billing service providing comprehensive billing functionality
	
	Features:
	- Multi-tenant subscription management
	- Real-time usage tracking and aggregation
	- Intelligent invoice generation
	- Payment processing integration
	- Revenue optimization with AI
	- Tax calculation and compliance
	"""
	
	def __init__(self):
		self.customers: Dict[str, BLCustomer] = {}
		self.plans: Dict[str, BLPlan] = {}
		self.subscriptions: Dict[str, BLSubscription] = {}
		self.usage_records: List[BLUsage] = []
		self.invoices: Dict[str, BLInvoice] = {}
		self.payments: Dict[str, BLPayment] = {}
		self.pricing_rules: Dict[str, BLPricingRule] = {}
		self.discounts: Dict[str, BLDiscount] = {}
		self.revenue_records: List[BLRevenue] = []
		
		# Service components
		self.logger = logging.getLogger(f"{__name__}.BillingService")
		self.payment_manager = get_payment_processor_manager()
		self.tax_manager = get_tax_service_manager()
		self.email_manager = get_email_service_manager()
		self.pricing_engine = get_pricing_engine()
		self.revenue_engine = get_revenue_recognition_engine()
		self.webhook_system = get_webhook_system()
		self.cac_engine = get_cac_analytics_engine()
		self.churn_engine = get_churn_prediction_engine()
		self.dunning_system = get_dunning_management_system()
		self.audit_system = get_audit_compliance_system()
		
		# User-to-customer mapping for multi-tenant access control
		self.user_customer_mapping: Dict[str, str] = {}  # user_id -> customer_id
		self.customer_users_mapping: Dict[str, List[str]] = {}  # customer_id -> list of user_ids
		
		# APG integration flags
		self._auth_service_available = False
		self._audit_service_available = False
		self._ai_orchestration_available = False
		
		# Start background tasks
		asyncio.create_task(self._initialize_billing_integrations())
		asyncio.create_task(self._start_billing_engine())
		asyncio.create_task(self._start_revenue_recognition())
	
	async def _initialize_billing_integrations(self) -> None:
		"""Initialize APG billing integrations"""
		try:
			await self._initialize_auth_service()
			await self._initialize_audit_service()
			await self._initialize_ai_orchestration()
		except Exception as e:
			self.logger.warning(f"Some billing integrations not available: {e}")
	
	async def _initialize_auth_service(self) -> None:
		"""Initialize auth service integration"""
		try:
			from capabilities.common.auth_rbac import get_auth_service
			self.auth_service = get_auth_service()
			self._auth_service_available = True
			self.logger.info("✅ auth_rbac integration initialized")
		except ImportError:
			self.logger.warning("⚠️  auth_rbac service not available")
	
	async def _initialize_audit_service(self) -> None:
		"""Initialize audit service integration"""
		try:
			from capabilities.common.audit_compliance import get_audit_service
			self.audit_service = get_audit_service()
			self._audit_service_available = True
			self.logger.info("✅ audit_compliance integration initialized")
		except ImportError:
			self.logger.warning("⚠️  audit_compliance service not available")
	
	async def _initialize_ai_orchestration(self) -> None:
		"""Initialize AI orchestration for billing intelligence"""
		try:
			from capabilities.common.ai_orchestration import get_orchestration_service
			self.ai_orchestration = get_orchestration_service()
			self._ai_orchestration_available = True
			self.logger.info("✅ ai_orchestration integration initialized")
		except ImportError:
			self.logger.warning("⚠️  ai_orchestration service not available")
	
	async def _check_billing_permissions(self, user_id: str, operation: str, resource_id: str = None) -> bool:
		"""Check user permissions for billing operations"""
		if not self._auth_service_available:
			self.logger.warning("Auth service unavailable, allowing operation")
			return True
		
		try:
			has_permission = await self.auth_service.check_permission(user_id, f"billing:{operation}")
			if not has_permission:
				self.logger.warning(f"Permission denied: {user_id} cannot {operation} billing resource {resource_id}")
			return has_permission
		except Exception as e:
			self.logger.error(f"Error checking billing permissions: {e}")
			return False
	
	async def _audit_billing_action(self, action: str, user_id: str, resource_id: str, details: Dict[str, Any]) -> None:
		"""Audit billing actions for compliance using real audit system"""
		try:
			# Map billing actions to audit event types
			event_type_mapping = {
				'create_customer': AuditEventType.CUSTOMER_CREATED,
				'update_customer': AuditEventType.CUSTOMER_UPDATED,
				'delete_customer': AuditEventType.CUSTOMER_DELETED,
				'create_subscription': AuditEventType.SUBSCRIPTION_CREATED,
				'update_subscription': AuditEventType.SUBSCRIPTION_UPDATED,
				'cancel_subscription': AuditEventType.SUBSCRIPTION_CANCELLED,
				'process_payment': AuditEventType.PAYMENT_PROCESSED,
				'refund_payment': AuditEventType.PAYMENT_REFUNDED,
				'generate_invoice': AuditEventType.INVOICE_GENERATED,
				'void_invoice': AuditEventType.INVOICE_VOIDED
			}
			
			event_type = event_type_mapping.get(action, 'unknown')
			
			# Create audit event
			audit_data = {
				'event_type': event_type.value if hasattr(event_type, 'value') else 'unknown',
				'user_id': user_id,
				'tenant_id': await self._get_user_tenant(user_id),
				'resource_type': 'billing',
				'resource_id': resource_id,
				'action': action,
				'description': f'Billing action: {action}',
				'metadata': details,
				'severity': 'medium'
			}
			
			# Add sensitive data encryption if needed
			sensitive_fields = ['payment_method', 'card_data', 'bank_account']
			sensitive_data = {}
			for field in sensitive_fields:
				if field in details:
					sensitive_data[field] = details.pop(field)
			
			if sensitive_data:
				audit_data['sensitive_data'] = sensitive_data
			
			await self.audit_system.log_audit_event(audit_data)
			
		except Exception as e:
			self.logger.error(f"Error auditing billing action: {e}")
	
	# Customer Management
	
	async def create_customer(self, user_id: str, customer_data: Dict[str, Any]) -> BLCustomer:
		"""Create new billing customer"""
		assert user_id, "user_id is required"
		assert customer_data, "customer_data is required"
		
		if not await self._check_billing_permissions(user_id, "create_customer"):
			raise BillingError(f"User {user_id} not authorized to create customers", "permission_denied")
		
		try:
			# Add tenant and creation info
			customer_data.update({
				"tenant_id": await self._get_user_tenant(user_id)
			})
			
			customer = BLCustomer(**customer_data)
			self.customers[customer.id] = customer
			
			# Audit the creation
			await self._audit_billing_action(
				action="create_customer",
				user_id=user_id,
				resource_id=customer.id,
				details={
					"customer_name": customer.name,
					"customer_email": customer.email,
					"currency": customer.currency.value
				}
			)
			
			self.logger.info(f"Customer created: {customer.name} ({customer.id})")
			return customer
			
		except Exception as e:
			self.logger.error(f"Failed to create customer: {e}")
			raise BillingError(f"Customer creation failed: {e}", "creation_failed")
	
	async def get_customer(self, user_id: str, customer_id: str) -> Optional[BLCustomer]:
		"""Get customer by ID"""
		assert user_id, "user_id is required"
		assert customer_id, "customer_id is required"
		
		if not await self._check_billing_permissions(user_id, "read_customer", customer_id):
			raise BillingError(f"User {user_id} not authorized to read customer {customer_id}", "permission_denied")
		
		return self.customers.get(customer_id)
	
	async def list_customers(self, user_id: str, filters: Dict[str, Any] = None) -> List[BLCustomer]:
		"""List customers with optional filtering"""
		assert user_id, "user_id is required"
		
		if not await self._check_billing_permissions(user_id, "list_customers"):
			raise BillingError(f"User {user_id} not authorized to list customers", "permission_denied")
		
		user_tenant = await self._get_user_tenant(user_id)
		accessible_customers = []
		
		for customer in self.customers.values():
			# Filter by tenant
			if customer.tenant_id != user_tenant:
				continue
			
			# Apply filters
			if filters:
				if "active" in filters and customer.active != filters["active"]:
					continue
				if "currency" in filters and customer.currency != filters["currency"]:
					continue
			
			accessible_customers.append(customer)
		
		return accessible_customers
	
	# Plan Management
	
	async def create_plan(self, user_id: str, plan_data: Dict[str, Any]) -> BLPlan:
		"""Create billing plan"""
		assert user_id, "user_id is required"
		assert plan_data, "plan_data is required"
		
		if not await self._check_billing_permissions(user_id, "create_plan"):
			raise BillingError(f"User {user_id} not authorized to create plans", "permission_denied")
		
		try:
			plan_data.update({
				"tenant_id": await self._get_user_tenant(user_id)
			})
			
			plan = BLPlan(**plan_data)
			self.plans[plan.id] = plan
			
			await self._audit_billing_action(
				action="create_plan",
				user_id=user_id,
				resource_id=plan.id,
				details={
					"plan_name": plan.name,
					"pricing_model": plan.pricing_model.value,
					"base_price": str(plan.base_price),
					"billing_period": plan.billing_period.value
				}
			)
			
			self.logger.info(f"Plan created: {plan.name} ({plan.id})")
			return plan
			
		except Exception as e:
			self.logger.error(f"Failed to create plan: {e}")
			raise BillingError(f"Plan creation failed: {e}", "creation_failed")
	
	async def get_plan(self, user_id: str, plan_id: str) -> Optional[BLPlan]:
		"""Get plan by ID"""
		assert user_id, "user_id is required"
		assert plan_id, "plan_id is required"
		
		if not await self._check_billing_permissions(user_id, "read_plan", plan_id):
			raise BillingError(f"User {user_id} not authorized to read plan {plan_id}", "permission_denied")
		
		return self.plans.get(plan_id)
	
	# Subscription Management
	
	async def create_subscription(self, user_id: str, request: CreateSubscriptionRequest) -> BLSubscription:
		"""Create new subscription"""
		assert user_id, "user_id is required"
		assert request, "request is required"
		
		if not await self._check_billing_permissions(user_id, "create_subscription"):
			raise SubscriptionError(f"User {user_id} not authorized to create subscriptions", "permission_denied")
		
		# Validate customer and plan exist
		customer = await self.get_customer(user_id, request.customer_id)
		if not customer:
			raise SubscriptionError(f"Customer not found: {request.customer_id}", "customer_not_found")
		
		plan = await self.get_plan(user_id, request.plan_id)
		if not plan:
			raise SubscriptionError(f"Plan not found: {request.plan_id}", "plan_not_found")
		
		try:
			# Calculate subscription dates
			now = datetime.utcnow()
			current_period_start = now
			
			# Determine if trial applies
			trial_days = request.trial_period_days or plan.trial_period_days
			if trial_days and trial_days > 0:
				trial_start = now
				trial_end = now + timedelta(days=trial_days)
				current_period_end = trial_end
				status = SubscriptionStatus.TRIAL
			else:
				trial_start = None
				trial_end = None
				current_period_end = self._calculate_period_end(now, plan.billing_period)
				status = SubscriptionStatus.ACTIVE
			
			subscription_data = {
				"tenant_id": customer.tenant_id,
				"customer_id": request.customer_id,
				"plan_id": request.plan_id,
				"status": status,
				"current_period_start": current_period_start,
				"current_period_end": current_period_end,
				"trial_start": trial_start,
				"trial_end": trial_end,
				"default_payment_method": request.payment_method_id,
				"metadata": request.metadata
			}
			
			subscription = BLSubscription(**subscription_data)
			self.subscriptions[subscription.id] = subscription
			
			# Initialize usage tracking
			if plan.billing_period == "usage_based":
				await self._initialize_usage_tracking(subscription.id)
			
			await self._audit_billing_action(
				action="create_subscription",
				user_id=user_id,
				resource_id=subscription.id,
				details={
					"customer_id": request.customer_id,
					"plan_id": request.plan_id,
					"status": status.value,
					"trial_days": trial_days
				}
			)
			
			self.logger.info(f"Subscription created: {subscription.id} for customer {customer.name}")
			return subscription
			
		except Exception as e:
			self.logger.error(f"Failed to create subscription: {e}")
			raise SubscriptionError(f"Subscription creation failed: {e}", "creation_failed")
	
	async def get_subscription(self, user_id: str, subscription_id: str) -> Optional[BLSubscription]:
		"""Get subscription by ID"""
		assert user_id, "user_id is required"
		assert subscription_id, "subscription_id is required"
		
		if not await self._check_billing_permissions(user_id, "read_subscription", subscription_id):
			raise SubscriptionError(f"User {user_id} not authorized to read subscription {subscription_id}", "permission_denied")
		
		return self.subscriptions.get(subscription_id)
	
	async def update_subscription(self, user_id: str, subscription_id: str, updates: Dict[str, Any]) -> BLSubscription:
		"""Update subscription"""
		assert user_id, "user_id is required"
		assert subscription_id, "subscription_id is required"
		assert updates, "updates are required"
		
		if not await self._check_billing_permissions(user_id, "update_subscription", subscription_id):
			raise SubscriptionError(f"User {user_id} not authorized to update subscription {subscription_id}", "permission_denied")
		
		subscription = await self.get_subscription(user_id, subscription_id)
		if not subscription:
			raise SubscriptionError(f"Subscription not found: {subscription_id}", "subscription_not_found")
		
		try:
			# Apply updates
			for key, value in updates.items():
				if hasattr(subscription, key):
					setattr(subscription, key, value)
			
			subscription.updated_at = datetime.utcnow()
			
			await self._audit_billing_action(
				action="update_subscription",
				user_id=user_id,
				resource_id=subscription_id,
				details={"updated_fields": list(updates.keys()), "updates": updates}
			)
			
			return subscription
			
		except Exception as e:
			self.logger.error(f"Failed to update subscription {subscription_id}: {e}")
			raise SubscriptionError(f"Subscription update failed: {e}", "update_failed")
	
	async def cancel_subscription(self, user_id: str, subscription_id: str, cancel_at_period_end: bool = True, reason: str = None) -> BLSubscription:
		"""Cancel subscription"""
		assert user_id, "user_id is required"
		assert subscription_id, "subscription_id is required"
		
		if not await self._check_billing_permissions(user_id, "cancel_subscription", subscription_id):
			raise SubscriptionError(f"User {user_id} not authorized to cancel subscription {subscription_id}", "permission_denied")
		
		subscription = await self.get_subscription(user_id, subscription_id)
		if not subscription:
			raise SubscriptionError(f"Subscription not found: {subscription_id}", "subscription_not_found")
		
		if subscription.status == SubscriptionStatus.CANCELLED:
			raise SubscriptionError(f"Subscription already cancelled: {subscription_id}", "already_cancelled")
		
		try:
			subscription.cancel_at_period_end = cancel_at_period_end
			subscription.cancellation_reason = reason
			subscription.updated_at = datetime.utcnow()
			
			if not cancel_at_period_end:
				subscription.status = SubscriptionStatus.CANCELLED
				subscription.cancelled_at = datetime.utcnow()
			
			await self._audit_billing_action(
				action="cancel_subscription",
				user_id=user_id,
				resource_id=subscription_id,
				details={
					"cancel_at_period_end": cancel_at_period_end,
					"reason": reason,
					"immediate": not cancel_at_period_end
				}
			)
			
			self.logger.info(f"Subscription cancelled: {subscription_id} (immediate: {not cancel_at_period_end})")
			return subscription
			
		except Exception as e:
			self.logger.error(f"Failed to cancel subscription {subscription_id}: {e}")
			raise SubscriptionError(f"Subscription cancellation failed: {e}", "cancellation_failed")
	
	# Usage Tracking
	
	async def submit_usage(self, user_id: str, request: UsageSubmissionRequest) -> BLUsage:
		"""Submit usage data for billing"""
		assert user_id, "user_id is required"
		assert request, "request is required"
		
		if not await self._check_billing_permissions(user_id, "submit_usage"):
			raise UsageError(f"User {user_id} not authorized to submit usage", "permission_denied")
		
		# Validate subscription exists
		subscription = await self.get_subscription(user_id, request.subscription_id)
		if not subscription:
			raise UsageError(f"Subscription not found: {request.subscription_id}", "subscription_not_found")
		
		try:
			# Calculate billing period
			current_period_start = subscription.current_period_start
			current_period_end = subscription.current_period_end
			
			usage_data = {
				"tenant_id": subscription.tenant_id,
				"subscription_id": request.subscription_id,
				"customer_id": subscription.customer_id,
				"metric_name": request.metric_name,
				"quantity": request.quantity,
				"unit": self._get_metric_unit(request.metric_name),
				"timestamp": request.timestamp or datetime.utcnow(),
				"billing_period_start": current_period_start,
				"billing_period_end": current_period_end,
				"source_system": "apg_billing",
				"metadata": request.metadata
			}
			
			usage = BLUsage(**usage_data)
			self.usage_records.append(usage)
			
			# Process usage for real-time billing if applicable
			await self._process_usage_for_billing(usage)
			
			await self._audit_billing_action(
				action="submit_usage",
				user_id=user_id,
				resource_id=usage.id,
				details={
					"subscription_id": request.subscription_id,
					"metric_name": request.metric_name,
					"quantity": str(request.quantity)
				}
			)
			
			self.logger.debug(f"Usage submitted: {request.metric_name} = {request.quantity} for subscription {request.subscription_id}")
			return usage
			
		except Exception as e:
			self.logger.error(f"Failed to submit usage: {e}")
			raise UsageError(f"Usage submission failed: {e}", "submission_failed")
	
	async def get_usage_summary(self, user_id: str, subscription_id: str, period_start: datetime = None, period_end: datetime = None) -> Dict[str, Any]:
		"""Get usage summary for subscription"""
		assert user_id, "user_id is required"
		assert subscription_id, "subscription_id is required"
		
		if not await self._check_billing_permissions(user_id, "read_usage", subscription_id):
			raise UsageError(f"User {user_id} not authorized to read usage for subscription {subscription_id}", "permission_denied")
		
		subscription = await self.get_subscription(user_id, subscription_id)
		if not subscription:
			raise UsageError(f"Subscription not found: {subscription_id}", "subscription_not_found")
		
		# Default to current billing period
		if not period_start:
			period_start = subscription.current_period_start
		if not period_end:
			period_end = subscription.current_period_end
		
		# Aggregate usage by metric
		usage_summary = {}
		
		for usage in self.usage_records:
			if (usage.subscription_id == subscription_id and
				usage.timestamp >= period_start and
				usage.timestamp <= period_end):
				
				metric = usage.metric_name
				if metric not in usage_summary:
					usage_summary[metric] = {
						"total_quantity": Decimal('0'),
						"unit": usage.unit,
						"count": 0,
						"first_usage": usage.timestamp,
						"last_usage": usage.timestamp
					}
				
				summary = usage_summary[metric]
				summary["total_quantity"] += usage.quantity
				summary["count"] += 1
				
				if usage.timestamp < summary["first_usage"]:
					summary["first_usage"] = usage.timestamp
				if usage.timestamp > summary["last_usage"]:
					summary["last_usage"] = usage.timestamp
		
		return {
			"subscription_id": subscription_id,
			"period_start": period_start.isoformat(),
			"period_end": period_end.isoformat(),
			"usage_by_metric": usage_summary,
			"total_metrics": len(usage_summary)
		}
	
	# Invoice Management
	
	async def generate_invoice(self, user_id: str, request: InvoiceGenerationRequest) -> BLInvoice:
		"""Generate invoice for subscription"""
		assert user_id, "user_id is required"
		assert request, "request is required"
		
		if not await self._check_billing_permissions(user_id, "generate_invoice"):
			raise InvoiceError(f"User {user_id} not authorized to generate invoices", "permission_denied")
		
		subscription = await self.get_subscription(user_id, request.subscription_id)
		if not subscription:
			raise InvoiceError(f"Subscription not found: {request.subscription_id}", "subscription_not_found")
		
		customer = await self.get_customer(user_id, subscription.customer_id)
		plan = await self.get_plan(user_id, subscription.plan_id)
		
		try:
			# Calculate invoice amounts
			line_items = []
			subtotal = Decimal('0')
			
			# Add subscription charges
			if plan.base_price > 0:
				line_items.append({
					"type": "subscription",
					"description": f"{plan.name} - {request.billing_period_start.strftime('%Y-%m-%d')} to {request.billing_period_end.strftime('%Y-%m-%d')}",
					"quantity": 1,
					"unit_price": plan.base_price,
					"amount": plan.base_price,
					"currency": plan.currency.value
				})
				subtotal += plan.base_price
			
			# Add usage charges if requested
			if request.include_usage:
				usage_charges = await self._calculate_usage_charges(
					subscription.id,
					request.billing_period_start,
					request.billing_period_end
				)
				
				for charge in usage_charges:
					line_items.append(charge)
					subtotal += charge["amount"]
			
			# Apply discounts
			discount_amount = await self._calculate_discounts(subscription, subtotal)
			
			# Calculate taxes
			tax_amount = await self._calculate_taxes(customer, subtotal - discount_amount)
			
			# Generate invoice
			total = subtotal - discount_amount + tax_amount
			due_date = datetime.utcnow() + timedelta(days=customer.payment_terms or 30)
			
			invoice_data = {
				"tenant_id": subscription.tenant_id,
				"customer_id": subscription.customer_id,
				"subscription_id": subscription.id,
				"invoice_number": await self._generate_invoice_number(),
				"status": InvoiceStatus.DRAFT,
				"subtotal": subtotal,
				"tax_amount": tax_amount,
				"discount_amount": discount_amount,
				"total": total,
				"amount_due": total,
				"currency": plan.currency,
				"period_start": request.billing_period_start,
				"period_end": request.billing_period_end,
				"due_date": due_date,
				"line_items": line_items,
				"collection_method": subscription.collection_method
			}
			
			invoice = BLInvoice(**invoice_data)
			self.invoices[invoice.id] = invoice
			
			# Auto-finalize if requested
			if request.auto_finalize:
				await self._finalize_invoice(invoice)
			
			await self._audit_billing_action(
				action="generate_invoice",
				user_id=user_id,
				resource_id=invoice.id,
				details={
					"subscription_id": request.subscription_id,
					"total": str(total),
					"currency": plan.currency.value,
					"period": f"{request.billing_period_start.date()} to {request.billing_period_end.date()}"
				}
			)
			
			self.logger.info(f"Invoice generated: {invoice.invoice_number} for subscription {subscription.id}")
			return invoice
			
		except Exception as e:
			self.logger.error(f"Failed to generate invoice: {e}")
			raise InvoiceError(f"Invoice generation failed: {e}", "generation_failed")
	
	async def get_invoice(self, user_id: str, invoice_id: str) -> Optional[BLInvoice]:
		"""Get invoice by ID"""
		assert user_id, "user_id is required"
		assert invoice_id, "invoice_id is required"
		
		if not await self._check_billing_permissions(user_id, "read_invoice", invoice_id):
			raise InvoiceError(f"User {user_id} not authorized to read invoice {invoice_id}", "permission_denied")
		
		return self.invoices.get(invoice_id)
	
	# Payment Processing
	
	async def process_payment(self, user_id: str, payment_data: Dict[str, Any]) -> BLPayment:
		"""Process payment for invoice"""
		assert user_id, "user_id is required"
		assert payment_data, "payment_data is required"
		
		if not await self._check_billing_permissions(user_id, "process_payment"):
			raise PaymentError(f"User {user_id} not authorized to process payments", "permission_denied")
		
		try:
			payment_data.update({
				"tenant_id": await self._get_user_tenant(user_id)
			})
			
			payment = BLPayment(**payment_data)
			self.payments[payment.id] = payment
			
			# Set initial status
			payment.status = PaymentStatus.PROCESSING
			payment.processed_at = datetime.utcnow()
			
			# Extract payment method and processor info
			payment_method = payment_data.get("payment_method", {})
			processor_name = payment_data.get("processor", "stripe")
			user_context = payment_data.get("user_context", {})
			
			# Process payment with real payment gateway
			success, result = await self.payment_manager.process_payment_with_fraud_check(
				payment, payment_method, processor_name, user_context
			)
			
			if success:
				payment.status = PaymentStatus.SUCCEEDED
				payment.settled_at = datetime.utcnow()
				payment.external_id = result.get("external_id")
				payment.fee_amount = result.get("fee_amount", Decimal('0'))
				payment.net_amount = result.get("net_amount", payment.amount)
				
				# Update invoice if applicable
				if payment.invoice_id:
					await self._apply_payment_to_invoice(payment)
				
				# Emit payment success webhook
				await emit_payment_succeeded(self.webhook_system, {
					'payment': {
						'id': payment.id,
						'amount': str(payment.amount),
						'currency': payment.currency.value,
						'customer_id': payment.customer_id,
						'invoice_id': payment.invoice_id,
						'external_id': payment.external_id,
						'processed_at': payment.processed_at.isoformat() if payment.processed_at else None
					}
				}, payment.tenant_id)
			else:
				payment.status = PaymentStatus.FAILED
				payment.failure_reason = result.get("failure_reason", "Payment processing failed")
				payment.failure_code = result.get("failure_code")
				
				# Handle fraud detection
				if result.get("failure_code") == "fraud_detected":
					payment.risk_score = result.get("fraud_assessment", {}).get("risk_score")
					payment.risk_level = result.get("fraud_assessment", {}).get("risk_level")
				
				# Emit payment failure webhook
				await emit_payment_failed(self.webhook_system, {
					'payment': {
						'id': payment.id,
						'amount': str(payment.amount),
						'currency': payment.currency.value,
						'customer_id': payment.customer_id,
						'invoice_id': payment.invoice_id,
						'failure_reason': payment.failure_reason,
						'failure_code': payment.failure_code
					}
				}, payment.tenant_id)
			
			await self._audit_billing_action(
				action="process_payment",
				user_id=user_id,
				resource_id=payment.id,
				details={
					"amount": str(payment.amount),
					"currency": payment.currency.value,
					"status": payment.status.value,
					"invoice_id": payment.invoice_id
				}
			)
			
			self.logger.info(f"Payment processed: {payment.id} - Status: {payment.status.value}")
			return payment
			
		except Exception as e:
			self.logger.error(f"Failed to process payment: {e}")
			raise PaymentError(f"Payment processing failed: {e}", "processing_failed")
	
	# Analytics and Reporting
	
	async def get_billing_analytics(self, user_id: str, tenant_id: str = None, period_start: datetime = None, period_end: datetime = None) -> Dict[str, Any]:
		"""Get comprehensive billing analytics"""
		assert user_id, "user_id is required"
		
		if not await self._check_billing_permissions(user_id, "read_analytics"):
			raise BillingError(f"User {user_id} not authorized to read analytics", "permission_denied")
		
		if not tenant_id:
			tenant_id = await self._get_user_tenant(user_id)
		
		# Default to last 30 days
		if not period_end:
			period_end = datetime.utcnow()
		if not period_start:
			period_start = period_end - timedelta(days=30)
		
		# Calculate metrics
		total_revenue = Decimal('0')
		total_invoices = 0
		paid_invoices = 0
		active_subscriptions = 0
		total_customers = 0
		
		revenue_by_day = {}
		currency_breakdown = {}
		
		for invoice in self.invoices.values():
			if (invoice.tenant_id == tenant_id and
				invoice.invoice_date >= period_start and
				invoice.invoice_date <= period_end):
				
				total_invoices += 1
				
				if invoice.status == InvoiceStatus.PAID:
					paid_invoices += 1
					total_revenue += invoice.total
					
					# Revenue by day
					day_key = invoice.invoice_date.date().isoformat()
					if day_key not in revenue_by_day:
						revenue_by_day[day_key] = Decimal('0')
					revenue_by_day[day_key] += invoice.total
					
					# Currency breakdown
					currency = invoice.currency.value
					if currency not in currency_breakdown:
						currency_breakdown[currency] = Decimal('0')
					currency_breakdown[currency] += invoice.total
		
		# Count active subscriptions and customers
		for subscription in self.subscriptions.values():
			if subscription.tenant_id == tenant_id:
				if subscription.status in [SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIAL]:
					active_subscriptions += 1
		
		for customer in self.customers.values():
			if customer.tenant_id == tenant_id and customer.active:
				total_customers += 1
		
		return {
			"period_start": period_start.isoformat(),
			"period_end": period_end.isoformat(),
			"metrics": {
				"total_revenue": str(total_revenue),
				"total_invoices": total_invoices,
				"paid_invoices": paid_invoices,
				"payment_success_rate": paid_invoices / max(total_invoices, 1),
				"active_subscriptions": active_subscriptions,
				"total_customers": total_customers
			},
			"revenue_by_day": {k: str(v) for k, v in revenue_by_day.items()},
			"currency_breakdown": {k: str(v) for k, v in currency_breakdown.items()},
			"generated_at": datetime.utcnow().isoformat()
		}
	
	# Helper Methods
	
	def _calculate_period_end(self, start_date: datetime, billing_period: str) -> datetime:
		"""Calculate billing period end date"""
		if billing_period == "daily":
			return start_date + timedelta(days=1)
		elif billing_period == "weekly":
			return start_date + timedelta(weeks=1)
		elif billing_period == "monthly":
			if start_date.month == 12:
				return start_date.replace(year=start_date.year + 1, month=1)
			else:
				return start_date.replace(month=start_date.month + 1)
		elif billing_period == "quarterly":
			return start_date + timedelta(days=90)
		elif billing_period == "yearly":
			return start_date.replace(year=start_date.year + 1)
		else:
			return start_date + timedelta(days=30)  # Default to monthly
	
	def _get_metric_unit(self, metric_name: str) -> str:
		"""Get unit for usage metric"""
		metric_units = {
			"api_calls": "requests",
			"storage": "GB",
			"bandwidth": "GB",
			"compute_hours": "hours",
			"ai_tokens": "tokens",
			"agent_hours": "hours"
		}
		return metric_units.get(metric_name, "units")
	
	async def _initialize_usage_tracking(self, subscription_id: str) -> None:
		"""Initialize usage tracking for subscription"""
		self.logger.debug(f"Initialized usage tracking for subscription {subscription_id}")
	
	async def _process_usage_for_billing(self, usage: BLUsage) -> None:
		"""Process usage for real-time billing"""
		try:
			# Get subscription to determine billing model
			subscription = self.subscriptions.get(usage.subscription_id)
			if not subscription:
				self.logger.warning(f"Subscription not found for usage: {usage.subscription_id}")
				return
			
			# Get plan to check if it has usage-based billing
			plan = self.plans.get(subscription.plan_id)
			if not plan:
				self.logger.warning(f"Plan not found: {subscription.plan_id}")
				return
			
			# Check if this usage metric is billable
			billable_metrics = [metric.metric_name for metric in plan.usage_based_billing.billable_metrics] if plan.usage_based_billing else []
			
			if usage.metric_name not in billable_metrics:
				self.logger.debug(f"Metric {usage.metric_name} is not billable for plan {plan.id}")
				return
			
			# Calculate charges if usage exceeds included allowances
			current_period_usage = self._get_current_period_usage(usage.subscription_id, usage.metric_name)
			total_usage = current_period_usage + usage.quantity
			
			# Check usage limits and generate overage charges
			for metric in plan.usage_based_billing.billable_metrics:
				if metric.metric_name == usage.metric_name:
					if metric.included_quantity and total_usage > metric.included_quantity:
						overage = total_usage - metric.included_quantity
						overage_cost = overage * metric.unit_price
						
						# Create usage charge record
						charge_record = {
							'subscription_id': usage.subscription_id,
							'customer_id': subscription.customer_id,
							'metric_name': usage.metric_name,
							'usage_quantity': usage.quantity,
							'overage_quantity': max(0, overage),
							'overage_cost': overage_cost,
							'billing_period_start': subscription.current_period_start,
							'billing_period_end': subscription.current_period_end,
							'created_at': datetime.utcnow()
						}
						
						# Store for next invoice generation
						if not hasattr(self, 'pending_usage_charges'):
							self.pending_usage_charges = []
						self.pending_usage_charges.append(charge_record)
						
						self.logger.info(f"Generated usage charge: {usage.metric_name} overage {overage} = ${overage_cost}")
					break
			
			# Update usage totals
			self._update_usage_totals(usage)
			
			self.logger.debug(f"Processed usage for billing: {usage.metric_name} = {usage.quantity}")
			
		except Exception as e:
			self.logger.error(f"Failed to process usage for billing: {e}")
	
	def _get_current_period_usage(self, subscription_id: str, metric_name: str) -> Decimal:
		"""Get current billing period usage for a metric"""
		subscription = self.subscriptions.get(subscription_id)
		if not subscription:
			return Decimal('0')
		
		period_start = subscription.current_period_start
		period_end = subscription.current_period_end
		
		total = Decimal('0')
		for usage in self.usage_records.values():
			if (usage.subscription_id == subscription_id and 
				usage.metric_name == metric_name and
				period_start <= usage.timestamp <= period_end):
				total += usage.quantity
		
		return total
	
	def _update_usage_totals(self, usage: BLUsage) -> None:
		"""Update running usage totals"""
		try:
			# Update subscription usage totals
			subscription = self.subscriptions.get(usage.subscription_id)
			if subscription:
				if not hasattr(subscription, 'usage_totals'):
					subscription.usage_totals = {}
				
				current_total = subscription.usage_totals.get(usage.metric_name, Decimal('0'))
				subscription.usage_totals[usage.metric_name] = current_total + usage.quantity
				
		except Exception as e:
			self.logger.error(f"Failed to update usage totals: {e}")
	
	async def _calculate_usage_charges(self, subscription_id: str, period_start: datetime, period_end: datetime) -> List[Dict[str, Any]]:
		"""Calculate usage-based charges for billing period using pricing engine"""
		try:
			# Get usage records for the period
			usage_records = [
				usage for usage in self.usage_records
				if (usage.subscription_id == subscription_id and
					usage.timestamp >= period_start and
					usage.timestamp <= period_end)
			]
			
			if not usage_records:
				return []
			
			# Get subscription and customer context
			subscription = self.subscriptions.get(subscription_id)
			customer_context = {}
			if subscription:
				customer = self.customers.get(subscription.customer_id)
				plan = self.plans.get(subscription.plan_id)
				customer_context = {
					'customer_id': subscription.customer_id,
					'subscription_id': subscription_id,
					'plan_id': subscription.plan_id,
					'customer_tier': customer.tier if customer else 'standard',
					'plan_name': plan.name if plan else 'default'
				}
			
			# Calculate pricing using the pricing engine
			pricing_result = await self.pricing_engine.calculate_usage_price(
				usage_records,
				plan=plan if subscription else None,
				customer_context=customer_context
			)
			
			# Convert pricing breakdown to charges format
			charges = []
			for breakdown in pricing_result.get('pricing_breakdown', []):
				charges.append({
					'type': 'usage',
					'description': f"{breakdown['metric_name']} usage - {breakdown['quantity']} units",
					'quantity': breakdown['quantity'],
					'unit_price': breakdown['details']['effective_rate'],
					'amount': breakdown['price'],
					'metric_name': breakdown['metric_name'],
					'pricing_rule': breakdown['rule_name'],
					'pricing_details': breakdown['details']
				})
			
			# Add applied discounts as line items
			for discount in pricing_result.get('applied_discounts', []):
				charges.append({
					'type': 'discount',
					'description': f"Discount: {discount['discount_name']}",
					'quantity': 1,
					'unit_price': -discount['discount_amount'],
					'amount': -discount['discount_amount'],
					'discount_code': discount.get('discount_code'),
					'discount_type': discount['discount_type']
				})
			
			return charges
			
		except Exception as e:
			self.logger.error(f"Usage charge calculation failed: {e}")
			# Fallback to simple calculation
			return await self._calculate_usage_charges_fallback(subscription_id, period_start, period_end)
	
	async def _calculate_usage_charges_fallback(self, subscription_id: str, period_start: datetime, period_end: datetime) -> List[Dict[str, Any]]:
		"""Fallback usage charge calculation with simple pricing"""
		charges = []
		
		# Aggregate usage by metric
		usage_by_metric = {}
		for usage in self.usage_records:
			if (usage.subscription_id == subscription_id and
				usage.timestamp >= period_start and
				usage.timestamp <= period_end):
				
				metric = usage.metric_name
				if metric not in usage_by_metric:
					usage_by_metric[metric] = {
						"total_quantity": Decimal('0'),
						"unit": usage.unit
					}
				usage_by_metric[metric]["total_quantity"] += usage.quantity
		
		# Calculate charges with fallback pricing
		for metric, data in usage_by_metric.items():
			unit_price = self._get_fallback_unit_price(metric)
			if unit_price > 0:
				amount = data["total_quantity"] * unit_price
				charges.append({
					"type": "usage",
					"description": f"{metric} usage - {data['total_quantity']} {data['unit']} (fallback pricing)",
					"quantity": data["total_quantity"],
					"unit_price": unit_price,
					"amount": amount,
					"metric_name": metric
				})
		
		return charges
	
	def _get_fallback_unit_price(self, metric_name: str) -> Decimal:
		"""Get fallback unit price for usage metric"""
		pricing_map = {
			"api_calls": Decimal('0.001'),
			"storage": Decimal('0.10'),
			"bandwidth": Decimal('0.05'),
			"compute_hours": Decimal('2.00'),
			"ai_tokens": Decimal('0.0001'),
			"agent_hours": Decimal('5.00')
		}
		return pricing_map.get(metric_name, Decimal('0'))
	
	async def _calculate_discounts(self, subscription: BLSubscription, subtotal: Decimal) -> Decimal:
		"""Calculate applicable discounts"""
		total_discount = Decimal('0')
		
		for discount_id in subscription.applied_discounts:
			discount = self.discounts.get(discount_id)
			if discount and discount.active:
				if discount.discount_type == "percentage":
					discount_amount = subtotal * (discount.discount_value / 100)
				elif discount.discount_type == "fixed":
					discount_amount = discount.discount_value
				else:
					discount_amount = Decimal('0')
				
				total_discount += discount_amount
		
		return total_discount
	
	async def _calculate_taxes(self, customer: BLCustomer, taxable_amount: Decimal) -> Decimal:
		"""Calculate taxes for customer using real tax services"""
		try:
			# Prepare transaction data for tax calculation
			transaction_data = {
				'customer_code': customer.id,
				'line_items': [{
					'amount': float(taxable_amount),
					'description': 'Billing Service Charges',
					'tax_code': 'PS081282'  # Software as a Service tax code
				}],
				'ship_from': {
					'country': 'US',
					'state': 'CA',
					'city': 'San Francisco',
					'postal_code': '94105',
					'street': '123 Business St'
				},
				'ship_to': {
					'country': customer.billing_address.get('country', 'US'),
					'state': customer.billing_address.get('state', ''),
					'city': customer.billing_address.get('city', ''),
					'postal_code': customer.billing_address.get('postal_code', ''),
					'street': customer.billing_address.get('street', '')
				}
			}
			
			# Calculate taxes using real tax service
			tax_result = await self.tax_manager.calculate_tax_with_fallback(transaction_data)
			
			if tax_result.get('success'):
				return tax_result.get('total_tax', Decimal('0'))
			else:
				self.logger.warning(f"Tax calculation failed: {tax_result.get('error', 'Unknown error')}")
				# Fallback to default rate if tax service fails
				return taxable_amount * Decimal('0.08')  # 8% fallback rate
			
		except Exception as e:
			self.logger.error(f"Tax calculation error: {e}")
			# Fallback to default rate
			return taxable_amount * Decimal('0.08')
	
	async def _generate_invoice_number(self) -> str:
		"""Generate unique invoice number"""
		invoice_count = len(self.invoices) + 1
		return f"INV-{datetime.utcnow().strftime('%Y%m')}-{invoice_count:06d}"
	
	async def _finalize_invoice(self, invoice: BLInvoice) -> None:
		"""Finalize invoice for payment collection"""
		invoice.status = InvoiceStatus.PENDING
		invoice.updated_at = datetime.utcnow()
		
		# Send invoice email to customer
		try:
			customer = self.customers.get(invoice.customer_id)
			if customer:
				email_manager = self.email_manager.get_billing_email_manager()
				await email_manager.send_invoice_email(customer, invoice)
				self.logger.info(f"Invoice email sent to {customer.email} for invoice {invoice.invoice_number}")
		except Exception as e:
			self.logger.error(f"Failed to send invoice email: {e}")
		
		# Trigger payment collection if auto-collection is enabled
		if invoice.collection_method == "charge_automatically":
			asyncio.create_task(self._auto_collect_payment(invoice))
		else:
			# For manual collection, set invoice to pending
			self.logger.info(f"Invoice {invoice.invoice_number} ready for manual payment collection")
	
	async def _auto_collect_payment(self, invoice: BLInvoice) -> None:
		"""Automatically collect payment for invoice"""
		try:
			# Get customer's default payment method
			customer = self.customers.get(invoice.customer_id)
			if not customer or not customer.default_payment_method:
				self.logger.warning(f"No default payment method for customer {invoice.customer_id}")
				return
			
			# Create payment data
			payment_data = {
				"invoice_id": invoice.id,
				"customer_id": invoice.customer_id,
				"amount": invoice.amount_due,
				"currency": invoice.currency,
				"payment_method": customer.default_payment_method,
				"tenant_id": invoice.tenant_id,
				"processor": customer.preferred_payment_processor or "stripe"
			}
			
			# Process auto-payment
			payment = BLPayment(**payment_data)
			self.payments[payment.id] = payment
			
			# Use real payment processing
			success, result = await self.payment_manager.process_payment_with_fraud_check(
				payment, payment_data["payment_method"], payment_data["processor"]
			)
			
			if success:
				payment.status = PaymentStatus.SUCCEEDED
				payment.processed_at = datetime.utcnow()
				payment.external_id = result.get("external_id")
				await self._apply_payment_to_invoice(payment)
				
				# Send payment confirmation email
				try:
					email_manager = self.email_manager.get_billing_email_manager()
					await email_manager.send_payment_confirmation_email(customer, payment, invoice)
				except Exception as email_error:
					self.logger.error(f"Failed to send payment confirmation email: {email_error}")
				
				self.logger.info(f"Auto-payment successful for invoice {invoice.invoice_number}")
			else:
				payment.status = PaymentStatus.FAILED
				payment.failure_reason = result.get("failure_reason")
				
				# Send payment failure email
				try:
					email_manager = self.email_manager.get_billing_email_manager()
					await email_manager.send_payment_failed_email(customer, payment, payment.failure_reason)
				except Exception as email_error:
					self.logger.error(f"Failed to send payment failure email: {email_error}")
				
				self.logger.warning(f"Auto-payment failed for invoice {invoice.invoice_number}: {payment.failure_reason}")
			
		except Exception as e:
			self.logger.error(f"Auto-collection failed for invoice {invoice.invoice_number}: {e}")
	
	async def _process_with_payment_gateway(self, payment: BLPayment) -> bool:
		"""Process payment with external gateway - DEPRECATED: Use payment_manager directly"""
		# This method is deprecated - payment processing now goes through payment_manager
		self.logger.warning("Using deprecated _process_with_payment_gateway method")
		return False
	
	async def _apply_payment_to_invoice(self, payment: BLPayment) -> None:
		"""Apply payment to invoice and trigger revenue recognition"""
		if payment.invoice_id:
			invoice = self.invoices.get(payment.invoice_id)
			if invoice:
				invoice.amount_paid += payment.amount
				invoice.amount_due = max(Decimal('0'), invoice.total - invoice.amount_paid)
				
				if invoice.amount_due == 0:
					invoice.status = InvoiceStatus.PAID
					invoice.paid_at = payment.processed_at
					
					# Trigger revenue recognition for paid invoice
					await self._trigger_revenue_recognition(invoice)
					
					# Emit invoice paid webhook
					await emit_invoice_paid(self.webhook_system, {
						'invoice': {
							'id': invoice.id,
							'invoice_number': invoice.invoice_number,
							'customer_id': invoice.customer_id,
							'total': str(invoice.total),
							'currency': invoice.currency.value,
							'paid_at': invoice.paid_at.isoformat() if invoice.paid_at else None
						}
					}, invoice.tenant_id)
				
				invoice.updated_at = datetime.utcnow()
	
	async def _trigger_revenue_recognition(self, invoice: BLInvoice) -> None:
		"""Trigger revenue recognition for paid invoice"""
		try:
			subscription = self.subscriptions.get(invoice.subscription_id) if invoice.subscription_id else None
			customer = self.customers.get(invoice.customer_id)
			
			if customer:
				revenue_records = await self.revenue_engine.recognize_invoice_revenue(
					invoice, subscription, customer
				)
				
				# Store revenue records
				for record in revenue_records:
					self.revenue_records.append(record)
				
				self.logger.info(f"Revenue recognition completed for invoice {invoice.invoice_number}: {len(revenue_records)} records created")
		except Exception as e:
			self.logger.error(f"Revenue recognition failed for invoice {invoice.invoice_number}: {e}")
	
	async def _get_user_tenant(self, user_id: str) -> str:
		"""Get user's tenant ID"""
		if self._auth_service_available:
			try:
				user_info = await self.auth_service.get_user(user_id)
				return user_info.get("tenant_id", "default")
			except Exception as e:
				self.logger.warning(f"Could not get user tenant: {e}")
		
		return "default"
	
	# User-to-Customer Mapping
	
	async def map_user_to_customer(self, user_id: str, customer_id: str) -> bool:
		"""Map a user to a customer for access control"""
		try:
			# Validate customer exists
			if customer_id not in self.customers:
				self.logger.warning(f"Cannot map user {user_id} to non-existent customer {customer_id}")
				return False
			
			# Remove existing mapping if present
			if user_id in self.user_customer_mapping:
				old_customer_id = self.user_customer_mapping[user_id]
				if old_customer_id in self.customer_users_mapping:
					self.customer_users_mapping[old_customer_id] = [
						uid for uid in self.customer_users_mapping[old_customer_id] if uid != user_id
					]
			
			# Add new mapping
			self.user_customer_mapping[user_id] = customer_id
			
			if customer_id not in self.customer_users_mapping:
				self.customer_users_mapping[customer_id] = []
			if user_id not in self.customer_users_mapping[customer_id]:
				self.customer_users_mapping[customer_id].append(user_id)
			
			# Audit the mapping
			await self.audit_system.log_audit_event({
				'event_type': AuditEventType.USER_ACCESS_GRANTED.value,
				'user_id': user_id,
				'resource_type': 'customer',
				'resource_id': customer_id,
				'action': 'map_user_to_customer',
				'description': f'User {user_id} mapped to customer {customer_id}',
				'metadata': {
					'mapping_type': 'user_customer',
					'operation': 'create'
				}
			})
			
			self.logger.info(f"User {user_id} mapped to customer {customer_id}")
			return True
			
		except Exception as e:
			self.logger.error(f"Failed to map user {user_id} to customer {customer_id}: {e}")
			return False
	
	async def unmap_user_from_customer(self, user_id: str) -> bool:
		"""Remove user-to-customer mapping"""
		try:
			if user_id not in self.user_customer_mapping:
				return True  # Already unmapped
			
			customer_id = self.user_customer_mapping[user_id]
			
			# Remove from both mappings
			del self.user_customer_mapping[user_id]
			
			if customer_id in self.customer_users_mapping:
				self.customer_users_mapping[customer_id] = [
					uid for uid in self.customer_users_mapping[customer_id] if uid != user_id
				]
			
			# Audit the unmapping
			await self.audit_system.log_audit_event({
				'event_type': AuditEventType.USER_ACCESS_REVOKED.value,
				'user_id': user_id,
				'resource_type': 'customer',
				'resource_id': customer_id,
				'action': 'unmap_user_from_customer',
				'description': f'User {user_id} unmapped from customer {customer_id}',
				'metadata': {
					'mapping_type': 'user_customer',
					'operation': 'delete'
				}
			})
			
			self.logger.info(f"User {user_id} unmapped from customer {customer_id}")
			return True
			
		except Exception as e:
			self.logger.error(f"Failed to unmap user {user_id}: {e}")
			return False
	
	def get_customer_for_user(self, user_id: str) -> Optional[str]:
		"""Get the customer ID for a user"""
		return self.user_customer_mapping.get(user_id)
	
	def get_users_for_customer(self, customer_id: str) -> List[str]:
		"""Get all users mapped to a customer"""
		return self.customer_users_mapping.get(customer_id, [])
	
	async def check_user_access_to_customer(self, user_id: str, customer_id: str) -> bool:
		"""Check if user has access to customer"""
		mapped_customer = self.get_customer_for_user(user_id)
		return mapped_customer == customer_id
	
	async def get_accessible_customers_for_user(self, user_id: str) -> List[str]:
		"""Get all customers accessible to a user"""
		customer_id = self.get_customer_for_user(user_id)
		return [customer_id] if customer_id else []
	
	async def bulk_map_users_to_customer(self, user_ids: List[str], customer_id: str) -> Dict[str, bool]:
		"""Map multiple users to a customer"""
		results = {}
		
		for user_id in user_ids:
			results[user_id] = await self.map_user_to_customer(user_id, customer_id)
		
		return results
	
	async def get_user_customer_mapping_stats(self) -> Dict[str, Any]:
		"""Get statistics about user-customer mappings"""
		total_mappings = len(self.user_customer_mapping)
		customers_with_users = len([c for c in self.customer_users_mapping.values() if c])
		
		# Calculate distribution
		users_per_customer = {}
		for customer_id, users in self.customer_users_mapping.items():
			count = len(users)
			if count > 0:
				users_per_customer[customer_id] = count
		
		return {
			'total_user_mappings': total_mappings,
			'customers_with_users': customers_with_users,
			'total_customers': len(self.customers),
			'unmapped_customers': len(self.customers) - customers_with_users,
			'avg_users_per_customer': sum(users_per_customer.values()) / len(users_per_customer) if users_per_customer else 0,
			'max_users_per_customer': max(users_per_customer.values()) if users_per_customer else 0,
			'users_per_customer_distribution': users_per_customer
		}
	
	# Background Tasks
	
	async def _start_billing_engine(self) -> None:
		"""Start background billing engine"""
		while True:
			try:
				await self._process_subscription_renewals()
				await self._process_trial_endings()
				await self._process_overdue_invoices()
				await asyncio.sleep(3600)  # Run every hour
			except Exception as e:
				self.logger.error(f"Billing engine error: {e}")
				await asyncio.sleep(3600)
	
	async def _start_revenue_recognition(self) -> None:
		"""Start revenue recognition processing"""
		while True:
			try:
				await self._process_revenue_recognition()
				await asyncio.sleep(86400)  # Run daily
			except Exception as e:
				self.logger.error(f"Revenue recognition error: {e}")
				await asyncio.sleep(86400)
	
	async def _process_subscription_renewals(self) -> None:
		"""Process subscription renewals"""
		now = datetime.utcnow()
		
		for subscription in self.subscriptions.values():
			if (subscription.status == SubscriptionStatus.ACTIVE and
				subscription.current_period_end <= now):
				
				try:
					# Generate renewal invoice
					await self._renew_subscription(subscription)
				except Exception as e:
					self.logger.error(f"Failed to renew subscription {subscription.id}: {e}")
	
	async def _renew_subscription(self, subscription: BLSubscription) -> None:
		"""Renew subscription and generate invoice"""
		plan = self.plans.get(subscription.plan_id)
		if not plan:
			return
		
		# Calculate new period
		new_start = subscription.current_period_end
		new_end = self._calculate_period_end(new_start, plan.billing_period.value)
		
		# Update subscription
		subscription.current_period_start = new_start
		subscription.current_period_end = new_end
		subscription.updated_at = datetime.utcnow()
		
		# Generate renewal invoice
		invoice_request = InvoiceGenerationRequest(
			subscription_id=subscription.id,
			billing_period_start=new_start,
			billing_period_end=new_end,
			include_usage=True,
			auto_finalize=True
		)
		
		# Use system user for renewal
		system_user = "system"
		await self.generate_invoice(system_user, invoice_request)
		
		self.logger.info(f"Subscription renewed: {subscription.id}")
	
	async def _process_trial_endings(self) -> None:
		"""Process ending trials"""
		now = datetime.utcnow()
		
		for subscription in self.subscriptions.values():
			if (subscription.status == SubscriptionStatus.TRIAL and
				subscription.trial_end and
				subscription.trial_end <= now):
				
				# Convert trial to active subscription
				subscription.status = SubscriptionStatus.ACTIVE
				subscription.updated_at = now
				
				self.logger.info(f"Trial ended for subscription {subscription.id}")
	
	async def _process_overdue_invoices(self) -> None:
		"""Process overdue invoices"""
		now = datetime.utcnow()
		
		for invoice in self.invoices.values():
			if (invoice.status == InvoiceStatus.PENDING and
				invoice.due_date < now):
				
				invoice.status = InvoiceStatus.OVERDUE
				invoice.updated_at = now
				
				# Trigger dunning process
				await self._trigger_dunning(invoice)
	
	async def _trigger_dunning(self, invoice: BLInvoice) -> None:
		"""Trigger dunning process for overdue invoice"""
		try:
			# Import dunning management system
			from .dunning_management import get_dunning_management_system
			
			dunning_system = get_dunning_management_system()
			
			# Get customer for dunning case creation
			customer = self.customers.get(invoice.customer_id)
			if not customer:
				self.logger.error(f"Customer not found for invoice {invoice.invoice_number}")
				return
			
			# Create dunning case data
			dunning_case_data = {
				'customer_id': invoice.customer_id,
				'invoice_id': invoice.id,
				'subscription_id': getattr(invoice, 'subscription_id', None),
				'sequence_id': 'standard_sequence',  # Use default sequence
				'outstanding_amount': invoice.amount_due,
				'currency': invoice.currency.value if hasattr(invoice.currency, 'value') else str(invoice.currency)
			}
			
			# Create dunning case
			dunning_case = await dunning_system.create_case(dunning_case_data)
			
			if dunning_case:
				self.logger.info(f"Created dunning case {dunning_case.id} for overdue invoice {invoice.invoice_number}")
				
				# Update invoice with dunning case reference
				if not hasattr(invoice, 'dunning_case_id'):
					invoice.dunning_case_id = dunning_case.id
				
				# Process first dunning action
				await dunning_system.process_next_action(dunning_case.id)
			else:
				self.logger.error(f"Failed to create dunning case for invoice {invoice.invoice_number}")
			
		except Exception as e:
			self.logger.error(f"Failed to trigger dunning for invoice {invoice.invoice_number}: {e}")
	
	async def _process_revenue_recognition(self) -> None:
		"""Process revenue recognition according to ASC 606"""
		try:
			# Import revenue recognition system
			from .revenue_recognition import get_revenue_recognition_service
			
			revenue_service = get_revenue_recognition_service()
			
			# Get all active subscriptions for revenue recognition
			active_subscriptions = [
				sub for sub in self.subscriptions.values() 
				if sub.status == SubscriptionStatus.ACTIVE
			]
			
			# Process revenue recognition for each subscription
			for subscription in active_subscriptions:
				try:
					# Get the plan for performance obligations
					plan = self.plans.get(subscription.plan_id)
					if not plan:
						continue
					
					# Recognize subscription revenue
					recognition_data = {
						'subscription_id': subscription.id,
						'customer_id': subscription.customer_id,
						'plan_id': subscription.plan_id,
						'amount': subscription.amount,
						'currency': subscription.currency.value if hasattr(subscription.currency, 'value') else str(subscription.currency),
						'period_start': subscription.current_period_start,
						'period_end': subscription.current_period_end,
						'recognition_date': datetime.utcnow().date(),
						'tenant_id': getattr(subscription, 'tenant_id', 'default')
					}
					
					# Process subscription revenue recognition
					revenue_record = await revenue_service.recognize_subscription_revenue(
						recognition_data, plan
					)
					
					if revenue_record:
						self.logger.debug(f"Recognized revenue for subscription {subscription.id}: ${revenue_record.amount}")
					
				except Exception as sub_error:
					self.logger.error(f"Failed to recognize revenue for subscription {subscription.id}: {sub_error}")
			
			# Process usage-based revenue recognition
			if hasattr(self, 'pending_usage_charges'):
				for charge in getattr(self, 'pending_usage_charges', []):
					try:
						usage_recognition_data = {
							'usage_charge': charge,
							'recognition_date': datetime.utcnow().date(),
							'tenant_id': 'default'
						}
						
						usage_revenue = await revenue_service.recognize_usage_revenue(usage_recognition_data)
						if usage_revenue:
							self.logger.debug(f"Recognized usage revenue: ${usage_revenue.amount}")
						
					except Exception as usage_error:
						self.logger.error(f"Failed to recognize usage revenue: {usage_error}")
			
			# Process monthly close if it's month-end
			today = datetime.utcnow()
			if today.day == 1:  # First day of month - process previous month close
				prev_month = today.replace(day=1) - timedelta(days=1)
				for tenant_id in set(getattr(sub, 'tenant_id', 'default') for sub in active_subscriptions):
					try:
						month_close = await revenue_service.process_monthly_close(
							prev_month.year, prev_month.month, tenant_id
						)
						self.logger.info(f"Processed monthly close for {prev_month.year}-{prev_month.month}: ${month_close.get('total_recognized', 0)}")
					except Exception as close_error:
						self.logger.error(f"Failed to process monthly close: {close_error}")
			
			self.logger.debug("Completed revenue recognition processing")
			
		except Exception as e:
			self.logger.error(f"Revenue recognition processing failed: {e}")
	
	async def get_service_status(self) -> Dict[str, Any]:
		"""Get billing service status"""
		return {
			"service": "BillingService",
			"status": "healthy",
			"customers": len(self.customers),
			"plans": len(self.plans),
			"subscriptions": {
				"total": len(self.subscriptions),
				"active": sum(1 for s in self.subscriptions.values() if s.status == SubscriptionStatus.ACTIVE),
				"trial": sum(1 for s in self.subscriptions.values() if s.status == SubscriptionStatus.TRIAL),
				"cancelled": sum(1 for s in self.subscriptions.values() if s.status == SubscriptionStatus.CANCELLED)
			},
			"invoices": {
				"total": len(self.invoices),
				"paid": sum(1 for i in self.invoices.values() if i.status == InvoiceStatus.PAID),
				"pending": sum(1 for i in self.invoices.values() if i.status == InvoiceStatus.PENDING),
				"overdue": sum(1 for i in self.invoices.values() if i.status == InvoiceStatus.OVERDUE)
			},
			"usage_records": len(self.usage_records),
			"payments": len(self.payments),
			"integrations": {
				"auth_service": self._auth_service_available,
				"audit_service": self._audit_service_available,
				"ai_orchestration": self._ai_orchestration_available
			},
			"timestamp": datetime.utcnow().isoformat()
		}


# Global service instance
_billing_service_instance: Optional[BillingService] = None

def get_billing_service() -> BillingService:
	"""Get global billing service instance"""
	global _billing_service_instance
	if _billing_service_instance is None:
		_billing_service_instance = BillingService()
	return _billing_service_instance


__all__ = [
	"BillingService",
	"BillingError", "SubscriptionError", "UsageError", "InvoiceError", "PaymentError",
	"get_billing_service"
]