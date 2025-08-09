"""
Subscription & Recurring Payments Service - APG Payment Gateway

Enterprise-grade subscription management with flexible billing cycles,
prorations, dunning management, and comprehensive lifecycle tracking.

© 2025 Datacraft. All rights reserved.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from dataclasses import dataclass, asdict
from decimal import Decimal
from uuid_extensions import uuid7str

from .models import PaymentTransaction, PaymentMethod, PaymentStatus, PaymentMethodType
from .database import DatabaseService


class SubscriptionStatus(str, Enum):
	"""Subscription status enumeration"""
	ACTIVE = "active"
	PENDING = "pending"
	PAUSED = "paused"
	CANCELLED = "cancelled"
	EXPIRED = "expired"
	PAST_DUE = "past_due"
	UNPAID = "unpaid"
	INCOMPLETE = "incomplete"
	INCOMPLETE_EXPIRED = "incomplete_expired"
	TRIALING = "trialing"


class BillingCycle(str, Enum):
	"""Billing cycle enumeration"""
	DAILY = "daily"
	WEEKLY = "weekly"
	MONTHLY = "monthly"
	QUARTERLY = "quarterly"
	SEMI_ANNUALLY = "semi_annually"
	ANNUALLY = "annually"
	CUSTOM = "custom"


class ProrationBehavior(str, Enum):
	"""Proration behavior for subscription changes"""
	CREATE_PRORATIONS = "create_prorations"
	NONE = "none"
	ALWAYS_INVOICE = "always_invoice"


class DunningAction(str, Enum):
	"""Dunning management actions"""
	EMAIL_REMINDER = "email_reminder"
	SMS_REMINDER = "sms_reminder"
	PAYMENT_RETRY = "payment_retry"
	SUBSCRIPTION_PAUSE = "subscription_pause"
	SUBSCRIPTION_CANCEL = "subscription_cancel"
	ESCALATE_TO_COLLECTIONS = "escalate_to_collections"


@dataclass
class SubscriptionPlan:
	"""Subscription plan definition"""
	id: str
	name: str
	description: str
	amount: int  # Amount in cents
	currency: str
	billing_cycle: BillingCycle
	trial_period_days: int = 0
	setup_fee: int = 0  # Setup fee in cents
	
	# Advanced features
	usage_based: bool = False
	metered_usage_tiers: List[Dict[str, Any]] = None
	
	# Plan metadata
	metadata: Dict[str, Any] = None
	active: bool = True
	created_at: datetime = None
	updated_at: datetime = None
	
	def __post_init__(self):
		if self.created_at is None:
			self.created_at = datetime.now(timezone.utc)
		if self.updated_at is None:
			self.updated_at = self.created_at
		if self.metadata is None:
			self.metadata = {}
		if self.metered_usage_tiers is None:
			self.metered_usage_tiers = []
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			"id": self.id,
			"name": self.name,
			"description": self.description,
			"amount": self.amount,
			"currency": self.currency,
			"billing_cycle": self.billing_cycle.value,
			"trial_period_days": self.trial_period_days,
			"setup_fee": self.setup_fee,
			"usage_based": self.usage_based,
			"metered_usage_tiers": self.metered_usage_tiers,
			"metadata": self.metadata,
			"active": self.active,
			"created_at": self.created_at.isoformat() if self.created_at else None,
			"updated_at": self.updated_at.isoformat() if self.updated_at else None
		}


@dataclass
class Subscription:
	"""Subscription instance"""
	id: str
	customer_id: str
	merchant_id: str
	plan_id: str
	payment_method_id: str
	status: SubscriptionStatus
	
	# Billing details
	current_period_start: datetime
	current_period_end: datetime
	billing_cycle_anchor: datetime
	
	# Trial information
	trial_start: Optional[datetime] = None
	trial_end: Optional[datetime] = None
	
	# Cancellation details
	cancel_at_period_end: bool = False
	canceled_at: Optional[datetime] = None
	
	# Usage tracking
	usage_records: List[Dict[str, Any]] = None
	
	# Financial tracking
	discount_id: Optional[str] = None
	tax_rate: Optional[Decimal] = None
	
	# Metadata
	metadata: Dict[str, Any] = None
	created_at: datetime = None
	updated_at: datetime = None
	
	def __post_init__(self):
		if self.created_at is None:
			self.created_at = datetime.now(timezone.utc)
		if self.updated_at is None:
			self.updated_at = self.created_at
		if self.metadata is None:
			self.metadata = {}
		if self.usage_records is None:
			self.usage_records = []
	
	@property
	def is_active(self) -> bool:
		"""Check if subscription is currently active"""
		return self.status in [SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING]
	
	@property
	def is_past_due(self) -> bool:
		"""Check if subscription is past due"""
		return self.status == SubscriptionStatus.PAST_DUE
	
	@property
	def days_until_renewal(self) -> int:
		"""Calculate days until next renewal"""
		now = datetime.now(timezone.utc)
		if self.current_period_end > now:
			return (self.current_period_end - now).days
		return 0
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			"id": self.id,
			"customer_id": self.customer_id,
			"merchant_id": self.merchant_id,
			"plan_id": self.plan_id,
			"payment_method_id": self.payment_method_id,
			"status": self.status.value,
			"current_period_start": self.current_period_start.isoformat(),
			"current_period_end": self.current_period_end.isoformat(),
			"billing_cycle_anchor": self.billing_cycle_anchor.isoformat(),
			"trial_start": self.trial_start.isoformat() if self.trial_start else None,
			"trial_end": self.trial_end.isoformat() if self.trial_end else None,
			"cancel_at_period_end": self.cancel_at_period_end,
			"canceled_at": self.canceled_at.isoformat() if self.canceled_at else None,
			"usage_records": self.usage_records,
			"discount_id": self.discount_id,
			"tax_rate": float(self.tax_rate) if self.tax_rate else None,
			"metadata": self.metadata,
			"created_at": self.created_at.isoformat(),
			"updated_at": self.updated_at.isoformat(),
			"is_active": self.is_active,
			"is_past_due": self.is_past_due,
			"days_until_renewal": self.days_until_renewal
		}


@dataclass
class Invoice:
	"""Subscription invoice"""
	id: str
	subscription_id: str
	customer_id: str
	merchant_id: str
	
	# Financial details
	amount_due: int  # Amount in cents
	amount_paid: int = 0
	amount_remaining: int = 0
	currency: str = "USD"
	
	# Invoice details
	number: str = None
	description: str = None
	period_start: datetime = None
	period_end: datetime = None
	due_date: datetime = None
	
	# Status tracking
	status: str = "draft"  # draft, open, paid, void, uncollectible
	paid: bool = False
	attempted: bool = False
	attempt_count: int = 0
	next_payment_attempt: Optional[datetime] = None
	
	# Line items
	line_items: List[Dict[str, Any]] = None
	
	# Metadata
	metadata: Dict[str, Any] = None
	created_at: datetime = None
	updated_at: datetime = None
	
	def __post_init__(self):
		if self.created_at is None:
			self.created_at = datetime.now(timezone.utc)
		if self.updated_at is None:
			self.updated_at = self.created_at
		if self.metadata is None:
			self.metadata = {}
		if self.line_items is None:
			self.line_items = []
		if self.amount_remaining == 0:
			self.amount_remaining = self.amount_due - self.amount_paid
		if self.number is None:
			self.number = f"INV-{self.id[:8].upper()}"
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			"id": self.id,
			"subscription_id": self.subscription_id,
			"customer_id": self.customer_id,
			"merchant_id": self.merchant_id,
			"amount_due": self.amount_due,
			"amount_paid": self.amount_paid,
			"amount_remaining": self.amount_remaining,
			"currency": self.currency,
			"number": self.number,
			"description": self.description,
			"period_start": self.period_start.isoformat() if self.period_start else None,
			"period_end": self.period_end.isoformat() if self.period_end else None,
			"due_date": self.due_date.isoformat() if self.due_date else None,
			"status": self.status,
			"paid": self.paid,
			"attempted": self.attempted,
			"attempt_count": self.attempt_count,
			"next_payment_attempt": self.next_payment_attempt.isoformat() if self.next_payment_attempt else None,
			"line_items": self.line_items,
			"metadata": self.metadata,
			"created_at": self.created_at.isoformat(),
			"updated_at": self.updated_at.isoformat()
		}


@dataclass
class DunningRule:
	"""Dunning management rule"""
	id: str
	name: str
	description: str
	merchant_id: str
	
	# Trigger conditions
	trigger_after_days: int
	max_attempts: int
	
	# Actions to take
	actions: List[DunningAction]
	
	# Schedule configuration
	retry_schedule: List[int]  # Days between retries
	
	# Escalation rules
	escalation_rules: List[Dict[str, Any]] = None
	
	# Status
	active: bool = True
	created_at: datetime = None
	updated_at: datetime = None
	
	def __post_init__(self):
		if self.created_at is None:
			self.created_at = datetime.now(timezone.utc)
		if self.updated_at is None:
			self.updated_at = self.created_at
		if self.escalation_rules is None:
			self.escalation_rules = []
	
	def to_dict(self) -> Dict[str, Any]:
		return asdict(self)


class SubscriptionService:
	"""
	Comprehensive subscription and recurring payments service
	"""
	
	def __init__(self, database_service: DatabaseService):
		self._database_service = database_service
		self._payment_service = None  # Will be injected
		self._analytics_engine = None  # Will be injected
		
		# APG services (imported from parent)
		try:
			from apg.notification_engine import NotificationService
			self.notification_service = NotificationService()
		except ImportError:
			self.notification_service = None
		
		# Service state
		self._initialized = False
		self._billing_scheduler_task = None
		self._dunning_processor_task = None
		
		# In-memory caches
		self._plans_cache = {}
		self._subscriptions_cache = {}
		self._dunning_rules_cache = {}
		
		self._log_subscription_service_created()
	
	async def initialize(self, payment_service=None, analytics_engine=None):
		"""Initialize the subscription service"""
		if self._initialized:
			return
		
		self._payment_service = payment_service
		self._analytics_engine = analytics_engine
		
		# Start background tasks
		await self._start_billing_scheduler()
		await self._start_dunning_processor()
		
		self._initialized = True
		self._log_subscription_service_initialized()
	
	async def shutdown(self):
		"""Shutdown the subscription service"""
		if not self._initialized:
			return
		
		# Stop background tasks
		if self._billing_scheduler_task:
			self._billing_scheduler_task.cancel()
		if self._dunning_processor_task:
			self._dunning_processor_task.cancel()
		
		self._initialized = False
		self._log_subscription_service_shutdown()
	
	# Plan Management
	
	async def create_plan(self, plan_data: Dict[str, Any]) -> SubscriptionPlan:
		"""Create a new subscription plan"""
		plan = SubscriptionPlan(
			id=plan_data.get("id", uuid7str()),
			name=plan_data["name"],
			description=plan_data["description"],
			amount=plan_data["amount"],
			currency=plan_data.get("currency", "USD"),
			billing_cycle=BillingCycle(plan_data["billing_cycle"]),
			trial_period_days=plan_data.get("trial_period_days", 0),
			setup_fee=plan_data.get("setup_fee", 0),
			usage_based=plan_data.get("usage_based", False),
			metered_usage_tiers=plan_data.get("metered_usage_tiers", []),
			metadata=plan_data.get("metadata", {})
		)
		
		# Save to database
		await self._database_service.create_subscription_plan(plan)
		
		# Cache the plan
		self._plans_cache[plan.id] = plan
		
		self._log_plan_created(plan.id, plan.name)
		return plan
	
	async def get_plan(self, plan_id: str) -> Optional[SubscriptionPlan]:
		"""Get a subscription plan by ID"""
		# Check cache first
		if plan_id in self._plans_cache:
			return self._plans_cache[plan_id]
		
		# Query database
		plan = await self._database_service.get_subscription_plan(plan_id)
		if plan:
			self._plans_cache[plan_id] = plan
		
		return plan
	
	async def update_plan(self, plan_id: str, updates: Dict[str, Any]) -> SubscriptionPlan:
		"""Update a subscription plan"""
		plan = await self.get_plan(plan_id)
		if not plan:
			raise ValueError(f"Plan {plan_id} not found")
		
		# Apply updates
		for key, value in updates.items():
			if hasattr(plan, key):
				setattr(plan, key, value)
		
		plan.updated_at = datetime.now(timezone.utc)
		
		# Save to database
		await self._database_service.update_subscription_plan(plan_id, updates)
		
		# Update cache
		self._plans_cache[plan_id] = plan
		
		self._log_plan_updated(plan_id)
		return plan
	
	async def list_plans(self, merchant_id: Optional[str] = None, active_only: bool = True) -> List[SubscriptionPlan]:
		"""List subscription plans"""
		return await self._database_service.list_subscription_plans(merchant_id, active_only)
	
	# Subscription Management
	
	async def create_subscription(self, subscription_data: Dict[str, Any]) -> Subscription:
		"""Create a new subscription"""
		plan = await self.get_plan(subscription_data["plan_id"])
		if not plan:
			raise ValueError(f"Plan {subscription_data['plan_id']} not found")
		
		now = datetime.now(timezone.utc)
		
		# Calculate billing periods
		current_period_start = now
		if plan.trial_period_days > 0:
			trial_start = now
			trial_end = now + timedelta(days=plan.trial_period_days)
			current_period_end = trial_end
		else:
			trial_start = None
			trial_end = None
			current_period_end = self._calculate_next_billing_date(now, plan.billing_cycle)
		
		subscription = Subscription(
			id=subscription_data.get("id", uuid7str()),
			customer_id=subscription_data["customer_id"],
			merchant_id=subscription_data["merchant_id"],
			plan_id=plan.id,
			payment_method_id=subscription_data["payment_method_id"],
			status=SubscriptionStatus.TRIALING if plan.trial_period_days > 0 else SubscriptionStatus.ACTIVE,
			current_period_start=current_period_start,
			current_period_end=current_period_end,
			billing_cycle_anchor=current_period_start,
			trial_start=trial_start,
			trial_end=trial_end,
			metadata=subscription_data.get("metadata", {})
		)
		
		# Process setup fee if applicable
		if plan.setup_fee > 0:
			await self._process_setup_fee(subscription, plan)
		
		# Save to database
		await self._database_service.create_subscription(subscription)
		
		# Cache the subscription
		self._subscriptions_cache[subscription.id] = subscription
		
		# Record analytics
		if self._analytics_engine:
			await self._analytics_engine.record_subscription_metric(
				"subscription_created",
				1.0,
				{
					"subscription_id": subscription.id,
					"plan_id": plan.id,
					"customer_id": subscription.customer_id,
					"trial_period": plan.trial_period_days > 0
				}
			)
		
		self._log_subscription_created(subscription.id, plan.name)
		return subscription
	
	async def get_subscription(self, subscription_id: str) -> Optional[Subscription]:
		"""Get a subscription by ID"""
		# Check cache first
		if subscription_id in self._subscriptions_cache:
			return self._subscriptions_cache[subscription_id]
		
		# Query database
		subscription = await self._database_service.get_subscription(subscription_id)
		if subscription:
			self._subscriptions_cache[subscription_id] = subscription
		
		return subscription
	
	async def update_subscription(self, subscription_id: str, updates: Dict[str, Any]) -> Subscription:
		"""Update a subscription"""
		subscription = await self.get_subscription(subscription_id)
		if not subscription:
			raise ValueError(f"Subscription {subscription_id} not found")
		
		# Handle special updates
		if "plan_id" in updates and updates["plan_id"] != subscription.plan_id:
			return await self._change_subscription_plan(subscription, updates["plan_id"], updates)
		
		# Apply regular updates
		for key, value in updates.items():
			if hasattr(subscription, key):
				setattr(subscription, key, value)
		
		subscription.updated_at = datetime.now(timezone.utc)
		
		# Save to database
		await self._database_service.update_subscription(subscription_id, updates)
		
		# Update cache
		self._subscriptions_cache[subscription_id] = subscription
		
		self._log_subscription_updated(subscription_id)
		return subscription
	
	async def cancel_subscription(self, subscription_id: str, cancel_at_period_end: bool = True, reason: str = None) -> Subscription:
		"""Cancel a subscription"""
		subscription = await self.get_subscription(subscription_id)
		if not subscription:
			raise ValueError(f"Subscription {subscription_id} not found")
		
		now = datetime.now(timezone.utc)
		
		if cancel_at_period_end:
			# Cancel at end of current period
			subscription.cancel_at_period_end = True
			subscription.canceled_at = now
		else:
			# Cancel immediately
			subscription.status = SubscriptionStatus.CANCELLED
			subscription.canceled_at = now
			subscription.current_period_end = now
		
		# Save to database
		await self._database_service.update_subscription(
			subscription_id,
			{
				"status": subscription.status.value,
				"cancel_at_period_end": subscription.cancel_at_period_end,
				"canceled_at": subscription.canceled_at,
				"current_period_end": subscription.current_period_end,
				"updated_at": now
			}
		)
		
		# Update cache
		self._subscriptions_cache[subscription_id] = subscription
		
		# Record analytics
		if self._analytics_engine:
			await self._analytics_engine.record_subscription_metric(
				"subscription_cancelled",
				1.0,
				{
					"subscription_id": subscription.id,
					"cancel_at_period_end": cancel_at_period_end,
					"reason": reason
				}
			)
		
		self._log_subscription_cancelled(subscription_id, cancel_at_period_end)
		return subscription
	
	async def pause_subscription(self, subscription_id: str, resume_at: Optional[datetime] = None) -> Subscription:
		"""Pause a subscription"""
		subscription = await self.get_subscription(subscription_id)
		if not subscription:
			raise ValueError(f"Subscription {subscription_id} not found")
		
		subscription.status = SubscriptionStatus.PAUSED
		subscription.updated_at = datetime.now(timezone.utc)
		
		if resume_at:
			subscription.metadata["paused_until"] = resume_at.isoformat()
		
		# Save to database
		await self._database_service.update_subscription(
			subscription_id,
			{
				"status": subscription.status.value,
				"metadata": subscription.metadata,
				"updated_at": subscription.updated_at
			}
		)
		
		# Update cache
		self._subscriptions_cache[subscription_id] = subscription
		
		self._log_subscription_paused(subscription_id)
		return subscription
	
	async def resume_subscription(self, subscription_id: str) -> Subscription:
		"""Resume a paused subscription"""
		subscription = await self.get_subscription(subscription_id)
		if not subscription:
			raise ValueError(f"Subscription {subscription_id} not found")
		
		if subscription.status != SubscriptionStatus.PAUSED:
			raise ValueError(f"Subscription {subscription_id} is not paused")
		
		subscription.status = SubscriptionStatus.ACTIVE
		subscription.updated_at = datetime.now(timezone.utc)
		
		# Remove pause metadata
		if "paused_until" in subscription.metadata:
			del subscription.metadata["paused_until"]
		
		# Save to database
		await self._database_service.update_subscription(
			subscription_id,
			{
				"status": subscription.status.value,
				"metadata": subscription.metadata,
				"updated_at": subscription.updated_at
			}
		)
		
		# Update cache
		self._subscriptions_cache[subscription_id] = subscription
		
		self._log_subscription_resumed(subscription_id)
		return subscription
	
	# Billing and Invoice Management
	
	async def create_invoice(self, subscription_id: str, amount: int, description: str = None) -> Invoice:
		"""Create an invoice for a subscription"""
		subscription = await self.get_subscription(subscription_id)
		if not subscription:
			raise ValueError(f"Subscription {subscription_id} not found")
		
		now = datetime.now(timezone.utc)
		
		invoice = Invoice(
			id=uuid7str(),
			subscription_id=subscription.id,
			customer_id=subscription.customer_id,
			merchant_id=subscription.merchant_id,
			amount_due=amount,
			currency="USD",  # Could be from plan
			description=description,
			period_start=subscription.current_period_start,
			period_end=subscription.current_period_end,
			due_date=now + timedelta(days=7),  # 7 days to pay
			status="open"
		)
		
		# Save to database
		await self._database_service.create_invoice(invoice)
		
		self._log_invoice_created(invoice.id, subscription_id, amount)
		return invoice
	
	async def process_invoice_payment(self, invoice_id: str) -> Dict[str, Any]:
		"""Process payment for an invoice"""
		invoice = await self._database_service.get_invoice(invoice_id)
		if not invoice:
			raise ValueError(f"Invoice {invoice_id} not found")
		
		if invoice.paid:
			return {"success": True, "message": "Invoice already paid"}
		
		subscription = await self.get_subscription(invoice.subscription_id)
		if not subscription:
			raise ValueError(f"Subscription {invoice.subscription_id} not found")
		
		# Attempt payment using payment service
		if self._payment_service:
			try:
				# Create payment transaction
				payment_result = await self._payment_service.process_payment(
					merchant_id=subscription.merchant_id,
					amount=invoice.amount_remaining,
					currency=invoice.currency,
					payment_method_id=subscription.payment_method_id,
					customer_id=subscription.customer_id,
					description=f"Invoice {invoice.number} payment",
					metadata={
						"invoice_id": invoice.id,
						"subscription_id": subscription.id,
						"type": "subscription_payment"
					}
				)
				
				if payment_result.status == PaymentStatus.COMPLETED:
					# Mark invoice as paid
					await self._mark_invoice_paid(invoice, payment_result.id)
					
					# Update subscription status if needed
					if subscription.status == SubscriptionStatus.PAST_DUE:
						await self.update_subscription(subscription.id, {"status": SubscriptionStatus.ACTIVE})
					
					return {"success": True, "transaction_id": payment_result.id}
				else:
					# Payment failed - increment attempt count
					await self._handle_failed_payment(invoice, str(payment_result.status))
					return {"success": False, "error": f"Payment failed: {payment_result.status}"}
			
			except Exception as e:
				await self._handle_failed_payment(invoice, str(e))
				return {"success": False, "error": str(e)}
		
		return {"success": False, "error": "Payment service not available"}
	
	# Background Processing
	
	async def _start_billing_scheduler(self):
		"""Start the billing scheduler background task"""
		self._billing_scheduler_task = asyncio.create_task(self._billing_scheduler_loop())
		self._log_billing_scheduler_started()
	
	async def _start_dunning_processor(self):
		"""Start the dunning processor background task"""
		self._dunning_processor_task = asyncio.create_task(self._dunning_processor_loop())
		self._log_dunning_processor_started()
	
	async def _billing_scheduler_loop(self):
		"""Main billing scheduler loop"""
		while self._initialized:
			try:
				await self._process_scheduled_billings()
				await asyncio.sleep(3600)  # Check every hour
			except asyncio.CancelledError:
				break
			except Exception as e:
				self._log_billing_scheduler_error(str(e))
				await asyncio.sleep(1800)  # Wait 30 minutes on error
	
	async def _dunning_processor_loop(self):
		"""Main dunning processor loop"""
		while self._initialized:
			try:
				await self._process_dunning_actions()
				await asyncio.sleep(3600)  # Check every hour
			except asyncio.CancelledError:
				break
			except Exception as e:
				self._log_dunning_processor_error(str(e))
				await asyncio.sleep(1800)  # Wait 30 minutes on error
	
	async def _process_scheduled_billings(self):
		"""Process subscriptions that need billing"""
		now = datetime.now(timezone.utc)
		
		# Find subscriptions due for billing
		due_subscriptions = await self._database_service.get_subscriptions_due_for_billing(now)
		
		for subscription in due_subscriptions:
			try:
				await self._process_subscription_billing(subscription)
			except Exception as e:
				self._log_subscription_billing_error(subscription.id, str(e))
	
	async def _process_subscription_billing(self, subscription: Subscription):
		"""Process billing for a single subscription"""
		plan = await self.get_plan(subscription.plan_id)
		if not plan:
			self._log_plan_not_found_error(subscription.plan_id)
			return
		
		# Calculate billing amount
		amount = plan.amount
		if plan.usage_based and subscription.usage_records:
			amount += self._calculate_usage_charges(subscription, plan)
		
		# Create invoice
		invoice = await self.create_invoice(
			subscription.id,
			amount,
			f"Subscription billing for {plan.name}"
		)
		
		# Attempt payment
		payment_result = await self.process_invoice_payment(invoice.id)
		
		if payment_result["success"]:
			# Advance billing period
			await self._advance_subscription_period(subscription, plan)
		else:
			# Handle failed payment
			await self._handle_subscription_payment_failure(subscription)
	
	async def _process_dunning_actions(self):
		"""Process dunning management actions"""
		# Find overdue invoices
		overdue_invoices = await self._database_service.get_overdue_invoices()
		
		for invoice in overdue_invoices:
			try:
				await self._process_dunning_for_invoice(invoice)
			except Exception as e:
				self._log_dunning_processing_error(invoice.id, str(e))
	
	# Helper Methods
	
	def _calculate_next_billing_date(self, start_date: datetime, billing_cycle: BillingCycle) -> datetime:
		"""Calculate the next billing date based on cycle with accurate date arithmetic"""
		if billing_cycle == BillingCycle.DAILY:
			return start_date + timedelta(days=1)
		elif billing_cycle == BillingCycle.WEEKLY:
			return start_date + timedelta(weeks=1)
		elif billing_cycle == BillingCycle.MONTHLY:
			# Accurate monthly calculation
			if start_date.month == 12:
				next_month = start_date.replace(year=start_date.year + 1, month=1)
			else:
				next_month = start_date.replace(month=start_date.month + 1)
			
			# Handle end-of-month edge cases
			try:
				return next_month
			except ValueError:
				# If the day doesn't exist in the next month (e.g., Jan 31 -> Feb 31)
				# Use the last day of the next month
				import calendar
				last_day = calendar.monthrange(next_month.year, next_month.month)[1]
				return next_month.replace(day=min(start_date.day, last_day))
		
		elif billing_cycle == BillingCycle.QUARTERLY:
			# Add 3 months
			month = start_date.month
			year = start_date.year
			
			month += 3
			if month > 12:
				year += month // 12
				month = month % 12
				if month == 0:
					month = 12
					year -= 1
			
			try:
				return start_date.replace(year=year, month=month)
			except ValueError:
				import calendar
				last_day = calendar.monthrange(year, month)[1]
				return start_date.replace(year=year, month=month, day=min(start_date.day, last_day))
		
		elif billing_cycle == BillingCycle.SEMI_ANNUALLY:
			# Add 6 months
			month = start_date.month
			year = start_date.year
			
			month += 6
			if month > 12:
				year += 1
				month -= 12
			
			try:
				return start_date.replace(year=year, month=month)
			except ValueError:
				import calendar
				last_day = calendar.monthrange(year, month)[1]
				return start_date.replace(year=year, month=month, day=min(start_date.day, last_day))
		
		elif billing_cycle == BillingCycle.ANNUALLY:
			# Add 1 year
			try:
				return start_date.replace(year=start_date.year + 1)
			except ValueError:
				# Handle leap year edge case (Feb 29)
				return start_date.replace(year=start_date.year + 1, month=2, day=28)
		
		else:
			# Default to monthly for unknown cycles
			return self._calculate_next_billing_date(start_date, BillingCycle.MONTHLY)
	
	def _calculate_usage_charges(self, subscription: Subscription, plan: SubscriptionPlan) -> int:
		"""Calculate usage-based charges"""
		total_usage_charge = 0
		
		for usage_record in subscription.usage_records:
			quantity = usage_record.get("quantity", 0)
			
			# Apply tiered pricing
			for tier in plan.metered_usage_tiers:
				tier_min = tier.get("up_to", 0)
				tier_max = tier.get("up_to_inf", False)
				unit_amount = tier.get("unit_amount", 0)
				
				if tier_max or quantity <= tier_min:
					applicable_quantity = quantity if tier_max else min(quantity, tier_min)
					total_usage_charge += applicable_quantity * unit_amount
					break
		
		return total_usage_charge
	
	async def _process_setup_fee(self, subscription: Subscription, plan: SubscriptionPlan):
		"""Process setup fee for a new subscription"""
		if self._payment_service:
			try:
				await self._payment_service.process_payment(
					merchant_id=subscription.merchant_id,
					amount=plan.setup_fee,
					currency=plan.currency,
					payment_method_id=subscription.payment_method_id,
					customer_id=subscription.customer_id,
					description=f"Setup fee for {plan.name}",
					metadata={
						"subscription_id": subscription.id,
						"type": "setup_fee"
					}
				)
			except Exception as e:
				self._log_setup_fee_error(subscription.id, str(e))
	
	async def _change_subscription_plan(self, subscription: Subscription, new_plan_id: str, updates: Dict[str, Any]) -> Subscription:
		"""Change subscription plan with proration"""
		new_plan = await self.get_plan(new_plan_id)
		if not new_plan:
			raise ValueError(f"Plan {new_plan_id} not found")
		
		old_plan = await self.get_plan(subscription.plan_id)
		
		# Calculate proration if needed
		proration_behavior = updates.get("proration_behavior", ProrationBehavior.CREATE_PRORATIONS)
		
		if proration_behavior == ProrationBehavior.CREATE_PRORATIONS:
			await self._create_proration_invoice(subscription, old_plan, new_plan)
		
		# Update subscription
		subscription.plan_id = new_plan_id
		subscription.updated_at = datetime.now(timezone.utc)
		
		# Apply other updates
		for key, value in updates.items():
			if hasattr(subscription, key) and key != "plan_id":
				setattr(subscription, key, value)
		
		# Save to database
		await self._database_service.update_subscription(subscription.id, {
			"plan_id": new_plan_id,
			"updated_at": subscription.updated_at
		})
		
		self._log_subscription_plan_changed(subscription.id, old_plan.name, new_plan.name)
		return subscription
	
	async def _create_proration_invoice(self, subscription: Subscription, old_plan: SubscriptionPlan, new_plan: SubscriptionPlan):
		"""Create proration invoice for plan change"""
		# Calculate proration amount (simplified)
		now = datetime.now(timezone.utc)
		period_remaining = (subscription.current_period_end - now).days
		total_period_days = (subscription.current_period_end - subscription.current_period_start).days
		
		if total_period_days > 0:
			unused_amount = int((old_plan.amount * period_remaining) / total_period_days)
			new_period_amount = int((new_plan.amount * period_remaining) / total_period_days)
			proration_amount = new_period_amount - unused_amount
			
			if proration_amount != 0:
				await self.create_invoice(
					subscription.id,
					proration_amount,
					f"Proration for plan change from {old_plan.name} to {new_plan.name}"
				)
	
	async def _advance_subscription_period(self, subscription: Subscription, plan: SubscriptionPlan):
		"""Advance subscription to next billing period"""
		subscription.current_period_start = subscription.current_period_end
		subscription.current_period_end = self._calculate_next_billing_date(
			subscription.current_period_end,
			plan.billing_cycle
		)
		subscription.updated_at = datetime.now(timezone.utc)
		
		# Save to database
		await self._database_service.update_subscription(subscription.id, {
			"current_period_start": subscription.current_period_start,
			"current_period_end": subscription.current_period_end,
			"updated_at": subscription.updated_at
		})
		
		# Update cache
		self._subscriptions_cache[subscription.id] = subscription
	
	async def _handle_subscription_payment_failure(self, subscription: Subscription):
		"""Handle failed subscription payment"""
		subscription.status = SubscriptionStatus.PAST_DUE
		subscription.updated_at = datetime.now(timezone.utc)
		
		await self._database_service.update_subscription(subscription.id, {
			"status": subscription.status.value,
			"updated_at": subscription.updated_at
		})
		
		# Update cache
		self._subscriptions_cache[subscription.id] = subscription
		
		self._log_subscription_payment_failed(subscription.id)
	
	async def _mark_invoice_paid(self, invoice: Invoice, transaction_id: str):
		"""Mark an invoice as paid"""
		invoice.paid = True
		invoice.amount_paid = invoice.amount_due
		invoice.amount_remaining = 0
		invoice.status = "paid"
		invoice.updated_at = datetime.now(timezone.utc)
		invoice.metadata["transaction_id"] = transaction_id
		
		await self._database_service.update_invoice(invoice.id, {
			"paid": True,
			"amount_paid": invoice.amount_paid,
			"amount_remaining": 0,
			"status": "paid",
			"metadata": invoice.metadata,
			"updated_at": invoice.updated_at
		})
	
	async def _handle_failed_payment(self, invoice: Invoice, error: str):
		"""Handle failed invoice payment"""
		invoice.attempted = True
		invoice.attempt_count += 1
		invoice.next_payment_attempt = datetime.now(timezone.utc) + timedelta(days=1)
		invoice.updated_at = datetime.now(timezone.utc)
		invoice.metadata["last_error"] = error
		
		await self._database_service.update_invoice(invoice.id, {
			"attempted": True,
			"attempt_count": invoice.attempt_count,
			"next_payment_attempt": invoice.next_payment_attempt,
			"metadata": invoice.metadata,
			"updated_at": invoice.updated_at
		})
	
	async def _process_dunning_for_invoice(self, invoice: Invoice):
		"""Process dunning actions for an overdue invoice"""
		try:
			# Get dunning rules for the merchant
			dunning_rules = await self._get_dunning_rules_for_merchant(invoice.merchant_id)
			
			if not dunning_rules:
				# Use default dunning rules
				dunning_rules = self._get_default_dunning_rules()
			
			# Calculate days overdue
			now = datetime.now(timezone.utc)
			days_overdue = (now - invoice.due_date).days if invoice.due_date else 0
			
			# Find applicable dunning rule
			applicable_rule = None
			for rule in dunning_rules:
				if rule.active and days_overdue >= rule.trigger_after_days:
					if invoice.attempt_count < rule.max_attempts:
						applicable_rule = rule
						break
			
			if not applicable_rule:
				self._log_no_applicable_dunning_rule(invoice.id, days_overdue)
				return
			
			# Execute dunning actions
			for action in applicable_rule.actions:
				try:
					await self._execute_dunning_action(invoice, action, applicable_rule)
				except Exception as e:
					self._log_dunning_action_error(invoice.id, action.value, str(e))
			
			# Schedule next retry if needed
			await self._schedule_next_dunning_attempt(invoice, applicable_rule)
			
		except Exception as e:
			self._log_dunning_processing_error(invoice.id, str(e))
	
	async def _get_dunning_rules_for_merchant(self, merchant_id: str) -> List[DunningRule]:
		"""Get dunning rules for a specific merchant"""
		# In production, this would query the database
		# For now, return cached rules if any
		merchant_rules = []
		for rule_id, rule in self._dunning_rules_cache.items():
			if rule.merchant_id == merchant_id:
				merchant_rules.append(rule)
		return merchant_rules
	
	def _get_default_dunning_rules(self) -> List[DunningRule]:
		"""Get default dunning rules"""
		default_rule = DunningRule(
			id="default_dunning_rule",
			name="Default Dunning Rule",
			description="Standard payment retry and reminder sequence",
			merchant_id="default",
			trigger_after_days=1,
			max_attempts=5,
			actions=[
				DunningAction.EMAIL_REMINDER,
				DunningAction.PAYMENT_RETRY,
				DunningAction.SMS_REMINDER
			],
			retry_schedule=[1, 3, 7, 14, 30],  # Days between retries
			escalation_rules=[
				{"after_attempts": 3, "action": "email_escalation"},
				{"after_attempts": 5, "action": "subscription_pause"}
			]
		)
		return [default_rule]
	
	async def _execute_dunning_action(self, invoice: Invoice, action: DunningAction, rule: DunningRule):
		"""Execute a specific dunning action"""
		if action == DunningAction.EMAIL_REMINDER:
			await self._send_email_reminder(invoice)
		
		elif action == DunningAction.SMS_REMINDER:
			await self._send_sms_reminder(invoice)
		
		elif action == DunningAction.PAYMENT_RETRY:
			await self._retry_payment(invoice)
		
		elif action == DunningAction.SUBSCRIPTION_PAUSE:
			await self._pause_subscription_for_non_payment(invoice)
		
		elif action == DunningAction.SUBSCRIPTION_CANCEL:
			await self._cancel_subscription_for_non_payment(invoice)
		
		elif action == DunningAction.ESCALATE_TO_COLLECTIONS:
			await self._escalate_to_collections(invoice)
		
		self._log_dunning_action_executed(invoice.id, action.value)
	
	async def _send_email_reminder(self, invoice: Invoice):
		"""Send email reminder for overdue invoice"""
		if self.notification_service:
			try:
				await self.notification_service.send_notification(
					type="email",
					recipient=invoice.customer_id,
					template="invoice_overdue_reminder",
					data={
						"invoice_id": invoice.id,
						"invoice_number": invoice.number,
						"amount_due": invoice.amount_remaining,
						"currency": invoice.currency,
						"due_date": invoice.due_date.isoformat() if invoice.due_date else None,
						"payment_url": f"/pay-invoice/{invoice.id}"
					}
				)
			except Exception as e:
				self._log_notification_error("email", invoice.id, str(e))
	
	async def _send_sms_reminder(self, invoice: Invoice):
		"""Send SMS reminder for overdue invoice"""
		if self.notification_service:
			try:
				await self.notification_service.send_notification(
					type="sms",
					recipient=invoice.customer_id,
					template="invoice_overdue_sms",
					data={
						"invoice_number": invoice.number,
						"amount_due": invoice.amount_remaining / 100,  # Convert cents to currency
						"currency": invoice.currency,
						"payment_url": f"/pay/{invoice.id}"
					}
				)
			except Exception as e:
				self._log_notification_error("sms", invoice.id, str(e))
	
	async def _retry_payment(self, invoice: Invoice):
		"""Retry payment for overdue invoice"""
		try:
			# Get the subscription to access payment method
			subscription = await self.get_subscription(invoice.subscription_id)
			if not subscription:
				raise ValueError(f"Subscription {invoice.subscription_id} not found")
			
			# Attempt payment using payment service
			if self._payment_service:
				payment_result = await self._payment_service.process_payment(
					merchant_id=subscription.merchant_id,
					amount=invoice.amount_remaining,
					currency=invoice.currency,
					payment_method_id=subscription.payment_method_id,
					customer_id=subscription.customer_id,
					description=f"Retry payment for invoice {invoice.number}",
					metadata={
						"invoice_id": invoice.id,
						"subscription_id": subscription.id,
						"type": "dunning_retry",
						"attempt_number": invoice.attempt_count + 1
					}
				)
				
				if payment_result.status == PaymentStatus.COMPLETED:
					# Mark invoice as paid
					await self._mark_invoice_paid(invoice, payment_result.id)
					self._log_dunning_payment_successful(invoice.id, payment_result.id)
				else:
					# Payment failed - increment attempt count
					await self._handle_failed_payment(invoice, f"Retry failed: {payment_result.status}")
					self._log_dunning_payment_failed(invoice.id, str(payment_result.status))
		
		except Exception as e:
			await self._handle_failed_payment(invoice, str(e))
			self._log_dunning_payment_error(invoice.id, str(e))
	
	async def _pause_subscription_for_non_payment(self, invoice: Invoice):
		"""Pause subscription due to non-payment"""
		try:
			subscription = await self.get_subscription(invoice.subscription_id)
			if subscription and subscription.status != SubscriptionStatus.PAUSED:
				# Set resume date to 30 days from now
				resume_date = datetime.now(timezone.utc) + timedelta(days=30)
				await self.pause_subscription(subscription.id, resume_date)
				
				# Update invoice metadata
				invoice.metadata["paused_for_non_payment"] = datetime.now(timezone.utc).isoformat()
				await self._database_service.update_invoice(invoice.id, {"metadata": invoice.metadata})
				
				self._log_subscription_paused_for_non_payment(invoice.subscription_id)
		
		except Exception as e:
			self._log_subscription_pause_error(invoice.subscription_id, str(e))
	
	async def _cancel_subscription_for_non_payment(self, invoice: Invoice):
		"""Cancel subscription due to non-payment"""
		try:
			subscription = await self.get_subscription(invoice.subscription_id)
			if subscription and subscription.status != SubscriptionStatus.CANCELLED:
				await self.cancel_subscription(
					subscription.id,
					cancel_at_period_end=False,
					reason="Non-payment after dunning process"
				)
				
				# Update invoice metadata
				invoice.metadata["cancelled_for_non_payment"] = datetime.now(timezone.utc).isoformat()
				await self._database_service.update_invoice(invoice.id, {"metadata": invoice.metadata})
				
				self._log_subscription_cancelled_for_non_payment(invoice.subscription_id)
		
		except Exception as e:
			self._log_subscription_cancel_error(invoice.subscription_id, str(e))
	
	async def _escalate_to_collections(self, invoice: Invoice):
		"""Escalate invoice to collections"""
		try:
			# Update invoice status
			invoice.status = "collections"
			invoice.metadata["escalated_to_collections"] = datetime.now(timezone.utc).isoformat()
			
			await self._database_service.update_invoice(invoice.id, {
				"status": invoice.status,
				"metadata": invoice.metadata
			})
			
			# Send notification to collections team
			if self.notification_service:
				await self.notification_service.send_notification(
					type="email",
					recipient="collections@company.com",
					template="invoice_escalated_to_collections",
					data={
						"invoice_id": invoice.id,
						"invoice_number": invoice.number,
						"customer_id": invoice.customer_id,
						"amount_due": invoice.amount_remaining,
						"currency": invoice.currency,
						"days_overdue": (datetime.now(timezone.utc) - invoice.due_date).days if invoice.due_date else 0
					}
				)
			
			self._log_invoice_escalated_to_collections(invoice.id)
		
		except Exception as e:
			self._log_collections_escalation_error(invoice.id, str(e))
	
	async def _schedule_next_dunning_attempt(self, invoice: Invoice, rule: DunningRule):
		"""Schedule the next dunning attempt"""
		try:
			if invoice.attempt_count < rule.max_attempts:
				# Calculate next attempt date based on retry schedule
				schedule_index = min(invoice.attempt_count, len(rule.retry_schedule) - 1)
				days_until_next = rule.retry_schedule[schedule_index]
				
				next_attempt = datetime.now(timezone.utc) + timedelta(days=days_until_next)
				
				# Update invoice with next attempt time
				await self._database_service.update_invoice(invoice.id, {
					"next_payment_attempt": next_attempt,
					"attempt_count": invoice.attempt_count + 1
				})
				
				self._log_next_dunning_attempt_scheduled(invoice.id, next_attempt)
		
		except Exception as e:
			self._log_dunning_scheduling_error(invoice.id, str(e))
	
	# Logging methods
	def _log_subscription_service_created(self):
		print("💳 Subscription Service created")
	
	def _log_subscription_service_initialized(self):
		print("🚀 Subscription Service initialized with billing scheduler")
	
	def _log_subscription_service_shutdown(self):
		print("🛑 Subscription Service shutdown")
	
	def _log_plan_created(self, plan_id: str, plan_name: str):
		print(f"📋 Subscription plan created: {plan_name} ({plan_id})")
	
	def _log_plan_updated(self, plan_id: str):
		print(f"✏️  Subscription plan updated: {plan_id}")
	
	def _log_subscription_created(self, subscription_id: str, plan_name: str):
		print(f"✅ Subscription created: {subscription_id} for plan {plan_name}")
	
	def _log_subscription_updated(self, subscription_id: str):
		print(f"✏️  Subscription updated: {subscription_id}")
	
	def _log_subscription_cancelled(self, subscription_id: str, at_period_end: bool):
		action = "at period end" if at_period_end else "immediately"
		print(f"❌ Subscription cancelled {action}: {subscription_id}")
	
	def _log_subscription_paused(self, subscription_id: str):
		print(f"⏸️  Subscription paused: {subscription_id}")
	
	def _log_subscription_resumed(self, subscription_id: str):
		print(f"▶️  Subscription resumed: {subscription_id}")
	
	def _log_invoice_created(self, invoice_id: str, subscription_id: str, amount: int):
		print(f"🧾 Invoice created: {invoice_id} for subscription {subscription_id}, amount: {amount/100:.2f}")
	
	def _log_billing_scheduler_started(self):
		print("⏰ Billing scheduler started")
	
	def _log_dunning_processor_started(self):
		print("📞 Dunning processor started")
	
	def _log_billing_scheduler_error(self, error: str):
		print(f"❌ Billing scheduler error: {error}")
	
	def _log_dunning_processor_error(self, error: str):
		print(f"❌ Dunning processor error: {error}")
	
	def _log_subscription_billing_error(self, subscription_id: str, error: str):
		print(f"❌ Subscription billing error [{subscription_id}]: {error}")
	
	def _log_plan_not_found_error(self, plan_id: str):
		print(f"❌ Plan not found: {plan_id}")
	
	def _log_setup_fee_error(self, subscription_id: str, error: str):
		print(f"❌ Setup fee error [{subscription_id}]: {error}")
	
	def _log_subscription_plan_changed(self, subscription_id: str, old_plan: str, new_plan: str):
		print(f"🔄 Subscription plan changed [{subscription_id}]: {old_plan} → {new_plan}")
	
	def _log_subscription_payment_failed(self, subscription_id: str):
		print(f"💸 Subscription payment failed: {subscription_id}")
	
	def _log_dunning_processing_error(self, invoice_id: str, error: str):
		print(f"❌ Dunning processing error [{invoice_id}]: {error}")
	
	def _log_dunning_action_needed(self, invoice_id: str):
		print(f"📞 Dunning action needed for invoice: {invoice_id}")
	
	def _log_no_applicable_dunning_rule(self, invoice_id: str, days_overdue: int):
		print(f"⚠️  No applicable dunning rule for invoice: {invoice_id} ({days_overdue} days overdue)")
	
	def _log_dunning_action_error(self, invoice_id: str, action: str, error: str):
		print(f"❌ Dunning action error [{invoice_id}] {action}: {error}")
	
	def _log_dunning_action_executed(self, invoice_id: str, action: str):
		print(f"✅ Dunning action executed [{invoice_id}]: {action}")
	
	def _log_notification_error(self, notification_type: str, invoice_id: str, error: str):
		print(f"❌ {notification_type.title()} notification error [{invoice_id}]: {error}")
	
	def _log_dunning_payment_successful(self, invoice_id: str, transaction_id: str):
		print(f"💰 Dunning payment successful [{invoice_id}]: {transaction_id}")
	
	def _log_dunning_payment_failed(self, invoice_id: str, error: str):
		print(f"💸 Dunning payment failed [{invoice_id}]: {error}")
	
	def _log_dunning_payment_error(self, invoice_id: str, error: str):
		print(f"❌ Dunning payment error [{invoice_id}]: {error}")
	
	def _log_subscription_paused_for_non_payment(self, subscription_id: str):
		print(f"⏸️  Subscription paused for non-payment: {subscription_id}")
	
	def _log_subscription_pause_error(self, subscription_id: str, error: str):
		print(f"❌ Subscription pause error [{subscription_id}]: {error}")
	
	def _log_subscription_cancelled_for_non_payment(self, subscription_id: str):
		print(f"❌ Subscription cancelled for non-payment: {subscription_id}")
	
	def _log_subscription_cancel_error(self, subscription_id: str, error: str):
		print(f"❌ Subscription cancel error [{subscription_id}]: {error}")
	
	def _log_invoice_escalated_to_collections(self, invoice_id: str):
		print(f"🚨 Invoice escalated to collections: {invoice_id}")
	
	def _log_collections_escalation_error(self, invoice_id: str, error: str):
		print(f"❌ Collections escalation error [{invoice_id}]: {error}")
	
	def _log_next_dunning_attempt_scheduled(self, invoice_id: str, next_attempt: datetime):
		print(f"📅 Next dunning attempt scheduled [{invoice_id}]: {next_attempt.isoformat()}")
	
	def _log_dunning_scheduling_error(self, invoice_id: str, error: str):
		print(f"❌ Dunning scheduling error [{invoice_id}]: {error}")


def create_subscription_service(database_service: DatabaseService) -> SubscriptionService:
	"""Create and return configured subscription service"""
	return SubscriptionService(database_service)


def _log_subscription_module_loaded():
	"""Log subscription module loaded"""
	print("💳 Subscription & Recurring Payments module loaded")
	print("   - Flexible billing cycles and trial periods")
	print("   - Automated billing and invoice management")
	print("   - Dunning management and payment retry logic")
	print("   - Usage-based billing and proration support")
	print("   - Comprehensive subscription lifecycle tracking")


# Execute module loading log
_log_subscription_module_loaded()