"""
Subscription Service Tests

Comprehensive tests for subscription and recurring payment functionality.

© 2025 Datacraft. All rights reserved.
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from uuid_extensions import uuid7str

from ..subscription_service import (
	SubscriptionService,
	SubscriptionPlan,
	Subscription,
	Invoice,
	BillingCycle,
	SubscriptionStatus,
	ProrationBehavior
)


class TestSubscriptionService:
	"""Test subscription service functionality"""
	
	@pytest.fixture
	async def subscription_service(self, temp_database):
		"""Create subscription service for testing"""
		from ..subscription_service import create_subscription_service
		service = create_subscription_service(temp_database)
		await service.initialize()
		yield service
		await service.shutdown()
	
	@pytest.fixture
	def sample_plan_data(self):
		"""Sample subscription plan data"""
		return {
			"name": "Premium Plan",
			"description": "Premium subscription with advanced features",
			"amount": 2999,  # $29.99
			"currency": "USD",
			"billing_cycle": "monthly",
			"trial_period_days": 14,
			"setup_fee": 500,  # $5.00
			"metadata": {"features": ["advanced_analytics", "priority_support"]}
		}
	
	@pytest.fixture
	def sample_subscription_data(self):
		"""Sample subscription data"""
		return {
			"customer_id": "cust_123",
			"merchant_id": "merchant_456",
			"plan_id": "plan_789",
			"payment_method_id": "pm_test_card",
			"metadata": {"source": "website", "campaign": "summer_promo"}
		}
	
	async def test_create_plan(self, subscription_service, sample_plan_data):
		"""Test creating a subscription plan"""
		plan = await subscription_service.create_plan(sample_plan_data)
		
		assert plan.id is not None
		assert plan.name == sample_plan_data["name"]
		assert plan.description == sample_plan_data["description"]
		assert plan.amount == sample_plan_data["amount"]
		assert plan.currency == sample_plan_data["currency"]
		assert plan.billing_cycle == BillingCycle.MONTHLY
		assert plan.trial_period_days == sample_plan_data["trial_period_days"]
		assert plan.setup_fee == sample_plan_data["setup_fee"]
		assert plan.active is True
		assert plan.created_at is not None
		assert plan.updated_at is not None
	
	async def test_get_plan(self, subscription_service, sample_plan_data):
		"""Test retrieving a subscription plan"""
		# Create a plan first
		created_plan = await subscription_service.create_plan(sample_plan_data)
		
		# Retrieve the plan
		retrieved_plan = await subscription_service.get_plan(created_plan.id)
		
		assert retrieved_plan is not None
		assert retrieved_plan.id == created_plan.id
		assert retrieved_plan.name == created_plan.name
		assert retrieved_plan.amount == created_plan.amount
	
	async def test_get_nonexistent_plan(self, subscription_service):
		"""Test retrieving a non-existent plan"""
		plan = await subscription_service.get_plan("nonexistent_plan")
		assert plan is None
	
	async def test_update_plan(self, subscription_service, sample_plan_data):
		"""Test updating a subscription plan"""
		# Create a plan first
		plan = await subscription_service.create_plan(sample_plan_data)
		
		# Update the plan
		updates = {
			"name": "Updated Premium Plan",
			"amount": 3499,  # $34.99
			"description": "Updated description"
		}
		
		updated_plan = await subscription_service.update_plan(plan.id, updates)
		
		assert updated_plan.name == "Updated Premium Plan"
		assert updated_plan.amount == 3499
		assert updated_plan.description == "Updated description"
		assert updated_plan.updated_at > plan.updated_at
	
	async def test_list_plans(self, subscription_service, sample_plan_data):
		"""Test listing subscription plans"""
		# Create multiple plans
		plan1_data = sample_plan_data.copy()
		plan1_data["name"] = "Basic Plan"
		plan1 = await subscription_service.create_plan(plan1_data)
		
		plan2_data = sample_plan_data.copy()
		plan2_data["name"] = "Premium Plan" 
		plan2 = await subscription_service.create_plan(plan2_data)
		
		# List all plans
		plans = await subscription_service.list_plans()
		
		assert len(plans) >= 2
		plan_names = [p.name for p in plans]
		assert "Basic Plan" in plan_names
		assert "Premium Plan" in plan_names
	
	async def test_create_subscription(self, subscription_service, sample_plan_data, sample_subscription_data):
		"""Test creating a subscription"""
		# Create a plan first
		plan = await subscription_service.create_plan(sample_plan_data)
		sample_subscription_data["plan_id"] = plan.id
		
		# Create subscription
		subscription = await subscription_service.create_subscription(sample_subscription_data)
		
		assert subscription.id is not None
		assert subscription.customer_id == sample_subscription_data["customer_id"]
		assert subscription.merchant_id == sample_subscription_data["merchant_id"]
		assert subscription.plan_id == plan.id
		assert subscription.payment_method_id == sample_subscription_data["payment_method_id"]
		assert subscription.status == SubscriptionStatus.TRIALING  # Has trial period
		assert subscription.current_period_start is not None
		assert subscription.current_period_end is not None
		assert subscription.trial_start is not None
		assert subscription.trial_end is not None
		assert subscription.is_active is True
	
	async def test_create_subscription_without_trial(self, subscription_service, sample_plan_data, sample_subscription_data):
		"""Test creating a subscription without trial period"""
		# Create plan without trial
		plan_data = sample_plan_data.copy()
		plan_data["trial_period_days"] = 0
		plan = await subscription_service.create_plan(plan_data)
		sample_subscription_data["plan_id"] = plan.id
		
		# Create subscription
		subscription = await subscription_service.create_subscription(sample_subscription_data)
		
		assert subscription.status == SubscriptionStatus.ACTIVE
		assert subscription.trial_start is None
		assert subscription.trial_end is None
		assert subscription.is_active is True
	
	async def test_get_subscription(self, subscription_service, sample_plan_data, sample_subscription_data):
		"""Test retrieving a subscription"""
		# Create plan and subscription
		plan = await subscription_service.create_plan(sample_plan_data)
		sample_subscription_data["plan_id"] = plan.id
		created_subscription = await subscription_service.create_subscription(sample_subscription_data)
		
		# Retrieve subscription
		retrieved_subscription = await subscription_service.get_subscription(created_subscription.id)
		
		assert retrieved_subscription is not None
		assert retrieved_subscription.id == created_subscription.id
		assert retrieved_subscription.customer_id == created_subscription.customer_id
		assert retrieved_subscription.plan_id == created_subscription.plan_id
	
	async def test_update_subscription(self, subscription_service, sample_plan_data, sample_subscription_data):
		"""Test updating a subscription"""
		# Create plan and subscription
		plan = await subscription_service.create_plan(sample_plan_data) 
		sample_subscription_data["plan_id"] = plan.id
		subscription = await subscription_service.create_subscription(sample_subscription_data)
		
		# Update subscription
		updates = {
			"metadata": {"updated": True, "new_field": "value"}
		}
		
		updated_subscription = await subscription_service.update_subscription(subscription.id, updates)
		
		assert updated_subscription.metadata["updated"] is True
		assert updated_subscription.metadata["new_field"] == "value"
		assert updated_subscription.updated_at > subscription.updated_at
	
	async def test_cancel_subscription_at_period_end(self, subscription_service, sample_plan_data, sample_subscription_data):
		"""Test cancelling a subscription at period end"""
		# Create plan and subscription
		plan = await subscription_service.create_plan(sample_plan_data)
		sample_subscription_data["plan_id"] = plan.id
		subscription = await subscription_service.create_subscription(sample_subscription_data)
		
		# Cancel at period end
		cancelled_subscription = await subscription_service.cancel_subscription(
			subscription.id, 
			cancel_at_period_end=True,
			reason="Customer request"
		)
		
		assert cancelled_subscription.cancel_at_period_end is True
		assert cancelled_subscription.canceled_at is not None
		assert cancelled_subscription.status == SubscriptionStatus.TRIALING  # Still active until period end
	
	async def test_cancel_subscription_immediately(self, subscription_service, sample_plan_data, sample_subscription_data):
		"""Test cancelling a subscription immediately"""
		# Create plan and subscription
		plan = await subscription_service.create_plan(sample_plan_data)
		sample_subscription_data["plan_id"] = plan.id
		subscription = await subscription_service.create_subscription(sample_subscription_data)
		
		# Cancel immediately
		cancelled_subscription = await subscription_service.cancel_subscription(
			subscription.id,
			cancel_at_period_end=False,
			reason="Immediate cancellation"
		)
		
		assert cancelled_subscription.status == SubscriptionStatus.CANCELLED
		assert cancelled_subscription.canceled_at is not None
		assert cancelled_subscription.is_active is False
	
	async def test_pause_subscription(self, subscription_service, sample_plan_data, sample_subscription_data):
		"""Test pausing a subscription"""
		# Create plan and subscription
		plan = await subscription_service.create_plan(sample_plan_data)
		sample_subscription_data["plan_id"] = plan.id
		subscription = await subscription_service.create_subscription(sample_subscription_data)
		
		# Pause subscription
		paused_subscription = await subscription_service.pause_subscription(subscription.id)
		
		assert paused_subscription.status == SubscriptionStatus.PAUSED
		assert paused_subscription.is_active is False
	
	async def test_resume_subscription(self, subscription_service, sample_plan_data, sample_subscription_data):
		"""Test resuming a paused subscription"""
		# Create plan and subscription
		plan = await subscription_service.create_plan(sample_plan_data)
		sample_subscription_data["plan_id"] = plan.id
		subscription = await subscription_service.create_subscription(sample_subscription_data)
		
		# Pause then resume
		await subscription_service.pause_subscription(subscription.id)
		resumed_subscription = await subscription_service.resume_subscription(subscription.id)
		
		assert resumed_subscription.status == SubscriptionStatus.ACTIVE
		assert resumed_subscription.is_active is True
	
	async def test_resume_non_paused_subscription(self, subscription_service, sample_plan_data, sample_subscription_data):
		"""Test resuming a non-paused subscription should fail"""
		# Create plan and subscription
		plan = await subscription_service.create_plan(sample_plan_data)
		sample_subscription_data["plan_id"] = plan.id
		subscription = await subscription_service.create_subscription(sample_subscription_data)
		
		# Try to resume non-paused subscription
		with pytest.raises(ValueError, match="is not paused"):
			await subscription_service.resume_subscription(subscription.id)
	
	async def test_create_invoice(self, subscription_service, sample_plan_data, sample_subscription_data):
		"""Test creating an invoice"""
		# Create plan and subscription
		plan = await subscription_service.create_plan(sample_plan_data)
		sample_subscription_data["plan_id"] = plan.id
		subscription = await subscription_service.create_subscription(sample_subscription_data)
		
		# Create invoice
		amount = 2999  # $29.99
		description = "Monthly subscription fee"
		invoice = await subscription_service.create_invoice(subscription.id, amount, description)
		
		assert invoice.id is not None
		assert invoice.subscription_id == subscription.id
		assert invoice.customer_id == subscription.customer_id
		assert invoice.merchant_id == subscription.merchant_id
		assert invoice.amount_due == amount
		assert invoice.description == description
		assert invoice.status == "open"
		assert invoice.paid is False
		assert invoice.number is not None
	
	async def test_billing_cycle_calculations(self, subscription_service):
		"""Test billing cycle date calculations"""
		now = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
		
		# Test different billing cycles
		daily_next = subscription_service._calculate_next_billing_date(now, BillingCycle.DAILY)
		assert daily_next == now + timedelta(days=1)
		
		weekly_next = subscription_service._calculate_next_billing_date(now, BillingCycle.WEEKLY)
		assert weekly_next == now + timedelta(weeks=1)
		
		monthly_next = subscription_service._calculate_next_billing_date(now, BillingCycle.MONTHLY)
		assert monthly_next == now + timedelta(days=30)
		
		quarterly_next = subscription_service._calculate_next_billing_date(now, BillingCycle.QUARTERLY)
		assert quarterly_next == now + timedelta(days=90)
		
		annually_next = subscription_service._calculate_next_billing_date(now, BillingCycle.ANNUALLY)
		assert annually_next == now + timedelta(days=365)
	
	async def test_subscription_properties(self, subscription_service, sample_plan_data, sample_subscription_data):
		"""Test subscription computed properties"""
		# Create plan and subscription
		plan = await subscription_service.create_plan(sample_plan_data)
		sample_subscription_data["plan_id"] = plan.id
		subscription = await subscription_service.create_subscription(sample_subscription_data)
		
		# Test properties
		assert subscription.is_active is True
		assert subscription.is_past_due is False
		assert subscription.days_until_renewal >= 0
		
		# Test to_dict method
		subscription_dict = subscription.to_dict()
		assert "id" in subscription_dict
		assert "customer_id" in subscription_dict
		assert "status" in subscription_dict
		assert "is_active" in subscription_dict
		assert "days_until_renewal" in subscription_dict
	
	async def test_plan_to_dict(self, subscription_service, sample_plan_data):
		"""Test plan to_dict method"""
		plan = await subscription_service.create_plan(sample_plan_data)
		
		plan_dict = plan.to_dict()
		
		assert "id" in plan_dict
		assert "name" in plan_dict
		assert "description" in plan_dict
		assert "amount" in plan_dict
		assert "currency" in plan_dict
		assert "billing_cycle" in plan_dict
		assert "trial_period_days" in plan_dict
		assert "setup_fee" in plan_dict
		assert "usage_based" in plan_dict
		assert "active" in plan_dict
		assert "created_at" in plan_dict
		assert "updated_at" in plan_dict
	
	async def test_invoice_to_dict(self, subscription_service, sample_plan_data, sample_subscription_data):
		"""Test invoice to_dict method"""
		# Create plan and subscription
		plan = await subscription_service.create_plan(sample_plan_data)
		sample_subscription_data["plan_id"] = plan.id
		subscription = await subscription_service.create_subscription(sample_subscription_data)
		
		# Create invoice
		invoice = await subscription_service.create_invoice(subscription.id, 2999, "Test invoice")
		
		invoice_dict = invoice.to_dict()
		
		assert "id" in invoice_dict
		assert "subscription_id" in invoice_dict
		assert "customer_id" in invoice_dict
		assert "merchant_id" in invoice_dict
		assert "amount_due" in invoice_dict
		assert "amount_paid" in invoice_dict
		assert "amount_remaining" in invoice_dict
		assert "status" in invoice_dict
		assert "paid" in invoice_dict
		assert "number" in invoice_dict
		assert "created_at" in invoice_dict
		assert "updated_at" in invoice_dict
	
	async def test_usage_based_billing(self, subscription_service):
		"""Test usage-based billing calculations"""
		# Create usage-based plan
		plan_data = {
			"name": "Usage Plan",
			"description": "Pay per API call",
			"amount": 1000,  # Base fee $10.00
			"currency": "USD",
			"billing_cycle": "monthly",
			"usage_based": True,
			"metered_usage_tiers": [
				{"up_to": 1000, "unit_amount": 10},  # $0.10 per call up to 1000
				{"up_to_inf": True, "unit_amount": 5}  # $0.05 per call above 1000
			]
		}
		
		plan = await subscription_service.create_plan(plan_data)
		
		# Create subscription with usage
		subscription_data = {
			"customer_id": "cust_usage",
			"merchant_id": "merchant_usage",
			"plan_id": plan.id,
			"payment_method_id": "pm_test",
			"usage_records": [
				{"quantity": 1500, "unit": "api_calls"}
			]
		}
		
		subscription = await subscription_service.create_subscription(subscription_data)
		
		# Calculate usage charges
		usage_charges = subscription_service._calculate_usage_charges(subscription, plan)
		
		# Expected: 1000 * $0.10 + 500 * $0.05 = $100 + $25 = $125 = 12500 cents
		assert usage_charges == 12500
	
	async def test_concurrent_subscription_operations(self, subscription_service, sample_plan_data):
		"""Test concurrent subscription operations"""
		# Create a plan first
		plan = await subscription_service.create_plan(sample_plan_data)
		
		# Create multiple subscriptions concurrently
		tasks = []
		for i in range(10):
			subscription_data = {
				"customer_id": f"cust_{i}",
				"merchant_id": f"merchant_{i}",
				"plan_id": plan.id,
				"payment_method_id": f"pm_{i}"
			}
			tasks.append(subscription_service.create_subscription(subscription_data))
		
		# Execute all tasks concurrently
		results = await asyncio.gather(*tasks, return_exceptions=True)
		
		# Verify all subscriptions were created successfully
		successful_creates = [r for r in results if not isinstance(r, Exception)]
		assert len(successful_creates) == 10
		
		# Verify each subscription has unique ID
		subscription_ids = [s.id for s in successful_creates]
		assert len(set(subscription_ids)) == 10
	
	async def test_subscription_service_initialization(self, temp_database):
		"""Test subscription service initialization and shutdown"""
		from ..subscription_service import create_subscription_service
		
		service = create_subscription_service(temp_database)
		assert service._initialized is False
		
		# Initialize service
		await service.initialize()
		assert service._initialized is True
		assert service._billing_scheduler_task is not None
		assert service._dunning_processor_task is not None
		
		# Shutdown service
		await service.shutdown()
		assert service._initialized is False
	
	async def test_plan_caching(self, subscription_service, sample_plan_data):
		"""Test plan caching functionality"""
		# Create a plan
		plan = await subscription_service.create_plan(sample_plan_data)
		
		# First retrieval should cache the plan
		retrieved_plan_1 = await subscription_service.get_plan(plan.id)
		assert plan.id in subscription_service._plans_cache
		
		# Second retrieval should use cache
		retrieved_plan_2 = await subscription_service.get_plan(plan.id)
		assert retrieved_plan_1 is retrieved_plan_2  # Same object from cache
	
	async def test_subscription_caching(self, subscription_service, sample_plan_data, sample_subscription_data):
		"""Test subscription caching functionality"""
		# Create plan and subscription
		plan = await subscription_service.create_plan(sample_plan_data)
		sample_subscription_data["plan_id"] = plan.id
		subscription = await subscription_service.create_subscription(sample_subscription_data)
		
		# First retrieval should cache the subscription
		retrieved_subscription_1 = await subscription_service.get_subscription(subscription.id)
		assert subscription.id in subscription_service._subscriptions_cache
		
		# Second retrieval should use cache
		retrieved_subscription_2 = await subscription_service.get_subscription(subscription.id)
		assert retrieved_subscription_1 is retrieved_subscription_2  # Same object from cache


class TestSubscriptionDataClasses:
	"""Test subscription data classes"""
	
	def test_subscription_plan_creation(self):
		"""Test SubscriptionPlan creation"""
		plan = SubscriptionPlan(
			id="plan_123",
			name="Test Plan",
			description="Test description",
			amount=1999,
			currency="USD",
			billing_cycle=BillingCycle.MONTHLY,
			trial_period_days=7
		)
		
		assert plan.id == "plan_123"
		assert plan.name == "Test Plan"
		assert plan.amount == 1999
		assert plan.billing_cycle == BillingCycle.MONTHLY
		assert plan.trial_period_days == 7
		assert plan.active is True
		assert plan.created_at is not None
		assert plan.updated_at is not None
	
	def test_subscription_creation(self):
		"""Test Subscription creation"""
		now = datetime.now(timezone.utc)
		
		subscription = Subscription(
			id="sub_123",
			customer_id="cust_456",
			merchant_id="merchant_789",
			plan_id="plan_abc",
			payment_method_id="pm_def",
			status=SubscriptionStatus.ACTIVE,
			current_period_start=now,
			current_period_end=now + timedelta(days=30),
			billing_cycle_anchor=now
		)
		
		assert subscription.id == "sub_123"
		assert subscription.customer_id == "cust_456"
		assert subscription.status == SubscriptionStatus.ACTIVE
		assert subscription.is_active is True
		assert subscription.is_past_due is False
		assert subscription.days_until_renewal >= 0
	
	def test_invoice_creation(self):
		"""Test Invoice creation"""
		invoice = Invoice(
			id="inv_123",
			subscription_id="sub_456",
			customer_id="cust_789",
			merchant_id="merchant_abc",
			amount_due=2999,
			currency="USD",
			description="Monthly subscription"
		)
		
		assert invoice.id == "inv_123"
		assert invoice.subscription_id == "sub_456"
		assert invoice.amount_due == 2999
		assert invoice.amount_remaining == 2999  # Initially equals amount_due
		assert invoice.paid is False
		assert invoice.number is not None
		assert invoice.number.startswith("INV-")
	
	def test_billing_cycle_enum(self):
		"""Test BillingCycle enum values"""
		assert BillingCycle.DAILY.value == "daily"
		assert BillingCycle.WEEKLY.value == "weekly"
		assert BillingCycle.MONTHLY.value == "monthly"
		assert BillingCycle.QUARTERLY.value == "quarterly"
		assert BillingCycle.SEMI_ANNUALLY.value == "semi_annually"
		assert BillingCycle.ANNUALLY.value == "annually"
		assert BillingCycle.CUSTOM.value == "custom"
	
	def test_subscription_status_enum(self):
		"""Test SubscriptionStatus enum values"""
		assert SubscriptionStatus.ACTIVE.value == "active"
		assert SubscriptionStatus.PENDING.value == "pending"
		assert SubscriptionStatus.PAUSED.value == "paused"
		assert SubscriptionStatus.CANCELLED.value == "cancelled"
		assert SubscriptionStatus.EXPIRED.value == "expired"
		assert SubscriptionStatus.PAST_DUE.value == "past_due"
		assert SubscriptionStatus.UNPAID.value == "unpaid"
		assert SubscriptionStatus.TRIALING.value == "trialing"