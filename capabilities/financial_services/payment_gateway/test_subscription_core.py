#!/usr/bin/env python3
"""
Quick test script for subscription service core functionality
"""

import asyncio
from datetime import datetime, timezone, timedelta
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional


class BillingCycle(str, Enum):
    """Billing cycle enumeration"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"


class SubscriptionStatus(str, Enum):
    """Subscription status enumeration"""
    ACTIVE = "active"
    PENDING = "pending"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    TRIALING = "trialing"


@dataclass
class SubscriptionPlan:
    """Simplified subscription plan"""
    id: str
    name: str
    amount: int  # Amount in cents
    currency: str
    billing_cycle: BillingCycle
    trial_period_days: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "amount": self.amount,
            "currency": self.currency,
            "billing_cycle": self.billing_cycle.value,
            "trial_period_days": self.trial_period_days
        }


@dataclass
class Subscription:
    """Simplified subscription"""
    id: str
    customer_id: str
    plan_id: str
    status: SubscriptionStatus
    current_period_start: datetime
    current_period_end: datetime
    trial_end: Optional[datetime] = None
    
    @property
    def is_active(self) -> bool:
        return self.status in [SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING]
    
    @property
    def days_until_renewal(self) -> int:
        now = datetime.now(timezone.utc)
        if self.current_period_end > now:
            return (self.current_period_end - now).days
        return 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "customer_id": self.customer_id,
            "plan_id": self.plan_id,
            "status": self.status.value,
            "current_period_start": self.current_period_start.isoformat(),
            "current_period_end": self.current_period_end.isoformat(),
            "trial_end": self.trial_end.isoformat() if self.trial_end else None,
            "is_active": self.is_active,
            "days_until_renewal": self.days_until_renewal
        }


class SimpleSubscriptionService:
    """Simplified subscription service for testing"""
    
    def __init__(self):
        self._plans = {}
        self._subscriptions = {}
        self._initialized = False
    
    async def initialize(self):
        self._initialized = True
        print("💳 Subscription service initialized")
    
    async def create_plan(self, plan_data: Dict[str, Any]) -> SubscriptionPlan:
        """Create a subscription plan"""
        plan = SubscriptionPlan(
            id=f"plan_{len(self._plans) + 1}",
            name=plan_data["name"],
            amount=plan_data["amount"],
            currency=plan_data.get("currency", "USD"),
            billing_cycle=BillingCycle(plan_data["billing_cycle"]),
            trial_period_days=plan_data.get("trial_period_days", 0)
        )
        
        self._plans[plan.id] = plan
        print(f"📋 Created plan: {plan.name} (${plan.amount/100:.2f}/{plan.billing_cycle.value})")
        
        return plan
    
    async def get_plan(self, plan_id: str) -> Optional[SubscriptionPlan]:
        """Get a plan by ID"""
        return self._plans.get(plan_id)
    
    async def create_subscription(self, subscription_data: Dict[str, Any]) -> Subscription:
        """Create a subscription"""
        plan = await self.get_plan(subscription_data["plan_id"])
        if not plan:
            raise ValueError(f"Plan {subscription_data['plan_id']} not found")
        
        now = datetime.now(timezone.utc)
        
        # Calculate periods
        if plan.trial_period_days > 0:
            trial_end = now + timedelta(days=plan.trial_period_days)
            current_period_end = trial_end
            status = SubscriptionStatus.TRIALING
        else:
            trial_end = None
            current_period_end = self._calculate_next_billing_date(now, plan.billing_cycle)
            status = SubscriptionStatus.ACTIVE
        
        subscription = Subscription(
            id=f"sub_{len(self._subscriptions) + 1}",
            customer_id=subscription_data["customer_id"],
            plan_id=plan.id,
            status=status,
            current_period_start=now,
            current_period_end=current_period_end,
            trial_end=trial_end
        )
        
        self._subscriptions[subscription.id] = subscription
        print(f"✅ Created subscription: {subscription.id} for customer {subscription.customer_id}")
        
        return subscription
    
    async def cancel_subscription(self, subscription_id: str, immediately: bool = False) -> Subscription:
        """Cancel a subscription"""
        subscription = self._subscriptions.get(subscription_id)
        if not subscription:
            raise ValueError(f"Subscription {subscription_id} not found")
        
        if immediately:
            subscription.status = SubscriptionStatus.CANCELLED
            subscription.current_period_end = datetime.now(timezone.utc)
        
        print(f"❌ Cancelled subscription: {subscription_id}")
        return subscription
    
    def _calculate_next_billing_date(self, start_date: datetime, billing_cycle: BillingCycle) -> datetime:
        """Calculate next billing date"""
        if billing_cycle == BillingCycle.DAILY:
            return start_date + timedelta(days=1)
        elif billing_cycle == BillingCycle.WEEKLY:
            return start_date + timedelta(weeks=1)
        elif billing_cycle == BillingCycle.MONTHLY:
            return start_date + timedelta(days=30)
        elif billing_cycle == BillingCycle.QUARTERLY:
            return start_date + timedelta(days=90)
        elif billing_cycle == BillingCycle.ANNUALLY:
            return start_date + timedelta(days=365)
        else:
            return start_date + timedelta(days=30)


async def test_subscription_service():
    """Test the subscription service functionality"""
    print("🧪 Testing Subscription Service")
    print("=" * 50)
    
    service = SimpleSubscriptionService()
    await service.initialize()
    
    # Create test plans
    basic_plan_data = {
        "name": "Basic Plan",
        "amount": 999,  # $9.99
        "currency": "USD",
        "billing_cycle": "monthly",
        "trial_period_days": 0
    }
    
    premium_plan_data = {
        "name": "Premium Plan",
        "amount": 2999,  # $29.99
        "currency": "USD", 
        "billing_cycle": "monthly",
        "trial_period_days": 14
    }
    
    basic_plan = await service.create_plan(basic_plan_data)
    premium_plan = await service.create_plan(premium_plan_data)
    
    # Create test subscriptions
    basic_subscription_data = {
        "customer_id": "cust_001",
        "plan_id": basic_plan.id
    }
    
    premium_subscription_data = {
        "customer_id": "cust_002", 
        "plan_id": premium_plan.id
    }
    
    basic_sub = await service.create_subscription(basic_subscription_data)
    premium_sub = await service.create_subscription(premium_subscription_data)
    
    # Test subscription properties
    print(f"\n📊 Subscription Analysis:")
    print(f"   Basic subscription active: {basic_sub.is_active}")
    print(f"   Premium subscription active: {premium_sub.is_active}")
    print(f"   Premium subscription status: {premium_sub.status.value}")
    print(f"   Premium trial ends: {premium_sub.trial_end}")
    print(f"   Days until renewal: {premium_sub.days_until_renewal}")
    
    # Test cancellation
    await service.cancel_subscription(basic_sub.id, immediately=True)
    print(f"   Basic subscription active after cancellation: {basic_sub.is_active}")
    
    # Test serialization
    print(f"\n📋 Plan Details:")
    basic_plan_dict = basic_plan.to_dict()
    print(f"   Basic Plan: {basic_plan_dict['name']} - ${basic_plan_dict['amount']/100:.2f}")
    
    premium_sub_dict = premium_sub.to_dict()
    print(f"   Premium Subscription: {premium_sub_dict['id']} - {premium_sub_dict['status']}")
    
    print(f"\n✅ Subscription service test completed successfully!")
    print(f"   Plans created: {len(service._plans)}")
    print(f"   Subscriptions created: {len(service._subscriptions)}")


if __name__ == "__main__":
    asyncio.run(test_subscription_service())