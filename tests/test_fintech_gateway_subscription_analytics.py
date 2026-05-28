"""Subscription API analytics regressions without Flask runtime dependencies."""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path


def _identity_decorator(*_args, **_kwargs):
	def decorator(func):
		return func
	return decorator


def _load_subscription_api_module():
	module_name = "capabilities.fintech.gateway.subscription_api"
	for name in [
		module_name,
		"capabilities.fintech.gateway.subscription_service",
		"capabilities.fintech.gateway.database",
		"capabilities.fintech.gateway.auth",
		"flask",
		"flask_appbuilder",
		"flask_appbuilder.security",
		"flask_appbuilder.security.decorators",
	]:
		sys.modules.pop(name, None)

	for package_name in ["capabilities", "capabilities.fintech", "capabilities.fintech.gateway"]:
		package = types.ModuleType(package_name)
		package.__path__ = []
		sys.modules[package_name] = package

	flask_module = types.ModuleType("flask")
	flask_module.request = types.SimpleNamespace(args={}, get_json=lambda: {})
	flask_module.jsonify = lambda payload: payload
	sys.modules["flask"] = flask_module

	fab_module = types.ModuleType("flask_appbuilder")
	fab_module.BaseView = object
	fab_module.expose = _identity_decorator
	sys.modules["flask_appbuilder"] = fab_module
	sys.modules["flask_appbuilder.security"] = types.ModuleType("flask_appbuilder.security")
	decorators_module = types.ModuleType("flask_appbuilder.security.decorators")
	decorators_module.has_access = lambda func: func
	sys.modules["flask_appbuilder.security.decorators"] = decorators_module

	service_module = types.ModuleType("capabilities.fintech.gateway.subscription_service")

	class SubscriptionStatus(str, Enum):
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
		DAILY = "daily"
		WEEKLY = "weekly"
		MONTHLY = "monthly"
		QUARTERLY = "quarterly"
		SEMI_ANNUALLY = "semi_annually"
		ANNUALLY = "annually"
		CUSTOM = "custom"

	class SubscriptionService:
		pass

	service_module.SubscriptionService = SubscriptionService
	service_module.SubscriptionStatus = SubscriptionStatus
	service_module.BillingCycle = BillingCycle
	sys.modules["capabilities.fintech.gateway.subscription_service"] = service_module

	database_module = types.ModuleType("capabilities.fintech.gateway.database")
	database_module.get_database_service = lambda: None
	sys.modules["capabilities.fintech.gateway.database"] = database_module

	auth_module = types.ModuleType("capabilities.fintech.gateway.auth")
	auth_module.authenticate_api_key = lambda _request: {"success": True, "user": {"role": "admin"}}
	auth_module.require_permission = lambda _user, _permission: True
	sys.modules["capabilities.fintech.gateway.auth"] = auth_module

	module_path = (
		Path(__file__).resolve().parents[1]
		/ "capabilities"
		/ "fintech"
		/ "gateway"
		/ "subscription_api.py"
	)
	spec = importlib.util.spec_from_file_location(module_name, module_path)
	module = importlib.util.module_from_spec(spec)
	sys.modules[module_name] = module
	assert spec.loader is not None
	spec.loader.exec_module(module)
	return module


@dataclass
class _Plan:
	id: str
	name: str
	amount: int
	billing_cycle: object


@dataclass
class _Subscription:
	id: str
	merchant_id: str
	plan_id: str
	status: object
	created_at: datetime
	current_period_start: datetime
	canceled_at: datetime | None = None


@dataclass
class _Invoice:
	id: str
	merchant_id: str
	amount_paid: int
	paid: bool
	created_at: datetime
	updated_at: datetime | None = None


class _Database:
	def __init__(self, plans, invoices):
		self._plans = plans
		self._invoices = {invoice.id: invoice for invoice in invoices}

	async def list_subscription_plans(self, active_only=True):
		return self._plans

	async def get_invoice(self, invoice_id):
		return self._invoices.get(invoice_id)


class _SubscriptionService:
	def __init__(self, plans, subscriptions, invoices):
		self._plans_cache = {plan.id: plan for plan in plans}
		self._subscriptions_cache = {subscription.id: subscription for subscription in subscriptions}
		self._database_service = _Database(plans, invoices)


def test_subscription_analytics_are_computed_from_runtime_state() -> None:
	module = _load_subscription_api_module()
	now = datetime(2026, 5, 29, tzinfo=timezone.utc)
	start_date = now - timedelta(days=30)
	plans = [
		_Plan("basic", "Basic", 10_000, module.BillingCycle.MONTHLY),
		_Plan("annual", "Annual", 120_000, module.BillingCycle.ANNUALLY),
	]
	subscriptions = [
		_Subscription("sub-active", "merchant-a", "basic", module.SubscriptionStatus.ACTIVE, now - timedelta(days=20), now - timedelta(days=20)),
		_Subscription("sub-trial", "merchant-a", "annual", module.SubscriptionStatus.TRIALING, now - timedelta(days=10), now - timedelta(days=10)),
		_Subscription("sub-cancelled", "merchant-a", "basic", module.SubscriptionStatus.CANCELLED, now - timedelta(days=40), now - timedelta(days=40), now - timedelta(days=5)),
		_Subscription("sub-paused", "merchant-a", "basic", module.SubscriptionStatus.PAUSED, now - timedelta(days=12), now - timedelta(days=12)),
		_Subscription("sub-other", "merchant-b", "basic", module.SubscriptionStatus.ACTIVE, now - timedelta(days=3), now - timedelta(days=3)),
	]
	invoices = [
		_Invoice("inv-paid", "merchant-a", 12_500, True, now - timedelta(days=3)),
		_Invoice("inv-unpaid", "merchant-a", 9_000, False, now - timedelta(days=2)),
		_Invoice("inv-other", "merchant-b", 77_000, True, now - timedelta(days=1)),
	]

	api = module.SubscriptionAPI.__new__(module.SubscriptionAPI)
	api._subscription_service = _SubscriptionService(plans, subscriptions, invoices)
	api._get_subscription_service = lambda: api._subscription_service

	analytics = module.asyncio.run(
		api._get_subscription_analytics("merchant-a", start_date, now)
	)

	assert analytics["summary"] == {
		"total_subscriptions": 4,
		"active_subscriptions": 1,
		"cancelled_subscriptions": 1,
		"paused_subscriptions": 1,
		"trial_subscriptions": 1,
	}
	assert analytics["revenue"] == {
		"monthly_recurring_revenue": 200.0,
		"annual_recurring_revenue": 2400.0,
		"average_revenue_per_user": 100.0,
		"total_revenue_period": 125.0,
	}
	assert analytics["churn"]["churn_rate"] == 25.0
	assert analytics["churn"]["retention_rate"] == 75.0
	assert analytics["billing_cycles"] == {"monthly": 3, "annually": 1}
	assert analytics["top_plans"][0] == {
		"plan_name": "Basic",
		"subscribers": 3,
		"revenue": 100.0,
	}


def test_subscription_analytics_empty_state_has_zeroes_not_mock_values() -> None:
	module = _load_subscription_api_module()
	now = datetime(2026, 5, 29, tzinfo=timezone.utc)
	api = module.SubscriptionAPI.__new__(module.SubscriptionAPI)
	api._subscription_service = _SubscriptionService([], [], [])
	api._get_subscription_service = lambda: api._subscription_service

	analytics = module.asyncio.run(
		api._get_subscription_analytics(None, now - timedelta(days=30), now)
	)

	assert analytics["summary"]["total_subscriptions"] == 0
	assert analytics["revenue"]["monthly_recurring_revenue"] == 0.0
	assert analytics["revenue"]["total_revenue_period"] == 0
	assert analytics["churn"]["retention_rate"] == 100.0
	assert analytics["billing_cycles"] == {}
	assert analytics["top_plans"] == []
