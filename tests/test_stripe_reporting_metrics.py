"""Focused coverage for Stripe reporting metric calculations."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
import zipfile
from datetime import datetime, timezone
from decimal import Decimal
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTING_PATH = REPO_ROOT / "capabilities" / "fintech" / "gateway" / "stripe_reporting.py"


def _load_reporting_module():
	stripe_module = types.ModuleType("stripe")
	for name in ["PaymentIntent", "Charge", "Customer", "Subscription", "Dispute", "StripeClient"]:
		setattr(stripe_module, name, type(name, (), {}))
	sys.modules["stripe"] = stripe_module

	spec = importlib.util.spec_from_file_location("stripe_reporting_under_test", REPORTING_PATH)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules["stripe_reporting_under_test"] = module
	spec.loader.exec_module(module)
	return module


def _payment(customer: str, amount: int, status: str = "succeeded"):
	return SimpleNamespace(
		id=f"pi_{customer}_{amount}",
		customer=customer,
		amount=amount,
		status=status,
		currency="usd",
		payment_method_types=["card"],
		created=1_700_000_000,
		description=None,
		metadata={},
	)


def _charge(amount: int, status: str = "succeeded", refunded: int = 0, risk_score: int = 0):
	return SimpleNamespace(
		amount=amount,
		amount_refunded=refunded,
		status=status,
		outcome=SimpleNamespace(type="authorized", risk_score=risk_score, seller_message=""),
		payment_method_details=SimpleNamespace(type="card"),
	)


def _subscription(status: str, cents: int, interval: str = "month", metadata: dict | None = None):
	return SimpleNamespace(
		status=status,
		created=1_600_000_000,
		ended_at=1_600_000_000 + 60 * 24 * 60 * 60,
		trial_start=1_600_000_000,
		trial_end=1_600_000_000 + 14 * 24 * 60 * 60,
		metadata=metadata or {},
		items=SimpleNamespace(
			data=[
				SimpleNamespace(
					quantity=1,
					price=SimpleNamespace(
						id=f"price_{cents}_{interval}",
						nickname=f"{interval}_plan",
						unit_amount=cents,
						recurring=SimpleNamespace(interval=interval, interval_count=1),
					),
				)
			]
		),
	)


def test_stripe_reporting_replaces_placeholder_metric_values():
	module = _load_reporting_module()
	service = module.StripeReportingService(
		SimpleNamespace(),
		{
			"marketing_spend": "300",
			"high_value_customer_threshold": "200",
			"medium_value_customer_threshold": "100",
		},
	)
	current_start = datetime(2026, 1, 1, tzinfo=timezone.utc)
	current_end = datetime(2026, 2, 1, tzinfo=timezone.utc)
	filters = module.ReportFilter(start_date=current_start, end_date=current_end)

	current_payments = [_payment("cus_1", 10000), _payment("cus_1", 5000), _payment("cus_2", 20000)]
	previous_payments = [_payment("cus_1", 8000), _payment("cus_3", 9000)]
	charges = [_charge(10000, risk_score=10), _charge(5000, refunded=1000, risk_score=50), _charge(15000, risk_score=85), _charge(2000, status="failed", risk_score=90)]
	disputes = [SimpleNamespace(amount=5000, reason="fraudulent"), SimpleNamespace(amount=2500, reason="duplicate")]
	customers = [
		SimpleNamespace(id="cus_1", created=int(current_start.timestamp()), invoice_settings=SimpleNamespace(default_payment_method=SimpleNamespace(type="card")), metadata={}),
		SimpleNamespace(id="cus_2", created=int(current_start.timestamp()), invoice_settings=SimpleNamespace(default_payment_method=SimpleNamespace(type="bank_transfer")), metadata={}),
		SimpleNamespace(id="cus_3", created=int(current_start.timestamp()), invoice_settings=SimpleNamespace(default_payment_method=None), default_source=None, metadata={}),
	]
	subscriptions = [
		_subscription("active", 10000, "month", {"change_type": "upgrades"}),
		_subscription("active", 120000, "year", {"change_type": "downgrades"}),
		_subscription("canceled", 5000, "month"),
	]

	async def get_payments(request_filters=None, limit=100):
		if request_filters and request_filters.end_date == current_start:
			return previous_payments
		return current_payments

	async def get_charges(request_filters=None, limit=100):
		return charges

	async def get_disputes(request_filters=None, limit=100):
		return disputes

	async def get_customers(request_filters=None, limit=100):
		return customers

	async def get_subscriptions(request_filters=None, limit=100):
		if request_filters and request_filters.end_date == current_start:
			return [_subscription("active", 5000)]
		return subscriptions

	service._get_payment_intents_data = get_payments
	service._get_charges_data = get_charges
	service._get_disputes_data = get_disputes
	service._get_customers_data = get_customers
	service._get_subscriptions_data = get_subscriptions

	async def exercise():
		customer_map = service._map_customers_to_payments(customers, current_payments)
		return {
			"chargeback_rate": await service._calculate_chargeback_rate(filters),
			"refund_rate": await service._calculate_refund_rate(filters),
			"cac": await service._calculate_customer_acquisition_cost(filters),
			"clv": await service._calculate_customer_lifetime_value(filters),
			"retention": await service._calculate_retention_rate(filters, module.ReportPeriod.MONTH),
			"top_customers": await service._get_top_customers_by_revenue(customer_map),
			"segments": service._segment_customers(customer_map),
			"adoption": await service._calculate_payment_method_adoption(customers),
			"mrr": await service._calculate_mrr(subscriptions),
			"subscription_churn": await service._calculate_subscription_churn_rate(filters, module.ReportPeriod.MONTH),
			"subscription_growth": await service._calculate_subscription_growth_rate(filters, module.ReportPeriod.MONTH),
			"subscription_ltv": await service._calculate_subscription_ltv(subscriptions),
			"trial_conversion": await service._calculate_trial_conversion_rate(filters),
			"subscription_changes": await service._get_subscription_changes(filters),
			"revenue_by_plan": await service._calculate_revenue_by_plan(subscriptions),
			"radar": await service._get_radar_analytics(filters),
			"fraud_indicators": await service._get_top_fraud_indicators(disputes),
			"custom_revenue": await service._calculate_custom_metric("total_revenue", module.ReportPeriod.MONTH, filters),
			"custom_refund": await service._calculate_custom_metric("refund_rate", module.ReportPeriod.MONTH, filters),
		}

	results = asyncio.run(exercise())

	assert results["chargeback_rate"] == 2 / 3
	assert results["refund_rate"] == 1000 / 30000
	assert results["cac"] == Decimal("100")
	assert results["clv"] == Decimal("175")
	assert results["retention"] == 0.5
	assert results["top_customers"][0]["customer_id"] == "cus_2"
	assert results["segments"] == {"high_value": 1, "medium_value": 1, "low_value": 0, "inactive": 1}
	assert results["adoption"] == {"card": 1 / 3, "bank_transfer": 1 / 3, "unknown": 1 / 3}
	assert results["mrr"] == Decimal("200")
	assert results["subscription_churn"] == 1 / 3
	assert results["subscription_growth"] == 1.0
	assert results["trial_conversion"] == 2 / 3
	assert results["subscription_changes"] == {"upgrades": 1, "downgrades": 1}
	assert results["revenue_by_plan"] == {"month_plan": Decimal("100"), "year_plan": Decimal("100")}
	assert results["radar"]["high_risk_transactions"] == 2
	assert results["fraud_indicators"][0] == {"indicator": "fraudulent", "count": 1}
	assert results["custom_revenue"]["value"] == 350.0
	assert results["custom_refund"]["type"] == "percentage"


def test_stripe_report_excel_export_is_valid_xlsx():
	module = _load_reporting_module()
	service = module.StripeReportingService(SimpleNamespace(), {})
	report_bytes = service._format_as_excel(
		{
			"period": "month",
			"metrics": {
				"total_revenue": {"value": 350.0, "type": "revenue", "period": "month"},
				"refund_rate": {"value": 0.033, "type": "percentage", "period": "month"},
			},
		}
	)

	with zipfile.ZipFile(BytesIO(report_bytes)) as workbook:
		names = set(workbook.namelist())
		sheet = workbook.read("xl/worksheets/sheet1.xml").decode()

	assert "[Content_Types].xml" in names
	assert "xl/workbook.xml" in names
	assert "total_revenue" in sheet
	assert "refund_rate" in sheet
