"""Tests for Pydantic v2 models — telecom/bil.

© 2025 Datacraft. All rights reserved.
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
	sys.path.insert(0, str(PACKAGE_DIR))


def _load(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec and spec.loader
	mod = importlib.util.module_from_spec(spec)
	sys.modules[name] = mod
	spec.loader.exec_module(mod)  # type: ignore[union-attr]
	return mod


_m = _load("_models_test", PACKAGE_DIR / "models.py")

from _models_test import (  # type: ignore[import]
	BilBase, BillingAccount, BillingAccountCreate, BillingAccountStatus, BillingAccountType,
	Bundle, BundleCreate, BundleStatus, BundleType,
	CDR, CDRCreate, CDRStatus, CDRType, CallDirection,
	ChargeType, ConvergentMode, CreditLimit, CreditLimitCreate,
	Discount, DiscountCreate, DiscountType,
	DisputeCreate, DisputeStatus, DisputeType,
	DunningStep, InvoiceCreate, InvoiceStatus,
	InterconnectSettlement, InterconnectSettlementCreate,
	PaymentAllocation, PaymentAllocationCreate, PaymentMethod,
	Promotion, PromotionCreate, PromotionStatus,
	RatingResult, RatingResultCreate,
	Roaming, RoamingCreate, RoamingZone,
	TariffPlan, TariffPlanCreate, TariffPlanType,
	TaxType, SettlementStatus,
	RevenueReport, BillingDashboardKPI, CDRRatingReport,
	SpendAlert, ConvergentBillingSummary, TaxBreakdown, RevenueAssuranceResult,
)


# ---------------------------------------------------------------------------
# BilBase
# ---------------------------------------------------------------------------

def test_bilbase_defaults():
	b = BilBase(tenant_id="t1")
	assert b.id  # uuid7 generated
	assert b.tenant_id == "t1"
	assert not b.is_deleted
	assert isinstance(b.created_at, datetime)


def test_bilbase_rejects_blank_tenant():
	with pytest.raises(ValidationError):
		BilBase(tenant_id="   ")


def test_bilbase_forbids_extra_fields():
	with pytest.raises(ValidationError):
		BilBase(tenant_id="t1", unknown_field="x")


# ---------------------------------------------------------------------------
# CDR
# ---------------------------------------------------------------------------

def test_cdr_create_minimal():
	c = CDRCreate(
		tenant_id="t1",
		cdr_type=CDRType.VOICE,
		direction=CallDirection.ORIGINATING,
		source="MSC-01",
		msisdn="+254712345678",
		recorded_at=datetime.now(timezone.utc),
	)
	assert c.cdr_type == CDRType.VOICE
	assert c.duration_seconds == 0


def test_cdr_full():
	c = CDR(
		tenant_id="t1",
		cdr_type=CDRType.DATA,
		direction=CallDirection.ORIGINATING,
		source="GGSN-01",
		msisdn="+254712345678",
		recorded_at=datetime.now(timezone.utc),
		data_volume_bytes=1_048_576,
	)
	assert c.mediation_status == CDRStatus.RAW
	assert c.data_volume_bytes == 1_048_576


def test_cdr_negative_duration_rejected():
	with pytest.raises(ValidationError):
		CDRCreate(
			tenant_id="t1",
			cdr_type=CDRType.VOICE,
			direction=CallDirection.ORIGINATING,
			source="MSC",
			msisdn="+254700000001",
			recorded_at=datetime.now(timezone.utc),
			duration_seconds=-1,
		)


def test_cdr_status_enum_values():
	statuses = [s.value for s in CDRStatus]
	assert "raw" in statuses
	assert "rated" in statuses
	assert "billed" in statuses


def test_cdr_type_enum_values():
	types = [t.value for t in CDRType]
	assert "voice" in types
	assert "data" in types
	assert "roaming" in types


# ---------------------------------------------------------------------------
# BillingAccount
# ---------------------------------------------------------------------------

def test_billing_account_create():
	ba = BillingAccountCreate(
		tenant_id="t1",
		account_type=BillingAccountType.POSTPAID,
		customer_id="cust-001",
	)
	assert ba.billing_day == 1
	assert ba.payment_terms_days == 30
	assert ba.currency == "KES"


def test_billing_account_billing_day_bounds():
	with pytest.raises(ValidationError):
		BillingAccountCreate(
			tenant_id="t1",
			account_type=BillingAccountType.POSTPAID,
			customer_id="cust-001",
			billing_day=29,
		)
	with pytest.raises(ValidationError):
		BillingAccountCreate(
			tenant_id="t1",
			account_type=BillingAccountType.POSTPAID,
			customer_id="cust-001",
			billing_day=0,
		)


def test_billing_account_status_transitions():
	statuses = [s.value for s in BillingAccountStatus]
	assert "active" in statuses
	assert "suspended" in statuses
	assert "barred" in statuses


# ---------------------------------------------------------------------------
# TariffPlan
# ---------------------------------------------------------------------------

def test_tariff_plan_create():
	tp = TariffPlanCreate(
		tenant_id="t1",
		name="Standard Voice",
		plan_type=TariffPlanType.FLAT_RATE,
		base_rate=Decimal("0"),
		rate_per_second=Decimal("0.05"),
		valid_from=datetime.now(timezone.utc),
	)
	assert tp.rate_per_second == Decimal("0.05")
	assert tp.rate_per_kb == Decimal("0")


def test_tariff_plan_negative_rate_rejected():
	with pytest.raises(ValidationError):
		TariffPlanCreate(
			tenant_id="t1",
			name="Bad Plan",
			plan_type=TariffPlanType.FLAT_RATE,
			base_rate=Decimal("-1"),
			valid_from=datetime.now(timezone.utc),
		)


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------

def test_bundle_remaining_units():
	b = Bundle(
		tenant_id="t1",
		account_id="acc-1",
		bundle_type=BundleType.VOICE,
		name="100min Bundle",
		total_units=Decimal("100"),
		consumed_units=Decimal("30"),
		unit="minutes",
		price=Decimal("50"),
		valid_from=datetime(2026, 6, 1, tzinfo=timezone.utc),
		valid_to=datetime(2026, 6, 30, tzinfo=timezone.utc),
	)
	assert b.remaining_units == Decimal("70")


def test_bundle_status_enum():
	statuses = [s.value for s in BundleStatus]
	assert "active" in statuses
	assert "exhausted" in statuses
	assert "expired" in statuses


def test_bundle_zero_total_units_rejected():
	with pytest.raises(ValidationError):
		BundleCreate(
			tenant_id="t1",
			account_id="acc-1",
			bundle_type=BundleType.DATA,
			name="Bad Bundle",
			total_units=Decimal("0"),
			unit="MB",
			price=Decimal("0"),
			valid_from=datetime(2026, 6, 1, tzinfo=timezone.utc),
			valid_to=datetime(2026, 6, 30, tzinfo=timezone.utc),
		)


# ---------------------------------------------------------------------------
# Discount
# ---------------------------------------------------------------------------

def test_discount_pct_max_50():
	with pytest.raises(ValidationError):
		DiscountCreate(
			tenant_id="t1",
			account_id="acc-1",
			discount_type=DiscountType.LOYALTY,
			discount_pct=Decimal("51"),
			approval_reference="AUTH-001",
			valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
			valid_to=datetime(2026, 12, 31, tzinfo=timezone.utc),
		)


def test_discount_pct_at_50_allowed():
	d = DiscountCreate(
		tenant_id="t1",
		account_id="acc-1",
		discount_type=DiscountType.LOYALTY,
		discount_pct=Decimal("50"),
		approval_reference="AUTH-001",
		valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
		valid_to=datetime(2026, 12, 31, tzinfo=timezone.utc),
	)
	assert d.discount_pct == Decimal("50")


# ---------------------------------------------------------------------------
# CreditLimit
# ---------------------------------------------------------------------------

def test_credit_limit_positive_only():
	with pytest.raises(ValidationError):
		CreditLimitCreate(
			tenant_id="t1",
			account_id="acc-1",
			hard_limit=Decimal("0"),
			soft_limit=Decimal("0"),
			approval_reference="AUTH-CL-001",
		)


# ---------------------------------------------------------------------------
# Invoice
# ---------------------------------------------------------------------------

def test_invoice_create():
	inv = InvoiceCreate(
		tenant_id="t1",
		account_id="acc-1",
		period_start=datetime(2026, 5, 1, tzinfo=timezone.utc),
		period_end=datetime(2026, 5, 31, tzinfo=timezone.utc),
		due_date=datetime(2026, 6, 15, tzinfo=timezone.utc),
	)
	assert inv.currency == "KES"


def test_invoice_status_enum():
	statuses = [s.value for s in InvoiceStatus]
	assert "draft" in statuses
	assert "paid" in statuses
	assert "written_off" in statuses
	assert "disputed" in statuses


# ---------------------------------------------------------------------------
# PaymentAllocation
# ---------------------------------------------------------------------------

def test_payment_allocation_positive_amount():
	with pytest.raises(ValidationError):
		PaymentAllocationCreate(
			tenant_id="t1",
			account_id="acc-1",
			invoice_id="inv-1",
			payment_method=PaymentMethod.MOBILE_MONEY,
			amount=Decimal("0"),
			reference="REF-001",
			paid_at=datetime.now(timezone.utc),
		)


def test_payment_method_enum():
	methods = [m.value for m in PaymentMethod]
	assert "mobile_money" in methods
	assert "bank_transfer" in methods
	assert "credit_card" in methods


# ---------------------------------------------------------------------------
# Roaming
# ---------------------------------------------------------------------------

def test_roaming_create():
	r = RoamingCreate(
		tenant_id="t1",
		account_id="acc-1",
		cdr_id="cdr-1",
		zone=RoamingZone.ZONE_A,
		visited_network="SAFARICOM-KE",
		home_network="MTN-UG",
		service_type=CDRType.VOICE,
		duration_seconds=60,
		base_charge=Decimal("5.00"),
	)
	assert r.zone == RoamingZone.ZONE_A


def test_roaming_zone_enum():
	zones = [z.value for z in RoamingZone]
	assert "zone_a" in zones
	assert "premium" in zones
	assert "global" in zones


# ---------------------------------------------------------------------------
# InterconnectSettlement
# ---------------------------------------------------------------------------

def test_interconnect_settlement_create():
	s = InterconnectSettlementCreate(
		tenant_id="t1",
		carrier_id="CARRIER-001",
		carrier_name="Airtel Kenya",
		period_start=datetime(2026, 5, 1, tzinfo=timezone.utc),
		period_end=datetime(2026, 5, 31, tzinfo=timezone.utc),
		receivable_amount=Decimal("10000.00"),
		payable_amount=Decimal("8000.00"),
		reference_number="IC-2026-05-001",
	)
	assert s.receivable_amount == Decimal("10000.00")


def test_settlement_status_enum():
	statuses = [s.value for s in SettlementStatus]
	assert "draft" in statuses
	assert "paid" in statuses


# ---------------------------------------------------------------------------
# Dispute
# ---------------------------------------------------------------------------

def test_dispute_create():
	d = DisputeCreate(
		tenant_id="t1",
		account_id="acc-1",
		invoice_id="inv-1",
		dispute_type=DisputeType.BILLING_ERROR,
		disputed_amount=Decimal("500.00"),
		reason="Charged for calls I did not make",
	)
	assert d.dispute_type == DisputeType.BILLING_ERROR


def test_dispute_status_enum():
	statuses = [s.value for s in DisputeStatus]
	assert "open" in statuses
	assert "resolved_upheld" in statuses
	assert "arbitration" in statuses


# ---------------------------------------------------------------------------
# Report models
# ---------------------------------------------------------------------------

def test_revenue_report_model():
	r = RevenueReport(
		tenant_id="t1",
		period_start=datetime(2026, 5, 1, tzinfo=timezone.utc),
		period_end=datetime(2026, 5, 31, tzinfo=timezone.utc),
		total_revenue=Decimal("500000"),
		voice_revenue=Decimal("200000"),
		data_revenue=Decimal("200000"),
		sms_revenue=Decimal("50000"),
		roaming_revenue=Decimal("30000"),
		interconnect_revenue=Decimal("20000"),
		other_revenue=Decimal("0"),
		tax_collected=Decimal("80000"),
		discounts_given=Decimal("5000"),
		net_revenue=Decimal("495000"),
		currency="KES",
		invoice_count=100,
		paid_invoice_count=85,
		disputed_amount=Decimal("10000"),
		written_off_amount=Decimal("2000"),
	)
	assert r.total_revenue == Decimal("500000")


def test_dashboard_kpi_model():
	k = BillingDashboardKPI(
		tenant_id="t1",
		total_accounts=1000,
		active_accounts=950,
		suspended_accounts=30,
		total_invoices=5000,
		draft_invoices=10,
		overdue_invoices=50,
		open_disputes=5,
		total_revenue_mtd=Decimal("1000000"),
		collection_rate_pct=Decimal("92.5"),
		average_revenue_per_account=Decimal("1052.63"),
		credit_utilisation_pct=Decimal("45.2"),
		currency="KES",
	)
	assert k.total_accounts == 1000


def test_spend_alert_model():
	a = SpendAlert(
		account_id="acc-1",
		tenant_id="t1",
		alert_type="soft_limit",
		current_usage=Decimal("4500"),
		limit=Decimal("5000"),
		utilisation_pct=Decimal("90"),
		currency="KES",
	)
	assert a.alert_type == "soft_limit"


def test_convergent_billing_summary():
	c = ConvergentBillingSummary(
		master_account_id="master-1",
		tenant_id="t1",
		mode=ConvergentMode.SINGLE_BILL,
		member_account_ids=["acc-1", "acc-2"],
		total_fixed_charges=Decimal("500"),
		total_mobile_charges=Decimal("300"),
		total_data_charges=Decimal("200"),
		combined_total=Decimal("1000"),
		currency="KES",
		period_start=datetime(2026, 5, 1, tzinfo=timezone.utc),
		period_end=datetime(2026, 5, 31, tzinfo=timezone.utc),
	)
	assert c.combined_total == Decimal("1000")


def test_tax_breakdown_model():
	t = TaxBreakdown(
		tenant_id="t1",
		account_id="acc-1",
		invoice_id="inv-1",
		jurisdiction="KE",
		pre_tax_amount=Decimal("1000"),
		tax_components=[
			{"name": "vat", "amount": "160"},
			{"name": "excise_duty", "amount": "150"},
		],
		total_tax=Decimal("310"),
		total_with_tax=Decimal("1310"),
		currency="KES",
	)
	assert t.total_tax == Decimal("310")


def test_revenue_assurance_result():
	r = RevenueAssuranceResult(
		tenant_id="t1",
		period_start=datetime(2026, 5, 1, tzinfo=timezone.utc),
		period_end=datetime(2026, 5, 31, tzinfo=timezone.utc),
		total_cdrs=10000,
		unrated_cdrs=50,
		leakage_pct=Decimal("0.5"),
		collection_rate_pct=Decimal("94.2"),
		dso_days=Decimal("22.5"),
		arpu=Decimal("1250.00"),
		anomalies=[{"type": "unrated_cdr", "count": 50}],
		currency="KES",
	)
	assert r.leakage_pct == Decimal("0.5")
