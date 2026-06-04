"""Async service tests for telecom/bil — rating, charging, invoicing, disputes.

Uses importlib.util pattern consistent with existing test_contract.py.
Plain async functions with loop.run_until_complete — no @pytest.mark.asyncio.

© 2025 Datacraft. All rights reserved.
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from decimal import Decimal
from pathlib import Path

import pytest

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


_cc = _load("_cc_svc_async", PACKAGE_DIR / "capability_contract.py")
_svc_mod = _load("_svc_async", PACKAGE_DIR / "service.py")
TelecomBillingService = _svc_mod.TelecomBillingService


def run(coro):
	loop = asyncio.get_event_loop()
	return loop.run_until_complete(coro)


@pytest.fixture
def svc():
	return TelecomBillingService(tenant_id="t1", actor_id="tester")


# ---------------------------------------------------------------------------
# Rating: voice
# ---------------------------------------------------------------------------

def test_rate_voice_on_net(svc):
	result = run(svc.rate_voice_call({
		"subscriber_id": "sub-001",
		"duration_seconds": 60,
		"call_type": "on_net",
		"tariff_plan_id": "tp-standard",
		"currency": "KES",
	}))
	assert result["call_type"] == "on_net"
	assert Decimal(result["total_charge"]) > Decimal("0")
	assert Decimal(result["tax_amount"]) > Decimal("0")


def test_rate_voice_off_net_more_expensive(svc):
	on_net = run(svc.rate_voice_call({
		"subscriber_id": "sub-001",
		"duration_seconds": 60,
		"call_type": "on_net",
	}))
	off_net = run(svc.rate_voice_call({
		"subscriber_id": "sub-001",
		"duration_seconds": 60,
		"call_type": "off_net",
	}))
	assert Decimal(off_net["total_charge"]) > Decimal(on_net["total_charge"])


def test_rate_voice_international_most_expensive(svc):
	intl = run(svc.rate_voice_call({
		"subscriber_id": "sub-001",
		"duration_seconds": 60,
		"call_type": "international",
	}))
	off_net = run(svc.rate_voice_call({
		"subscriber_id": "sub-001",
		"duration_seconds": 60,
		"call_type": "off_net",
	}))
	assert Decimal(intl["total_charge"]) > Decimal(off_net["total_charge"])


def test_rate_voice_zero_duration(svc):
	result = run(svc.rate_voice_call({
		"subscriber_id": "sub-001",
		"duration_seconds": 0,
		"call_type": "on_net",
	}))
	assert Decimal(result["total_charge"]) == Decimal("0")


def test_rate_voice_with_bundle_deduction(svc):
	svc._bundles["bundle-voice-1"] = {
		"bundle_id": "bundle-voice-1",
		"bundle_type": "voice",
		"subscriber_id": "sub-bundle",
		"remaining_units": Decimal("10"),
		"consumed_units": Decimal("0"),
		"status": "active",
	}
	result = run(svc.rate_voice_call({
		"subscriber_id": "sub-bundle",
		"duration_seconds": 300,  # 5 minutes — within bundle
		"call_type": "on_net",
		"bundle_id": "bundle-voice-1",
	}))
	assert Decimal(result["bundle_units_used_minutes"]) > Decimal("0")
	assert Decimal(result["bundle_deduction"]) > Decimal("0")
	assert svc._bundles["bundle-voice-1"]["remaining_units"] < Decimal("10")


def test_rate_voice_roaming_surcharge(svc):
	domestic = run(svc.rate_voice_call({
		"subscriber_id": "sub-001",
		"duration_seconds": 60,
		"call_type": "on_net",
	}))
	roaming = run(svc.rate_voice_call({
		"subscriber_id": "sub-001",
		"duration_seconds": 60,
		"call_type": "on_net",
		"roaming_zone": "zone_a",
	}))
	assert Decimal(roaming["total_charge"]) > Decimal(domestic["total_charge"])


# ---------------------------------------------------------------------------
# Rating: data
# ---------------------------------------------------------------------------

def test_rate_data_session(svc):
	result = run(svc.rate_data_session({
		"subscriber_id": "sub-001",
		"data_volume_bytes": 10 * 1024 * 1024,
	}))
	assert Decimal(result["total_charge"]) > Decimal("0")
	assert result["subscriber_id"] == "sub-001"


def test_rate_data_zero_bytes(svc):
	result = run(svc.rate_data_session({
		"subscriber_id": "sub-001",
		"data_volume_bytes": 0,
	}))
	assert Decimal(result["total_charge"]) == Decimal("0")


def test_rate_data_with_bundle(svc):
	svc._bundles["bundle-data-1"] = {
		"bundle_id": "bundle-data-1",
		"bundle_type": "data",
		"subscriber_id": "sub-data",
		"remaining_units": Decimal("100"),
		"consumed_units": Decimal("0"),
		"status": "active",
	}
	result = run(svc.rate_data_session({
		"subscriber_id": "sub-data",
		"data_volume_bytes": 50 * 1024 * 1024,
		"bundle_id": "bundle-data-1",
	}))
	assert Decimal(result["bundle_mb_used"]) > Decimal("0")
	assert Decimal(result["bundle_deduction"]) > Decimal("0")


def test_rate_data_tiered(svc):
	result = run(svc.rate_data_session({
		"subscriber_id": "sub-001",
		"data_volume_bytes": 10 * 1024 * 1024 * 1024,
	}))
	assert Decimal(result["tiered_charge"]) > Decimal("0")


# ---------------------------------------------------------------------------
# Rating: SMS
# ---------------------------------------------------------------------------

def test_rate_sms_on_net(svc):
	result = run(svc.rate_sms({
		"subscriber_id": "sub-001",
		"sms_count": 5,
		"sms_type": "on_net",
	}))
	assert result["billable_count"] == "5"
	assert Decimal(result["total_charge"]) > Decimal("0")


def test_rate_sms_international_more_expensive(svc):
	local = run(svc.rate_sms({"subscriber_id": "sub-001", "sms_count": 1, "sms_type": "on_net"}))
	intl = run(svc.rate_sms({"subscriber_id": "sub-001", "sms_count": 1, "sms_type": "international"}))
	assert Decimal(intl["total_charge"]) > Decimal(local["total_charge"])


# ---------------------------------------------------------------------------
# Rating: roaming
# ---------------------------------------------------------------------------

def test_rate_roaming_voice(svc):
	result = run(svc.rate_roaming_event({
		"subscriber_id": "sub-roam",
		"visited_network": "SAFARICOM-KE",
		"home_network": "MTN-UG",
		"zone": "zone_a",
		"service_type": "voice",
		"duration_seconds": 120,
	}))
	assert Decimal(result["total_charge"]) > Decimal("0")
	assert result["zone"] == "zone_a"


def test_rate_roaming_data(svc):
	result = run(svc.rate_roaming_event({
		"subscriber_id": "sub-roam",
		"visited_network": "SAFARICOM-KE",
		"home_network": "MTN-UG",
		"zone": "zone_b",
		"service_type": "data",
		"data_volume_bytes": 1024 * 1024,
	}))
	assert Decimal(result["total_charge"]) > Decimal("0")


def test_rate_roaming_premium_zone_most_expensive(svc):
	a = run(svc.rate_roaming_event({
		"subscriber_id": "s", "visited_network": "N", "zone": "zone_a",
		"service_type": "voice", "duration_seconds": 60,
	}))
	p = run(svc.rate_roaming_event({
		"subscriber_id": "s", "visited_network": "N", "zone": "premium",
		"service_type": "voice", "duration_seconds": 60,
	}))
	assert Decimal(p["total_charge"]) > Decimal(a["total_charge"])


# ---------------------------------------------------------------------------
# Real-time balance
# ---------------------------------------------------------------------------

def test_balance_check_sufficient(svc):
	svc._balances["sub-prepaid"]["main_balance"] = Decimal("500")
	result = run(svc.real_time_balance_check("sub-prepaid", "voice", Decimal("100")))
	assert result["sufficient"] is True
	assert result["deficit"] == "0"


def test_balance_check_insufficient(svc):
	svc._balances["sub-broke"]["main_balance"] = Decimal("10")
	result = run(svc.real_time_balance_check("sub-broke", "voice", Decimal("100")))
	assert result["sufficient"] is False
	assert Decimal(result["deficit"]) == Decimal("90")


def test_balance_check_zero_amount(svc):
	result = run(svc.real_time_balance_check("sub-any", "voice", Decimal("0")))
	assert result["sufficient"] is True


# ---------------------------------------------------------------------------
# Bundle consumption
# ---------------------------------------------------------------------------

def test_bundle_consumption_partial(svc):
	svc._bundles["b-voice-1"] = {
		"bundle_id": "b-voice-1",
		"bundle_type": "voice",
		"subscriber_id": "sub-bundle",
		"remaining_units": Decimal("100"),
		"consumed_units": Decimal("0"),
		"status": "active",
	}
	result = run(svc.bundle_consumption("sub-bundle", "voice", Decimal("30")))
	assert result["bundle_found"] is True
	assert result["consumed"] == "30"
	assert result["remaining_units"] == "70"
	assert result["overage_units"] == "0"
	assert result["exhausted"] is False


def test_bundle_consumption_with_overage(svc):
	svc._bundles["b-data-1"] = {
		"bundle_id": "b-data-1",
		"bundle_type": "data",
		"subscriber_id": "sub-bundle2",
		"remaining_units": Decimal("50"),
		"consumed_units": Decimal("0"),
		"status": "active",
	}
	result = run(svc.bundle_consumption("sub-bundle2", "data", Decimal("80")))
	assert result["consumed"] == "50"
	assert result["overage_units"] == "30"
	assert result["exhausted"] is True


def test_bundle_consumption_no_bundle(svc):
	result = run(svc.bundle_consumption("sub-nobody", "voice", Decimal("10")))
	assert result["bundle_found"] is False
	assert result["overage_units"] == "10"


# ---------------------------------------------------------------------------
# Overage charging
# ---------------------------------------------------------------------------

def test_overage_charging(svc):
	svc._bundles["b-over-1"] = {
		"bundle_id": "b-over-1",
		"bundle_type": "data",
		"subscriber_id": "sub-over",
		"remaining_units": Decimal("0"),
		"consumed_units": Decimal("100"),
		"status": "exhausted",
	}
	svc._balances["sub-over"]["main_balance"] = Decimal("200")
	result = run(svc.overage_charging("sub-over", "b-over-1", Decimal("10")))
	assert Decimal(result["total_charge"]) > Decimal("0")
	assert svc._balances["sub-over"]["main_balance"] < Decimal("200")


# ---------------------------------------------------------------------------
# Invoice generation
# ---------------------------------------------------------------------------

def test_generate_bill(svc):
	svc.record_charge("chg-1", "cust-gen", "recurring", "flat_rate", 1000.0, "KES", 160.0)
	result = run(svc.generate_bill("cust-gen", {"start": "2026-05-01", "end": "2026-05-31"}))
	assert result["account_id"] == "cust-gen"
	assert result["status"] == "draft"
	assert Decimal(result["total_amount"]) > Decimal("0")


def test_bill_calculation_breakdown(svc):
	svc.record_charge("chg-a", "cust-breakdown", "recurring", "flat_rate", 500.0, "KES", 80.0)
	svc.record_charge("chg-b", "cust-breakdown", "usage_based", "tiered", 300.0, "KES", 48.0)
	result = run(svc.bill_calculation("cust-breakdown", {"start": "2026-05-01", "end": "2026-05-31"}))
	assert len(result["line_items"]) == 2
	assert "recurring" in result["by_charge_type"]
	assert "usage_based" in result["by_charge_type"]


def test_apply_adjustments_credit(svc):
	result = run(svc.apply_adjustments("inv-adj-1", "credit", Decimal("50"), "Goodwill credit"))
	assert result["adjustment_type"] == "credit"
	assert result["invoice_id"] == "inv-adj-1"
	assert result["amount"] == "50"


def test_apply_adjustments_invalid_type(svc):
	with pytest.raises(AssertionError):
		run(svc.apply_adjustments("inv-1", "bribe", Decimal("50"), "bad"))


def test_bill_delivery_valid(svc):
	svc.generate_invoice("inv-del-1", "cust-del", "cyc-1", 500.0, "KES", "2026-06-15")
	result = run(svc.bill_delivery("inv-del-1", "email"))
	assert result["status"] == "delivered"
	assert result["channel"] == "email"


def test_bill_delivery_invalid_channel(svc):
	with pytest.raises(AssertionError):
		run(svc.bill_delivery("inv-1", "telegram"))


def test_view_bill_not_found(svc):
	result = run(svc.view_bill("inv-nonexistent"))
	assert result["found"] is False


def test_generate_bill_run(svc):
	svc.record_charge("chg-run-1", "cust-run-a", "recurring", "flat_rate", 100.0, "KES", 16.0)
	svc.record_charge("chg-run-2", "cust-run-b", "recurring", "flat_rate", 200.0, "KES", 32.0)
	result = run(svc.generate_bill_run("2026-05-31"))
	assert result["status"] in {"completed", "completed_with_errors"}
	assert result["accounts_processed"] >= 2


# ---------------------------------------------------------------------------
# Payment processing
# ---------------------------------------------------------------------------

def test_payment_processing(svc):
	result = run(svc.payment_processing("acc-pay-1", Decimal("500"), "mobile_money", "MPESA-REF-001"))
	assert result["status"] == "received"
	assert result["payment_method"] == "mobile_money"
	assert svc._balances["acc-pay-1"]["main_balance"] == Decimal("500")


def test_payment_processing_invalid_method(svc):
	with pytest.raises(ValueError, match="not supported"):
		run(svc.payment_processing("acc-1", Decimal("500"), "bitcoin_lightning", "ref"))


def test_payment_processing_zero_amount(svc):
	with pytest.raises(AssertionError):
		run(svc.payment_processing("acc-1", Decimal("0"), "mobile_money", "ref"))


# ---------------------------------------------------------------------------
# Dunning workflow
# ---------------------------------------------------------------------------

def test_dunning_zero_dpd(svc):
	result = run(svc.dunning_workflow("acc-good", 0))
	assert result["action"] == "none"


def test_dunning_reminder_1(svc):
	result = run(svc.dunning_workflow("acc-late", 5))
	assert result["dunning_step"] == "reminder_1"
	assert result["suspended"] is False


def test_dunning_reminder_2(svc):
	result = run(svc.dunning_workflow("acc-late", 10))
	assert result["dunning_step"] == "reminder_2"


def test_dunning_suspension_warning(svc):
	result = run(svc.dunning_workflow("acc-late", 17))
	assert result["dunning_step"] == "suspension_warning"


def test_dunning_service_suspended(svc):
	result = run(svc.dunning_workflow("acc-very-late", 25))
	assert result["dunning_step"] == "service_suspended"
	assert result["suspended"] is True
	assert "acc-very-late" in svc._suspended_accounts


def test_dunning_legal_notice(svc):
	result = run(svc.dunning_workflow("acc-writeoff", 60))
	assert result["dunning_step"] == "legal_notice"


# ---------------------------------------------------------------------------
# Service suspension / restoration
# ---------------------------------------------------------------------------

def test_service_suspension_and_restoration(svc):
	run(svc.service_suspension("acc-sus-1", "non_payment"))
	assert "acc-sus-1" in svc._suspended_accounts

	svc.record_payment("pay-restore", "inv-x", "mobile_money", 500.0, "KES", "MPESA-X", "2026-06-01")
	result = run(svc.service_restoration("acc-sus-1", "pay-restore"))
	assert result["action"] == "restored"
	assert "acc-sus-1" not in svc._suspended_accounts


def test_restore_non_suspended_account(svc):
	svc.record_payment("pay-ns", "inv-y", "cash", 100.0, "KES", "CASH-Y", "2026-06-01")
	result = run(svc.service_restoration("acc-not-suspended", "pay-ns"))
	assert result["action"] == "not_suspended"


# ---------------------------------------------------------------------------
# Disputes
# ---------------------------------------------------------------------------

def test_raise_billing_dispute(svc):
	svc.generate_invoice("inv-disp-1", "cust-disp", "cyc-1", 1000.0, "KES", "2026-06-15")
	result = run(svc.raise_billing_dispute("acc-disp", "inv-disp-1", Decimal("200"), "Double billing"))
	assert result["status"] == "open"
	assert result["disputed_amount"] == "200"


def test_investigate_dispute(svc):
	svc.generate_invoice("inv-disp-2", "cust-disp2", "cyc-1", 1000.0, "KES", "2026-06-15")
	dispute = run(svc.raise_billing_dispute("acc-disp2", "inv-disp-2", Decimal("100"), "Wrong rate"))
	result = run(svc.investigate_dispute(dispute["dispute_id"], {"findings": "rate correct"}))
	assert result["status"] == "under_review"


def test_resolve_dispute_upheld(svc):
	svc.generate_invoice("inv-disp-3", "cust-disp3", "cyc-1", 1000.0, "KES", "2026-06-15")
	dispute = run(svc.raise_billing_dispute("acc-disp3", "inv-disp-3", Decimal("300"), "Overcharge"))
	run(svc.investigate_dispute(dispute["dispute_id"], {}))
	result = run(svc.resolve_dispute(dispute["dispute_id"], "upheld", Decimal("300")))
	assert result["status"] == "resolved_upheld"
	assert result["credit_applied"] is True


def test_resolve_dispute_rejected(svc):
	svc.generate_invoice("inv-disp-4", "cust-disp4", "cyc-1", 1000.0, "KES", "2026-06-15")
	dispute = run(svc.raise_billing_dispute("acc-disp4", "inv-disp-4", Decimal("100"), "Claim"))
	run(svc.investigate_dispute(dispute["dispute_id"], {}))
	result = run(svc.resolve_dispute(dispute["dispute_id"], "rejected", Decimal("0")))
	assert result["status"] == "resolved_rejected"
	assert result["credit_applied"] is False


def test_dispute_analytics(svc):
	for i in range(3):
		svc.generate_invoice(f"inv-da-{i}", f"cust-da-{i}", "cyc-1", 500.0, "KES", "2026-06-15")
		run(svc.raise_billing_dispute(f"acc-da-{i}", f"inv-da-{i}", Decimal("100"), "Test"))
	result = run(svc.dispute_analytics({"start": "2026-01-01", "end": "2026-12-31"}))
	assert result["total_disputes"] >= 3


# ---------------------------------------------------------------------------
# Revenue assurance
# ---------------------------------------------------------------------------

def test_revenue_leakage_detection(svc):
	svc.record_cdr("cdr-leak-1", "MSC-01", "raw", "+254700000001", 60, 0, "2026-05-01T10:00:00Z")
	svc.record_cdr("cdr-leak-2", "MSC-01", "rated", "+254700000002", 30, 0, "2026-05-01T11:00:00Z")
	result = run(svc.revenue_leakage_detection({"start": "2026-05-01", "end": "2026-05-31"}))
	assert result["total_cdrs"] >= 2
	assert "leakage_pct" in result
	assert result["currency"] == "KES"


def test_interconnect_reconciliation(svc):
	svc.record_charge("ic-chg-1", "carrier-001", "interconnect", "flat_rate", 10000.0, "KES", 0.0)
	result = run(svc.interconnect_reconciliation("carrier-001", {"start": "2026-05-01", "end": "2026-05-31"}))
	assert "our_receivable" in result
	assert result["status"] in {"agreed", "disputed"}


def test_revenue_report(svc):
	svc.record_charge("rpt-chg-1", "cust-rpt", "recurring", "flat_rate", 5000.0, "KES", 800.0)
	result = run(svc.revenue_report({"start": "2026-05-01", "end": "2026-05-31"}))
	assert Decimal(result["total_revenue"]) > Decimal("0")
	assert "by_charge_type" in result


def test_arpu_analysis(svc):
	for i in range(5):
		svc.record_charge(f"arpu-chg-{i}", f"sub-arpu-{i}", "recurring", "flat_rate", 1000.0, "KES", 160.0)
	result = run(svc.arpu_analysis({"start": "2026-05-01", "end": "2026-05-31"}))
	assert result["unique_subscribers"] >= 5
	assert Decimal(result["arpu"]) == Decimal("1000.00")


def test_churn_revenue_impact(svc):
	result = run(svc.churn_revenue_impact({"start": "2026-05-01", "end": "2026-05-31"}))
	assert "active_subscriber_count" in result
	assert "churn_revenue_impact_pct" in result


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------

def test_apply_promotion(svc):
	result = run(svc.apply_promotion("sub-promo", "LAUNCH10", "2026-01-01", "2026-12-31"))
	assert result["promo_code"] == "LAUNCH10"
	assert result["subscriber_id"] == "sub-promo"


def test_promotion_max_redemptions(svc):
	svc._promotions["LIMITED"] = {
		"promo_code": "LIMITED",
		"discount_pct": Decimal("5"),
		"bonus_units": Decimal("0"),
		"status": "active",
		"redemptions": 1,
		"max_redemptions": 1,
	}
	with pytest.raises(ValueError, match="redemption limit"):
		run(svc.apply_promotion("sub-1", "LIMITED", "2026-01-01", "2026-12-31"))


def test_promotion_inactive(svc):
	svc._promotions["EXPIRED"] = {
		"promo_code": "EXPIRED",
		"discount_pct": Decimal("5"),
		"bonus_units": Decimal("0"),
		"status": "expired",
		"redemptions": 0,
	}
	with pytest.raises(ValueError, match="not active"):
		run(svc.apply_promotion("sub-1", "EXPIRED", "2026-01-01", "2026-12-31"))


# ---------------------------------------------------------------------------
# Multi-tenant isolation
# ---------------------------------------------------------------------------

def test_async_tenant_isolation():
	svc_a = TelecomBillingService(tenant_id="tenant-iso-a", actor_id="actor-a")
	svc_b = TelecomBillingService(tenant_id="tenant-iso-b", actor_id="actor-b")

	run(svc_a.rate_voice_call({"subscriber_id": "sub-a", "duration_seconds": 60, "call_type": "on_net"}))
	run(svc_b.rate_voice_call({"subscriber_id": "sub-b", "duration_seconds": 60, "call_type": "on_net"}))

	a_events = [e for e in svc_a.audit_events if e["tenant_id"] == "tenant-iso-a"]
	b_events = [e for e in svc_b.audit_events if e["tenant_id"] == "tenant-iso-b"]
	assert len(a_events) == 1
	assert len(b_events) == 1
