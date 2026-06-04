"""Tests for domain rules and calculations — telecom/bil.

© 2025 Datacraft. All rights reserved.
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
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


_rules = _load("_rules_test", PACKAGE_DIR / "domain" / "rules.py")
_calcs = _load("_calcs_test", PACKAGE_DIR / "domain" / "calculations.py")

RuleViolation = _rules.RuleViolation
assert_tenant_context = _rules.assert_tenant_context
assert_write_policy = _rules.assert_write_policy
assert_no_cross_tenant_access = _rules.assert_no_cross_tenant_access
assert_actor_present = _rules.assert_actor_present
assert_cdr_source_present = _rules.assert_cdr_source_present
assert_cdr_msisdn_present = _rules.assert_cdr_msisdn_present
assert_cdr_type_supported = _rules.assert_cdr_type_supported
assert_cdr_not_duplicate = _rules.assert_cdr_not_duplicate
assert_cdr_duration_non_negative = _rules.assert_cdr_duration_non_negative
assert_cdr_mediation_status_valid = _rules.assert_cdr_mediation_status_valid
assert_tariff_plan_active = _rules.assert_tariff_plan_active
assert_tariff_plan_valid_date = _rules.assert_tariff_plan_valid_date
assert_rate_non_negative = _rules.assert_rate_non_negative
assert_bundle_active = _rules.assert_bundle_active
assert_bundle_not_expired = _rules.assert_bundle_not_expired
assert_bundle_has_units = _rules.assert_bundle_has_units
assert_invoice_in_draft = _rules.assert_invoice_in_draft
assert_invoice_approvable = _rules.assert_invoice_approvable
assert_invoice_not_paid = _rules.assert_invoice_not_paid
assert_invoice_amount_positive = _rules.assert_invoice_amount_positive
assert_approval_reference_present = _rules.assert_approval_reference_present
assert_payment_method_supported = _rules.assert_payment_method_supported
assert_payment_amount_positive = _rules.assert_payment_amount_positive
assert_discount_pct_within_limit = _rules.assert_discount_pct_within_limit
assert_discount_not_expired = _rules.assert_discount_not_expired
assert_credit_limit_sane = _rules.assert_credit_limit_sane
assert_within_credit_limit = _rules.assert_within_credit_limit
assert_roaming_zone_valid = _rules.assert_roaming_zone_valid
assert_settlement_period_valid = _rules.assert_settlement_period_valid
assert_carrier_id_present = _rules.assert_carrier_id_present
assert_dispute_amount_valid = _rules.assert_dispute_amount_valid
assert_dispute_resolution_valid = _rules.assert_dispute_resolution_valid
assert_dispute_open = _rules.assert_dispute_open
assert_dunning_step_valid = _rules.assert_dunning_step_valid
assert_dunning_sequence = _rules.assert_dunning_sequence
assert_leakage_pct_acceptable = _rules.assert_leakage_pct_acceptable
assert_collection_rate_acceptable = _rules.assert_collection_rate_acceptable
calculate_dpd = _rules.calculate_dpd
calculate_outstanding_balance = _rules.calculate_outstanding_balance
calculate_penalty_accrual = _rules.calculate_penalty_accrual
calculate_realtime_spend_headroom = _rules.calculate_realtime_spend_headroom

round_currency = _calcs.round_currency
calculate_voice_charge = _calcs.calculate_voice_charge
calculate_tiered_voice_charge = _calcs.calculate_tiered_voice_charge
calculate_time_of_day_voice_charge = _calcs.calculate_time_of_day_voice_charge
calculate_data_charge = _calcs.calculate_data_charge
calculate_data_charge_gb = _calcs.calculate_data_charge_gb
calculate_volume_data_charge = _calcs.calculate_volume_data_charge
calculate_sms_charge = _calcs.calculate_sms_charge
calculate_tax = _calcs.calculate_tax
calculate_tax_inclusive_split = _calcs.calculate_tax_inclusive_split
calculate_multi_tax = _calcs.calculate_multi_tax
calculate_jurisdiction_tax = _calcs.calculate_jurisdiction_tax
apply_percentage_discount = _calcs.apply_percentage_discount
apply_flat_discount = _calcs.apply_flat_discount
calculate_combined_discount = _calcs.calculate_combined_discount
calculate_bundle_overage = _calcs.calculate_bundle_overage
calculate_roaming_charge = _calcs.calculate_roaming_charge
calculate_tap_settlement_amount = _calcs.calculate_tap_settlement_amount
calculate_interconnect_net = _calcs.calculate_interconnect_net
calculate_termination_charge = _calcs.calculate_termination_charge
calculate_transit_charge = _calcs.calculate_transit_charge
credit_utilisation_pct = _calcs.credit_utilisation_pct
headroom = _calcs.headroom
is_over_soft_limit = _calcs.is_over_soft_limit
is_over_hard_limit = _calcs.is_over_hard_limit
aggregate_invoice_totals = _calcs.aggregate_invoice_totals
calculate_late_payment_penalty = _calcs.calculate_late_payment_penalty
leakage_rate_pct = _calcs.leakage_rate_pct
collection_rate_pct = _calcs.collection_rate_pct
arpu = _calcs.arpu
days_sales_outstanding = _calcs.days_sales_outstanding
calculate_convergent_bill = _calcs.calculate_convergent_bill
calculate_spend_velocity = _calcs.calculate_spend_velocity
VOICE_PULSE_SECONDS = _calcs.VOICE_PULSE_SECONDS
DEFAULT_VAT_PCT = _calcs.DEFAULT_VAT_PCT


# ---------------------------------------------------------------------------
# Rules: tenant / auth
# ---------------------------------------------------------------------------

def test_tenant_context_required():
	with pytest.raises(RuleViolation, match="tenant_context_required"):
		assert_tenant_context({})


def test_tenant_context_blank_rejected():
	with pytest.raises(RuleViolation):
		assert_tenant_context({"tenant_id": "  "})


def test_tenant_context_valid():
	assert_tenant_context({"tenant_id": "t1"})  # no exception


def test_write_requires_policy():
	with pytest.raises(RuleViolation, match="write_requires_policy"):
		assert_write_policy({"operation_type": "write", "policy_attached": False})


def test_read_no_policy_required():
	assert_write_policy({"operation_type": "read"})  # no exception


def test_cross_tenant_access_denied():
	with pytest.raises(RuleViolation, match="cross_tenant_access_denied"):
		assert_no_cross_tenant_access("tenant-a", "tenant-b")


def test_same_tenant_ok():
	assert_no_cross_tenant_access("t1", "t1")  # no exception


def test_actor_required():
	with pytest.raises(RuleViolation, match="actor_required"):
		assert_actor_present("")


# ---------------------------------------------------------------------------
# Rules: CDR
# ---------------------------------------------------------------------------

def test_cdr_source_required():
	with pytest.raises(RuleViolation, match="cdr_source_required"):
		assert_cdr_source_present("")


def test_cdr_msisdn_required():
	with pytest.raises(RuleViolation, match="cdr_msisdn_required"):
		assert_cdr_msisdn_present(None)


def test_cdr_type_unknown():
	with pytest.raises(RuleViolation, match="cdr_type_not_supported"):
		assert_cdr_type_supported("fax")


def test_cdr_type_valid():
	for t in ("voice", "sms", "data", "roaming"):
		assert_cdr_type_supported(t)  # no exception


def test_cdr_duplicate_detected():
	with pytest.raises(RuleViolation, match="duplicate_cdr"):
		assert_cdr_not_duplicate("cdr-001", {"cdr-001", "cdr-002"})


def test_cdr_not_duplicate():
	assert_cdr_not_duplicate("cdr-003", {"cdr-001", "cdr-002"})  # no exception


def test_cdr_negative_duration():
	with pytest.raises(RuleViolation, match="cdr_negative_duration"):
		assert_cdr_duration_non_negative(-1)


def test_cdr_mediation_status_invalid():
	with pytest.raises(RuleViolation, match="cdr_mediation_status_invalid"):
		assert_cdr_mediation_status_valid("processed")


# ---------------------------------------------------------------------------
# Rules: tariff
# ---------------------------------------------------------------------------

def test_tariff_plan_inactive_blocked():
	with pytest.raises(RuleViolation, match="tariff_plan_inactive"):
		assert_tariff_plan_active(False, "tp-001")


def test_tariff_plan_active_ok():
	assert_tariff_plan_active(True, "tp-001")


def test_tariff_not_yet_effective():
	future = datetime(2030, 1, 1, tzinfo=timezone.utc)
	now = datetime(2026, 1, 1, tzinfo=timezone.utc)
	with pytest.raises(RuleViolation, match="tariff_not_yet_effective"):
		assert_tariff_plan_valid_date(future, None, now)


def test_tariff_expired():
	past = datetime(2020, 1, 1, tzinfo=timezone.utc)
	now = datetime(2026, 1, 1, tzinfo=timezone.utc)
	with pytest.raises(RuleViolation, match="tariff_expired"):
		assert_tariff_plan_valid_date(past, datetime(2021, 1, 1, tzinfo=timezone.utc), now)


def test_negative_rate_rejected():
	with pytest.raises(RuleViolation, match="negative_rate"):
		assert_rate_non_negative(Decimal("-0.01"), "rate_per_second")


# ---------------------------------------------------------------------------
# Rules: bundle
# ---------------------------------------------------------------------------

def test_bundle_inactive_blocked():
	with pytest.raises(RuleViolation, match="bundle_not_active"):
		assert_bundle_active("exhausted", "bundle-001")


def test_bundle_expired():
	with pytest.raises(RuleViolation, match="bundle_expired"):
		assert_bundle_not_expired(
			datetime(2026, 1, 1, tzinfo=timezone.utc),
			datetime(2026, 6, 1, tzinfo=timezone.utc),
			"bundle-001",
		)


def test_bundle_no_units():
	with pytest.raises(RuleViolation, match="bundle_exhausted"):
		assert_bundle_has_units(Decimal("0"), "bundle-001")


# ---------------------------------------------------------------------------
# Rules: invoice
# ---------------------------------------------------------------------------

def test_invoice_not_draft():
	with pytest.raises(RuleViolation, match="invoice_not_draft"):
		assert_invoice_in_draft("approved", "inv-001")


def test_invoice_not_approvable():
	with pytest.raises(RuleViolation, match="invoice_not_approvable"):
		assert_invoice_approvable("paid", "inv-001")


def test_invoice_already_closed():
	with pytest.raises(RuleViolation, match="invoice_already_closed"):
		assert_invoice_not_paid("paid", "inv-001")


def test_approval_reference_required():
	with pytest.raises(RuleViolation, match="approval_reference_required"):
		assert_approval_reference_present(None)


# ---------------------------------------------------------------------------
# Rules: payment
# ---------------------------------------------------------------------------

def test_payment_method_unsupported():
	with pytest.raises(RuleViolation, match="payment_method_not_supported"):
		assert_payment_method_supported("barter")


def test_payment_methods_supported():
	for m in ("mobile_money", "bank_transfer", "credit_card", "cash"):
		assert_payment_method_supported(m)


def test_payment_non_positive_rejected():
	with pytest.raises(RuleViolation, match="payment_amount_must_be_positive"):
		assert_payment_amount_positive(Decimal("0"))


# ---------------------------------------------------------------------------
# Rules: discount
# ---------------------------------------------------------------------------

def test_discount_exceeds_max():
	with pytest.raises(RuleViolation, match="discount_exceeds_max_allowed"):
		assert_discount_pct_within_limit(Decimal("51"))


def test_discount_at_boundary_ok():
	assert_discount_pct_within_limit(Decimal("50"))


# ---------------------------------------------------------------------------
# Rules: credit limit
# ---------------------------------------------------------------------------

def test_soft_must_be_below_hard():
	with pytest.raises(RuleViolation, match="soft_limit_must_be_below_hard_limit"):
		assert_credit_limit_sane(Decimal("10000"), Decimal("10000"))


def test_credit_hard_limit_breached():
	with pytest.raises(RuleViolation, match="credit_hard_limit_breached"):
		assert_within_credit_limit(Decimal("10000"), Decimal("10000"), "acc-1")


# ---------------------------------------------------------------------------
# Rules: roaming
# ---------------------------------------------------------------------------

def test_invalid_roaming_zone():
	with pytest.raises(RuleViolation, match="invalid_roaming_zone"):
		assert_roaming_zone_valid("moon")


def test_valid_roaming_zones():
	for z in ("domestic", "zone_a", "zone_b", "zone_c", "premium", "global"):
		assert_roaming_zone_valid(z)


# ---------------------------------------------------------------------------
# Rules: settlement
# ---------------------------------------------------------------------------

def test_settlement_period_invalid():
	with pytest.raises(RuleViolation, match="settlement_period_invalid"):
		assert_settlement_period_valid(
			datetime(2026, 5, 31, tzinfo=timezone.utc),
			datetime(2026, 5, 1, tzinfo=timezone.utc),
		)


def test_carrier_id_required():
	with pytest.raises(RuleViolation, match="carrier_id_required"):
		assert_carrier_id_present("")


# ---------------------------------------------------------------------------
# Rules: dispute
# ---------------------------------------------------------------------------

def test_disputed_amount_exceeds_invoice():
	with pytest.raises(RuleViolation, match="disputed_amount_exceeds_invoice"):
		assert_dispute_amount_valid(Decimal("1001"), Decimal("1000"))


def test_invalid_dispute_resolution():
	with pytest.raises(RuleViolation, match="invalid_dispute_resolution"):
		assert_dispute_resolution_valid("maybe")


def test_dispute_already_closed():
	with pytest.raises(RuleViolation, match="dispute_already_closed"):
		assert_dispute_open("resolved_upheld", "disp-001")


# ---------------------------------------------------------------------------
# Rules: dunning
# ---------------------------------------------------------------------------

def test_dunning_step_invalid():
	with pytest.raises(RuleViolation, match="dunning_step_not_supported"):
		assert_dunning_step_valid("nudge")


def test_dunning_de_escalation_blocked():
	with pytest.raises(RuleViolation, match="dunning_de_escalation_not_permitted"):
		assert_dunning_sequence("service_suspended", "reminder_1")


def test_dunning_escalation_allowed():
	assert_dunning_sequence("reminder_1", "reminder_2")  # no exception
	assert_dunning_sequence(None, "reminder_1")  # first step — no exception


# ---------------------------------------------------------------------------
# Rules: revenue assurance
# ---------------------------------------------------------------------------

def test_leakage_threshold():
	with pytest.raises(RuleViolation, match="revenue_leakage_threshold_breached"):
		assert_leakage_pct_acceptable(Decimal("3"))


def test_collection_rate_threshold():
	with pytest.raises(RuleViolation, match="collection_rate_below_threshold"):
		assert_collection_rate_acceptable(Decimal("79"))


# ---------------------------------------------------------------------------
# Rules: calculations
# ---------------------------------------------------------------------------

def test_calculate_dpd_zero_when_not_due():
	due = datetime(2026, 6, 30, tzinfo=timezone.utc)
	now = datetime(2026, 6, 15, tzinfo=timezone.utc)
	assert calculate_dpd(due, now) == 0


def test_calculate_dpd_positive_when_overdue():
	due = datetime(2026, 6, 1, tzinfo=timezone.utc)
	now = datetime(2026, 6, 15, tzinfo=timezone.utc)
	assert calculate_dpd(due, now) == 14


def test_calculate_outstanding_balance():
	balance = calculate_outstanding_balance(Decimal("1000"), Decimal("400"), Decimal("100"))
	assert balance == Decimal("500")


def test_outstanding_never_negative():
	balance = calculate_outstanding_balance(Decimal("100"), Decimal("200"))
	assert balance == Decimal("0")


def test_calculate_penalty_accrual_capped():
	penalty = calculate_penalty_accrual(Decimal("1000"), 200)  # 200 days — should cap
	assert penalty <= Decimal("100")  # cap is 10%


def test_spend_headroom_sufficient():
	result = calculate_realtime_spend_headroom(Decimal("4000"), Decimal("5000"), Decimal("500"))
	assert result["can_proceed"] is True
	assert result["headroom"] == Decimal("1000")


def test_spend_headroom_insufficient():
	result = calculate_realtime_spend_headroom(Decimal("4800"), Decimal("5000"), Decimal("500"))
	assert result["can_proceed"] is False


# ---------------------------------------------------------------------------
# Calculations: voice
# ---------------------------------------------------------------------------

def test_voice_charge_zero_duration():
	charge = calculate_voice_charge(0, Decimal("0.05"), minimum_charge=Decimal("0.50"))
	assert charge == Decimal("0.50")


def test_voice_charge_pulse_billing():
	# 10 seconds at 6s pulse = 2 pulses
	charge = calculate_voice_charge(10, Decimal("0.10"), pulse_seconds=6)
	assert charge == Decimal("0.12")  # 12 seconds * 0.10/s


def test_voice_charge_exact_pulse():
	charge = calculate_voice_charge(6, Decimal("0.10"), pulse_seconds=6)
	assert charge == Decimal("0.60")


def test_tiered_voice_charge():
	tiers = [
		{"up_to_seconds": 60, "rate_per_second": "0.10"},
		{"up_to_seconds": None, "rate_per_second": "0.05"},
	]
	charge = calculate_tiered_voice_charge(90, tiers)
	# 60s * 0.10 + 30s * 0.05 = 6.00 + 1.50 = 7.50
	assert charge == Decimal("7.50")


def test_time_of_day_peak():
	# peak 8-18, call at 10h
	peak = calculate_time_of_day_voice_charge(
		60, Decimal("0.10"), Decimal("0.05"), 8, 18, 10
	)
	off_peak = calculate_time_of_day_voice_charge(
		60, Decimal("0.10"), Decimal("0.05"), 8, 18, 22
	)
	assert peak > off_peak


# ---------------------------------------------------------------------------
# Calculations: data
# ---------------------------------------------------------------------------

def test_data_charge_kb():
	# 2048 bytes = 2 KB, rate 1.00/KB
	charge = calculate_data_charge(2048, Decimal("1.00"))
	assert charge == Decimal("2.00")


def test_data_charge_ceiling():
	# 1025 bytes → ceil to 2 KB
	charge = calculate_data_charge(1025, Decimal("1.00"))
	assert charge == Decimal("2.00")


def test_data_charge_zero():
	charge = calculate_data_charge(0, Decimal("1.00"), minimum_charge=Decimal("0.50"))
	assert charge == Decimal("0.50")


def test_data_charge_gb():
	charge = calculate_data_charge_gb(1_073_741_824, Decimal("100.00"))  # exactly 1 GB
	assert charge == Decimal("100.00")


def test_volume_data_tiers():
	tiers = [
		{"up_to_gb": 1, "rate_per_gb": "100"},
		{"up_to_gb": 10, "rate_per_gb": "80"},
		{"up_to_gb": None, "rate_per_gb": "60"},
	]
	# 2 GB: 1*100 + 1*80 = 180
	charge = calculate_volume_data_charge(2 * 1_073_741_824, tiers)
	assert charge == Decimal("180.00")


# ---------------------------------------------------------------------------
# Calculations: SMS
# ---------------------------------------------------------------------------

def test_sms_charge():
	charge = calculate_sms_charge(5, Decimal("1.00"))
	assert charge == Decimal("5.00")


def test_sms_charge_minimum():
	charge = calculate_sms_charge(0, Decimal("1.00"), minimum_charge=Decimal("0.50"))
	assert charge == Decimal("0.50")


# ---------------------------------------------------------------------------
# Calculations: tax
# ---------------------------------------------------------------------------

def test_calculate_tax():
	tax = calculate_tax(Decimal("1000"), Decimal("16"))
	assert tax == Decimal("160.00")


def test_calculate_tax_zero():
	assert calculate_tax(Decimal("0"), Decimal("16")) == Decimal("0")


def test_tax_inclusive_split():
	net, tax = calculate_tax_inclusive_split(Decimal("1160"), Decimal("16"))
	assert net == Decimal("1000.00")
	assert tax == Decimal("160.00")


def test_multi_tax():
	components = [
		{"name": "vat", "rate_pct": "16"},
		{"name": "excise", "rate_pct": "15"},
	]
	result = calculate_multi_tax(Decimal("1000"), components)
	assert result["vat"] == Decimal("160.00")
	assert result["excise"] == Decimal("150.00")
	assert result["total_tax"] == Decimal("310.00")


def test_jurisdiction_tax_ke():
	result = calculate_jurisdiction_tax(Decimal("1000"), "KE")
	assert result["jurisdiction"] == "KE"
	assert result["total_tax"] > Decimal("0")


def test_jurisdiction_tax_unknown_zero():
	result = calculate_jurisdiction_tax(Decimal("1000"), "XX")
	assert result["total_tax"] == Decimal("0")


# ---------------------------------------------------------------------------
# Calculations: discounts
# ---------------------------------------------------------------------------

def test_percentage_discount():
	result = apply_percentage_discount(Decimal("1000"), Decimal("10"))
	assert result == Decimal("900.00")


def test_flat_discount():
	result = apply_flat_discount(Decimal("1000"), Decimal("200"))
	assert result == Decimal("800.00")


def test_flat_discount_floor_zero():
	result = apply_flat_discount(Decimal("100"), Decimal("200"))
	assert result == Decimal("0.00")


def test_combined_discount_cascade():
	result = calculate_combined_discount(Decimal("1000"), Decimal("10"), Decimal("50"))
	# 10% off 1000 = 900, then 50 flat = 850
	assert result["final_amount"] == Decimal("850.00")


def test_bundle_overage():
	overage = calculate_bundle_overage(Decimal("110"), Decimal("100"), Decimal("2.00"))
	assert overage == Decimal("20.00")


def test_bundle_no_overage():
	overage = calculate_bundle_overage(Decimal("90"), Decimal("100"), Decimal("2.00"))
	assert overage == Decimal("0")


# ---------------------------------------------------------------------------
# Calculations: roaming
# ---------------------------------------------------------------------------

def test_roaming_charge():
	zone_rates = {
		"zone_a": {
			"voice_rate_per_second": "0.15",
			"data_rate_per_kb": "0.05",
			"sms_rate": "3.00",
			"surcharge_pct": "10",
		}
	}
	result = calculate_roaming_charge(60, 0, 0, zone_rates, "zone_a")
	assert result["voice"] > Decimal("0")
	assert result["surcharge"] > Decimal("0")
	assert result["total"] > result["voice"]


def test_tap_settlement():
	result = calculate_tap_settlement_amount(
		Decimal("100"), Decimal("80"), Decimal("2.00")
	)
	assert result["receivable"] == Decimal("200.00")
	assert result["payable"] == Decimal("160.00")
	assert result["net"] == Decimal("40.00")


# ---------------------------------------------------------------------------
# Calculations: interconnect
# ---------------------------------------------------------------------------

def test_interconnect_net():
	net = calculate_interconnect_net(Decimal("10000"), Decimal("8000"))
	assert net == Decimal("2000.00")


def test_termination_charge():
	charge = calculate_termination_charge(Decimal("100"), Decimal("0.50"))
	assert charge == Decimal("50.00")


# ---------------------------------------------------------------------------
# Calculations: credit limit
# ---------------------------------------------------------------------------

def test_credit_utilisation():
	pct = credit_utilisation_pct(Decimal("4000"), Decimal("10000"))
	assert pct == Decimal("40.00")


def test_headroom_calc():
	h = headroom(Decimal("3000"), Decimal("5000"))
	assert h == Decimal("2000.00")


def test_over_soft_limit_true():
	assert is_over_soft_limit(Decimal("5000"), Decimal("4000")) is True


def test_over_hard_limit_false():
	assert is_over_hard_limit(Decimal("3000"), Decimal("5000")) is False


# ---------------------------------------------------------------------------
# Calculations: invoice aggregation
# ---------------------------------------------------------------------------

def test_aggregate_invoice_totals():
	items = [
		{"amount": "500"},
		{"amount": "300"},
		{"amount": "200"},
	]
	result = aggregate_invoice_totals(items, discount_pct=Decimal("10"), tax_rate_pct=Decimal("16"))
	assert result["subtotal"] == Decimal("1000.00")
	assert result["discount_amount"] == Decimal("100.00")
	assert result["tax_amount"] == Decimal("144.00")  # 16% of 900
	assert result["total_amount"] == Decimal("1044.00")


def test_late_payment_penalty_capped():
	penalty = calculate_late_payment_penalty(Decimal("10000"), 500)
	assert penalty <= Decimal("1000.00")  # 10% cap


# ---------------------------------------------------------------------------
# Calculations: revenue assurance
# ---------------------------------------------------------------------------

def test_leakage_rate():
	rate = leakage_rate_pct(Decimal("9500"), Decimal("10000"))
	assert rate == Decimal("5.00")


def test_collection_rate():
	rate = collection_rate_pct(Decimal("9200"), Decimal("10000"))
	assert rate == Decimal("92.00")


def test_arpu():
	avg = arpu(Decimal("100000"), 100)
	assert avg == Decimal("1000.00")


def test_dso():
	dso = days_sales_outstanding(Decimal("30000"), Decimal("360000"), 30)
	assert dso == Decimal("2.50")


# ---------------------------------------------------------------------------
# Calculations: convergent
# ---------------------------------------------------------------------------

def test_convergent_bill():
	fixed = [{"description": "Line rental", "amount": "500"}]
	mobile = [{"description": "Voice", "amount": "300"}]
	data = [{"description": "Data", "amount": "200"}]
	result = calculate_convergent_bill(fixed, mobile, data, shared_discount_pct=Decimal("5"))
	assert result["fixed_line_total"] == Decimal("500.00")
	assert result["mobile_total"] == Decimal("300.00")
	assert result["data_total"] == Decimal("200.00")
	assert result["combined_subtotal"] == Decimal("1000.00")
	assert result["shared_discount"] == Decimal("50.00")
	assert result["total_amount"] > Decimal("0")
