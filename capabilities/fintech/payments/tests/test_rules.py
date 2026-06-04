"""Tests for domain/rules.py — all pure functions, no async."""
from __future__ import annotations

import sys
from pathlib import Path
from decimal import Decimal

import pytest

PAYMENTS_DIR = Path(__file__).resolve().parents[1]
if str(PAYMENTS_DIR) not in sys.path:
	sys.path.insert(0, str(PAYMENTS_DIR))

from domain.rules import (
	RuleViolation,
	assert_tenant_context,
	assert_write_policy,
	assert_no_cross_tenant_access,
	assert_positive_amount,
	assert_amount_precision,
	assert_kyc_per_txn_limit,
	assert_kyc_daily_limit,
	assert_kyc_monthly_limit,
	assert_no_duplicate,
	assert_retry_window,
	assert_mpesa_amount,
	assert_mpesa_phone,
	assert_mpesa_float_sufficient,
	assert_mpesa_reference_length,
	assert_momo_amount,
	assert_card_token_not_pan,
	assert_card_cvv_not_stored,
	assert_3ds_result,
	assert_swift_bic,
	assert_iban,
	assert_swift_purpose_code,
	assert_refund_amount,
	assert_refund_not_duplicate,
	assert_reversal_window,
	assert_refundable_status,
	assert_fx_rate_freshness,
	assert_settlement_variance,
	assert_supported_currency,
	assert_aml_velocity,
	assert_batch_size,
	assert_batch_lists_aligned,
	assert_webhook_url,
	assert_mcc_code,
	calculate_ctr_obligation,
	calculate_mpesa_fee,
	calculate_excise_ke,
	calculate_vat_ke,
	calculate_total_charge,
	calculate_fx_amount,
	calculate_settlement_net,
	calculate_late_settlement_penalty,
)


# ---------------------------------------------------------------------------
# Tenant context
# ---------------------------------------------------------------------------

def test_assert_tenant_context_passes():
	assert_tenant_context({"tenant_id": "t1"})


def test_assert_tenant_context_fails():
	with pytest.raises(RuleViolation) as exc:
		assert_tenant_context({})
	assert exc.value.rule_name == "tenant_context_required"


def test_assert_write_policy_passes():
	assert_write_policy({"operation_type": "write", "policy_attached": True})


def test_assert_write_policy_fails():
	with pytest.raises(RuleViolation) as exc:
		assert_write_policy({"operation_type": "write", "policy_attached": False})
	assert exc.value.rule_name == "write_requires_policy"


def test_assert_write_policy_read_no_policy_ok():
	assert_write_policy({"operation_type": "read", "policy_attached": False})


def test_assert_no_cross_tenant_access_passes():
	assert_no_cross_tenant_access("t1", "t1")


def test_assert_no_cross_tenant_access_fails():
	with pytest.raises(RuleViolation) as exc:
		assert_no_cross_tenant_access("t1", "t2")
	assert exc.value.rule_name == "cross_tenant_access_denied"


# ---------------------------------------------------------------------------
# Amount guards
# ---------------------------------------------------------------------------

def test_assert_positive_amount_passes():
	assert_positive_amount(Decimal("0.01"))
	assert_positive_amount("100")
	assert_positive_amount(1)


def test_assert_positive_amount_fails_zero():
	with pytest.raises(RuleViolation) as exc:
		assert_positive_amount(Decimal("0"))
	assert exc.value.rule_name == "non_positive_amount"


def test_assert_positive_amount_fails_negative():
	with pytest.raises(RuleViolation):
		assert_positive_amount("-5")


# ---------------------------------------------------------------------------
# KYC limits
# ---------------------------------------------------------------------------

def test_kyc_per_txn_limit_passes():
	assert_kyc_per_txn_limit(Decimal("100000"), "basic")


def test_kyc_per_txn_limit_fails():
	with pytest.raises(RuleViolation) as exc:
		assert_kyc_per_txn_limit(Decimal("200000"), "basic")
	assert exc.value.rule_name == "kyc_per_txn_limit_exceeded"


def test_kyc_full_kyc_allows_large():
	assert_kyc_per_txn_limit(Decimal("900000"), "full_kyc")


def test_kyc_daily_limit_fails():
	with pytest.raises(RuleViolation) as exc:
		assert_kyc_daily_limit(Decimal("290000"), Decimal("20000"), "basic")
	assert "daily_limit" in exc.value.rule_name


def test_kyc_monthly_limit_fails():
	with pytest.raises(RuleViolation):
		assert_kyc_monthly_limit(Decimal("2990000"), Decimal("20000"), "basic")


# ---------------------------------------------------------------------------
# Duplicate / idempotency
# ---------------------------------------------------------------------------

def test_no_duplicate_passes():
	assert_no_duplicate("key-1", {"key-2", "key-3"})


def test_no_duplicate_fails():
	with pytest.raises(RuleViolation) as exc:
		assert_no_duplicate("key-1", {"key-1", "key-2"})
	assert exc.value.rule_name == "duplicate_payment_detected"


def test_no_duplicate_empty_key_always_passes():
	assert_no_duplicate("", {"key-1", ""})  # empty key is not tracked


def test_retry_window_passes():
	assert_retry_window(2, 3)


def test_retry_window_fails():
	with pytest.raises(RuleViolation) as exc:
		assert_retry_window(3, 3)
	assert exc.value.rule_name == "max_retries_exceeded"


# ---------------------------------------------------------------------------
# M-Pesa rules
# ---------------------------------------------------------------------------

def test_mpesa_amount_valid():
	assert_mpesa_amount(Decimal("500"))
	assert_mpesa_amount(Decimal("1"))
	assert_mpesa_amount(Decimal("300000"))


def test_mpesa_amount_below_min():
	with pytest.raises(RuleViolation) as exc:
		assert_mpesa_amount(Decimal("0"))
	assert "below_minimum" in exc.value.rule_name


def test_mpesa_amount_above_max():
	with pytest.raises(RuleViolation) as exc:
		assert_mpesa_amount(Decimal("300001"))
	assert "above_maximum" in exc.value.rule_name


def test_mpesa_phone_valid():
	assert_mpesa_phone("254712345678")
	assert_mpesa_phone("254112345678")


def test_mpesa_phone_invalid():
	with pytest.raises(RuleViolation) as exc:
		assert_mpesa_phone("0712345678")
	assert exc.value.rule_name == "mpesa_invalid_phone"


def test_mpesa_float_sufficient():
	assert_mpesa_float_sufficient(Decimal("10000"), Decimal("5000"))


def test_mpesa_float_insufficient():
	with pytest.raises(RuleViolation) as exc:
		assert_mpesa_float_sufficient(Decimal("1000"), Decimal("5000"))
	assert exc.value.rule_name == "mpesa_insufficient_float"


def test_mpesa_reference_valid():
	assert_mpesa_reference_length("REF001")
	assert_mpesa_reference_length("X" * 12)


def test_mpesa_reference_too_long():
	with pytest.raises(RuleViolation):
		assert_mpesa_reference_length("X" * 13)


def test_mpesa_reference_empty():
	with pytest.raises(RuleViolation):
		assert_mpesa_reference_length("")


# ---------------------------------------------------------------------------
# Card rules
# ---------------------------------------------------------------------------

def test_card_token_not_pan_passes():
	assert_card_token_not_pan("tok_visa_4242424242424242")
	assert_card_token_not_pan("vault://card/abc123")


def test_card_token_pan_rejected():
	with pytest.raises(RuleViolation) as exc:
		assert_card_token_not_pan("4242424242424242")
	assert exc.value.rule_name == "raw_pan_storage_forbidden"


def test_card_cvv_not_stored_passes():
	assert_card_cvv_not_stored(None)
	assert_card_cvv_not_stored("M")  # result code, not raw CVV


def test_card_cvv_storage_rejected():
	with pytest.raises(RuleViolation) as exc:
		assert_card_cvv_not_stored("123")
	assert exc.value.rule_name == "cvv_storage_forbidden"


def test_3ds_not_required_small_amount():
	assert_3ds_result(None, Decimal("5000"))


def test_3ds_required_large_amount():
	with pytest.raises(RuleViolation) as exc:
		assert_3ds_result(None, Decimal("15000"))
	assert "3ds_required" in exc.value.rule_name


def test_3ds_provided_large_amount_passes():
	assert_3ds_result("Y", Decimal("50000"))


# ---------------------------------------------------------------------------
# SWIFT / IBAN rules
# ---------------------------------------------------------------------------

def test_swift_bic_valid_8():
	assert_swift_bic("KCBLKENX")


def test_swift_bic_valid_11():
	assert_swift_bic("KCBLKENXXXX")


def test_swift_bic_invalid():
	with pytest.raises(RuleViolation) as exc:
		assert_swift_bic("INVALID")
	assert exc.value.rule_name == "invalid_swift_bic"


def test_iban_valid():
	assert_iban("GB29NWBK60161331926819")
	assert_iban("DE89370400440532013000")


def test_iban_invalid():
	with pytest.raises(RuleViolation):
		assert_iban("not-an-iban")


def test_swift_purpose_code_valid():
	assert_swift_purpose_code("OTH")
	assert_swift_purpose_code("SAL")


def test_swift_purpose_code_invalid():
	with pytest.raises(RuleViolation):
		assert_swift_purpose_code("TO")


# ---------------------------------------------------------------------------
# Refund / reversal rules
# ---------------------------------------------------------------------------

def test_refund_amount_passes():
	assert_refund_amount(Decimal("500"), Decimal("1000"))
	assert_refund_amount(Decimal("1000"), Decimal("1000"))


def test_refund_amount_exceeds():
	with pytest.raises(RuleViolation) as exc:
		assert_refund_amount(Decimal("1001"), Decimal("1000"))
	assert exc.value.rule_name == "refund_exceeds_original"


def test_refund_not_duplicate_passes():
	assert_refund_not_duplicate(Decimal("200"), Decimal("300"), Decimal("500"))


def test_refund_duplicate_fails():
	with pytest.raises(RuleViolation) as exc:
		assert_refund_not_duplicate(Decimal("400"), Decimal("200"), Decimal("500"))
	assert "cumulative_refund" in exc.value.rule_name


def test_reversal_window_expired():
	from datetime import datetime, timezone, timedelta
	old_ts = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
	with pytest.raises(RuleViolation) as exc:
		assert_reversal_window(old_ts, window_hours=24)
	assert "window_expired" in exc.value.rule_name


def test_reversal_window_valid():
	from datetime import datetime, timezone, timedelta
	recent_ts = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
	assert_reversal_window(recent_ts, window_hours=24)


def test_refundable_status_passes():
	assert_refundable_status("completed")
	assert_refundable_status("settled")


def test_refundable_status_fails():
	with pytest.raises(RuleViolation) as exc:
		assert_refundable_status("pending")
	assert "not_refundable" in exc.value.rule_name


# ---------------------------------------------------------------------------
# FX / settlement rules
# ---------------------------------------------------------------------------

def test_fx_rate_freshness_passes():
	assert_fx_rate_freshness(100.0, max_age_seconds=300)


def test_fx_rate_stale():
	with pytest.raises(RuleViolation) as exc:
		assert_fx_rate_freshness(400.0, max_age_seconds=300)
	assert exc.value.rule_name == "fx_rate_stale"


def test_settlement_variance_passes():
	assert_settlement_variance(Decimal("1"), Decimal("10000"), threshold_bps=10)


def test_settlement_variance_exceeds():
	with pytest.raises(RuleViolation) as exc:
		assert_settlement_variance(Decimal("100"), Decimal("10000"), threshold_bps=5)
	assert "variance_exceeded" in exc.value.rule_name


def test_supported_currency_passes():
	assert_supported_currency("KES")
	assert_supported_currency("USD")


def test_unsupported_currency_fails():
	with pytest.raises(RuleViolation) as exc:
		assert_supported_currency("XYZ")
	assert exc.value.rule_name == "unsupported_currency"


# ---------------------------------------------------------------------------
# AML velocity
# ---------------------------------------------------------------------------

def test_aml_velocity_low_risk():
	# assert_aml_velocity(transactions_last_24h, amount_last_24h, amount, currency)
	assert_aml_velocity(2, Decimal("50000"), Decimal("5000"), "KES")


def test_aml_velocity_structuring_detected():
	with pytest.raises(RuleViolation) as exc:
		assert_aml_velocity(6, Decimal("900000"), Decimal("100000"), "KES")
	assert exc.value.rule_name == "aml_velocity_threshold"


# ---------------------------------------------------------------------------
# Batch rules
# ---------------------------------------------------------------------------

def test_batch_size_passes():
	assert_batch_size(100, max_batch=10000)


def test_batch_size_fails():
	with pytest.raises(RuleViolation) as exc:
		assert_batch_size(10001, max_batch=10000)
	assert exc.value.rule_name == "batch_too_large"


def test_batch_lists_aligned_passes():
	assert_batch_lists_aligned(["a", "b"], [Decimal("1"), Decimal("2")], ["r1", "r2"])


def test_batch_lists_misaligned_fails():
	with pytest.raises(RuleViolation) as exc:
		assert_batch_lists_aligned(["a"], [Decimal("1"), Decimal("2")], ["r1"])
	assert "misaligned" in exc.value.rule_name


# ---------------------------------------------------------------------------
# Webhook / MCC rules
# ---------------------------------------------------------------------------

def test_webhook_url_https_passes():
	assert_webhook_url("https://example.com/webhook")


def test_webhook_url_http_fails():
	with pytest.raises(RuleViolation) as exc:
		assert_webhook_url("http://example.com/webhook")
	assert exc.value.rule_name == "webhook_url_must_use_https"


def test_mcc_valid():
	assert_mcc_code("7372")
	assert_mcc_code("5411")


def test_mcc_invalid():
	with pytest.raises(RuleViolation):
		assert_mcc_code("ABC1")


# ---------------------------------------------------------------------------
# CTR obligation
# ---------------------------------------------------------------------------

def test_ctr_cbk_triggered():
	result = calculate_ctr_obligation(Decimal("1500000"), "KES")
	assert result["requires_ctr"] is True
	assert result["regulator"] == "CBK"


def test_ctr_cbk_not_triggered():
	result = calculate_ctr_obligation(Decimal("500000"), "KES")
	assert result["requires_ctr"] is False


def test_ctr_cbn_nigeria():
	result = calculate_ctr_obligation(Decimal("6000000"), "NGN")
	assert result["requires_ctr"] is True
	assert result["regulator"] == "CBN"


# ---------------------------------------------------------------------------
# Calculation helpers
# ---------------------------------------------------------------------------

def test_calculate_mpesa_fee_free_tier():
	assert calculate_mpesa_fee(Decimal("50")) == Decimal("0")


def test_calculate_mpesa_fee_mid_tier():
	assert calculate_mpesa_fee(Decimal("5000")) == Decimal("57")


def test_calculate_mpesa_fee_max():
	assert calculate_mpesa_fee(Decimal("100000")) == Decimal("108")


def test_calculate_excise_ke():
	assert calculate_excise_ke(Decimal("100")) == Decimal("20.00")


def test_calculate_vat_ke():
	assert calculate_vat_ke(Decimal("100")) == Decimal("16.00")


def test_calculate_total_charge():
	assert calculate_total_charge(Decimal("57"), Decimal("11.40")) == Decimal("68.40")


def test_calculate_fx_amount_identity():
	result = calculate_fx_amount(Decimal("100"), Decimal("1"), spread_bps=0)
	assert result == Decimal("100.00")


def test_calculate_fx_amount_with_spread():
	result = calculate_fx_amount(Decimal("1000"), Decimal("130"), spread_bps=200, direction="buy")
	# effective rate = 130 * (1 - 0.01) = 128.70
	assert result < Decimal("130000")
	assert result > Decimal("125000")


def test_calculate_settlement_net():
	net = calculate_settlement_net(Decimal("100000"), processing_fee_rate_bps=200)
	assert net == Decimal("98000.00")


def test_calculate_late_penalty():
	penalty = calculate_late_settlement_penalty(Decimal("100000"), days_overdue=5)
	assert penalty == Decimal("500.00")
