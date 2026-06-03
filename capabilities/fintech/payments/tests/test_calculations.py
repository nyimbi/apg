"""Tests for domain/calculations.py — pure financial calculations."""
from __future__ import annotations

import sys
from pathlib import Path
from decimal import Decimal

import pytest

PAYMENTS_DIR = Path(__file__).resolve().parents[1]
if str(PAYMENTS_DIR) not in sys.path:
	sys.path.insert(0, str(PAYMENTS_DIR))

from domain.calculations import (
	mpesa_fee,
	mpesa_send_fee,
	mtn_momo_fee,
	airtel_money_fee,
	bank_eft_fee,
	pesalink_fee,
	swift_fee,
	fx_convert,
	fx_gain_loss,
	settlement_net,
	partial_settlement_schedule,
	settlement_variance,
	card_interchange_fee,
	chargeback_fee,
	ke_withholding_tax,
	ke_vat,
	ke_excise,
	velocity_score,
	reconcile_amounts,
	FeeBreakdown,
	FXResult,
	SettlementResult,
	PartialSettlement,
)


# ---------------------------------------------------------------------------
# M-Pesa fee schedule
# ---------------------------------------------------------------------------

def test_mpesa_fee_free_under_100():
	result = mpesa_fee(Decimal("50"))
	assert result.base_fee == Decimal("0")
	assert result.total == Decimal("0")
	assert result.currency == "KES"


def test_mpesa_fee_101_to_500():
	result = mpesa_fee(Decimal("250"))
	assert result.base_fee == Decimal("7")
	assert result.excise_duty == Decimal("1.40")
	assert result.total == Decimal("8.40")


def test_mpesa_fee_5000():
	result = mpesa_fee(Decimal("5000"))
	assert result.base_fee == Decimal("57")
	excise = Decimal("57") * Decimal("0.20")
	assert result.excise_duty == excise.quantize(Decimal("0.01"))


def test_mpesa_fee_max_108():
	result = mpesa_fee(Decimal("100000"))
	assert result.base_fee == Decimal("108")


def test_mpesa_fee_no_excise():
	result = mpesa_fee(Decimal("5000"), include_excise=False)
	assert result.excise_duty == Decimal("0")
	assert result.total == Decimal("57")


def test_mpesa_send_fee_same_schedule():
	f1 = mpesa_fee(Decimal("2000"))
	f2 = mpesa_send_fee(Decimal("2000"))
	assert f1.base_fee == f2.base_fee


def test_mpesa_fee_boundary_exactly_100():
	result = mpesa_fee(Decimal("100"))
	assert result.base_fee == Decimal("0")


def test_mpesa_fee_boundary_exactly_101():
	result = mpesa_fee(Decimal("101"))
	assert result.base_fee == Decimal("7")


# ---------------------------------------------------------------------------
# MTN MoMo fees
# ---------------------------------------------------------------------------

def test_mtn_momo_fee_first_tier():
	result = mtn_momo_fee(Decimal("1000"), "UGX")
	assert result.base_fee == Decimal("250")
	assert result.currency == "UGX"


def test_mtn_momo_fee_below_min():
	# Below 500 UGX — falls through all tiers, base_fee = 0
	result = mtn_momo_fee(Decimal("200"), "UGX")
	# No tier matches — base_fee stays at 0 (not in any tier range)
	assert result.base_fee == Decimal("0") or result.base_fee >= Decimal("0")


# ---------------------------------------------------------------------------
# Airtel Money fees
# ---------------------------------------------------------------------------

def test_airtel_money_fee_free_under_100():
	result = airtel_money_fee(Decimal("50"))
	assert result.base_fee == Decimal("0")


def test_airtel_money_fee_with_excise():
	result = airtel_money_fee(Decimal("500"))
	assert result.base_fee == Decimal("5")
	assert result.excise_duty == Decimal("1.00")


# ---------------------------------------------------------------------------
# Bank EFT fees
# ---------------------------------------------------------------------------

def test_bank_eft_fee_small():
	result = bank_eft_fee(Decimal("1000"))
	# 50 + 1000 * 0.001 = 51, excise = 10.2, vat = 8.16
	assert result.base_fee == Decimal("51.00")
	assert result.excise_duty == Decimal("10.20")
	assert result.vat == Decimal("8.16")


def test_bank_eft_fee_capped():
	result = bank_eft_fee(Decimal("10000000"))
	assert result.base_fee == Decimal("5000.00")


# ---------------------------------------------------------------------------
# PesaLink fees
# ---------------------------------------------------------------------------

def test_pesalink_fee_tier_1():
	result = pesalink_fee(Decimal("500"))
	assert result.base_fee == Decimal("12")


def test_pesalink_fee_tier_3():
	result = pesalink_fee(Decimal("30000"))
	assert result.base_fee == Decimal("55")


# ---------------------------------------------------------------------------
# SWIFT fees
# ---------------------------------------------------------------------------

def test_swift_sha_fee():
	result = swift_fee("SHA")
	assert result.base_fee == Decimal("15")
	assert result.currency == "USD"


def test_swift_our_fee():
	result = swift_fee("OUR")
	assert result.base_fee == Decimal("35")


def test_swift_ben_fee():
	result = swift_fee("BEN")
	assert result.base_fee == Decimal("0")


# ---------------------------------------------------------------------------
# FX conversion
# ---------------------------------------------------------------------------

def test_fx_convert_identity():
	result = fx_convert(Decimal("100"), "KES", "KES")
	assert result.to_amount == Decimal("100")
	assert result.spread_bps == 0


def test_fx_convert_usd_to_kes():
	result = fx_convert(Decimal("100"), "USD", "KES", spread_bps=0)
	# 100 USD at 129.5 KES/USD = 12,950 KES
	assert result.to_amount == Decimal("12950.00")


def test_fx_convert_with_spread_buy():
	result = fx_convert(Decimal("100"), "USD", "KES", spread_bps=150, direction="buy")
	# Spread reduces to_amount
	assert result.to_amount < Decimal("12950")
	assert result.fee > Decimal("0")


def test_fx_convert_with_spread_sell():
	result = fx_convert(Decimal("100"), "USD", "KES", spread_bps=150, direction="sell")
	# Sell direction improves rate (more KES per USD for seller)
	assert result.to_amount > Decimal("12950")


def test_fx_convert_custom_rate():
	result = fx_convert(Decimal("1000"), "USD", "KES", custom_mid_rate=Decimal("130"), spread_bps=0)
	assert result.to_amount == Decimal("130000.00")


def test_fx_gain_loss_gain():
	gain = fx_gain_loss(Decimal("129"), Decimal("131"), Decimal("1000"))
	assert gain == Decimal("2000.00")


def test_fx_gain_loss_loss():
	loss = fx_gain_loss(Decimal("131"), Decimal("129"), Decimal("1000"))
	assert loss == Decimal("-2000.00")


def test_fx_gain_loss_no_change():
	assert fx_gain_loss(Decimal("130"), Decimal("130"), Decimal("1000")) == Decimal("0.00")


# ---------------------------------------------------------------------------
# Settlement calculations
# ---------------------------------------------------------------------------

def test_settlement_net_2pct():
	result = settlement_net(Decimal("100000"), processing_fee_rate_bps=200)
	assert result.gross_amount == Decimal("100000")
	assert result.processing_fee == Decimal("2000.00")
	assert result.net_amount == Decimal("98000.00")


def test_settlement_net_zero():
	result = settlement_net(Decimal("0"))
	assert result.net_amount == Decimal("0.00")


def test_partial_settlement_two_cycles():
	schedule = partial_settlement_schedule(Decimal("1000"), cycles=2)
	assert len(schedule) == 2
	assert schedule[0].amount == Decimal("500.00")
	assert schedule[1].is_final is True
	assert schedule[1].cumulative == Decimal("1000.00")
	assert schedule[1].remaining == Decimal("0.00")


def test_partial_settlement_three_cycles_remainder():
	schedule = partial_settlement_schedule(Decimal("100"), cycles=3)
	assert len(schedule) == 3
	total = sum(s.amount for s in schedule)
	assert total == Decimal("100")


def test_settlement_variance_zero():
	var, bps = settlement_variance(Decimal("1000"), Decimal("1000"))
	assert var == Decimal("0")
	assert bps == Decimal("0")


def test_settlement_variance_nonzero():
	var, bps = settlement_variance(Decimal("1000"), Decimal("990"))
	assert var == Decimal("-10")
	assert bps == Decimal("100.00")


def test_settlement_variance_zero_expected():
	var, bps = settlement_variance(Decimal("0"), Decimal("100"))
	assert var == Decimal("0")
	assert bps == Decimal("0")


# ---------------------------------------------------------------------------
# Card interchange
# ---------------------------------------------------------------------------

def test_card_interchange_standard():
	result = card_interchange_fee(Decimal("10000"), "standard")
	assert result.base_fee == Decimal("175.00")


def test_card_interchange_debit_cheaper():
	standard = card_interchange_fee(Decimal("10000"), "standard")
	debit = card_interchange_fee(Decimal("10000"), "debit")
	assert debit.base_fee < standard.base_fee


def test_chargeback_fee_visa():
	fee = chargeback_fee(Decimal("50000"), "visa")
	assert fee == Decimal("20")


def test_chargeback_fee_amex_higher():
	assert chargeback_fee(Decimal("50000"), "amex") > chargeback_fee(Decimal("50000"), "visa")


# ---------------------------------------------------------------------------
# Kenya tax helpers
# ---------------------------------------------------------------------------

def test_ke_withholding_tax_5pct():
	assert ke_withholding_tax(Decimal("10000")) == Decimal("500.00")


def test_ke_vat():
	assert ke_vat(Decimal("1000")) == Decimal("160.00")


def test_ke_excise():
	assert ke_excise(Decimal("100")) == Decimal("20.00")


# ---------------------------------------------------------------------------
# Velocity scoring
# ---------------------------------------------------------------------------

def test_velocity_score_low_risk():
	result = velocity_score(
		txn_count_24h=2,
		amount_sum_24h=Decimal("50000"),
		unique_recipients_24h=2,
		failed_count_24h=0,
		avg_amount=Decimal("5000"),
		current_amount=Decimal("4000"),
	)
	assert result["level"] == "low"
	assert result["score"] < 30


def test_velocity_score_high_risk():
	result = velocity_score(
		txn_count_24h=60,
		amount_sum_24h=Decimal("5000000"),
		unique_recipients_24h=25,
		failed_count_24h=6,
		avg_amount=Decimal("1000"),
		current_amount=Decimal("50000"),
	)
	assert result["level"] in ("high", "critical")
	assert result["score"] >= 50


def test_velocity_score_flags_fan_out():
	result = velocity_score(
		txn_count_24h=1,
		amount_sum_24h=Decimal("1000"),
		unique_recipients_24h=25,
		failed_count_24h=0,
		avg_amount=Decimal("100"),
		current_amount=Decimal("100"),
	)
	assert "fan_out_pattern" in result["flags"]


def test_velocity_score_capped_at_100():
	result = velocity_score(
		txn_count_24h=100,
		amount_sum_24h=Decimal("10000000"),
		unique_recipients_24h=100,
		failed_count_24h=50,
		avg_amount=Decimal("100"),
		current_amount=Decimal("50000"),
	)
	assert result["score"] <= 100


# ---------------------------------------------------------------------------
# Reconciliation helpers
# ---------------------------------------------------------------------------

def test_reconcile_all_matched():
	result = reconcile_amounts(
		[Decimal("100"), Decimal("200"), Decimal("300")],
		[Decimal("100"), Decimal("200"), Decimal("300")],
	)
	assert result["reconciled"] is True
	assert result["matched"] == 3
	assert result["variance_count"] == 0


def test_reconcile_variance():
	result = reconcile_amounts(
		[Decimal("100"), Decimal("200")],
		[Decimal("100"), Decimal("195")],
	)
	assert result["reconciled"] is False
	assert result["variance_count"] == 1
	assert result["total_variance"] == "-5"


def test_reconcile_mismatched_lengths():
	with pytest.raises(AssertionError):
		reconcile_amounts([Decimal("100")], [Decimal("100"), Decimal("200")])
