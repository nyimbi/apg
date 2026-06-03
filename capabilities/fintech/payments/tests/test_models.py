"""Tests for Pydantic v2 models — no async needed."""
from __future__ import annotations

import sys
from pathlib import Path
from decimal import Decimal

import pytest

PAYMENTS_DIR = Path(__file__).resolve().parents[1]
if str(PAYMENTS_DIR) not in sys.path:
	sys.path.insert(0, str(PAYMENTS_DIR))

from models import (
	PaymentStatus, PaymentMethod, CurrencyCode, KYCTier, RiskLevel,
	PaymentOrder, PaymentTransaction, PaymentLimit, MobileMoneyPayment,
	CardPayment, BankTransfer, SWIFTPayment, PaymentRefund, PaymentReversal,
	FXConversion, FXRateType, SettlementBatch, MerchantAccount, VirtualAccount,
	PaymentReceipt, ChargebackCase, PaymentDispute, DisputeStatus,
	BulkPaymentBatch, SplitPayment, PaymentLeg, ReconciliationRecord,
	PaymentFee, WebhookEvent, KYC_LIMITS, MPESA_FEE_TIERS,
	uuid7str, utcnow, money,
)


# ---------------------------------------------------------------------------
# uuid7str / utcnow / money helpers
# ---------------------------------------------------------------------------

def test_uuid7str_produces_unique_strings():
	ids = {uuid7str() for _ in range(100)}
	assert len(ids) == 100
	for i in ids:
		assert len(i) == 36


def test_money_stable():
	assert money(Decimal("1.10")) == "1.10"
	assert money("123.456") == "123.456"
	assert money(0) == "0"


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

def test_payment_status_values():
	assert PaymentStatus.pending.value == "pending"
	assert PaymentStatus.completed.value == "completed"
	assert PaymentStatus.reversed.value == "reversed"


def test_payment_method_values():
	assert PaymentMethod.mpesa_stk.value == "mpesa_stk"
	assert PaymentMethod.swift.value == "swift"
	assert PaymentMethod.card_visa.value == "card_visa"


def test_currency_codes_present():
	assert "KES" in [c.value for c in CurrencyCode]
	assert "NGN" in [c.value for c in CurrencyCode]
	assert "USD" in [c.value for c in CurrencyCode]


# ---------------------------------------------------------------------------
# PaymentLimit — model validators
# ---------------------------------------------------------------------------

def test_payment_limit_consistency():
	lim = PaymentLimit(
		kyc_tier=KYCTier.basic,
		daily_limit=Decimal("300000"),
		monthly_limit=Decimal("3000000"),
		per_txn_limit=Decimal("150000"),
	)
	assert lim.per_txn_limit <= lim.daily_limit
	assert lim.daily_limit <= lim.monthly_limit


def test_payment_limit_invalid_per_txn_exceeds_daily():
	with pytest.raises(Exception):
		PaymentLimit(
			kyc_tier=KYCTier.basic,
			daily_limit=Decimal("100"),
			monthly_limit=Decimal("10000"),
			per_txn_limit=Decimal("200"),  # exceeds daily
		)


# ---------------------------------------------------------------------------
# KYC_LIMITS constants
# ---------------------------------------------------------------------------

def test_kyc_limits_all_tiers_present():
	assert KYCTier.basic in KYC_LIMITS
	assert KYCTier.standard in KYC_LIMITS
	assert KYCTier.full_kyc in KYC_LIMITS
	assert KYCTier.enhanced in KYC_LIMITS


def test_kyc_limits_enhanced_unlimited():
	lim = KYC_LIMITS[KYCTier.enhanced]
	# Enhanced tier sentinel — effectively unlimited (> 10M daily)
	assert lim.daily_limit > Decimal("10000000")


# ---------------------------------------------------------------------------
# ReconciliationRecord — auto-variance
# ---------------------------------------------------------------------------

def test_reconciliation_variance_computed():
	rec = ReconciliationRecord(
		id=uuid7str(),
		tenant_id="t1",
		settlement_id="s1",
		transaction_id="tx1",
		expected_amount=Decimal("1000"),
		actual_amount=Decimal("995"),
	)
	assert rec.variance == Decimal("-5")
	assert rec.status == "variance"


def test_reconciliation_matched():
	rec = ReconciliationRecord(
		id=uuid7str(),
		tenant_id="t1",
		settlement_id="s1",
		transaction_id="tx1",
		expected_amount=Decimal("1000"),
		actual_amount=Decimal("1000"),
	)
	assert rec.variance == Decimal("0")
	assert rec.status == "matched"


# ---------------------------------------------------------------------------
# PaymentFee — total sum
# ---------------------------------------------------------------------------

def test_payment_fee_total():
	f = PaymentFee(
		id=uuid7str(),
		method=PaymentMethod.mpesa_stk,
		amount=Decimal("5000"),
		currency=CurrencyCode.KES,
		fee_amount=Decimal("57"),
		excise_tax=Decimal("11.40"),
	)
	assert f.total_charge == Decimal("68.40")


# ---------------------------------------------------------------------------
# BulkPaymentBatch — list alignment
# ---------------------------------------------------------------------------

def test_bulk_batch_lists_aligned():
	b = BulkPaymentBatch(
		id=uuid7str(),
		tenant_id="t1",
		payment_date="2025-12-01",
		method=PaymentMethod.mpesa_b2c,
		currency=CurrencyCode.KES,
		recipients=["2547001", "2547002"],
		amounts=[Decimal("100"), Decimal("200")],
		references=["ref-1", "ref-2"],
	)
	assert b.total_amount == Decimal("300")


def test_bulk_batch_misaligned_raises():
	with pytest.raises(Exception):
		BulkPaymentBatch(
			id=uuid7str(),
			tenant_id="t1",
			payment_date="2025-12-01",
			method=PaymentMethod.mpesa_b2c,
			currency=CurrencyCode.KES,
			recipients=["2547001"],
			amounts=[Decimal("100"), Decimal("200")],
			references=["ref-1"],
		)


# ---------------------------------------------------------------------------
# SplitPayment — leg sum check
# ---------------------------------------------------------------------------

def test_split_payment_leg_sum_matches():
	legs = [
		PaymentLeg(transaction_id="tx1", merchant_id="m1", amount=Decimal("600"), currency=CurrencyCode.KES),
		PaymentLeg(transaction_id="tx1", merchant_id="m2", amount=Decimal("400"), currency=CurrencyCode.KES),
	]
	sp = SplitPayment(
		id=uuid7str(),
		transaction_id="tx1",
		tenant_id="t1",
		legs=legs,
		total_amount=Decimal("1000"),
		currency=CurrencyCode.KES,
	)
	assert len(sp.legs) == 2


def test_split_payment_leg_sum_mismatch_raises():
	legs = [
		PaymentLeg(transaction_id="tx1", merchant_id="m1", amount=Decimal("600"), currency=CurrencyCode.KES),
	]
	from pydantic import ValidationError
	with pytest.raises((AssertionError, ValidationError)):
		SplitPayment(
			id=uuid7str(),
			transaction_id="tx1",
			tenant_id="t1",
			legs=legs,
			total_amount=Decimal("1000"),
			currency=CurrencyCode.KES,
		)


# ---------------------------------------------------------------------------
# VirtualAccount — available balance
# ---------------------------------------------------------------------------

def test_virtual_account_available():
	va = VirtualAccount(
		id=uuid7str(),
		tenant_id="t1",
		owner_id="owner1",
		currency=CurrencyCode.KES,
		balance=Decimal("10000"),
		reserved=Decimal("3000"),
	)
	assert va.available == Decimal("7000")


# ---------------------------------------------------------------------------
# MPESA_FEE_TIERS — spot checks
# ---------------------------------------------------------------------------

def test_mpesa_fee_tiers_free_under_100():
	for lo, hi, fee in MPESA_FEE_TIERS:
		if lo == Decimal("1") and hi == Decimal("100"):
			assert fee == Decimal("0")


def test_mpesa_fee_tiers_max_108():
	for _, _, fee in MPESA_FEE_TIERS:
		assert fee <= Decimal("108")
