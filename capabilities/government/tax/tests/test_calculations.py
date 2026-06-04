"""Unit tests for tax domain calculations.

Pure function tests — no I/O, no service layer.
"""
from __future__ import annotations

import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]
if str(PKG) not in sys.path:
	sys.path.insert(0, str(PKG))

from domain.calculations import (
	calculate_income_tax,
	calculate_vat_payable,
	calculate_vat_on_amount,
	extract_vat_from_inclusive,
	calculate_withholding_tax,
	calculate_turnover_tax,
	calculate_capital_gain,
	calculate_capital_gains_tax,
	calculate_late_filing_penalty,
	calculate_late_payment_penalty,
	calculate_understatement_penalty,
	calculate_interest,
	calculate_compound_interest,
	calculate_total_debt,
	calculate_debt_balance,
	calculate_refund_interest,
	calculate_compliance_risk_score,
	calculate_return_due_date,
	calculate_certificate_expiry,
)


# ---------------------------------------------------------------------------
# Income Tax
# ---------------------------------------------------------------------------

def test_income_tax_zero():
	assert calculate_income_tax(Decimal("0")) == Decimal("0")


def test_income_tax_first_band():
	# Income within first band: 288000 × 10% = 28800; personal relief 28800 → 0
	tax = calculate_income_tax(Decimal("288000"))
	assert tax == Decimal("0")


def test_income_tax_second_band():
	# bands: [0-288000 @ 10%, 288001-388000 @ 25%]
	# 288000 * 10% = 28800; (388000 - 288001) * 25% = 99999 * 0.25 = 24999.75
	# total = 53799.75 - personal_relief 28800 = 24999.75
	tax = calculate_income_tax(Decimal("388000"))
	assert tax == Decimal("24999.75")


def test_income_tax_high_income():
	tax = calculate_income_tax(Decimal("10000000"))
	assert tax > Decimal("0")
	assert tax < Decimal("10000000")


def test_income_tax_negative_raises():
	with pytest.raises(AssertionError):
		calculate_income_tax(Decimal("-1"))


def test_income_tax_custom_bands():
	bands = [(Decimal("0"), Decimal("100000"), Decimal("0.10"))]
	tax = calculate_income_tax(Decimal("100000"), bands=bands, personal_relief=Decimal("0"))
	assert tax == Decimal("10000.00")


def test_income_tax_personal_relief_larger_than_tax():
	# very low income, relief wipes out all tax
	tax = calculate_income_tax(Decimal("100000"), personal_relief=Decimal("50000"))
	# 100000 * 10% = 10000 - 50000 = max(0, ...) = 0
	assert tax == Decimal("0")


# ---------------------------------------------------------------------------
# VAT
# ---------------------------------------------------------------------------

def test_vat_payable_positive():
	result = calculate_vat_payable(Decimal("16000"), Decimal("8000"))
	assert result == Decimal("8000.00")


def test_vat_payable_refund():
	result = calculate_vat_payable(Decimal("5000"), Decimal("10000"))
	assert result == Decimal("-5000.00")


def test_vat_payable_zero():
	assert calculate_vat_payable(Decimal("0"), Decimal("0")) == Decimal("0.00")


def test_vat_on_amount():
	vat, gross = calculate_vat_on_amount(Decimal("100000"))
	assert vat == Decimal("16000.00")
	assert gross == Decimal("116000.00")


def test_vat_on_amount_custom_rate():
	vat, gross = calculate_vat_on_amount(Decimal("100000"), rate=Decimal("0.08"))
	assert vat == Decimal("8000.00")
	assert gross == Decimal("108000.00")


def test_extract_vat_from_inclusive():
	vat, net = extract_vat_from_inclusive(Decimal("116000"))
	assert net == Decimal("100000.00")
	assert vat == Decimal("16000.00")


def test_extract_vat_roundtrip():
	vat_added, gross = calculate_vat_on_amount(Decimal("500000"))
	vat_extracted, net_back = extract_vat_from_inclusive(gross)
	# within 1 cent due to rounding
	assert abs(net_back - Decimal("500000")) <= Decimal("1.00")


def test_vat_negative_raises():
	with pytest.raises(AssertionError):
		calculate_vat_payable(Decimal("-1"), Decimal("0"))


# ---------------------------------------------------------------------------
# Withholding Tax
# ---------------------------------------------------------------------------

def test_wht_standard():
	assert calculate_withholding_tax(Decimal("100000"), Decimal("0.05")) == Decimal("5000.00")


def test_wht_zero_rate():
	assert calculate_withholding_tax(Decimal("100000"), Decimal("0")) == Decimal("0.00")


def test_wht_full_rate():
	assert calculate_withholding_tax(Decimal("100000"), Decimal("1")) == Decimal("100000.00")


def test_wht_invalid_rate():
	with pytest.raises(AssertionError):
		calculate_withholding_tax(Decimal("100000"), Decimal("1.5"))


def test_wht_negative_gross():
	with pytest.raises(AssertionError):
		calculate_withholding_tax(Decimal("-1"), Decimal("0.05"))


# ---------------------------------------------------------------------------
# Turnover Tax
# ---------------------------------------------------------------------------

def test_turnover_tax_standard():
	assert calculate_turnover_tax(Decimal("1000000")) == Decimal("15000.00")


def test_turnover_tax_custom_rate():
	assert calculate_turnover_tax(Decimal("1000000"), rate=Decimal("0.02")) == Decimal("20000.00")


def test_turnover_tax_zero():
	assert calculate_turnover_tax(Decimal("0")) == Decimal("0.00")


# ---------------------------------------------------------------------------
# Capital Gains Tax
# ---------------------------------------------------------------------------

def test_capital_gain_positive():
	gain = calculate_capital_gain(Decimal("5000000"), Decimal("2000000"))
	assert gain == Decimal("3000000.00")


def test_capital_gain_no_gain():
	gain = calculate_capital_gain(Decimal("1000000"), Decimal("2000000"))
	assert gain == Decimal("0.00")


def test_capital_gains_tax():
	tax = calculate_capital_gains_tax(Decimal("3000000"))
	assert tax == Decimal("450000.00")


def test_capital_gains_tax_with_improvements():
	gain = calculate_capital_gain(
		Decimal("5000000"), Decimal("2000000"),
		improvements=Decimal("500000"), inflation_adjustment=Decimal("200000"),
	)
	assert gain == Decimal("2300000.00")


# ---------------------------------------------------------------------------
# Late Filing Penalty
# ---------------------------------------------------------------------------

def test_late_filing_penalty_above_minimum():
	pen = calculate_late_filing_penalty(Decimal("200000"))
	assert pen == Decimal("10000.00")  # 5% of 200000


def test_late_filing_penalty_minimum_enforced():
	pen = calculate_late_filing_penalty(Decimal("10000"), minimum_penalty=Decimal("2000"))
	# 5% of 10000 = 500, minimum 2000
	assert pen == Decimal("2000.00")


def test_late_filing_penalty_capped():
	pen = calculate_late_filing_penalty(
		Decimal("10000000"), maximum_penalty=Decimal("50000")
	)
	assert pen == Decimal("50000.00")


def test_late_filing_penalty_zero_tax():
	pen = calculate_late_filing_penalty(Decimal("0"), minimum_penalty=Decimal("2000"))
	assert pen == Decimal("2000.00")


# ---------------------------------------------------------------------------
# Understatement Penalty
# ---------------------------------------------------------------------------

def test_understatement_no_reasonable_care():
	pen = calculate_understatement_penalty(Decimal("100000"), "no_reasonable_care")
	assert pen == Decimal("25000.00")


def test_understatement_careless():
	pen = calculate_understatement_penalty(Decimal("100000"), "careless")
	assert pen == Decimal("50000.00")


def test_understatement_deliberate():
	pen = calculate_understatement_penalty(Decimal("100000"), "deliberate")
	assert pen == Decimal("75000.00")


def test_understatement_fraud():
	pen = calculate_understatement_penalty(Decimal("100000"), "fraud")
	assert pen == Decimal("100000.00")


def test_understatement_unknown_tier_defaults_careless():
	pen = calculate_understatement_penalty(Decimal("100000"), "unknown_tier")
	assert pen == Decimal("50000.00")


# ---------------------------------------------------------------------------
# Interest
# ---------------------------------------------------------------------------

def test_interest_simple():
	interest = calculate_interest(
		Decimal("100000"), Decimal("0.12"),
		date(2025, 1, 1), date(2025, 4, 1),
	)
	# 90 days / 365 × 12% × 100000
	expected = Decimal("100000") * Decimal("0.12") * Decimal("90") / Decimal("365")
	assert abs(interest - expected.quantize(Decimal("0.01"))) <= Decimal("0.01")


def test_interest_zero_days():
	interest = calculate_interest(
		Decimal("100000"), Decimal("0.12"),
		date(2025, 1, 1), date(2025, 1, 1),
	)
	assert interest == Decimal("0")


def test_interest_negative_days():
	interest = calculate_interest(
		Decimal("100000"), Decimal("0.12"),
		date(2025, 4, 1), date(2025, 1, 1),
	)
	assert interest == Decimal("0")


def test_interest_invalid_rate():
	with pytest.raises(AssertionError):
		calculate_interest(Decimal("100000"), Decimal("1.5"), date(2025, 1, 1), date(2025, 12, 31))


def test_compound_interest():
	ci = calculate_compound_interest(Decimal("100000"), Decimal("0.01"), 12)
	# 100000 × (1.01)^12 - 100000 ≈ 12682.50
	assert ci > Decimal("12000")
	assert ci < Decimal("13000")


def test_compound_interest_zero_months():
	assert calculate_compound_interest(Decimal("100000"), Decimal("0.01"), 0) == Decimal("0.00")


# ---------------------------------------------------------------------------
# Debt
# ---------------------------------------------------------------------------

def test_total_debt():
	total = calculate_total_debt(Decimal("100000"), Decimal("5000"), Decimal("1000"))
	assert total == Decimal("106000.00")


def test_debt_balance():
	balance = calculate_debt_balance(Decimal("106000"), Decimal("50000"))
	assert balance == Decimal("56000.00")


def test_debt_balance_overpayment():
	# overpayment clamps to 0
	balance = calculate_debt_balance(Decimal("50000"), Decimal("60000"))
	assert balance == Decimal("0.00")


# ---------------------------------------------------------------------------
# Compliance Risk Score
# ---------------------------------------------------------------------------

def test_risk_score_perfect_compliance():
	score, cat = calculate_compliance_risk_score(
		years_registered=5,
		returns_filed=12,
		returns_due=12,
		payments_on_time=12,
		payments_due=12,
		open_audits=0,
		prior_fraud_flags=0,
		days_avg_late_filing=0.0,
		debt_to_turnover_ratio=0.0,
	)
	assert score == Decimal("0.00")
	assert cat == "low"


def test_risk_score_high_risk():
	score, cat = calculate_compliance_risk_score(
		years_registered=1,
		returns_filed=0,
		returns_due=12,
		payments_on_time=0,
		payments_due=12,
		open_audits=3,
		prior_fraud_flags=2,
		days_avg_late_filing=90.0,
		debt_to_turnover_ratio=0.5,
	)
	assert score > Decimal("50")
	assert cat in ("high", "critical")


def test_risk_score_capped_at_100():
	score, _ = calculate_compliance_risk_score(
		years_registered=0,
		returns_filed=0,
		returns_due=100,
		payments_on_time=0,
		payments_due=100,
		open_audits=10,
		prior_fraud_flags=10,
		days_avg_late_filing=999.0,
		debt_to_turnover_ratio=10.0,
	)
	assert score <= Decimal("100")


# ---------------------------------------------------------------------------
# Return Due Date
# ---------------------------------------------------------------------------

def test_return_due_date_monthly():
	period_end = date(2025, 1, 31)
	due = calculate_return_due_date(period_end, "monthly", due_day=20)
	assert due == date(2025, 2, 20)


def test_return_due_date_annual():
	period_end = date(2025, 12, 31)
	due = calculate_return_due_date(period_end, "annually", due_day=90)
	assert due == date(2026, 3, 31)


def test_return_due_date_december_monthly():
	period_end = date(2025, 12, 31)
	due = calculate_return_due_date(period_end, "monthly", due_day=20)
	assert due == date(2026, 1, 20)


# ---------------------------------------------------------------------------
# Certificate Expiry
# ---------------------------------------------------------------------------

def test_certificate_expiry_6months():
	expiry = calculate_certificate_expiry(date(2025, 1, 15), 6)
	assert expiry == date(2025, 7, 15)


def test_certificate_expiry_year_boundary():
	expiry = calculate_certificate_expiry(date(2025, 10, 31), 6)
	assert expiry == date(2026, 4, 30)


def test_certificate_expiry_12months():
	expiry = calculate_certificate_expiry(date(2025, 3, 1), 12)
	assert expiry == date(2026, 3, 1)


# ---------------------------------------------------------------------------
# Refund Interest
# ---------------------------------------------------------------------------

def test_refund_interest_positive():
	interest = calculate_refund_interest(
		Decimal("100000"), Decimal("0.12"),
		date(2025, 1, 1), date(2025, 7, 1),
	)
	assert interest > Decimal("0")
