"""Unit tests for domain/calculations.py — pure financial math."""
from __future__ import annotations

import pytest
from decimal import Decimal

from capabilities.fin.glr.general_ledger.domain.calculations import (
	net_balance,
	debit_credit_split,
	validate_balance,
	convert_currency,
	functional_amount,
	revaluation_gain_loss,
	hyperinflationary_restatement,
	straight_line_accrual,
	prepaid_remaining,
	straight_line_depreciation,
	declining_balance_depreciation,
	calculate_ratios,
	FinancialRatios,
	variance,
	variance_pct,
	variance_indicator,
	minority_interest_amount,
	goodwill_on_acquisition,
	withholding_tax,
	vat_exclusive_to_inclusive,
	vat_inclusive_to_exclusive,
)


# ---------------------------------------------------------------------------
# Core double-entry
# ---------------------------------------------------------------------------

def test_net_balance_debit_normal():
	result = net_balance(Decimal("1000"), Decimal("500"), Decimal("200"), "debit")
	assert result == Decimal("1300")


def test_net_balance_credit_normal():
	result = net_balance(Decimal("1000"), Decimal("200"), Decimal("500"), "credit")
	assert result == Decimal("1300")


def test_net_balance_zero_opening():
	result = net_balance(Decimal("0"), Decimal("1000"), Decimal("1000"), "debit")
	assert result == Decimal("0")


def test_debit_credit_split_debit():
	d, c = debit_credit_split(Decimal("500"), is_debit=True)
	assert d == Decimal("500")
	assert c == Decimal("0")


def test_debit_credit_split_credit():
	d, c = debit_credit_split(Decimal("500"), is_debit=False)
	assert d == Decimal("0")
	assert c == Decimal("500")


def test_debit_credit_split_negative_raises():
	with pytest.raises(ValueError):
		debit_credit_split(Decimal("-1"), is_debit=True)


def test_validate_balance_ok():
	assert validate_balance(Decimal("1000"), Decimal("1000")) is True


def test_validate_balance_unequal():
	assert validate_balance(Decimal("1000"), Decimal("900")) is False


def test_validate_balance_zero():
	assert validate_balance(Decimal("0"), Decimal("0")) is False


# ---------------------------------------------------------------------------
# Multi-currency
# ---------------------------------------------------------------------------

def test_convert_currency():
	result = convert_currency(Decimal("1300"), Decimal("130"))
	assert result == Decimal("10.00")


def test_convert_currency_zero_rate_raises():
	with pytest.raises(ValueError):
		convert_currency(Decimal("100"), Decimal("0"))


def test_functional_amount():
	result = functional_amount(Decimal("100"), Decimal("130"))
	assert result == Decimal("13000.00")


def test_functional_amount_zero_rate_raises():
	with pytest.raises(ValueError):
		functional_amount(Decimal("100"), Decimal("0"))


def test_revaluation_gain():
	gain = revaluation_gain_loss(Decimal("1000"), Decimal("120"), Decimal("130"))
	assert gain == Decimal("10000.00")


def test_revaluation_loss():
	loss = revaluation_gain_loss(Decimal("1000"), Decimal("130"), Decimal("120"))
	assert loss == Decimal("-10000.00")


def test_revaluation_invalid_rates_raises():
	with pytest.raises(ValueError):
		revaluation_gain_loss(Decimal("1000"), Decimal("0"), Decimal("130"))


def test_hyperinflationary_restatement():
	result = hyperinflationary_restatement(
		Decimal("100000"),
		Decimal("100"),
		Decimal("150"),
	)
	assert result == Decimal("150000.00")


def test_hyperinflationary_zero_index_raises():
	with pytest.raises(ValueError):
		hyperinflationary_restatement(Decimal("100"), Decimal("0"), Decimal("150"))


# ---------------------------------------------------------------------------
# Accruals
# ---------------------------------------------------------------------------

def test_straight_line_accrual_full():
	result = straight_line_accrual(Decimal("12000"), 12, 12)
	assert result == Decimal("12000.00")


def test_straight_line_accrual_half():
	result = straight_line_accrual(Decimal("12000"), 12, 6)
	assert result == Decimal("6000.00")


def test_straight_line_accrual_zero_elapsed():
	result = straight_line_accrual(Decimal("12000"), 12, 0)
	assert result == Decimal("0.00")


def test_straight_line_accrual_invalid_periods_raises():
	with pytest.raises(ValueError):
		straight_line_accrual(Decimal("12000"), 0, 0)


def test_straight_line_accrual_out_of_range_raises():
	with pytest.raises(ValueError):
		straight_line_accrual(Decimal("12000"), 12, 13)


def test_prepaid_remaining():
	result = prepaid_remaining(Decimal("12000"), 12, 3)
	assert result == Decimal("9000.00")


# ---------------------------------------------------------------------------
# Depreciation
# ---------------------------------------------------------------------------

def test_straight_line_depreciation():
	result = straight_line_depreciation(Decimal("100000"), Decimal("10000"), 5)
	assert result == Decimal("18000.00")


def test_straight_line_depreciation_zero_life_raises():
	with pytest.raises(ValueError):
		straight_line_depreciation(Decimal("100000"), Decimal("0"), 0)


def test_straight_line_depreciation_residual_exceeds_cost_raises():
	with pytest.raises(ValueError):
		straight_line_depreciation(Decimal("10000"), Decimal("20000"), 5)


def test_declining_balance_depreciation():
	result = declining_balance_depreciation(Decimal("100000"), Decimal("0.20"))
	assert result == Decimal("20000.00")


def test_declining_balance_invalid_rate_raises():
	with pytest.raises(ValueError):
		declining_balance_depreciation(Decimal("100000"), Decimal("0"))
	with pytest.raises(ValueError):
		declining_balance_depreciation(Decimal("100000"), Decimal("1.5"))


# ---------------------------------------------------------------------------
# Financial ratios
# ---------------------------------------------------------------------------

def test_calculate_ratios_basic():
	ratios = calculate_ratios(
		revenue=Decimal("1000000"),
		pat=Decimal("200000"),
		total_assets=Decimal("2000000"),
		total_equity=Decimal("1000000"),
		current_assets=Decimal("500000"),
		current_liabilities=Decimal("250000"),
		total_debt=Decimal("500000"),
	)
	assert ratios.net_profit_margin_pct == Decimal("20.00")
	assert ratios.return_on_assets_pct == Decimal("10.00")
	assert ratios.return_on_equity_pct == Decimal("20.00")
	assert ratios.current_ratio == Decimal("2.0000")
	assert ratios.debt_to_equity == Decimal("0.5000")


def test_calculate_ratios_zero_denominators():
	ratios = calculate_ratios(
		revenue=Decimal("0"),
		pat=Decimal("0"),
		total_assets=Decimal("0"),
		total_equity=Decimal("0"),
		current_assets=Decimal("0"),
		current_liabilities=Decimal("0"),
		total_debt=Decimal("0"),
	)
	assert ratios.net_profit_margin_pct == Decimal("0")
	assert ratios.current_ratio == Decimal("0")


# ---------------------------------------------------------------------------
# Variance
# ---------------------------------------------------------------------------

def test_variance_positive():
	assert variance(Decimal("5000"), Decimal("4000")) == Decimal("1000.00")


def test_variance_negative():
	assert variance(Decimal("3000"), Decimal("4000")) == Decimal("-1000.00")


def test_variance_pct_basic():
	assert variance_pct(Decimal("5000"), Decimal("4000")) == Decimal("25.00")


def test_variance_pct_zero_budget():
	assert variance_pct(Decimal("5000"), Decimal("0")) == Decimal("0")


def test_variance_indicator_revenue_favourable():
	assert variance_indicator("revenue", Decimal("5000"), Decimal("4000")) == "F"


def test_variance_indicator_revenue_adverse():
	assert variance_indicator("revenue", Decimal("3000"), Decimal("4000")) == "A"


def test_variance_indicator_expense_favourable():
	assert variance_indicator("expense", Decimal("3000"), Decimal("4000")) == "F"


def test_variance_indicator_expense_adverse():
	assert variance_indicator("expense", Decimal("5000"), Decimal("4000")) == "A"


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------

def test_minority_interest_amount():
	result = minority_interest_amount(Decimal("1000000"), Decimal("0.30"))
	assert result == Decimal("300000.00")


def test_minority_interest_zero():
	result = minority_interest_amount(Decimal("1000000"), Decimal("0"))
	assert result == Decimal("0.00")


def test_minority_interest_invalid_raises():
	with pytest.raises(ValueError):
		minority_interest_amount(Decimal("1000000"), Decimal("1.5"))


def test_goodwill_on_acquisition():
	result = goodwill_on_acquisition(
		Decimal("500000"),  # consideration paid
		Decimal("400000"),  # fair value net assets
		Decimal("1.0"),     # 100% ownership
	)
	assert result == Decimal("100000.00")


def test_goodwill_negative_is_bargain_purchase():
	result = goodwill_on_acquisition(
		Decimal("300000"),
		Decimal("400000"),
		Decimal("1.0"),
	)
	assert result == Decimal("-100000.00")


# ---------------------------------------------------------------------------
# Tax
# ---------------------------------------------------------------------------

def test_withholding_tax():
	result = withholding_tax(Decimal("100000"), Decimal("0.05"))
	assert result == Decimal("5000.00")


def test_withholding_tax_invalid_rate_raises():
	with pytest.raises(ValueError):
		withholding_tax(Decimal("100000"), Decimal("1.5"))


def test_vat_exclusive_to_inclusive():
	vat, gross = vat_exclusive_to_inclusive(Decimal("1000"), Decimal("0.16"))
	assert vat == Decimal("160.00")
	assert gross == Decimal("1160.00")


def test_vat_inclusive_to_exclusive():
	net, vat = vat_inclusive_to_exclusive(Decimal("1160"), Decimal("0.16"))
	assert net == Decimal("1000.00")
	assert vat == Decimal("160.00")
