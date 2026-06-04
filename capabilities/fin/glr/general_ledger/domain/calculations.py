"""
General Ledger — financial calculations.

All functions are pure (no I/O), fully type-annotated, and handle edge cases
explicitly.  Callers pass in already-validated Decimal values.

© 2025 Datacraft. All rights reserved.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import NamedTuple


TWO = Decimal("0.01")
FOUR = Decimal("0.0001")
SIX = Decimal("0.000001")


def _q2(v: Decimal) -> Decimal:
	"""Round to 2 decimal places (monetary)."""
	return v.quantize(TWO, rounding=ROUND_HALF_UP)


def _q4(v: Decimal) -> Decimal:
	"""Round to 4 decimal places (exchange rates, unit costs)."""
	return v.quantize(FOUR, rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Core double-entry helpers
# ---------------------------------------------------------------------------

def net_balance(opening: Decimal, debits: Decimal, credits: Decimal, normal_balance: str) -> Decimal:
	"""
	Compute the closing balance for an account.

	For debit-normal accounts (assets, expenses): closing = opening + debits - credits
	For credit-normal accounts (liabilities, equity, revenue): closing = opening + credits - debits
	"""
	if normal_balance == "debit":
		return _q2(opening + debits - credits)
	return _q2(opening + credits - debits)


def debit_credit_split(amount: Decimal, is_debit: bool) -> tuple[Decimal, Decimal]:
	"""Return (debit, credit) from a signed amount + side indicator."""
	if amount < 0:
		raise ValueError(f"amount must be non-negative; got {amount}")
	if is_debit:
		return amount, Decimal("0")
	return Decimal("0"), amount


def validate_balance(total_debit: Decimal, total_credit: Decimal) -> bool:
	"""Return True iff the journal balances."""
	return total_debit == total_credit and total_debit > 0


# ---------------------------------------------------------------------------
# Multi-currency
# ---------------------------------------------------------------------------

def convert_currency(
	amount: Decimal,
	exchange_rate: Decimal,
	rounding_places: int = 2,
) -> Decimal:
	"""
	Convert *amount* in foreign currency to functional currency.

	exchange_rate = units of functional currency per 1 unit of foreign currency.
	e.g. USD/KES = 130.00 means 1 USD = 130 KES; to convert 500 KES to USD pass
	rate=130 and divide (amount / rate).
	"""
	if exchange_rate <= 0:
		raise ValueError(f"exchange_rate must be > 0; got {exchange_rate}")
	quantizer = Decimal("0." + "0" * rounding_places)
	return (amount / exchange_rate).quantize(quantizer, rounding=ROUND_HALF_UP)


def functional_amount(
	foreign_amount: Decimal,
	exchange_rate: Decimal,
) -> Decimal:
	"""Multiply foreign_amount by exchange_rate → functional currency amount."""
	if exchange_rate <= 0:
		raise ValueError(f"exchange_rate must be > 0; got {exchange_rate}")
	return _q2(foreign_amount * exchange_rate)


def revaluation_gain_loss(
	foreign_balance: Decimal,
	old_rate: Decimal,
	new_rate: Decimal,
) -> Decimal:
	"""
	Compute unrealised FX revaluation gain or loss.

	Positive = gain, negative = loss.
	For monetary asset (debit-normal): gain if rate rises.
	revaluation = foreign_balance * (new_rate - old_rate)
	"""
	if old_rate <= 0 or new_rate <= 0:
		raise ValueError("exchange rates must be positive")
	return _q2(foreign_balance * (new_rate - old_rate))


def hyperinflationary_restatement(
	historical_amount: Decimal,
	historical_price_index: Decimal,
	current_price_index: Decimal,
) -> Decimal:
	"""
	IAS 29 restatement of non-monetary items.

	restated = historical_amount * (current_index / historical_index)
	"""
	if historical_price_index <= 0 or current_price_index <= 0:
		raise ValueError("price indices must be positive")
	factor = current_price_index / historical_price_index
	return _q2(historical_amount * factor)


# ---------------------------------------------------------------------------
# Period-end accruals
# ---------------------------------------------------------------------------

def straight_line_accrual(
	total_amount: Decimal,
	total_periods: int,
	periods_elapsed: int,
) -> Decimal:
	"""Accrued amount under straight-line method."""
	if total_periods <= 0:
		raise ValueError("total_periods must be > 0")
	if periods_elapsed < 0 or periods_elapsed > total_periods:
		raise ValueError("periods_elapsed out of range [0, total_periods]")
	return _q2(total_amount * Decimal(periods_elapsed) / Decimal(total_periods))


def prepaid_remaining(
	total_amount: Decimal,
	total_periods: int,
	periods_elapsed: int,
) -> Decimal:
	"""Unexpired prepaid amount = total − accrued so far."""
	accrued = straight_line_accrual(total_amount, total_periods, periods_elapsed)
	return _q2(total_amount - accrued)


# ---------------------------------------------------------------------------
# Depreciation
# ---------------------------------------------------------------------------

def straight_line_depreciation(
	cost: Decimal,
	residual_value: Decimal,
	useful_life_periods: int,
) -> Decimal:
	"""Annual/period depreciation charge under SL method."""
	if useful_life_periods <= 0:
		raise ValueError("useful_life_periods must be > 0")
	depreciable = cost - residual_value
	if depreciable < 0:
		raise ValueError("residual_value exceeds cost")
	return _q2(depreciable / Decimal(useful_life_periods))


def declining_balance_depreciation(
	book_value: Decimal,
	rate: Decimal,
) -> Decimal:
	"""
	Depreciation for one period under declining balance method.
	rate is expressed as a fraction (e.g. 0.20 for 20%).
	"""
	if rate <= 0 or rate > 1:
		raise ValueError("rate must be in (0, 1]")
	return _q2(book_value * rate)


# ---------------------------------------------------------------------------
# Financial ratios
# ---------------------------------------------------------------------------

class FinancialRatios(NamedTuple):
	net_profit_margin_pct: Decimal
	return_on_assets_pct: Decimal
	return_on_equity_pct: Decimal
	current_ratio: Decimal
	debt_to_equity: Decimal


def calculate_ratios(
	revenue: Decimal,
	pat: Decimal,
	total_assets: Decimal,
	total_equity: Decimal,
	current_assets: Decimal,
	current_liabilities: Decimal,
	total_debt: Decimal,
) -> FinancialRatios:
	"""Compute standard financial ratios; return 0 where denominator is zero."""

	def _safe_ratio(numerator: Decimal, denominator: Decimal) -> Decimal:
		if denominator == 0:
			return Decimal("0")
		return _q2(numerator / denominator * 100)

	def _safe_plain(numerator: Decimal, denominator: Decimal) -> Decimal:
		if denominator == 0:
			return Decimal("0")
		return _q4(numerator / denominator)

	return FinancialRatios(
		net_profit_margin_pct=_safe_ratio(pat, revenue),
		return_on_assets_pct=_safe_ratio(pat, total_assets),
		return_on_equity_pct=_safe_ratio(pat, total_equity),
		current_ratio=_safe_plain(current_assets, current_liabilities),
		debt_to_equity=_safe_plain(total_debt, total_equity),
	)


# ---------------------------------------------------------------------------
# Variance analysis
# ---------------------------------------------------------------------------

def variance(actual: Decimal, budget: Decimal) -> Decimal:
	"""actual − budget; positive = overspend for expense, over-delivery for revenue."""
	return _q2(actual - budget)


def variance_pct(actual: Decimal, budget: Decimal) -> Decimal:
	"""Percentage variance; returns 0 if budget is 0."""
	if budget == 0:
		return Decimal("0")
	return _q2((actual - budget) / abs(budget) * 100)


def variance_indicator(account_type: str, actual: Decimal, budget: Decimal) -> str:
	"""
	Return 'F' (favourable) or 'A' (adverse).

	For revenue:  actual > budget → F
	For expense:  actual < budget → F
	"""
	var = actual - budget
	if account_type in ("revenue", "equity", "liability"):
		return "F" if var >= 0 else "A"
	# asset, expense, contra
	return "F" if var <= 0 else "A"


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------

def minority_interest_amount(
	subsidiary_equity: Decimal,
	minority_percentage_decimal: Decimal,
) -> Decimal:
	"""NCI = subsidiary net assets × minority ownership fraction."""
	if not (0 <= minority_percentage_decimal <= 1):
		raise ValueError("minority_percentage_decimal must be in [0, 1]")
	return _q2(subsidiary_equity * minority_percentage_decimal)


def goodwill_on_acquisition(
	consideration_paid: Decimal,
	fair_value_net_assets: Decimal,
	ownership_fraction: Decimal,
) -> Decimal:
	"""IFRS 3 goodwill = consideration − (fair_value_net_assets × ownership_fraction)."""
	return _q2(consideration_paid - fair_value_net_assets * ownership_fraction)


# ---------------------------------------------------------------------------
# Tax
# ---------------------------------------------------------------------------

def withholding_tax(gross_amount: Decimal, rate: Decimal) -> Decimal:
	"""WHT amount = gross × rate (rate as fraction, e.g. 0.05 for 5%)."""
	if not (0 <= rate <= 1):
		raise ValueError("rate must be in [0, 1]")
	return _q2(gross_amount * rate)


def vat_exclusive_to_inclusive(net_amount: Decimal, vat_rate: Decimal) -> tuple[Decimal, Decimal]:
	"""Returns (vat_amount, gross_amount) from a net amount."""
	vat = _q2(net_amount * vat_rate)
	return vat, _q2(net_amount + vat)


def vat_inclusive_to_exclusive(gross_amount: Decimal, vat_rate: Decimal) -> tuple[Decimal, Decimal]:
	"""Returns (net_amount, vat_amount) from a gross amount."""
	net = _q2(gross_amount / (1 + vat_rate))
	vat = _q2(gross_amount - net)
	return net, vat
