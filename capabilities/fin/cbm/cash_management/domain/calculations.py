"""
Cash Management — domain calculations.

All financial formulas are here: interest accrual, FX gain/loss, liquidity
ratios, hedge effectiveness, stress haircuts, float/value-date adjustments,
revolving credit headroom, and forecast accuracy metrics.

All inputs/outputs use Decimal for precision. Never use float arithmetic on
monetary amounts — callers must convert first.
"""

from __future__ import annotations

from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from typing import Sequence


TWO_DP = Decimal("0.01")
FOUR_DP = Decimal("0.0001")
SIX_DP = Decimal("0.000001")


# ---------------------------------------------------------------------------
# Interest
# ---------------------------------------------------------------------------

def calculate_simple_interest(
	principal: Decimal,
	annual_rate: Decimal,
	days: int,
	day_count: int = 365,
) -> Decimal:
	"""Simple interest accrual.

	Args:
		principal: Face value.
		annual_rate: Annual rate expressed as a decimal (e.g. 0.05 for 5 %).
		days: Number of days to accrue.
		day_count: Day-count convention (365 or 360).
	"""
	assert principal >= Decimal("0"), "principal must be non-negative"
	assert days >= 0, "days must be non-negative"
	assert day_count in (360, 365), "day_count must be 360 or 365"
	if days == 0 or principal == Decimal("0"):
		return Decimal("0")
	return (principal * annual_rate * Decimal(days) / Decimal(day_count)).quantize(TWO_DP, rounding=ROUND_HALF_UP)


def calculate_compound_interest(
	principal: Decimal,
	annual_rate: Decimal,
	days: int,
	compounds_per_year: int = 365,
) -> Decimal:
	"""Compound interest (money-market funds, notice accounts).

	Returns the interest amount (not total value).
	"""
	assert principal >= Decimal("0")
	assert compounds_per_year > 0
	if days == 0:
		return Decimal("0")
	n = Decimal(compounds_per_year)
	rate = annual_rate / n
	periods = Decimal(days) * n / Decimal(365)
	# Use float for exponentiation then convert — precision is recovered by quantize
	factor = Decimal(str(float((1 + float(rate)) ** float(periods))))
	interest = principal * (factor - Decimal("1"))
	return interest.quantize(TWO_DP, rounding=ROUND_HALF_UP)


def calculate_accrued_interest(
	principal: Decimal,
	annual_rate: Decimal,
	start_date: date,
	accrual_date: date,
	day_count: int = 365,
) -> Decimal:
	"""Accrued interest from start_date to accrual_date (exclusive end)."""
	days = (accrual_date - start_date).days
	if days <= 0:
		return Decimal("0")
	return calculate_simple_interest(principal, annual_rate, days, day_count)


# ---------------------------------------------------------------------------
# FX
# ---------------------------------------------------------------------------

def convert_currency(
	amount: Decimal,
	spot_rate: Decimal,
	inverse: bool = False,
) -> Decimal:
	"""Apply spot rate.  If inverse=True, divide (quote→base).

	spot_rate = units of quote per 1 unit of base.
	"""
	assert spot_rate > Decimal("0"), "spot_rate must be positive"
	if inverse:
		return (amount / spot_rate).quantize(TWO_DP, rounding=ROUND_HALF_UP)
	return (amount * spot_rate).quantize(TWO_DP, rounding=ROUND_HALF_UP)


def calculate_fx_unrealised_pnl(
	notional: Decimal,
	contracted_rate: Decimal,
	current_rate: Decimal,
	long: bool = True,
) -> Decimal:
	"""Mark-to-market unrealised P&L for an FX position or hedge.

	long=True  → bought base currency (will benefit from appreciation).
	long=False → sold base currency (will benefit from depreciation).
	"""
	assert contracted_rate > Decimal("0")
	assert current_rate > Decimal("0")
	rate_diff = current_rate - contracted_rate if long else contracted_rate - current_rate
	return (notional * rate_diff).quantize(TWO_DP, rounding=ROUND_HALF_UP)


def calculate_net_fx_exposure(
	long_amount: Decimal,
	short_amount: Decimal,
) -> Decimal:
	"""Net FX exposure.  Positive = net long, negative = net short."""
	return long_amount - short_amount


def calculate_hedge_ratio(hedged_amount: Decimal, total_exposure: Decimal) -> float:
	"""Hedge ratio as a fraction (0-1)."""
	if total_exposure == Decimal("0"):
		return 0.0
	ratio = float(hedged_amount / total_exposure)
	return min(max(ratio, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Liquidity
# ---------------------------------------------------------------------------

def calculate_days_cash_on_hand(
	available_cash: Decimal,
	average_daily_outflow: Decimal,
) -> int | None:
	"""Days of operations covered by current cash.  None if outflow is zero."""
	if average_daily_outflow <= Decimal("0"):
		return None
	days = int(available_cash / average_daily_outflow)
	return max(days, 0)


def calculate_liquidity_coverage_ratio(
	high_quality_liquid_assets: Decimal,
	total_net_cash_outflows_30d: Decimal,
) -> float | None:
	"""Basel III LCR: HQLA / net 30-day outflows.  Returns None if outflows = 0."""
	if total_net_cash_outflows_30d <= Decimal("0"):
		return None
	return float(high_quality_liquid_assets / total_net_cash_outflows_30d)


def apply_stress_haircut(
	amount: Decimal,
	scenario: str,
) -> Decimal:
	"""Apply scenario-specific liquidity haircut.

	Haircuts (illustrative — replace with institution policy):
	  base      → 0 %
	  optimistic → -5 % (less stress)
	  pessimistic → +20 %
	  stress    → +40 %
	"""
	haircuts: dict[str, Decimal] = {
		"base": Decimal("0.00"),
		"optimistic": Decimal("-0.05"),
		"pessimistic": Decimal("0.20"),
		"stress": Decimal("0.40"),
	}
	haircut = haircuts.get(scenario, Decimal("0.00"))
	return (amount * (Decimal("1") + haircut)).quantize(TWO_DP, rounding=ROUND_HALF_UP)


def calculate_concentration_risk(
	account_balance: Decimal,
	total_cash: Decimal,
) -> float:
	"""Single-bank or single-account concentration as a percentage (0-100)."""
	if total_cash <= Decimal("0"):
		return 0.0
	return float((account_balance / total_cash * Decimal("100")).quantize(TWO_DP, rounding=ROUND_HALF_UP))


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------

def calculate_reconciliation_variance(
	bank_balance: Decimal,
	book_balance: Decimal,
) -> Decimal:
	"""Bank statement balance minus book (ledger) balance.

	Positive variance → more cash per bank than books.
	Negative variance → shortfall per bank vs books.
	"""
	return bank_balance - book_balance


def is_variance_material(variance: Decimal, threshold: Decimal) -> bool:
	"""True if abs(variance) exceeds the materiality threshold."""
	return abs(variance) > threshold


def adjust_for_outstanding_items(
	bank_balance: Decimal,
	outstanding_deposits: Decimal,
	outstanding_payments: Decimal,
	bank_errors: Decimal,
	book_errors: Decimal,
) -> Decimal:
	"""Compute the adjusted bank balance for reconciliation.

	Adjusted bank = bank + outstanding deposits - outstanding payments +/- errors.
	"""
	return bank_balance + outstanding_deposits - outstanding_payments + bank_errors - book_errors


# ---------------------------------------------------------------------------
# Cash pooling
# ---------------------------------------------------------------------------

def calculate_notional_pool_balance(account_balances: Sequence[Decimal]) -> Decimal:
	"""Sum of all participating account balances (notional offset)."""
	return sum(account_balances, Decimal("0"))


def calculate_physical_sweep_amount(
	account_balance: Decimal,
	target_balance: Decimal,
	method: str = "zero_balance",
) -> Decimal:
	"""Amount to sweep from/to an account.

	method = 'zero_balance' → sweep all above zero.
	method = 'target_balance' → sweep to maintain target.
	Returns positive amount to sweep OUT, negative to fund IN.
	"""
	if method == "zero_balance":
		return max(account_balance, Decimal("0"))
	elif method == "target_balance":
		return account_balance - target_balance
	elif method == "threshold":
		return max(account_balance - target_balance, Decimal("0"))
	return Decimal("0")


def calculate_interest_savings_from_pooling(
	gross_debit_balance: Decimal,
	gross_credit_balance: Decimal,
	debit_rate: Decimal,
	credit_rate: Decimal,
	days: int = 1,
) -> Decimal:
	"""Estimate daily interest saving from notional pooling.

	Without pooling: pay debit_rate on overdrafts + earn credit_rate on credits.
	With pooling: net them — saving = gross interest cost that's eliminated.
	"""
	net = gross_credit_balance - gross_debit_balance
	if net >= Decimal("0"):
		saving = calculate_simple_interest(gross_debit_balance, debit_rate, days)
	else:
		saving = calculate_simple_interest(abs(net), debit_rate, days)
	return saving


# ---------------------------------------------------------------------------
# Revolving credit
# ---------------------------------------------------------------------------

def calculate_credit_headroom(
	credit_limit: Decimal,
	utilised: Decimal,
) -> Decimal:
	"""Available headroom on a revolving credit facility."""
	return max(credit_limit - utilised, Decimal("0"))


def calculate_overdraft_interest(
	overdrawn_amount: Decimal,
	annual_rate: Decimal,
	days: int,
) -> Decimal:
	"""Interest charge on an overdrawn account balance."""
	assert overdrawn_amount >= Decimal("0")
	return calculate_simple_interest(overdrawn_amount, annual_rate, days)


# ---------------------------------------------------------------------------
# Value dating
# ---------------------------------------------------------------------------

def apply_value_date_offset(transaction_date: date, offset_days: int) -> date:
	"""Return value date given transaction date and bank's float offset."""
	from datetime import timedelta
	assert offset_days >= 0, "offset_days must be non-negative"
	return transaction_date + timedelta(days=offset_days)


def is_same_day_value(transaction_date: date, value_date: date) -> bool:
	"""True when transaction clears with same-day value (offset = 0)."""
	return transaction_date == value_date


# ---------------------------------------------------------------------------
# Forecast accuracy
# ---------------------------------------------------------------------------

def calculate_forecast_mape(
	actuals: Sequence[Decimal],
	forecasts: Sequence[Decimal],
) -> float:
	"""Mean Absolute Percentage Error (MAPE) for forecast evaluation.

	Returns a percentage (0-100+).  Skips periods where actual = 0.
	"""
	assert len(actuals) == len(forecasts), "actuals and forecasts must be same length"
	errors = []
	for a, f in zip(actuals, forecasts):
		if a == Decimal("0"):
			continue
		errors.append(float(abs(a - f) / abs(a)) * 100)
	if not errors:
		return 0.0
	return sum(errors) / len(errors)


def calculate_forecast_bias(
	actuals: Sequence[Decimal],
	forecasts: Sequence[Decimal],
) -> float:
	"""Mean signed error — positive means systematic over-forecast."""
	assert len(actuals) == len(forecasts)
	diffs = [float(f - a) for a, f in zip(actuals, forecasts)]
	return sum(diffs) / len(diffs) if diffs else 0.0


# ---------------------------------------------------------------------------
# M-Pesa / mobile money specifics
# ---------------------------------------------------------------------------

def parse_mpesa_amount(raw: str) -> Decimal:
	"""Parse an M-Pesa amount string like 'KES 10,250.00' → Decimal."""
	cleaned = raw.replace(",", "").strip()
	# Strip currency prefix if present
	for prefix in ("KES", "UGX", "TZS", "GHS", "NGN"):
		if cleaned.upper().startswith(prefix):
			cleaned = cleaned[len(prefix):].strip()
			break
	return Decimal(cleaned).quantize(TWO_DP, rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Weighted average cost
# ---------------------------------------------------------------------------

def weighted_average_rate(
	amounts: Sequence[Decimal],
	rates: Sequence[Decimal],
) -> Decimal:
	"""Weighted average rate across multiple positions/instruments."""
	assert len(amounts) == len(rates)
	total_amount = sum(amounts, Decimal("0"))
	if total_amount == Decimal("0"):
		return Decimal("0")
	weighted_sum = sum(a * r for a, r in zip(amounts, rates))
	return (weighted_sum / total_amount).quantize(FOUR_DP, rounding=ROUND_HALF_UP)
