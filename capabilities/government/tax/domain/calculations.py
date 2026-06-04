"""Financial and tax-domain calculations for Tax Administration.

All functions are pure (no I/O, no side effects).
Decimal arithmetic throughout — never float for money.
"""
from __future__ import annotations

from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from typing import Any


CENT = Decimal("0.01")
ZERO = Decimal("0")
DAYS_PER_YEAR = Decimal("365")


def _round(v: Decimal) -> Decimal:
	return v.quantize(CENT, rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Income tax (individual, PAYE-style progressive)
# ---------------------------------------------------------------------------

# Kenya-style illustrative bands — configurable at runtime
_DEFAULT_INCOME_BANDS: list[tuple[Decimal, Decimal, Decimal]] = [
	# (lower, upper, rate)  — upper=None means top band
	(ZERO, Decimal("288000"), Decimal("0.10")),
	(Decimal("288001"), Decimal("388000"), Decimal("0.25")),
	(Decimal("388001"), Decimal("6000000"), Decimal("0.30")),
	(Decimal("6000001"), Decimal("9600000"), Decimal("0.325")),
	(Decimal("9600001"), Decimal("999999999999"), Decimal("0.35")),
]


def calculate_income_tax(
	taxable_income: Decimal,
	bands: list[tuple[Decimal, Decimal, Decimal]] | None = None,
	personal_relief: Decimal = Decimal("28800"),
) -> Decimal:
	"""Progressive income tax using configurable bands.

	Args:
		taxable_income: Annual taxable income.
		bands: List of (lower, upper, rate) tuples (inclusive bounds).
		personal_relief: Fixed annual personal relief amount.

	Returns:
		Net income tax payable (>= 0).
	"""
	assert taxable_income >= ZERO, "taxable_income must be non-negative"
	active_bands = bands or _DEFAULT_INCOME_BANDS
	tax = ZERO
	for lower, upper, rate in active_bands:
		if taxable_income <= lower:
			break
		band_income = min(taxable_income, upper) - lower
		tax += band_income * rate
	net = max(ZERO, tax - personal_relief)
	return _round(net)


# ---------------------------------------------------------------------------
# VAT
# ---------------------------------------------------------------------------

def calculate_vat_payable(
	output_vat: Decimal,
	input_vat: Decimal,
	rate: Decimal = Decimal("0.16"),
) -> Decimal:
	"""Net VAT payable = output VAT - input VAT credit.

	Returns positive = payment due; negative = refund due.
	"""
	assert output_vat >= ZERO
	assert input_vat >= ZERO
	return _round(output_vat - input_vat)


def calculate_vat_on_amount(
	amount_exclusive: Decimal,
	rate: Decimal = Decimal("0.16"),
) -> tuple[Decimal, Decimal]:
	"""Returns (vat_amount, gross_inclusive)."""
	assert amount_exclusive >= ZERO
	vat = _round(amount_exclusive * rate)
	return vat, _round(amount_exclusive + vat)


def extract_vat_from_inclusive(
	amount_inclusive: Decimal,
	rate: Decimal = Decimal("0.16"),
) -> tuple[Decimal, Decimal]:
	"""Returns (vat_amount, net_exclusive) from a VAT-inclusive amount."""
	assert amount_inclusive >= ZERO
	net = _round(amount_inclusive / (1 + rate))
	vat = _round(amount_inclusive - net)
	return vat, net


# ---------------------------------------------------------------------------
# Withholding Tax
# ---------------------------------------------------------------------------

def calculate_withholding_tax(
	gross_payment: Decimal,
	wht_rate: Decimal,
) -> Decimal:
	"""WHT = gross_payment × rate."""
	assert gross_payment >= ZERO
	assert ZERO <= wht_rate <= Decimal("1"), "WHT rate must be in [0, 1]"
	return _round(gross_payment * wht_rate)


# ---------------------------------------------------------------------------
# Turnover Tax (simplified presumptive)
# ---------------------------------------------------------------------------

def calculate_turnover_tax(
	gross_turnover: Decimal,
	rate: Decimal = Decimal("0.015"),
) -> Decimal:
	"""Turnover tax = gross_turnover × rate."""
	assert gross_turnover >= ZERO
	return _round(gross_turnover * rate)


# ---------------------------------------------------------------------------
# Capital Gains Tax
# ---------------------------------------------------------------------------

def calculate_capital_gain(
	proceeds: Decimal,
	cost: Decimal,
	improvements: Decimal = ZERO,
	inflation_adjustment: Decimal = ZERO,
) -> Decimal:
	"""Net gain = proceeds − (cost + improvements + inflation_adjustment)."""
	gain = proceeds - cost - improvements - inflation_adjustment
	return _round(max(ZERO, gain))


def calculate_capital_gains_tax(
	net_gain: Decimal,
	rate: Decimal = Decimal("0.15"),
) -> Decimal:
	return _round(net_gain * rate)


# ---------------------------------------------------------------------------
# Late filing penalty
# ---------------------------------------------------------------------------

def calculate_late_filing_penalty(
	tax_due: Decimal,
	rate: Decimal = Decimal("0.05"),
	minimum_penalty: Decimal = Decimal("2000"),
	maximum_penalty: Decimal | None = None,
) -> Decimal:
	"""Late filing penalty = max(minimum, tax_due × rate), capped at maximum if given."""
	assert tax_due >= ZERO
	penalty = max(minimum_penalty, _round(tax_due * rate))
	if maximum_penalty is not None:
		penalty = min(penalty, maximum_penalty)
	return _round(penalty)


# ---------------------------------------------------------------------------
# Late payment penalty
# ---------------------------------------------------------------------------

def calculate_late_payment_penalty(
	tax_due: Decimal,
	rate: Decimal = Decimal("0.05"),
) -> Decimal:
	"""One-off late payment penalty = tax_due × rate (typically 5%)."""
	assert tax_due >= ZERO
	return _round(tax_due * rate)


# ---------------------------------------------------------------------------
# Understatement penalty
# ---------------------------------------------------------------------------

def calculate_understatement_penalty(
	understated_amount: Decimal,
	penalty_tier: str = "careless",
) -> Decimal:
	"""
	Tiered understatement penalty.

	Tiers: no_reasonable_care → 25%, careless → 50%, deliberate → 75%, fraud → 100%.
	"""
	RATES: dict[str, Decimal] = {
		"no_reasonable_care": Decimal("0.25"),
		"careless": Decimal("0.50"),
		"deliberate": Decimal("0.75"),
		"fraud": Decimal("1.00"),
	}
	rate = RATES.get(penalty_tier, Decimal("0.50"))
	return _round(understated_amount * rate)


# ---------------------------------------------------------------------------
# Interest on overdue tax
# ---------------------------------------------------------------------------

def calculate_interest(
	principal: Decimal,
	annual_rate: Decimal,
	from_date: date,
	to_date: date,
) -> Decimal:
	"""Simple interest = principal × annual_rate × days / 365.

	Returns 0 if to_date <= from_date.
	"""
	assert principal >= ZERO
	assert ZERO <= annual_rate <= Decimal("1"), "annual_rate must be in [0, 1]"
	days = (to_date - from_date).days
	if days <= 0:
		return ZERO
	return _round(principal * annual_rate * Decimal(str(days)) / DAYS_PER_YEAR)


def calculate_compound_interest(
	principal: Decimal,
	monthly_rate: Decimal,
	months: int,
) -> Decimal:
	"""Compound interest over N months: principal × (1 + r)^N - principal."""
	assert principal >= ZERO
	assert months >= 0
	total = principal * (1 + monthly_rate) ** months
	return _round(total - principal)


# ---------------------------------------------------------------------------
# Debt total
# ---------------------------------------------------------------------------

def calculate_total_debt(
	principal: Decimal,
	penalty: Decimal,
	interest: Decimal,
) -> Decimal:
	assert principal >= ZERO
	assert penalty >= ZERO
	assert interest >= ZERO
	return _round(principal + penalty + interest)


def calculate_debt_balance(total: Decimal, amount_paid: Decimal) -> Decimal:
	assert amount_paid >= ZERO
	return _round(max(ZERO, total - amount_paid))


# ---------------------------------------------------------------------------
# Refund interest (payable by tax authority to taxpayer)
# ---------------------------------------------------------------------------

def calculate_refund_interest(
	refund_amount: Decimal,
	annual_rate: Decimal,
	approval_date: date,
	payment_date: date,
) -> Decimal:
	"""Interest the authority owes on delayed refunds."""
	return calculate_interest(refund_amount, annual_rate, approval_date, payment_date)


# ---------------------------------------------------------------------------
# Compliance risk scoring
# ---------------------------------------------------------------------------

def calculate_compliance_risk_score(
	*,
	years_registered: int,
	returns_filed: int,
	returns_due: int,
	payments_on_time: int,
	payments_due: int,
	open_audits: int,
	prior_fraud_flags: int,
	days_avg_late_filing: float,
	debt_to_turnover_ratio: float,
) -> tuple[Decimal, str]:
	"""Composite risk score 0–100 (higher = riskier).

	Returns (score, risk_category) where category is low/medium/high/critical.
	"""
	score = Decimal("0")

	# Filing compliance (0–30 pts)
	if returns_due > 0:
		filing_rate = Decimal(str(returns_filed / returns_due))
		score += (1 - filing_rate) * 30

	# Payment compliance (0–25 pts)
	if payments_due > 0:
		payment_rate = Decimal(str(payments_on_time / payments_due))
		score += (1 - payment_rate) * 25

	# Active audits (0–15 pts)
	score += min(Decimal(str(open_audits * 5)), Decimal("15"))

	# Prior fraud (0–20 pts)
	score += min(Decimal(str(prior_fraud_flags * 10)), Decimal("20"))

	# Average late filing days (0–5 pts)
	score += min(Decimal(str(days_avg_late_filing / 30)), Decimal("5"))

	# Debt burden (0–5 pts)
	score += min(Decimal(str(debt_to_turnover_ratio * 5)), Decimal("5"))

	score = _round(min(score, Decimal("100")))

	if score < Decimal("25"):
		category = "low"
	elif score < Decimal("50"):
		category = "medium"
	elif score < Decimal("75"):
		category = "high"
	else:
		category = "critical"

	return score, category


# ---------------------------------------------------------------------------
# Return due date
# ---------------------------------------------------------------------------

def calculate_return_due_date(period_end: date, filing_frequency: str, due_day: int = 20) -> date:
	"""Calculate the return due date based on period end and frequency.

	For monthly: 20th of the following month.
	For quarterly/annual: due_day days after period end.
	"""
	import calendar

	if filing_frequency == "monthly":
		# Next month, day = due_day
		year = period_end.year + (period_end.month // 12)
		month = (period_end.month % 12) + 1
		max_day = calendar.monthrange(year, month)[1]
		return date(year, month, min(due_day, max_day))
	else:
		# Annual / quarterly: due_day days after period end
		import datetime as dt
		return period_end + dt.timedelta(days=due_day)


# ---------------------------------------------------------------------------
# Certificate validity
# ---------------------------------------------------------------------------

def calculate_certificate_expiry(issue_date: date, validity_months: int) -> date:
	"""Expiry = issue_date + validity_months months."""
	month = issue_date.month - 1 + validity_months
	year = issue_date.year + month // 12
	month = month % 12 + 1
	import calendar
	day = min(issue_date.day, calendar.monthrange(year, month)[1])
	return date(year, month, day)
