"""
AP financial calculations — pure functions, no side effects.

All Decimal arithmetic uses ROUND_HALF_UP per standard accounting convention.
Currency conversion always passes through a provided exchange_rate rather than
fetching live rates (separation of concerns).

© 2025 Datacraft. All rights reserved.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from typing import Any


_2DP = Decimal("0.01")
_4DP = Decimal("0.0001")
_6DP = Decimal("0.000001")


def _round2(v: Decimal) -> Decimal:
	return v.quantize(_2DP, rounding=ROUND_HALF_UP)


def _round4(v: Decimal) -> Decimal:
	return v.quantize(_4DP, rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Line-level calculations
# ---------------------------------------------------------------------------

def calculate_line_subtotal(quantity: Decimal, unit_price: Decimal) -> Decimal:
	"""quantity × unit_price, rounded to 2dp."""
	return _round2(quantity * unit_price)


def calculate_line_tax(subtotal: Decimal, tax_rate_pct: Decimal) -> Decimal:
	"""Tax on a single line."""
	return _round2(subtotal * tax_rate_pct / Decimal("100"))


def calculate_retention_amount(subtotal: Decimal, retention_pct: Decimal) -> Decimal:
	"""Retention withheld from a line (construction)."""
	return _round2(subtotal * retention_pct / Decimal("100"))


def calculate_line_total(
	subtotal: Decimal,
	tax_amount: Decimal,
	retention_amount: Decimal = Decimal("0"),
) -> Decimal:
	"""Net payable for a line after adding tax and deducting retention."""
	return _round2(subtotal + tax_amount - retention_amount)


# ---------------------------------------------------------------------------
# Invoice-level calculations
# ---------------------------------------------------------------------------

def calculate_invoice_subtotal(lines: list[dict[str, Any]]) -> Decimal:
	"""Sum of all line_subtotal values."""
	return _round2(sum(Decimal(str(ln["line_subtotal"])) for ln in lines))


def calculate_invoice_tax(lines: list[dict[str, Any]]) -> Decimal:
	"""Sum of all line tax_amount values."""
	return _round2(sum(Decimal(str(ln.get("tax_amount", "0"))) for ln in lines))


def calculate_invoice_retention(lines: list[dict[str, Any]]) -> Decimal:
	"""Sum of all retention_amount values."""
	return _round2(sum(Decimal(str(ln.get("retention_amount", "0"))) for ln in lines))


def calculate_invoice_total(subtotal: Decimal, tax: Decimal, retention: Decimal = Decimal("0")) -> Decimal:
	return _round2(subtotal + tax - retention)


def calculate_outstanding(total: Decimal, paid: Decimal) -> Decimal:
	"""Outstanding balance, floored at zero (overpayments handled separately)."""
	return _round2(max(Decimal("0"), total - paid))


# ---------------------------------------------------------------------------
# Currency conversion
# ---------------------------------------------------------------------------

def convert_to_base(amount: Decimal, exchange_rate: Decimal) -> Decimal:
	"""Convert invoice-currency amount to functional (base) currency."""
	assert exchange_rate > Decimal("0"), "exchange_rate must be positive"
	return _round2(amount * exchange_rate)


def convert_from_base(amount_base: Decimal, exchange_rate: Decimal) -> Decimal:
	"""Convert base-currency amount back to invoice currency."""
	assert exchange_rate > Decimal("0"), "exchange_rate must be positive"
	return _round2(amount_base / exchange_rate)


def calculate_fx_gain_loss(
	invoice_amount_base: Decimal,
	payment_amount_base: Decimal,
) -> Decimal:
	"""FX realised gain (positive) or loss (negative)."""
	return _round2(payment_amount_base - invoice_amount_base)


# ---------------------------------------------------------------------------
# Due date and early-payment discount
# ---------------------------------------------------------------------------

def calculate_due_date(invoice_date: date, net_days: int, end_of_month: bool = False) -> date:
	"""Compute due date from invoice date and net days."""
	from datetime import timedelta
	import calendar

	due = invoice_date + timedelta(days=net_days)
	if end_of_month:
		# Snap to last day of the resulting month
		last_day = calendar.monthrange(due.year, due.month)[1]
		due = due.replace(day=last_day)
	return due


def calculate_discount_due_date(invoice_date: date, discount_days: int) -> date:
	from datetime import timedelta
	return invoice_date + timedelta(days=discount_days)


def calculate_early_discount_amount(invoice_total: Decimal, discount_pct: Decimal) -> Decimal:
	"""Gross discount amount available under terms like 2/10."""
	return _round2(invoice_total * discount_pct / Decimal("100"))


def calculate_net_payment_with_discount(invoice_total: Decimal, discount_pct: Decimal) -> Decimal:
	"""Amount to pay when capturing early discount."""
	return _round2(invoice_total - calculate_early_discount_amount(invoice_total, discount_pct))


def calculate_annualised_discount_return(
	discount_pct: Decimal,
	discount_days: int,
	net_days: int,
) -> Decimal:
	"""
	Annualised return of capturing an early payment discount.

	Formula: (discount_pct / (1 - discount_pct/100)) * (365 / (net_days - discount_days))

	A 2/10 Net30 term yields ~36.7% annualised — almost always worth taking.
	"""
	days_saved = net_days - discount_days
	if days_saved <= 0 or discount_pct <= Decimal("0"):
		return Decimal("0.00")
	rate = discount_pct / (Decimal("100") - discount_pct)
	annualised = rate * Decimal("365") / Decimal(str(days_saved))
	return _round2(annualised * Decimal("100"))  # return as percentage


def is_within_discount_window(invoice_date: date, discount_days: int, payment_date: date | None = None) -> bool:
	"""True if payment_date (default today) is within the discount capture window."""
	if payment_date is None:
		payment_date = date.today()
	deadline = calculate_discount_due_date(invoice_date, discount_days)
	return payment_date <= deadline


def calculate_late_payment_penalty(
	outstanding: Decimal,
	penalty_pct_per_month: Decimal,
	days_overdue: int,
) -> Decimal:
	"""
	Simple interest late payment penalty.
	penalty = outstanding × (penalty_pct_per_month / 100) × (days_overdue / 30)
	"""
	if days_overdue <= 0 or penalty_pct_per_month <= Decimal("0"):
		return Decimal("0.00")
	return _round2(outstanding * penalty_pct_per_month / Decimal("100") * Decimal(str(days_overdue)) / Decimal("30"))


def calculate_days_overdue(due_date: date, reference_date: date | None = None) -> int:
	"""Days past due. Negative means not yet due."""
	if reference_date is None:
		reference_date = date.today()
	return (reference_date - due_date).days


# ---------------------------------------------------------------------------
# Three-way match variance
# ---------------------------------------------------------------------------

def calculate_price_variance_pct(
	invoice_unit_price: Decimal,
	po_unit_price: Decimal,
) -> Decimal:
	"""
	Signed price variance as a percentage of PO price.
	Positive means invoice is MORE expensive than PO.
	"""
	if po_unit_price == Decimal("0"):
		return Decimal("0.0000")
	variance = (invoice_unit_price - po_unit_price) / po_unit_price * Decimal("100")
	return _round4(variance)


def calculate_qty_variance_pct(
	invoice_qty: Decimal,
	po_qty: Decimal,
) -> Decimal:
	"""
	Signed quantity variance as a percentage of PO quantity.
	Positive means invoice claims more than PO.
	"""
	if po_qty == Decimal("0"):
		return Decimal("0.0000")
	variance = (invoice_qty - po_qty) / po_qty * Decimal("100")
	return _round4(variance)


# ---------------------------------------------------------------------------
# AP aging
# ---------------------------------------------------------------------------

def assign_aging_bucket(days_overdue: int) -> str:
	"""Map days-overdue to a standard aging bucket label."""
	if days_overdue <= 0:
		return "current"
	elif days_overdue <= 30:
		return "days_1_30"
	elif days_overdue <= 60:
		return "days_31_60"
	elif days_overdue <= 90:
		return "days_61_90"
	elif days_overdue <= 120:
		return "days_91_120"
	else:
		return "over_120"


# ---------------------------------------------------------------------------
# DPO (Days Payable Outstanding)
# ---------------------------------------------------------------------------

def calculate_dpo(
	accounts_payable_balance: Decimal,
	cost_of_goods_sold: Decimal,
	period_days: int = 365,
) -> Decimal:
	"""
	DPO = (AP Balance / COGS) × period_days.
	Lower DPO = paying too fast (losing float).
	Higher DPO = potential strain on supplier relationships.
	"""
	if cost_of_goods_sold <= Decimal("0"):
		return Decimal("0.00")
	return _round2(accounts_payable_balance / cost_of_goods_sold * Decimal(str(period_days)))


# ---------------------------------------------------------------------------
# Withholding tax
# ---------------------------------------------------------------------------

def calculate_withholding_tax(
	gross_amount: Decimal,
	wht_rate_pct: Decimal,
) -> tuple[Decimal, Decimal]:
	"""
	Returns (wht_amount, net_payable).
	net_payable = gross_amount − wht_amount
	"""
	wht = _round2(gross_amount * wht_rate_pct / Decimal("100"))
	net = _round2(gross_amount - wht)
	return wht, net


# ---------------------------------------------------------------------------
# Retention release
# ---------------------------------------------------------------------------

def calculate_retention_release(
	original_retention: Decimal,
	release_pct: Decimal,
) -> tuple[Decimal, Decimal]:
	"""
	Returns (released_amount, remaining_retention).
	release_pct is the percentage of the held retention to release (0-100).
	"""
	released = _round2(original_retention * release_pct / Decimal("100"))
	remaining = _round2(original_retention - released)
	return released, remaining


# ---------------------------------------------------------------------------
# Payment run selection scoring
# ---------------------------------------------------------------------------

def score_invoice_for_payment(
	outstanding: Decimal,
	days_overdue: int,
	discount_available: bool,
	discount_days_remaining: int,
	penalty_rate_per_month: Decimal,
) -> Decimal:
	"""
	Composite priority score for payment selection.
	Higher score = higher priority.

	Factors:
	- Overdue: +1 point per day overdue
	- Discount: +5 points if discount capture window open, scaled by days remaining
	- Penalty: +penalty_rate * outstanding / 1000
	"""
	score = Decimal("0")

	# Overdue penalty adds urgency
	if days_overdue > 0:
		score += Decimal(str(days_overdue))

	# Discount opportunity: strong incentive
	if discount_available and discount_days_remaining > 0:
		score += Decimal("5") + Decimal(str(max(0, 10 - discount_days_remaining)))

	# Penalty accrual rate (risk of further cost)
	if penalty_rate_per_month > Decimal("0") and days_overdue > 0:
		score += penalty_rate_per_month * outstanding / Decimal("1000")

	return _round2(score)


# ---------------------------------------------------------------------------
# Duplicate detection scoring (exact + fuzzy signals)
# ---------------------------------------------------------------------------

def calculate_duplicate_score(
	same_supplier: bool,
	same_amount: bool,
	amount_difference_pct: Decimal,
	same_invoice_ref: bool,
	invoice_date_delta_days: int,
	same_currency: bool,
) -> float:
	"""
	Heuristic duplicate probability in [0, 1].

	Signals and weights:
	- same_supplier: 0.20
	- same_invoice_ref: 0.40 (strongest signal — supplier controls this)
	- same_amount (exact): 0.20
	- amount within 1%: 0.10
	- invoice date within 7 days: 0.05
	- same_currency: 0.05
	"""
	score = 0.0

	if same_supplier:
		score += 0.20
	if same_invoice_ref:
		score += 0.40
	if same_amount:
		score += 0.20
	elif amount_difference_pct <= Decimal("1"):
		score += 0.10
	if abs(invoice_date_delta_days) <= 7:
		score += 0.05
	if same_currency:
		score += 0.05

	return min(1.0, score)
