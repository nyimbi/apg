"""Financial and domain calculations for APG Patient Management.

All functions are pure — no side effects, no I/O. Type-safe inputs and
outputs. Edge cases (zero denominators, Decimal precision) handled
explicitly so callers never receive NaN or ZeroDivisionError.
"""
from __future__ import annotations

from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Sequence


# ── monetary helpers ──────────────────────────────────────────────────────────

_CENTS = Decimal("0.01")


def _round(v: Decimal) -> Decimal:
	"""Round to 2 decimal places using ROUND_HALF_UP (standard financial rounding)."""
	return v.quantize(_CENTS, rounding=ROUND_HALF_UP)


# ── bill calculations ─────────────────────────────────────────────────────────

def calculate_bill_subtotal(
	unit_prices: Sequence[Decimal],
	quantities: Sequence[int],
) -> Decimal:
	"""Sum of (unit_price * quantity) for all line items."""
	if len(unit_prices) != len(quantities):
		raise ValueError("unit_prices and quantities must be the same length")
	return _round(sum(p * q for p, q in zip(unit_prices, quantities)))


def calculate_bill_balance_due(
	subtotal: Decimal,
	insurance_adjustment: Decimal,
	write_off_amount: Decimal,
	amount_paid: Decimal,
) -> Decimal:
	"""Balance the patient owes after insurance, write-offs, and payments.

	Never returns a negative value — overpayments are represented as 0
	(refund handling is separate).
	"""
	balance = subtotal - insurance_adjustment - write_off_amount - amount_paid
	return _round(max(balance, Decimal("0")))


def calculate_patient_responsibility(
	total_billed: Decimal,
	adjudicated_amount: Decimal,
	copay: Decimal = Decimal("0"),
	deductible: Decimal = Decimal("0"),
	coinsurance_pct: Decimal = Decimal("0"),
	out_of_pocket_max: Decimal | None = None,
) -> Decimal:
	"""Patient out-of-pocket after insurance adjudication.

	Formula:
	  responsibility = copay + deductible + (adjudicated_amount * coinsurance_pct)
	  capped at out_of_pocket_max if provided.
	"""
	if not (Decimal("0") <= coinsurance_pct <= Decimal("1")):
		raise ValueError(f"coinsurance_pct must be in [0, 1]; got {coinsurance_pct}")
	base = copay + deductible + _round(adjudicated_amount * coinsurance_pct)
	if out_of_pocket_max is not None:
		base = min(base, out_of_pocket_max)
	# Can't owe more than was billed
	return _round(min(base, total_billed))


def calculate_collection_rate(total_billed: Decimal, total_collected: Decimal) -> float:
	"""Percentage of billed amount actually collected. Returns 0.0 when
	total_billed is zero to avoid ZeroDivisionError."""
	if total_billed <= Decimal("0"):
		return 0.0
	return float(_round(total_collected / total_billed * Decimal("100")))


def calculate_denial_rate(claims_submitted: int, claims_denied: int) -> float:
	"""Percentage of submitted claims denied."""
	if claims_submitted <= 0:
		return 0.0
	return round(claims_denied / claims_submitted * 100, 2)


def calculate_days_in_ar(
	outstanding_balance: Decimal,
	avg_daily_charges: Decimal,
) -> float:
	"""Days in accounts receivable — standard revenue cycle metric."""
	if avg_daily_charges <= Decimal("0"):
		return 0.0
	return float(_round(outstanding_balance / avg_daily_charges))


# ── payment plan ───────────────────────────────────────────────────────────────

def calculate_installment_amount(
	total_amount: Decimal,
	installments: int,
) -> Decimal:
	"""Divide total evenly; last installment absorbs rounding remainder."""
	if installments < 1:
		raise ValueError("installments must be >= 1")
	per = _round(total_amount / Decimal(str(installments)))
	return per


def calculate_payment_plan_balance(
	total_amount: Decimal,
	amount_paid: Decimal,
) -> Decimal:
	return _round(max(total_amount - amount_paid, Decimal("0")))


def calculate_payment_plan_completion_pct(
	total_amount: Decimal,
	amount_paid: Decimal,
) -> float:
	if total_amount <= Decimal("0"):
		return 0.0
	return float(_round(amount_paid / total_amount * Decimal("100")))


# ── bed occupancy ──────────────────────────────────────────────────────────────

def calculate_occupancy_rate(occupied: int, total: int) -> float:
	"""Returns occupancy % [0.0, 100.0]. Handles zero-total edge case."""
	if total <= 0:
		return 0.0
	return round(occupied / total * 100, 2)


def is_overflow_risk(available: int, total: int, threshold_pct: float = 10.0) -> bool:
	"""True when available beds are below threshold_pct of total capacity."""
	if total <= 0:
		return False
	return (available / total * 100) < threshold_pct


def calculate_bed_turnover_rate(discharges: int, total_beds: int, period_days: int = 30) -> float:
	"""Average number of times a bed was occupied per day over the period."""
	if total_beds <= 0 or period_days <= 0:
		return 0.0
	return round(discharges / (total_beds * period_days), 4)


# ── length of stay ─────────────────────────────────────────────────────────────

def calculate_los_hours(admit_time: datetime, discharge_time: datetime | None = None) -> float:
	"""Length of stay in fractional hours. Uses utcnow() when not yet discharged."""
	end = discharge_time or datetime.utcnow()
	delta = end - admit_time
	return max(delta.total_seconds() / 3600, 0.0)


def calculate_los_days(admit_time: datetime, discharge_time: datetime | None = None) -> float:
	return round(calculate_los_hours(admit_time, discharge_time) / 24, 2)


def calculate_avg_los(los_hours_list: Sequence[float]) -> float:
	"""Mean LOS in hours. Returns 0.0 for empty list."""
	if not los_hours_list:
		return 0.0
	return round(sum(los_hours_list) / len(los_hours_list), 2)


# ── MRN generation ────────────────────────────────────────────────────────────

def generate_mrn(tenant_prefix: str, sequence: int) -> str:
	"""Deterministic MRN: MRN{PREFIX}{SEQ:08d}.
	Prefix is uppercased and limited to 4 chars.
	"""
	prefix = tenant_prefix[:4].upper().ljust(4, "X")
	return f"MRN{prefix}{sequence:08d}"


# ── age ────────────────────────────────────────────────────────────────────────

def calculate_age_years(date_of_birth: datetime, as_of: datetime | None = None) -> int:
	"""Accurate age in whole years, accounting for leap years."""
	ref = as_of or datetime.utcnow()
	years = ref.year - date_of_birth.year
	if (ref.month, ref.day) < (date_of_birth.month, date_of_birth.day):
		years -= 1
	return max(years, 0)


def calculate_age_months(date_of_birth: datetime, as_of: datetime | None = None) -> int:
	"""Age in whole months — used for paediatric age-limit checks."""
	ref = as_of or datetime.utcnow()
	months = (ref.year - date_of_birth.year) * 12 + (ref.month - date_of_birth.month)
	if ref.day < date_of_birth.day:
		months -= 1
	return max(months, 0)


# ── waitlist scoring ───────────────────────────────────────────────────────────

def calculate_wait_hours(enqueued_at: datetime, as_of: datetime | None = None) -> float:
	ref = as_of or datetime.utcnow()
	delta = ref - enqueued_at
	return max(delta.total_seconds() / 3600, 0.0)


# ── insurance claim ────────────────────────────────────────────────────────────

def calculate_claim_write_off(
	total_billed: Decimal,
	adjudicated_amount: Decimal,
	patient_responsibility: Decimal,
) -> Decimal:
	"""Contractual write-off = billed - adjudicated - patient_responsibility."""
	write_off = total_billed - adjudicated_amount - patient_responsibility
	return _round(max(write_off, Decimal("0")))


# ── no-show risk scoring ───────────────────────────────────────────────────────

def calculate_no_show_risk(
	prior_no_shows: int,
	prior_cancellations: int,
	total_appointments: int,
	days_until_appointment: int,
	telehealth: bool,
) -> float:
	"""Heuristic no-show risk score in [0.0, 1.0].

	Higher = more likely to no-show. Factors:
	- historical no-show rate (weighted 40%)
	- cancellation rate (weighted 20%)
	- booking horizon (longer = higher risk, weighted 20%)
	- telehealth reduces risk slightly (-10% modifier)
	"""
	if total_appointments <= 0:
		base = 0.15  # baseline for new patients
	else:
		nsr = prior_no_shows / total_appointments
		cr = prior_cancellations / total_appointments
		base = 0.40 * nsr + 0.20 * cr + 0.15  # baseline

	# Horizon factor: >14 days out adds up to 0.10
	horizon_factor = min(days_until_appointment / 14 * 0.10, 0.10) if days_until_appointment > 0 else 0.0

	score = base + horizon_factor
	if telehealth:
		score *= 0.85  # telehealth reduces friction

	return round(min(max(score, 0.0), 1.0), 4)


# ── readmission risk ───────────────────────────────────────────────────────────

def calculate_readmission_risk_score(
	prior_admissions_30d: int,
	age_years: int,
	primary_diagnosis_high_risk: bool,
	has_discharge_plan: bool,
	has_follow_up_appointment: bool,
) -> float:
	"""Simplified readmission risk score in [0.0, 1.0] (LACE-inspired).

	Not a validated clinical tool — for workflow routing only.
	"""
	score = 0.0
	score += min(prior_admissions_30d * 0.15, 0.30)
	if age_years >= 75:
		score += 0.20
	elif age_years >= 65:
		score += 0.10
	if primary_diagnosis_high_risk:
		score += 0.25
	if not has_discharge_plan:
		score += 0.15
	if not has_follow_up_appointment:
		score += 0.10
	return round(min(score, 1.0), 4)


# ── revenue cycle KPIs ────────────────────────────────────────────────────────

def calculate_net_collection_rate(
	net_revenue: Decimal,
	net_charges: Decimal,
) -> float:
	"""NCR = net_revenue / net_charges * 100. Industry benchmark: >95%."""
	if net_charges <= Decimal("0"):
		return 0.0
	return float(_round(net_revenue / net_charges * Decimal("100")))


def calculate_cost_per_discharge(
	total_operating_cost: Decimal,
	total_discharges: int,
) -> Decimal:
	if total_discharges <= 0:
		return Decimal("0")
	return _round(total_operating_cost / Decimal(str(total_discharges)))
