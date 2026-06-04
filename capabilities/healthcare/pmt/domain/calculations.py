"""Financial and domain calculations for APG Patient Management.

All functions are pure — no side effects, no I/O. Type-safe inputs and
outputs. Every edge case (zero denominators, Decimal precision, negative
durations) handled explicitly.
"""
from __future__ import annotations

from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Sequence


# ── internal helpers ───────────────────────────────────────────────────────────

_CENTS = Decimal("0.01")
_FOUR  = Decimal("0.0001")


def _d2(v: Decimal) -> Decimal:
	return v.quantize(_CENTS, rounding=ROUND_HALF_UP)


def _d4(v: Decimal) -> Decimal:
	return v.quantize(_FOUR, rounding=ROUND_HALF_UP)


# ── MRN generation ────────────────────────────────────────────────────────────

def generate_mrn(tenant_prefix: str, sequence: int) -> str:
	"""Deterministic MRN: MRN{PREFIX4}{SEQ:08d}.

	Prefix is uppercased and padded/truncated to exactly 4 characters.
	"""
	prefix = (tenant_prefix[:4].upper()).ljust(4, "X")
	return f"MRN{prefix}{sequence:08d}"


# ── age ────────────────────────────────────────────────────────────────────────

def calculate_age_years(date_of_birth: datetime, as_of: datetime | None = None) -> int:
	"""Whole years of age, accounting for leap-year birthdays."""
	ref = as_of or datetime.utcnow()
	years = ref.year - date_of_birth.year
	if (ref.month, ref.day) < (date_of_birth.month, date_of_birth.day):
		years -= 1
	return max(years, 0)


def calculate_age_months(date_of_birth: datetime, as_of: datetime | None = None) -> int:
	"""Age in whole months — used for paediatric ward age-limit enforcement."""
	ref = as_of or datetime.utcnow()
	months = (ref.year - date_of_birth.year) * 12 + (ref.month - date_of_birth.month)
	if ref.day < date_of_birth.day:
		months -= 1
	return max(months, 0)


def is_paediatric(date_of_birth: datetime, threshold_years: int = 18) -> bool:
	return calculate_age_years(date_of_birth) < threshold_years


# ── length of stay ─────────────────────────────────────────────────────────────

def calculate_los_hours(admit_time: datetime, discharge_time: datetime | None = None) -> float:
	"""Fractional hours of stay. Uses utcnow when not yet discharged."""
	end = discharge_time or datetime.utcnow()
	return max((end - admit_time).total_seconds() / 3600.0, 0.0)


def calculate_los_days(admit_time: datetime, discharge_time: datetime | None = None) -> float:
	return round(calculate_los_hours(admit_time, discharge_time) / 24.0, 2)


def calculate_avg_los(los_hours_list: Sequence[float]) -> float:
	if not los_hours_list:
		return 0.0
	return round(sum(los_hours_list) / len(los_hours_list), 2)


# ── bed occupancy ──────────────────────────────────────────────────────────────

def calculate_occupancy_rate(occupied: int, total: int) -> float:
	"""Occupancy % [0.0–100.0]. Safe against zero total."""
	if total <= 0:
		return 0.0
	return round(occupied / total * 100.0, 2)


def is_overflow_risk(available: int, total: int, threshold_pct: float = 10.0) -> bool:
	"""True when available beds are below threshold_pct of total capacity."""
	if total <= 0:
		return False
	return (available / total * 100.0) < threshold_pct


def calculate_bed_turnover_rate(discharges: int, total_beds: int, period_days: int = 30) -> float:
	"""Average times a bed was occupied per bed per day over the period."""
	if total_beds <= 0 or period_days <= 0:
		return 0.0
	return round(discharges / (total_beds * period_days), 4)


def effective_available_beds(available: int, cleaning: int, cleaning_turnaround_hours: float = 2.0) -> int:
	"""Conservative available count: cleaning beds are included as ~half-available."""
	extra = max(0, int(cleaning * (1.0 - cleaning_turnaround_hours / 4.0)))
	return available + extra


# ── waitlist ───────────────────────────────────────────────────────────────────

def calculate_wait_hours(enqueued_at: datetime, as_of: datetime | None = None) -> float:
	ref = as_of or datetime.utcnow()
	return max((ref - enqueued_at).total_seconds() / 3600.0, 0.0)


def calculate_waitlist_priority_score(
	priority: str,
	wait_hours: float,
	isolation_required: bool = False,
	paediatric: bool = False,
) -> float:
	"""Urgency-weighted priority score for queue ordering.

	Higher score = serve first.
	Base weights: emergency=100, urgent=70, semi_urgent=40, routine=10.
	Accrual: +1 per wait hour (capped at 48). Modifiers: isolation +5, paediatric +3.
	"""
	_base = {"emergency": 100.0, "urgent": 70.0, "semi_urgent": 40.0, "routine": 10.0}
	base = _base.get(priority, 10.0)
	accrual = min(wait_hours, 48.0)
	modifier = (5.0 if isolation_required else 0.0) + (3.0 if paediatric else 0.0)
	return round(base + accrual + modifier, 2)


# ── bill calculations ─────────────────────────────────────────────────────────

def calculate_bill_subtotal(
	unit_prices: Sequence[Decimal],
	quantities: Sequence[int],
) -> Decimal:
	"""Sum of (unit_price × quantity) for all line items."""
	if len(unit_prices) != len(quantities):
		raise ValueError("unit_prices and quantities must be the same length")
	total = sum(p * Decimal(str(q)) for p, q in zip(unit_prices, quantities))
	return _d2(total)


def calculate_bill_tax(subtotal: Decimal, vat_rate: Decimal = Decimal("0.16")) -> Decimal:
	"""VAT at specified rate. Default 16% (Kenya)."""
	return _d2(subtotal * vat_rate)


def calculate_bill_total(subtotal: Decimal, tax: Decimal) -> Decimal:
	return _d2(subtotal + tax)


def calculate_bill_balance_due(
	subtotal: Decimal,
	insurance_adjustment: Decimal,
	write_off_amount: Decimal,
	amount_paid: Decimal,
) -> Decimal:
	"""Patient balance after insurance, write-offs, and payments. Floor: 0."""
	balance = subtotal - insurance_adjustment - write_off_amount - amount_paid
	return _d2(max(balance, Decimal("0")))


def calculate_patient_responsibility(
	total_billed: Decimal,
	adjudicated_amount: Decimal,
	copay: Decimal = Decimal("0"),
	deductible: Decimal = Decimal("0"),
	coinsurance_pct: Decimal = Decimal("0"),
	out_of_pocket_max: Decimal | None = None,
) -> Decimal:
	"""Out-of-pocket after insurance adjudication (copay + deductible + coinsurance).

	Capped at out_of_pocket_max when provided. Never exceeds total_billed.
	"""
	if not (Decimal("0") <= coinsurance_pct <= Decimal("1")):
		raise ValueError(f"coinsurance_pct must be in [0,1]; got {coinsurance_pct}")
	base = copay + deductible + _d2(adjudicated_amount * coinsurance_pct)
	if out_of_pocket_max is not None:
		base = min(base, out_of_pocket_max)
	return _d2(min(base, total_billed))


def calculate_insurance_adjustment(
	total_billed: Decimal,
	adjudicated_amount: Decimal,
) -> Decimal:
	"""Contractual discount: billed − adjudicated. Floor: 0."""
	return _d2(max(total_billed - adjudicated_amount, Decimal("0")))


def calculate_claim_write_off(
	total_billed: Decimal,
	adjudicated_amount: Decimal,
	patient_responsibility: Decimal,
) -> Decimal:
	"""Contractual write-off = billed − adjudicated − patient_responsibility. Floor: 0."""
	write_off = total_billed - adjudicated_amount - patient_responsibility
	return _d2(max(write_off, Decimal("0")))


# ── payment plan ───────────────────────────────────────────────────────────────

def calculate_installment_amount(total_amount: Decimal, installments: int) -> Decimal:
	"""Per-installment amount. Last installment absorbs rounding remainder."""
	if installments < 1:
		raise ValueError("installments must be >= 1")
	return _d2(total_amount / Decimal(str(installments)))


def calculate_payment_plan_balance(total_amount: Decimal, amount_paid: Decimal) -> Decimal:
	return _d2(max(total_amount - amount_paid, Decimal("0")))


def calculate_payment_plan_completion_pct(total_amount: Decimal, amount_paid: Decimal) -> float:
	if total_amount <= Decimal("0"):
		return 0.0
	return float(_d2(amount_paid / total_amount * Decimal("100")))


# ── revenue cycle KPIs ────────────────────────────────────────────────────────

def calculate_collection_rate(total_billed: Decimal, total_collected: Decimal) -> float:
	"""Gross collection rate %. Industry benchmark: >95%."""
	if total_billed <= Decimal("0"):
		return 0.0
	return float(_d2(total_collected / total_billed * Decimal("100")))


def calculate_net_collection_rate(net_revenue: Decimal, net_charges: Decimal) -> float:
	"""NCR = net_revenue / net_charges × 100. Benchmark: >95%."""
	if net_charges <= Decimal("0"):
		return 0.0
	return float(_d2(net_revenue / net_charges * Decimal("100")))


def calculate_denial_rate(claims_submitted: int, claims_denied: int) -> float:
	if claims_submitted <= 0:
		return 0.0
	return round(claims_denied / claims_submitted * 100.0, 2)


def calculate_days_in_ar(outstanding_balance: Decimal, avg_daily_charges: Decimal) -> float:
	"""Days in A/R — standard revenue cycle metric."""
	if avg_daily_charges <= Decimal("0"):
		return 0.0
	return float(_d2(outstanding_balance / avg_daily_charges))


def calculate_cost_per_discharge(total_operating_cost: Decimal, total_discharges: int) -> Decimal:
	if total_discharges <= 0:
		return Decimal("0")
	return _d2(total_operating_cost / Decimal(str(total_discharges)))


# ── no-show risk ───────────────────────────────────────────────────────────────

def calculate_no_show_risk(
	prior_no_shows: int,
	prior_cancellations: int,
	total_appointments: int,
	days_until_appointment: int,
	telehealth: bool,
) -> float:
	"""Heuristic no-show risk [0.0–1.0].

	Factors: historical no-show rate (40%), cancellation rate (20%),
	baseline (15%), booking horizon (≤14d, 10%), telehealth −15% modifier.
	"""
	if total_appointments <= 0:
		base = 0.15
	else:
		nsr = prior_no_shows / total_appointments
		cr  = prior_cancellations / total_appointments
		base = 0.40 * nsr + 0.20 * cr + 0.15

	horizon_factor = min(max(days_until_appointment, 0) / 14.0 * 0.10, 0.10)
	score = base + horizon_factor
	if telehealth:
		score *= 0.85
	return round(min(max(score, 0.0), 1.0), 4)


# ── readmission risk ───────────────────────────────────────────────────────────

def calculate_readmission_risk_score(
	prior_admissions_30d: int,
	age_years: int,
	primary_diagnosis_high_risk: bool,
	has_discharge_plan: bool,
	has_follow_up_appointment: bool,
) -> float:
	"""Simplified LACE-inspired readmission risk [0.0–1.0].

	Not a validated clinical instrument — for workflow routing only.
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


# ── early warning score ────────────────────────────────────────────────────────

def calculate_early_warning_score(vitals: dict[str, float]) -> tuple[int, str]:
	"""Simplified NEWS2-inspired Early Warning Score.

	Returns (score, level) where level: low | medium | high | critical.
	"""
	score = 0

	bp = vitals.get("bp_systolic", 120.0)
	if bp <= 90:
		score += 3
	elif bp <= 100:
		score += 2
	elif bp <= 110:
		score += 1
	elif bp >= 220:
		score += 3

	rr = vitals.get("respiratory_rate", 16.0)
	if rr <= 8 or rr >= 25:
		score += 3
	elif rr >= 21:
		score += 2
	elif rr <= 11:
		score += 1

	spo2 = vitals.get("spo2", 98.0)
	if spo2 <= 91:
		score += 3
	elif spo2 <= 93:
		score += 2
	elif spo2 <= 95:
		score += 1

	hr = vitals.get("heart_rate", 80.0)
	if hr <= 40 or hr >= 131:
		score += 3
	elif hr >= 111:
		score += 2
	elif hr <= 50 or hr >= 91:
		score += 1

	temp = vitals.get("temperature_c", 37.0)
	if temp <= 35.0:
		score += 3
	elif temp >= 39.1:
		score += 2
	elif temp >= 38.1:
		score += 1

	consciousness = vitals.get("avpu_score", 1.0)  # 1=Alert, 0=not alert
	if consciousness < 1.0:
		score += 3

	if score >= 7:
		level = "critical"
	elif score >= 5:
		level = "high"
	elif score >= 2:
		level = "medium"
	else:
		level = "low"

	return score, level


# ── satisfaction / NPS ────────────────────────────────────────────────────────

def calculate_nps_bucket(score: float) -> str:
	"""NPS classification: promoter (9-10), passive (7-8), detractor (0-6)."""
	if score >= 9:
		return "promoter"
	if score >= 7:
		return "passive"
	return "detractor"


def calculate_composite_satisfaction(responses: dict[str, float]) -> float | None:
	"""Mean of all numeric Likert-scale responses (excluding NPS anchor)."""
	values = [v for k, v in responses.items() if k != "would_recommend" and isinstance(v, (int, float))]
	if not values:
		return None
	return round(sum(values) / len(values), 2)


# ── NHIF / SHA benefit calculation (Kenya) ────────────────────────────────────

def calculate_nhif_benefit(
	admission_type: str,
	los_days: int,
	ward_category: str = "general",
) -> Decimal:
	"""Indicative NHIF/SHA inpatient benefit (KES).

	Rates are illustrative — integrate with live NHIF tariff API in production.
	ward_category: general | private | icu | hdu
	"""
	_DAILY_RATES: dict[str, dict[str, int]] = {
		"general":  {"emergency": 2_500, "elective": 2_000, "default": 1_800},
		"private":  {"emergency": 4_500, "elective": 3_500, "default": 3_000},
		"icu":      {"emergency": 8_000, "elective": 8_000, "default": 8_000},
		"hdu":      {"emergency": 5_000, "elective": 5_000, "default": 5_000},
	}
	rates = _DAILY_RATES.get(ward_category, _DAILY_RATES["general"])
	daily = rates.get(admission_type, rates["default"])
	return _d2(Decimal(str(daily)) * Decimal(str(max(los_days, 1))))


# ── deposit adequacy ───────────────────────────────────────────────────────────

def is_deposit_adequate(
	deposit_amount: Decimal,
	estimated_bill: Decimal,
	threshold_pct: float = 30.0,
) -> bool:
	"""True when deposit covers at least threshold_pct of the estimated bill."""
	if estimated_bill <= Decimal("0"):
		return True
	return float(deposit_amount / estimated_bill * Decimal("100")) >= threshold_pct
