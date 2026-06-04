"""
Employee Data Management — financial and domain calculations.

All functions are pure (no I/O, no side effects).

© 2025 Datacraft. All rights reserved.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from typing import Any


_CENT = Decimal("0.01")


def _round(v: Decimal) -> Decimal:
	return v.quantize(_CENT, rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Compensation calculations
# ---------------------------------------------------------------------------

def annual_to_monthly(annual: Decimal) -> Decimal:
	"""Convert annual salary to monthly."""
	return _round(annual / 12)


def monthly_to_annual(monthly: Decimal) -> Decimal:
	return _round(monthly * 12)


def daily_rate(annual: Decimal, working_days_per_year: int = 260) -> Decimal:
	"""Standard daily rate for leave encashment, pro-rata, etc."""
	if working_days_per_year <= 0:
		raise ValueError("working_days_per_year must be positive")
	return _round(annual / working_days_per_year)


def hourly_rate(annual: Decimal, hours_per_year: int = 2080) -> Decimal:
	"""Standard hourly rate (52 weeks × 40 hours)."""
	if hours_per_year <= 0:
		raise ValueError("hours_per_year must be positive")
	return _round(annual / hours_per_year)


def pro_rata_salary(
	annual: Decimal,
	start: date,
	end: date,
	year_start: date | None = None,
) -> Decimal:
	"""
	Pro-rata salary for a partial year period [start, end] (inclusive).

	year_start defaults to Jan 1 of start.year.
	"""
	ys = year_start or date(start.year, 1, 1)
	ye = date(ys.year, 12, 31)
	total_days = (ye - ys).days + 1
	period_days = (end - start).days + 1
	if total_days <= 0:
		return Decimal("0.00")
	return _round(annual * Decimal(period_days) / Decimal(total_days))


def severance_pay(
	annual_salary: Decimal,
	service_years: float,
	country_code: str = "KE",
) -> Decimal:
	"""
	Statutory severance estimate.

	Kenya Employment Act 2007: 15 days gross pay per year of service.
	Default for unknown jurisdictions: 1 month per year, capped at 12 months.
	"""
	if service_years <= 0:
		return Decimal("0.00")

	if country_code.upper() == "KE":
		daily = daily_rate(annual_salary)
		return _round(daily * 15 * Decimal(str(service_years)))

	# Generic: 1 month per year, max 12 months
	months = min(service_years, 12)
	return _round(annual_to_monthly(annual_salary) * Decimal(str(months)))


def notice_pay(
	daily_rate_val: Decimal,
	notice_days_owed: int,
) -> Decimal:
	"""Pay in lieu of notice."""
	return _round(daily_rate_val * Decimal(notice_days_owed))


def leave_encashment(
	daily_rate_val: Decimal,
	accrued_days: float,
) -> Decimal:
	"""Cash value of accrued but untaken leave."""
	return _round(daily_rate_val * Decimal(str(accrued_days)))


# ---------------------------------------------------------------------------
# Headcount & attrition
# ---------------------------------------------------------------------------

def headcount_variance(
	authorized: int,
	actual: int,
) -> dict[str, Any]:
	"""Return variance between authorized and actual headcount."""
	variance = actual - authorized
	utilization = round(actual / authorized * 100, 1) if authorized else 0.0
	return {
		"authorized": authorized,
		"actual": actual,
		"variance": variance,
		"utilization_pct": utilization,
		"status": "over" if variance > 0 else ("under" if variance < 0 else "met"),
	}


def rolling_attrition(
	terminations_per_month: list[int],
	headcount_per_month: list[int],
) -> float:
	"""
	12-month rolling attrition rate.

	ATR = (sum of monthly terminations / average monthly headcount) × 100
	"""
	if not terminations_per_month or not headcount_per_month:
		return 0.0
	n = min(len(terminations_per_month), len(headcount_per_month))
	total_terms = sum(terminations_per_month[:n])
	avg_hc = sum(headcount_per_month[:n]) / n
	if avg_hc == 0:
		return 0.0
	return round(total_terms / avg_hc * 100, 2)


# ---------------------------------------------------------------------------
# Salary benchmarking
# ---------------------------------------------------------------------------

def market_premium_pct(
	employee_salary: Decimal,
	market_median: Decimal,
) -> float:
	"""
	How far employee salary is above/below market median.

	Positive = above market; negative = below market.
	"""
	if market_median <= 0:
		return 0.0
	return round(float((employee_salary - market_median) / market_median * 100), 2)


def salary_range_penetration(
	salary: Decimal,
	grade_min: Decimal,
	grade_max: Decimal,
) -> float:
	"""
	Range penetration = (salary - min) / (max - min).
	0.0 = at minimum; 1.0 = at maximum.
	"""
	span = grade_max - grade_min
	if span <= 0:
		return 0.0
	return round(float((salary - grade_min) / span), 4)


# ---------------------------------------------------------------------------
# Benefit cost
# ---------------------------------------------------------------------------

def total_benefit_cost_per_employee(
	enrollments: list[dict[str, Any]],
) -> Decimal:
	"""
	Sum employer_contribution across active benefit enrollments.

	Each item in enrollments must have keys: status, employer_contribution.
	"""
	total = Decimal("0.00")
	for e in enrollments:
		if e.get("status") in ("active", "enrolled"):
			total += Decimal(str(e.get("employer_contribution", 0)))
	return _round(total)


# ---------------------------------------------------------------------------
# Succession readiness
# ---------------------------------------------------------------------------

def succession_readiness_score(
	performance_rating_num: int,   # 1-5
	service_years: float,
	skill_match_pct: float,        # 0-100
	has_leadership_training: bool,
) -> float:
	"""
	Composite succession readiness score in [0, 100].
	"""
	# Performance contributes up to 30 points (scale 1-5 → 6-30)
	perf_pts = performance_rating_num * 6

	# Service contributes up to 20 points (capped at 10 years)
	service_pts = min(service_years, 10) * 2

	# Skill match contributes up to 40 points
	skill_pts = skill_match_pct * 0.4

	# Leadership training adds 10 points
	leadership_pts = 10 if has_leadership_training else 0

	total = perf_pts + service_pts + skill_pts + leadership_pts
	return round(min(total, 100), 1)
