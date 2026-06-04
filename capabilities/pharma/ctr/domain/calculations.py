"""Clinical trials domain calculations.

All formulas used in enrollment projection, safety signal detection,
data quality scoring, and interim analysis power calculations.

All functions are pure (no side effects) and type-safe.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Enrollment projections
# ─────────────────────────────────────────────────────────────────────────────

def calculate_enrollment_rate(
	enrolled: int,
	days_active: int,
) -> float:
	"""Subjects per day across the entire trial.

	Args:
		enrolled: Number of subjects enrolled so far.
		days_active: Days since first patient first visit.

	Returns:
		Enrollment rate as subjects/day (0.0 if days_active <= 0).
	"""
	if days_active <= 0 or enrolled <= 0:
		return 0.0
	return enrolled / days_active


def project_enrollment_completion(
	enrolled: int,
	target: int,
	days_active: int,
	today: datetime | None = None,
) -> datetime | None:
	"""Project the date when target enrollment will be reached.

	Returns None if rate is zero (cannot project).
	"""
	rate = calculate_enrollment_rate(enrolled, days_active)
	if rate <= 0 or enrolled >= target:
		return None
	remaining = target - enrolled
	days_needed = math.ceil(remaining / rate)
	base = today or datetime.utcnow()
	return base + timedelta(days=days_needed)


def calculate_screen_failure_rate(screened: int, enrolled: int) -> float:
	"""Proportion of screened subjects who failed screening.

	Returns value in [0, 1].  Returns 0.0 if screened == 0.
	"""
	if screened <= 0:
		return 0.0
	failed = max(screened - enrolled, 0)
	return failed / screened


def calculate_dropout_rate(enrolled: int, withdrawn: int, lost: int) -> float:
	"""Proportion of enrolled subjects who discontinued.

	Returns value in [0, 1].  Returns 0.0 if enrolled == 0.
	"""
	if enrolled <= 0:
		return 0.0
	return min((withdrawn + lost) / enrolled, 1.0)


def calculate_site_performance_score(
	enrolled: int,
	target: int,
	days_active: int,
	open_queries: int,
	protocol_deviations: int,
) -> float:
	"""Composite site performance score in [0, 1].

	Weights:
	  - Enrollment progress:  40%
	  - Enrollment velocity:  30%
	  - Data quality:         20%
	  - Protocol adherence:   10%
	"""
	# Enrollment progress (capped at 1.0)
	progress = min(enrolled / target, 1.0) if target > 0 else 0.0

	# Enrollment velocity: compare actual to expected (uniform distribution)
	expected_per_day = target / max(days_active, 1)
	actual_per_day = enrolled / max(days_active, 1)
	velocity = min(actual_per_day / expected_per_day, 1.0) if expected_per_day > 0 else 0.0

	# Data quality: penalise open queries (assume max 10 queries is poor)
	data_quality = max(1.0 - (open_queries / 10.0), 0.0)

	# Protocol adherence: penalise deviations (assume max 5 deviations is poor)
	adherence = max(1.0 - (protocol_deviations / 5.0), 0.0)

	return (
		0.40 * progress
		+ 0.30 * velocity
		+ 0.20 * data_quality
		+ 0.10 * adherence
	)


# ─────────────────────────────────────────────────────────────────────────────
# Safety signal detection
# ─────────────────────────────────────────────────────────────────────────────

def calculate_ae_incidence_rate(
	ae_count: int,
	patient_days: float,
) -> float:
	"""AE incidence per 100 patient-days.

	Args:
		ae_count: Number of adverse events.
		patient_days: Total accumulated patient exposure in days.

	Returns:
		Rate per 100 patient-days.
	"""
	if patient_days <= 0:
		return 0.0
	return (ae_count / patient_days) * 100.0


def calculate_relative_risk(
	events_treatment: int,
	n_treatment: int,
	events_control: int,
	n_control: int,
) -> float | None:
	"""Relative risk (risk ratio) between treatment and control arms.

	Returns None if either denominator is zero or control rate is zero.
	"""
	if n_treatment <= 0 or n_control <= 0:
		return None
	rate_t = events_treatment / n_treatment
	rate_c = events_control / n_control
	if rate_c == 0:
		return None
	return rate_t / rate_c


def calculate_number_needed_to_treat(
	absolute_risk_reduction: float,
) -> float | None:
	"""NNT = 1 / ARR.

	Returns None if ARR is zero or negative.
	"""
	if absolute_risk_reduction <= 0:
		return None
	return 1.0 / absolute_risk_reduction


def assess_safety_signal_strength(
	observed_ae_count: int,
	expected_ae_count: float,
) -> dict[str, Any]:
	"""Observed-to-Expected (O/E) ratio as a simple disproportionality metric.

	Returns a dict with ratio, signal_level, and interpretation.
	"""
	if expected_ae_count <= 0:
		return {"ratio": None, "signal_level": "insufficient_data", "interpretation": "no expected count"}
	ratio = observed_ae_count / expected_ae_count
	if ratio >= 3.0:
		level = "strong"
		interpretation = "potential safety signal — expedited review warranted"
	elif ratio >= 2.0:
		level = "moderate"
		interpretation = "elevated O/E ratio — enhanced monitoring recommended"
	elif ratio >= 1.5:
		level = "weak"
		interpretation = "slightly elevated — routine monitoring sufficient"
	else:
		level = "none"
		interpretation = "within expected range"
	return {"ratio": round(ratio, 3), "signal_level": level, "interpretation": interpretation}


# ─────────────────────────────────────────────────────────────────────────────
# Reporting timelines
# ─────────────────────────────────────────────────────────────────────────────

def hours_since(event_time: datetime, reference: datetime | None = None) -> float:
	"""Hours elapsed since event_time."""
	ref = reference or datetime.utcnow()
	delta = ref - event_time
	return delta.total_seconds() / 3600.0


def days_since(event_time: datetime, reference: datetime | None = None) -> float:
	"""Days elapsed since event_time."""
	return hours_since(event_time, reference) / 24.0


def calculate_susar_reporting_compliance(
	onset_date: datetime,
	reported_to_authority_date: datetime | None,
	deadline_days: int = 15,
) -> dict[str, Any]:
	"""Assess SUSAR reporting compliance.

	Returns dict with days_elapsed, deadline_days, compliant, days_overdue.
	"""
	if reported_to_authority_date is None:
		elapsed = days_since(onset_date)
		return {
			"days_elapsed": round(elapsed, 1),
			"deadline_days": deadline_days,
			"compliant": elapsed <= deadline_days,
			"days_overdue": round(max(elapsed - deadline_days, 0), 1),
			"pending": True,
		}
	elapsed = days_since(onset_date, reported_to_authority_date)
	return {
		"days_elapsed": round(elapsed, 1),
		"deadline_days": deadline_days,
		"compliant": elapsed <= deadline_days,
		"days_overdue": round(max(elapsed - deadline_days, 0), 1),
		"pending": False,
	}


# ─────────────────────────────────────────────────────────────────────────────
# Data quality metrics
# ─────────────────────────────────────────────────────────────────────────────

def calculate_query_resolution_rate(resolved: int, total: int) -> float:
	"""Proportion of data queries resolved (0–1)."""
	if total <= 0:
		return 1.0  # No queries → perfect
	return min(resolved / total, 1.0)


def calculate_missing_data_rate(
	expected_data_points: int,
	missing_data_points: int,
) -> float:
	"""Proportion of expected data points that are missing (0–1)."""
	if expected_data_points <= 0:
		return 0.0
	return min(missing_data_points / expected_data_points, 1.0)


def calculate_data_quality_score(
	query_resolution_rate: float,
	missing_data_rate: float,
	sdv_rate: float,
	protocol_deviation_rate: float,
) -> float:
	"""Composite data quality score in [0, 1].

	Weights:
	  - Query resolution:       30%
	  - Missing data (inverse): 30%
	  - SDV completion:         20%
	  - Protocol adherence:     20%
	"""
	return (
		0.30 * query_resolution_rate
		+ 0.30 * (1.0 - missing_data_rate)
		+ 0.20 * sdv_rate
		+ 0.20 * (1.0 - protocol_deviation_rate)
	)


# ─────────────────────────────────────────────────────────────────────────────
# Statistical power (simplified for interim analyses)
# ─────────────────────────────────────────────────────────────────────────────

def _norm_inv_approx(p: float) -> float:
	"""Rational approximation to the inverse normal CDF (Abramowitz & Stegun 26.2.17).

	Accurate to ~3 decimal places for p in (0.001, 0.999).
	"""
	if p <= 0 or p >= 1:
		raise ValueError(f"p must be in (0, 1), got {p}")
	q = p if p < 0.5 else 1 - p
	t = math.sqrt(-2 * math.log(q))
	c = [2.515517, 0.802853, 0.010328]
	d = [1.432788, 0.189269, 0.001308]
	num = c[0] + c[1] * t + c[2] * t * t
	den = 1 + d[0] * t + d[1] * t * t + d[2] * t * t * t
	z = t - num / den
	return z if p < 0.5 else -z


def calculate_sample_size_two_proportions(
	p1: float,
	p2: float,
	alpha: float = 0.05,
	power: float = 0.80,
	two_sided: bool = True,
) -> int:
	"""Minimum sample size per arm for comparing two proportions.

	Uses the standard formula: n = (z_alpha/2 + z_beta)^2 * (p1(1-p1) + p2(1-p2)) / (p1-p2)^2

	Args:
		p1: Event rate in treatment arm.
		p2: Event rate in control arm.
		alpha: Type I error rate.
		power: Statistical power (1 - beta).
		two_sided: Whether to use a two-sided test.

	Returns:
		Required sample size per arm (rounded up).
	"""
	if p1 == p2:
		raise ValueError("p1 and p2 must differ")
	alpha_adj = alpha / 2 if two_sided else alpha
	z_alpha = _norm_inv_approx(1 - alpha_adj)
	z_beta = _norm_inv_approx(power)
	effect = (p1 - p2) ** 2
	variance = p1 * (1 - p1) + p2 * (1 - p2)
	n = ((z_alpha + z_beta) ** 2 * variance) / effect
	return math.ceil(n)


def calculate_study_power(
	n_per_arm: int,
	p1: float,
	p2: float,
	alpha: float = 0.05,
	two_sided: bool = True,
) -> float:
	"""Achieved power for a two-proportion study given the current sample size.

	Returns power in [0, 1].
	"""
	if p1 == p2 or n_per_arm <= 0:
		return 0.0
	alpha_adj = alpha / 2 if two_sided else alpha
	z_alpha = _norm_inv_approx(1 - alpha_adj)
	effect = abs(p1 - p2)
	se = math.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n_per_arm)
	if se == 0:
		return 1.0
	z_beta = effect / se - z_alpha
	# Power ≈ Φ(z_beta) — approximate using logistic sigmoid (good enough for display)
	return 1 / (1 + math.exp(-1.7 * z_beta))


def calculate_interim_analysis_oc_spending(
	information_fraction: float,
	alpha: float = 0.05,
	spending_function: str = "obrien_fleming",
) -> float:
	"""Alpha spending at a given information fraction for group sequential designs.

	Supported functions:
	  - "obrien_fleming": Conservative early stopping
	  - "pocock":         Uniform spending
	  - "linear":         Simple linear alpha spending

	Args:
		information_fraction: Proportion of total information accrued (0–1).
		alpha: Overall type I error rate.
		spending_function: Name of the alpha spending function.

	Returns:
		Cumulative alpha spent at this interim.
	"""
	t = max(0.0, min(information_fraction, 1.0))
	if spending_function == "obrien_fleming":
		# Lan-DeMets O'Brien-Fleming approximation
		if t <= 0:
			return 0.0
		z_alpha = _norm_inv_approx(1 - alpha / 2)
		return 2 * (1 - 1 / (1 + math.exp(z_alpha / math.sqrt(t))))
	elif spending_function == "pocock":
		return alpha * math.log(1 + (math.e - 1) * t)
	else:
		# Linear
		return alpha * t


# ─────────────────────────────────────────────────────────────────────────────
# TMF completeness
# ─────────────────────────────────────────────────────────────────────────────

def calculate_tmf_completeness(
	total_expected: int,
	total_filed: int,
	overdue: int,
) -> dict[str, Any]:
	"""TMF completeness metrics.

	Returns completeness_rate, overdue_rate, health (green/amber/red).
	"""
	if total_expected <= 0:
		return {"completeness_rate": 1.0, "overdue_rate": 0.0, "health": "green"}
	completeness = min(total_filed / total_expected, 1.0)
	overdue_rate = overdue / total_expected
	if completeness >= 0.95 and overdue_rate < 0.05:
		health = "green"
	elif completeness >= 0.80 and overdue_rate < 0.15:
		health = "amber"
	else:
		health = "red"
	return {
		"completeness_rate": round(completeness, 4),
		"overdue_rate": round(overdue_rate, 4),
		"health": health,
		"total_expected": total_expected,
		"total_filed": total_filed,
		"overdue": overdue,
	}
