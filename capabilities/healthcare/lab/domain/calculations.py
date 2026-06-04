"""Domain calculations for the APG Laboratory Information System.

All functions are pure — no I/O, no side effects.
Inputs and outputs are fully typed.
"""

from __future__ import annotations

import math
import statistics
from datetime import datetime
from typing import Sequence

from ..models import (
	AbnormalFlag,
	CollectionPriority,
	QCStatus,
)


# ── Reference range evaluation ────────────────────────────────────────────────

def classify_numeric_result(
	value: float,
	low: float | None,
	high: float | None,
	critical_low: float | None = None,
	critical_high: float | None = None,
) -> tuple[AbnormalFlag | None, bool]:
	"""Classify a numeric result against reference and critical limits.

	Returns
	-------
	(flag, is_critical)
	    flag        — AbnormalFlag or None if within normal limits
	    is_critical — True when value crosses a critical threshold
	"""
	# Critical high
	if critical_high is not None and value >= critical_high:
		return AbnormalFlag.CRITICAL_HIGH, True
	# Critical low
	if critical_low is not None and value <= critical_low:
		return AbnormalFlag.CRITICAL_LOW, True
	# Very high (> 1.5× upper limit, no explicit critical_high)
	if high is not None and value > high:
		if critical_high is None and value > high * 1.5:
			return AbnormalFlag.VERY_HIGH, True
		return AbnormalFlag.HIGH, False
	# Very low (< 0.5× lower limit, no explicit critical_low)
	if low is not None and value < low:
		if critical_low is None and value < low * 0.5:
			return AbnormalFlag.VERY_LOW, True
		return AbnormalFlag.LOW, False
	return None, False


def is_within_reference_range(
	value: float,
	low: float | None,
	high: float | None,
) -> bool:
	"""Return True when value falls within [low, high] (inclusive)."""
	if low is not None and value < low:
		return False
	if high is not None and value > high:
		return False
	return True


# ── Delta check ───────────────────────────────────────────────────────────────

def delta_check(
	current: float,
	previous: float,
	delta_pct_threshold: float = 25.0,
) -> bool:
	"""Return True when the percent change from previous exceeds threshold.

	Delta check flags analytically implausible acute changes.
	Standard threshold: 25% for most chemistry analytes.
	"""
	if previous == 0:
		return current != 0
	pct_change = abs((current - previous) / previous) * 100
	return pct_change > delta_pct_threshold


# ── Westgard multi-rule evaluation ────────────────────────────────────────────

def calculate_z_score(measured: float, target: float, sd: float) -> float:
	"""Z-score = (measured - target) / SD."""
	if sd == 0:
		return 0.0
	return round((measured - target) / sd, 4)


def calculate_cv_percent(measured: float, sd: float) -> float:
	"""Coefficient of variation as a percentage."""
	if measured == 0:
		return 0.0
	return round((sd / measured) * 100, 2)


def evaluate_westgard_rules(
	z_scores: list[float],
) -> tuple[list[str], QCStatus]:
	"""Apply Westgard multi-rules to a run history of z-scores.

	The list is ordered oldest-to-newest; the most recent z-score is last.

	Rules evaluated
	---------------
	1-2s  : warning  — 1 point outside ±2 SD
	1-3s  : reject   — 1 point outside ±3 SD
	2-2s  : reject   — 2 consecutive points on same side of ±2 SD
	R-4s  : reject   — 2 consecutive points, range > 4 SD
	4-1s  : reject   — 4 consecutive points outside ±1 SD on same side
	10x   : reject   — 10 consecutive points on same side of mean

	Returns (violation_names, qc_status).
	"""
	if not z_scores:
		return [], QCStatus.PENDING_REVIEW

	violations: list[str] = []
	latest = z_scores[-1]

	# 1-3s: immediate reject
	if abs(latest) > 3.0:
		violations.append("1-3s")

	# 1-2s: warning (informational, does not reject by itself)
	if abs(latest) > 2.0:
		violations.append("1-2s")

	if len(z_scores) >= 2:
		prev = z_scores[-2]
		# R-4s: range between consecutive points > 4 SD
		if abs(latest - prev) > 4.0:
			violations.append("R-4s")
		# 2-2s: two consecutive outside ±2 SD on same side
		if abs(latest) > 2.0 and abs(prev) > 2.0:
			if (latest > 0 and prev > 0) or (latest < 0 and prev < 0):
				violations.append("2-2s")

	if len(z_scores) >= 4:
		last4 = z_scores[-4:]
		# 4-1s: four consecutive points beyond ±1 SD on same side
		all_pos = all(z > 1.0 for z in last4)
		all_neg = all(z < -1.0 for z in last4)
		if all_pos or all_neg:
			violations.append("4-1s")

	if len(z_scores) >= 10:
		last10 = z_scores[-10:]
		# 10x: ten consecutive points on same side of mean
		all_above = all(z > 0 for z in last10)
		all_below = all(z < 0 for z in last10)
		if all_above or all_below:
			violations.append("10x")

	# Determine status
	reject_rules = {"1-3s", "2-2s", "R-4s", "4-1s", "10x"}
	if any(r in reject_rules for r in violations):
		status = QCStatus.FAILED
	elif violations:  # only warnings
		status = QCStatus.PENDING_REVIEW
	else:
		status = QCStatus.PASSED

	return violations, status


# ── TAT calculations ──────────────────────────────────────────────────────────

def calculate_tat_minutes(ordered_at: datetime, completed_at: datetime) -> float:
	"""Return elapsed minutes between order placement and result completion."""
	delta = completed_at - ordered_at
	return max(0.0, delta.total_seconds() / 60)


def calculate_tat_metrics(
	tat_values: Sequence[float],
	stat_tat_values: Sequence[float],
	target_tat: float = 120.0,
) -> dict[str, float | int | None]:
	"""Compute descriptive TAT statistics for a cohort.

	Parameters
	----------
	tat_values      : all completed TAT values in minutes
	stat_tat_values : STAT-only TAT values in minutes
	target_tat      : on-time threshold in minutes (default 2 h)
	"""
	total = len(tat_values)
	if total == 0:
		return {
			"total_completed": 0,
			"median_tat_minutes": None,
			"p90_tat_minutes": None,
			"stat_median_tat_minutes": None,
			"overdue_count": 0,
			"on_time_rate_pct": None,
		}

	sorted_tats = sorted(tat_values)
	median = statistics.median(sorted_tats)
	p90_idx = math.ceil(0.9 * total) - 1
	p90 = sorted_tats[min(p90_idx, total - 1)]
	overdue = sum(1 for t in tat_values if t > target_tat)
	on_time_rate = round((1 - overdue / total) * 100, 1) if total else None

	stat_median: float | None = None
	if stat_tat_values:
		stat_median = statistics.median(sorted(stat_tat_values))

	return {
		"total_completed": total,
		"median_tat_minutes": round(median, 1),
		"p90_tat_minutes": round(p90, 1),
		"stat_median_tat_minutes": round(stat_median, 1) if stat_median else None,
		"overdue_count": overdue,
		"on_time_rate_pct": on_time_rate,
	}


# ── Workload statistics ───────────────────────────────────────────────────────

def calculate_rejection_rate(
	total_specimens: int,
	rejected_specimens: int,
) -> float:
	"""Return specimen rejection rate as a percentage."""
	if total_specimens == 0:
		return 0.0
	return round((rejected_specimens / total_specimens) * 100, 2)


def calculate_critical_value_response_time(
	notified_at: datetime,
	acknowledged_at: datetime | None,
) -> float | None:
	"""Minutes from critical value notification to physician acknowledgement."""
	if acknowledged_at is None:
		return None
	delta = acknowledged_at - notified_at
	return max(0.0, round(delta.total_seconds() / 60, 1))


def calculate_pass_rate(
	total_qc_runs: int,
	passed_runs: int,
) -> float:
	"""QC pass rate as a percentage."""
	if total_qc_runs == 0:
		return 0.0
	return round((passed_runs / total_qc_runs) * 100, 2)


# ── Reference range selection ─────────────────────────────────────────────────

def select_reference_range(
	ranges: list[dict],
	age_years: float | None,
	sex: str | None,
) -> dict | None:
	"""Select the most specific matching reference range for a patient.

	Preference order: age + sex > age only > sex only > universal.
	Returns None if no matching range exists.

	Parameters
	----------
	ranges    : list of range dicts with keys age_min_years, age_max_years, sex
	age_years : patient age in years (may be None)
	sex       : patient sex ('M', 'F', 'O', 'U', or None)
	"""
	def _age_matches(r: dict) -> bool:
		if age_years is None:
			return r.get("age_min_years") is None and r.get("age_max_years") is None
		lo = r.get("age_min_years")
		hi = r.get("age_max_years")
		if lo is not None and age_years < lo:
			return False
		if hi is not None and age_years > hi:
			return False
		return True

	def _sex_matches(r: dict) -> bool:
		r_sex = r.get("sex")
		if r_sex is None:
			return True
		return r_sex == sex

	# Score: age_specificity + sex_specificity
	def _score(r: dict) -> int:
		score = 0
		if r.get("age_min_years") is not None or r.get("age_max_years") is not None:
			score += 2
		if r.get("sex") is not None:
			score += 1
		return score

	candidates = [r for r in ranges if _age_matches(r) and _sex_matches(r)]
	if not candidates:
		return None
	return max(candidates, key=_score)
