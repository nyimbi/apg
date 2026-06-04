"""Domain calculations for Monitoring and Observability.

All formulas are pure functions — no I/O, no side effects.
Type-safe inputs; comprehensive edge-case handling throughout.

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""
from __future__ import annotations

import math
import statistics
from datetime import datetime, timedelta
from typing import Sequence


# ─── SLO & Error-Budget ───────────────────────────────────────────────────────

def calculate_error_budget_minutes(
	objective_percent: float,
	window_days: int,
) -> float:
	"""Total allowed downtime minutes for an SLO window.

	Args:
		objective_percent: e.g. 99.9 for three-nines.
		window_days: Rolling window in calendar days.

	Returns:
		Allowed error budget in minutes.
	"""
	assert 0.0 < objective_percent <= 100.0, "objective_percent must be in (0, 100]"
	assert window_days >= 1, "window_days must be at least 1"
	window_minutes = window_days * 24 * 60
	allowed_fraction = (100.0 - objective_percent) / 100.0
	return window_minutes * allowed_fraction


def calculate_remaining_budget_minutes(
	total_budget_minutes: float,
	consumed_minutes: float,
) -> float:
	"""Remaining error budget after accounting for observed downtime.

	Clamps to [0, total_budget_minutes] — never negative or above total.
	"""
	assert total_budget_minutes >= 0.0, "total_budget_minutes must be non-negative"
	assert consumed_minutes >= 0.0, "consumed_minutes must be non-negative"
	return max(0.0, min(total_budget_minutes, total_budget_minutes - consumed_minutes))


def calculate_slo_compliance_percent(
	good_requests: int,
	total_requests: int,
) -> float:
	"""Compliance percentage from a good/total request ratio.

	Returns 100.0 when total_requests == 0 (vacuously compliant).
	"""
	assert good_requests >= 0, "good_requests must be non-negative"
	assert total_requests >= 0, "total_requests must be non-negative"
	assert good_requests <= total_requests, "good_requests cannot exceed total_requests"
	if total_requests == 0:
		return 100.0
	return (good_requests / total_requests) * 100.0


def calculate_error_budget_remaining_percent(
	current_compliance: float,
	objective_percent: float,
) -> float:
	"""Remaining error budget as a percentage of the original budget.

	Formula: remaining% = (compliance - objective) / (100 - objective) * 100
	Clamps to [0, 100].
	"""
	assert 0.0 <= current_compliance <= 100.0
	assert 0.0 < objective_percent <= 100.0
	allowed_error = 100.0 - objective_percent
	if allowed_error <= 0.0:
		# 100% objective — any error exhausts the budget
		return 100.0 if current_compliance >= objective_percent else 0.0
	consumed_error = objective_percent - current_compliance
	remaining = 1.0 - (consumed_error / allowed_error)
	return max(0.0, min(100.0, remaining * 100.0))


def calculate_burn_rate(
	consumed_minutes: float,
	total_budget_minutes: float,
	elapsed_fraction: float,
) -> float:
	"""Burn rate: how fast error budget is consumed relative to time elapsed.

	A burn rate of 1.0 means budget is being consumed at exactly the allowed pace.
	>1.0 means burning faster than allowed; <1.0 means margin exists.

	elapsed_fraction must be in (0, 1].
	"""
	assert total_budget_minutes > 0.0, "total_budget_minutes must be positive"
	assert 0.0 < elapsed_fraction <= 1.0, "elapsed_fraction must be in (0, 1]"
	if total_budget_minutes == 0.0:
		return 0.0
	normalized_consumed = consumed_minutes / total_budget_minutes
	return normalized_consumed / elapsed_fraction


def calculate_time_to_exhaustion_hours(
	remaining_minutes: float,
	burn_rate_per_hour: float,
) -> float | None:
	"""Hours until error budget is exhausted at the current burn rate.

	Returns None if burn_rate_per_hour <= 0 (budget not being consumed).
	"""
	assert remaining_minutes >= 0.0
	if burn_rate_per_hour <= 0.0:
		return None
	return remaining_minutes / 60.0 / burn_rate_per_hour


# ─── Anomaly detection ────────────────────────────────────────────────────────

def calculate_z_score(
	observed: float,
	baseline_mean: float,
	baseline_std: float,
) -> float:
	"""Z-score magnitude of an observation relative to the baseline.

	Returns 0.0 when std == 0 (constant series — treat as no anomaly).
	"""
	if baseline_std <= 0.0:
		return 0.0
	return abs((observed - baseline_mean) / baseline_std)


def calculate_anomaly_score(
	z_score: float,
	sensitivity: float = 0.8,
) -> float:
	"""Normalised anomaly score in [0, 1] from a z-score.

	Higher sensitivity lowers the z-score threshold for a given score.
	Uses sigmoid: score = 1 / (1 + exp(-(z - threshold)))

	sensitivity in [0, 1]: 1.0 → threshold≈1, 0.0 → threshold≈5.
	"""
	assert 0.0 <= sensitivity <= 1.0, "sensitivity must be in [0, 1]"
	threshold = 1.0 + (1.0 - sensitivity) * 4.0  # maps [0,1] → [5,1]
	exponent = -(z_score - threshold)
	# Guard against overflow
	if exponent > 500:
		return 0.0
	if exponent < -500:
		return 1.0
	return 1.0 / (1.0 + math.exp(exponent))


def calculate_iqr_bounds(
	values: Sequence[float],
	iqr_multiplier: float = 1.5,
) -> tuple[float, float]:
	"""Tukey IQR-based outlier bounds (lower, upper).

	iqr_multiplier=1.5 is classic Tukey; 3.0 for extreme outliers.
	"""
	assert len(values) >= 4, "need at least 4 values for IQR bounds"
	assert iqr_multiplier > 0.0
	sorted_vals = sorted(values)
	n = len(sorted_vals)
	q1 = sorted_vals[n // 4]
	q3 = sorted_vals[(3 * n) // 4]
	iqr = q3 - q1
	return q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr


def calculate_moving_average(
	values: Sequence[float],
	window: int,
) -> list[float]:
	"""Simple moving average with the given window size.

	Returns a list of the same length; early entries use available samples.
	"""
	assert window >= 1, "window must be at least 1"
	result: list[float] = []
	vals = list(values)
	for i in range(len(vals)):
		start = max(0, i - window + 1)
		result.append(statistics.mean(vals[start : i + 1]))
	return result


def calculate_rate_of_change(
	current_value: float,
	previous_value: float,
	elapsed_seconds: float,
) -> float:
	"""Rate of change per second between two observations.

	Returns 0.0 when elapsed_seconds == 0.
	"""
	assert elapsed_seconds >= 0.0, "elapsed_seconds must be non-negative"
	if elapsed_seconds == 0.0:
		return 0.0
	return (current_value - previous_value) / elapsed_seconds


# ─── Health scoring ───────────────────────────────────────────────────────────

def calculate_health_score(
	critical_alert_count: int,
	high_alert_count: int,
	slo_breached: bool,
	has_recent_data: bool,
	rule_effectiveness_avg: float = 1.0,
) -> float:
	"""Composite health score in [0.0, 1.0] for a tenant or service.

	Penalties:
	  - Each critical alert: −0.10, capped at −0.50
	  - Each high alert: −0.05, capped at −0.30
	  - SLO breached: −0.20
	  - No recent data: −0.30
	  - Poor rule effectiveness (<0.5): −0.20
	"""
	assert critical_alert_count >= 0
	assert high_alert_count >= 0
	assert 0.0 <= rule_effectiveness_avg <= 1.0

	score = 1.0
	score -= min(0.50, critical_alert_count * 0.10)
	score -= min(0.30, high_alert_count * 0.05)
	if slo_breached:
		score -= 0.20
	if not has_recent_data:
		score -= 0.30
	if rule_effectiveness_avg < 0.5:
		score -= 0.20
	return max(0.0, min(1.0, score))


def calculate_alert_impact_score(
	severity: str,
	affected_services_count: int,
	affected_users_count: int,
) -> float:
	"""Impact score in [0.0, 1.0] for an alert.

	Combines severity weight with blast radius.
	"""
	severity_weights = {
		"critical": 1.0,
		"high": 0.75,
		"medium": 0.50,
		"low": 0.25,
		"info": 0.10,
	}
	base = severity_weights.get(severity, 0.25)
	service_factor = min(1.0, affected_services_count / 10.0) if affected_services_count else 0.0
	user_factor = min(1.0, affected_users_count / 1000.0) if affected_users_count else 0.0
	# Weighted combination
	impact = base * 0.5 + service_factor * 0.3 + user_factor * 0.2
	return max(0.0, min(1.0, impact))


# ─── Metric aggregations ──────────────────────────────────────────────────────

def calculate_percentile(
	values: Sequence[float],
	percentile: float,
) -> float:
	"""Linear-interpolation percentile (0–100).

	Requires at least one value.
	"""
	assert len(values) >= 1, "values must be non-empty"
	assert 0.0 <= percentile <= 100.0, "percentile must be in [0, 100]"
	sorted_vals = sorted(values)
	n = len(sorted_vals)
	if n == 1:
		return sorted_vals[0]
	rank = percentile / 100.0 * (n - 1)
	lower = int(rank)
	upper = lower + 1
	if upper >= n:
		return sorted_vals[-1]
	frac = rank - lower
	return sorted_vals[lower] + frac * (sorted_vals[upper] - sorted_vals[lower])


def calculate_apdex(
	satisfied_count: int,
	tolerating_count: int,
	total_count: int,
) -> float:
	"""Apdex score in [0.0, 1.0].

	Apdex = (satisfied + tolerating/2) / total
	Returns 1.0 when total == 0.
	"""
	assert satisfied_count >= 0
	assert tolerating_count >= 0
	assert total_count >= 0
	if total_count == 0:
		return 1.0
	return (satisfied_count + tolerating_count / 2.0) / total_count


def calculate_ingestion_rate_per_minute(
	sample_count: int,
	window_seconds: float,
) -> float:
	"""Convert a sample count over a window to a per-minute rate.

	Returns 0.0 when window_seconds == 0.
	"""
	assert sample_count >= 0
	assert window_seconds >= 0.0
	if window_seconds == 0.0:
		return 0.0
	return sample_count / window_seconds * 60.0


# ─── EWMA helpers ─────────────────────────────────────────────────────────────

def ewma_update(
	current: float,
	new_value: float,
	alpha: float = 0.1,
) -> float:
	"""Single-step exponentially-weighted moving average update.

	alpha is the weight given to the new value (0 < alpha <= 1).
	"""
	assert 0.0 < alpha <= 1.0, "alpha must be in (0, 1]"
	return current * (1.0 - alpha) + new_value * alpha


def ewma_false_positive_decay(
	current_rate: float,
	is_false_positive: bool,
	decay: float = 0.95,
	increment: float = 0.05,
) -> float:
	"""Update false-positive rate with EWMA: decay on correct, increment on FP."""
	assert 0.0 <= current_rate <= 1.0
	if is_false_positive:
		return min(1.0, current_rate * decay + increment)
	return current_rate * decay


__all__ = [
	# SLO / error-budget
	"calculate_error_budget_minutes",
	"calculate_remaining_budget_minutes",
	"calculate_slo_compliance_percent",
	"calculate_error_budget_remaining_percent",
	"calculate_burn_rate",
	"calculate_time_to_exhaustion_hours",
	# anomaly
	"calculate_z_score",
	"calculate_anomaly_score",
	"calculate_iqr_bounds",
	"calculate_moving_average",
	"calculate_rate_of_change",
	# health
	"calculate_health_score",
	"calculate_alert_impact_score",
	# metric aggregations
	"calculate_percentile",
	"calculate_apdex",
	"calculate_ingestion_rate_per_minute",
	# EWMA
	"ewma_update",
	"ewma_false_positive_decay",
]
