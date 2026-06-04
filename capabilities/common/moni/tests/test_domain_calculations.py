"""CI tests for MONI domain calculations.

All functions are pure — no I/O, no mocks needed.
Covers normal paths, edge cases, and assertion guards.

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""
from __future__ import annotations

import math
import pytest

from capabilities.common.moni.domain.calculations import (
	calculate_error_budget_minutes,
	calculate_remaining_budget_minutes,
	calculate_slo_compliance_percent,
	calculate_error_budget_remaining_percent,
	calculate_burn_rate,
	calculate_time_to_exhaustion_hours,
	calculate_z_score,
	calculate_anomaly_score,
	calculate_iqr_bounds,
	calculate_moving_average,
	calculate_rate_of_change,
	calculate_health_score,
	calculate_alert_impact_score,
	calculate_percentile,
	calculate_apdex,
	calculate_ingestion_rate_per_minute,
	ewma_update,
	ewma_false_positive_decay,
)


# ─── Error budget ─────────────────────────────────────────────────────────────

def test_error_budget_three_nines_30_days():
	# 99.9% over 30 days → 0.1% × 30d × 1440m/d = 43.2 minutes
	budget = calculate_error_budget_minutes(99.9, 30)
	assert abs(budget - 43.2) < 0.01


def test_error_budget_99_availability():
	# 99% over 30 days → 1% × 43200m = 432 minutes
	budget = calculate_error_budget_minutes(99.0, 30)
	assert abs(budget - 432.0) < 0.01


def test_error_budget_100_percent_zero_budget():
	budget = calculate_error_budget_minutes(100.0, 30)
	assert budget == 0.0


def test_error_budget_single_day():
	budget = calculate_error_budget_minutes(99.0, 1)
	assert abs(budget - 14.4) < 0.01


def test_error_budget_invalid_objective_raises():
	with pytest.raises(AssertionError):
		calculate_error_budget_minutes(0.0, 30)


def test_error_budget_invalid_window_raises():
	with pytest.raises(AssertionError):
		calculate_error_budget_minutes(99.9, 0)


# ─── Remaining budget ─────────────────────────────────────────────────────────

def test_remaining_budget_partial_consumption():
	remaining = calculate_remaining_budget_minutes(43.2, 10.0)
	assert abs(remaining - 33.2) < 0.01


def test_remaining_budget_fully_consumed():
	assert calculate_remaining_budget_minutes(43.2, 43.2) == 0.0


def test_remaining_budget_over_consumed_clamps_to_zero():
	assert calculate_remaining_budget_minutes(43.2, 100.0) == 0.0


def test_remaining_budget_zero_consumed():
	assert calculate_remaining_budget_minutes(43.2, 0.0) == 43.2


# ─── Compliance percent ───────────────────────────────────────────────────────

def test_compliance_percent_perfect():
	assert calculate_slo_compliance_percent(1000, 1000) == 100.0


def test_compliance_percent_zero_total():
	assert calculate_slo_compliance_percent(0, 0) == 100.0


def test_compliance_percent_half():
	assert calculate_slo_compliance_percent(500, 1000) == 50.0


def test_compliance_percent_invalid_raises():
	with pytest.raises(AssertionError):
		calculate_slo_compliance_percent(1001, 1000)


# ─── Error budget remaining percent ──────────────────────────────────────────

def test_budget_remaining_full():
	# compliance == objective → budget fully intact
	pct = calculate_error_budget_remaining_percent(99.9, 99.9)
	assert abs(pct - 100.0) < 0.1


def test_budget_remaining_halfway():
	# compliance halfway between 0 and objective (99.9)
	# consumed half of the 0.1% error → 50% remaining
	pct = calculate_error_budget_remaining_percent(99.85, 99.9)
	assert abs(pct - 50.0) < 1.0


def test_budget_remaining_exhausted():
	pct = calculate_error_budget_remaining_percent(0.0, 99.9)
	assert pct == 0.0


# ─── Burn rate ────────────────────────────────────────────────────────────────

def test_burn_rate_on_pace():
	# consumed exactly elapsed fraction of budget
	rate = calculate_burn_rate(21.6, 43.2, 0.5)
	assert abs(rate - 1.0) < 0.001


def test_burn_rate_double_speed():
	# consumed entire budget at halfway mark
	rate = calculate_burn_rate(43.2, 43.2, 0.5)
	assert abs(rate - 2.0) < 0.001


def test_burn_rate_zero_consumed():
	rate = calculate_burn_rate(0.0, 43.2, 0.5)
	assert rate == 0.0


def test_burn_rate_invalid_elapsed_raises():
	with pytest.raises(AssertionError):
		calculate_burn_rate(10.0, 43.2, 0.0)


# ─── Time to exhaustion ───────────────────────────────────────────────────────

def test_time_to_exhaustion_normal():
	hours = calculate_time_to_exhaustion_hours(60.0, 2.0)
	assert hours is not None
	assert abs(hours - 0.5) < 0.001


def test_time_to_exhaustion_zero_burn_rate():
	assert calculate_time_to_exhaustion_hours(60.0, 0.0) is None


def test_time_to_exhaustion_negative_burn_rate():
	assert calculate_time_to_exhaustion_hours(60.0, -1.0) is None


# ─── Z-score ──────────────────────────────────────────────────────────────────

def test_z_score_normal():
	z = calculate_z_score(115.0, 100.0, 10.0)
	assert abs(z - 1.5) < 0.001


def test_z_score_below_mean():
	z = calculate_z_score(85.0, 100.0, 10.0)
	assert abs(z - 1.5) < 0.001  # magnitude


def test_z_score_zero_std():
	assert calculate_z_score(200.0, 100.0, 0.0) == 0.0


# ─── Anomaly score ────────────────────────────────────────────────────────────

def test_anomaly_score_low_z():
	score = calculate_anomaly_score(0.5, sensitivity=0.8)
	assert 0.0 <= score < 0.5


def test_anomaly_score_high_z():
	score = calculate_anomaly_score(5.0, sensitivity=0.8)
	assert score > 0.7


def test_anomaly_score_in_unit_range():
	for z in [0.0, 1.0, 2.0, 3.0, 5.0, 10.0]:
		score = calculate_anomaly_score(z)
		assert 0.0 <= score <= 1.0


def test_anomaly_score_sensitivity_1_lower_threshold():
	# high sensitivity → lower z needed to score high
	high_sens = calculate_anomaly_score(1.5, sensitivity=1.0)
	low_sens = calculate_anomaly_score(1.5, sensitivity=0.0)
	assert high_sens > low_sens


# ─── IQR bounds ───────────────────────────────────────────────────────────────

def test_iqr_bounds_symmetric():
	values = list(range(100))
	lower, upper = calculate_iqr_bounds(values)
	assert lower < 0
	assert upper > 99


def test_iqr_bounds_too_few_values_raises():
	with pytest.raises(AssertionError):
		calculate_iqr_bounds([1, 2, 3])


# ─── Moving average ───────────────────────────────────────────────────────────

def test_moving_average_window_1_returns_values():
	vals = [1.0, 2.0, 3.0, 4.0]
	result = calculate_moving_average(vals, 1)
	assert result == vals


def test_moving_average_window_3():
	vals = [1.0, 2.0, 3.0, 4.0, 5.0]
	result = calculate_moving_average(vals, 3)
	assert len(result) == 5
	assert abs(result[2] - 2.0) < 0.001  # (1+2+3)/3
	assert abs(result[4] - 4.0) < 0.001  # (3+4+5)/3


def test_moving_average_preserves_length():
	vals = list(range(10))
	assert len(calculate_moving_average(vals, 4)) == 10


# ─── Rate of change ───────────────────────────────────────────────────────────

def test_rate_of_change_positive():
	rate = calculate_rate_of_change(110.0, 100.0, 10.0)
	assert abs(rate - 1.0) < 0.001


def test_rate_of_change_negative():
	rate = calculate_rate_of_change(90.0, 100.0, 10.0)
	assert abs(rate - (-1.0)) < 0.001


def test_rate_of_change_zero_elapsed():
	assert calculate_rate_of_change(200.0, 100.0, 0.0) == 0.0


# ─── Health score ─────────────────────────────────────────────────────────────

def test_health_score_perfect():
	score = calculate_health_score(0, 0, False, True, 1.0)
	assert score == 1.0


def test_health_score_critical_alerts_reduce_score():
	score = calculate_health_score(3, 0, False, True, 1.0)
	assert score < 1.0
	assert score >= 0.5


def test_health_score_many_critical_caps_at_50_percent_penalty():
	score = calculate_health_score(100, 0, False, True, 1.0)
	assert score == 0.5  # capped penalty


def test_health_score_slo_breach_penalises():
	with_breach = calculate_health_score(0, 0, True, True, 1.0)
	without_breach = calculate_health_score(0, 0, False, True, 1.0)
	assert with_breach < without_breach


def test_health_score_no_recent_data_penalises():
	score = calculate_health_score(0, 0, False, False, 1.0)
	assert score < 1.0


def test_health_score_clamps_to_zero():
	score = calculate_health_score(100, 100, True, False, 0.0)
	assert score == 0.0


# ─── Alert impact score ───────────────────────────────────────────────────────

def test_alert_impact_score_critical():
	score = calculate_alert_impact_score("critical", 5, 200)
	assert 0.0 <= score <= 1.0
	assert score > 0.4


def test_alert_impact_score_info():
	score = calculate_alert_impact_score("info", 0, 0)
	assert score < 0.2


def test_alert_impact_score_unknown_severity_uses_default():
	score = calculate_alert_impact_score("unknown", 0, 0)
	assert 0.0 <= score <= 1.0


# ─── Percentile ───────────────────────────────────────────────────────────────

def test_percentile_p50():
	vals = list(range(1, 101))
	p50 = calculate_percentile(vals, 50)
	assert abs(p50 - 50.0) < 1.0


def test_percentile_p100():
	vals = [1.0, 2.0, 3.0]
	assert calculate_percentile(vals, 100) == 3.0


def test_percentile_p0():
	vals = [1.0, 2.0, 3.0]
	assert calculate_percentile(vals, 0) == 1.0


def test_percentile_single_value():
	assert calculate_percentile([42.0], 75) == 42.0


# ─── Apdex ───────────────────────────────────────────────────────────────────

def test_apdex_perfect():
	assert calculate_apdex(1000, 0, 1000) == 1.0


def test_apdex_zero_total():
	assert calculate_apdex(0, 0, 0) == 1.0


def test_apdex_half_satisfied():
	score = calculate_apdex(500, 0, 1000)
	assert abs(score - 0.5) < 0.001


def test_apdex_tolerating_counts_half():
	score = calculate_apdex(0, 1000, 1000)
	assert abs(score - 0.5) < 0.001


# ─── Ingestion rate ───────────────────────────────────────────────────────────

def test_ingestion_rate_per_minute_60s():
	rate = calculate_ingestion_rate_per_minute(120, 60.0)
	assert abs(rate - 120.0) < 0.001


def test_ingestion_rate_zero_window():
	assert calculate_ingestion_rate_per_minute(100, 0.0) == 0.0


# ─── EWMA ────────────────────────────────────────────────────────────────────

def test_ewma_update_zero_alpha_keeps_current():
	# alpha must be > 0 by contract, so use near-zero
	updated = ewma_update(100.0, 200.0, alpha=0.001)
	assert abs(updated - 100.1) < 0.01


def test_ewma_update_alpha_one_returns_new():
	updated = ewma_update(100.0, 200.0, alpha=1.0)
	assert updated == 200.0


def test_ewma_update_invalid_alpha_raises():
	with pytest.raises(AssertionError):
		ewma_update(100.0, 200.0, alpha=0.0)


def test_ewma_false_positive_decay_no_fp():
	rate = ewma_false_positive_decay(0.2, is_false_positive=False)
	assert rate < 0.2


def test_ewma_false_positive_decay_with_fp():
	rate = ewma_false_positive_decay(0.0, is_false_positive=True)
	assert rate > 0.0


def test_ewma_false_positive_decay_clamps_to_one():
	rate = ewma_false_positive_decay(1.0, is_false_positive=True)
	assert rate <= 1.0
