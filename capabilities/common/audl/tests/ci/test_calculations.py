"""
Tests for domain/calculations.py.

© 2025 Datacraft  www.datacraft.co.ke
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from capabilities.common.audl.domain.calculations import (
	anomaly_z_score,
	classify_risk_tier,
	compliance_coverage_pct,
	days_until_expiry,
	dsr_deadline,
	evidence_manifest_hash,
	event_volume_trend,
	is_expired,
	retention_expiry,
	score_event_risk,
	score_to_probability,
	violation_rate,
)


# ---------------------------------------------------------------------------
# score_event_risk
# ---------------------------------------------------------------------------

def test_score_zero():
	assert score_event_risk() == 0.0


def test_score_failed_auth():
	s = score_event_risk(is_failed_auth=True)
	assert s == 0.35


def test_score_all_factors():
	s = score_event_risk(
		is_failed_auth=True, is_privileged_actor=True,
		is_off_hours=True, is_external_ip=True,
		is_sensitive_data=True, is_error_event=True,
		anomaly_hint=1.0,
	)
	assert s == 1.0


def test_score_capped():
	s = score_event_risk(
		is_failed_auth=True, is_privileged_actor=True,
		is_off_hours=True, is_external_ip=True,
		is_sensitive_data=True, is_error_event=True,
	)
	assert s <= 1.0


def test_score_anomaly_contribution():
	s = score_event_risk(anomaly_hint=1.0)
	assert s == 0.20


def test_score_anomaly_clamped():
	s = score_event_risk(anomaly_hint=99.0)
	assert s == 0.20   # clamps at 1.0 before multiplying


# ---------------------------------------------------------------------------
# classify_risk_tier
# ---------------------------------------------------------------------------

def test_tier_critical():
	assert classify_risk_tier(0.8) == "critical"
	assert classify_risk_tier(1.0) == "critical"


def test_tier_high():
	assert classify_risk_tier(0.6) == "high"
	assert classify_risk_tier(0.79) == "high"


def test_tier_medium():
	assert classify_risk_tier(0.3) == "medium"
	assert classify_risk_tier(0.59) == "medium"


def test_tier_low():
	assert classify_risk_tier(0.0) == "low"
	assert classify_risk_tier(0.29) == "low"


def test_tier_out_of_range():
	with pytest.raises(AssertionError):
		classify_risk_tier(1.5)


# ---------------------------------------------------------------------------
# retention_expiry / days_until_expiry / is_expired
# ---------------------------------------------------------------------------

def test_retention_expiry():
	created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
	exp = retention_expiry(created_at, 365)
	assert exp == datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_days_until_expiry_positive():
	created_at = datetime.now(timezone.utc) - timedelta(days=5)
	d = days_until_expiry(created_at, 30)
	assert d >= 24


def test_days_until_expiry_negative():
	created_at = datetime.now(timezone.utc) - timedelta(days=100)
	d = days_until_expiry(created_at, 30)
	assert d < 0


def test_is_expired_true():
	created_at = datetime.now(timezone.utc) - timedelta(days=50)
	assert is_expired(created_at, 30) is True


def test_is_expired_false():
	created_at = datetime.now(timezone.utc) - timedelta(days=5)
	assert is_expired(created_at, 30) is False


# ---------------------------------------------------------------------------
# event_volume_trend
# ---------------------------------------------------------------------------

def test_volume_trend_empty():
	result = event_volume_trend([])
	assert result["trend"] == "flat"
	assert result["mean"] == 0.0


def test_volume_trend_rising():
	result = event_volume_trend([1, 2, 3, 4, 10, 20, 30])
	assert result["trend"] == "rising"


def test_volume_trend_falling():
	result = event_volume_trend([30, 20, 10, 4, 3, 2, 1])
	assert result["trend"] == "falling"


def test_volume_trend_flat():
	result = event_volume_trend([10, 10, 10, 10, 10, 10])
	assert result["trend"] == "flat"


def test_volume_trend_stats():
	counts = [1, 2, 3, 4, 5]
	result = event_volume_trend(counts)
	assert result["min"] == 1
	assert result["max"] == 5
	assert result["mean"] == 3.0


# ---------------------------------------------------------------------------
# anomaly_z_score
# ---------------------------------------------------------------------------

def test_z_score_zero_stddev():
	assert anomaly_z_score(5.0, 5.0, 0.0) == 0.0


def test_z_score_above_mean():
	z = anomaly_z_score(10.0, 5.0, 2.0)
	assert z == 2.5


def test_z_score_below_mean():
	z = anomaly_z_score(3.0, 5.0, 2.0)
	assert z == -1.0


# ---------------------------------------------------------------------------
# score_to_probability
# ---------------------------------------------------------------------------

def test_prob_midpoint():
	p = score_to_probability(0.5)
	assert 0.49 <= p <= 0.51   # sigmoid(0) ≈ 0.5


def test_prob_high_risk():
	p = score_to_probability(1.0)
	assert p > 0.9


def test_prob_low_risk():
	p = score_to_probability(0.0)
	assert p < 0.1


# ---------------------------------------------------------------------------
# compliance_coverage_pct
# ---------------------------------------------------------------------------

def test_coverage_zero_total():
	assert compliance_coverage_pct(0, 0) == 0.0


def test_coverage_full():
	assert compliance_coverage_pct(100, 100) == 100.0


def test_coverage_partial():
	assert compliance_coverage_pct(100, 75) == 75.0


# ---------------------------------------------------------------------------
# violation_rate
# ---------------------------------------------------------------------------

def test_violation_rate_zero():
	assert violation_rate(0, 0) == 0.0


def test_violation_rate_ten_pct():
	assert violation_rate(100, 10) == 10.0


# ---------------------------------------------------------------------------
# evidence_manifest_hash
# ---------------------------------------------------------------------------

def test_manifest_hash_deterministic():
	cs = ["aaa" * 21 + "a", "bbb" * 21 + "b"]
	h1 = evidence_manifest_hash(cs)
	h2 = evidence_manifest_hash(cs)
	assert h1 == h2
	assert len(h1) == 64


def test_manifest_hash_order_independent():
	cs = ["a" * 64, "b" * 64]
	h1 = evidence_manifest_hash(cs)
	h2 = evidence_manifest_hash(list(reversed(cs)))
	assert h1 == h2   # sorted internally


def test_manifest_hash_differs_for_different_checksums():
	h1 = evidence_manifest_hash(["a" * 64])
	h2 = evidence_manifest_hash(["b" * 64])
	assert h1 != h2


# ---------------------------------------------------------------------------
# dsr_deadline
# ---------------------------------------------------------------------------

def test_dsr_deadline_gdpr():
	created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
	dl = dsr_deadline(created_at, "GDPR")
	assert dl == datetime(2025, 1, 31, tzinfo=timezone.utc)


def test_dsr_deadline_ccpa():
	created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
	dl = dsr_deadline(created_at, "CCPA")
	assert dl == datetime(2025, 2, 15, tzinfo=timezone.utc)


def test_dsr_deadline_unknown_defaults_30():
	created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
	dl = dsr_deadline(created_at, "UNKNOWN")
	assert dl == datetime(2025, 1, 31, tzinfo=timezone.utc)
