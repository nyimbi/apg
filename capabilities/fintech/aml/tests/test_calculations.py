"""Tests for AML domain calculations."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
	sys.path.insert(0, str(PACKAGE_DIR))

from domain.calculations import (  # type: ignore
	calculate_false_positive_rate,
	calculate_network_risk_score,
	calculate_risk_score,
	calculate_sar_priority,
	detect_layering,
	detect_round_trip,
	detect_structuring,
	detect_velocity_anomaly,
	requires_ctr,
	risk_segment_from_score,
	severity_from_score,
)


def _txn(
	amount: float,
	sender: str = "A",
	receiver: str = "B",
	minutes_ago: int = 0,
) -> dict:
	return {
		"amount": amount,
		"currency": "USD",
		"sender_account": sender,
		"receiver_account": receiver,
		"created_at": (datetime.utcnow() - timedelta(minutes=minutes_ago)).isoformat(),
	}


# ---------------------------------------------------------------------------
# Risk scoring
# ---------------------------------------------------------------------------

def test_risk_score_sanctions_overrides():
	score = calculate_risk_score(
		amount=100, large_threshold=10000, structuring_threshold=9500,
		velocity_count=0, velocity_window_hours=24,
		sanctions_hit=True, pep_hit=False,
		high_risk_country=False, adverse_media=False,
	)
	assert score == 100


def test_risk_score_pep_adds_30():
	score = calculate_risk_score(
		amount=100, large_threshold=10000, structuring_threshold=9500,
		velocity_count=0, velocity_window_hours=24,
		sanctions_hit=False, pep_hit=True,
		high_risk_country=False, adverse_media=False,
		base_kyc_score=0,
	)
	assert score == 30


def test_risk_score_large_tx():
	score = calculate_risk_score(
		amount=15000, large_threshold=10000, structuring_threshold=9500,
		velocity_count=0, velocity_window_hours=24,
		sanctions_hit=False, pep_hit=False,
		high_risk_country=False, adverse_media=False,
	)
	assert score >= 25


def test_risk_score_clamped():
	score = calculate_risk_score(
		amount=100000, large_threshold=10000, structuring_threshold=9500,
		velocity_count=100, velocity_window_hours=1,
		sanctions_hit=False, pep_hit=True,
		high_risk_country=True, adverse_media=True,
		base_kyc_score=50,
	)
	assert 0 <= score <= 100


def test_severity_from_score():
	assert severity_from_score(95) == "critical"
	assert severity_from_score(75) == "high"
	assert severity_from_score(50) == "medium"
	assert severity_from_score(10) == "low"


def test_risk_segment_from_score():
	assert risk_segment_from_score(95) == "prohibited"
	assert risk_segment_from_score(75) == "very_high"
	assert risk_segment_from_score(50) == "high"
	assert risk_segment_from_score(30) == "medium"
	assert risk_segment_from_score(5) == "low"


# ---------------------------------------------------------------------------
# Structuring detection
# ---------------------------------------------------------------------------

def test_detect_structuring_detected():
	txns = [_txn(9600, minutes_ago=i * 60) for i in range(5)]
	result = detect_structuring(txns, reporting_threshold=10_000, min_occurrences=3)
	assert result["detected"] is True
	assert result["count"] == 5


def test_detect_structuring_not_triggered_below_band():
	txns = [_txn(1000, minutes_ago=i * 60) for i in range(10)]
	result = detect_structuring(txns, reporting_threshold=10_000, min_occurrences=3)
	assert result["detected"] is False


def test_detect_structuring_empty():
	result = detect_structuring([])
	assert result["detected"] is False
	assert result["count"] == 0


def test_detect_structuring_single_txn_not_flagged():
	result = detect_structuring([_txn(9600)], min_occurrences=3)
	assert result["detected"] is False


# ---------------------------------------------------------------------------
# Velocity detection
# ---------------------------------------------------------------------------

def test_detect_velocity_count_threshold():
	txns = [_txn(100, minutes_ago=i * 5) for i in range(15)]
	result = detect_velocity_anomaly(txns, window_hours=24, count_threshold=10)
	assert result["detected"] is True
	assert result["count"] == 15


def test_detect_velocity_amount_threshold():
	txns = [_txn(10_000, minutes_ago=i * 30) for i in range(6)]
	result = detect_velocity_anomaly(txns, window_hours=24, amount_threshold=50_000)
	assert result["detected"] is True


def test_detect_velocity_empty():
	result = detect_velocity_anomaly([])
	assert result["detected"] is False


# ---------------------------------------------------------------------------
# Round-trip detection
# ---------------------------------------------------------------------------

def test_detect_round_trip_simple():
	txns = [
		_txn(10_000, sender="A", receiver="B", minutes_ago=120),
		_txn(9_800, sender="B", receiver="A", minutes_ago=60),
	]
	result = detect_round_trip(txns, tolerance_pct=0.05, max_hops=3)
	assert result["detected"] is True


def test_detect_round_trip_empty():
	result = detect_round_trip([])
	assert result["detected"] is False


# ---------------------------------------------------------------------------
# Layering detection
# ---------------------------------------------------------------------------

def test_detect_layering_chain():
	txns = [
		_txn(10_000, sender="A", receiver="B", minutes_ago=300),
		_txn(9_900, sender="B", receiver="C", minutes_ago=200),
		_txn(9_800, sender="C", receiver="D", minutes_ago=100),
	]
	result = detect_layering(txns, min_layers=3)
	assert result["detected"] is True
	assert result["layers"] >= 3


def test_detect_layering_insufficient_chain():
	txns = [
		_txn(10_000, sender="A", receiver="B", minutes_ago=120),
	]
	result = detect_layering(txns, min_layers=3)
	assert result["detected"] is False


# ---------------------------------------------------------------------------
# Network risk
# ---------------------------------------------------------------------------

def test_calculate_network_risk_base():
	score = calculate_network_risk_score(
		direct_risk_scores=[50, 60],
		indirect_risk_scores=[30, 40],
		round_trip_detected=False,
		layering_detected=False,
	)
	assert 0 <= score <= 100


def test_calculate_network_risk_with_patterns():
	score = calculate_network_risk_score(
		direct_risk_scores=[60],
		indirect_risk_scores=[],
		round_trip_detected=True,
		layering_detected=True,
	)
	# round-trip (+20) + layering (+25) should push score higher
	assert score >= 80


# ---------------------------------------------------------------------------
# CTR threshold
# ---------------------------------------------------------------------------

def test_requires_ctr_us():
	assert requires_ctr(15_000.0, "USD", "US") is True
	assert requires_ctr(9_999.0, "USD", "US") is False


def test_requires_ctr_ke():
	assert requires_ctr(1_500_000.0, "KES", "KE") is True
	assert requires_ctr(500_000.0, "KES", "KE") is False


# ---------------------------------------------------------------------------
# SAR priority
# ---------------------------------------------------------------------------

def test_calculate_sar_priority_high():
	priority = calculate_sar_priority(
		risk_score=95, days_since_suspicious=2,
		typology_count=4, amount=2_000_000,
	)
	assert priority == 1


def test_calculate_sar_priority_low():
	priority = calculate_sar_priority(
		risk_score=20, days_since_suspicious=60,
		typology_count=0, amount=500,
	)
	assert priority >= 4


# ---------------------------------------------------------------------------
# False positive rate
# ---------------------------------------------------------------------------

def test_false_positive_rate():
	assert calculate_false_positive_rate(100, 25) == 0.25
	assert calculate_false_positive_rate(0, 0) == 0.0
	assert calculate_false_positive_rate(10, 15) == 1.0  # clamped


def test_false_positive_rate_zero_alerts():
	assert calculate_false_positive_rate(0, 5) == 0.0
