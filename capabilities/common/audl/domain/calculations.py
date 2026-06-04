"""
APG Audit Logging — Domain Calculations.

All numeric formulas, scoring models, and statistical aggregations that are
specific to the audit-logging domain.  Pure functions, no I/O, no side effects.

© 2025 Datacraft  www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Risk scoring weights — tunable without touching service.py
# ---------------------------------------------------------------------------

RISK_WEIGHTS: dict[str, float] = {
	"failed_auth":      0.35,
	"privileged_actor": 0.20,
	"off_hours":        0.15,
	"external_ip":      0.15,
	"sensitive_data":   0.10,
	"error_event":      0.05,
	"anomaly_max":      0.20,   # max anomaly contribution
}


def score_event_risk(
	*,
	is_failed_auth:      bool  = False,
	is_privileged_actor: bool  = False,
	is_off_hours:        bool  = False,
	is_external_ip:      bool  = False,
	is_sensitive_data:   bool  = False,
	is_error_event:      bool  = False,
	anomaly_hint:        float = 0.0,
) -> float:
	"""
	Compute a risk score in [0.0, 1.0] using the MITRE ATT&CK-aligned weight
	table in :data:`RISK_WEIGHTS`.

	Parameters
	----------
	is_failed_auth : bool
	    True for authentication failures (USER_FAILED_LOGIN, ACCESS_DENIED).
	is_privileged_actor : bool
	    True when actor_type is 'admin' or 'service'.
	is_off_hours : bool
	    True when the event timestamp falls outside 07:00–20:00 UTC.
	is_external_ip : bool
	    True when the source IP is not RFC-1918 / loopback.
	is_sensitive_data : bool
	    True when data_classification is 'confidential', 'restricted', or 'secret'.
	is_error_event : bool
	    True for any non-success event.
	anomaly_hint : float
	    External anomaly signal in [0, 1], e.g. from an ML model.
	    Scaled to at most RISK_WEIGHTS["anomaly_max"].

	Returns
	-------
	float
	    Score clamped to [0.0, 1.0], rounded to 4 decimal places.
	"""
	score = 0.0
	if is_failed_auth:      score += RISK_WEIGHTS["failed_auth"]
	if is_privileged_actor: score += RISK_WEIGHTS["privileged_actor"]
	if is_off_hours:        score += RISK_WEIGHTS["off_hours"]
	if is_external_ip:      score += RISK_WEIGHTS["external_ip"]
	if is_sensitive_data:   score += RISK_WEIGHTS["sensitive_data"]
	if is_error_event:      score += RISK_WEIGHTS["error_event"]
	score += max(0.0, min(1.0, anomaly_hint)) * RISK_WEIGHTS["anomaly_max"]
	return round(min(1.0, score), 4)


def classify_risk_tier(score: float) -> str:
	"""
	Map a numeric risk score to a categorical tier.

	Tiers:
	  critical  [0.8, 1.0]
	  high      [0.6, 0.8)
	  medium    [0.3, 0.6)
	  low       [0.0, 0.3)
	"""
	assert 0.0 <= score <= 1.0, f"risk score out of range: {score}"
	if score >= 0.8:   return "critical"
	if score >= 0.6:   return "high"
	if score >= 0.3:   return "medium"
	return "low"


# ---------------------------------------------------------------------------
# Retention / expiry calculations
# ---------------------------------------------------------------------------

def retention_expiry(created_at: datetime, retain_days: int) -> datetime:
	"""
	Return the UTC expiry datetime for a record.

	``expiry = created_at + retain_days``
	"""
	return created_at + timedelta(days=retain_days)


def days_until_expiry(created_at: datetime, retain_days: int) -> int:
	"""
	Return the number of full days until the retention window closes.

	Negative means already expired.
	"""
	now    = datetime.now(timezone.utc)
	expiry = retention_expiry(created_at, retain_days)
	delta  = expiry - now
	return int(delta.total_seconds() // 86_400)


def is_expired(created_at: datetime, retain_days: int) -> bool:
	"""Return True if the retention window has passed."""
	return datetime.now(timezone.utc) > retention_expiry(created_at, retain_days)


# ---------------------------------------------------------------------------
# Statistical aggregation helpers
# ---------------------------------------------------------------------------

def event_volume_trend(
	hourly_counts: list[int],
) -> dict[str, Any]:
	"""
	Compute basic volume trend statistics over a list of hourly event counts.

	Parameters
	----------
	hourly_counts : list[int]
	    Ordered count per hour (oldest first).

	Returns
	-------
	dict with: mean, stddev, min, max, trend ('rising'|'falling'|'flat').
	"""
	if not hourly_counts:
		return {"mean": 0.0, "stddev": 0.0, "min": 0, "max": 0, "trend": "flat"}
	n      = len(hourly_counts)
	mean   = sum(hourly_counts) / n
	var    = sum((x - mean) ** 2 for x in hourly_counts) / n
	stddev = math.sqrt(var)
	# Simple linear trend: compare second half to first half
	mid    = n // 2
	first  = sum(hourly_counts[:mid]) / max(mid, 1)
	second = sum(hourly_counts[mid:]) / max(n - mid, 1)
	if second > first * 1.1:
		trend = "rising"
	elif second < first * 0.9:
		trend = "falling"
	else:
		trend = "flat"
	return {
		"mean":   round(mean, 2),
		"stddev": round(stddev, 2),
		"min":    min(hourly_counts),
		"max":    max(hourly_counts),
		"trend":  trend,
	}


def anomaly_z_score(
	value: float,
	mean:  float,
	stddev: float,
) -> float:
	"""
	Compute z-score for anomaly detection.

	Returns 0.0 when stddev is zero (prevents division by zero).
	"""
	if stddev == 0.0:
		return 0.0
	return (value - mean) / stddev


def score_to_probability(score: float, scale: float = 6.0) -> float:
	"""
	Map a linear risk score [0, 1] to a log-odds probability via sigmoid.

	Useful for displaying risk as a percentage probability to end users.
	``scale`` controls the steepness of the sigmoid (default 6 ≈ 0.5 → 50%).
	"""
	x = (score - 0.5) * scale
	return round(1.0 / (1.0 + math.exp(-x)), 4)


# ---------------------------------------------------------------------------
# Compliance gap analysis
# ---------------------------------------------------------------------------

def compliance_coverage_pct(
	total_events:      int,
	tagged_events:     int,
) -> float:
	"""
	Return the percentage of events that carry a compliance tag.

	0.0 when total_events is 0.
	"""
	if total_events == 0:
		return 0.0
	return round((tagged_events / total_events) * 100, 2)


def violation_rate(
	total_events:     int,
	violation_events: int,
) -> float:
	"""
	Return violations as a percentage of total events.

	0.0 when total_events is 0.
	"""
	if total_events == 0:
		return 0.0
	return round((violation_events / total_events) * 100, 4)


# ---------------------------------------------------------------------------
# Evidence package integrity
# ---------------------------------------------------------------------------

def evidence_manifest_hash(event_checksums: list[str]) -> str:
	"""
	Produce a deterministic SHA-256 hash of an ordered list of event checksums.

	The manifest hash uniquely identifies the set and order of events included
	in an evidence package.  Changing any checksum or their order invalidates
	the manifest.
	"""
	import hashlib, json
	canonical = json.dumps(sorted(event_checksums), sort_keys=True, separators=(",", ":"))
	return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def dsr_deadline(created_at: datetime, regulation: str = "GDPR") -> datetime:
	"""
	Return the regulatory deadline for a data-subject request.

	  GDPR   Art. 12(3) — 30 days  (extendable to 90, not modelled here)
	  CCPA   Cal. Civ. Code §1798.100 — 45 days
	  HIPAA  45 CFR §164.524 — 30 days

	Defaults to 30 days for unknown regulations.
	"""
	days_map = {
		"GDPR":  30,
		"CCPA":  45,
		"HIPAA": 30,
	}
	days = days_map.get(regulation.upper(), 30)
	return created_at + timedelta(days=days)


__all__ = [
	"RISK_WEIGHTS",
	"score_event_risk",
	"classify_risk_tier",
	"retention_expiry",
	"days_until_expiry",
	"is_expired",
	"event_volume_trend",
	"anomaly_z_score",
	"score_to_probability",
	"compliance_coverage_pct",
	"violation_rate",
	"evidence_manifest_hash",
	"dsr_deadline",
]
