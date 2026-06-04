"""Domain calculations for Threat Intelligence.

All formulas are pure functions with type-safe inputs and comprehensive edge
case handling. No I/O, no side effects.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any


def decay_factor(age_days: float, half_life_days: float) -> float:
	"""Exponential decay factor in [0.0, 1.0]; 1.0 = brand new."""
	if half_life_days <= 0:
		return 0.0
	return round(math.exp(-0.693 * max(age_days, 0.0) / half_life_days), 6)


def indicator_staleness(
	valid_from: datetime,
	valid_until: datetime | None,
	indicator_type: str,
) -> float:
	"""Return staleness in [0.0, 1.0]. 1.0 = completely stale.

	Uses type-specific half-lives (days):
	  ip_address=30, domain=60, url=14, file_hash_*=365,
	  certificate=180, network_signature=90, yara_rule=365, default=90.
	"""
	half_lives: dict[str, float] = {
		"ip_address": 30.0,
		"domain": 60.0,
		"url": 14.0,
		"file_hash_md5": 365.0,
		"file_hash_sha1": 365.0,
		"file_hash_sha256": 365.0,
		"certificate": 180.0,
		"network_signature": 90.0,
		"yara_rule": 365.0,
		"email_address": 45.0,
		"user_agent": 30.0,
		"mutex": 180.0,
		"registry_key": 365.0,
	}
	half_life = half_lives.get(indicator_type, 90.0)
	now = datetime.now(timezone.utc)
	if valid_until:
		vu = valid_until.replace(tzinfo=timezone.utc) if valid_until.tzinfo is None else valid_until
		if vu < now:
			return 1.0
	vf = valid_from.replace(tzinfo=timezone.utc) if valid_from.tzinfo is None else valid_from
	age_days = max((now - vf).total_seconds() / 86400.0, 0.0)
	stale = 1.0 - decay_factor(age_days, half_life)
	return round(min(1.0, max(0.0, stale)), 4)


def admiralty_confidence(
	source_reliability: float,
	information_credibility: float,
	corroboration_count: int = 0,
	age_days: float = 0.0,
	max_age_days: float = 365.0,
) -> float:
	"""Admiralty system-inspired confidence score in [0.0, 1.0].

	source_reliability: 0.0 (untested) to 1.0 (completely reliable)
	information_credibility: 0.0 (improbable) to 1.0 (confirmed)
	corroboration_count: number of independent corroborating sources
	age_days: days since the information was collected
	max_age_days: age at which score decays to near zero
	"""
	assert 0.0 <= source_reliability <= 1.0, "source_reliability must be in [0,1]"
	assert 0.0 <= information_credibility <= 1.0, "information_credibility must be in [0,1]"

	base = (source_reliability + information_credibility) / 2.0
	corroboration_bonus = min(corroboration_count * 0.05, 0.20)
	recency = decay_factor(age_days, max(max_age_days / 2.0, 1.0))
	raw = base * (1.0 + corroboration_bonus) * recency
	return round(min(1.0, max(0.0, raw)), 4)


def attribution_confidence(evidence_scores: list[float]) -> float:
	"""Geometric mean of evidence confidence scores with corroboration penalty.

	Returns 0.0 for empty input. Penalises fewer than 3 pieces of evidence.
	"""
	if not evidence_scores:
		return 0.0
	clamped = [max(min(float(s), 1.0), 1e-9) for s in evidence_scores]
	geo_mean = math.exp(sum(math.log(s) for s in clamped) / len(clamped))
	corroboration = min(1.0, len(evidence_scores) / 3.0)
	return round(min(1.0, geo_mean * corroboration), 4)


def threat_risk_score(
	actor_confidence: float,
	campaign_active: bool,
	indicator_count: int,
	max_indicator_count: int = 100,
	critical_indicator_present: bool = False,
) -> float:
	"""Composite threat risk score in [0.0, 1.0].

	Higher score = more immediate risk.
	"""
	base = actor_confidence
	activity_bonus = 0.15 if campaign_active else 0.0
	ioc_density = min(indicator_count / max(max_indicator_count, 1), 1.0) * 0.20
	critical_bonus = 0.25 if critical_indicator_present else 0.0
	raw = base + activity_bonus + ioc_density + critical_bonus
	return round(min(1.0, max(0.0, raw)), 4)


def feed_ingestion_quality_score(
	indicators_ingested: int,
	false_positives: int,
	stale_count: int,
	total: int,
) -> float:
	"""Quality score of a threat feed in [0.0, 1.0].

	Penalises false positives and stale indicators.
	"""
	if total <= 0:
		return 0.0
	valid = max(total - false_positives - stale_count, 0)
	return round(valid / total, 4)


def mitre_coverage_percentage(
	observed_techniques: set[str],
	total_techniques: int = 201,  # ATT&CK Enterprise v14 technique count
) -> float:
	"""Percentage of MITRE ATT&CK techniques covered by observed intelligence."""
	if total_techniques <= 0:
		return 0.0
	return round(min(1.0, len(observed_techniques) / total_techniques) * 100.0, 2)


def days_since(dt: datetime) -> float:
	"""Return days elapsed since *dt* (UTC-aware)."""
	now = datetime.now(timezone.utc)
	ref = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
	return max((now - ref).total_seconds() / 86400.0, 0.0)


def indicator_priority_score(
	confidence_score: float,
	staleness_score: float,
	actor_threat_level: float,
	campaign_active: bool,
) -> float:
	"""Triage priority score for analyst queue ordering in [0.0, 1.0].

	High priority = high confidence, low staleness, high-threat actor, active campaign.
	"""
	freshness = 1.0 - staleness_score
	activity = 0.2 if campaign_active else 0.0
	raw = (confidence_score * 0.35 + freshness * 0.35 + actor_threat_level * 0.20 + activity)
	return round(min(1.0, max(0.0, raw)), 4)


def confidence_score_breakdown(
	source_reliability: float,
	information_credibility: float,
	corroboration_count: int,
	age_days: float,
	max_age_days: float = 365.0,
) -> dict[str, float]:
	"""Return a fully factored breakdown of confidence score components.

	Useful for explaining scoring to analysts and for auditability.
	"""
	base = (source_reliability + information_credibility) / 2.0
	corroboration_bonus = min(corroboration_count * 0.05, 0.20)
	recency = decay_factor(age_days, max(max_age_days / 2.0, 1.0))
	raw = base * (1.0 + corroboration_bonus) * recency
	final = round(min(1.0, max(0.0, raw)), 4)

	return {
		"source_reliability": round(source_reliability, 4),
		"information_credibility": round(information_credibility, 4),
		"base_score": round(base, 4),
		"corroboration_bonus": round(corroboration_bonus, 4),
		"recency_factor": round(recency, 4),
		"final_score": final,
	}


def stix_object_confidence(
	stix_obj: dict[str, Any],
	feed_weight: float = 0.8,
) -> float:
	"""Extract or estimate a confidence score from a STIX 2.1 object.

	STIX confidence is expressed as 0-100; we normalise to [0.0, 1.0].
	If not present, uses the feed weight as a default.
	"""
	raw = stix_obj.get("confidence")
	if raw is not None:
		try:
			return round(min(1.0, max(0.0, float(raw) / 100.0)) * feed_weight, 4)
		except (TypeError, ValueError):
			pass
	return round(max(0.0, min(1.0, feed_weight)), 4)


def campaign_duration_days(first_seen: datetime | None, last_seen: datetime | None) -> float | None:
	"""Return campaign duration in days, or None if insufficient data."""
	if first_seen is None or last_seen is None:
		return None
	fs = first_seen.replace(tzinfo=timezone.utc) if first_seen.tzinfo is None else first_seen
	ls = last_seen.replace(tzinfo=timezone.utc) if last_seen.tzinfo is None else last_seen
	delta = (ls - fs).total_seconds() / 86400.0
	return round(max(0.0, delta), 2)


def weighted_actor_confidence(
	direct_indicator_scores: list[float],
	campaign_risk_weight: float,
	attribution_evidence_scores: list[float],
) -> float:
	"""Weighted composite actor confidence score.

	Blends:
	  - Mean of direct indicator confidence scores (40%)
	  - Campaign risk weight normalised to [0,1] (20%)
	  - Attribution evidence geometric mean (40%)

	Returns float in [0.0, 1.0].
	"""
	ioc_mean = (
		sum(direct_indicator_scores) / len(direct_indicator_scores)
		if direct_indicator_scores else 0.0
	)
	attr_conf = attribution_confidence(attribution_evidence_scores) if attribution_evidence_scores else 0.0
	campaign_w = max(0.0, min(1.0, campaign_risk_weight))
	raw = ioc_mean * 0.40 + campaign_w * 0.20 + attr_conf * 0.40
	return round(min(1.0, max(0.0, raw)), 4)
