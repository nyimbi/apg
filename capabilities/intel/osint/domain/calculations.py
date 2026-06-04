"""Domain calculations for Open Source Intelligence.

All formulas are pure functions with type-safe inputs.  No I/O, no side
effects — the service layer owns orchestration.
"""

from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Timeliness / freshness
# ---------------------------------------------------------------------------

def calculate_timeliness_score(captured_at: datetime, reference_now: datetime | None = None) -> float:
	"""Freshness score in [0.0, 1.0].  Half-life of 7 days — item older than
	60 days scores near zero.

	Args:
		captured_at: UTC datetime the intelligence was captured.
		reference_now: Optional override for 'now' (useful in tests).

	Returns:
		float in [0.0, 1.0].
	"""
	now = reference_now or datetime.now(timezone.utc)
	if captured_at.tzinfo is None:
		captured_at = captured_at.replace(tzinfo=timezone.utc)
	age_days = max((now - captured_at).total_seconds() / 86400.0, 0.0)
	# Exponential decay: half-life = 7 days
	half_life = 7.0
	score = math.exp(-math.log(2) * age_days / half_life)
	return round(min(max(score, 0.0), 1.0), 4)


# ---------------------------------------------------------------------------
# Content fingerprinting
# ---------------------------------------------------------------------------

def compute_content_fingerprint(content: str, algorithm: str = "sha256") -> str:
	"""Deterministic content fingerprint for deduplication.

	Args:
		content: Raw text/bytes content as string.
		algorithm: Hash algorithm name understood by hashlib.

	Returns:
		Hex digest string.
	"""
	h = hashlib.new(algorithm)
	h.update(content.encode("utf-8", errors="replace"))
	return h.hexdigest()


def compute_url_fingerprint(url: str, query_params_to_ignore: list[str] | None = None) -> str:
	"""Canonical URL fingerprint that strips volatile query parameters
	(e.g. session tokens, tracking IDs) before hashing.
	"""
	from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

	ignore = set(query_params_to_ignore or ["utm_source", "utm_medium", "utm_campaign", "session", "token"])
	parsed = urlparse(url.strip().lower())
	params = {k: v for k, v in parse_qs(parsed.query).items() if k not in ignore}
	clean_query = urlencode(sorted(params.items()), doseq=True)
	canonical = urlunparse(parsed._replace(query=clean_query, fragment=""))
	return compute_content_fingerprint(canonical)


# ---------------------------------------------------------------------------
# Composite credibility
# ---------------------------------------------------------------------------

def composite_intel_credibility(
	source_credibility: float,
	corroboration_count: int,
	analyst_confidence: float,
	captured_at: datetime,
	reference_now: datetime | None = None,
) -> dict[str, float]:
	"""Full composite credibility breakdown for a processed intel item.

	Returns a dict with individual factor scores plus the composite.
	Weights: source=0.40, corroboration=0.25, analyst=0.25, timeliness=0.10.
	"""
	timeliness = calculate_timeliness_score(captured_at, reference_now)
	corroboration_factor = min(corroboration_count / 5.0, 1.0)

	composite = (
		source_credibility * 0.40
		+ corroboration_factor * 0.25
		+ min(max(analyst_confidence, 0.0), 1.0) * 0.25
		+ timeliness * 0.10
	)
	composite = round(min(max(composite, 0.0), 1.0), 4)

	return {
		"composite": composite,
		"source_credibility": round(source_credibility, 4),
		"corroboration_factor": round(corroboration_factor, 4),
		"analyst_confidence": round(min(max(analyst_confidence, 0.0), 1.0), 4),
		"timeliness": round(timeliness, 4),
	}


# ---------------------------------------------------------------------------
# Entity de-duplication similarity
# ---------------------------------------------------------------------------

def name_similarity(name_a: str, name_b: str) -> float:
	"""Simple Jaro-Winkler inspired token-level similarity in [0.0, 1.0].

	For production use, replace with `rapidfuzz.distance.JaroWinkler.similarity`.
	This implementation is stdlib-only for zero-dependency portability.
	"""
	a = set(name_a.lower().split())
	b = set(name_b.lower().split())
	if not a or not b:
		return 0.0
	intersection = len(a & b)
	union = len(a | b)
	return round(intersection / union, 4) if union else 0.0


def deduplicate_entities(
	entities: list[dict[str, Any]],
	similarity_threshold: float = 0.80,
) -> list[dict[str, Any]]:
	"""Greedy deduplication: merge entities whose names exceed the threshold.

	Returns the deduplicated list; merging strategy is union of aliases.
	"""
	merged: list[dict[str, Any]] = []
	used: set[int] = set()

	for i, entity in enumerate(entities):
		if i in used:
			continue
		cluster = entity.copy()
		aliases: set[str] = set(entity.get("aliases", []))
		for j, other in enumerate(entities):
			if j <= i or j in used:
				continue
			sim = name_similarity(entity["name"], other["name"])
			if sim >= similarity_threshold:
				used.add(j)
				aliases.add(other["name"])
				aliases.update(other.get("aliases", []))
				# Keep higher confidence score
				if other.get("confidence_score", 0.0) > cluster.get("confidence_score", 0.0):
					cluster["confidence_score"] = other["confidence_score"]
		cluster["aliases"] = sorted(aliases - {cluster["name"]})
		merged.append(cluster)

	return merged


# ---------------------------------------------------------------------------
# Network / relationship graph metrics
# ---------------------------------------------------------------------------

def calculate_entity_centrality(
	entity_id: str,
	relationships: list[dict[str, Any]],
) -> dict[str, float]:
	"""Degree, in-degree and out-degree centrality for an entity.

	relationships: list of dicts with keys 'source_entity_id', 'target_entity_id'.
	"""
	out_degree = sum(1 for r in relationships if r.get("source_entity_id") == entity_id)
	in_degree = sum(1 for r in relationships if r.get("target_entity_id") == entity_id)
	total = len(relationships)
	normaliser = total if total else 1

	return {
		"entity_id": entity_id,
		"in_degree": in_degree,
		"out_degree": out_degree,
		"total_degree": in_degree + out_degree,
		"in_degree_centrality": round(in_degree / normaliser, 4),
		"out_degree_centrality": round(out_degree / normaliser, 4),
		"degree_centrality": round((in_degree + out_degree) / (2 * normaliser), 4),
	}


def find_connected_clusters(
	entity_ids: list[str],
	relationships: list[dict[str, Any]],
) -> list[list[str]]:
	"""Union-Find connected components over the entity relationship graph.

	Returns a list of clusters (lists of entity IDs).
	"""
	parent = {eid: eid for eid in entity_ids}

	def find(x: str) -> str:
		while parent[x] != x:
			parent[x] = parent[parent[x]]
			x = parent[x]
		return x

	def union(x: str, y: str) -> None:
		px, py = find(x), find(y)
		if px != py:
			parent[px] = py

	for rel in relationships:
		src = rel.get("source_entity_id", "")
		tgt = rel.get("target_entity_id", "")
		if src in parent and tgt in parent:
			union(src, tgt)

	clusters: dict[str, list[str]] = {}
	for eid in entity_ids:
		root = find(eid)
		clusters.setdefault(root, []).append(eid)

	return sorted(clusters.values(), key=len, reverse=True)


# ---------------------------------------------------------------------------
# IP reputation scoring
# ---------------------------------------------------------------------------

def calculate_ip_threat_score(
	is_tor: bool,
	is_vpn: bool,
	is_proxy: bool,
	is_datacenter: bool,
	abuse_reports: int,
	open_ports_count: int,
) -> float:
	"""Aggregate IP threat score in [0.0, 1.0].

	Higher = more suspicious.
	"""
	score = 0.0
	if is_tor:
		score += 0.40
	if is_vpn:
		score += 0.20
	if is_proxy:
		score += 0.15
	if is_datacenter:
		score += 0.05
	score += min(abuse_reports / 100.0, 1.0) * 0.15
	score += min(open_ports_count / 50.0, 1.0) * 0.05
	return round(min(max(score, 0.0), 1.0), 4)


# ---------------------------------------------------------------------------
# Domain age helper
# ---------------------------------------------------------------------------

def domain_age_years(created_date: datetime | None, reference_now: datetime | None = None) -> float:
	"""Age of a domain in decimal years.  Returns 0.0 if created_date is None."""
	if created_date is None:
		return 0.0
	now = reference_now or datetime.now(timezone.utc)
	if created_date.tzinfo is None:
		created_date = created_date.replace(tzinfo=timezone.utc)
	delta = now - created_date
	return max(delta.total_seconds() / (365.25 * 86400.0), 0.0)


# ---------------------------------------------------------------------------
# Sentiment interpretation
# ---------------------------------------------------------------------------

def interpret_sentiment(score: float) -> str:
	"""Map a sentiment score in [-1.0, 1.0] to a human label."""
	if score >= 0.5:
		return "strongly_positive"
	if score >= 0.1:
		return "positive"
	if score > -0.1:
		return "neutral"
	if score > -0.5:
		return "negative"
	return "strongly_negative"
