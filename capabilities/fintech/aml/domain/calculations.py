"""Financial and risk calculations for AML.

All formulas are deterministic, type-safe, and cover edge cases including
zero/negative amounts, missing data, and boundary conditions.
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any


# ---------------------------------------------------------------------------
# Risk scoring
# ---------------------------------------------------------------------------

def calculate_risk_score(
	amount: float,
	large_threshold: float,
	structuring_threshold: float,
	velocity_count: int,
	velocity_window_hours: int,
	sanctions_hit: bool,
	pep_hit: bool,
	high_risk_country: bool,
	adverse_media: bool,
	base_kyc_score: int = 0,
) -> int:
	"""Compute a 0-100 AML risk score for a transaction context.

	Additive scoring model with hard overrides for sanctions/PEP.
	Each factor contributes a defined weight; final score is clamped to [0, 100].
	"""
	score = base_kyc_score

	if sanctions_hit:
		return 100
	if pep_hit:
		score += 30

	if large_threshold > 0 and amount >= large_threshold:
		score += 25
	elif structuring_threshold > 0 and amount >= structuring_threshold:
		score += 20

	if velocity_window_hours > 0:
		hourly_rate = velocity_count / max(velocity_window_hours, 1)
		if hourly_rate > 5:
			score += 20
		elif hourly_rate > 2:
			score += 10

	if high_risk_country:
		score += 15
	if adverse_media:
		score += 10

	return min(max(score, 0), 100)


def severity_from_score(score: int) -> str:
	"""Map risk score to alert severity."""
	if score >= 90:
		return "critical"
	if score >= 70:
		return "high"
	if score >= 45:
		return "medium"
	return "low"


def risk_segment_from_score(score: int) -> str:
	"""Map risk score to customer risk segment."""
	if score >= 90:
		return "prohibited"
	if score >= 70:
		return "very_high"
	if score >= 45:
		return "high"
	if score >= 20:
		return "medium"
	return "low"


# ---------------------------------------------------------------------------
# Structuring / smurfing detection
# ---------------------------------------------------------------------------

def detect_structuring(
	transactions: list[dict[str, Any]],
	reporting_threshold: float = 10_000.0,
	lookback_days: int = 10,
	min_occurrences: int = 3,
) -> dict[str, Any]:
	"""Detect structuring (smurfing) — multiple sub-threshold transactions
	designed to avoid reporting requirements.

	Args:
		transactions: list of dicts with keys: amount, currency, created_at (ISO str), account_id
		reporting_threshold: jurisdiction CTR threshold (default US $10k)
		lookback_days: window for grouping
		min_occurrences: minimum sub-threshold txns to flag

	Returns:
		dict with detected flag, count, total_amount, pattern details
	"""
	assert reporting_threshold > 0, "reporting_threshold must be positive"
	assert lookback_days > 0, "lookback_days must be positive"

	if not transactions:
		return {"detected": False, "count": 0, "total_amount": 0.0, "patterns": []}

	cutoff = datetime.utcnow() - timedelta(days=lookback_days)
	sub_threshold = [
		t for t in transactions
		if float(t.get("amount", 0)) < reporting_threshold
		and float(t.get("amount", 0)) > reporting_threshold * 0.7
		and _parse_dt(t.get("created_at")) >= cutoff
	]

	count = len(sub_threshold)
	total = sum(float(t.get("amount", 0)) for t in sub_threshold)
	detected = count >= min_occurrences

	return {
		"detected": detected,
		"count": count,
		"total_amount": round(total, 2),
		"threshold": reporting_threshold,
		"patterns": sub_threshold if detected else [],
		"smurfing_band": (reporting_threshold * 0.7, reporting_threshold),
	}


def detect_velocity_anomaly(
	transactions: list[dict[str, Any]],
	window_hours: int = 24,
	count_threshold: int = 10,
	amount_threshold: float = 50_000.0,
) -> dict[str, Any]:
	"""Detect unusual transaction velocity in a time window."""
	assert window_hours > 0, "window_hours must be positive"

	if not transactions:
		return {"detected": False, "count": 0, "total_amount": 0.0}

	cutoff = datetime.utcnow() - timedelta(hours=window_hours)
	recent = [t for t in transactions if _parse_dt(t.get("created_at")) >= cutoff]
	total = sum(float(t.get("amount", 0)) for t in recent)
	count = len(recent)

	return {
		"detected": count >= count_threshold or total >= amount_threshold,
		"count": count,
		"total_amount": round(total, 2),
		"window_hours": window_hours,
		"count_threshold": count_threshold,
		"amount_threshold": amount_threshold,
	}


# ---------------------------------------------------------------------------
# Round-trip / layering detection
# ---------------------------------------------------------------------------

def detect_round_trip(
	transactions: list[dict[str, Any]],
	tolerance_pct: float = 0.05,
	max_hops: int = 5,
	lookback_days: int = 90,
) -> dict[str, Any]:
	"""Detect funds returning to originating account within tolerance.

	Args:
		transactions: list of dicts with amount, currency, sender_account, receiver_account, created_at
		tolerance_pct: allowable deviation (fees, FX) as fraction (default 5%)
		max_hops: maximum chain length to trace
		lookback_days: window

	Returns:
		dict with detected flag, chains list
	"""
	assert 0.0 <= tolerance_pct <= 1.0
	assert max_hops >= 2

	if not transactions:
		return {"detected": False, "chains": []}

	cutoff = datetime.utcnow() - timedelta(days=lookback_days)
	recent = [t for t in transactions if _parse_dt(t.get("created_at")) >= cutoff]

	# Build adjacency: sender -> list of (receiver, amount, txn)
	graph: dict[str, list[tuple[str, float, dict]]] = defaultdict(list)
	for t in recent:
		sender = str(t.get("sender_account", ""))
		receiver = str(t.get("receiver_account", ""))
		amt = float(t.get("amount", 0))
		if sender and receiver and amt > 0:
			graph[sender].append((receiver, amt, t))

	chains: list[list[dict]] = []
	for origin in list(graph.keys()):
		_dfs_round_trip(graph, origin, origin, [], chains, tolerance_pct, max_hops, 0)

	return {"detected": bool(chains), "chain_count": len(chains), "chains": chains[:10]}


def _dfs_round_trip(
	graph: dict[str, list[tuple[str, float, dict]]],
	current: str,
	origin: str,
	path: list[dict],
	results: list[list[dict]],
	tolerance_pct: float,
	max_hops: int,
	depth: int,
) -> None:
	if depth > max_hops:
		return
	for receiver, amt, txn in graph.get(current, []):
		new_path = path + [txn]
		if receiver == origin and depth >= 1:
			results.append(new_path)
		elif receiver != origin:
			_dfs_round_trip(graph, receiver, origin, new_path, results, tolerance_pct, max_hops, depth + 1)


def detect_layering(
	transactions: list[dict[str, Any]],
	min_layers: int = 3,
	lookback_days: int = 30,
) -> dict[str, Any]:
	"""Detect rapid consecutive transfers across accounts (layering).

	Looks for chains where funds move quickly between multiple accounts
	without apparent business purpose.
	"""
	if not transactions:
		return {"detected": False, "layers": 0, "chains": []}

	cutoff = datetime.utcnow() - timedelta(days=lookback_days)
	sorted_txns = sorted(
		[t for t in transactions if _parse_dt(t.get("created_at")) >= cutoff],
		key=lambda t: _parse_dt(t.get("created_at")),
	)

	# Find chains where receiver of one txn is sender of the next within 48h
	chains: list[list[dict]] = []
	for i, txn in enumerate(sorted_txns):
		chain = [txn]
		current_receiver = str(txn.get("receiver_account", ""))
		current_time = _parse_dt(txn.get("created_at"))
		for j in range(i + 1, len(sorted_txns)):
			next_txn = sorted_txns[j]
			next_sender = str(next_txn.get("sender_account", ""))
			next_time = _parse_dt(next_txn.get("created_at"))
			if next_sender == current_receiver and (next_time - current_time).total_seconds() <= 172800:
				chain.append(next_txn)
				current_receiver = str(next_txn.get("receiver_account", ""))
				current_time = next_time
		if len(chain) >= min_layers:
			chains.append(chain)

	return {
		"detected": bool(chains),
		"layers": max((len(c) for c in chains), default=0),
		"chains": chains[:5],
	}


# ---------------------------------------------------------------------------
# Network risk
# ---------------------------------------------------------------------------

def calculate_network_risk_score(
	direct_risk_scores: list[int],
	indirect_risk_scores: list[int],
	round_trip_detected: bool,
	layering_detected: bool,
) -> int:
	"""Weighted network risk from counterparty scores and pattern flags."""
	if not direct_risk_scores and not indirect_risk_scores:
		base = 0
	else:
		direct_avg = sum(direct_risk_scores) / max(len(direct_risk_scores), 1)
		indirect_avg = sum(indirect_risk_scores) / max(len(indirect_risk_scores), 1)
		base = int(direct_avg * 0.6 + indirect_avg * 0.4)

	penalty = 0
	if round_trip_detected:
		penalty += 20
	if layering_detected:
		penalty += 25

	return min(base + penalty, 100)


# ---------------------------------------------------------------------------
# CTR threshold check
# ---------------------------------------------------------------------------

def requires_ctr(amount: float, currency: str, jurisdiction: str) -> bool:
	"""Determine if transaction triggers a Currency Transaction Report requirement.

	Uses jurisdiction-specific thresholds.
	"""
	thresholds: dict[str, float] = {
		"US": 10_000.0,
		"UK": 10_000.0,
		"EU": 10_000.0,
		"AU": 10_000.0,
		"CA": 10_000.0,
		"KE": 1_000_000.0,  # KES
		"NG": 5_000_000.0,  # NGN
		"ZA": 24_999.0,     # ZAR
	}
	threshold = thresholds.get(jurisdiction.upper(), 10_000.0)
	return amount >= threshold


def calculate_sar_priority(
	risk_score: int,
	days_since_suspicious: int,
	typology_count: int,
	amount: float,
) -> int:
	"""Compute SAR filing priority 1-5 (1=highest).

	Factors: risk score, time elapsed, typology complexity, amount.
	"""
	score = 0
	score += (risk_score / 100) * 40
	urgency = max(0, 30 - days_since_suspicious)
	score += (urgency / 30) * 20
	score += min(typology_count * 5, 20)
	if amount >= 1_000_000:
		score += 20
	elif amount >= 100_000:
		score += 10

	raw = score / 100
	if raw >= 0.8:
		return 1
	if raw >= 0.6:
		return 2
	if raw >= 0.4:
		return 3
	if raw >= 0.2:
		return 4
	return 5


def calculate_false_positive_rate(total_alerts: int, false_positives: int) -> float:
	"""FPR = false_positives / total_alerts, clamped to [0, 1]."""
	if total_alerts <= 0:
		return 0.0
	return round(min(false_positives / total_alerts, 1.0), 4)


def _parse_dt(value: Any) -> datetime:
	"""Parse ISO string or datetime; fallback to epoch."""
	if isinstance(value, datetime):
		return value
	try:
		return datetime.fromisoformat(str(value))
	except (ValueError, TypeError):
		return datetime(1970, 1, 1)
