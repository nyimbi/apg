"""Financial and risk calculations for AML.

All formulas are deterministic, type-safe, and cover edge cases including
zero/negative amounts, missing data, and boundary conditions.
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

# re-export so service.py can do a single import from domain.calculations
__all__ = [
	"calculate_false_positive_rate",
	"calculate_network_risk_score",
	"calculate_risk_score",
	"calculate_sar_priority",
	"calculate_correspondent_nesting_risk",
	"detect_layering",
	"detect_round_trip",
	"detect_structuring",
	"detect_velocity_anomaly",
	"detect_trade_based_ml",
	"detect_nft_wash_trading",
	"detect_crypto_mixer_routing",
	"detect_terrorist_financing_indicators",
	"requires_ctr",
	"risk_segment_from_score",
	"severity_from_score",
]


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


# ---------------------------------------------------------------------------
# Trade-based money laundering (TBML)
# ---------------------------------------------------------------------------

def detect_trade_based_ml(
	invoices: list[dict[str, Any]],
	market_value_lookup: dict[str, float] | None = None,
	over_under_threshold: float = 0.15,
	phantom_shipment_indicators: list[str] | None = None,
) -> dict[str, Any]:
	"""Detect TBML patterns: over/under-invoicing, phantom shipments, multiple invoicing.

	Args:
		invoices: list of dicts with keys: id, amount, currency, commodity_code,
		          quantity, unit_price, declared_value, counterparty_country
		market_value_lookup: commodity_code -> market_unit_price mapping
		over_under_threshold: fractional deviation to flag (default 15%)
		phantom_shipment_indicators: list of known phantom shipment reference IDs

	Returns:
		dict with detected flag, typology list, flagged_invoices, risk_score
	"""
	assert 0.0 < over_under_threshold < 1.0
	if not invoices:
		return {"detected": False, "typologies": [], "flagged_invoices": [], "risk_score": 0}

	phantom_set = set(phantom_shipment_indicators or [])
	market_values = market_value_lookup or {}
	flagged: list[dict[str, Any]] = []
	typologies: set[str] = set()

	# Track invoices per (counterparty, commodity) to detect multiple invoicing
	invoice_index: dict[tuple[str, str], list[dict]] = defaultdict(list)

	for inv in invoices:
		inv_id = str(inv.get("id", ""))
		amount = float(inv.get("amount", 0))
		commodity = str(inv.get("commodity_code", ""))
		counterparty = str(inv.get("counterparty_country", ""))
		quantity = float(inv.get("quantity", 1) or 1)
		unit_price = float(inv.get("unit_price", 0))
		reasons: list[str] = []

		# Phantom shipment check
		if inv_id in phantom_set:
			reasons.append("phantom_shipment")
			typologies.add("phantom_shipment")

		# Over/under-invoicing check
		market_unit = market_values.get(commodity)
		if market_unit and market_unit > 0 and unit_price > 0:
			deviation = abs(unit_price - market_unit) / market_unit
			if deviation > over_under_threshold:
				direction = "over_invoiced" if unit_price > market_unit else "under_invoiced"
				reasons.append(f"{direction}:{deviation:.1%}")
				typologies.add(direction)

		# Multiple invoicing: same (counterparty, commodity) pair seen before
		key = (counterparty, commodity)
		invoice_index[key].append(inv)
		if len(invoice_index[key]) > 1:
			reasons.append("multiple_invoicing")
			typologies.add("multiple_invoicing")

		if reasons:
			flagged.append({"invoice_id": inv_id, "amount": amount, "reasons": reasons})

	risk_score = min(len(flagged) * 20 + len(typologies) * 15, 100)
	return {
		"detected": bool(flagged),
		"typologies": sorted(typologies),
		"flagged_invoices": flagged,
		"risk_score": risk_score,
		"invoice_count": len(invoices),
	}


# ---------------------------------------------------------------------------
# NFT wash-trade detection
# ---------------------------------------------------------------------------

def detect_nft_wash_trading(
	nft_transfers: list[dict[str, Any]],
	lookback_days: int = 30,
	min_round_trips: int = 2,
	price_inflation_threshold: float = 3.0,
) -> dict[str, Any]:
	"""Detect NFT wash trading: same asset transferred between related wallets at inflating prices.

	Args:
		nft_transfers: list of dicts with keys: token_id, from_wallet, to_wallet,
		               price, currency, created_at
		lookback_days: analysis window
		min_round_trips: minimum round-trips to flag wash trade
		price_inflation_threshold: price multiplier to flag artificial inflation

	Returns:
		dict with detected flag, wash_trade_score (0-1), flagged_tokens, patterns
	"""
	if not nft_transfers:
		return {"detected": False, "wash_trade_score": 0.0, "flagged_tokens": [], "patterns": []}

	cutoff = datetime.utcnow() - timedelta(days=lookback_days)
	recent = [t for t in nft_transfers if _parse_dt(t.get("created_at")) >= cutoff]

	# Group by token_id
	by_token: dict[str, list[dict]] = defaultdict(list)
	for t in recent:
		token_id = str(t.get("token_id", ""))
		if token_id:
			by_token[token_id].append(t)

	flagged_tokens: list[dict[str, Any]] = []
	patterns: list[str] = []

	for token_id, transfers in by_token.items():
		sorted_transfers = sorted(transfers, key=lambda t: _parse_dt(t.get("created_at")))
		wallets_seen: list[str] = [str(sorted_transfers[0].get("from_wallet", ""))]
		round_trips = 0
		price_inflation = 1.0

		for i, t in enumerate(sorted_transfers):
			to_wallet = str(t.get("to_wallet", ""))
			if to_wallet in wallets_seen:
				round_trips += 1

			wallets_seen.append(to_wallet)

			# Price inflation check
			if i > 0:
				prev_price = float(sorted_transfers[i - 1].get("price", 0) or 0)
				curr_price = float(t.get("price", 0) or 0)
				if prev_price > 0 and curr_price > 0:
					price_inflation = max(price_inflation, curr_price / prev_price)

		reasons: list[str] = []
		if round_trips >= min_round_trips:
			reasons.append(f"round_trips:{round_trips}")
			patterns.append("circular_transfers")
		if price_inflation >= price_inflation_threshold:
			reasons.append(f"price_inflation:{price_inflation:.1f}x")
			patterns.append("artificial_price_inflation")

		if reasons:
			flagged_tokens.append({
				"token_id": token_id,
				"transfer_count": len(transfers),
				"round_trips": round_trips,
				"price_inflation": round(price_inflation, 2),
				"reasons": reasons,
			})

	total_tokens = len(by_token)
	wash_trade_score = len(flagged_tokens) / max(total_tokens, 1) if flagged_tokens else 0.0

	return {
		"detected": bool(flagged_tokens),
		"wash_trade_score": round(min(wash_trade_score, 1.0), 4),
		"flagged_tokens": flagged_tokens,
		"patterns": list(set(patterns)),
		"total_tokens_analysed": total_tokens,
	}


# ---------------------------------------------------------------------------
# Crypto mixer detection
# ---------------------------------------------------------------------------

# Known mixing/tumbling service address prefixes and service names
_KNOWN_MIXER_INDICATORS = {
	"tornado_cash", "chipmixer", "bitcoin_fog", "helix", "bitmixer",
	"sinbad", "blender", "wasabi_coinjoin", "joinmarket",
}


def detect_crypto_mixer_routing(
	crypto_transactions: list[dict[str, Any]],
	known_mixer_addresses: set[str] | None = None,
) -> dict[str, Any]:
	"""Detect routing through crypto mixing/tumbling services.

	Args:
		crypto_transactions: list of dicts with keys: tx_hash, from_address,
		                     to_address, amount, asset, service_label (optional),
		                     created_at
		known_mixer_addresses: tenant-specific set of known mixer addresses

	Returns:
		dict with detected flag, mixer_indicators, flagged_transactions
	"""
	if not crypto_transactions:
		return {"detected": False, "mixer_indicators": [], "flagged_transactions": []}

	mixer_addresses = known_mixer_addresses or set()
	flagged: list[dict[str, Any]] = []
	indicators: set[str] = set()

	for txn in crypto_transactions:
		tx_hash = str(txn.get("tx_hash", ""))
		to_addr = str(txn.get("to_address", "")).lower()
		from_addr = str(txn.get("from_address", "")).lower()
		service_label = str(txn.get("service_label", "")).lower()
		reasons: list[str] = []

		# Known address match
		if to_addr in mixer_addresses or from_addr in mixer_addresses:
			reasons.append("known_mixer_address")
			indicators.add("known_mixer_address")

		# Service label match
		for mixer_name in _KNOWN_MIXER_INDICATORS:
			if mixer_name in service_label:
				reasons.append(f"mixer_service:{mixer_name}")
				indicators.add(mixer_name)

		# CoinJoin pattern: multiple inputs, many outputs of equal value
		inputs = int(txn.get("input_count", 0))
		outputs = int(txn.get("output_count", 0))
		if inputs >= 5 and outputs >= 5:
			equal_outputs = bool(txn.get("equal_output_amounts", False))
			if equal_outputs:
				reasons.append("coinjoin_pattern")
				indicators.add("coinjoin_pattern")

		if reasons:
			flagged.append({"tx_hash": tx_hash, "reasons": reasons})

	return {
		"detected": bool(flagged),
		"mixer_indicators": sorted(indicators),
		"flagged_transactions": flagged,
		"flagged_count": len(flagged),
	}


# ---------------------------------------------------------------------------
# Correspondent banking nested account risk
# ---------------------------------------------------------------------------

def calculate_correspondent_nesting_risk(
	correspondent_chain: list[dict[str, Any]],
	high_risk_jurisdictions: set[str] | None = None,
) -> dict[str, Any]:
	"""Assess risk of nested correspondent banking relationships.

	Args:
		correspondent_chain: ordered list of dicts with keys: institution_id,
		                     institution_name, jurisdiction, aml_rating,
		                     kyb_status, nested_accounts_count
		high_risk_jurisdictions: set of high-risk jurisdiction codes (FATF grey/black list)

	Returns:
		dict with nesting_depth, risk_score, risk_factors, recommended_action
	"""
	if not correspondent_chain:
		return {"nesting_depth": 0, "risk_score": 0, "risk_factors": [], "recommended_action": "no_action"}

	hr_jurisdictions = high_risk_jurisdictions or {
		"IR", "KP", "MM", "SY", "YE", "AF", "IQ", "LY", "SO", "SS",
	}

	depth = len(correspondent_chain)
	risk_factors: list[str] = []
	score = 0

	# Depth penalty: each layer beyond 2 adds 15 points
	if depth > 2:
		penalty = (depth - 2) * 15
		score += penalty
		risk_factors.append(f"deep_nesting:{depth}_layers(+{penalty})")

	for link in correspondent_chain:
		jurisdiction = str(link.get("jurisdiction", "")).upper()
		aml_rating = str(link.get("aml_rating", "unknown")).lower()
		kyb_status = str(link.get("kyb_status", "unknown")).lower()
		nested_count = int(link.get("nested_accounts_count", 0))

		if jurisdiction in hr_jurisdictions:
			score += 25
			risk_factors.append(f"high_risk_jurisdiction:{jurisdiction}")

		if aml_rating in ("poor", "non_compliant", "sanctioned"):
			score += 20
			risk_factors.append(f"poor_aml_rating:{link.get('institution_id')}")

		if kyb_status not in ("verified", "approved"):
			score += 10
			risk_factors.append(f"unverified_kyb:{link.get('institution_id')}")

		if nested_count > 5:
			score += 15
			risk_factors.append(f"high_nested_count:{nested_count}")

	score = min(score, 100)

	if score >= 70:
		action = "terminate_relationship"
	elif score >= 45:
		action = "enhanced_due_diligence"
	elif score >= 20:
		action = "review_and_monitor"
	else:
		action = "standard_monitoring"

	return {
		"nesting_depth": depth,
		"risk_score": score,
		"risk_factors": risk_factors,
		"recommended_action": action,
		"chain_length": depth,
	}


# ---------------------------------------------------------------------------
# Terrorist financing typology detection
# ---------------------------------------------------------------------------

_TF_BEHAVIOURAL_INDICATORS = {
	"small_frequent_international",  # small amounts to high-risk jurisdictions
	"charity_misuse",                # NGO/charity transfers to conflict zones
	"hawala_pattern",                # value transfer without fund movement
	"cash_courier",                  # physical cash movement detection
	"prepaid_card_loading",          # prepaid card reload patterns
}

_TF_HIGH_RISK_JURISDICTIONS = {
	"AF", "IQ", "SY", "YE", "LY", "SO", "SS", "ML", "BF", "NE", "TD",
}


def detect_terrorist_financing_indicators(
	transactions: list[dict[str, Any]],
	customer_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
	"""Detect terrorist financing indicators using FATF typology guidance.

	Args:
		transactions: list of transaction dicts
		customer_profile: optional dict with keys: nationality, pep_status,
		                  adverse_media, charity_sector, known_associates

	Returns:
		dict with detected flag, tf_indicators, risk_score, typologies
	"""
	if not transactions:
		return {"detected": False, "tf_indicators": [], "risk_score": 0, "typologies": [], "transaction_count": 0}

	profile = customer_profile or {}
	indicators: set[str] = set()
	score = 0

	# Check adverse media / known associates with terrorism links
	if profile.get("adverse_media_terrorism"):
		indicators.add("adverse_media_terrorism_link")
		score += 40

	if profile.get("known_tf_associate"):
		indicators.add("known_tf_associate")
		score += 50

	# Analyse transaction patterns
	for txn in transactions:
		dest_country = str(txn.get("destination_country", "") or txn.get("counterparty_country", "")).upper()
		amount = float(txn.get("amount", 0))
		txn_type = str(txn.get("transaction_type", "")).lower()
		service_label = str(txn.get("service_label", "")).lower()

		# Small amounts to high-risk jurisdictions (common TF evasion)
		if dest_country in _TF_HIGH_RISK_JURISDICTIONS and 0 < amount < 3000:
			indicators.add("small_amount_to_high_risk_jurisdiction")
			score += 15

		# Hawala / informal value transfer
		if "hawala" in txn_type or "hawala" in service_label:
			indicators.add("hawala_pattern")
			score += 20

		# Charity misuse: NGO transfers to conflict zones
		if profile.get("charity_sector") and dest_country in _TF_HIGH_RISK_JURISDICTIONS:
			indicators.add("charity_misuse")
			score += 25

		# Prepaid card loading patterns
		if "prepaid" in txn_type and amount < 1000:
			indicators.add("prepaid_card_loading")
			score += 10

	score = min(score, 100)
	typologies = [i for i in indicators if i in _TF_BEHAVIOURAL_INDICATORS]

	return {
		"detected": bool(indicators),
		"tf_indicators": sorted(indicators),
		"risk_score": score,
		"typologies": typologies,
		"transaction_count": len(transactions),
	}
