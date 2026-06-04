"""KYC domain calculations — risk scoring, expiry, name matching, transliteration.

All functions are pure (no side-effects), type-safe, and handle edge cases
explicitly.  No external dependencies — works in any Python 3.12+ environment.
"""
from __future__ import annotations

import unicodedata
from datetime import date, timedelta
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Risk score weights
# ─────────────────────────────────────────────────────────────────────────────

_RISK_WEIGHTS: dict[str, int] = {
	# Regulatory / screening factors (heaviest)
	"is_pep": 30,
	"is_sanctioned": 50,
	"is_adverse_media": 15,
	# Geography
	"high_risk_country": 20,
	"non_resident": 10,
	# Product / channel / structure
	"high_risk_industry": 15,
	"cash_intensive_business": 10,
	"complex_ownership_structure": 15,
	"nominee_shareholders_present": 12,
	# Document quality
	"low_confidence_documents": 8,
	"missing_address_document": 5,
	"expired_document": 10,
	# Biometrics
	"liveness_fail": 20,
	"biometric_mismatch": 15,
	# Account / transaction history
	"dormant_account": 8,
	"high_cash_transactions": 10,
	"unusual_transaction_patterns": 12,
	# Customer profile
	"is_refugee": 5,
	"is_informal_sector": 5,
	# Edge cases
	"deceased_id_flag": 40,
	"synthetic_identity_flag": 45,
	"id_number_reuse": 35,
}

# FATF grey/black list + OFAC heightened-scrutiny countries (ISO-3166-1 alpha-2)
_HIGH_RISK_COUNTRIES: frozenset[str] = frozenset({
	"AF", "BY", "BI", "CF", "CG", "CD", "CU", "ER", "ET", "GN", "GW",
	"HT", "IR", "IQ", "KP", "LB", "LY", "ML", "MM", "MZ", "NI", "NG",
	"PK", "PS", "RU", "SO", "SS", "SD", "SY", "TN", "UA", "VE", "YE", "ZW",
})

# High-risk ISIC / NAICS-equivalent industry codes
_HIGH_RISK_INDUSTRIES: frozenset[str] = frozenset({
	"6492",  # Money changing
	"6499",  # Other monetary intermediation
	"9200",  # Gambling / betting
	"9001",  # Weapons / defence
	"7911",  # Travel agencies (cash-intensive)
	"4723",  # Precious stones / metals dealers
	"6411",  # Real estate (high-value)
})

# KYC validity periods by risk band
_EXPIRY_DAYS: dict[str, int] = {
	"low": 730,       # 2 years
	"medium": 365,    # 1 year
	"high": 180,      # 6 months
	"very_high": 90,  # 3 months — mandatory periodic refresh
	"unacceptable": 0,
}

# CBK Kenya simplified KYC tier limits (2023 Mobile Money Regulations)
CBK_TIER_THRESHOLDS: dict[str, dict[str, float | None]] = {
	"tier_1": {
		"max_single_txn_kes": 1_000,
		"max_daily_kes": 300_000,
		"max_balance_kes": 100_000,
	},
	"tier_2": {
		"max_single_txn_kes": 70_000,
		"max_daily_kes": 500_000,
		"max_balance_kes": 500_000,
	},
	"tier_3": {
		"max_single_txn_kes": None,
		"max_daily_kes": None,
		"max_balance_kes": None,  # Full KYC — no limits
	},
}


# ─────────────────────────────────────────────────────────────────────────────
# Geography / industry helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_high_risk_country(country_code: str) -> bool:
	"""Return True if the country is on the FATF/OFAC high-risk list."""
	return country_code.upper().strip() in _HIGH_RISK_COUNTRIES


def is_high_risk_industry(industry_code: str) -> bool:
	"""Return True if the ISIC/NAICS industry code is flagged as high-risk."""
	return industry_code.strip() in _HIGH_RISK_INDUSTRIES


def get_cbk_tier(kyc_tier: str) -> dict[str, float | None]:
	"""Return CBK mobile money tier limits for the given KYC tier string."""
	tier_map: dict[str, str] = {
		"simplified": "tier_1",
		"basic": "tier_2",
		"standard": "tier_3",
		"tier_1": "tier_1",
		"tier_2": "tier_2",
		"tier_3": "tier_3",
	}
	mapped = tier_map.get(kyc_tier.lower(), "tier_3")
	return CBK_TIER_THRESHOLDS[mapped]


# ─────────────────────────────────────────────────────────────────────────────
# Risk score calculation
# ─────────────────────────────────────────────────────────────────────────────

def calculate_risk_score(factors: dict[str, bool | int | float]) -> tuple[int, dict[str, int]]:
	"""Compute composite KYC risk score in [0, 100].

	Args:
		factors: mapping of factor name → bool/numeric value.
			Truthy bool values activate the corresponding weight.
			Numeric values scale the weight proportionally.

	Returns:
		(score, breakdown) — score is capped at 100.
		breakdown shows each active factor's point contribution.
	"""
	breakdown: dict[str, int] = {}
	raw = 0
	for factor, weight in _RISK_WEIGHTS.items():
		value = factors.get(factor)
		if not value:
			continue
		contribution = weight if isinstance(value, bool) else int(weight * float(value))
		breakdown[factor] = contribution
		raw += contribution
	score = min(raw, 100)
	return score, breakdown


def calculate_risk_band(score: int) -> str:
	"""Map numeric risk score to a named risk band.

	Bands:
		0–29   → low
		30–54  → medium
		55–74  → high
		75–89  → very_high
		90–100 → unacceptable
	"""
	assert 0 <= score <= 100, f"score must be in [0, 100], got {score}"
	if score >= 90:
		return "unacceptable"
	if score >= 75:
		return "very_high"
	if score >= 55:
		return "high"
	if score >= 30:
		return "medium"
	return "low"


def calculate_expiry_date(risk_band: str, approval_date: date | None = None) -> date | None:
	"""Return the KYC expiry date given risk band and optional approval date.

	Returns None for 'unacceptable' band (approval should be blocked upstream).
	"""
	days = _EXPIRY_DAYS.get(risk_band)
	if days is None:
		raise ValueError(f"unknown risk band: {risk_band!r}")
	if days == 0:
		return None
	base = approval_date or date.today()
	return base + timedelta(days=days)


def days_until_expiry(expiry_date: date | None, reference: date | None = None) -> int | None:
	"""Days until KYC expiry. Negative = already expired. None = no expiry date set."""
	if expiry_date is None:
		return None
	ref = reference or date.today()
	return (expiry_date - ref).days


def expiring_within(expiry_date: date | None, days: int, reference: date | None = None) -> bool:
	"""Return True if KYC expires within ``days`` days from reference date."""
	remaining = days_until_expiry(expiry_date, reference)
	if remaining is None:
		return False
	return 0 <= remaining <= days


# ─────────────────────────────────────────────────────────────────────────────
# Name script detection and transliteration
# ─────────────────────────────────────────────────────────────────────────────

def detect_name_script(name: str) -> str:
	"""Detect the primary Unicode script of a name string.

	Returns one of: 'arabic', 'chinese', 'cyrillic', 'devanagari', 'latin',
	'mixed', 'unknown'.
	"""
	if not name.strip():
		return "unknown"

	counts: dict[str, int] = {
		"arabic": 0, "chinese": 0, "cyrillic": 0,
		"devanagari": 0, "latin": 0, "other": 0,
	}
	for ch in name:
		if ch.isspace() or ch in "'-.,":
			continue
		cp = ord(ch)
		if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F:
			counts["arabic"] += 1
		elif (
			0x4E00 <= cp <= 0x9FFF
			or 0x3400 <= cp <= 0x4DBF
			or 0xF900 <= cp <= 0xFAFF
			or 0x20000 <= cp <= 0x2A6DF
		):
			counts["chinese"] += 1
		elif 0x0400 <= cp <= 0x04FF:
			counts["cyrillic"] += 1
		elif 0x0900 <= cp <= 0x097F:
			counts["devanagari"] += 1
		elif ch.isascii() and ch.isalpha():
			counts["latin"] += 1
		else:
			counts["other"] += 1

	dominant = max(counts, key=lambda k: counts[k])
	non_zero = sum(1 for v in counts.values() if v > 0)
	if non_zero > 2:
		return "mixed"
	return dominant if counts[dominant] > 0 else "unknown"


def transliterate_to_latin(name: str) -> str:
	"""Best-effort transliteration of a non-Latin name to Latin characters.

	Uses Unicode NFKD decomposition for accented Latin and Cyrillic.
	For Arabic and CJK, returns empty string — the caller must use a
	dedicated transliteration service (``camel-tools`` for Arabic,
	``pypinyin`` for Chinese) in production.

	This function is intentionally conservative: it never corrupts names.
	"""
	if not name.strip():
		return ""
	script = detect_name_script(name)
	if script == "latin":
		return name
	if script in ("arabic", "chinese"):
		# Cannot safely transliterate without an external library
		return ""
	try:
		normalized = unicodedata.normalize("NFKD", name)
		return "".join(c for c in normalized if not unicodedata.combining(c))
	except Exception:
		return ""


def normalize_name_for_matching(name: str) -> str:
	"""Normalise a name for fuzzy matching: lowercase, strip accents, collapse spaces."""
	if not name:
		return ""
	transliterated = transliterate_to_latin(name) or name
	normalized = unicodedata.normalize("NFKD", transliterated)
	ascii_only = "".join(c for c in normalized if not unicodedata.combining(c))
	return " ".join(ascii_only.lower().split())


# ─────────────────────────────────────────────────────────────────────────────
# Fuzzy name matching — Jaro-Winkler (no external deps)
# ─────────────────────────────────────────────────────────────────────────────

def _jaro(s1: str, s2: str) -> float:
	if s1 == s2:
		return 1.0
	len1, len2 = len(s1), len(s2)
	if not len1 or not len2:
		return 0.0
	match_dist = max(len1, len2) // 2 - 1
	s1_matches = [False] * len1
	s2_matches = [False] * len2
	matches = 0
	transpositions = 0
	for i in range(len1):
		start = max(0, i - match_dist)
		end = min(i + match_dist + 1, len2)
		for j in range(start, end):
			if s2_matches[j] or s1[i] != s2[j]:
				continue
			s1_matches[i] = s2_matches[j] = True
			matches += 1
			break
	if not matches:
		return 0.0
	k = 0
	for i in range(len1):
		if not s1_matches[i]:
			continue
		while not s2_matches[k]:
			k += 1
		if s1[i] != s2[k]:
			transpositions += 1
		k += 1
	return (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3


def jaro_winkler(s1: str, s2: str, prefix_weight: float = 0.1) -> float:
	"""Compute Jaro-Winkler similarity in [0, 1]."""
	jaro = _jaro(s1, s2)
	prefix = 0
	for c1, c2 in zip(s1[:4], s2[:4]):
		if c1 == c2:
			prefix += 1
		else:
			break
	return jaro + prefix * prefix_weight * (1 - jaro)


def name_match_score(name_a: str, name_b: str) -> float:
	"""Return Jaro-Winkler score between two names after normalisation.

	Returns 0.0 if either name normalises to empty (e.g. script detection
	failure for Arabic/CJK without a transliteration service).
	"""
	a = normalize_name_for_matching(name_a)
	b = normalize_name_for_matching(name_b)
	if not a or not b:
		return 0.0
	return jaro_winkler(a, b)


# ─────────────────────────────────────────────────────────────────────────────
# Ownership / UBO calculations
# ─────────────────────────────────────────────────────────────────────────────

def calculate_total_declared_ownership(ownership_percentages: list[float]) -> float:
	"""Sum all declared ownership percentages. Must not exceed 100%."""
	return round(sum(ownership_percentages), 4)


def has_controlling_ubo(ownership_pct: float, threshold: float = 25.0) -> bool:
	"""Return True if a single UBO meets or exceeds the controlling threshold."""
	return ownership_pct >= threshold


def effective_ownership(direct_pct: float, indirect_chain: list[float]) -> float:
	"""Compute effective ownership through an indirect chain.

	Example: A owns 60% of B, B owns 40% of C
		→ A's effective ownership of C = 60% × 40% = 24%

	Args:
		direct_pct: immediate ownership percentage (0–100)
		indirect_chain: list of intermediate ownership percentages (0–100)

	Returns:
		effective ownership percentage (0–100)
	"""
	result = direct_pct / 100.0
	for pct in indirect_chain:
		result *= pct / 100.0
	return round(result * 100.0, 4)


def calculate_ubo_control_flags(
	ubos: list[dict[str, Any]],
	threshold: float = 25.0,
) -> dict[str, Any]:
	"""Analyse a list of UBO records and return control summary.

	Each UBO dict must have 'ownership_percentage', 'is_nominee', 'full_name'.
	Returns: controlling_ubos, nominee_count, total_declared, ownership_gap,
	complex_structure_flag.
	"""
	total = 0.0
	controlling: list[dict[str, Any]] = []
	nominees: int = 0
	for ubo in ubos:
		pct = float(ubo.get("ownership_percentage", 0))
		total += pct
		if ubo.get("is_nominee"):
			nominees += 1
		if pct >= threshold:
			controlling.append({"full_name": ubo.get("full_name"), "pct": pct})
	gap = max(0.0, 100.0 - total)
	return {
		"total_declared_pct": round(total, 4),
		"ownership_gap_pct": round(gap, 4),
		"controlling_ubos": controlling,
		"controlling_ubo_count": len(controlling),
		"nominee_count": nominees,
		"complex_structure_flag": nominees > 0 or gap > 10.0 or len(controlling) == 0,
	}


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic identity detection helpers
# ─────────────────────────────────────────────────────────────────────────────

def synthetic_identity_score(
	name_doc_match: float,
	dob_doc_match: float,
	liveness_score: float,
	biometric_match: float,
	id_number_age_days: int | None,
) -> float:
	"""Estimate the probability of a synthetic identity (0.0 = clean, 1.0 = fraud).

	Factors:
	- Low name match between docs → indicator
	- Low DOB match → indicator
	- Low liveness → possible spoofing
	- Low biometric match → mismatch
	- Very young ID number (< 30 days) → newly-minted for fraud

	Returns a float in [0, 1].  Score ≥ 0.6 triggers the
	``assert_no_synthetic_identity`` rule in rules.py.
	"""
	score = 0.0
	if name_doc_match < 0.75:
		score += 0.30
	if dob_doc_match < 0.80:
		score += 0.20
	if liveness_score < 0.70:
		score += 0.20
	if biometric_match < 0.80:
		score += 0.20
	if id_number_age_days is not None and id_number_age_days < 30:
		score += 0.15
	return round(min(score, 1.0), 4)


def deceased_id_risk_score(
	id_number: str,
	dob: date | None,
	registry_deceased_flag: bool,
) -> float:
	"""Return a risk score for deceased-ID fraud (0.0 = clean, 1.0 = confirmed).

	Registry flag is authoritative; other heuristics are supplementary.
	"""
	if registry_deceased_flag:
		return 1.0
	score = 0.0
	# Very old DOB without a death date in the registry is suspicious if
	# the person would be over 110 years old.
	if dob is not None:
		age_years = (date.today() - dob).days / 365.25
		if age_years > 110:
			score += 0.60
		elif age_years > 90:
			score += 0.20
	# ID numbers with very old issue dates (heuristic — numeric prefix ≤ 5000000 for KE)
	if id_number and id_number.isdigit() and int(id_number[:7].ljust(7, "0")) < 1_000_000:
		score += 0.10
	return round(min(score, 1.0), 4)
