"""Domain calculations for Intelligence Fusion.

All formulas are pure functions — no side-effects, no I/O.  Type-safe inputs,
comprehensive edge-case handling, and explicit comments referencing the
analytic methodology used.

Methodologies referenced:
  - IC Standards for Analytic Confidence (ICD 203)
  - Analysis of Competing Hypotheses (Heuer, 1999)
  - Bayesian confidence update
  - Dempster-Shafer evidence combination (simplified)
  - F-measure for source corroboration
  - ACE (Analysis, Confidence, Evidence) method

© 2025 Datacraft — Nyimbi Odero
"""
from __future__ import annotations

import math
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Confidence calibration (ICD 203)
# ─────────────────────────────────────────────────────────────────────────────

_CONFIDENCE_WORDS: list[tuple[float, str]] = [
	(0.93, "almost_certain"),
	(0.80, "highly_likely"),
	(0.55, "likely"),
	(0.45, "roughly_even"),
	(0.20, "unlikely"),
	(0.07, "highly_unlikely"),
	(0.00, "remote"),
]


def score_to_confidence_level(score: float) -> str:
	"""Convert a numeric probability to an ICD-203 estimative word."""
	score = max(0.0, min(1.0, score))
	for threshold, word in _CONFIDENCE_WORDS:
		if score >= threshold:
			return word
	return "remote"


def bayesian_update(
	prior: float,
	likelihood_given_true: float,
	likelihood_given_false: float,
) -> float:
	"""
	Bayesian posterior update.

	P(H|E) = P(E|H) * P(H) / P(E)

	Args:
		prior: P(H) — prior probability of hypothesis being true [0,1]
		likelihood_given_true: P(E|H) — probability of this evidence if H is true [0,1]
		likelihood_given_false: P(E|¬H) — probability of this evidence if H is false [0,1]

	Returns:
		Posterior probability P(H|E) clamped to [0.001, 0.999]
	"""
	prior = max(1e-9, min(1.0 - 1e-9, prior))
	if likelihood_given_false <= 0:
		return min(0.999, prior * likelihood_given_true)
	p_evidence = likelihood_given_true * prior + likelihood_given_false * (1.0 - prior)
	if p_evidence <= 0:
		return prior
	posterior = (likelihood_given_true * prior) / p_evidence
	return max(0.001, min(0.999, posterior))


def likelihood_ratio(likelihood_given_true: float, likelihood_given_false: float) -> float:
	"""Compute the Bayes factor (likelihood ratio) for evidence."""
	if likelihood_given_false <= 0:
		return 999.0
	return likelihood_given_true / likelihood_given_false


def calibrated_confidence(
	scores: list[float],
	weights: list[float] | None = None,
) -> float:
	"""
	Weighted harmonic mean of multiple confidence scores.

	Harmonic mean is conservative — pulled toward the lowest scores,
	appropriate for fusion where a single weak source drags the whole picture.

	Args:
		scores: list of confidence values in [0,1]
		weights: optional weights (same length as scores; will be normalised)

	Returns:
		Calibrated composite confidence in [0,1]
	"""
	if not scores:
		return 0.0
	scores = [max(1e-6, min(1.0, s)) for s in scores]
	if weights is None:
		weights = [1.0] * len(scores)
	assert len(weights) == len(scores), "weights and scores must have the same length"
	total_weight = sum(weights)
	if total_weight <= 0:
		return 0.0
	w = [wt / total_weight for wt in weights]
	harmonic = 1.0 / sum(wi / si for wi, si in zip(w, scores))
	return max(0.0, min(1.0, harmonic))


def confidence_calibration_report(
	prior: float,
	likelihood_given_true: float,
	likelihood_given_false: float,
) -> dict[str, Any]:
	"""Full Bayesian confidence calibration including word equivalent."""
	lr = likelihood_ratio(likelihood_given_true, likelihood_given_false)
	posterior = bayesian_update(prior, likelihood_given_true, likelihood_given_false)
	level = score_to_confidence_level(posterior)
	return {
		"prior": round(prior, 4),
		"likelihood_ratio": round(lr, 4),
		"posterior": round(posterior, 4),
		"confidence_level": level,
		"word_equivalent": level.replace("_", " "),
	}


# ─────────────────────────────────────────────────────────────────────────────
# Analysis of Competing Hypotheses (ACH — Heuer, 1999)
# ─────────────────────────────────────────────────────────────────────────────

def build_ach_matrix(
	hypotheses: list[str],
	evidence_items: list[dict[str, Any]],
) -> dict[str, Any]:
	"""
	Construct an ACH matrix from hypotheses and evidence.

	Each evidence item should have:
		{
			"label": str,
			"consistencies": list[float]  # one per hypothesis, values in [-1, 0, 1]
		}

	Consistency convention (Heuer):
		 1 = consistent with hypothesis
		 0 = irrelevant / not applicable
		-1 = inconsistent with hypothesis

	Returns a dict with the matrix and ranked hypotheses by total inconsistency.
	The hypothesis with the *least* total inconsistency is diagnostically leading.
	"""
	if not hypotheses:
		return {
			"hypotheses": [],
			"evidence_labels": [],
			"matrix": [],
			"inconsistency_scores": [],
			"hypothesis_confidence": [],
			"leading_idx": 0,
			"leading_hypothesis": "",
		}

	n_h = len(hypotheses)
	matrix: list[list[float]] = []
	labels: list[str] = []

	for ev in evidence_items:
		row = ev.get("consistencies", [0.0] * n_h)
		# Pad or trim to n_h
		row = (list(row) + [0.0] * n_h)[:n_h]
		matrix.append(row)
		labels.append(ev.get("label", ""))

	# Score = sum of squared inconsistencies (negative values only)
	# Lower score → fewer inconsistencies → more likely hypothesis
	scores: list[float] = []
	for h_idx in range(n_h):
		total = sum(
			row[h_idx] ** 2 if row[h_idx] < 0 else 0.0
			for row in matrix
		)
		scores.append(total)

	leading_idx = int(min(range(n_h), key=lambda i: scores[i]))
	confidence_values = ach_hypothesis_confidence(scores)

	return {
		"hypotheses": hypotheses,
		"evidence_labels": labels,
		"matrix": matrix,
		"inconsistency_scores": scores,
		"hypothesis_confidence": confidence_values,
		"leading_idx": leading_idx,
		"leading_hypothesis": hypotheses[leading_idx],
		"confidence": round(confidence_values[leading_idx], 4) if confidence_values else 0.5,
	}


def ach_hypothesis_confidence(inconsistency_scores: list[float]) -> list[float]:
	"""
	Convert ACH inconsistency scores to normalised confidence probabilities.

	Lower inconsistency → higher confidence.
	Uses softmax over negated scores so they sum to 1.
	"""
	if not inconsistency_scores:
		return []
	negated = [-s for s in inconsistency_scores]
	max_val = max(negated)
	exps = [math.exp(v - max_val) for v in negated]
	total = sum(exps)
	return [e / total for e in exps] if total > 0 else [1.0 / len(negated)] * len(negated)


# ─────────────────────────────────────────────────────────────────────────────
# Key Assumptions Check (KAC)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_assumptions(
	assumptions: list[str],
	confidence_scores: list[float],
) -> dict[str, Any]:
	"""
	Rate the robustness of a set of key assumptions.

	Returns the weakest assumption (lowest confidence) and an overall
	analytic robustness score (geometric mean of assumption confidences).
	A geometric mean is used because one collapsed assumption invalidates
	the whole analytic picture.
	"""
	if not assumptions or not confidence_scores:
		return {
			"assumptions": [],
			"robustness": 0.0,
			"weakest_assumption": "",
			"weakest_confidence": 0.0,
			"analytic_recommendation": "no_assumptions_provided",
		}
	paired = list(zip(assumptions, confidence_scores))
	weakest = min(paired, key=lambda p: p[1])
	scores = [max(1e-9, min(1.0, s)) for s in confidence_scores]
	geo_mean = math.exp(sum(math.log(s) for s in scores) / len(scores))

	if geo_mean < 0.30:
		recommendation = "revisit_core_assumptions_before_proceeding"
	elif geo_mean < 0.60:
		recommendation = "stress_test_weakest_assumptions"
	else:
		recommendation = "assumptions_sufficiently_robust"

	return {
		"assumptions": [{"assumption": a, "confidence": c} for a, c in paired],
		"robustness": round(geo_mean, 4),
		"weakest_assumption": weakest[0],
		"weakest_confidence": weakest[1],
		"analytic_recommendation": recommendation,
	}


# ─────────────────────────────────────────────────────────────────────────────
# ACE Method — Analysis, Confidence, Evidence
# ─────────────────────────────────────────────────────────────────────────────

def ace_assessment(
	analysis_statement: str,
	confidence_score: float,
	evidence_count: int,
	evidence_types: list[str],
	cross_source_confirmed: bool,
) -> dict[str, Any]:
	"""
	ACE (Analysis, Confidence, Evidence) structured assessment.

	Returns the structured ACE output with diagnostics about
	evidence sufficiency and analytic confidence.
	"""
	confidence_level = score_to_confidence_level(confidence_score)
	evidence_diversity = len(set(t.lower() for t in evidence_types))

	# Evidence sufficiency tiers
	if evidence_count == 0:
		sufficiency = "insufficient"
	elif evidence_count < 3:
		sufficiency = "minimal"
	elif evidence_count < 6:
		sufficiency = "adequate"
	else:
		sufficiency = "strong"

	corroboration_bonus = 0.10 if cross_source_confirmed else 0.0
	adjusted_confidence = min(0.999, confidence_score + corroboration_bonus * (1.0 - confidence_score))

	return {
		"analysis": analysis_statement,
		"confidence_score": round(adjusted_confidence, 4),
		"confidence_level": score_to_confidence_level(adjusted_confidence),
		"confidence_word": confidence_level.replace("_", " "),
		"evidence_count": evidence_count,
		"evidence_diversity": evidence_diversity,
		"evidence_sufficiency": sufficiency,
		"cross_source_confirmed": cross_source_confirmed,
	}


# ─────────────────────────────────────────────────────────────────────────────
# Source corroboration (Dempster-Shafer, simplified)
# ─────────────────────────────────────────────────────────────────────────────

def source_corroboration_score(
	source_confidences: list[float],
	source_types: list[str],
) -> dict[str, Any]:
	"""
	Multi-source corroboration score.

	Uses Dempster-Shafer combination: p_combined = 1 − ∏(1 − p_i)
	Independent sources confirming the same observation increase combined
	confidence non-linearly.  Diverse source types are weighted higher.

	Returns combined confidence and diversity bonus.
	"""
	if not source_confidences:
		return {
			"combined_confidence": 0.0,
			"source_count": 0,
			"unique_source_types": 0,
			"diversity_bonus": 0.0,
			"confidence_level": "remote",
		}

	cs = [max(0.01, min(0.99, s)) for s in source_confidences]
	n = len(cs)

	combined = 1.0 - math.prod(1.0 - c for c in cs)

	unique_types = len(set(t.lower() for t in source_types))
	diversity_bonus = min(0.15, unique_types * 0.03)
	combined = min(0.99, combined + diversity_bonus * (1.0 - combined))

	return {
		"combined_confidence": round(combined, 4),
		"source_count": n,
		"unique_source_types": unique_types,
		"diversity_bonus": round(diversity_bonus, 4),
		"confidence_level": score_to_confidence_level(combined),
	}


# ─────────────────────────────────────────────────────────────────────────────
# Risk scoring
# ─────────────────────────────────────────────────────────────────────────────

_RISK_MAP = {"low": 1, "medium": 2, "high": 3, "critical": 4}
_RISK_REVERSE = {v: k for k, v in _RISK_MAP.items()}


def composite_risk_score(
	risk_levels: list[str],
	weights: list[float] | None = None,
) -> dict[str, Any]:
	"""
	Compute composite risk from multiple risk levels.

	Uses weighted mean of numeric risk, then rounds to the nearest level.
	Also reports the maximum risk level seen — for escalation logic.
	"""
	if not risk_levels:
		return {"numeric": 0.0, "level": "low", "max_seen": "low"}
	numeric = [_RISK_MAP.get(r.lower(), 1) for r in risk_levels]
	if weights is None:
		weights = [1.0] * len(numeric)
	assert len(weights) == len(numeric)
	total_w = sum(weights)
	score = sum(n * w for n, w in zip(numeric, weights)) / (total_w or 1.0)
	rounded = max(1, min(4, round(score)))
	return {
		"numeric": round(score, 3),
		"level": _RISK_REVERSE[rounded],
		"max_seen": _RISK_REVERSE[max(numeric)],
	}


def escalation_threshold_met(
	risk_level: str,
	confidence: float,
	threshold_risk: str = "high",
	threshold_confidence: float = 0.60,
) -> bool:
	"""True if both risk level and confidence exceed the escalation thresholds."""
	return (
		_RISK_MAP.get(risk_level.lower(), 0) >= _RISK_MAP.get(threshold_risk.lower(), 3)
		and confidence >= threshold_confidence
	)


# ─────────────────────────────────────────────────────────────────────────────
# Time-window correlation
# ─────────────────────────────────────────────────────────────────────────────

def time_overlap_score(
	start_a: float,
	end_a: float,
	start_b: float,
	end_b: float,
) -> float:
	"""
	Intersection-over-Union (IoU) score for two time windows (Unix timestamps).

	Returns 0.0 if no overlap, 1.0 if perfectly aligned.
	"""
	if end_a <= start_a or end_b <= start_b:
		return 0.0
	intersection = max(0.0, min(end_a, end_b) - max(start_a, start_b))
	union = max(end_a, end_b) - min(start_a, start_b)
	return intersection / union if union > 0 else 0.0


def items_within_time_window(
	timestamps: list[float],
	window_start: float,
	window_end: float,
) -> list[int]:
	"""Return indices of timestamps that fall within [window_start, window_end]."""
	return [i for i, t in enumerate(timestamps) if window_start <= t <= window_end]


def temporal_clustering_score(
	timestamps: list[float],
	window_seconds: float = 3600.0,
) -> dict[str, Any]:
	"""
	Measure how temporally clustered a set of intelligence items are.

	Items tightly clustered in time suggest coordinated activity.
	Returns a clustering coefficient in [0,1] where 1 = all in same second.
	"""
	if len(timestamps) < 2:
		return {"coefficient": 0.0, "span_seconds": 0.0, "count": len(timestamps)}
	ts = sorted(timestamps)
	span = ts[-1] - ts[0]
	if span == 0:
		return {"coefficient": 1.0, "span_seconds": 0.0, "count": len(ts)}
	# Normalise: coefficient approaches 1 as span approaches 0 relative to window
	coefficient = max(0.0, 1.0 - (span / window_seconds))
	return {
		"coefficient": round(coefficient, 4),
		"span_seconds": round(span, 2),
		"count": len(ts),
	}


# ─────────────────────────────────────────────────────────────────────────────
# Dissemination / TLP
# ─────────────────────────────────────────────────────────────────────────────

_TLP_ORDER = {"TLP:WHITE": 0, "TLP:CLEAR": 0, "TLP:GREEN": 1, "TLP:AMBER": 2, "TLP:RED": 3}


def tlp_compatible(product_tlp: str, recipient_max_tlp: str) -> bool:
	"""
	True if a product's TLP level does not exceed what the recipient can receive.

	TLP:WHITE / TLP:CLEAR ≤ TLP:GREEN ≤ TLP:AMBER ≤ TLP:RED
	"""
	p = _TLP_ORDER.get(product_tlp.upper(), 99)
	r = _TLP_ORDER.get(recipient_max_tlp.upper(), -1)
	return p <= r


def effective_classification(
	classification_levels: list[str],
) -> str:
	"""
	The effective classification of a fused product is the highest of all
	constituent items (domination principle).
	"""
	_order = {"unclassified": 0, "confidential": 1, "secret": 2, "top_secret": 3}
	levels = [c.lower() for c in classification_levels if c]
	if not levels:
		return "unclassified"
	return max(levels, key=lambda c: _order.get(c, 0))


# ─────────────────────────────────────────────────────────────────────────────
# Fusion quality score
# ─────────────────────────────────────────────────────────────────────────────

def fusion_quality_score(
	source_count: int,
	unique_source_types: int,
	avg_confidence: float,
	has_cross_source_confirmation: bool,
	has_structured_analytic_technique: bool,
) -> dict[str, Any]:
	"""
	Composite quality score for a fusion product [0,1].

	Dimensions:
	  source_breadth  (0–0.25) — more sources from more disciplines
	  confidence      (0–0.35) — average confidence of inputs
	  corroboration   (0–0.25) — cross-source confirmation bonus
	  rigour          (0–0.15) — use of a structured analytic technique
	"""
	breadth = min(0.25, source_count * 0.05 + unique_source_types * 0.03)
	conf_component = avg_confidence * 0.35
	corroboration = 0.25 if has_cross_source_confirmation else 0.0
	rigour = 0.15 if has_structured_analytic_technique else 0.0
	total = breadth + conf_component + corroboration + rigour

	quality = min(1.0, total)
	if quality >= 0.80:
		recommendation = "publication_ready"
	elif quality >= 0.55:
		recommendation = "additional_corroboration_recommended"
	else:
		recommendation = "insufficient_quality_for_dissemination"

	return {
		"quality_score": round(quality, 4),
		"dimensions": {
			"source_breadth": round(breadth, 4),
			"confidence": round(conf_component, 4),
			"corroboration": round(corroboration, 4),
			"analytic_rigour": round(rigour, 4),
		},
		"recommendation": recommendation,
	}


# ─────────────────────────────────────────────────────────────────────────────
# Cross-domain correlation
# ─────────────────────────────────────────────────────────────────────────────

def cross_domain_correlation_score(
	domain_scores: dict[str, float],
	domain_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
	"""
	Aggregate correlation scores across intelligence domains.

	Used in correlate_across_domains() to combine OSINT/SIGINT/HUMINT signals.
	Domains with no data contribute 0.  Missing weights default to 1.0.

	Returns overall score and per-domain contributions.
	"""
	if not domain_scores:
		return {"overall": 0.0, "domains": {}, "dominant_domain": ""}

	weights = domain_weights or {d: 1.0 for d in domain_scores}
	total_w = sum(weights.get(d, 1.0) for d in domain_scores)
	if total_w <= 0:
		return {"overall": 0.0, "domains": domain_scores, "dominant_domain": ""}

	weighted_sum = sum(
		domain_scores[d] * weights.get(d, 1.0) for d in domain_scores
	)
	overall = min(1.0, weighted_sum / total_w)
	dominant = max(domain_scores, key=lambda d: domain_scores[d])

	return {
		"overall": round(overall, 4),
		"confidence_level": score_to_confidence_level(overall),
		"domains": {d: round(v, 4) for d, v in domain_scores.items()},
		"dominant_domain": dominant,
	}
