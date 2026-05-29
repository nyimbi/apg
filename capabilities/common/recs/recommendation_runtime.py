"""Deterministic runtime helpers for recommender system operations."""

from __future__ import annotations

from hashlib import sha256
from typing import Any


ALGORITHMS = ("collaborative_filtering", "content_based", "hybrid", "contextual_bandit")
IMPACT_LEVELS = ("low", "medium", "high")


def stable_id(prefix: str, *parts: object) -> str:
	seed = "|".join(str(part) for part in parts if part is not None)
	digest = sha256(seed.encode("utf-8")).hexdigest()[:12]
	return f"{prefix}_{digest}"


def normalize_algorithm(algorithm: str | None) -> str:
	value = (algorithm or "hybrid").strip().lower().replace("-", "_")
	if value not in ALGORITHMS:
		raise ValueError(f"unsupported_recommendation_algorithm:{value}")
	return value


def normalize_impact_level(impact_level: str | None) -> str:
	value = (impact_level or "low").strip().lower()
	if value not in IMPACT_LEVELS:
		raise ValueError(f"unsupported_recommendation_impact:{value}")
	return value


def normalize_features(features: dict[str, Any] | None) -> dict[str, float]:
	normalized: dict[str, float] = {}
	for key, value in (features or {}).items():
		try:
			normalized[str(key).strip().lower()] = round(float(value), 4)
		except (TypeError, ValueError):
			continue
	return {key: value for key, value in normalized.items() if key}


def normalize_labels(labels: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
	return tuple(sorted({str(label).strip().lower() for label in labels or () if str(label).strip()}))


def score_item(
	model_id: str,
	profile_features: dict[str, float],
	item_features: dict[str, float],
	tags: tuple[str, ...],
	segments: tuple[str, ...],
) -> float:
	shared_keys = set(profile_features).intersection(item_features)
	dot_score = sum(profile_features[key] * item_features[key] for key in shared_keys)
	tag_bonus = 0.05 * len(set(tags).intersection(segments))
	seed = int(sha256(f"{model_id}:{sorted(item_features.items())}".encode("utf-8")).hexdigest()[:4], 16)
	jitter = (seed % 100) / 1000
	return round(min(1.0, max(0.0, dot_score + tag_bonus + jitter)), 4)


def confidence_for_score(score: float, algorithm: str) -> float:
	algorithm_weight = {
		"collaborative_filtering": 0.04,
		"content_based": 0.03,
		"hybrid": 0.06,
		"contextual_bandit": 0.05,
	}[normalize_algorithm(algorithm)]
	return round(min(0.99, max(0.0, score + algorithm_weight)), 4)


def recommendation_reason(score: float, tags: tuple[str, ...], segments: tuple[str, ...]) -> str:
	overlap = sorted(set(tags).intersection(segments))
	if overlap:
		return f"matched profile segment {overlap[0]} at score {score:.2f}"
	return f"ranked by feature affinity at score {score:.2f}"


def drift_status(baseline_metric: float, current_metric: float) -> str:
	delta = abs(float(baseline_metric) - float(current_metric))
	if delta >= 0.20:
		return "critical"
	if delta >= 0.10:
		return "watch"
	return "stable"
