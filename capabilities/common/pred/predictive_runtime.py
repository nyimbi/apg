"""Deterministic runtime helpers for predictive analytics operations."""

from __future__ import annotations

from hashlib import sha256
from typing import Any


ENVIRONMENTS = ("development", "staging", "production")
IMPACT_LEVELS = ("low", "medium", "high")


def stable_id(prefix: str, *parts: object) -> str:
	seed = "|".join(str(part) for part in parts if part is not None)
	digest = sha256(seed.encode("utf-8")).hexdigest()[:12]
	return f"{prefix}_{digest}"


def normalize_environment(environment: str | None) -> str:
	value = (environment or "development").strip().lower()
	if value not in ENVIRONMENTS:
		raise ValueError(f"unsupported_prediction_environment:{value}")
	return value


def normalize_impact(impact: str | None) -> str:
	value = (impact or "low").strip().lower()
	if value not in IMPACT_LEVELS:
		raise ValueError(f"unsupported_prediction_impact:{value}")
	return value


def normalize_names(values: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
	return tuple(sorted({str(value).strip().lower() for value in values or () if str(value).strip()}))


def numeric_feature_values(values: dict[str, Any] | None) -> tuple[float, ...]:
	numbers: list[float] = []
	for value in (values or {}).values():
		if isinstance(value, bool):
			numbers.append(1.0 if value else 0.0)
		elif isinstance(value, (int, float)):
			numbers.append(float(value))
	return tuple(numbers)


def deterministic_score(model_id: str, feature_values: dict[str, Any] | None) -> float:
	numbers = numeric_feature_values(feature_values)
	if not numbers:
		return 0.0
	bias = int(sha256(model_id.encode("utf-8")).hexdigest()[:4], 16) % 17
	raw = (sum(numbers) / len(numbers)) + bias
	return round(max(0.0, min(100.0, raw)), 4)


def forecast_series(history_values: list[float] | tuple[float, ...], horizon_days: int) -> tuple[float, ...]:
	if not history_values:
		return ()
	points = [float(value) for value in history_values]
	window = points[-min(6, len(points)):]
	baseline = sum(window) / len(window)
	slope = 0.0
	if len(points) >= 2:
		slope = (points[-1] - points[max(0, len(points) - 6)]) / min(5, len(points) - 1)
	values = [round(baseline + slope * (index + 1), 4) for index in range(horizon_days)]
	return tuple(values)


def scenario_projection(baseline_score: float, adjustments: dict[str, Any] | None) -> tuple[float, float]:
	total_adjustment = sum(numeric_feature_values(adjustments))
	scenario_score = round(max(0.0, min(100.0, float(baseline_score) + total_adjustment)), 4)
	return scenario_score, round(scenario_score - float(baseline_score), 4)


def drift_status(drift_score: float, threshold: float) -> str:
	return "review_required" if float(drift_score) > float(threshold) else "within_threshold"
