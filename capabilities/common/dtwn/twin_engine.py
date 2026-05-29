"""Deterministic digital-twin helpers for the APG DTWN capability."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_digest(payload: Any) -> str:
	"""Return a stable digest for twin state, model evidence, and audit events."""
	if isinstance(payload, str):
		raw = payload
	else:
		raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
	return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def state_version_for(twin_id: str, state: dict[str, Any], sequence: int) -> str:
	digest = stable_digest({"twin_id": twin_id, "state": state, "sequence": sequence})
	return f"v{sequence:06d}-{digest[:10]}"


def fuse_state(current: dict[str, Any], measurements: dict[str, Any]) -> dict[str, Any]:
	"""Merge telemetry measurements into the twin state without mutating input."""
	state = dict(current)
	for key, value in measurements.items():
		state[key] = value
	return state


def simulation_outputs(state: dict[str, Any], model_confidence: float, scenario: str) -> dict[str, Any]:
	"""Generate deterministic simulation summary outputs from current twin state."""
	numeric_values = [float(value) for value in state.values() if isinstance(value, int | float)]
	load = sum(numeric_values) / len(numeric_values) if numeric_values else 0.0
	normalized_load = min(1.0, max(0.0, load / 100.0))
	risk_score = round(min(0.99, max(0.01, (normalized_load * 0.7) + ((1.0 - model_confidence) * 0.3))), 3)
	return {
		"scenario": scenario,
		"state_digest": stable_digest(state),
		"normalized_load": round(normalized_load, 3),
		"risk_score": risk_score,
		"recommendation": recommendation_for_risk(risk_score),
	}


def recommendation_for_risk(risk_score: float) -> str:
	if risk_score > 0.8:
		return "review_before_operational_change"
	if risk_score > 0.55:
		return "monitor_and_schedule_maintenance"
	return "continue_normal_operation"
