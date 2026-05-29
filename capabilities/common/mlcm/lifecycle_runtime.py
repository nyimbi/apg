"""Deterministic runtime helpers for AI model lifecycle operations."""

from __future__ import annotations

from hashlib import sha256
from typing import Any


MODEL_STAGES = ("dev", "staging", "production")
MODEL_STATUSES = ("registered", "candidate", "promoted", "serving", "retired", "rolled_back")
DEPLOYMENT_STATUSES = ("serving", "paused", "failed", "rolled_back")


def stable_id(prefix: str, *parts: object) -> str:
	"""Create short deterministic identifiers for repeatable APG tests and demos."""
	seed = "|".join(str(part) for part in parts if part is not None)
	digest = sha256(seed.encode("utf-8")).hexdigest()[:12]
	return f"{prefix}_{digest}"


def normalize_stage(stage: str | None) -> str:
	value = (stage or "dev").strip().lower()
	if value not in MODEL_STAGES:
		raise ValueError(f"unsupported_model_stage:{value}")
	return value


def normalize_status(status: str | None, default: str = "registered") -> str:
	value = (status or default).strip().lower()
	if value not in MODEL_STATUSES:
		raise ValueError(f"unsupported_model_status:{value}")
	return value


def normalize_deployment_status(status: str | None) -> str:
	value = (status or "serving").strip().lower()
	if value not in DEPLOYMENT_STATUSES:
		raise ValueError(f"unsupported_deployment_status:{value}")
	return value


def normalize_score(score: float | int | str | None) -> float:
	if score is None:
		return 0.0
	value = float(score)
	if value < 0:
		return 0.0
	if value > 1:
		return 1.0
	return value


def evaluation_status(score: float, minimum_score: float) -> str:
	return "passed" if score >= minimum_score else "failed"


def promotion_status(
	target_stage: str,
	score: float | None,
	minimum_score: float,
	approval_recorded: bool,
) -> tuple[str, list[str]]:
	reasons: list[str] = []
	if target_stage == "production" and not approval_recorded:
		reasons.append("promotion_approval_required")
	if score is None:
		reasons.append("evaluation_required")
	elif score < minimum_score:
		reasons.append("evaluation_score_too_low")
	return ("blocked" if reasons else "approved", reasons)


def model_card_complete(model_card: dict[str, Any] | None) -> bool:
	card = model_card or {}
	required = ("purpose", "owner", "training_data", "limitations")
	return all(bool(str(card.get(key, "")).strip()) for key in required)


def drift_status(score: float, threshold: float, review_recorded: bool = False) -> tuple[bool, str]:
	detected = score >= threshold
	if detected and not review_recorded:
		return True, "review_required"
	if detected and review_recorded:
		return True, "reviewed"
	return False, "within_threshold"


def deployment_posture(
	stage: str,
	model_card_present: bool,
	score: float | None,
	minimum_score: float,
	unresolved_drift_count: int,
) -> tuple[str, list[str]]:
	reasons: list[str] = []
	if not model_card_present:
		reasons.append("model_card_required")
	if stage == "production" and score is not None and score < minimum_score:
		reasons.append("evaluation_score_too_low")
	if unresolved_drift_count:
		reasons.append("drift_review_required")
	return ("blocked" if reasons else "deployable", reasons)
