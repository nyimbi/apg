"""Deterministic runtime helpers for Platform Foundation."""

from __future__ import annotations

from hashlib import sha256
from typing import Any


TIERS = ("core", "shared", "edge", "integration")
HEALTH_STATUSES = ("healthy", "degraded", "unhealthy", "unknown")
BASELINE_TYPES = ("configuration", "tenant", "auth", "audit")


def stable_id(prefix: str, *parts: object) -> str:
	seed = "|".join(str(part) for part in parts if part is not None)
	digest = sha256(seed.encode("utf-8")).hexdigest()[:12]
	return f"{prefix}_{digest}"


def normalize_tier(tier: str | None) -> str:
	value = (tier or "shared").strip().lower()
	if value not in TIERS:
		raise ValueError(f"unsupported_foundation_tier:{value}")
	return value


def normalize_health(status: str | None) -> str:
	value = (status or "unknown").strip().lower()
	if value not in HEALTH_STATUSES:
		raise ValueError(f"unsupported_health_status:{value}")
	return value


def normalize_baseline_type(baseline_type: str | None) -> str:
	value = (baseline_type or "").strip().lower()
	if value not in BASELINE_TYPES:
		raise ValueError(f"unsupported_baseline_type:{value}")
	return value


def normalize_score(score: float | int | str | None) -> float:
	if score is None:
		return 0.0
	value = float(score)
	if value < 0:
		return 0.0
	if value > 100:
		return 100.0
	return round(value, 2)


def service_baselines_complete(baselines: list[dict[str, Any]]) -> bool:
	approved_types = {baseline["baseline_type"] for baseline in baselines if baseline["status"] == "approved"}
	return set(BASELINE_TYPES).issubset(approved_types)


def dependencies_are_healthy(dependencies: list[dict[str, Any]]) -> bool:
	return all(not item["required"] or item["health_status"] == "healthy" for item in dependencies)


def readiness_posture(
	score: float,
	dependencies_healthy: bool,
	baselines_complete: bool,
	monitoring_ready: bool,
	rollback_ready: bool,
	change_window_ready: bool,
) -> tuple[str, list[str]]:
	issues: list[str] = []
	if score < 80:
		issues.append("readiness_score_below_threshold")
	if not dependencies_healthy:
		issues.append("dependencies_unhealthy")
	if not baselines_complete:
		issues.append("baselines_incomplete")
	if not monitoring_ready:
		issues.append("monitoring_required")
	if not rollback_ready:
		issues.append("rollback_plan_required")
	if not change_window_ready:
		issues.append("change_window_required")
	if issues:
		return "blocked", issues
	return "ready", issues


def change_review_status(affected_capability_count: int, broad_review_recorded: bool) -> str:
	if affected_capability_count > 10 and not broad_review_recorded:
		return "review_required"
	return "pending_approval"
