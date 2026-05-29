"""Deterministic helpers for the SBOX sandbox/testing runtime."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from typing import Any


SANDBOX_STATES = {"draft", "ready", "running", "completed", "failed", "expired", "quarantined"}
DATASET_TYPES = {"synthetic", "masked", "fixture", "production_sample"}
RUN_TYPES = {"unit", "integration", "plugin", "agent", "migration", "load"}
RESULT_STATUSES = {"queued", "running", "passed", "failed", "blocked", "cancelled"}
ISOLATION_LEVELS = {"basic", "network_locked", "data_masked", "strict", "air_gapped"}


def stable_id(prefix: str, *parts: object) -> str:
	"""Return a stable APG-local identifier for deterministic tests."""
	seed = "|".join(str(part).strip().lower() for part in parts if str(part).strip())
	digest = sha256(seed.encode("utf-8")).hexdigest()[:12]
	return f"{prefix}_{digest}"


def utc_now() -> datetime:
	"""Return an aware UTC timestamp."""
	return datetime.now(timezone.utc)


def normalize_tags(tags: list[str] | tuple[str, ...] | None) -> list[str]:
	"""Normalize tag strings for stable comparisons."""
	seen: set[str] = set()
	normalized: list[str] = []
	for tag in tags or []:
		value = str(tag).strip().lower().replace(" ", "_")
		if value and value not in seen:
			seen.add(value)
			normalized.append(value)
	return normalized


def normalize_dataset_type(dataset_type: str) -> str:
	value = str(dataset_type or "").strip().lower()
	if value not in DATASET_TYPES:
		raise ValueError(f"unsupported_dataset_type:{value}")
	return value


def normalize_run_type(run_type: str) -> str:
	value = str(run_type or "").strip().lower()
	if value not in RUN_TYPES:
		raise ValueError(f"unsupported_run_type:{value}")
	return value


def normalize_isolation_level(level: str) -> str:
	value = str(level or "").strip().lower()
	if value not in ISOLATION_LEVELS:
		raise ValueError(f"unsupported_isolation_level:{value}")
	return value


def sandbox_state(ttl_hours: int, approved: bool, has_run: bool = False) -> str:
	if ttl_hours <= 0:
		return "expired"
	if approved and has_run:
		return "running"
	if approved:
		return "ready"
	return "draft"


def run_status(passed: int, failed: int, blocked: int = 0) -> str:
	if blocked > 0:
		return "blocked"
	if failed > 0:
		return "failed"
	if passed > 0:
		return "passed"
	return "queued"


def risk_score(
	ttl_hours: int,
	outbound_network: bool,
	secret_access: bool,
	dataset_type: str,
	isolation_level: str,
) -> int:
	score = 0
	if ttl_hours > 48:
		score += 25
	if outbound_network:
		score += 25
	if secret_access:
		score += 20
	if dataset_type == "production_sample":
		score += 25
	if isolation_level in {"basic", "network_locked"}:
		score += 10
	return min(score, 100)


def summarize_decision(result: dict[str, Any]) -> str:
	if result.get("decision") == "allow":
		return "Sandbox policy allows the requested operation."
	reasons = [action.get("reason", "policy_review_required") for action in result.get("actions", [])]
	return "; ".join(reasons) if reasons else "sandbox_policy_blocked"
