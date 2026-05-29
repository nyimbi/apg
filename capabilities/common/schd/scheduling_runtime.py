"""Runtime helpers for the APG Scheduling and Job Orchestration capability."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from typing import Any


TRIGGER_TYPES = {"cron", "interval", "calendar", "event", "manual"}
JOB_CRITICALITIES = {"low", "normal", "high", "critical"}
RUN_STATUSES = {"queued", "running", "succeeded", "failed", "blocked", "cancelled", "dead_lettered"}
WORKER_STATES = {"ready", "degraded", "draining", "offline"}
RETRY_STRATEGIES = {"none", "fixed", "linear", "exponential"}


def stable_id(prefix: str, *parts: object) -> str:
	"""Build a deterministic short ID for local package runtime objects."""
	digest = sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:12]
	return f"{prefix}_{digest}"


def utc_now() -> datetime:
	return datetime.now(timezone.utc)


def normalize_tags(tags: list[str] | None) -> list[str]:
	return sorted({tag.strip().lower() for tag in tags or [] if tag and tag.strip()})


def normalize_trigger_type(trigger_type: str) -> str:
	value = trigger_type.strip().lower()
	if value not in TRIGGER_TYPES:
		raise ValueError(f"unsupported_trigger_type:{trigger_type}")
	return value


def normalize_criticality(criticality: str) -> str:
	value = criticality.strip().lower()
	if value not in JOB_CRITICALITIES:
		raise ValueError(f"unsupported_job_criticality:{criticality}")
	return value


def normalize_retry_strategy(strategy: str) -> str:
	value = strategy.strip().lower()
	if value not in RETRY_STRATEGIES:
		raise ValueError(f"unsupported_retry_strategy:{strategy}")
	return value


def normalize_worker_state(state: str) -> str:
	value = state.strip().lower()
	if value not in WORKER_STATES:
		raise ValueError(f"unsupported_worker_state:{state}")
	return value


def schedule_state(enabled: bool, paused: bool = False) -> str:
	if not enabled:
		return "disabled"
	if paused:
		return "paused"
	return "active"


def run_status(succeeded: bool, failed: bool, blocked: int = 0) -> str:
	if blocked:
		return "blocked"
	if failed:
		return "failed"
	if succeeded:
		return "succeeded"
	return "queued"


def next_run_hint(trigger_type: str, timezone_name: str, interval_minutes: int | None = None) -> str:
	if trigger_type == "interval" and interval_minutes:
		return f"every {interval_minutes} minutes in {timezone_name}"
	if trigger_type == "cron":
		return f"next cron match in {timezone_name}"
	if trigger_type == "calendar":
		return f"next business-calendar window in {timezone_name}"
	if trigger_type == "event":
		return f"on matching event in {timezone_name}"
	return f"manual trigger in {timezone_name}"


def backoff_seconds(strategy: str, attempt: int) -> int:
	attempt = max(1, int(attempt))
	strategy = normalize_retry_strategy(strategy)
	if strategy == "none":
		return 0
	if strategy == "fixed":
		return 60
	if strategy == "linear":
		return min(3600, 60 * attempt)
	return min(3600, 60 * (2 ** (attempt - 1)))


def summarize_decision(result: dict[str, Any]) -> str:
	actions = result.get("actions") or []
	if not actions:
		return result.get("decision", "allow")
	return ",".join(action.get("reason", action.get("decision", "policy_action")) for action in actions)
