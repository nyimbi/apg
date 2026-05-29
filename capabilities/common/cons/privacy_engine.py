"""Deterministic privacy calculations for the APG CONS capability."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from typing import Any


def stable_digest(payload: dict[str, Any]) -> str:
	encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
	return hashlib.sha256(encoded).hexdigest()


def consent_age_days(captured_at: datetime, now: datetime | None = None) -> int:
	now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
	captured = captured_at.astimezone(timezone.utc)
	return max(0, (now - captured).days)


def request_due_at(submitted_at: datetime, sla_days: int = 30) -> datetime:
	return submitted_at.astimezone(timezone.utc) + timedelta(days=sla_days)


def request_sla_state(due_at: datetime, now: datetime | None = None) -> str:
	now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
	due = due_at.astimezone(timezone.utc)
	if now > due:
		return "overdue"
	if (due - now).days <= 3:
		return "due_soon"
	return "on_track"


def consent_coverage(active_count: int, withdrawn_count: int, purpose_count: int) -> dict[str, Any]:
	total = active_count + withdrawn_count
	if purpose_count <= 0:
		return {"coverage_percent": 0, "posture": "no_purposes"}
	if total <= 0:
		return {"coverage_percent": 0, "posture": "no_consents"}
	coverage = round((active_count / max(purpose_count, 1)) * 100, 2)
	if withdrawn_count:
		posture = "withdrawals_present"
	elif coverage >= 100:
		posture = "fully_consented"
	elif coverage >= 70:
		posture = "mostly_consented"
	else:
		posture = "coverage_gap"
	return {"coverage_percent": coverage, "posture": posture}
