"""Deterministic compliance calculations for the APG COMP capability."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any


def stable_digest(payload: dict[str, Any]) -> str:
	"""Return a stable SHA-256 digest for compliance evidence and audit payloads."""
	encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
	return hashlib.sha256(encoded).hexdigest()


def evidence_age_days(collected_at: datetime, now: datetime | None = None) -> int:
	now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
	collected = collected_at.astimezone(timezone.utc)
	return max(0, (now - collected).days)


def finding_age_days(created_at: datetime, now: datetime | None = None) -> int:
	now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
	created = created_at.astimezone(timezone.utc)
	return max(0, (now - created).days)


def assessment_result(evidence_age: int, freshness_days: int, has_open_finding: bool) -> str:
	if evidence_age > freshness_days:
		return "stale_evidence"
	if has_open_finding:
		return "needs_remediation"
	return "effective"


def framework_coverage(control_count: int, tested_count: int, failing_count: int) -> dict[str, Any]:
	if control_count <= 0:
		return {"coverage_percent": 0, "assurance": "unmapped"}
	coverage = round((tested_count / control_count) * 100, 2)
	if failing_count:
		assurance = "findings_open"
	elif coverage >= 95:
		assurance = "ready_for_attestation"
	elif coverage >= 70:
		assurance = "testing_in_progress"
	else:
		assurance = "coverage_gap"
	return {"coverage_percent": coverage, "assurance": assurance}
