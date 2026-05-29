"""Deterministic DLP inspection helpers for the APG DLPD capability."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


PATTERNS: dict[str, re.Pattern[str]] = {
	"pii": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+|\b\d{3}-\d{2}-\d{4}\b", re.IGNORECASE),
	"phi": re.compile(r"\b(patient|diagnosis|medical record|prescription|insurance id)\b", re.IGNORECASE),
	"pci": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
	"secrets": re.compile(r"\b(api[_-]?key|secret|password|token)\b\s*[:=]\s*['\"]?[A-Za-z0-9._-]{8,}", re.IGNORECASE),
	"financial_records": re.compile(r"\b(iban|swift|routing number|account number|ledger|invoice)\b", re.IGNORECASE),
	"source_code": re.compile(r"\b(def |class |import |SELECT |INSERT |BEGIN RSA PRIVATE KEY)\b", re.IGNORECASE),
}

SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3}


def stable_digest(payload: Any) -> str:
	"""Return a stable digest for content, audit metadata, and decisions."""
	if isinstance(payload, str):
		raw = payload
	else:
		raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
	return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def detect_classifier_hits(content: str, enabled_patterns: list[str]) -> list[dict[str, Any]]:
	"""Detect sensitive data classes using deterministic local patterns."""
	hits: list[dict[str, Any]] = []
	for key in enabled_patterns:
		pattern = PATTERNS.get(key)
		if pattern is None:
			continue
		matches = pattern.findall(content)
		if not matches:
			continue
		hits.append({
			"classifier": key,
			"match_count": len(matches),
			"confidence": confidence_for(key, len(matches)),
			"severity": severity_for_classifier(key),
			"sensitivity_label": sensitivity_label_for_classifier(key),
		})
	return hits


def confidence_for(classifier: str, match_count: int) -> float:
	base = {
		"pii": 0.9,
		"phi": 0.88,
		"pci": 0.94,
		"secrets": 0.96,
		"financial_records": 0.86,
		"source_code": 0.84,
	}.get(classifier, 0.82)
	return min(0.99, base + (max(match_count, 1) - 1) * 0.01)


def severity_for_classifier(classifier: str) -> str:
	if classifier in {"pci", "phi", "secrets"}:
		return "high"
	if classifier in {"pii", "financial_records", "source_code"}:
		return "medium"
	return "low"


def sensitivity_label_for_classifier(classifier: str) -> str:
	if classifier in {"pci", "phi", "secrets"}:
		return "restricted"
	if classifier in {"pii", "financial_records", "source_code"}:
		return "confidential"
	return "internal"


def highest_severity(hits: list[dict[str, Any]]) -> str:
	severity = "low"
	for hit in hits:
		if SEVERITY_RANK[hit["severity"]] > SEVERITY_RANK[severity]:
			severity = hit["severity"]
	return severity


def highest_sensitivity_label(hits: list[dict[str, Any]]) -> str | None:
	if not hits:
		return None
	if any(hit["sensitivity_label"] == "restricted" for hit in hits):
		return "restricted"
	if any(hit["sensitivity_label"] == "confidential" for hit in hits):
		return "confidential"
	return "internal"


def action_for(policy_action: str, severity: str, review_required: bool) -> str:
	"""Return the runtime response action for an inspection."""
	if review_required:
		return "require_review"
	if severity == "high" and policy_action in {"quarantine", "block"}:
		return policy_action
	if severity in {"medium", "high"} and policy_action == "alert":
		return "alert"
	return "allow"
