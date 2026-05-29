"""Runtime helpers for the APG Scraper/Data Harvesting capability."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from typing import Any


SOURCE_TYPES = {"api", "website", "database", "file", "feed"}
EXTRACTOR_TYPES = {"html", "json", "csv", "xml", "text", "api"}
HARVEST_MODES = {"full", "incremental", "sample"}
RUN_STATUSES = {"queued", "running", "succeeded", "failed", "blocked", "cancelled"}
DLP_STATUSES = {"not_required", "pending", "passed", "failed"}


def stable_id(prefix: str, *parts: object) -> str:
	"""Build a deterministic short ID for local package runtime objects."""
	digest = sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:12]
	return f"{prefix}_{digest}"


def utc_now() -> datetime:
	return datetime.now(timezone.utc)


def normalize_tags(tags: list[str] | None) -> list[str]:
	return sorted({tag.strip().lower() for tag in tags or [] if tag and tag.strip()})


def normalize_source_type(source_type: str) -> str:
	value = source_type.strip().lower()
	if value not in SOURCE_TYPES:
		raise ValueError(f"unsupported_source_type:{source_type}")
	return value


def normalize_extractor_type(extractor_type: str) -> str:
	value = extractor_type.strip().lower()
	if value not in EXTRACTOR_TYPES:
		raise ValueError(f"unsupported_extractor_type:{extractor_type}")
	return value


def normalize_harvest_mode(mode: str) -> str:
	value = mode.strip().lower()
	if value not in HARVEST_MODES:
		raise ValueError(f"unsupported_harvest_mode:{mode}")
	return value


def classify_dlp_status(pii_expected: bool, scanned: bool, violations: int = 0) -> str:
	if not pii_expected:
		return "not_required"
	if not scanned:
		return "pending"
	if violations:
		return "failed"
	return "passed"


def run_status(records_extracted: int, errors: int = 0, blocked: bool = False) -> str:
	if blocked:
		return "blocked"
	if errors:
		return "failed"
	if records_extracted >= 0:
		return "succeeded"
	return "failed"


def result_retention_until(retention_days: int) -> str:
	if retention_days <= 0:
		raise ValueError("result_retention_days_must_be_positive")
	return f"+{retention_days}d"


def summarize_decision(result: dict[str, Any]) -> str:
	actions = result.get("actions") or []
	if not actions:
		return result.get("decision", "allow")
	return ",".join(action.get("reason", action.get("decision", "policy_action")) for action in actions)
