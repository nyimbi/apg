"""Dependency-light Security Operations runtime primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any


SEVERITIES = {"low", "medium", "high", "critical"}
DETECTION_STATES = {"new", "review_required", "triaged", "linked"}
INCIDENT_STATES = {"open", "escalated", "responding", "contained", "closed"}
RESPONSE_STATES = {"planned", "executed", "blocked"}


def utc_now() -> str:
	return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_id(prefix: str, *parts: object) -> str:
	seed = "|".join(str(part).strip().lower() for part in parts if str(part).strip())
	digest = sha256(seed.encode("utf-8")).hexdigest()[:16]
	return f"{prefix}_{digest}"


def normalize_severity(severity: str) -> str:
	value = str(severity or "").strip().lower()
	if value in {"info", "informational"}:
		value = "low"
	if value not in SEVERITIES:
		raise ValueError(f"unsupported_incident_severity:{severity}")
	return value


def normalize_confidence(confidence: float | int) -> float:
	value = float(confidence)
	if value < 0 or value > 1:
		raise ValueError(f"anomaly_confidence_out_of_range:{confidence}")
	return round(value, 3)


def response_required_actions(rule_result: dict[str, Any]) -> list[str]:
	return [
		str(action["required_action"])
		for action in rule_result.get("actions", [])
		if action.get("required_action")
	]


def serialize(record: object) -> dict[str, Any]:
	return asdict(record)


@dataclass(slots=True)
class DetectionRecord:
	id: str
	tenant_id: str
	title: str
	alert_source: str
	severity: str
	anomaly_confidence: float
	status: str
	signal_refs: list[str] = field(default_factory=list)
	owner: str | None = None
	matched_rules: list[str] = field(default_factory=list)
	required_actions: list[str] = field(default_factory=list)
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class IncidentRecord:
	id: str
	tenant_id: str
	title: str
	owner: str
	severity: str
	status: str
	detection_ids: list[str] = field(default_factory=list)
	evidence_refs: list[str] = field(default_factory=list)
	escalation_recorded: bool = False
	created_at: str = field(default_factory=utc_now)
	closed_at: str | None = None

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class PlaybookRecord:
	id: str
	tenant_id: str
	name: str
	owner: str
	steps: list[str]
	approved_by: str
	approved: bool = True
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class ResponseActionRecord:
	id: str
	tenant_id: str
	incident_id: str
	playbook_id: str
	action: str
	actor: str
	status: str
	required_actions: list[str] = field(default_factory=list)
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class PostureControlRecord:
	id: str
	tenant_id: str
	control_id: str
	domain: str
	coverage: float
	owner: str
	status: str
	updated_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class OpsAuditEventRecord:
	id: str
	tenant_id: str
	event_type: str
	subject_id: str
	message: str
	actor: str
	severity: str = "low"
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


__all__ = [
	"DETECTION_STATES",
	"INCIDENT_STATES",
	"RESPONSE_STATES",
	"SEVERITIES",
	"DetectionRecord",
	"IncidentRecord",
	"OpsAuditEventRecord",
	"PlaybookRecord",
	"PostureControlRecord",
	"ResponseActionRecord",
	"normalize_confidence",
	"normalize_severity",
	"response_required_actions",
	"stable_id",
	"utc_now",
]
