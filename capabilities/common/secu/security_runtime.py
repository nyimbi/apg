"""Dependency-light SECU security runtime primitives.

The larger SECU package contains async integration engines for APG platform
services. This module provides the deterministic local surface that capability
composition, generated apps, tests, and publish tooling can execute without
live identity, SIEM, EDR, compliance, or policy-provider integrations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any


SECURITY_LEVELS = {"public", "internal", "confidential", "restricted", "critical"}
DEVICE_TRUST_STATES = {"trusted", "known", "unknown", "suspicious", "compromised"}
THREAT_SEVERITIES = {"info", "low", "medium", "high", "critical"}
CONTROL_STATUSES = {"implemented", "evidence_required", "non_compliant", "waived"}
EXCEPTION_STATUSES = {"pending", "approved", "rejected", "expired"}
INCIDENT_STATUSES = {"open", "contained", "resolved"}
SECU_AGENT_STATUSES = {"active", "suspended", "retired"}


def utc_now() -> str:
	"""Return a compact UTC timestamp for deterministic package records."""
	return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_id(prefix: str, *parts: object) -> str:
	"""Build a stable package-local identifier from meaningful record parts."""
	seed = "|".join(str(part).strip().lower() for part in parts if str(part).strip())
	digest = sha256(seed.encode("utf-8")).hexdigest()[:16]
	return f"{prefix}_{digest}"


def normalize_security_level(level: str) -> str:
	value = str(level or "").strip().lower()
	if value == "secret":
		value = "restricted"
	if value == "top_secret":
		value = "critical"
	if value not in SECURITY_LEVELS:
		raise ValueError(f"unsupported_security_level:{level}")
	return value


def normalize_device_trust(state: str) -> str:
	value = str(state or "").strip().lower()
	if value not in DEVICE_TRUST_STATES:
		raise ValueError(f"unsupported_device_trust:{state}")
	return value


def normalize_threat_severity(severity: str) -> str:
	value = str(severity or "").strip().lower()
	if value not in THREAT_SEVERITIES:
		raise ValueError(f"unsupported_threat_severity:{severity}")
	return value


def normalize_tags(tags: list[str] | None) -> list[str]:
	return sorted({str(tag).strip().lower().replace(" ", "_") for tag in tags or [] if str(tag).strip()})


def clamp_score(score: float | int) -> int:
	value = int(round(float(score)))
	if value < 0 or value > 100:
		raise ValueError(f"risk_score_out_of_range:{score}")
	return value


def risk_band(score: float | int) -> str:
	value = clamp_score(score)
	if value >= 90:
		return "critical"
	if value >= 70:
		return "high"
	if value >= 50:
		return "elevated"
	return "normal"


def required_actions(rule_result: dict[str, Any]) -> list[str]:
	return [
		str(action["required_action"])
		for action in rule_result.get("actions", [])
		if action.get("required_action")
	]


def summarize_decision(rule_result: dict[str, Any]) -> str:
	decision = rule_result.get("decision", "allow")
	if decision == "deny":
		return "access_denied"
	if decision == "quarantine":
		return "device_quarantine_required"
	if decision == "challenge":
		return "step_up_or_evidence_required"
	return "access_allowed"


def control_status(compliant: bool, evidence_attached: bool, waived: bool = False) -> str:
	if waived:
		return "waived"
	if compliant and evidence_attached:
		return "implemented"
	if not compliant and not evidence_attached:
		return "evidence_required"
	return "non_compliant"


def serialize_record(record: object) -> dict[str, Any]:
	return asdict(record)


@dataclass(slots=True)
class SecurityPolicyRecord:
	id: str
	tenant_id: str
	name: str
	owner: str
	security_level: str
	required_controls: list[str] = field(default_factory=list)
	applies_to: list[str] = field(default_factory=list)
	enabled: bool = True
	tags: list[str] = field(default_factory=list)
	created_at: str = field(default_factory=utc_now)
	updated_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize_record(self)


@dataclass(slots=True)
class DevicePostureRecord:
	id: str
	tenant_id: str
	device_id: str
	user_id: str
	trust_state: str
	managed: bool
	risk_score: int
	indicators: list[str] = field(default_factory=list)
	quarantined: bool = False
	last_seen_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize_record(self)


@dataclass(slots=True)
class ThreatIndicatorRecord:
	id: str
	tenant_id: str
	name: str
	indicator_type: str
	value: str
	severity: str
	source: str
	ttl_hours: int
	active: bool = True
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize_record(self)


@dataclass(slots=True)
class RiskAssessmentRecord:
	id: str
	tenant_id: str
	subject_id: str
	subject_type: str
	risk_score: int
	risk_band: str
	decision: str
	summary: str
	matched_rules: list[str]
	required_actions: list[str]
	device_id: str | None = None
	challenge_completed: bool = False
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize_record(self)


@dataclass(slots=True)
class ComplianceControlRecord:
	id: str
	tenant_id: str
	framework: str
	control_id: str
	owner: str
	status: str
	compliant: bool
	audit_evidence_attached: bool
	evidence_ref: str | None = None
	assessed_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize_record(self)


@dataclass(slots=True)
class SecurityAuditEventRecord:
	id: str
	tenant_id: str
	event_type: str
	subject_id: str
	message: str
	actor: str
	severity: str = "info"
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize_record(self)


@dataclass(slots=True)
class PolicyExceptionRecord:
	id: str
	tenant_id: str
	policy_id: str
	requested_by: str
	reason: str
	expires_at: str
	status: str = "pending"
	decision: str = ""
	reviewer: str = ""
	notes: str = ""
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize_record(self)


@dataclass(slots=True)
class SecurityIncidentRecord:
	id: str
	tenant_id: str
	title: str
	severity: str
	opened_by: str
	status: str = "open"
	containment_action: str = ""
	containment_evidence: str = ""
	resolution: str = ""
	resolved_by: str = ""
	opened_at: str = field(default_factory=utc_now)
	contained_at: str = ""
	resolved_at: str = ""

	def to_dict(self) -> dict[str, Any]:
		return serialize_record(self)


@dataclass(slots=True)
class SecurityAgentRecord:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str
	owner: str
	purpose: str
	contribution_disclosed: bool
	human_approval_required: bool
	policy_ref: str | None = None
	status: str = "active"
	registered_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize_record(self)


__all__ = [
	"CONTROL_STATUSES",
	"DEVICE_TRUST_STATES",
	"EXCEPTION_STATUSES",
	"INCIDENT_STATUSES",
	"SECURITY_LEVELS",
	"SECU_AGENT_STATUSES",
	"THREAT_SEVERITIES",
	"ComplianceControlRecord",
	"DevicePostureRecord",
	"PolicyExceptionRecord",
	"RiskAssessmentRecord",
	"SecurityAuditEventRecord",
	"SecurityAgentRecord",
	"SecurityIncidentRecord",
	"SecurityPolicyRecord",
	"ThreatIndicatorRecord",
	"clamp_score",
	"control_status",
	"normalize_device_trust",
	"normalize_security_level",
	"normalize_tags",
	"normalize_threat_severity",
	"required_actions",
	"risk_band",
	"stable_id",
	"summarize_decision",
	"utc_now",
]
