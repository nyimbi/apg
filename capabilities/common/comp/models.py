"""Compliance-domain models for the APG COMP capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> datetime:
	"""Return a timezone-aware timestamp for deterministic local state."""
	return datetime.now(timezone.utc)


def isoformat(value: datetime) -> str:
	return value.astimezone(timezone.utc).isoformat()


@dataclass
class ComplianceFramework:
	id: str
	tenant_id: str
	name: str
	owner: str
	obligations: list[str]
	policy_version: str
	status: str = "active"
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"owner": self.owner,
			"obligations": list(self.obligations),
			"policy_version": self.policy_version,
			"status": self.status,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class ComplianceControl:
	id: str
	tenant_id: str
	framework_id: str
	name: str
	owner: str
	control_type: str
	regulated_data_scope: bool
	dlp_policy_linked: bool
	testing_frequency_days: int
	status: str = "active"
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"framework_id": self.framework_id,
			"name": self.name,
			"owner": self.owner,
			"control_type": self.control_type,
			"regulated_data_scope": self.regulated_data_scope,
			"dlp_policy_linked": self.dlp_policy_linked,
			"testing_frequency_days": self.testing_frequency_days,
			"status": self.status,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class EvidenceRecord:
	id: str
	tenant_id: str
	control_id: str
	source: str
	collected_by: str
	encrypted: bool
	immutable_reference: str
	collected_at: datetime = field(default_factory=utc_now)
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"control_id": self.control_id,
			"source": self.source,
			"collected_by": self.collected_by,
			"encrypted": self.encrypted,
			"immutable_reference": self.immutable_reference,
			"collected_at": isoformat(self.collected_at),
			"metadata": dict(self.metadata),
		}


@dataclass
class ControlAssessment:
	id: str
	tenant_id: str
	control_id: str
	evidence_id: str
	result: str
	tested_by: str
	evidence_age_days: int
	findings: list[str] = field(default_factory=list)
	assessed_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"control_id": self.control_id,
			"evidence_id": self.evidence_id,
			"result": self.result,
			"tested_by": self.tested_by,
			"evidence_age_days": self.evidence_age_days,
			"findings": list(self.findings),
			"assessed_at": isoformat(self.assessed_at),
		}


@dataclass
class ComplianceFinding:
	id: str
	tenant_id: str
	control_id: str
	severity: str
	description: str
	owner: str
	due_at: datetime
	status: str = "open"
	escalated: bool = False
	created_at: datetime = field(default_factory=utc_now)
	remediation_plan: str = ""

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"control_id": self.control_id,
			"severity": self.severity,
			"description": self.description,
			"owner": self.owner,
			"due_at": isoformat(self.due_at),
			"status": self.status,
			"escalated": self.escalated,
			"created_at": isoformat(self.created_at),
			"remediation_plan": self.remediation_plan,
		}


@dataclass
class ComplianceReport:
	id: str
	tenant_id: str
	framework_id: str
	period: str
	prepared_by: str
	status: str = "draft"
	approved_by: str | None = None
	approved_at: datetime | None = None
	published_at: datetime | None = None
	control_count: int = 0
	finding_count: int = 0

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"framework_id": self.framework_id,
			"period": self.period,
			"prepared_by": self.prepared_by,
			"status": self.status,
			"approved_by": self.approved_by,
			"approved_at": isoformat(self.approved_at) if self.approved_at else None,
			"published_at": isoformat(self.published_at) if self.published_at else None,
			"control_count": self.control_count,
			"finding_count": self.finding_count,
		}


@dataclass
class AttestationRecord:
	id: str
	tenant_id: str
	report_id: str
	attested_by: str
	statement: str
	attested_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"report_id": self.report_id,
			"attested_by": self.attested_by,
			"statement": self.statement,
			"attested_at": isoformat(self.attested_at),
		}


@dataclass
class ComplianceAgentRecord:
	"""First-class AI agent assigned to a governed compliance scope."""

	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str
	owner: str
	purpose: str
	contribution_disclosed: bool = True
	human_approval_required: bool = False
	status: str = "active"
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"runtime": self.runtime,
			"role": self.role,
			"scope": self.scope,
			"owner": self.owner,
			"purpose": self.purpose,
			"contribution_disclosed": self.contribution_disclosed,
			"human_approval_required": self.human_approval_required,
			"status": self.status,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class CompLifecycleBatchRecord:
	"""Bytewax lifecycle batch evidence for compliance mutations."""

	id: str
	tenant_id: str
	event_stream: str
	mutation_count: int
	operation: str
	accepted: bool
	decision: str
	matched_rules: list[str] = field(default_factory=list)
	status: str = "accepted"
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"event_stream": self.event_stream,
			"mutation_count": self.mutation_count,
			"operation": self.operation,
			"accepted": self.accepted,
			"decision": self.decision,
			"matched_rules": list(self.matched_rules),
			"status": self.status,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class ComplianceAuditEvent:
	id: str
	tenant_id: str
	event_type: str
	subject_id: str
	actor: str
	payload_hash: str
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"event_type": self.event_type,
			"subject_id": self.subject_id,
			"actor": self.actor,
			"payload_hash": self.payload_hash,
			"created_at": isoformat(self.created_at),
		}
