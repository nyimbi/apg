"""Dependency-light APG models for the RCM capability.

The package keeps production database, analytics, and collaboration adapters
behind explicit integration boundaries. These dataclasses are the executable
APG package surface used by tests, API helpers, and generated applications.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now_iso() -> str:
	"""Return a stable UTC timestamp for package audit records."""
	return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class GRCRiskLevel(str, Enum):
	"""Risk severity levels derived from residual risk score."""

	CRITICAL = "critical"
	HIGH = "high"
	MEDIUM = "medium"
	LOW = "low"
	MINIMAL = "minimal"


class GRCRiskStatus(str, Enum):
	"""Risk lifecycle states."""

	IDENTIFIED = "identified"
	ASSESSED = "assessed"
	TREATED = "treated"
	MONITORED = "monitored"
	CLOSED = "closed"
	ESCALATED = "escalated"


class GRCComplianceStatus(str, Enum):
	"""Compliance assessment states."""

	COMPLIANT = "compliant"
	NON_COMPLIANT = "non_compliant"
	PARTIALLY_COMPLIANT = "partially_compliant"
	NOT_ASSESSED = "not_assessed"
	PENDING_REVIEW = "pending_review"
	EXCEPTION_APPROVED = "exception_approved"


class GRCControlType(str, Enum):
	"""Control classifications."""

	PREVENTIVE = "preventive"
	DETECTIVE = "detective"
	CORRECTIVE = "corrective"
	COMPENSATING = "compensating"
	DIRECTIVE = "directive"


class GRCGovernanceDecisionType(str, Enum):
	"""Governance decision categories."""

	POLICY_APPROVAL = "policy_approval"
	RISK_ACCEPTANCE = "risk_acceptance"
	BUDGET_ALLOCATION = "budget_allocation"
	STRATEGIC_DIRECTION = "strategic_direction"
	COMPLIANCE_EXCEPTION = "compliance_exception"
	OPERATIONAL_CHANGE = "operational_change"


def risk_level_from_score(score: float) -> GRCRiskLevel:
	"""Map a 0-100 residual risk score to an APG RCM risk level."""
	if score >= 90:
		return GRCRiskLevel.CRITICAL
	if score >= 70:
		return GRCRiskLevel.HIGH
	if score >= 40:
		return GRCRiskLevel.MEDIUM
	if score >= 20:
		return GRCRiskLevel.LOW
	return GRCRiskLevel.MINIMAL


def compliance_status_from_control(
	design_effective: bool,
	operating_effective: bool,
	findings: list[str],
) -> GRCComplianceStatus:
	"""Determine compliance status from control testing evidence."""
	if design_effective and operating_effective and not findings:
		return GRCComplianceStatus.COMPLIANT
	if design_effective or operating_effective:
		return GRCComplianceStatus.PARTIALLY_COMPLIANT
	return GRCComplianceStatus.NON_COMPLIANT


@dataclass(slots=True)
class RCMRisk:
	"""Tenant-scoped enterprise risk register entry."""

	id: str
	tenant_id: str
	title: str
	category: str
	owner_id: str
	probability: float
	impact: float
	control_effectiveness: float = 0.0
	status: GRCRiskStatus = GRCRiskStatus.IDENTIFIED
	tags: list[str] = field(default_factory=list)
	metadata: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=utc_now_iso)

	@property
	def inherent_score(self) -> float:
		return round(self.probability * self.impact * 100, 2)

	@property
	def residual_score(self) -> float:
		return round(self.inherent_score * (1 - self.control_effectiveness), 2)

	@property
	def level(self) -> GRCRiskLevel:
		return risk_level_from_score(self.residual_score)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"kind": "risk",
			"tenant_id": self.tenant_id,
			"title": self.title,
			"category": self.category,
			"owner_id": self.owner_id,
			"probability": self.probability,
			"impact": self.impact,
			"control_effectiveness": self.control_effectiveness,
			"inherent_score": self.inherent_score,
			"residual_score": self.residual_score,
			"risk_level": self.level.value,
			"status": self.status.value,
			"tags": list(self.tags),
			"metadata": dict(self.metadata),
			"created_at": self.created_at,
		}


@dataclass(slots=True)
class RCMControl:
	"""Control mapped to one or more tenant risks."""

	id: str
	tenant_id: str
	name: str
	owner_id: str
	control_type: GRCControlType
	mapped_risk_ids: list[str]
	effectiveness: float
	test_frequency_days: int = 90
	last_test_status: GRCComplianceStatus = GRCComplianceStatus.NOT_ASSESSED
	metadata: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"kind": "control",
			"tenant_id": self.tenant_id,
			"name": self.name,
			"owner_id": self.owner_id,
			"control_type": self.control_type.value,
			"mapped_risk_ids": list(self.mapped_risk_ids),
			"effectiveness": self.effectiveness,
			"test_frequency_days": self.test_frequency_days,
			"last_test_status": self.last_test_status.value,
			"metadata": dict(self.metadata),
			"created_at": self.created_at,
		}


@dataclass(slots=True)
class RCMComplianceObligation:
	"""Regulatory or policy obligation mapped to controls."""

	id: str
	tenant_id: str
	framework: str
	requirement: str
	owner_id: str
	jurisdiction: str
	due_date: str
	mapped_control_ids: list[str]
	status: GRCComplianceStatus = GRCComplianceStatus.PENDING_REVIEW
	metadata: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"kind": "obligation",
			"tenant_id": self.tenant_id,
			"framework": self.framework,
			"requirement": self.requirement,
			"owner_id": self.owner_id,
			"jurisdiction": self.jurisdiction,
			"due_date": self.due_date,
			"mapped_control_ids": list(self.mapped_control_ids),
			"status": self.status.value,
			"metadata": dict(self.metadata),
			"created_at": self.created_at,
		}


@dataclass(slots=True)
class RCMControlAssessment:
	"""Point-in-time control test result."""

	id: str
	tenant_id: str
	control_id: str
	assessor_id: str
	design_effective: bool
	operating_effective: bool
	evidence_refs: list[str]
	findings: list[str] = field(default_factory=list)
	created_at: str = field(default_factory=utc_now_iso)

	@property
	def status(self) -> GRCComplianceStatus:
		return compliance_status_from_control(self.design_effective, self.operating_effective, self.findings)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"kind": "assessment",
			"tenant_id": self.tenant_id,
			"control_id": self.control_id,
			"assessor_id": self.assessor_id,
			"design_effective": self.design_effective,
			"operating_effective": self.operating_effective,
			"evidence_refs": list(self.evidence_refs),
			"findings": list(self.findings),
			"status": self.status.value,
			"created_at": self.created_at,
		}


@dataclass(slots=True)
class RCMGovernanceDecision:
	"""Governance decision with approval evidence."""

	id: str
	tenant_id: str
	title: str
	decision_type: GRCGovernanceDecisionType
	approver_id: str
	related_risk_ids: list[str]
	rationale: str
	approved: bool
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"kind": "governance_decision",
			"tenant_id": self.tenant_id,
			"title": self.title,
			"decision_type": self.decision_type.value,
			"approver_id": self.approver_id,
			"related_risk_ids": list(self.related_risk_ids),
			"rationale": self.rationale,
			"approved": self.approved,
			"created_at": self.created_at,
		}


@dataclass(slots=True)
class RCMEvidence:
	"""Evidence artifact linked to a control or obligation."""

	id: str
	tenant_id: str
	source: str
	linked_control_id: str | None = None
	linked_obligation_id: str | None = None
	encrypted: bool = True
	retention_days: int = 2555
	metadata: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"kind": "evidence",
			"tenant_id": self.tenant_id,
			"source": self.source,
			"linked_control_id": self.linked_control_id,
			"linked_obligation_id": self.linked_obligation_id,
			"encrypted": self.encrypted,
			"retention_days": self.retention_days,
			"metadata": dict(self.metadata),
			"created_at": self.created_at,
		}


@dataclass(slots=True)
class RCMAuditEvent:
	"""Lightweight audit event emitted by the RCM service facade."""

	id: str
	tenant_id: str
	action: str
	subject_id: str
	details: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"kind": "audit_event",
			"tenant_id": self.tenant_id,
			"action": self.action,
			"subject_id": self.subject_id,
			"details": dict(self.details),
			"created_at": self.created_at,
		}


# Compatibility aliases for older local imports.
GRCRisk = RCMRisk
GRCControl = RCMControl
GRCRegulation = RCMComplianceObligation
GRCRiskAssessment = RCMControlAssessment
GRCPolicy = RCMComplianceObligation
GRCGovernanceDecision = RCMGovernanceDecision


__all__ = [
	"GRCRiskLevel",
	"GRCRiskStatus",
	"GRCComplianceStatus",
	"GRCControlType",
	"GRCGovernanceDecisionType",
	"RCMRisk",
	"RCMControl",
	"RCMComplianceObligation",
	"RCMControlAssessment",
	"RCMGovernanceDecision",
	"RCMEvidence",
	"RCMAuditEvent",
	"GRCRisk",
	"GRCControl",
	"GRCRegulation",
	"GRCRiskAssessment",
	"GRCPolicy",
	"GRCGovernanceDecision",
	"risk_level_from_score",
	"compliance_status_from_control",
]
