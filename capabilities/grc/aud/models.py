"""Pydantic v2 models for grc_aud capability.

© 2025 Datacraft  |  Author: Nyimbi Odero
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid6 import uuid7

from pydantic import BaseModel, ConfigDict, Field

uuid7str = lambda: str(uuid7())


def _now() -> str:
	return datetime.now(timezone.utc).isoformat()


# ── Enums ─────────────────────────────────────────────────────────────────────

class AuditType(str, Enum):
	internal = "internal"
	external = "external"
	regulatory = "regulatory"
	iso_certification = "iso_certification"
	soc2 = "soc2"
	penetration_test = "penetration_test"
	supplier = "supplier"
	it_general_controls = "it_general_controls"


class AuditStatus(str, Enum):
	planned = "planned"
	in_progress = "in_progress"
	fieldwork = "fieldwork"
	review = "review"
	report_draft = "report_draft"
	report_final = "report_final"
	closed = "closed"
	cancelled = "cancelled"


class FindingSeverity(str, Enum):
	observation = "observation"
	minor = "minor"
	major = "major"
	critical = "critical"


class FindingStatus(str, Enum):
	open = "open"
	in_remediation = "in_remediation"
	remediated = "remediated"
	accepted = "accepted"
	closed = "closed"
	disputed = "disputed"


class QARating(str, Enum):
	satisfactory = "satisfactory"
	needs_improvement = "needs_improvement"
	unsatisfactory = "unsatisfactory"


class FollowUpStatus(str, Enum):
	verified_closed = "verified_closed"
	partially_remediated = "partially_remediated"
	not_remediated = "not_remediated"


# ── Core models ───────────────────────────────────────────────────────────────

class _Base(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)


class AuditEngagement(_Base):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	created_at: str = Field(default_factory=_now)
	plan_id: str
	entity_id: str | None = None
	area: str
	audit_type: AuditType = AuditType.internal
	scope: str = "process"
	objectives: list[str] = Field(default_factory=list)
	lead_auditor_id: str
	auditee_id: str | None = None
	start_date: str
	end_date: str
	planned_hours: int = 80
	actual_hours: int = 0
	status: AuditStatus = AuditStatus.planned
	finding_ids: list[str] = Field(default_factory=list)
	report_id: str | None = None
	updated_at: str = Field(default_factory=_now)


class AuditFinding(_Base):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	created_at: str = Field(default_factory=_now)
	engagement_id: str
	plan_id: str | None = None
	entity_id: str | None = None
	area_tested: str
	finding_type: str
	observation: str
	criteria: str
	risk_rating: FindingSeverity
	evidence_ids: list[str] = Field(default_factory=list)
	status: FindingStatus = FindingStatus.open
	owner_id: str | None = None
	management_response: str | None = None
	remediation_deadline: str | None = None
	raised_at: str = Field(default_factory=_now)
	updated_at: str = Field(default_factory=_now)


class ManagementResponse(_Base):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str | None = None
	created_at: str = Field(default_factory=_now)
	finding_id: str
	response_text: str
	action_plan: str
	owner_id: str
	deadline: str
	received_at: str = Field(default_factory=_now)


class FollowUp(_Base):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	created_at: str = Field(default_factory=_now)
	finding_id: str
	follow_up_date: str
	status: FollowUpStatus
	evidence_ids: list[str] = Field(default_factory=list)


# ── Request / Response ────────────────────────────────────────────────────────

class CreateAuditPlanRequest(_Base):
	entity_id: str
	year: int
	risk_based_areas: list[str]
	approved_by: str
	plan_type: str = "annual"
	methodology: str = "risk_based"


class CreateEngagementRequest(_Base):
	plan_id: str
	area: str
	objectives: list[str]
	start_date: str
	end_date: str
	lead_auditor_id: str
	audit_type: AuditType = AuditType.internal
	scope: str = "process"
	auditee_id: str | None = None
	planned_hours: int = 80


class FieldworkRequest(_Base):
	area_tested: str
	finding_type: str
	observation: str
	criteria: str
	evidence: list[dict[str, Any]]
	risk_rating: FindingSeverity
	auditor_id: str | None = None


class DraftReportRequest(_Base):
	findings: list[str]
	recommendations: list[str]
	auditor_id: str


class ManagementResponseRequest(_Base):
	response_text: str
	action_plan: str
	owner_id: str
	deadline: str


class FinaliseReportRequest(_Base):
	chief_audit_executive_id: str
	sign_off_date: str


class IssueTrackingRequest(_Base):
	status: FindingStatus
	progress_notes: str
	updated_by: str


class FollowUpRequest(_Base):
	follow_up_date: str
	status: FollowUpStatus
	evidence: list[dict[str, Any]]


class QAReviewRequest(_Base):
	reviewer_id: str
	rating: QARating
	observations: str
