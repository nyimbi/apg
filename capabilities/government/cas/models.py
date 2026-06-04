"""In-memory models for APG Case Management."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class CitizenCase:
	id: str
	tenant_id: str
	case_type: str
	intake_channel: str
	citizen_id: str
	priority: str
	status: str
	subject: str
	description: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CaseAssignment:
	id: str
	tenant_id: str
	case_id: str
	assignment_type: str
	assignee_id: str
	assigned_by: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CaseEscalation:
	id: str
	tenant_id: str
	case_id: str
	escalation_reason: str
	escalated_to: str
	supervisor_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class SlaRecord:
	id: str
	tenant_id: str
	case_id: str
	sla_category: str
	due_date: str
	breached: bool
	breach_notified: bool

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CaseOutcome:
	id: str
	tenant_id: str
	case_id: str
	outcome_type: str
	description: str
	approval_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CaseNotification:
	id: str
	tenant_id: str
	case_id: str
	notification_type: str
	recipient_id: str
	message: str
	sent: bool

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CaseReview:
	id: str
	tenant_id: str
	reference_id: str
	reviewer_id: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CaseAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
