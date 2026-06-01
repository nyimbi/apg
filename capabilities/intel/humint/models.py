"""In-memory models for APG Human Intelligence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class SourceAuthority:
	id: str
	tenant_id: str
	authority_type: str
	scope_reference: str
	classification: str
	approver_id: str
	expires_at: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class HumanSource:
	id: str
	tenant_id: str
	source_type: str
	handling_status: str
	risk_level: str
	owner_id: str
	authority_id: str
	protection_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ContactPlan:
	id: str
	tenant_id: str
	authority_id: str
	source_id: str
	contact_method: str
	objective_reference: str
	safety_plan_reference: str
	approval_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ContactReport:
	id: str
	tenant_id: str
	plan_id: str
	report_reference: str
	handler_id: str
	source_welfare_score: float
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class Debriefing:
	id: str
	tenant_id: str
	report_id: str
	topic: str
	classification: str
	credibility_score: float
	analyst_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ReliabilityAssessment:
	id: str
	tenant_id: str
	source_id: str
	reliability_grade: str
	confidence_score: float
	analyst_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class HUMINTLead:
	id: str
	tenant_id: str
	debriefing_id: str
	lead_type: str
	priority: str
	analyst_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class HUMINTDissemination:
	id: str
	tenant_id: str
	lead_id: str
	audience: str
	release_marking: str
	approval_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class HUMINTReview:
	id: str
	tenant_id: str
	reference_id: str
	reviewer_id: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class HUMINTAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
