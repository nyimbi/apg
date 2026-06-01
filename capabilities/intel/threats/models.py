"""In-memory models for APG Threat Intelligence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ThreatAuthority:
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
class ThreatWorkspace:
	id: str
	tenant_id: str
	workspace_type: str
	name: str
	classification: str
	authority_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ThreatSource:
	id: str
	tenant_id: str
	workspace_id: str
	source_type: str
	source_reference: str
	custodian_id: str
	lineage_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ThreatIndicator:
	id: str
	tenant_id: str
	source_id: str
	indicator_type: str
	indicator_reference: str
	confidence_score: float
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ThreatActor:
	id: str
	tenant_id: str
	workspace_id: str
	actor_type: str
	actor_reference: str
	confidence_score: float
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ThreatCampaign:
	id: str
	tenant_id: str
	actor_id: str
	campaign_type: str
	campaign_reference: str
	risk_level: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ThreatAssessment:
	id: str
	tenant_id: str
	campaign_id: str
	assessment_type: str
	risk_level: str
	confidence_score: float
	analyst_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ThreatReport:
	id: str
	tenant_id: str
	assessment_id: str
	report_type: str
	report_reference: str
	approval_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ThreatMitigation:
	id: str
	tenant_id: str
	assessment_id: str
	mitigation_type: str
	action_reference: str
	approval_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ThreatReview:
	id: str
	tenant_id: str
	reference_id: str
	reviewer_id: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ThreatAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)

