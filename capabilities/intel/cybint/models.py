"""In-memory models for APG Cyber Intelligence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class CyberAuthority:
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
class Indicator:
	id: str
	tenant_id: str
	indicator_type: str
	indicator_value: str
	tlp: str
	confidence_score: float
	authority_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class Sighting:
	id: str
	tenant_id: str
	indicator_id: str
	source_reference: str
	observed_at: str
	severity: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class Enrichment:
	id: str
	tenant_id: str
	indicator_id: str
	enrichment_type: str
	provider_reference: str
	confidence_score: float
	analyst_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ThreatProfile:
	id: str
	tenant_id: str
	profile_type: str
	name: str
	classification: str
	confidence_score: float
	analyst_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CyberRiskAssessment:
	id: str
	tenant_id: str
	indicator_id: str
	profile_id: str
	risk_level: str
	confidence_score: float
	analyst_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class IncidentLink:
	id: str
	tenant_id: str
	assessment_id: str
	incident_reference: str
	response_priority: str
	owner_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CYBINTDissemination:
	id: str
	tenant_id: str
	assessment_id: str
	audience: str
	release_marking: str
	approval_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CYBINTReview:
	id: str
	tenant_id: str
	reference_id: str
	reviewer_id: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CYBINTAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
