"""In-memory models for APG Emergency Management."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class EmergencyIncident:
	id: str
	tenant_id: str
	incident_type: str
	severity: str
	phase: str
	location_reference: str
	commander_id: str
	description: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ResourceMobilisation:
	id: str
	tenant_id: str
	incident_id: str
	resource_type: str
	quantity: int
	unit: str
	responsible_agency: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class AgencyActivation:
	id: str
	tenant_id: str
	incident_id: str
	agency_type: str
	agency_name: str
	contact_reference: str
	role: str
	activated_at: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class EocRecord:
	id: str
	tenant_id: str
	incident_id: str
	eoc_status: str
	command_structure: str
	activation_authority: str
	activated_at: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class SituationReport:
	id: str
	tenant_id: str
	incident_id: str
	period: str
	author_id: str
	summary: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class AfterActionReview:
	id: str
	tenant_id: str
	incident_id: str
	reviewer_id: str
	lessons_learned: str
	recommendations: str
	evidence_reference: str
	status: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class EmergencyReview:
	id: str
	tenant_id: str
	reference_id: str
	reviewer_id: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class EmergencyAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
