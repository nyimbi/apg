"""In-memory models for APG Law Enforcement & Justice."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class IncidentReport:
	id: str
	tenant_id: str
	incident_type: str
	ob_number: str
	reporting_officer_id: str
	location_reference: str
	complainant_id: str
	description: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CaseDocket:
	id: str
	tenant_id: str
	incident_id: str
	investigating_officer_id: str
	status: str
	docket_number: str
	opened_date: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class EvidenceItem:
	id: str
	tenant_id: str
	docket_id: str
	evidence_type: str
	description: str
	custodian_id: str
	evidence_reference: str
	current_location: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CustodyAction:
	id: str
	tenant_id: str
	evidence_id: str
	custody_action: str
	actor_id: str
	from_location: str
	to_location: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CourtHearing:
	id: str
	tenant_id: str
	docket_id: str
	court_type: str
	hearing_type: str
	court_reference: str
	hearing_date: str
	presiding_judge: str
	outcome: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ProsecutionRecord:
	id: str
	tenant_id: str
	docket_id: str
	dpp_reference: str
	prosecution_status: str
	charges: str
	prosecutor_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class LawEnforcementReview:
	id: str
	tenant_id: str
	reference_id: str
	reviewer_id: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class LawEnforcementAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
