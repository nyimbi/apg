"""In-memory models for APG Permits Management."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class PermitApplication:
	id: str
	tenant_id: str
	permit_type: str
	applicant_id: str
	site_reference: str
	status: str
	fee_paid: bool
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class Permit:
	id: str
	tenant_id: str
	application_id: str
	permit_type: str
	permit_number: str
	holder_id: str
	site_reference: str
	issued_date: str
	expiry_date: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class PermitCondition:
	id: str
	tenant_id: str
	permit_id: str
	condition_type: str
	description: str
	due_date: str
	responsible_party: str
	fulfilled: bool
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class PermitInspection:
	id: str
	tenant_id: str
	permit_id: str
	inspection_type: str
	inspector_id: str
	scheduled_date: str
	outcome: str
	findings: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ComplianceRecord:
	id: str
	tenant_id: str
	permit_id: str
	compliance_status: str
	officer_id: str
	assessment_date: str
	narrative: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class EnforcementAction:
	id: str
	tenant_id: str
	permit_id: str
	compliance_id: str
	action_type: str
	officer_id: str
	description: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class PermitReview:
	id: str
	tenant_id: str
	reference_id: str
	reviewer_id: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class PermitsAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
