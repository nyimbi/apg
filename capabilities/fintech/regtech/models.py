"""In-memory models for APG Regulatory Technology."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class RegulatorySource:
	id: str
	tenant_id: str
	regulator: str
	jurisdiction: str
	source_reference: str
	owner_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class RegulatoryChange:
	id: str
	tenant_id: str
	source_id: str
	framework: str
	change_type: str
	title: str
	effective_date: str
	severity: str
	evidence_reference: str
	status: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ObligationMapping:
	id: str
	tenant_id: str
	change_id: str
	obligation_reference: str
	policy_reference: str
	owner_id: str
	due_date: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ImpactAssessment:
	id: str
	tenant_id: str
	change_id: str
	impacted_capability: str
	risk_rating: str
	evidence_reference: str
	reviewer_id: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class RegulatoryFiling:
	id: str
	tenant_id: str
	framework: str
	filing_type: str
	period: str
	evidence_reference: str
	owner_id: str
	status: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class RegulatorySubmission:
	id: str
	tenant_id: str
	filing_id: str
	channel: str
	submitted_by: str
	submitted_at: str
	acknowledgment_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class RegulatoryInquiry:
	id: str
	tenant_id: str
	regulator: str
	reference_id: str
	severity: str
	due_date: str
	evidence_reference: str
	status: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class RegulatoryResponse:
	id: str
	tenant_id: str
	inquiry_id: str
	responder_id: str
	response_reference: str
	approval_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class RegTechReview:
	id: str
	tenant_id: str
	reference_id: str
	reviewer_id: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class RegTechAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
