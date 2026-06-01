"""In-memory models for APG Intelligence Reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ReportingAuthority:
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
class ReportingWorkspace:
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
class ReportingTemplate:
	id: str
	tenant_id: str
	workspace_id: str
	template_type: str
	template_reference: str
	classification: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ReportingProduct:
	id: str
	tenant_id: str
	template_id: str
	product_type: str
	title: str
	author_id: str
	classification: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ReportingSection:
	id: str
	tenant_id: str
	product_id: str
	section_type: str
	section_reference: str
	confidence_score: float
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ReportingCitation:
	id: str
	tenant_id: str
	section_id: str
	citation_type: str
	source_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ReportingApproval:
	id: str
	tenant_id: str
	product_id: str
	approval_type: str
	approver_id: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ReportingDistribution:
	id: str
	tenant_id: str
	product_id: str
	distribution_type: str
	recipient_reference: str
	approval_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ReportingPublication:
	id: str
	tenant_id: str
	distribution_id: str
	publication_type: str
	publication_reference: str
	approval_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ReportingReview:
	id: str
	tenant_id: str
	reference_id: str
	reviewer_id: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ReportingAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)

