"""In-memory models for APG Government Contracts & Procurement."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Tender:
	id: str
	tenant_id: str
	procurement_method: str
	ppda_threshold: str
	title: str
	description: str
	approver_id: str
	evidence_reference: str
	status: str
	justification: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class TenderEvaluation:
	id: str
	tenant_id: str
	tender_id: str
	bidder_id: str
	criteria: str
	score: float
	evaluator_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ContractAward:
	id: str
	tenant_id: str
	tender_id: str
	awarded_to: str
	awarded_amount: float
	ppda_notification_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class GovernmentContract:
	id: str
	tenant_id: str
	award_id: str
	contract_type: str
	contract_value: float
	start_date: str
	end_date: str
	signed_by: str
	contractor_reference: str
	evidence_reference: str
	status: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ContractVariation:
	id: str
	tenant_id: str
	contract_id: str
	variation_type: str
	description: str
	value_change: float
	approval_reference: str
	ppda_notification_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ContractPerformance:
	id: str
	tenant_id: str
	contract_id: str
	performance_status: str
	reviewer_id: str
	period: str
	narrative: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class PpdaCompliance:
	id: str
	tenant_id: str
	tender_id: str
	threshold_category: str
	submission_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class DebarredBidder:
	id: str
	tenant_id: str
	bidder_id: str
	reason: str
	debarred_until: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ProcurementReview:
	id: str
	tenant_id: str
	reference_id: str
	reviewer_id: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ProcurementAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
