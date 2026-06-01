"""Dependency-light data models for APG Crowdfunding Platform."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class IssuerProfile:
	id: str
	tenant_id: str
	name: str
	kyc_reference: str
	beneficial_owner_reference: str
	risk_rating_reference: str
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class Campaign:
	id: str
	tenant_id: str
	issuer_id: str
	name: str
	campaign_type: str
	target_amount_minor: int
	currency: str
	disclosure_reference: str
	status: str = "published"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class DisclosureRecord:
	id: str
	tenant_id: str
	campaign_id: str
	disclosure_type: str
	evidence_reference: str
	status: str = "recorded"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class InvestorCommitment:
	id: str
	tenant_id: str
	campaign_id: str
	investor_id: str
	amount_minor: int
	currency: str
	investor_kyc_reference: str
	risk_ack_reference: str
	status: str = "pledged"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class EscrowFunding:
	id: str
	tenant_id: str
	commitment_id: str
	wallet_reference: str
	amount_minor: int
	status: str = "funded"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class MilestoneRecord:
	id: str
	tenant_id: str
	campaign_id: str
	name: str
	evidence_reference: str
	status: str = "recorded"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class PayoutAuthorization:
	id: str
	tenant_id: str
	campaign_id: str
	milestone_id: str
	amount_minor: int
	approval_reference: str
	status: str = "authorized"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class InvestorUpdate:
	id: str
	tenant_id: str
	campaign_id: str
	disclosure_reference: str
	recipient_scope: str
	status: str = "published"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class ComplianceAlert:
	id: str
	tenant_id: str
	campaign_id: str
	severity: str
	evidence_reference: str
	review_required: bool = True
	status: str = "open"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class CrowdfundingReview:
	id: str
	tenant_id: str
	reference_id: str
	reviewer_id: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class CrowdfundingEvidence:
	id: str
	tenant_id: str
	kind: str
	reference_id: str
	status: str
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "kind": self.kind, "reference_id": self.reference_id, "status": self.status, "metadata": dict(self.metadata)}
