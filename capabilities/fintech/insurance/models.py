"""Dependency-light data models for APG InsurTech."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Policyholder:
	id: str
	tenant_id: str
	name: str
	kyc_reference: str
	contact_reference: str
	risk_profile_reference: str
	status: str = "active"
	def to_dict(self) -> dict[str, Any]: return self.__dict__.copy()


@dataclass
class InsuranceProduct:
	id: str
	tenant_id: str
	name: str
	product_line: str
	coverage_terms_reference: str
	pricing_reference: str
	status: str = "published"
	def to_dict(self) -> dict[str, Any]: return self.__dict__.copy()


@dataclass
class Quote:
	id: str
	tenant_id: str
	policyholder_id: str
	product_id: str
	premium_minor: int
	currency: str
	underwriting_reference: str
	status: str = "quoted"
	def to_dict(self) -> dict[str, Any]: return self.__dict__.copy()


@dataclass
class Policy:
	id: str
	tenant_id: str
	quote_id: str
	effective_date: str
	payment_reference: str
	status: str = "bound"
	def to_dict(self) -> dict[str, Any]: return self.__dict__.copy()


@dataclass
class PremiumRecord:
	id: str
	tenant_id: str
	policy_id: str
	amount_minor: int
	currency: str
	payment_reference: str
	status: str = "recorded"
	def to_dict(self) -> dict[str, Any]: return self.__dict__.copy()


@dataclass
class ClaimRecord:
	id: str
	tenant_id: str
	policy_id: str
	claim_type: str
	amount_minor: int
	loss_date: str
	evidence_reference: str
	status: str = "open"
	def to_dict(self) -> dict[str, Any]: return self.__dict__.copy()


@dataclass
class InsuranceDocument:
	id: str
	tenant_id: str
	reference_id: str
	document_type: str
	evidence_reference: str
	status: str = "recorded"
	def to_dict(self) -> dict[str, Any]: return self.__dict__.copy()


@dataclass
class RiskAssessment:
	id: str
	tenant_id: str
	policyholder_id: str
	score: float
	source_reference: str
	status: str = "recorded"
	def to_dict(self) -> dict[str, Any]: return self.__dict__.copy()


@dataclass
class ReinsuranceAttachment:
	id: str
	tenant_id: str
	policy_id: str
	treaty_reference: str
	share_percent: float
	status: str = "attached"
	def to_dict(self) -> dict[str, Any]: return self.__dict__.copy()


@dataclass
class InsuranceAlert:
	id: str
	tenant_id: str
	reference_id: str
	severity: str
	evidence_reference: str
	status: str = "open"
	def to_dict(self) -> dict[str, Any]: return self.__dict__.copy()


@dataclass
class InsuranceReview:
	id: str
	tenant_id: str
	reference_id: str
	reviewer_id: str
	status: str
	evidence_reference: str
	def to_dict(self) -> dict[str, Any]: return self.__dict__.copy()


@dataclass
class InsuranceEvidence:
	id: str
	tenant_id: str
	kind: str
	reference_id: str
	status: str
	metadata: dict[str, Any] = field(default_factory=dict)
	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "kind": self.kind, "reference_id": self.reference_id, "status": self.status, "metadata": dict(self.metadata)}
