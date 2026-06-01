"""Dependency-light data models for APG Buy Now Pay Later."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MerchantProgram:
	id: str
	tenant_id: str
	name: str
	owner_id: str
	country: str
	currency: str
	settlement_policy_reference: str
	fee_disclosure_reference: str
	max_installments: int
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "name": self.name, "owner_id": self.owner_id, "country": self.country, "currency": self.currency, "settlement_policy_reference": self.settlement_policy_reference, "fee_disclosure_reference": self.fee_disclosure_reference, "max_installments": self.max_installments, "status": self.status}


@dataclass
class BNPLConsumer:
	id: str
	tenant_id: str
	customer_reference: str
	kyc_profile_id: str
	country: str
	consent_reference: str
	aml_reference: str
	fraud_reference: str
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "customer_reference": self.customer_reference, "kyc_profile_id": self.kyc_profile_id, "country": self.country, "consent_reference": self.consent_reference, "aml_reference": self.aml_reference, "fraud_reference": self.fraud_reference, "status": self.status}


@dataclass
class MerchantProfile:
	id: str
	tenant_id: str
	program_id: str
	legal_entity_reference: str
	category: str
	country: str
	risk_tier: str
	settlement_account: str
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "program_id": self.program_id, "legal_entity_reference": self.legal_entity_reference, "category": self.category, "country": self.country, "risk_tier": self.risk_tier, "settlement_account": self.settlement_account, "status": self.status}


@dataclass
class CheckoutSession:
	id: str
	tenant_id: str
	merchant_id: str
	consumer_id: str
	channel: str
	category: str
	amount: float
	currency: str
	payment_reference: str
	fraud_reference: str
	aml_reference: str
	consent_reference: str
	human_review: str = ""
	status: str = "created"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "merchant_id": self.merchant_id, "consumer_id": self.consumer_id, "channel": self.channel, "category": self.category, "amount": self.amount, "currency": self.currency, "payment_reference": self.payment_reference, "fraud_reference": self.fraud_reference, "aml_reference": self.aml_reference, "consent_reference": self.consent_reference, "human_review": self.human_review, "status": self.status}


@dataclass
class AffordabilityDecision:
	id: str
	tenant_id: str
	checkout_id: str
	score: int
	decision: str
	evidence_references: list[str]
	human_approval: str
	adverse_reason: str = ""
	status: str = "recorded"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "checkout_id": self.checkout_id, "score": self.score, "decision": self.decision, "evidence_references": list(self.evidence_references), "human_approval": self.human_approval, "adverse_reason": self.adverse_reason, "status": self.status}


@dataclass
class BNPLPlan:
	id: str
	tenant_id: str
	checkout_id: str
	affordability_id: str
	plan_type: str
	principal: float
	currency: str
	term_days: int
	down_payment: float
	fee_disclosure_reference: str
	customer_acceptance_reference: str
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "checkout_id": self.checkout_id, "affordability_id": self.affordability_id, "plan_type": self.plan_type, "principal": self.principal, "currency": self.currency, "term_days": self.term_days, "down_payment": self.down_payment, "fee_disclosure_reference": self.fee_disclosure_reference, "customer_acceptance_reference": self.customer_acceptance_reference, "status": self.status}


@dataclass
class InstallmentSchedule:
	id: str
	tenant_id: str
	plan_id: str
	due_amount: float
	currency: str
	due_date: str
	status: str
	sequence: int

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "plan_id": self.plan_id, "due_amount": self.due_amount, "currency": self.currency, "due_date": self.due_date, "status": self.status, "sequence": self.sequence}


@dataclass
class MerchantSettlement:
	id: str
	tenant_id: str
	merchant_id: str
	plan_id: str
	gross_amount: float
	net_amount: float
	currency: str
	status: str
	reconciliation_reference: str
	payment_rail_reference: str
	human_approval: str = ""

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "merchant_id": self.merchant_id, "plan_id": self.plan_id, "gross_amount": self.gross_amount, "net_amount": self.net_amount, "currency": self.currency, "status": self.status, "reconciliation_reference": self.reconciliation_reference, "payment_rail_reference": self.payment_rail_reference, "human_approval": self.human_approval}


@dataclass
class BNPLDispute:
	id: str
	tenant_id: str
	plan_id: str
	reason: str
	reviewer_id: str
	evidence_references: list[str]
	status: str = "open"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "plan_id": self.plan_id, "reason": self.reason, "reviewer_id": self.reviewer_id, "evidence_references": list(self.evidence_references), "status": self.status}


@dataclass
class BNPLevidence:
	id: str
	tenant_id: str
	kind: str
	reference_id: str
	status: str
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "kind": self.kind, "reference_id": self.reference_id, "status": self.status, "metadata": dict(self.metadata)}
