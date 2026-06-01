"""Dependency-light data models for APG Digital Lending."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LoanProduct:
	id: str
	tenant_id: str
	name: str
	owner_id: str
	product_type: str
	currency: str
	min_amount: float
	max_amount: float
	min_term_days: int
	max_term_days: int
	annual_rate: float
	repayment_frequency: str
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "name": self.name, "owner_id": self.owner_id, "product_type": self.product_type, "currency": self.currency, "min_amount": self.min_amount, "max_amount": self.max_amount, "min_term_days": self.min_term_days, "max_term_days": self.max_term_days, "annual_rate": self.annual_rate, "repayment_frequency": self.repayment_frequency, "status": self.status}


@dataclass
class BorrowerProfile:
	id: str
	tenant_id: str
	customer_reference: str
	kyc_profile_id: str
	country: str
	income_evidence_id: str
	consent_reference: str
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "customer_reference": self.customer_reference, "kyc_profile_id": self.kyc_profile_id, "country": self.country, "income_evidence_id": self.income_evidence_id, "consent_reference": self.consent_reference, "status": self.status}


@dataclass
class LoanApplication:
	id: str
	tenant_id: str
	borrower_id: str
	product_id: str
	requested_amount: float
	currency: str
	purpose: str
	affordability_reference: str
	bank_statement_reference: str
	aml_reference: str
	fraud_reference: str
	behavior_evidence_reference: str
	status: str = "submitted"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "borrower_id": self.borrower_id, "product_id": self.product_id, "requested_amount": self.requested_amount, "currency": self.currency, "purpose": self.purpose, "affordability_reference": self.affordability_reference, "bank_statement_reference": self.bank_statement_reference, "aml_reference": self.aml_reference, "fraud_reference": self.fraud_reference, "behavior_evidence_reference": self.behavior_evidence_reference, "status": self.status}


@dataclass
class UnderwritingDecision:
	id: str
	tenant_id: str
	application_id: str
	score: int
	decision: str
	evidence_references: list[str]
	human_approval: str
	adverse_reason: str = ""
	status: str = "recorded"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "application_id": self.application_id, "score": self.score, "decision": self.decision, "evidence_references": list(self.evidence_references), "human_approval": self.human_approval, "adverse_reason": self.adverse_reason, "status": self.status}


@dataclass
class LoanOffer:
	id: str
	tenant_id: str
	application_id: str
	underwriting_id: str
	amount: float
	currency: str
	apr: float
	term_days: int
	expiry_date: str
	status: str
	borrower_acceptance_reference: str = ""

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "application_id": self.application_id, "underwriting_id": self.underwriting_id, "amount": self.amount, "currency": self.currency, "apr": self.apr, "term_days": self.term_days, "expiry_date": self.expiry_date, "status": self.status, "borrower_acceptance_reference": self.borrower_acceptance_reference}


@dataclass
class Disbursement:
	id: str
	tenant_id: str
	offer_id: str
	amount: float
	currency: str
	rail: str
	funding_account: str
	destination_reference: str
	human_approval: str
	status: str = "recorded"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "offer_id": self.offer_id, "amount": self.amount, "currency": self.currency, "rail": self.rail, "funding_account": self.funding_account, "destination_reference": self.destination_reference, "human_approval": self.human_approval, "status": self.status}


@dataclass
class RepaymentSchedule:
	id: str
	tenant_id: str
	offer_id: str
	due_amount: float
	currency: str
	due_date: str
	frequency: str
	installment_count: int
	status: str = "scheduled"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "offer_id": self.offer_id, "due_amount": self.due_amount, "currency": self.currency, "due_date": self.due_date, "frequency": self.frequency, "installment_count": self.installment_count, "status": self.status}


@dataclass
class CollectionCase:
	id: str
	tenant_id: str
	overdue_account_reference: str
	reason: str
	reviewer_id: str
	contact_policy_reference: str
	status: str = "open"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "overdue_account_reference": self.overdue_account_reference, "reason": self.reason, "reviewer_id": self.reviewer_id, "contact_policy_reference": self.contact_policy_reference, "status": self.status}


@dataclass
class LendingEvidence:
	id: str
	tenant_id: str
	kind: str
	reference_id: str
	status: str
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "kind": self.kind, "reference_id": self.reference_id, "status": self.status, "metadata": dict(self.metadata)}
