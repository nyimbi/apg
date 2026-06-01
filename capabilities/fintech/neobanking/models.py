"""Dependency-light data models for APG Digital Neobanking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BankProgram:
	id: str
	tenant_id: str
	name: str
	owner_id: str
	country: str
	base_currency: str
	settlement_account: str
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "name": self.name, "owner_id": self.owner_id, "country": self.country, "base_currency": self.base_currency, "settlement_account": self.settlement_account, "status": self.status}


@dataclass
class CustomerProfile:
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
class DepositAccount:
	id: str
	tenant_id: str
	program_id: str
	customer_id: str
	account_type: str
	currency: str
	account_number: str
	balance: float = 0.0
	status: str = "open"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "program_id": self.program_id, "customer_id": self.customer_id, "account_type": self.account_type, "currency": self.currency, "account_number": self.account_number, "balance": self.balance, "status": self.status}


@dataclass
class PaymentRailLink:
	id: str
	tenant_id: str
	account_id: str
	rail: str
	provider_reference: str
	wallet_reference: str = ""
	card_reference: str = ""
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "account_id": self.account_id, "rail": self.rail, "provider_reference": self.provider_reference, "wallet_reference": self.wallet_reference, "card_reference": self.card_reference, "status": self.status}


@dataclass
class AccountTransaction:
	id: str
	tenant_id: str
	account_id: str
	kind: str
	amount: float
	currency: str
	direction: str
	reference: str
	risk_reference: str
	status: str = "posted"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "account_id": self.account_id, "kind": self.kind, "amount": self.amount, "currency": self.currency, "direction": self.direction, "reference": self.reference, "risk_reference": self.risk_reference, "status": self.status}


@dataclass
class SavingsPot:
	id: str
	tenant_id: str
	account_id: str
	name: str
	target_amount: float
	currency: str
	balance: float = 0.0
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "account_id": self.account_id, "name": self.name, "target_amount": self.target_amount, "currency": self.currency, "balance": self.balance, "status": self.status}


@dataclass
class StatementRecord:
	id: str
	tenant_id: str
	account_id: str
	period_start: str
	period_end: str
	transaction_count: int
	closing_balance: float
	status: str = "issued"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "account_id": self.account_id, "period_start": self.period_start, "period_end": self.period_end, "transaction_count": self.transaction_count, "closing_balance": self.closing_balance, "status": self.status}


@dataclass
class ServiceCase:
	id: str
	tenant_id: str
	customer_id: str
	account_id: str
	reason: str
	reviewer_id: str
	evidence_references: list[str]
	status: str = "open"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "customer_id": self.customer_id, "account_id": self.account_id, "reason": self.reason, "reviewer_id": self.reviewer_id, "evidence_references": list(self.evidence_references), "status": self.status}


@dataclass
class NeobankingEvidence:
	id: str
	tenant_id: str
	kind: str
	reference_id: str
	status: str
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "kind": self.kind, "reference_id": self.reference_id, "status": self.status, "metadata": dict(self.metadata)}
