"""Dependency-light data models for APG Digital Cards."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CardProgram:
	id: str
	tenant_id: str
	name: str
	owner_id: str
	bin_range: str
	currency: str
	settlement_account: str
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "name": self.name, "owner_id": self.owner_id, "bin_range": self.bin_range, "currency": self.currency, "settlement_account": self.settlement_account, "status": self.status}


@dataclass
class Cardholder:
	id: str
	tenant_id: str
	customer_reference: str
	kyc_profile_id: str
	country: str
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "customer_reference": self.customer_reference, "kyc_profile_id": self.kyc_profile_id, "country": self.country, "status": self.status}


@dataclass
class Card:
	id: str
	tenant_id: str
	program_id: str
	cardholder_id: str
	card_type: str
	product: str
	wallet_reference: str
	funding_account: str
	masked_pan: str
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "program_id": self.program_id, "cardholder_id": self.cardholder_id, "card_type": self.card_type, "product": self.product, "wallet_reference": self.wallet_reference, "funding_account": self.funding_account, "masked_pan": self.masked_pan, "status": self.status}


@dataclass
class CardToken:
	id: str
	tenant_id: str
	card_id: str
	token_type: str
	token_reference: str
	key_domain_id: str
	device_or_merchant_reference: str
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "card_id": self.card_id, "token_type": self.token_type, "token_reference": self.token_reference, "key_domain_id": self.key_domain_id, "device_or_merchant_reference": self.device_or_merchant_reference, "status": self.status}


@dataclass
class CardAuthorization:
	id: str
	tenant_id: str
	card_id: str
	amount: float
	currency: str
	merchant_category: str
	fraud_reference: str
	aml_reference: str
	decision: str
	status: str = "recorded"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "card_id": self.card_id, "amount": self.amount, "currency": self.currency, "merchant_category": self.merchant_category, "fraud_reference": self.fraud_reference, "aml_reference": self.aml_reference, "decision": self.decision, "status": self.status}


@dataclass
class CardDispute:
	id: str
	tenant_id: str
	transaction_reference: str
	reason: str
	evidence_references: list[str]
	reviewer_id: str
	status: str = "filed"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "transaction_reference": self.transaction_reference, "reason": self.reason, "evidence_references": list(self.evidence_references), "reviewer_id": self.reviewer_id, "status": self.status}


@dataclass
class CardEvidence:
	id: str
	tenant_id: str
	kind: str
	reference_id: str
	status: str
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "kind": self.kind, "reference_id": self.reference_id, "status": self.status, "metadata": dict(self.metadata)}
