"""Dependency-light data models for APG Mobile Banking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MobileProgram:
	id: str
	tenant_id: str
	name: str
	owner_id: str
	country: str
	currency: str
	platforms: list[str]
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "name": self.name, "owner_id": self.owner_id, "country": self.country, "currency": self.currency, "platforms": list(self.platforms), "status": self.status}


@dataclass
class MobileCustomer:
	id: str
	tenant_id: str
	customer_reference: str
	country: str
	kyc_reference: str
	consent_reference: str
	aml_reference: str
	fraud_reference: str
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "customer_reference": self.customer_reference, "country": self.country, "kyc_reference": self.kyc_reference, "consent_reference": self.consent_reference, "aml_reference": self.aml_reference, "fraud_reference": self.fraud_reference, "status": self.status}


@dataclass
class TrustedDevice:
	id: str
	tenant_id: str
	customer_id: str
	platform: str
	fingerprint: str
	attestation_reference: str
	risk_tier: str
	status: str = "trusted"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "customer_id": self.customer_id, "platform": self.platform, "fingerprint": self.fingerprint, "attestation_reference": self.attestation_reference, "risk_tier": self.risk_tier, "status": self.status}


@dataclass
class AuthFactor:
	id: str
	tenant_id: str
	customer_id: str
	device_id: str
	factor_type: str
	strength_reference: str
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "customer_id": self.customer_id, "device_id": self.device_id, "factor_type": self.factor_type, "strength_reference": self.strength_reference, "status": self.status}


@dataclass
class AccountLink:
	id: str
	tenant_id: str
	customer_id: str
	link_type: str
	account_reference: str
	currency: str
	provider_reference: str
	status: str = "linked"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "customer_id": self.customer_id, "link_type": self.link_type, "account_reference": self.account_reference, "currency": self.currency, "provider_reference": self.provider_reference, "status": self.status}


@dataclass
class MobilePayment:
	id: str
	tenant_id: str
	customer_id: str
	device_id: str
	account_link_id: str
	payment_type: str
	amount: float
	currency: str
	recipient_reference: str
	risk_reference: str
	human_approval: str = ""
	status: str = "initiated"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "customer_id": self.customer_id, "device_id": self.device_id, "account_link_id": self.account_link_id, "payment_type": self.payment_type, "amount": self.amount, "currency": self.currency, "recipient_reference": self.recipient_reference, "risk_reference": self.risk_reference, "human_approval": self.human_approval, "status": self.status}


@dataclass
class BillPayment:
	id: str
	tenant_id: str
	payment_id: str
	biller_reference: str
	bill_account_reference: str
	status: str = "recorded"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "payment_id": self.payment_id, "biller_reference": self.biller_reference, "bill_account_reference": self.bill_account_reference, "status": self.status}


@dataclass
class AirtimePurchase:
	id: str
	tenant_id: str
	payment_id: str
	operator_reference: str
	phone_reference: str
	status: str = "recorded"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "payment_id": self.payment_id, "operator_reference": self.operator_reference, "phone_reference": self.phone_reference, "status": self.status}


@dataclass
class ServiceRequest:
	id: str
	tenant_id: str
	customer_id: str
	reason: str
	reviewer_id: str
	evidence_references: list[str]
	status: str = "open"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "customer_id": self.customer_id, "reason": self.reason, "reviewer_id": self.reviewer_id, "evidence_references": list(self.evidence_references), "status": self.status}


@dataclass
class NotificationPreference:
	id: str
	tenant_id: str
	customer_id: str
	channel: str
	consent_reference: str
	enabled: bool = True

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "customer_id": self.customer_id, "channel": self.channel, "consent_reference": self.consent_reference, "enabled": self.enabled}


@dataclass
class FraudEvent:
	id: str
	tenant_id: str
	customer_id: str
	severity: str
	evidence_references: list[str]
	human_approval: str = ""
	status: str = "open"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "customer_id": self.customer_id, "severity": self.severity, "evidence_references": list(self.evidence_references), "human_approval": self.human_approval, "status": self.status}


@dataclass
class MobileEvidence:
	id: str
	tenant_id: str
	kind: str
	reference_id: str
	status: str
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "kind": self.kind, "reference_id": self.reference_id, "status": self.status, "metadata": dict(self.metadata)}
