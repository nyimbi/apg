"""Dependency-light data models for APG Cross-Border Remittance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RemittanceQuote:
	id: str
	tenant_id: str
	source_country: str
	destination_country: str
	source_currency: str
	destination_currency: str
	send_amount: float
	fx_rate: float
	fee_amount: float
	expiry: str
	status: str = "quoted"

	def to_dict(self) -> dict[str, Any]:
		receive_amount = round(self.send_amount * self.fx_rate - self.fee_amount, 2)
		return {"id": self.id, "tenant_id": self.tenant_id, "source_country": self.source_country, "destination_country": self.destination_country, "source_currency": self.source_currency, "destination_currency": self.destination_currency, "send_amount": self.send_amount, "fx_rate": self.fx_rate, "fee_amount": self.fee_amount, "receive_amount": receive_amount, "expiry": self.expiry, "status": self.status}


@dataclass
class RemittanceTransfer:
	id: str
	tenant_id: str
	quote_id: str
	sender_reference: str
	beneficiary_reference: str
	sender_kyc_id: str
	beneficiary_kyc_id: str
	funding_reference: str
	payout_method: str
	purpose_code: str
	source_of_funds: str
	aml_screen_id: str
	fraud_decision: str
	status: str = "created"
	human_approval: str = ""
	settlement_reference: str = ""
	provider_receipt: str = ""

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "quote_id": self.quote_id, "sender_reference": self.sender_reference, "beneficiary_reference": self.beneficiary_reference, "sender_kyc_id": self.sender_kyc_id, "beneficiary_kyc_id": self.beneficiary_kyc_id, "funding_reference": self.funding_reference, "payout_method": self.payout_method, "purpose_code": self.purpose_code, "source_of_funds": self.source_of_funds, "aml_screen_id": self.aml_screen_id, "fraud_decision": self.fraud_decision, "status": self.status, "human_approval": self.human_approval, "settlement_reference": self.settlement_reference, "provider_receipt": self.provider_receipt}


@dataclass
class RemittanceRefund:
	id: str
	tenant_id: str
	transfer_id: str
	reason: str
	reviewer_id: str
	status: str = "filed"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "transfer_id": self.transfer_id, "reason": self.reason, "reviewer_id": self.reviewer_id, "status": self.status}


@dataclass
class RemittanceEvidence:
	id: str
	tenant_id: str
	kind: str
	reference_id: str
	status: str
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "kind": self.kind, "reference_id": self.reference_id, "status": self.status, "metadata": dict(self.metadata)}
