"""Dependency-light data models for APG Agency Banking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgencyProgram:
	id: str
	tenant_id: str
	name: str
	owner_id: str
	country: str
	currency: str
	settlement_model: str
	services: list[str]
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "name": self.name, "owner_id": self.owner_id, "country": self.country, "currency": self.currency, "settlement_model": self.settlement_model, "services": list(self.services), "status": self.status}


@dataclass
class AgencyOutlet:
	id: str
	tenant_id: str
	program_id: str
	name: str
	outlet_type: str
	country: str
	license_reference: str
	location_reference: str
	security_plan_reference: str
	primary_channel: str
	initial_float: float
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "program_id": self.program_id, "name": self.name, "outlet_type": self.outlet_type, "country": self.country, "license_reference": self.license_reference, "location_reference": self.location_reference, "security_plan_reference": self.security_plan_reference, "primary_channel": self.primary_channel, "initial_float": self.initial_float, "status": self.status}


@dataclass
class AccreditedAgent:
	id: str
	tenant_id: str
	outlet_id: str
	name: str
	identity_reference: str
	training_reference: str
	background_check_reference: str
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "outlet_id": self.outlet_id, "name": self.name, "identity_reference": self.identity_reference, "training_reference": self.training_reference, "background_check_reference": self.background_check_reference, "status": self.status}


@dataclass
class FloatAccount:
	id: str
	tenant_id: str
	outlet_id: str
	currency: str
	available_balance: float
	ledger_reference: str
	reserved_balance: float = 0
	status: str = "open"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "outlet_id": self.outlet_id, "currency": self.currency, "available_balance": self.available_balance, "reserved_balance": self.reserved_balance, "ledger_reference": self.ledger_reference, "status": self.status}


@dataclass
class AgencyCustomer:
	id: str
	tenant_id: str
	customer_reference: str
	tier: str
	kyc_reference: str
	consent_reference: str
	aml_reference: str
	fraud_reference: str
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "customer_reference": self.customer_reference, "tier": self.tier, "kyc_reference": self.kyc_reference, "consent_reference": self.consent_reference, "aml_reference": self.aml_reference, "fraud_reference": self.fraud_reference, "status": self.status}


@dataclass
class AgencyTransaction:
	id: str
	tenant_id: str
	outlet_id: str
	agent_id: str
	customer_id: str
	float_account_id: str
	service: str
	amount: float
	currency: str
	channel: str
	customer_reference: str
	risk_reference: str
	status: str = "posted"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "outlet_id": self.outlet_id, "agent_id": self.agent_id, "customer_id": self.customer_id, "float_account_id": self.float_account_id, "service": self.service, "amount": self.amount, "currency": self.currency, "channel": self.channel, "customer_reference": self.customer_reference, "risk_reference": self.risk_reference, "status": self.status}


@dataclass
class CashMovement:
	id: str
	tenant_id: str
	outlet_id: str
	movement_type: str
	amount: float
	currency: str
	custodian_reference: str
	human_approval: str = ""
	status: str = "recorded"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "outlet_id": self.outlet_id, "movement_type": self.movement_type, "amount": self.amount, "currency": self.currency, "custodian_reference": self.custodian_reference, "human_approval": self.human_approval, "status": self.status}


@dataclass
class CommissionSettlement:
	id: str
	tenant_id: str
	outlet_id: str
	period: str
	amount: float
	currency: str
	reconciliation_reference: str
	payment_reference: str
	status: str = "settled"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "outlet_id": self.outlet_id, "period": self.period, "amount": self.amount, "currency": self.currency, "reconciliation_reference": self.reconciliation_reference, "payment_reference": self.payment_reference, "status": self.status}


@dataclass
class AgencyDispute:
	id: str
	tenant_id: str
	transaction_id: str
	reason: str
	reviewer_id: str
	evidence_references: list[str]
	status: str = "open"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "transaction_id": self.transaction_id, "reason": self.reason, "reviewer_id": self.reviewer_id, "evidence_references": list(self.evidence_references), "status": self.status}


@dataclass
class SupervisionVisit:
	id: str
	tenant_id: str
	outlet_id: str
	supervisor_id: str
	outcome: str
	evidence_references: list[str]
	findings: list[str] = field(default_factory=list)
	remediation_plan_reference: str = ""
	status: str = "recorded"

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "outlet_id": self.outlet_id, "supervisor_id": self.supervisor_id, "outcome": self.outcome, "evidence_references": list(self.evidence_references), "findings": list(self.findings), "remediation_plan_reference": self.remediation_plan_reference, "status": self.status}


@dataclass
class AgencyEvidence:
	id: str
	tenant_id: str
	kind: str
	reference_id: str
	status: str
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "kind": self.kind, "reference_id": self.reference_id, "status": self.status, "metadata": dict(self.metadata)}
