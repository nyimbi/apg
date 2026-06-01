"""Dependency-light data models for APG Digital Payments."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any


def money(value: Decimal | int | str) -> str:
	"""Return stable JSON money text without floating-point rounding."""
	return str(Decimal(str(value)))


@dataclass
class PaymentAccount:
	id: str
	tenant_id: str
	owner_reference: str
	currency: str
	status: str = "active"
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"owner_reference": self.owner_reference,
			"currency": self.currency,
			"status": self.status,
			"metadata": dict(self.metadata),
		}


@dataclass
class PaymentInstrument:
	id: str
	tenant_id: str
	account_id: str
	instrument_type: str
	token_reference: str
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"account_id": self.account_id,
			"instrument_type": self.instrument_type,
			"token_reference": self.token_reference,
			"status": self.status,
		}


@dataclass
class PaymentOrder:
	id: str
	tenant_id: str
	account_id: str
	instrument_id: str
	amount: Decimal
	currency: str
	counterparty_reference: str
	purpose: str
	status: str = "created"
	authorized_amount: Decimal = Decimal("0")
	captured_amount: Decimal = Decimal("0")
	refunded_amount: Decimal = Decimal("0")
	risk_level: str = "medium"
	risk_score: Decimal = Decimal("0")

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"account_id": self.account_id,
			"instrument_id": self.instrument_id,
			"amount": money(self.amount),
			"currency": self.currency,
			"counterparty_reference": self.counterparty_reference,
			"purpose": self.purpose,
			"status": self.status,
			"authorized_amount": money(self.authorized_amount),
			"captured_amount": money(self.captured_amount),
			"refunded_amount": money(self.refunded_amount),
			"risk_level": self.risk_level,
			"risk_score": money(self.risk_score),
		}


@dataclass
class PaymentEvidence:
	id: str
	tenant_id: str
	kind: str
	reference_id: str
	status: str
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"kind": self.kind,
			"reference_id": self.reference_id,
			"status": self.status,
			"metadata": dict(self.metadata),
		}
