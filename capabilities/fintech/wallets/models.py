"""Dependency-light data models for APG Digital Wallets."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any


def money(value: Decimal | int | str) -> str:
	return str(Decimal(str(value)))


@dataclass
class Wallet:
	id: str
	tenant_id: str
	owner_reference: str
	wallet_type: str
	currency: str
	balance: Decimal = Decimal("0")
	held_balance: Decimal = Decimal("0")
	status: str = "active"
	metadata: dict[str, Any] = field(default_factory=dict)

	@property
	def available_balance(self) -> Decimal:
		return self.balance - self.held_balance

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"owner_reference": self.owner_reference,
			"wallet_type": self.wallet_type,
			"currency": self.currency,
			"balance": money(self.balance),
			"held_balance": money(self.held_balance),
			"available_balance": money(self.available_balance),
			"status": self.status,
			"metadata": dict(self.metadata),
		}


@dataclass
class WalletInstrument:
	id: str
	tenant_id: str
	wallet_id: str
	instrument_type: str
	token_reference: str
	verified_by: str
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"wallet_id": self.wallet_id,
			"instrument_type": self.instrument_type,
			"token_reference": self.token_reference,
			"verified_by": self.verified_by,
			"status": self.status,
		}


@dataclass
class WalletLedgerEntry:
	id: str
	tenant_id: str
	wallet_id: str
	entry_type: str
	amount: Decimal
	currency: str
	description: str
	idempotency_key: str
	status: str = "posted"

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"wallet_id": self.wallet_id,
			"entry_type": self.entry_type,
			"amount": money(self.amount),
			"currency": self.currency,
			"description": self.description,
			"idempotency_key": self.idempotency_key,
			"status": self.status,
		}


@dataclass
class WalletEvidence:
	id: str
	tenant_id: str
	kind: str
	reference_id: str
	status: str
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "kind": self.kind, "reference_id": self.reference_id, "status": self.status, "metadata": dict(self.metadata)}
