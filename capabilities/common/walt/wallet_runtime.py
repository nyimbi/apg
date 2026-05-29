"""Dependency-light wallet and payment runtime primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from hashlib import sha256
from typing import Any


WALLET_STATUSES = {"active", "disabled", "frozen"}
INSTRUMENT_TYPES = {"card", "bank_account", "mobile_money", "token", "external"}
INSTRUMENT_STATUSES = {"verified", "blocked", "expired"}
TRANSACTION_STATUSES = {"authorized", "captured", "review_required", "declined", "settled"}
SETTLEMENT_STATUSES = {"ready", "reconciled", "exception_review"}
RECONCILIATION_STATUSES = {"matched", "exceptions"}


def utc_now() -> str:
	return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_id(prefix: str, *parts: object) -> str:
	seed = "|".join(str(part).strip().lower() for part in parts if str(part).strip())
	digest = sha256(seed.encode("utf-8")).hexdigest()[:16]
	return f"{prefix}_{digest}"


def normalize_currency(currency: str) -> str:
	value = str(currency or "").strip().upper()
	if len(value) != 3 or not value.isalpha():
		raise ValueError(f"unsupported_currency:{currency}")
	return value


def normalize_instrument_type(instrument_type: str) -> str:
	value = str(instrument_type or "external").strip().lower()
	if value not in INSTRUMENT_TYPES:
		raise ValueError(f"unsupported_instrument_type:{instrument_type}")
	return value


def money_to_minor_units(amount: int | float | str | Decimal) -> int:
	value = Decimal(str(amount)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
	return int(value * 100)


def money_from_minor_units(amount_minor: int) -> float:
	return float((Decimal(int(amount_minor)) / Decimal(100)).quantize(Decimal("0.01")))


def rule_required_actions(rule_result: dict[str, Any]) -> list[str]:
	return [
		str(action["required_action"])
		for action in rule_result.get("actions", [])
		if action.get("required_action")
	]


def serialize(record: object) -> dict[str, Any]:
	data = asdict(record)
	for key in ("balance_minor", "hold_minor", "amount_minor", "total_minor"):
		if key in data:
			data[key.replace("_minor", "")] = money_from_minor_units(int(data[key]))
	return data


@dataclass(slots=True)
class WalletRecord:
	id: str
	tenant_id: str
	owner_ref: str
	currency: str
	ledger_ref: str
	compliance_policy_ref: str
	balance_minor: int = 0
	hold_minor: int = 0
	status: str = "active"
	created_at: str = field(default_factory=utc_now)
	updated_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class PaymentInstrumentRecord:
	id: str
	tenant_id: str
	wallet_id: str
	instrument_ref: str
	instrument_type: str
	token_ref: str
	encrypted: bool
	verified_by: str
	status: str = "verified"
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class TransactionRecord:
	id: str
	tenant_id: str
	wallet_id: str
	instrument_id: str
	direction: str
	amount_minor: int
	currency: str
	status: str
	risk_score: float
	mfa_completed: bool
	risk_review_recorded: bool
	idempotency_key: str
	required_actions: list[str] = field(default_factory=list)
	matched_rules: list[str] = field(default_factory=list)
	authorized_at: str = field(default_factory=utc_now)
	captured_at: str | None = None
	settled_at: str | None = None

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class SettlementBatchRecord:
	id: str
	tenant_id: str
	transaction_ids: list[str]
	settlement_account_ref: str
	total_minor: int
	currency: str
	reconciliation_completed: bool
	status: str = "ready"
	created_by: str = ""
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class ReconciliationRecord:
	id: str
	tenant_id: str
	settlement_batch_id: str
	reconciliation_ref: str
	matched_count: int
	exception_count: int
	status: str
	recorded_by: str
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class WalletAuditEventRecord:
	id: str
	tenant_id: str
	event_type: str
	subject_id: str
	message: str
	actor: str
	severity: str = "low"
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


__all__ = [
	"INSTRUMENT_STATUSES",
	"INSTRUMENT_TYPES",
	"RECONCILIATION_STATUSES",
	"SETTLEMENT_STATUSES",
	"TRANSACTION_STATUSES",
	"WALLET_STATUSES",
	"PaymentInstrumentRecord",
	"ReconciliationRecord",
	"SettlementBatchRecord",
	"TransactionRecord",
	"WalletAuditEventRecord",
	"WalletRecord",
	"money_from_minor_units",
	"money_to_minor_units",
	"normalize_currency",
	"normalize_instrument_type",
	"rule_required_actions",
	"stable_id",
	"utc_now",
]
