"""Pydantic v2 models for fintech_terminal capability.

© 2025 Datacraft  |  Author: Nyimbi Odero
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid6 import uuid7

from pydantic import BaseModel, ConfigDict, Field

uuid7str = lambda: str(uuid7())


def _now() -> str:
	return datetime.now(timezone.utc).isoformat()


# ── Enums ────────────────────────────────────────────────────────────────────

class TerminalStatus(str, Enum):
	active = "active"
	inactive = "inactive"
	suspended = "suspended"
	decommissioned = "decommissioned"
	key_injection_pending = "key_injection_pending"
	configuration_pending = "configuration_pending"
	maintenance = "maintenance"


class TerminalType(str, Enum):
	pos = "pos"
	atm = "atm"
	mpos = "mpos"
	android_pos = "android_pos"
	web_pos = "web_pos"
	kiosk = "kiosk"
	unattended = "unattended"
	soft_pos = "soft_pos"
	tap_on_phone = "tap_on_phone"
	agent_terminal = "agent_terminal"


class TransactionType(str, Enum):
	cash_deposit = "cash_deposit"
	cash_withdrawal = "cash_withdrawal"
	funds_transfer = "funds_transfer"
	bill_payment = "bill_payment"
	balance_inquiry = "balance_inquiry"
	mini_statement = "mini_statement"
	purchase = "purchase"


class FloatOperationType(str, Enum):
	top_up = "top_up"
	withdrawal = "withdrawal"
	reconcile = "reconcile"


class ReceiptFormat(str, Enum):
	thermal = "thermal"
	sms = "sms"
	pdf = "pdf"
	qr = "qr"


# ── Core models ──────────────────────────────────────────────────────────────

class _Base(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)


class Terminal(_Base):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	created_at: str = Field(default_factory=_now)
	terminal_type: TerminalType
	connectivity: str
	location: dict[str, Any]
	agent_id: str
	serial_number: str | None = None
	merchant_id: str | None = None
	model: str | None = None
	status: TerminalStatus = TerminalStatus.configuration_pending
	registered_at: str = Field(default_factory=_now)
	updated_at: str = Field(default_factory=_now)
	float_balance: float = 0.0
	transaction_count: int = 0
	last_heartbeat: str | None = None
	offline_queue: list[dict[str, Any]] = Field(default_factory=list)
	pci_dss_compliant: bool = False
	tamper_detection_enabled: bool = False


class TerminalTransaction(_Base):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	created_at: str = Field(default_factory=_now)
	terminal_id: str
	agent_id: str | None = None
	transaction_type: TransactionType
	amount: float
	currency: str
	customer_id: str
	reference: str
	status: str = "approved"
	timestamp: str = Field(default_factory=_now)
	metadata: dict[str, Any] = Field(default_factory=dict)


class FloatManagement(_Base):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	created_at: str = Field(default_factory=_now)
	terminal_id: str
	operation_type: FloatOperationType
	float_amount: float
	previous_balance: float
	new_balance: float
	authorised_by: str | None = None
	timestamp: str = Field(default_factory=_now)


class AgentNetwork(_Base):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	created_at: str = Field(default_factory=_now)
	network_id: str
	period: str
	total_terminals: int = 0
	active_terminals: int = 0
	total_transactions: int = 0
	total_volume_kes: float = 0.0
	volume_by_type: dict[str, float] = Field(default_factory=dict)
	count_by_type: dict[str, int] = Field(default_factory=dict)
	top_agents: list[dict[str, Any]] = Field(default_factory=list)
	generated_at: str = Field(default_factory=_now)


# ── Request / Response ───────────────────────────────────────────────────────

class RegisterTerminalRequest(_Base):
	terminal_id: str
	location: dict[str, Any]
	agent_id: str
	terminal_type: TerminalType
	connectivity: str
	serial_number: str | None = None
	merchant_id: str | None = None
	model: str | None = None
	tenant_id: str | None = None


class ActivateTerminalRequest(_Base):
	activated_by: str
	pci_dss_compliant: bool = True
	tamper_detection_enabled: bool = True
	software_integrity_verified: bool = True


class TerminalTransactionRequest(_Base):
	transaction_type: TransactionType
	amount: float
	currency: str
	customer_id: str
	reference: str
	metadata: dict[str, Any] | None = None


class FloatManagementRequest(_Base):
	float_amount: float
	operation_type: FloatOperationType
	authorised_by: str | None = None


class OfflineSyncRequest(_Base):
	queued_transactions: list[dict[str, Any]]


class CommissionReportRequest(_Base):
	period: str


class FraudAlertRequest(_Base):
	event_type: str
	details: dict[str, Any]


class CustomerEnrolmentRequest(_Base):
	biometric_data: dict[str, Any]
	id_number: str
	customer_name: str | None = None
	phone: str | None = None
