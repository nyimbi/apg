"""Pydantic v2 models for fintech_switch capability.

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


# ── Enums ─────────────────────────────────────────────────────────────────────

class SwitchTransactionStatus(str, Enum):
	routed = "routed"
	authorised = "authorised"
	settled = "settled"
	reversed = "reversed"
	failed = "failed"
	pending = "pending"


class NetworkType(str, Enum):
	visa = "visa"
	mastercard = "mastercard"
	interswitch = "interswitch"
	pesalink = "pesalink"
	mpesa = "mpesa"
	amex = "amex"
	interbank = "interbank"
	rtgs = "rtgs"
	eft = "eft"


class Channel(str, Enum):
	pos = "pos"
	atm = "atm"
	web = "web"
	mobile = "mobile"
	ussd = "ussd"
	ecommerce = "ecommerce"


class SchemeStatus(str, Enum):
	pending_activation = "pending_activation"
	active = "active"
	suspended = "suspended"
	deregistered = "deregistered"


class ClearingFileStatus(str, Enum):
	generated = "generated"
	submitted = "submitted"
	acknowledged = "acknowledged"
	rejected = "rejected"


# ── Core models ───────────────────────────────────────────────────────────────

class _Base(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)


class SwitchRoute(_Base):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	created_at: str = Field(default_factory=_now)
	stan: str
	rrn: str
	amount: float
	currency: str
	transaction_type: str = "purchase"
	channel: Channel = Channel.pos
	pan_masked: str | None = None
	merchant_id: str | None = None
	network: NetworkType = NetworkType.interbank
	status: SwitchTransactionStatus = SwitchTransactionStatus.routed
	hops: int = 1
	routed_at: str = Field(default_factory=_now)
	message_type: str = "0100"
	route: dict[str, Any] = Field(default_factory=dict)


class SwitchTransaction(_Base):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	created_at: str = Field(default_factory=_now)
	pan_or_phone: str
	amount: float
	currency: str
	merchant_id: str
	transaction_type: str = "purchase"
	channel: Channel = Channel.pos
	stan: str
	rrn: str
	auth_number: str
	response_code: str
	response_message: str
	authorised: bool
	timestamp: str = Field(default_factory=_now)


class ClearingFile(_Base):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	created_at: str = Field(default_factory=_now)
	settlement_date: str
	scheme: str
	transaction_count: int = 0
	total_amount: float = 0.0
	net_positions: dict[str, float] = Field(default_factory=dict)
	status: ClearingFileStatus = ClearingFileStatus.generated
	generated_at: str = Field(default_factory=_now)
	file_ref: str = ""


class Scheme(_Base):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	created_at: str = Field(default_factory=_now)
	scheme_name: str
	credential_hash: str
	effective_date: str
	status: SchemeStatus = SchemeStatus.pending_activation
	registered_at: str = Field(default_factory=_now)


# ── Request / Response ────────────────────────────────────────────────────────

class RouteTransactionRequest(_Base):
	transaction_data: dict[str, Any]
	routing_rules: list[dict[str, Any]]


class AuthorisationRequest(_Base):
	pan_or_phone: str
	amount: float
	merchant_id: str
	currency: str
	transaction_type: str = "purchase"
	channel: str = "pos"


class SchemeComplianceRequest(_Base):
	transaction_id: str
	scheme: str


class ClearingFileRequest(_Base):
	settlement_date: str
	scheme: str


class FraudVelocityRequest(_Base):
	pan_or_phone: str
	window_seconds: int
	max_attempts: int


class CardNotPresentRequest(_Base):
	token: str
	amount: float
	cvv_result: str
	avs_result: str


class Auth3DSRequest(_Base):
	pan: str
	amount: float
	eci: str
	cavv: str


class SchemeRegistrationRequest(_Base):
	scheme_name: str
	credentials: dict[str, Any]
	effective_date: str


class SimulatorRequest(_Base):
	scenario: str
	expected_response: str
