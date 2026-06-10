"""Pydantic v2 models for Mobile Banking USSD capability."""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

try:
	from uuid6 import uuid7
	def uuid7str() -> str:
		return str(uuid7())
except ImportError:
	import uuid
	def uuid7str() -> str:  # type: ignore[misc]
		return str(uuid.uuid4())


# ── Account models ────────────────────────────────────────────────────────────

class MobAccountCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	phone_number: str
	account_number: str
	account_type: str = "savings"
	customer_name: str
	national_id: str
	pin: str
	currency: str = "KES"
	tenant_id: str = "default"


class MobAccountUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	customer_name: str | None = None
	pin: str | None = None
	status: str | None = None


class MobAccountResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	phone_number: str
	account_number: str
	account_type: str
	customer_name: str
	national_id: str
	currency: str
	balance: Decimal = Decimal("0")
	available_balance: Decimal = Decimal("0")
	daily_limit: Decimal = Decimal("100000")
	status: str = "active"
	tenant_id: str
	created_at: str
	updated_at: str | None = None


class MobAccountListResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	items: list[MobAccountResponse]
	total: int
	page: int = 1
	page_size: int = 50


class MobAccountFilter(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	phone_number: str | None = None
	account_type: str | None = None
	status: str | None = None
	tenant_id: str = "default"


# ── Transaction / Transfer models ─────────────────────────────────────────────

class MobTransferCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	from_account: str
	to_account: str
	amount: Decimal
	currency: str = "KES"
	narration: str = ""
	pin: str
	tenant_id: str = "default"


class MobTransferUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	status: str | None = None
	reversal_reason: str | None = None


class MobTransferResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	from_account: str
	to_account: str
	amount: Decimal
	currency: str
	narration: str
	reference: str
	status: str = "pending"
	tenant_id: str
	created_at: str
	settled_at: str | None = None


# ── Mini-statement models ─────────────────────────────────────────────────────

class MobStatementEntry(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	account_number: str
	transaction_type: str
	amount: Decimal
	currency: str
	balance_after: Decimal
	narration: str
	reference: str
	created_at: str


class MobMiniStatementResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	account_number: str
	entries: list[MobStatementEntry]
	generated_at: str


# ── Standing order models ─────────────────────────────────────────────────────

class MobStandingOrderCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	from_account: str
	to_account: str
	amount: Decimal
	frequency: str  # daily | weekly | monthly
	start_date: str
	end_date: str | None = None
	narration: str = ""
	pin: str
	tenant_id: str = "default"


class MobStandingOrderUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	amount: Decimal | None = None
	frequency: str | None = None
	end_date: str | None = None
	status: str | None = None


class MobStandingOrderResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	from_account: str
	to_account: str
	amount: Decimal
	frequency: str
	start_date: str
	end_date: str | None
	narration: str
	next_execution_date: str
	executions_count: int = 0
	status: str = "active"
	tenant_id: str
	created_at: str


# ── PIN management models ─────────────────────────────────────────────────────

class MobPinChangeRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	account_number: str
	old_pin: str
	new_pin: str
	confirm_pin: str
	tenant_id: str = "default"


class MobPinResetRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	phone_number: str
	national_id: str
	new_pin: str
	otp: str
	tenant_id: str = "default"


# ── Audit event model ─────────────────────────────────────────────────────────

class MobAuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	record_id: str
	record_type: str
	actor: str | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)
	emitted_at: str


# ── USSD session models ───────────────────────────────────────────────────────

class MobUssdSessionCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	session_id: str
	phone_number: str
	service_code: str
	input_text: str = ""
	tenant_id: str = "default"


class MobUssdSessionResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	session_id: str
	phone_number: str
	menu_level: int = 0
	response_text: str
	continues: bool = True
	tenant_id: str
	created_at: str
	last_activity: str
