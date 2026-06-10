"""Pydantic v2 models for Payment USSD App capability."""
from __future__ import annotations

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


# ── Bill payment models ───────────────────────────────────────────────────────

class PayBillCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	phone_number: str
	biller_code: str
	account_reference: str
	amount: Decimal
	pin: str
	narration: str = ""
	tenant_id: str = "default"


class PayBillUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	status: str | None = None
	reversal_reason: str | None = None


class PayBillResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	phone_number: str
	biller_code: str
	biller_name: str
	account_reference: str
	amount: Decimal
	currency: str = "KES"
	narration: str
	receipt_number: str
	status: str = "pending"
	tenant_id: str
	created_at: str
	completed_at: str | None = None


# ── Merchant payment models ───────────────────────────────────────────────────

class PayMerchantCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	phone_number: str
	merchant_till: str
	amount: Decimal
	pin: str
	narration: str = ""
	tenant_id: str = "default"


class PayMerchantResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	phone_number: str
	merchant_till: str
	merchant_name: str
	amount: Decimal
	currency: str = "KES"
	narration: str
	receipt_number: str
	status: str = "completed"
	tenant_id: str
	created_at: str


# ── Airtime top-up models ─────────────────────────────────────────────────────

class PayAirtimeCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	phone_number: str
	recipient_phone: str
	amount: Decimal
	telco: str = "safaricom"
	pin: str
	tenant_id: str = "default"


class PayAirtimeResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	phone_number: str
	recipient_phone: str
	telco: str
	amount: Decimal
	currency: str = "KES"
	receipt_number: str
	status: str = "completed"
	tenant_id: str
	created_at: str


# ── Utility payment models ────────────────────────────────────────────────────

class PayUtilityCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	phone_number: str
	utility_code: str
	meter_number: str
	amount: Decimal
	pin: str
	tenant_id: str = "default"


class PayUtilityResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	phone_number: str
	utility_code: str
	utility_name: str
	meter_number: str
	amount: Decimal
	currency: str = "KES"
	units_purchased: str | None = None
	token: str | None = None
	receipt_number: str
	status: str = "completed"
	tenant_id: str
	created_at: str


# ── Send money models ─────────────────────────────────────────────────────────

class PaySendMoneyCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	from_phone: str
	to_phone: str
	amount: Decimal
	pin: str
	narration: str = ""
	tenant_id: str = "default"


class PaySendMoneyConfirmation(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	transaction_id: str
	pin: str
	tenant_id: str = "default"


class PaySendMoneyResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	from_phone: str
	to_phone: str
	amount: Decimal
	currency: str = "KES"
	narration: str
	receipt_number: str
	status: str = "pending"
	requires_confirmation: bool = False
	tenant_id: str
	created_at: str
	confirmed_at: str | None = None


# ── Biller registry model ─────────────────────────────────────────────────────

class PayBillerCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	biller_code: str
	biller_name: str
	category: str  # utility | insurance | tax | school | government | telco | water | internet
	paybill_number: str
	account_mask: str = ""  # regex or format hint e.g. "XXXX-XXXX"
	min_amount: Decimal = Decimal("1")
	max_amount: Decimal = Decimal("9999999")
	tenant_id: str = "default"


class PayBillerResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	biller_code: str
	biller_name: str
	category: str
	paybill_number: str
	account_mask: str
	min_amount: Decimal
	max_amount: Decimal
	status: str = "active"
	tenant_id: str
	created_at: str


# ── Filter model ──────────────────────────────────────────────────────────────

class PayPaymentFilter(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	payment_type: str | None = None  # bill | merchant | airtime | utility | send_money
	status: str | None = None
	phone_number: str | None = None
	date_from: str | None = None
	date_to: str | None = None
	tenant_id: str = "default"


# ── Audit model ───────────────────────────────────────────────────────────────

class PayAuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	record_id: str
	record_type: str
	phone_number: str | None = None
	amount: str | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)
	emitted_at: str


# ── USSD session model ────────────────────────────────────────────────────────

class PayUssdSessionCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	session_id: str
	phone_number: str
	service_code: str
	input_text: str = ""
	tenant_id: str = "default"


class PayUssdSessionResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	session_id: str
	phone_number: str
	menu_level: int = 0
	response_text: str
	continues: bool = True
	pending_transaction_id: str | None = None
	tenant_id: str
	created_at: str
	last_activity: str
