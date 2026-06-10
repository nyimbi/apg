"""Legal Billing & Time Tracking — Pydantic v2 models."""
from __future__ import annotations

from typing import Any
from uuid_extensions import uuid7str
from pydantic import BaseModel, ConfigDict, Field


class BilTimeEntryCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	matter_id: str
	attorney_id: str
	date: str
	hours: float
	rate: float
	activity_code: str  # L110, L120, A101 ... (ABA codes)
	description: str
	billable: bool = True
	currency: str = "KES"


class BilTimeEntryUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	hours: float | None = None
	rate: float | None = None
	description: str | None = None
	billable: bool | None = None
	status: str | None = None


class BilTimeEntryResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	matter_id: str
	attorney_id: str
	date: str
	hours: float
	rate: float
	amount: float
	activity_code: str
	description: str
	billable: bool
	currency: str
	status: str  # draft, submitted, approved, billed, written_off
	invoice_id: str | None = None
	created_at: str
	updated_at: str | None = None


class BilTimeEntryListResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	items: list[BilTimeEntryResponse]
	total: int
	total_hours: float
	total_amount: float
	page: int = 1
	page_size: int = 50


class BilTimeEntryFilter(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	matter_id: str | None = None
	attorney_id: str | None = None
	status: str | None = None
	billable: bool | None = None
	date_from: str | None = None
	date_to: str | None = None
	invoice_id: str | None = None


class BilDisbursementCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	matter_id: str
	recorded_by_id: str
	date: str
	amount: float
	currency: str = "KES"
	disbursement_type: str  # court_fee, expert_fee, travel, postage, copy, other
	description: str
	receipt_reference: str = ""
	billable: bool = True


class BilDisbursementResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	matter_id: str
	recorded_by_id: str
	date: str
	amount: float
	currency: str
	disbursement_type: str
	description: str
	receipt_reference: str
	billable: bool
	status: str
	invoice_id: str | None = None
	created_at: str


class BilInvoiceCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	matter_id: str
	client_id: str
	billing_period_start: str
	billing_period_end: str
	due_date: str
	time_entry_ids: list[str] = Field(default_factory=list)
	disbursement_ids: list[str] = Field(default_factory=list)
	discount_amount: float = 0.0
	discount_reason: str = ""
	notes: str = ""
	currency: str = "KES"


class BilInvoiceResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	invoice_number: str
	matter_id: str
	client_id: str
	billing_period_start: str
	billing_period_end: str
	due_date: str
	time_entry_ids: list[str]
	disbursement_ids: list[str]
	fees_amount: float
	disbursements_amount: float
	discount_amount: float
	subtotal: float
	tax_amount: float
	total_amount: float
	currency: str
	notes: str
	status: str  # draft, submitted, approved, sent, paid, overdue, written_off
	approved_by_id: str | None = None
	sent_at: str | None = None
	paid_at: str | None = None
	created_at: str


class BilTrustAccountCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	matter_id: str
	client_id: str
	account_name: str
	bank_name: str
	account_number: str
	currency: str = "KES"


class BilTrustAccountResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	matter_id: str
	client_id: str
	account_name: str
	bank_name: str
	account_number: str
	currency: str
	balance: float
	status: str
	created_at: str


class BilTrustTransactionCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	trust_account_id: str
	transaction_type: str  # deposit, withdrawal, transfer, fee_application
	amount: float
	date: str
	description: str
	reference: str = ""
	authorized_by_id: str


class BilTrustTransactionResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	trust_account_id: str
	tenant_id: str
	transaction_type: str
	amount: float
	running_balance: float
	date: str
	description: str
	reference: str
	authorized_by_id: str
	status: str
	created_at: str


class BilAuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	matter_id: str | None
	event_type: str
	actor_id: str | None
	details: dict[str, Any]
	created_at: str
