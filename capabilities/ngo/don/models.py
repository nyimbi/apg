"""Donor Relationship Management — Pydantic v2 models."""
from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

try:
	from uuid6 import uuid7
	def uuid7str() -> str:
		return str(uuid7())
except ImportError:
	from uuid import uuid4
	def uuid7str() -> str:  # type: ignore[misc]
		return str(uuid4())


_cfg = ConfigDict(extra="forbid", validate_by_name=True)


class DonDonorCreate(BaseModel):
	model_config = _cfg
	name: str
	donor_type: str = "individual"
	email: str = ""
	phone: str = ""
	country: str = "KE"
	address: str = ""
	tax_id: str = ""
	notes: str = ""
	tags: list[str] = Field(default_factory=list)


class DonDonorUpdate(BaseModel):
	model_config = _cfg
	name: str | None = None
	email: str | None = None
	phone: str | None = None
	address: str | None = None
	status: str | None = None
	notes: str | None = None
	tags: list[str] | None = None


class DonDonorResponse(BaseModel):
	model_config = _cfg
	id: str = Field(default_factory=uuid7str)
	name: str
	donor_type: str
	email: str
	phone: str
	country: str
	address: str
	tax_id: str
	notes: str
	tags: list[str]
	total_pledged: Decimal = Decimal("0")
	total_received: Decimal = Decimal("0")
	status: str
	tenant_id: str
	created_at: str
	updated_at: str | None = None


class DonCommunicationCreate(BaseModel):
	model_config = _cfg
	donor_id: str
	channel: str = "email"
	direction: str = "outbound"
	subject: str
	body: str
	staff_member: str
	communication_date: str
	tags: list[str] = Field(default_factory=list)


class DonCommunicationResponse(BaseModel):
	model_config = _cfg
	id: str = Field(default_factory=uuid7str)
	donor_id: str
	channel: str
	direction: str
	subject: str
	body: str
	staff_member: str
	communication_date: str
	tags: list[str]
	tenant_id: str
	created_at: str


class DonPledgeCreate(BaseModel):
	model_config = _cfg
	donor_id: str
	amount: Decimal
	currency: str = "KES"
	pledge_date: str
	due_date: str
	purpose: str = ""
	frequency: str = "one_time"
	notes: str = ""


class DonPledgeResponse(BaseModel):
	model_config = _cfg
	id: str = Field(default_factory=uuid7str)
	donor_id: str
	amount: Decimal
	received_amount: Decimal = Decimal("0")
	currency: str
	pledge_date: str
	due_date: str
	purpose: str
	frequency: str
	notes: str
	status: str
	tenant_id: str
	created_at: str


class DonReceiptCreate(BaseModel):
	model_config = _cfg
	donor_id: str
	pledge_id: str | None = None
	amount: Decimal
	currency: str = "KES"
	receipt_date: str
	payment_method: str = "bank_transfer"
	reference: str
	issued_by: str


class DonReceiptResponse(BaseModel):
	model_config = _cfg
	id: str = Field(default_factory=uuid7str)
	receipt_number: str
	donor_id: str
	pledge_id: str | None
	amount: Decimal
	currency: str
	receipt_date: str
	payment_method: str
	reference: str
	issued_by: str
	status: str
	tenant_id: str
	created_at: str


class DonStewardshipPlanCreate(BaseModel):
	model_config = _cfg
	donor_id: str
	tier: str = "standard"
	touchpoints_per_year: int = 4
	assigned_to: str
	notes: str = ""


class DonStewardshipPlanResponse(BaseModel):
	model_config = _cfg
	id: str = Field(default_factory=uuid7str)
	donor_id: str
	tier: str
	touchpoints_per_year: int
	assigned_to: str
	notes: str
	status: str
	tenant_id: str
	created_at: str


class DonDonorFilter(BaseModel):
	model_config = _cfg
	status: str | None = None
	donor_type: str | None = None
	country: str | None = None
	tags: list[str] | None = None


class DonAuditEvent(BaseModel):
	model_config = _cfg
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	record_id: str
	record_type: str
	details: dict[str, Any] = Field(default_factory=dict)
	emitted_at: str


__all__ = [
	"DonDonorCreate", "DonDonorUpdate", "DonDonorResponse",
	"DonCommunicationCreate", "DonCommunicationResponse",
	"DonPledgeCreate", "DonPledgeResponse",
	"DonReceiptCreate", "DonReceiptResponse",
	"DonStewardshipPlanCreate", "DonStewardshipPlanResponse",
	"DonDonorFilter", "DonAuditEvent",
]
