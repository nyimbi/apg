"""Pydantic v2 models for USSD Government Services (gov_usd)."""
from __future__ import annotations

from datetime import datetime
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


_CONFIG = ConfigDict(extra="forbid", validate_by_name=True)


# ── Session models ────────────────────────────────────────────────────────────

class USSDSessionCreate(BaseModel):
	model_config = _CONFIG
	msisdn: str = Field(..., description="Subscriber phone number (MSISDN)")
	service_code: str = Field(..., description="USSD service code e.g. *384#")
	tenant_id: str = Field(default="default")
	session_data: dict[str, Any] = Field(default_factory=dict)


class USSDSessionUpdate(BaseModel):
	model_config = _CONFIG
	input_text: str = Field(..., description="User USSD input text")
	menu_level: int = Field(default=1)
	session_data: dict[str, Any] | None = None


class USSDSessionResponse(BaseModel):
	model_config = _CONFIG
	id: str = Field(default_factory=uuid7str)
	msisdn: str
	service_code: str
	tenant_id: str
	menu_level: int = 1
	session_data: dict[str, Any] = Field(default_factory=dict)
	status: str = "active"
	created_at: str
	updated_at: str | None = None


# ── Permit enquiry models ─────────────────────────────────────────────────────

class PermitEnquiryCreate(BaseModel):
	model_config = _CONFIG
	msisdn: str
	permit_number: str
	permit_type: str
	tenant_id: str = Field(default="default")


class PermitEnquiryResponse(BaseModel):
	model_config = _CONFIG
	id: str = Field(default_factory=uuid7str)
	msisdn: str
	permit_number: str
	permit_type: str
	tenant_id: str
	holder_name: str | None = None
	issue_date: str | None = None
	expiry_date: str | None = None
	status: str = "pending"
	created_at: str


# ── Tax balance models ────────────────────────────────────────────────────────

class TaxBalanceEnquiryCreate(BaseModel):
	model_config = _CONFIG
	msisdn: str
	tax_pin: str = Field(..., description="KRA PIN or equivalent tax identifier")
	tax_type: str = Field(default="income_tax")
	tenant_id: str = Field(default="default")


class TaxBalanceEnquiryResponse(BaseModel):
	model_config = _CONFIG
	id: str = Field(default_factory=uuid7str)
	msisdn: str
	tax_pin: str
	tax_type: str
	tenant_id: str
	outstanding_balance: float = 0.0
	currency: str = "KES"
	last_payment_date: str | None = None
	due_date: str | None = None
	status: str = "fetched"
	created_at: str


# ── ID verification models ────────────────────────────────────────────────────

class IDVerificationCreate(BaseModel):
	model_config = _CONFIG
	msisdn: str
	id_number: str
	id_type: str = Field(default="national_id")
	full_name: str | None = None
	tenant_id: str = Field(default="default")


class IDVerificationResponse(BaseModel):
	model_config = _CONFIG
	id: str = Field(default_factory=uuid7str)
	msisdn: str
	id_number: str
	id_type: str
	full_name: str | None = None
	tenant_id: str
	verified: bool = False
	verification_details: dict[str, Any] = Field(default_factory=dict)
	status: str = "pending"
	created_at: str


# ── Certificate request models ────────────────────────────────────────────────

class CertificateRequestCreate(BaseModel):
	model_config = _CONFIG
	msisdn: str
	certificate_type: str
	applicant_id: str
	applicant_name: str
	reference_number: str | None = None
	tenant_id: str = Field(default="default")
	metadata: dict[str, Any] = Field(default_factory=dict)


class CertificateRequestUpdate(BaseModel):
	model_config = _CONFIG
	status: str | None = None
	certificate_number: str | None = None
	issued_by: str | None = None
	notes: str | None = None


class CertificateRequestResponse(BaseModel):
	model_config = _CONFIG
	id: str = Field(default_factory=uuid7str)
	msisdn: str
	certificate_type: str
	applicant_id: str
	applicant_name: str
	reference_number: str | None = None
	certificate_number: str | None = None
	tenant_id: str
	metadata: dict[str, Any] = Field(default_factory=dict)
	status: str = "submitted"
	issued_by: str | None = None
	created_at: str
	updated_at: str | None = None


# ── USSD Menu models ──────────────────────────────────────────────────────────

class USSDMenuCreate(BaseModel):
	model_config = _CONFIG
	service_code: str
	menu_key: str
	menu_text: str
	menu_level: int = 1
	parent_key: str | None = None
	action: str | None = None
	tenant_id: str = Field(default="default")


class USSDMenuResponse(BaseModel):
	model_config = _CONFIG
	id: str = Field(default_factory=uuid7str)
	service_code: str
	menu_key: str
	menu_text: str
	menu_level: int
	parent_key: str | None = None
	action: str | None = None
	tenant_id: str
	status: str = "active"
	created_at: str


# ── Audit models ──────────────────────────────────────────────────────────────

class USSDSessionFilter(BaseModel):
	model_config = _CONFIG
	msisdn: str | None = None
	service_code: str | None = None
	status: str | None = None
	tenant_id: str = "default"
	page: int = 1
	page_size: int = 50


class USSDSessionList(BaseModel):
	model_config = _CONFIG
	items: list[USSDSessionResponse] = Field(default_factory=list)
	total: int = 0
	page: int = 1
	page_size: int = 50


class USSDEventAudit(BaseModel):
	model_config = _CONFIG
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	msisdn: str | None = None
	resource_id: str | None = None
	details: dict[str, Any] = Field(default_factory=dict)
	created_at: str
