"""Pydantic v2 models for County / Devolved Services (gov_cty)."""
from __future__ import annotations

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


# ── Revenue collection models ─────────────────────────────────────────────────

class RevenueCollectionCreate(BaseModel):
	model_config = _CONFIG
	payer_id: str
	payer_name: str
	revenue_type: str  # land_rates, business_permit, parking, market_fee, etc.
	amount_kes: float
	period: str  # e.g. "2025-Q1" or "2025-01"
	payment_method: str = Field(default="mpesa")
	receipt_number: str | None = None
	tenant_id: str = Field(default="default")
	metadata: dict[str, Any] = Field(default_factory=dict)


class RevenueCollectionResponse(BaseModel):
	model_config = _CONFIG
	id: str = Field(default_factory=uuid7str)
	payer_id: str
	payer_name: str
	revenue_type: str
	amount_kes: float
	period: str
	payment_method: str
	receipt_number: str | None = None
	tenant_id: str
	metadata: dict[str, Any] = Field(default_factory=dict)
	status: str = "pending"
	created_at: str
	confirmed_at: str | None = None


# ── Permit issuance models ────────────────────────────────────────────────────

class CountyPermitCreate(BaseModel):
	model_config = _CONFIG
	applicant_id: str
	applicant_name: str
	business_name: str
	permit_type: str
	location: str
	sub_county: str
	fee_paid_kes: float
	supporting_documents: list[str] = Field(default_factory=list)
	tenant_id: str = Field(default="default")
	metadata: dict[str, Any] = Field(default_factory=dict)


class CountyPermitUpdate(BaseModel):
	model_config = _CONFIG
	status: str | None = None
	permit_number: str | None = None
	issue_date: str | None = None
	expiry_date: str | None = None
	issued_by: str | None = None
	rejection_reason: str | None = None


class CountyPermitResponse(BaseModel):
	model_config = _CONFIG
	id: str = Field(default_factory=uuid7str)
	applicant_id: str
	applicant_name: str
	business_name: str
	permit_type: str
	location: str
	sub_county: str
	fee_paid_kes: float
	permit_number: str | None = None
	issue_date: str | None = None
	expiry_date: str | None = None
	issued_by: str | None = None
	tenant_id: str
	metadata: dict[str, Any] = Field(default_factory=dict)
	status: str = "submitted"
	created_at: str
	updated_at: str | None = None


# ── Social welfare models ─────────────────────────────────────────────────────

class SocialWelfareApplicationCreate(BaseModel):
	model_config = _CONFIG
	applicant_id: str
	applicant_name: str
	id_number: str
	programme_type: str  # cash_transfer, food_subsidy, elderly_grant, disability_grant
	sub_county: str
	ward: str
	household_size: int
	monthly_income_kes: float = 0.0
	needs_assessment: str | None = None
	tenant_id: str = Field(default="default")
	metadata: dict[str, Any] = Field(default_factory=dict)


class SocialWelfareApplicationUpdate(BaseModel):
	model_config = _CONFIG
	status: str | None = None
	approved_amount_kes: float | None = None
	payment_frequency: str | None = None
	case_worker_id: str | None = None
	notes: str | None = None


class SocialWelfareApplicationResponse(BaseModel):
	model_config = _CONFIG
	id: str = Field(default_factory=uuid7str)
	applicant_id: str
	applicant_name: str
	id_number: str
	programme_type: str
	sub_county: str
	ward: str
	household_size: int
	monthly_income_kes: float
	approved_amount_kes: float | None = None
	payment_frequency: str | None = None
	case_worker_id: str | None = None
	tenant_id: str
	metadata: dict[str, Any] = Field(default_factory=dict)
	status: str = "submitted"
	created_at: str
	updated_at: str | None = None


# ── Health service models ─────────────────────────────────────────────────────

class HealthFacilityCreate(BaseModel):
	model_config = _CONFIG
	facility_code: str
	facility_name: str
	facility_type: str  # dispensary, health_centre, county_hospital, sub_county_hospital
	sub_county: str
	ward: str
	beds: int = 0
	services: list[str] = Field(default_factory=list)
	tenant_id: str = Field(default="default")


class HealthFacilityResponse(BaseModel):
	model_config = _CONFIG
	id: str = Field(default_factory=uuid7str)
	facility_code: str
	facility_name: str
	facility_type: str
	sub_county: str
	ward: str
	beds: int
	services: list[str]
	tenant_id: str
	status: str = "active"
	created_at: str


class PatientRegistrationCreate(BaseModel):
	model_config = _CONFIG
	facility_id: str
	patient_name: str
	id_number: str
	date_of_birth: str
	gender: str
	sub_county: str
	phone: str | None = None
	tenant_id: str = Field(default="default")


class PatientRegistrationResponse(BaseModel):
	model_config = _CONFIG
	id: str = Field(default_factory=uuid7str)
	facility_id: str
	patient_name: str
	id_number: str
	date_of_birth: str
	gender: str
	sub_county: str
	phone: str | None = None
	patient_number: str
	tenant_id: str
	status: str = "active"
	created_at: str


# ── Public works ticketing models ─────────────────────────────────────────────

class PublicWorksTicketCreate(BaseModel):
	model_config = _CONFIG
	reporter_id: str
	reporter_name: str
	reporter_phone: str | None = None
	ticket_type: str  # road_repair, drainage, streetlight, water_supply, waste_collection
	description: str
	location: str
	sub_county: str
	ward: str
	priority: str = Field(default="normal")  # low, normal, high, critical
	tenant_id: str = Field(default="default")
	metadata: dict[str, Any] = Field(default_factory=dict)


class PublicWorksTicketUpdate(BaseModel):
	model_config = _CONFIG
	status: str | None = None
	assigned_to: str | None = None
	resolution_notes: str | None = None
	estimated_completion: str | None = None


class PublicWorksTicketResponse(BaseModel):
	model_config = _CONFIG
	id: str = Field(default_factory=uuid7str)
	reporter_id: str
	reporter_name: str
	reporter_phone: str | None = None
	ticket_type: str
	description: str
	location: str
	sub_county: str
	ward: str
	priority: str
	assigned_to: str | None = None
	resolution_notes: str | None = None
	estimated_completion: str | None = None
	tenant_id: str
	metadata: dict[str, Any] = Field(default_factory=dict)
	status: str = "open"
	created_at: str
	updated_at: str | None = None
	resolved_at: str | None = None


# ── Filter / list / audit models ──────────────────────────────────────────────

class CountyServiceFilter(BaseModel):
	model_config = _CONFIG
	sub_county: str | None = None
	status: str | None = None
	tenant_id: str = "default"
	page: int = 1
	page_size: int = 50


class CountyEventAudit(BaseModel):
	model_config = _CONFIG
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	resource_id: str | None = None
	details: dict[str, Any] = Field(default_factory=dict)
	created_at: str
