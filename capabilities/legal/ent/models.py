"""Entity & Corporate Secretary — Pydantic v2 models."""
from __future__ import annotations

from typing import Any
from uuid_extensions import uuid7str
from pydantic import BaseModel, ConfigDict, Field


class EntEntityCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	legal_name: str
	entity_type: str  # limited_company, llp, branch, holding, subsidiary, ngo
	registration_number: str
	jurisdiction: str
	incorporation_date: str
	registered_address: str
	business_address: str = ""
	tax_pin: str = ""
	vat_number: str = ""
	financial_year_end: str = "12-31"  # MM-DD
	description: str = ""
	metadata: dict[str, Any] = Field(default_factory=dict)


class EntEntityUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	legal_name: str | None = None
	status: str | None = None
	registered_address: str | None = None
	business_address: str | None = None
	tax_pin: str | None = None
	vat_number: str | None = None
	financial_year_end: str | None = None
	description: str | None = None
	metadata: dict[str, Any] | None = None


class EntEntityResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	legal_name: str
	entity_type: str
	registration_number: str
	jurisdiction: str
	incorporation_date: str
	registered_address: str
	business_address: str
	tax_pin: str
	vat_number: str
	financial_year_end: str
	description: str
	status: str
	director_count: int
	shareholder_count: int
	metadata: dict[str, Any]
	created_at: str
	updated_at: str | None = None


class EntEntityListResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	items: list[EntEntityResponse]
	total: int
	page: int = 1
	page_size: int = 50


class EntEntityFilter(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	entity_type: str | None = None
	jurisdiction: str | None = None
	status: str | None = None


class EntDirectorCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	entity_id: str
	full_name: str
	id_number: str
	nationality: str
	appointment_date: str
	role: str = "director"  # director, chairperson, secretary, ceo, cfo
	address: str = ""
	email: str = ""


class EntDirectorResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	entity_id: str
	tenant_id: str
	full_name: str
	id_number: str
	nationality: str
	appointment_date: str
	cessation_date: str | None = None
	role: str
	address: str
	email: str
	status: str
	created_at: str


class EntShareholderCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	entity_id: str
	full_name: str
	id_number: str
	share_class: str = "ordinary"
	shares_held: int
	nominal_value: float
	consideration_paid: float
	nationality: str = ""


class EntShareholderResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	entity_id: str
	tenant_id: str
	full_name: str
	id_number: str
	share_class: str
	shares_held: int
	nominal_value: float
	consideration_paid: float
	nationality: str
	status: str
	created_at: str


class EntFilingCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	entity_id: str
	filing_type: str  # annual_return, change_of_directors, change_of_address, share_allotment
	due_date: str
	filing_period: str = ""
	filed_by_id: str = ""
	reference_number: str = ""
	notes: str = ""


class EntFilingResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	entity_id: str
	tenant_id: str
	filing_type: str
	due_date: str
	filing_period: str
	filed_by_id: str
	reference_number: str
	notes: str
	status: str
	filed_at: str | None = None
	created_at: str


class EntBoardResolutionCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	entity_id: str
	resolution_number: str
	resolution_date: str
	resolution_type: str  # ordinary, special, written
	subject: str
	body: str
	passed_by: list[str] = Field(default_factory=list)


class EntBoardResolutionResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	entity_id: str
	tenant_id: str
	resolution_number: str
	resolution_date: str
	resolution_type: str
	subject: str
	body: str
	passed_by: list[str]
	status: str
	created_at: str


class EntAuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	entity_id: str | None
	event_type: str
	actor_id: str | None
	details: dict[str, Any]
	created_at: str
