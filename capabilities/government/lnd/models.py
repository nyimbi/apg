"""Pydantic v2 models for Land Registry (gov_lnd)."""
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


# ── Parcel / cadastre models ──────────────────────────────────────────────────

class ParcelCreate(BaseModel):
	model_config = _CONFIG
	parcel_number: str = Field(..., description="Unique cadastral parcel number")
	county: str
	sub_county: str
	location: str
	area_hectares: float
	land_use: str = Field(default="residential")
	coordinates: dict[str, Any] | None = None
	tenant_id: str = Field(default="default")
	metadata: dict[str, Any] = Field(default_factory=dict)


class ParcelUpdate(BaseModel):
	model_config = _CONFIG
	land_use: str | None = None
	area_hectares: float | None = None
	coordinates: dict[str, Any] | None = None
	metadata: dict[str, Any] | None = None
	status: str | None = None


class ParcelResponse(BaseModel):
	model_config = _CONFIG
	id: str = Field(default_factory=uuid7str)
	parcel_number: str
	county: str
	sub_county: str
	location: str
	area_hectares: float
	land_use: str
	coordinates: dict[str, Any] | None = None
	owner_id: str | None = None
	title_number: str | None = None
	tenant_id: str
	metadata: dict[str, Any] = Field(default_factory=dict)
	status: str = "unregistered"
	created_at: str
	updated_at: str | None = None


# ── Title models ──────────────────────────────────────────────────────────────

class TitleCreate(BaseModel):
	model_config = _CONFIG
	parcel_id: str
	title_number: str
	owner_id: str
	owner_name: str
	owner_type: str = Field(default="individual")  # individual, company, trust
	issue_date: str
	tenure_type: str = Field(default="freehold")  # freehold, leasehold, community
	lease_term_years: int | None = None
	issued_by: str
	tenant_id: str = Field(default="default")


class TitleUpdate(BaseModel):
	model_config = _CONFIG
	owner_id: str | None = None
	owner_name: str | None = None
	status: str | None = None
	notes: str | None = None


class TitleResponse(BaseModel):
	model_config = _CONFIG
	id: str = Field(default_factory=uuid7str)
	parcel_id: str
	title_number: str
	owner_id: str
	owner_name: str
	owner_type: str
	issue_date: str
	tenure_type: str
	lease_term_years: int | None = None
	issued_by: str
	tenant_id: str
	status: str = "active"
	created_at: str
	updated_at: str | None = None


# ── Land transfer models ──────────────────────────────────────────────────────

class TransferCreate(BaseModel):
	model_config = _CONFIG
	title_id: str
	transferor_id: str
	transferor_name: str
	transferee_id: str
	transferee_name: str
	consideration_kes: float
	transfer_date: str
	instrument_number: str
	approved_by: str
	tenant_id: str = Field(default="default")
	metadata: dict[str, Any] = Field(default_factory=dict)


class TransferResponse(BaseModel):
	model_config = _CONFIG
	id: str = Field(default_factory=uuid7str)
	title_id: str
	transferor_id: str
	transferor_name: str
	transferee_id: str
	transferee_name: str
	consideration_kes: float
	transfer_date: str
	instrument_number: str
	approved_by: str
	tenant_id: str
	metadata: dict[str, Any] = Field(default_factory=dict)
	status: str = "pending"
	created_at: str
	completed_at: str | None = None


# ── Adjudication models ───────────────────────────────────────────────────────

class AdjudicationCreate(BaseModel):
	model_config = _CONFIG
	parcel_id: str
	claimant_id: str
	claimant_name: str
	claim_basis: str  # adverse_possession, inheritance, purchase, etc.
	evidence_reference: str
	adjudicator_id: str
	tenant_id: str = Field(default="default")
	metadata: dict[str, Any] = Field(default_factory=dict)


class AdjudicationResponse(BaseModel):
	model_config = _CONFIG
	id: str = Field(default_factory=uuid7str)
	parcel_id: str
	claimant_id: str
	claimant_name: str
	claim_basis: str
	evidence_reference: str
	adjudicator_id: str
	tenant_id: str
	outcome: str | None = None
	outcome_notes: str | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)
	status: str = "submitted"
	created_at: str
	decided_at: str | None = None


# ── Encumbrance models ────────────────────────────────────────────────────────

class EncumbranceCreate(BaseModel):
	model_config = _CONFIG
	title_id: str
	encumbrance_type: str  # mortgage, caveat, charge, easement, restriction
	holder_id: str
	holder_name: str
	amount_kes: float | None = None
	start_date: str
	end_date: str | None = None
	instrument_reference: str
	registered_by: str
	tenant_id: str = Field(default="default")


class EncumbranceUpdate(BaseModel):
	model_config = _CONFIG
	status: str | None = None
	end_date: str | None = None
	discharge_reference: str | None = None
	discharged_by: str | None = None


class EncumbranceResponse(BaseModel):
	model_config = _CONFIG
	id: str = Field(default_factory=uuid7str)
	title_id: str
	encumbrance_type: str
	holder_id: str
	holder_name: str
	amount_kes: float | None = None
	start_date: str
	end_date: str | None = None
	instrument_reference: str
	registered_by: str
	discharge_reference: str | None = None
	discharged_by: str | None = None
	tenant_id: str
	status: str = "active"
	created_at: str
	discharged_at: str | None = None


# ── Valuation roll models ─────────────────────────────────────────────────────

class ValuationCreate(BaseModel):
	model_config = _CONFIG
	parcel_id: str
	valuation_date: str
	market_value_kes: float
	annual_rental_value_kes: float
	unimproved_site_value_kes: float
	valuer_id: str
	valuation_method: str = Field(default="market_comparison")
	tenant_id: str = Field(default="default")


class ValuationResponse(BaseModel):
	model_config = _CONFIG
	id: str = Field(default_factory=uuid7str)
	parcel_id: str
	valuation_date: str
	market_value_kes: float
	annual_rental_value_kes: float
	unimproved_site_value_kes: float
	valuer_id: str
	valuation_method: str
	tenant_id: str
	status: str = "draft"
	approved_by: str | None = None
	created_at: str
	approved_at: str | None = None


# ── Filter / list models ──────────────────────────────────────────────────────

class ParcelFilter(BaseModel):
	model_config = _CONFIG
	county: str | None = None
	land_use: str | None = None
	status: str | None = None
	tenant_id: str = "default"
	page: int = 1
	page_size: int = 50


class LandEventAudit(BaseModel):
	model_config = _CONFIG
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	resource_id: str | None = None
	details: dict[str, Any] = Field(default_factory=dict)
	created_at: str
