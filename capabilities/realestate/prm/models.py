"""Pydantic v2 models for Property Management (prm)."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from uuid6 import uuid7


def uuid7str() -> str:
	return str(uuid7())


class PropertyType(str, Enum):
	office = "office"
	retail = "retail"
	industrial = "industrial"
	residential = "residential"
	mixed_use = "mixed_use"
	hotel = "hotel"
	student_accommodation = "student_accommodation"
	data_centre = "data_centre"
	healthcare_facility = "healthcare_facility"
	land = "land"
	special_purpose = "special_purpose"


class PropertyStatus(str, Enum):
	development = "development"
	pre_completion = "pre_completion"
	active = "active"
	partially_let = "partially_let"
	vacant = "vacant"
	under_refurbishment = "under_refurbishment"
	for_sale = "for_sale"
	sold = "sold"
	demolished = "demolished"


class UnitType(str, Enum):
	office_suite = "office_suite"
	retail_unit = "retail_unit"
	industrial_unit = "industrial_unit"
	apartment = "apartment"
	penthouse = "penthouse"
	studio = "studio"
	car_park = "car_park"
	storage = "storage"
	roof_terrace = "roof_terrace"
	amenity = "amenity"


class UnitStatus(str, Enum):
	available = "available"
	under_offer = "under_offer"
	let = "let"
	owner_occupied = "owner_occupied"
	under_refurbishment = "under_refurbishment"
	held_back = "held_back"
	not_available = "not_available"


class OwnershipStructure(str, Enum):
	freehold = "freehold"
	leasehold = "leasehold"
	commonhold = "commonhold"
	joint_venture = "joint_venture"
	spv = "spv"
	reit = "reit"
	unit_trust = "unit_trust"
	managed_fund = "managed_fund"


class PortfolioTier(str, Enum):
	core = "core"
	core_plus = "core_plus"
	value_add = "value_add"
	opportunistic = "opportunistic"
	development = "development"


class OwnerType(str, Enum):
	institutional = "institutional"
	private_individual = "private_individual"
	corporate = "corporate"
	pension_fund = "pension_fund"
	sovereign_wealth = "sovereign_wealth"
	family_office = "family_office"
	reit = "reit"
	government = "government"


class ManagementModel(str, Enum):
	full_service = "full_service"
	facilities_only = "facilities_only"
	lease_management_only = "lease_management_only"
	financial_only = "financial_only"
	owner_managed = "owner_managed"


# ── Owner ─────────────────────────────────────────────────────────────────────

class OwnerCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	name: str
	owner_type: OwnerType
	registration_number: str | None = None
	email: str
	phone: str | None = None
	address: str | None = None
	bank_account_details: dict[str, str] = Field(default_factory=dict)
	created_by: str


class OwnerResponse(OwnerCreate):
	id: str = Field(default_factory=uuid7str)
	property_ids: list[str] = Field(default_factory=list)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class OwnerUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	email: str | None = None
	phone: str | None = None
	address: str | None = None
	bank_account_details: dict[str, str] | None = None


# ── Property ──────────────────────────────────────────────────────────────────

class PropertyAddress(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	street: str
	city: str
	county: str | None = None
	country: str
	postcode: str | None = None
	latitude: float | None = None
	longitude: float | None = None


class PropertyCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	name: str
	property_type: PropertyType
	address: PropertyAddress
	owner_id: str
	ownership_structure: OwnershipStructure
	portfolio_tier: PortfolioTier
	management_model: ManagementModel
	grade: str | None = None
	gross_area: Decimal | None = None
	net_lettable_area: Decimal | None = None
	area_unit: str = "sqm"
	year_built: int | None = None
	number_of_floors: int | None = None
	currency: str = "KES"
	created_by: str


class PropertyResponse(PropertyCreate):
	id: str = Field(default_factory=uuid7str)
	status: PropertyStatus = PropertyStatus.active
	units: list[str] = Field(default_factory=list)
	occupancy_rate: Decimal = Decimal("0")
	current_valuation: Decimal | None = None
	valuation_date: date | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class PropertyUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	name: str | None = None
	status: PropertyStatus | None = None
	portfolio_tier: PortfolioTier | None = None
	management_model: ManagementModel | None = None
	grade: str | None = None
	current_valuation: Decimal | None = None
	valuation_date: date | None = None


# ── Unit ──────────────────────────────────────────────────────────────────────

class UnitCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	property_id: str
	unit_ref: str
	unit_type: UnitType
	floor: str | None = None
	gross_area: Decimal | None = None
	net_lettable_area: Decimal | None = None
	area_unit: str = "sqm"
	description: str | None = None
	created_by: str


class UnitResponse(UnitCreate):
	id: str = Field(default_factory=uuid7str)
	status: UnitStatus = UnitStatus.available
	current_lease_id: str | None = None
	current_tenant_id: str | None = None
	current_rent: Decimal | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class UnitUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	status: UnitStatus | None = None
	current_lease_id: str | None = None
	current_tenant_id: str | None = None
	current_rent: Decimal | None = None
	description: str | None = None


# ── Performance KPI ───────────────────────────────────────────────────────────

class KpiCalculationRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	property_id: str | None = None  # None = portfolio-wide
	kpi_names: list[str]
	period: str  # YYYY-MM or YYYY
	requested_by: str


class KpiResult(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	kpi_name: str
	value: Decimal
	unit: str
	period: str
	property_id: str | None = None
	calculated_at: datetime = Field(default_factory=datetime.utcnow)


class KpiResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	property_id: str | None = None
	period: str
	results: list[KpiResult]
	created_at: datetime = Field(default_factory=datetime.utcnow)


# ── Owner Distribution ────────────────────────────────────────────────────────

class DistributionCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	owner_id: str
	property_id: str
	period: str  # YYYY-MM
	gross_income: Decimal
	deductions: Decimal = Decimal("0")
	net_distribution: Decimal
	currency: str = "KES"
	payment_date: date
	created_by: str

	@field_validator("net_distribution")
	@classmethod
	def _net_non_negative(cls, v: Decimal) -> Decimal:
		if v < 0:
			raise ValueError("net_distribution must be non-negative")
		return v


class DistributionResponse(DistributionCreate):
	id: str = Field(default_factory=uuid7str)
	status: str = "pending"
	second_approver: str | None = None
	paid_at: datetime | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Handover ──────────────────────────────────────────────────────────────────

class HandoverCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	property_id: str
	unit_id: str | None = None
	handover_type: str
	from_party: str
	to_party: str
	scheduled_date: date
	notes: str | None = None
	created_by: str


class HandoverResponse(HandoverCreate):
	id: str = Field(default_factory=uuid7str)
	status: str = "scheduled"
	completed_at: datetime | None = None
	checklist_items: list[dict[str, Any]] = Field(default_factory=list)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
