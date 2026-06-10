"""Crop Management models — Pydantic v2."""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from uuid_extensions import uuid7str


class CropStatus(str, Enum):
	PLANNED = "planned"
	PLANTED = "planted"
	GROWING = "growing"
	HARVESTED = "harvested"
	FAILED = "failed"


class GrowthStage(str, Enum):
	GERMINATION = "germination"
	SEEDLING = "seedling"
	VEGETATIVE = "vegetative"
	FLOWERING = "flowering"
	FRUITING = "fruiting"
	MATURITY = "maturity"
	HARVEST_READY = "harvest_ready"


class RotationStrategy(str, Enum):
	LEGUME_CEREAL = "legume_cereal"
	THREE_YEAR = "three_year"
	FOUR_YEAR = "four_year"
	CONTINUOUS = "continuous"
	CUSTOM = "custom"


# --- Variety ---

class VarietyCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	name: str
	crop_type: str
	maturity_days: int
	yield_potential_kg_ha: float
	drought_tolerance: str | None = None
	disease_resistance: list[str] = Field(default_factory=list)
	optimal_rainfall_mm: float | None = None
	optimal_temp_min_c: float | None = None
	optimal_temp_max_c: float | None = None
	notes: str | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)


class VarietyUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	name: str | None = None
	maturity_days: int | None = None
	yield_potential_kg_ha: float | None = None
	drought_tolerance: str | None = None
	disease_resistance: list[str] | None = None
	notes: str | None = None
	metadata: dict[str, Any] | None = None


class VarietyResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str
	crop_type: str
	maturity_days: int
	yield_potential_kg_ha: float
	drought_tolerance: str | None = None
	disease_resistance: list[str] = Field(default_factory=list)
	optimal_rainfall_mm: float | None = None
	optimal_temp_min_c: float | None = None
	optimal_temp_max_c: float | None = None
	notes: str | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)
	created_at: str
	updated_at: str


# --- Planting Calendar ---

class PlantingCalendarCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	crop_type: str
	variety_id: str | None = None
	region: str
	planting_window_start: str  # MM-DD
	planting_window_end: str    # MM-DD
	harvest_window_start: str | None = None
	harvest_window_end: str | None = None
	recommended_density_plants_ha: float | None = None
	input_requirements: dict[str, Any] = Field(default_factory=dict)
	notes: str | None = None


class PlantingCalendarUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	planting_window_start: str | None = None
	planting_window_end: str | None = None
	harvest_window_start: str | None = None
	harvest_window_end: str | None = None
	recommended_density_plants_ha: float | None = None
	input_requirements: dict[str, Any] | None = None
	notes: str | None = None


class PlantingCalendarResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	crop_type: str
	variety_id: str | None = None
	region: str
	planting_window_start: str
	planting_window_end: str
	harvest_window_start: str | None = None
	harvest_window_end: str | None = None
	recommended_density_plants_ha: float | None = None
	input_requirements: dict[str, Any] = Field(default_factory=dict)
	notes: str | None = None
	created_at: str
	updated_at: str


# --- Crop Record ---

class CropCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	farm_parcel_id: str
	crop_type: str
	variety_id: str | None = None
	season: str
	planting_date: str
	expected_harvest_date: str | None = None
	area_ha: float
	status: CropStatus = CropStatus.PLANNED
	target_yield_kg: float | None = None
	seed_lot_reference: str | None = None
	notes: str | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)


class CropUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	status: CropStatus | None = None
	expected_harvest_date: str | None = None
	actual_harvest_date: str | None = None
	actual_yield_kg: float | None = None
	notes: str | None = None
	metadata: dict[str, Any] | None = None


class CropResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	farm_parcel_id: str
	crop_type: str
	variety_id: str | None = None
	season: str
	planting_date: str
	expected_harvest_date: str | None = None
	actual_harvest_date: str | None = None
	area_ha: float
	status: CropStatus
	target_yield_kg: float | None = None
	actual_yield_kg: float | None = None
	seed_lot_reference: str | None = None
	notes: str | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)
	created_at: str
	updated_at: str


# --- Phenology Observation ---

class PhenologyCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	crop_id: str
	observed_at: str
	growth_stage: GrowthStage
	observer_id: str | None = None
	notes: str | None = None
	images: list[str] = Field(default_factory=list)
	measurements: dict[str, Any] = Field(default_factory=dict)


class PhenologyResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	crop_id: str
	observed_at: str
	growth_stage: GrowthStage
	observer_id: str | None = None
	notes: str | None = None
	images: list[str] = Field(default_factory=list)
	measurements: dict[str, Any] = Field(default_factory=dict)
	created_at: str


# --- Rotation Plan ---

class RotationPlanCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	farm_parcel_id: str
	strategy: RotationStrategy
	start_season: str
	crop_sequence: list[str]
	rationale: str | None = None
	notes: str | None = None


class RotationPlanResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	farm_parcel_id: str
	strategy: RotationStrategy
	start_season: str
	crop_sequence: list[str]
	rationale: str | None = None
	notes: str | None = None
	created_at: str
	updated_at: str


# --- Yield Record ---

class YieldRecordCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	crop_id: str
	harvest_date: str
	gross_yield_kg: float
	net_yield_kg: float | None = None
	moisture_pct: float | None = None
	grade: str | None = None
	storage_location: str | None = None
	notes: str | None = None


class YieldRecordResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	crop_id: str
	harvest_date: str
	gross_yield_kg: float
	net_yield_kg: float | None = None
	moisture_pct: float | None = None
	grade: str | None = None
	storage_location: str | None = None
	notes: str | None = None
	created_at: str


# --- List / Filter ---

class CropListFilter(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	farm_parcel_id: str | None = None
	crop_type: str | None = None
	season: str | None = None
	status: CropStatus | None = None
	limit: int = 50
	offset: int = 0


class AuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	entity_id: str
	entity_type: str
	actor_id: str | None = None
	payload: dict[str, Any] = Field(default_factory=dict)
	occurred_at: str
