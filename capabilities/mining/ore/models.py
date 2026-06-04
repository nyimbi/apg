"""Pydantic v2 models for APG Ore Processing & Metallurgy."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from uuid6 import uuid7


def uuid7str() -> str:
	return str(uuid7())


# ── Enums ─────────────────────────────────────────────────────────────────────

class FeedSource(str, Enum):
	ROM_ORE = "rom_ore"
	CRUSHED_ORE = "crushed_ore"
	STOCKPILE_BLEND = "stockpile_blend"
	RECLAIMED = "reclaimed"
	PURCHASED_ORE = "purchased_ore"
	REPROCESSED_TAILINGS = "reprocessed_tailings"


class CircuitStatus(str, Enum):
	RUNNING = "running"
	STANDBY = "standby"
	MAINTENANCE = "maintenance"
	SHUTDOWN = "shutdown"
	COMMISSIONING = "commissioning"
	RAMPUP = "rampup"


class ReagentType(str, Enum):
	CYANIDE = "cyanide"
	LIME = "lime"
	SULPHURIC_ACID = "sulphuric_acid"
	XANTHATE = "xanthate"
	FROTHER = "frother"
	FLOCCULANT = "flocculant"
	ACTIVATED_CARBON = "activated_carbon"
	STEEL_MEDIA = "steel_media"
	GRINDING_MEDIA = "grinding_media"
	COAGULANT = "coagulant"
	DISPERSANT = "dispersant"
	DIESEL = "diesel"
	HYDROGEN_PEROXIDE = "hydrogen_peroxide"
	FERRIC_SULPHATE = "ferric_sulphate"


class ProductType(str, Enum):
	GOLD_DORE = "gold_dore"
	COPPER_CONCENTRATE = "copper_concentrate"
	ZINC_CONCENTRATE = "zinc_concentrate"
	LEAD_CONCENTRATE = "lead_concentrate"
	NICKEL_CONCENTRATE = "nickel_concentrate"
	IRON_ORE_LUMP = "iron_ore_lump"
	IRON_ORE_FINES = "iron_ore_fines"
	COAL_PRODUCT = "coal_product"
	SILVER_DORE = "silver_dore"
	LITHIUM_CARBONATE = "lithium_carbonate"


class BalanceType(str, Enum):
	DAILY = "daily"
	WEEKLY = "weekly"
	MONTHLY = "monthly"
	CAMPAIGN = "campaign"
	ANNUAL = "annual"


class RecoveryMethod(str, Enum):
	ASSAY_BASED = "assay_based"
	MASS_BALANCE = "mass_balance"
	ATTRIBUTABLE_METAL = "attributable_metal"
	RECONCILIATION = "reconciliation"


class ReconciliationStatus(str, Enum):
	OPEN = "open"
	SUBMITTED = "submitted"
	APPROVED = "approved"
	FINALISED = "finalised"


class DeviationType(str, Enum):
	GRADE_DEVIATION = "grade_deviation"
	RECOVERY_DEVIATION = "recovery_deviation"
	THROUGHPUT_DEVIATION = "throughput_deviation"
	REAGENT_DEVIATION = "reagent_deviation"
	QUALITY_DEVIATION = "quality_deviation"


class AlertLevel(str, Enum):
	CRITICAL = "critical"
	HIGH = "high"
	MEDIUM = "medium"
	LOW = "low"


# ── Base ───────────────────────────────────────────────────────────────────────

class OreBase(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)


# ── Plant Feed Record ──────────────────────────────────────────────────────────

class PlantFeedCreate(OreBase):
	tenant_id: str
	feed_source: FeedSource
	period_start: datetime
	period_end: datetime
	dry_tonnes: float = Field(..., gt=0)
	feed_grade: float = Field(..., ge=0)
	grade_units: str
	moisture_pct: float = Field(..., ge=0, le=100)
	particle_size_p80_mm: float | None = Field(None, ge=0)
	density_t_m3: float | None = Field(None, gt=0)
	source_stockpile_id: str | None = None
	entered_by: str

	@model_validator(mode="after")
	def start_before_end(self) -> "PlantFeedCreate":
		assert self.period_start < self.period_end, "period_start must be before period_end"
		return self


class PlantFeedResponse(OreBase):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	feed_source: FeedSource
	period_start: datetime
	period_end: datetime
	dry_tonnes: float
	feed_grade: float
	grade_units: str
	moisture_pct: float
	particle_size_p80_mm: float | None
	density_t_m3: float | None
	source_stockpile_id: str | None
	entered_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str


# ── Process Circuit ────────────────────────────────────────────────────────────

class CircuitStatusUpdateCreate(OreBase):
	tenant_id: str
	circuit_id: str
	circuit_name: str
	circuit_type: str
	status: CircuitStatus
	throughput_tph: float | None = Field(None, ge=0)
	power_kw: float | None = Field(None, ge=0)
	downtime_reason: str | None = None
	updated_by: str
	updated_at: datetime


class CircuitStatusResponse(OreBase):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	circuit_id: str
	circuit_name: str
	circuit_type: str
	status: CircuitStatus
	throughput_tph: float | None
	power_kw: float | None
	downtime_reason: str | None
	updated_by: str
	updated_at: datetime
	created_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str


# ── Reagent Usage ──────────────────────────────────────────────────────────────

class ReagentUsageCreate(OreBase):
	tenant_id: str
	reagent_type: ReagentType
	period_start: datetime
	period_end: datetime
	quantity_kg: float = Field(..., gt=0)
	dosage_rate_g_t: float = Field(..., ge=0)
	circuit_id: str
	batch_number: str | None = None
	supplier: str | None = None
	unit_cost: float | None = Field(None, ge=0)
	entered_by: str


class ReagentUsageResponse(OreBase):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	reagent_type: ReagentType
	period_start: datetime
	period_end: datetime
	quantity_kg: float
	dosage_rate_g_t: float
	circuit_id: str
	batch_number: str | None
	supplier: str | None
	unit_cost: float | None
	total_cost: float | None = None
	entered_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str


# ── Metallurgical Balance ──────────────────────────────────────────────────────

class StreamAssay(OreBase):
	sample_point: str
	dry_tonnes: float = Field(..., ge=0)
	grade_value: float = Field(..., ge=0)
	grade_units: str
	moisture_pct: float | None = Field(None, ge=0, le=100)


class MetallurgicalBalanceCreate(OreBase):
	tenant_id: str
	balance_type: BalanceType
	period_start: datetime
	period_end: datetime
	commodity: str
	recovery_method: RecoveryMethod
	feed_stream: StreamAssay
	concentrate_stream: StreamAssay | None = None
	tailings_stream: StreamAssay | None = None
	additional_streams: list[StreamAssay] = Field(default_factory=list)
	calculated_recovery_pct: float | None = Field(None, ge=0, le=100)
	mass_pull_pct: float | None = Field(None, ge=0, le=100)
	prepared_by: str
	notes: str | None = None

	@model_validator(mode="after")
	def start_before_end(self) -> "MetallurgicalBalanceCreate":
		assert self.period_start < self.period_end, "period_start must be before period_end"
		return self


class MetallurgicalBalanceResponse(OreBase):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	balance_type: BalanceType
	period_start: datetime
	period_end: datetime
	commodity: str
	recovery_method: RecoveryMethod
	feed_stream: dict[str, Any]
	concentrate_stream: dict[str, Any] | None
	tailings_stream: dict[str, Any] | None
	additional_streams: list[dict[str, Any]]
	calculated_recovery_pct: float | None
	mass_pull_pct: float | None
	prepared_by: str
	notes: str | None
	status: ReconciliationStatus = ReconciliationStatus.OPEN
	approved_by: str | None = None
	approved_at: datetime | None = None
	published: bool = False
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str


# ── Product Quality ────────────────────────────────────────────────────────────

class ProductQualityCreate(OreBase):
	tenant_id: str
	product_type: ProductType
	lot_number: str
	sampled_at: datetime
	dry_weight_tonnes: float = Field(..., gt=0)
	moisture_pct: float = Field(..., ge=0, le=100)
	commodity_grade: float = Field(..., ge=0)
	grade_units: str
	deleterious_elements: dict[str, float] = Field(default_factory=dict, description="element -> value mapping")
	particle_size_p80_mm: float | None = Field(None, ge=0)
	meets_specification: bool
	specification_ref: str | None = None
	sampled_by: str
	lab_ref: str | None = None


class ProductQualityResponse(OreBase):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	product_type: ProductType
	lot_number: str
	sampled_at: datetime
	dry_weight_tonnes: float
	moisture_pct: float
	commodity_grade: float
	grade_units: str
	deleterious_elements: dict[str, float]
	particle_size_p80_mm: float | None
	meets_specification: bool
	specification_ref: str | None
	sampled_by: str
	lab_ref: str | None
	dispatched: bool = False
	dispatch_approved_by: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str


# ── Deviation Alert ────────────────────────────────────────────────────────────

class DeviationAlertCreate(OreBase):
	tenant_id: str
	deviation_type: DeviationType
	alert_level: AlertLevel
	circuit_id: str | None = None
	description: str
	actual_value: float
	target_value: float
	units: str
	detected_at: datetime
	detected_by: str


class DeviationAlertResponse(OreBase):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	deviation_type: DeviationType
	alert_level: AlertLevel
	circuit_id: str | None
	description: str
	actual_value: float
	target_value: float
	variance_pct: float
	units: str
	detected_at: datetime
	detected_by: str
	acknowledged: bool = False
	acknowledged_by: str | None = None
	acknowledged_at: datetime | None = None
	resolved: bool = False
	resolved_at: datetime | None = None
	resolution_notes: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
