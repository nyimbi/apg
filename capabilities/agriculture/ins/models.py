"""Crop Insurance models — Pydantic v2."""
from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field
from uuid_extensions import uuid7str


class PolicyStatus(str, Enum):
	QUOTED = "quoted"
	ACTIVE = "active"
	LAPSED = "lapsed"
	EXPIRED = "expired"
	CLAIMED = "claimed"
	CANCELLED = "cancelled"


class ClaimStatus(str, Enum):
	SUBMITTED = "submitted"
	UNDER_REVIEW = "under_review"
	APPROVED = "approved"
	PAID = "paid"
	REJECTED = "rejected"


class TriggerType(str, Enum):
	RAINFALL_DEFICIT = "rainfall_deficit"
	RAINFALL_EXCESS = "rainfall_excess"
	TEMPERATURE_EXTREME = "temperature_extreme"
	DROUGHT_INDEX = "drought_index"
	NDVI_DECLINE = "ndvi_decline"
	WIND_SPEED = "wind_speed"


class ProductCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	name: str
	trigger_type: TriggerType
	trigger_threshold: float
	trigger_unit: str
	payout_per_unit: float
	max_payout: float
	coverage_period_months: int
	eligible_crops: list[str] = Field(default_factory=list)
	eligible_regions: list[str] = Field(default_factory=list)
	base_premium_rate_pct: float
	notes: str | None = None


class ProductResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str
	trigger_type: TriggerType
	trigger_threshold: float
	trigger_unit: str
	payout_per_unit: float
	max_payout: float
	coverage_period_months: int
	eligible_crops: list[str]
	eligible_regions: list[str]
	base_premium_rate_pct: float
	notes: str | None = None
	active: bool = True
	created_at: str
	updated_at: str


class PolicyCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	farmer_id: str
	product_id: str
	crop_id: str
	farm_parcel_id: str
	sum_insured: float
	currency: str = "KES"
	coverage_start: str
	coverage_end: str
	season: str
	notes: str | None = None


class PolicyUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	status: PolicyStatus | None = None
	premium_paid_at: str | None = None
	notes: str | None = None


class PolicyResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	farmer_id: str
	product_id: str
	crop_id: str
	farm_parcel_id: str
	sum_insured: float
	premium_amount: float
	currency: str
	coverage_start: str
	coverage_end: str
	season: str
	status: PolicyStatus = PolicyStatus.QUOTED
	premium_paid_at: str | None = None
	notes: str | None = None
	created_at: str
	updated_at: str


class ClaimCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	policy_id: str
	trigger_event: str
	trigger_value: float
	observed_at: str
	evidence_source: str
	evidence_reference: str | None = None
	notes: str | None = None


class ClaimUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	status: ClaimStatus | None = None
	verified_trigger_value: float | None = None
	approved_payout: float | None = None
	rejection_reason: str | None = None
	paid_at: str | None = None
	notes: str | None = None


class ClaimResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	policy_id: str
	farmer_id: str
	trigger_event: str
	trigger_value: float
	observed_at: str
	evidence_source: str
	evidence_reference: str | None = None
	status: ClaimStatus = ClaimStatus.SUBMITTED
	verified_trigger_value: float | None = None
	approved_payout: float | None = None
	rejection_reason: str | None = None
	paid_at: str | None = None
	notes: str | None = None
	created_at: str
	updated_at: str


class PremiumCalculation(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	product_id: str
	farmer_id: str
	sum_insured: float
	base_premium: float
	risk_adjustment: float
	final_premium: float
	currency: str
	rate_pct: float


class AuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	entity_id: str
	entity_type: str
	payload: dict[str, Any] = Field(default_factory=dict)
	occurred_at: str
