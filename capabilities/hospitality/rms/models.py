"""Pydantic v2 models for Revenue Management & Rates."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _uid() -> str:
	return uuid4().hex


class RatePlanCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	code: str
	name: str
	description: str | None = None
	base_rate: float
	room_type: str
	min_stay: int = 1
	max_stay: int | None = None
	meal_plan: str = "room_only"  # room_only|bed_breakfast|half_board|full_board|all_inclusive
	cancellation_policy: str = "flexible"
	advance_purchase_days: int = 0
	is_public: bool = True


class RatePlanUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	name: str | None = None
	base_rate: float | None = None
	min_stay: int | None = None
	is_active: bool | None = None
	description: str | None = None


class RatePlanResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str
	tenant_id: str
	code: str
	name: str
	base_rate: float
	room_type: str
	min_stay: int
	meal_plan: str
	cancellation_policy: str
	is_public: bool
	is_active: bool
	status: str
	created_at: str


class DemandForecastCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	forecast_date: str
	room_type: str
	predicted_demand: float  # 0.0 to 1.0 occupancy ratio
	confidence: float = 0.8
	events: list[str] = Field(default_factory=list)
	notes: str | None = None


class DemandForecastResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str
	tenant_id: str
	forecast_date: str
	room_type: str
	predicted_demand: float
	confidence: float
	recommended_rate: float
	events: list[str]
	created_at: str


class CompetitorRateCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	competitor_name: str
	room_type: str
	rate: float
	date: str
	source: str = "manual"  # manual|scraper|ota
	channel: str | None = None


class CompetitorRateResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str
	tenant_id: str
	competitor_name: str
	room_type: str
	rate: float
	date: str
	source: str
	channel: str | None
	created_at: str


class YieldOptimisationRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	date_from: str
	date_to: str
	room_type: str
	current_occupancy: float
	target_occupancy: float = 0.85


class YieldOptimisationResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	tenant_id: str
	date_from: str
	date_to: str
	room_type: str
	current_occupancy: float
	recommended_rate: float
	rate_change_pct: float
	strategy: str
	generated_at: str


class RateParityAlert(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=_uid)
	tenant_id: str
	room_type: str
	date: str
	our_rate: float
	competitor_rate: float
	channel: str
	variance_pct: float
	severity: str  # low|medium|high
	created_at: str


class RMSListFilter(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	room_type: str | None = None
	date_from: str | None = None
	date_to: str | None = None
	status: str | None = None
	limit: int = 100
	offset: int = 0


class AuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=_uid)
	tenant_id: str
	event_type: str
	record_id: str
	record_type: str
	details: dict[str, Any] = Field(default_factory=dict)
	created_at: str
