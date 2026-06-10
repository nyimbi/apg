"""Pydantic v2 models for Hospitality Analytics."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _uid() -> str:
	return uuid4().hex


class KPISnapshot(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	tenant_id: str
	date: str
	occupancy_rate: float
	adr: float  # Average Daily Rate
	revpar: float  # Revenue Per Available Room
	goppar: float | None = None  # Gross Operating Profit Per Available Room
	total_rooms: int
	occupied_rooms: int
	total_revenue: float
	room_revenue: float
	ancillary_revenue: float
	generated_at: str


class SegmentReport(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	tenant_id: str
	period: str
	segment: str  # leisure|corporate|group|government|airline|other
	room_nights: int
	revenue: float
	avg_rate: float
	share_pct: float
	generated_at: str


class PaceReport(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	tenant_id: str
	report_date: str
	future_date: str
	booked_rooms: int
	booked_revenue: float
	vs_same_time_last_year_pct: float | None
	pickup_last_7_days: int
	on_the_books_adr: float
	generated_at: str


class GuestSatisfactionCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	reservation_id: str
	guest_name: str
	overall_score: float  # 1-10
	room_score: float | None = None
	service_score: float | None = None
	food_score: float | None = None
	cleanliness_score: float | None = None
	comments: str | None = None
	channel: str = "post_stay_email"


class GuestSatisfactionResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str
	tenant_id: str
	reservation_id: str
	guest_name: str
	overall_score: float
	room_score: float | None
	service_score: float | None
	food_score: float | None
	cleanliness_score: float | None
	comments: str | None
	channel: str
	nps_category: str  # promoter|passive|detractor
	created_at: str


class ANAListFilter(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	date_from: str | None = None
	date_to: str | None = None
	segment: str | None = None
	room_type: str | None = None
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
