"""Pydantic v2 models for Guest Loyalty Programme."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _uid() -> str:
	return uuid4().hex


class LoyaltyMemberCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	guest_id: str
	first_name: str
	last_name: str
	email: str
	phone: str | None = None
	enrollment_source: str = "front_desk"  # front_desk|online|mobile|partner


class LoyaltyMemberUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	phone: str | None = None
	email: str | None = None
	preferences: dict[str, Any] | None = None
	status: str | None = None


class LoyaltyMemberResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str
	tenant_id: str
	guest_id: str
	membership_number: str
	first_name: str
	last_name: str
	email: str
	tier: str  # bronze|silver|gold|platinum
	points_balance: int
	lifetime_points: int
	lifetime_spend: float
	tier_qualifying_nights: int
	tier_qualifying_spend: float
	enrollment_source: str
	status: str
	created_at: str


class PointsTransactionCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	member_id: str
	transaction_type: str  # earn|redeem|expire|adjust|bonus
	points: int
	description: str
	reference_id: str | None = None  # reservation_id, folio_id, etc.
	spend_amount: float = 0.0


class PointsTransactionResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str
	tenant_id: str
	member_id: str
	transaction_type: str
	points: int
	running_balance: int
	description: str
	reference_id: str | None
	spend_amount: float
	created_at: str


class TierRuleCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	tier: str
	min_points: int = 0
	min_lifetime_spend: float = 0.0
	min_nights: int = 0
	benefits: list[str] = Field(default_factory=list)
	points_multiplier: float = 1.0
	base_earn_rate: float = 1.0  # points per KES spent


class PartnerCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	partner_name: str
	partner_type: str  # airline|car_rental|restaurant|retail|entertainment
	earn_rate: float = 1.0  # points per unit of partner spend
	redeem_rate: float = 1.0  # points per unit of partner redemption


class PartnerResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str
	tenant_id: str
	partner_name: str
	partner_type: str
	earn_rate: float
	redeem_rate: float
	is_active: bool
	created_at: str


class LOYListFilter(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	tier: str | None = None
	status: str | None = None
	date_from: str | None = None
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
