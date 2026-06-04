"""Pydantic v2 models for APG Loyalty & Rewards."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any
from uuid6 import uuid7

from pydantic import BaseModel, ConfigDict, Field, AfterValidator


def uuid7str() -> str:
	return str(uuid7())


def _non_empty_str(v: str) -> str:
	assert v and v.strip(), "must be non-empty string"
	return v.strip()


def _valid_points(v: int) -> int:
	assert v >= 0, "points must be non-negative"
	return v


def _valid_confidence(v: float) -> float:
	assert 0.0 <= v <= 1.0, "confidence must be between 0 and 1"
	return v


NonEmptyStr = Annotated[str, AfterValidator(_non_empty_str)]
NonNegativeInt = Annotated[int, AfterValidator(_valid_points)]


_CFG = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)


# ---------------------------------------------------------------------------
# Programme
# ---------------------------------------------------------------------------

class LoyProgrammeCreate(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	name: NonEmptyStr
	programme_type: str
	points_currency: str = "PTS"
	points_to_currency_rate: float = 0.01  # 1 PTS = $0.01
	max_earn_per_transaction: NonNegativeInt = 100000
	max_redeem_per_transaction: NonNegativeInt = 50000
	created_by: NonEmptyStr


class LoyProgrammeResponse(LoyProgrammeCreate):
	id: str = Field(default_factory=uuid7str)
	is_active: bool = True
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Member
# ---------------------------------------------------------------------------

class LoyMemberCreate(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	programme_id: NonEmptyStr
	external_customer_id: NonEmptyStr
	first_name: NonEmptyStr
	last_name: NonEmptyStr
	email: str | None = None
	mobile: str | None = None
	consent_recorded: bool = False
	identity_verified: bool = False
	created_by: NonEmptyStr


class LoyMemberUpdate(BaseModel):
	model_config = _CFG
	first_name: str | None = None
	last_name: str | None = None
	email: str | None = None
	mobile: str | None = None
	status: str | None = None
	updated_by: NonEmptyStr


class LoyMemberResponse(LoyMemberCreate):
	id: str = Field(default_factory=uuid7str)
	member_number: str = Field(default_factory=lambda: f"M{uuid7str()[:8].upper()}")
	current_tier_id: str | None = None
	current_tier_name: str = "bronze"
	points_balance: NonNegativeInt = 0
	lifetime_points_earned: NonNegativeInt = 0
	lifetime_points_redeemed: NonNegativeInt = 0
	status: str = "active"
	clv_segment: str | None = None
	enrolled_at: datetime = Field(default_factory=datetime.utcnow)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Tier
# ---------------------------------------------------------------------------

class LoyTierCreate(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	programme_id: NonEmptyStr
	tier_name: NonEmptyStr
	earn_multiplier: float = 1.0
	qualification_points: NonNegativeInt = 0
	qualification_window_days: int = 365
	downgrade_grace_days: int = 90
	created_by: NonEmptyStr


class LoyTierResponse(LoyTierCreate):
	id: str = Field(default_factory=uuid7str)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Points Transaction
# ---------------------------------------------------------------------------

class LoyTransactionCreate(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	member_id: NonEmptyStr
	programme_id: NonEmptyStr
	transaction_type: str
	points: int  # can be negative for redeems/adjustments
	earn_mechanism: str | None = None
	redeem_mechanism: str | None = None
	reference_id: str | None = None  # POS transaction, order, etc.
	receipt_reference: str | None = None
	notes: str | None = None
	created_by: NonEmptyStr


class LoyTransactionResponse(LoyTransactionCreate):
	id: str = Field(default_factory=uuid7str)
	balance_after: NonNegativeInt = 0
	tier_at_time: str = "bronze"
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Campaign
# ---------------------------------------------------------------------------

class LoyCampaignCreate(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	programme_id: NonEmptyStr
	name: NonEmptyStr
	campaign_type: str
	points_multiplier: float = 1.0
	bonus_points: NonNegativeInt = 0
	audience_type: str = "all_customers"
	channel: str | None = None
	start_date: datetime
	end_date: datetime
	budget_cap_points: NonNegativeInt = 0
	approval_status: str = "draft"
	created_by: NonEmptyStr


class LoyCampaignResponse(LoyCampaignCreate):
	id: str = Field(default_factory=uuid7str)
	points_issued_to_date: NonNegativeInt = 0
	redemption_count: NonNegativeInt = 0
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Partner
# ---------------------------------------------------------------------------

class LoyPartnerCreate(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	programme_id: NonEmptyStr
	partner_name: NonEmptyStr
	partner_role: str
	earn_rate: float | None = None  # points per currency unit
	redeem_rate: float | None = None
	sla_reference: NonEmptyStr
	settlement_frequency_days: int = 30
	created_by: NonEmptyStr


class LoyPartnerResponse(LoyPartnerCreate):
	id: str = Field(default_factory=uuid7str)
	is_active: bool = True
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class LoyRewardCreate(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	programme_id: NonEmptyStr
	reward_name: NonEmptyStr
	points_cost: NonNegativeInt
	redeem_mechanism: str
	stock_available: int | None = None  # None = unlimited
	valid_from: datetime
	valid_to: datetime
	created_by: NonEmptyStr


class LoyRewardResponse(LoyRewardCreate):
	id: str = Field(default_factory=uuid7str)
	status: str = "available"
	redemptions_count: NonNegativeInt = 0
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# CLV Segment
# ---------------------------------------------------------------------------

class LoyClvSegmentRecord(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	member_id: NonEmptyStr
	programme_id: NonEmptyStr
	clv_score: float
	clv_segment: str
	predicted_12m_revenue: float
	recency_days: int
	frequency_transactions: int
	monetary_value: float
	calculated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: NonEmptyStr


class LoyClvSegmentResponse(LoyClvSegmentRecord):
	id: str = Field(default_factory=uuid7str)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
