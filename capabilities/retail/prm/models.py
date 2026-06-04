"""Pydantic v2 models for APG Promotions Management."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid6 import uuid7

from pydantic import BaseModel, ConfigDict, Field, AfterValidator
from typing import Annotated


def uuid7str() -> str:
	return str(uuid7())


def _non_empty(v: str) -> str:
	assert v and v.strip(), "must be non-empty"
	return v.strip()


NonEmptyStr = Annotated[str, AfterValidator(_non_empty)]
_CFG = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------

class PrmPromotionCreate(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	name: NonEmptyStr
	description: str | None = None
	promotion_type: str
	discount_type: str  # "percentage" | "fixed_amount"
	discount_value: float
	channel_restriction: str = "all_channels"
	audience_type: str = "all_customers"
	stack_policy: str = "best_of"
	budget_strategy: str = "total_cap"
	budget_cap: float = 0.0
	margin_floor_pct: float = 5.0
	start_date: datetime
	end_date: datetime
	approval_status: str = "draft"
	excluded_skus: list[str] = Field(default_factory=list)
	excluded_categories: list[str] = Field(default_factory=list)
	created_by: NonEmptyStr


class PrmPromotionUpdate(BaseModel):
	model_config = _CFG
	name: str | None = None
	description: str | None = None
	discount_value: float | None = None
	budget_cap: float | None = None
	start_date: datetime | None = None
	end_date: datetime | None = None
	approval_status: str | None = None
	updated_by: NonEmptyStr


class PrmPromotionResponse(PrmPromotionCreate):
	id: str = Field(default_factory=uuid7str)
	promotion_code: str = Field(default_factory=lambda: f"PROMO-{uuid7str()[:8].upper()}")
	redemption_count: int = 0
	total_discount_issued: float = 0.0
	budget_consumed: float = 0.0
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Promotion Trigger
# ---------------------------------------------------------------------------

class PrmTriggerCreate(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	promotion_id: NonEmptyStr
	trigger_type: str
	trigger_value: Any  # threshold amount, quantity, tier name, etc.
	trigger_operator: str = "gte"  # gte, lte, eq, in
	created_by: NonEmptyStr


class PrmTriggerResponse(PrmTriggerCreate):
	id: str = Field(default_factory=uuid7str)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Coupon
# ---------------------------------------------------------------------------

class PrmCouponCreate(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	promotion_id: NonEmptyStr
	coupon_type: str
	coupon_code: NonEmptyStr
	max_uses: int = 1
	customer_id: str | None = None
	channel_restriction: str | None = None
	valid_from: datetime
	valid_to: datetime
	created_by: NonEmptyStr


class PrmCouponResponse(PrmCouponCreate):
	id: str = Field(default_factory=uuid7str)
	times_used: int = 0
	status: str = "active"
	first_redeemed_at: datetime | None = None
	last_redeemed_at: datetime | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Coupon Redemption
# ---------------------------------------------------------------------------

class PrmCouponRedemptionCreate(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	coupon_id: NonEmptyStr
	promotion_id: NonEmptyStr
	order_id: str | None = None
	transaction_id: str | None = None
	channel_id: NonEmptyStr
	customer_id: str | None = None
	discount_applied: float
	created_by: NonEmptyStr


class PrmCouponRedemptionResponse(PrmCouponRedemptionCreate):
	id: str = Field(default_factory=uuid7str)
	redeemed_at: datetime = Field(default_factory=datetime.utcnow)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Pricing Rule
# ---------------------------------------------------------------------------

class PrmPricingRuleCreate(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	name: NonEmptyStr
	rule_type: str
	sku_pattern: str | None = None
	category_path: list[str] = Field(default_factory=list)
	channel_restriction: str = "all_channels"
	adjustment_type: str  # "percentage" | "fixed_amount"
	adjustment_value: float
	priority: int = 100
	valid_from: datetime
	valid_to: datetime | None = None
	created_by: NonEmptyStr


class PrmPricingRuleResponse(PrmPricingRuleCreate):
	id: str = Field(default_factory=uuid7str)
	is_active: bool = True
	times_applied: int = 0
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

class PrmMarkdownCreate(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	name: NonEmptyStr
	markdown_type: str
	sku_list: list[str] = Field(default_factory=list)
	category_path: list[str] = Field(default_factory=list)
	markdown_pct: float
	floor_margin_pct: float = 5.0
	cascade_enabled: bool = False
	cascade_interval_days: int | None = None
	cascade_increment_pct: float | None = None
	effective_from: datetime
	effective_to: datetime | None = None
	approval_status: str = "draft"
	created_by: NonEmptyStr


class PrmMarkdownResponse(PrmMarkdownCreate):
	id: str = Field(default_factory=uuid7str)
	items_affected: int = 0
	total_margin_impact: float = 0.0
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Effectiveness Report
# ---------------------------------------------------------------------------

class PrmEffectivenessRecord(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	promotion_id: NonEmptyStr
	measurement_period_start: datetime
	measurement_period_end: datetime
	redemption_rate: float
	incremental_revenue: float
	margin_impact: float
	basket_uplift_pct: float
	new_customer_acquisitions: int
	repeat_purchase_rate: float
	roi: float
	calculated_by: NonEmptyStr


class PrmEffectivenessResponse(PrmEffectivenessRecord):
	id: str = Field(default_factory=uuid7str)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
