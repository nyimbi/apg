"""Pydantic v2 models for APG Omnichannel Commerce."""

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
# Channel
# ---------------------------------------------------------------------------

class OmcChannelCreate(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	name: NonEmptyStr
	channel_type: str
	is_active: bool = True
	currency_code: str = "USD"
	locale: str = "en"
	created_by: NonEmptyStr


class OmcChannelResponse(OmcChannelCreate):
	id: str = Field(default_factory=uuid7str)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Catalogue Item (cross-channel)
# ---------------------------------------------------------------------------

class OmcCatalogueItemCreate(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	sku: NonEmptyStr
	name: NonEmptyStr
	description: str | None = None
	base_price: float
	currency_code: str = "USD"
	category_path: list[str] = Field(default_factory=list)
	brand: str | None = None
	barcode: str | None = None
	weight_kg: float | None = None
	is_active: bool = True
	created_by: NonEmptyStr


class OmcCatalogueItemResponse(OmcCatalogueItemCreate):
	id: str = Field(default_factory=uuid7str)
	channel_prices: dict[str, float] = Field(default_factory=dict)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

class OmcInventoryRecord(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	sku: NonEmptyStr
	location_id: NonEmptyStr
	channel_id: NonEmptyStr
	on_hand_qty: int = 0
	reserved_qty: int = 0
	available_qty: int = 0
	safety_stock_qty: int = 0
	visibility_mode: str = "real_time"
	updated_by: NonEmptyStr


class OmcInventoryResponse(OmcInventoryRecord):
	id: str = Field(default_factory=uuid7str)
	last_synced_at: datetime = Field(default_factory=datetime.utcnow)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Cart
# ---------------------------------------------------------------------------

class OmcCartLineItem(BaseModel):
	model_config = _CFG
	sku: NonEmptyStr
	quantity: int
	unit_price: float
	line_total: float
	discount_applied: float = 0.0
	promotion_ids: list[str] = Field(default_factory=list)


class OmcCartCreate(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	channel_id: NonEmptyStr
	customer_id: str | None = None  # guest if None
	session_id: NonEmptyStr
	currency_code: str = "USD"
	items: list[OmcCartLineItem] = Field(default_factory=list)
	created_by: NonEmptyStr


class OmcCartResponse(OmcCartCreate):
	id: str = Field(default_factory=uuid7str)
	subtotal: float = 0.0
	discount_total: float = 0.0
	tax_total: float = 0.0
	grand_total: float = 0.0
	state: str = "active"
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Order
# ---------------------------------------------------------------------------

class OmcOrderCreate(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	channel_id: NonEmptyStr
	cart_id: str | None = None
	customer_id: str | None = None
	fulfilment_mode: str
	store_id: str | None = None  # for C&C / ship-from-store
	delivery_address: dict[str, Any] | None = None
	currency_code: str = "USD"
	items: list[OmcCartLineItem] = Field(default_factory=list)
	payment_method: str
	coupon_codes: list[str] = Field(default_factory=list)
	notes: str | None = None
	created_by: NonEmptyStr


class OmcOrderUpdate(BaseModel):
	model_config = _CFG
	status: str | None = None
	carrier_tracking_number: str | None = None
	estimated_delivery_at: datetime | None = None
	updated_by: NonEmptyStr


class OmcOrderResponse(OmcOrderCreate):
	id: str = Field(default_factory=uuid7str)
	order_number: str = Field(default_factory=lambda: f"ORD-{uuid7str()[:8].upper()}")
	status: str = "draft"
	subtotal: float = 0.0
	discount_total: float = 0.0
	tax_total: float = 0.0
	shipping_total: float = 0.0
	grand_total: float = 0.0
	fraud_check_passed: bool = False
	carrier_tracking_number: str | None = None
	estimated_delivery_at: datetime | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Return
# ---------------------------------------------------------------------------

class OmcReturnCreate(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	order_id: NonEmptyStr
	channel_id: NonEmptyStr
	return_reason: str
	items: list[dict[str, Any]] = Field(default_factory=list)
	refund_method: str
	notes: str | None = None
	created_by: NonEmptyStr


class OmcReturnResponse(OmcReturnCreate):
	id: str = Field(default_factory=uuid7str)
	return_number: str = Field(default_factory=lambda: f"RTN-{uuid7str()[:8].upper()}")
	status: str = "pending"
	refund_amount: float = 0.0
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Journey Event
# ---------------------------------------------------------------------------

class OmcJourneyEventCreate(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	session_id: NonEmptyStr
	customer_id: str | None = None
	channel_id: NonEmptyStr
	journey_stage: str
	event_type: str
	event_payload: dict[str, Any] = Field(default_factory=dict)
	created_by: NonEmptyStr


class OmcJourneyEventResponse(OmcJourneyEventCreate):
	id: str = Field(default_factory=uuid7str)
	occurred_at: datetime = Field(default_factory=datetime.utcnow)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Pricing Rule
# ---------------------------------------------------------------------------

class OmcPricingRuleCreate(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	name: NonEmptyStr
	rule_type: str
	channel_id: str | None = None
	sku_pattern: str | None = None
	category_path: list[str] = Field(default_factory=list)
	adjustment_type: str  # "percentage" or "fixed"
	adjustment_value: float
	priority: int = 100
	valid_from: datetime
	valid_to: datetime | None = None
	created_by: NonEmptyStr


class OmcPricingRuleResponse(OmcPricingRuleCreate):
	id: str = Field(default_factory=uuid7str)
	is_active: bool = True
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
