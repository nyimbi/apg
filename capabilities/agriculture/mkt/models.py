"""Agri-Marketplace models — Pydantic v2."""
from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field
from uuid_extensions import uuid7str


class ListingStatus(str, Enum):
	DRAFT = "draft"
	ACTIVE = "active"
	MATCHED = "matched"
	SOLD = "sold"
	EXPIRED = "expired"
	CANCELLED = "cancelled"


class BidStatus(str, Enum):
	PENDING = "pending"
	ACCEPTED = "accepted"
	REJECTED = "rejected"
	COUNTERED = "countered"
	WITHDRAWN = "withdrawn"


class EscrowStatus(str, Enum):
	FUNDED = "funded"
	RELEASED = "released"
	DISPUTED = "disputed"
	REFUNDED = "refunded"


class AuctionStatus(str, Enum):
	SCHEDULED = "scheduled"
	OPEN = "open"
	CLOSED = "closed"
	SETTLED = "settled"
	CANCELLED = "cancelled"


class ListingCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	farmer_id: str
	product_type: str
	variety: str | None = None
	quantity_kg: float
	asking_price_per_kg: float
	currency: str = "KES"
	harvest_date: str
	available_from: str
	available_to: str
	location: str
	quality_grade: str | None = None
	description: str | None = None
	images: list[str] = Field(default_factory=list)


class ListingUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	asking_price_per_kg: float | None = None
	quantity_kg: float | None = None
	available_to: str | None = None
	status: ListingStatus | None = None
	description: str | None = None


class ListingResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	farmer_id: str
	product_type: str
	variety: str | None = None
	quantity_kg: float
	asking_price_per_kg: float
	currency: str
	harvest_date: str
	available_from: str
	available_to: str
	location: str
	quality_grade: str | None = None
	description: str | None = None
	images: list[str] = Field(default_factory=list)
	status: ListingStatus = ListingStatus.DRAFT
	views_count: int = 0
	bids_count: int = 0
	created_at: str
	updated_at: str


class BidCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	listing_id: str
	buyer_id: str
	offered_price_per_kg: float
	quantity_kg: float
	message: str | None = None


class BidResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	listing_id: str
	buyer_id: str
	offered_price_per_kg: float
	quantity_kg: float
	total_value: float
	currency: str
	message: str | None = None
	status: BidStatus = BidStatus.PENDING
	counter_price: float | None = None
	created_at: str
	updated_at: str


class EscrowCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	bid_id: str
	listing_id: str
	buyer_id: str
	farmer_id: str
	amount: float
	currency: str = "KES"


class EscrowResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	bid_id: str
	listing_id: str
	buyer_id: str
	farmer_id: str
	amount: float
	currency: str
	status: EscrowStatus = EscrowStatus.FUNDED
	funded_at: str
	released_at: str | None = None
	notes: str | None = None
	created_at: str


class AuctionCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	listing_id: str
	start_at: str
	end_at: str
	reserve_price: float
	increment: float = 0.5
	notes: str | None = None


class AuctionResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	listing_id: str
	start_at: str
	end_at: str
	reserve_price: float
	increment: float
	current_bid: float | None = None
	current_bidder: str | None = None
	bid_count: int = 0
	status: AuctionStatus = AuctionStatus.SCHEDULED
	winner_id: str | None = None
	winning_bid: float | None = None
	notes: str | None = None
	created_at: str
	updated_at: str


class PriceDiscoveryResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	product_type: str
	region: str | None = None
	period: str
	avg_price_per_kg: float
	min_price_per_kg: float
	max_price_per_kg: float
	median_price_per_kg: float
	sample_size: int
	currency: str = "KES"


class AuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	entity_id: str
	entity_type: str
	payload: dict[str, Any] = Field(default_factory=dict)
	occurred_at: str
