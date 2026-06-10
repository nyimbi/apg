"""Cooperative Management models — Pydantic v2."""
from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field
from uuid_extensions import uuid7str


class MemberStatus(str, Enum):
	ACTIVE = "active"
	SUSPENDED = "suspended"
	WITHDRAWN = "withdrawn"
	DECEASED = "deceased"


class ShareTransactionType(str, Enum):
	PURCHASE = "purchase"
	TRANSFER = "transfer"
	REDEMPTION = "redemption"
	DIVIDEND = "dividend"


class CoopCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	name: str
	registration_number: str
	region: str
	crop_focus: list[str] = Field(default_factory=list)
	share_value: float
	currency: str = "KES"
	notes: str | None = None


class CoopResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str
	registration_number: str
	region: str
	crop_focus: list[str]
	share_value: float
	currency: str
	total_shares_issued: int = 0
	total_members: int = 0
	notes: str | None = None
	created_at: str
	updated_at: str


class MemberCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	coop_id: str
	farmer_id: str
	name: str
	id_number: str
	shares_purchased: int = 1
	join_date: str
	notes: str | None = None


class MemberResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	coop_id: str
	farmer_id: str
	name: str
	id_number: str
	shares_held: int
	share_value: float
	total_share_value: float
	status: MemberStatus = MemberStatus.ACTIVE
	join_date: str
	notes: str | None = None
	created_at: str
	updated_at: str


class InputPoolCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	coop_id: str
	product_name: str
	total_quantity: float
	unit: str
	unit_cost: float
	season: str
	notes: str | None = None


class InputPoolResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	coop_id: str
	product_name: str
	total_quantity: float
	unit: str
	unit_cost: float
	total_cost: float
	season: str
	allocated_quantity: float = 0
	remaining_quantity: float
	notes: str | None = None
	created_at: str


class DividendAllocationCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	coop_id: str
	financial_year: str
	total_profit: float
	dividend_rate_pct: float
	notes: str | None = None


class DividendAllocationResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	coop_id: str
	financial_year: str
	total_profit: float
	dividend_rate_pct: float
	total_dividend_paid: float
	allocations: list[dict[str, Any]] = Field(default_factory=list)
	notes: str | None = None
	created_at: str


class AnnualReturnCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	coop_id: str
	financial_year: str
	total_revenue: float
	total_expenses: float
	net_profit: float
	member_count: int
	notes: str | None = None


class AnnualReturnResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	coop_id: str
	financial_year: str
	total_revenue: float
	total_expenses: float
	net_profit: float
	member_count: int
	return_on_equity_pct: float | None = None
	notes: str | None = None
	created_at: str


class AuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	entity_id: str
	entity_type: str
	payload: dict[str, Any] = Field(default_factory=dict)
	occurred_at: str
