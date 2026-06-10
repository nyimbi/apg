"""Pydantic v2 models for F&B Management."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _uid() -> str:
	return uuid4().hex


class MenuItemCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	name: str
	category: str  # starter|main|dessert|beverage|cocktail|wine|special
	description: str | None = None
	price: float
	cost: float = 0.0
	allergens: list[str] = Field(default_factory=list)
	is_available: bool = True
	prep_time_mins: int = 15


class MenuItemUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	name: str | None = None
	price: float | None = None
	cost: float | None = None
	is_available: bool | None = None
	description: str | None = None
	prep_time_mins: int | None = None


class MenuItemResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str
	tenant_id: str
	name: str
	category: str
	description: str | None
	price: float
	cost: float
	gross_margin_pct: float
	allergens: list[str]
	is_available: bool
	prep_time_mins: int
	status: str
	created_at: str


class TableCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	table_number: str
	section: str
	capacity: int
	notes: str | None = None


class TableResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str
	tenant_id: str
	table_number: str
	section: str
	capacity: int
	status: str
	notes: str | None
	created_at: str


class OrderCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	table_id: str
	server_id: str
	items: list[dict[str, Any]]  # [{item_id, quantity, notes}]
	order_type: str = "dine_in"  # dine_in|room_service|takeaway|bar


class OrderItemCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	item_id: str
	quantity: int = 1
	notes: str | None = None
	modifiers: list[str] = Field(default_factory=list)


class OrderResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str
	tenant_id: str
	table_id: str
	server_id: str
	order_type: str
	items: list[dict[str, Any]]
	subtotal: float
	tax: float
	total: float
	status: str
	kitchen_status: str
	created_at: str


class KitchenTicketResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str
	order_id: str
	table_number: str
	items: list[dict[str, Any]]
	priority: str
	status: str
	sent_at: str
	completed_at: str | None


class FDBListFilter(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	category: str | None = None
	status: str | None = None
	date_from: str | None = None
	date_to: str | None = None
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
