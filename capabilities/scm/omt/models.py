"""Pydantic v2 models for Order Management & Tracking (scm_omt)."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field

try:
	from uuid_extensions import uuid7str
except ImportError:
	from uuid import uuid4
	def uuid7str() -> str:
		return str(uuid4())


class OrderLineCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	sku: str
	description: str | None = None
	quantity: float
	unit_price: float
	currency: str = "USD"
	warehouse_id: str | None = None


class OrderCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	customer_id: str
	customer_reference: str | None = None
	lines: list[OrderLineCreate]
	requested_delivery_date: str | None = None
	shipping_address: dict[str, Any] = Field(default_factory=dict)
	billing_address: dict[str, Any] = Field(default_factory=dict)
	priority: str = "normal"  # urgent | high | normal | low
	notes: str | None = None


class OrderUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	status: str | None = None
	requested_delivery_date: str | None = None
	shipping_address: dict[str, Any] | None = None
	priority: str | None = None
	notes: str | None = None


class OrderResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	order_number: str
	customer_id: str
	customer_reference: str | None
	lines: list[dict[str, Any]]
	total_value: float
	currency: str
	requested_delivery_date: str | None
	promised_delivery_date: str | None
	shipping_address: dict[str, Any]
	billing_address: dict[str, Any]
	priority: str
	notes: str | None
	status: str
	created_at: str
	updated_at: str | None = None


class BackorderCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	order_id: str
	sku: str
	backordered_quantity: float
	reason: str
	expected_fulfilment_date: str | None = None


class BackorderResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	order_id: str
	sku: str
	backordered_quantity: float
	reason: str
	expected_fulfilment_date: str | None
	status: str
	created_at: str


class SplitShipmentCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	order_id: str
	split_lines: list[dict[str, Any]]
	reason: str


class SplitShipmentResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	order_id: str
	split_lines: list[dict[str, Any]]
	reason: str
	status: str
	created_at: str


class OrderPromiseCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	order_id: str
	promised_date: str
	promised_by: str
	confidence_pct: float = 95.0


class OrderPromiseResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	order_id: str
	promised_date: str
	promised_by: str
	confidence_pct: float
	status: str
	created_at: str


class CustomerNotificationCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	order_id: str
	channel: str  # email | sms | push | webhook
	event_type: str
	message: str
	recipient: str


class CustomerNotificationResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	order_id: str
	channel: str
	event_type: str
	message: str
	recipient: str
	status: str
	sent_at: str | None
	created_at: str


class OmtAuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	record_id: str
	record_type: str
	status: str
	emitted_at: str
