"""Pydantic v2 models for Warehouse Management System (scm_wms)."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field

try:
	from uuid_extensions import uuid7str
except ImportError:
	from uuid import uuid4
	def uuid7str() -> str:
		return str(uuid4())


class BinCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	warehouse_id: str
	aisle: str
	bay: str
	level: str
	bin_code: str
	bin_type: str = "standard"  # standard | bulk | cold | hazmat | quarantine
	capacity_units: float | None = None
	capacity_weight_kg: float | None = None
	pick_sequence: int | None = None


class BinResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	warehouse_id: str
	aisle: str
	bay: str
	level: str
	bin_code: str
	bin_type: str
	capacity_units: float | None
	capacity_weight_kg: float | None
	pick_sequence: int | None
	current_qty: float
	status: str
	created_at: str


class PutAwayTaskCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	receipt_id: str
	sku: str
	quantity: float
	bin_id: str | None = None  # None = system will suggest
	assigned_to: str | None = None


class PutAwayTaskResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	receipt_id: str
	sku: str
	quantity: float
	suggested_bin_id: str | None
	confirmed_bin_id: str | None
	assigned_to: str | None
	status: str
	created_at: str
	completed_at: str | None = None


class PickTaskCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	order_id: str
	sku: str
	quantity: float
	bin_id: str
	assigned_to: str | None = None
	pick_method: str = "fifo"  # fifo | fefo | lifo | zone | wave | batch


class PickTaskResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	order_id: str
	sku: str
	quantity: float
	picked_quantity: float
	bin_id: str
	assigned_to: str | None
	pick_method: str
	status: str
	created_at: str
	completed_at: str | None = None


class PackTaskCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	order_id: str
	pick_task_ids: list[str]
	packing_station: str | None = None
	assigned_to: str | None = None


class PackTaskResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	order_id: str
	pick_task_ids: list[str]
	packing_station: str | None
	assigned_to: str | None
	cartons: list[dict[str, Any]]
	total_weight_kg: float | None
	status: str
	created_at: str
	completed_at: str | None = None


class CycleCountCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	warehouse_id: str
	bin_ids: list[str] = Field(default_factory=list)  # empty = full warehouse
	count_method: str = "spot"  # spot | abc | full | zone
	assigned_to: str | None = None


class CycleCountResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	warehouse_id: str
	bin_ids: list[str]
	count_method: str
	assigned_to: str | None
	results: list[dict[str, Any]]
	variance_items: int
	status: str
	created_at: str
	completed_at: str | None = None


class CrossDockCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	inbound_shipment_id: str
	outbound_order_id: str
	sku: str
	quantity: float
	dock_door: str | None = None


class CrossDockResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	inbound_shipment_id: str
	outbound_order_id: str
	sku: str
	quantity: float
	dock_door: str | None
	status: str
	created_at: str
	completed_at: str | None = None


class WmsAuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	record_id: str
	record_type: str
	status: str
	emitted_at: str
