"""In-memory models for APG Warehouse Operations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Warehouse:
	id: str
	tenant_id: str
	warehouse_type: str
	name: str
	location: str
	storage_condition: str
	capacity_sqm: float
	dock_door_count: int

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class GoodsReceipt:
	id: str
	tenant_id: str
	warehouse_id: str
	receipt_method: str
	supplier_id: str
	po_reference: str
	line_count: int
	temperature_checked: bool
	damage_inspection_completed: bool
	received_at: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class PutawayTask:
	id: str
	tenant_id: str
	receipt_id: str
	strategy: str
	slot_id: str
	confirmed: bool
	completed_at: str | None
	operator_id: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class PickTask:
	id: str
	tenant_id: str
	order_id: str
	pick_method: str
	warehouse_id: str
	lines_count: int
	priority: str
	operator_id: str
	completed_at: str | None

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class PackTask:
	id: str
	tenant_id: str
	pick_task_id: str
	pack_type: str
	weight_kg: float
	weight_checked: bool
	label_printed: bool
	packing_slip_printed: bool
	completed_at: str | None

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CycleCount:
	id: str
	tenant_id: str
	warehouse_id: str
	count_type: str
	initiated_at: str
	completed_at: str | None
	discrepancy_pct: float
	approved: bool
	approved_by: str | None

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class DockDoor:
	id: str
	tenant_id: str
	warehouse_id: str
	door_number: str
	status: str
	current_job_ref: str | None
	last_updated: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class InventoryAdjustment:
	id: str
	tenant_id: str
	warehouse_id: str
	sku: str
	quantity_before: int
	quantity_after: int
	reason: str
	approved_by: str
	adjusted_at: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class WarehouseAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
