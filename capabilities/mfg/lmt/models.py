"""Pydantic v2 models for APG Lot and Batch Management."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

try:
	from situ_cloudevents._uuid7 import uuid7str
except ImportError:
	from uuid6 import uuid7

	def uuid7str() -> str:
		return str(uuid7())


class MfLmtLot(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	lot_number: str
	lot_type: str = "production"  # production | purchase | process | sub_lot
	item_id: str
	item_code: str
	quantity: float
	uom: str = "EA"
	status: str = "available"
	manufactured_date: str | None = None
	expiry_date: str | None = None
	shelf_life_days: int | None = None
	supplier_id: str | None = None
	supplier_lot_number: str | None = None
	work_order_id: str | None = None
	parent_lot_id: str | None = None  # for sub-lots
	warehouse_id: str | None = None
	quarantine_reason: str = ""
	created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
	created_by: str = "system"
	metadata: dict[str, Any] = Field(default_factory=dict)


class MfLmtGenealogyLink(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	parent_lot_id: str
	child_lot_id: str
	quantity_consumed: float
	work_order_id: str | None = None
	linked_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
	metadata: dict[str, Any] = Field(default_factory=dict)


class MfLmtRecallEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	recall_number: str
	root_lot_id: str
	reason: str
	affected_lot_ids: list[str] = Field(default_factory=list)
	status: str = "initiated"  # initiated | in_progress | completed | cancelled
	initiated_by: str
	initiated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
	completed_at: str | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)
