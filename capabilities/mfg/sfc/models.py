"""Pydantic v2 models for APG Shop Floor Control."""

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


class MfSfcWorkCentre(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	code: str
	name: str
	wc_type: str = "machine"  # machine | labour | subcontract | inspection
	capacity_hours_per_day: float = 8.0
	efficiency_pct: float = 100.0
	queue_capacity: int = 999
	status: str = "active"
	created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
	metadata: dict[str, Any] = Field(default_factory=dict)


class MfSfcRouting(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	item_id: str
	item_code: str
	version: str = "1"
	status: str = "active"
	total_lead_time_days: float = 0.0
	created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
	metadata: dict[str, Any] = Field(default_factory=dict)


class MfSfcOperation(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	routing_id: str
	sequence: int
	operation_code: str
	operation_name: str
	work_centre_id: str
	setup_time_hrs: float = 0.0
	run_time_hrs: float = 0.0
	teardown_time_hrs: float = 0.0
	status: str = "queued"
	work_order_id: str | None = None
	started_at: str | None = None
	completed_at: str | None = None
	operator_id: str | None = None
	created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
	metadata: dict[str, Any] = Field(default_factory=dict)


class MfSfcLabourRecord(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	operation_id: str
	work_order_id: str
	operator_id: str
	hours_logged: float
	labour_type: str = "direct"  # direct | indirect | setup
	recorded_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
	notes: str = ""
	metadata: dict[str, Any] = Field(default_factory=dict)
