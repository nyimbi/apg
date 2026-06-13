"""Pydantic v2 models for APG Capacity Planning."""

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


class MfCapWorkCentreCapacity(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	work_centre_id: str
	work_centre_code: str
	capacity_type: str = "machine"
	period_start: str
	period_end: str
	available_hours: float
	efficiency_pct: float = 100.0
	effective_hours: float = 0.0  # available_hours * efficiency_pct / 100
	created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
	metadata: dict[str, Any] = Field(default_factory=dict)


class MfCapLoadRecord(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	work_centre_id: str
	period_start: str
	period_end: str
	load_source: str  # production_order | planned_order | forecast
	source_id: str
	required_hours: float
	utilisation_pct: float | None = None
	is_overloaded: bool = False
	created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
	metadata: dict[str, Any] = Field(default_factory=dict)
