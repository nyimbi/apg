"""Pydantic v2 models for APG Computer-Aided Manufacturing."""

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


class MfCamNcProgram(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	program_number: str
	program_name: str
	machine_type: str
	machine_id: str | None = None
	item_id: str | None = None
	item_code: str | None = None
	operation_code: str | None = None
	version: str = "1"
	status: str = "draft"
	nc_code: str = ""  # The actual G-code / CNC program text
	simulation_passed: bool = False
	approved_by: str | None = None
	approved_at: str | None = None
	released_at: str | None = None
	created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
	created_by: str = "system"
	metadata: dict[str, Any] = Field(default_factory=dict)


class MfCamTool(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	tool_number: str
	tool_name: str
	tool_type: str
	diameter_mm: float | None = None
	length_mm: float | None = None
	material: str = ""
	cutting_speed_mpm: float | None = None
	feed_rate_mmpr: float | None = None
	tool_life_minutes: float | None = None
	used_minutes: float = 0.0
	status: str = "active"
	created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
	metadata: dict[str, Any] = Field(default_factory=dict)
