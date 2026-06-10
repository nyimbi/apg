"""Programme & Project Monitoring — Pydantic v2 models."""
from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

try:
	from uuid6 import uuid7
	def uuid7str() -> str:
		return str(uuid7())
except ImportError:
	from uuid import uuid4
	def uuid7str() -> str:  # type: ignore[misc]
		return str(uuid4())


_cfg = ConfigDict(extra="forbid", validate_by_name=True)


class PrgProgrammeCreate(BaseModel):
	model_config = _cfg
	name: str
	code: str
	description: str = ""
	sector: str = ""
	start_date: str
	end_date: str
	budget: Decimal = Decimal("0")
	currency: str = "KES"
	lead_staff: str = ""
	geographic_focus: str = ""


class PrgProgrammeUpdate(BaseModel):
	model_config = _cfg
	name: str | None = None
	description: str | None = None
	end_date: str | None = None
	budget: Decimal | None = None
	status: str | None = None
	lead_staff: str | None = None


class PrgProgrammeResponse(BaseModel):
	model_config = _cfg
	id: str = Field(default_factory=uuid7str)
	name: str
	code: str
	description: str
	sector: str
	start_date: str
	end_date: str
	budget: Decimal
	currency: str
	lead_staff: str
	geographic_focus: str
	status: str
	tenant_id: str
	created_at: str
	updated_at: str | None = None


class PrgLogframeCreate(BaseModel):
	model_config = _cfg
	programme_id: str
	goal: str
	purpose: str
	outputs: list[str] = Field(default_factory=list)
	activities: list[str] = Field(default_factory=list)
	assumptions: list[str] = Field(default_factory=list)
	version: str = "1.0"


class PrgLogframeResponse(BaseModel):
	model_config = _cfg
	id: str = Field(default_factory=uuid7str)
	programme_id: str
	goal: str
	purpose: str
	outputs: list[str]
	activities: list[str]
	assumptions: list[str]
	version: str
	status: str
	tenant_id: str
	created_at: str


class PrgActivityCreate(BaseModel):
	model_config = _cfg
	programme_id: str
	logframe_id: str | None = None
	name: str
	description: str = ""
	responsible_person: str = ""
	planned_start: str
	planned_end: str
	budget: Decimal = Decimal("0")
	currency: str = "KES"


class PrgActivityResponse(BaseModel):
	model_config = _cfg
	id: str = Field(default_factory=uuid7str)
	programme_id: str
	logframe_id: str | None
	name: str
	description: str
	responsible_person: str
	planned_start: str
	planned_end: str
	actual_start: str | None
	actual_end: str | None
	budget: Decimal
	currency: str
	completion_pct: float
	status: str
	tenant_id: str
	created_at: str


class PrgOutputCreate(BaseModel):
	model_config = _cfg
	activity_id: str
	programme_id: str
	output_type: str = "quantitative"
	description: str
	target_value: float
	unit: str = ""
	reporting_date: str
	recorded_by: str


class PrgOutputResponse(BaseModel):
	model_config = _cfg
	id: str = Field(default_factory=uuid7str)
	activity_id: str
	programme_id: str
	output_type: str
	description: str
	target_value: float
	achieved_value: float = 0.0
	unit: str
	reporting_date: str
	recorded_by: str
	achievement_pct: float = 0.0
	status: str
	tenant_id: str
	created_at: str


class PrgFieldDataCreate(BaseModel):
	model_config = _cfg
	programme_id: str
	activity_id: str | None = None
	collector: str
	collection_date: str
	location: str = ""
	data_type: str = "observation"
	data: dict[str, Any] = Field(default_factory=dict)
	notes: str = ""


class PrgFieldDataResponse(BaseModel):
	model_config = _cfg
	id: str = Field(default_factory=uuid7str)
	programme_id: str
	activity_id: str | None
	collector: str
	collection_date: str
	location: str
	data_type: str
	data: dict[str, Any]
	notes: str
	verified: bool = False
	tenant_id: str
	created_at: str


class PrgProgrammeFilter(BaseModel):
	model_config = _cfg
	status: str | None = None
	sector: str | None = None
	lead_staff: str | None = None


class PrgAuditEvent(BaseModel):
	model_config = _cfg
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	record_id: str
	record_type: str
	details: dict[str, Any] = Field(default_factory=dict)
	emitted_at: str


__all__ = [
	"PrgProgrammeCreate", "PrgProgrammeUpdate", "PrgProgrammeResponse",
	"PrgLogframeCreate", "PrgLogframeResponse",
	"PrgActivityCreate", "PrgActivityResponse",
	"PrgOutputCreate", "PrgOutputResponse",
	"PrgFieldDataCreate", "PrgFieldDataResponse",
	"PrgProgrammeFilter", "PrgAuditEvent",
]
