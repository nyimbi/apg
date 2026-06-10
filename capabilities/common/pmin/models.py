"""Process Mining — Pydantic v2 models."""
from __future__ import annotations

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


class EventLogCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	name: str
	description: str = ""
	subject_filter: str = ""  # NATS subject pattern e.g. "orders.>"
	case_id_field: str = "case_id"
	activity_field: str = "activity"
	timestamp_field: str = "timestamp"
	resource_field: str = "resource"

class EventLogResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str
	description: str
	subject_filter: str
	case_id_field: str
	activity_field: str
	timestamp_field: str
	resource_field: str
	event_count: int = 0
	case_count: int = 0
	status: str = "active"
	created_at: str

class ProcessEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	case_id: str
	activity: str
	timestamp: str
	resource: str = ""
	attributes: dict[str, Any] = Field(default_factory=dict)

class BPMNModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_log_id: str
	algorithm: str = "alpha_miner"
	nodes: list[dict[str, Any]] = Field(default_factory=list)
	edges: list[dict[str, Any]] = Field(default_factory=list)
	start_activities: list[str] = Field(default_factory=list)
	end_activities: list[str] = Field(default_factory=list)
	discovered_at: str

class ConformanceResult(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_log_id: str
	model_id: str
	fitness: float = 0.0
	precision: float = 0.0
	generalization: float = 0.0
	simplicity: float = 0.0
	deviating_cases: list[str] = Field(default_factory=list)
	checked_at: str

class BottleneckReport(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_log_id: str
	bottlenecks: list[dict[str, Any]] = Field(default_factory=list)
	generated_at: str

class PminAuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	entity_id: str
	payload: dict[str, Any] = Field(default_factory=dict)
	created_at: str

class PminFilter(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	event_log_id: str | None = None
	algorithm: str | None = None
	status: str | None = None
