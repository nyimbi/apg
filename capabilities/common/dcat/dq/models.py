"""Data Quality — Pydantic v2 models."""
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


class DQRuleCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	dataset_id: str
	column: str = ""
	rule_type: str  # completeness | uniqueness | accuracy | range | regex | referential | custom
	expression: str = ""
	threshold: float = 1.0
	severity: str = "warning"  # info | warning | error | critical
	description: str = ""

class DQRuleUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	expression: str | None = None
	threshold: float | None = None
	severity: str | None = None
	description: str | None = None
	active: bool | None = None

class DQRuleResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	dataset_id: str
	column: str
	rule_type: str
	expression: str
	threshold: float
	severity: str
	description: str
	active: bool = True
	created_at: str

class DQProfileCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	dataset_id: str
	row_count: int = 0
	column_profiles: list[dict[str, Any]] = Field(default_factory=list)
	sample_size: int = 0

class DQProfileResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	dataset_id: str
	row_count: int
	column_profiles: list[dict[str, Any]]
	sample_size: int
	profiled_at: str

class DQRunResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	dataset_id: str
	rules_evaluated: int
	passed: int
	failed: int
	warnings: int
	overall_score: float
	status: str
	results: list[dict[str, Any]]
	run_at: str

class DQReportResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	dataset_id: str
	period_start: str
	period_end: str
	runs_total: int
	avg_score: float
	trend: str  # improving | degrading | stable
	anomalies: list[dict[str, Any]]
	generated_at: str

class DQAuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	entity_id: str
	entity_type: str
	payload: dict[str, Any] = Field(default_factory=dict)
	created_at: str

class DQFilter(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	dataset_id: str | None = None
	rule_type: str | None = None
	severity: str | None = None
	active: bool | None = None
