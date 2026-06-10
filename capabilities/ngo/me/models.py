"""M&E (Monitoring & Evaluation) — Pydantic v2 models."""
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


_cfg = ConfigDict(extra="forbid", validate_by_name=True)


class MeIndicatorCreate(BaseModel):
	model_config = _cfg
	programme_id: str
	name: str
	code: str
	description: str = ""
	indicator_type: str = "output"
	unit: str = ""
	baseline_value: float = 0.0
	baseline_date: str = ""
	target_value: float
	target_date: str
	disaggregation: list[str] = Field(default_factory=list)
	data_source: str = ""
	collection_method: str = ""


class MeIndicatorUpdate(BaseModel):
	model_config = _cfg
	name: str | None = None
	description: str | None = None
	target_value: float | None = None
	target_date: str | None = None
	status: str | None = None
	data_source: str | None = None


class MeIndicatorResponse(BaseModel):
	model_config = _cfg
	id: str = Field(default_factory=uuid7str)
	programme_id: str
	name: str
	code: str
	description: str
	indicator_type: str
	unit: str
	baseline_value: float
	baseline_date: str
	target_value: float
	target_date: str
	current_value: float
	achievement_pct: float
	disaggregation: list[str]
	data_source: str
	collection_method: str
	status: str
	tenant_id: str
	created_at: str
	updated_at: str | None = None


class MeDataCollectionCreate(BaseModel):
	model_config = _cfg
	indicator_id: str
	programme_id: str
	value: float
	collection_date: str
	collected_by: str
	period: str = ""
	disaggregation_values: dict[str, Any] = Field(default_factory=dict)
	notes: str = ""


class MeDataCollectionResponse(BaseModel):
	model_config = _cfg
	id: str = Field(default_factory=uuid7str)
	indicator_id: str
	programme_id: str
	value: float
	collection_date: str
	collected_by: str
	period: str
	disaggregation_values: dict[str, Any]
	notes: str
	verified: bool = False
	tenant_id: str
	created_at: str


class MeProgressReportCreate(BaseModel):
	model_config = _cfg
	programme_id: str
	report_period: str
	period_start: str
	period_end: str
	prepared_by: str
	narrative: str = ""
	key_achievements: list[str] = Field(default_factory=list)
	challenges: list[str] = Field(default_factory=list)
	lessons_learned: list[str] = Field(default_factory=list)


class MeProgressReportResponse(BaseModel):
	model_config = _cfg
	id: str = Field(default_factory=uuid7str)
	programme_id: str
	report_period: str
	period_start: str
	period_end: str
	prepared_by: str
	narrative: str
	key_achievements: list[str]
	challenges: list[str]
	lessons_learned: list[str]
	indicator_snapshots: list[dict[str, Any]] = Field(default_factory=list)
	status: str
	tenant_id: str
	created_at: str


class MeEvaluationCreate(BaseModel):
	model_config = _cfg
	programme_id: str
	evaluation_type: str = "mid_term"
	evaluator: str
	evaluation_date: str
	scope: str = ""
	methodology: str = ""
	findings: str = ""
	recommendations: str = ""
	rating: str = "satisfactory"


class MeEvaluationResponse(BaseModel):
	model_config = _cfg
	id: str = Field(default_factory=uuid7str)
	programme_id: str
	evaluation_type: str
	evaluator: str
	evaluation_date: str
	scope: str
	methodology: str
	findings: str
	recommendations: str
	rating: str
	status: str
	tenant_id: str
	created_at: str


class MeLearningCycleCreate(BaseModel):
	model_config = _cfg
	programme_id: str
	cycle_name: str
	start_date: str
	end_date: str
	facilitator: str
	learning_questions: list[str] = Field(default_factory=list)
	notes: str = ""


class MeLearningCycleResponse(BaseModel):
	model_config = _cfg
	id: str = Field(default_factory=uuid7str)
	programme_id: str
	cycle_name: str
	start_date: str
	end_date: str
	facilitator: str
	learning_questions: list[str]
	findings: list[str] = Field(default_factory=list)
	action_points: list[str] = Field(default_factory=list)
	notes: str
	status: str
	tenant_id: str
	created_at: str


class MeIndicatorFilter(BaseModel):
	model_config = _cfg
	programme_id: str | None = None
	indicator_type: str | None = None
	status: str | None = None


class MeAuditEvent(BaseModel):
	model_config = _cfg
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	record_id: str
	record_type: str
	details: dict[str, Any] = Field(default_factory=dict)
	emitted_at: str


__all__ = [
	"MeIndicatorCreate", "MeIndicatorUpdate", "MeIndicatorResponse",
	"MeDataCollectionCreate", "MeDataCollectionResponse",
	"MeProgressReportCreate", "MeProgressReportResponse",
	"MeEvaluationCreate", "MeEvaluationResponse",
	"MeLearningCycleCreate", "MeLearningCycleResponse",
	"MeIndicatorFilter", "MeAuditEvent",
]
