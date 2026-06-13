"""Pydantic v2 models for APG Enterprise Service Bus."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

try:
	from situ_cloudevents._uuid7 import uuid7str  # type: ignore[import]
except ImportError:
	from uuid6 import uuid7  # type: ignore[import]
	def uuid7str() -> str:
		return str(uuid7())


class FlowStatus(str, Enum):
	DRAFT = "draft"
	ACTIVE = "active"
	PAUSED = "paused"
	ARCHIVED = "archived"


class FlowRunStatus(str, Enum):
	RUNNING = "running"
	COMPLETED = "completed"
	FAILED = "failed"
	RETRYING = "retrying"
	DEAD_LETTERED = "dead_lettered"


class StepType(str, Enum):
	SOURCE = "source"
	TRANSFORM = "transform"
	ROUTER = "router"
	SINK = "sink"
	DELAY = "delay"
	FILTER = "filter"


class EsbFlowStep(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	name: str
	step_type: StepType
	connector_id: str = ""
	config: dict[str, Any] = Field(default_factory=dict)
	transformation: str = ""  # JMESPath or Jinja2 template
	next_steps: list[str] = Field(default_factory=list)
	error_next_step: str | None = None


class EsbFlow(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str
	description: str = ""
	status: FlowStatus = FlowStatus.DRAFT
	trigger: dict[str, Any] = Field(default_factory=dict)  # NATS subject, webhook, schedule
	steps: list[EsbFlowStep] = Field(default_factory=list)
	retry_attempts: int = 3
	timeout_seconds: int = 30
	tags: list[str] = Field(default_factory=list)
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EsbFlowRun(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	flow_id: str
	status: FlowRunStatus = FlowRunStatus.RUNNING
	trigger_payload: dict[str, Any] = Field(default_factory=dict)
	step_results: dict[str, Any] = Field(default_factory=dict)
	error_message: str | None = None
	attempt_number: int = 1
	started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	completed_at: datetime | None = None
	duration_ms: int | None = None


class EsbDeadLetter(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	flow_id: str
	flow_run_id: str
	subject: str
	payload: dict[str, Any] = Field(default_factory=dict)
	error_message: str
	attempts: int = 0
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	resolved: bool = False
	resolved_at: datetime | None = None
