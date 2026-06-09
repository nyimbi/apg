"""Temporal workflow capability — Pydantic v2 models."""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class WorkflowStatus(str, Enum):
	running = "RUNNING"
	completed = "COMPLETED"
	failed = "FAILED"
	cancelled = "CANCELLED"
	terminated = "TERMINATED"
	timed_out = "TIMED_OUT"
	not_found = "NOT_FOUND"


class StartWorkflowRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	workflow_type: str = Field(..., description="Registered workflow type name")
	workflow_id: str | None = Field(default=None, description="Optional deterministic ID")
	input_data: dict[str, Any] = Field(default_factory=dict)
	task_queue: str = Field(default="apg-workflows")
	execution_timeout_seconds: int = Field(default=3600, ge=1)


class SignalWorkflowRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	workflow_id: str
	signal_name: str
	payload: dict[str, Any] = Field(default_factory=dict)


class QueryWorkflowRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	workflow_id: str
	query_type: str
	args: dict[str, Any] = Field(default_factory=dict)


class CompleteTaskRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	task_token: str
	result: dict[str, Any] = Field(default_factory=dict)


class FailTaskRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	task_token: str
	error: str


class ScheduleWorkflowRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	schedule_id: str
	workflow_type: str
	cron_expression: str = Field(..., description="Standard cron expression (5 or 6 fields)")
	input_data: dict[str, Any] = Field(default_factory=dict)


class WorkflowInfo(BaseModel):
	model_config = ConfigDict(extra="ignore", validate_by_name=True)

	workflow_id: str
	workflow_type: str = ""
	status: WorkflowStatus = WorkflowStatus.not_found
	started_at: str | None = None
	closed_at: str | None = None
	input_data: dict[str, Any] = Field(default_factory=dict)
	result: dict[str, Any] | None = None


class TaskQueueInfo(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	task_queue: str
	pollers: int = 0
	backlog_count: int = 0


class HealthStatus(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	status: str
	host: str
	namespace: str
