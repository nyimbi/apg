"""Temporal workflow capability — Flask-AppBuilder views and Pydantic schemas."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .models import (
	StartWorkflowRequest,
	SignalWorkflowRequest,
	QueryWorkflowRequest,
	CompleteTaskRequest,
	FailTaskRequest,
	ScheduleWorkflowRequest,
	WorkflowInfo,
	TaskQueueInfo,
	HealthStatus,
	WorkflowStatus,
)


class WorkflowListResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	workflows: list[dict[str, Any]] = Field(default_factory=list)
	total: int = 0
	has_more: bool = False


class WorkflowCountResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	count: int
	status: str


class ScheduleListResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	schedules: list[dict[str, Any]] = Field(default_factory=list)
	total: int = 0


class NamespaceInfo(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	namespace: str
	state: str = "REGISTERED"
	description: str = ""


class SystemInfo(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	host: str
	namespace: str
	version: str


__all__ = [
	"StartWorkflowRequest",
	"SignalWorkflowRequest",
	"QueryWorkflowRequest",
	"CompleteTaskRequest",
	"FailTaskRequest",
	"ScheduleWorkflowRequest",
	"WorkflowInfo",
	"TaskQueueInfo",
	"HealthStatus",
	"WorkflowStatus",
	"WorkflowListResponse",
	"WorkflowCountResponse",
	"ScheduleListResponse",
	"NamespaceInfo",
	"SystemInfo",
]
