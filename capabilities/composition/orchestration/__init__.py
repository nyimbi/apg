"""APG workflow orchestration capability package."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .capability_contract import (
	ORCHESTRATION_EVENT_STREAM,
	SUPPORTED_ORCHESTRATION_AGENT_ROLES,
	SUPPORTED_ORCHESTRATION_AGENT_RUNTIMES,
	evaluate_capability_rules,
	event_stream_name,
	get_capability_contract,
	streaming_manifest,
)
from .service import WorkflowOrchestrationService


CAPABILITY_ID = "composition_orchestration"
CAPABILITY_NAME = "Workflow Orchestration"
CAPABILITY_VERSION = "2.1.0"


class WorkflowStatus(str, Enum):
	"""Workflow execution status values kept for composition compatibility."""

	DRAFT = "draft"
	VALIDATED = "validated"
	RELEASED = "released"
	RUNNING = "running"
	WAITING = "waiting"
	COMPLETED = "completed"
	FAILED = "failed"
	RETIRED = "retired"


class TaskStatus(str, Enum):
	"""Task execution status values kept for composition compatibility."""

	PENDING = "pending"
	ASSIGNED = "assigned"
	RUNNING = "running"
	COMPLETED = "completed"
	FAILED = "failed"
	SKIPPED = "skipped"
	ESCALATED = "escalated"


class TaskType(str, Enum):
	"""Task type values supported by orchestration definitions."""

	AUTOMATED = "automated"
	HUMAN = "human"
	APPROVAL = "approval"
	INTEGRATION = "integration"
	PARALLEL = "parallel"
	TERMINAL = "terminal"


@dataclass
class WorkflowTask:
	"""Small compatibility task definition."""

	id: str
	name: str
	type: str = TaskType.AUTOMATED.value
	handler: str | None = None
	depends_on: list[str] = field(default_factory=list)
	metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowDefinition:
	"""Small compatibility workflow definition."""

	id: str
	tenant_id: str
	name: str
	owner: str
	version: str
	tasks: list[WorkflowTask] = field(default_factory=list)
	start_event: str = "manual"
	terminal_state: str = "completed"


@dataclass
class WorkflowInstance:
	"""Small compatibility workflow instance."""

	id: str
	tenant_id: str
	workflow_definition_id: str
	status: WorkflowStatus = WorkflowStatus.RUNNING
	current_tasks: list[str] = field(default_factory=list)
	completed_tasks: list[str] = field(default_factory=list)


class WorkflowEngine:
	"""Compatibility facade backed by the package service."""

	def __init__(self) -> None:
		self.service = WorkflowOrchestrationService()

	def define(self, definition: WorkflowDefinition) -> dict[str, object]:
		return self.service.define_workflow(
			definition.id,
			definition.tenant_id,
			definition.name,
			definition.owner,
			definition.version,
			[
				{
					"id": task.id,
					"name": task.name,
					"type": task.type,
					"handler": task.handler,
					"depends_on": task.depends_on,
					**task.metadata,
				}
				for task in definition.tasks
			],
			definition.start_event,
			definition.terminal_state,
		)


def get_workflow_engine() -> WorkflowEngine:
	"""Return a compatibility workflow engine facade."""
	return WorkflowEngine()


WORKFLOW_TEMPLATES: dict[str, dict[str, object]] = {
	"approval_flow": {
		"name": "Approval Flow",
		"tasks": ["intake", "review", "approve"],
	},
	"integration_flow": {
		"name": "Integration Flow",
		"tasks": ["receive", "transform", "publish"],
	},
}


def register_capability() -> dict[str, object]:
	"""Return APG registration metadata for this capability."""
	contract = get_capability_contract()
	return {
		"capability": CAPABILITY_ID,
		"display_name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"provides": contract["provides"],
		"requires": contract["requires"],
		"ui": contract["ui"],
		"theme": contract["theme"],
		"streaming": contract["streaming"],
	}


__all__ = [
	"CAPABILITY_ID",
	"CAPABILITY_NAME",
	"CAPABILITY_VERSION",
	"ORCHESTRATION_EVENT_STREAM",
	"SUPPORTED_ORCHESTRATION_AGENT_ROLES",
	"SUPPORTED_ORCHESTRATION_AGENT_RUNTIMES",
	"TaskStatus",
	"TaskType",
	"WorkflowDefinition",
	"WorkflowEngine",
	"WorkflowInstance",
	"WorkflowOrchestrationService",
	"WorkflowStatus",
	"WorkflowTask",
	"WORKFLOW_TEMPLATES",
	"evaluate_capability_rules",
	"event_stream_name",
	"get_workflow_engine",
	"get_capability_contract",
	"register_capability",
	"streaming_manifest",
]
