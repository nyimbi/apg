"""APG Workflow Orchestration capability.

Standalone package: ``pip install apg-composition-orchestration``

Quick start::

    from apg_composition_orchestration import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : composition_orchestration
Provides      : workflow_definition_lifecycle, workflow_graph_validation, workflow_execution_lifecycle, human_task_coordination, workflow_release_governance, workflow_rule_enforcement
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-composition-orchestration"
__capability_id__ = "composition_orchestration"

from .capability_contract import (  # noqa: E402
    get_capability_contract,
    evaluate_capability_rules,
)
from .service import WorkflowOrchestrationService  # noqa: E402

# ── Backward-compatibility stubs ──────────────────────────────────────────────
# These were present in older versions of this module.  They are re-exported
# here so that code written against the previous API continues to import cleanly.

from enum import Enum
from dataclasses import dataclass, field
from typing import Any


class WorkflowStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WAITING = "waiting"


class TaskType(str, Enum):
    HUMAN = "human"
    SERVICE = "service"
    DECISION = "decision"
    PARALLEL = "parallel"
    TIMER = "timer"


@dataclass
class WorkflowTask:
    id: str
    name: str
    task_type: TaskType = TaskType.SERVICE
    status: TaskStatus = TaskStatus.PENDING
    assignee: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowDefinition:
    id: str
    name: str
    version: str = "1.0.0"
    steps: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowInstance:
    id: str
    definition_id: str
    status: WorkflowStatus = WorkflowStatus.DRAFT
    current_step: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


WORKFLOW_TEMPLATES: dict[str, WorkflowDefinition] = {}


WorkflowEngine = WorkflowOrchestrationService

def get_workflow_engine() -> WorkflowOrchestrationService:
    """Return a default workflow engine instance."""
    return WorkflowOrchestrationService(tenant_id="default")


__all__ = [
    "__version__",
    "__capability_id__",
    "get_capability_contract",
    "evaluate_capability_rules",
    "WorkflowOrchestrationService",
    "WorkflowStatus",
    "TaskStatus",
    "TaskType",
    "WorkflowTask",
    "WorkflowDefinition",
    "WorkflowInstance",
    "WORKFLOW_TEMPLATES",
    "WorkflowEngine",
    "get_workflow_engine",
]

NativeWorkflowService = WorkflowOrchestrationService
