"""Domain models for the Workflow Orchestration capability."""

from __future__ import annotations

from .workflow_runtime import (
	WorkflowApprovalRecord,
	WorkflowAuditEventRecord,
	WorkflowDefinitionRecord,
	WorkflowEventRecord,
	WorkflowExecutionRecord,
	WorkflowStepRecord,
	WorkflowTaskRecord,
)


WfloRecord = WorkflowDefinitionRecord


__all__ = [
	"WorkflowApprovalRecord",
	"WorkflowAuditEventRecord",
	"WorkflowDefinitionRecord",
	"WorkflowEventRecord",
	"WorkflowExecutionRecord",
	"WorkflowStepRecord",
	"WorkflowTaskRecord",
	"WfloRecord",
]
