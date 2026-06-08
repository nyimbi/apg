"""APG Temporal.io capability — durable workflow execution.

Provides TemporalWorkflowAdapter implementing the WorkflowAdapter Protocol.
Drop-in replacement for NullWorkflowAdapter — same interface, crash-resilient
workflow execution backed by Temporal's durable execution engine.

Activated automatically when TEMPORAL_HOST env var is set.

Usage::

    # Auto-wired via get_workflow_adapter() factory
    adapter = get_temporal_workflow_adapter()
    result = await adapter.start_workflow("PayRunProcess", {"tenant_id": "t1", ...})
    # Workflow continues even if the process restarts
"""
from .temporal_adapter import TemporalWorkflowAdapter, get_temporal_workflow_adapter
from .apg_workflow import APGStateMachineWorkflow, APGWorkflowInput
from .apg_activities import APGActivities
from .temporal_worker import start_worker

__all__ = [
	"TemporalWorkflowAdapter",
	"get_temporal_workflow_adapter",
	"APGStateMachineWorkflow",
	"APGWorkflowInput",
	"APGActivities",
	"start_worker",
]
