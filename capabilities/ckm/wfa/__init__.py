"""APG CKM Workflow Automation capability package."""

from __future__ import annotations

from .capability_contract import (
	SUPPORTED_TASK_TYPES,
	SUPPORTED_WFA_AGENT_ROLES,
	SUPPORTED_WFA_AGENT_RUNTIMES,
	SUPPORTED_WORKFLOW_TRIGGERS,
	evaluate_capability_rules,
	get_capability_contract,
	streaming_manifest,
)
from .lifecycle import (
	WfaAgent,
	WfaApproval,
	WfaLifecycleService,
	WfaProcess,
	WfaProcessInstance,
	WfaTask,
)


__version__ = "1.0.0"
__author__ = "Datacraft"

APG_CAPABILITY_INFO = {
	"id": "ckm_wfa",
	"name": "Workflow Automation",
	"version": __version__,
	"description": "Tenant-scoped workflow definitions, instances, tasks, approvals, exceptions, analytics, and AI-agent guardrails for generated APG applications.",
	"category": "ckm",
	"provides": get_capability_contract()["provides"],
	"requires": get_capability_contract()["requires"],
	"supported_triggers": SUPPORTED_WORKFLOW_TRIGGERS,
	"supported_task_types": SUPPORTED_TASK_TYPES,
	"supported_agent_runtimes": SUPPORTED_WFA_AGENT_RUNTIMES,
	"streaming": streaming_manifest(),
}

__all__ = [
	"APG_CAPABILITY_INFO",
	"SUPPORTED_TASK_TYPES",
	"SUPPORTED_WFA_AGENT_ROLES",
	"SUPPORTED_WFA_AGENT_RUNTIMES",
	"SUPPORTED_WORKFLOW_TRIGGERS",
	"WfaAgent",
	"WfaApproval",
	"WfaLifecycleService",
	"WfaProcess",
	"WfaProcessInstance",
	"WfaTask",
	"evaluate_capability_rules",
	"get_capability_contract",
	"streaming_manifest",
]
