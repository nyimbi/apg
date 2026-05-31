"""APG Workflow Orchestration (WFLO) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract

__version__ = "1.0.0"
__capability_id__ = "wflo"
__capability_name__ = "Workflow Orchestration"
__apg_dependencies__ = ["mqeb", "auth", "audl", "aicr"]

capability_metadata: dict[str, Any] = {
	"name": "wflo",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant-aware workflow definitions, triggers, approvals, tasks, first-class provider-neutral workflow agents, compensation, Bytewax lifecycle orchestration, and execution governance",
	"category": "workflow_automation",
	"subcategory": "workflow_orchestration",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["workflow_definitions", "event_orchestration", "task_routing", "approval_flows", "execution_monitoring", "workflow_agent_composition", "compensation_controls", "bytewax_workflow_lifecycle"],
	"permissions": ["wflo:view", "wflo:design", "wflo:execute", "wflo:approve", "wflo:audit", "wflo:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register WFLO with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "wflo",
		"aliases": ["workflow", "workflow_orchestration", "process_automation"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["schd", "ntfy", "comp", "scpt", "them"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"workflow_definitions": "Design, version, publish, and retire tenant-scoped workflow definitions",
			"event_orchestration": "Bind triggers, queues, tasks, and events into executable process graphs",
			"task_routing": "Assign human and automated work with due dates, ownership, and escalation",
			"approval_flows": "Enforce policy-backed approval gates for risky workflow paths",
			"workflow_agent_composition": "Register first-class provider-neutral workflow agents for design, execution, approval, compensation, integration, lifecycle, and runtime observation",
			"compensation_controls": "Cancel, fail, and compensate executions through governed runtime state changes",
			"bytewax_workflow_lifecycle": "Validate workflow lifecycle batches through Bytewax-only processor metadata",
			"capability_rules": "Evaluate deterministic workflow-governance rules",
			"visual_theming": "Apply workflow-studio theme tokens and components"
		},
		"endpoints": {
			"definitions": "/wflo/api/v1/definitions",
			"executions": "/wflo/api/v1/executions",
			"tasks": "/wflo/api/v1/tasks",
			"approvals": "/wflo/api/v1/approvals",
			"agents": "/wflo/api/v1/agents",
			"lifecycle": "/wflo/api/v1/lifecycle",
			"events": "/wflo/api/v1/events",
			"streaming": "/wflo/api/v1/streaming"
		},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get WFLO capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = ["capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
