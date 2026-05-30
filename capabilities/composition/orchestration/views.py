"""View models for APG workflow orchestration screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_ORCHESTRATION_AGENT_ROLES,
		SUPPORTED_ORCHESTRATION_AGENT_RUNTIMES,
		get_capability_contract,
	)
	from .service import WorkflowOrchestrationService
except ImportError:
	from capability_contract import (
		SUPPORTED_ORCHESTRATION_AGENT_ROLES,
		SUPPORTED_ORCHESTRATION_AGENT_RUNTIMES,
		get_capability_contract,
	)
	from service import WorkflowOrchestrationService


def navigation_model(tenant_id: str = "default") -> dict[str, Any]:
	"""Return route and theme metadata for generated navigation."""
	contract = get_capability_contract(tenant_id)
	return {
		"capability": contract["capability"],
		"routes": contract["ui"]["routes"],
		"theme": contract["theme"],
		"api_prefix": contract["ui"]["api_prefix"],
	}


def dashboard_model(service: WorkflowOrchestrationService, tenant_id: str = "default") -> dict[str, Any]:
	"""Return the dashboard screen model."""
	return {
		"screen": "dashboard",
		"title": "Workflow Orchestration",
		"summary": service.dashboard_summary(tenant_id),
		"sections": ["definition_health", "execution_health", "human_tasks", "agent_activity"],
	}


def definition_library_model(service: WorkflowOrchestrationService, tenant_id: str = "default") -> dict[str, Any]:
	"""Return the workflow definition library model."""
	return {
		"screen": "definitions",
		"records": service.list_workflow_definitions(tenant_id),
		"columns": ["workflow_id", "name", "version", "owner", "status", "updated_at"],
		"actions": ["define_workflow", "validate_graph", "prepare_release"],
	}


def designer_model(tenant_id: str = "default") -> dict[str, Any]:
	"""Return workflow designer metadata."""
	return {
		"screen": "designer",
		"tenant_id": tenant_id,
		"node_types": ["start", "automated", "human", "approval", "integration", "parallel", "terminal"],
		"edge_rules": ["dependency", "branch", "join", "compensation"],
		"validation": ["cycle_detection", "handler_presence", "owner_presence", "terminal_state"],
	}


def execution_console_model(service: WorkflowOrchestrationService, tenant_id: str = "default") -> dict[str, Any]:
	"""Return workflow execution console state."""
	return {
		"screen": "executions",
		"records": service.list_executions(tenant_id),
		"columns": ["execution_id", "workflow_definition_id", "status", "current_tasks", "updated_at"],
		"actions": ["start_execution", "complete_task", "assign_human_task"],
	}


def task_console_model(service: WorkflowOrchestrationService, tenant_id: str = "default") -> dict[str, Any]:
	"""Return human task assignment state."""
	return {
		"screen": "tasks",
		"records": service.list_task_assignments(tenant_id),
		"columns": ["task_id", "execution_record_id", "assignee", "due_at", "status"],
		"actions": ["assign", "complete", "escalate"],
	}


def release_console_model(service: WorkflowOrchestrationService, tenant_id: str = "default") -> dict[str, Any]:
	"""Return workflow release governance state."""
	return {
		"screen": "releases",
		"records": service.list_releases(tenant_id),
		"columns": ["release_id", "workflow_definition_id", "dry_run_passed", "approved_by", "status"],
		"actions": ["validate", "dry_run", "release", "rollback"],
	}


def rule_center_model(tenant_id: str = "default") -> dict[str, Any]:
	"""Return deterministic rule metadata."""
	contract = get_capability_contract(tenant_id)
	return {
		"screen": "rules",
		"rules": contract["rule_engine"]["rules"],
		"guardrails": contract["streaming"]["guardrails"],
	}


def agent_workbench_model(service: WorkflowOrchestrationService, tenant_id: str = "default") -> dict[str, Any]:
	"""Return workflow agent workbench state."""
	return {
		"screen": "agents",
		"records": service.list_workflow_agents(tenant_id),
		"supported_runtimes": SUPPORTED_ORCHESTRATION_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_ORCHESTRATION_AGENT_ROLES,
		"actions": ["register_agent", "validate_action", "record_human_approval"],
	}
