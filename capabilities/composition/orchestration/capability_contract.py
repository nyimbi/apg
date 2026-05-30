"""Executable capability contract for APG workflow orchestration."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


SUPPORTED_ORCHESTRATION_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_ORCHESTRATION_AGENT_ROLES = [
	"workflow_architect",
	"bpml_reviewer",
	"release_reviewer",
	"incident_reviewer",
	"compliance_reviewer",
	"optimization_reviewer",
]
ORCHESTRATION_EVENT_STREAM = "apg.composition.orchestration.lifecycle"


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"workflow_definitions": {
		"owner_required": True,
		"version_required": True,
		"start_event_required": True,
		"terminal_state_required": True,
		"task_graph_required": True,
		"cycle_detection_enabled": True,
	},
	"tasks": {
		"handler_required": True,
		"human_assignee_required": True,
		"approval_policy_required": True,
		"cross_capability_contract_required": True,
		"sla_escalation_required": True,
	},
	"execution": {
		"bytewax_required": True,
		"idempotency_required": True,
		"retry_limit_required": True,
		"compensation_required_for_transactions": True,
		"max_parallel_branches": 64,
	},
	"releases": {
		"validation_required": True,
		"dry_run_required": True,
		"rollback_plan_required": True,
		"approval_required_for_high_risk": True,
	},
	"automation_agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_ORCHESTRATION_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_ORCHESTRATION_AGENT_ROLES,
		"human_approval_required": True,
		"max_autonomous_scope": "recommend_validate_and_prepare",
	},
	"governance": {
		"require_tenant_context": True,
		"audit_state_changes": True,
		"policy_attached_for_writes": True,
		"privileged_workflow_changes_reviewed": True,
	},
	"observability": {
		"event_stream": ORCHESTRATION_EVENT_STREAM,
		"stream_processor": "bytewax",
		"emit_definition_events": True,
		"emit_execution_events": True,
		"emit_release_events": True,
		"emit_agent_events": True,
	},
	"adapters": {
		"authorization": "adapter",
		"audit": "adapter",
		"event_stream": "bytewax",
		"notification": "adapter",
		"registry": "adapter",
		"theme": "adapter",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_definition_library": True,
		"enable_designer": True,
		"enable_execution_console": True,
		"enable_task_console": True,
		"enable_release_console": True,
		"enable_rule_center": True,
		"enable_agent_workbench": True,
		"enable_settings": True,
	},
	"theme": {"default_theme": "composition_orchestration_control", "allow_tenant_overrides": True},
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"workflow_definitions",
		"tasks",
		"execution",
		"releases",
		"automation_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {
		"tenant_id": {"type": "string", "minLength": 1},
		"workflow_definitions": {"type": "object"},
		"tasks": {"type": "object"},
		"execution": {"type": "object"},
		"releases": {"type": "object"},
		"automation_agents": {"type": "object"},
		"governance": {"type": "object"},
		"observability": {"type": "object"},
		"adapters": {"type": "object"},
		"ui": {"type": "object"},
		"theme": {"type": "object"},
	},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "Workflow orchestration operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "workflow_write_requires_policy", "description": "Workflow definition and execution write operations require policy attachment.", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "operation_policy_required", "required_action": "attach_operation_policy"}},
	{"name": "workflow_requires_owner", "description": "Workflow definitions require an accountable owner.", "condition": {"operation": "define_workflow", "workflow_owner_assigned": False}, "effect": {"decision": "deny", "reason": "workflow_owner_required", "required_action": "assign_workflow_owner"}},
	{"name": "workflow_requires_version", "description": "Workflow definitions require a version.", "condition": {"operation": "define_workflow", "workflow_version_present": False}, "effect": {"decision": "deny", "reason": "workflow_version_required", "required_action": "set_workflow_version"}},
	{"name": "workflow_requires_start_event", "description": "Workflow definitions require at least one start event.", "condition": {"operation": "define_workflow", "start_event_present": False}, "effect": {"decision": "deny", "reason": "workflow_start_event_required", "required_action": "add_start_event"}},
	{"name": "workflow_requires_task_graph", "description": "Workflow definitions require at least one executable task.", "condition": {"operation": "define_workflow", "task_graph_present": False}, "effect": {"decision": "deny", "reason": "workflow_task_graph_required", "required_action": "add_executable_task_graph"}},
	{"name": "workflow_requires_terminal_state", "description": "Workflow definitions require a terminal state.", "condition": {"operation": "define_workflow", "terminal_state_present": False}, "effect": {"decision": "deny", "reason": "workflow_terminal_state_required", "required_action": "add_terminal_state"}},
	{"name": "task_requires_handler", "description": "Automated and integration tasks require a handler.", "condition": {"operation": "define_task", "handler_present": False}, "effect": {"decision": "deny", "reason": "task_handler_required", "required_action": "attach_task_handler"}},
	{"name": "human_task_requires_assignee", "description": "Human tasks require an assignee, group, or role.", "condition": {"operation": "define_task", "human_task": True, "assignee_present": False}, "effect": {"decision": "deny", "reason": "human_task_assignee_required", "required_action": "assign_human_task"}},
	{"name": "approval_task_requires_policy", "description": "Approval tasks require an approval policy.", "condition": {"operation": "define_task", "approval_task": True, "approval_policy_present": False}, "effect": {"decision": "deny", "reason": "approval_policy_required", "required_action": "attach_approval_policy"}},
	{"name": "cross_capability_task_requires_contract", "description": "Cross-capability tasks require an APG capability contract reference.", "condition": {"operation": "define_task", "cross_capability_task": True, "capability_contract_present": False}, "effect": {"decision": "deny", "reason": "capability_contract_required", "required_action": "attach_capability_contract"}},
	{"name": "execution_requires_bytewax_stream", "description": "Workflow execution lifecycle events must use Bytewax.", "condition": {"operation": "start_execution", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_execution_lifecycle_to_bytewax"}},
	{"name": "execution_requires_idempotency_key", "description": "Workflow execution starts require idempotency keys.", "condition": {"operation": "start_execution", "idempotency_key_present": False}, "effect": {"decision": "deny", "reason": "idempotency_key_required", "required_action": "attach_idempotency_key"}},
	{"name": "high_risk_execution_requires_review", "description": "High-risk workflow executions require review.", "condition": {"operation": "start_execution", "risk_level": "high", "review_recorded": False}, "effect": {"decision": "require_review", "reason": "high_risk_review_required", "required_action": "record_execution_review"}},
	{"name": "release_requires_validation", "description": "Workflow releases require validation evidence.", "condition": {"operation": "release_workflow", "validation_evidence_present": False}, "effect": {"decision": "deny", "reason": "release_validation_required", "required_action": "attach_validation_evidence"}},
	{"name": "release_requires_dry_run", "description": "Workflow releases require a dry-run result.", "condition": {"operation": "release_workflow", "dry_run_passed": False}, "effect": {"decision": "deny", "reason": "release_dry_run_required", "required_action": "run_release_dry_run"}},
	{"name": "release_requires_rollback_plan", "description": "Workflow releases require a rollback plan.", "condition": {"operation": "release_workflow", "rollback_plan_present": False}, "effect": {"decision": "deny", "reason": "release_rollback_plan_required", "required_action": "attach_rollback_plan"}},
	{"name": "retry_policy_requires_limit", "description": "Retry policies require bounded attempts.", "condition": {"operation": "define_task", "retry_policy_present": True, "retry_limit_present": False}, "effect": {"decision": "deny", "reason": "retry_limit_required", "required_action": "set_retry_limit"}},
	{"name": "sla_task_requires_escalation", "description": "SLA-bound tasks require escalation rules.", "condition": {"operation": "define_task", "sla_present": True, "escalation_present": False}, "effect": {"decision": "require_review", "reason": "sla_escalation_required", "required_action": "attach_escalation_rule"}},
	{"name": "compensation_required_for_transaction", "description": "Transactional workflows require compensation steps.", "condition": {"operation": "define_workflow", "transactional_workflow": True, "compensation_present": False}, "effect": {"decision": "deny", "reason": "compensation_required", "required_action": "attach_compensation_steps"}},
	{"name": "batch_schedule_requires_bytewax", "description": "Batch execution scheduling requires Bytewax coordination.", "condition": {"operation": "batch_schedule", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_batch_schedule_to_bytewax"}},
	{"name": "workflow_agent_runtime_supported", "description": "Workflow agents must use an approved runtime.", "condition": {"operation": "register_workflow_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "workflow_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"}},
	{"name": "workflow_agent_role_supported", "description": "Workflow agents must use an approved role.", "condition": {"operation": "register_workflow_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "workflow_agent_role_not_supported", "required_action": "select_supported_agent_role"}},
	{"name": "privileged_agent_workflow_action_requires_human_approval", "description": "Privileged workflow actions proposed by agents require human approval.", "condition": {"operation": "agent_workflow_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/composition-orchestration/dashboard", "component": "OrchestrationDashboard", "permission": "composition_orchestration:view", "nav_group": "Overview"},
	{"name": "definitions", "path": "/composition-orchestration/definitions", "component": "WorkflowDefinitionLibrary", "permission": "composition_orchestration:manage_definitions", "nav_group": "Definitions"},
	{"name": "designer", "path": "/composition-orchestration/designer", "component": "WorkflowDesigner", "permission": "composition_orchestration:design", "nav_group": "Design"},
	{"name": "executions", "path": "/composition-orchestration/executions", "component": "WorkflowExecutionConsole", "permission": "composition_orchestration:operate", "nav_group": "Operations"},
	{"name": "tasks", "path": "/composition-orchestration/tasks", "component": "WorkflowTaskConsole", "permission": "composition_orchestration:manage_tasks", "nav_group": "Operations"},
	{"name": "releases", "path": "/composition-orchestration/releases", "component": "WorkflowReleaseConsole", "permission": "composition_orchestration:release", "nav_group": "Release"},
	{"name": "rules", "path": "/composition-orchestration/rules", "component": "WorkflowRuleCenter", "permission": "composition_orchestration:govern", "nav_group": "Governance"},
	{"name": "agents", "path": "/composition-orchestration/agents", "component": "WorkflowAgentWorkbench", "permission": "composition_orchestration:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/composition-orchestration/settings", "component": "WorkflowSettings", "permission": "composition_orchestration:admin", "nav_group": "Administration"},
]

THEME: dict[str, Any] = {
	"name": "composition_orchestration_control",
	"tokens": {"color.primary": "#28536B", "color.accent": "#C44536", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {
		"definition_library": {"icon": "workflow", "status_indicator": "definition-pill", "risk_style": "validation-band"},
		"designer": {"visual": "graph-canvas", "status_style": "graph-chip"},
		"execution_console": {"visual": "run-lanes", "status_style": "execution-chip"},
		"task_console": {"visual": "task-queue", "status_style": "sla-chip"},
		"release_console": {"visual": "release-checklist", "status_style": "evidence-chip"},
		"rule_center": {"visual": "rule-grid", "status_style": "guardrail-chip"},
		"agent_workbench": {"visual": "review-lane", "status_style": "approval-chip"},
	},
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "composition_orchestration",
		"display_name": "Workflow Orchestration",
		"provides": [
			"workflow_definition_lifecycle",
			"workflow_graph_validation",
			"workflow_execution_lifecycle",
			"human_task_coordination",
			"workflow_release_governance",
			"workflow_rule_enforcement",
			"workflow_agents",
		],
		"requires": ["auth", "audl", "ntfy", "registry", "composition_events", "composition_config"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {"shell": "apg_python", "view_module": "views.py", "api_prefix": "/composition-orchestration/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True},
		"theme": deepcopy(THEME),
		"streaming": streaming_manifest(),
	}


def streaming_manifest() -> dict[str, Any]:
	return {
		"processor": "bytewax",
		"stream": ORCHESTRATION_EVENT_STREAM,
		"key": "tenant_id",
		"events": [
			"workflow_defined",
			"workflow_validated",
			"workflow_released",
			"workflow_execution_started",
			"workflow_execution_advanced",
			"workflow_execution_completed",
			"workflow_task_assigned",
			"workflow_agent_registered",
		],
		"states": ["draft", "validated", "released", "running", "waiting", "completed", "failed", "retired"],
		"guardrails": [
			"execution_requires_bytewax_stream",
			"batch_schedule_requires_bytewax",
			"privileged_agent_workflow_action_requires_human_approval",
		],
	}


def event_stream_name() -> str:
	return ORCHESTRATION_EVENT_STREAM


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	matched: list[str] = []
	actions: list[dict[str, Any]] = []
	decision = "allow"
	for rule in RULES:
		if _matches(rule["condition"], context):
			matched.append(rule["name"])
			effect = rule["effect"]
			actions.append(effect)
			if effect["decision"] == "deny":
				decision = "deny"
			elif effect["decision"] == "require_review" and decision != "deny":
				decision = "require_review"
	return {"decision": decision, "matched_rules": matched, "actions": actions, "context": context}


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_lte"):
			if not context.get(key[:-4], 0) <= expected:
				return False
		elif key.endswith("_lt"):
			if not context.get(key[:-3], 0) < expected:
				return False
		elif key.endswith("_gte"):
			if not context.get(key[:-4], 0) >= expected:
				return False
		elif key.endswith("_gt"):
			if not context.get(key[:-3], 0) > expected:
				return False
		elif key.endswith("_ne"):
			if context.get(key[:-3]) == expected:
				return False
		elif context.get(key) != expected:
			return False
	return True


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
	for key, value in source.items():
		if isinstance(value, dict) and isinstance(target.get(key), dict):
			_deep_merge(target[key], value)
		else:
			target[key] = value
