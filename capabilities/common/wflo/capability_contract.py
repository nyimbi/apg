"""Executable capability contract for APG Workflow Orchestration."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"definitions": {
		"workflow_owner_required": True,
		"workflow_name_required": True,
		"versioning_enabled": True,
		"publish_approval_required": True,
		"max_steps_per_workflow": 250,
		"duplicate_step_ids_blocked": True,
		"retirement_approval_required": True,
	},
	"steps": {
		"supported_step_types": ["human", "automation", "approval", "ai", "event"],
		"step_name_required": True,
		"human_assignee_required": True,
		"automation_policy_required": True,
		"ai_policy_required": True,
		"event_policy_required": True,
		"compensation_supported": True,
	},
	"execution": {
		"event_bus_required": True,
		"event_stream": "bytewax",
		"max_runtime_minutes": 1440,
		"retry_policy_required": True,
		"compensation_supported": True,
		"correlation_id_required": True,
		"state_change_audit_required": True,
	},
	"tasks": {
		"assignee_required": True,
		"claim_required_for_completion": True,
		"due_date_supported": True,
		"escalation_reason_required": True,
	},
	"approvals": {
		"human_approval_for_high_risk": True,
		"delegation_supported": True,
		"approval_audit_required": True,
		"timeout_escalation_enabled": True,
		"decision_evidence_required": True,
		"reason_required": True,
	},
	"workflow_agents": {
		"agent_assist_enabled": True,
		"agent_registration_required": True,
		"agent_scope_required": True,
		"agent_contribution_disclosure_required": True,
		"supported_runtimes": ["codex", "claude_code", "opencode", "pi"],
		"allowed_roles": ["designer", "step_runner", "approver_assist", "compensation_planner", "runtime_observer"],
	},
	"governance": {
		"require_tenant_context": True,
		"audit_execution": True,
		"ai_step_policy_required": True,
		"external_trigger_policy_required": True,
		"tenant_isolation_required": True,
		"batch_event_stream": "bytewax",
	},
	"observability": {
		"audit_required": True,
		"execution_metrics_required": True,
		"task_metrics_required": True,
		"approval_metrics_required": True,
		"event_stream": "bytewax",
	},
	"adapters": {
		"generated_app_runtime": "service.WfloService",
		"runtime_models": "workflow_runtime.py",
		"api_helpers": "api.py",
		"view_models": "views.py",
		"event_stream": "bytewax",
		"message_bus": "mqeb",
		"identity": "auth",
		"audit_sink": "audl",
		"ai_core": "aicr",
		"scheduler": "schd",
		"notifications": "ntfy",
		"script_runtime": "scpt",
		"composition": "comp",
		"theme": "them",
	},
	"ui": {
		"enable_workflow_studio": True,
		"enable_execution_monitor": True,
		"enable_task_inbox": True,
		"enable_approval_center": True,
		"enable_agent_panel": True,
		"enable_audit": True,
		"enable_analytics": True,
	},
	"theme": {
		"default_theme": "wflo_workflow_studio",
		"allow_tenant_overrides": True,
	},
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"definitions",
		"steps",
		"execution",
		"tasks",
		"approvals",
		"workflow_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {key: {"type": "object"} for key in [
		"definitions",
		"steps",
		"execution",
		"tasks",
		"approvals",
		"workflow_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]} | {
		"tenant_id": {"type": "string", "minLength": 1}
	},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All workflow operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "workflow_requires_owner", "description": "Workflow definitions require an accountable owner.", "condition": {"operation": "create_workflow", "workflow_owner_assigned": False}, "effect": {"decision": "deny", "reason": "workflow_owner_required", "required_action": "assign_workflow_owner"}},
	{"name": "workflow_requires_name", "description": "Workflow definitions require a readable name.", "condition": {"operation": "create_workflow", "workflow_name_present": False}, "effect": {"decision": "deny", "reason": "workflow_name_required", "required_action": "name_workflow"}},
	{"name": "workflow_requires_steps", "description": "Workflow definitions require at least one step.", "condition": {"operation": "create_workflow", "step_count_lte": 0}, "effect": {"decision": "deny", "reason": "workflow_steps_required", "required_action": "add_workflow_step"}},
	{"name": "workflow_step_limit_review", "description": "Large workflow definitions require review.", "condition": {"operation": "create_workflow", "step_count_gt": 250, "workflow_size_review_recorded": False}, "effect": {"decision": "require_review", "reason": "workflow_size_review_required", "required_action": "review_workflow_size"}},
	{"name": "workflow_duplicate_step_ids_blocked", "description": "Workflow definitions may not reuse step IDs.", "condition": {"operation": "create_workflow", "duplicate_step_ids_present": True}, "effect": {"decision": "deny", "reason": "duplicate_step_ids_blocked", "required_action": "deduplicate_step_ids"}},
	{"name": "workflow_requires_retry_policy", "description": "Workflow definitions require a retry policy.", "condition": {"operation": "create_workflow", "retry_policy_attached": False}, "effect": {"decision": "deny", "reason": "retry_policy_required", "required_action": "attach_retry_policy"}},
	{"name": "publish_requires_approval", "description": "Workflow publication requires approval.", "condition": {"operation": "publish_workflow", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "workflow_publish_approval_required", "required_action": "record_publication_approval"}},
	{"name": "retire_requires_approval", "description": "Workflow retirement requires approval.", "condition": {"operation": "retire_workflow", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "workflow_retirement_approval_required", "required_action": "record_retirement_approval"}},
	{"name": "external_trigger_requires_policy", "description": "External triggers require a tenant trigger policy.", "condition": {"external_trigger": True, "trigger_policy_attached": False}, "effect": {"decision": "deny", "reason": "external_trigger_policy_required", "required_action": "attach_trigger_policy"}},
	{"name": "ai_step_requires_policy", "description": "AI workflow steps require an AI execution policy.", "condition": {"ai_step_present": True, "ai_policy_attached": False}, "effect": {"decision": "deny", "reason": "ai_step_policy_required", "required_action": "attach_ai_policy"}},
	{"name": "automation_step_requires_policy", "description": "Automation workflow steps require execution policy.", "condition": {"automation_step_present": True, "automation_policy_attached": False}, "effect": {"decision": "deny", "reason": "automation_policy_required", "required_action": "attach_automation_policy"}},
	{"name": "event_step_requires_policy", "description": "Event workflow steps require event policy.", "condition": {"event_step_present": True, "event_policy_attached": False}, "effect": {"decision": "deny", "reason": "event_policy_required", "required_action": "attach_event_policy"}},
	{"name": "long_running_execution_requires_review", "description": "Long-running workflow executions require review.", "condition": {"expected_runtime_minutes_gt": 1440, "runtime_review_recorded": False}, "effect": {"decision": "require_review", "reason": "long_running_execution_review_required", "required_action": "review_runtime"}},
	{"name": "execution_requires_published_definition", "description": "Executions require a published workflow definition.", "condition": {"operation": "start_execution", "definition_published": False}, "effect": {"decision": "deny", "reason": "workflow_not_published", "required_action": "publish_workflow"}},
	{"name": "execution_requires_correlation_id", "description": "Executions require a correlation ID.", "condition": {"operation": "start_execution", "correlation_id_present": False}, "effect": {"decision": "deny", "reason": "correlation_id_required", "required_action": "attach_correlation_id"}},
	{"name": "execution_requires_event_stream", "description": "Workflow execution events must use Bytewax streams.", "condition": {"operation": "emit_event", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
	{"name": "task_requires_assignee", "description": "Workflow tasks require an assignee.", "condition": {"operation": "create_task", "task_assignee_present": False}, "effect": {"decision": "deny", "reason": "task_assignee_required", "required_action": "assign_task"}},
	{"name": "task_completion_requires_claim", "description": "Workflow task completion requires a claim when configured.", "condition": {"operation": "complete_task", "task_claimed": False}, "effect": {"decision": "deny", "reason": "task_claim_required", "required_action": "claim_task"}},
	{"name": "task_escalation_requires_reason", "description": "Workflow task escalation requires a reason.", "condition": {"operation": "escalate_task", "escalation_reason_present": False}, "effect": {"decision": "deny", "reason": "task_escalation_reason_required", "required_action": "record_escalation_reason"}},
	{"name": "approval_requires_approver", "description": "Approval requests require an approver.", "condition": {"operation": "request_approval", "approver_present": False}, "effect": {"decision": "deny", "reason": "approval_approver_required", "required_action": "assign_approver"}},
	{"name": "approval_requires_reason", "description": "Approval requests require a reason.", "condition": {"operation": "request_approval", "approval_reason_present": False}, "effect": {"decision": "deny", "reason": "approval_reason_required", "required_action": "record_approval_reason"}},
	{"name": "approval_decision_requires_evidence", "description": "Approval decisions require evidence.", "condition": {"operation": "record_approval", "decision_evidence_present": False}, "effect": {"decision": "deny", "reason": "approval_decision_evidence_required", "required_action": "attach_decision_evidence"}},
	{"name": "approval_delegation_requires_delegate", "description": "Delegated approvals require a delegate.", "condition": {"operation": "record_approval", "approval_delegated": True, "delegate_present": False}, "effect": {"decision": "deny", "reason": "approval_delegate_required", "required_action": "select_delegate"}},
	{"name": "completion_blocks_open_tasks", "description": "Workflow completion is blocked by open tasks.", "condition": {"operation": "complete_execution", "open_tasks_present": True}, "effect": {"decision": "deny", "reason": "open_tasks_block_completion", "required_action": "complete_open_tasks"}},
	{"name": "completion_blocks_pending_approvals", "description": "Workflow completion is blocked by pending approvals.", "condition": {"operation": "complete_execution", "pending_approvals_present": True}, "effect": {"decision": "deny", "reason": "pending_approvals_block_completion", "required_action": "resolve_pending_approvals"}},
	{"name": "execution_state_change_requires_reason", "description": "Execution cancellation and failure require a reason.", "condition": {"operation": "change_execution_state", "state_change_reason_present": False}, "effect": {"decision": "deny", "reason": "execution_state_change_reason_required", "required_action": "record_state_change_reason"}},
	{"name": "compensation_requires_plan", "description": "Compensation execution requires a compensation plan.", "condition": {"operation": "run_compensation", "compensation_plan_present": False}, "effect": {"decision": "deny", "reason": "compensation_plan_required", "required_action": "attach_compensation_plan"}},
	{"name": "workflow_agent_requires_registration", "description": "AI workflow agents must be registered.", "condition": {"workflow_agent_present": True, "agent_registered": False}, "effect": {"decision": "deny", "reason": "workflow_agent_registration_required", "required_action": "register_workflow_agent"}},
	{"name": "workflow_agent_runtime_supported", "description": "AI workflow agents must use a configured runtime.", "condition": {"workflow_agent_present": True, "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "workflow_agent_runtime_not_supported", "required_action": "choose_supported_workflow_agent_runtime"}},
	{"name": "workflow_agent_requires_scope", "description": "AI workflow agents require workflow or execution scope.", "condition": {"workflow_agent_present": True, "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "workflow_agent_scope_required", "required_action": "set_workflow_agent_scope"}},
	{"name": "workflow_agent_requires_disclosure", "description": "AI workflow agent contributions require disclosure.", "condition": {"workflow_agent_present": True, "agent_contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "workflow_agent_disclosure_required", "required_action": "disclose_workflow_agent"}},
	{"name": "workflow_state_change_requires_audit", "description": "Workflow state changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "workflow_audit_event_required", "required_action": "record_workflow_audit_event"}},
	{"name": "cross_tenant_workflow_access_denied", "description": "Workflow records may not cross tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_workflow_access_denied", "required_action": "use_tenant_local_context"}},
	{"name": "batch_workflow_mutation_requires_bytewax", "description": "Batch workflow mutations must use Bytewax event streams.", "condition": {"operation": "batch_workflow_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/wflo/dashboard", "component": "WFLODashboard", "permission": "wflo:view", "nav_group": "Overview"},
	{"name": "designer", "path": "/wflo/designer", "component": "WorkflowStudio", "permission": "wflo:design", "nav_group": "Design"},
	{"name": "definitions", "path": "/wflo/definitions", "component": "DefinitionLibrary", "permission": "wflo:design", "nav_group": "Design"},
	{"name": "executions", "path": "/wflo/executions", "component": "ExecutionMonitor", "permission": "wflo:view", "nav_group": "Runtime"},
	{"name": "tasks", "path": "/wflo/tasks", "component": "TaskInbox", "permission": "wflo:execute", "nav_group": "Runtime"},
	{"name": "approvals", "path": "/wflo/approvals", "component": "ApprovalCenter", "permission": "wflo:approve", "nav_group": "Governance"},
	{"name": "agents", "path": "/wflo/agents", "component": "WorkflowAgentPanel", "permission": "wflo:execute", "nav_group": "Runtime"},
	{"name": "audit", "path": "/wflo/audit", "component": "WorkflowAuditTrail", "permission": "wflo:audit", "nav_group": "Governance"},
	{"name": "analytics", "path": "/wflo/analytics", "component": "WorkflowAnalytics", "permission": "wflo:view", "nav_group": "Operations"},
	{"name": "settings", "path": "/wflo/settings", "component": "WFLOSettings", "permission": "wflo:admin", "nav_group": "Administration"},
]

THEME: dict[str, Any] = {
	"name": "wflo_workflow_studio",
	"tokens": {
		"color.primary": "#2C5282",
		"color.accent": "#DD6B20",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F7F8FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact",
	},
	"components": {
		"workflow_canvas": {"icon": "workflow", "status_indicator": "definition-pill", "risk_style": "approval-band"},
		"step_palette": {"visual": "step-type-list", "status_style": "policy-chip"},
		"execution_timeline": {"visual": "step-timeline", "highlight": "runtime-chip"},
		"task_inbox": {"visual": "assignment-list", "status_style": "sla-chip"},
		"approval_center": {"visual": "approval-queue", "status_style": "decision-chip"},
		"agent_panel": {"visual": "agent-roster", "status_style": "scope-chip"},
		"audit_timeline": {"visual": "event-timeline", "status_style": "workflow-chip"},
	},
}

STREAMING: dict[str, Any] = {
	"processor": "bytewax",
	"topic": "apg.wflo.lifecycle",
	"state": ["workflow_definitions", "workflow_executions", "workflow_tasks", "workflow_approvals", "workflow_agents"],
	"events": [
		"workflow_created",
		"workflow_published",
		"workflow_retired",
		"workflow_started",
		"task_created",
		"task_claimed",
		"task_completed",
		"task_escalated",
		"approval_requested",
		"approval_approved",
		"approval_rejected",
		"approval_delegated",
		"workflow_completed",
		"workflow_cancelled",
		"workflow_failed",
		"compensation_completed",
		"workflow_agent_registered",
	],
	"batch_mutation_guardrail": "batch_workflow_mutation_requires_bytewax",
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable WFLO capability contract."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "wflo",
		"display_name": "Workflow Orchestration",
		"provides": ["workflow_definitions", "event_orchestration", "task_routing", "approval_flows", "execution_monitoring", "workflow_agents", "compensation_controls"],
		"requires": ["mqeb", "auth", "audl", "aicr"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": config["adapters"]["view_models"],
			"api_prefix": "/wflo/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True,
		},
		"theme": deepcopy(THEME),
		"streaming": deepcopy(STREAMING),
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate default WFLO governance rules."""
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
			actual = context.get(key[:-4])
			if not isinstance(actual, Number) or not actual <= expected:
				return False
		elif key.endswith("_lt"):
			actual = context.get(key[:-3])
			if not isinstance(actual, Number) or not actual < expected:
				return False
		elif key.endswith("_gte"):
			actual = context.get(key[:-4])
			if not isinstance(actual, Number) or not actual >= expected:
				return False
		elif key.endswith("_gt"):
			actual = context.get(key[:-3])
			if not isinstance(actual, Number) or not actual > expected:
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
