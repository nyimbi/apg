"""Executable capability contract for APG CKM Workflow Automation."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any


SUPPORTED_WFA_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_WFA_AGENT_ROLES = [
	"process_designer",
	"approval_reviewer",
	"exception_reviewer",
	"sla_reviewer",
	"optimization_reviewer",
]
SUPPORTED_WORKFLOW_TRIGGERS = ["manual", "schedule", "event", "api", "form_submission"]
SUPPORTED_TASK_TYPES = ["human", "approval", "service", "decision", "notification", "subprocess"]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"definitions": {
		"owner_required": True,
		"version_required": True,
		"trigger_required": True,
		"approval_required_for_activation": True,
		"supported_triggers": SUPPORTED_WORKFLOW_TRIGGERS,
	},
	"instances": {
		"initiator_required": True,
		"definition_must_be_active": True,
		"context_required": True,
		"state_change_requires_audit": True,
	},
	"tasks": {
		"assignee_required_for_human_tasks": True,
		"supported_task_types": SUPPORTED_TASK_TYPES,
		"sla_required": True,
		"due_at_required_for_sla": True,
		"completion_evidence_required": True,
	},
	"approvals": {
		"approval_trace_required": True,
		"independent_reviewer_required": True,
		"decision_reason_required": True,
		"rejection_reason_required": True,
	},
	"exceptions": {
		"exception_owner_required": True,
		"escalation_policy_required": True,
		"sla_breach_requires_review": True,
	},
	"wfa_agents": {
		"agent_assist_enabled": True,
		"agent_registration_required": True,
		"agent_runtime_required": True,
		"agent_scope_required": True,
		"agent_contribution_disclosure_required": True,
		"supported_runtimes": SUPPORTED_WFA_AGENT_RUNTIMES,
		"allowed_roles": SUPPORTED_WFA_AGENT_ROLES,
	},
	"governance": {
		"audit_workflow_events": True,
		"decision_trace_required": True,
		"batch_event_stream": "bytewax",
	},
	"observability": {
		"audit_required": True,
		"trace_required": True,
		"workflow_metrics_required": True,
		"agent_activity_required": True,
		"event_stream": "bytewax",
	},
	"adapters": {
		"generated_app_runtime": "lifecycle.WfaLifecycleService",
		"api_helpers": "api.py",
		"view_models": "views.py",
		"event_stream": "bytewax",
		"audit_sink": "audl",
		"identity": "auth",
		"notification": "ckm_not",
		"collaboration": "ckm_rtc",
		"configuration": "conf",
		"scheduler": "schd",
		"monitoring": "moni",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_designer": True,
		"enable_definitions": True,
		"enable_instances": True,
		"enable_task_queue": True,
		"enable_approvals": True,
		"enable_exceptions": True,
		"enable_agent_panel": True,
		"enable_rules": True,
		"enable_audit": True,
		"enable_analytics": True,
	},
	"theme": {
		"default_theme": "ckm_wfa_workflow_ops",
		"allow_tenant_overrides": True,
	},
}


CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"definitions",
		"instances",
		"tasks",
		"approvals",
		"exceptions",
		"wfa_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {
		key: {"type": "object"}
		for key in [
			"definitions",
			"instances",
			"tasks",
			"approvals",
			"exceptions",
			"wfa_agents",
			"governance",
			"observability",
			"adapters",
			"ui",
			"theme",
		]
	}
	| {"tenant_id": {"type": "string", "minLength": 1}},
}


RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "WFA operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "definition_requires_owner", "description": "Workflow definitions require an accountable owner.", "condition": {"operation": "create_definition", "owner_present": False}, "effect": {"decision": "deny", "reason": "workflow_owner_required", "required_action": "assign_workflow_owner"}},
	{"name": "definition_requires_version", "description": "Workflow definitions require version metadata.", "condition": {"operation": "create_definition", "version_present": False}, "effect": {"decision": "deny", "reason": "workflow_version_required", "required_action": "set_workflow_version"}},
	{"name": "activation_requires_approval", "description": "Workflow activation requires approval evidence.", "condition": {"operation": "activate_definition", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "workflow_activation_approval_required", "required_action": "record_workflow_approval"}},
	{"name": "instance_requires_active_definition", "description": "Workflow instances require an active definition.", "condition": {"operation": "start_instance", "definition_active": False}, "effect": {"decision": "deny", "reason": "active_definition_required", "required_action": "activate_workflow_definition"}},
	{"name": "instance_requires_initiator", "description": "Workflow instances require an initiator.", "condition": {"operation": "start_instance", "initiator_present": False}, "effect": {"decision": "deny", "reason": "workflow_initiator_required", "required_action": "attach_workflow_initiator"}},
	{"name": "human_task_requires_assignee", "description": "Human workflow tasks require an assignee.", "condition": {"operation": "create_task", "task_type": "human", "assignee_present": False}, "effect": {"decision": "deny", "reason": "task_assignee_required", "required_action": "assign_task"}},
	{"name": "sla_task_requires_due_at", "description": "SLA-tracked workflow tasks require a due time.", "condition": {"operation": "create_task", "sla_tracked": True, "due_at_present": False}, "effect": {"decision": "deny", "reason": "task_due_at_required", "required_action": "set_task_due_at"}},
	{"name": "task_completion_requires_evidence", "description": "Task completion requires completion evidence.", "condition": {"operation": "complete_task", "completion_evidence_present": False}, "effect": {"decision": "deny", "reason": "task_completion_evidence_required", "required_action": "attach_completion_evidence"}},
	{"name": "approval_requires_independent_reviewer", "description": "Workflow approvals require an independent reviewer.", "condition": {"operation": "record_approval", "reviewer_same_as_requester": True}, "effect": {"decision": "deny", "reason": "independent_reviewer_required", "required_action": "route_to_independent_reviewer"}},
	{"name": "approval_requires_decision_reason", "description": "Workflow approval decisions require a reason.", "condition": {"operation": "record_approval", "decision_reason_present": False}, "effect": {"decision": "deny", "reason": "approval_decision_reason_required", "required_action": "record_approval_decision_reason"}},
	{"name": "rejection_requires_reason", "description": "Rejected workflow approvals require a reason.", "condition": {"operation": "record_approval", "decision": "rejected", "decision_reason_present": False}, "effect": {"decision": "deny", "reason": "rejection_reason_required", "required_action": "record_rejection_reason"}},
	{"name": "sla_breach_requires_review", "description": "SLA breaches require review.", "condition": {"operation": "escalate_task", "sla_breached": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "sla_breach_review_required", "required_action": "record_sla_review"}},
	{"name": "exception_requires_owner", "description": "Workflow exceptions require an owner.", "condition": {"operation": "raise_exception", "exception_owner_present": False}, "effect": {"decision": "deny", "reason": "exception_owner_required", "required_action": "assign_exception_owner"}},
	{"name": "wfa_agent_requires_registration", "description": "AI workflow agents must be registered.", "condition": {"wfa_agent_present": True, "agent_registered": False}, "effect": {"decision": "deny", "reason": "wfa_agent_registration_required", "required_action": "register_wfa_agent"}},
	{"name": "wfa_agent_runtime_supported", "description": "AI workflow agents must use a supported runtime.", "condition": {"wfa_agent_present": True, "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "wfa_agent_runtime_not_supported", "required_action": "choose_supported_wfa_agent_runtime"}},
	{"name": "wfa_agent_role_supported", "description": "AI workflow agents must use a supported role.", "condition": {"wfa_agent_present": True, "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "wfa_agent_role_not_supported", "required_action": "choose_supported_wfa_agent_role"}},
	{"name": "wfa_agent_requires_scope", "description": "AI workflow agents require explicit scope.", "condition": {"wfa_agent_present": True, "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "wfa_agent_scope_required", "required_action": "set_wfa_agent_scope"}},
	{"name": "wfa_agent_requires_disclosure", "description": "AI workflow-agent contributions require disclosure.", "condition": {"wfa_agent_present": True, "agent_contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "wfa_agent_disclosure_required", "required_action": "disclose_wfa_agent"}},
	{"name": "workflow_state_change_requires_audit", "description": "Workflow lifecycle state changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "workflow_audit_event_required", "required_action": "record_workflow_audit_event"}},
	{"name": "batch_workflow_mutation_requires_bytewax", "description": "Batch workflow mutations must use Bytewax event streams.", "condition": {"requested_operation": "batch_workflow_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
]


UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/ckm-wfa/dashboard", "component": "WfaDashboard", "permission": "ckm_wfa:view", "nav_group": "Overview"},
	{"name": "designer", "path": "/ckm-wfa/designer", "component": "WorkflowDesigner", "permission": "ckm_wfa:design", "nav_group": "Design"},
	{"name": "definitions", "path": "/ckm-wfa/definitions", "component": "WorkflowDefinitions", "permission": "ckm_wfa:design", "nav_group": "Design"},
	{"name": "instances", "path": "/ckm-wfa/instances", "component": "WorkflowInstances", "permission": "ckm_wfa:operate", "nav_group": "Operations"},
	{"name": "tasks", "path": "/ckm-wfa/tasks", "component": "WorkflowTaskQueue", "permission": "ckm_wfa:participate", "nav_group": "Operations"},
	{"name": "approvals", "path": "/ckm-wfa/approvals", "component": "WorkflowApprovals", "permission": "ckm_wfa:approve", "nav_group": "Governance"},
	{"name": "exceptions", "path": "/ckm-wfa/exceptions", "component": "WorkflowExceptions", "permission": "ckm_wfa:operate", "nav_group": "Operations"},
	{"name": "agents", "path": "/ckm-wfa/agents", "component": "WfaAgentPanel", "permission": "ckm_wfa:govern", "nav_group": "Governance"},
	{"name": "rules", "path": "/ckm-wfa/rules", "component": "WfaRules", "permission": "ckm_wfa:govern", "nav_group": "Governance"},
	{"name": "analytics", "path": "/ckm-wfa/analytics", "component": "WorkflowAnalytics", "permission": "ckm_wfa:view", "nav_group": "Insights"},
	{"name": "audit", "path": "/ckm-wfa/audit", "component": "WorkflowAudit", "permission": "ckm_wfa:view", "nav_group": "Governance"},
	{"name": "settings", "path": "/ckm-wfa/settings", "component": "WfaSettings", "permission": "ckm_wfa:admin", "nav_group": "Administration"},
]


THEME: dict[str, Any] = {
	"name": "ckm_wfa_workflow_ops",
	"tokens": {
		"color.primary": "#24405A",
		"color.accent": "#2A9D8F",
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
		"workflow_designer": {"icon": "workflow", "status_indicator": "version-pill", "risk_style": "activation-band"},
		"definition_registry": {"icon": "file-cog", "status_indicator": "approval-chip"},
		"instance_console": {"icon": "play-circle", "status_indicator": "state-chip"},
		"task_queue": {"icon": "list-checks", "status_indicator": "sla-chip"},
		"approval_queue": {"icon": "clipboard-check", "status_indicator": "review-chip"},
		"exception_queue": {"icon": "triangle-alert", "status_indicator": "escalation-chip"},
		"wfa_agent_panel": {"icon": "bot", "status_indicator": "scope-chip"},
		"stream_health": {"visual": "event-lane", "status_style": "stream-chip"},
		"audit": {"visual": "event-ledger", "status_style": "decision-chip"},
	},
}


def streaming_manifest() -> dict[str, Any]:
	return {
		"processor": "bytewax",
		"topic": "apg.ckm_wfa.lifecycle",
		"state": ["definitions", "instances", "tasks", "approvals", "exceptions", "wfa_agents", "audit_events"],
		"events": [
			"workflow_definition_created",
			"workflow_definition_activated",
			"workflow_instance_started",
			"workflow_task_created",
			"workflow_task_completed",
			"workflow_task_approved",
			"workflow_task_rejected",
			"workflow_exception_raised",
			"workflow_agent_registered",
		],
		"batch_mutation_guardrail": "batch_workflow_mutation_requires_bytewax",
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "ckm_wfa",
		"display_name": "Workflow Automation",
		"version": "1.0.0",
		"provides": [
			"workflow_definitions",
			"workflow_instances",
			"task_orchestration",
			"approval_governance",
			"exception_management",
			"workflow_analytics",
			"wfa_agents",
		],
		"requires": ["auth", "conf", "audl", "ckm_not", "ckm_rtc"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/ckm-wfa/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True,
		},
		"theme": deepcopy(THEME),
		"streaming": streaming_manifest(),
	}


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
			actual = context.get(key[:-4])
			if not isinstance(actual, Number) or not actual <= expected:
				return False
		elif key.endswith("_lt"):
			actual = context.get(key[:-3])
			if not isinstance(actual, Number) or not actual < expected:
				return False
		elif key.endswith("_gt"):
			actual = context.get(key[:-3])
			if not isinstance(actual, Number) or not actual > expected:
				return False
		elif key.endswith("_gte"):
			actual = context.get(key[:-4])
			if not isinstance(actual, Number) or not actual >= expected:
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
