"""Executable capability contract for APG Workflow Orchestration."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"definitions": {
		"workflow_owner_required": True,
		"versioning_enabled": True,
		"publish_approval_required": True,
		"max_steps_per_workflow": 250
	},
	"execution": {
		"event_bus_required": True,
		"max_runtime_minutes": 1440,
		"retry_policy_required": True,
		"compensation_supported": True
	},
	"approvals": {
		"human_approval_for_high_risk": True,
		"delegation_supported": True,
		"approval_audit_required": True,
		"timeout_escalation_enabled": True
	},
	"governance": {
		"require_tenant_context": True,
		"audit_execution": True,
		"ai_step_policy_required": True,
		"external_trigger_policy_required": True
	},
	"ui": {
		"enable_workflow_studio": True,
		"enable_execution_monitor": True,
		"enable_task_inbox": True,
		"enable_approval_center": True
	},
	"theme": {
		"default_theme": "wflo_workflow_studio",
		"allow_tenant_overrides": True
	}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "definitions", "execution", "approvals", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["definitions", "execution", "approvals", "governance", "ui", "theme"]} | {
		"tenant_id": {"type": "string", "minLength": 1}
	}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All workflow operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "workflow_requires_owner", "description": "Workflow definitions require an accountable owner.", "condition": {"operation": "create_workflow", "workflow_owner_assigned": False}, "effect": {"decision": "deny", "reason": "workflow_owner_required", "required_action": "assign_workflow_owner"}},
	{"name": "publish_requires_approval", "description": "Workflow publication requires approval.", "condition": {"operation": "publish_workflow", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "workflow_publish_approval_required", "required_action": "record_publication_approval"}},
	{"name": "external_trigger_requires_policy", "description": "External triggers require a tenant trigger policy.", "condition": {"external_trigger": True, "trigger_policy_attached": False}, "effect": {"decision": "deny", "reason": "external_trigger_policy_required", "required_action": "attach_trigger_policy"}},
	{"name": "ai_step_requires_policy", "description": "AI workflow steps require an AI execution policy.", "condition": {"ai_step_present": True, "ai_policy_attached": False}, "effect": {"decision": "deny", "reason": "ai_step_policy_required", "required_action": "attach_ai_policy"}},
	{"name": "long_running_execution_requires_review", "description": "Long-running workflow executions require review.", "condition": {"expected_runtime_minutes_gt": 1440, "runtime_review_recorded": False}, "effect": {"decision": "require_review", "reason": "long_running_execution_review_required", "required_action": "review_runtime"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/wflo/dashboard", "component": "WFLODashboard", "permission": "wflo:view", "nav_group": "Overview"},
	{"name": "designer", "path": "/wflo/designer", "component": "WorkflowStudio", "permission": "wflo:design", "nav_group": "Design"},
	{"name": "definitions", "path": "/wflo/definitions", "component": "DefinitionLibrary", "permission": "wflo:design", "nav_group": "Design"},
	{"name": "executions", "path": "/wflo/executions", "component": "ExecutionMonitor", "permission": "wflo:view", "nav_group": "Runtime"},
	{"name": "tasks", "path": "/wflo/tasks", "component": "TaskInbox", "permission": "wflo:execute", "nav_group": "Runtime"},
	{"name": "approvals", "path": "/wflo/approvals", "component": "ApprovalCenter", "permission": "wflo:approve", "nav_group": "Governance"},
	{"name": "analytics", "path": "/wflo/analytics", "component": "WorkflowAnalytics", "permission": "wflo:view", "nav_group": "Operations"},
	{"name": "settings", "path": "/wflo/settings", "component": "WFLOSettings", "permission": "wflo:admin", "nav_group": "Administration"}
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
		"density": "compact"
	},
	"components": {
		"workflow_canvas": {"icon": "workflow", "status_indicator": "definition-pill", "risk_style": "approval-band"},
		"execution_timeline": {"visual": "step-timeline", "highlight": "runtime-chip"},
		"task_inbox": {"visual": "assignment-list", "status_style": "sla-chip"},
		"approval_center": {"visual": "approval-queue", "status_style": "decision-chip"}
	}
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
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/wflo/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True
		},
		"theme": deepcopy(THEME)
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
		if key.endswith("_lt"):
			if not context.get(key[:-3], 0) < expected:
				return False
		elif key.endswith("_gt"):
			if not context.get(key[:-3], 0) > expected:
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
