"""Executable capability contract for APG AI Core Framework."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"services": {
		"registry_enabled": True,
		"service_owner_required": True,
		"endpoint_health_required": True,
		"max_services_per_tenant": 100
	},
	"inference": {
		"default_timeout_seconds": 60,
		"max_concurrent_requests": 10000,
		"model_policy_required": True,
		"prompt_audit_enabled": True
	},
	"orchestration": {
		"workflow_orchestration_enabled": True,
		"multi_modal_fusion_enabled": True,
		"edge_cloud_mesh_enabled": True,
		"human_approval_for_high_risk": True
	},
	"governance": {
		"require_tenant_context": True,
		"auth_required": True,
		"monitoring_required": True,
		"ai_audit_events_required": True
	},
	"ui": {
		"enable_service_registry": True,
		"enable_inference_console": True,
		"enable_workflow_designer": True,
		"enable_governance_center": True
	},
	"theme": {
		"default_theme": "aicr_ai_control_console",
		"allow_tenant_overrides": True
	}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "services", "inference", "orchestration", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["services", "inference", "orchestration", "governance", "ui", "theme"]} | {
		"tenant_id": {"type": "string", "minLength": 1}
	}
}

RULES: list[dict[str, Any]] = [
	{
		"name": "tenant_context_required",
		"description": "All AI core operations require tenant context.",
		"condition": {"tenant_context_present": False},
		"effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}
	},
	{
		"name": "service_registration_requires_owner",
		"description": "AI service registration requires an owner.",
		"condition": {"operation": "register_service", "owner_assigned": False},
		"effect": {"decision": "deny", "reason": "service_owner_required", "required_action": "assign_service_owner"}
	},
	{
		"name": "inference_requires_model_policy",
		"description": "Inference requires an attached model policy.",
		"condition": {"operation": "run_inference", "model_policy_attached": False},
		"effect": {"decision": "deny", "reason": "model_policy_required", "required_action": "attach_model_policy"}
	},
	{
		"name": "high_risk_workflow_requires_approval",
		"description": "High-risk AI workflows require approval.",
		"condition": {"workflow_risk": "high", "approval_recorded": False},
		"effect": {"decision": "deny", "reason": "workflow_approval_required", "required_action": "record_human_approval"}
	},
	{
		"name": "unhealthy_service_blocks_routing",
		"description": "Unhealthy AI services cannot receive routed work.",
		"condition": {"service_health": "unhealthy", "routing_requested": True},
		"effect": {"decision": "deny", "reason": "service_unhealthy", "required_action": "restore_service_health"}
	},
	{
		"name": "large_context_requires_review",
		"description": "Large context windows require cost and safety review.",
		"condition": {"context_tokens_gt": 128000, "review_recorded": False},
		"effect": {"decision": "require_review", "reason": "large_context_review_required", "required_action": "record_context_review"}
	}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/aicr/dashboard", "component": "AICRDashboard", "permission": "aicr:view", "nav_group": "Overview"},
	{"name": "services", "path": "/aicr/services", "component": "AIServiceRegistry", "permission": "aicr:manage_services", "nav_group": "Services"},
	{"name": "inference", "path": "/aicr/inference", "component": "InferenceConsole", "permission": "aicr:run_inference", "nav_group": "Runtime"},
	{"name": "models", "path": "/aicr/models", "component": "ModelCatalog", "permission": "aicr:view_models", "nav_group": "Runtime"},
	{"name": "workflows", "path": "/aicr/workflows", "component": "AIWorkflowDesigner", "permission": "aicr:manage_workflows", "nav_group": "Orchestration"},
	{"name": "governance", "path": "/aicr/governance", "component": "AIGovernanceCenter", "permission": "aicr:govern", "nav_group": "Governance"},
	{"name": "metrics", "path": "/aicr/metrics", "component": "AICRMetrics", "permission": "aicr:view_metrics", "nav_group": "Operations"},
	{"name": "settings", "path": "/aicr/settings", "component": "AICRSettings", "permission": "aicr:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "aicr_ai_control_console",
	"tokens": {
		"color.primary": "#243B6B",
		"color.accent": "#7C3AED",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F6F7FB",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact"
	},
	"components": {
		"ai_service_card": {"icon": "brain-circuit", "status_indicator": "health-pill", "risk_style": "policy-band"},
		"inference_trace_panel": {"visual": "request-timeline", "highlight": "latency-chip"},
		"workflow_graph": {"visual": "directed-agent-graph", "edge_style": "handoff-line"},
		"governance_rule_stack": {"visual": "rule-ladder", "status_style": "decision-chip"}
	}
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable AICR capability contract."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "aicr",
		"display_name": "AI Core Framework",
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "dashboard.py",
			"api_prefix": "/aicr/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True
		},
		"theme": deepcopy(THEME)
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate default AICR governance rules."""
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
		if key.endswith("_gt"):
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
