"""Executable capability contract for APG Deployment Management."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any

SUPPORTED_DEPLOYMENT_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_DEPLOYMENT_AGENT_ROLES = ["release_planner", "rollout_operator", "health_reviewer", "rollback_coordinator", "incident_reviewer"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"releases": {"release_owner_required": True, "manifest_required": True, "approval_required": True, "artifact_signature_required": True},
	"rollouts": {"supported_strategies": ["rolling", "blue_green", "canary"], "health_gate_required": True, "rollback_plan_required": True, "max_canary_percent": 25},
	"evidence": {"log_trace_link_required": True, "health_report_required": True, "deployment_audit_required": True, "change_ticket_required": True},
	"deployment_agents": {
		"agent_assist_enabled": True,
		"agent_registration_required": True,
		"agent_runtime_required": True,
		"agent_scope_required": True,
		"agent_contribution_disclosure_required": True,
		"supported_runtimes": SUPPORTED_DEPLOYMENT_AGENT_RUNTIMES,
		"allowed_roles": SUPPORTED_DEPLOYMENT_AGENT_ROLES,
	},
	"governance": {"require_tenant_context": True, "environment_policy_required": True, "production_approval_required": True, "separation_of_duties_required": True, "tenant_isolation_required": True, "state_change_reason_required": True, "batch_event_stream": "bytewax"},
	"observability": {"audit_required": True, "trace_required": True, "health_metrics_required": True, "agent_activity_required": True, "event_stream": "bytewax"},
	"adapters": {"generated_app_runtime": "service.DeplService", "api_helpers": "api.py", "view_models": "views.py", "event_stream": "bytewax", "ci_cd": "cicd", "environment": "envm", "logs": "logt", "monitoring": "moni", "notifications": "ntfy", "audit_sink": "audl", "composition": "comp"},
	"ui": {"enable_release_console": True, "enable_rollout_monitor": True, "enable_health_gate_view": True, "enable_rollback_center": True, "enable_agent_panel": True, "enable_audit": True, "enable_analytics": True},
	"theme": {"default_theme": "depl_release_ops", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {"type": "object", "required": ["tenant_id", "releases", "rollouts", "evidence", "deployment_agents", "governance", "observability", "adapters", "ui", "theme"], "properties": {key: {"type": "object"} for key in ["releases", "rollouts", "evidence", "deployment_agents", "governance", "observability", "adapters", "ui", "theme"]} | {"tenant_id": {"type": "string", "minLength": 1}}}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All deployment operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "release_requires_owner", "description": "Releases require an accountable owner.", "condition": {"operation": "create_release", "release_owner_assigned": False}, "effect": {"decision": "deny", "reason": "release_owner_required", "required_action": "assign_release_owner"}},
	{"name": "release_requires_manifest", "description": "Releases require manifest evidence.", "condition": {"operation": "create_release", "manifest_attached": False}, "effect": {"decision": "deny", "reason": "manifest_required", "required_action": "attach_release_manifest"}},
	{"name": "release_requires_signature", "description": "Release artifacts require signature evidence.", "condition": {"operation": "create_release", "artifact_signature_attached": False}, "effect": {"decision": "deny", "reason": "artifact_signature_required", "required_action": "attach_artifact_signature"}},
	{"name": "release_requires_change_ticket", "description": "Releases require change-ticket evidence.", "condition": {"operation": "create_release", "change_ticket_attached": False}, "effect": {"decision": "deny", "reason": "change_ticket_required", "required_action": "attach_change_ticket"}},
	{"name": "health_gate_requires_checks", "description": "Health gates require at least one check.", "condition": {"operation": "record_health_gate", "check_count_lte": 0}, "effect": {"decision": "deny", "reason": "health_checks_required", "required_action": "record_health_checks"}},
	{"name": "deployment_requires_health_gate", "description": "Deployments require a passing health gate.", "condition": {"operation": "deploy", "health_gate_passed": False}, "effect": {"decision": "deny", "reason": "health_gate_required", "required_action": "pass_health_gate"}},
	{"name": "production_requires_approval", "description": "Production deployment requires approval.", "condition": {"target_environment": "production", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "production_approval_required", "required_action": "record_production_approval"}},
	{"name": "rollback_requires_plan", "description": "Deployments require rollback plans.", "condition": {"operation": "deploy", "rollback_plan_attached": False}, "effect": {"decision": "deny", "reason": "rollback_plan_required", "required_action": "attach_rollback_plan"}},
	{"name": "large_canary_requires_review", "description": "Large canary deployments require review.", "condition": {"canary_percent_gt": 25, "canary_review_recorded": False}, "effect": {"decision": "require_review", "reason": "canary_review_required", "required_action": "review_canary_scope"}},
	{"name": "deployment_requires_trace", "description": "Deployment runs require log and trace evidence.", "condition": {"operation": "deploy", "log_trace_captured": False}, "effect": {"decision": "deny", "reason": "log_trace_link_required", "required_action": "attach_log_trace_link"}},
	{"name": "deployment_agent_requires_registration", "description": "AI deployment agents must be registered.", "condition": {"deployment_agent_present": True, "agent_registered": False}, "effect": {"decision": "deny", "reason": "deployment_agent_registration_required", "required_action": "register_deployment_agent"}},
	{"name": "deployment_agent_runtime_supported", "description": "AI deployment agents must use a supported runtime.", "condition": {"deployment_agent_present": True, "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "deployment_agent_runtime_not_supported", "required_action": "choose_supported_deployment_agent_runtime"}},
	{"name": "deployment_agent_role_supported", "description": "AI deployment agents must use a supported role.", "condition": {"deployment_agent_present": True, "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "deployment_agent_role_not_supported", "required_action": "choose_supported_deployment_agent_role"}},
	{"name": "deployment_agent_requires_scope", "description": "AI deployment agents require explicit scope.", "condition": {"deployment_agent_present": True, "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "deployment_agent_scope_required", "required_action": "set_deployment_agent_scope"}},
	{"name": "deployment_agent_requires_disclosure", "description": "AI deployment-agent contributions require disclosure.", "condition": {"deployment_agent_present": True, "agent_contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "deployment_agent_disclosure_required", "required_action": "disclose_deployment_agent"}},
	{"name": "depl_state_change_requires_reason", "description": "Deployment lifecycle state changes require a reason.", "condition": {"state_change_requested": True, "state_change_reason_present": False}, "effect": {"decision": "deny", "reason": "depl_state_change_reason_required", "required_action": "record_state_change_reason"}},
	{"name": "depl_state_change_requires_audit", "description": "Deployment lifecycle state changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "depl_audit_event_required", "required_action": "record_deployment_audit_event"}},
	{"name": "cross_tenant_deployment_access_denied", "description": "Deployment records may not cross tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_deployment_access_denied", "required_action": "use_tenant_local_context"}},
	{"name": "batch_deployment_mutation_requires_bytewax", "description": "Batch deployment mutations must use Bytewax event streams.", "condition": {"operation": "batch_deployment_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/depl/dashboard", "component": "DEPLDashboard", "permission": "depl:view", "nav_group": "Overview"},
	{"name": "releases", "path": "/depl/releases", "component": "ReleaseConsole", "permission": "depl:plan", "nav_group": "Releases"},
	{"name": "deployments", "path": "/depl/deployments", "component": "DeploymentMonitor", "permission": "depl:deploy", "nav_group": "Runtime"},
	{"name": "rollouts", "path": "/depl/rollouts", "component": "RolloutStrategies", "permission": "depl:deploy", "nav_group": "Runtime"},
	{"name": "health", "path": "/depl/health", "component": "HealthGates", "permission": "depl:view", "nav_group": "Quality"},
	{"name": "rollback", "path": "/depl/rollback", "component": "RollbackCenter", "permission": "depl:rollback", "nav_group": "Recovery"},
	{"name": "agents", "path": "/depl/agents", "component": "DeploymentAgentPanel", "permission": "depl:deploy", "nav_group": "Agents"},
	{"name": "evidence", "path": "/depl/evidence", "component": "DeploymentEvidence", "permission": "depl:view", "nav_group": "Governance"},
	{"name": "audit", "path": "/depl/audit", "component": "DeploymentAuditTrail", "permission": "depl:audit", "nav_group": "Governance"},
	{"name": "analytics", "path": "/depl/analytics", "component": "DeploymentAnalytics", "permission": "depl:view", "nav_group": "Operations"},
	{"name": "settings", "path": "/depl/settings", "component": "DEPLSettings", "permission": "depl:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {"name": "depl_release_ops", "tokens": {"color.primary": "#2C5282", "color.accent": "#38A169", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"}, "components": {"release_board": {"icon": "rocket", "status_indicator": "release-pill", "risk_style": "approval-band"}, "rollout_monitor": {"visual": "progress-lanes", "highlight": "canary-chip"}, "health_gate": {"visual": "gate-checklist", "status_style": "health-chip"}, "rollback_center": {"visual": "recovery-timeline", "status_style": "rollback-chip"}, "agent_panel": {"icon": "bot", "status_style": "scope-chip"}, "audit_timeline": {"icon": "list-checks", "status_style": "governance-chip"}}}

STREAMING: dict[str, Any] = {
	"processor": "bytewax",
	"topic": "apg.depl.lifecycle",
	"state": ["environments", "releases", "rollback_plans", "health_gates", "deployment_plans", "deployment_runs", "rollback_events", "deployment_agents", "audit_events"],
	"events": ["environment_registered", "release_created", "rollback_plan_attached", "health_gate_recorded", "deployment_plan_created", "deployment_review_approved", "deployment_plan_state_changed", "deployment_executed", "rollback_executed", "deployment_agent_registered"],
	"batch_mutation_guardrail": "batch_deployment_mutation_requires_bytewax",
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "depl", "display_name": "Deployment Management", "provides": ["release_management", "deployment_rollouts", "health_gates", "rollback_control", "deployment_audit", "deployment_agents"], "requires": ["logt", "moni", "hlth"], "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "view_module": config["adapters"]["view_models"], "api_prefix": "/depl/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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
