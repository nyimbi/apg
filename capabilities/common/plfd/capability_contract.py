"""Executable capability contract for APG Platform Foundation."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any


SUPPORTED_PLFD_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_PLFD_AGENT_ROLES = [
	"foundation_reviewer",
	"dependency_reviewer",
	"baseline_reviewer",
	"readiness_reviewer",
	"change_reviewer",
	"security_reviewer",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"foundation": {
		"service_owner_required": True,
		"tier_classification_required": True,
		"dependency_map_required": True,
		"readiness_score_required": True,
	},
	"baselines": {
		"configuration_baseline_required": True,
		"tenant_baseline_required": True,
		"auth_baseline_required": True,
		"audit_baseline_required": True,
		"baseline_evidence_required": True,
		"baseline_approver_required": True,
	},
	"operations": {
		"health_gate_required": True,
		"monitoring_required": True,
		"rollback_plan_required": True,
		"change_window_required": True,
		"event_stream": "bytewax",
	},
	"plfd_agents": {
		"agent_assist_enabled": True,
		"agent_registration_required": True,
		"agent_runtime_required": True,
		"agent_role_required": True,
		"agent_scope_required": True,
		"agent_contribution_disclosure_required": True,
		"supported_runtimes": SUPPORTED_PLFD_AGENT_RUNTIMES,
		"allowed_roles": SUPPORTED_PLFD_AGENT_ROLES,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_foundation_changes": True,
		"broad_change_review_required": True,
		"security_review_required": True,
		"state_change_audit_required": True,
		"tenant_isolation_required": True,
		"batch_event_stream": "bytewax",
	},
	"observability": {
		"audit_required": True,
		"foundation_metrics_required": True,
		"readiness_metrics_required": True,
		"dependency_metrics_required": True,
		"agent_activity_required": True,
		"event_stream": "bytewax",
	},
	"adapters": {
		"generated_app_runtime": "service.PlfdService",
		"api_helpers": "api.py",
		"view_models": "views.py",
		"event_stream": "bytewax",
		"configuration": "conf",
		"multi_tenant": "mten",
		"identity": "auth",
		"audit_sink": "audl",
		"monitoring": "moni",
		"health": "hlth",
		"registry": "regy",
		"security": "secu",
		"plugins": "plgn",
	},
	"ui": {
		"enable_foundation_dashboard": True,
		"enable_dependency_map": True,
		"enable_baseline_manager": True,
		"enable_readiness_gate": True,
		"enable_agent_panel": True,
		"enable_audit": True,
	},
	"theme": {
		"default_theme": "plfd_platform_foundation",
		"allow_tenant_overrides": True,
	},
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"foundation",
		"baselines",
		"operations",
		"plfd_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {
		key: {"type": "object"}
		for key in [
			"foundation",
			"baselines",
			"operations",
			"plfd_agents",
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
	{"name": "tenant_context_required", "description": "All platform-foundation operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "foundation_service_requires_owner", "description": "Foundation services require owners.", "condition": {"operation": "register_foundation_service", "service_owner_assigned": False}, "effect": {"decision": "deny", "reason": "service_owner_required", "required_action": "assign_service_owner"}},
	{"name": "foundation_service_requires_tier", "description": "Foundation services require tier classification.", "condition": {"operation": "register_foundation_service", "tier_classified": False}, "effect": {"decision": "deny", "reason": "tier_classification_required", "required_action": "classify_service_tier"}},
	{"name": "foundation_service_requires_readiness_score", "description": "Foundation services require readiness score.", "condition": {"operation": "register_foundation_service", "readiness_score_present": False}, "effect": {"decision": "deny", "reason": "readiness_score_required", "required_action": "record_readiness_score"}},
	{"name": "dependency_requires_evidence", "description": "Foundation dependencies require health evidence.", "condition": {"operation": "record_dependency", "dependency_evidence_present": False}, "effect": {"decision": "deny", "reason": "dependency_evidence_required", "required_action": "attach_dependency_evidence"}},
	{"name": "baseline_requires_evidence", "description": "Foundation baselines require evidence.", "condition": {"operation": "attach_baseline", "baseline_evidence_present": False}, "effect": {"decision": "deny", "reason": "baseline_evidence_required", "required_action": "attach_baseline_evidence"}},
	{"name": "baseline_requires_approver", "description": "Foundation baselines require approver identity.", "condition": {"operation": "attach_baseline", "baseline_approver_present": False}, "effect": {"decision": "deny", "reason": "baseline_approver_required", "required_action": "attach_baseline_approver"}},
	{"name": "dependency_health_required", "description": "Foundation changes require healthy dependencies.", "condition": {"operation": "approve_platform_change", "dependencies_healthy": False}, "effect": {"decision": "deny", "reason": "dependency_health_required", "required_action": "restore_dependency_health"}},
	{"name": "configuration_baseline_required", "description": "Foundation services require configuration baselines.", "condition": {"configuration_baseline_present": False}, "effect": {"decision": "deny", "reason": "configuration_baseline_required", "required_action": "attach_configuration_baseline"}},
	{"name": "platform_change_requires_owner", "description": "Platform foundation changes require owner identity.", "condition": {"operation": "propose_platform_change", "change_owner_present": False}, "effect": {"decision": "deny", "reason": "change_owner_required", "required_action": "attach_change_owner"}},
	{"name": "platform_change_requires_affected_capability", "description": "Platform changes require affected capability scope.", "condition": {"operation": "propose_platform_change", "affected_capability_count_lte": 0}, "effect": {"decision": "deny", "reason": "affected_capability_required", "required_action": "set_affected_capability_scope"}},
	{"name": "platform_change_requires_approval", "description": "Platform foundation changes require approval.", "condition": {"operation": "approve_platform_change", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "platform_change_approval_required", "required_action": "record_platform_approval"}},
	{"name": "platform_change_requires_security_review", "description": "Platform changes require security review.", "condition": {"operation": "approve_platform_change", "security_review_recorded": False}, "effect": {"decision": "deny", "reason": "security_review_required", "required_action": "record_security_review"}},
	{"name": "platform_change_requires_window", "description": "Platform changes require change window.", "condition": {"operation": "approve_platform_change", "change_window_present": False}, "effect": {"decision": "deny", "reason": "change_window_required", "required_action": "attach_change_window"}},
	{"name": "platform_change_requires_rollback", "description": "Platform changes require rollback plan.", "condition": {"operation": "approve_platform_change", "rollback_plan_present": False}, "effect": {"decision": "deny", "reason": "rollback_plan_required", "required_action": "attach_rollback_plan"}},
	{"name": "platform_change_requires_bytewax_stream", "description": "Platform change lifecycle events require Bytewax event streams.", "condition": {"operation": "approve_platform_change", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
	{"name": "broad_platform_change_requires_review", "description": "Broad platform changes require review.", "condition": {"affected_capability_count_gt": 10, "broad_review_recorded": False}, "effect": {"decision": "require_review", "reason": "broad_platform_review_required", "required_action": "review_platform_change"}},
	{"name": "plfd_agent_requires_registration", "description": "AI foundation agents must be registered.", "condition": {"plfd_agent_present": True, "agent_registered": False}, "effect": {"decision": "deny", "reason": "plfd_agent_registration_required", "required_action": "register_plfd_agent"}},
	{"name": "plfd_agent_runtime_supported", "description": "AI foundation agents must use a supported runtime.", "condition": {"plfd_agent_present": True, "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "plfd_agent_runtime_not_supported", "required_action": "choose_supported_plfd_agent_runtime"}},
	{"name": "plfd_agent_role_supported", "description": "AI foundation agents must use a supported role.", "condition": {"plfd_agent_present": True, "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "plfd_agent_role_not_supported", "required_action": "choose_supported_plfd_agent_role"}},
	{"name": "plfd_agent_requires_scope", "description": "AI foundation agents require explicit scope.", "condition": {"plfd_agent_present": True, "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "plfd_agent_scope_required", "required_action": "set_plfd_agent_scope"}},
	{"name": "plfd_agent_requires_disclosure", "description": "AI foundation-agent contributions require disclosure.", "condition": {"plfd_agent_present": True, "agent_contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "plfd_agent_disclosure_required", "required_action": "disclose_plfd_agent"}},
	{"name": "plfd_state_change_requires_audit", "description": "Platform foundation lifecycle state changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "plfd_audit_event_required", "required_action": "record_plfd_audit_event"}},
	{"name": "batch_foundation_mutation_requires_bytewax", "description": "Batch foundation mutations must use Bytewax event streams.", "condition": {"requested_operation": "batch_foundation_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/plfd/dashboard", "component": "PLFDDashboard", "permission": "plfd:view", "nav_group": "Overview"},
	{"name": "services", "path": "/plfd/services", "component": "FoundationServices", "permission": "plfd:manage_services", "nav_group": "Services"},
	{"name": "dependencies", "path": "/plfd/dependencies", "component": "DependencyMap", "permission": "plfd:view", "nav_group": "Readiness"},
	{"name": "baselines", "path": "/plfd/baselines", "component": "BaselineManager", "permission": "plfd:manage_baselines", "nav_group": "Baselines"},
	{"name": "readiness", "path": "/plfd/readiness", "component": "ReadinessGate", "permission": "plfd:view", "nav_group": "Readiness"},
	{"name": "changes", "path": "/plfd/changes", "component": "PlatformChangeQueue", "permission": "plfd:approve_changes", "nav_group": "Governance"},
	{"name": "agents", "path": "/plfd/agents", "component": "PLFDAgentPanel", "permission": "plfd:admin", "nav_group": "Operations"},
	{"name": "governance", "path": "/plfd/governance", "component": "FoundationGovernance", "permission": "plfd:admin", "nav_group": "Governance"},
	{"name": "audit", "path": "/plfd/audit", "component": "FoundationAuditTrail", "permission": "plfd:admin", "nav_group": "Governance"},
	{"name": "settings", "path": "/plfd/settings", "component": "PLFDSettings", "permission": "plfd:admin", "nav_group": "Administration"},
]

THEME: dict[str, Any] = {
	"name": "plfd_platform_foundation",
	"tokens": {
		"color.primary": "#2A4365",
		"color.accent": "#38A169",
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
		"foundation_card": {"icon": "layers", "status_indicator": "tier-pill", "risk_style": "readiness-band"},
		"dependency_map": {"visual": "service-graph", "highlight": "health-chip"},
		"baseline_manager": {"visual": "policy-grid", "status_style": "baseline-chip"},
		"change_queue": {"visual": "approval-lanes", "status_style": "risk-chip"},
		"agent_panel": {"visual": "agent-roster", "status_style": "scope-chip"},
		"audit_timeline": {"visual": "event-timeline", "status_style": "foundation-chip"},
	},
}


def streaming_manifest() -> dict[str, Any]:
	return {
		"processor": "bytewax",
		"topic": "apg.plfd.lifecycle",
		"state": ["services", "dependencies", "baselines", "readiness_assessments", "changes", "plfd_agents", "audit_events"],
		"events": [
			"plfd_service_registered",
			"plfd_dependency_recorded",
			"plfd_baseline_attached",
			"plfd_readiness_assessed",
			"plfd_change_proposed",
			"plfd_change_approved",
			"plfd_agent_registered",
		],
		"batch_mutation_guardrail": "batch_foundation_mutation_requires_bytewax",
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "plfd",
		"display_name": "Platform Foundation",
		"version": "1.0.0",
		"provides": [
			"foundation_registry",
			"dependency_posture",
			"configuration_baselines",
			"readiness_gates",
			"platform_governance",
			"plfd_agents",
		],
		"requires": ["conf", "mten", "auth", "audl"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/plfd/api/v1",
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


def event_stream_name(value: str) -> str:
	return value.strip().lower().split("://", 1)[0]


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
			actual = context.get(key[:-3])
			if actual is None or actual == expected:
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
