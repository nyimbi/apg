"""Executable capability contract for APG API service mesh."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


SUPPORTED_GATEWAY_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_GATEWAY_AGENT_ROLES = [
	"mesh_architect",
	"route_reviewer",
	"policy_reviewer",
	"traffic_reviewer",
	"certificate_reviewer",
	"incident_reviewer",
]
GATEWAY_EVENT_STREAM = "apg.composition.gateway.lifecycle"


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"services": {
		"owner_required": True,
		"endpoint_required": True,
		"health_check_required": True,
		"capability_binding_required": True,
	},
	"routes": {
		"service_required": True,
		"match_rule_required": True,
		"policy_required_for_public": True,
		"approval_required_for_public": True,
	},
	"traffic": {
		"canary_supported": True,
		"canary_evidence_required": True,
		"rate_limit_required_for_public": True,
		"circuit_breaker_required": True,
	},
	"security": {
		"tls_required_for_public": True,
		"certificate_owner_required": True,
		"m_tls_supported": True,
		"secret_reference_required": True,
	},
	"gateway_agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_GATEWAY_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_GATEWAY_AGENT_ROLES,
		"human_approval_required": True,
		"max_autonomous_scope": "recommend_and_validate",
	},
	"governance": {
		"require_tenant_context": True,
		"audit_state_changes": True,
		"policy_attached_for_writes": True,
		"privileged_route_changes_reviewed": True,
	},
	"observability": {
		"event_stream": GATEWAY_EVENT_STREAM,
		"stream_processor": "bytewax",
		"emit_service_events": True,
		"emit_route_events": True,
		"emit_traffic_events": True,
		"emit_policy_events": True,
	},
	"adapters": {
		"authorization": "adapter",
		"audit": "adapter",
		"event_stream": "bytewax",
		"notification": "adapter",
		"secrets": "adapter",
		"theme": "adapter",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_service_registry": True,
		"enable_route_console": True,
		"enable_policy_center": True,
		"enable_traffic_console": True,
		"enable_certificate_console": True,
		"enable_agent_workbench": True,
		"enable_settings": True,
	},
	"theme": {"default_theme": "composition_gateway_control", "allow_tenant_overrides": True},
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"services",
		"routes",
		"traffic",
		"security",
		"gateway_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {
		"tenant_id": {"type": "string", "minLength": 1},
		"services": {"type": "object"},
		"routes": {"type": "object"},
		"traffic": {"type": "object"},
		"security": {"type": "object"},
		"gateway_agents": {"type": "object"},
		"governance": {"type": "object"},
		"observability": {"type": "object"},
		"adapters": {"type": "object"},
		"ui": {"type": "object"},
		"theme": {"type": "object"},
	},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All gateway operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "gateway_write_requires_policy", "description": "Gateway write operations require policy attachment.", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "operation_policy_required", "required_action": "attach_operation_policy"}},
	{"name": "service_requires_owner", "description": "Mesh services require an accountable owner.", "condition": {"operation": "register_service", "service_owner_assigned": False}, "effect": {"decision": "deny", "reason": "service_owner_required", "required_action": "assign_service_owner"}},
	{"name": "service_requires_endpoint", "description": "Mesh services require at least one endpoint.", "condition": {"operation": "register_service", "endpoint_present": False}, "effect": {"decision": "deny", "reason": "service_endpoint_required", "required_action": "attach_service_endpoint"}},
	{"name": "service_requires_health_check", "description": "Mesh services require health-check configuration.", "condition": {"operation": "register_service", "health_check_present": False}, "effect": {"decision": "deny", "reason": "service_health_check_required", "required_action": "attach_health_check"}},
	{"name": "public_route_requires_policy", "description": "Public routes require an attached policy.", "condition": {"operation": "create_route", "public_route": True, "route_policy_attached": False}, "effect": {"decision": "deny", "reason": "public_route_policy_required", "required_action": "attach_route_policy"}},
	{"name": "public_route_requires_approval", "description": "Public routes require approval.", "condition": {"operation": "create_route", "public_route": True, "approval_recorded": False}, "effect": {"decision": "deny", "reason": "public_route_approval_required", "required_action": "record_route_approval"}},
	{"name": "public_route_requires_tls", "description": "Public routes require TLS.", "condition": {"operation": "create_route", "public_route": True, "tls_enabled": False}, "effect": {"decision": "deny", "reason": "public_route_tls_required", "required_action": "enable_tls"}},
	{"name": "route_requires_bytewax_stream", "description": "Route lifecycle events must use Bytewax.", "condition": {"operation": "create_route", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_gateway_lifecycle_to_bytewax"}},
	{"name": "traffic_shift_requires_canary_evidence", "description": "Canary traffic shifts require evidence.", "condition": {"operation": "shift_traffic", "canary_shift": True, "canary_evidence_present": False}, "effect": {"decision": "require_review", "reason": "canary_evidence_required", "required_action": "attach_canary_evidence"}},
	{"name": "traffic_shift_requires_bytewax_stream", "description": "Traffic-shift events must use Bytewax.", "condition": {"operation": "shift_traffic", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_traffic_shift_to_bytewax"}},
	{"name": "public_service_requires_rate_limit", "description": "Public services require rate limits.", "condition": {"operation": "attach_policy", "public_service": True, "rate_limit_configured": False}, "effect": {"decision": "deny", "reason": "rate_limit_required", "required_action": "configure_rate_limit"}},
	{"name": "public_service_requires_circuit_breaker", "description": "Public services require circuit breakers.", "condition": {"operation": "attach_policy", "public_service": True, "circuit_breaker_configured": False}, "effect": {"decision": "deny", "reason": "circuit_breaker_required", "required_action": "configure_circuit_breaker"}},
	{"name": "certificate_requires_owner", "description": "Certificates require an owner.", "condition": {"operation": "register_certificate", "certificate_owner_assigned": False}, "effect": {"decision": "deny", "reason": "certificate_owner_required", "required_action": "assign_certificate_owner"}},
	{"name": "certificate_requires_secret_reference", "description": "Certificates require a secret reference.", "condition": {"operation": "register_certificate", "secret_reference_present": False}, "effect": {"decision": "deny", "reason": "certificate_secret_reference_required", "required_action": "attach_secret_reference"}},
	{"name": "batch_route_change_requires_bytewax", "description": "Batch route changes require Bytewax coordination.", "condition": {"operation": "batch_route_change", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_batch_gateway_changes_to_bytewax"}},
	{"name": "gateway_agent_runtime_supported", "description": "Gateway agents must use an approved runtime.", "condition": {"operation": "register_gateway_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "gateway_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"}},
	{"name": "gateway_agent_role_supported", "description": "Gateway agents must use an approved role.", "condition": {"operation": "register_gateway_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "gateway_agent_role_not_supported", "required_action": "select_supported_agent_role"}},
	{"name": "privileged_agent_gateway_action_requires_human_approval", "description": "Privileged gateway actions proposed by agents require human approval.", "condition": {"operation": "agent_gateway_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/composition-gateway/dashboard", "component": "GatewayDashboard", "permission": "composition_gateway:view", "nav_group": "Overview"},
	{"name": "services", "path": "/composition-gateway/services", "component": "GatewayServiceRegistry", "permission": "composition_gateway:manage_services", "nav_group": "Services"},
	{"name": "routes", "path": "/composition-gateway/routes", "component": "GatewayRouteConsole", "permission": "composition_gateway:manage_routes", "nav_group": "Routes"},
	{"name": "policies", "path": "/composition-gateway/policies", "component": "GatewayPolicyCenter", "permission": "composition_gateway:govern", "nav_group": "Governance"},
	{"name": "traffic", "path": "/composition-gateway/traffic", "component": "GatewayTrafficConsole", "permission": "composition_gateway:operate", "nav_group": "Operations"},
	{"name": "certificates", "path": "/composition-gateway/certificates", "component": "GatewayCertificateConsole", "permission": "composition_gateway:admin", "nav_group": "Security"},
	{"name": "agents", "path": "/composition-gateway/agents", "component": "GatewayAgentWorkbench", "permission": "composition_gateway:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/composition-gateway/settings", "component": "GatewaySettings", "permission": "composition_gateway:admin", "nav_group": "Administration"},
]

THEME: dict[str, Any] = {
	"name": "composition_gateway_control",
	"tokens": {"color.primary": "#28536B", "color.accent": "#C44536", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {
		"service_registry": {"icon": "server", "status_indicator": "service-pill", "risk_style": "health-band"},
		"route_console": {"visual": "route-table", "status_style": "route-chip"},
		"policy_center": {"visual": "policy-grid", "status_style": "guardrail-chip"},
		"traffic_console": {"visual": "traffic-lanes", "status_style": "canary-chip"},
		"certificate_console": {"visual": "certificate-list", "status_style": "expiry-chip"},
		"agent_workbench": {"visual": "review-lane", "status_style": "approval-chip"},
	},
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "composition_gateway",
		"display_name": "API Service Mesh",
		"provides": [
			"service_mesh_registry",
			"gateway_route_lifecycle",
			"traffic_management",
			"gateway_policy_enforcement",
			"certificate_lifecycle",
			"mesh_health_observability",
			"gateway_agents",
		],
		"requires": ["auth", "audl", "ntfy", "registry", "composition_access", "composition_events"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {"shell": "apg_python", "view_module": "views.py", "api_prefix": "/composition-gateway/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True},
		"theme": deepcopy(THEME),
		"streaming": streaming_manifest(),
	}


def streaming_manifest() -> dict[str, Any]:
	return {
		"processor": "bytewax",
		"stream": GATEWAY_EVENT_STREAM,
		"key": "tenant_id",
		"events": [
			"service_registered",
			"route_created",
			"policy_attached",
			"traffic_shifted",
			"certificate_registered",
			"health_recorded",
			"gateway_agent_registered",
		],
		"states": ["draft", "active", "healthy", "degraded", "canary", "blocked", "retired"],
		"guardrails": [
			"route_requires_bytewax_stream",
			"traffic_shift_requires_bytewax_stream",
			"batch_route_change_requires_bytewax",
			"privileged_agent_gateway_action_requires_human_approval",
		],
	}


def event_stream_name() -> str:
	return GATEWAY_EVENT_STREAM


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
