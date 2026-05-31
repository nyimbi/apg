"""Executable capability contract for APG API Gateway and Management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SUPPORTED_APIG_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_APIG_AGENT_ROLES = [
	"route_reviewer",
	"security_policy_reviewer",
	"traffic_reviewer",
	"quota_reviewer",
	"canary_reviewer",
	"deployment_reviewer",
	"edge_filter_reviewer",
	"retirement_reviewer",
]
PRIVILEGED_APIG_AGENT_ROLES = [
	"route_reviewer",
	"security_policy_reviewer",
	"traffic_reviewer",
	"quota_reviewer",
	"canary_reviewer",
	"deployment_reviewer",
	"edge_filter_reviewer",
]


@dataclass(frozen=True)
class CapabilityConfiguration:
	"""Tenant-scoped APIG configuration defaults and schema."""

	defaults: dict[str, Any] = field(default_factory=lambda: {
		"tenant_id": "default",
		"upstreams": {
			"service_discovery_required": True,
			"owner_required": True,
			"https_required": True,
			"health_required": True,
			"max_upstreams_per_tenant": 500,
		},
		"consumers": {
			"consumer_registration_required": True,
			"owner_required": True,
			"credential_rotation_required": True,
			"restricted_consumer_review_required": True,
		},
		"routes": {
			"default_strategy": "weighted",
			"route_owner_required": True,
			"absolute_path_required": True,
			"supported_methods": ["GET", "HEAD", "OPTIONS", "POST", "PUT", "PATCH", "DELETE"],
			"public_auth_required": True,
			"unsafe_method_threat_policy_required": True,
			"mtls_required_for_external": True,
		},
		"traffic": {
			"rate_limits_enabled": True,
			"default_rps_limit": 1000,
			"max_rps_without_review": 100000,
			"circuit_breaking_enabled": True,
			"rollback_plan_required": True,
		},
		"security": {
			"require_tenant_context": True,
			"auth_required_by_default": True,
			"m_tls_enabled": True,
			"blocked_without_threat_policy": True,
			"waf_policy_required_for_public": True,
		},
		"edge": {
			"wasm_filters_enabled": True,
			"filter_signing_required": True,
			"edge_deployment_enabled": True,
			"allowed_regions": ["local", "edge-east", "edge-west", "edge-africa", "edge-eu"],
		},
		"canary": {
			"canary_release_enabled": True,
			"review_threshold_percent": 10,
			"max_percent_without_review": 25,
			"max_percent": 50,
		},
		"deployments": {
			"observability_required": True,
			"trace_propagation_required": True,
			"access_logs_required": True,
			"approval_required_for_production": True,
		},
		"governance": {
			"policy_review_required": True,
			"audit_policy_changes": True,
			"retirement_impact_review_required": True,
			"lineage_capture_required": True,
		},
		"observability": {
			"emit_access_logs": True,
			"emit_metrics": True,
			"trace_propagation_enabled": True,
			"audit_policy_changes": True,
		},
		"adapters": {
			"production_runtime": "service.ProductionAPGIntelligentGatewayService",
			"generated_app_runtime": "gateway_runtime.ApigService",
			"reverse_proxy": "adapter",
			"service_discovery": "conf",
			"auth_provider": "auth",
			"credential_vault": "keym",
			"metrics_sink": "moni",
			"audit_sink": "audl",
			"event_stream": "bytewax",
			"cache_store": "cach",
			"edge_runtime": "wasm_runtime",
		},
		"agents": {
			"first_class": True,
			"supported_runtimes": SUPPORTED_APIG_AGENT_RUNTIMES,
			"supported_roles": SUPPORTED_APIG_AGENT_ROLES,
			"privileged_roles": PRIVILEGED_APIG_AGENT_ROLES,
			"require_scope": True,
			"require_owner": True,
			"require_purpose": True,
			"require_contribution_disclosure": True,
			"human_approval_required_for_privileged_roles": True,
		},
		"streaming": {
			"engine": "bytewax",
			"required_processor": "bytewax",
			"lifecycle_stream": "apig.lifecycle",
			"watermark": "event_time",
			"operations": [
				"upstream_batch",
				"consumer_batch",
				"route_batch",
				"policy_batch",
				"traffic_shift_batch",
				"deployment_batch",
				"gateway_agent_batch",
			],
			"topics": [
				"apig.upstreams",
				"apig.consumers",
				"apig.routes",
				"apig.policies",
				"apig.traffic",
				"apig.deployments",
				"apig.agents",
			],
		},
		"ui": {
			"enable_route_designer": True,
			"enable_upstream_manager": True,
			"enable_consumer_manager": True,
			"enable_traffic_console": True,
			"enable_security_policies": True,
			"enable_edge_filters": True,
			"enable_quota_reviews": True,
			"enable_canary_releases": True,
			"enable_deployment_gates": True,
			"enable_audit_timeline": True,
			"enable_gateway_agent_roster": True,
			"enable_lifecycle_batch_monitor": True,
		},
		"theme": {
			"default_theme": "apig_gateway_console",
			"allow_tenant_overrides": True,
		},
	})
	schema: dict[str, Any] = field(default_factory=lambda: {
		"type": "object",
		"required": [
			"tenant_id",
			"upstreams",
			"consumers",
			"routes",
			"traffic",
			"security",
			"edge",
			"canary",
			"deployments",
			"governance",
			"observability",
			"adapters",
			"agents",
			"streaming",
			"ui",
			"theme",
		],
		"properties": {
			"tenant_id": {"type": "string", "minLength": 1},
			"upstreams": {"type": "object"},
			"consumers": {"type": "object"},
			"routes": {"type": "object"},
			"traffic": {"type": "object"},
			"security": {"type": "object"},
			"edge": {"type": "object"},
			"canary": {"type": "object"},
			"deployments": {"type": "object"},
			"governance": {"type": "object"},
			"observability": {"type": "object"},
			"adapters": {"type": "object"},
			"agents": {"type": "object"},
			"streaming": {"type": "object"},
			"ui": {"type": "object"},
			"theme": {"type": "object"},
		},
	})

	def for_tenant(self, tenant_id: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
		assert isinstance(tenant_id, str) and tenant_id, "tenant_id is required"
		merged = _deep_copy(self.defaults)
		merged["tenant_id"] = tenant_id
		if overrides:
			_deep_merge(merged, overrides)
		return merged


@dataclass(frozen=True)
class CapabilityRule:
	name: str
	description: str
	condition: dict[str, Any]
	effect: dict[str, Any]


class CapabilityRuleEngine:
	"""Deterministic APIG rule engine for gateway control decisions."""

	def __init__(self, rules: list[CapabilityRule] | None = None):
		self.rules = rules or default_rules()

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		assert isinstance(context, dict), "context must be a dictionary"
		matched: list[str] = []
		actions: list[dict[str, Any]] = []
		decision = "allow"
		for rule in self.rules:
			if _matches(rule.condition, context):
				matched.append(rule.name)
				actions.append(rule.effect)
				if rule.effect.get("decision") == "deny":
					decision = "deny"
				elif rule.effect.get("decision") == "require_review" and decision != "deny":
					decision = "require_review"
		return {"decision": decision, "matched_rules": matched, "actions": actions, "context": context}


@dataclass(frozen=True)
class CapabilityUIRoute:
	name: str
	path: str
	component: str
	permission: str
	nav_group: str


@dataclass(frozen=True)
class CapabilityTheme:
	name: str = "apig_gateway_console"
	tokens: dict[str, str] = field(default_factory=lambda: {
		"color.primary": "#1F4E79",
		"color.accent": "#F18F01",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F5F7FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact",
	})
	components: dict[str, dict[str, str]] = field(default_factory=lambda: {
		"route_status_card": {"icon": "route", "status_indicator": "traffic-pill", "risk_style": "latency-band"},
		"upstream_health_panel": {"visual": "service-list", "status_indicator": "health-pill"},
		"consumer_access_panel": {"visual": "credential-table", "status_style": "rotation-chip"},
		"traffic_policy_panel": {"visual": "rule-stack", "highlight": "limit-chip"},
		"gateway_topology_map": {"visual": "edge-route-graph", "edge_style": "weighted-route-line"},
		"security_policy_matrix": {"visual": "policy-grid", "status_indicator": "control-chip"},
		"wasm_filter_trace": {"visual": "filter-chain", "status_style": "signature-pill"},
		"quota_review_queue": {"visual": "review-list", "highlight": "quota-chip"},
		"canary_release_panel": {"visual": "traffic-split", "status_indicator": "canary-pill"},
		"deployment_gate_panel": {"visual": "environment-lane", "status_indicator": "gate-pill"},
		"audit_timeline": {"visual": "event-timeline", "status_style": "decision-pill"},
		"gateway_agent_roster": {"visual": "agent-role-grid", "status_indicator": "approval-pill"},
		"bytewax_lifecycle_panel": {"visual": "stream-batch-ledger", "status_indicator": "processor-pill"},
	})


def default_rules() -> list[CapabilityRule]:
	return [
		CapabilityRule("tenant_context_required", "All gateway operations require tenant context.", {"tenant_context_present": False}, {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}),
		CapabilityRule("upstream_requires_owner", "Upstream services require an accountable owner.", {"operation": "register_upstream", "upstream_owner_assigned": False}, {"decision": "deny", "reason": "upstream_owner_required", "required_action": "assign_upstream_owner"}),
		CapabilityRule("upstream_requires_https", "Upstream services require HTTPS by default.", {"operation": "register_upstream", "https_enabled": False}, {"decision": "deny", "reason": "https_upstream_required", "required_action": "use_https_upstream"}),
		CapabilityRule("upstream_requires_health_check", "Upstreams require health status evidence.", {"operation": "register_upstream", "health_check_configured": False}, {"decision": "deny", "reason": "upstream_health_required", "required_action": "configure_upstream_health"}),
		CapabilityRule("consumer_requires_owner", "API consumers require an accountable owner.", {"operation": "register_consumer", "consumer_owner_assigned": False}, {"decision": "deny", "reason": "consumer_owner_required", "required_action": "assign_consumer_owner"}),
		CapabilityRule("consumer_requires_credential_rotation", "API consumers require credential rotation evidence.", {"operation": "register_consumer", "credential_rotation_recorded": False}, {"decision": "deny", "reason": "credential_rotation_required", "required_action": "record_credential_rotation"}),
		CapabilityRule("restricted_consumer_requires_rbac", "Restricted consumers require RBAC approval.", {"operation": "register_consumer", "access_tier": "restricted", "rbac_approval_recorded": False}, {"decision": "deny", "reason": "consumer_rbac_approval_required", "required_action": "record_consumer_rbac_approval"}),
		CapabilityRule("route_requires_owner", "Routes require an accountable owner.", {"operation": "create_route", "route_owner_assigned": False}, {"decision": "deny", "reason": "route_owner_required", "required_action": "assign_route_owner"}),
		CapabilityRule("route_path_must_be_absolute", "Route paths must start with a slash.", {"operation": "create_route", "absolute_path": False}, {"decision": "deny", "reason": "absolute_route_path_required", "required_action": "use_absolute_route_path"}),
		CapabilityRule("route_requires_registered_service", "Routes require a registered upstream service.", {"operation": "create_route", "service_registered": False}, {"decision": "deny", "reason": "registered_service_required", "required_action": "register_upstream_service"}),
		CapabilityRule("route_requires_methods", "Routes require at least one HTTP method.", {"operation": "create_route", "methods_present": False}, {"decision": "deny", "reason": "route_methods_required", "required_action": "declare_route_methods"}),
		CapabilityRule("public_route_requires_auth_policy", "Public routes require an explicit auth policy.", {"operation": "create_route", "route_exposure": "public", "auth_policy_attached": False}, {"decision": "deny", "reason": "auth_policy_required", "required_action": "attach_auth_policy"}),
		CapabilityRule("external_route_requires_mtls", "External routes require mTLS evidence.", {"operation": "create_route", "route_exposure": "external", "mtls_enabled": False}, {"decision": "deny", "reason": "mtls_required", "required_action": "enable_mtls"}),
		CapabilityRule("unsafe_method_requires_threat_policy", "Unsafe methods require a threat policy.", {"operation": "create_route", "unsafe_http_method_enabled": True, "threat_policy_attached": False}, {"decision": "deny", "reason": "threat_policy_required", "required_action": "attach_threat_policy"}),
		CapabilityRule("route_requires_rate_limit", "Routes require rate-limit policy evidence.", {"operation": "create_route", "rate_limit_configured": False}, {"decision": "deny", "reason": "rate_limit_required", "required_action": "configure_rate_limit"}),
		CapabilityRule("wasm_filter_requires_signature", "WASM edge filters require signature verification.", {"operation": "create_route", "wasm_filter_attached": True, "filter_signature_verified": False}, {"decision": "deny", "reason": "filter_signature_required", "required_action": "verify_filter_signature"}),
		CapabilityRule("high_quota_requires_review", "High gateway quotas require review.", {"operation": "create_route", "requested_rps_limit_gt": 100000, "quota_review_recorded": False}, {"decision": "require_review", "reason": "quota_review_required", "required_action": "record_quota_review"}),
		CapabilityRule("canary_requires_review", "Canary traffic shifts above threshold require review.", {"operation": "shift_traffic", "canary_percent_gt": 10, "canary_review_recorded": False}, {"decision": "require_review", "reason": "canary_review_required", "required_action": "record_canary_review"}),
		CapabilityRule("canary_percent_limit_enforced", "Canary traffic shifts cannot exceed the configured limit.", {"operation": "shift_traffic", "canary_percent_gt": 50}, {"decision": "deny", "reason": "canary_percent_limit_exceeded", "required_action": "reduce_canary_percent"}),
		CapabilityRule("traffic_shift_requires_rollback_plan", "Traffic shifts require a rollback plan.", {"operation": "shift_traffic", "rollback_plan_recorded": False}, {"decision": "deny", "reason": "rollback_plan_required", "required_action": "record_rollback_plan"}),
		CapabilityRule("deployment_requires_region", "Edge deployments require an allowed target region.", {"operation": "deploy_gateway", "allowed_region": False}, {"decision": "deny", "reason": "allowed_region_required", "required_action": "choose_allowed_region"}),
		CapabilityRule("deployment_requires_observability", "Gateway deployments require metrics, logs, and tracing.", {"operation": "deploy_gateway", "observability_configured": False}, {"decision": "deny", "reason": "observability_required", "required_action": "configure_gateway_observability"}),
		CapabilityRule("production_deployment_requires_approval", "Production deployments require approval evidence.", {"operation": "deploy_gateway", "environment": "production", "deployment_approval_recorded": False}, {"decision": "require_review", "reason": "deployment_approval_required", "required_action": "record_deployment_approval"}),
		CapabilityRule("policy_change_requires_review", "Gateway policy changes require review.", {"operation": "change_policy", "policy_review_recorded": False}, {"decision": "require_review", "reason": "policy_review_required", "required_action": "record_policy_review"}),
		CapabilityRule("route_retirement_requires_impact_review", "Retiring a route requires impact review.", {"operation": "retire_route", "impact_review_recorded": False}, {"decision": "deny", "reason": "impact_review_required", "required_action": "record_route_impact_review"}),
		CapabilityRule("gateway_agent_runtime_supported", "Gateway agents must use an approved runtime.", {"operation": "register_gateway_agent", "unsupported_agent_runtime": True}, {"decision": "deny", "reason": "unsupported_gateway_agent_runtime", "required_action": "choose_supported_agent_runtime"}),
		CapabilityRule("gateway_agent_role_supported", "Gateway agents must use an approved APIG role.", {"operation": "register_gateway_agent", "unsupported_agent_role": True}, {"decision": "deny", "reason": "unsupported_gateway_agent_role", "required_action": "choose_supported_agent_role"}),
		CapabilityRule("gateway_agent_requires_scope", "Gateway agents require bounded route, traffic, security, edge, or deployment scope.", {"operation": "register_gateway_agent", "agent_scope_present": False}, {"decision": "deny", "reason": "gateway_agent_scope_required", "required_action": "define_agent_scope"}),
		CapabilityRule("gateway_agent_requires_owner", "Gateway agents require an accountable owner.", {"operation": "register_gateway_agent", "agent_owner_present": False}, {"decision": "deny", "reason": "gateway_agent_owner_required", "required_action": "assign_agent_owner"}),
		CapabilityRule("gateway_agent_requires_purpose", "Gateway agents require a declared purpose.", {"operation": "register_gateway_agent", "agent_purpose_present": False}, {"decision": "deny", "reason": "gateway_agent_purpose_required", "required_action": "declare_agent_purpose"}),
		CapabilityRule("gateway_agent_requires_contribution_disclosure", "Gateway agents must disclose machine contribution before participating in gateway decisions.", {"operation": "register_gateway_agent", "agent_contribution_disclosed": False}, {"decision": "deny", "reason": "gateway_agent_contribution_disclosure_required", "required_action": "disclose_machine_contribution"}),
		CapabilityRule("gateway_agent_privileged_role_requires_human_approval", "Privileged gateway-agent roles require human approval.", {"operation": "register_gateway_agent", "privileged_agent_role": True, "human_approval_required": False}, {"decision": "require_review", "reason": "privileged_gateway_agent_human_approval_required", "required_action": "record_human_approval_requirement"}),
		CapabilityRule("bytewax_apig_stream_required", "APIG lifecycle batches must be routed through Bytewax.", {"operation": "validate_apig_lifecycle_batch", "event_stream_ne": "bytewax"}, {"decision": "deny", "reason": "bytewax_required", "required_action": "route_batch_to_bytewax"}),
	]


def ui_manifest() -> dict[str, Any]:
	routes = [
		CapabilityUIRoute("dashboard", "/apig/dashboard", "APIGDashboard", "apig:view", "Overview"),
		CapabilityUIRoute("routes", "/apig/routes", "RouteDesigner", "apig:manage_routes", "Gateway"),
		CapabilityUIRoute("upstreams", "/apig/upstreams", "UpstreamServices", "apig:manage_routes", "Gateway"),
		CapabilityUIRoute("consumers", "/apig/consumers", "APIConsumers", "apig:manage_security", "Security"),
		CapabilityUIRoute("traffic", "/apig/traffic", "TrafficConsole", "apig:manage_traffic", "Gateway"),
		CapabilityUIRoute("security", "/apig/security", "GatewaySecurityPolicies", "apig:manage_security", "Security"),
		CapabilityUIRoute("edge", "/apig/edge", "EdgeFilterManager", "apig:manage_edge", "Edge"),
		CapabilityUIRoute("quota_reviews", "/apig/quota-reviews", "QuotaReviewQueue", "apig:manage_traffic", "Governance"),
		CapabilityUIRoute("canary", "/apig/canary", "CanaryReleaseConsole", "apig:manage_traffic", "Gateway"),
		CapabilityUIRoute("deployments", "/apig/deployments", "GatewayDeployments", "apig:admin", "Operations"),
		CapabilityUIRoute("analytics", "/apig/analytics", "GatewayAnalytics", "apig:view_metrics", "Operations"),
		CapabilityUIRoute("agents", "/apig/agents", "GatewayAgentRoster", "apig:admin", "Governance"),
		CapabilityUIRoute("lifecycle", "/apig/lifecycle", "GatewayLifecycleBatchMonitor", "apig:view_metrics", "Operations"),
		CapabilityUIRoute("audit", "/apig/audit", "GatewayAuditTimeline", "apig:view_metrics", "Governance"),
		CapabilityUIRoute("settings", "/apig/settings", "APIGSettings", "apig:admin", "Administration"),
	]
	return {"shell": "apg_python", "view_module": "view_models.py", "api_prefix": "/apig/api/v1", "routes": [route.__dict__ for route in routes], "template_roots": ["templates/", "static/"], "requires_theme": True}


def agent_manifest() -> dict[str, Any]:
	config = CapabilityConfiguration().defaults["agents"]
	return {
		"first_class": config["first_class"],
		"supported_runtimes": list(config["supported_runtimes"]),
		"supported_roles": list(config["supported_roles"]),
		"privileged_roles": list(config["privileged_roles"]),
		"requires_scope": config["require_scope"],
		"requires_owner": config["require_owner"],
		"requires_purpose": config["require_purpose"],
		"requires_contribution_disclosure": config["require_contribution_disclosure"],
	}


def streaming_manifest() -> dict[str, Any]:
	config = CapabilityConfiguration().defaults["streaming"]
	return {
		"engine": config["engine"],
		"required_processor": config["required_processor"],
		"lifecycle_stream": config["lifecycle_stream"],
		"watermark": config["watermark"],
		"operations": list(config["operations"]),
		"topics": list(config["topics"]),
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = CapabilityConfiguration()
	theme = CapabilityTheme()
	return {
		"capability": "apig",
		"display_name": "API Gateway & Management",
		"provides": ["api_gateway", "traffic_management", "gateway_agent_composition"],
		"requires": ["auth", "moni", "mqeb", "conf"],
		"configuration": config.for_tenant(tenant_id, overrides),
		"configuration_schema": config.schema,
		"rule_engine": {"type": "deterministic", "rules": [rule.__dict__ for rule in default_rules()]},
		"ui": ui_manifest(),
		"theme": {"name": theme.name, "tokens": theme.tokens, "components": theme.components},
		"agents": agent_manifest(),
		"streaming": streaming_manifest(),
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	return CapabilityRuleEngine().evaluate(context)


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_gt"):
			if not context.get(key[:-3], 0) > expected:
				return False
		elif key.endswith("_lt"):
			if not context.get(key[:-3], 0) < expected:
				return False
		elif key.endswith("_ne"):
			if context.get(key[:-3]) == expected:
				return False
		elif context.get(key) != expected:
			return False
	return True


def _deep_copy(value: dict[str, Any]) -> dict[str, Any]:
	copied: dict[str, Any] = {}
	for key, item in value.items():
		if isinstance(item, dict):
			copied[key] = _deep_copy(item)
		elif isinstance(item, list):
			copied[key] = list(item)
		else:
			copied[key] = item
	return copied


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
	for key, value in source.items():
		if isinstance(value, dict) and isinstance(target.get(key), dict):
			_deep_merge(target[key], value)
		else:
			target[key] = value
