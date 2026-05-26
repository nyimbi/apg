"""
Executable capability contract for APG Intelligent Gateway.

APIG is a first-class APG capability. This module exposes tenant-scoped
configuration, deterministic gateway-governance rules, UI surfaces, and theme
tokens so composition tooling can integrate with APIG without starting gateway
runtime services.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CapabilityConfiguration:
	"""Tenant-scoped APIG configuration defaults and schema."""

	defaults: dict[str, Any] = field(default_factory=lambda: {
		"tenant_id": "default",
		"routing": {
			"default_strategy": "weighted",
			"service_discovery_required": True,
			"route_owner_required": True,
			"canary_release_enabled": True
		},
		"security": {
			"require_tenant_context": True,
			"auth_required_by_default": True,
			"m_tls_enabled": True,
			"blocked_without_threat_policy": True
		},
		"traffic": {
			"rate_limits_enabled": True,
			"default_rps_limit": 1000,
			"circuit_breaking_enabled": True,
			"quota_review_threshold": 100000
		},
		"observability": {
			"emit_access_logs": True,
			"emit_metrics": True,
			"trace_propagation_enabled": True,
			"audit_policy_changes": True
		},
		"edge": {
			"wasm_filters_enabled": True,
			"filter_signing_required": True,
			"edge_deployment_enabled": True
		},
		"ui": {
			"enable_route_designer": True,
			"enable_traffic_console": True,
			"enable_security_policies": True,
			"enable_edge_filters": True
		},
		"theme": {
			"default_theme": "apig_gateway_console",
			"allow_tenant_overrides": True
		}
	})
	schema: dict[str, Any] = field(default_factory=lambda: {
		"type": "object",
		"required": [
			"tenant_id",
			"routing",
			"security",
			"traffic",
			"observability",
			"edge",
			"ui",
			"theme"
		],
		"properties": {
			"tenant_id": {"type": "string", "minLength": 1},
			"routing": {"type": "object"},
			"security": {"type": "object"},
			"traffic": {"type": "object"},
			"observability": {"type": "object"},
			"edge": {"type": "object"},
			"ui": {"type": "object"},
			"theme": {"type": "object"}
		}
	})

	def for_tenant(self, tenant_id: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
		"""Return configuration with tenant-specific overrides applied."""
		assert isinstance(tenant_id, str) and tenant_id, "tenant_id is required"
		merged = _deep_copy(self.defaults)
		merged["tenant_id"] = tenant_id
		if overrides:
			_deep_merge(merged, overrides)
		return merged


@dataclass(frozen=True)
class CapabilityRule:
	"""Simple APIG policy rule definition."""

	name: str
	description: str
	condition: dict[str, Any]
	effect: dict[str, Any]


class CapabilityRuleEngine:
	"""Deterministic APIG rule engine for gateway control decisions."""

	def __init__(self, rules: list[CapabilityRule] | None = None):
		self.rules = rules or default_rules()

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		"""Evaluate matching gateway governance rules."""
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
	"""UI route exposed by APIG."""

	name: str
	path: str
	component: str
	permission: str
	nav_group: str


@dataclass(frozen=True)
class CapabilityTheme:
	"""Visual theme contract for APIG UI surfaces."""

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
		"density": "compact"
	})
	components: dict[str, dict[str, str]] = field(default_factory=lambda: {
		"route_status_card": {"icon": "route", "status_indicator": "traffic-pill", "risk_style": "latency-band"},
		"traffic_policy_panel": {"visual": "rule-stack", "highlight": "limit-chip"},
		"gateway_topology_map": {"visual": "edge-route-graph", "edge_style": "weighted-route-line"},
		"wasm_filter_trace": {"visual": "filter-chain", "status_style": "signature-pill"}
	})


def default_rules() -> list[CapabilityRule]:
	"""Default APIG rules available to every tenant."""
	return [
		CapabilityRule("tenant_context_required", "All gateway operations require tenant context.", {"tenant_context_present": False}, {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}),
		CapabilityRule("route_requires_registered_service", "Routes require a registered upstream service.", {"operation": "create_route", "service_registered": False}, {"decision": "deny", "reason": "registered_service_required", "required_action": "register_upstream_service"}),
		CapabilityRule("public_route_requires_auth_policy", "Public routes require an explicit auth policy.", {"route_exposure": "public", "auth_policy_attached": False}, {"decision": "deny", "reason": "auth_policy_required", "required_action": "attach_auth_policy"}),
		CapabilityRule("unsafe_method_requires_threat_policy", "Unsafe methods require a threat policy.", {"unsafe_http_method_enabled": True, "threat_policy_attached": False}, {"decision": "deny", "reason": "threat_policy_required", "required_action": "attach_threat_policy"}),
		CapabilityRule("wasm_filter_requires_signature", "WASM edge filters require signature verification.", {"wasm_filter_attached": True, "filter_signature_verified": False}, {"decision": "deny", "reason": "filter_signature_required", "required_action": "verify_filter_signature"}),
		CapabilityRule("high_quota_requires_review", "High gateway quotas require review.", {"requested_rps_limit_gt": 100000, "quota_review_recorded": False}, {"decision": "require_review", "reason": "quota_review_required", "required_action": "record_quota_review"})
	]


def ui_manifest() -> dict[str, Any]:
	"""Return APIG UI surface manifest."""
	routes = [
		CapabilityUIRoute("dashboard", "/apig/dashboard", "APIGDashboard", "apig:view", "Overview"),
		CapabilityUIRoute("routes", "/apig/routes", "RouteDesigner", "apig:manage_routes", "Gateway"),
		CapabilityUIRoute("traffic", "/apig/traffic", "TrafficConsole", "apig:manage_traffic", "Gateway"),
		CapabilityUIRoute("security", "/apig/security", "GatewaySecurityPolicies", "apig:manage_security", "Security"),
		CapabilityUIRoute("upstreams", "/apig/upstreams", "UpstreamServices", "apig:manage_routes", "Gateway"),
		CapabilityUIRoute("edge", "/apig/edge", "EdgeFilterManager", "apig:manage_edge", "Edge"),
		CapabilityUIRoute("analytics", "/apig/analytics", "GatewayAnalytics", "apig:view_metrics", "Operations"),
		CapabilityUIRoute("settings", "/apig/settings", "APIGSettings", "apig:admin", "Administration")
	]
	return {"shell": "flask_appbuilder", "view_module": "control_plane.py", "api_prefix": "/apig/api/v1", "routes": [route.__dict__ for route in routes], "template_roots": ["templates/", "static/"], "requires_theme": True}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable APIG capability contract."""
	config = CapabilityConfiguration()
	theme = CapabilityTheme()
	return {"capability": "apig", "display_name": "APG Intelligent Gateway", "configuration": config.for_tenant(tenant_id, overrides), "configuration_schema": config.schema, "rule_engine": {"type": "deterministic", "rules": [rule.__dict__ for rule in default_rules()]}, "ui": ui_manifest(), "theme": {"name": theme.name, "tokens": theme.tokens, "components": theme.components}}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Convenience wrapper for default APIG rule evaluation."""
	return CapabilityRuleEngine().evaluate(context)


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_gt"):
			if not context.get(key[:-3], 0) > expected:
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
