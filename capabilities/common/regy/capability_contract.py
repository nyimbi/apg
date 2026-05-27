"""
Executable capability contract for APG API/Service Registry.

REGY is a first-class APG capability. This module exposes tenant-scoped
configuration, deterministic registry-governance rules, UI surfaces, and theme
tokens so composition tooling can integrate with REGY consistently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CapabilityConfiguration:
	"""Tenant-scoped REGY configuration defaults and schema."""

	defaults: dict[str, Any] = field(default_factory=lambda: {
		"tenant_id": "default",
		"registration": {
			"owner_required": True,
			"health_endpoint_required": True,
			"api_version_required": True,
			"contract_schema_required": True
		},
		"discovery": {
			"service_discovery_enabled": True,
			"cache_ttl_seconds": 60,
			"prefer_healthy_instances": True,
			"cross_tenant_discovery_allowed": False
		},
		"health": {
			"active_health_checks_enabled": True,
			"default_interval_seconds": 30,
			"failure_threshold": 3,
			"degraded_blocks_gateway_publish": True
		},
		"governance": {
			"require_tenant_context": True,
			"audit_registration_events": True,
			"breaking_change_review_required": True,
			"duplicate_service_names_blocked": True
		},
		"routing": {
			"gateway_sync_enabled": True,
			"load_balancing_metadata_required": True,
			"circuit_breaking_enabled": True
		},
		"ui": {
			"enable_service_catalog": True,
			"enable_discovery_console": True,
			"enable_health_dashboard": True,
			"enable_version_manager": True
		},
		"theme": {
			"default_theme": "regy_service_catalog",
			"allow_tenant_overrides": True
		}
	})
	schema: dict[str, Any] = field(default_factory=lambda: {
		"type": "object",
		"required": [
			"tenant_id",
			"registration",
			"discovery",
			"health",
			"governance",
			"routing",
			"ui",
			"theme"
		],
		"properties": {
			"tenant_id": {"type": "string", "minLength": 1},
			"registration": {"type": "object"},
			"discovery": {"type": "object"},
			"health": {"type": "object"},
			"governance": {"type": "object"},
			"routing": {"type": "object"},
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
	"""Simple REGY policy rule definition."""

	name: str
	description: str
	condition: dict[str, Any]
	effect: dict[str, Any]


class CapabilityRuleEngine:
	"""Deterministic REGY rule engine for registry control decisions."""

	def __init__(self, rules: list[CapabilityRule] | None = None):
		self.rules = rules or default_rules()

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		"""Evaluate matching registry governance rules."""
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
	"""UI route exposed by REGY."""

	name: str
	path: str
	component: str
	permission: str
	nav_group: str


@dataclass(frozen=True)
class CapabilityTheme:
	"""Visual theme contract for REGY UI surfaces."""

	name: str = "regy_service_catalog"
	tokens: dict[str, str] = field(default_factory=lambda: {
		"color.primary": "#355070",
		"color.accent": "#6D597A",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F7F8FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact"
	})
	components: dict[str, dict[str, str]] = field(default_factory=lambda: {
		"service_catalog_row": {"icon": "network", "status_indicator": "health-pill", "risk_style": "version-band"},
		"discovery_result_card": {"visual": "instance-stack", "highlight": "endpoint-chip"},
		"health_check_timeline": {"visual": "probe-timeline", "status_style": "failure-threshold"},
		"version_compatibility_panel": {"visual": "version-matrix", "highlight": "breaking-change-chip"}
	})


def default_rules() -> list[CapabilityRule]:
	"""Default REGY rules available to every tenant."""
	return [
		CapabilityRule("tenant_context_required", "All registry operations require tenant context.", {"tenant_context_present": False}, {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}),
		CapabilityRule("service_registration_requires_owner", "Service registration requires an owner.", {"operation": "register_service", "owner_assigned": False}, {"decision": "deny", "reason": "service_owner_required", "required_action": "assign_service_owner"}),
		CapabilityRule("service_registration_requires_health_endpoint", "Service registration requires health endpoint metadata.", {"operation": "register_service", "health_endpoint_present": False}, {"decision": "deny", "reason": "health_endpoint_required", "required_action": "attach_health_endpoint"}),
		CapabilityRule("duplicate_service_name_blocked", "Duplicate service names are blocked within tenant scope.", {"duplicate_service_name": True}, {"decision": "deny", "reason": "duplicate_service_name", "required_action": "choose_unique_service_name"}),
		CapabilityRule("breaking_change_requires_review", "Breaking API changes require compatibility review.", {"breaking_change_detected": True, "compatibility_review_recorded": False}, {"decision": "require_review", "reason": "compatibility_review_required", "required_action": "record_compatibility_review"}),
		CapabilityRule("cross_tenant_discovery_denied", "Cross-tenant discovery is denied by default.", {"cross_tenant_discovery": True}, {"decision": "deny", "reason": "cross_tenant_discovery_denied", "required_action": "use_tenant_scoped_discovery"})
	]


def ui_manifest() -> dict[str, Any]:
	"""Return REGY UI surface manifest."""
	routes = [
		CapabilityUIRoute("dashboard", "/regy/dashboard", "RegistryDashboard", "regy:view", "Overview"),
		CapabilityUIRoute("services", "/regy/services", "ServiceCatalog", "regy:view_services", "Catalog"),
		CapabilityUIRoute("register", "/regy/register", "ServiceRegistration", "regy:register_service", "Catalog"),
		CapabilityUIRoute("discovery", "/regy/discovery", "DiscoveryConsole", "regy:discover", "Discovery"),
		CapabilityUIRoute("health", "/regy/health", "ServiceHealthDashboard", "regy:view_health", "Reliability"),
		CapabilityUIRoute("versions", "/regy/versions", "ServiceVersionManager", "regy:manage_versions", "Governance"),
		CapabilityUIRoute("gateway_sync", "/regy/gateway-sync", "GatewaySyncView", "regy:sync_gateway", "Integration"),
		CapabilityUIRoute("settings", "/regy/settings", "RegistrySettings", "regy:admin", "Administration")
	]
	return {"shell": "apg_python", "view_module": "views.py", "api_prefix": "/regy/api/v1", "routes": [route.__dict__ for route in routes], "template_roots": ["templates/", "static/"], "requires_theme": True}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable REGY capability contract."""
	config = CapabilityConfiguration()
	theme = CapabilityTheme()
	return {"capability": "regy", "display_name": "API/Service Registry", "configuration": config.for_tenant(tenant_id, overrides), "configuration_schema": config.schema, "rule_engine": {"type": "deterministic", "rules": [rule.__dict__ for rule in default_rules()]}, "ui": ui_manifest(), "theme": {"name": theme.name, "tokens": theme.tokens, "components": theme.components}}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Convenience wrapper for default REGY rule evaluation."""
	return CapabilityRuleEngine().evaluate(context)


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if context.get(key) != expected:
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
