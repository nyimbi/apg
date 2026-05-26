"""
Executable capability contract for APG Multi-Tenant Management.

MTEN is a first-class APG capability. This module exposes tenant-scoped
configuration, deterministic governance rules, UI surfaces, and theme tokens so
composition tooling can integrate with MTEN without loading the full runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CapabilityConfiguration:
	"""Tenant-scoped MTEN configuration defaults and schema."""

	defaults: dict[str, Any] = field(default_factory=lambda: {
		"tenant_id": "default",
		"provisioning": {
			"default_tier": "free",
			"provisioning_timeout_seconds": 60,
			"max_concurrent_provisions": 10,
			"require_template_for_custom_tiers": True
		},
		"isolation": {
			"require_tenant_context": True,
			"allow_cross_tenant_operations": False,
			"enforce_encrypted_boundaries": True,
			"suspend_on_isolation_breach": True
		},
		"resources": {
			"auto_rightsize_enabled": True,
			"burst_capacity_enabled": True,
			"quota_alert_threshold_percent": 85,
			"require_capacity_approval_for_overcommit": True
		},
		"orchestration": {
			"enabled_cloud_providers": ["aws", "azure", "gcp"],
			"multi_cloud_enabled": True,
			"live_migration_enabled": True,
			"dns_validation_required": True
		},
		"analytics": {
			"real_time_analytics_enabled": True,
			"optimization_recommendations_enabled": True,
			"anomaly_detection_enabled": True,
			"provisioning_sla_seconds": 60
		},
		"ui": {
			"enable_dashboard": True,
			"enable_template_library": True,
			"enable_cost_workspace": True,
			"enable_isolation_map": True
		},
		"theme": {
			"default_theme": "mten_control_fabric",
			"allow_tenant_overrides": True
		}
	})
	schema: dict[str, Any] = field(default_factory=lambda: {
		"type": "object",
		"required": [
			"tenant_id",
			"provisioning",
			"isolation",
			"resources",
			"orchestration",
			"analytics",
			"ui",
			"theme"
		],
		"properties": {
			"tenant_id": {"type": "string", "minLength": 1},
			"provisioning": {"type": "object"},
			"isolation": {"type": "object"},
			"resources": {"type": "object"},
			"orchestration": {"type": "object"},
			"analytics": {"type": "object"},
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
	"""Simple MTEN policy rule definition."""

	name: str
	description: str
	condition: dict[str, Any]
	effect: dict[str, Any]


class CapabilityRuleEngine:
	"""Deterministic MTEN rule engine for tenancy and capacity decisions."""

	def __init__(self, rules: list[CapabilityRule] | None = None):
		self.rules = rules or default_rules()

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		"""Evaluate all matching rules against a multi-tenant workload context."""
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

		return {
			"decision": decision,
			"matched_rules": matched,
			"actions": actions,
			"context": context
		}


@dataclass(frozen=True)
class CapabilityUIRoute:
	"""UI route exposed by MTEN."""

	name: str
	path: str
	component: str
	permission: str
	nav_group: str


@dataclass(frozen=True)
class CapabilityTheme:
	"""Visual theme contract for MTEN UI surfaces."""

	name: str = "mten_control_fabric"
	tokens: dict[str, str] = field(default_factory=lambda: {
		"color.primary": "#0F5A5C",
		"color.accent": "#C67B2F",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F5F7FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#102A43",
		"text.secondary": "#52606D",
		"border.radius": "12px",
		"density": "comfortable"
	})
	components: dict[str, dict[str, str]] = field(default_factory=lambda: {
		"tenant_health_card": {
			"icon": "building-2",
			"shape": "rounded-rectangle",
			"status_indicator": "tier-ribbon"
		},
		"provisioning_timeline": {
			"orientation": "horizontal",
			"milestone_style": "stacked-checkpoints"
		},
		"isolation_boundary_map": {
			"visual": "zoned-topology",
			"breach_indicator": "alert-ring"
		},
		"quota_burn_indicator": {
			"visual": "segmented-usage-bar",
			"threshold_highlight": "capacity-band"
		}
	})


def default_rules() -> list[CapabilityRule]:
	"""Default MTEN rules available to every tenant."""
	return [
		CapabilityRule(
			name="tenant_context_required",
			description="All tenant management operations require tenant context.",
			condition={"tenant_context_present": False},
			effect={
				"decision": "deny",
				"reason": "tenant_context_required",
				"required_action": "attach_tenant_context"
			}
		),
		CapabilityRule(
			name="cross_tenant_access_requires_membership",
			description="Cross-tenant operations require confirmed tenant membership.",
			condition={"cross_tenant_operation": True, "tenant_membership_confirmed": False},
			effect={
				"decision": "deny",
				"reason": "tenant_membership_required",
				"required_action": "confirm_tenant_membership"
			}
		),
		CapabilityRule(
			name="suspended_tenants_block_mutations",
			description="Suspended tenants cannot be mutated until reactivated.",
			condition={"tenant_status": "suspended", "requested_operation_is_mutation": True},
			effect={
				"decision": "deny",
				"reason": "tenant_suspended",
				"required_action": "reactivate_tenant_or_abort"
			}
		),
		CapabilityRule(
			name="custom_domain_requires_dns_validation",
			description="Custom domains require ownership validation before activation.",
			condition={"custom_domain_requested": True, "dns_validated": False},
			effect={
				"decision": "deny",
				"reason": "dns_validation_required",
				"required_action": "validate_dns_ownership"
			}
		),
		CapabilityRule(
			name="capacity_overcommit_requires_review",
			description="High projected capacity usage requires explicit approval.",
			condition={"projected_compute_units_gt": 1000, "capacity_approval_recorded": False},
			effect={
				"decision": "require_review",
				"reason": "capacity_review_required",
				"required_action": "record_capacity_approval"
			}
		),
		CapabilityRule(
			name="live_migration_requires_runbook",
			description="Live migrations require an attached runbook before execution.",
			condition={"requested_operation": "live_migration", "runbook_attached": False},
			effect={
				"decision": "deny",
				"reason": "live_migration_runbook_required",
				"required_action": "attach_migration_runbook"
			}
		)
	]


def ui_manifest() -> dict[str, Any]:
	"""Return MTEN UI surface manifest."""
	routes = [
		CapabilityUIRoute("dashboard", "/mten/dashboard", "MultiTenantDashboard", "mten:view", "Overview"),
		CapabilityUIRoute("tenants", "/mten/tenants", "TenantPortfolioView", "mten:view", "Operations"),
		CapabilityUIRoute("provisioning", "/mten/provisioning", "TenantProvisioningPipeline", "mten:provision", "Operations"),
		CapabilityUIRoute("templates", "/mten/templates", "TenantTemplateCatalog", "mten:manage_templates", "Build"),
		CapabilityUIRoute("analytics", "/mten/analytics", "TenantAnalyticsHub", "mten:view_analytics", "Intelligence"),
		CapabilityUIRoute("optimization", "/mten/optimization", "ResourceOptimizationWorkbench", "mten:optimize", "Intelligence"),
		CapabilityUIRoute("settings", "/mten/settings", "TenantGovernanceSettings", "mten:admin", "Administration")
	]
	return {
		"shell": "flask_appbuilder",
		"view_module": "blueprint.py",
		"api_prefix": "/mten/api/v1",
		"routes": [route.__dict__ for route in routes],
		"template_roots": ["templates/", "static/"],
		"requires_theme": True
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable MTEN capability contract."""
	config = CapabilityConfiguration()
	theme = CapabilityTheme()
	return {
		"capability": "mten",
		"display_name": "Multi-Tenant Management",
		"configuration": config.for_tenant(tenant_id, overrides),
		"configuration_schema": config.schema,
		"rule_engine": {
			"type": "deterministic",
			"rules": [rule.__dict__ for rule in default_rules()]
		},
		"ui": ui_manifest(),
		"theme": {
			"name": theme.name,
			"tokens": theme.tokens,
			"components": theme.components
		}
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Convenience wrapper for default MTEN rule evaluation."""
	return CapabilityRuleEngine().evaluate(context)


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_gt"):
			field_name = key[:-3]
			if not context.get(field_name, 0) > expected:
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
