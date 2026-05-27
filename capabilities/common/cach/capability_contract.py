"""
Executable capability contract for APG Cache Management.

CACH is a first-class APG capability. This module exposes tenant-scoped
configuration, deterministic cache-governance rules, UI surfaces, and theme
tokens so composition tooling can integrate with CACH without loading optional
compression or UI runtime dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CapabilityConfiguration:
	"""Tenant-scoped CACH configuration defaults and schema."""

	defaults: dict[str, Any] = field(default_factory=lambda: {
		"tenant_id": "default",
		"hierarchy": {
			"default_tier": "memory",
			"tiers": ["memory", "distributed", "edge"],
			"max_memory_mb": 1024,
			"multi_tenant_isolation": True
		},
		"policy": {
			"default_ttl_seconds": 3600,
			"default_eviction_policy": "adaptive_lru",
			"namespace_required": True,
			"critical_reads_require_freshness": True
		},
		"warming": {
			"predictive_prefetching_enabled": True,
			"source_registration_required": True,
			"max_warming_batch_size": 10000
		},
		"security": {
			"require_tenant_context": True,
			"sensitive_entries_require_encryption": True,
			"cross_tenant_access_allowed": False,
			"quantum_ready_security": True
		},
		"optimization": {
			"ai_optimization_enabled": True,
			"memory_pressure_review_threshold_percent": 90,
			"auto_evict_on_pressure": True
		},
		"telemetry": {
			"metrics_enabled": True,
			"track_access_patterns": True,
			"emit_cache_events": True
		},
		"ui": {
			"enable_dashboard": True,
			"enable_policy_manager": True,
			"enable_warming_console": True,
			"enable_hierarchy_map": True
		},
		"theme": {
			"default_theme": "cach_memory_fabric",
			"allow_tenant_overrides": True
		}
	})
	schema: dict[str, Any] = field(default_factory=lambda: {
		"type": "object",
		"required": [
			"tenant_id",
			"hierarchy",
			"policy",
			"warming",
			"security",
			"optimization",
			"telemetry",
			"ui",
			"theme"
		],
		"properties": {
			"tenant_id": {"type": "string", "minLength": 1},
			"hierarchy": {"type": "object"},
			"policy": {"type": "object"},
			"warming": {"type": "object"},
			"security": {"type": "object"},
			"optimization": {"type": "object"},
			"telemetry": {"type": "object"},
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
	"""Simple CACH policy rule definition."""

	name: str
	description: str
	condition: dict[str, Any]
	effect: dict[str, Any]


class CapabilityRuleEngine:
	"""Deterministic CACH rule engine for cache governance decisions."""

	def __init__(self, rules: list[CapabilityRule] | None = None):
		self.rules = rules or default_rules()

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		"""Evaluate matching cache governance rules."""
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
	"""UI route exposed by CACH."""

	name: str
	path: str
	component: str
	permission: str
	nav_group: str


@dataclass(frozen=True)
class CapabilityTheme:
	"""Visual theme contract for CACH UI surfaces."""

	name: str = "cach_memory_fabric"
	tokens: dict[str, str] = field(default_factory=lambda: {
		"color.primary": "#31572C",
		"color.accent": "#4D908E",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F7FAF8",
		"surface.panel": "#FFFFFF",
		"text.primary": "#152017",
		"text.secondary": "#53635A",
		"border.radius": "8px",
		"density": "compact"
	})
	components: dict[str, dict[str, str]] = field(default_factory=lambda: {
		"cache_hit_card": {
			"icon": "gauge",
			"status_indicator": "hit-rate-pill",
			"risk_style": "trend-sparkline"
		},
		"tier_hierarchy_map": {
			"visual": "layered-topology",
			"edge_style": "promotion-path"
		},
		"warming_plan_timeline": {
			"visual": "batch-timeline",
			"threshold_style": "source-readiness"
		},
		"namespace_policy_trace": {
			"visual": "rule-ladder",
			"highlight": "ttl-chip"
		}
	})


def default_rules() -> list[CapabilityRule]:
	"""Default CACH rules available to every tenant."""
	return [
		CapabilityRule(
			name="tenant_context_required",
			description="All cache operations require tenant context.",
			condition={"tenant_context_present": False},
			effect={
				"decision": "deny",
				"reason": "tenant_context_required",
				"required_action": "attach_tenant_context"
			}
		),
		CapabilityRule(
			name="write_requires_namespace",
			description="Cache writes require an explicit namespace.",
			condition={"operation": "write", "namespace_present": False},
			effect={
				"decision": "deny",
				"reason": "namespace_required",
				"required_action": "select_cache_namespace"
			}
		),
		CapabilityRule(
			name="sensitive_entry_requires_encryption",
			description="Sensitive cache entries require encryption at rest.",
			condition={"data_classification": "sensitive", "entry_encrypted": False},
			effect={
				"decision": "deny",
				"reason": "cache_entry_encryption_required",
				"required_action": "enable_entry_encryption"
			}
		),
		CapabilityRule(
			name="cross_tenant_cache_access_denied",
			description="Cross-tenant cache access is denied by default.",
			condition={"cross_tenant_access": True},
			effect={
				"decision": "deny",
				"reason": "cross_tenant_cache_access_denied",
				"required_action": "use_tenant_scoped_namespace"
			}
		),
		CapabilityRule(
			name="critical_stale_read_requires_refresh",
			description="Critical stale reads require refresh before serving.",
			condition={"operation": "read", "data_criticality": "critical", "entry_stale": True},
			effect={
				"decision": "deny",
				"reason": "critical_entry_refresh_required",
				"required_action": "refresh_cache_entry"
			}
		),
		CapabilityRule(
			name="high_memory_pressure_requires_review",
			description="High memory pressure requires eviction or review.",
			condition={"memory_utilization_percent_gt": 90, "eviction_plan_ready": False},
			effect={
				"decision": "require_review",
				"reason": "memory_pressure_review_required",
				"required_action": "prepare_eviction_or_capacity_plan"
			}
		)
	]


def ui_manifest() -> dict[str, Any]:
	"""Return CACH UI surface manifest."""
	routes = [
		CapabilityUIRoute("dashboard", "/cach/dashboard", "CacheDashboard", "cach:view", "Overview"),
		CapabilityUIRoute("entries", "/cach/entries", "CacheEntryExplorer", "cach:read", "Operations"),
		CapabilityUIRoute("policies", "/cach/policies", "CachePolicyManager", "cach:manage_policies", "Governance"),
		CapabilityUIRoute("warming", "/cach/warming", "CacheWarmingConsole", "cach:warm", "Operations"),
		CapabilityUIRoute("hierarchy", "/cach/hierarchy", "CacheHierarchyMap", "cach:view", "Architecture"),
		CapabilityUIRoute("analytics", "/cach/analytics", "CacheAnalytics", "cach:view_analytics", "Intelligence"),
		CapabilityUIRoute("security", "/cach/security", "CacheSecurityView", "cach:admin", "Governance"),
		CapabilityUIRoute("settings", "/cach/settings", "CacheSettings", "cach:admin", "Administration")
	]
	return {
		"shell": "apg_python",
		"view_module": "dashboard.py",
		"api_prefix": "/cach/api/v1",
		"routes": [route.__dict__ for route in routes],
		"template_roots": ["templates/", "static/"],
		"requires_theme": True
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable CACH capability contract."""
	config = CapabilityConfiguration()
	theme = CapabilityTheme()
	return {
		"capability": "cach",
		"display_name": "Cache Management",
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
	"""Convenience wrapper for default CACH rule evaluation."""
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
