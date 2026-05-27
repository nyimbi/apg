"""
Executable capability contract for APG Connection Management.

Each APG capability should expose capability-specific configuration, a rule
engine, UI surfaces, and visual theming. This module makes that contract
machine-readable for CONN and gives tests/callers a stable integration point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CapabilityConfiguration:
	"""Tenant-scoped CONN configuration defaults and schema."""

	defaults: dict[str, Any] = field(default_factory=lambda: {
		"tenant_id": "default",
		"singer": {
			"default_batch_size": 1000,
			"max_batch_size": 100000,
			"sync_mode": "incremental",
			"health_check_interval_seconds": 60
		},
		"security": {
			"encrypt_credentials": True,
			"audit_enabled": True,
			"require_connection_test_before_activation": True
		},
		"ai": {
			"enabled": True,
			"model": "qwen3:1.7b",
			"schema_mapping_confidence_threshold": 0.75
		},
		"ui": {
			"enable_visual_designer": True,
			"enable_marketplace": True,
			"enable_lineage_view": True,
			"enable_data_quality_view": True
		},
		"theme": {
			"default_theme": "conn_enterprise",
			"allow_tenant_overrides": True
		}
	})
	schema: dict[str, Any] = field(default_factory=lambda: {
		"type": "object",
		"required": ["tenant_id", "singer", "security", "ai", "ui", "theme"],
		"properties": {
			"tenant_id": {"type": "string", "minLength": 1},
			"singer": {"type": "object"},
			"security": {"type": "object"},
			"ai": {"type": "object"},
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
	"""Simple capability rule definition."""

	name: str
	description: str
	condition: dict[str, Any]
	effect: dict[str, Any]


class CapabilityRuleEngine:
	"""Deterministic rule engine for CONN policy and workflow decisions."""

	def __init__(self, rules: list[CapabilityRule] | None = None):
		self.rules = rules or default_rules()

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		"""Evaluate all matching rules against a connection context."""
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

		return {
			"decision": decision,
			"matched_rules": matched,
			"actions": actions,
			"context": context
		}


@dataclass(frozen=True)
class CapabilityUIRoute:
	"""UI route exposed by the capability."""

	name: str
	path: str
	component: str
	permission: str
	nav_group: str


@dataclass(frozen=True)
class CapabilityTheme:
	"""Visual theme contract for CONN UI surfaces."""

	name: str = "conn_enterprise"
	tokens: dict[str, str] = field(default_factory=lambda: {
		"color.primary": "#176B87",
		"color.accent": "#8B5E34",
		"color.success": "#2E7D32",
		"color.warning": "#B26A00",
		"color.danger": "#B42318",
		"surface.canvas": "#F7F9FB",
		"surface.panel": "#FFFFFF",
		"text.primary": "#1F2933",
		"text.secondary": "#52616B",
		"border.radius": "8px",
		"density": "compact"
	})
	components: dict[str, dict[str, str]] = field(default_factory=lambda: {
		"connection_node": {
			"icon": "plug",
			"shape": "rounded-rectangle",
			"status_indicator": "left-border"
		},
		"data_flow_edge": {
			"line_style": "solid",
			"animated_when_active": "true"
		},
		"rule_badge": {
			"icon": "shield-check",
			"variant": "subtle"
		}
	})


def default_rules() -> list[CapabilityRule]:
	"""Default CONN rules available to every tenant."""
	return [
		CapabilityRule(
			name="require_connection_test_before_activation",
			description="Connections must pass a test before activation.",
			condition={"requested_status": "active", "last_test_passed": False},
			effect={
				"decision": "deny",
				"reason": "connection_test_required",
				"required_action": "run_connection_test"
			}
		),
		CapabilityRule(
			name="encrypt_credentials",
			description="Credential-bearing connectors require encrypted storage.",
			condition={"contains_credentials": True, "credentials_encrypted": False},
			effect={
				"decision": "deny",
				"reason": "credentials_must_be_encrypted",
				"required_action": "enable_encryption"
			}
		),
		CapabilityRule(
			name="large_batch_requires_monitoring",
			description="Large synchronization batches require monitoring.",
			condition={"batch_size_gt": 10000, "monitoring_enabled": False},
			effect={
				"decision": "deny",
				"reason": "large_batch_requires_monitoring",
				"required_action": "enable_monitoring"
			}
		)
	]


def ui_manifest() -> dict[str, Any]:
	"""Return CONN UI surface manifest."""
	routes = [
		CapabilityUIRoute("dashboard", "/conn/dashboard", "ConnectionDashboard", "conn:view", "Operations"),
		CapabilityUIRoute("designer", "/conn/designer", "VisualFlowDesigner", "conn:create", "Build"),
		CapabilityUIRoute("marketplace", "/conn/marketplace", "ConnectorMarketplace", "conn:view", "Extend"),
		CapabilityUIRoute("lineage", "/conn/lineage", "DataLineageView", "conn:view", "Governance"),
		CapabilityUIRoute("data_quality", "/conn/data-quality", "DataQualityWorkbench", "conn:view", "Governance"),
		CapabilityUIRoute("rules", "/conn/rules", "CapabilityRuleWorkbench", "conn:admin", "Governance"),
		CapabilityUIRoute("settings", "/conn/settings", "CapabilitySettings", "conn:admin", "Administration")
	]
	return {
		"shell": "apg_python",
		"frontend_bundle": "frontend/src/App.tsx",
		"routes": [route.__dict__ for route in routes],
		"template_roots": ["templates/", "frontend/"],
		"requires_theme": True
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable CONN capability contract."""
	config = CapabilityConfiguration()
	theme = CapabilityTheme()
	return {
		"capability": "conn",
		"display_name": "Connection Management",
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
	"""Convenience wrapper for default CONN rule evaluation."""
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
		copied[key] = _deep_copy(item) if isinstance(item, dict) else item
	return copied


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
	for key, value in source.items():
		if isinstance(value, dict) and isinstance(target.get(key), dict):
			_deep_merge(target[key], value)
		else:
			target[key] = value
