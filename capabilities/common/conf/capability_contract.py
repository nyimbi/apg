"""
Executable capability contract for APG Configuration Management.

This module exposes a machine-readable contract for the CONF capability so the
APG composition layer can discover tenant configuration, deterministic rules,
UI surfaces, and theming without importing the full runtime stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CapabilityConfiguration:
	"""Tenant-scoped CONF configuration defaults and schema."""

	defaults: dict[str, Any] = field(default_factory=lambda: {
		"tenant_id": "default",
		"gitops": {
			"default_branch": "main",
			"auto_create_pull_requests": True,
			"drift_detection_interval_seconds": 300,
			"enforce_signed_commits": True
		},
		"security": {
			"require_secret_encryption": True,
			"require_change_approval_for_production": True,
			"record_audit_trail": True
		},
		"automation": {
			"enable_ai_assistance": True,
			"default_deployment_strategy": "rolling",
			"auto_remediation_enabled": False,
			"validation_mode": "strict"
		},
		"ui": {
			"enable_topology_view": True,
			"enable_policy_workbench": True,
			"enable_drift_dashboard": True,
			"enable_gitops_center": True
		},
		"theme": {
			"default_theme": "conf_control_room",
			"allow_tenant_overrides": True
		}
	})
	schema: dict[str, Any] = field(default_factory=lambda: {
		"type": "object",
		"required": ["tenant_id", "gitops", "security", "automation", "ui", "theme"],
		"properties": {
			"tenant_id": {"type": "string", "minLength": 1},
			"gitops": {"type": "object"},
			"security": {"type": "object"},
			"automation": {"type": "object"},
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
	"""Deterministic rule engine for CONF governance and rollout decisions."""

	def __init__(self, rules: list[CapabilityRule] | None = None):
		self.rules = rules or default_rules()

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		"""Evaluate all matching rules against a configuration change context."""
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
	"""Visual theme contract for CONF UI surfaces."""

	name: str = "conf_control_room"
	tokens: dict[str, str] = field(default_factory=lambda: {
		"color.primary": "#0F766E",
		"color.accent": "#A16207",
		"color.success": "#2F855A",
		"color.warning": "#C2410C",
		"color.danger": "#B91C1C",
		"surface.canvas": "#F4F7F6",
		"surface.panel": "#FFFFFF",
		"text.primary": "#16302B",
		"text.secondary": "#4B635D",
		"border.radius": "10px",
		"density": "comfortable"
	})
	components: dict[str, dict[str, str]] = field(default_factory=lambda: {
		"configuration_resource_card": {
			"icon": "server-stack",
			"shape": "rounded-rectangle",
			"status_indicator": "top-bar"
		},
		"deployment_timeline": {
			"line_style": "segmented",
			"highlight_current_stage": "true"
		},
		"drift_badge": {
			"icon": "radar",
			"variant": "attention"
		}
	})


def default_rules() -> list[CapabilityRule]:
	"""Default CONF rules available to every tenant."""
	return [
		CapabilityRule(
			name="validate_before_apply",
			description="Configuration changes must pass validation before apply.",
			condition={"requested_operation": "apply", "validation_passed": False},
			effect={
				"decision": "deny",
				"reason": "validation_required",
				"required_action": "run_validation"
			}
		),
		CapabilityRule(
			name="production_changes_require_approval",
			description="Production changes require explicit approval.",
			condition={"target_environment": "production", "change_approved": False},
			effect={
				"decision": "deny",
				"reason": "production_approval_required",
				"required_action": "collect_change_approval"
			}
		),
		CapabilityRule(
			name="encrypted_secrets_required",
			description="Secret-bearing configurations require encrypted secret storage.",
			condition={"contains_secrets": True, "secrets_encrypted": False},
			effect={
				"decision": "deny",
				"reason": "secret_encryption_required",
				"required_action": "encrypt_secrets"
			}
		),
		CapabilityRule(
			name="drift_requires_remediation_plan",
			description="Drifted resources need a remediation plan before rollout.",
			condition={"drift_detected": True, "remediation_plan_available": False},
			effect={
				"decision": "deny",
				"reason": "drift_remediation_required",
				"required_action": "generate_remediation_plan"
			}
		)
	]


def ui_manifest() -> dict[str, Any]:
	"""Return CONF UI surface manifest."""
	routes = [
		CapabilityUIRoute("dashboard", "/config/dashboard", "ConfigurationDashboard", "conf:view", "Operations"),
		CapabilityUIRoute("resources", "/config/resources", "ConfigurationResourceCatalog", "conf:view", "Author"),
		CapabilityUIRoute("templates", "/config/templates", "ConfigurationTemplateLibrary", "conf:create", "Author"),
		CapabilityUIRoute("policies", "/config/policies", "ConfigurationPolicyWorkbench", "conf:admin", "Governance"),
		CapabilityUIRoute("deployments", "/config/deployments", "ConfigurationDeploymentCenter", "conf:deploy", "Operations"),
		CapabilityUIRoute("drift", "/config/drift", "ConfigurationDriftConsole", "conf:view", "Governance"),
		CapabilityUIRoute("gitops", "/config/gitops", "ConfigurationGitOpsCenter", "conf:deploy", "Operations"),
		CapabilityUIRoute("settings", "/config/settings", "ConfigurationSettings", "conf:admin", "Administration")
	]
	return {
		"shell": "apg_python",
		"blueprint_module": "blueprints/blueprint.py",
		"api_prefix": "/api/v1/config",
		"routes": [route.__dict__ for route in routes],
		"template_roots": ["blueprints/"],
		"requires_theme": True
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable CONF capability contract."""
	config = CapabilityConfiguration()
	theme = CapabilityTheme()
	return {
		"capability": "conf",
		"display_name": "Configuration Management",
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
	"""Convenience wrapper for default CONF rule evaluation."""
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
