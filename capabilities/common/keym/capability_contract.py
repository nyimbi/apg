"""
Executable capability contract for APG Key Management.

KEYM is a first-class APG capability. This module exposes tenant-scoped
configuration, deterministic key-governance rules, UI surfaces, and theme
tokens so composition tooling can integrate with KEYM without initializing the
full key-management runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CapabilityConfiguration:
	"""Tenant-scoped KEYM configuration defaults and schema."""

	defaults: dict[str, Any] = field(default_factory=lambda: {
		"tenant_id": "default",
		"key_domains": {
			"default_domain": "tenant-root",
			"tenant_isolation_required": True,
			"root_keys_require_hsm": True,
			"allow_cross_region_replication": True
		},
		"lifecycle": {
			"default_rotation_days": 90,
			"auto_rotation_enabled": True,
			"compromise_response": "disable_and_rotate",
			"backup_required": True
		},
		"access": {
			"require_tenant_context": True,
			"require_policy_for_key_creation": True,
			"require_dual_control_for_export": True,
			"max_failed_attempts": 3
		},
		"hsm": {
			"software_hsm_enabled": True,
			"hardware_hsm_preferred": True,
			"attestation_required_for_root_keys": True
		},
		"compliance": {
			"frameworks": ["FIPS_140_2", "GDPR", "HIPAA", "PCI_DSS", "ISO_27001"],
			"immutable_audit_required": True,
			"audit_retention_days": 2555
		},
		"automation": {
			"ai_lifecycle_recommendations": True,
			"anomaly_detection_enabled": True,
			"notify_on_policy_violation": True
		},
		"ui": {
			"enable_inventory": True,
			"enable_policy_manager": True,
			"enable_hsm_console": True,
			"enable_audit_viewer": True
		},
		"theme": {
			"default_theme": "keym_vault_console",
			"allow_tenant_overrides": True
		}
	})
	schema: dict[str, Any] = field(default_factory=lambda: {
		"type": "object",
		"required": [
			"tenant_id",
			"key_domains",
			"lifecycle",
			"access",
			"hsm",
			"compliance",
			"automation",
			"ui",
			"theme"
		],
		"properties": {
			"tenant_id": {"type": "string", "minLength": 1},
			"key_domains": {"type": "object"},
			"lifecycle": {"type": "object"},
			"access": {"type": "object"},
			"hsm": {"type": "object"},
			"compliance": {"type": "object"},
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
	"""Simple KEYM policy rule definition."""

	name: str
	description: str
	condition: dict[str, Any]
	effect: dict[str, Any]


class CapabilityRuleEngine:
	"""Deterministic KEYM rule engine for key lifecycle decisions."""

	def __init__(self, rules: list[CapabilityRule] | None = None):
		self.rules = rules or default_rules()

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		"""Evaluate matching key-management governance rules."""
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
	"""UI route exposed by KEYM."""

	name: str
	path: str
	component: str
	permission: str
	nav_group: str


@dataclass(frozen=True)
class CapabilityTheme:
	"""Visual theme contract for KEYM UI surfaces."""

	name: str = "keym_vault_console"
	tokens: dict[str, str] = field(default_factory=lambda: {
		"color.primary": "#24415F",
		"color.accent": "#B7791F",
		"color.success": "#2F855A",
		"color.warning": "#C05621",
		"color.danger": "#B83232",
		"surface.canvas": "#F7F8FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#16202A",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact"
	})
	components: dict[str, dict[str, str]] = field(default_factory=lambda: {
		"key_inventory_row": {
			"icon": "key-round",
			"status_indicator": "lifecycle-pill",
			"risk_style": "right-aligned-score"
		},
		"rotation_timeline": {
			"visual": "deadline-track",
			"threshold_style": "expiry-bands"
		},
		"hsm_attestation_panel": {
			"visual": "signed-attestation-stack",
			"status_style": "seal-chip"
		},
		"policy_violation_trace": {
			"visual": "rule-ladder",
			"highlight": "deny-marker"
		}
	})


def default_rules() -> list[CapabilityRule]:
	"""Default KEYM rules available to every tenant."""
	return [
		CapabilityRule(
			name="tenant_context_required",
			description="All key operations require tenant context.",
			condition={"tenant_context_present": False},
			effect={
				"decision": "deny",
				"reason": "tenant_context_required",
				"required_action": "attach_tenant_context"
			}
		),
		CapabilityRule(
			name="key_creation_requires_policy",
			description="Key creation requires an attached key policy.",
			condition={"operation": "create_key", "policy_attached": False},
			effect={
				"decision": "deny",
				"reason": "key_policy_required",
				"required_action": "attach_key_policy"
			}
		),
		CapabilityRule(
			name="root_key_requires_hsm_attestation",
			description="Root keys require HSM attestation before activation.",
			condition={"key_class": "root", "hsm_attested": False},
			effect={
				"decision": "deny",
				"reason": "hsm_attestation_required",
				"required_action": "complete_hsm_attestation"
			}
		),
		CapabilityRule(
			name="export_requires_dual_control",
			description="Key export requires dual-control approval and wrapping.",
			condition={"operation": "export_key", "dual_control_approved": False},
			effect={
				"decision": "deny",
				"reason": "dual_control_required",
				"required_action": "record_dual_control_approval"
			}
		),
		CapabilityRule(
			name="overdue_rotation_requires_review",
			description="Overdue key rotation requires review before continued use.",
			condition={"rotation_age_days_gt": 90, "rotation_exception_recorded": False},
			effect={
				"decision": "require_review",
				"reason": "rotation_overdue",
				"required_action": "rotate_key_or_record_exception"
			}
		),
		CapabilityRule(
			name="compromised_key_blocks_use",
			description="Compromised keys cannot be used for cryptographic operations.",
			condition={"key_status": "compromised", "operation_is_cryptographic": True},
			effect={
				"decision": "deny",
				"reason": "key_compromised",
				"required_action": "disable_and_rotate_key"
			}
		)
	]


def ui_manifest() -> dict[str, Any]:
	"""Return KEYM UI surface manifest."""
	routes = [
		CapabilityUIRoute("dashboard", "/keym/dashboard", "KeyManagementDashboard", "keym.read_key", "Overview"),
		CapabilityUIRoute("inventory", "/keym/keys", "KeyInventoryView", "keym.read_key", "Operations"),
		CapabilityUIRoute("lifecycle", "/keym/lifecycle", "KeyLifecycleWorkbench", "keym.rotate_key", "Operations"),
		CapabilityUIRoute("policies", "/keym/policies", "PolicyManagerView", "keym.manage_policies", "Governance"),
		CapabilityUIRoute("hsm", "/keym/hsm", "HSMConsole", "keym.manage_hsm", "Security"),
		CapabilityUIRoute("audit", "/keym/audit", "AuditLogsView", "keym.view_audit_logs", "Governance"),
		CapabilityUIRoute("analytics", "/keym/analytics", "SecurityAnalyticsView", "keym.admin", "Intelligence"),
		CapabilityUIRoute("settings", "/keym/settings", "KeyManagementSettings", "keym.admin", "Administration")
	]
	return {
		"shell": "apg_python",
		"view_module": "views.py",
		"api_prefix": "/keym/api/v1",
		"routes": [route.__dict__ for route in routes],
		"template_roots": ["templates/", "static/"],
		"requires_theme": True
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable KEYM capability contract."""
	config = CapabilityConfiguration()
	theme = CapabilityTheme()
	return {
		"capability": "keym",
		"display_name": "Key Management",
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
	"""Convenience wrapper for default KEYM rule evaluation."""
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
