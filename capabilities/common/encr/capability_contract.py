"""
Executable capability contract for APG Encryption Services.

ENCR is a first-class APG capability. This module exposes tenant-scoped
configuration, deterministic cryptographic governance rules, UI surfaces, and
theme tokens so composition tooling can integrate with ENCR without loading the
full encryption runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CapabilityConfiguration:
	"""Tenant-scoped ENCR configuration defaults and schema."""

	defaults: dict[str, Any] = field(default_factory=lambda: {
		"tenant_id": "default",
		"cryptography": {
			"default_symmetric_algorithm": "AES-256-GCM",
			"post_quantum_enabled": True,
			"zero_knowledge_enabled": True,
			"homomorphic_computation_enabled": True,
			"minimum_entropy_quality": 0.95
		},
		"key_lifecycle": {
			"autonomous_rotation_enabled": True,
			"default_rotation_days": 90,
			"tenant_key_domains_required": True,
			"external_key_manager": "keym"
		},
		"policy": {
			"require_tenant_context": True,
			"deny_plaintext_exports": True,
			"require_quantum_safe_for_restricted_data": True,
			"require_review_for_legacy_algorithms": True
		},
		"threat_adaptive": {
			"enabled": True,
			"escalate_on_active_threat": True,
			"rotate_keys_on_compromise_signal": True
		},
		"compliance": {
			"frameworks": ["GDPR", "HIPAA", "PCI_DSS", "FIPS_140_2"],
			"audit_all_crypto_operations": True,
			"evidence_retention_days": 2555
		},
		"ui": {
			"enable_dashboard": True,
			"enable_policy_designer": True,
			"enable_entropy_console": True,
			"enable_homomorphic_workspace": True
		},
		"theme": {
			"default_theme": "encr_quantum_guard",
			"allow_tenant_overrides": True
		}
	})
	schema: dict[str, Any] = field(default_factory=lambda: {
		"type": "object",
		"required": [
			"tenant_id",
			"cryptography",
			"key_lifecycle",
			"policy",
			"threat_adaptive",
			"compliance",
			"ui",
			"theme"
		],
		"properties": {
			"tenant_id": {"type": "string", "minLength": 1},
			"cryptography": {"type": "object"},
			"key_lifecycle": {"type": "object"},
			"policy": {"type": "object"},
			"threat_adaptive": {"type": "object"},
			"compliance": {"type": "object"},
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
	"""Simple ENCR policy rule definition."""

	name: str
	description: str
	condition: dict[str, Any]
	effect: dict[str, Any]


class CapabilityRuleEngine:
	"""Deterministic ENCR rule engine for cryptographic control decisions."""

	def __init__(self, rules: list[CapabilityRule] | None = None):
		self.rules = rules or default_rules()

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		"""Evaluate matching cryptographic governance rules."""
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
	"""UI route exposed by ENCR."""

	name: str
	path: str
	component: str
	permission: str
	nav_group: str


@dataclass(frozen=True)
class CapabilityTheme:
	"""Visual theme contract for ENCR UI surfaces."""

	name: str = "encr_quantum_guard"
	tokens: dict[str, str] = field(default_factory=lambda: {
		"color.primary": "#143C5C",
		"color.accent": "#23A6A6",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F4F7FB",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#53627A",
		"border.radius": "8px",
		"density": "compact"
	})
	components: dict[str, dict[str, str]] = field(default_factory=lambda: {
		"crypto_posture_card": {
			"icon": "shield-check",
			"status_indicator": "algorithm-badge",
			"risk_style": "left-rail"
		},
		"entropy_quality_meter": {
			"visual": "segmented-meter",
			"threshold_style": "quality-bands"
		},
		"policy_decision_trace": {
			"visual": "stacked-rule-list",
			"highlight": "decision-chip"
		},
		"homomorphic_workspace": {
			"visual": "locked-data-flow",
			"result_style": "sealed-output"
		}
	})


def default_rules() -> list[CapabilityRule]:
	"""Default ENCR rules available to every tenant."""
	return [
		CapabilityRule(
			name="tenant_context_required",
			description="All encryption operations require tenant context.",
			condition={"tenant_context_present": False},
			effect={
				"decision": "deny",
				"reason": "tenant_context_required",
				"required_action": "attach_tenant_context"
			}
		),
		CapabilityRule(
			name="restricted_data_requires_quantum_safe_algorithm",
			description="Restricted data must use quantum-safe encryption.",
			condition={"data_classification": "restricted", "algorithm_quantum_safe": False},
			effect={
				"decision": "deny",
				"reason": "quantum_safe_algorithm_required",
				"required_action": "select_quantum_safe_algorithm"
			}
		),
		CapabilityRule(
			name="plaintext_export_blocked",
			description="Plaintext export requests are blocked by default.",
			condition={"plaintext_export_requested": True},
			effect={
				"decision": "deny",
				"reason": "plaintext_export_blocked",
				"required_action": "use_wrapped_or_encrypted_export"
			}
		),
		CapabilityRule(
			name="low_entropy_blocks_key_generation",
			description="Key generation requires high-quality entropy.",
			condition={"entropy_quality_lt": 0.95, "operation": "generate_key"},
			effect={
				"decision": "deny",
				"reason": "entropy_quality_too_low",
				"required_action": "refresh_entropy_source"
			}
		),
		CapabilityRule(
			name="legacy_algorithm_requires_review",
			description="Legacy cryptographic algorithms require explicit review.",
			condition={"algorithm_family": "legacy", "security_review_recorded": False},
			effect={
				"decision": "require_review",
				"reason": "legacy_algorithm_review_required",
				"required_action": "record_crypto_exception"
			}
		),
		CapabilityRule(
			name="active_threat_requires_key_rotation",
			description="Active threat signals require key rotation before sensitive operations.",
			condition={"active_threat_signal": True, "key_rotation_completed": False},
			effect={
				"decision": "deny",
				"reason": "threat_adaptive_rotation_required",
				"required_action": "rotate_affected_keys"
			}
		)
	]


def ui_manifest() -> dict[str, Any]:
	"""Return ENCR UI surface manifest."""
	routes = [
		CapabilityUIRoute("dashboard", "/encr/dashboard", "EncryptionDashboard", "encr:view", "Overview"),
		CapabilityUIRoute("operations", "/encr/operations", "CryptoOperationsConsole", "encr:operate", "Operations"),
		CapabilityUIRoute("keys", "/encr/keys", "EncryptionKeyDomains", "encr:view_keys", "Operations"),
		CapabilityUIRoute("policies", "/encr/policies", "CryptoPolicyDesigner", "encr:manage_policies", "Governance"),
		CapabilityUIRoute("entropy", "/encr/entropy", "EntropyQualityConsole", "encr:view_entropy", "Governance"),
		CapabilityUIRoute("homomorphic", "/encr/homomorphic", "HomomorphicWorkspace", "encr:compute", "Advanced"),
		CapabilityUIRoute("analytics", "/encr/analytics", "CryptoAnalytics", "encr:view_analytics", "Intelligence"),
		CapabilityUIRoute("settings", "/encr/settings", "EncryptionSettings", "encr:admin", "Administration")
	]
	return {
		"shell": "apg_python",
		"view_module": "web_ui.py",
		"api_prefix": "/encr/api/v1",
		"routes": [route.__dict__ for route in routes],
		"template_roots": ["templates/", "static/"],
		"requires_theme": True
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable ENCR capability contract."""
	config = CapabilityConfiguration()
	theme = CapabilityTheme()
	return {
		"capability": "encr",
		"display_name": "Encryption Services",
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
	"""Convenience wrapper for default ENCR rule evaluation."""
	return CapabilityRuleEngine().evaluate(context)


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_gt"):
			field_name = key[:-3]
			if not context.get(field_name, 0) > expected:
				return False
		elif key.endswith("_lt"):
			field_name = key[:-3]
			if not context.get(field_name, 0) < expected:
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
