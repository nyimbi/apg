"""
Executable capability contract for APG Authentication & RBAC.

AUTH is a first-class APG capability. This module exposes tenant-scoped
configuration, deterministic authorization/authentication rules, UI surfaces,
and theme tokens so composition tooling can reason about AUTH without loading
the full runtime stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CapabilityConfiguration:
	"""Tenant-scoped AUTH configuration defaults and schema."""

	defaults: dict[str, Any] = field(default_factory=lambda: {
		"tenant_id": "default",
		"authentication": {
			"enable_behavioral_auth": True,
			"enable_biometric_fusion": True,
			"enable_quantum_safe_auth": True,
			"enable_zero_knowledge_proofs": True,
			"require_mfa_for_privileged_access": True
		},
		"authorization": {
			"default_access_model": "hybrid",
			"enforce_least_privilege": True,
			"enable_role_templates": True,
			"require_approval_for_admin_role_assignment": True
		},
		"sessions": {
			"idle_timeout_minutes": 30,
			"absolute_timeout_hours": 8,
			"require_device_binding": True,
			"continuous_risk_evaluation": True
		},
		"federation": {
			"mesh_enabled": True,
			"require_trusted_issuer": True,
			"allow_cross_tenant_federation": False,
			"minimum_trust_level": "medium"
		},
		"privacy": {
			"enable_privacy_analytics": True,
			"default_privacy_budget": 1.0,
			"analytics_retention_days": 90,
			"allow_behavioral_data_export": False
		},
		"ui": {
			"enable_trust_dashboard": True,
			"enable_role_workbench": True,
			"enable_behavioral_console": True,
			"enable_federation_console": True
		},
		"theme": {
			"default_theme": "auth_trust_fabric",
			"allow_tenant_overrides": True
		}
	})
	schema: dict[str, Any] = field(default_factory=lambda: {
		"type": "object",
		"required": [
			"tenant_id",
			"authentication",
			"authorization",
			"sessions",
			"federation",
			"privacy",
			"ui",
			"theme"
		],
		"properties": {
			"tenant_id": {"type": "string", "minLength": 1},
			"authentication": {"type": "object"},
			"authorization": {"type": "object"},
			"sessions": {"type": "object"},
			"federation": {"type": "object"},
			"privacy": {"type": "object"},
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
	"""Simple AUTH policy rule definition."""

	name: str
	description: str
	condition: dict[str, Any]
	effect: dict[str, Any]


class CapabilityRuleEngine:
	"""Deterministic AUTH rule engine for access and session decisions."""

	def __init__(self, rules: list[CapabilityRule] | None = None):
		self.rules = rules or default_rules()

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		"""Evaluate all matching rules against an authentication context."""
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
	"""UI route exposed by AUTH."""

	name: str
	path: str
	component: str
	permission: str
	nav_group: str


@dataclass(frozen=True)
class CapabilityTheme:
	"""Visual theme contract for AUTH UI surfaces."""

	name: str = "auth_trust_fabric"
	tokens: dict[str, str] = field(default_factory=lambda: {
		"color.primary": "#12344D",
		"color.accent": "#0F8B8D",
		"color.success": "#2D6A4F",
		"color.warning": "#B7791F",
		"color.danger": "#C05621",
		"surface.canvas": "#F4F7FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#102A43",
		"text.secondary": "#486581",
		"border.radius": "12px",
		"density": "comfortable"
	})
	components: dict[str, dict[str, str]] = field(default_factory=lambda: {
		"identity_signal_card": {
			"icon": "shield-lock",
			"shape": "rounded-rectangle",
			"status_indicator": "trust-ring"
		},
		"risk_posture_meter": {
			"visual": "segmented-gauge",
			"highlight": "threshold-bands"
		},
		"role_assignment_timeline": {
			"line_style": "stepped",
			"approval_badge": "inline"
		},
		"session_trust_badge": {
			"icon": "fingerprint",
			"variant": "elevated"
		}
	})


def default_rules() -> list[CapabilityRule]:
	"""Default AUTH rules available to every tenant."""
	return [
		CapabilityRule(
			name="locked_accounts_denied",
			description="Locked or suspended accounts cannot authenticate.",
			condition={"user_locked": True},
			effect={
				"decision": "deny",
				"reason": "account_locked",
				"required_action": "unlock_or_reset_account"
			}
		),
		CapabilityRule(
			name="privileged_access_requires_mfa",
			description="Privileged access requires MFA verification.",
			condition={"requested_permission_tier": "privileged", "mfa_verified": False},
			effect={
				"decision": "deny",
				"reason": "mfa_required",
				"required_action": "complete_mfa_challenge"
			}
		),
		CapabilityRule(
			name="high_risk_sessions_require_step_up",
			description="High-risk sessions require step-up authentication.",
			condition={"risk_level": "high", "step_up_completed": False},
			effect={
				"decision": "deny",
				"reason": "step_up_authentication_required",
				"required_action": "perform_step_up_authentication"
			}
		),
		CapabilityRule(
			name="elevated_role_assignment_requires_approval",
			description="Administrative role assignments require recorded approval.",
			condition={"requested_operation": "assign_role", "role_tier": "admin", "approval_recorded": False},
			effect={
				"decision": "deny",
				"reason": "approval_required_for_admin_role_assignment",
				"required_action": "record_role_assignment_approval"
			}
		),
		CapabilityRule(
			name="untrusted_federation_denied",
			description="Federated logins require a trusted issuer.",
			condition={"auth_source": "federated", "issuer_trusted": False},
			effect={
				"decision": "deny",
				"reason": "trusted_issuer_required",
				"required_action": "approve_or_rotate_identity_provider"
			}
		),
		CapabilityRule(
			name="cross_tenant_access_requires_membership",
			description="Cross-tenant access requires confirmed tenant membership.",
			condition={"tenant_mismatch": True, "tenant_membership_confirmed": False},
			effect={
				"decision": "deny",
				"reason": "tenant_membership_required",
				"required_action": "confirm_tenant_membership"
			}
		),
		CapabilityRule(
			name="privacy_queries_require_budget",
			description="Privacy-preserving analytics queries require remaining budget.",
			condition={"requested_operation": "privacy_analytics_query", "privacy_budget_available": False},
			effect={
				"decision": "require_review",
				"reason": "privacy_budget_exhausted",
				"required_action": "approve_budget_replenishment"
			}
		)
	]


def ui_manifest() -> dict[str, Any]:
	"""Return AUTH UI surface manifest."""
	routes = [
		CapabilityUIRoute("login", "/auth/revolutionary/login", "RevolutionaryLoginScreen", "public", "Access"),
		CapabilityUIRoute("dashboard", "/auth/revolutionary/dashboard", "RevolutionaryAuthenticationDashboard", "auth:view", "Overview"),
		CapabilityUIRoute("biometric_enrollment", "/auth/biometric/enroll", "BiometricEnrollmentStudio", "auth:manage_biometrics", "Assurance"),
		CapabilityUIRoute("biometric_management", "/auth/biometric/manage", "BiometricManagementConsole", "auth:manage_biometrics", "Assurance"),
		CapabilityUIRoute("quantum_keys", "/auth/quantum/keys", "QuantumKeyVault", "auth:manage_keys", "Cryptography"),
		CapabilityUIRoute("quantum_generate", "/auth/quantum/generate", "QuantumKeyGenerationFlow", "auth:manage_keys", "Cryptography"),
		CapabilityUIRoute("behavioral_analysis", "/auth/behavioral/analysis", "BehavioralTrustWorkbench", "auth:view_risk", "Intelligence"),
		CapabilityUIRoute("behavioral_training", "/auth/behavioral/training", "BehavioralBaselineTraining", "auth:view_risk", "Intelligence"),
		CapabilityUIRoute("privacy_settings", "/auth/privacy/settings", "PrivacyPreferenceCenter", "auth:manage_privacy", "Governance"),
		CapabilityUIRoute("privacy_analytics", "/auth/privacy/analytics", "PrivacyAnalyticsCenter", "auth:view_privacy", "Governance"),
		CapabilityUIRoute("neuromorphic_dashboard", "/auth/neuromorphic/dashboard", "NeuromorphicDecisionConsole", "auth:view_risk", "Intelligence"),
		CapabilityUIRoute("federated_mesh", "/auth/federated/mesh", "FederatedIdentityMeshConsole", "auth:manage_federation", "Federation"),
		CapabilityUIRoute("metrics", "/auth/metrics/overview", "AuthenticationMetricsOverview", "auth:admin", "Operations")
	]
	return {
		"shell": "flask_appbuilder",
		"view_module": "views.py",
		"api_prefix": "/api",
		"routes": [route.__dict__ for route in routes],
		"template_roots": ["templates/auth/"],
		"requires_theme": True
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable AUTH capability contract."""
	config = CapabilityConfiguration()
	theme = CapabilityTheme()
	return {
		"capability": "auth",
		"display_name": "Authentication & RBAC",
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
	"""Convenience wrapper for default AUTH rule evaluation."""
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
