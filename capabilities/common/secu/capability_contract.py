"""
Executable capability contract for APG Security Framework.

SECU is a first-class APG capability: composition tooling can inspect its
tenant configuration, deterministic rules, UI surfaces, and visual theme
without initializing the full security runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CapabilityConfiguration:
	"""Tenant-scoped SECU configuration defaults and schema."""

	defaults: dict[str, Any] = field(default_factory=lambda: {
		"tenant_id": "default",
		"zero_trust": {
			"enabled": True,
			"default_security_level": "confidential",
			"continuous_verification": True,
			"deny_unknown_devices": True
		},
		"risk": {
			"critical_threshold": 90,
			"high_threshold": 70,
			"challenge_threshold": 50,
			"auto_quarantine_threshold": 85
		},
		"threat_detection": {
			"enabled": True,
			"ai_detection_enabled": True,
			"indicator_ttl_hours": 24,
			"alert_on_unknown_threats": True
		},
		"compliance": {
			"frameworks": ["iso_27001", "soc2", "nist"],
			"assessment_interval_days": 30,
			"require_audit_evidence": True
		},
		"ui": {
			"enable_security_dashboard": True,
			"enable_policy_workbench": True,
			"enable_threat_console": True,
			"enable_compliance_console": True
		},
		"theme": {
			"default_theme": "secu_zero_trust",
			"allow_tenant_overrides": True
		}
	})
	schema: dict[str, Any] = field(default_factory=lambda: {
		"type": "object",
		"required": ["tenant_id", "zero_trust", "risk", "threat_detection", "compliance", "ui", "theme"],
		"properties": {
			"tenant_id": {"type": "string", "minLength": 1},
			"zero_trust": {"type": "object"},
			"risk": {"type": "object"},
			"threat_detection": {"type": "object"},
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
	"""Simple SECU policy rule definition."""

	name: str
	description: str
	condition: dict[str, Any]
	effect: dict[str, Any]


class CapabilityRuleEngine:
	"""Deterministic SECU rule engine for security posture decisions."""

	def __init__(self, rules: list[CapabilityRule] | None = None):
		self.rules = rules or default_rules()

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		"""Evaluate all matching rules against a security context."""
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
				elif rule.effect.get("decision") == "quarantine" and decision not in {"deny"}:
					decision = "quarantine"
				elif rule.effect.get("decision") == "challenge" and decision == "allow":
					decision = "challenge"

		return {
			"decision": decision,
			"matched_rules": matched,
			"actions": actions,
			"context": context
		}


@dataclass(frozen=True)
class CapabilityUIRoute:
	"""UI route exposed by SECU."""

	name: str
	path: str
	component: str
	permission: str
	nav_group: str


@dataclass(frozen=True)
class CapabilityTheme:
	"""Visual theme contract for SECU UI surfaces."""

	name: str = "secu_zero_trust"
	tokens: dict[str, str] = field(default_factory=lambda: {
		"color.primary": "#1C3D5A",
		"color.accent": "#C2410C",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#B42318",
		"surface.canvas": "#F6F8FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#102A43",
		"text.secondary": "#52616B",
		"border.radius": "8px",
		"density": "compact"
	})
	components: dict[str, dict[str, str]] = field(default_factory=lambda: {
		"risk_score_meter": {
			"visual": "threshold-gauge",
			"critical_band": "color.danger",
			"safe_band": "color.success"
		},
		"threat_indicator": {
			"icon": "shield-alert",
			"severity_style": "left-border"
		},
		"policy_card": {
			"icon": "lock-keyhole",
			"status_indicator": "top-right"
		},
		"compliance_badge": {
			"icon": "badge-check",
			"variant": "subtle"
		}
	})


def default_rules() -> list[CapabilityRule]:
	"""Default SECU rules available to every tenant."""
	return [
		CapabilityRule(
			name="known_malicious_network_denied",
			description="Known malicious network origins are denied.",
			condition={"is_known_malicious": True},
			effect={
				"decision": "deny",
				"reason": "malicious_network_origin",
				"required_action": "block_request"
			}
		),
		CapabilityRule(
			name="compromised_device_quarantined",
			description="Compromised devices require quarantine.",
			condition={"device_trust": "compromised"},
			effect={
				"decision": "quarantine",
				"reason": "compromised_device",
				"required_action": "isolate_device"
			}
		),
		CapabilityRule(
			name="critical_risk_denied",
			description="Critical risk scores are denied by default.",
			condition={"risk_score_gte": 90},
			effect={
				"decision": "deny",
				"reason": "critical_risk_score",
				"required_action": "investigate_security_event"
			}
		),
		CapabilityRule(
			name="high_risk_requires_challenge",
			description="High risk requests require challenge before access.",
			condition={"risk_score_gte": 70, "challenge_completed": False},
			effect={
				"decision": "challenge",
				"reason": "step_up_required",
				"required_action": "complete_security_challenge"
			}
		),
		CapabilityRule(
			name="compliance_violation_alert",
			description="Compliance violations require audit evidence and alerting.",
			condition={"compliance_violation": True, "audit_evidence_attached": False},
			effect={
				"decision": "challenge",
				"reason": "audit_evidence_required",
				"required_action": "attach_audit_evidence"
			}
		)
	]


def ui_manifest() -> dict[str, Any]:
	"""Return SECU UI surface manifest."""
	routes = [
		CapabilityUIRoute("dashboard", "/secu/dashboard", "SecurityDashboard", "secu:view", "Operations"),
		CapabilityUIRoute("risk", "/secu/risk", "RiskAssessmentConsole", "secu:view_risk", "Operations"),
		CapabilityUIRoute("threats", "/secu/threats", "ThreatDetectionConsole", "secu:view_threats", "Operations"),
		CapabilityUIRoute("policies", "/secu/policies", "SecurityPolicyWorkbench", "secu:manage_policies", "Governance"),
		CapabilityUIRoute("compliance", "/secu/compliance", "ComplianceConsole", "secu:view_compliance", "Governance"),
		CapabilityUIRoute("rules", "/secu/rules", "SecurityRuleWorkbench", "secu:admin", "Governance"),
		CapabilityUIRoute("settings", "/secu/settings", "SecuritySettings", "secu:admin", "Administration")
	]
	return {
		"shell": "flask_appbuilder",
		"frontend_bundle": "views.py",
		"routes": [route.__dict__ for route in routes],
		"template_roots": ["templates/"],
		"requires_theme": True
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable SECU capability contract."""
	config = CapabilityConfiguration()
	theme = CapabilityTheme()
	return {
		"capability": "secu",
		"display_name": "Security Framework",
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
	"""Convenience wrapper for default SECU rule evaluation."""
	return CapabilityRuleEngine().evaluate(context)


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_gte"):
			field_name = key[:-4]
			if not context.get(field_name, 0) >= expected:
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
