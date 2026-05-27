"""
Executable capability contract for APG System Health Management.

HLTH is a first-class APG capability. This module exposes tenant-scoped
configuration, deterministic health-governance rules, UI surfaces, and theme
tokens so composition tooling can integrate with HLTH without starting the
health runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CapabilityConfiguration:
	"""Tenant-scoped HLTH configuration defaults and schema."""

	defaults: dict[str, Any] = field(default_factory=lambda: {
		"tenant_id": "default",
		"assessment": {
			"health_check_interval_seconds": 60,
			"component_discovery_enabled": True,
			"contextual_health_scoring": True,
			"business_impact_weighting": True
		},
		"baselines": {
			"learning_period_days": 7,
			"auto_update_enabled": True,
			"stale_baseline_days": 30
		},
		"alerts": {
			"correlation_window_minutes": 5,
			"critical_health_score_threshold": 40,
			"auto_acknowledge_recovered_alerts": True
		},
		"prediction": {
			"prediction_window_hours": 24,
			"failure_forecast_enabled": True,
			"minimum_prediction_confidence": 0.75
		},
		"remediation": {
			"auto_remediation_enabled": True,
			"runbook_required": True,
			"production_approval_required": True
		},
		"incidents": {
			"block_deploy_on_unresolved_critical": True,
			"incident_owner_required": True,
			"postmortem_required_for_sev1": True
		},
		"ui": {
			"enable_dashboard": True,
			"enable_component_map": True,
			"enable_prediction_console": True,
			"enable_remediation_console": True
		},
		"theme": {
			"default_theme": "hlth_health_console",
			"allow_tenant_overrides": True
		}
	})
	schema: dict[str, Any] = field(default_factory=lambda: {
		"type": "object",
		"required": [
			"tenant_id",
			"assessment",
			"baselines",
			"alerts",
			"prediction",
			"remediation",
			"incidents",
			"ui",
			"theme"
		],
		"properties": {
			"tenant_id": {"type": "string", "minLength": 1},
			"assessment": {"type": "object"},
			"baselines": {"type": "object"},
			"alerts": {"type": "object"},
			"prediction": {"type": "object"},
			"remediation": {"type": "object"},
			"incidents": {"type": "object"},
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
	"""Simple HLTH policy rule definition."""

	name: str
	description: str
	condition: dict[str, Any]
	effect: dict[str, Any]


class CapabilityRuleEngine:
	"""Deterministic HLTH rule engine for health control decisions."""

	def __init__(self, rules: list[CapabilityRule] | None = None):
		self.rules = rules or default_rules()

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		"""Evaluate matching health governance rules."""
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
	"""UI route exposed by HLTH."""

	name: str
	path: str
	component: str
	permission: str
	nav_group: str


@dataclass(frozen=True)
class CapabilityTheme:
	"""Visual theme contract for HLTH UI surfaces."""

	name: str = "hlth_health_console"
	tokens: dict[str, str] = field(default_factory=lambda: {
		"color.primary": "#245C4E",
		"color.accent": "#E0A458",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F7FAF8",
		"surface.panel": "#FFFFFF",
		"text.primary": "#14261F",
		"text.secondary": "#52635B",
		"border.radius": "8px",
		"density": "compact"
	})
	components: dict[str, dict[str, str]] = field(default_factory=lambda: {
		"health_score_card": {
			"icon": "heart-pulse",
			"status_indicator": "grade-pill",
			"risk_style": "score-band"
		},
		"component_dependency_map": {
			"visual": "dependency-topology",
			"edge_style": "health-impact-line"
		},
		"prediction_risk_panel": {
			"visual": "forecast-sparkline",
			"threshold_style": "confidence-bands"
		},
		"remediation_action_trace": {
			"visual": "runbook-timeline",
			"status_style": "approval-gate"
		}
	})


def default_rules() -> list[CapabilityRule]:
	"""Default HLTH rules available to every tenant."""
	return [
		CapabilityRule(
			name="tenant_context_required",
			description="All health operations require tenant context.",
			condition={"tenant_context_present": False},
			effect={
				"decision": "deny",
				"reason": "tenant_context_required",
				"required_action": "attach_tenant_context"
			}
		),
		CapabilityRule(
			name="component_health_requires_component_id",
			description="Component health updates require a component identifier.",
			condition={"operation": "track_component_health", "component_id_present": False},
			effect={
				"decision": "deny",
				"reason": "component_id_required",
				"required_action": "attach_component_id"
			}
		),
		CapabilityRule(
			name="critical_health_score_creates_alert",
			description="Critical health scores require a tracked alert.",
			condition={"health_score_lt": 40, "alert_created": False},
			effect={
				"decision": "deny",
				"reason": "critical_health_alert_required",
				"required_action": "create_critical_health_alert"
			}
		),
		CapabilityRule(
			name="remediation_requires_runbook",
			description="Remediation actions require an attached runbook.",
			condition={"remediation_requested": True, "runbook_attached": False},
			effect={
				"decision": "deny",
				"reason": "remediation_runbook_required",
				"required_action": "attach_remediation_runbook"
			}
		),
		CapabilityRule(
			name="stale_baseline_requires_review",
			description="Stale health baselines require review before prediction use.",
			condition={"baseline_age_days_gt": 30, "baseline_review_recorded": False},
			effect={
				"decision": "require_review",
				"reason": "baseline_review_required",
				"required_action": "review_or_refresh_baseline"
			}
		),
		CapabilityRule(
			name="unresolved_critical_incident_blocks_deploy",
			description="Deployments are blocked while critical incidents remain unresolved.",
			condition={"deployment_requested": True, "unresolved_critical_incidents_gt": 0},
			effect={
				"decision": "deny",
				"reason": "critical_incident_unresolved",
				"required_action": "resolve_or_waive_critical_incident"
			}
		)
	]


def ui_manifest() -> dict[str, Any]:
	"""Return HLTH UI surface manifest."""
	routes = [
		CapabilityUIRoute("dashboard", "/hlth/dashboard", "HealthDashboard", "health.view", "Overview"),
		CapabilityUIRoute("components", "/hlth/components", "ComponentHealthMap", "health.view", "Assessment"),
		CapabilityUIRoute("alerts", "/hlth/alerts", "HealthAlertCenter", "health.alerts.acknowledge", "Assessment"),
		CapabilityUIRoute("incidents", "/hlth/incidents", "HealthIncidentManager", "health.incidents.manage", "Response"),
		CapabilityUIRoute("predictions", "/hlth/predictions", "HealthPredictionConsole", "health.view", "Intelligence"),
		CapabilityUIRoute("remediation", "/hlth/remediation", "RemediationWorkbench", "health.remediate", "Response"),
		CapabilityUIRoute("reports", "/hlth/reports", "HealthReportStudio", "health.reports.generate", "Reports"),
		CapabilityUIRoute("settings", "/hlth/settings", "HealthSettings", "health.admin", "Administration")
	]
	return {
		"shell": "apg_python",
		"view_module": "views.py",
		"api_prefix": "/hlth/api/v1",
		"routes": [route.__dict__ for route in routes],
		"template_roots": ["templates/", "static/"],
		"requires_theme": True
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable HLTH capability contract."""
	config = CapabilityConfiguration()
	theme = CapabilityTheme()
	return {
		"capability": "hlth",
		"display_name": "System Health Management",
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
	"""Convenience wrapper for default HLTH rule evaluation."""
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
