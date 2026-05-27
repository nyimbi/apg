"""
Executable capability contract for APG Monitoring and Observability.

MONI is a first-class APG capability. This module exposes tenant-scoped
configuration, deterministic observability-governance rules, UI surfaces, and
theme tokens so composition tooling can integrate with MONI without starting
the monitoring runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CapabilityConfiguration:
	"""Tenant-scoped MONI configuration defaults and schema."""

	defaults: dict[str, Any] = field(default_factory=lambda: {
		"tenant_id": "default",
		"collection": {
			"metrics_enabled": True,
			"logs_enabled": True,
			"traces_enabled": True,
			"tenant_label_required": True,
			"max_cardinality_per_metric": 10000
		},
		"alerts": {
			"default_severity": "medium",
			"critical_alert_route_required": True,
			"deduplication_window_minutes": 5,
			"notification_capability": "ntfy"
		},
		"analytics": {
			"anomaly_detection_enabled": True,
			"predictive_issue_prevention_enabled": True,
			"business_impact_correlation": True
		},
		"retention": {
			"metrics_days": 90,
			"logs_days": 30,
			"traces_days": 14,
			"compliance_evidence_days": 2555
		},
		"remediation": {
			"autonomous_remediation_enabled": True,
			"require_approval_for_production": True,
			"runbook_required": True
		},
		"security": {
			"require_tenant_context": True,
			"block_pii_in_logs": True,
			"audit_rule_changes": True
		},
		"ui": {
			"enable_dashboard": True,
			"enable_alert_center": True,
			"enable_trace_explorer": True,
			"enable_remediation_console": True
		},
		"theme": {
			"default_theme": "moni_signal_console",
			"allow_tenant_overrides": True
		}
	})
	schema: dict[str, Any] = field(default_factory=lambda: {
		"type": "object",
		"required": [
			"tenant_id",
			"collection",
			"alerts",
			"analytics",
			"retention",
			"remediation",
			"security",
			"ui",
			"theme"
		],
		"properties": {
			"tenant_id": {"type": "string", "minLength": 1},
			"collection": {"type": "object"},
			"alerts": {"type": "object"},
			"analytics": {"type": "object"},
			"retention": {"type": "object"},
			"remediation": {"type": "object"},
			"security": {"type": "object"},
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
	"""Simple MONI policy rule definition."""

	name: str
	description: str
	condition: dict[str, Any]
	effect: dict[str, Any]


class CapabilityRuleEngine:
	"""Deterministic MONI rule engine for observability control decisions."""

	def __init__(self, rules: list[CapabilityRule] | None = None):
		self.rules = rules or default_rules()

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		"""Evaluate matching observability governance rules."""
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
	"""UI route exposed by MONI."""

	name: str
	path: str
	component: str
	permission: str
	nav_group: str


@dataclass(frozen=True)
class CapabilityTheme:
	"""Visual theme contract for MONI UI surfaces."""

	name: str = "moni_signal_console"
	tokens: dict[str, str] = field(default_factory=lambda: {
		"color.primary": "#1B4965",
		"color.accent": "#5FA8D3",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F5F8FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#13293D",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact"
	})
	components: dict[str, dict[str, str]] = field(default_factory=lambda: {
		"signal_overview_card": {
			"icon": "activity",
			"status_indicator": "slo-pill",
			"risk_style": "burn-rate-band"
		},
		"alert_correlation_stack": {
			"visual": "grouped-alert-list",
			"highlight": "incident-chip"
		},
		"metric_query_panel": {
			"visual": "time-series-grid",
			"threshold_style": "slo-lines"
		},
		"remediation_runbook_trace": {
			"visual": "step-timeline",
			"status_style": "approval-gate"
		}
	})


def default_rules() -> list[CapabilityRule]:
	"""Default MONI rules available to every tenant."""
	return [
		CapabilityRule(
			name="tenant_context_required",
			description="All observability events require tenant context.",
			condition={"tenant_context_present": False},
			effect={
				"decision": "deny",
				"reason": "tenant_context_required",
				"required_action": "attach_tenant_context"
			}
		),
		CapabilityRule(
			name="metric_ingestion_requires_source",
			description="Metric ingestion requires an explicit source identifier.",
			condition={"operation": "ingest_metric", "source_present": False},
			effect={
				"decision": "deny",
				"reason": "metric_source_required",
				"required_action": "attach_metric_source"
			}
		),
		CapabilityRule(
			name="critical_alert_requires_route",
			description="Critical alerts require an escalation route.",
			condition={"alert_severity": "critical", "notification_route_configured": False},
			effect={
				"decision": "deny",
				"reason": "critical_alert_route_required",
				"required_action": "configure_alert_route"
			}
		),
		CapabilityRule(
			name="pii_logs_blocked",
			description="Logs containing PII are blocked unless redacted.",
			condition={"log_contains_pii": True, "pii_redacted": False},
			effect={
				"decision": "deny",
				"reason": "pii_redaction_required",
				"required_action": "redact_or_drop_log_event"
			}
		),
		CapabilityRule(
			name="high_cardinality_metric_requires_review",
			description="High-cardinality metrics require review before ingestion.",
			condition={"metric_cardinality_gt": 10000, "cardinality_exception_recorded": False},
			effect={
				"decision": "require_review",
				"reason": "cardinality_review_required",
				"required_action": "record_cardinality_exception"
			}
		),
		CapabilityRule(
			name="production_remediation_requires_runbook",
			description="Production remediation requires an approved runbook.",
			condition={"environment": "production", "remediation_requested": True, "runbook_approved": False},
			effect={
				"decision": "deny",
				"reason": "approved_runbook_required",
				"required_action": "approve_remediation_runbook"
			}
		)
	]


def ui_manifest() -> dict[str, Any]:
	"""Return MONI UI surface manifest."""
	routes = [
		CapabilityUIRoute("dashboard", "/moni/dashboard", "MonitoringDashboard", "moni:view", "Overview"),
		CapabilityUIRoute("metrics", "/moni/metrics", "MetricExplorer", "moni:view_metrics", "Signals"),
		CapabilityUIRoute("alerts", "/moni/alerts", "AlertCenter", "moni:manage_alerts", "Signals"),
		CapabilityUIRoute("traces", "/moni/traces", "TraceExplorer", "moni:view_traces", "Signals"),
		CapabilityUIRoute("analytics", "/moni/analytics", "ObservabilityAnalytics", "moni:view_analytics", "Intelligence"),
		CapabilityUIRoute("rules", "/moni/rules", "MonitoringRuleManager", "moni:manage_rules", "Governance"),
		CapabilityUIRoute("remediation", "/moni/remediation", "RemediationConsole", "moni:remediate", "Reliability"),
		CapabilityUIRoute("settings", "/moni/settings", "MonitoringSettings", "moni:admin", "Administration")
	]
	return {
		"shell": "apg_python",
		"view_module": "views.py",
		"api_prefix": "/moni/api/v1",
		"routes": [route.__dict__ for route in routes],
		"template_roots": ["templates/", "static/"],
		"requires_theme": True
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable MONI capability contract."""
	config = CapabilityConfiguration()
	theme = CapabilityTheme()
	return {
		"capability": "moni",
		"display_name": "Monitoring and Observability",
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
	"""Convenience wrapper for default MONI rule evaluation."""
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
