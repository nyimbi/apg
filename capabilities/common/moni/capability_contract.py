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


SUPPORTED_MONI_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_MONI_AGENT_ROLES = [
	"slo_reviewer",
	"alert_reviewer",
	"incident_reviewer",
	"anomaly_triage",
	"metric_quality_reviewer",
	"trace_correlation_reviewer",
	"dashboard_reviewer",
]
PRIVILEGED_MONI_AGENT_ROLES = [
	"slo_reviewer",
	"alert_reviewer",
	"incident_reviewer",
	"anomaly_triage",
]


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
			"source_registration_required": True,
			"disabled_sources_block_ingestion": True,
			"max_cardinality_per_metric": 10000
		},
		"slo": {
			"default_window_minutes": 60,
			"alert_route_required": True,
			"owner_required": True,
			"burn_rate_alerting_enabled": True
		},
		"alerts": {
			"default_severity": "medium",
			"critical_alert_route_required": True,
			"deduplication_window_minutes": 5,
			"notification_capability": "ntfy",
			"critical_incident_owner_required": True
		},
		"incidents": {
			"auto_open_from_critical_alerts": True,
			"owner_required": True,
			"postmortem_required_for_severity": ["critical"]
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
			"runbook_required": True,
			"require_independent_reviewer": True,
			"review_notes_required": True
		},
		"adapters": {
			"supported_collectors": ["opentelemetry", "prometheus", "apg_native"],
			"metrics_store": "adapter",
			"log_store": "adapter",
			"trace_store": "adapter",
			"notification_adapter_required_for_critical": True
		},
		"agents": {
			"first_class": True,
			"supported_runtimes": SUPPORTED_MONI_AGENT_RUNTIMES,
			"supported_roles": SUPPORTED_MONI_AGENT_ROLES,
			"privileged_roles": PRIVILEGED_MONI_AGENT_ROLES,
			"require_owner": True,
			"require_purpose": True,
			"require_scope": True,
			"require_contribution_disclosure": True,
			"require_human_approval_for_privileged_roles": True
		},
		"streaming": {
			"processor": "bytewax",
			"stream": "apg.common.moni.lifecycle",
			"key": "tenant_id",
			"events": [
				"metric_recorded",
				"alert_fired",
				"alert_resolved",
				"incident_created",
				"incident_closed",
				"slo_breach_detected",
				"health_check_failed",
				"anomaly_detected"
			],
			"guardrails": [
				"monitoring_batch_requires_bytewax"
			],
			"engine": "bytewax",
			"lifecycle_stream": "moni.lifecycle",
			"watermark": "event_time",
			"required_operations": [
				"metric_batch",
				"alert_batch",
				"incident_batch",
				"slo_batch",
				"monitoring_agent_batch"
			],
			"topics": [
				"moni.metrics",
				"moni.alerts",
				"moni.incidents",
				"moni.slos",
				"moni.agents"
			]
		},
		"security": {
			"require_tenant_context": True,
			"block_pii_in_logs": True,
			"audit_rule_changes": True,
			"record_lifecycle_audit": True
		},
		"ui": {
			"enable_dashboard": True,
			"enable_source_inventory": True,
			"enable_alert_center": True,
			"enable_log_explorer": True,
			"enable_trace_explorer": True,
			"enable_slo_console": True,
			"enable_incident_console": True,
			"enable_remediation_console": True,
			"enable_adapter_health": True,
			"enable_audit_timeline": True,
			"enable_monitoring_agent_roster": True,
			"enable_lifecycle_batch_monitor": True
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
			"slo",
			"alerts",
			"incidents",
			"analytics",
			"retention",
			"remediation",
			"adapters",
			"agents",
			"streaming",
			"security",
			"ui",
			"theme"
		],
		"properties": {
			"tenant_id": {"type": "string", "minLength": 1},
			"collection": {"type": "object"},
			"slo": {"type": "object"},
			"alerts": {"type": "object"},
			"incidents": {"type": "object"},
			"analytics": {"type": "object"},
			"retention": {"type": "object"},
			"remediation": {"type": "object"},
			"adapters": {"type": "object"},
			"agents": {"type": "object"},
			"streaming": {"type": "object"},
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
		},
		"source_health_panel": {
			"visual": "source-grid",
			"status_indicator": "source-state"
		},
		"slo_burn_rate_panel": {
			"visual": "burn-rate-chart",
			"threshold_style": "budget-lines"
		},
		"incident_timeline": {
			"visual": "event-timeline",
			"highlight": "severity-chip"
		},
		"adapter_status_panel": {
			"visual": "backend-grid",
			"status_indicator": "adapter-state"
		},
		"audit_decision_timeline": {
			"visual": "decision-timeline",
			"highlight": "matched-rule-chip"
		},
		"monitoring_agent_roster": {
			"icon": "bot",
			"status_indicator": "approval-state",
			"variant": "agent-governance"
		},
		"bytewax_lifecycle_panel": {
			"icon": "activity",
			"status_indicator": "processor-state",
			"variant": "stream-lifecycle"
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
			name="signal_requires_registered_source",
			description="Signal ingestion requires a registered source.",
			condition={"operation": "ingest_signal", "source_registered": False},
			effect={
				"decision": "deny",
				"reason": "source_registration_required",
				"required_action": "register_signal_source"
			}
		),
		CapabilityRule(
			name="disabled_source_blocks_ingestion",
			description="Disabled sources cannot emit telemetry signals.",
			condition={"operation": "ingest_signal", "source_status": "disabled"},
			effect={
				"decision": "deny",
				"reason": "source_disabled",
				"required_action": "reactivate_or_select_source"
			}
		),
		CapabilityRule(
			name="trace_requires_trace_id",
			description="Trace ingestion requires a trace identifier.",
			condition={"signal_type": "trace", "trace_id_present": False},
			effect={
				"decision": "deny",
				"reason": "trace_id_required",
				"required_action": "attach_trace_id"
			}
		),
		CapabilityRule(
			name="trace_requires_service_name",
			description="Trace ingestion requires service name evidence.",
			condition={"signal_type": "trace", "service_name_present": False},
			effect={
				"decision": "deny",
				"reason": "service_name_required",
				"required_action": "attach_service_name"
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
			name="critical_alert_requires_owner",
			description="Critical alerts require an assigned owner.",
			condition={"alert_severity": "critical", "alert_owner_present": False},
			effect={
				"decision": "deny",
				"reason": "alert_owner_required",
				"required_action": "assign_alert_owner"
			}
		),
		CapabilityRule(
			name="critical_incident_requires_owner",
			description="Critical incidents require an assigned owner.",
			condition={"incident_severity": "critical", "incident_owner_present": False},
			effect={
				"decision": "deny",
				"reason": "incident_owner_required",
				"required_action": "assign_incident_owner"
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
			name="slo_requires_alert_route",
			description="SLO definitions require alert route evidence.",
			condition={"operation": "create_slo", "notification_route_configured": False},
			effect={
				"decision": "deny",
				"reason": "slo_alert_route_required",
				"required_action": "configure_slo_alert_route"
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
			name="retention_above_limit_requires_review",
			description="Telemetry retention above tenant limits requires review.",
			condition={"retention_above_limit": True, "retention_exception_recorded": False},
			effect={
				"decision": "require_review",
				"reason": "retention_review_required",
				"required_action": "record_retention_exception"
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
		),
		CapabilityRule(
			name="remediation_review_requires_independent_reviewer",
			description="Remediation approvals require independent review.",
			condition={"reviewer_same_as_requester": True},
			effect={
				"decision": "deny",
				"reason": "independent_reviewer_required",
				"required_action": "assign_independent_reviewer"
			}
		),
		CapabilityRule(
			name="review_notes_required",
			description="Remediation and exception reviews require notes.",
			condition={"review_notes_attached": False},
			effect={
				"decision": "deny",
				"reason": "review_notes_required",
				"required_action": "attach_review_notes"
			}
		),
		CapabilityRule(
			name="monitoring_agent_runtime_supported",
			description="Monitoring agents must use a supported runtime adapter.",
			condition={"operation": "register_monitoring_agent", "agent_runtime_supported": False},
			effect={
				"decision": "deny",
				"reason": "unsupported_monitoring_agent_runtime",
				"required_action": "select_supported_agent_runtime"
			}
		),
		CapabilityRule(
			name="monitoring_agent_role_supported",
			description="Monitoring agents must use a supported observability role.",
			condition={"operation": "register_monitoring_agent", "agent_role_supported": False},
			effect={
				"decision": "deny",
				"reason": "unsupported_monitoring_agent_role",
				"required_action": "select_supported_agent_role"
			}
		),
		CapabilityRule(
			name="monitoring_agent_requires_scope",
			description="Monitoring agents require an explicit operating scope.",
			condition={"operation": "register_monitoring_agent", "agent_scope_present": False},
			effect={
				"decision": "deny",
				"reason": "monitoring_agent_scope_required",
				"required_action": "attach_agent_scope"
			}
		),
		CapabilityRule(
			name="monitoring_agent_requires_owner",
			description="Monitoring agents require an accountable owner.",
			condition={"operation": "register_monitoring_agent", "agent_owner_present": False},
			effect={
				"decision": "deny",
				"reason": "monitoring_agent_owner_required",
				"required_action": "attach_agent_owner"
			}
		),
		CapabilityRule(
			name="monitoring_agent_requires_purpose",
			description="Monitoring agents require a declared purpose.",
			condition={"operation": "register_monitoring_agent", "agent_purpose_present": False},
			effect={
				"decision": "deny",
				"reason": "monitoring_agent_purpose_required",
				"required_action": "attach_agent_purpose"
			}
		),
		CapabilityRule(
			name="monitoring_agent_requires_contribution_disclosure",
			description="Monitoring agents must disclose machine contribution in observability decisions.",
			condition={"operation": "register_monitoring_agent", "contribution_disclosed": False},
			effect={
				"decision": "deny",
				"reason": "monitoring_agent_contribution_disclosure_required",
				"required_action": "enable_agent_contribution_disclosure"
			}
		),
		CapabilityRule(
			name="monitoring_agent_privileged_role_requires_human_approval",
			description="Privileged monitoring-agent roles require human approval evidence or review.",
			condition={"operation": "register_monitoring_agent", "privileged_agent_role": True, "human_approval_required": False},
			effect={
				"decision": "require_review",
				"reason": "monitoring_agent_human_approval_required",
				"required_action": "require_human_approval_for_agent"
			}
		),
		CapabilityRule(
			name="bytewax_monitoring_stream_required",
			description="MONI lifecycle batches must declare Bytewax as the observability lifecycle processor.",
			condition={"operation": "validate_monitoring_lifecycle_batch", "event_stream_ne": "bytewax"},
			effect={
				"decision": "deny",
				"reason": "bytewax_monitoring_stream_required",
				"required_action": "route_batch_through_bytewax"
			}
		)
	]


def ui_manifest() -> dict[str, Any]:
	"""Return MONI UI surface manifest."""
	routes = [
		CapabilityUIRoute("dashboard", "/moni/dashboard", "MonitoringDashboard", "moni:view", "Overview"),
		CapabilityUIRoute("sources", "/moni/sources", "SignalSourceInventory", "moni:manage_sources", "Signals"),
		CapabilityUIRoute("metrics", "/moni/metrics", "MetricExplorer", "moni:view_metrics", "Signals"),
		CapabilityUIRoute("logs", "/moni/logs", "LogExplorer", "moni:view_logs", "Signals"),
		CapabilityUIRoute("alerts", "/moni/alerts", "AlertCenter", "moni:manage_alerts", "Signals"),
		CapabilityUIRoute("traces", "/moni/traces", "TraceExplorer", "moni:view_traces", "Signals"),
		CapabilityUIRoute("slos", "/moni/slos", "SLOConsole", "moni:manage_slos", "Reliability"),
		CapabilityUIRoute("incidents", "/moni/incidents", "IncidentConsole", "moni:manage_incidents", "Reliability"),
		CapabilityUIRoute("analytics", "/moni/analytics", "ObservabilityAnalytics", "moni:view_analytics", "Intelligence"),
		CapabilityUIRoute("rules", "/moni/rules", "MonitoringRuleManager", "moni:manage_rules", "Governance"),
		CapabilityUIRoute("remediation", "/moni/remediation", "RemediationConsole", "moni:remediate", "Reliability"),
		CapabilityUIRoute("audit", "/moni/audit", "MonitoringAuditTimeline", "moni:admin", "Governance"),
		CapabilityUIRoute("adapters", "/moni/adapters", "MonitoringAdapterHealth", "moni:admin", "Runtime"),
		CapabilityUIRoute("agents", "/moni/agents", "MonitoringAgentRoster", "moni:admin", "Administration"),
		CapabilityUIRoute("lifecycle", "/moni/lifecycle", "MonitoringLifecycleBatchMonitor", "moni:admin", "Runtime"),
		CapabilityUIRoute("settings", "/moni/settings", "MonitoringSettings", "moni:admin", "Administration")
	]
	return {
		"shell": "apg_python",
		"view_module": "view_models.py",
		"api_prefix": "/moni/api/v1",
		"routes": [route.__dict__ for route in routes],
		"template_roots": ["templates/", "static/"],
		"requires_theme": True
	}


def agent_manifest() -> dict[str, Any]:
	"""Return first-class MONI agent composition manifest."""
	return {
		"first_class": True,
		"supported_runtimes": list(SUPPORTED_MONI_AGENT_RUNTIMES),
		"supported_roles": list(SUPPORTED_MONI_AGENT_ROLES),
		"privileged_roles": list(PRIVILEGED_MONI_AGENT_ROLES),
		"required_fields": ["tenant_id", "agent_id", "name", "runtime", "role", "scope", "owner", "purpose"],
		"guardrails": [
			"supported_runtime",
			"supported_role",
			"explicit_scope",
			"accountable_owner",
			"declared_purpose",
			"machine_contribution_disclosure",
			"human_approval_for_privileged_roles"
		]
	}


def streaming_manifest() -> dict[str, Any]:
	"""Return MONI lifecycle stream-processing contract."""
	return {
		"engine": "bytewax",
		"lifecycle_stream": "moni.lifecycle",
		"watermark": "event_time",
		"required_processor": "bytewax",
		"required_operations": [
			"metric_batch",
			"alert_batch",
			"incident_batch",
			"slo_batch",
			"monitoring_agent_batch"
		],
		"topics": [
			"moni.metrics",
			"moni.alerts",
			"moni.incidents",
			"moni.slos",
			"moni.agents"
		],
		"broker_core_dependency_allowed": False
	}


STREAMING: dict[str, Any] = {
	"processor": "bytewax",
	"stream": "apg.moni.lifecycle",
	"key": "tenant_id",
	"events": [
		"metric_recorded",
		"metric_threshold_breached",
		"alert_triggered",
		"alert_resolved",
		"health_check_failed",
		"dashboard_created",
		"dashboard_updated",
		"trace_captured",
		"slo_breached",
		"incident_raised",
		"incident_resolved",
		"agent_registered",
	],
	"guardrails": [
		"moni_batch_requires_bytewax",
		"moni_privileged_action_requires_human_approval",
	],
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable MONI capability contract."""
	config = CapabilityConfiguration()
	theme = CapabilityTheme()
	return {
		"capability": "moni",
		"display_name": "Monitoring and Observability",
		"provides": ["observability_governance", "metrics_lifecycle", "monitoring_agent_composition", "review_evidence"],
		"requires": ["conf", "audl", "mqeb"],
		"configuration": config.for_tenant(tenant_id, overrides),
		"configuration_schema": config.schema,
		"rule_engine": {
			"type": "deterministic",
			"rules": [rule.__dict__ for rule in default_rules()]
		},
		"ui": ui_manifest(),
		"agents": agent_manifest(),
		"streaming": STREAMING,
		"review_evidence": {
			"durable_statuses": [
				"pending",
				"pending_review",
				"review_required",
				"denied",
				"accepted",
				"active",
				"open",
				"resolved",
				"approved",
				"rejected"
			],
			"policy_fields": [
				"policy_decision",
				"matched_rules",
				"review_reasons",
				"review_evidence"
			],
			"pending_queues": [
				"signals",
				"alerts",
				"incidents",
				"remediation_requests",
				"monitoring_agents",
				"lifecycle_batches"
			],
			"deny_behavior": "Denied MONI lifecycle batches persist evidence before PermissionError"
		},
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
		elif key.endswith("_ne"):
			field_name = key[:-3]
			if context.get(field_name) == expected:
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
