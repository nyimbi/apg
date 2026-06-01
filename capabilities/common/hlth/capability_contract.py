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


SUPPORTED_HLTH_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_HLTH_AGENT_ROLES = [
	"component_health_reviewer",
	"baseline_reviewer",
	"prediction_reviewer",
	"incident_reviewer",
	"remediation_reviewer",
	"deployment_gate_reviewer",
	"dependency_map_reviewer",
]
PRIVILEGED_HLTH_AGENT_ROLES = [
	"prediction_reviewer",
	"incident_reviewer",
	"remediation_reviewer",
	"deployment_gate_reviewer",
]


@dataclass(frozen=True)
class CapabilityConfiguration:
	"""Tenant-scoped HLTH configuration defaults and schema."""

	defaults: dict[str, Any] = field(default_factory=lambda: {
		"tenant_id": "default",
		"assessment": {
			"health_check_interval_seconds": 60,
			"component_discovery_enabled": True,
			"contextual_health_scoring": True,
			"business_impact_weighting": True,
			"registered_component_required": True,
			"disabled_components_block_checks": True,
			"minimum_score": 0,
			"maximum_score": 100
		},
		"baselines": {
			"learning_period_days": 7,
			"auto_update_enabled": True,
			"stale_baseline_days": 30,
			"minimum_sample_count": 20,
			"review_required_for_stale": True
		},
		"alerts": {
			"correlation_window_minutes": 5,
			"critical_health_score_threshold": 40,
			"auto_acknowledge_recovered_alerts": True,
			"critical_alert_owner_required": True,
			"critical_alert_route_required": True
		},
		"prediction": {
			"prediction_window_hours": 24,
			"failure_forecast_enabled": True,
			"minimum_prediction_confidence": 0.75,
			"baseline_required": True,
			"review_low_confidence_predictions": True
		},
		"remediation": {
			"auto_remediation_enabled": True,
			"runbook_required": True,
			"production_approval_required": True,
			"require_independent_reviewer": True,
			"review_notes_required": True
		},
		"incidents": {
			"block_deploy_on_unresolved_critical": True,
			"incident_owner_required": True,
			"incident_route_required": True,
			"postmortem_required_for_sev1": True
		},
		"deployment_gates": {
			"block_on_unresolved_critical": True,
			"waiver_requires_review": True,
			"record_gate_audit": True
		},
		"adapters": {
			"supported_probe_sources": ["apg_native", "moni", "opentelemetry", "kubernetes"],
			"notification_adapter_required_for_critical": True,
			"remediation_executor": "adapter",
			"deployment_gate_adapter": "adapter",
			"prediction_engine": "adapter"
		},
		"agents": {
			"first_class": True,
			"supported_runtimes": SUPPORTED_HLTH_AGENT_RUNTIMES,
			"supported_roles": SUPPORTED_HLTH_AGENT_ROLES,
			"privileged_roles": PRIVILEGED_HLTH_AGENT_ROLES,
			"require_owner": True,
			"require_purpose": True,
			"require_scope": True,
			"require_contribution_disclosure": True,
			"require_human_approval_for_privileged_roles": True
		},
		"streaming": {
			"engine": "bytewax",
			"lifecycle_stream": "hlth.lifecycle",
			"watermark": "event_time",
			"required_operations": [
				"component_batch",
				"health_check_batch",
				"baseline_batch",
				"prediction_batch",
				"incident_batch",
				"health_agent_batch"
			],
			"topics": [
				"hlth.components",
				"hlth.checks",
				"hlth.baselines",
				"hlth.predictions",
				"hlth.incidents",
				"hlth.agents"
			]
		},
		"security": {
			"require_tenant_context": True,
			"record_lifecycle_audit": True,
			"audit_rule_changes": True
		},
		"ui": {
			"enable_dashboard": True,
			"enable_component_map": True,
			"enable_check_timeline": True,
			"enable_baseline_console": True,
			"enable_prediction_console": True,
			"enable_alert_center": True,
			"enable_incident_console": True,
			"enable_remediation_console": True,
			"enable_deployment_gates": True,
			"enable_adapter_health": True,
			"enable_audit_timeline": True,
			"enable_health_agent_roster": True,
			"enable_lifecycle_batch_monitor": True
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
			"deployment_gates",
			"adapters",
			"agents",
			"streaming",
			"security",
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
			"deployment_gates": {"type": "object"},
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
		},
		"component_inventory_panel": {
			"visual": "component-grid",
			"status_indicator": "component-state"
		},
		"health_check_timeline": {
			"visual": "score-timeline",
			"threshold_style": "critical-band"
		},
		"baseline_freshness_panel": {
			"visual": "baseline-age-list",
			"status_indicator": "freshness-chip"
		},
		"incident_impact_panel": {
			"visual": "incident-impact-list",
			"highlight": "criticality-chip"
		},
		"deployment_gate_panel": {
			"visual": "gate-decision-list",
			"status_indicator": "gate-state"
		},
		"adapter_status_panel": {
			"visual": "backend-grid",
			"status_indicator": "adapter-state"
		},
		"audit_decision_timeline": {
			"visual": "decision-timeline",
			"highlight": "matched-rule-chip"
		},
		"health_agent_roster": {
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
			name="component_must_be_registered",
			description="Health checks require a registered component.",
			condition={"operation": "track_component_health", "component_registered": False},
			effect={
				"decision": "deny",
				"reason": "component_registration_required",
				"required_action": "register_component"
			}
		),
		CapabilityRule(
			name="disabled_component_blocks_health_check",
			description="Disabled components cannot accept health check updates.",
			condition={"operation": "track_component_health", "component_status": "disabled"},
			effect={
				"decision": "deny",
				"reason": "component_disabled",
				"required_action": "reactivate_or_select_component"
			}
		),
		CapabilityRule(
			name="health_score_below_range_denied",
			description="Health scores below zero are denied.",
			condition={"health_score_lt": 0},
			effect={
				"decision": "deny",
				"reason": "health_score_below_range",
				"required_action": "provide_score_0_to_100"
			}
		),
		CapabilityRule(
			name="health_score_above_range_denied",
			description="Health scores above one hundred are denied.",
			condition={"health_score_gt": 100},
			effect={
				"decision": "deny",
				"reason": "health_score_above_range",
				"required_action": "provide_score_0_to_100"
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
			name="critical_alert_requires_owner",
			description="Critical health alerts require owner evidence.",
			condition={"alert_severity": "critical", "alert_owner_present": False},
			effect={
				"decision": "deny",
				"reason": "critical_alert_owner_required",
				"required_action": "assign_alert_owner"
			}
		),
		CapabilityRule(
			name="critical_alert_requires_route",
			description="Critical health alerts require notification route evidence.",
			condition={"alert_severity": "critical", "notification_route_configured": False},
			effect={
				"decision": "deny",
				"reason": "critical_alert_route_required",
				"required_action": "configure_alert_route"
			}
		),
		CapabilityRule(
			name="critical_incident_requires_owner",
			description="Critical health incidents require owner evidence.",
			condition={"incident_severity": "critical", "incident_owner_present": False},
			effect={
				"decision": "deny",
				"reason": "critical_incident_owner_required",
				"required_action": "assign_incident_owner"
			}
		),
		CapabilityRule(
			name="critical_incident_requires_route",
			description="Critical health incidents require notification route evidence.",
			condition={"incident_severity": "critical", "notification_route_configured": False},
			effect={
				"decision": "deny",
				"reason": "critical_incident_route_required",
				"required_action": "configure_incident_route"
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
			name="production_remediation_requires_approval",
			description="Production remediation requires approval evidence.",
			condition={"environment": "production", "remediation_requested": True, "production_approved": False},
			effect={
				"decision": "deny",
				"reason": "production_approval_required",
				"required_action": "attach_production_approval"
			}
		),
		CapabilityRule(
			name="remediation_review_requires_independent_reviewer",
			description="Remediation reviews require an independent reviewer.",
			condition={"reviewer_same_as_requester": True},
			effect={
				"decision": "deny",
				"reason": "independent_reviewer_required",
				"required_action": "assign_independent_reviewer"
			}
		),
		CapabilityRule(
			name="review_notes_required",
			description="Health remediation and waiver reviews require notes.",
			condition={"review_notes_attached": False},
			effect={
				"decision": "deny",
				"reason": "review_notes_required",
				"required_action": "attach_review_notes"
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
			name="prediction_requires_baseline",
			description="Health predictions require baseline evidence.",
			condition={"operation": "predict_health", "baseline_present": False},
			effect={
				"decision": "deny",
				"reason": "prediction_baseline_required",
				"required_action": "attach_health_baseline"
			}
		),
		CapabilityRule(
			name="low_confidence_prediction_requires_review",
			description="Low-confidence predictions require review before action.",
			condition={"operation": "predict_health", "prediction_confidence_lt": 0.75},
			effect={
				"decision": "require_review",
				"reason": "prediction_confidence_review_required",
				"required_action": "review_prediction_evidence"
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
		),
		CapabilityRule(
			name="deployment_waiver_requires_review",
			description="Deployment waivers require review evidence.",
			condition={"deployment_waiver_requested": True, "waiver_review_recorded": False},
			effect={
				"decision": "deny",
				"reason": "deployment_waiver_review_required",
				"required_action": "record_deployment_waiver_review"
			}
		),
		CapabilityRule(
			name="health_agent_runtime_supported",
			description="Health agents must use a supported runtime adapter.",
			condition={"operation": "register_health_agent", "agent_runtime_supported": False},
			effect={
				"decision": "deny",
				"reason": "unsupported_health_agent_runtime",
				"required_action": "select_supported_agent_runtime"
			}
		),
		CapabilityRule(
			name="health_agent_role_supported",
			description="Health agents must use a supported reliability governance role.",
			condition={"operation": "register_health_agent", "agent_role_supported": False},
			effect={
				"decision": "deny",
				"reason": "unsupported_health_agent_role",
				"required_action": "select_supported_agent_role"
			}
		),
		CapabilityRule(
			name="health_agent_requires_scope",
			description="Health agents require an explicit operating scope.",
			condition={"operation": "register_health_agent", "agent_scope_present": False},
			effect={
				"decision": "deny",
				"reason": "health_agent_scope_required",
				"required_action": "attach_agent_scope"
			}
		),
		CapabilityRule(
			name="health_agent_requires_owner",
			description="Health agents require an accountable owner.",
			condition={"operation": "register_health_agent", "agent_owner_present": False},
			effect={
				"decision": "deny",
				"reason": "health_agent_owner_required",
				"required_action": "attach_agent_owner"
			}
		),
		CapabilityRule(
			name="health_agent_requires_purpose",
			description="Health agents require a declared purpose.",
			condition={"operation": "register_health_agent", "agent_purpose_present": False},
			effect={
				"decision": "deny",
				"reason": "health_agent_purpose_required",
				"required_action": "attach_agent_purpose"
			}
		),
		CapabilityRule(
			name="health_agent_requires_contribution_disclosure",
			description="Health agents must disclose machine contribution in reliability decisions.",
			condition={"operation": "register_health_agent", "contribution_disclosed": False},
			effect={
				"decision": "deny",
				"reason": "health_agent_contribution_disclosure_required",
				"required_action": "enable_agent_contribution_disclosure"
			}
		),
		CapabilityRule(
			name="health_agent_privileged_role_requires_human_approval",
			description="Privileged health-agent roles require human approval evidence or review.",
			condition={"operation": "register_health_agent", "privileged_agent_role": True, "human_approval_required": False},
			effect={
				"decision": "require_review",
				"reason": "health_agent_human_approval_required",
				"required_action": "require_human_approval_for_agent"
			}
		),
		CapabilityRule(
			name="bytewax_health_stream_required",
			description="HLTH lifecycle batches must declare Bytewax as the health lifecycle processor.",
			condition={"operation": "validate_health_lifecycle_batch", "event_stream_ne": "bytewax"},
			effect={
				"decision": "deny",
				"reason": "bytewax_health_stream_required",
				"required_action": "route_batch_through_bytewax"
			}
		)
	]


def ui_manifest() -> dict[str, Any]:
	"""Return HLTH UI surface manifest."""
	routes = [
		CapabilityUIRoute("dashboard", "/hlth/dashboard", "HealthDashboard", "health.view", "Overview"),
		CapabilityUIRoute("components", "/hlth/components", "ComponentHealthMap", "health.view", "Assessment"),
		CapabilityUIRoute("checks", "/hlth/checks", "HealthCheckTimeline", "health.view", "Assessment"),
		CapabilityUIRoute("baselines", "/hlth/baselines", "HealthBaselineConsole", "health.manage", "Assessment"),
		CapabilityUIRoute("alerts", "/hlth/alerts", "HealthAlertCenter", "health.alerts.acknowledge", "Assessment"),
		CapabilityUIRoute("incidents", "/hlth/incidents", "HealthIncidentManager", "health.incidents.manage", "Response"),
		CapabilityUIRoute("predictions", "/hlth/predictions", "HealthPredictionConsole", "health.view", "Intelligence"),
		CapabilityUIRoute("remediation", "/hlth/remediation", "RemediationWorkbench", "health.remediate", "Response"),
		CapabilityUIRoute("deployment_gates", "/hlth/deployment-gates", "DeploymentGateConsole", "health.deployments.review", "Response"),
		CapabilityUIRoute("reports", "/hlth/reports", "HealthReportStudio", "health.reports.generate", "Reports"),
		CapabilityUIRoute("audit", "/hlth/audit", "HealthAuditTimeline", "health.admin", "Governance"),
		CapabilityUIRoute("adapters", "/hlth/adapters", "HealthAdapterHealth", "health.admin", "Runtime"),
		CapabilityUIRoute("agents", "/hlth/agents", "HealthAgentRoster", "health.admin", "Administration"),
		CapabilityUIRoute("lifecycle", "/hlth/lifecycle", "HealthLifecycleBatchMonitor", "health.admin", "Runtime"),
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


def agent_manifest() -> dict[str, Any]:
	"""Return first-class HLTH agent composition manifest."""
	return {
		"first_class": True,
		"supported_runtimes": list(SUPPORTED_HLTH_AGENT_RUNTIMES),
		"supported_roles": list(SUPPORTED_HLTH_AGENT_ROLES),
		"privileged_roles": list(PRIVILEGED_HLTH_AGENT_ROLES),
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
	"""Return HLTH lifecycle stream-processing contract."""
	return {
		"engine": "bytewax",
		"lifecycle_stream": "hlth.lifecycle",
		"watermark": "event_time",
		"required_processor": "bytewax",
		"required_operations": [
			"component_batch",
			"health_check_batch",
			"baseline_batch",
			"prediction_batch",
			"incident_batch",
			"health_agent_batch"
		],
		"topics": [
			"hlth.components",
			"hlth.checks",
			"hlth.baselines",
			"hlth.predictions",
			"hlth.incidents",
			"hlth.agents"
		],
		"broker_core_dependency_allowed": False
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable HLTH capability contract."""
	config = CapabilityConfiguration()
	theme = CapabilityTheme()
	return {
		"capability": "hlth",
		"display_name": "Health Checks and Diagnostics",
		"provides": ["health_governance", "diagnostic_lifecycle", "health_agent_composition", "review_evidence"],
		"requires": ["moni", "mqeb", "conf"],
		"configuration": config.for_tenant(tenant_id, overrides),
		"configuration_schema": config.schema,
		"rule_engine": {
			"type": "deterministic",
			"rules": [rule.__dict__ for rule in default_rules()]
		},
		"ui": ui_manifest(),
		"agents": agent_manifest(),
		"streaming": streaming_manifest(),
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
				"rejected",
				"allowed",
				"blocked"
			],
			"policy_fields": [
				"policy_decision",
				"matched_rules",
				"review_reasons",
				"review_evidence"
			],
			"pending_queues": [
				"checks",
				"predictions",
				"alerts",
				"incidents",
				"remediation_requests",
				"deployment_gates",
				"health_agents",
				"lifecycle_batches"
			],
			"deny_behavior": "Denied HLTH lifecycle batches persist evidence before PermissionError"
		},
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
