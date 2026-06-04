"""Executable capability contract for APG Performance Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "telecom_per"
CAPABILITY_NAME = "Performance Management"
CAPABILITY_VERSION = "1.0.0"
PER_EVENT_STREAM = "apg.telecom.per.lifecycle"

SUPPORTED_KPI_CATEGORIES = ["radio_access", "core_network", "transmission", "ims_voice", "data_services", "customer_experience", "revenue", "service_quality"]
SUPPORTED_KPI_STATUSES = ["nominal", "degraded", "critical", "under_maintenance", "no_data"]
SUPPORTED_SLA_COMPLIANCE_STATUSES = ["compliant", "at_risk", "breached", "grace_period", "disputed"]
SUPPORTED_CAPACITY_STATES = ["under_utilised", "optimal", "high", "near_capacity", "congested", "overloaded"]
SUPPORTED_TREND_DIRECTIONS = ["improving", "stable", "degrading", "volatile", "recovering"]
SUPPORTED_REPORT_PERIODS = ["hourly", "daily", "weekly", "monthly", "quarterly", "annual", "custom"]
SUPPORTED_THRESHOLD_ACTIONS = ["alert_only", "escalate", "trigger_capacity_plan", "trigger_optimisation", "auto_remediate"]
SUPPORTED_BENCHMARK_TYPES = ["internal_target", "regulatory", "industry_standard", "competitor", "historical_best"]
SUPPORTED_NETWORK_LAYERS = ["ran", "core", "transport", "ims", "value_added_services", "cdn"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["kpi_analyst", "sla_compliance_analyst", "capacity_analyst", "trend_analyst", "report_generator"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"kpis": {"supported_categories": SUPPORTED_KPI_CATEGORIES, "supported_statuses": SUPPORTED_KPI_STATUSES, "collection_interval_seconds": 300, "retention_days": 365, "baseline_required": True},
	"sla_compliance": {"supported_statuses": SUPPORTED_SLA_COMPLIANCE_STATUSES, "breach_notification": True, "grace_period_minutes": 60, "penalty_reporting": True},
	"capacity": {"supported_states": SUPPORTED_CAPACITY_STATES, "utilisation_threshold_pct": 80, "forecast_horizon_days": 90, "auto_planning_trigger": True},
	"trends": {"supported_directions": SUPPORTED_TREND_DIRECTIONS, "lookback_days": 30, "anomaly_detection": True, "ml_based_forecasting": True},
	"reports": {"supported_periods": SUPPORTED_REPORT_PERIODS, "approval_required": True, "scheduled_delivery": True},
	"thresholds": {"supported_actions": SUPPORTED_THRESHOLD_ACTIONS, "per_layer_thresholds": True, "hysteresis_enabled": True},
	"benchmarks": {"supported_types": SUPPORTED_BENCHMARK_TYPES, "gap_analysis": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "unapproved_threshold_change_denied": True, "cross_tenant_data_denied": True},
	"observability": {"event_stream": PER_EVENT_STREAM, "stream_processor": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_kpis": True, "enable_sla": True, "enable_capacity": True, "enable_trends": True, "enable_reports": True, "enable_benchmarks": True, "enable_agents": True},
	"theme": {"default_theme": "telecom_per_control", "allow_tenant_overrides": True},
}

PROVIDES = ["kpi_monitoring_workflow", "sla_compliance_workflow", "capacity_utilisation_workflow", "trend_reporting_workflow", "performance_reporting_workflow", "threshold_management_workflow", "benchmark_analysis_workflow", "per_agent_workflow"]
REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "moni", "mqeb", "schd", "nlpc"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/telecom-per/dashboard", "component": "PerDashboard", "permission": "telecom_per:view", "nav_group": "Overview"},
	{"name": "kpis", "path": "/telecom-per/kpis", "component": "PerKpiConsole", "permission": "telecom_per:kpis", "nav_group": "KPIs"},
	{"name": "kpi_detail", "path": "/telecom-per/kpis/<id>", "component": "PerKpiDetail", "permission": "telecom_per:kpis", "nav_group": "KPIs"},
	{"name": "sla_compliance", "path": "/telecom-per/sla", "component": "PerSlaConsole", "permission": "telecom_per:sla", "nav_group": "SLA"},
	{"name": "capacity", "path": "/telecom-per/capacity", "component": "PerCapacityConsole", "permission": "telecom_per:capacity", "nav_group": "Capacity"},
	{"name": "trends", "path": "/telecom-per/trends", "component": "PerTrendConsole", "permission": "telecom_per:trends", "nav_group": "Analytics"},
	{"name": "thresholds", "path": "/telecom-per/thresholds", "component": "PerThresholdConsole", "permission": "telecom_per:thresholds", "nav_group": "Configuration"},
	{"name": "benchmarks", "path": "/telecom-per/benchmarks", "component": "PerBenchmarkConsole", "permission": "telecom_per:benchmarks", "nav_group": "Analytics"},
	{"name": "reports", "path": "/telecom-per/reports", "component": "PerReportConsole", "permission": "telecom_per:reports", "nav_group": "Reporting"},
	{"name": "forecasts", "path": "/telecom-per/forecasts", "component": "PerForecastConsole", "permission": "telecom_per:trends", "nav_group": "Analytics"},
	{"name": "agents", "path": "/telecom-per/agents", "component": "PerAgentWorkbench", "permission": "telecom_per:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/telecom-per/settings", "component": "PerSettings", "permission": "telecom_per:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "telecom_per_control",
	"tokens": {"color.primary": "#0369A1", "color.accent": "#0891B2", "color.success": "#15803D", "color.warning": "#B45309", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"kpis": {"icon": "trending-up", "status_indicator": "kpi-status-chip"}, "sla_compliance": {"icon": "target", "status_indicator": "sla-compliance-chip"}, "capacity": {"icon": "database", "status_indicator": "capacity-state-chip"}, "trends": {"icon": "bar-chart", "status_indicator": "trend-direction-chip"}, "thresholds": {"icon": "sliders", "status_indicator": "threshold-action-chip"}, "benchmarks": {"icon": "award", "status_indicator": "benchmark-type-chip"}, "reports": {"icon": "file-text", "status_indicator": "report-period-chip"}, "forecasts": {"icon": "calendar", "status_indicator": "forecast-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {"processor": "bytewax", "stream": PER_EVENT_STREAM, "key": "tenant_id", "events": ["kpi_threshold_breached", "sla_breach_detected", "capacity_congestion_alert", "trend_degradation_detected", "report_generated", "forecast_computed", "benchmark_gap_detected", "threshold_changed", "per_agent_registered"], "guardrails": ["per_batch_requires_bytewax", "privileged_per_agent_action_requires_human_approval", "unapproved_threshold_change_denied", "cross_tenant_data_denied"]}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "per_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "per_policy_required", "required_action": "attach_per_policy"}},
	{"name": "kpi_category_supported", "condition": {"operation": "record_kpi", "kpi_category_supported": False}, "effect": {"decision": "deny", "reason": "kpi_category_not_supported", "required_action": "select_supported_kpi_category"}},
	{"name": "kpi_baseline_required", "condition": {"operation": "record_kpi", "baseline_present": False}, "effect": {"decision": "deny", "reason": "kpi_baseline_required", "required_action": "set_kpi_baseline"}},
	{"name": "kpi_status_supported", "condition": {"operation": "update_kpi_status", "kpi_status_supported": False}, "effect": {"decision": "deny", "reason": "kpi_status_not_supported", "required_action": "select_supported_kpi_status"}},
	{"name": "sla_compliance_status_supported", "condition": {"operation": "record_sla_compliance", "sla_status_supported": False}, "effect": {"decision": "deny", "reason": "sla_compliance_status_not_supported", "required_action": "select_supported_sla_status"}},
	{"name": "sla_breach_notification_required", "condition": {"operation": "record_sla_compliance", "sla_breached": True, "notification_sent": False}, "effect": {"decision": "deny", "reason": "sla_breach_notification_required", "required_action": "send_breach_notification"}},
	{"name": "capacity_state_supported", "condition": {"operation": "record_capacity", "capacity_state_supported": False}, "effect": {"decision": "deny", "reason": "capacity_state_not_supported", "required_action": "select_supported_capacity_state"}},
	{"name": "trend_direction_supported", "condition": {"operation": "record_trend", "trend_direction_supported": False}, "effect": {"decision": "deny", "reason": "trend_direction_not_supported", "required_action": "select_supported_trend_direction"}},
	{"name": "threshold_action_supported", "condition": {"operation": "set_threshold", "threshold_action_supported": False}, "effect": {"decision": "deny", "reason": "threshold_action_not_supported", "required_action": "select_supported_threshold_action"}},
	{"name": "threshold_change_requires_approval", "condition": {"operation": "set_threshold", "approval_present": False}, "effect": {"decision": "deny", "reason": "threshold_change_approval_required", "required_action": "attach_threshold_approval"}},
	{"name": "benchmark_type_supported", "condition": {"operation": "record_benchmark", "benchmark_type_supported": False}, "effect": {"decision": "deny", "reason": "benchmark_type_not_supported", "required_action": "select_supported_benchmark_type"}},
	{"name": "report_period_supported", "condition": {"operation": "generate_report", "report_period_supported": False}, "effect": {"decision": "deny", "reason": "report_period_not_supported", "required_action": "select_supported_report_period"}},
	{"name": "report_approval_required", "condition": {"operation": "generate_report", "approval_present": False}, "effect": {"decision": "deny", "reason": "report_approval_required", "required_action": "attach_report_approval"}},
	{"name": "cross_tenant_data_denied", "condition": {"operation": "per_agent_action", "cross_tenant_data_scope": True}, "effect": {"decision": "deny", "reason": "cross_tenant_data_denied", "required_action": "remove_cross_tenant_data_scope"}},
	{"name": "unapproved_threshold_change_denied", "condition": {"operation": "per_agent_action", "unapproved_threshold_change_scope": True}, "effect": {"decision": "deny", "reason": "unapproved_threshold_change_denied", "required_action": "remove_unapproved_threshold_scope"}},
	{"name": "per_batch_requires_bytewax", "condition": {"operation": "per_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_per_batch_to_bytewax"}},
	{"name": "per_agent_runtime_supported", "condition": {"operation": "register_per_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "per_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "per_agent_role_supported", "condition": {"operation": "register_per_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "per_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "per_agent_name_required", "condition": {"operation": "register_per_agent", "agent_name_present": False}, "effect": {"decision": "deny", "reason": "per_agent_name_required", "required_action": "name_per_agent"}},
	{"name": "per_agent_scope_required", "condition": {"operation": "register_per_agent", "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "per_agent_scope_required", "required_action": "bound_per_agent_scope"}},
	{"name": "privileged_per_agent_action_requires_human_approval", "condition": {"operation": "per_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "provides": list(PROVIDES), "requires": list(REQUIRES), "configuration": configuration, "configuration_schema": {"type": "object", "required": list(configuration), "properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}}, "rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/telecom-per/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	actions: list[dict[str, Any]] = []
	for rule in RULES:
		if _matches(rule["condition"], context):
			actions.append(rule["effect"] | {"rule": rule["name"]})
	if not actions:
		return {"decision": "allow", "actions": [], "context": dict(context)}
	return {"decision": "deny", "actions": actions, "context": dict(context)}


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_ne"):
			if context.get(key[:-3]) == expected:
				return False
			continue
		if context.get(key) != expected:
			return False
	return True
