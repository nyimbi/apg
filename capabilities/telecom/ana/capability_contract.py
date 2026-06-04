"""Executable capability contract for APG Telecom Analytics."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "telecom_ana"
CAPABILITY_NAME = "Telecom Analytics"
CAPABILITY_VERSION = "1.0.0"
ANA_EVENT_STREAM = "apg.telecom.ana.lifecycle"

SUPPORTED_ANALYSIS_TYPES = ["churn_prediction", "arpu_analysis", "usage_pattern", "revenue_assurance", "network_performance", "customer_segmentation", "fraud_analytics", "roaming_analytics", "congestion_analysis", "capacity_forecast"]
SUPPORTED_METRIC_TYPES = ["kpi", "counter", "gauge", "histogram", "derived", "composite", "predictive", "benchmark"]
SUPPORTED_REPORT_FORMATS = ["json", "csv", "pdf", "excel", "dashboard", "api", "stream"]
SUPPORTED_AGGREGATION_TYPES = ["sum", "avg", "max", "min", "count", "percentile", "stddev", "rate"]
SUPPORTED_TIME_GRANULARITIES = ["realtime", "minute", "hour", "day", "week", "month", "quarter", "year"]
SUPPORTED_SEGMENTS = ["prepaid", "postpaid", "enterprise", "wholesale", "roaming", "iot", "broadband", "voice_only"]
SUPPORTED_CHURN_RISK_LEVELS = ["low", "medium", "high", "critical"]
SUPPORTED_REVENUE_CATEGORIES = ["voice", "data", "sms", "roaming", "interconnect", "value_added_services", "equipment", "penalties"]
SUPPORTED_NETWORK_LAYERS = ["core", "radio", "transport", "ims", "cdn", "edge"]
SUPPORTED_ANOMALY_TYPES = ["revenue_leak", "usage_spike", "churn_signal", "fraud_pattern", "network_degradation", "billing_discrepancy"]
SUPPORTED_MODEL_TYPES = ["regression", "classification", "clustering", "time_series", "ensemble", "neural_network"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["data_analyst", "model_trainer", "report_generator", "anomaly_detector", "forecast_reviewer"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"analysis": {"supported_analysis_types": SUPPORTED_ANALYSIS_TYPES, "supported_metric_types": SUPPORTED_METRIC_TYPES, "supported_time_granularities": SUPPORTED_TIME_GRANULARITIES, "owner_required": True, "evidence_required": True},
	"metrics": {"supported_metric_types": SUPPORTED_METRIC_TYPES, "supported_aggregation_types": SUPPORTED_AGGREGATION_TYPES, "baseline_required": True},
	"reports": {"supported_formats": SUPPORTED_REPORT_FORMATS, "approval_required": True, "evidence_required": True},
	"segments": {"supported_segments": SUPPORTED_SEGMENTS, "criteria_required": True},
	"churn": {"supported_risk_levels": SUPPORTED_CHURN_RISK_LEVELS, "model_required": True, "threshold_required": True},
	"revenue": {"supported_categories": SUPPORTED_REVENUE_CATEGORIES, "assurance_enabled": True, "leak_detection": True},
	"network": {"supported_layers": SUPPORTED_NETWORK_LAYERS, "performance_threshold_required": True},
	"anomalies": {"supported_anomaly_types": SUPPORTED_ANOMALY_TYPES, "confidence_required": True, "evidence_required": True},
	"models": {"supported_model_types": SUPPORTED_MODEL_TYPES, "validation_required": True, "versioning_enabled": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "cross_tenant_data_denied": True, "unapproved_model_deployment_denied": True, "raw_data_export_requires_approval": True},
	"observability": {"event_stream": ANA_EVENT_STREAM, "stream_processor": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_analysis": True, "enable_metrics": True, "enable_reports": True, "enable_churn": True, "enable_revenue": True, "enable_anomalies": True, "enable_models": True, "enable_agents": True},
	"theme": {"default_theme": "telecom_ana_control", "allow_tenant_overrides": True},
}

PROVIDES = ["analytics_pipeline", "churn_prediction_workflow", "arpu_analysis_workflow", "usage_pattern_workflow", "revenue_assurance_workflow", "network_performance_analytics", "customer_segmentation_workflow", "anomaly_detection_workflow", "model_management_workflow", "analytics_reporting_workflow", "analytics_agent_workflow"]
REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "nlpc", "moni", "mqeb", "schd"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/telecom-ana/dashboard", "component": "AnaDashboard", "permission": "telecom_ana:view", "nav_group": "Overview"},
	{"name": "analysis", "path": "/telecom-ana/analysis", "component": "AnaAnalysisConsole", "permission": "telecom_ana:analysis", "nav_group": "Analysis"},
	{"name": "metrics", "path": "/telecom-ana/metrics", "component": "AnaMetricLedger", "permission": "telecom_ana:metrics", "nav_group": "Analysis"},
	{"name": "churn", "path": "/telecom-ana/churn", "component": "AnaChurnWorkbench", "permission": "telecom_ana:churn", "nav_group": "Predictions"},
	{"name": "revenue", "path": "/telecom-ana/revenue", "component": "AnaRevenueAssurance", "permission": "telecom_ana:revenue", "nav_group": "Revenue"},
	{"name": "segments", "path": "/telecom-ana/segments", "component": "AnaSegmentConsole", "permission": "telecom_ana:segments", "nav_group": "Customers"},
	{"name": "network_analytics", "path": "/telecom-ana/network", "component": "AnaNetworkPerformance", "permission": "telecom_ana:network", "nav_group": "Network"},
	{"name": "anomalies", "path": "/telecom-ana/anomalies", "component": "AnaAnomalyQueue", "permission": "telecom_ana:anomalies", "nav_group": "Monitoring"},
	{"name": "models", "path": "/telecom-ana/models", "component": "AnaModelRegistry", "permission": "telecom_ana:models", "nav_group": "ML"},
	{"name": "reports", "path": "/telecom-ana/reports", "component": "AnaReportConsole", "permission": "telecom_ana:reports", "nav_group": "Reporting"},
	{"name": "agents", "path": "/telecom-ana/agents", "component": "AnaAgentWorkbench", "permission": "telecom_ana:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/telecom-ana/settings", "component": "AnaSettings", "permission": "telecom_ana:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "telecom_ana_control",
	"tokens": {"color.primary": "#1D4ED8", "color.accent": "#0891B2", "color.success": "#15803D", "color.warning": "#B45309", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"analysis": {"icon": "bar-chart-2", "status_indicator": "analysis-chip"}, "metrics": {"icon": "activity", "status_indicator": "metric-chip"}, "churn": {"icon": "user-minus", "status_indicator": "churn-risk-chip"}, "revenue": {"icon": "dollar-sign", "status_indicator": "revenue-chip"}, "segments": {"icon": "users", "status_indicator": "segment-chip"}, "network_analytics": {"icon": "network", "status_indicator": "layer-chip"}, "anomalies": {"icon": "alert-triangle", "status_indicator": "anomaly-chip"}, "models": {"icon": "cpu", "status_indicator": "model-chip"}, "reports": {"icon": "file-text", "status_indicator": "report-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {"processor": "bytewax", "stream": ANA_EVENT_STREAM, "key": "tenant_id", "events": ["analysis_run_recorded", "metric_recorded", "churn_prediction_recorded", "revenue_assurance_event_recorded", "segment_recorded", "network_analytics_recorded", "anomaly_detected", "model_registered", "report_generated", "ana_agent_registered"], "guardrails": ["ana_batch_requires_bytewax", "privileged_ana_agent_action_requires_human_approval", "unapproved_model_deployment_denied", "raw_data_export_requires_approval", "cross_tenant_data_denied"]}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "ana_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "ana_policy_required", "required_action": "attach_ana_policy"}},
	{"name": "analysis_type_supported", "condition": {"operation": "record_analysis", "analysis_type_supported": False}, "effect": {"decision": "deny", "reason": "analysis_type_not_supported", "required_action": "select_supported_analysis_type"}},
	{"name": "analysis_owner_required", "condition": {"operation": "record_analysis", "owner_present": False}, "effect": {"decision": "deny", "reason": "analysis_owner_required", "required_action": "assign_analysis_owner"}},
	{"name": "analysis_evidence_required", "condition": {"operation": "record_analysis", "evidence_present": False}, "effect": {"decision": "deny", "reason": "analysis_evidence_required", "required_action": "attach_analysis_evidence"}},
	{"name": "metric_type_supported", "condition": {"operation": "record_metric", "metric_type_supported": False}, "effect": {"decision": "deny", "reason": "metric_type_not_supported", "required_action": "select_supported_metric_type"}},
	{"name": "metric_baseline_required", "condition": {"operation": "record_metric", "baseline_present": False}, "effect": {"decision": "deny", "reason": "metric_baseline_required", "required_action": "set_metric_baseline"}},
	{"name": "churn_risk_level_supported", "condition": {"operation": "record_churn_prediction", "risk_level_supported": False}, "effect": {"decision": "deny", "reason": "churn_risk_level_not_supported", "required_action": "select_supported_risk_level"}},
	{"name": "churn_model_required", "condition": {"operation": "record_churn_prediction", "model_present": False}, "effect": {"decision": "deny", "reason": "churn_model_required", "required_action": "select_churn_model"}},
	{"name": "churn_confidence_valid", "condition": {"operation": "record_churn_prediction", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "revenue_category_supported", "condition": {"operation": "record_revenue_event", "category_supported": False}, "effect": {"decision": "deny", "reason": "revenue_category_not_supported", "required_action": "select_supported_revenue_category"}},
	{"name": "revenue_evidence_required", "condition": {"operation": "record_revenue_event", "evidence_present": False}, "effect": {"decision": "deny", "reason": "revenue_evidence_required", "required_action": "attach_revenue_evidence"}},
	{"name": "segment_criteria_required", "condition": {"operation": "record_segment", "criteria_present": False}, "effect": {"decision": "deny", "reason": "segment_criteria_required", "required_action": "define_segment_criteria"}},
	{"name": "network_layer_supported", "condition": {"operation": "record_network_analytics", "layer_supported": False}, "effect": {"decision": "deny", "reason": "network_layer_not_supported", "required_action": "select_supported_network_layer"}},
	{"name": "anomaly_type_supported", "condition": {"operation": "record_anomaly", "anomaly_type_supported": False}, "effect": {"decision": "deny", "reason": "anomaly_type_not_supported", "required_action": "select_supported_anomaly_type"}},
	{"name": "anomaly_confidence_required", "condition": {"operation": "record_anomaly", "confidence_present": False}, "effect": {"decision": "deny", "reason": "anomaly_confidence_required", "required_action": "set_anomaly_confidence"}},
	{"name": "anomaly_evidence_required", "condition": {"operation": "record_anomaly", "evidence_present": False}, "effect": {"decision": "deny", "reason": "anomaly_evidence_required", "required_action": "attach_anomaly_evidence"}},
	{"name": "model_type_supported", "condition": {"operation": "register_model", "model_type_supported": False}, "effect": {"decision": "deny", "reason": "model_type_not_supported", "required_action": "select_supported_model_type"}},
	{"name": "model_validation_required", "condition": {"operation": "register_model", "validation_present": False}, "effect": {"decision": "deny", "reason": "model_validation_required", "required_action": "attach_model_validation"}},
	{"name": "report_format_supported", "condition": {"operation": "generate_report", "format_supported": False}, "effect": {"decision": "deny", "reason": "report_format_not_supported", "required_action": "select_supported_format"}},
	{"name": "report_approval_required", "condition": {"operation": "generate_report", "approval_present": False}, "effect": {"decision": "deny", "reason": "report_approval_required", "required_action": "attach_report_approval"}},
	{"name": "ana_batch_requires_bytewax", "condition": {"operation": "ana_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_ana_batch_to_bytewax"}},
	{"name": "ana_agent_runtime_supported", "condition": {"operation": "register_ana_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "ana_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "ana_agent_role_supported", "condition": {"operation": "register_ana_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "ana_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "ana_agent_name_required", "condition": {"operation": "register_ana_agent", "agent_name_present": False}, "effect": {"decision": "deny", "reason": "ana_agent_name_required", "required_action": "name_ana_agent"}},
	{"name": "ana_agent_scope_required", "condition": {"operation": "register_ana_agent", "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "ana_agent_scope_required", "required_action": "bound_ana_agent_scope"}},
	{"name": "privileged_ana_agent_action_requires_human_approval", "condition": {"operation": "ana_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
	{"name": "unapproved_model_deployment_denied", "condition": {"operation": "ana_agent_action", "unapproved_model_deployment_scope": True}, "effect": {"decision": "deny", "reason": "unapproved_model_deployment_scope_denied", "required_action": "remove_unapproved_model_deployment_scope"}},
	{"name": "raw_data_export_requires_approval", "condition": {"operation": "ana_agent_action", "raw_data_export_scope": True, "export_approval_present": False}, "effect": {"decision": "deny", "reason": "raw_data_export_approval_required", "required_action": "attach_export_approval"}},
	{"name": "cross_tenant_data_denied", "condition": {"operation": "ana_agent_action", "cross_tenant_data_scope": True}, "effect": {"decision": "deny", "reason": "cross_tenant_data_scope_denied", "required_action": "remove_cross_tenant_data_scope"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "provides": list(PROVIDES), "requires": list(REQUIRES), "configuration": configuration, "configuration_schema": {"type": "object", "required": list(configuration), "properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}}, "rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/telecom-ana/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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
