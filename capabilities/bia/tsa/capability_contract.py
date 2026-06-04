"""Executable capability contract for APG Time Series Analytics (bia_tsa)."""

from __future__ import annotations
from copy import deepcopy
from typing import Any

CAPABILITY_ID = "bia_tsa"
CAPABILITY_NAME = "Time Series Analytics"
CAPABILITY_VERSION = "1.0.0"
TSA_EVENT_STREAM = "apg.bia.tsa.lifecycle"

SUPPORTED_INGESTION_PROTOCOLS = ["mqtt", "websocket", "http_push", "polling", "file_watch", "grpc"]
SUPPORTED_FREQUENCIES = ["tick", "1s", "5s", "10s", "30s", "1m", "5m", "15m", "1h", "4h", "1d", "1w"]
SUPPORTED_ANOMALY_METHODS = ["zscore", "iqr", "isolation_forest", "lstm_autoencoder", "prophet_residual", "mad", "seasonal_decomposition", "custom"]
SUPPORTED_DECOMPOSITION_COMPONENTS = ["trend", "seasonality", "residual", "cyclical"]
SUPPORTED_FORECAST_MODELS = ["arima", "sarima", "prophet", "exponential_smoothing", "lstm", "transformer", "ensemble"]
SUPPORTED_WINDOW_TYPES = ["tumbling", "sliding", "session", "hopping"]
SUPPORTED_STREAM_STATES = ["active", "paused", "error", "archived"]
SUPPORTED_AGGREGATION_FUNCTIONS = ["sum", "avg", "min", "max", "count", "first", "last", "stddev", "percentile"]
SUPPORTED_INTERPOLATION_METHODS = ["linear", "forward_fill", "backward_fill", "cubic_spline", "zero", "none"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["stream_engineer", "anomaly_analyst", "forecast_builder", "decomposition_analyst", "alert_configurator"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"streams": {"supported_protocols": SUPPORTED_INGESTION_PROTOCOLS, "supported_frequencies": SUPPORTED_FREQUENCIES, "supported_states": SUPPORTED_STREAM_STATES, "max_streams_per_tenant": 200, "require_owner": True},
	"anomaly_detection": {"supported_methods": SUPPORTED_ANOMALY_METHODS, "default_method": "zscore", "alert_on_anomaly": True, "sensitivity_default": 0.95},
	"decomposition": {"supported_components": SUPPORTED_DECOMPOSITION_COMPONENTS, "additive_model": True, "multiplicative_model": True},
	"forecasting": {"supported_models": SUPPORTED_FORECAST_MODELS, "default_model": "prophet", "max_horizon_periods": 365, "confidence_interval_default": 0.95},
	"windowing": {"supported_types": SUPPORTED_WINDOW_TYPES, "default_type": "tumbling", "max_window_size_seconds": 86400},
	"aggregation": {"supported_functions": SUPPORTED_AGGREGATION_FUNCTIONS, "allow_multi_function": True},
	"interpolation": {"supported_methods": SUPPORTED_INTERPOLATION_METHODS, "default_method": "forward_fill"},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "cross_tenant_stream_access_denied": True},
	"observability": {"event_stream": TSA_EVENT_STREAM, "stream_processor": "bytewax"},
	"theme": {"default_theme": "bia_tsa_timeseries", "allow_tenant_overrides": True},
}

PROVIDES = ["high_frequency_time_series_ingestion", "anomaly_detection", "seasonality_decomposition", "time_series_forecasting", "stream_windowing", "multi_stream_correlation", "gap_filling_interpolation", "real_time_alerting"]

REQUIRES = ["auth", "audl", "mten", "conf", "mqeb", "moni", "ntfy", "schd"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/bia/tsa/dashboard", "component": "TimeSeriesDashboard", "permission": "bia_tsa:view", "nav_group": "Overview"},
	{"name": "streams", "path": "/bia/tsa/streams", "component": "StreamManager", "permission": "bia_tsa:streams", "nav_group": "Streams"},
	{"name": "stream_detail", "path": "/bia/tsa/streams/<id>", "component": "StreamDetail", "permission": "bia_tsa:streams", "nav_group": "Streams"},
	{"name": "stream_explorer", "path": "/bia/tsa/streams/<id>/explore", "component": "StreamExplorer", "permission": "bia_tsa:streams", "nav_group": "Streams"},
	{"name": "anomaly_detection", "path": "/bia/tsa/anomalies", "component": "AnomalyDetectionConsole", "permission": "bia_tsa:anomalies", "nav_group": "Analysis"},
	{"name": "anomaly_detail", "path": "/bia/tsa/anomalies/<id>", "component": "AnomalyDetail", "permission": "bia_tsa:anomalies", "nav_group": "Analysis"},
	{"name": "decomposition", "path": "/bia/tsa/decomposition", "component": "DecompositionAnalyser", "permission": "bia_tsa:decompose", "nav_group": "Analysis"},
	{"name": "forecasts", "path": "/bia/tsa/forecasts", "component": "ForecastManager", "permission": "bia_tsa:forecast", "nav_group": "Forecasting"},
	{"name": "forecast_detail", "path": "/bia/tsa/forecasts/<id>", "component": "ForecastDetail", "permission": "bia_tsa:forecast", "nav_group": "Forecasting"},
	{"name": "windows", "path": "/bia/tsa/windows", "component": "WindowManager", "permission": "bia_tsa:streams", "nav_group": "Processing"},
	{"name": "alerts", "path": "/bia/tsa/alerts", "component": "StreamAlertManager", "permission": "bia_tsa:alerts", "nav_group": "Alerting"},
	{"name": "audit_log", "path": "/bia/tsa/audit", "component": "TSAAuditLog", "permission": "bia_tsa:admin", "nav_group": "Governance"},
	{"name": "settings", "path": "/bia/tsa/settings", "component": "TSASettings", "permission": "bia_tsa:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "bia_tsa_timeseries",
	"tokens": {"color.primary": "#0E4D6D", "color.accent": "#00BFA5", "color.success": "#1B5E20", "color.warning": "#E65100", "color.danger": "#B71C1C", "surface.canvas": "#E8F5F9", "surface.panel": "#FFFFFF", "text.primary": "#0A1A24", "text.secondary": "#37474F", "border.radius": "6px", "density": "compact"},
	"components": {
		"stream": {"icon": "activity", "status_indicator": "stream-state-chip"},
		"anomaly": {"icon": "alert-triangle", "status_indicator": "anomaly-severity-chip"},
		"decomposition": {"icon": "layers", "status_indicator": "component-chip"},
		"forecast": {"icon": "trending-up", "status_indicator": "model-chip"},
		"window": {"icon": "square", "status_indicator": "window-type-chip"},
	},
}

STREAMING = {
	"processor": "bytewax", "stream": TSA_EVENT_STREAM, "key": "tenant_id",
	"events": ["stream_registered", "stream_data_ingested", "anomaly_detected", "anomaly_confirmed", "decomposition_completed", "forecast_generated", "window_opened", "window_closed", "alert_triggered", "gap_filled", "stream_paused", "stream_resumed"],
	"guardrails": ["cross_tenant_stream_access_denied", "max_streams_per_tenant_enforced", "anomaly_alert_gated", "high_frequency_rate_limited"],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "write_policy_required", "required_action": "attach_policy"}},
	{"name": "cross_tenant_stream_access_denied", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_stream_access_not_permitted", "required_action": "restrict_to_tenant"}},
	{"name": "ingestion_protocol_supported", "condition": {"operation": "register_stream", "protocol_supported": False}, "effect": {"decision": "deny", "reason": "ingestion_protocol_not_supported", "required_action": "select_supported_protocol"}},
	{"name": "stream_frequency_supported", "condition": {"operation": "register_stream", "frequency_supported": False}, "effect": {"decision": "deny", "reason": "stream_frequency_not_supported", "required_action": "select_supported_frequency"}},
	{"name": "stream_owner_required", "condition": {"operation": "register_stream", "owner_present": False}, "effect": {"decision": "deny", "reason": "stream_owner_required", "required_action": "attach_stream_owner"}},
	{"name": "max_streams_enforced", "condition": {"operation": "register_stream", "stream_limit_exceeded": True}, "effect": {"decision": "deny", "reason": "max_streams_per_tenant_exceeded", "required_action": "archive_unused_streams"}},
	{"name": "anomaly_method_supported", "condition": {"operation": "configure_anomaly_detection", "method_supported": False}, "effect": {"decision": "deny", "reason": "anomaly_method_not_supported", "required_action": "select_supported_anomaly_method"}},
	{"name": "decomposition_component_supported", "condition": {"operation": "run_decomposition", "component_supported": False}, "effect": {"decision": "deny", "reason": "decomposition_component_not_supported", "required_action": "select_supported_component"}},
	{"name": "forecast_model_supported", "condition": {"operation": "create_forecast", "model_supported": False}, "effect": {"decision": "deny", "reason": "forecast_model_not_supported", "required_action": "select_supported_forecast_model"}},
	{"name": "window_type_supported", "condition": {"operation": "create_window", "window_type_supported": False}, "effect": {"decision": "deny", "reason": "window_type_not_supported", "required_action": "select_supported_window_type"}},
	{"name": "window_size_limit_enforced", "condition": {"operation": "create_window", "window_size_exceeded": True}, "effect": {"decision": "deny", "reason": "window_size_exceeds_maximum", "required_action": "reduce_window_size"}},
	{"name": "aggregation_function_supported", "condition": {"operation": "aggregate_stream", "function_supported": False}, "effect": {"decision": "deny", "reason": "aggregation_function_not_supported", "required_action": "select_supported_aggregation_function"}},
	{"name": "interpolation_method_supported", "condition": {"operation": "fill_gaps", "method_supported": False}, "effect": {"decision": "deny", "reason": "interpolation_method_not_supported", "required_action": "select_supported_interpolation_method"}},
	{"name": "paused_stream_cannot_ingest", "condition": {"operation": "ingest_data", "stream_state": "paused"}, "effect": {"decision": "deny", "reason": "paused_stream_cannot_accept_data", "required_action": "resume_stream_first"}},
	{"name": "archived_stream_read_only", "condition": {"operation": "ingest_data", "stream_state": "archived"}, "effect": {"decision": "deny", "reason": "archived_stream_is_read_only", "required_action": "create_new_stream"}},
	{"name": "forecast_requires_sufficient_history", "condition": {"operation": "create_forecast", "history_sufficient": False}, "effect": {"decision": "deny", "reason": "insufficient_history_for_forecast", "required_action": "collect_more_data_before_forecasting"}},
	{"name": "anomaly_alert_gated", "condition": {"operation": "trigger_alert", "alert_rate_exceeded": True}, "effect": {"decision": "deny", "reason": "anomaly_alert_rate_limit_exceeded", "required_action": "adjust_sensitivity_or_suppress_window"}},
	{"name": "audit_anomaly_detections", "condition": {"operation": "detect_anomaly", "audit_enabled": True}, "effect": {"decision": "allow", "reason": "anomaly_detection_audited", "required_action": "emit_anomaly_detected_event"}},
	{"name": "horizon_limit_enforced", "condition": {"operation": "create_forecast", "horizon_exceeded": True}, "effect": {"decision": "deny", "reason": "forecast_horizon_exceeds_maximum", "required_action": "reduce_forecast_horizon"}},
	{"name": "error_stream_requires_remediation", "condition": {"operation": "ingest_data", "stream_state": "error"}, "effect": {"decision": "deny", "reason": "stream_in_error_state", "required_action": "resolve_stream_error_first"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION,
		"configuration": config,
		"configuration_schema": {"required": ["tenant_id", "ui", "theme"], "properties": {"tenant_id": {"type": "string"}, "ui": {"type": "object"}, "theme": {"type": "object"}}},
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": RULES},
		"ui": {"shell": "apg_python", "requires_theme": True, "template_roots": ["bia/tsa/templates"], "routes": UI_ROUTES},
		"theme": THEME, "streaming": STREAMING, "provides": PROVIDES, "requires": REQUIRES,
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	for rule in RULES:
		if all(context.get(k) == v for k, v in rule["condition"].items()):
			return {"matched_rule": rule["name"], "decision": rule["effect"]["decision"], "reason": rule["effect"]["reason"], "required_action": rule["effect"]["required_action"]}
	return {"matched_rule": None, "decision": "allow", "reason": "no_rule_matched", "required_action": None}
