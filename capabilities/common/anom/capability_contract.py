"""Executable capability contract for APG Anomaly Detection."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"sources": {
		"owner_required": True,
		"name_required": True,
		"kind_required": True,
		"allowed_kinds": ["metric", "event", "trace", "forecast", "security"],
	},
	"detection": {
		"metric_detection_enabled": True,
		"event_detection_enabled": True,
		"forecast_residual_detection_enabled": True,
		"minimum_baseline_points": 50,
		"default_sensitivity": "medium",
		"allowed_sensitivities": ["low", "medium", "high"],
	},
	"baselines": {
		"auto_baseline_enabled": True,
		"baseline_review_required": True,
		"drift_reset_requires_approval": True,
		"minimum_refresh_points": 50,
	},
	"signals": {
		"critical_requires_owner": True,
		"high_requires_owner": False,
		"root_cause_hints_enabled": True,
		"cross_tenant_detection_allowed": False,
	},
	"investigation": {
		"critical_anomalies_require_owner": True,
		"closure_evidence_required": True,
		"feedback_loop_enabled": True,
		"root_cause_hints_enabled": True,
	},
	"feedback": {
		"reviewer_required": True,
		"allowed_labels": ["true_positive", "false_positive", "expected_change"],
		"false_positive_review_threshold": 0.2,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_detections": True,
		"monitoring_source_required": True,
		"auth_required": True,
	},
	"observability": {
		"metrics_required": True,
		"trace_required": True,
		"audit_required": True,
		"event_stream": "bytewax",
		"quality_metrics_required": True,
	},
	"adapters": {
		"generated_app_runtime": "service.AnomService",
		"helper_runtime": "anomaly_engine.py",
		"production_runtime": "service.AnomService",
		"http_api": "api.py",
		"event_stream": "bytewax",
		"predictive_analytics": "pred",
		"ai_core": "aicr",
		"monitoring": "moni",
		"workflow": "wflo",
		"notification": "ntfy",
		"health": "hlth",
		"configuration": "conf",
		"auth_provider": "auth",
		"audit_sink": "audl",
		"metrics_sink": "moni",
		"cache": "cach",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_sources": True,
		"enable_baseline_console": True,
		"enable_detector": True,
		"enable_signals": True,
		"enable_investigation_queue": True,
		"enable_alerts": True,
		"enable_rules": True,
		"enable_feedback_review": True,
		"enable_quality": True,
		"enable_audit": True,
		"enable_settings": True,
	},
	"theme": {"default_theme": "anom_signal_console", "allow_tenant_overrides": True},
}


CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"sources",
		"detection",
		"baselines",
		"signals",
		"investigation",
		"feedback",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {key: {"type": "object"} for key in [
		"sources",
		"detection",
		"baselines",
		"signals",
		"investigation",
		"feedback",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]} | {"tenant_id": {"type": "string", "minLength": 1}},
}


RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All anomaly detection operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "source_requires_name", "description": "Monitoring sources require a display name.", "condition": {"operation": "register_source", "source_name_present": False}, "effect": {"decision": "deny", "reason": "source_name_required", "required_action": "attach_source_name"}},
	{"name": "source_requires_owner", "description": "Monitoring sources require an accountable owner.", "condition": {"operation": "register_source", "source_owner_present": False}, "effect": {"decision": "deny", "reason": "source_owner_required", "required_action": "assign_source_owner"}},
	{"name": "source_requires_kind", "description": "Monitoring sources require a kind.", "condition": {"operation": "register_source", "source_kind_present": False}, "effect": {"decision": "deny", "reason": "source_kind_required", "required_action": "attach_source_kind"}},
	{"name": "source_kind_requires_review", "description": "Unknown monitoring source kinds require review.", "condition": {"operation": "register_source", "source_kind_known": False}, "effect": {"decision": "require_review", "reason": "source_kind_review_required", "required_action": "review_source_kind"}},
	{"name": "baseline_requires_source", "description": "Baseline creation requires a registered monitoring source.", "condition": {"operation": "create_baseline", "monitoring_source_present": False}, "effect": {"decision": "deny", "reason": "baseline_source_required", "required_action": "select_monitoring_source"}},
	{"name": "baseline_requires_metric", "description": "Baselines require a metric name.", "condition": {"operation": "create_baseline", "metric_present": False}, "effect": {"decision": "deny", "reason": "baseline_metric_required", "required_action": "attach_metric"}},
	{"name": "baseline_requires_history", "description": "Baselines require enough historical observations.", "condition": {"operation": "create_baseline", "history_points_lt": 50}, "effect": {"decision": "deny", "reason": "insufficient_baseline_history", "required_action": "collect_more_history"}},
	{"name": "baseline_requires_sensitivity", "description": "Baselines require sensitivity metadata.", "condition": {"operation": "create_baseline", "sensitivity_present": False}, "effect": {"decision": "deny", "reason": "baseline_sensitivity_required", "required_action": "choose_sensitivity"}},
	{"name": "baseline_sensitivity_requires_review", "description": "Unknown sensitivity values require tuning review.", "condition": {"operation": "create_baseline", "sensitivity_known": False}, "effect": {"decision": "require_review", "reason": "baseline_sensitivity_review_required", "required_action": "review_sensitivity"}},
	{"name": "detection_requires_monitoring_source", "description": "Anomaly detection requires a monitoring source.", "condition": {"operation": "detect", "monitoring_source_present": False}, "effect": {"decision": "deny", "reason": "monitoring_source_required", "required_action": "attach_monitoring_source"}},
	{"name": "detection_requires_baseline", "description": "Anomaly detection requires a baseline.", "condition": {"operation": "detect", "baseline_present": False}, "effect": {"decision": "deny", "reason": "baseline_required", "required_action": "select_baseline"}},
	{"name": "detection_requires_metric", "description": "Anomaly detection requires a metric.", "condition": {"operation": "detect", "metric_present": False}, "effect": {"decision": "deny", "reason": "detection_metric_required", "required_action": "attach_metric"}},
	{"name": "detection_requires_value", "description": "Anomaly detection requires an observed value.", "condition": {"operation": "detect", "value_present": False}, "effect": {"decision": "deny", "reason": "observation_value_required", "required_action": "attach_observation_value"}},
	{"name": "critical_anomaly_requires_owner", "description": "Critical anomalies require an investigation owner.", "condition": {"operation": "detect", "severity": "critical", "owner_assigned": False}, "effect": {"decision": "deny", "reason": "investigation_owner_required", "required_action": "assign_owner"}},
	{"name": "high_anomaly_requires_triage", "description": "High-severity anomalies require triage evidence.", "condition": {"operation": "detect", "severity": "high", "triage_recorded": False}, "effect": {"decision": "require_review", "reason": "high_anomaly_triage_required", "required_action": "record_triage"}},
	{"name": "cross_tenant_detection_denied", "description": "Cross-tenant source or baseline use is denied.", "condition": {"cross_tenant_detection": True}, "effect": {"decision": "deny", "reason": "cross_tenant_detection_denied", "required_action": "use_tenant_scoped_detection"}},
	{"name": "investigation_requires_signal", "description": "Investigations require an anomaly signal.", "condition": {"operation": "open_investigation", "signal_present": False}, "effect": {"decision": "deny", "reason": "investigation_signal_required", "required_action": "select_signal"}},
	{"name": "investigation_requires_owner", "description": "Investigations require an owner.", "condition": {"operation": "open_investigation", "owner_assigned": False}, "effect": {"decision": "deny", "reason": "investigation_owner_required", "required_action": "assign_owner"}},
	{"name": "investigation_close_requires_resolution", "description": "Investigation closure requires a resolution.", "condition": {"operation": "close_investigation", "resolution_present": False}, "effect": {"decision": "deny", "reason": "investigation_resolution_required", "required_action": "attach_resolution"}},
	{"name": "investigation_close_requires_actor", "description": "Investigation closure requires the closing actor.", "condition": {"operation": "close_investigation", "closed_by_present": False}, "effect": {"decision": "deny", "reason": "investigation_closer_required", "required_action": "attach_closer"}},
	{"name": "investigation_close_requires_evidence", "description": "Investigation closure requires resolution evidence.", "condition": {"operation": "close_investigation", "resolution_evidence_present": False}, "effect": {"decision": "deny", "reason": "investigation_resolution_evidence_required", "required_action": "attach_resolution_evidence"}},
	{"name": "feedback_requires_signal", "description": "Feedback requires an anomaly signal.", "condition": {"operation": "record_feedback", "signal_present": False}, "effect": {"decision": "deny", "reason": "feedback_signal_required", "required_action": "select_signal"}},
	{"name": "feedback_requires_reviewer", "description": "Feedback requires a reviewer.", "condition": {"operation": "record_feedback", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "feedback_reviewer_required", "required_action": "assign_reviewer"}},
	{"name": "feedback_requires_label", "description": "Feedback requires a label.", "condition": {"operation": "record_feedback", "label_present": False}, "effect": {"decision": "deny", "reason": "feedback_label_required", "required_action": "choose_feedback_label"}},
	{"name": "feedback_label_requires_review", "description": "Unknown feedback labels require review.", "condition": {"operation": "record_feedback", "label_known": False}, "effect": {"decision": "require_review", "reason": "feedback_label_review_required", "required_action": "review_feedback_label"}},
	{"name": "high_false_positive_rate_requires_tuning", "description": "High false-positive rates require tuning review.", "condition": {"operation": "record_feedback", "false_positive_rate_gt": 0.2, "tuning_review_recorded": False}, "effect": {"decision": "require_review", "reason": "tuning_review_required", "required_action": "record_tuning_review"}},
	{"name": "baseline_reset_requires_approval", "description": "Baseline reset after drift requires approval.", "condition": {"operation": "reset_baseline", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "baseline_reset_approval_required", "required_action": "record_approval"}},
	{"name": "batch_detection_requires_bytewax", "description": "Batch anomaly detection streams must use Bytewax.", "condition": {"operation": "configure_batch_detection", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
	{"name": "alert_dispatch_requires_notification_adapter", "description": "Alert dispatch requires a notification adapter.", "condition": {"operation": "dispatch_alert", "notification_adapter_present": False}, "effect": {"decision": "deny", "reason": "notification_adapter_required", "required_action": "configure_notification_adapter"}},
	{"name": "anomaly_state_change_requires_audit", "description": "Anomaly state changes require audit events.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "audit_event_required", "required_action": "record_audit_event"}},
]


UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/anom/dashboard", "component": "ANOMDashboard", "permission": "anom:view", "nav_group": "Overview"},
	{"name": "sources", "path": "/anom/sources", "component": "MonitoringSourceRegistry", "permission": "anom:tune", "nav_group": "Sources"},
	{"name": "baselines", "path": "/anom/baselines", "component": "BaselineConsole", "permission": "anom:tune", "nav_group": "Baselines"},
	{"name": "detector", "path": "/anom/detector", "component": "DetectionWorkbench", "permission": "anom:detect", "nav_group": "Detection"},
	{"name": "signals", "path": "/anom/signals", "component": "SignalBoard", "permission": "anom:detect", "nav_group": "Signals"},
	{"name": "investigations", "path": "/anom/investigations", "component": "InvestigationQueue", "permission": "anom:investigate", "nav_group": "Investigations"},
	{"name": "alerts", "path": "/anom/alerts", "component": "AlertDispatchQueue", "permission": "anom:investigate", "nav_group": "Investigations"},
	{"name": "rules", "path": "/anom/rules", "component": "AnomalyRuleManager", "permission": "anom:manage_rules", "nav_group": "Governance"},
	{"name": "feedback", "path": "/anom/feedback", "component": "FeedbackReview", "permission": "anom:tune", "nav_group": "Quality"},
	{"name": "quality", "path": "/anom/quality", "component": "DetectionQuality", "permission": "anom:tune", "nav_group": "Quality"},
	{"name": "audit", "path": "/anom/audit", "component": "AnomalyAuditTimeline", "permission": "anom:audit", "nav_group": "Governance"},
	{"name": "settings", "path": "/anom/settings", "component": "ANOMSettings", "permission": "anom:admin", "nav_group": "Administration"},
]


THEME: dict[str, Any] = {
	"name": "anom_signal_console",
	"tokens": {
		"color.primary": "#2D5A87",
		"color.accent": "#D1495B",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F6F8FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact",
	},
	"components": {
		"signal_card": {"icon": "activity", "status_indicator": "severity-pill", "risk_style": "anomaly-band"},
		"baseline_chart": {"visual": "threshold-band", "highlight": "drift-marker"},
		"source_registry": {"visual": "source-table", "status_style": "kind-chip"},
		"detection_workbench": {"visual": "score-panel", "status_style": "threshold-chip"},
		"investigation_timeline": {"visual": "event-timeline", "status_style": "owner-chip"},
		"alert_queue": {"visual": "queue-table", "status_style": "notification-chip"},
		"feedback_panel": {"visual": "review-stack", "threshold_style": "false-positive-meter"},
		"quality_dashboard": {"visual": "quality-grid", "status_style": "tuning-chip"},
		"audit_timeline": {"visual": "event-timeline", "status_style": "evidence-chip"},
	},
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable ANOM capability contract."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "anom",
		"display_name": "Anomaly Detection",
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/anom/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True,
		},
		"theme": deepcopy(THEME),
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate default ANOM governance rules."""
	matched: list[str] = []
	actions: list[dict[str, Any]] = []
	decision = "allow"
	for rule in RULES:
		if _matches(rule["condition"], context):
			matched.append(rule["name"])
			effect = rule["effect"]
			actions.append(effect)
			if effect["decision"] == "deny":
				decision = "deny"
			elif effect["decision"] == "require_review" and decision != "deny":
				decision = "require_review"
	return {"decision": decision, "matched_rules": matched, "actions": actions, "context": context}


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_lt"):
			if key[:-3] not in context or not context[key[:-3]] < expected:
				return False
		elif key.endswith("_gt"):
			if key[:-3] not in context or not context[key[:-3]] > expected:
				return False
		elif key.endswith("_ne"):
			if key[:-3] not in context or context[key[:-3]] == expected:
				return False
		elif key not in context or context[key] != expected:
			return False
	return True


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
	for key, value in source.items():
		if isinstance(value, dict) and isinstance(target.get(key), dict):
			_deep_merge(target[key], value)
		else:
			target[key] = value
