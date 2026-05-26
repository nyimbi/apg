"""Executable capability contract for APG Anomaly Detection."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"detection": {"metric_detection_enabled": True, "event_detection_enabled": True, "minimum_baseline_points": 50, "default_sensitivity": "medium"},
	"baselines": {"auto_baseline_enabled": True, "baseline_review_required": True, "drift_reset_requires_approval": True},
	"investigation": {"critical_anomalies_require_owner": True, "feedback_loop_enabled": True, "root_cause_hints_enabled": True},
	"governance": {"require_tenant_context": True, "audit_detections": True, "monitoring_source_required": True},
	"ui": {"enable_signal_board": True, "enable_baseline_console": True, "enable_investigation_queue": True, "enable_feedback_review": True},
	"theme": {"default_theme": "anom_signal_console", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "detection", "baselines", "investigation", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["detection", "baselines", "investigation", "governance", "ui", "theme"]} | {
		"tenant_id": {"type": "string", "minLength": 1}
	}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All anomaly detection operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "detection_requires_monitoring_source", "description": "Anomaly detection requires a monitoring source.", "condition": {"operation": "detect", "monitoring_source_present": False}, "effect": {"decision": "deny", "reason": "monitoring_source_required", "required_action": "attach_monitoring_source"}},
	{"name": "baseline_requires_history", "description": "Baselines require enough historical observations.", "condition": {"operation": "create_baseline", "history_points_lt": 50}, "effect": {"decision": "deny", "reason": "insufficient_baseline_history", "required_action": "collect_more_history"}},
	{"name": "critical_anomaly_requires_owner", "description": "Critical anomalies require an investigation owner.", "condition": {"severity": "critical", "owner_assigned": False}, "effect": {"decision": "deny", "reason": "investigation_owner_required", "required_action": "assign_owner"}},
	{"name": "baseline_reset_requires_approval", "description": "Baseline reset after drift requires approval.", "condition": {"operation": "reset_baseline", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "baseline_reset_approval_required", "required_action": "record_approval"}},
	{"name": "high_false_positive_rate_requires_tuning", "description": "High false-positive rates require tuning review.", "condition": {"false_positive_rate_gt": 0.2, "tuning_review_recorded": False}, "effect": {"decision": "require_review", "reason": "tuning_review_required", "required_action": "record_tuning_review"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/anom/dashboard", "component": "ANOMDashboard", "permission": "anom:view", "nav_group": "Overview"},
	{"name": "signals", "path": "/anom/signals", "component": "SignalBoard", "permission": "anom:detect", "nav_group": "Signals"},
	{"name": "baselines", "path": "/anom/baselines", "component": "BaselineConsole", "permission": "anom:tune", "nav_group": "Baselines"},
	{"name": "investigations", "path": "/anom/investigations", "component": "InvestigationQueue", "permission": "anom:investigate", "nav_group": "Investigations"},
	{"name": "rules", "path": "/anom/rules", "component": "AnomalyRuleManager", "permission": "anom:manage_rules", "nav_group": "Governance"},
	{"name": "feedback", "path": "/anom/feedback", "component": "FeedbackReview", "permission": "anom:tune", "nav_group": "Quality"},
	{"name": "settings", "path": "/anom/settings", "component": "ANOMSettings", "permission": "anom:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "anom_signal_console",
	"tokens": {"color.primary": "#334E68", "color.accent": "#D1495B", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F6F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {
		"signal_card": {"icon": "activity", "status_indicator": "severity-pill", "risk_style": "anomaly-band"},
		"baseline_chart": {"visual": "threshold-band", "highlight": "drift-marker"},
		"investigation_timeline": {"visual": "event-timeline", "status_style": "owner-chip"},
		"feedback_panel": {"visual": "review-stack", "threshold_style": "false-positive-meter"}
	}
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "anom", "display_name": "Anomaly Detection", "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "flask_appbuilder", "view_module": "__init__.py", "api_prefix": "/anom/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME)}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
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
			if not context.get(key[:-3], 0) < expected:
				return False
		elif key.endswith("_gt"):
			if not context.get(key[:-3], 0) > expected:
				return False
		elif context.get(key) != expected:
			return False
	return True


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
	for key, value in source.items():
		if isinstance(value, dict) and isinstance(target.get(key), dict):
			_deep_merge(target[key], value)
		else:
			target[key] = value
