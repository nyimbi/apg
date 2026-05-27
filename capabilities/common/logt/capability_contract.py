"""Executable capability contract for APG Logging and Tracing."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"ingestion": {"pipeline_owner_required": True, "schema_validation_required": True, "event_bus_required": True, "max_batch_size": 10000},
	"tracing": {"trace_context_required": True, "sampling_policy_required": True, "span_retention_days": 30, "service_map_enabled": True},
	"privacy": {"pii_redaction_required": True, "restricted_log_filtering": True, "retention_policy_required": True, "export_approval_required": True},
	"governance": {"require_tenant_context": True, "audit_query_access": True, "monitoring_integration_required": True, "configuration_policy_required": True},
	"ui": {"enable_log_search": True, "enable_trace_explorer": True, "enable_pipeline_manager": True, "enable_retention_center": True},
	"theme": {"default_theme": "logt_observability_console", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "ingestion", "tracing", "privacy", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["ingestion", "tracing", "privacy", "governance", "ui", "theme"]} | {"tenant_id": {"type": "string", "minLength": 1}}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All diagnostic operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "pipeline_requires_owner", "description": "Ingestion pipelines require an accountable owner.", "condition": {"operation": "create_pipeline", "pipeline_owner_assigned": False}, "effect": {"decision": "deny", "reason": "pipeline_owner_required", "required_action": "assign_pipeline_owner"}},
	{"name": "trace_context_required", "description": "Trace ingestion requires trace context.", "condition": {"operation": "ingest_trace", "trace_context_present": False}, "effect": {"decision": "deny", "reason": "trace_context_required", "required_action": "attach_trace_context"}},
	{"name": "sensitive_log_requires_redaction", "description": "Sensitive logs must be redacted.", "condition": {"sensitive_log_content": True, "redaction_applied": False}, "effect": {"decision": "deny", "reason": "log_redaction_required", "required_action": "apply_log_redaction"}},
	{"name": "log_export_requires_approval", "description": "Diagnostic exports require approval.", "condition": {"operation": "export_logs", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "export_approval_required", "required_action": "record_export_approval"}},
	{"name": "large_query_requires_review", "description": "Large diagnostic queries require review.", "condition": {"query_window_hours_gt": 168, "query_review_recorded": False}, "effect": {"decision": "require_review", "reason": "large_query_review_required", "required_action": "review_query_scope"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/logt/dashboard", "component": "LOGTDashboard", "permission": "logt:view", "nav_group": "Overview"},
	{"name": "logs", "path": "/logt/logs", "component": "LogSearch", "permission": "logt:query", "nav_group": "Diagnostics"},
	{"name": "traces", "path": "/logt/traces", "component": "TraceExplorer", "permission": "logt:query", "nav_group": "Diagnostics"},
	{"name": "spans", "path": "/logt/spans", "component": "SpanDetails", "permission": "logt:query", "nav_group": "Diagnostics"},
	{"name": "pipelines", "path": "/logt/pipelines", "component": "PipelineManager", "permission": "logt:manage_pipelines", "nav_group": "Pipelines"},
	{"name": "retention", "path": "/logt/retention", "component": "RetentionCenter", "permission": "logt:manage_retention", "nav_group": "Governance"},
	{"name": "analytics", "path": "/logt/analytics", "component": "TraceAnalytics", "permission": "logt:view", "nav_group": "Operations"},
	{"name": "settings", "path": "/logt/settings", "component": "LOGTSettings", "permission": "logt:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "logt_observability_console",
	"tokens": {"color.primary": "#2A4365", "color.accent": "#DD6B20", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {"trace_waterfall": {"icon": "activity", "status_indicator": "latency-pill", "risk_style": "sampling-band"}, "log_table": {"visual": "structured-log-grid", "highlight": "redaction-chip"}, "pipeline_graph": {"visual": "ingestion-flow", "status_style": "health-chip"}, "retention_panel": {"visual": "policy-list", "status_style": "expiry-chip"}}
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "logt", "display_name": "Logging and Tracing", "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "view_module": "views.py", "api_prefix": "/logt/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME)}


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
