"""Executable capability contract for APG Logging and Tracing."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any


SUPPORTED_LOGT_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_LOGT_AGENT_ROLES = [
	"pipeline_reviewer",
	"log_reviewer",
	"trace_reviewer",
	"incident_reviewer",
	"privacy_reviewer",
	"retention_reviewer",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"ingestion": {
		"pipeline_owner_required": True,
		"schema_validation_required": True,
		"event_stream": "bytewax",
		"event_stream_required": True,
		"max_batch_size": 10000,
	},
	"tracing": {
		"trace_context_required": True,
		"sampling_policy_required": True,
		"span_retention_days": 30,
		"service_map_enabled": True,
		"span_duration_validation_required": True,
	},
	"privacy": {
		"pii_redaction_required": True,
		"restricted_log_filtering": True,
		"retention_policy_required": True,
		"export_approval_required": True,
		"export_reference_required": True,
	},
	"logt_agents": {
		"agent_assist_enabled": True,
		"agent_registration_required": True,
		"agent_runtime_required": True,
		"agent_role_required": True,
		"agent_scope_required": True,
		"agent_contribution_disclosure_required": True,
		"supported_runtimes": SUPPORTED_LOGT_AGENT_RUNTIMES,
		"allowed_roles": SUPPORTED_LOGT_AGENT_ROLES,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_query_access": True,
		"monitoring_integration_required": True,
		"configuration_policy_required": True,
		"state_change_audit_required": True,
		"tenant_isolation_required": True,
		"batch_event_stream": "bytewax",
	},
	"observability": {
		"audit_required": True,
		"log_metrics_required": True,
		"trace_metrics_required": True,
		"latency_metrics_required": True,
		"agent_activity_required": True,
		"event_stream": "bytewax",
	},
	"adapters": {
		"generated_app_runtime": "service.LogtService",
		"api_helpers": "api.py",
		"view_models": "views.py",
		"event_stream": "bytewax",
		"monitoring": "moni",
		"configuration": "conf",
		"audit_sink": "audl",
		"search": "srch",
		"anomaly_detection": "anom",
		"compliance": "comp",
	},
	"ui": {
		"enable_log_search": True,
		"enable_trace_explorer": True,
		"enable_pipeline_manager": True,
		"enable_retention_center": True,
		"enable_agent_panel": True,
		"enable_audit": True,
		"enable_analytics": True,
	},
	"theme": {
		"default_theme": "logt_observability_console",
		"allow_tenant_overrides": True,
	},
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"ingestion",
		"tracing",
		"privacy",
		"logt_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {
		key: {"type": "object"}
		for key in [
			"ingestion",
			"tracing",
			"privacy",
			"logt_agents",
			"governance",
			"observability",
			"adapters",
			"ui",
			"theme",
		]
	}
	| {"tenant_id": {"type": "string", "minLength": 1}},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All diagnostic operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "pipeline_requires_owner", "description": "Ingestion pipelines require an accountable owner.", "condition": {"operation": "create_pipeline", "pipeline_owner_assigned": False}, "effect": {"decision": "deny", "reason": "pipeline_owner_required", "required_action": "assign_pipeline_owner"}},
	{"name": "pipeline_requires_schema", "description": "Ingestion pipelines require schema validation reference.", "condition": {"operation": "create_pipeline", "schema_ref_present": False}, "effect": {"decision": "deny", "reason": "schema_validation_required", "required_action": "attach_schema_reference"}},
	{"name": "pipeline_requires_bytewax_stream", "description": "Diagnostic ingestion pipelines require Bytewax event streams.", "condition": {"operation": "create_pipeline", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
	{"name": "pipeline_requires_sampling_policy", "description": "Ingestion pipelines require sampling policy.", "condition": {"operation": "create_pipeline", "sampling_policy_present": False}, "effect": {"decision": "deny", "reason": "sampling_policy_required", "required_action": "attach_sampling_policy"}},
	{"name": "trace_context_required", "description": "Trace ingestion requires trace context.", "condition": {"operation": "ingest_trace", "trace_context_present": False}, "effect": {"decision": "deny", "reason": "trace_context_required", "required_action": "attach_trace_context"}},
	{"name": "trace_requires_identifier", "description": "Trace ingestion requires trace identifier.", "condition": {"operation": "ingest_trace", "trace_id_present": False}, "effect": {"decision": "deny", "reason": "trace_id_required", "required_action": "set_trace_id"}},
	{"name": "span_requires_service", "description": "Spans require service identity.", "condition": {"operation": "record_span", "span_service_present": False}, "effect": {"decision": "deny", "reason": "span_service_required", "required_action": "set_span_service"}},
	{"name": "span_requires_valid_duration", "description": "Spans require non-negative duration.", "condition": {"operation": "record_span", "span_duration_valid": False}, "effect": {"decision": "deny", "reason": "span_duration_invalid", "required_action": "fix_span_duration"}},
	{"name": "sensitive_log_requires_redaction", "description": "Sensitive logs must be redacted.", "condition": {"sensitive_log_content": True, "redaction_applied": False}, "effect": {"decision": "deny", "reason": "log_redaction_required", "required_action": "apply_log_redaction"}},
	{"name": "log_requires_service", "description": "Logs require service identity.", "condition": {"operation": "ingest_log", "service_name_present": False}, "effect": {"decision": "deny", "reason": "service_name_required", "required_action": "set_service_name"}},
	{"name": "query_requires_actor", "description": "Diagnostic queries require requester identity.", "condition": {"operation": "search_logs", "query_actor_present": False}, "effect": {"decision": "deny", "reason": "query_actor_required", "required_action": "set_query_actor"}},
	{"name": "large_query_requires_review", "description": "Large diagnostic queries require review.", "condition": {"query_window_hours_gt": 168, "query_review_recorded": False}, "effect": {"decision": "require_review", "reason": "large_query_review_required", "required_action": "review_query_scope"}},
	{"name": "log_export_requires_approval", "description": "Diagnostic exports require approval.", "condition": {"operation": "export_logs", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "export_approval_required", "required_action": "record_export_approval"}},
	{"name": "log_export_requires_approval_reference", "description": "Diagnostic exports require approval reference.", "condition": {"operation": "export_logs", "approval_ref_present": False}, "effect": {"decision": "deny", "reason": "export_approval_reference_required", "required_action": "attach_export_approval_reference"}},
	{"name": "logt_agent_requires_registration", "description": "AI observability agents must be registered.", "condition": {"logt_agent_present": True, "agent_registered": False}, "effect": {"decision": "deny", "reason": "logt_agent_registration_required", "required_action": "register_logt_agent"}},
	{"name": "logt_agent_runtime_supported", "description": "AI observability agents must use a supported runtime.", "condition": {"logt_agent_present": True, "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "logt_agent_runtime_not_supported", "required_action": "choose_supported_logt_agent_runtime"}},
	{"name": "logt_agent_role_supported", "description": "AI observability agents must use a supported role.", "condition": {"logt_agent_present": True, "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "logt_agent_role_not_supported", "required_action": "choose_supported_logt_agent_role"}},
	{"name": "logt_agent_requires_scope", "description": "AI observability agents require explicit scope.", "condition": {"logt_agent_present": True, "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "logt_agent_scope_required", "required_action": "set_logt_agent_scope"}},
	{"name": "logt_agent_requires_disclosure", "description": "AI observability-agent contributions require disclosure.", "condition": {"logt_agent_present": True, "agent_contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "logt_agent_disclosure_required", "required_action": "disclose_logt_agent"}},
	{"name": "logt_state_change_requires_audit", "description": "Diagnostic lifecycle state changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "logt_audit_event_required", "required_action": "record_logt_audit_event"}},
	{"name": "batch_diagnostic_mutation_requires_bytewax", "description": "Batch diagnostic mutations must use Bytewax event streams.", "condition": {"requested_operation": "batch_diagnostic_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/logt/dashboard", "component": "LOGTDashboard", "permission": "logt:view", "nav_group": "Overview"},
	{"name": "logs", "path": "/logt/logs", "component": "LogSearch", "permission": "logt:query", "nav_group": "Diagnostics"},
	{"name": "traces", "path": "/logt/traces", "component": "TraceExplorer", "permission": "logt:query", "nav_group": "Diagnostics"},
	{"name": "spans", "path": "/logt/spans", "component": "SpanDetails", "permission": "logt:query", "nav_group": "Diagnostics"},
	{"name": "pipelines", "path": "/logt/pipelines", "component": "PipelineManager", "permission": "logt:manage_pipelines", "nav_group": "Pipelines"},
	{"name": "retention", "path": "/logt/retention", "component": "RetentionCenter", "permission": "logt:manage_retention", "nav_group": "Governance"},
	{"name": "agents", "path": "/logt/agents", "component": "LOGTAgentPanel", "permission": "logt:admin", "nav_group": "Operations"},
	{"name": "analytics", "path": "/logt/analytics", "component": "TraceAnalytics", "permission": "logt:view", "nav_group": "Operations"},
	{"name": "audit", "path": "/logt/audit", "component": "DiagnosticAuditTrail", "permission": "logt:admin", "nav_group": "Governance"},
	{"name": "settings", "path": "/logt/settings", "component": "LOGTSettings", "permission": "logt:admin", "nav_group": "Administration"},
]

THEME: dict[str, Any] = {
	"name": "logt_observability_console",
	"tokens": {
		"color.primary": "#2A4365",
		"color.accent": "#DD6B20",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F7F8FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact",
	},
	"components": {
		"trace_waterfall": {"icon": "activity", "status_indicator": "latency-pill", "risk_style": "sampling-band"},
		"log_table": {"visual": "structured-log-grid", "highlight": "redaction-chip"},
		"pipeline_graph": {"visual": "ingestion-flow", "status_style": "health-chip"},
		"retention_panel": {"visual": "policy-list", "status_style": "expiry-chip"},
		"agent_panel": {"visual": "agent-roster", "status_style": "scope-chip"},
		"audit_timeline": {"visual": "event-timeline", "status_style": "diagnostic-chip"},
	},
}


def streaming_manifest() -> dict[str, Any]:
	return {
		"processor": "bytewax",
		"topic": "apg.logt.lifecycle",
		"state": ["pipelines", "logs", "traces", "spans", "queries", "exports", "retention_policies", "logt_agents", "audit_events"],
		"events": [
			"logt_pipeline_created",
			"logt_log_ingested",
			"logt_trace_ingested",
			"logt_span_recorded",
			"logt_query_executed",
			"logt_export_created",
			"logt_agent_registered",
		],
		"batch_mutation_guardrail": "batch_diagnostic_mutation_requires_bytewax",
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "logt",
		"display_name": "Logging and Tracing",
		"version": "1.0.0",
		"provides": [
			"structured_logging",
			"distributed_tracing",
			"trace_correlation",
			"log_search",
			"diagnostic_retention",
			"diagnostic_exports",
			"logt_agents",
		],
		"requires": ["moni", "conf", "audl"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/logt/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True,
		},
		"theme": deepcopy(THEME),
		"streaming": streaming_manifest(),
	}


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


def event_stream_name(value: str) -> str:
	return value.strip().lower().split("://", 1)[0]


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_lte"):
			actual = context.get(key[:-4])
			if not isinstance(actual, Number) or not actual <= expected:
				return False
		elif key.endswith("_lt"):
			actual = context.get(key[:-3])
			if not isinstance(actual, Number) or not actual < expected:
				return False
		elif key.endswith("_gt"):
			actual = context.get(key[:-3])
			if not isinstance(actual, Number) or not actual > expected:
				return False
		elif key.endswith("_gte"):
			actual = context.get(key[:-4])
			if not isinstance(actual, Number) or not actual >= expected:
				return False
		elif key.endswith("_ne"):
			actual = context.get(key[:-3])
			if actual is None or actual == expected:
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
