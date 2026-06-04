"""Executable capability contract for APG Analytics Engine (BIA)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "bia_anl"
CAPABILITY_NAME = "Analytics Engine"
CAPABILITY_VERSION = "1.0.0"
ANALYTICS_EVENT_STREAM = "apg.bia.anl.lifecycle"

SUPPORTED_QUERY_TYPES = ["ad_hoc", "scheduled", "parameterised", "saved", "template", "federated", "streaming"]
SUPPORTED_AGGREGATIONS = ["sum", "avg", "min", "max", "count", "distinct_count", "median", "percentile", "stddev", "variance"]
SUPPORTED_DIMENSIONS = ["time", "geography", "product", "customer", "channel", "organisation", "category", "custom"]
SUPPORTED_CUBE_STATES = ["building", "active", "stale", "refreshing", "error", "archived"]
SUPPORTED_METRIC_TYPES = ["kpi", "derived", "ratio", "cumulative", "moving_average", "year_on_year", "budget_vs_actual", "custom"]
SUPPORTED_ACCESS_LEVELS = ["public", "team", "private", "restricted"]
SUPPORTED_OUTPUT_FORMATS = ["json", "csv", "parquet", "arrow", "xlsx", "html"]
SUPPORTED_CACHE_POLICIES = ["no_cache", "session", "hourly", "daily", "weekly", "manual"]
SUPPORTED_JOIN_TYPES = ["inner", "left", "right", "full", "cross", "semi", "anti"]
SUPPORTED_FILTER_OPERATORS = ["eq", "neq", "gt", "gte", "lt", "lte", "in", "not_in", "like", "between", "is_null", "is_not_null"]
SUPPORTED_SORT_DIRECTIONS = ["asc", "desc"]
SUPPORTED_DATASOURCE_TYPES = ["postgresql", "mysql", "bigquery", "snowflake", "redshift", "duckdb", "parquet_file", "csv_file", "api_endpoint"]
SUPPORTED_REVIEW_STATUSES = ["pending", "approved", "rejected", "needs_changes"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["query_author", "metric_steward", "cube_builder", "access_reviewer", "result_validator"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"queries": {
		"supported_query_types": SUPPORTED_QUERY_TYPES,
		"max_rows_per_query": 100_000,
		"timeout_seconds": 300,
		"require_owner": True,
		"require_evidence": False,
	},
	"cubes": {
		"supported_states": SUPPORTED_CUBE_STATES,
		"supported_dimensions": SUPPORTED_DIMENSIONS,
		"supported_aggregations": SUPPORTED_AGGREGATIONS,
		"max_dimensions": 20,
		"max_measures": 50,
	},
	"metrics": {
		"supported_metric_types": SUPPORTED_METRIC_TYPES,
		"require_owner": True,
		"require_formula": True,
	},
	"access": {
		"supported_levels": SUPPORTED_ACCESS_LEVELS,
		"default_level": "private",
		"require_approval_for_public": True,
	},
	"cache": {
		"supported_policies": SUPPORTED_CACHE_POLICIES,
		"default_policy": "session",
		"max_cache_size_mb": 512,
	},
	"datasources": {
		"supported_types": SUPPORTED_DATASOURCE_TYPES,
		"require_connection_test": True,
		"credentials_vault_required": True,
	},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_events": True,
		"cross_tenant_query_denied": True,
		"unapproved_public_access_denied": True,
	},
	"observability": {
		"event_stream": ANALYTICS_EVENT_STREAM,
		"stream_processor": "bytewax",
	},
	"ui": {
		"enable_query_builder": True,
		"enable_cube_explorer": True,
		"enable_metric_library": True,
		"enable_datasources": True,
		"enable_saved_queries": True,
	},
	"theme": {"default_theme": "bia_anl_analytics", "allow_tenant_overrides": True},
}

PROVIDES = [
	"ad_hoc_query_execution",
	"olap_cube_management",
	"metric_definition_registry",
	"analytical_data_access",
	"query_result_cache",
	"datasource_connectivity",
	"saved_query_library",
	"query_scheduling",
	"result_export",
]

REQUIRES = ["auth", "audl", "mten", "conf", "schd", "mqeb", "moni", "nlpc"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/bia/anl/dashboard", "component": "AnalyticsDashboard", "permission": "bia_anl:view", "nav_group": "Overview"},
	{"name": "query_builder", "path": "/bia/anl/query-builder", "component": "QueryBuilder", "permission": "bia_anl:query", "nav_group": "Querying"},
	{"name": "saved_queries", "path": "/bia/anl/saved-queries", "component": "SavedQueryLibrary", "permission": "bia_anl:query", "nav_group": "Querying"},
	{"name": "query_detail", "path": "/bia/anl/saved-queries/<id>", "component": "QueryDetail", "permission": "bia_anl:query", "nav_group": "Querying"},
	{"name": "cube_explorer", "path": "/bia/anl/cubes", "component": "CubeExplorer", "permission": "bia_anl:cubes", "nav_group": "OLAP"},
	{"name": "cube_detail", "path": "/bia/anl/cubes/<id>", "component": "CubeDetail", "permission": "bia_anl:cubes", "nav_group": "OLAP"},
	{"name": "metric_library", "path": "/bia/anl/metrics", "component": "MetricLibrary", "permission": "bia_anl:metrics", "nav_group": "Metrics"},
	{"name": "metric_detail", "path": "/bia/anl/metrics/<id>", "component": "MetricDetail", "permission": "bia_anl:metrics", "nav_group": "Metrics"},
	{"name": "datasources", "path": "/bia/anl/datasources", "component": "DatasourceManager", "permission": "bia_anl:admin", "nav_group": "Configuration"},
	{"name": "datasource_detail", "path": "/bia/anl/datasources/<id>", "component": "DatasourceDetail", "permission": "bia_anl:admin", "nav_group": "Configuration"},
	{"name": "results", "path": "/bia/anl/results/<query_id>", "component": "QueryResults", "permission": "bia_anl:query", "nav_group": "Querying"},
	{"name": "schedules", "path": "/bia/anl/schedules", "component": "QueryScheduleManager", "permission": "bia_anl:schedule", "nav_group": "Automation"},
	{"name": "audit_log", "path": "/bia/anl/audit", "component": "AnalyticsAuditLog", "permission": "bia_anl:admin", "nav_group": "Governance"},
	{"name": "settings", "path": "/bia/anl/settings", "component": "AnalyticsSettings", "permission": "bia_anl:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "bia_anl_analytics",
	"tokens": {
		"color.primary": "#1D4ED8",
		"color.accent": "#0891B2",
		"color.success": "#15803D",
		"color.warning": "#B45309",
		"color.danger": "#B91C1C",
		"surface.canvas": "#F0F9FF",
		"surface.panel": "#FFFFFF",
		"text.primary": "#0F172A",
		"text.secondary": "#475569",
		"border.radius": "6px",
		"density": "comfortable",
	},
	"components": {
		"query": {"icon": "database", "status_indicator": "query-status-chip"},
		"cube": {"icon": "box", "status_indicator": "cube-state-chip"},
		"metric": {"icon": "trending-up", "status_indicator": "metric-type-chip"},
		"datasource": {"icon": "server", "status_indicator": "connection-chip"},
		"schedule": {"icon": "clock", "status_indicator": "schedule-chip"},
		"result": {"icon": "table", "status_indicator": "result-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": ANALYTICS_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"query_executed",
		"query_saved",
		"query_scheduled",
		"cube_created",
		"cube_refreshed",
		"cube_archived",
		"metric_defined",
		"metric_updated",
		"datasource_registered",
		"datasource_tested",
		"result_exported",
	],
	"guardrails": [
		"cross_tenant_query_denied",
		"unapproved_public_access_denied",
		"query_timeout_enforced",
		"max_row_limit_enforced",
		"credentials_vault_required",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "write_policy_required", "required_action": "attach_policy"}},
	{"name": "query_type_supported", "condition": {"operation": "execute_query", "query_type_supported": False}, "effect": {"decision": "deny", "reason": "query_type_not_supported", "required_action": "select_supported_query_type"}},
	{"name": "query_owner_required", "condition": {"operation": "save_query", "owner_present": False}, "effect": {"decision": "deny", "reason": "query_owner_required", "required_action": "attach_owner"}},
	{"name": "cross_tenant_query_denied", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_query_not_permitted", "required_action": "restrict_to_tenant"}},
	{"name": "query_timeout_enforced", "condition": {"query_timeout_exceeded": True}, "effect": {"decision": "deny", "reason": "query_timeout_exceeded", "required_action": "optimise_or_paginate_query"}},
	{"name": "max_rows_enforced", "condition": {"rows_exceed_limit": True}, "effect": {"decision": "deny", "reason": "row_limit_exceeded", "required_action": "add_filters_or_pagination"}},
	{"name": "cube_dimension_supported", "condition": {"operation": "create_cube", "dimension_supported": False}, "effect": {"decision": "deny", "reason": "dimension_type_not_supported", "required_action": "select_supported_dimension"}},
	{"name": "cube_owner_required", "condition": {"operation": "create_cube", "owner_present": False}, "effect": {"decision": "deny", "reason": "cube_owner_required", "required_action": "attach_cube_owner"}},
	{"name": "cube_refresh_requires_active", "condition": {"operation": "refresh_cube", "cube_state": "archived"}, "effect": {"decision": "deny", "reason": "archived_cube_cannot_be_refreshed", "required_action": "restore_cube_first"}},
	{"name": "metric_type_supported", "condition": {"operation": "define_metric", "metric_type_supported": False}, "effect": {"decision": "deny", "reason": "metric_type_not_supported", "required_action": "select_supported_metric_type"}},
	{"name": "metric_formula_required", "condition": {"operation": "define_metric", "formula_present": False}, "effect": {"decision": "deny", "reason": "metric_formula_required", "required_action": "attach_metric_formula"}},
	{"name": "metric_owner_required", "condition": {"operation": "define_metric", "owner_present": False}, "effect": {"decision": "deny", "reason": "metric_owner_required", "required_action": "attach_metric_owner"}},
	{"name": "public_access_requires_approval", "condition": {"access_level": "public", "access_approved": False}, "effect": {"decision": "deny", "reason": "public_access_requires_approval", "required_action": "submit_for_approval"}},
	{"name": "datasource_credentials_vault_required", "condition": {"operation": "register_datasource", "credentials_in_vault": False}, "effect": {"decision": "deny", "reason": "credentials_must_be_in_vault", "required_action": "store_credentials_in_vault"}},
	{"name": "datasource_type_supported", "condition": {"operation": "register_datasource", "datasource_type_supported": False}, "effect": {"decision": "deny", "reason": "datasource_type_not_supported", "required_action": "select_supported_datasource_type"}},
	{"name": "datasource_connection_test_required", "condition": {"operation": "register_datasource", "connection_tested": False}, "effect": {"decision": "deny", "reason": "connection_test_required", "required_action": "test_connection_before_saving"}},
	{"name": "scheduled_query_requires_owner", "condition": {"operation": "schedule_query", "owner_present": False}, "effect": {"decision": "deny", "reason": "scheduled_query_must_have_owner", "required_action": "attach_owner_before_scheduling"}},
	{"name": "export_access_check", "condition": {"operation": "export_result", "export_permitted": False}, "effect": {"decision": "deny", "reason": "export_not_permitted_for_access_level", "required_action": "request_export_permission"}},
	{"name": "audit_all_queries", "condition": {"operation": "execute_query", "audit_enabled": True}, "effect": {"decision": "allow", "reason": "query_execution_audited", "required_action": "emit_query_executed_event"}},
	{"name": "delete_shared_query_requires_owner", "condition": {"operation": "delete_query", "requester_is_owner": False, "access_level": "team"}, "effect": {"decision": "deny", "reason": "only_owner_can_delete_shared_query", "required_action": "transfer_ownership_first"}},
	{"name": "stale_cube_read_allowed_with_warning", "condition": {"operation": "query_cube", "cube_state": "stale"}, "effect": {"decision": "allow", "reason": "stale_cube_read_allowed_with_staleness_warning", "required_action": "attach_staleness_metadata"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	config["configuration"] = config
	return {
		"capability": CAPABILITY_ID,
		"display_name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"configuration": config,
		"configuration_schema": {
			"required": ["tenant_id", "ui", "theme"],
			"properties": {
				"tenant_id": {"type": "string"},
				"ui": {"type": "object"},
				"theme": {"type": "object"},
			},
		},
		"rule_engine": {
			"type": "deterministic",
			"default_decision": "allow",
			"rules": RULES,
		},
		"ui": {
			"shell": "apg_python",
			"requires_theme": True,
			"template_roots": ["bia/anl/templates"],
			"routes": UI_ROUTES,
		},
		"theme": THEME,
		"streaming": STREAMING,
		"provides": PROVIDES,
		"requires": REQUIRES,
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	for rule in RULES:
		cond = rule["condition"]
		match = all(context.get(k) == v for k, v in cond.items())
		if match:
			return {
				"matched_rule": rule["name"],
				"decision": rule["effect"]["decision"],
				"reason": rule["effect"]["reason"],
				"required_action": rule["effect"]["required_action"],
			}
	return {"matched_rule": None, "decision": "allow", "reason": "no_rule_matched", "required_action": None}
