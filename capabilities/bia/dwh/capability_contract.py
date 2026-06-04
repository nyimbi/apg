"""Executable capability contract for APG Data Warehouse (bia_dwh)."""

from __future__ import annotations
from copy import deepcopy
from typing import Any

CAPABILITY_ID = "bia_dwh"
CAPABILITY_NAME = "Data Warehouse"
CAPABILITY_VERSION = "1.0.0"
DWH_EVENT_STREAM = "apg.bia.dwh.lifecycle"

SUPPORTED_SCHEMA_TYPES = ["star", "snowflake", "galaxy", "flat", "data_vault"]
SUPPORTED_TABLE_TYPES = ["fact", "dimension", "bridge", "aggregate", "staging", "raw", "quarantine"]
SUPPORTED_ETL_STATES = ["pending", "running", "completed", "failed", "cancelled", "retrying"]
SUPPORTED_PARTITION_STRATEGIES = ["range", "list", "hash", "composite", "none"]
SUPPORTED_DATA_QUALITY_RULES = ["not_null", "unique", "referential_integrity", "range_check", "pattern_match", "freshness", "completeness", "consistency"]
SUPPORTED_LOAD_STRATEGIES = ["full_refresh", "incremental", "scd_type1", "scd_type2", "scd_type3", "merge", "append"]
SUPPORTED_COMPRESSION_TYPES = ["none", "zstd", "lz4", "snappy", "gzip"]
SUPPORTED_STORAGE_TIERS = ["hot", "warm", "cold", "archive"]
SUPPORTED_GRAIN_TYPES = ["transaction", "periodic_snapshot", "accumulating_snapshot"]
SUPPORTED_REVIEW_STATUSES = ["pending", "approved", "rejected", "needs_changes"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["schema_designer", "etl_author", "quality_steward", "lineage_tracer", "partition_manager"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"schemas": {"supported_types": SUPPORTED_SCHEMA_TYPES, "require_owner": True, "require_grain": True},
	"tables": {"supported_types": SUPPORTED_TABLE_TYPES, "supported_partition_strategies": SUPPORTED_PARTITION_STRATEGIES, "supported_compression": SUPPORTED_COMPRESSION_TYPES, "require_owner": True},
	"etl": {"supported_states": SUPPORTED_ETL_STATES, "supported_load_strategies": SUPPORTED_LOAD_STRATEGIES, "require_source": True, "require_target": True, "max_parallel_jobs": 10},
	"quality": {"supported_rules": SUPPORTED_DATA_QUALITY_RULES, "quarantine_on_failure": True, "alert_on_failure": True},
	"storage": {"supported_tiers": SUPPORTED_STORAGE_TIERS, "default_tier": "hot", "auto_tiering_enabled": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "cross_tenant_access_denied": True, "lineage_tracking_required": True},
	"observability": {"event_stream": DWH_EVENT_STREAM, "stream_processor": "bytewax"},
	"theme": {"default_theme": "bia_dwh_warehouse", "allow_tenant_overrides": True},
}

PROVIDES = ["dimensional_schema_management", "star_snowflake_schema_design", "etl_orchestration", "data_partitioning", "data_quality_enforcement", "lineage_tracking", "storage_tier_management", "warehouse_catalogue"]

REQUIRES = ["auth", "audl", "mten", "conf", "schd", "mqeb", "moni", "comp"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/bia/dwh/dashboard", "component": "WarehouseDashboard", "permission": "bia_dwh:view", "nav_group": "Overview"},
	{"name": "schemas", "path": "/bia/dwh/schemas", "component": "SchemaExplorer", "permission": "bia_dwh:schemas", "nav_group": "Schema"},
	{"name": "schema_detail", "path": "/bia/dwh/schemas/<id>", "component": "SchemaDetail", "permission": "bia_dwh:schemas", "nav_group": "Schema"},
	{"name": "tables", "path": "/bia/dwh/tables", "component": "TableCatalogue", "permission": "bia_dwh:tables", "nav_group": "Tables"},
	{"name": "table_detail", "path": "/bia/dwh/tables/<id>", "component": "TableDetail", "permission": "bia_dwh:tables", "nav_group": "Tables"},
	{"name": "etl_jobs", "path": "/bia/dwh/etl", "component": "ETLJobManager", "permission": "bia_dwh:etl", "nav_group": "ETL"},
	{"name": "etl_detail", "path": "/bia/dwh/etl/<id>", "component": "ETLJobDetail", "permission": "bia_dwh:etl", "nav_group": "ETL"},
	{"name": "quality", "path": "/bia/dwh/quality", "component": "DataQualityConsole", "permission": "bia_dwh:quality", "nav_group": "Quality"},
	{"name": "lineage", "path": "/bia/dwh/lineage", "component": "LineageViewer", "permission": "bia_dwh:lineage", "nav_group": "Governance"},
	{"name": "partitions", "path": "/bia/dwh/partitions", "component": "PartitionManager", "permission": "bia_dwh:admin", "nav_group": "Storage"},
	{"name": "storage_tiers", "path": "/bia/dwh/storage", "component": "StorageTierManager", "permission": "bia_dwh:admin", "nav_group": "Storage"},
	{"name": "audit_log", "path": "/bia/dwh/audit", "component": "WarehouseAuditLog", "permission": "bia_dwh:admin", "nav_group": "Governance"},
	{"name": "settings", "path": "/bia/dwh/settings", "component": "WarehouseSettings", "permission": "bia_dwh:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "bia_dwh_warehouse",
	"tokens": {"color.primary": "#0F4C81", "color.accent": "#00897B", "color.success": "#2E7D32", "color.warning": "#E65100", "color.danger": "#B71C1C", "surface.canvas": "#F1F8FE", "surface.panel": "#FFFFFF", "text.primary": "#0A1929", "text.secondary": "#455A64", "border.radius": "4px", "density": "compact"},
	"components": {
		"schema": {"icon": "git-branch", "status_indicator": "schema-type-chip"},
		"table": {"icon": "table-2", "status_indicator": "table-type-chip"},
		"etl_job": {"icon": "workflow", "status_indicator": "etl-state-chip"},
		"quality_rule": {"icon": "shield-check", "status_indicator": "quality-chip"},
		"lineage": {"icon": "link", "status_indicator": "lineage-chip"},
	},
}

STREAMING = {
	"processor": "bytewax", "stream": DWH_EVENT_STREAM, "key": "tenant_id",
	"events": ["schema_created", "schema_updated", "table_registered", "table_updated", "etl_job_started", "etl_job_completed", "etl_job_failed", "quality_rule_violated", "lineage_recorded", "partition_created", "storage_tier_changed"],
	"guardrails": ["cross_tenant_access_denied", "lineage_tracking_required", "quarantine_on_quality_failure", "etl_parallel_limit_enforced"],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "write_policy_required", "required_action": "attach_policy"}},
	{"name": "cross_tenant_access_denied", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_not_permitted", "required_action": "restrict_to_tenant"}},
	{"name": "schema_type_supported", "condition": {"operation": "create_schema", "schema_type_supported": False}, "effect": {"decision": "deny", "reason": "schema_type_not_supported", "required_action": "select_supported_schema_type"}},
	{"name": "schema_owner_required", "condition": {"operation": "create_schema", "owner_present": False}, "effect": {"decision": "deny", "reason": "schema_owner_required", "required_action": "attach_schema_owner"}},
	{"name": "schema_grain_required", "condition": {"operation": "create_schema", "grain_present": False}, "effect": {"decision": "deny", "reason": "schema_grain_required", "required_action": "define_schema_grain"}},
	{"name": "table_type_supported", "condition": {"operation": "register_table", "table_type_supported": False}, "effect": {"decision": "deny", "reason": "table_type_not_supported", "required_action": "select_supported_table_type"}},
	{"name": "table_owner_required", "condition": {"operation": "register_table", "owner_present": False}, "effect": {"decision": "deny", "reason": "table_owner_required", "required_action": "attach_table_owner"}},
	{"name": "partition_strategy_supported", "condition": {"operation": "set_partition", "strategy_supported": False}, "effect": {"decision": "deny", "reason": "partition_strategy_not_supported", "required_action": "select_supported_partition_strategy"}},
	{"name": "load_strategy_supported", "condition": {"operation": "create_etl_job", "load_strategy_supported": False}, "effect": {"decision": "deny", "reason": "load_strategy_not_supported", "required_action": "select_supported_load_strategy"}},
	{"name": "etl_source_required", "condition": {"operation": "create_etl_job", "source_present": False}, "effect": {"decision": "deny", "reason": "etl_source_required", "required_action": "attach_etl_source"}},
	{"name": "etl_target_required", "condition": {"operation": "create_etl_job", "target_present": False}, "effect": {"decision": "deny", "reason": "etl_target_required", "required_action": "attach_etl_target"}},
	{"name": "etl_parallel_limit_enforced", "condition": {"operation": "start_etl_job", "parallel_limit_exceeded": True}, "effect": {"decision": "deny", "reason": "etl_parallel_limit_exceeded", "required_action": "wait_for_running_jobs_to_complete"}},
	{"name": "quality_rule_type_supported", "condition": {"operation": "add_quality_rule", "rule_type_supported": False}, "effect": {"decision": "deny", "reason": "quality_rule_type_not_supported", "required_action": "select_supported_quality_rule_type"}},
	{"name": "quarantine_on_quality_failure", "condition": {"operation": "etl_quality_check", "quality_check_failed": True}, "effect": {"decision": "deny", "reason": "data_quarantined_due_to_quality_failure", "required_action": "review_quarantined_data"}},
	{"name": "lineage_tracking_required", "condition": {"operation": "register_table", "lineage_tracked": False}, "effect": {"decision": "deny", "reason": "lineage_tracking_required_for_all_tables", "required_action": "attach_lineage_reference"}},
	{"name": "storage_tier_supported", "condition": {"operation": "set_storage_tier", "tier_supported": False}, "effect": {"decision": "deny", "reason": "storage_tier_not_supported", "required_action": "select_supported_storage_tier"}},
	{"name": "drop_table_requires_no_dependents", "condition": {"operation": "drop_table", "has_dependents": True}, "effect": {"decision": "deny", "reason": "cannot_drop_table_with_dependents", "required_action": "remove_dependent_tables_first"}},
	{"name": "scd2_requires_surrogate_key", "condition": {"operation": "create_etl_job", "load_strategy": "scd_type2", "surrogate_key_present": False}, "effect": {"decision": "deny", "reason": "scd_type2_requires_surrogate_key", "required_action": "add_surrogate_key_to_dimension"}},
	{"name": "audit_all_etl_runs", "condition": {"operation": "start_etl_job", "audit_enabled": True}, "effect": {"decision": "allow", "reason": "etl_execution_audited", "required_action": "emit_etl_started_event"}},
	{"name": "cold_tier_requires_archival_policy", "condition": {"operation": "set_storage_tier", "tier": "archive", "archival_policy_present": False}, "effect": {"decision": "deny", "reason": "archive_tier_requires_archival_policy", "required_action": "attach_archival_policy"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION,
		"configuration": config,
		"configuration_schema": {"required": ["tenant_id", "ui", "theme"], "properties": {"tenant_id": {"type": "string"}, "ui": {"type": "object"}, "theme": {"type": "object"}}},
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": RULES},
		"ui": {"shell": "apg_python", "requires_theme": True, "template_roots": ["bia/dwh/templates"], "routes": UI_ROUTES},
		"theme": THEME, "streaming": STREAMING, "provides": PROVIDES, "requires": REQUIRES,
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	for rule in RULES:
		if all(context.get(k) == v for k, v in rule["condition"].items()):
			return {"matched_rule": rule["name"], "decision": rule["effect"]["decision"], "reason": rule["effect"]["reason"], "required_action": rule["effect"]["required_action"]}
	return {"matched_rule": None, "decision": "allow", "reason": "no_rule_matched", "required_action": None}
