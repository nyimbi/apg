"""Executable capability contract for APG Data Virtualization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SUPPORTED_DVRL_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_DVRL_AGENT_ROLES = [
	"source_reviewer",
	"schema_reviewer",
	"virtual_table_reviewer",
	"query_policy_reviewer",
	"cache_policy_reviewer",
	"lineage_reviewer",
	"federation_reviewer",
	"publish_gate_reviewer",
]
PRIVILEGED_DVRL_AGENT_ROLES = [
	"source_reviewer",
	"virtual_table_reviewer",
	"query_policy_reviewer",
	"cache_policy_reviewer",
	"publish_gate_reviewer",
]


@dataclass(frozen=True)
class CapabilityConfiguration:
	"""Tenant-scoped DVRL configuration defaults and schema."""

	defaults: dict[str, Any] = field(default_factory=lambda: {
		"tenant_id": "default",
		"sources": {
			"source_registration_required": True,
			"connection_encryption_required": True,
			"credential_vault": "keym",
			"max_sources_per_tenant": 100,
			"owner_required": True,
			"approval_required": True,
			"supported_source_types": ["database", "warehouse", "lakehouse", "api", "file", "stream", "singer_tap"],
		},
		"schemas": {
			"schema_discovery_required": True,
			"schema_refresh_review_days": 30,
			"virtual_table_owner_required": True,
			"classification_required": True,
		},
		"queries": {
			"federated_query_enabled": True,
			"default_timeout_seconds": 300,
			"max_result_rows": 100000,
			"cost_estimation_required": True,
			"parameterization_required": True,
			"write_queries_allowed": False,
		},
		"cache": {
			"query_cache_enabled": True,
			"sensitive_result_cache_allowed": False,
			"default_ttl_seconds": 900,
			"max_ttl_seconds": 3600,
		},
		"governance": {
			"require_tenant_context": True,
			"rbac_required": True,
			"audit_all_queries": True,
			"lineage_capture_required": True,
			"policy_review_required": True,
		},
		"optimization": {
			"pushdown_enabled": True,
			"join_rewrite_enabled": True,
			"cost_review_threshold": 1000.0,
			"cross_source_join_review_threshold": 3,
		},
		"adapters": {
			"production_runtime": "service.DVRLService",
			"generated_app_runtime": "service.DVRLLifecycleService",
			"connector_registry": "adapter",
			"query_planner": "adapter",
			"execution_engine": "adapter",
			"metadata_catalog": "meta",
			"cache_store": "cach",
			"credential_vault": "keym",
			"audit_sink": "audl",
			"event_stream": "bytewax",
		},
		"agents": {
			"first_class": True,
			"supported_runtimes": SUPPORTED_DVRL_AGENT_RUNTIMES,
			"supported_roles": SUPPORTED_DVRL_AGENT_ROLES,
			"privileged_roles": PRIVILEGED_DVRL_AGENT_ROLES,
			"require_scope": True,
			"require_owner": True,
			"require_purpose": True,
			"require_contribution_disclosure": True,
			"human_approval_required_for_privileged_roles": True,
		},
		"streaming": {
			"engine": "bytewax",
			"required_processor": "bytewax",
			"lifecycle_stream": "dvrl.lifecycle",
			"watermark": "event_time",
			"operations": [
				"source_batch",
				"schema_batch",
				"virtual_table_batch",
				"query_batch",
				"cache_batch",
				"policy_batch",
				"virtualization_agent_batch",
			],
			"topics": [
				"dvrl.sources",
				"dvrl.schemas",
				"dvrl.virtual_tables",
				"dvrl.queries",
				"dvrl.cache",
				"dvrl.policies",
				"dvrl.agents",
			],
		},
		"ui": {
			"enable_query_workbench": True,
			"enable_source_manager": True,
			"enable_schema_browser": True,
			"enable_federation_map": True,
			"enable_cache_console": True,
			"enable_policy_review": True,
			"enable_adapter_health": True,
			"enable_audit_timeline": True,
			"enable_virtualization_agent_roster": True,
			"enable_lifecycle_batch_monitor": True,
		},
		"theme": {
			"default_theme": "dvrl_federation_console",
			"allow_tenant_overrides": True,
		},
	})
	schema: dict[str, Any] = field(default_factory=lambda: {
		"type": "object",
		"required": [
			"tenant_id",
			"sources",
			"schemas",
			"queries",
			"cache",
			"governance",
			"optimization",
			"adapters",
			"agents",
			"streaming",
			"ui",
			"theme",
		],
		"properties": {
			"tenant_id": {"type": "string", "minLength": 1},
			"sources": {"type": "object"},
			"schemas": {"type": "object"},
			"queries": {"type": "object"},
			"cache": {"type": "object"},
			"governance": {"type": "object"},
			"optimization": {"type": "object"},
			"adapters": {"type": "object"},
			"agents": {"type": "object"},
			"streaming": {"type": "object"},
			"ui": {"type": "object"},
			"theme": {"type": "object"},
		},
	})

	def for_tenant(self, tenant_id: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
		assert isinstance(tenant_id, str) and tenant_id, "tenant_id is required"
		merged = _deep_copy(self.defaults)
		merged["tenant_id"] = tenant_id
		if overrides:
			_deep_merge(merged, overrides)
		return merged


@dataclass(frozen=True)
class CapabilityRule:
	name: str
	description: str
	condition: dict[str, Any]
	effect: dict[str, Any]


class CapabilityRuleEngine:
	"""Deterministic DVRL rule engine for virtualization control decisions."""

	def __init__(self, rules: list[CapabilityRule] | None = None):
		self.rules = rules or default_rules()

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
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
		return {"decision": decision, "matched_rules": matched, "actions": actions, "context": context}


@dataclass(frozen=True)
class CapabilityUIRoute:
	name: str
	path: str
	component: str
	permission: str
	nav_group: str


@dataclass(frozen=True)
class CapabilityTheme:
	name: str = "dvrl_federation_console"
	tokens: dict[str, str] = field(default_factory=lambda: {
		"color.primary": "#274060",
		"color.accent": "#4ECDC4",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F6F8FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact",
	})
	components: dict[str, dict[str, str]] = field(default_factory=lambda: {
		"virtual_source_card": {"icon": "database-zap", "status_indicator": "connectivity-pill", "risk_style": "policy-band"},
		"federation_map": {"visual": "source-topology", "edge_style": "join-path"},
		"query_plan_viewer": {"visual": "execution-tree", "highlight": "cost-chip"},
		"cache_result_panel": {"visual": "cache-hit-stack", "status_style": "ttl-pill"},
		"schema_browser": {"visual": "table-tree", "status_indicator": "classification-chip"},
		"policy_review_queue": {"visual": "rule-decision-list", "highlight": "review-chip"},
		"adapter_health_panel": {"visual": "adapter-grid", "status_indicator": "health-pill"},
		"audit_timeline": {"visual": "event-timeline", "status_style": "decision-pill"},
		"virtualization_agent_roster": {"visual": "agent-role-grid", "status_indicator": "approval-pill"},
		"bytewax_lifecycle_panel": {"visual": "stream-batch-ledger", "status_indicator": "processor-pill"},
	})


def default_rules() -> list[CapabilityRule]:
	return [
		CapabilityRule("tenant_context_required", "All virtualization operations require tenant context.", {"tenant_context_present": False}, {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}),
		CapabilityRule("source_registration_requires_owner", "Virtual sources require an accountable owner.", {"operation": "register_source", "source_owner_assigned": False}, {"decision": "deny", "reason": "source_owner_required", "required_action": "assign_source_owner"}),
		CapabilityRule("source_type_must_be_supported", "Virtual source type must be configured for the tenant.", {"operation": "register_source", "unsupported_source_type": True}, {"decision": "deny", "reason": "unsupported_source_type", "required_action": "choose_supported_source_type"}),
		CapabilityRule("source_registration_requires_credentials", "Virtual sources require vaulted credentials.", {"operation": "register_source", "credentials_vaulted": False}, {"decision": "deny", "reason": "vaulted_credentials_required", "required_action": "store_credentials_in_keym"}),
		CapabilityRule("source_connection_requires_encryption", "Virtual source connections require encryption.", {"operation": "register_source", "connection_encrypted": False}, {"decision": "deny", "reason": "connection_encryption_required", "required_action": "enable_connection_encryption"}),
		CapabilityRule("source_activation_requires_approval", "Virtual sources require approval before activation.", {"operation": "activate_source", "source_approval_recorded": False}, {"decision": "require_review", "reason": "source_approval_required", "required_action": "record_source_approval"}),
		CapabilityRule("schema_refresh_requires_review", "Stale virtual schemas require refresh review.", {"operation": "refresh_schema", "schema_age_days_gt": 30, "schema_review_recorded": False}, {"decision": "require_review", "reason": "schema_refresh_review_required", "required_action": "record_schema_refresh_review"}),
		CapabilityRule("virtual_table_requires_owner", "Virtual tables require an accountable owner.", {"operation": "publish_virtual_table", "virtual_table_owner_assigned": False}, {"decision": "deny", "reason": "virtual_table_owner_required", "required_action": "assign_virtual_table_owner"}),
		CapabilityRule("virtual_table_requires_classification", "Virtual tables require data classification before publication.", {"operation": "publish_virtual_table", "classification_complete": False}, {"decision": "deny", "reason": "classification_required", "required_action": "classify_virtual_table"}),
		CapabilityRule("query_requires_parameterization", "Federated queries require parameterization evidence.", {"operation": "execute_query", "parameterized": False}, {"decision": "deny", "reason": "parameterization_required", "required_action": "parameterize_query"}),
		CapabilityRule("write_query_blocked", "DVRL generated-app queries are read-only by default.", {"operation": "execute_query", "write_query": True}, {"decision": "deny", "reason": "write_query_blocked", "required_action": "route_to_write_capability"}),
		CapabilityRule("restricted_query_requires_rbac", "Restricted data queries require RBAC authorization.", {"operation": "execute_query", "data_classification": "restricted", "rbac_authorized": False}, {"decision": "deny", "reason": "rbac_authorization_required", "required_action": "authorize_query_access"}),
		CapabilityRule("sensitive_results_block_cache", "Sensitive query results cannot be cached by default.", {"operation": "execute_query", "result_contains_sensitive_data": True, "cache_requested": True}, {"decision": "deny", "reason": "sensitive_result_cache_blocked", "required_action": "disable_result_cache"}),
		CapabilityRule("query_requires_lineage_capture", "Federated queries require lineage capture.", {"operation": "execute_query", "lineage_capture_enabled": False}, {"decision": "deny", "reason": "lineage_capture_required", "required_action": "enable_lineage_capture"}),
		CapabilityRule("high_cost_query_requires_review", "High cost federated queries require review.", {"operation": "execute_query", "estimated_query_cost_gt": 1000.0, "cost_review_recorded": False}, {"decision": "require_review", "reason": "query_cost_review_required", "required_action": "record_query_cost_review"}),
		CapabilityRule("cross_source_join_requires_review", "Cross-source joins above the tenant threshold require review.", {"operation": "execute_query", "join_source_count_gt": 3, "join_review_recorded": False}, {"decision": "require_review", "reason": "cross_source_join_review_required", "required_action": "record_join_review"}),
		CapabilityRule("query_result_limit_enforced", "Federated query result limits cannot exceed tenant configuration.", {"operation": "execute_query", "requested_rows_gt": 100000}, {"decision": "deny", "reason": "query_result_limit_exceeded", "required_action": "reduce_result_limit"}),
		CapabilityRule("cache_ttl_requires_limit", "Query cache TTL cannot exceed tenant configuration.", {"operation": "cache_result", "cache_ttl_seconds_gt": 3600}, {"decision": "deny", "reason": "cache_ttl_limit_exceeded", "required_action": "reduce_cache_ttl"}),
		CapabilityRule("policy_change_requires_review", "Virtualization policy changes require review.", {"operation": "change_policy", "policy_review_recorded": False}, {"decision": "require_review", "reason": "policy_review_required", "required_action": "record_policy_review"}),
		CapabilityRule("source_retirement_requires_impact_review", "Retiring a virtual source requires impact review.", {"operation": "retire_source", "impact_review_recorded": False}, {"decision": "deny", "reason": "impact_review_required", "required_action": "record_source_impact_review"}),
		CapabilityRule("virtualization_agent_runtime_supported", "Virtualization agents must use an approved runtime.", {"operation": "register_virtualization_agent", "unsupported_agent_runtime": True}, {"decision": "deny", "reason": "unsupported_agent_runtime", "required_action": "choose_supported_agent_runtime"}),
		CapabilityRule("virtualization_agent_role_supported", "Virtualization agents must use an approved DVRL role.", {"operation": "register_virtualization_agent", "unsupported_agent_role": True}, {"decision": "deny", "reason": "unsupported_agent_role", "required_action": "choose_supported_agent_role"}),
		CapabilityRule("virtualization_agent_requires_scope", "Virtualization agents require bounded source, schema, query, or policy scope.", {"operation": "register_virtualization_agent", "agent_scope_present": False}, {"decision": "deny", "reason": "agent_scope_required", "required_action": "define_agent_scope"}),
		CapabilityRule("virtualization_agent_requires_owner", "Virtualization agents require an accountable owner.", {"operation": "register_virtualization_agent", "agent_owner_present": False}, {"decision": "deny", "reason": "agent_owner_required", "required_action": "assign_agent_owner"}),
		CapabilityRule("virtualization_agent_requires_purpose", "Virtualization agents require a declared purpose.", {"operation": "register_virtualization_agent", "agent_purpose_present": False}, {"decision": "deny", "reason": "agent_purpose_required", "required_action": "declare_agent_purpose"}),
		CapabilityRule("virtualization_agent_requires_contribution_disclosure", "Virtualization agents must disclose machine contribution before participating in DVRL decisions.", {"operation": "register_virtualization_agent", "agent_contribution_disclosed": False}, {"decision": "deny", "reason": "agent_contribution_disclosure_required", "required_action": "disclose_machine_contribution"}),
		CapabilityRule("virtualization_agent_privileged_role_requires_human_approval", "Privileged virtualization-agent roles require human approval.", {"operation": "register_virtualization_agent", "privileged_agent_role": True, "human_approval_required": False}, {"decision": "require_review", "reason": "privileged_agent_human_approval_required", "required_action": "record_human_approval_requirement"}),
		CapabilityRule("bytewax_dvrl_stream_required", "DVRL lifecycle batches must be routed through Bytewax.", {"operation": "validate_dvrl_lifecycle_batch", "event_stream_ne": "bytewax"}, {"decision": "deny", "reason": "bytewax_required", "required_action": "route_batch_to_bytewax"}),
	]


def ui_manifest() -> dict[str, Any]:
	routes = [
		CapabilityUIRoute("dashboard", "/dvrl/dashboard", "DVRLDashboard", "dvrl:view", "Overview"),
		CapabilityUIRoute("query", "/dvrl/query", "QueryWorkbench", "dvrl:query", "Query"),
		CapabilityUIRoute("sources", "/dvrl/sources", "VirtualSourceManager", "dvrl:manage_sources", "Sources"),
		CapabilityUIRoute("schemas", "/dvrl/schemas", "SchemaBrowser", "dvrl:view", "Sources"),
		CapabilityUIRoute("virtual_tables", "/dvrl/virtual-tables", "VirtualTableCatalog", "dvrl:manage_sources", "Sources"),
		CapabilityUIRoute("federation", "/dvrl/federation", "FederationMap", "dvrl:view_lineage", "Architecture"),
		CapabilityUIRoute("policies", "/dvrl/policies", "VirtualizationPolicies", "dvrl:manage_policies", "Governance"),
		CapabilityUIRoute("cache", "/dvrl/cache", "CacheConsole", "dvrl:manage_policies", "Operations"),
		CapabilityUIRoute("metrics", "/dvrl/metrics", "DVRLMetrics", "dvrl:view_metrics", "Operations"),
		CapabilityUIRoute("adapters", "/dvrl/adapters", "DVRLAdapterHealth", "dvrl:admin", "Administration"),
		CapabilityUIRoute("agents", "/dvrl/agents", "DVRLVirtualizationAgentRoster", "dvrl:admin", "Governance"),
		CapabilityUIRoute("lifecycle", "/dvrl/lifecycle", "DVRLLifecycleBatchMonitor", "dvrl:view_metrics", "Operations"),
		CapabilityUIRoute("audit", "/dvrl/audit", "DVRLAuditTimeline", "dvrl:view_metrics", "Governance"),
		CapabilityUIRoute("settings", "/dvrl/settings", "DVRLSettings", "dvrl:admin", "Administration"),
	]
	return {"shell": "apg_python", "view_module": "view_models.py", "api_prefix": "/dvrl/api/v1", "routes": [route.__dict__ for route in routes], "template_roots": ["templates/", "static/"], "requires_theme": True}


def agent_manifest() -> dict[str, Any]:
	config = CapabilityConfiguration().defaults["agents"]
	return {
		"first_class": config["first_class"],
		"supported_runtimes": list(config["supported_runtimes"]),
		"supported_roles": list(config["supported_roles"]),
		"privileged_roles": list(config["privileged_roles"]),
		"requires_scope": config["require_scope"],
		"requires_owner": config["require_owner"],
		"requires_purpose": config["require_purpose"],
		"requires_contribution_disclosure": config["require_contribution_disclosure"],
	}


def streaming_manifest() -> dict[str, Any]:
	config = CapabilityConfiguration().defaults["streaming"]
	return {
		"engine": config["engine"],
		"required_processor": config["required_processor"],
		"lifecycle_stream": config["lifecycle_stream"],
		"watermark": config["watermark"],
		"operations": list(config["operations"]),
		"topics": list(config["topics"]),
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = CapabilityConfiguration()
	theme = CapabilityTheme()
	return {
		"capability": "dvrl",
		"display_name": "Data Virtualization",
		"provides": ["data_virtualization", "federated_query_lifecycle", "virtualization_agent_composition"],
		"requires": ["mdm", "etlp", "meta"],
		"configuration": config.for_tenant(tenant_id, overrides),
		"configuration_schema": config.schema,
		"rule_engine": {"type": "deterministic", "rules": [rule.__dict__ for rule in default_rules()]},
		"ui": ui_manifest(),
		"theme": {"name": theme.name, "tokens": theme.tokens, "components": theme.components},
		"agents": agent_manifest(),
		"streaming": streaming_manifest(),
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	return CapabilityRuleEngine().evaluate(context)


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_gt"):
			if not context.get(key[:-3], 0) > expected:
				return False
		elif key.endswith("_lt"):
			if not context.get(key[:-3], 0) < expected:
				return False
		elif key.endswith("_ne"):
			if context.get(key[:-3]) == expected:
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
