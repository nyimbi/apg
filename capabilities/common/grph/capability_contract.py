"""Executable capability contract for APG Graph Data Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


SUPPORTED_GRPH_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_GRPH_AGENT_ROLES = [
	"schema_reviewer",
	"node_quality_reviewer",
	"edge_policy_reviewer",
	"traversal_reviewer",
	"lineage_reviewer",
	"impact_reviewer",
	"quality_reviewer",
	"lifecycle_batch_reviewer",
	"graph_steward",
]
PRIVILEGED_GRPH_AGENT_ROLES = [
	"edge_policy_reviewer",
	"traversal_reviewer",
	"lineage_reviewer",
	"impact_reviewer",
	"lifecycle_batch_reviewer",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"schemas": {
		"id_required": True,
		"name_required": True,
		"kind_required": True,
		"allowed_graph_kinds": ["property", "lineage", "knowledge", "dependency"],
		"node_types_required": True,
		"edge_types_required": True,
		"retire_requires_review": True,
	},
	"nodes": {
		"id_required": True,
		"type_required": True,
		"owner_required": True,
		"max_labels": 20,
		"allowed_label_prefixes": ["apg", "data", "asset", "entity", "process", "service", "customer"],
		"property_validation_enabled": True,
	},
	"edges": {
		"id_required": True,
		"type_required": True,
		"owner_required": True,
		"source_required": True,
		"target_required": True,
		"allowed_classifications": ["public", "internal", "confidential", "restricted"],
		"restricted_review_required": True,
		"self_edge_review_required": True,
	},
	"traversal": {
		"allowed_query_types": ["traverse", "lineage", "impact", "neighborhood"],
		"max_depth": 8,
		"max_result_window": 1000,
		"rbac_filter_required": True,
	},
	"lineage": {
		"enabled": True,
		"source_asset_required": True,
		"dependency_edges_enabled": True,
	},
	"quality": {
		"enabled": True,
		"orphan_review_threshold": 50,
		"restricted_edge_review_threshold": 100,
		"missing_owner_threshold": 1,
	},
	"security": {
		"cross_tenant_edges_allowed": False,
		"restricted_relationship_filter_required": True,
		"rbac_filter_required": True,
	},
	"agents": {
		"first_class": True,
		"supported_runtimes": SUPPORTED_GRPH_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_GRPH_AGENT_ROLES,
		"privileged_roles": PRIVILEGED_GRPH_AGENT_ROLES,
		"require_owner": True,
		"require_purpose": True,
		"require_scope": True,
		"require_contribution_disclosure": True,
		"require_human_approval_for_privileged_roles": True,
		"adapter_contract": "aicr_provider_neutral_graph_agent_adapter",
	},
	"streaming": {
		"engine": "bytewax",
		"lifecycle_stream": "grph.lifecycle",
		"watermark": "event_time",
		"required_processor": "bytewax",
		"required_operations": [
			"schema_batch",
			"node_batch",
			"edge_batch",
			"traversal_batch",
			"lineage_batch",
			"impact_batch",
			"quality_batch",
			"graph_agent_batch",
		],
		"topics": [
			"grph.schemas",
			"grph.nodes",
			"grph.edges",
			"grph.traversals",
			"grph.lineage",
			"grph.impact",
			"grph.quality",
			"grph.agents",
		],
		"broker_core_dependency_allowed": False,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_mutations": True,
		"audit_queries": True,
		"review_required_for_restricted_relationships": True,
	},
	"observability": {
		"metrics_required": True,
		"trace_required": True,
		"audit_required": True,
		"quality_metrics_required": True,
		"event_stream": "bytewax",
	},
	"adapters": {
		"generated_app_runtime": "service.GrphService",
		"helper_runtime": "graph_runtime.py",
		"production_runtime": "service.GrphService",
		"http_api": "api.py",
		"event_stream": "bytewax",
		"master_data": "mdm",
		"metadata": "meta",
		"data_pipeline": "etlp",
		"search": "srch",
		"knowledge_graph": "kngr",
		"ai_core": "aicr",
		"auth_provider": "auth",
		"audit_sink": "audl",
		"cache": "cach",
		"metrics_sink": "moni",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_explorer": True,
		"enable_schema_manager": True,
		"enable_node_manager": True,
		"enable_edge_manager": True,
		"enable_traversal": True,
		"enable_lineage": True,
		"enable_impact": True,
		"enable_quality": True,
		"enable_governance": True,
		"enable_graph_agent_roster": True,
		"enable_lifecycle_batch_monitor": True,
		"enable_audit": True,
		"enable_settings": True,
	},
	"theme": {"default_theme": "grph_relationship_console", "allow_tenant_overrides": True},
}


CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"schemas",
		"nodes",
		"edges",
		"traversal",
		"lineage",
		"quality",
		"security",
		"agents",
		"streaming",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {key: {"type": "object"} for key in [
		"schemas",
		"nodes",
		"edges",
		"traversal",
		"lineage",
		"quality",
		"security",
		"agents",
		"streaming",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]} | {"tenant_id": {"type": "string", "minLength": 1}},
}


RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All graph operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "schema_requires_id", "description": "Graph schemas require a stable identifier.", "condition": {"operation": "write_schema", "schema_id_present": False}, "effect": {"decision": "deny", "reason": "schema_id_required", "required_action": "attach_schema_id"}},
	{"name": "schema_requires_name", "description": "Graph schemas require a display name.", "condition": {"operation": "write_schema", "schema_name_present": False}, "effect": {"decision": "deny", "reason": "schema_name_required", "required_action": "attach_schema_name"}},
	{"name": "schema_requires_kind", "description": "Graph schemas require a graph kind.", "condition": {"operation": "write_schema", "graph_kind_present": False}, "effect": {"decision": "deny", "reason": "graph_kind_required", "required_action": "choose_graph_kind"}},
	{"name": "schema_kind_requires_review", "description": "Unknown graph kinds require review.", "condition": {"operation": "write_schema", "graph_kind_known": False, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "graph_kind_review_required", "required_action": "review_graph_kind"}},
	{"name": "schema_requires_node_types", "description": "Graph schemas require node type declarations.", "condition": {"operation": "write_schema", "node_type_count_lt": 1}, "effect": {"decision": "deny", "reason": "schema_node_types_required", "required_action": "declare_node_types"}},
	{"name": "schema_requires_edge_types", "description": "Graph schemas require edge type declarations.", "condition": {"operation": "write_schema", "edge_type_count_lt": 1}, "effect": {"decision": "deny", "reason": "schema_edge_types_required", "required_action": "declare_edge_types"}},
	{"name": "lineage_schema_requires_source_asset", "description": "Lineage schemas require source asset linkage.", "condition": {"operation": "write_schema", "graph_type": "lineage", "source_asset_present": False}, "effect": {"decision": "deny", "reason": "source_asset_required", "required_action": "attach_source_asset"}},
	{"name": "node_requires_schema", "description": "Node writes require a registered schema.", "condition": {"operation": "write_node", "schema_present": False}, "effect": {"decision": "deny", "reason": "schema_required", "required_action": "select_schema"}},
	{"name": "node_requires_id", "description": "Node writes require a stable identifier.", "condition": {"operation": "write_node", "node_id_present": False}, "effect": {"decision": "deny", "reason": "node_id_required", "required_action": "attach_node_id"}},
	{"name": "node_requires_type", "description": "Node writes require a node type.", "condition": {"operation": "write_node", "node_type_present": False}, "effect": {"decision": "deny", "reason": "node_type_required", "required_action": "attach_node_type"}},
	{"name": "node_write_requires_owner", "description": "Graph node writes require an owner.", "condition": {"operation": "write_node", "owner_assigned": False}, "effect": {"decision": "deny", "reason": "node_owner_required", "required_action": "assign_owner"}},
	{"name": "node_type_requires_schema_membership", "description": "Node types must be declared in the schema.", "condition": {"operation": "write_node", "node_type_allowed": False}, "effect": {"decision": "deny", "reason": "node_type_not_in_schema", "required_action": "update_schema_node_types"}},
	{"name": "node_label_requires_review", "description": "Labels outside the configured prefixes require review.", "condition": {"operation": "write_node", "labels_allowed": False, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "node_label_review_required", "required_action": "review_node_labels"}},
	{"name": "edge_requires_schema", "description": "Edge writes require a registered schema.", "condition": {"operation": "write_edge", "schema_present": False}, "effect": {"decision": "deny", "reason": "schema_required", "required_action": "select_schema"}},
	{"name": "edge_requires_id", "description": "Edge writes require a stable identifier.", "condition": {"operation": "write_edge", "edge_id_present": False}, "effect": {"decision": "deny", "reason": "edge_id_required", "required_action": "attach_edge_id"}},
	{"name": "edge_requires_source", "description": "Edge writes require a source node.", "condition": {"operation": "write_edge", "source_node_present": False}, "effect": {"decision": "deny", "reason": "source_node_required", "required_action": "attach_source_node"}},
	{"name": "edge_requires_target", "description": "Edge writes require a target node.", "condition": {"operation": "write_edge", "target_node_present": False}, "effect": {"decision": "deny", "reason": "target_node_required", "required_action": "attach_target_node"}},
	{"name": "edge_write_requires_type", "description": "Graph edge writes require an edge type.", "condition": {"operation": "write_edge", "edge_type_present": False}, "effect": {"decision": "deny", "reason": "edge_type_required", "required_action": "attach_edge_type"}},
	{"name": "edge_requires_owner", "description": "Graph edge writes require an owner.", "condition": {"operation": "write_edge", "owner_assigned": False}, "effect": {"decision": "deny", "reason": "edge_owner_required", "required_action": "assign_owner"}},
	{"name": "edge_requires_classification", "description": "Graph edges require relationship classification.", "condition": {"operation": "write_edge", "classification_present": False}, "effect": {"decision": "deny", "reason": "edge_classification_required", "required_action": "attach_classification"}},
	{"name": "edge_classification_requires_review", "description": "Unknown relationship classifications require review.", "condition": {"operation": "write_edge", "classification_known": False, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "edge_classification_review_required", "required_action": "review_edge_classification"}},
	{"name": "edge_type_requires_schema_membership", "description": "Edge types must be declared in the schema.", "condition": {"operation": "write_edge", "edge_type_allowed": False}, "effect": {"decision": "deny", "reason": "edge_type_not_in_schema", "required_action": "update_schema_edge_types"}},
	{"name": "cross_tenant_edge_denied", "description": "Edges may not connect nodes across tenants.", "condition": {"operation": "write_edge", "cross_tenant_edge": True}, "effect": {"decision": "deny", "reason": "cross_tenant_edge_denied", "required_action": "use_tenant_local_nodes"}},
	{"name": "restricted_relationship_requires_review", "description": "Restricted relationships require governance review.", "condition": {"operation": "write_edge", "relationship_classification": "restricted", "review_recorded": False}, "effect": {"decision": "require_review", "reason": "restricted_relationship_review_required", "required_action": "record_relationship_review"}},
	{"name": "self_edge_requires_review", "description": "Self-referential edges require review.", "condition": {"operation": "write_edge", "self_edge": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "self_edge_review_required", "required_action": "review_self_edge"}},
	{"name": "traversal_requires_start_node", "description": "Traversals require a tenant-local start node.", "condition": {"operation": "traverse", "start_node_present": False}, "effect": {"decision": "deny", "reason": "start_node_required", "required_action": "attach_start_node"}},
	{"name": "traversal_depth_requires_positive_value", "description": "Traversal depth must be positive.", "condition": {"operation": "traverse", "traversal_depth_lt": 1}, "effect": {"decision": "deny", "reason": "traversal_depth_required", "required_action": "choose_positive_depth"}},
	{"name": "deep_traversal_requires_review", "description": "Deep graph traversals require review.", "condition": {"operation": "traverse", "traversal_depth_gt": 8, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "deep_traversal_review_required", "required_action": "record_traversal_review"}},
	{"name": "restricted_traversal_requires_rbac_filter", "description": "Restricted relationship traversals require RBAC filtering.", "condition": {"operation": "traverse", "restricted_relationships_in_scope": True, "rbac_filter_applied": False}, "effect": {"decision": "deny", "reason": "rbac_filter_required", "required_action": "apply_rbac_filter"}},
	{"name": "lineage_query_requires_source_asset", "description": "Lineage queries require source asset context.", "condition": {"operation": "lineage_query", "source_asset_present": False}, "effect": {"decision": "deny", "reason": "source_asset_required", "required_action": "attach_source_asset"}},
	{"name": "quality_threshold_requires_review", "description": "Poor graph quality above thresholds requires review.", "condition": {"operation": "quality_report", "quality_issue_count_gt": 50, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "quality_review_required", "required_action": "review_graph_quality"}},
	{"name": "batch_mutation_requires_bytewax", "description": "Batch graph mutations must use Bytewax event streams.", "condition": {"operation": "batch_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
	{"name": "schema_retire_requires_review", "description": "Schema retirement requires review evidence.", "condition": {"operation": "retire_schema", "review_recorded": False}, "effect": {"decision": "deny", "reason": "schema_retire_review_required", "required_action": "record_schema_retire_review"}},
	{"name": "graph_state_change_requires_audit", "description": "Graph state changes require audit events.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "audit_event_required", "required_action": "record_audit_event"}},
	{"name": "graph_agent_runtime_supported", "description": "Graph agents must use supported runtimes.", "condition": {"operation": "register_graph_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_graph_agent_runtime", "required_action": "choose_supported_graph_agent_runtime"}},
	{"name": "graph_agent_role_supported", "description": "Graph agents must use supported graph-governance roles.", "condition": {"operation": "register_graph_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_graph_agent_role", "required_action": "choose_supported_graph_agent_role"}},
	{"name": "graph_agent_requires_scope", "description": "Graph agents require an explicit bounded graph scope.", "condition": {"operation": "register_graph_agent", "scope_present": False}, "effect": {"decision": "deny", "reason": "graph_agent_scope_required", "required_action": "declare_graph_agent_scope"}},
	{"name": "graph_agent_requires_owner", "description": "Graph agents require an accountable owner.", "condition": {"operation": "register_graph_agent", "owner_present": False}, "effect": {"decision": "deny", "reason": "graph_agent_owner_required", "required_action": "assign_graph_agent_owner"}},
	{"name": "graph_agent_requires_purpose", "description": "Graph agents require a documented purpose.", "condition": {"operation": "register_graph_agent", "purpose_present": False}, "effect": {"decision": "deny", "reason": "graph_agent_purpose_required", "required_action": "document_graph_agent_purpose"}},
	{"name": "graph_agent_requires_contribution_disclosure", "description": "Graph agents must disclose machine-authored graph-governance contributions.", "condition": {"operation": "register_graph_agent", "contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "graph_agent_contribution_disclosure_required", "required_action": "disclose_machine_contribution"}},
	{"name": "graph_agent_privileged_role_requires_human_approval", "description": "Privileged graph-agent roles require human approval evidence.", "condition": {"operation": "register_graph_agent", "privileged_role": True, "human_approval_required": False}, "effect": {"decision": "require_review", "reason": "graph_agent_human_approval_required", "required_action": "record_human_graph_agent_approval"}},
	{"name": "bytewax_grph_stream_required", "description": "GRPH lifecycle batches must be routed through Bytewax.", "condition": {"operation": "validate_grph_lifecycle_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_lifecycle_stream_required", "required_action": "route_grph_lifecycle_batch_to_bytewax"}},
]


UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/grph/dashboard", "component": "GRPHDashboard", "permission": "grph:view", "nav_group": "Overview"},
	{"name": "explorer", "path": "/grph/explorer", "component": "GraphExplorer", "permission": "grph:query", "nav_group": "Graph"},
	{"name": "schemas", "path": "/grph/schemas", "component": "GraphSchemaManager", "permission": "grph:manage_schema", "nav_group": "Schema"},
	{"name": "nodes", "path": "/grph/nodes", "component": "GraphNodeManager", "permission": "grph:write", "nav_group": "Graph"},
	{"name": "edges", "path": "/grph/edges", "component": "GraphEdgeManager", "permission": "grph:write", "nav_group": "Graph"},
	{"name": "traversal", "path": "/grph/traversal", "component": "GraphTraversalWorkbench", "permission": "grph:query", "nav_group": "Graph"},
	{"name": "lineage", "path": "/grph/lineage", "component": "LineageGraphViewer", "permission": "grph:view", "nav_group": "Lineage"},
	{"name": "impact", "path": "/grph/impact", "component": "GraphImpactAnalysis", "permission": "grph:query", "nav_group": "Lineage"},
	{"name": "quality", "path": "/grph/quality", "component": "GraphQualityConsole", "permission": "grph:govern", "nav_group": "Quality"},
	{"name": "governance", "path": "/grph/governance", "component": "GraphGovernance", "permission": "grph:govern", "nav_group": "Governance"},
	{"name": "agents", "path": "/grph/agents", "component": "GraphAgentRoster", "permission": "grph:govern", "nav_group": "Governance"},
	{"name": "lifecycle", "path": "/grph/lifecycle", "component": "GRPHLifecycleBatchMonitor", "permission": "grph:admin", "nav_group": "Operations"},
	{"name": "audit", "path": "/grph/audit", "component": "GraphAuditTimeline", "permission": "grph:audit", "nav_group": "Governance"},
	{"name": "settings", "path": "/grph/settings", "component": "GRPHSettings", "permission": "grph:admin", "nav_group": "Administration"},
]


THEME: dict[str, Any] = {
	"name": "grph_relationship_console",
	"tokens": {
		"color.primary": "#2A5D67",
		"color.accent": "#D98E04",
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
		"graph_canvas": {"icon": "network", "visual": "node-link", "status_indicator": "schema-chip"},
		"node_panel": {"visual": "property-list", "risk_style": "classification-band"},
		"edge_panel": {"visual": "relationship-list", "highlight": "type-chip"},
		"traversal_panel": {"visual": "depth-control", "status_style": "review-chip"},
		"lineage_path": {"visual": "path-trace", "threshold_style": "depth-band"},
		"impact_map": {"visual": "dependency-map", "status_style": "blast-radius-chip"},
		"quality_panel": {"visual": "quality-scorecard", "status_style": "health-chip"},
		"graph_agent_roster": {"icon": "bot", "status_indicator": "agent-approval-chip", "risk_style": "relationship-scope-band"},
		"bytewax_lifecycle_panel": {"icon": "git-branch", "status_indicator": "stream-chip", "risk_style": "processor-band"},
		"audit_timeline": {"visual": "event-timeline", "status_style": "evidence-chip"},
	},
}


def agent_manifest() -> dict[str, Any]:
	"""Return first-class GRPH agent composition manifest."""
	return {
		"first_class": True,
		"supported_runtimes": list(SUPPORTED_GRPH_AGENT_RUNTIMES),
		"supported_roles": list(SUPPORTED_GRPH_AGENT_ROLES),
		"privileged_roles": list(PRIVILEGED_GRPH_AGENT_ROLES),
		"required_fields": ["tenant_id", "agent_id", "name", "runtime", "role", "scope", "owner", "purpose"],
		"guardrails": [
			"supported_runtime",
			"supported_role",
			"explicit_scope",
			"accountable_owner",
			"declared_purpose",
			"machine_contribution_disclosure",
			"human_approval_for_privileged_roles",
		],
		"adapter_contract": "aicr_provider_neutral_graph_agent_adapter",
	}


def streaming_manifest() -> dict[str, Any]:
	"""Return the GRPH Bytewax lifecycle stream contract."""
	return {
		"engine": "bytewax",
		"lifecycle_stream": "grph.lifecycle",
		"watermark": "event_time",
		"required_processor": "bytewax",
		"required_operations": [
			"schema_batch",
			"node_batch",
			"edge_batch",
			"traversal_batch",
			"lineage_batch",
			"impact_batch",
			"quality_batch",
			"graph_agent_batch",
		],
		"topics": [
			"grph.schemas",
			"grph.nodes",
			"grph.edges",
			"grph.traversals",
			"grph.lineage",
			"grph.impact",
			"grph.quality",
			"grph.agents",
		],
		"broker_core_dependency_allowed": False,
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable GRPH capability contract."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "grph",
		"display_name": "Graph Data Management",
		"provides": ["graph_data_management", "relationship_intelligence", "graph_agent_composition"],
		"requires": ["mdm", "meta", "etlp", "srch", "aicr", "conf"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/grph/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True,
		},
		"agents": agent_manifest(),
		"streaming": streaming_manifest(),
		"theme": deepcopy(THEME),
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate default GRPH governance rules."""
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
