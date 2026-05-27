"""Executable capability contract for APG Graph Data Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"graph": {"schema_required": True, "max_traversal_depth": 8, "lineage_graphs_enabled": True},
	"storage": {"node_owner_required": True, "edge_type_required": True, "property_validation_enabled": True},
	"governance": {"require_tenant_context": True, "audit_mutations": True, "restricted_relationship_review_required": True},
	"ui": {"enable_graph_explorer": True, "enable_schema_manager": True, "enable_lineage_viewer": True, "enable_quality_console": True},
	"theme": {"default_theme": "grph_relationship_console", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {"type": "object", "required": ["tenant_id", "graph", "storage", "governance", "ui", "theme"], "properties": {key: {"type": "object"} for key in ["graph", "storage", "governance", "ui", "theme"]} | {"tenant_id": {"type": "string", "minLength": 1}}}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All graph operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "node_write_requires_owner", "description": "Graph node writes require an owner.", "condition": {"operation": "write_node", "owner_assigned": False}, "effect": {"decision": "deny", "reason": "node_owner_required", "required_action": "assign_owner"}},
	{"name": "edge_write_requires_type", "description": "Graph edge writes require an edge type.", "condition": {"operation": "write_edge", "edge_type_present": False}, "effect": {"decision": "deny", "reason": "edge_type_required", "required_action": "attach_edge_type"}},
	{"name": "restricted_relationship_requires_review", "description": "Restricted relationships require governance review.", "condition": {"relationship_classification": "restricted", "review_recorded": False}, "effect": {"decision": "require_review", "reason": "restricted_relationship_review_required", "required_action": "record_relationship_review"}},
	{"name": "deep_traversal_requires_review", "description": "Deep graph traversals require review.", "condition": {"traversal_depth_gt": 8, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "deep_traversal_review_required", "required_action": "record_traversal_review"}},
	{"name": "lineage_graph_requires_source_asset", "description": "Lineage graph mutations require source asset linkage.", "condition": {"graph_type": "lineage", "source_asset_present": False}, "effect": {"decision": "deny", "reason": "source_asset_required", "required_action": "attach_source_asset"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/grph/dashboard", "component": "GRPHDashboard", "permission": "grph:view", "nav_group": "Overview"},
	{"name": "explorer", "path": "/grph/explorer", "component": "GraphExplorer", "permission": "grph:query", "nav_group": "Graph"},
	{"name": "schema", "path": "/grph/schema", "component": "GraphSchemaManager", "permission": "grph:manage_schema", "nav_group": "Schema"},
	{"name": "lineage", "path": "/grph/lineage", "component": "LineageGraphViewer", "permission": "grph:view", "nav_group": "Lineage"},
	{"name": "quality", "path": "/grph/quality", "component": "GraphQualityConsole", "permission": "grph:govern", "nav_group": "Quality"},
	{"name": "settings", "path": "/grph/settings", "component": "GRPHSettings", "permission": "grph:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {"name": "grph_relationship_console", "tokens": {"color.primary": "#2A5D67", "color.accent": "#D98E04", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F6F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"}, "components": {"graph_canvas": {"icon": "network", "visual": "node-link", "status_indicator": "schema-chip"}, "node_panel": {"visual": "property-list", "risk_style": "classification-band"}, "edge_panel": {"visual": "relationship-list", "highlight": "type-chip"}, "lineage_path": {"visual": "path-trace", "threshold_style": "depth-band"}}}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "grph", "display_name": "Graph Data Management", "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "view_module": "__init__.py", "api_prefix": "/grph/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME)}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	return _evaluate(RULES, context)


def _evaluate(rules: list[dict[str, Any]], context: dict[str, Any]) -> dict[str, Any]:
	matched: list[str] = []
	actions: list[dict[str, Any]] = []
	decision = "allow"
	for rule in rules:
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
		if key.endswith("_gt"):
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
