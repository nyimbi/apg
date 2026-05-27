"""Executable capability contract for APG Search Engine."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"indexing": {"enabled": True, "owner_required": True, "classification_required": True, "max_documents_per_batch": 10000},
	"query": {"keyword_enabled": True, "semantic_enabled": True, "rbac_filter_required": True, "max_result_window": 1000},
	"governance": {"require_tenant_context": True, "audit_queries": True, "restricted_content_filter_required": True},
	"ui": {"enable_search_console": True, "enable_index_manager": True, "enable_query_analytics": True, "enable_governance": True},
	"theme": {"default_theme": "srch_discovery_console", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {"type": "object", "required": ["tenant_id", "indexing", "query", "governance", "ui", "theme"], "properties": {key: {"type": "object"} for key in ["indexing", "query", "governance", "ui", "theme"]} | {"tenant_id": {"type": "string", "minLength": 1}}}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All search operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "indexing_requires_owner", "description": "Search indices require an owner.", "condition": {"operation": "create_index", "owner_assigned": False}, "effect": {"decision": "deny", "reason": "index_owner_required", "required_action": "assign_owner"}},
	{"name": "restricted_query_requires_rbac_filter", "description": "Restricted content queries require RBAC filters.", "condition": {"content_classification": "restricted", "rbac_filter_applied": False}, "effect": {"decision": "deny", "reason": "rbac_filter_required", "required_action": "apply_rbac_filter"}},
	{"name": "semantic_query_requires_embeddings", "description": "Semantic search requires an embedding index.", "condition": {"query_type": "semantic", "embedding_index_ready": False}, "effect": {"decision": "deny", "reason": "embedding_index_required", "required_action": "build_embedding_index"}},
	{"name": "large_result_window_requires_review", "description": "Large result windows require review.", "condition": {"result_window_gt": 1000, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "large_result_window_review_required", "required_action": "record_query_review"}},
	{"name": "bulk_index_requires_lineage", "description": "Bulk indexing requires source lineage.", "condition": {"operation": "bulk_index", "source_lineage_present": False}, "effect": {"decision": "deny", "reason": "source_lineage_required", "required_action": "attach_source_lineage"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/srch/dashboard", "component": "SRCHDashboard", "permission": "srch:view", "nav_group": "Overview"},
	{"name": "search", "path": "/srch/search", "component": "SearchConsole", "permission": "srch:query", "nav_group": "Search"},
	{"name": "indices", "path": "/srch/indices", "component": "IndexManager", "permission": "srch:manage_indices", "nav_group": "Indexes"},
	{"name": "documents", "path": "/srch/documents", "component": "DocumentIndexer", "permission": "srch:index", "nav_group": "Indexes"},
	{"name": "analytics", "path": "/srch/analytics", "component": "QueryAnalytics", "permission": "srch:view", "nav_group": "Operations"},
	{"name": "governance", "path": "/srch/governance", "component": "SearchGovernance", "permission": "srch:govern", "nav_group": "Governance"},
	{"name": "settings", "path": "/srch/settings", "component": "SRCHSettings", "permission": "srch:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {"name": "srch_discovery_console", "tokens": {"color.primary": "#235789", "color.accent": "#F1A208", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F6F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"}, "components": {"result_card": {"icon": "search", "status_indicator": "classification-pill", "risk_style": "access-band"}, "facet_panel": {"visual": "filter-stack", "highlight": "active-chip"}, "index_health": {"visual": "coverage-meter", "status_style": "freshness-pill"}, "query_trace": {"visual": "retrieval-timeline", "threshold_style": "latency-band"}}}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "srch", "display_name": "Search Engine", "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "view_module": "__init__.py", "api_prefix": "/srch/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME)}


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
