"""Executable capability contract for APG Search Engine."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"indices": {
		"owner_required": True,
		"name_required": True,
		"classification_required": True,
		"content_type_required": True,
		"allowed_classifications": ["public", "internal", "confidential", "restricted"],
		"allowed_content_types": ["document", "article", "record", "event", "profile"],
	},
	"documents": {
		"id_required": True,
		"title_required": True,
		"body_required": True,
		"source_lineage_required": True,
		"classification_required": True,
	},
	"indexing": {
		"enabled": True,
		"owner_required": True,
		"classification_required": True,
		"max_documents_per_batch": 10000,
		"bulk_lineage_required": True,
	},
	"query": {
		"keyword_enabled": True,
		"semantic_enabled": True,
		"hybrid_enabled": True,
		"rbac_filter_required": True,
		"max_result_window": 1000,
		"allowed_query_types": ["keyword", "semantic", "hybrid"],
	},
	"ranking": {
		"keyword_weight": 1.0,
		"semantic_weight": 1.0,
		"freshness_boost_enabled": True,
		"explain_ranking": True,
	},
	"facets": {
		"enabled": True,
		"max_facets": 50,
		"allowed_facet_keys": ["module", "kind", "owner", "classification", "source"],
	},
	"security": {
		"restricted_content_filter_required": True,
		"cross_tenant_search_allowed": False,
		"rbac_filter_required": True,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_queries": True,
		"audit_indexing": True,
		"restricted_content_filter_required": True,
	},
	"observability": {
		"metrics_required": True,
		"trace_required": True,
		"audit_required": True,
		"event_stream": "bytewax",
		"quality_metrics_required": True,
	},
	"adapters": {
		"generated_app_runtime": "service.SrchService",
		"helper_runtime": "search_runtime.py",
		"production_runtime": "service.SrchService",
		"http_api": "api.py",
		"event_stream": "bytewax",
		"data_pipeline": "etlp",
		"metadata": "meta",
		"nlp": "nlpc",
		"ai_core": "aicr",
		"auth_provider": "auth",
		"audit_sink": "audl",
		"cache": "cach",
		"metrics_sink": "moni",
		"vector_index": "aicr",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_search_console": True,
		"enable_index_manager": True,
		"enable_document_indexer": True,
		"enable_bulk_indexing": True,
		"enable_facets": True,
		"enable_query_analytics": True,
		"enable_ranking": True,
		"enable_access_review": True,
		"enable_governance": True,
		"enable_audit": True,
		"enable_settings": True,
	},
	"theme": {"default_theme": "srch_discovery_console", "allow_tenant_overrides": True},
}


CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"indices",
		"documents",
		"indexing",
		"query",
		"ranking",
		"facets",
		"security",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {key: {"type": "object"} for key in [
		"indices",
		"documents",
		"indexing",
		"query",
		"ranking",
		"facets",
		"security",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]} | {"tenant_id": {"type": "string", "minLength": 1}},
}


RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All search operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "index_requires_name", "description": "Search indices require a name.", "condition": {"operation": "create_index", "index_name_present": False}, "effect": {"decision": "deny", "reason": "index_name_required", "required_action": "attach_index_name"}},
	{"name": "indexing_requires_owner", "description": "Search indices require an owner.", "condition": {"operation": "create_index", "owner_assigned": False}, "effect": {"decision": "deny", "reason": "index_owner_required", "required_action": "assign_owner"}},
	{"name": "index_requires_content_type", "description": "Search indices require a content type.", "condition": {"operation": "create_index", "content_type_present": False}, "effect": {"decision": "deny", "reason": "index_content_type_required", "required_action": "attach_content_type"}},
	{"name": "index_content_type_requires_review", "description": "Unknown content types require review.", "condition": {"operation": "create_index", "content_type_known": False, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "index_content_type_review_required", "required_action": "review_content_type"}},
	{"name": "index_requires_classification", "description": "Search indices require a content classification.", "condition": {"operation": "create_index", "classification_present": False}, "effect": {"decision": "deny", "reason": "index_classification_required", "required_action": "attach_classification"}},
	{"name": "index_classification_requires_review", "description": "Unknown classifications require review.", "condition": {"operation": "create_index", "classification_known": False, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "index_classification_review_required", "required_action": "review_classification"}},
	{"name": "restricted_index_requires_lineage", "description": "Restricted indices require source lineage.", "condition": {"operation": "create_index", "content_classification": "restricted", "source_lineage_present": False}, "effect": {"decision": "deny", "reason": "restricted_index_lineage_required", "required_action": "attach_source_lineage"}},
	{"name": "document_requires_index", "description": "Documents require a registered search index.", "condition": {"operation": "index_document", "index_present": False}, "effect": {"decision": "deny", "reason": "document_index_required", "required_action": "select_index"}},
	{"name": "document_requires_id", "description": "Documents require an identifier.", "condition": {"operation": "index_document", "document_id_present": False}, "effect": {"decision": "deny", "reason": "document_id_required", "required_action": "attach_document_id"}},
	{"name": "document_requires_title", "description": "Documents require a title.", "condition": {"operation": "index_document", "title_present": False}, "effect": {"decision": "deny", "reason": "document_title_required", "required_action": "attach_title"}},
	{"name": "document_requires_body", "description": "Documents require searchable body text.", "condition": {"operation": "index_document", "body_present": False}, "effect": {"decision": "deny", "reason": "document_body_required", "required_action": "attach_body"}},
	{"name": "document_requires_lineage", "description": "Documents require source lineage.", "condition": {"operation": "index_document", "source_lineage_present": False}, "effect": {"decision": "deny", "reason": "source_lineage_required", "required_action": "attach_source_lineage"}},
	{"name": "document_requires_classification", "description": "Documents require classification metadata.", "condition": {"operation": "index_document", "classification_present": False}, "effect": {"decision": "deny", "reason": "document_classification_required", "required_action": "attach_classification"}},
	{"name": "bulk_index_requires_documents", "description": "Bulk indexing requires documents.", "condition": {"operation": "bulk_index", "document_count_lt": 1}, "effect": {"decision": "deny", "reason": "bulk_documents_required", "required_action": "attach_documents"}},
	{"name": "bulk_index_requires_lineage", "description": "Bulk indexing requires source lineage.", "condition": {"operation": "bulk_index", "source_lineage_present": False}, "effect": {"decision": "deny", "reason": "source_lineage_required", "required_action": "attach_source_lineage"}},
	{"name": "bulk_index_batch_requires_review", "description": "Large indexing batches require review.", "condition": {"operation": "bulk_index", "document_count_gt": 10000, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "bulk_index_review_required", "required_action": "record_bulk_review"}},
	{"name": "query_requires_text", "description": "Queries require query text.", "condition": {"operation": "query", "query_text_present": False}, "effect": {"decision": "deny", "reason": "query_text_required", "required_action": "attach_query_text"}},
	{"name": "query_requires_index", "description": "Queries require at least one index.", "condition": {"operation": "query", "index_ids_present": False}, "effect": {"decision": "deny", "reason": "query_index_required", "required_action": "select_index"}},
	{"name": "query_requires_type", "description": "Queries require a type.", "condition": {"operation": "query", "query_type_present": False}, "effect": {"decision": "deny", "reason": "query_type_required", "required_action": "choose_query_type"}},
	{"name": "query_type_requires_review", "description": "Unknown query types require review.", "condition": {"operation": "query", "query_type_known": False, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "query_type_review_required", "required_action": "review_query_type"}},
	{"name": "restricted_query_requires_rbac_filter", "description": "Restricted content queries require RBAC filters.", "condition": {"operation": "query", "content_classification": "restricted", "rbac_filter_applied": False}, "effect": {"decision": "deny", "reason": "rbac_filter_required", "required_action": "apply_rbac_filter"}},
	{"name": "semantic_query_requires_embeddings", "description": "Semantic search requires an embedding index.", "condition": {"operation": "query", "query_type": "semantic", "embedding_index_ready": False}, "effect": {"decision": "deny", "reason": "embedding_index_required", "required_action": "build_embedding_index"}},
	{"name": "hybrid_query_requires_embeddings", "description": "Hybrid search requires an embedding index.", "condition": {"operation": "query", "query_type": "hybrid", "embedding_index_ready": False}, "effect": {"decision": "deny", "reason": "embedding_index_required", "required_action": "build_embedding_index"}},
	{"name": "result_window_requires_positive_value", "description": "Result windows must be positive.", "condition": {"operation": "query", "result_window_lt": 1}, "effect": {"decision": "deny", "reason": "result_window_required", "required_action": "choose_positive_result_window"}},
	{"name": "large_result_window_requires_review", "description": "Large result windows require review.", "condition": {"operation": "query", "result_window_review_check": True, "result_window_gt": 1000, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "large_result_window_review_required", "required_action": "record_query_review"}},
	{"name": "facet_key_requires_allowlist", "description": "Facet keys outside the allowlist require review.", "condition": {"operation": "index_document", "facet_keys_allowed": False, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "facet_key_review_required", "required_action": "review_facet_key"}},
	{"name": "cross_tenant_search_denied", "description": "Cross-tenant search is denied by default.", "condition": {"cross_tenant_search": True}, "effect": {"decision": "deny", "reason": "cross_tenant_search_denied", "required_action": "use_tenant_scoped_indices"}},
	{"name": "batch_indexing_requires_bytewax", "description": "Batch indexing streams must use Bytewax.", "condition": {"operation": "configure_batch_indexing", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
	{"name": "index_retire_requires_review", "description": "Index retirement requires review evidence.", "condition": {"operation": "retire_index", "review_recorded": False}, "effect": {"decision": "deny", "reason": "index_retire_review_required", "required_action": "record_retire_review"}},
	{"name": "search_state_change_requires_audit", "description": "Search state changes require audit events.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "audit_event_required", "required_action": "record_audit_event"}},
]


UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/srch/dashboard", "component": "SRCHDashboard", "permission": "srch:view", "nav_group": "Overview"},
	{"name": "search", "path": "/srch/search", "component": "SearchConsole", "permission": "srch:query", "nav_group": "Search"},
	{"name": "indices", "path": "/srch/indices", "component": "IndexManager", "permission": "srch:manage_indices", "nav_group": "Indexes"},
	{"name": "documents", "path": "/srch/documents", "component": "DocumentIndexer", "permission": "srch:index", "nav_group": "Indexes"},
	{"name": "bulk", "path": "/srch/bulk", "component": "BulkIndexQueue", "permission": "srch:index", "nav_group": "Indexes"},
	{"name": "facets", "path": "/srch/facets", "component": "FacetExplorer", "permission": "srch:view", "nav_group": "Search"},
	{"name": "analytics", "path": "/srch/analytics", "component": "QueryAnalytics", "permission": "srch:view", "nav_group": "Operations"},
	{"name": "ranking", "path": "/srch/ranking", "component": "RankingWorkbench", "permission": "srch:govern", "nav_group": "Operations"},
	{"name": "access", "path": "/srch/access", "component": "SearchAccessReview", "permission": "srch:govern", "nav_group": "Governance"},
	{"name": "governance", "path": "/srch/governance", "component": "SearchGovernance", "permission": "srch:govern", "nav_group": "Governance"},
	{"name": "audit", "path": "/srch/audit", "component": "SearchAuditTimeline", "permission": "srch:audit", "nav_group": "Governance"},
	{"name": "settings", "path": "/srch/settings", "component": "SRCHSettings", "permission": "srch:admin", "nav_group": "Administration"},
]


THEME: dict[str, Any] = {
	"name": "srch_discovery_console",
	"tokens": {
		"color.primary": "#235789",
		"color.accent": "#F1A208",
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
		"result_card": {"icon": "search", "status_indicator": "classification-pill", "risk_style": "access-band"},
		"facet_panel": {"visual": "filter-stack", "highlight": "active-chip"},
		"index_health": {"visual": "coverage-meter", "status_style": "freshness-pill"},
		"bulk_queue": {"visual": "queue-table", "status_style": "bytewax-chip"},
		"ranking_panel": {"visual": "weight-grid", "status_style": "explain-chip"},
		"access_review": {"visual": "policy-table", "status_style": "rbac-chip"},
		"query_trace": {"visual": "retrieval-timeline", "threshold_style": "latency-band"},
		"audit_timeline": {"visual": "event-timeline", "status_style": "evidence-chip"},
	},
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable SRCH capability contract."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "srch",
		"display_name": "Search Engine",
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/srch/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True,
		},
		"theme": deepcopy(THEME),
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate default SRCH governance rules."""
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
