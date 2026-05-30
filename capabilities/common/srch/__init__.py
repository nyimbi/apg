"""APG Search Engine (SRCH) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract

__version__ = "1.0.0"
__capability_id__ = "srch"
__capability_name__ = "Search Engine"
__apg_dependencies__ = ["etlp", "meta", "nlpc"]

capability_metadata: dict[str, Any] = {
	"name": "srch",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant-aware enterprise search, indexing, semantic retrieval, and query governance",
	"category": "knowledge_search",
	"subcategory": "search_engine",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["indexing", "keyword_search", "semantic_search", "hybrid_search", "faceted_search", "access_filtered_retrieval", "query_analytics"],
	"permissions": ["srch:view", "srch:query", "srch:index", "srch:manage_indices", "srch:govern", "srch:audit", "srch:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register SRCH with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "srch",
		"aliases": ["search", "enterprise_search", "semantic_search"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["auth", "audl", "cach", "aicr"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"adapters": contract["configuration"]["adapters"],
		"capabilities": {
			"indexing": "Index tenant-scoped structured and unstructured content",
			"bulk_indexing": "Coordinate lineage-protected Bytewax-backed indexing batches",
			"keyword_search": "Run governed full-text search with facets",
			"semantic_search": "Run embedding-backed semantic retrieval through NLPC/AICR",
			"hybrid_search": "Combine lexical and embedding-backed retrieval",
			"access_filtered_retrieval": "Apply tenant, RBAC, and classification filters before result return",
			"facet_navigation": "Expose governed facet counts and filters",
			"query_analytics": "Track query decisions, reviews, denials, and retrieval health",
			"capability_rules": "Evaluate deterministic search-governance rules",
			"visual_theming": "Apply search-console theme tokens and components"
		},
		"endpoints": {
			"query": "/srch/api/v1/query",
			"indices": "/srch/api/v1/indices",
			"documents": "/srch/api/v1/documents",
			"bulk": "/srch/api/v1/bulk",
			"facets": "/srch/api/v1/facets",
			"analytics": "/srch/api/v1/analytics",
			"audit": "/srch/api/v1/audit"
		},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get SRCH capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = ["capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
