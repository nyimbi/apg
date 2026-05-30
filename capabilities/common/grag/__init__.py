"""APG Graph-based RAG (GRAG) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract

__version__ = "1.0.0"
__capability_id__ = "grag"
__capability_name__ = "Graph-based RAG"
__apg_dependencies__ = ["ragn", "kngr", "grph"]

capability_metadata: dict[str, Any] = {
	"name": "grag",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Hybrid vector-graph retrieval, multi-hop reasoning, provenance, and graph-grounded generation",
	"category": "knowledge_search",
	"subcategory": "graph_rag",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["hybrid_retrieval", "multi_hop_reasoning", "graph_grounded_generation", "reasoning_explanations", "knowledge_curation", "answer_publication"],
	"permissions": ["grag:view", "grag:query", "grag:reason", "grag:generate", "grag:curate", "grag:manage_graphs", "grag:manage_sources", "grag:govern", "grag:audit", "grag:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register GRAG with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "grag",
		"aliases": ["graph_rag", "graphrag", "graph_augmented_generation"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["srch", "nlpc", "aicr", "onto", "meta", "auth", "audl", "cach", "moni"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"graph_source_management": "Register governed graph sources with provenance and ownership",
			"vector_source_management": "Register vector indexes and embedding-model provenance",
			"hybrid_retrieval": "Fuse vector retrieval with graph traversal for answer context",
			"multi_hop_reasoning": "Run bounded reasoning paths across graph relationships",
			"graph_grounded_generation": "Generate cited answers grounded in graph evidence",
			"answer_publication": "Curate and publish graph-grounded answers with audit evidence",
			"reasoning_explanations": "Expose answer paths, confidence, and provenance",
			"capability_rules": "Evaluate deterministic GraphRAG governance rules",
			"visual_theming": "Apply graph-RAG reasoning theme tokens and components"
		},
		"endpoints": {
			"status": "/grag/api/v1/status",
			"graph_sources": "/grag/api/v1/graph-sources",
			"vector_sources": "/grag/api/v1/vector-sources",
			"query": "/grag/api/v1/query",
			"hybrid_retrieval": "/grag/api/v1/hybrid-retrieval",
			"reasoning": "/grag/api/v1/reasoning",
			"provenance": "/grag/api/v1/provenance",
			"generation": "/grag/api/v1/generation",
			"curation": "/grag/api/v1/curation",
			"publications": "/grag/api/v1/publications",
			"audit": "/grag/api/v1/audit"
		},
		"adapters": contract["configuration"]["adapters"],
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get GRAG capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = ["capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
