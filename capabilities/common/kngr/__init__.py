"""APG Knowledge Graph (KNGR) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract

__version__ = "1.0.0"
__capability_id__ = "kngr"
__capability_name__ = "Knowledge Graph"
__apg_dependencies__ = ["grph", "nlpc", "meta", "srch", "onto"]

capability_metadata: dict[str, Any] = {
	"name": "kngr",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant-aware semantic knowledge graph construction, enrichment, reasoning, and governance",
	"category": "knowledge_search",
	"subcategory": "knowledge_graph",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["source_registration", "entity_resolution", "relationship_linking", "semantic_enrichment", "knowledge_graphs", "reasoning_paths", "contextual_relationships", "graph_publication", "audit_evidence"],
	"permissions": ["kngr:view", "kngr:source", "kngr:query", "kngr:enrich", "kngr:curate", "kngr:publish", "kngr:reason", "kngr:govern", "kngr:audit", "kngr:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register KNGR with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "kngr",
		"aliases": ["knowledge_graph", "semantic_graph", "entity_graph"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["aicr", "audl", "auth", "cach", "moni"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"source_registration": "Register tenant-scoped source assets with evidence and confidence",
			"entity_resolution": "Resolve entities into curated graph identities",
			"relationship_linking": "Link evidence-backed semantic relationships between tenant-local entities",
			"semantic_enrichment": "Attach semantic labels and extracted relationships from NLPC/META",
			"knowledge_graphs": "Build tenant-scoped knowledge graphs over graph data",
			"reasoning_paths": "Expose bounded reasoning paths and contextual neighborhoods",
			"graph_publication": "Publish curated graph snapshots for generated applications",
			"audit_evidence": "Capture audit events for knowledge source, entity, relationship, reasoning, and publication operations",
			"capability_rules": "Evaluate deterministic knowledge-graph governance rules",
			"visual_theming": "Apply knowledge-graph theme tokens and components"
		},
		"endpoints": {
			"sources": "/kngr/api/v1/sources",
			"entities": "/kngr/api/v1/entities",
			"relationships": "/kngr/api/v1/relationships",
			"enrichment": "/kngr/api/v1/enrichment",
			"reasoning": "/kngr/api/v1/reasoning",
			"curation": "/kngr/api/v1/curation",
			"context": "/kngr/api/v1/context",
			"publication": "/kngr/api/v1/publication",
			"audit": "/kngr/api/v1/audit"
		},
		"adapters": contract["configuration"]["adapters"],
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get KNGR capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = ["capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
