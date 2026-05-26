"""APG Graph Data Management (GRPH) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract

__version__ = "1.0.0"
__capability_id__ = "grph"
__capability_name__ = "Graph Data Management"
__apg_dependencies__ = ["mdm", "meta", "etlp"]

capability_metadata: dict[str, Any] = {
	"name": "grph",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant-aware graph storage, relationship modeling, traversal, and graph-governance foundation",
	"category": "knowledge_search",
	"subcategory": "graph_data",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["graph_store", "relationship_modeling", "graph_traversal", "lineage_graphs", "graph_quality"],
	"permissions": ["grph:view", "grph:query", "grph:write", "grph:manage_schema", "grph:govern", "grph:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register GRPH with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "grph",
		"aliases": ["graph_data", "relationship_graph", "graph_store"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["auth", "audl", "cach", "moni"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"graph_store": "Persist tenant-scoped nodes, edges, properties, and labels",
			"relationship_modeling": "Govern relationship schema and entity links",
			"graph_traversal": "Run bounded graph traversals and path queries",
			"lineage_graphs": "Represent lineage and dependency graphs for data assets",
			"capability_rules": "Evaluate deterministic graph-governance rules",
			"visual_theming": "Apply graph-console theme tokens and components"
		},
		"endpoints": {
			"nodes": "/grph/api/v1/nodes",
			"edges": "/grph/api/v1/edges",
			"queries": "/grph/api/v1/queries",
			"schema": "/grph/api/v1/schema",
			"lineage": "/grph/api/v1/lineage"
		},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get GRPH capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = ["capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
