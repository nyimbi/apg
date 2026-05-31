"""APG Ontology Management (ONTO) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract

__version__ = "1.0.0"
__capability_id__ = "onto"
__capability_name__ = "Ontology Management"
__apg_dependencies__ = ["kngr", "meta", "nlpc", "grph", "srch", "aicr", "conf", "auth", "audl"]

capability_metadata: dict[str, Any] = {
	"name": "onto",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant-aware ontology, taxonomy, vocabulary, mapping, and semantic-governance management",
	"category": "knowledge_search",
	"subcategory": "ontology_management",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["ontology_registry", "namespace_management", "taxonomy_management", "vocabulary_governance", "semantic_mapping", "ontology_validation", "term_curation", "ontology_exchange", "ontology_agent_composition", "bytewax_lifecycle_batches"],
	"permissions": ["onto:view", "onto:edit", "onto:map", "onto:publish", "onto:govern", "onto:audit", "onto:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register ONTO with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "onto",
		"aliases": ["ontology", "taxonomy", "vocabulary_governance"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["cach", "moni"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"ontology_registry": "Register tenant-scoped ontologies and versions",
			"namespace_management": "Govern ontology namespace prefixes and URIs",
			"taxonomy_management": "Curate hierarchical taxonomies and controlled vocabularies",
			"semantic_mapping": "Map terms, entities, and metadata concepts across domains",
			"ontology_validation": "Validate duplicates, draft terms, taxonomy integrity, and mapping reviews",
			"ontology_exchange": "Prepare ontology export artifacts in configured interchange formats",
			"ontology_agent_composition": "Compose Codex, Claude Code, opencode, and Pi style ontology agents behind provider-neutral guardrails",
			"bytewax_lifecycle_batches": "Validate ontology lifecycle batches through Bytewax-first processor contracts",
			"term_curation": "Govern term ownership, status, synonyms, and publication",
			"capability_rules": "Evaluate deterministic ontology-governance rules",
			"visual_theming": "Apply ontology-workbench theme tokens and components"
		},
		"endpoints": {
			"status": "/onto/api/v1/status",
			"ontologies": "/onto/api/v1/ontologies",
			"namespaces": "/onto/api/v1/namespaces",
			"terms": "/onto/api/v1/terms",
			"mappings": "/onto/api/v1/mappings",
			"taxonomies": "/onto/api/v1/taxonomies",
			"validation": "/onto/api/v1/validation",
			"exports": "/onto/api/v1/exports",
			"publication": "/onto/api/v1/publication",
			"agents": "/onto/api/v1/agents",
			"lifecycle": "/onto/api/v1/lifecycle",
			"audit": "/onto/api/v1/audit"
		},
		"adapters": contract["configuration"]["adapters"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get ONTO capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = ["capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
