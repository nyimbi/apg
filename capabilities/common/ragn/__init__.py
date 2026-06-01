"""APG Retrieval-Augmented Generation (RAGN) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract

__version__ = "1.0.0"
__capability_id__ = "ragn"
__capability_name__ = "Retrieval-Augmented Generation"
__apg_dependencies__ = ["srch", "nlpc", "aicr", "conf", "audl"]

capability_metadata: dict[str, Any] = {
	"name": "ragn",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant-scoped retrieval-augmented generation with governed knowledge bases, citations, conversations, and answer review",
	"category": "knowledge_search",
	"subcategory": "retrieval_augmented_generation",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": [
		"knowledge_base_management",
		"document_ingestion",
		"context_retrieval",
		"grounded_generation",
		"conversation_memory",
		"citation_governance",
		"answer_curation",
		"rag_agent_composition",
		"bytewax_lifecycle_batches",
	],
	"permissions": [
		"ragn:view",
		"ragn:query",
		"ragn:manage_kb",
		"ragn:curate",
		"ragn:govern",
		"ragn:audit",
		"ragn:admin",
	],
}


def register_capability() -> dict[str, Any]:
	"""Register RAGN with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "ragn",
		"aliases": ["rag", "retrieval_augmented_generation", "grounded_generation"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["mlcm", "auth", "audl", "cach", "kngr", "grph", "meta", "moni"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"knowledge_base_management": "Create tenant-scoped knowledge bases with owners and source attribution",
			"document_ingestion": "Ingest classified source documents with content hashes and source URIs",
			"context_retrieval": "Retrieve governed context with confidence, RBAC, and restricted-source filtering",
			"grounded_generation": "Generate cited answers from approved retrieval context",
			"conversation_memory": "Record governed conversation turns and answer traces",
			"citation_governance": "Attach source, document, and chunk-level citations",
			"answer_curation": "Review low-confidence or high-risk RAG answers",
			"rag_agent_composition": "Compose Codex, Claude Code, opencode, and Pi style RAG agents behind provider-neutral guardrails",
			"bytewax_lifecycle_batches": "Validate RAG lifecycle batches through Bytewax-first processor contracts",
			"review_evidence": "Persist review-required RAG lifecycle outcomes as pending-review records with policy evidence",
			"capability_rules": "Evaluate deterministic RAG governance rules",
			"visual_theming": "Apply RAG studio theme tokens and components",
		},
		"endpoints": {
			"status": "/ragn/api/v1/status",
			"knowledge_bases": "/ragn/api/v1/knowledge-bases",
			"documents": "/ragn/api/v1/documents",
			"retrieval": "/ragn/api/v1/retrieval",
			"generation": "/ragn/api/v1/generation",
			"conversations": "/ragn/api/v1/conversations",
			"curation": "/ragn/api/v1/curation",
			"pending_reviews": "/ragn/api/v1/pending-reviews",
			"agents": "/ragn/api/v1/agents",
			"lifecycle": "/ragn/api/v1/lifecycle",
			"audit": "/ragn/api/v1/audit",
		},
		"adapters": contract["configuration"]["adapters"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": capability_metadata["permissions"],
	}


def get_capability_info() -> dict[str, Any]:
	"""Get RAGN capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


def get_supported_modalities() -> list[str]:
	"""Return modalities supported by the generated-app contract."""
	return ["text", "structured_data", "code", "image", "audio"]


__all__ = [
	"capability_metadata",
	"register_capability",
	"get_capability_info",
	"get_capability_contract",
	"evaluate_capability_rules",
	"get_supported_modalities",
	"__version__",
	"__capability_id__",
	"__capability_name__",
	"__apg_dependencies__",
]
