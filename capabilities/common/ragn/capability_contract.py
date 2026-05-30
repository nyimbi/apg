"""Executable capability contract for APG Retrieval-Augmented Generation."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"knowledge_bases": {
		"id_required": True,
		"name_required": True,
		"owner_required": True,
		"source_attribution_required": True,
		"default_classification": "internal",
		"retire_requires_review": True,
	},
	"documents": {
		"title_required": True,
		"content_hash_required": True,
		"knowledge_base_required": True,
		"allowed_classifications": ["public", "internal", "confidential", "restricted"],
		"max_documents_per_ingest": 5000,
		"large_ingest_review_threshold": 1000,
	},
	"chunking": {
		"chunk_size": 1000,
		"chunk_overlap": 200,
		"minimum_chunk_size": 100,
		"maximum_chunk_size": 8000,
	},
	"retrieval": {
		"semantic_retrieval_enabled": True,
		"keyword_fallback_enabled": True,
		"minimum_context_confidence": 0.7,
		"max_results": 50,
		"rbac_filter_required": True,
	},
	"generation": {
		"citations_required": True,
		"model_policy_required": True,
		"streaming_enabled": True,
		"external_model_requires_policy": True,
		"max_answer_tokens": 4000,
	},
	"conversations": {
		"conversation_id_required": True,
		"user_id_required": True,
		"memory_enabled": True,
		"max_turns": 100,
		"retention_days": 30,
	},
	"citations": {
		"source_id_required": True,
		"document_id_required": True,
		"chunk_id_required": True,
		"minimum_citation_count": 1,
	},
	"curation": {
		"review_required_for_low_confidence": True,
		"curator_required": True,
		"evidence_required": True,
		"allowed_decisions": ["approved", "rejected", "needs_revision"],
	},
	"security": {
		"cross_tenant_access_allowed": False,
		"restricted_source_filter_required": True,
		"prompt_injection_scan_required": True,
		"unsafe_generation_blocking": True,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_queries": True,
		"audit_ingestion": True,
		"audit_generation": True,
		"audit_curation": True,
	},
	"observability": {
		"metrics_required": True,
		"trace_required": True,
		"audit_required": True,
		"event_stream": "bytewax",
	},
	"adapters": {
		"generated_app_runtime": "rag_runtime.RagnService",
		"production_runtime": "service.RAGService",
		"helper_runtime": "rag_runtime.py",
		"http_api": "api.py",
		"event_stream": "bytewax",
		"search": "srch",
		"nlp": "nlpc",
		"ai_core": "aicr",
		"model_lifecycle": "mlcm",
		"knowledge_graph": "kngr",
		"graph": "grph",
		"metadata": "meta",
		"auth_provider": "auth",
		"audit_sink": "audl",
		"cache": "cach",
		"metrics_sink": "moni",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_studio": True,
		"enable_knowledge_bases": True,
		"enable_documents": True,
		"enable_retrieval": True,
		"enable_generation": True,
		"enable_conversations": True,
		"enable_citations": True,
		"enable_curation": True,
		"enable_governance": True,
		"enable_audit": True,
		"enable_settings": True,
	},
	"theme": {"default_theme": "ragn_answer_studio", "allow_tenant_overrides": True},
}


CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"knowledge_bases",
		"documents",
		"chunking",
		"retrieval",
		"generation",
		"conversations",
		"citations",
		"curation",
		"security",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {key: {"type": "object"} for key in [
		"knowledge_bases",
		"documents",
		"chunking",
		"retrieval",
		"generation",
		"conversations",
		"citations",
		"curation",
		"security",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]} | {"tenant_id": {"type": "string", "minLength": 1}},
}


RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All RAG operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "knowledge_base_requires_id", "description": "Knowledge bases require stable identifiers.", "condition": {"operation": "create_knowledge_base", "knowledge_base_id_present": False}, "effect": {"decision": "deny", "reason": "knowledge_base_id_required", "required_action": "attach_knowledge_base_id"}},
	{"name": "knowledge_base_requires_name", "description": "Knowledge bases require names.", "condition": {"operation": "create_knowledge_base", "knowledge_base_name_present": False}, "effect": {"decision": "deny", "reason": "knowledge_base_name_required", "required_action": "attach_knowledge_base_name"}},
	{"name": "knowledge_base_requires_owner", "description": "Knowledge bases require an owner.", "condition": {"operation": "create_knowledge_base", "owner_assigned": False}, "effect": {"decision": "deny", "reason": "knowledge_base_owner_required", "required_action": "assign_owner"}},
	{"name": "knowledge_base_requires_source_attribution", "description": "Knowledge bases require source attribution policy.", "condition": {"operation": "create_knowledge_base", "source_attribution_present": False}, "effect": {"decision": "deny", "reason": "source_attribution_required", "required_action": "attach_source_attribution_policy"}},
	{"name": "document_requires_knowledge_base", "description": "Document ingestion requires a knowledge base.", "condition": {"operation": "ingest_document", "knowledge_base_present": False}, "effect": {"decision": "deny", "reason": "knowledge_base_required", "required_action": "select_knowledge_base"}},
	{"name": "document_requires_title", "description": "Document ingestion requires a title.", "condition": {"operation": "ingest_document", "document_title_present": False}, "effect": {"decision": "deny", "reason": "document_title_required", "required_action": "attach_document_title"}},
	{"name": "document_requires_content_hash", "description": "Document ingestion requires a content hash.", "condition": {"operation": "ingest_document", "content_hash_present": False}, "effect": {"decision": "deny", "reason": "content_hash_required", "required_action": "attach_content_hash"}},
	{"name": "document_requires_source_uri", "description": "Document ingestion requires a source URI.", "condition": {"operation": "ingest_document", "source_uri_present": False}, "effect": {"decision": "deny", "reason": "source_uri_required", "required_action": "attach_source_uri"}},
	{"name": "document_classification_requires_allowed_value", "description": "Document classifications must be configured values.", "condition": {"operation": "ingest_document", "classification_allowed": False}, "effect": {"decision": "deny", "reason": "document_classification_invalid", "required_action": "choose_allowed_classification"}},
	{"name": "large_ingest_requires_review", "description": "Large document ingestion batches require review.", "condition": {"operation": "ingest_document", "document_count_gt": 1000, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "large_ingest_review_required", "required_action": "record_ingest_review"}},
	{"name": "chunk_size_requires_minimum", "description": "Chunk size must be within configured bounds.", "condition": {"operation": "configure_chunking", "chunk_size_lt": 100}, "effect": {"decision": "deny", "reason": "chunk_size_too_small", "required_action": "increase_chunk_size"}},
	{"name": "chunk_size_requires_maximum", "description": "Chunk size must be within configured bounds.", "condition": {"operation": "configure_chunking", "chunk_size_gt": 8000}, "effect": {"decision": "deny", "reason": "chunk_size_too_large", "required_action": "reduce_chunk_size"}},
	{"name": "retrieval_requires_query", "description": "Retrieval requires a query.", "condition": {"operation": "retrieve_context", "query_present": False}, "effect": {"decision": "deny", "reason": "retrieval_query_required", "required_action": "attach_query"}},
	{"name": "retrieval_requires_knowledge_base", "description": "Retrieval requires a knowledge base.", "condition": {"operation": "retrieve_context", "knowledge_base_present": False}, "effect": {"decision": "deny", "reason": "knowledge_base_required", "required_action": "select_knowledge_base"}},
	{"name": "retrieval_result_window_requires_review", "description": "Large retrieval windows require review.", "condition": {"operation": "retrieve_context", "result_window_gt": 50, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "large_retrieval_window_review_required", "required_action": "record_retrieval_review"}},
	{"name": "restricted_sources_require_filter", "description": "Restricted sources require access filters.", "condition": {"source_classification": "restricted", "access_filter_applied": False}, "effect": {"decision": "deny", "reason": "access_filter_required", "required_action": "apply_access_filter"}},
	{"name": "low_context_confidence_requires_review", "description": "Low confidence context requires review.", "condition": {"context_confidence_lt": 0.7, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "low_context_confidence_review_required", "required_action": "record_context_review"}},
	{"name": "generation_requires_query", "description": "Answer generation requires a user query.", "condition": {"operation": "generate_answer", "query_present": False}, "effect": {"decision": "deny", "reason": "generation_query_required", "required_action": "attach_query"}},
	{"name": "generation_requires_context", "description": "Answer generation requires retrieved context.", "condition": {"operation": "generate_answer", "context_present": False}, "effect": {"decision": "deny", "reason": "retrieval_context_required", "required_action": "retrieve_context"}},
	{"name": "generation_requires_answer_text", "description": "Answer generation requires non-empty answer text.", "condition": {"operation": "generate_answer", "answer_text_present": False}, "effect": {"decision": "deny", "reason": "answer_text_required", "required_action": "attach_answer_text"}},
	{"name": "generation_requires_citations", "description": "Generated answers require citations.", "condition": {"operation": "generate_answer", "citations_attached": False}, "effect": {"decision": "deny", "reason": "citations_required", "required_action": "attach_citations"}},
	{"name": "external_model_requires_policy", "description": "External generation models require policy approval.", "condition": {"operation": "generate_answer", "model_location": "external", "model_policy_attached": False}, "effect": {"decision": "deny", "reason": "model_policy_required", "required_action": "attach_model_policy"}},
	{"name": "prompt_injection_requires_block", "description": "Detected prompt injection blocks generation.", "condition": {"operation": "generate_answer", "prompt_injection_detected": True}, "effect": {"decision": "deny", "reason": "prompt_injection_blocked", "required_action": "sanitize_or_reject_prompt"}},
	{"name": "unsafe_generation_requires_block", "description": "Unsafe answer classifications block generation.", "condition": {"operation": "generate_answer", "unsafe_answer_detected": True}, "effect": {"decision": "deny", "reason": "unsafe_generation_blocked", "required_action": "block_answer"}},
	{"name": "conversation_requires_id", "description": "Conversation turns require a conversation id.", "condition": {"operation": "record_turn", "conversation_id_present": False}, "effect": {"decision": "deny", "reason": "conversation_id_required", "required_action": "attach_conversation_id"}},
	{"name": "conversation_requires_user", "description": "Conversation turns require a user id.", "condition": {"operation": "record_turn", "user_id_present": False}, "effect": {"decision": "deny", "reason": "user_id_required", "required_action": "attach_user_id"}},
	{"name": "conversation_turn_limit_requires_review", "description": "Long conversations require review or summarization.", "condition": {"operation": "record_turn", "turn_count_gt": 100, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "conversation_turn_review_required", "required_action": "summarize_or_review_conversation"}},
	{"name": "citation_requires_source", "description": "Citations require source ids.", "condition": {"operation": "attach_citation", "source_id_present": False}, "effect": {"decision": "deny", "reason": "citation_source_required", "required_action": "attach_source_id"}},
	{"name": "citation_requires_document", "description": "Citations require document ids.", "condition": {"operation": "attach_citation", "document_id_present": False}, "effect": {"decision": "deny", "reason": "citation_document_required", "required_action": "attach_document_id"}},
	{"name": "citation_requires_chunk", "description": "Citations require chunk ids.", "condition": {"operation": "attach_citation", "chunk_id_present": False}, "effect": {"decision": "deny", "reason": "citation_chunk_required", "required_action": "attach_chunk_id"}},
	{"name": "curation_requires_curator", "description": "Curation requires a curator.", "condition": {"operation": "curate_answer", "curator_present": False}, "effect": {"decision": "deny", "reason": "curator_required", "required_action": "assign_curator"}},
	{"name": "curation_requires_decision", "description": "Curation requires a decision.", "condition": {"operation": "curate_answer", "curation_decision_present": False}, "effect": {"decision": "deny", "reason": "curation_decision_required", "required_action": "choose_curation_decision"}},
	{"name": "curation_requires_evidence", "description": "Curation requires evidence.", "condition": {"operation": "curate_answer", "evidence_present": False}, "effect": {"decision": "deny", "reason": "curation_evidence_required", "required_action": "attach_curation_evidence"}},
	{"name": "batch_rag_mutation_requires_bytewax", "description": "Batch RAG mutations must use Bytewax event streams.", "condition": {"operation": "batch_rag_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
	{"name": "cross_tenant_rag_access_denied", "description": "RAG operations may not cross tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_rag_access_denied", "required_action": "use_tenant_local_context"}},
	{"name": "rag_state_change_requires_audit", "description": "RAG state changes require audit events.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "audit_event_required", "required_action": "record_audit_event"}},
]


UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/ragn/dashboard", "component": "RAGNDashboard", "permission": "ragn:view", "nav_group": "Overview"},
	{"name": "studio", "path": "/ragn/studio", "component": "RAGStudio", "permission": "ragn:query", "nav_group": "Ask"},
	{"name": "knowledge_bases", "path": "/ragn/knowledge-bases", "component": "KnowledgeBaseManager", "permission": "ragn:manage_kb", "nav_group": "Knowledge"},
	{"name": "documents", "path": "/ragn/documents", "component": "DocumentIngestion", "permission": "ragn:manage_kb", "nav_group": "Knowledge"},
	{"name": "retrieval", "path": "/ragn/retrieval", "component": "RetrievalWorkbench", "permission": "ragn:query", "nav_group": "Ask"},
	{"name": "generation", "path": "/ragn/generation", "component": "GenerationWorkbench", "permission": "ragn:query", "nav_group": "Ask"},
	{"name": "conversations", "path": "/ragn/conversations", "component": "ConversationMemory", "permission": "ragn:query", "nav_group": "Ask"},
	{"name": "citations", "path": "/ragn/citations", "component": "CitationInspector", "permission": "ragn:view", "nav_group": "Evidence"},
	{"name": "curation", "path": "/ragn/curation", "component": "KnowledgeCuration", "permission": "ragn:curate", "nav_group": "Governance"},
	{"name": "governance", "path": "/ragn/governance", "component": "RAGNGovernance", "permission": "ragn:govern", "nav_group": "Governance"},
	{"name": "audit", "path": "/ragn/audit", "component": "RAGNAuditTimeline", "permission": "ragn:audit", "nav_group": "Governance"},
	{"name": "settings", "path": "/ragn/settings", "component": "RAGNSettings", "permission": "ragn:admin", "nav_group": "Administration"},
]


THEME: dict[str, Any] = {
	"name": "ragn_answer_studio",
	"tokens": {
		"color.primary": "#324A5F",
		"color.accent": "#F2A541",
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
		"answer_panel": {"icon": "message-square-text", "status_indicator": "citation-pill", "risk_style": "grounding-band"},
		"source_stack": {"visual": "evidence-list", "highlight": "source-chip"},
		"knowledge_base_card": {"visual": "collection-summary", "status_style": "classification-chip"},
		"document_panel": {"visual": "document-list", "status_style": "ingest-chip"},
		"retrieval_debug": {"visual": "ranked-results", "threshold_style": "confidence-band"},
		"generation_panel": {"visual": "answer-draft", "status_style": "policy-chip"},
		"conversation_trace": {"visual": "turn-timeline", "status_style": "memory-chip"},
		"citation_stack": {"visual": "citation-list", "status_style": "source-chip"},
		"curation_queue": {"visual": "review-list", "threshold_style": "quality-band"},
		"audit_timeline": {"visual": "event-timeline", "status_style": "evidence-chip"},
	},
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "ragn",
		"display_name": "Retrieval-Augmented Generation",
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/ragn/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True,
		},
		"theme": deepcopy(THEME),
	}


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
		if key.endswith("_lte"):
			actual = context.get(key[:-4])
			if not isinstance(actual, Number) or not actual <= expected:
				return False
		elif key.endswith("_lt"):
			actual = context.get(key[:-3])
			if not isinstance(actual, Number) or not actual < expected:
				return False
		elif key.endswith("_gte"):
			actual = context.get(key[:-4])
			if not isinstance(actual, Number) or not actual >= expected:
				return False
		elif key.endswith("_gt"):
			actual = context.get(key[:-3])
			if not isinstance(actual, Number) or not actual > expected:
				return False
		elif key.endswith("_ne"):
			if context.get(key[:-3]) == expected:
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
