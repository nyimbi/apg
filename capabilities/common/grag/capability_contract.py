"""Executable capability contract for APG Graph-based RAG."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any


SUPPORTED_GRAG_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_GRAG_AGENT_ROLES = [
	"graph_retrieval_reviewer",
	"vector_retrieval_reviewer",
	"fusion_reviewer",
	"reasoning_path_reviewer",
	"provenance_reviewer",
	"grounded_generation_reviewer",
	"citation_reviewer",
	"safety_reviewer",
	"lifecycle_batch_reviewer",
	"graphrag_steward",
]
PRIVILEGED_GRAG_AGENT_ROLES = [
	"fusion_reviewer",
	"reasoning_path_reviewer",
	"provenance_reviewer",
	"grounded_generation_reviewer",
	"safety_reviewer",
	"lifecycle_batch_reviewer",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"graph_sources": {
		"id_required": True,
		"name_required": True,
		"owner_required": True,
		"registered_graph_required": True,
		"provenance_required": True,
		"default_classification": "internal",
		"retire_requires_review": True,
	},
	"vector_sources": {
		"id_required": True,
		"index_required": True,
		"embedding_model_required": True,
		"source_documents_required": True,
		"default_embedding_model": "text-embedding-3-large",
		"minimum_documents": 1,
	},
	"hybrid_retrieval": {
		"enabled": True,
		"vector_index_required": True,
		"graph_index_required": True,
		"minimum_retrieval_confidence": 0.7,
		"max_result_window": 50,
		"large_result_window_review_threshold": 50,
		"restricted_source_filter_required": True,
	},
	"reasoning": {
		"max_hops": 4,
		"minimum_hops": 1,
		"evidence_path_required": True,
		"start_node_required": True,
		"explanation_required": True,
		"multi_hop_review_threshold": 4,
	},
	"generation": {
		"citations_required": True,
		"provenance_required": True,
		"reasoning_path_required": True,
		"model_policy_required": True,
		"external_model_requires_policy": True,
		"minimum_answer_confidence": 0.75,
		"unsafe_generation_blocking": True,
	},
	"provenance": {
		"source_refs_required": True,
		"path_refs_required": True,
		"citation_refs_required": True,
		"lineage_export_enabled": True,
	},
	"curation": {
		"review_required_for_low_confidence": True,
		"curator_required": True,
		"evidence_required": True,
		"allowed_decisions": ["approved", "rejected", "needs_revision"],
		"publication_requires_approval": True,
	},
	"security": {
		"cross_tenant_access_allowed": False,
		"restricted_source_filter_required": True,
		"unsafe_generation_blocking": True,
		"tenant_isolation_required": True,
	},
	"agents": {
		"first_class": True,
		"supported_runtimes": SUPPORTED_GRAG_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_GRAG_AGENT_ROLES,
		"privileged_roles": PRIVILEGED_GRAG_AGENT_ROLES,
		"require_owner": True,
		"require_purpose": True,
		"require_scope": True,
		"require_contribution_disclosure": True,
		"require_human_approval_for_privileged_roles": True,
		"adapter_contract": "aicr_provider_neutral_graphrag_agent_adapter",
	},
	"streaming": {
		"engine": "bytewax",
		"lifecycle_stream": "grag.lifecycle",
		"watermark": "event_time",
		"required_processor": "bytewax",
		"required_operations": [
			"graph_source_batch",
			"vector_source_batch",
			"hybrid_query_batch",
			"reasoning_path_batch",
			"provenance_batch",
			"generation_batch",
			"curation_batch",
			"publication_batch",
			"graphrag_agent_batch",
		],
		"topics": [
			"grag.graph_sources",
			"grag.vector_sources",
			"grag.hybrid_queries",
			"grag.reasoning_paths",
			"grag.provenance",
			"grag.generations",
			"grag.curations",
			"grag.publications",
			"grag.agents",
		],
		"broker_core_dependency_allowed": False,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_graph_sources": True,
		"audit_vector_sources": True,
		"audit_queries": True,
		"audit_reasoning": True,
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
		"generated_app_runtime": "grag_runtime.GragService",
		"production_runtime": "service.GraphRAGService",
		"helper_runtime": "grag_runtime.py",
		"http_api": "api.py",
		"event_stream": "bytewax",
		"rag": "ragn",
		"knowledge_graph": "kngr",
		"graph": "grph",
		"search": "srch",
		"nlp": "nlpc",
		"ai_core": "aicr",
		"ontology": "onto",
		"metadata": "meta",
		"auth_provider": "auth",
		"audit_sink": "audl",
		"cache": "cach",
		"metrics_sink": "moni",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_query_console": True,
		"enable_graph_sources": True,
		"enable_vector_sources": True,
		"enable_hybrid_retrieval": True,
		"enable_reasoning_paths": True,
		"enable_provenance": True,
		"enable_generation": True,
		"enable_curation": True,
		"enable_governance": True,
		"enable_graphrag_agent_roster": True,
		"enable_lifecycle_batch_monitor": True,
		"enable_audit": True,
		"enable_settings": True,
	},
	"theme": {"default_theme": "grag_reasoning_console", "allow_tenant_overrides": True},
}


CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"graph_sources",
		"vector_sources",
		"hybrid_retrieval",
		"reasoning",
		"generation",
		"provenance",
		"curation",
		"security",
		"agents",
		"streaming",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {key: {"type": "object"} for key in [
		"graph_sources",
		"vector_sources",
		"hybrid_retrieval",
		"reasoning",
		"generation",
		"provenance",
		"curation",
		"security",
		"agents",
		"streaming",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]} | {"tenant_id": {"type": "string", "minLength": 1}},
}


RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All GraphRAG operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "graph_source_requires_id", "description": "Graph sources require stable identifiers.", "condition": {"operation": "register_graph_source", "graph_source_id_present": False}, "effect": {"decision": "deny", "reason": "graph_source_id_required", "required_action": "attach_graph_source_id"}},
	{"name": "graph_source_requires_name", "description": "Graph sources require names.", "condition": {"operation": "register_graph_source", "graph_source_name_present": False}, "effect": {"decision": "deny", "reason": "graph_source_name_required", "required_action": "attach_graph_source_name"}},
	{"name": "graph_source_requires_owner", "description": "Graph sources require accountable owners.", "condition": {"operation": "register_graph_source", "owner_assigned": False}, "effect": {"decision": "deny", "reason": "graph_source_owner_required", "required_action": "assign_owner"}},
	{"name": "graph_source_requires_registered_graph", "description": "Graph sources must reference a registered graph.", "condition": {"operation": "register_graph_source", "registered_graph_present": False}, "effect": {"decision": "deny", "reason": "registered_graph_required", "required_action": "attach_graph_id"}},
	{"name": "graph_source_requires_provenance", "description": "Graph sources require provenance references.", "condition": {"operation": "register_graph_source", "provenance_attached": False}, "effect": {"decision": "deny", "reason": "graph_source_provenance_required", "required_action": "attach_provenance"}},
	{"name": "graph_source_retire_requires_review", "description": "Retiring graph sources requires review.", "condition": {"operation": "retire_graph_source", "review_recorded": False}, "effect": {"decision": "require_review", "reason": "graph_source_retire_review_required", "required_action": "record_graph_source_review"}},
	{"name": "vector_source_requires_id", "description": "Vector sources require stable identifiers.", "condition": {"operation": "register_vector_source", "vector_source_id_present": False}, "effect": {"decision": "deny", "reason": "vector_source_id_required", "required_action": "attach_vector_source_id"}},
	{"name": "vector_source_requires_index", "description": "Vector sources require an index id.", "condition": {"operation": "register_vector_source", "vector_index_present": False}, "effect": {"decision": "deny", "reason": "vector_index_required", "required_action": "attach_vector_index"}},
	{"name": "vector_source_requires_embedding_model", "description": "Vector sources require an embedding model.", "condition": {"operation": "register_vector_source", "embedding_model_present": False}, "effect": {"decision": "deny", "reason": "embedding_model_required", "required_action": "attach_embedding_model"}},
	{"name": "vector_source_requires_source_documents", "description": "Vector sources require source document references.", "condition": {"operation": "register_vector_source", "source_documents_present": False}, "effect": {"decision": "deny", "reason": "source_documents_required", "required_action": "attach_source_documents"}},
	{"name": "hybrid_query_requires_query", "description": "Hybrid retrieval requires a query.", "condition": {"operation": "hybrid_query", "query_present": False}, "effect": {"decision": "deny", "reason": "hybrid_query_required", "required_action": "attach_query"}},
	{"name": "hybrid_query_requires_graph_source", "description": "Hybrid retrieval requires a graph source.", "condition": {"operation": "hybrid_query", "graph_source_present": False}, "effect": {"decision": "deny", "reason": "graph_source_required", "required_action": "select_graph_source"}},
	{"name": "hybrid_query_requires_vector_source", "description": "Hybrid retrieval requires a vector source.", "condition": {"operation": "hybrid_query", "vector_source_present": False}, "effect": {"decision": "deny", "reason": "vector_source_required", "required_action": "select_vector_source"}},
	{"name": "hybrid_query_requires_vector_index", "description": "Hybrid retrieval requires vector index readiness.", "condition": {"operation": "hybrid_query", "vector_index_ready": False}, "effect": {"decision": "deny", "reason": "vector_index_required", "required_action": "build_vector_index"}},
	{"name": "hybrid_query_requires_graph_index", "description": "Hybrid retrieval requires graph index readiness.", "condition": {"operation": "hybrid_query", "graph_index_ready": False}, "effect": {"decision": "deny", "reason": "graph_index_required", "required_action": "build_graph_index"}},
	{"name": "hybrid_query_requires_access_filter_for_restricted", "description": "Restricted graph/vector sources require access filters.", "condition": {"source_classification": "restricted", "access_filter_applied": False}, "effect": {"decision": "deny", "reason": "access_filter_required", "required_action": "apply_access_filter"}},
	{"name": "hybrid_query_requires_result_window_review", "description": "Large hybrid result windows require review.", "condition": {"operation": "hybrid_query", "result_window_gt": 50, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "large_result_window_review_required", "required_action": "record_retrieval_review"}},
	{"name": "low_retrieval_confidence_requires_review", "description": "Low confidence hybrid retrieval requires review.", "condition": {"operation": "hybrid_query", "retrieval_confidence_lt": 0.7, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "low_retrieval_confidence_review_required", "required_action": "record_retrieval_review"}},
	{"name": "reasoning_requires_query", "description": "Reasoning paths require a recorded hybrid query.", "condition": {"operation": "build_reasoning_path", "query_present": False}, "effect": {"decision": "deny", "reason": "reasoning_query_required", "required_action": "select_hybrid_query"}},
	{"name": "reasoning_requires_start_node", "description": "Reasoning paths require a graph start node.", "condition": {"operation": "build_reasoning_path", "start_node_present": False}, "effect": {"decision": "deny", "reason": "start_node_required", "required_action": "attach_start_node"}},
	{"name": "reasoning_requires_evidence_path", "description": "Graph reasoning requires evidence paths.", "condition": {"operation": "build_reasoning_path", "evidence_path_present": False}, "effect": {"decision": "deny", "reason": "evidence_path_required", "required_action": "attach_evidence_path"}},
	{"name": "reasoning_requires_positive_hops", "description": "Reasoning paths require at least one hop.", "condition": {"operation": "build_reasoning_path", "hop_count_lte": 0}, "effect": {"decision": "deny", "reason": "positive_hop_count_required", "required_action": "increase_hop_count"}},
	{"name": "multi_hop_requires_review", "description": "Deep multi-hop reasoning requires review.", "condition": {"operation": "build_reasoning_path", "hop_count_gt": 4, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "multi_hop_review_required", "required_action": "record_reasoning_review"}},
	{"name": "reasoning_requires_explanation", "description": "Reasoning paths require a human-readable explanation.", "condition": {"operation": "build_reasoning_path", "explanation_present": False}, "effect": {"decision": "deny", "reason": "reasoning_explanation_required", "required_action": "attach_explanation"}},
	{"name": "generation_requires_query", "description": "Graph-grounded generation requires a user query.", "condition": {"operation": "generate_answer", "query_present": False}, "effect": {"decision": "deny", "reason": "generation_query_required", "required_action": "attach_query"}},
	{"name": "generation_requires_retrieval_context", "description": "Graph-grounded generation requires hybrid retrieval context.", "condition": {"operation": "generate_answer", "retrieval_context_present": False}, "effect": {"decision": "deny", "reason": "retrieval_context_required", "required_action": "run_hybrid_query"}},
	{"name": "generation_requires_reasoning_path", "description": "Graph-grounded generation requires a reasoning path.", "condition": {"operation": "generate_answer", "reasoning_path_present": False}, "effect": {"decision": "deny", "reason": "reasoning_path_required", "required_action": "build_reasoning_path"}},
	{"name": "generation_requires_answer_text", "description": "Graph-grounded generation requires answer text.", "condition": {"operation": "generate_answer", "answer_text_present": False}, "effect": {"decision": "deny", "reason": "answer_text_required", "required_action": "attach_answer_text"}},
	{"name": "answer_requires_provenance", "description": "Graph-grounded answers require provenance.", "condition": {"operation": "generate_answer", "provenance_attached": False}, "effect": {"decision": "deny", "reason": "provenance_required", "required_action": "attach_provenance"}},
	{"name": "generation_requires_citations", "description": "Graph-grounded answers require citations.", "condition": {"operation": "generate_answer", "citations_attached": False}, "effect": {"decision": "deny", "reason": "citations_required", "required_action": "attach_citations"}},
	{"name": "external_model_requires_policy", "description": "External generation models require policy approval.", "condition": {"operation": "generate_answer", "model_location": "external", "model_policy_attached": False}, "effect": {"decision": "deny", "reason": "model_policy_required", "required_action": "attach_model_policy"}},
	{"name": "unsafe_generation_requires_block", "description": "Unsafe answer classifications block generation.", "condition": {"operation": "generate_answer", "unsafe_answer_detected": True}, "effect": {"decision": "deny", "reason": "unsafe_generation_blocked", "required_action": "block_answer"}},
	{"name": "low_answer_confidence_requires_review", "description": "Low confidence graph-grounded answers require review.", "condition": {"operation": "generate_answer", "answer_confidence_lt": 0.75, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "low_answer_confidence_review_required", "required_action": "record_answer_review"}},
	{"name": "curation_requires_curator", "description": "Curation requires a curator.", "condition": {"operation": "curate_answer", "curator_present": False}, "effect": {"decision": "deny", "reason": "curator_required", "required_action": "assign_curator"}},
	{"name": "curation_requires_decision", "description": "Curation requires an allowed decision.", "condition": {"operation": "curate_answer", "curation_decision_present": False}, "effect": {"decision": "deny", "reason": "curation_decision_required", "required_action": "choose_curation_decision"}},
	{"name": "curation_requires_evidence", "description": "Curation requires evidence.", "condition": {"operation": "curate_answer", "evidence_present": False}, "effect": {"decision": "deny", "reason": "curation_evidence_required", "required_action": "attach_curation_evidence"}},
	{"name": "publication_requires_curated_answer", "description": "Published graph-grounded answers require approval.", "condition": {"operation": "publish_answer", "curated_answer_present": False}, "effect": {"decision": "deny", "reason": "curated_answer_required", "required_action": "approve_answer"}},
	{"name": "batch_grag_mutation_requires_bytewax", "description": "Batch GraphRAG mutations must use Bytewax event streams.", "condition": {"operation": "batch_grag_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
	{"name": "cross_tenant_graph_access_denied", "description": "GraphRAG operations may not cross tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_graph_access_denied", "required_action": "use_tenant_local_context"}},
	{"name": "graph_state_change_requires_audit", "description": "GraphRAG state changes require audit events.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "audit_event_required", "required_action": "record_audit_event"}},
	{"name": "graphrag_agent_runtime_supported", "description": "GraphRAG agents must use supported provider-neutral runtimes.", "condition": {"operation": "register_grag_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_graphrag_agent_runtime", "required_action": "choose_supported_graphrag_agent_runtime"}},
	{"name": "graphrag_agent_role_supported", "description": "GraphRAG agents must use supported graph-RAG roles.", "condition": {"operation": "register_grag_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_graphrag_agent_role", "required_action": "choose_supported_graphrag_agent_role"}},
	{"name": "graphrag_agent_requires_scope", "description": "GraphRAG agents require an explicit bounded graph, retrieval, reasoning, or answer scope.", "condition": {"operation": "register_grag_agent", "scope_present": False}, "effect": {"decision": "deny", "reason": "graphrag_agent_scope_required", "required_action": "declare_graphrag_agent_scope"}},
	{"name": "graphrag_agent_requires_owner", "description": "GraphRAG agents require an accountable owner.", "condition": {"operation": "register_grag_agent", "owner_present": False}, "effect": {"decision": "deny", "reason": "graphrag_agent_owner_required", "required_action": "assign_graphrag_agent_owner"}},
	{"name": "graphrag_agent_requires_purpose", "description": "GraphRAG agents require a documented purpose.", "condition": {"operation": "register_grag_agent", "purpose_present": False}, "effect": {"decision": "deny", "reason": "graphrag_agent_purpose_required", "required_action": "document_graphrag_agent_purpose"}},
	{"name": "graphrag_agent_requires_contribution_disclosure", "description": "GraphRAG agents must disclose machine-authored retrieval, reasoning, and answer-review contributions.", "condition": {"operation": "register_grag_agent", "contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "graphrag_agent_contribution_disclosure_required", "required_action": "disclose_machine_contribution"}},
	{"name": "graphrag_agent_privileged_role_requires_human_approval", "description": "Privileged GraphRAG-agent roles require human approval evidence.", "condition": {"operation": "register_grag_agent", "privileged_role": True, "human_approval_required": False}, "effect": {"decision": "require_review", "reason": "graphrag_agent_human_approval_required", "required_action": "record_human_graphrag_agent_approval"}},
	{"name": "bytewax_grag_stream_required", "description": "GRAG lifecycle batches must be routed through Bytewax.", "condition": {"operation": "validate_grag_lifecycle_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_lifecycle_stream_required", "required_action": "route_grag_lifecycle_batch_to_bytewax"}},
]


UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/grag/dashboard", "component": "GRAGDashboard", "permission": "grag:view", "nav_group": "Overview"},
	{"name": "query", "path": "/grag/query", "component": "GraphRAGQueryConsole", "permission": "grag:query", "nav_group": "Ask"},
	{"name": "graph_sources", "path": "/grag/graph-sources", "component": "GraphSourceManager", "permission": "grag:manage_graphs", "nav_group": "Sources"},
	{"name": "vector_sources", "path": "/grag/vector-sources", "component": "VectorSourceManager", "permission": "grag:manage_sources", "nav_group": "Sources"},
	{"name": "hybrid_retrieval", "path": "/grag/hybrid-retrieval", "component": "HybridRetrievalWorkbench", "permission": "grag:query", "nav_group": "Ask"},
	{"name": "reasoning", "path": "/grag/reasoning", "component": "ReasoningPathExplorer", "permission": "grag:reason", "nav_group": "Reasoning"},
	{"name": "provenance", "path": "/grag/provenance", "component": "ProvenanceInspector", "permission": "grag:view", "nav_group": "Evidence"},
	{"name": "generation", "path": "/grag/generation", "component": "GroundedGenerationWorkbench", "permission": "grag:generate", "nav_group": "Ask"},
	{"name": "curation", "path": "/grag/curation", "component": "GraphAnswerCuration", "permission": "grag:curate", "nav_group": "Governance"},
	{"name": "governance", "path": "/grag/governance", "component": "GRAGGovernance", "permission": "grag:govern", "nav_group": "Governance"},
	{"name": "agents", "path": "/grag/agents", "component": "GraphRAGAgentRoster", "permission": "grag:govern", "nav_group": "Governance"},
	{"name": "lifecycle", "path": "/grag/lifecycle", "component": "GRAGLifecycleBatchMonitor", "permission": "grag:admin", "nav_group": "Operations"},
	{"name": "audit", "path": "/grag/audit", "component": "GRAGAuditTimeline", "permission": "grag:audit", "nav_group": "Governance"},
	{"name": "settings", "path": "/grag/settings", "component": "GRAGSettings", "permission": "grag:admin", "nav_group": "Administration"},
]


THEME: dict[str, Any] = {
	"name": "grag_reasoning_console",
	"tokens": {
		"color.primary": "#2F4858",
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
		"hybrid_result": {"icon": "network", "status_indicator": "fusion-pill", "risk_style": "evidence-band"},
		"graph_source_card": {"visual": "graph-summary", "status_style": "lineage-chip"},
		"vector_index_card": {"visual": "index-summary", "status_style": "embedding-chip"},
		"reasoning_path": {"visual": "multi-hop-path", "highlight": "hop-chip"},
		"provenance_panel": {"visual": "source-graph", "status_style": "confidence-pill"},
		"generation_panel": {"visual": "answer-draft", "status_style": "grounding-chip"},
		"curation_queue": {"visual": "review-list", "threshold_style": "quality-band"},
		"graphrag_agent_roster": {"icon": "bot", "status_indicator": "agent-approval-chip", "risk_style": "reasoning-scope-band"},
		"bytewax_lifecycle_panel": {"icon": "git-branch", "status_indicator": "stream-chip", "risk_style": "processor-band"},
		"audit_timeline": {"visual": "event-timeline", "status_style": "evidence-chip"},
		"query_console": {"visual": "ranked-context", "status_style": "retrieval-chip"},
	},
}


def agent_manifest() -> dict[str, Any]:
	"""Return first-class GRAG agent composition manifest."""
	return {
		"first_class": True,
		"supported_runtimes": list(SUPPORTED_GRAG_AGENT_RUNTIMES),
		"supported_roles": list(SUPPORTED_GRAG_AGENT_ROLES),
		"privileged_roles": list(PRIVILEGED_GRAG_AGENT_ROLES),
		"required_fields": ["tenant_id", "agent_id", "name", "runtime", "role", "scope", "owner", "purpose"],
		"guardrails": [
			"supported_runtime",
			"supported_role",
			"explicit_scope",
			"accountable_owner",
			"declared_purpose",
			"machine_contribution_disclosure",
			"human_approval_for_privileged_roles",
		],
		"adapter_contract": "aicr_provider_neutral_graphrag_agent_adapter",
	}


def streaming_manifest() -> dict[str, Any]:
	"""Return the GRAG Bytewax lifecycle stream contract."""
	return {
		"engine": "bytewax",
		"lifecycle_stream": "grag.lifecycle",
		"watermark": "event_time",
		"required_processor": "bytewax",
		"required_operations": [
			"graph_source_batch",
			"vector_source_batch",
			"hybrid_query_batch",
			"reasoning_path_batch",
			"provenance_batch",
			"generation_batch",
			"curation_batch",
			"publication_batch",
			"graphrag_agent_batch",
		],
		"topics": [
			"grag.graph_sources",
			"grag.vector_sources",
			"grag.hybrid_queries",
			"grag.reasoning_paths",
			"grag.provenance",
			"grag.generations",
			"grag.curations",
			"grag.publications",
			"grag.agents",
		],
		"broker_core_dependency_allowed": False,
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "grag",
		"display_name": "Graph-based RAG",
		"provides": ["graph_based_rag", "hybrid_graph_vector_retrieval", "graphrag_agent_composition"],
		"requires": ["ragn", "kngr", "grph", "srch", "nlpc", "aicr", "conf", "audl"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/grag/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True,
		},
		"agents": agent_manifest(),
		"streaming": streaming_manifest(),
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
