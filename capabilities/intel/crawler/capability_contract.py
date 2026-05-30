"""Executable capability contract for APG intelligence crawler."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


SUPPORTED_CRAWLER_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_CRAWLER_AGENT_ROLES = [
	"source_strategy_reviewer",
	"crawl_policy_reviewer",
	"extraction_quality_reviewer",
	"validation_reviewer",
	"rag_pipeline_reviewer",
	"risk_reviewer",
]
CRAWLER_EVENT_STREAM = "apg.intel.crawler.lifecycle"


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"sources": {
		"owner_required": True,
		"url_required": True,
		"allowed_domain_required": True,
		"robots_policy_review_required": True,
	},
	"crawl_jobs": {
		"source_required": True,
		"positive_rate_limit_required": True,
		"max_depth_limit": 8,
		"approval_required_for_high_risk": True,
	},
	"extraction": {
		"schema_required": True,
		"content_fingerprint_required": True,
		"minimum_quality_score": 0.7,
	},
	"datasets": {
		"lineage_required": True,
		"validation_required_before_publish": True,
		"pii_review_required": True,
	},
	"validation": {
		"reviewer_required": True,
		"minimum_confidence": 0.75,
		"disagreement_requires_review": True,
	},
	"rag": {"chunk_plan_required": True, "chunk_size_max": 4096, "embedding_model_required": True},
	"knowledge_graph": {"entity_schema_required": True, "relationship_evidence_required": True},
	"crawler_agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_CRAWLER_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_CRAWLER_AGENT_ROLES,
		"human_approval_required": True,
		"max_autonomous_scope": "recommend_validate_and_prepare",
	},
	"governance": {
		"require_tenant_context": True,
		"audit_state_changes": True,
		"policy_attached_for_writes": True,
		"respect_source_terms": True,
	},
	"observability": {
		"event_stream": CRAWLER_EVENT_STREAM,
		"stream_processor": "bytewax",
		"emit_source_events": True,
		"emit_crawl_events": True,
		"emit_dataset_events": True,
		"emit_validation_events": True,
		"emit_agent_events": True,
	},
	"adapters": {
		"authorization": "adapter",
		"audit": "adapter",
		"event_stream": "bytewax",
		"notification": "adapter",
		"storage": "adapter",
		"vector_index": "adapter",
		"theme": "adapter",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_sources": True,
		"enable_crawl_jobs": True,
		"enable_extractions": True,
		"enable_datasets": True,
		"enable_validation": True,
		"enable_rag": True,
		"enable_graph": True,
		"enable_agents": True,
		"enable_settings": True,
	},
	"theme": {"default_theme": "intel_crawler_control", "allow_tenant_overrides": True},
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"sources",
		"crawl_jobs",
		"extraction",
		"datasets",
		"validation",
		"rag",
		"knowledge_graph",
		"crawler_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {
		"tenant_id": {"type": "string", "minLength": 1},
		"sources": {"type": "object"},
		"crawl_jobs": {"type": "object"},
		"extraction": {"type": "object"},
		"datasets": {"type": "object"},
		"validation": {"type": "object"},
		"rag": {"type": "object"},
		"knowledge_graph": {"type": "object"},
		"crawler_agents": {"type": "object"},
		"governance": {"type": "object"},
		"observability": {"type": "object"},
		"adapters": {"type": "object"},
		"ui": {"type": "object"},
		"theme": {"type": "object"},
	},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "Crawler operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "crawler_write_requires_policy", "description": "Crawler writes require policy attachment.", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "operation_policy_required", "required_action": "attach_operation_policy"}},
	{"name": "source_requires_owner", "description": "Sources require an accountable owner.", "condition": {"operation": "register_source", "source_owner_assigned": False}, "effect": {"decision": "deny", "reason": "source_owner_required", "required_action": "assign_source_owner"}},
	{"name": "source_requires_url", "description": "Sources require at least one URL or feed endpoint.", "condition": {"operation": "register_source", "source_url_present": False}, "effect": {"decision": "deny", "reason": "source_url_required", "required_action": "attach_source_url"}},
	{"name": "source_requires_allowed_domain", "description": "Sources require explicit allowed domains.", "condition": {"operation": "register_source", "allowed_domain_present": False}, "effect": {"decision": "deny", "reason": "allowed_domain_required", "required_action": "set_allowed_domain"}},
	{"name": "source_requires_policy_review", "description": "Sources require crawl policy review before activation.", "condition": {"operation": "register_source", "policy_review_recorded": False}, "effect": {"decision": "require_review", "reason": "crawl_policy_review_required", "required_action": "record_source_policy_review"}},
	{"name": "crawl_job_requires_source", "description": "Crawl jobs require a registered source.", "condition": {"operation": "create_crawl_job", "source_present": False}, "effect": {"decision": "deny", "reason": "crawl_source_required", "required_action": "attach_source"}},
	{"name": "crawl_job_requires_cadence", "description": "Crawl jobs require a cadence.", "condition": {"operation": "create_crawl_job", "cadence_present": False}, "effect": {"decision": "deny", "reason": "crawl_cadence_required", "required_action": "set_cadence"}},
	{"name": "crawl_rate_limit_positive", "description": "Crawl rate limits must be positive.", "condition": {"operation": "create_crawl_job", "rate_limit_per_minute_lte": 0}, "effect": {"decision": "deny", "reason": "crawl_rate_limit_must_be_positive", "required_action": "set_positive_rate_limit"}},
	{"name": "crawl_depth_within_limit", "description": "Crawl depth must stay within the configured limit.", "condition": {"operation": "create_crawl_job", "max_depth_gt": 8}, "effect": {"decision": "deny", "reason": "crawl_depth_limit_exceeded", "required_action": "reduce_crawl_depth"}},
	{"name": "high_risk_crawl_requires_approval", "description": "High-risk crawl jobs require approval.", "condition": {"operation": "create_crawl_job", "high_risk": True, "approved": False}, "effect": {"decision": "require_review", "reason": "high_risk_crawl_approval_required", "required_action": "record_crawl_approval"}},
	{"name": "extraction_requires_schema", "description": "Extraction batches require a schema.", "condition": {"operation": "record_extraction", "schema_present": False}, "effect": {"decision": "deny", "reason": "extraction_schema_required", "required_action": "attach_extraction_schema"}},
	{"name": "extraction_requires_fingerprint", "description": "Extraction batches require content fingerprinting.", "condition": {"operation": "record_extraction", "fingerprint_present": False}, "effect": {"decision": "deny", "reason": "content_fingerprint_required", "required_action": "attach_content_fingerprint"}},
	{"name": "extraction_quality_minimum", "description": "Extraction quality must meet the configured minimum.", "condition": {"operation": "record_extraction", "quality_score_lt": 0.7}, "effect": {"decision": "require_review", "reason": "extraction_quality_review_required", "required_action": "review_extraction_quality"}},
	{"name": "dataset_requires_lineage", "description": "Datasets require source and job lineage.", "condition": {"operation": "publish_dataset", "lineage_present": False}, "effect": {"decision": "deny", "reason": "dataset_lineage_required", "required_action": "attach_dataset_lineage"}},
	{"name": "dataset_requires_validation", "description": "Datasets require validation before publication.", "condition": {"operation": "publish_dataset", "validation_recorded": False}, "effect": {"decision": "deny", "reason": "dataset_validation_required", "required_action": "record_dataset_validation"}},
	{"name": "pii_dataset_requires_review", "description": "Datasets containing PII require privacy review.", "condition": {"operation": "publish_dataset", "contains_pii": True, "privacy_review_recorded": False}, "effect": {"decision": "require_review", "reason": "privacy_review_required", "required_action": "record_privacy_review"}},
	{"name": "validation_requires_reviewer", "description": "Validation sessions require a reviewer.", "condition": {"operation": "open_validation_session", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "validation_reviewer_required", "required_action": "assign_reviewer"}},
	{"name": "validation_confidence_minimum", "description": "Validation confidence must meet the configured minimum.", "condition": {"operation": "complete_validation_session", "confidence_lt": 0.75}, "effect": {"decision": "require_review", "reason": "validation_confidence_review_required", "required_action": "review_validation_confidence"}},
	{"name": "rag_requires_chunk_plan", "description": "RAG preparation requires a chunk plan.", "condition": {"operation": "record_rag_plan", "chunk_plan_present": False}, "effect": {"decision": "deny", "reason": "rag_chunk_plan_required", "required_action": "attach_chunk_plan"}},
	{"name": "rag_chunk_size_within_limit", "description": "RAG chunk size must stay within the configured limit.", "condition": {"operation": "record_rag_plan", "chunk_size_gt": 4096}, "effect": {"decision": "deny", "reason": "rag_chunk_size_limit_exceeded", "required_action": "reduce_chunk_size"}},
	{"name": "rag_requires_embedding_model", "description": "RAG preparation requires an embedding model.", "condition": {"operation": "record_rag_plan", "embedding_model_present": False}, "effect": {"decision": "deny", "reason": "embedding_model_required", "required_action": "select_embedding_model"}},
	{"name": "graph_requires_entity_schema", "description": "Knowledge graph projection requires an entity schema.", "condition": {"operation": "record_graph_projection", "entity_schema_present": False}, "effect": {"decision": "deny", "reason": "entity_schema_required", "required_action": "attach_entity_schema"}},
	{"name": "graph_requires_relationship_evidence", "description": "Knowledge graph relationships require evidence.", "condition": {"operation": "record_graph_projection", "relationship_evidence_present": False}, "effect": {"decision": "deny", "reason": "relationship_evidence_required", "required_action": "attach_relationship_evidence"}},
	{"name": "crawler_batch_requires_bytewax", "description": "Crawler batches require Bytewax coordination.", "condition": {"operation": "crawler_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_crawler_batch_to_bytewax"}},
	{"name": "crawler_event_requires_bytewax", "description": "Crawler lifecycle events require Bytewax.", "condition": {"operation": "crawler_event", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_crawler_event_to_bytewax"}},
	{"name": "crawler_agent_runtime_supported", "description": "Crawler agents must use an approved runtime.", "condition": {"operation": "register_crawler_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "crawler_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"}},
	{"name": "crawler_agent_role_supported", "description": "Crawler agents must use an approved role.", "condition": {"operation": "register_crawler_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "crawler_agent_role_not_supported", "required_action": "select_supported_agent_role"}},
	{"name": "privileged_agent_crawler_action_requires_human_approval", "description": "Privileged crawler actions proposed by agents require human approval.", "condition": {"operation": "agent_crawler_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/intel-crawler/dashboard", "component": "CrawlerDashboard", "permission": "intel_crawler:view", "nav_group": "Overview"},
	{"name": "sources", "path": "/intel-crawler/sources", "component": "CrawlerSourceRegistry", "permission": "intel_crawler:manage_sources", "nav_group": "Sources"},
	{"name": "crawl_jobs", "path": "/intel-crawler/crawl-jobs", "component": "CrawlerJobConsole", "permission": "intel_crawler:operate", "nav_group": "Crawling"},
	{"name": "extractions", "path": "/intel-crawler/extractions", "component": "ExtractionWorkbench", "permission": "intel_crawler:extract", "nav_group": "Processing"},
	{"name": "datasets", "path": "/intel-crawler/datasets", "component": "DatasetPublicationConsole", "permission": "intel_crawler:publish", "nav_group": "Processing"},
	{"name": "validation", "path": "/intel-crawler/validation", "component": "ValidationWorkbench", "permission": "intel_crawler:validate", "nav_group": "Quality"},
	{"name": "rag", "path": "/intel-crawler/rag", "component": "RAGPreparationWorkbench", "permission": "intel_crawler:rag", "nav_group": "Knowledge"},
	{"name": "graph", "path": "/intel-crawler/graph", "component": "KnowledgeGraphProjection", "permission": "intel_crawler:graph", "nav_group": "Knowledge"},
	{"name": "agents", "path": "/intel-crawler/agents", "component": "CrawlerAgentWorkbench", "permission": "intel_crawler:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/intel-crawler/settings", "component": "CrawlerSettings", "permission": "intel_crawler:admin", "nav_group": "Administration"},
]

THEME: dict[str, Any] = {
	"name": "intel_crawler_control",
	"tokens": {"color.primary": "#28536B", "color.accent": "#C44536", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {
		"sources": {"icon": "radar", "status_indicator": "source-pill", "risk_style": "policy-band"},
		"crawl_jobs": {"visual": "job-queue", "status_style": "crawl-chip"},
		"extractions": {"visual": "schema-grid", "status_style": "quality-chip"},
		"datasets": {"visual": "lineage-table", "status_style": "publish-chip"},
		"validation": {"visual": "review-lane", "status_style": "confidence-chip"},
		"rag": {"visual": "chunk-map", "status_style": "embedding-chip"},
		"graph": {"visual": "entity-network", "status_style": "evidence-chip"},
		"agent_workbench": {"visual": "approval-lane", "status_style": "agent-chip"},
	},
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "intel_crawler",
		"display_name": "Intelligence Crawler",
		"provides": [
			"source_intelligence_registry",
			"crawl_job_lifecycle",
			"extraction_pipeline",
			"dataset_quality_control",
			"validation_workflow",
			"rag_graphrag_preparation",
			"crawler_agents",
		],
		"requires": ["auth", "audl", "ntfy", "composition_events", "composition_config", "document_processing"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {"shell": "apg_python", "view_module": "views.py", "api_prefix": "/intel-crawler/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True},
		"theme": deepcopy(THEME),
		"streaming": streaming_manifest(),
	}


def streaming_manifest() -> dict[str, Any]:
	return {
		"processor": "bytewax",
		"stream": CRAWLER_EVENT_STREAM,
		"key": "tenant_id",
		"events": [
			"source_registered",
			"crawl_job_created",
			"crawl_job_completed",
			"extraction_recorded",
			"dataset_published",
			"validation_session_opened",
			"rag_plan_recorded",
			"graph_projection_recorded",
			"crawler_agent_registered",
		],
		"states": ["draft", "active", "scheduled", "running", "review_required", "validated", "published", "blocked"],
		"guardrails": [
			"crawler_batch_requires_bytewax",
			"crawler_event_requires_bytewax",
			"privileged_agent_crawler_action_requires_human_approval",
		],
	}


def event_stream_name() -> str:
	return CRAWLER_EVENT_STREAM


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
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
		if key.endswith("_lte"):
			if not context.get(key[:-4], 0) <= expected:
				return False
		elif key.endswith("_lt"):
			if not context.get(key[:-3], 0) < expected:
				return False
		elif key.endswith("_gte"):
			if not context.get(key[:-4], 0) >= expected:
				return False
		elif key.endswith("_gt"):
			if not context.get(key[:-3], 0) > expected:
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
