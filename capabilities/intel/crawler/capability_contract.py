"""Executable capability contract for APG Intelligence Crawler."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "intel_crawler"
CAPABILITY_NAME = "Intelligence Crawler"
CAPABILITY_VERSION = "1.1.0"
CRAWLER_EVENT_STREAM = "apg.intel.crawler.lifecycle"

SUPPORTED_AUTHORITY_TYPES = ["mission_order", "legal_mandate", "partner_authority", "consent", "public_interest_authority", "data_processing_authority"]
SUPPORTED_CLASSIFICATIONS = ["unclassified", "confidential", "secret", "top_secret"]
SUPPORTED_SOURCE_TYPES = ["news_site", "web_site", "rss_feed", "api_endpoint", "partner_feed", "dataset_endpoint", "document_repository", "public_registry"]
SUPPORTED_CRAWL_MODES = ["full_crawl", "incremental", "sitemap_guided", "rss_pull", "api_poll", "scheduled_refresh", "manual_trigger"]
SUPPORTED_EXTRACTION_TYPES = ["structured_text", "tabular_data", "document", "metadata_only", "entity_mentions", "link_graph", "image_metadata"]
SUPPORTED_RETENTION_CLASSES = ["short", "standard", "extended", "legal_hold"]
SUPPORTED_VALIDATION_DECISIONS = ["pass", "fail", "needs_review", "escalated"]
SUPPORTED_RISK_TIERS = ["low", "medium", "high", "critical"]
SUPPORTED_DATASET_TYPES = ["text_corpus", "entity_dataset", "graph_dataset", "document_store", "embedding_store", "knowledge_graph"]
SUPPORTED_RAG_STRATEGIES = ["fixed_chunk", "semantic_chunk", "sentence_window", "document_summary", "hierarchical"]
SUPPORTED_GRAPH_ENTITY_TYPES = ["person", "organization", "location", "event", "topic", "product", "concept"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = [
	"source_strategy_reviewer",
	"crawl_policy_reviewer",
	"extraction_quality_reviewer",
	"validation_reviewer",
	"rag_pipeline_reviewer",
	"risk_reviewer",
]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"authorities": {
		"supported_authority_types": SUPPORTED_AUTHORITY_TYPES,
		"supported_classifications": SUPPORTED_CLASSIFICATIONS,
		"approver_required": True,
		"expiry_required": True,
		"evidence_required": True,
	},
	"sources": {
		"supported_source_types": SUPPORTED_SOURCE_TYPES,
		"supported_risk_tiers": SUPPORTED_RISK_TIERS,
		"owner_required": True,
		"url_required": True,
		"allowed_domain_required": True,
		"robots_policy_review_required": True,
		"authority_required": True,
		"evidence_required": True,
	},
	"crawl_jobs": {
		"supported_modes": SUPPORTED_CRAWL_MODES,
		"source_required": True,
		"positive_rate_limit_required": True,
		"max_depth_limit": 8,
		"approval_required_for_high_risk": True,
		"evidence_required": True,
	},
	"extraction": {
		"supported_types": SUPPORTED_EXTRACTION_TYPES,
		"schema_required": True,
		"content_fingerprint_required": True,
		"minimum_quality_score": 0.7,
		"evidence_required": True,
	},
	"datasets": {
		"supported_types": SUPPORTED_DATASET_TYPES,
		"supported_retention_classes": SUPPORTED_RETENTION_CLASSES,
		"lineage_required": True,
		"validation_required_before_publish": True,
		"pii_review_required": True,
		"owner_required": True,
		"evidence_required": True,
	},
	"validation": {
		"supported_decisions": SUPPORTED_VALIDATION_DECISIONS,
		"reviewer_required": True,
		"minimum_confidence": 0.75,
		"disagreement_requires_review": True,
		"evidence_required": True,
	},
	"rag": {
		"supported_strategies": SUPPORTED_RAG_STRATEGIES,
		"chunk_plan_required": True,
		"chunk_size_max": 4096,
		"embedding_model_required": True,
		"evidence_required": True,
	},
	"knowledge_graph": {
		"supported_entity_types": SUPPORTED_GRAPH_ENTITY_TYPES,
		"entity_schema_required": True,
		"relationship_evidence_required": True,
		"evidence_required": True,
	},
	"reviews": {
		"supported_statuses": SUPPORTED_REVIEW_STATUSES,
		"reviewer_required": True,
		"evidence_required": True,
	},
	"crawler_agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_AGENT_ROLES,
		"human_approval_required": True,
		"max_autonomous_scope": "recommend_validate_and_prepare",
	},
	"governance": {
		"require_tenant_context": True,
		"audit_state_changes": True,
		"policy_attached_for_writes": True,
		"respect_source_terms": True,
		"cross_tenant_crawl_denied": True,
		"privilege_escalation_denied": True,
		"unauthorized_pii_collection_denied": True,
		"scraping_beyond_authority_denied": True,
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
		"auth": "auth",
		"audit": "audl",
		"notifications": "ntfy",
		"nlp": "nlpc",
		"graph": "grph",
		"rag": "ragn",
		"event_stream": "bytewax",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_authorities": True,
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

PROVIDES = [
	"source_intelligence_registry",
	"crawl_job_lifecycle",
	"extraction_pipeline",
	"dataset_quality_control",
	"validation_workflow",
	"rag_graphrag_preparation",
	"crawler_authority_workflow",
	"crawler_governance_workflow",
	"crawler_review_workflow",
	"crawler_agents",
]
REQUIRES = ["auth", "audl", "ntfy", "nlpc", "grph", "ragn", "mten", "conf"]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/intel-crawler/dashboard", "component": "CrawlerDashboard", "permission": "intel_crawler:view", "nav_group": "Overview"},
	{"name": "authorities", "path": "/intel-crawler/authorities", "component": "CrawlerAuthorityConsole", "permission": "intel_crawler:authorities", "nav_group": "Governance"},
	{"name": "sources", "path": "/intel-crawler/sources", "component": "CrawlerSourceRegistry", "permission": "intel_crawler:manage_sources", "nav_group": "Sources"},
	{"name": "crawl_jobs", "path": "/intel-crawler/crawl-jobs", "component": "CrawlerJobConsole", "permission": "intel_crawler:operate", "nav_group": "Crawling"},
	{"name": "extractions", "path": "/intel-crawler/extractions", "component": "ExtractionWorkbench", "permission": "intel_crawler:extract", "nav_group": "Processing"},
	{"name": "datasets", "path": "/intel-crawler/datasets", "component": "DatasetPublicationConsole", "permission": "intel_crawler:publish", "nav_group": "Processing"},
	{"name": "validation", "path": "/intel-crawler/validation", "component": "ValidationWorkbench", "permission": "intel_crawler:validate", "nav_group": "Quality"},
	{"name": "rag", "path": "/intel-crawler/rag", "component": "RAGPreparationWorkbench", "permission": "intel_crawler:rag", "nav_group": "Knowledge"},
	{"name": "graph", "path": "/intel-crawler/graph", "component": "KnowledgeGraphProjection", "permission": "intel_crawler:graph", "nav_group": "Knowledge"},
	{"name": "reviews", "path": "/intel-crawler/reviews", "component": "CrawlerReviewConsole", "permission": "intel_crawler:reviews", "nav_group": "Governance"},
	{"name": "agents", "path": "/intel-crawler/agents", "component": "CrawlerAgentWorkbench", "permission": "intel_crawler:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/intel-crawler/settings", "component": "CrawlerSettings", "permission": "intel_crawler:admin", "nav_group": "Administration"},
]

THEME: dict[str, Any] = {
	"name": "intel_crawler_control",
	"tokens": {
		"color.primary": "#28536B",
		"color.accent": "#C44536",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F7F8FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact",
	},
	"components": {
		"authorities": {"icon": "shield-check", "status_indicator": "authority-chip"},
		"sources": {"icon": "radar", "status_indicator": "source-pill", "risk_style": "policy-band"},
		"crawl_jobs": {"icon": "calendar-clock", "visual": "job-queue", "status_style": "crawl-chip"},
		"extractions": {"icon": "file-search", "visual": "schema-grid", "status_style": "quality-chip"},
		"datasets": {"icon": "database", "visual": "lineage-table", "status_style": "publish-chip"},
		"validation": {"icon": "badge-check", "visual": "review-lane", "status_style": "confidence-chip"},
		"rag": {"icon": "layers", "visual": "chunk-map", "status_style": "embedding-chip"},
		"graph": {"icon": "network", "visual": "entity-network", "status_style": "evidence-chip"},
		"reviews": {"icon": "clipboard-check", "status_indicator": "review-chip"},
		"agent_workbench": {"icon": "bot", "visual": "approval-lane", "status_style": "agent-chip"},
	},
}

STREAMING: dict[str, Any] = {
	"processor": "bytewax",
	"stream": CRAWLER_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"crawler_authority_recorded",
		"source_registered",
		"crawl_job_created",
		"crawl_job_completed",
		"extraction_recorded",
		"dataset_published",
		"validation_session_opened",
		"validation_session_completed",
		"rag_plan_recorded",
		"graph_projection_recorded",
		"crawler_review_recorded",
		"crawler_agent_registered",
	],
	"states": ["draft", "active", "scheduled", "running", "review_required", "validated", "published", "blocked"],
	"guardrails": [
		"crawler_batch_requires_bytewax",
		"crawler_event_requires_bytewax",
		"privileged_agent_crawler_action_requires_human_approval",
		"cross_tenant_crawl_action_denied",
		"privilege_escalation_action_denied",
		"unauthorized_pii_collection_action_denied",
		"scraping_beyond_authority_action_denied",
		"unapproved_high_risk_crawl_action_denied",
	],
}

RULES: list[dict[str, Any]] = [
	# Core governance
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "crawler_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "operation_policy_required", "required_action": "attach_operation_policy"}},
	{"name": "cross_tenant_crawl_denied", "condition": {"operation_type": "write", "cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_crawl_denied", "required_action": "remove_cross_tenant_scope"}},
	{"name": "privilege_escalation_denied", "condition": {"operation": "crawler_agent_action", "privilege_escalation_scope": True}, "effect": {"decision": "deny", "reason": "privilege_escalation_denied", "required_action": "remove_privilege_escalation_scope"}},
	# Authority workflow
	{"name": "authority_type_supported", "condition": {"operation": "record_authority", "authority_type_supported": False}, "effect": {"decision": "deny", "reason": "authority_type_not_supported", "required_action": "select_supported_authority_type"}},
	{"name": "authority_scope_required", "condition": {"operation": "record_authority", "scope_present": False}, "effect": {"decision": "deny", "reason": "authority_scope_required", "required_action": "attach_scope_reference"}},
	{"name": "authority_classification_supported", "condition": {"operation": "record_authority", "classification_supported": False}, "effect": {"decision": "deny", "reason": "classification_not_supported", "required_action": "select_supported_classification"}},
	{"name": "authority_approver_required", "condition": {"operation": "record_authority", "approver_present": False}, "effect": {"decision": "deny", "reason": "authority_approver_required", "required_action": "record_approver"}},
	{"name": "authority_expiry_required", "condition": {"operation": "record_authority", "expiry_present": False}, "effect": {"decision": "deny", "reason": "authority_expiry_required", "required_action": "set_expiry"}},
	{"name": "authority_evidence_required", "condition": {"operation": "record_authority", "evidence_present": False}, "effect": {"decision": "deny", "reason": "authority_evidence_required", "required_action": "attach_authority_evidence"}},
	# Source registration
	{"name": "source_requires_owner", "condition": {"operation": "register_source", "source_owner_assigned": False}, "effect": {"decision": "deny", "reason": "source_owner_required", "required_action": "assign_source_owner"}},
	{"name": "source_requires_url", "condition": {"operation": "register_source", "source_url_present": False}, "effect": {"decision": "deny", "reason": "source_url_required", "required_action": "attach_source_url"}},
	{"name": "source_requires_allowed_domain", "condition": {"operation": "register_source", "allowed_domain_present": False}, "effect": {"decision": "deny", "reason": "allowed_domain_required", "required_action": "set_allowed_domain"}},
	{"name": "source_requires_policy_review", "condition": {"operation": "register_source", "policy_review_recorded": False}, "effect": {"decision": "require_review", "reason": "crawl_policy_review_required", "required_action": "record_source_policy_review"}},
	{"name": "source_authority_required", "condition": {"operation": "register_source", "authority_present": False}, "effect": {"decision": "deny", "reason": "lawful_authority_required", "required_action": "select_authority"}},
	{"name": "source_risk_tier_supported", "condition": {"operation": "register_source", "risk_tier_supported": False}, "effect": {"decision": "deny", "reason": "risk_tier_not_supported", "required_action": "select_supported_risk_tier"}},
	# Crawl job
	{"name": "crawl_job_requires_source", "condition": {"operation": "create_crawl_job", "source_present": False}, "effect": {"decision": "deny", "reason": "crawl_source_required", "required_action": "attach_source"}},
	{"name": "crawl_job_requires_cadence", "condition": {"operation": "create_crawl_job", "cadence_present": False}, "effect": {"decision": "deny", "reason": "crawl_cadence_required", "required_action": "set_cadence"}},
	{"name": "crawl_rate_limit_positive", "condition": {"operation": "create_crawl_job", "rate_limit_per_minute_lte": 0}, "effect": {"decision": "deny", "reason": "crawl_rate_limit_must_be_positive", "required_action": "set_positive_rate_limit"}},
	{"name": "crawl_depth_within_limit", "condition": {"operation": "create_crawl_job", "max_depth_gt": 8}, "effect": {"decision": "deny", "reason": "crawl_depth_limit_exceeded", "required_action": "reduce_crawl_depth"}},
	{"name": "crawl_mode_supported", "condition": {"operation": "create_crawl_job", "crawl_mode_supported": False}, "effect": {"decision": "deny", "reason": "crawl_mode_not_supported", "required_action": "select_supported_crawl_mode"}},
	{"name": "high_risk_crawl_requires_approval", "condition": {"operation": "create_crawl_job", "high_risk": True, "approved": False}, "effect": {"decision": "require_review", "reason": "high_risk_crawl_approval_required", "required_action": "record_crawl_approval"}},
	# Extraction
	{"name": "extraction_requires_schema", "condition": {"operation": "record_extraction", "schema_present": False}, "effect": {"decision": "deny", "reason": "extraction_schema_required", "required_action": "attach_extraction_schema"}},
	{"name": "extraction_requires_fingerprint", "condition": {"operation": "record_extraction", "fingerprint_present": False}, "effect": {"decision": "deny", "reason": "content_fingerprint_required", "required_action": "attach_content_fingerprint"}},
	{"name": "extraction_type_supported", "condition": {"operation": "record_extraction", "extraction_type_supported": False}, "effect": {"decision": "deny", "reason": "extraction_type_not_supported", "required_action": "select_supported_extraction_type"}},
	{"name": "extraction_quality_minimum", "condition": {"operation": "record_extraction", "quality_score_lt": 0.7}, "effect": {"decision": "require_review", "reason": "extraction_quality_review_required", "required_action": "review_extraction_quality"}},
	# Dataset publication
	{"name": "dataset_requires_lineage", "condition": {"operation": "publish_dataset", "lineage_present": False}, "effect": {"decision": "deny", "reason": "dataset_lineage_required", "required_action": "attach_dataset_lineage"}},
	{"name": "dataset_requires_validation", "condition": {"operation": "publish_dataset", "validation_recorded": False}, "effect": {"decision": "deny", "reason": "dataset_validation_required", "required_action": "record_dataset_validation"}},
	{"name": "dataset_type_supported", "condition": {"operation": "publish_dataset", "dataset_type_supported": False}, "effect": {"decision": "deny", "reason": "dataset_type_not_supported", "required_action": "select_supported_dataset_type"}},
	{"name": "dataset_owner_required", "condition": {"operation": "publish_dataset", "owner_present": False}, "effect": {"decision": "deny", "reason": "dataset_owner_required", "required_action": "assign_dataset_owner"}},
	{"name": "dataset_retention_supported", "condition": {"operation": "publish_dataset", "retention_class_supported": False}, "effect": {"decision": "deny", "reason": "retention_class_not_supported", "required_action": "select_supported_retention_class"}},
	{"name": "pii_dataset_requires_review", "condition": {"operation": "publish_dataset", "contains_pii": True, "privacy_review_recorded": False}, "effect": {"decision": "require_review", "reason": "privacy_review_required", "required_action": "record_privacy_review"}},
	# Validation
	{"name": "validation_requires_reviewer", "condition": {"operation": "open_validation_session", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "validation_reviewer_required", "required_action": "assign_reviewer"}},
	{"name": "validation_decision_supported", "condition": {"operation": "complete_validation_session", "decision_supported": False}, "effect": {"decision": "deny", "reason": "validation_decision_not_supported", "required_action": "select_supported_decision"}},
	{"name": "validation_confidence_minimum", "condition": {"operation": "complete_validation_session", "confidence_lt": 0.75}, "effect": {"decision": "require_review", "reason": "validation_confidence_review_required", "required_action": "review_validation_confidence"}},
	# RAG preparation
	{"name": "rag_requires_chunk_plan", "condition": {"operation": "record_rag_plan", "chunk_plan_present": False}, "effect": {"decision": "deny", "reason": "rag_chunk_plan_required", "required_action": "attach_chunk_plan"}},
	{"name": "rag_chunk_size_within_limit", "condition": {"operation": "record_rag_plan", "chunk_size_gt": 4096}, "effect": {"decision": "deny", "reason": "rag_chunk_size_limit_exceeded", "required_action": "reduce_chunk_size"}},
	{"name": "rag_requires_embedding_model", "condition": {"operation": "record_rag_plan", "embedding_model_present": False}, "effect": {"decision": "deny", "reason": "embedding_model_required", "required_action": "select_embedding_model"}},
	{"name": "rag_strategy_supported", "condition": {"operation": "record_rag_plan", "rag_strategy_supported": False}, "effect": {"decision": "deny", "reason": "rag_strategy_not_supported", "required_action": "select_supported_rag_strategy"}},
	# Knowledge graph
	{"name": "graph_requires_entity_schema", "condition": {"operation": "record_graph_projection", "entity_schema_present": False}, "effect": {"decision": "deny", "reason": "entity_schema_required", "required_action": "attach_entity_schema"}},
	{"name": "graph_entity_type_supported", "condition": {"operation": "record_graph_projection", "entity_type_supported": False}, "effect": {"decision": "deny", "reason": "entity_type_not_supported", "required_action": "select_supported_entity_type"}},
	{"name": "graph_requires_relationship_evidence", "condition": {"operation": "record_graph_projection", "relationship_evidence_present": False}, "effect": {"decision": "deny", "reason": "relationship_evidence_required", "required_action": "attach_relationship_evidence"}},
	# Review workflow
	{"name": "review_status_supported", "condition": {"operation": "record_review", "status_supported": False}, "effect": {"decision": "deny", "reason": "review_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "review_reviewer_required", "condition": {"operation": "record_review", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "reviewer_required", "required_action": "assign_reviewer"}},
	{"name": "review_evidence_required", "condition": {"operation": "record_review", "evidence_present": False}, "effect": {"decision": "deny", "reason": "review_evidence_required", "required_action": "attach_review_evidence"}},
	# Streaming / batch guardrails
	{"name": "crawler_batch_requires_bytewax", "condition": {"operation": "crawler_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_crawler_batch_to_bytewax"}},
	{"name": "crawler_event_requires_bytewax", "condition": {"operation": "crawler_event", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_crawler_event_to_bytewax"}},
	# Agent governance
	{"name": "crawler_agent_runtime_supported", "condition": {"operation": "register_crawler_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "crawler_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"}},
	{"name": "crawler_agent_role_supported", "condition": {"operation": "register_crawler_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "crawler_agent_role_not_supported", "required_action": "select_supported_agent_role"}},
	{"name": "privileged_agent_crawler_action_requires_human_approval", "condition": {"operation": "agent_crawler_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
	{"name": "cross_tenant_crawl_action_denied", "condition": {"operation": "agent_crawler_action", "cross_tenant_crawl_scope": True}, "effect": {"decision": "deny", "reason": "cross_tenant_crawl_scope_denied", "required_action": "remove_cross_tenant_crawl_scope"}},
	{"name": "privilege_escalation_action_denied", "condition": {"operation": "agent_crawler_action", "privilege_escalation_scope": True}, "effect": {"decision": "deny", "reason": "privilege_escalation_scope_denied", "required_action": "remove_privilege_escalation_scope"}},
	{"name": "unauthorized_pii_collection_action_denied", "condition": {"operation": "agent_crawler_action", "unauthorized_pii_collection_scope": True}, "effect": {"decision": "deny", "reason": "unauthorized_pii_collection_scope_denied", "required_action": "remove_pii_collection_scope"}},
	{"name": "scraping_beyond_authority_action_denied", "condition": {"operation": "agent_crawler_action", "scraping_beyond_authority_scope": True}, "effect": {"decision": "deny", "reason": "scraping_beyond_authority_scope_denied", "required_action": "remove_out_of_scope_scraping"}},
	{"name": "unapproved_high_risk_crawl_action_denied", "condition": {"operation": "agent_crawler_action", "unapproved_high_risk_crawl_scope": True}, "effect": {"decision": "deny", "reason": "unapproved_high_risk_crawl_scope_denied", "required_action": "obtain_crawl_approval"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID,
		"display_name": CAPABILITY_NAME,
		"name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"provides": list(PROVIDES),
		"requires": list(REQUIRES),
		"configuration": configuration,
		"configuration_schema": {
			"type": "object",
			"required": list(configuration),
			"properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}},
		},
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/intel-crawler/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True,
		},
		"theme": deepcopy(THEME),
		"streaming": deepcopy(STREAMING),
	}


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
