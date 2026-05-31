"""Executable capability contract for APG NLP Core."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


SUPPORTED_LANGUAGES: list[str] = [
	"en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "ar", "hi",
	"af", "aa", "ak", "am", "bm", "ee", "ff", "ha", "ig", "kr", "ki", "rw",
	"rn", "kg", "ln", "lg", "mg", "ny", "om", "sg", "sn", "so", "st", "sw",
	"ss", "ti", "ts", "tn", "tw", "ve", "wo", "xh", "yo", "zu", "kab", "kam",
	"luo", "mas", "mer", "mos", "nus", "suk", "tzm", "tig", "umb",
]
SUPPORTED_NLPC_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_NLPC_AGENT_ROLES = [
	"document_reviewer",
	"language_reviewer",
	"pii_reviewer",
	"generation_safety_reviewer",
	"annotation_reviewer",
	"pipeline_reviewer",
	"model_release_reviewer",
	"semantic_search_reviewer",
	"language_steward",
]
PRIVILEGED_NLPC_AGENT_ROLES = [
	"pii_reviewer",
	"generation_safety_reviewer",
	"annotation_reviewer",
	"pipeline_reviewer",
	"model_release_reviewer",
	"semantic_search_reviewer",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"processing": {
		"default_language": "auto",
		"supported_languages": SUPPORTED_LANGUAGES,
		"max_document_chars": 250000,
		"async_threshold_documents": 25,
		"language_detection_required": True,
		"document_hash_required": True,
	},
	"languages": {
		"auto_detection_enabled": True,
		"minimum_detection_confidence": 0.70,
		"african_language_coverage_required": 40,
		"language_policy_required": True,
	},
	"tasks": {
		"enabled": [
			"text_classification",
			"sentiment_analysis",
			"entity_recognition",
			"semantic_search",
			"summarization",
			"text_generation",
			"pii_detection",
			"translation",
			"topic_modeling",
			"keyword_extraction",
		],
		"generation_safety_required": True,
		"minimum_confidence_score": 0.75,
		"human_review_for_low_confidence": True,
	},
	"pipelines": {
		"pipeline_owner_required": True,
		"registered_model_required": True,
		"versioned_pipeline_required": True,
		"batch_async_required": True,
	},
	"annotation": {
		"annotation_guidelines_required": True,
		"minimum_consensus_score": 0.80,
		"review_required_for_disagreement": True,
		"golden_dataset_required": True,
	},
	"model_registry": {
		"mlcm_link_required": True,
		"model_policy_required": True,
		"evaluation_required": True,
		"release_approval_required": True,
	},
	"agents": {
		"first_class": True,
		"supported_runtimes": SUPPORTED_NLPC_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_NLPC_AGENT_ROLES,
		"privileged_roles": PRIVILEGED_NLPC_AGENT_ROLES,
		"require_owner": True,
		"require_purpose": True,
		"require_scope": True,
		"require_contribution_disclosure": True,
		"require_human_approval_for_privileged_roles": True,
		"adapter_contract": "aicr_provider_neutral_agent_adapter",
	},
	"streaming": {
		"engine": "bytewax",
		"lifecycle_stream": "nlpc.lifecycle",
		"watermark": "event_time",
		"required_processor": "bytewax",
		"required_operations": [
			"document_batch",
			"processing_batch",
			"pipeline_batch",
			"annotation_batch",
			"model_batch",
			"lexicon_batch",
			"language_registry_batch",
			"nlp_agent_batch",
		],
		"topics": [
			"nlpc.documents",
			"nlpc.processing",
			"nlpc.pipelines",
			"nlpc.annotations",
			"nlpc.models",
			"nlpc.lexicons",
			"nlpc.languages",
			"nlpc.agents",
		],
		"broker_core_dependency_allowed": False,
	},
	"governance": {
		"require_tenant_context": True,
		"auth_required": True,
		"audit_processing": True,
		"pii_redaction_policy_required": True,
		"model_policy_required": True,
		"cross_tenant_processing_allowed": False,
	},
	"observability": {
		"metrics_required": True,
		"trace_required": True,
		"audit_required": True,
		"event_stream": "bytewax",
		"quality_metrics_required": True,
	},
	"adapters": {
		"generated_app_runtime": "nlpc_runtime.NlpcService",
		"production_runtime": "service.NLPCService",
		"http_api": "api.py",
		"event_stream": "bytewax",
		"ai_core": "aicr",
		"model_lifecycle": "mlcm",
		"configuration": "conf",
		"auth_provider": "auth",
		"audit_sink": "audl",
		"metrics_sink": "moni",
		"search_index": "srch",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_processing_console": True,
		"enable_document_workbench": True,
		"enable_pipeline_designer": True,
		"enable_batch_queue": True,
		"enable_annotation_workbench": True,
		"enable_review_console": True,
		"enable_model_registry": True,
		"enable_language_coverage": True,
		"enable_lexicon_manager": True,
		"enable_semantic_search": True,
		"enable_nlp_agent_roster": True,
		"enable_lifecycle_batch_monitor": True,
		"enable_governance": True,
		"enable_audit_timeline": True,
		"enable_settings": True,
	},
	"theme": {"default_theme": "nlpc_text_intelligence", "allow_tenant_overrides": True},
}


CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"processing",
		"languages",
		"tasks",
		"pipelines",
		"annotation",
		"model_registry",
		"agents",
		"streaming",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {key: {"type": "object"} for key in [
		"processing",
		"languages",
		"tasks",
		"pipelines",
		"annotation",
		"model_registry",
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
	{"name": "tenant_context_required", "description": "All NLP operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "document_requires_content", "description": "Document ingestion requires non-empty content.", "condition": {"operation": "ingest_document", "content_present": False}, "effect": {"decision": "deny", "reason": "document_content_required", "required_action": "provide_document_text"}},
	{"name": "document_size_within_limit", "description": "Documents must fit the configured processing size limit.", "condition": {"operation": "ingest_document", "document_chars_gt": 250000}, "effect": {"decision": "deny", "reason": "document_too_large", "required_action": "split_document"}},
	{"name": "document_requires_language_or_detection", "description": "Document ingestion requires a language or detection permission.", "condition": {"operation": "ingest_document", "language_known": False, "language_detection_enabled": False}, "effect": {"decision": "deny", "reason": "language_required", "required_action": "run_language_detection"}},
	{"name": "language_required_or_detected", "description": "Processing requires a declared or detected language.", "condition": {"operation": "process_document", "language_known": False}, "effect": {"decision": "deny", "reason": "language_required", "required_action": "run_language_detection"}},
	{"name": "language_must_be_supported", "description": "Processing language must be supported by NLPC.", "condition": {"operation": "process_document", "language_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_language", "required_action": "choose_supported_language"}},
	{"name": "language_detection_low_confidence_requires_review", "description": "Low-confidence language detection requires review.", "condition": {"operation": "detect_language", "confidence_score_lt": 0.70, "human_review_recorded": False}, "effect": {"decision": "require_review", "reason": "language_detection_review_required", "required_action": "record_language_review"}},
	{"name": "task_must_be_enabled", "description": "NLP tasks must be enabled for the tenant.", "condition": {"operation": "process_document", "task_enabled": False}, "effect": {"decision": "deny", "reason": "nlp_task_not_enabled", "required_action": "enable_task_or_change_pipeline"}},
	{"name": "pii_requires_redaction_policy", "description": "PII extraction requires a redaction policy.", "condition": {"task": "pii_detection", "redaction_policy_attached": False}, "effect": {"decision": "deny", "reason": "pii_redaction_policy_required", "required_action": "attach_redaction_policy"}},
	{"name": "generation_requires_safety_policy", "description": "Text generation requires a model safety policy.", "condition": {"task": "text_generation", "safety_policy_attached": False}, "effect": {"decision": "deny", "reason": "generation_safety_policy_required", "required_action": "attach_safety_policy"}},
	{"name": "generation_requires_model_policy", "description": "Text generation requires model policy metadata.", "condition": {"task": "text_generation", "model_policy_attached": False}, "effect": {"decision": "deny", "reason": "model_policy_required", "required_action": "attach_model_policy"}},
	{"name": "translation_requires_source_and_target", "description": "Translation requires source and target language evidence.", "condition": {"task": "translation", "translation_pair_present": False}, "effect": {"decision": "deny", "reason": "translation_pair_required", "required_action": "attach_source_and_target_languages"}},
	{"name": "semantic_search_requires_index", "description": "Semantic search requires a search index binding.", "condition": {"task": "semantic_search", "search_index_attached": False}, "effect": {"decision": "deny", "reason": "search_index_required", "required_action": "attach_search_index"}},
	{"name": "summarization_requires_length_budget", "description": "Summarization requires an output length budget.", "condition": {"task": "summarization", "length_budget_present": False}, "effect": {"decision": "require_review", "reason": "summary_length_budget_required", "required_action": "record_summary_budget"}},
	{"name": "low_confidence_requires_review", "description": "Low-confidence NLP results require review.", "condition": {"confidence_score_lt": 0.75, "human_review_recorded": False}, "effect": {"decision": "require_review", "reason": "low_confidence_review_required", "required_action": "record_human_review"}},
	{"name": "large_batch_requires_async_queue", "description": "Large NLP batches must run through the async queue.", "condition": {"document_count_gt": 25, "async_queue_enabled": False}, "effect": {"decision": "require_review", "reason": "large_batch_requires_async_queue", "required_action": "enable_async_queue"}},
	{"name": "batch_requires_bytewax_stream", "description": "Batch processing event streams must use Bytewax.", "condition": {"operation": "configure_batch_events", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
	{"name": "pipeline_requires_owner", "description": "NLP pipelines require an owner.", "condition": {"operation": "register_pipeline", "owner_assigned": False}, "effect": {"decision": "deny", "reason": "pipeline_owner_required", "required_action": "assign_pipeline_owner"}},
	{"name": "pipeline_requires_registered_model", "description": "NLP pipelines require registered model linkage.", "condition": {"operation": "register_pipeline", "registered_model_attached": False}, "effect": {"decision": "deny", "reason": "registered_model_required", "required_action": "attach_registered_model"}},
	{"name": "pipeline_requires_version", "description": "NLP pipelines require version metadata.", "condition": {"operation": "register_pipeline", "pipeline_version_present": False}, "effect": {"decision": "deny", "reason": "pipeline_version_required", "required_action": "version_pipeline"}},
	{"name": "model_requires_mlcm_link", "description": "NLP model registrations require MLCM linkage.", "condition": {"operation": "register_model", "mlcm_model_ref_present": False}, "effect": {"decision": "deny", "reason": "mlcm_model_ref_required", "required_action": "link_mlcm_model"}},
	{"name": "model_release_requires_evaluation", "description": "NLP model release requires evaluation evidence.", "condition": {"operation": "release_model", "evaluation_recorded": False}, "effect": {"decision": "deny", "reason": "model_evaluation_required", "required_action": "record_model_evaluation"}},
	{"name": "model_release_requires_approval", "description": "NLP model release requires approval.", "condition": {"operation": "release_model", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "model_release_approval_required", "required_action": "record_release_approval"}},
	{"name": "annotation_requires_guidelines", "description": "Annotation projects require guidelines.", "condition": {"operation": "create_annotation_project", "guidelines_present": False}, "effect": {"decision": "deny", "reason": "annotation_guidelines_required", "required_action": "attach_annotation_guidelines"}},
	{"name": "annotation_low_consensus_requires_review", "description": "Low annotation consensus requires adjudication.", "condition": {"operation": "submit_annotation", "consensus_score_lt": 0.80, "adjudication_recorded": False}, "effect": {"decision": "require_review", "reason": "annotation_adjudication_required", "required_action": "record_annotation_adjudication"}},
	{"name": "lexicon_requires_language", "description": "Tenant lexicons require language metadata.", "condition": {"operation": "register_lexicon", "language_known": False}, "effect": {"decision": "deny", "reason": "lexicon_language_required", "required_action": "attach_lexicon_language"}},
	{"name": "quality_metric_requires_owner", "description": "NLP quality metrics require accountable ownership.", "condition": {"operation": "record_quality_metric", "owner_assigned": False}, "effect": {"decision": "require_review", "reason": "quality_metric_owner_required", "required_action": "assign_metric_owner"}},
	{"name": "cross_tenant_processing_denied", "description": "Cross-tenant text processing is denied by default.", "condition": {"cross_tenant_processing": True}, "effect": {"decision": "deny", "reason": "cross_tenant_processing_denied", "required_action": "use_tenant_scoped_document"}},
	{"name": "audit_event_required_for_processing", "description": "NLP processing state changes require audit events.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "audit_event_required", "required_action": "record_audit_event"}},
	{"name": "african_language_coverage_required", "description": "The language registry must keep at least 40 African language codes.", "condition": {"operation": "validate_language_registry", "african_language_count_lt": 40}, "effect": {"decision": "deny", "reason": "african_language_coverage_required", "required_action": "restore_african_language_codes"}},
	{"name": "nlp_agent_runtime_supported", "description": "NLP agents must use supported runtimes.", "condition": {"operation": "register_nlp_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_nlp_agent_runtime", "required_action": "choose_supported_nlp_agent_runtime"}},
	{"name": "nlp_agent_role_supported", "description": "NLP agents must use supported text-intelligence roles.", "condition": {"operation": "register_nlp_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_nlp_agent_role", "required_action": "choose_supported_nlp_agent_role"}},
	{"name": "nlp_agent_requires_scope", "description": "NLP agents require an explicit bounded scope.", "condition": {"operation": "register_nlp_agent", "scope_present": False}, "effect": {"decision": "deny", "reason": "nlp_agent_scope_required", "required_action": "declare_nlp_agent_scope"}},
	{"name": "nlp_agent_requires_owner", "description": "NLP agents require an accountable owner.", "condition": {"operation": "register_nlp_agent", "owner_present": False}, "effect": {"decision": "deny", "reason": "nlp_agent_owner_required", "required_action": "assign_nlp_agent_owner"}},
	{"name": "nlp_agent_requires_purpose", "description": "NLP agents require a documented purpose.", "condition": {"operation": "register_nlp_agent", "purpose_present": False}, "effect": {"decision": "deny", "reason": "nlp_agent_purpose_required", "required_action": "document_nlp_agent_purpose"}},
	{"name": "nlp_agent_requires_contribution_disclosure", "description": "NLP agents must disclose machine-authored text-intelligence contributions.", "condition": {"operation": "register_nlp_agent", "contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "nlp_agent_contribution_disclosure_required", "required_action": "disclose_machine_contribution"}},
	{"name": "nlp_agent_privileged_role_requires_human_approval", "description": "Privileged NLP-agent roles require human approval evidence.", "condition": {"operation": "register_nlp_agent", "privileged_role": True, "human_approval_required": False}, "effect": {"decision": "require_review", "reason": "nlp_agent_human_approval_required", "required_action": "record_human_nlp_agent_approval"}},
	{"name": "bytewax_nlpc_stream_required", "description": "NLPC lifecycle batches must be routed through Bytewax.", "condition": {"operation": "validate_nlpc_lifecycle_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_lifecycle_stream_required", "required_action": "route_nlpc_lifecycle_batch_to_bytewax"}},
]


UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/nlpc/dashboard", "component": "NLPCDashboard", "permission": "nlpc:view", "nav_group": "Overview"},
	{"name": "process", "path": "/nlpc/process", "component": "ProcessingConsole", "permission": "nlpc:process", "nav_group": "Process"},
	{"name": "documents", "path": "/nlpc/documents", "component": "DocumentWorkbench", "permission": "nlpc:process", "nav_group": "Process"},
	{"name": "pipelines", "path": "/nlpc/pipelines", "component": "PipelineDesigner", "permission": "nlpc:manage_models", "nav_group": "Process"},
	{"name": "batches", "path": "/nlpc/batches", "component": "BatchQueue", "permission": "nlpc:process", "nav_group": "Process"},
	{"name": "annotations", "path": "/nlpc/annotations", "component": "AnnotationWorkbench", "permission": "nlpc:annotate", "nav_group": "Quality"},
	{"name": "review", "path": "/nlpc/review", "component": "HumanReviewConsole", "permission": "nlpc:annotate", "nav_group": "Quality"},
	{"name": "models", "path": "/nlpc/models", "component": "NLPModelRegistry", "permission": "nlpc:manage_models", "nav_group": "Models"},
	{"name": "languages", "path": "/nlpc/languages", "component": "LanguageCoverage", "permission": "nlpc:view", "nav_group": "Coverage"},
	{"name": "lexicons", "path": "/nlpc/lexicons", "component": "LexiconManager", "permission": "nlpc:manage_models", "nav_group": "Coverage"},
	{"name": "search", "path": "/nlpc/search", "component": "SemanticSearchConsole", "permission": "nlpc:process", "nav_group": "Search"},
	{"name": "agents", "path": "/nlpc/agents", "component": "NLPAgentRoster", "permission": "nlpc:govern", "nav_group": "Governance"},
	{"name": "lifecycle", "path": "/nlpc/lifecycle", "component": "NLPCLifecycleBatchMonitor", "permission": "nlpc:govern", "nav_group": "Operations"},
	{"name": "governance", "path": "/nlpc/governance", "component": "NLPGovernance", "permission": "nlpc:govern", "nav_group": "Governance"},
	{"name": "audit", "path": "/nlpc/audit", "component": "NLPCAuditTimeline", "permission": "nlpc:view", "nav_group": "Governance"},
	{"name": "settings", "path": "/nlpc/settings", "component": "NLPCSettings", "permission": "nlpc:admin", "nav_group": "Administration"},
]


THEME: dict[str, Any] = {
	"name": "nlpc_text_intelligence",
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
		"document_queue": {"icon": "file-text", "status_indicator": "processing-pill", "risk_style": "policy-band"},
		"language_coverage_map": {"visual": "coverage-grid", "highlight": "african-language-chip"},
		"annotation_panel": {"visual": "span-highlighter", "status_style": "review-chip"},
		"model_result_card": {"visual": "confidence-meter", "threshold_style": "quality-band"},
		"pipeline_designer": {"visual": "task-chain", "status_style": "model-policy-chip"},
		"batch_queue": {"visual": "document-stack", "status_style": "async-chip"},
		"lexicon_manager": {"visual": "term-table", "status_style": "language-chip"},
		"semantic_search": {"visual": "ranked-results", "highlight": "similarity-chip"},
		"nlp_agent_roster": {"icon": "bot-message-square", "status_indicator": "agent-approval-chip", "risk_style": "scope-band"},
		"bytewax_lifecycle_panel": {"icon": "git-branch", "status_indicator": "stream-chip", "risk_style": "processor-band"},
		"audit_timeline": {"visual": "event-timeline", "status_style": "evidence-chip"},
	},
}


def agent_manifest() -> dict[str, Any]:
	"""Return first-class NLPC agent composition manifest."""
	return {
		"first_class": True,
		"supported_runtimes": list(SUPPORTED_NLPC_AGENT_RUNTIMES),
		"supported_roles": list(SUPPORTED_NLPC_AGENT_ROLES),
		"privileged_roles": list(PRIVILEGED_NLPC_AGENT_ROLES),
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
		"adapter_contract": "aicr_provider_neutral_agent_adapter",
	}


def streaming_manifest() -> dict[str, Any]:
	"""Return the NLPC Bytewax lifecycle stream contract."""
	return {
		"engine": "bytewax",
		"lifecycle_stream": "nlpc.lifecycle",
		"watermark": "event_time",
		"required_processor": "bytewax",
		"required_operations": [
			"document_batch",
			"processing_batch",
			"pipeline_batch",
			"annotation_batch",
			"model_batch",
			"lexicon_batch",
			"language_registry_batch",
			"nlp_agent_batch",
		],
		"topics": [
			"nlpc.documents",
			"nlpc.processing",
			"nlpc.pipelines",
			"nlpc.annotations",
			"nlpc.models",
			"nlpc.lexicons",
			"nlpc.languages",
			"nlpc.agents",
		],
		"broker_core_dependency_allowed": False,
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable NLPC capability contract."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "nlpc",
		"display_name": "NLP Core",
		"provides": ["text_intelligence", "multilingual_processing", "nlp_agent_composition"],
		"requires": ["aicr", "mlcm", "conf"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "view_models.py",
			"api_prefix": "/nlpc/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True,
		},
		"agents": agent_manifest(),
		"streaming": streaming_manifest(),
		"theme": deepcopy(THEME),
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate default NLPC governance rules."""
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
		if key.endswith("_lt"):
			if key[:-3] not in context or not context[key[:-3]] < expected:
				return False
		elif key.endswith("_gt"):
			if key[:-3] not in context or not context[key[:-3]] > expected:
				return False
		elif key.endswith("_ne"):
			if key[:-3] not in context or context[key[:-3]] == expected:
				return False
		elif key not in context or context[key] != expected:
			return False
	return True


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
	for key, value in source.items():
		if isinstance(value, dict) and isinstance(target.get(key), dict):
			_deep_merge(target[key], value)
		else:
			target[key] = value
