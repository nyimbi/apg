"""Executable capability contract for APG Help and Knowledge Base."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any


SUPPORTED_HELP_AGENT_RUNTIMES: list[str] = ["codex", "claude_code", "opencode", "pi"]

SUPPORTED_HELP_AGENT_ROLES: list[str] = [
	"source_reviewer",
	"article_reviewer",
	"answer_reviewer",
	"search_reviewer",
	"feedback_reviewer",
	"localization_reviewer",
	"curation_reviewer",
	"safety_reviewer",
	"lifecycle_batch_reviewer",
	"knowledge_steward",
]

PRIVILEGED_HELP_AGENT_ROLES: list[str] = [
	"source_reviewer",
	"article_reviewer",
	"answer_reviewer",
	"safety_reviewer",
	"lifecycle_batch_reviewer",
	"knowledge_steward",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"content": {
		"article_owner_required": True,
		"article_title_required": True,
		"article_body_required": True,
		"source_approval_required": True,
		"publication_approval_required": True,
		"freshness_review_days": 90,
		"freshness_review_required": True,
		"localization_supported": True,
		"supported_locales": ["en", "fr", "sw", "ar"],
	},
	"sources": {
		"source_owner_required": True,
		"source_uri_required": True,
		"source_approval_required": True,
		"restricted_source_rbac_required": True,
	},
	"answers": {
		"rag_enabled": True,
		"query_required": True,
		"citations_required": True,
		"minimum_answer_confidence": 0.76,
		"unsafe_answer_blocking": True,
		"restricted_content_filtering_required": True,
	},
	"search": {
		"semantic_search_enabled": True,
		"query_required": True,
		"restricted_content_filtering": True,
		"feedback_boosting_enabled": True,
		"query_logging_enabled": True,
	},
	"feedback": {
		"user_required": True,
		"rating_min": 1,
		"rating_max": 5,
		"low_rating_review_threshold": 2,
		"review_queue_enabled": True,
	},
	"localization": {
		"translator_required": True,
		"source_locale_required": True,
		"fallback_locale_required": True,
		"supported_locales": ["en", "fr", "sw", "ar"],
	},
	"governance": {
		"require_tenant_context": True,
		"audit_publication": True,
		"audit_feedback": True,
		"source_approval_required": True,
		"support_feedback_review_required": True,
		"tenant_isolation_required": True,
		"batch_event_stream": "bytewax",
	},
	"observability": {
		"audit_required": True,
		"query_metrics_required": True,
		"deflection_metrics_required": True,
		"event_stream": "bytewax",
	},
	"agents": {
		"first_class": True,
		"supported_runtimes": SUPPORTED_HELP_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_HELP_AGENT_ROLES,
		"privileged_roles": PRIVILEGED_HELP_AGENT_ROLES,
		"require_scope": True,
		"require_owner": True,
		"require_purpose": True,
		"require_contribution_disclosure": True,
		"require_human_approval_for_privileged_roles": True,
		"adapter_contract": "aicr_provider_neutral_help_agent_adapter",
	},
	"streaming": {
		"engine": "bytewax",
		"lifecycle_stream": "help.lifecycle",
		"watermark": "event_time",
		"required_processor": "bytewax",
		"required_operations": [
			"source_batch",
			"article_batch",
			"answer_batch",
			"search_batch",
			"feedback_batch",
			"localization_batch",
			"curation_batch",
			"help_agent_batch",
			"audit_batch",
		],
		"topics": [
			"help.sources",
			"help.articles",
			"help.answers",
			"help.search",
			"help.feedback",
			"help.localizations",
			"help.curation",
			"help.agents",
			"help.audit",
		],
		"broker_core_dependency_allowed": False,
	},
	"adapters": {
		"generated_app_runtime": "service.HelpService",
		"runtime_helpers": "help_runtime.py",
		"api_helpers": "api.py",
		"view_models": "views.py",
		"event_stream": "bytewax",
		"retrieval_augmented_generation": "ragn",
		"search": "srch",
		"natural_language": "nlpc",
		"identity": "auth",
		"audit_sink": "audl",
		"notification": "ntfy",
		"chat": "chat",
		"theme": "them",
		"ai_orchestration": "aicr",
		"agent_adapter": "aicr_provider_neutral_help_agent_adapter",
	},
	"ui": {
		"enable_help_center": True,
		"enable_article_library": True,
		"enable_article_editor": True,
		"enable_answer_console": True,
		"enable_source_registry": True,
		"enable_localization_workbench": True,
		"enable_curation_queue": True,
		"enable_help_agent_roster": True,
		"enable_lifecycle_batch_monitor": True,
		"enable_audit": True,
		"enable_analytics": True,
	},
	"theme": {"default_theme": "help_support_knowledge", "allow_tenant_overrides": True},
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"content",
		"sources",
		"answers",
		"search",
		"feedback",
		"localization",
		"governance",
		"observability",
		"agents",
		"streaming",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {key: {"type": "object"} for key in [
		"content",
		"sources",
		"answers",
		"search",
		"feedback",
		"localization",
		"governance",
		"observability",
		"agents",
		"streaming",
		"adapters",
		"ui",
		"theme",
	]} | {"tenant_id": {"type": "string", "minLength": 1}},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All help operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "source_requires_owner", "description": "Help sources require accountable owners.", "condition": {"operation": "register_source", "source_owner_assigned": False}, "effect": {"decision": "deny", "reason": "source_owner_required", "required_action": "assign_source_owner"}},
	{"name": "source_requires_uri", "description": "Help sources require source URI or reference.", "condition": {"operation": "register_source", "source_uri_present": False}, "effect": {"decision": "deny", "reason": "source_uri_required", "required_action": "attach_source_uri"}},
	{"name": "source_requires_approval", "description": "Help sources require approval before article publication.", "condition": {"source_approval_required": True, "source_approved": False}, "effect": {"decision": "deny", "reason": "source_approval_required", "required_action": "approve_source"}},
	{"name": "article_requires_owner", "description": "Knowledge articles require accountable owners.", "condition": {"operation": "create_article", "article_owner_assigned": False}, "effect": {"decision": "deny", "reason": "article_owner_required", "required_action": "assign_article_owner"}},
	{"name": "article_requires_title", "description": "Knowledge articles require readable titles.", "condition": {"operation": "create_article", "article_title_present": False}, "effect": {"decision": "deny", "reason": "article_title_required", "required_action": "title_article"}},
	{"name": "article_requires_body", "description": "Knowledge articles require useful body content.", "condition": {"operation": "create_article", "article_body_present": False}, "effect": {"decision": "deny", "reason": "article_body_required", "required_action": "write_article_body"}},
	{"name": "publication_requires_approval", "description": "Articles require approval before publication.", "condition": {"operation": "publish_article", "publication_approved": False}, "effect": {"decision": "deny", "reason": "publication_approval_required", "required_action": "approve_publication"}},
	{"name": "publication_requires_audit", "description": "Article publication requires audit evidence.", "condition": {"operation": "publish_article", "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "publication_audit_required", "required_action": "record_publication_audit"}},
	{"name": "answer_requires_query", "description": "Generated answers require a user query.", "condition": {"operation": "generate_answer", "query_present": False}, "effect": {"decision": "deny", "reason": "answer_query_required", "required_action": "enter_answer_query"}},
	{"name": "answer_requires_citations", "description": "Generated answers require source citations.", "condition": {"operation": "generate_answer", "citations_present": False}, "effect": {"decision": "deny", "reason": "citations_required", "required_action": "attach_answer_citations"}},
	{"name": "answer_confidence_requires_review", "description": "Low-confidence generated answers require review.", "condition": {"operation": "generate_answer", "answer_confidence_lt": 0.76}, "effect": {"decision": "require_review", "reason": "answer_confidence_review_required", "required_action": "review_answer_confidence"}},
	{"name": "unsafe_answer_blocked", "description": "Unsafe generated answers are blocked.", "condition": {"operation": "generate_answer", "unsafe_answer_detected": True}, "effect": {"decision": "deny", "reason": "unsafe_answer_blocked", "required_action": "revise_answer_sources"}},
	{"name": "search_requires_query", "description": "Help search requires a query.", "condition": {"operation": "search_articles", "query_present": False}, "effect": {"decision": "deny", "reason": "search_query_required", "required_action": "enter_search_query"}},
	{"name": "search_requires_query_logging", "description": "Help search requires query logging for analytics and improvement.", "condition": {"operation": "search_articles", "query_logging_enabled": False}, "effect": {"decision": "require_review", "reason": "query_logging_review_required", "required_action": "enable_or_approve_query_logging_exception"}},
	{"name": "restricted_content_requires_filtering", "description": "Restricted help content requires RBAC filtering.", "condition": {"restricted_content_present": True, "rbac_filter_applied": False}, "effect": {"decision": "deny", "reason": "rbac_filter_required", "required_action": "apply_rbac_filter"}},
	{"name": "stale_article_requires_review", "description": "Stale knowledge articles require curation review.", "condition": {"article_age_days_gt": 90, "freshness_review_recorded": False}, "effect": {"decision": "require_review", "reason": "freshness_review_required", "required_action": "review_article_freshness"}},
	{"name": "feedback_requires_user", "description": "Help feedback requires a user reference.", "condition": {"operation": "record_feedback", "feedback_user_present": False}, "effect": {"decision": "deny", "reason": "feedback_user_required", "required_action": "identify_feedback_user"}},
	{"name": "feedback_rating_minimum", "description": "Help feedback ratings must be within configured bounds.", "condition": {"operation": "record_feedback", "feedback_rating_lt": 1}, "effect": {"decision": "deny", "reason": "rating_out_of_range", "required_action": "choose_valid_rating"}},
	{"name": "feedback_rating_maximum", "description": "Help feedback ratings must be within configured bounds.", "condition": {"operation": "record_feedback", "feedback_rating_gt": 5}, "effect": {"decision": "deny", "reason": "rating_out_of_range", "required_action": "choose_valid_rating"}},
	{"name": "low_feedback_requires_review", "description": "Low support feedback requires curation review.", "condition": {"operation": "record_feedback", "feedback_rating_lte": 2, "feedback_review_opened": False}, "effect": {"decision": "require_review", "reason": "support_feedback_review_required", "required_action": "open_feedback_review"}},
	{"name": "localization_requires_supported_locale", "description": "Article localization must use a supported locale.", "condition": {"operation": "localize_article", "locale_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_locale", "required_action": "choose_supported_locale"}},
	{"name": "localization_requires_translator", "description": "Article localization requires an accountable translator.", "condition": {"operation": "localize_article", "translator_assigned": False}, "effect": {"decision": "deny", "reason": "translator_required", "required_action": "assign_translator"}},
	{"name": "localization_requires_fallback", "description": "Localized help needs a fallback locale.", "condition": {"operation": "localize_article", "fallback_locale_configured": False}, "effect": {"decision": "require_review", "reason": "fallback_locale_required", "required_action": "configure_fallback_locale"}},
		{"name": "curation_requires_reviewer", "description": "Curation decisions require a reviewer.", "condition": {"operation": "close_curation_item", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "curation_reviewer_required", "required_action": "assign_curation_reviewer"}},
		{"name": "curation_requires_evidence", "description": "Curation decisions require evidence.", "condition": {"operation": "close_curation_item", "curation_evidence_present": False}, "effect": {"decision": "deny", "reason": "curation_evidence_required", "required_action": "attach_curation_evidence"}},
		{"name": "help_agent_requires_id", "description": "First-class help agents require stable identifiers.", "condition": {"operation": "register_help_agent", "agent_id_present": False}, "effect": {"decision": "deny", "reason": "help_agent_id_required", "required_action": "assign_help_agent_id"}},
		{"name": "help_agent_requires_name", "description": "First-class help agents require readable names.", "condition": {"operation": "register_help_agent", "agent_name_present": False}, "effect": {"decision": "deny", "reason": "help_agent_name_required", "required_action": "name_help_agent"}},
		{"name": "help_agent_runtime_supported", "description": "First-class help agents must use supported provider-neutral runtimes.", "condition": {"operation": "register_help_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_help_agent_runtime", "required_action": "choose_supported_help_agent_runtime"}},
		{"name": "help_agent_role_supported", "description": "First-class help agents must use supported help-governance roles.", "condition": {"operation": "register_help_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_help_agent_role", "required_action": "choose_supported_help_agent_role"}},
	{"name": "help_agent_requires_scope", "description": "First-class help agents require explicit source, article, answer, search, feedback, localization, curation, safety, or lifecycle scope.", "condition": {"operation": "register_help_agent", "scope_present": False}, "effect": {"decision": "deny", "reason": "help_agent_scope_required", "required_action": "declare_help_agent_scope"}},
	{"name": "help_agent_requires_owner", "description": "First-class help agents require an accountable owner.", "condition": {"operation": "register_help_agent", "owner_present": False}, "effect": {"decision": "deny", "reason": "help_agent_owner_required", "required_action": "assign_help_agent_owner"}},
	{"name": "help_agent_requires_purpose", "description": "First-class help agents require a documented knowledge-governance purpose.", "condition": {"operation": "register_help_agent", "purpose_present": False}, "effect": {"decision": "deny", "reason": "help_agent_purpose_required", "required_action": "document_help_agent_purpose"}},
	{"name": "help_agent_requires_contribution_disclosure", "description": "First-class help agents must disclose machine-authored source, article, answer, feedback, localization, curation, and lifecycle contributions.", "condition": {"operation": "register_help_agent", "contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "help_agent_contribution_disclosure_required", "required_action": "disclose_machine_contribution"}},
	{"name": "help_agent_privileged_role_requires_human_approval", "description": "Privileged help-agent roles require human approval evidence.", "condition": {"operation": "register_help_agent", "privileged_role": True, "human_approval_required": False}, "effect": {"decision": "require_review", "reason": "help_agent_human_approval_required", "required_action": "record_human_help_agent_approval"}},
	{"name": "help_lifecycle_batch_requires_mutations", "description": "HELP lifecycle batches must include at least one mutation.", "condition": {"operation": "validate_help_lifecycle_batch", "mutation_count_lte": 0}, "effect": {"decision": "deny", "reason": "help_lifecycle_batch_empty", "required_action": "include_help_lifecycle_mutations"}},
	{"name": "help_lifecycle_operation_supported", "description": "HELP lifecycle batches must use configured lifecycle operations.", "condition": {"operation": "validate_help_lifecycle_batch", "lifecycle_operation_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_help_lifecycle_operation", "required_action": "choose_supported_help_lifecycle_operation"}},
	{"name": "bytewax_help_stream_required", "description": "HELP lifecycle batches must be routed through Bytewax.", "condition": {"operation": "validate_help_lifecycle_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_lifecycle_stream_required", "required_action": "route_help_lifecycle_batch_to_bytewax"}},
	{"name": "help_state_change_requires_audit", "description": "Help state changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "help_audit_event_required", "required_action": "record_help_audit_event"}},
	{"name": "cross_tenant_help_access_denied", "description": "Help records may not cross tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_help_access_denied", "required_action": "use_tenant_local_context"}},
	{"name": "batch_help_mutation_requires_bytewax", "description": "Batch help mutations must use Bytewax event streams.", "condition": {"operation": "batch_help_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/help/dashboard", "component": "HELPDashboard", "permission": "help:view", "nav_group": "Overview"},
	{"name": "home", "path": "/help/home", "component": "HelpCenter", "permission": "help:view", "nav_group": "Help"},
	{"name": "articles", "path": "/help/articles", "component": "ArticleLibrary", "permission": "help:view", "nav_group": "Help"},
	{"name": "editor", "path": "/help/editor", "component": "ArticleEditor", "permission": "help:edit_articles", "nav_group": "Authoring"},
	{"name": "sources", "path": "/help/sources", "component": "SourceRegistry", "permission": "help:publish", "nav_group": "Authoring"},
	{"name": "answers", "path": "/help/answers", "component": "AnswerConsole", "permission": "help:ask", "nav_group": "Assistant"},
	{"name": "localization", "path": "/help/localization", "component": "LocalizationWorkbench", "permission": "help:edit_articles", "nav_group": "Authoring"},
	{"name": "curation", "path": "/help/curation", "component": "CurationQueue", "permission": "help:publish", "nav_group": "Governance"},
	{"name": "agents", "path": "/help/agents", "component": "HelpAgentRoster", "permission": "help:publish", "nav_group": "Governance"},
	{"name": "lifecycle", "path": "/help/lifecycle", "component": "HELPLifecycleBatchMonitor", "permission": "help:admin", "nav_group": "Operations"},
	{"name": "audit", "path": "/help/audit", "component": "HelpAuditTrail", "permission": "help:audit", "nav_group": "Governance"},
	{"name": "analytics", "path": "/help/analytics", "component": "SupportAnalytics", "permission": "help:view", "nav_group": "Operations"},
	{"name": "settings", "path": "/help/settings", "component": "HELPSettings", "permission": "help:admin", "nav_group": "Administration"},
]

THEME: dict[str, Any] = {
	"name": "help_support_knowledge",
	"tokens": {
		"color.primary": "#28536B",
		"color.accent": "#38A169",
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
		"article_library": {"icon": "book-open", "status_indicator": "article-pill", "risk_style": "freshness-band"},
		"source_registry": {"visual": "source-table", "status_style": "approval-chip"},
		"answer_panel": {"visual": "cited-answer", "highlight": "confidence-chip"},
		"localization_workbench": {"visual": "locale-grid", "status_style": "locale-chip"},
		"curation_queue": {"visual": "review-list", "status_style": "approval-chip"},
		"feedback_table": {"visual": "feedback-grid", "status_style": "sentiment-chip"},
		"help_agent_roster": {"icon": "bot", "visual": "agent-roster", "status_style": "approval-chip"},
		"bytewax_lifecycle_panel": {"icon": "activity", "visual": "lifecycle-batch-list", "status_style": "stream-chip"},
		"audit_timeline": {"visual": "event-timeline", "status_style": "help-chip"},
	},
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable HELP capability contract."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "help",
		"display_name": "Help and Knowledge Base",
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": config["adapters"]["view_models"],
			"api_prefix": "/help/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True,
		},
		"theme": deepcopy(THEME),
		"agents": agent_manifest(config),
		"streaming": streaming_manifest(config),
	}


def agent_manifest(config: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return provider-neutral help-agent composition metadata."""
	agents = (config or DEFAULT_CONFIGURATION)["agents"]
	return {
		"first_class": bool(agents["first_class"]),
		"supported_runtimes": list(agents["supported_runtimes"]),
		"supported_roles": list(agents["supported_roles"]),
		"privileged_roles": list(agents["privileged_roles"]),
		"required_fields": ["tenant_id", "agent_id", "name", "runtime", "role", "scope", "owner", "purpose"],
		"guardrails": [
			"supported_runtime",
			"supported_role",
			"stable_identifier",
			"readable_name",
			"explicit_scope",
			"accountable_owner",
			"declared_purpose",
			"machine_contribution_disclosure",
			"human_approval_for_privileged_roles",
		],
		"adapter_contract": agents["adapter_contract"],
	}


def streaming_manifest(config: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return Bytewax lifecycle stream metadata for help composition."""
	streaming = (config or DEFAULT_CONFIGURATION)["streaming"]
	return {
		"engine": streaming["engine"],
		"lifecycle_stream": streaming["lifecycle_stream"],
		"watermark": streaming["watermark"],
		"required_processor": streaming["required_processor"],
		"required_operations": list(streaming["required_operations"]),
		"topics": list(streaming["topics"]),
		"broker_core_dependency_allowed": bool(streaming["broker_core_dependency_allowed"]),
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate default HELP governance rules."""
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
