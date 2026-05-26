"""Executable capability contract for APG Retrieval-Augmented Generation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"knowledge_bases": {"owner_required": True, "source_attribution_required": True, "max_documents_per_ingest": 5000},
	"retrieval": {"semantic_retrieval_enabled": True, "keyword_fallback_enabled": True, "minimum_context_confidence": 0.7},
	"generation": {"citations_required": True, "model_policy_required": True, "streaming_enabled": True},
	"governance": {"require_tenant_context": True, "audit_queries": True, "restricted_source_filter_required": True},
	"ui": {"enable_rag_studio": True, "enable_knowledge_bases": True, "enable_conversations": True, "enable_curation": True},
	"theme": {"default_theme": "ragn_answer_studio", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {"type": "object", "required": ["tenant_id", "knowledge_bases", "retrieval", "generation", "governance", "ui", "theme"], "properties": {key: {"type": "object"} for key in ["knowledge_bases", "retrieval", "generation", "governance", "ui", "theme"]} | {"tenant_id": {"type": "string", "minLength": 1}}}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All RAG operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "knowledge_base_requires_owner", "description": "Knowledge bases require an owner.", "condition": {"operation": "create_knowledge_base", "owner_assigned": False}, "effect": {"decision": "deny", "reason": "knowledge_base_owner_required", "required_action": "assign_owner"}},
	{"name": "restricted_sources_require_filter", "description": "Restricted sources require access filters.", "condition": {"source_classification": "restricted", "access_filter_applied": False}, "effect": {"decision": "deny", "reason": "access_filter_required", "required_action": "apply_access_filter"}},
	{"name": "generation_requires_citations", "description": "Generated answers require citations.", "condition": {"operation": "generate_answer", "citations_attached": False}, "effect": {"decision": "deny", "reason": "citations_required", "required_action": "attach_citations"}},
	{"name": "low_context_confidence_requires_review", "description": "Low confidence context requires review.", "condition": {"context_confidence_lt": 0.7, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "low_context_confidence_review_required", "required_action": "record_context_review"}},
	{"name": "external_model_requires_policy", "description": "External generation models require policy approval.", "condition": {"model_location": "external", "model_policy_attached": False}, "effect": {"decision": "deny", "reason": "model_policy_required", "required_action": "attach_model_policy"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/ragn/dashboard", "component": "RAGNDashboard", "permission": "ragn:view", "nav_group": "Overview"},
	{"name": "studio", "path": "/ragn/studio", "component": "RAGStudio", "permission": "ragn:query", "nav_group": "Ask"},
	{"name": "knowledge_bases", "path": "/ragn/knowledge-bases", "component": "KnowledgeBaseManager", "permission": "ragn:manage_kb", "nav_group": "Knowledge"},
	{"name": "documents", "path": "/ragn/documents", "component": "DocumentIngestion", "permission": "ragn:manage_kb", "nav_group": "Knowledge"},
	{"name": "conversations", "path": "/ragn/conversations", "component": "ConversationMemory", "permission": "ragn:query", "nav_group": "Ask"},
	{"name": "curation", "path": "/ragn/curation", "component": "KnowledgeCuration", "permission": "ragn:curate", "nav_group": "Governance"},
	{"name": "settings", "path": "/ragn/settings", "component": "RAGNSettings", "permission": "ragn:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {"name": "ragn_answer_studio", "tokens": {"color.primary": "#324A5F", "color.accent": "#F2A541", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F6F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"}, "components": {"answer_panel": {"icon": "message-square-text", "status_indicator": "citation-pill", "risk_style": "grounding-band"}, "source_stack": {"visual": "evidence-list", "highlight": "source-chip"}, "conversation_trace": {"visual": "turn-timeline", "status_style": "memory-chip"}, "retrieval_debug": {"visual": "ranked-results", "threshold_style": "confidence-band"}}}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "ragn", "display_name": "Retrieval-Augmented Generation", "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "flask_appbuilder", "view_module": "views.py", "api_prefix": "/ragn/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME)}


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
		if key.endswith("_lt"):
			if not context.get(key[:-3], 0) < expected:
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
