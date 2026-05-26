"""Executable capability contract for APG Help and Knowledge Base."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"content": {
		"article_owner_required": True,
		"publication_approval_required": True,
		"freshness_review_days": 90,
		"localization_supported": True
	},
	"answers": {
		"rag_enabled": True,
		"citations_required": True,
		"minimum_answer_confidence": 0.76,
		"unsafe_answer_blocking": True
	},
	"search": {
		"semantic_search_enabled": True,
		"restricted_content_filtering": True,
		"feedback_boosting_enabled": True,
		"query_logging_enabled": True
	},
	"governance": {
		"require_tenant_context": True,
		"audit_publication": True,
		"source_approval_required": True,
		"support_feedback_review_required": True
	},
	"ui": {
		"enable_help_center": True,
		"enable_article_editor": True,
		"enable_answer_console": True,
		"enable_curation_queue": True
	},
	"theme": {
		"default_theme": "help_support_knowledge",
		"allow_tenant_overrides": True
	}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "content", "answers", "search", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["content", "answers", "search", "governance", "ui", "theme"]} | {
		"tenant_id": {"type": "string", "minLength": 1}
	}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All help operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "article_requires_owner", "description": "Knowledge articles require accountable owners.", "condition": {"operation": "create_article", "article_owner_assigned": False}, "effect": {"decision": "deny", "reason": "article_owner_required", "required_action": "assign_article_owner"}},
	{"name": "publication_requires_approval", "description": "Articles require approval before publication.", "condition": {"operation": "publish_article", "publication_approved": False}, "effect": {"decision": "deny", "reason": "publication_approval_required", "required_action": "approve_publication"}},
	{"name": "answer_requires_citations", "description": "Generated answers require source citations.", "condition": {"operation": "generate_answer", "citations_present": False}, "effect": {"decision": "deny", "reason": "citations_required", "required_action": "attach_answer_citations"}},
	{"name": "restricted_content_requires_filtering", "description": "Restricted help content requires RBAC filtering.", "condition": {"restricted_content_present": True, "rbac_filter_applied": False}, "effect": {"decision": "deny", "reason": "rbac_filter_required", "required_action": "apply_rbac_filter"}},
	{"name": "stale_article_requires_review", "description": "Stale knowledge articles require curation review.", "condition": {"article_age_days_gt": 90, "freshness_review_recorded": False}, "effect": {"decision": "require_review", "reason": "freshness_review_required", "required_action": "review_article_freshness"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/help/dashboard", "component": "HELPDashboard", "permission": "help:view", "nav_group": "Overview"},
	{"name": "home", "path": "/help/home", "component": "HelpCenter", "permission": "help:view", "nav_group": "Help"},
	{"name": "articles", "path": "/help/articles", "component": "ArticleLibrary", "permission": "help:view", "nav_group": "Help"},
	{"name": "editor", "path": "/help/editor", "component": "ArticleEditor", "permission": "help:edit_articles", "nav_group": "Authoring"},
	{"name": "answers", "path": "/help/answers", "component": "AnswerConsole", "permission": "help:ask", "nav_group": "Assistant"},
	{"name": "curation", "path": "/help/curation", "component": "CurationQueue", "permission": "help:publish", "nav_group": "Governance"},
	{"name": "analytics", "path": "/help/analytics", "component": "SupportAnalytics", "permission": "help:view", "nav_group": "Operations"},
	{"name": "settings", "path": "/help/settings", "component": "HELPSettings", "permission": "help:admin", "nav_group": "Administration"}
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
		"density": "compact"
	},
	"components": {
		"article_library": {"icon": "book-open", "status_indicator": "article-pill", "risk_style": "freshness-band"},
		"answer_panel": {"visual": "cited-answer", "highlight": "confidence-chip"},
		"curation_queue": {"visual": "review-list", "status_style": "approval-chip"},
		"feedback_table": {"visual": "feedback-grid", "status_style": "sentiment-chip"}
	}
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
			"shell": "flask_appbuilder",
			"view_module": "views.py",
			"api_prefix": "/help/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True
		},
		"theme": deepcopy(THEME)
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
		if key.endswith("_lt"):
			if not context.get(key[:-3], 0) < expected:
				return False
		elif key.endswith("_gt"):
			if not context.get(key[:-3], 0) > expected:
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
