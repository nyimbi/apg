"""Executable capability contract for APG NLP Core."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


SUPPORTED_LANGUAGES: list[str] = [
	"en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "ar", "hi",
	"af", "aa", "ak", "am", "bm", "ee", "ff", "ha", "ig", "kr", "ki", "rw",
	"rn", "kg", "ln", "lg", "mg", "ny", "om", "sg", "sn", "so", "st", "sw",
	"ss", "ti", "ts", "tn", "tw", "ve", "wo", "xh", "yo", "zu", "kab", "kam",
	"luo", "mas", "mer", "mos", "nus", "suk", "tzm", "tig", "umb"
]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"processing": {
		"default_language": "auto",
		"supported_languages": SUPPORTED_LANGUAGES,
		"max_document_chars": 250000,
		"async_threshold_documents": 25,
		"language_detection_required": True
	},
	"tasks": {
		"enabled": [
			"text_classification",
			"sentiment_analysis",
			"entity_recognition",
			"semantic_search",
			"summarization",
			"text_generation",
			"pii_detection"
		],
		"generation_safety_required": True,
		"minimum_confidence_score": 0.75
	},
	"governance": {
		"require_tenant_context": True,
		"audit_processing": True,
		"pii_redaction_policy_required": True,
		"model_policy_required": True
	},
	"ui": {
		"enable_processing_console": True,
		"enable_annotation_workbench": True,
		"enable_model_registry": True,
		"enable_language_coverage": True
	},
	"theme": {
		"default_theme": "nlpc_text_intelligence",
		"allow_tenant_overrides": True
	}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "processing", "tasks", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["processing", "tasks", "governance", "ui", "theme"]} | {
		"tenant_id": {"type": "string", "minLength": 1}
	}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All NLP operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "language_required_or_detected", "description": "Processing requires a declared or detected language.", "condition": {"operation": "process_document", "language_known": False}, "effect": {"decision": "deny", "reason": "language_required", "required_action": "run_language_detection"}},
	{"name": "pii_requires_redaction_policy", "description": "PII extraction requires a redaction policy.", "condition": {"task": "pii_detection", "redaction_policy_attached": False}, "effect": {"decision": "deny", "reason": "pii_redaction_policy_required", "required_action": "attach_redaction_policy"}},
	{"name": "generation_requires_safety_policy", "description": "Text generation requires a model safety policy.", "condition": {"task": "text_generation", "safety_policy_attached": False}, "effect": {"decision": "deny", "reason": "generation_safety_policy_required", "required_action": "attach_safety_policy"}},
	{"name": "low_confidence_requires_review", "description": "Low-confidence NLP results require review.", "condition": {"confidence_score_lt": 0.75, "human_review_recorded": False}, "effect": {"decision": "require_review", "reason": "low_confidence_review_required", "required_action": "record_human_review"}},
	{"name": "large_batch_requires_async_queue", "description": "Large NLP batches must run through the async queue.", "condition": {"document_count_gt": 25, "async_queue_enabled": False}, "effect": {"decision": "require_review", "reason": "large_batch_requires_async_queue", "required_action": "enable_async_queue"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/nlpc/dashboard", "component": "NLPCDashboard", "permission": "nlpc:view", "nav_group": "Overview"},
	{"name": "process", "path": "/nlpc/process", "component": "ProcessingConsole", "permission": "nlpc:process", "nav_group": "Process"},
	{"name": "documents", "path": "/nlpc/documents", "component": "DocumentWorkbench", "permission": "nlpc:process", "nav_group": "Process"},
	{"name": "annotations", "path": "/nlpc/annotations", "component": "AnnotationWorkbench", "permission": "nlpc:annotate", "nav_group": "Quality"},
	{"name": "models", "path": "/nlpc/models", "component": "NLPModelRegistry", "permission": "nlpc:manage_models", "nav_group": "Models"},
	{"name": "languages", "path": "/nlpc/languages", "component": "LanguageCoverage", "permission": "nlpc:view", "nav_group": "Coverage"},
	{"name": "governance", "path": "/nlpc/governance", "component": "NLPGovernance", "permission": "nlpc:govern", "nav_group": "Governance"},
	{"name": "settings", "path": "/nlpc/settings", "component": "NLPCSettings", "permission": "nlpc:admin", "nav_group": "Administration"}
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
		"density": "compact"
	},
	"components": {
		"document_queue": {"icon": "file-text", "status_indicator": "processing-pill", "risk_style": "policy-band"},
		"language_coverage_map": {"visual": "coverage-grid", "highlight": "african-language-chip"},
		"annotation_panel": {"visual": "span-highlighter", "status_style": "review-chip"},
		"model_result_card": {"visual": "confidence-meter", "threshold_style": "quality-band"}
	}
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
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/nlpc/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True
		},
		"theme": deepcopy(THEME)
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
