"""Executable capability contract for APG Knowledge Graph."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"knowledge": {"entity_resolution_enabled": True, "semantic_enrichment_enabled": True, "curation_required": True},
	"reasoning": {"bounded_reasoning_enabled": True, "max_reasoning_depth": 5, "evidence_required": True},
	"governance": {"require_tenant_context": True, "audit_enrichment": True, "source_confidence_required": True},
	"ui": {"enable_graph_browser": True, "enable_entity_curation": True, "enable_reasoning_paths": True, "enable_context_explorer": True},
	"theme": {"default_theme": "kngr_semantic_console", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {"type": "object", "required": ["tenant_id", "knowledge", "reasoning", "governance", "ui", "theme"], "properties": {key: {"type": "object"} for key in ["knowledge", "reasoning", "governance", "ui", "theme"]} | {"tenant_id": {"type": "string", "minLength": 1}}}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All knowledge graph operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "entity_resolution_requires_source", "description": "Entity resolution requires source asset evidence.", "condition": {"operation": "resolve_entity", "source_evidence_present": False}, "effect": {"decision": "deny", "reason": "source_evidence_required", "required_action": "attach_source_evidence"}},
	{"name": "semantic_enrichment_requires_confidence", "description": "Semantic enrichment requires a minimum confidence score.", "condition": {"operation": "enrich", "confidence_score_lt": 0.7}, "effect": {"decision": "require_review", "reason": "low_confidence_enrichment_review_required", "required_action": "record_enrichment_review"}},
	{"name": "reasoning_requires_evidence", "description": "Reasoning paths require evidence links.", "condition": {"operation": "reason", "evidence_links_present": False}, "effect": {"decision": "deny", "reason": "reasoning_evidence_required", "required_action": "attach_evidence_links"}},
	{"name": "deep_reasoning_requires_review", "description": "Deep reasoning paths require review.", "condition": {"reasoning_depth_gt": 5, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "deep_reasoning_review_required", "required_action": "record_reasoning_review"}},
	{"name": "uncurated_public_graph_blocked", "description": "Public graph publication requires curation.", "condition": {"operation": "publish_graph", "curation_recorded": False}, "effect": {"decision": "deny", "reason": "curation_required", "required_action": "record_curation"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/kngr/dashboard", "component": "KNGRDashboard", "permission": "kngr:view", "nav_group": "Overview"},
	{"name": "entities", "path": "/kngr/entities", "component": "EntityBrowser", "permission": "kngr:query", "nav_group": "Knowledge"},
	{"name": "curation", "path": "/kngr/curation", "component": "EntityCuration", "permission": "kngr:curate", "nav_group": "Curation"},
	{"name": "reasoning", "path": "/kngr/reasoning", "component": "ReasoningPaths", "permission": "kngr:reason", "nav_group": "Reasoning"},
	{"name": "context", "path": "/kngr/context", "component": "ContextExplorer", "permission": "kngr:query", "nav_group": "Context"},
	{"name": "settings", "path": "/kngr/settings", "component": "KNGRSettings", "permission": "kngr:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {"name": "kngr_semantic_console", "tokens": {"color.primary": "#3A506B", "color.accent": "#6A994E", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F6F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"}, "components": {"entity_card": {"icon": "badge", "status_indicator": "curation-pill", "risk_style": "confidence-band"}, "semantic_graph": {"visual": "knowledge-network", "highlight": "entity-chip"}, "reasoning_path": {"visual": "evidence-path", "threshold_style": "depth-band"}, "context_panel": {"visual": "neighborhood-list", "status_style": "source-pill"}}}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "kngr", "display_name": "Knowledge Graph", "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "flask_appbuilder", "view_module": "__init__.py", "api_prefix": "/kngr/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME)}


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
