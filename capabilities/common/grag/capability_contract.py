"""Executable capability contract for APG Graph-based RAG."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"retrieval": {"hybrid_retrieval_enabled": True, "vector_index_required": True, "graph_index_required": True},
	"reasoning": {"max_hops": 4, "evidence_required": True, "explanation_required": True},
	"curation": {"expert_review_required": True, "confidence_threshold": 0.75, "provenance_required": True},
	"governance": {"require_tenant_context": True, "audit_reasoning": True, "graph_access_filter_required": True},
	"ui": {"enable_query_console": True, "enable_reasoning_paths": True, "enable_graph_curation": True, "enable_explanations": True},
	"theme": {"default_theme": "grag_reasoning_console", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {"type": "object", "required": ["tenant_id", "retrieval", "reasoning", "curation", "governance", "ui", "theme"], "properties": {key: {"type": "object"} for key in ["retrieval", "reasoning", "curation", "governance", "ui", "theme"]} | {"tenant_id": {"type": "string", "minLength": 1}}}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All GraphRAG operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "hybrid_query_requires_vector_and_graph", "description": "Hybrid GraphRAG requires vector and graph indexes.", "condition": {"query_type": "hybrid", "vector_index_ready": False}, "effect": {"decision": "deny", "reason": "vector_index_required", "required_action": "build_vector_index"}},
	{"name": "hybrid_query_requires_graph_index", "description": "Hybrid GraphRAG requires graph index readiness.", "condition": {"query_type": "hybrid", "graph_index_ready": False}, "effect": {"decision": "deny", "reason": "graph_index_required", "required_action": "build_graph_index"}},
	{"name": "reasoning_requires_evidence_path", "description": "Graph reasoning requires evidence paths.", "condition": {"operation": "reason", "evidence_path_present": False}, "effect": {"decision": "deny", "reason": "evidence_path_required", "required_action": "attach_evidence_path"}},
	{"name": "multi_hop_requires_review", "description": "Deep multi-hop reasoning requires review.", "condition": {"hop_count_gt": 4, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "multi_hop_review_required", "required_action": "record_reasoning_review"}},
	{"name": "answer_requires_provenance", "description": "Graph-grounded answers require provenance.", "condition": {"operation": "generate_answer", "provenance_attached": False}, "effect": {"decision": "deny", "reason": "provenance_required", "required_action": "attach_provenance"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/grag/dashboard", "component": "GRAGDashboard", "permission": "grag:view", "nav_group": "Overview"},
	{"name": "query", "path": "/grag/query", "component": "GraphRAGQuery", "permission": "grag:query", "nav_group": "Ask"},
	{"name": "reasoning", "path": "/grag/reasoning", "component": "ReasoningPathExplorer", "permission": "grag:reason", "nav_group": "Reasoning"},
	{"name": "graphs", "path": "/grag/graphs", "component": "GraphContextManager", "permission": "grag:manage_graphs", "nav_group": "Graphs"},
	{"name": "curation", "path": "/grag/curation", "component": "GraphCuration", "permission": "grag:curate", "nav_group": "Curation"},
	{"name": "explanations", "path": "/grag/explanations", "component": "ExplanationWorkbench", "permission": "grag:reason", "nav_group": "Reasoning"},
	{"name": "settings", "path": "/grag/settings", "component": "GRAGSettings", "permission": "grag:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {"name": "grag_reasoning_console", "tokens": {"color.primary": "#2F4858", "color.accent": "#86BBD8", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F6F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"}, "components": {"hybrid_result": {"icon": "network", "status_indicator": "fusion-pill", "risk_style": "evidence-band"}, "reasoning_path": {"visual": "multi-hop-path", "highlight": "hop-chip"}, "provenance_panel": {"visual": "source-graph", "status_style": "confidence-pill"}, "curation_queue": {"visual": "review-list", "threshold_style": "quality-band"}}}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "grag", "display_name": "Graph-based RAG", "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "view_module": "views.py", "api_prefix": "/grag/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME)}


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
		if key.endswith("_gt"):
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
