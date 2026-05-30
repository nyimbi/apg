"""Executable capability contract for APG Knowledge Graph."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"sources": {
		"registered_sources_required": True,
		"owner_required": True,
		"source_uri_required": True,
		"evidence_required": True,
		"minimum_confidence": 0.7,
		"low_confidence_review_required": True,
	},
	"entities": {
		"stable_id_required": True,
		"canonical_label_required": True,
		"entity_type_required": True,
		"source_required": True,
		"evidence_required": True,
		"curation_required_for_publication": True,
	},
	"relationships": {
		"subject_required": True,
		"object_required": True,
		"predicate_required": True,
		"source_required": True,
		"evidence_required": True,
		"minimum_confidence": 0.7,
		"low_confidence_review_required": True,
	},
	"enrichment": {
		"semantic_labels_required": True,
		"evidence_required": True,
		"minimum_confidence": 0.7,
		"low_confidence_review_required": True,
		"nlpc_adapter": "nlpc",
	},
	"reasoning": {
		"bounded_reasoning_enabled": True,
		"max_reasoning_depth": 5,
		"query_required": True,
		"entity_endpoints_required": True,
		"evidence_required": True,
		"deep_reasoning_review_required": True,
	},
	"curation": {
		"curator_required": True,
		"allowed_decisions": ["approved", "rejected", "needs_revision"],
		"evidence_required": True,
		"approved_entities_publishable": True,
	},
	"publication": {
		"curation_required": True,
		"publisher_required": True,
		"minimum_entity_count": 1,
		"relationship_validation_required": True,
	},
	"security": {
		"cross_tenant_graph_access_allowed": False,
		"rbac_filter_required": True,
		"public_graph_requires_curation": True,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_source_registration": True,
		"audit_entity_resolution": True,
		"audit_relationship_links": True,
		"audit_enrichment": True,
		"audit_reasoning": True,
		"audit_publication": True,
	},
	"observability": {
		"metrics_required": True,
		"trace_required": True,
		"audit_required": True,
		"event_stream": "bytewax",
	},
	"adapters": {
		"generated_app_runtime": "service.KngrService",
		"helper_runtime": "knowledge_runtime.py",
		"production_runtime": "service.KngrService",
		"http_api": "api.py",
		"event_stream": "bytewax",
		"graph": "grph",
		"nlp": "nlpc",
		"metadata": "meta",
		"search": "srch",
		"ontology": "onto",
		"ai_core": "aicr",
		"auth_provider": "auth",
		"audit_sink": "audl",
		"cache": "cach",
		"metrics_sink": "moni",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_source_manager": True,
		"enable_entity_browser": True,
		"enable_relationship_browser": True,
		"enable_enrichment_console": True,
		"enable_reasoning_paths": True,
		"enable_context_explorer": True,
		"enable_entity_curation": True,
		"enable_publication_console": True,
		"enable_governance": True,
		"enable_audit": True,
		"enable_settings": True,
	},
	"theme": {"default_theme": "kngr_semantic_console", "allow_tenant_overrides": True},
}


CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"sources",
		"entities",
		"relationships",
		"enrichment",
		"reasoning",
		"curation",
		"publication",
		"security",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {key: {"type": "object"} for key in [
		"sources",
		"entities",
		"relationships",
		"enrichment",
		"reasoning",
		"curation",
		"publication",
		"security",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]} | {"tenant_id": {"type": "string", "minLength": 1}},
}


RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All knowledge graph operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "source_requires_id", "description": "Knowledge sources require stable identifiers.", "condition": {"operation": "register_source", "source_id_present": False}, "effect": {"decision": "deny", "reason": "source_id_required", "required_action": "attach_source_id"}},
	{"name": "source_requires_name", "description": "Knowledge sources require display names.", "condition": {"operation": "register_source", "source_name_present": False}, "effect": {"decision": "deny", "reason": "source_name_required", "required_action": "attach_source_name"}},
	{"name": "source_requires_uri", "description": "Knowledge sources require a resolvable source URI.", "condition": {"operation": "register_source", "source_uri_present": False}, "effect": {"decision": "deny", "reason": "source_uri_required", "required_action": "attach_source_uri"}},
	{"name": "source_requires_owner", "description": "Knowledge sources require an accountable owner.", "condition": {"operation": "register_source", "source_owner_present": False}, "effect": {"decision": "deny", "reason": "source_owner_required", "required_action": "assign_source_owner"}},
	{"name": "source_requires_evidence", "description": "Knowledge sources require evidence references.", "condition": {"operation": "register_source", "source_evidence_present": False}, "effect": {"decision": "deny", "reason": "source_evidence_required", "required_action": "attach_source_evidence"}},
	{"name": "source_requires_confidence", "description": "Knowledge sources require a positive confidence score.", "condition": {"operation": "register_source", "confidence_score_lte": 0}, "effect": {"decision": "deny", "reason": "source_confidence_required", "required_action": "attach_source_confidence"}},
	{"name": "source_confidence_requires_review", "description": "Low-confidence knowledge sources require review.", "condition": {"operation": "register_source", "confidence_score_lt": 0.7, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "low_confidence_source_review_required", "required_action": "record_source_review"}},
	{"name": "entity_requires_source", "description": "Entity resolution requires a registered source.", "condition": {"operation": "resolve_entity", "source_present": False}, "effect": {"decision": "deny", "reason": "source_required", "required_action": "select_registered_source"}},
	{"name": "entity_requires_id", "description": "Resolved entities require stable identifiers.", "condition": {"operation": "resolve_entity", "entity_id_present": False}, "effect": {"decision": "deny", "reason": "entity_id_required", "required_action": "attach_entity_id"}},
	{"name": "entity_requires_label", "description": "Resolved entities require canonical labels.", "condition": {"operation": "resolve_entity", "canonical_label_present": False}, "effect": {"decision": "deny", "reason": "canonical_label_required", "required_action": "attach_canonical_label"}},
	{"name": "entity_requires_type", "description": "Resolved entities require entity types.", "condition": {"operation": "resolve_entity", "entity_type_present": False}, "effect": {"decision": "deny", "reason": "entity_type_required", "required_action": "attach_entity_type"}},
	{"name": "entity_resolution_requires_source", "description": "Entity resolution requires source asset evidence.", "condition": {"operation": "resolve_entity", "source_evidence_present": False}, "effect": {"decision": "deny", "reason": "source_evidence_required", "required_action": "attach_source_evidence"}},
	{"name": "entity_confidence_requires_review", "description": "Low-confidence entity resolution requires review.", "condition": {"operation": "resolve_entity", "confidence_score_lt": 0.7, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "low_confidence_entity_review_required", "required_action": "record_entity_review"}},
	{"name": "relationship_requires_subject", "description": "Relationship links require a subject entity.", "condition": {"operation": "link_relationship", "subject_present": False}, "effect": {"decision": "deny", "reason": "relationship_subject_required", "required_action": "attach_subject_entity"}},
	{"name": "relationship_requires_object", "description": "Relationship links require an object entity.", "condition": {"operation": "link_relationship", "object_present": False}, "effect": {"decision": "deny", "reason": "relationship_object_required", "required_action": "attach_object_entity"}},
	{"name": "relationship_requires_predicate", "description": "Relationship links require a predicate.", "condition": {"operation": "link_relationship", "predicate_present": False}, "effect": {"decision": "deny", "reason": "predicate_required", "required_action": "attach_predicate"}},
	{"name": "relationship_requires_source", "description": "Relationship links require a registered source.", "condition": {"operation": "link_relationship", "source_present": False}, "effect": {"decision": "deny", "reason": "relationship_source_required", "required_action": "select_registered_source"}},
	{"name": "relationship_requires_evidence", "description": "Relationship links require evidence.", "condition": {"operation": "link_relationship", "evidence_links_present": False}, "effect": {"decision": "deny", "reason": "relationship_evidence_required", "required_action": "attach_relationship_evidence"}},
	{"name": "relationship_confidence_requires_review", "description": "Low-confidence relationship links require review.", "condition": {"operation": "link_relationship", "confidence_score_lt": 0.7, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "low_confidence_relationship_review_required", "required_action": "record_relationship_review"}},
	{"name": "semantic_enrichment_requires_labels", "description": "Semantic enrichment requires labels.", "condition": {"operation": "enrich", "semantic_labels_present": False}, "effect": {"decision": "deny", "reason": "semantic_labels_required", "required_action": "attach_semantic_labels"}},
	{"name": "semantic_enrichment_requires_evidence", "description": "Semantic enrichment requires evidence links.", "condition": {"operation": "enrich", "evidence_links_present": False}, "effect": {"decision": "deny", "reason": "enrichment_evidence_required", "required_action": "attach_enrichment_evidence"}},
	{"name": "semantic_enrichment_requires_confidence", "description": "Low-confidence semantic enrichment requires review.", "condition": {"operation": "enrich", "confidence_score_lt": 0.7, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "low_confidence_enrichment_review_required", "required_action": "record_enrichment_review"}},
	{"name": "reasoning_requires_query", "description": "Reasoning paths require a natural-language or structured query.", "condition": {"operation": "reason", "query_present": False}, "effect": {"decision": "deny", "reason": "reasoning_query_required", "required_action": "attach_reasoning_query"}},
	{"name": "reasoning_requires_entities", "description": "Reasoning paths require start and end entities.", "condition": {"operation": "reason", "entity_endpoints_present": False}, "effect": {"decision": "deny", "reason": "reasoning_entities_required", "required_action": "attach_reasoning_entities"}},
	{"name": "reasoning_requires_evidence", "description": "Reasoning paths require evidence links.", "condition": {"operation": "reason", "evidence_links_present": False}, "effect": {"decision": "deny", "reason": "reasoning_evidence_required", "required_action": "attach_evidence_links"}},
	{"name": "deep_reasoning_requires_review", "description": "Deep reasoning paths require review.", "condition": {"operation": "reason", "reasoning_depth_gt": 5, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "deep_reasoning_review_required", "required_action": "record_reasoning_review"}},
	{"name": "curation_requires_curator", "description": "Curation decisions require a curator.", "condition": {"operation": "curate_entity", "curator_present": False}, "effect": {"decision": "deny", "reason": "curator_required", "required_action": "assign_curator"}},
	{"name": "curation_requires_decision", "description": "Curation decisions require a valid decision.", "condition": {"operation": "curate_entity", "curation_decision_present": False}, "effect": {"decision": "deny", "reason": "curation_decision_required", "required_action": "choose_curation_decision"}},
	{"name": "curation_decision_requires_allowed_value", "description": "Curation decisions must use configured values.", "condition": {"operation": "curate_entity", "curation_decision_allowed": False}, "effect": {"decision": "deny", "reason": "curation_decision_invalid", "required_action": "choose_allowed_curation_decision"}},
	{"name": "curation_requires_evidence", "description": "Curation decisions require evidence links.", "condition": {"operation": "curate_entity", "evidence_links_present": False}, "effect": {"decision": "deny", "reason": "curation_evidence_required", "required_action": "attach_curation_evidence"}},
	{"name": "publication_requires_name", "description": "Graph publications require a name.", "condition": {"operation": "publish_graph", "publication_name_present": False}, "effect": {"decision": "deny", "reason": "publication_name_required", "required_action": "attach_publication_name"}},
	{"name": "publication_requires_publisher", "description": "Graph publications require an accountable publisher.", "condition": {"operation": "publish_graph", "publisher_present": False}, "effect": {"decision": "deny", "reason": "publisher_required", "required_action": "assign_publisher"}},
	{"name": "uncurated_public_graph_blocked", "description": "Public graph publication requires curation.", "condition": {"operation": "publish_graph", "curation_recorded": False}, "effect": {"decision": "deny", "reason": "curation_required", "required_action": "record_curation"}},
	{"name": "publication_requires_entities", "description": "Graph publications require at least one curated entity.", "condition": {"operation": "publish_graph", "entity_count_lt": 1}, "effect": {"decision": "deny", "reason": "publication_entities_required", "required_action": "select_curated_entities"}},
	{"name": "batch_knowledge_mutation_requires_bytewax", "description": "Batch knowledge graph mutations must use Bytewax event streams.", "condition": {"operation": "batch_knowledge_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
	{"name": "cross_tenant_graph_access_denied", "description": "Knowledge graph operations may not cross tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_graph_access_denied", "required_action": "use_tenant_local_graph"}},
	{"name": "graph_state_change_requires_audit", "description": "Knowledge graph state changes require audit events.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "audit_event_required", "required_action": "record_audit_event"}},
]


UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/kngr/dashboard", "component": "KNGRDashboard", "permission": "kngr:view", "nav_group": "Overview"},
	{"name": "sources", "path": "/kngr/sources", "component": "KnowledgeSourceManager", "permission": "kngr:source", "nav_group": "Knowledge"},
	{"name": "entities", "path": "/kngr/entities", "component": "EntityBrowser", "permission": "kngr:query", "nav_group": "Knowledge"},
	{"name": "relationships", "path": "/kngr/relationships", "component": "RelationshipBrowser", "permission": "kngr:query", "nav_group": "Knowledge"},
	{"name": "enrichment", "path": "/kngr/enrichment", "component": "SemanticEnrichmentConsole", "permission": "kngr:enrich", "nav_group": "Knowledge"},
	{"name": "reasoning", "path": "/kngr/reasoning", "component": "ReasoningPaths", "permission": "kngr:reason", "nav_group": "Reasoning"},
	{"name": "context", "path": "/kngr/context", "component": "ContextExplorer", "permission": "kngr:query", "nav_group": "Context"},
	{"name": "curation", "path": "/kngr/curation", "component": "EntityCuration", "permission": "kngr:curate", "nav_group": "Curation"},
	{"name": "publication", "path": "/kngr/publication", "component": "GraphPublicationConsole", "permission": "kngr:publish", "nav_group": "Publication"},
	{"name": "governance", "path": "/kngr/governance", "component": "KnowledgeGovernance", "permission": "kngr:govern", "nav_group": "Governance"},
	{"name": "audit", "path": "/kngr/audit", "component": "KnowledgeAuditTimeline", "permission": "kngr:audit", "nav_group": "Governance"},
	{"name": "settings", "path": "/kngr/settings", "component": "KNGRSettings", "permission": "kngr:admin", "nav_group": "Administration"},
]


THEME: dict[str, Any] = {
	"name": "kngr_semantic_console",
	"tokens": {
		"color.primary": "#3A506B",
		"color.accent": "#6A994E",
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
		"source_panel": {"icon": "database", "status_indicator": "source-pill", "risk_style": "confidence-band"},
		"entity_card": {"icon": "badge", "status_indicator": "curation-pill", "risk_style": "confidence-band"},
		"relationship_panel": {"icon": "git-branch", "highlight": "predicate-chip", "status_style": "review-chip"},
		"semantic_graph": {"visual": "knowledge-network", "highlight": "entity-chip"},
		"enrichment_panel": {"visual": "label-stack", "status_style": "confidence-chip"},
		"reasoning_path": {"visual": "evidence-path", "threshold_style": "depth-band"},
		"curation_queue": {"visual": "review-list", "status_style": "decision-chip"},
		"publication_card": {"visual": "snapshot-summary", "status_style": "release-chip"},
		"context_panel": {"visual": "neighborhood-list", "status_style": "source-pill"},
		"audit_timeline": {"visual": "event-timeline", "status_style": "evidence-chip"},
	},
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "kngr",
		"display_name": "Knowledge Graph",
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/kngr/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True,
		},
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
		elif key.endswith("_gt"):
			actual = context.get(key[:-3])
			if not isinstance(actual, Number) or not actual > expected:
				return False
		elif key.endswith("_gte"):
			actual = context.get(key[:-4])
			if not isinstance(actual, Number) or not actual >= expected:
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
