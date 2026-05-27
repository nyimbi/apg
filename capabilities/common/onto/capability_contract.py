"""Executable capability contract for APG Ontology Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"ontology": {"versioning_enabled": True, "owner_required": True, "publication_approval_required": True},
	"vocabulary": {"duplicate_detection_enabled": True, "synonym_management_enabled": True, "term_status_required": True},
	"mapping": {"confidence_threshold": 0.8, "external_mapping_review_required": True, "breaking_change_review_required": True},
	"governance": {"require_tenant_context": True, "audit_term_changes": True, "curation_required": True},
	"ui": {"enable_ontology_registry": True, "enable_term_editor": True, "enable_mapping_workbench": True, "enable_publication_queue": True},
	"theme": {"default_theme": "onto_vocabulary_workbench", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {"type": "object", "required": ["tenant_id", "ontology", "vocabulary", "mapping", "governance", "ui", "theme"], "properties": {key: {"type": "object"} for key in ["ontology", "vocabulary", "mapping", "governance", "ui", "theme"]} | {"tenant_id": {"type": "string", "minLength": 1}}}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All ontology operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "term_requires_owner", "description": "Ontology terms require an owner.", "condition": {"operation": "create_term", "owner_assigned": False}, "effect": {"decision": "deny", "reason": "term_owner_required", "required_action": "assign_term_owner"}},
	{"name": "publication_requires_approval", "description": "Ontology publication requires approval.", "condition": {"operation": "publish_ontology", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "publication_approval_required", "required_action": "record_publication_approval"}},
	{"name": "breaking_change_requires_review", "description": "Breaking ontology changes require review.", "condition": {"change_type": "breaking", "review_recorded": False}, "effect": {"decision": "require_review", "reason": "breaking_change_review_required", "required_action": "record_breaking_change_review"}},
	{"name": "low_confidence_mapping_requires_review", "description": "Low-confidence semantic mappings require review.", "condition": {"mapping_confidence_lt": 0.8, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "mapping_review_required", "required_action": "record_mapping_review"}},
	{"name": "duplicate_term_blocks_publication", "description": "Duplicate terms block ontology publication.", "condition": {"operation": "publish_ontology", "duplicate_term_detected": True}, "effect": {"decision": "deny", "reason": "duplicate_term_detected", "required_action": "resolve_duplicate_term"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/onto/dashboard", "component": "ONTODashboard", "permission": "onto:view", "nav_group": "Overview"},
	{"name": "ontologies", "path": "/onto/ontologies", "component": "OntologyRegistry", "permission": "onto:view", "nav_group": "Registry"},
	{"name": "terms", "path": "/onto/terms", "component": "TermEditor", "permission": "onto:edit", "nav_group": "Vocabulary"},
	{"name": "mappings", "path": "/onto/mappings", "component": "MappingWorkbench", "permission": "onto:map", "nav_group": "Mappings"},
	{"name": "publication", "path": "/onto/publication", "component": "PublicationQueue", "permission": "onto:publish", "nav_group": "Governance"},
	{"name": "governance", "path": "/onto/governance", "component": "OntologyGovernance", "permission": "onto:govern", "nav_group": "Governance"},
	{"name": "settings", "path": "/onto/settings", "component": "ONTOSettings", "permission": "onto:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {"name": "onto_vocabulary_workbench", "tokens": {"color.primary": "#4B5563", "color.accent": "#7A9E7E", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F6F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"}, "components": {"term_card": {"icon": "book-open", "status_indicator": "term-status-pill", "risk_style": "publication-band"}, "taxonomy_tree": {"visual": "hierarchy-tree", "highlight": "selected-term-chip"}, "mapping_panel": {"visual": "concept-map", "threshold_style": "confidence-band"}, "publication_queue": {"visual": "approval-list", "status_style": "review-chip"}}}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "onto", "display_name": "Ontology Management", "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "view_module": "__init__.py", "api_prefix": "/onto/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME)}


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
