"""Executable capability contract for APG capability registry."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


SUPPORTED_REGISTRY_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_REGISTRY_AGENT_ROLES = [
	"capability_curator",
	"dependency_reviewer",
	"composition_reviewer",
	"version_reviewer",
	"marketplace_reviewer",
	"security_reviewer",
]
REGISTRY_EVENT_STREAM = "apg.composition.registry.lifecycle"


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"catalog": {
		"owner_required": True,
		"category_required": True,
		"version_required": True,
		"provides_required": True,
		"contract_required": True,
	},
	"dependencies": {
		"target_required": True,
		"dependency_type_required": True,
		"version_constraint_required": True,
		"cycle_detection_enabled": True,
	},
	"composition_blueprints": {
		"owner_required": True,
		"capability_required": True,
		"validation_required": True,
		"publish_validation_required": True,
	},
	"versions": {
		"compatibility_evidence_required": True,
		"migration_plan_required_for_deprecation": True,
		"release_review_required": True,
	},
	"marketplace": {
		"publication_review_required": True,
		"security_review_required": True,
		"documentation_required": True,
	},
	"registry_agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_REGISTRY_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_REGISTRY_AGENT_ROLES,
		"human_approval_required": True,
		"max_autonomous_scope": "recommend_validate_and_prepare",
	},
	"governance": {
		"require_tenant_context": True,
		"audit_state_changes": True,
		"policy_attached_for_writes": True,
		"privileged_registry_changes_reviewed": True,
	},
	"observability": {
		"event_stream": REGISTRY_EVENT_STREAM,
		"stream_processor": "bytewax",
		"emit_catalog_events": True,
		"emit_dependency_events": True,
		"emit_composition_events": True,
		"emit_agent_events": True,
	},
	"adapters": {
		"authorization": "adapter",
		"audit": "adapter",
		"event_stream": "bytewax",
		"notification": "adapter",
		"search_index": "adapter",
		"theme": "adapter",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_catalog": True,
		"enable_dependencies": True,
		"enable_compositions": True,
		"enable_versions": True,
		"enable_marketplace": True,
		"enable_rules": True,
		"enable_agents": True,
		"enable_settings": True,
	},
	"theme": {"default_theme": "composition_registry_control", "allow_tenant_overrides": True},
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"catalog",
		"dependencies",
		"composition_blueprints",
		"versions",
		"marketplace",
		"registry_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {
		"tenant_id": {"type": "string", "minLength": 1},
		"catalog": {"type": "object"},
		"dependencies": {"type": "object"},
		"composition_blueprints": {"type": "object"},
		"versions": {"type": "object"},
		"marketplace": {"type": "object"},
		"registry_agents": {"type": "object"},
		"governance": {"type": "object"},
		"observability": {"type": "object"},
		"adapters": {"type": "object"},
		"ui": {"type": "object"},
		"theme": {"type": "object"},
	},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "Registry operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "registry_write_requires_policy", "description": "Registry write operations require policy attachment.", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "operation_policy_required", "required_action": "attach_operation_policy"}},
	{"name": "capability_requires_owner", "description": "Registered capabilities require an owner.", "condition": {"operation": "register_capability", "capability_owner_assigned": False}, "effect": {"decision": "deny", "reason": "capability_owner_required", "required_action": "assign_capability_owner"}},
	{"name": "capability_requires_category", "description": "Registered capabilities require a category.", "condition": {"operation": "register_capability", "capability_category_present": False}, "effect": {"decision": "deny", "reason": "capability_category_required", "required_action": "set_capability_category"}},
	{"name": "capability_requires_version", "description": "Registered capabilities require a version.", "condition": {"operation": "register_capability", "capability_version_present": False}, "effect": {"decision": "deny", "reason": "capability_version_required", "required_action": "set_capability_version"}},
	{"name": "capability_requires_provides", "description": "Registered capabilities require at least one provided surface.", "condition": {"operation": "register_capability", "capability_provides_present": False}, "effect": {"decision": "deny", "reason": "capability_provides_required", "required_action": "declare_provided_surfaces"}},
	{"name": "capability_requires_contract", "description": "Registered capabilities require an executable contract reference.", "condition": {"operation": "register_capability", "capability_contract_present": False}, "effect": {"decision": "deny", "reason": "capability_contract_required", "required_action": "attach_capability_contract"}},
	{"name": "dependency_requires_target", "description": "Dependencies require a target capability.", "condition": {"operation": "add_dependency", "dependency_target_present": False}, "effect": {"decision": "deny", "reason": "dependency_target_required", "required_action": "attach_dependency_target"}},
	{"name": "dependency_requires_type", "description": "Dependencies require a dependency type.", "condition": {"operation": "add_dependency", "dependency_type_present": False}, "effect": {"decision": "deny", "reason": "dependency_type_required", "required_action": "set_dependency_type"}},
	{"name": "dependency_requires_version_constraint", "description": "Dependencies require a version constraint.", "condition": {"operation": "add_dependency", "version_constraint_present": False}, "effect": {"decision": "deny", "reason": "dependency_version_constraint_required", "required_action": "set_version_constraint"}},
	{"name": "composition_requires_owner", "description": "Composition blueprints require an owner.", "condition": {"operation": "create_composition", "composition_owner_assigned": False}, "effect": {"decision": "deny", "reason": "composition_owner_required", "required_action": "assign_composition_owner"}},
	{"name": "composition_requires_capabilities", "description": "Composition blueprints require capabilities.", "condition": {"operation": "create_composition", "composition_capabilities_present": False}, "effect": {"decision": "deny", "reason": "composition_capabilities_required", "required_action": "add_capabilities_to_composition"}},
	{"name": "composition_publish_requires_validation", "description": "Published compositions require validation evidence.", "condition": {"operation": "publish_composition", "validation_evidence_present": False}, "effect": {"decision": "deny", "reason": "composition_validation_required", "required_action": "attach_composition_validation"}},
	{"name": "version_release_requires_compatibility", "description": "Capability version releases require compatibility evidence.", "condition": {"operation": "release_version", "compatibility_evidence_present": False}, "effect": {"decision": "deny", "reason": "compatibility_evidence_required", "required_action": "attach_compatibility_evidence"}},
	{"name": "deprecation_requires_migration_plan", "description": "Capability deprecation requires a migration plan.", "condition": {"operation": "deprecate_capability", "migration_plan_present": False}, "effect": {"decision": "deny", "reason": "migration_plan_required", "required_action": "attach_migration_plan"}},
	{"name": "marketplace_publish_requires_review", "description": "Marketplace publication requires review.", "condition": {"operation": "publish_marketplace", "review_recorded": False}, "effect": {"decision": "require_review", "reason": "marketplace_review_required", "required_action": "record_marketplace_review"}},
	{"name": "marketplace_publish_requires_documentation", "description": "Marketplace publication requires documentation.", "condition": {"operation": "publish_marketplace", "documentation_present": False}, "effect": {"decision": "deny", "reason": "marketplace_documentation_required", "required_action": "attach_marketplace_documentation"}},
	{"name": "registry_import_requires_bytewax", "description": "Registry import batches require Bytewax coordination.", "condition": {"operation": "registry_import", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_registry_import_to_bytewax"}},
	{"name": "registry_event_requires_bytewax", "description": "Registry lifecycle events require Bytewax.", "condition": {"operation": "registry_event", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_registry_event_to_bytewax"}},
	{"name": "registry_agent_runtime_supported", "description": "Registry agents must use an approved runtime.", "condition": {"operation": "register_registry_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "registry_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"}},
	{"name": "registry_agent_role_supported", "description": "Registry agents must use an approved role.", "condition": {"operation": "register_registry_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "registry_agent_role_not_supported", "required_action": "select_supported_agent_role"}},
	{"name": "privileged_agent_registry_action_requires_human_approval", "description": "Privileged registry actions proposed by agents require human approval.", "condition": {"operation": "agent_registry_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/composition-registry/dashboard", "component": "RegistryDashboard", "permission": "composition_registry:view", "nav_group": "Overview"},
	{"name": "catalog", "path": "/composition-registry/catalog", "component": "CapabilityCatalog", "permission": "composition_registry:manage_catalog", "nav_group": "Catalog"},
	{"name": "dependencies", "path": "/composition-registry/dependencies", "component": "DependencyGraph", "permission": "composition_registry:manage_dependencies", "nav_group": "Graph"},
	{"name": "compositions", "path": "/composition-registry/compositions", "component": "CompositionBlueprints", "permission": "composition_registry:compose", "nav_group": "Compositions"},
	{"name": "versions", "path": "/composition-registry/versions", "component": "VersionGovernance", "permission": "composition_registry:release", "nav_group": "Release"},
	{"name": "marketplace", "path": "/composition-registry/marketplace", "component": "MarketplacePublication", "permission": "composition_registry:publish", "nav_group": "Marketplace"},
	{"name": "rules", "path": "/composition-registry/rules", "component": "RegistryRuleCenter", "permission": "composition_registry:govern", "nav_group": "Governance"},
	{"name": "agents", "path": "/composition-registry/agents", "component": "RegistryAgentWorkbench", "permission": "composition_registry:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/composition-registry/settings", "component": "RegistrySettings", "permission": "composition_registry:admin", "nav_group": "Administration"},
]

THEME: dict[str, Any] = {
	"name": "composition_registry_control",
	"tokens": {"color.primary": "#28536B", "color.accent": "#C44536", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {
		"catalog": {"icon": "blocks", "status_indicator": "capability-pill", "risk_style": "quality-band"},
		"dependencies": {"visual": "dependency-graph", "status_style": "edge-chip"},
		"compositions": {"visual": "composition-board", "status_style": "validation-chip"},
		"versions": {"visual": "version-lanes", "status_style": "compatibility-chip"},
		"marketplace": {"visual": "publication-list", "status_style": "review-chip"},
		"rule_center": {"visual": "rule-grid", "status_style": "guardrail-chip"},
		"agent_workbench": {"visual": "review-lane", "status_style": "approval-chip"},
	},
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "composition_registry",
		"display_name": "Capability Registry",
		"provides": [
			"capability_catalog_lifecycle",
			"dependency_graph_management",
			"composition_blueprint_validation",
			"version_compatibility_governance",
			"marketplace_publication_governance",
			"registry_discovery",
			"registry_agents",
		],
		"requires": ["auth", "audl", "ntfy", "composition_events", "composition_config", "composition_access"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {"shell": "apg_python", "view_module": "views.py", "api_prefix": "/composition-registry/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True},
		"theme": deepcopy(THEME),
		"streaming": streaming_manifest(),
	}


def streaming_manifest() -> dict[str, Any]:
	return {
		"processor": "bytewax",
		"stream": REGISTRY_EVENT_STREAM,
		"key": "tenant_id",
		"events": [
			"capability_registered",
			"dependency_added",
			"composition_created",
			"composition_validated",
			"version_released",
			"capability_deprecated",
			"marketplace_publication_prepared",
			"registry_agent_registered",
		],
		"states": ["draft", "registered", "validated", "released", "published", "deprecated", "retired"],
		"guardrails": [
			"registry_import_requires_bytewax",
			"registry_event_requires_bytewax",
			"privileged_agent_registry_action_requires_human_approval",
		],
	}


def event_stream_name() -> str:
	return REGISTRY_EVENT_STREAM


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
		if key.endswith("_ne"):
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
