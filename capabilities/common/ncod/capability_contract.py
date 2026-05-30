"""Executable capability contract for APG No-Code/Low-Code Builder."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any


SUPPORTED_AI_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = [
	"app_architect",
	"screen_designer",
	"workflow_designer",
	"rule_author",
	"theme_designer",
	"test_builder",
]
SUPPORTED_DEPLOYMENT_TARGETS = ["python", "container", "apg_runtime", "edge_worker"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"apps": {
		"app_owner_required": True,
		"versioning_enabled": True,
		"publish_approval_required": True,
		"production_change_review_required": True,
		"review_before_deploy_required": True,
		"retirement_reason_required": True,
	},
	"builder": {
		"component_catalog_enabled": True,
		"theme_policy_required": True,
		"accessibility_checks_required": True,
		"data_binding_validation_required": True,
		"data_model_required": True,
		"component_relationships_required": True,
		"screen_route_required": True,
	},
	"extensions": {
		"workflow_binding_enabled": True,
		"workflow_policy_required": True,
		"script_extension_policy_required": True,
		"external_connector_policy_required": True,
		"custom_component_review_required": True,
	},
	"ai_builder_agents": {
		"agent_assist_enabled": True,
		"agent_registration_required": True,
		"agent_runtime_required": True,
		"agent_scope_required": True,
		"agent_contribution_disclosure_required": True,
		"supported_runtimes": SUPPORTED_AI_RUNTIMES,
		"allowed_roles": SUPPORTED_AGENT_ROLES,
	},
	"deployments": {
		"deployment_target_required": True,
		"deployment_approval_required": True,
		"rollback_plan_required": True,
		"production_release_review_required": True,
		"supported_targets": SUPPORTED_DEPLOYMENT_TARGETS,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_app_changes": True,
		"rbac_policy_required": True,
		"data_residency_policy_required": True,
		"tenant_isolation_required": True,
		"state_change_reason_required": True,
		"batch_event_stream": "bytewax",
	},
	"observability": {
		"audit_required": True,
		"build_metrics_required": True,
		"deployment_metrics_required": True,
		"agent_activity_required": True,
		"event_stream": "bytewax",
	},
	"adapters": {
		"generated_app_runtime": "service.NcodService",
		"runtime_helpers": "builder_runtime.py",
		"api_helpers": "api.py",
		"view_models": "views.py",
		"event_stream": "bytewax",
		"workflow": "wflo",
		"script": "scpt",
		"connectors": "conn",
		"auth": "auth",
		"audit_sink": "audl",
		"theme": "them",
		"accessibility": "accs",
	},
	"ui": {
		"enable_app_builder": True,
		"enable_page_composer": True,
		"enable_data_modeler": True,
		"enable_component_catalog": True,
		"enable_workflow_designer": True,
		"enable_publish_center": True,
		"enable_deployment_center": True,
		"enable_agent_panel": True,
		"enable_audit": True,
		"enable_analytics": True,
	},
	"theme": {
		"default_theme": "ncod_app_builder",
		"allow_tenant_overrides": True,
	},
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"apps",
		"builder",
		"extensions",
		"ai_builder_agents",
		"deployments",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {key: {"type": "object"} for key in [
		"apps",
		"builder",
		"extensions",
		"ai_builder_agents",
		"deployments",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]} | {
		"tenant_id": {"type": "string", "minLength": 1},
	},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All no-code operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "app_requires_owner", "description": "Applications require an accountable owner.", "condition": {"operation": "create_app", "app_owner_assigned": False}, "effect": {"decision": "deny", "reason": "app_owner_required", "required_action": "assign_app_owner"}},
	{"name": "app_requires_name", "description": "Applications require a non-empty display name.", "condition": {"operation": "create_app", "app_name_present": False}, "effect": {"decision": "deny", "reason": "app_name_required", "required_action": "set_app_name"}},
	{"name": "app_requires_theme", "description": "Applications require a selected theme.", "condition": {"operation": "create_app", "theme_selected": False}, "effect": {"decision": "deny", "reason": "theme_required", "required_action": "select_theme"}},
	{"name": "app_requires_rbac_policy", "description": "Applications require an RBAC policy reference.", "condition": {"operation": "create_app", "rbac_policy_present": False}, "effect": {"decision": "deny", "reason": "rbac_policy_required", "required_action": "attach_rbac_policy"}},
	{"name": "app_requires_data_residency_policy", "description": "Applications require a data residency policy reference.", "condition": {"operation": "create_app", "data_residency_policy_present": False}, "effect": {"decision": "deny", "reason": "data_residency_policy_required", "required_action": "attach_data_residency_policy"}},
	{"name": "page_requires_route", "description": "Screens require a stable route.", "condition": {"operation": "add_page", "route_present": False}, "effect": {"decision": "deny", "reason": "screen_route_required", "required_action": "set_screen_route"}},
	{"name": "page_requires_relationship_policy", "description": "Composed screens require relationship metadata for nested elements.", "condition": {"operation": "add_page", "element_relationships_declared": False}, "effect": {"decision": "require_review", "reason": "element_relationships_required", "required_action": "declare_element_relationships"}},
	{"name": "component_requires_screen", "description": "Components must be attached to a screen.", "condition": {"operation": "add_component", "screen_present": False}, "effect": {"decision": "deny", "reason": "screen_required", "required_action": "attach_component_to_screen"}},
	{"name": "interactive_component_requires_accessibility", "description": "Interactive components require accessible labels.", "condition": {"operation": "add_component", "interactive_component": True, "accessibility_label_present": False}, "effect": {"decision": "deny", "reason": "accessibility_label_required", "required_action": "add_accessibility_label"}},
	{"name": "data_model_requires_name", "description": "Data models require a business name.", "condition": {"operation": "define_data_model", "data_model_name_present": False}, "effect": {"decision": "deny", "reason": "data_model_name_required", "required_action": "set_data_model_name"}},
	{"name": "data_model_requires_fields", "description": "Data models require one or more fields.", "condition": {"operation": "define_data_model", "data_model_fields_present": False}, "effect": {"decision": "deny", "reason": "data_model_fields_required", "required_action": "define_data_model_fields"}},
	{"name": "data_model_requires_policy", "description": "Data models require governance policy.", "condition": {"operation": "define_data_model", "data_model_policy_present": False}, "effect": {"decision": "deny", "reason": "data_model_policy_required", "required_action": "attach_data_model_policy"}},
	{"name": "data_binding_requires_schema", "description": "Data bindings require a valid schema.", "condition": {"operation": "bind_data_source", "binding_schema_valid": False}, "effect": {"decision": "deny", "reason": "data_binding_schema_required", "required_action": "provide_binding_schema"}},
	{"name": "workflow_requires_trigger", "description": "Workflow bindings require a trigger.", "condition": {"operation": "attach_workflow", "workflow_trigger_present": False}, "effect": {"decision": "deny", "reason": "workflow_trigger_required", "required_action": "set_workflow_trigger"}},
	{"name": "workflow_requires_reference", "description": "Workflow bindings require a workflow reference.", "condition": {"operation": "attach_workflow", "workflow_ref_present": False}, "effect": {"decision": "deny", "reason": "workflow_ref_required", "required_action": "set_workflow_ref"}},
	{"name": "workflow_requires_policy", "description": "Workflow bindings require an automation policy.", "condition": {"operation": "attach_workflow", "workflow_policy_attached": False}, "effect": {"decision": "deny", "reason": "workflow_policy_required", "required_action": "attach_workflow_policy"}},
	{"name": "publish_requires_approval", "description": "Publishing applications requires approval.", "condition": {"operation": "publish_app", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "publish_approval_required", "required_action": "record_publish_approval"}},
	{"name": "publish_requires_validation", "description": "Publishing requires a passing validation result.", "condition": {"operation": "publish_app", "validation_passed": False}, "effect": {"decision": "deny", "reason": "app_validation_required", "required_action": "validate_app"}},
	{"name": "script_extension_requires_policy", "description": "Script extensions require an approved policy.", "condition": {"script_extension_present": True, "script_policy_attached": False}, "effect": {"decision": "deny", "reason": "script_policy_required", "required_action": "attach_script_policy"}},
	{"name": "external_connector_requires_policy", "description": "External connectors require a connector policy.", "condition": {"external_connector_present": True, "connector_policy_attached": False}, "effect": {"decision": "deny", "reason": "connector_policy_required", "required_action": "attach_connector_policy"}},
	{"name": "production_change_requires_review", "description": "Production app changes require review.", "condition": {"production_change": True, "change_review_recorded": False}, "effect": {"decision": "require_review", "reason": "production_change_review_required", "required_action": "review_production_change"}},
	{"name": "deployment_requires_target", "description": "Deployments require a supported target.", "condition": {"operation": "deploy_release", "deployment_target_supported": False}, "effect": {"decision": "deny", "reason": "deployment_target_required", "required_action": "choose_supported_deployment_target"}},
	{"name": "deployment_requires_approval", "description": "Deployments require approval evidence.", "condition": {"operation": "deploy_release", "deployment_approval_recorded": False}, "effect": {"decision": "deny", "reason": "deployment_approval_required", "required_action": "record_deployment_approval"}},
	{"name": "deployment_requires_rollback_plan", "description": "Deployments require a rollback plan.", "condition": {"operation": "deploy_release", "rollback_plan_present": False}, "effect": {"decision": "deny", "reason": "rollback_plan_required", "required_action": "attach_rollback_plan"}},
	{"name": "ai_builder_agent_requires_registration", "description": "AI builder agents must be registered.", "condition": {"ai_builder_agent_present": True, "agent_registered": False}, "effect": {"decision": "deny", "reason": "ai_builder_agent_registration_required", "required_action": "register_ai_builder_agent"}},
	{"name": "ai_builder_agent_runtime_supported", "description": "AI builder agents must use a supported runtime.", "condition": {"ai_builder_agent_present": True, "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "ai_builder_agent_runtime_not_supported", "required_action": "choose_supported_ai_builder_runtime"}},
	{"name": "ai_builder_agent_requires_scope", "description": "AI builder agents require explicit scope.", "condition": {"ai_builder_agent_present": True, "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "ai_builder_agent_scope_required", "required_action": "set_ai_builder_agent_scope"}},
	{"name": "ai_builder_agent_requires_disclosure", "description": "AI builder agent contributions require disclosure.", "condition": {"ai_builder_agent_present": True, "agent_contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "ai_builder_agent_disclosure_required", "required_action": "disclose_ai_builder_agent"}},
	{"name": "state_change_requires_reason", "description": "Application lifecycle state changes require a reason.", "condition": {"state_change_requested": True, "state_change_reason_present": False}, "effect": {"decision": "deny", "reason": "state_change_reason_required", "required_action": "record_state_change_reason"}},
	{"name": "state_change_requires_audit", "description": "Application lifecycle state changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "app_audit_event_required", "required_action": "record_app_audit_event"}},
	{"name": "cross_tenant_builder_access_denied", "description": "Builder records may not cross tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_builder_access_denied", "required_action": "use_tenant_local_context"}},
	{"name": "batch_builder_mutation_requires_bytewax", "description": "Batch builder mutations must use Bytewax event streams.", "condition": {"operation": "batch_builder_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/ncod/dashboard", "component": "NCODDashboard", "permission": "ncod:view", "nav_group": "Overview"},
	{"name": "apps", "path": "/ncod/apps", "component": "AppLibrary", "permission": "ncod:manage_apps", "nav_group": "Apps"},
	{"name": "builder", "path": "/ncod/builder", "component": "AppBuilder", "permission": "ncod:build", "nav_group": "Build"},
	{"name": "pages", "path": "/ncod/pages", "component": "PageComposer", "permission": "ncod:build", "nav_group": "Build"},
	{"name": "data_models", "path": "/ncod/data-models", "component": "DataModeler", "permission": "ncod:build", "nav_group": "Build"},
	{"name": "components", "path": "/ncod/components", "component": "ComponentCatalog", "permission": "ncod:build", "nav_group": "Build"},
	{"name": "workflows", "path": "/ncod/workflows", "component": "WorkflowDesigner", "permission": "ncod:build", "nav_group": "Automation"},
	{"name": "publishing", "path": "/ncod/publishing", "component": "PublishCenter", "permission": "ncod:publish", "nav_group": "Release"},
	{"name": "deployments", "path": "/ncod/deployments", "component": "DeploymentCenter", "permission": "ncod:deploy", "nav_group": "Release"},
	{"name": "connectors", "path": "/ncod/connectors", "component": "ConnectorBindings", "permission": "ncod:build", "nav_group": "Integrations"},
	{"name": "agents", "path": "/ncod/agents", "component": "AIBuilderAgentPanel", "permission": "ncod:build", "nav_group": "Agents"},
	{"name": "audit", "path": "/ncod/audit", "component": "NCODAuditTrail", "permission": "ncod:audit", "nav_group": "Governance"},
	{"name": "analytics", "path": "/ncod/analytics", "component": "NCODAnalytics", "permission": "ncod:view", "nav_group": "Operations"},
	{"name": "settings", "path": "/ncod/settings", "component": "NCODSettings", "permission": "ncod:admin", "nav_group": "Administration"},
]

THEME: dict[str, Any] = {
	"name": "ncod_app_builder",
	"tokens": {
		"color.primary": "#2C5282",
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
		"app_library": {"icon": "layout-dashboard", "status_indicator": "app-pill", "risk_style": "release-band"},
		"page_composer": {"visual": "component-canvas", "highlight": "theme-chip"},
		"component_catalog": {"visual": "component-grid", "status_style": "accessibility-chip"},
		"data_modeler": {"visual": "entity-map", "status_style": "policy-chip"},
		"workflow_designer": {"visual": "automation-canvas", "status_style": "trigger-chip"},
		"publish_center": {"visual": "release-checklist", "status_style": "approval-chip"},
		"deployment_center": {"visual": "deployment-lane", "status_style": "target-chip"},
		"agent_panel": {"visual": "agent-roster", "status_style": "scope-chip"},
	},
}

STREAMING: dict[str, Any] = {
	"processor": "bytewax",
	"topic": "apg.ncod.lifecycle",
	"state": ["apps", "pages", "components", "data_models", "workflows", "releases", "deployments", "ai_builder_agents"],
	"events": [
		"app_created",
		"page_added",
		"component_added",
		"data_model_defined",
		"data_binding_added",
		"workflow_attached",
		"script_extension_added",
		"connector_bound",
		"ai_builder_agent_registered",
		"app_validated",
		"app_published",
		"release_deployed",
		"app_state_changed",
	],
	"batch_mutation_guardrail": "batch_builder_mutation_requires_bytewax",
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable NCOD capability contract."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "ncod",
		"display_name": "No-Code/Low-Code Builder",
		"provides": ["app_builder", "page_composer", "data_modeler", "workflow_binding", "script_extensions", "connector_bindings", "ai_builder_agents", "app_publishing", "app_deployment"],
		"requires": ["wflo", "scpt", "auth"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": config["adapters"]["view_models"],
			"api_prefix": "/ncod/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True,
		},
		"theme": deepcopy(THEME),
		"streaming": deepcopy(STREAMING),
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate default NCOD governance rules."""
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
