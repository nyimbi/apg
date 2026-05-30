"""Executable capability contract for APG Website Builder."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


SUPPORTED_WSBL_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_WSBL_AGENT_ROLES = [
	"site_reviewer",
	"component_reviewer",
	"accessibility_reviewer",
	"privacy_reviewer",
	"publish_reviewer",
	"seo_reviewer",
]
WSBL_EVENT_STREAM = "apg.wsbl.lifecycle"


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"sites": {
		"site_owner_required": True,
		"domain_validation_required": True,
		"multi_locale_enabled": True,
		"environment_preview_enabled": True,
		"preview_evidence_required": True,
	},
	"pages": {
		"structured_sections_required": True,
		"custom_component_review_required": True,
		"draft_autosave_enabled": True,
		"content_versioning_enabled": True,
		"page_review_required": True,
	},
	"publishing": {
		"approval_required": True,
		"accessibility_pass_required": True,
		"privacy_banner_policy_required": True,
		"rollback_supported": True,
		"publish_stream_required": True,
	},
	"wsbl_agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_WSBL_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_WSBL_AGENT_ROLES,
		"human_approval_required": True,
		"max_autonomous_scope": "non_privileged",
		"disclose_agent_recommendations": True,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_publication_changes": True,
		"component_policy_required": True,
		"public_site_controls_required": True,
		"state_change_audit_required": True,
	},
	"observability": {
		"event_stream": WSBL_EVENT_STREAM,
		"stream_processor": "bytewax",
		"emit_site_events": True,
		"emit_page_events": True,
		"emit_publish_events": True,
	},
	"adapters": {
		"theme": "adapter",
		"authorization": "adapter",
		"consent": "adapter",
		"accessibility": "adapter",
		"analytics": "adapter",
		"event_stream": "bytewax",
	},
	"ui": {
		"enable_site_console": True,
		"enable_page_editor": True,
		"enable_component_library": True,
		"enable_publish_queue": True,
		"enable_agent_workbench": True,
		"enable_policy_center": True,
	},
	"theme": {"default_theme": "wsbl_site_builder", "allow_tenant_overrides": True},
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"sites",
		"pages",
		"publishing",
		"wsbl_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {
		key: {"type": "object"}
		for key in [
			"sites",
			"pages",
			"publishing",
			"wsbl_agents",
			"governance",
			"observability",
			"adapters",
			"ui",
			"theme",
		]
	} | {"tenant_id": {"type": "string", "minLength": 1}},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All website-builder operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "site_requires_owner", "description": "Sites require an accountable owner.", "condition": {"operation": "create_site", "site_owner_assigned": False}, "effect": {"decision": "deny", "reason": "site_owner_required", "required_action": "assign_site_owner"}},
	{"name": "domain_requires_validation_before_publish", "description": "Publishing requires validated site domains.", "condition": {"operation": "publish_site", "domain_validation_complete": False}, "effect": {"decision": "deny", "reason": "domain_validation_required", "required_action": "validate_site_domains"}},
	{"name": "page_requires_structured_sections", "description": "Publishing requires at least one structured page section.", "condition": {"operation": "publish_site", "structured_sections_present": False}, "effect": {"decision": "deny", "reason": "structured_sections_required", "required_action": "add_structured_page_sections"}},
	{"name": "preview_requires_evidence", "description": "Publishing requires preview evidence for the target environment.", "condition": {"operation": "publish_site", "preview_evidence_present": False}, "effect": {"decision": "deny", "reason": "preview_evidence_required", "required_action": "attach_environment_preview"}},
	{"name": "publish_requires_approval", "description": "Site publishing requires approval.", "condition": {"operation": "publish_site", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "site_publish_approval_required", "required_action": "record_publish_approval"}},
	{"name": "publish_requires_bytewax_stream", "description": "Site publishing lifecycle events must be emitted through Bytewax.", "condition": {"operation": "publish_site", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_publish_lifecycle_to_bytewax"}},
	{"name": "custom_component_requires_review", "description": "Custom components require review before use.", "condition": {"operation": "add_page_section", "custom_component_present": True, "component_review_recorded": False}, "effect": {"decision": "deny", "reason": "component_review_required", "required_action": "review_custom_component"}},
	{"name": "custom_component_requires_policy", "description": "Custom components require component policy attribution.", "condition": {"operation": "review_component", "custom_component_present": True, "component_policy_attached": False}, "effect": {"decision": "deny", "reason": "component_policy_required", "required_action": "attach_component_policy"}},
	{"name": "public_site_requires_accessibility_pass", "description": "Public sites require an accessibility pass.", "condition": {"operation": "publish_site", "public_site": True, "accessibility_passed": False}, "effect": {"decision": "deny", "reason": "accessibility_pass_required", "required_action": "complete_accessibility_pass"}},
	{"name": "privacy_banner_requires_consent_policy", "description": "Privacy banners require an attached consent policy.", "condition": {"operation": "publish_site", "privacy_banner_required": True, "consent_policy_attached": False}, "effect": {"decision": "require_review", "reason": "consent_policy_required", "required_action": "attach_consent_policy"}},
	{"name": "rollback_requires_bytewax_stream", "description": "Rollback lifecycle events must be emitted through Bytewax.", "condition": {"operation": "rollback_site", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_rollback_lifecycle_to_bytewax"}},
	{"name": "batch_publish_requires_bytewax", "description": "Batch site publishing requires Bytewax stream coordination.", "condition": {"operation": "batch_publish", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_batch_publish_to_bytewax"}},
	{"name": "wsbl_agent_runtime_supported", "description": "Website builder agents must use an approved runtime.", "condition": {"operation": "register_wsbl_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "wsbl_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"}},
	{"name": "wsbl_agent_role_supported", "description": "Website builder agents must use an approved role.", "condition": {"operation": "register_wsbl_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "wsbl_agent_role_not_supported", "required_action": "select_supported_agent_role"}},
	{"name": "privileged_agent_publish_action_requires_human_approval", "description": "Privileged publishing actions proposed by agents require human approval.", "condition": {"operation": "agent_publish_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/wsbl/dashboard", "component": "WSBLDashboard", "permission": "wsbl:view", "nav_group": "Overview"},
	{"name": "sites", "path": "/wsbl/sites", "component": "SiteConsole", "permission": "wsbl:manage_sites", "nav_group": "Sites"},
	{"name": "pages", "path": "/wsbl/pages", "component": "PageLibrary", "permission": "wsbl:build", "nav_group": "Pages"},
	{"name": "editor", "path": "/wsbl/editor", "component": "PageEditor", "permission": "wsbl:build", "nav_group": "Build"},
	{"name": "components", "path": "/wsbl/components", "component": "ComponentLibrary", "permission": "wsbl:build", "nav_group": "Build"},
	{"name": "publishing", "path": "/wsbl/publishing", "component": "PublishQueue", "permission": "wsbl:publish", "nav_group": "Release"},
	{"name": "analytics", "path": "/wsbl/analytics", "component": "SiteAnalytics", "permission": "wsbl:view", "nav_group": "Operations"},
	{"name": "agents", "path": "/wsbl/agents", "component": "WSBLAgentWorkbench", "permission": "wsbl:admin", "nav_group": "Automation"},
	{"name": "policy", "path": "/wsbl/policy", "component": "WSBLPolicyCenter", "permission": "wsbl:admin", "nav_group": "Governance"},
	{"name": "settings", "path": "/wsbl/settings", "component": "WSBLSettings", "permission": "wsbl:admin", "nav_group": "Administration"},
]

THEME: dict[str, Any] = {
	"name": "wsbl_site_builder",
	"tokens": {"color.primary": "#2C5282", "color.accent": "#38A169", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {
		"site_card": {"icon": "layout-template", "status_indicator": "site-pill", "risk_style": "publish-band"},
		"page_editor": {"visual": "section-builder", "highlight": "component-chip"},
		"publish_queue": {"visual": "release-checklist", "status_style": "approval-chip"},
		"analytics_panel": {"visual": "traffic-grid", "status_style": "trend-chip"},
		"agent_workbench": {"visual": "review-lane", "status_style": "approval-chip"},
		"policy_center": {"visual": "rule-grid", "status_style": "guardrail-chip"},
	},
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "wsbl",
		"display_name": "Website Builder",
		"provides": [
			"site_management",
			"page_composition",
			"component_library",
			"publishing_workflows",
			"site_theming",
			"website_governance",
			"wsbl_agents",
		],
		"requires": ["them", "auth", "ncod", "accs", "cons"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {"shell": "apg_python", "view_module": "views.py", "api_prefix": "/wsbl/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True},
		"theme": deepcopy(THEME),
		"streaming": streaming_manifest(),
	}


def streaming_manifest() -> dict[str, Any]:
	return {
		"processor": "bytewax",
		"stream": WSBL_EVENT_STREAM,
		"key": "tenant_id",
		"events": [
			"site_created",
			"domain_registered",
			"domain_validated",
			"component_created",
			"component_reviewed",
			"page_created",
			"page_section_added",
			"publish_request_created",
			"site_published",
			"site_rolled_back",
			"wsbl_agent_registered",
		],
		"states": ["draft", "domain_pending", "ready", "review_required", "approved", "published", "rolled_back", "blocked"],
		"guardrails": [
			"publish_requires_bytewax_stream",
			"rollback_requires_bytewax_stream",
			"batch_publish_requires_bytewax",
			"privileged_agent_publish_action_requires_human_approval",
		],
	}


def event_stream_name() -> str:
	return WSBL_EVENT_STREAM


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
		if key.endswith("_lte"):
			if not context.get(key[:-4], 0) <= expected:
				return False
		elif key.endswith("_lt"):
			if not context.get(key[:-3], 0) < expected:
				return False
		elif key.endswith("_gte"):
			if not context.get(key[:-4], 0) >= expected:
				return False
		elif key.endswith("_gt"):
			if not context.get(key[:-3], 0) > expected:
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
