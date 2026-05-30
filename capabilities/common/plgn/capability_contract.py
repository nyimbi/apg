"""Executable capability contract for APG Plugin/Extension Framework."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any


SUPPORTED_PLGN_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_PLGN_AGENT_ROLES = [
	"marketplace_reviewer",
	"manifest_reviewer",
	"permission_reviewer",
	"sandbox_reviewer",
	"release_reviewer",
	"compatibility_reviewer",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"marketplace": {
		"curated_listing_required": True,
		"publisher_verification_required": True,
		"release_channel_policy_required": True,
		"tenant_install_policy_enabled": True,
		"compatibility_matrix_required": True,
	},
	"plugins": {
		"plugin_owner_required": True,
		"manifest_schema_required": True,
		"signature_required": True,
		"dependency_validation_required": True,
		"event_stream": "bytewax",
	},
	"security": {
		"permission_review_required": True,
		"sandbox_policy_required": True,
		"secret_access_denied_by_default": True,
		"supply_chain_scan_required": True,
		"runtime_isolation_required": True,
	},
	"plgn_agents": {
		"agent_assist_enabled": True,
		"agent_registration_required": True,
		"agent_runtime_required": True,
		"agent_role_required": True,
		"agent_scope_required": True,
		"agent_contribution_disclosure_required": True,
		"supported_runtimes": SUPPORTED_PLGN_AGENT_RUNTIMES,
		"allowed_roles": SUPPORTED_PLGN_AGENT_ROLES,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_plugin_changes": True,
		"external_plugin_review_required": True,
		"configuration_policy_required": True,
		"state_change_audit_required": True,
		"tenant_isolation_required": True,
		"batch_event_stream": "bytewax",
	},
	"observability": {
		"audit_required": True,
		"marketplace_metrics_required": True,
		"permission_metrics_required": True,
		"sandbox_metrics_required": True,
		"agent_activity_required": True,
		"event_stream": "bytewax",
	},
	"adapters": {
		"generated_app_runtime": "service.PlgnService",
		"api_helpers": "api.py",
		"view_models": "views.py",
		"event_stream": "bytewax",
		"identity": "auth",
		"security": "secu",
		"configuration": "conf",
		"audit_sink": "audl",
		"registry": "regy",
		"sandbox": "sbox",
		"workflow": "wflo",
	},
	"ui": {
		"enable_marketplace": True,
		"enable_plugin_registry": True,
		"enable_manifest_editor": True,
		"enable_permission_review": True,
		"enable_sandbox_policy": True,
		"enable_release_manager": True,
		"enable_agent_panel": True,
		"enable_audit": True,
	},
	"theme": {
		"default_theme": "plgn_extension_marketplace",
		"allow_tenant_overrides": True,
	},
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"marketplace",
		"plugins",
		"security",
		"plgn_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {
		key: {"type": "object"}
		for key in [
			"marketplace",
			"plugins",
			"security",
			"plgn_agents",
			"governance",
			"observability",
			"adapters",
			"ui",
			"theme",
		]
	}
	| {"tenant_id": {"type": "string", "minLength": 1}},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All plugin operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "plugin_requires_owner", "description": "Plugins require an accountable owner.", "condition": {"operation": "register_plugin", "plugin_owner_assigned": False}, "effect": {"decision": "deny", "reason": "plugin_owner_required", "required_action": "assign_plugin_owner"}},
	{"name": "plugin_requires_signature", "description": "Plugin packages require verified signatures.", "condition": {"operation": "register_plugin", "signature_verified": False}, "effect": {"decision": "deny", "reason": "plugin_signature_required", "required_action": "verify_plugin_signature"}},
	{"name": "plugin_requires_manifest_schema", "description": "Plugin manifests require schema validation.", "condition": {"operation": "register_plugin", "manifest_schema_valid": False}, "effect": {"decision": "deny", "reason": "manifest_schema_required", "required_action": "validate_plugin_manifest"}},
	{"name": "plugin_requires_dependency_validation", "description": "Plugin dependencies require validation.", "condition": {"operation": "register_plugin", "dependency_validation_passed": False}, "effect": {"decision": "deny", "reason": "dependency_validation_required", "required_action": "validate_plugin_dependencies"}},
	{"name": "plugin_requires_supply_chain_scan", "description": "Plugin packages require supply-chain scan evidence.", "condition": {"operation": "register_plugin", "supply_chain_scan_passed": False}, "effect": {"decision": "deny", "reason": "supply_chain_scan_required", "required_action": "scan_plugin_package"}},
	{"name": "permissions_require_review", "description": "Requested plugin permissions require review.", "condition": {"permissions_requested": True, "permission_review_recorded": False}, "effect": {"decision": "deny", "reason": "permission_review_required", "required_action": "review_plugin_permissions"}},
	{"name": "plugin_requires_sandbox", "description": "Plugins require sandbox policy before execution.", "condition": {"operation": "enable_plugin", "sandbox_policy_attached": False}, "effect": {"decision": "deny", "reason": "plugin_sandbox_required", "required_action": "attach_sandbox_policy"}},
	{"name": "external_plugin_requires_review", "description": "External plugins require review.", "condition": {"external_plugin": True, "external_review_recorded": False}, "effect": {"decision": "require_review", "reason": "external_plugin_review_required", "required_action": "review_external_plugin"}},
	{"name": "marketplace_requires_verified_publisher", "description": "Marketplace listings require verified publishers.", "condition": {"operation": "publish_listing", "publisher_verified": False}, "effect": {"decision": "deny", "reason": "publisher_verification_required", "required_action": "verify_publisher"}},
	{"name": "marketplace_requires_curated_listing", "description": "Marketplace listings require curation.", "condition": {"operation": "publish_listing", "curated_listing": False}, "effect": {"decision": "deny", "reason": "curated_listing_required", "required_action": "curate_listing"}},
	{"name": "release_requires_signature_reference", "description": "Plugin releases require signature reference.", "condition": {"operation": "create_release", "signature_ref_present": False}, "effect": {"decision": "deny", "reason": "release_signature_required", "required_action": "attach_release_signature"}},
	{"name": "release_requires_bytewax_stream", "description": "Plugin release lifecycle events require Bytewax streams.", "condition": {"operation": "create_release", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
	{"name": "install_requires_tenant_policy", "description": "Plugin installation requires tenant install policy.", "condition": {"operation": "install_plugin", "tenant_install_policy_present": False}, "effect": {"decision": "deny", "reason": "tenant_install_policy_required", "required_action": "attach_tenant_install_policy"}},
	{"name": "plgn_agent_requires_registration", "description": "AI plugin agents must be registered.", "condition": {"plgn_agent_present": True, "agent_registered": False}, "effect": {"decision": "deny", "reason": "plgn_agent_registration_required", "required_action": "register_plgn_agent"}},
	{"name": "plgn_agent_runtime_supported", "description": "AI plugin agents must use a supported runtime.", "condition": {"plgn_agent_present": True, "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "plgn_agent_runtime_not_supported", "required_action": "choose_supported_plgn_agent_runtime"}},
	{"name": "plgn_agent_role_supported", "description": "AI plugin agents must use a supported role.", "condition": {"plgn_agent_present": True, "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "plgn_agent_role_not_supported", "required_action": "choose_supported_plgn_agent_role"}},
	{"name": "plgn_agent_requires_scope", "description": "AI plugin agents require explicit scope.", "condition": {"plgn_agent_present": True, "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "plgn_agent_scope_required", "required_action": "set_plgn_agent_scope"}},
	{"name": "plgn_agent_requires_disclosure", "description": "AI plugin-agent contributions require disclosure.", "condition": {"plgn_agent_present": True, "agent_contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "plgn_agent_disclosure_required", "required_action": "disclose_plgn_agent"}},
	{"name": "plgn_state_change_requires_audit", "description": "Plugin lifecycle state changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "plgn_audit_event_required", "required_action": "record_plgn_audit_event"}},
	{"name": "batch_plugin_mutation_requires_bytewax", "description": "Batch plugin mutations must use Bytewax event streams.", "condition": {"requested_operation": "batch_plugin_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/plgn/dashboard", "component": "PLGNDashboard", "permission": "plgn:view", "nav_group": "Overview"},
	{"name": "marketplace", "path": "/plgn/marketplace", "component": "ExtensionMarketplace", "permission": "plgn:install", "nav_group": "Marketplace"},
	{"name": "plugins", "path": "/plgn/plugins", "component": "PluginRegistry", "permission": "plgn:view", "nav_group": "Plugins"},
	{"name": "manifests", "path": "/plgn/manifests", "component": "ManifestEditor", "permission": "plgn:publish", "nav_group": "Plugins"},
	{"name": "permissions", "path": "/plgn/permissions", "component": "PermissionReview", "permission": "plgn:review", "nav_group": "Security"},
	{"name": "sandbox", "path": "/plgn/sandbox", "component": "PluginSandboxPolicy", "permission": "plgn:review", "nav_group": "Security"},
	{"name": "releases", "path": "/plgn/releases", "component": "ReleaseManager", "permission": "plgn:publish", "nav_group": "Release"},
	{"name": "agents", "path": "/plgn/agents", "component": "PLGNAgentPanel", "permission": "plgn:admin", "nav_group": "Operations"},
	{"name": "audit", "path": "/plgn/audit", "component": "PluginAuditTrail", "permission": "plgn:admin", "nav_group": "Governance"},
	{"name": "settings", "path": "/plgn/settings", "component": "PLGNSettings", "permission": "plgn:admin", "nav_group": "Administration"},
]

THEME: dict[str, Any] = {
	"name": "plgn_extension_marketplace",
	"tokens": {
		"color.primary": "#2B4C7E",
		"color.accent": "#D69E2E",
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
		"plugin_card": {"icon": "package-plus", "status_indicator": "trust-pill", "risk_style": "permission-band"},
		"marketplace_grid": {"visual": "extension-grid", "highlight": "verified-chip"},
		"permission_review": {"visual": "scope-table", "status_style": "review-chip"},
		"release_manager": {"visual": "channel-lanes", "status_style": "signature-chip"},
		"agent_panel": {"visual": "agent-roster", "status_style": "scope-chip"},
		"audit_timeline": {"visual": "event-timeline", "status_style": "extension-chip"},
	},
}


def streaming_manifest() -> dict[str, Any]:
	return {
		"processor": "bytewax",
		"topic": "apg.plgn.lifecycle",
		"state": [
			"plugins",
			"permission_reviews",
			"sandbox_policies",
			"marketplace_listings",
			"releases",
			"installations",
			"plgn_agents",
			"audit_events",
		],
		"events": [
			"plgn_plugin_registered",
			"plgn_permission_review_recorded",
			"plgn_sandbox_policy_attached",
			"plgn_marketplace_listing_published",
			"plgn_plugin_released",
			"plgn_plugin_installed",
			"plgn_plugin_enabled",
			"plgn_agent_registered",
		],
		"batch_mutation_guardrail": "batch_plugin_mutation_requires_bytewax",
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "plgn",
		"display_name": "Plugin/Extension Framework",
		"version": "1.0.0",
		"provides": [
			"plugin_registry",
			"extension_marketplace",
			"permission_review",
			"sandbox_policy",
			"plugin_release_lifecycle",
			"plgn_agents",
		],
		"requires": ["auth", "secu", "conf", "audl"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/plgn/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True,
		},
		"theme": deepcopy(THEME),
		"streaming": streaming_manifest(),
	}


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


def event_stream_name(value: str) -> str:
	return value.strip().lower().split("://", 1)[0]


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
			actual = context.get(key[:-3])
			if actual is None or actual == expected:
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
