"""Executable capability contract for APG Identity Federation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"providers": {
		"enabled_provider_types": ["saml", "oidc", "ldap", "scim"],
		"signing_key_required": True,
		"metadata_refresh_hours": 24,
		"provider_owner_required": True
	},
	"protocols": {
		"saml_assertion_encryption_required": True,
		"oidc_redirect_allowlist_required": True,
		"pkce_required": True,
		"scim_group_sync_enabled": True
	},
	"sessions": {
		"max_session_hours": 12,
		"privileged_mfa_required": True,
		"session_revocation_supported": True,
		"risk_based_reauth_enabled": True
	},
	"governance": {
		"require_tenant_context": True,
		"audit_federation_events": True,
		"certificate_rotation_days": 90,
		"claim_mapping_review_required": True
	},
	"ui": {
		"enable_provider_console": True,
		"enable_protocol_workbench": True,
		"enable_session_monitor": True,
		"enable_certificate_center": True
	},
	"theme": {
		"default_theme": "idfd_federation_console",
		"allow_tenant_overrides": True
	}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "providers", "protocols", "sessions", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["providers", "protocols", "sessions", "governance", "ui", "theme"]} | {
		"tenant_id": {"type": "string", "minLength": 1}
	}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All federation operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "provider_requires_signing_key", "description": "Federation providers require a signing key.", "condition": {"operation": "register_provider", "signing_key_present": False}, "effect": {"decision": "deny", "reason": "signing_key_required", "required_action": "attach_signing_key"}},
	{"name": "saml_assertion_requires_encryption", "description": "SAML assertions require encryption.", "condition": {"protocol": "saml", "assertion_encrypted": False}, "effect": {"decision": "deny", "reason": "saml_assertion_encryption_required", "required_action": "enable_assertion_encryption"}},
	{"name": "oidc_client_requires_redirect_allowlist", "description": "OIDC clients require redirect URI allowlists.", "condition": {"protocol": "oidc", "redirect_allowlist_configured": False}, "effect": {"decision": "deny", "reason": "redirect_allowlist_required", "required_action": "configure_redirect_allowlist"}},
	{"name": "privileged_federation_requires_mfa", "description": "Privileged federation sessions require MFA.", "condition": {"session_privilege": "privileged", "mfa_completed": False}, "effect": {"decision": "deny", "reason": "privileged_mfa_required", "required_action": "complete_mfa"}},
	{"name": "stale_metadata_requires_refresh", "description": "Stale provider metadata requires refresh or review.", "condition": {"metadata_age_hours_gt": 24, "metadata_refresh_completed": False}, "effect": {"decision": "require_review", "reason": "metadata_refresh_required", "required_action": "refresh_provider_metadata"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/idfd/dashboard", "component": "IDFDDashboard", "permission": "idfd:view", "nav_group": "Overview"},
	{"name": "providers", "path": "/idfd/providers", "component": "FederationProviders", "permission": "idfd:manage_providers", "nav_group": "Providers"},
	{"name": "protocols", "path": "/idfd/protocols", "component": "ProtocolWorkbench", "permission": "idfd:manage_providers", "nav_group": "Providers"},
	{"name": "mappings", "path": "/idfd/mappings", "component": "IdentityMappings", "permission": "idfd:manage_mappings", "nav_group": "Mappings"},
	{"name": "sessions", "path": "/idfd/sessions", "component": "FederatedSessions", "permission": "idfd:view", "nav_group": "Operations"},
	{"name": "certificates", "path": "/idfd/certificates", "component": "CertificateCenter", "permission": "idfd:rotate_keys", "nav_group": "Security"},
	{"name": "audit", "path": "/idfd/audit", "component": "FederationAudit", "permission": "idfd:view", "nav_group": "Governance"},
	{"name": "settings", "path": "/idfd/settings", "component": "IDFDSettings", "permission": "idfd:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "idfd_federation_console",
	"tokens": {
		"color.primary": "#2A4365",
		"color.accent": "#9F7AEA",
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
		"provider_grid": {"icon": "network", "status_indicator": "provider-pill", "risk_style": "metadata-band"},
		"protocol_panel": {"visual": "protocol-tabs", "highlight": "crypto-chip"},
		"mapping_table": {"visual": "claim-map", "status_style": "review-chip"},
		"certificate_timeline": {"visual": "rotation-timeline", "status_style": "expiry-chip"}
	}
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable IDFD capability contract."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "idfd",
		"display_name": "Identity Federation",
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/idfd/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True
		},
		"theme": deepcopy(THEME)
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate default IDFD governance rules."""
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
