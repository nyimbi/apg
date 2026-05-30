"""Executable capability contract for APG Identity Federation."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"providers": {
		"enabled_provider_types": ["saml", "oidc", "ldap", "scim"],
		"signing_key_required": True,
		"metadata_refresh_hours": 24,
		"provider_owner_required": True,
		"provider_disable_supported": True,
	},
	"protocols": {
		"saml_assertion_encryption_required": True,
		"oidc_redirect_allowlist_required": True,
		"pkce_required": True,
		"ldap_tls_required": True,
		"scim_group_sync_enabled": True,
	},
	"claims": {
		"claim_mapping_review_required": True,
		"source_claim_required": True,
		"target_claim_required": True,
		"sensitive_claim_review_required": True,
	},
	"sessions": {
		"max_session_hours": 12,
		"privileged_mfa_required": True,
		"session_revocation_supported": True,
		"risk_based_reauth_enabled": True,
		"high_risk_reauth_threshold": 0.7,
	},
	"scim": {
		"group_sync_enabled": True,
		"deprovisioning_required": True,
		"external_id_required": True,
	},
	"certificates": {
		"rotation_days": 90,
		"expiry_review_days": 30,
		"active_certificate_required": True,
		"revocation_supported": True,
	},
	"reviews": {
		"independent_reviewer_required": True,
		"review_notes_required": True,
		"duplicate_pending_review_blocked": True,
	},
	"security": {
		"tenant_isolation_required": True,
		"signed_metadata_required": True,
		"deny_unencrypted_saml": True,
		"deny_unlisted_oidc_redirects": True,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_federation_events": True,
		"certificate_rotation_days": 90,
		"claim_mapping_review_required": True,
	},
	"observability": {
		"audit_required": True,
		"metrics_required": True,
		"trace_required": True,
		"event_stream": "bytewax",
	},
	"adapters": {
		"generated_app_runtime": "service.IdfdService",
		"helper_runtime": "federation_runtime.py",
		"api_helpers": "api.py",
		"view_models": "views.py",
		"production_runtime": "service.py",
		"production_api": "api.py",
		"production_views": "views.py",
		"event_stream": "bytewax",
		"authentication": "auth",
		"mfa_provider": "mfau",
		"encryption": "encr",
		"audit_sink": "audl",
		"security_framework": "secu",
		"key_management": "keym",
		"monitoring": "moni",
		"cache": "cach",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_provider_console": True,
		"enable_protocol_workbench": True,
		"enable_claim_mapping": True,
		"enable_session_monitor": True,
		"enable_certificate_center": True,
		"enable_scim_directory": True,
		"enable_risk_console": True,
		"enable_review_queue": True,
		"enable_audit": True,
		"enable_settings": True,
	},
	"theme": {"default_theme": "idfd_federation_console", "allow_tenant_overrides": True},
}


CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"providers",
		"protocols",
		"claims",
		"sessions",
		"scim",
		"certificates",
		"reviews",
		"security",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {key: {"type": "object"} for key in [
		"providers",
		"protocols",
		"claims",
		"sessions",
		"scim",
		"certificates",
		"reviews",
		"security",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]} | {"tenant_id": {"type": "string", "minLength": 1}},
}


RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All federation operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "provider_requires_owner", "description": "Federation providers require an owning identity team or service.", "condition": {"operation": "register_provider", "owner_present": False}, "effect": {"decision": "deny", "reason": "provider_owner_required", "required_action": "assign_provider_owner"}},
	{"name": "provider_requires_signing_key", "description": "Federation providers require a signing key.", "condition": {"operation": "register_provider", "signing_key_present": False}, "effect": {"decision": "deny", "reason": "signing_key_required", "required_action": "attach_signing_key"}},
	{"name": "provider_protocol_must_be_enabled", "description": "Provider protocol must be enabled for the tenant.", "condition": {"operation": "register_provider", "protocol_enabled": False}, "effect": {"decision": "deny", "reason": "provider_protocol_not_enabled", "required_action": "enable_provider_protocol"}},
	{"name": "provider_metadata_url_required", "description": "Federation providers require metadata location or metadata payload evidence.", "condition": {"operation": "register_provider", "metadata_present": False}, "effect": {"decision": "require_review", "reason": "provider_metadata_required", "required_action": "attach_metadata"}},
	{"name": "provider_metadata_signature_required", "description": "Provider metadata must be signed or independently verified.", "condition": {"operation": "register_provider", "metadata_signed": False}, "effect": {"decision": "require_review", "reason": "metadata_signature_required", "required_action": "verify_metadata_signature"}},
	{"name": "saml_assertion_requires_encryption", "description": "SAML assertions require encryption.", "condition": {"protocol": "saml", "assertion_encrypted": False}, "effect": {"decision": "deny", "reason": "saml_assertion_encryption_required", "required_action": "enable_assertion_encryption"}},
	{"name": "saml_requires_signed_response", "description": "SAML responses require signature validation.", "condition": {"protocol": "saml", "response_signature_validated": False}, "effect": {"decision": "deny", "reason": "saml_response_signature_required", "required_action": "validate_saml_response_signature"}},
	{"name": "oidc_client_requires_redirect_allowlist", "description": "OIDC clients require redirect URI allowlists.", "condition": {"protocol": "oidc", "redirect_allowlist_configured": False}, "effect": {"decision": "deny", "reason": "redirect_allowlist_required", "required_action": "configure_redirect_allowlist"}},
	{"name": "oidc_requires_pkce", "description": "OIDC public clients require PKCE.", "condition": {"protocol": "oidc", "pkce_required": False}, "effect": {"decision": "deny", "reason": "pkce_required", "required_action": "enable_pkce"}},
	{"name": "ldap_requires_tls", "description": "LDAP federation requires TLS.", "condition": {"protocol": "ldap", "tls_enabled": False}, "effect": {"decision": "deny", "reason": "ldap_tls_required", "required_action": "enable_ldap_tls"}},
	{"name": "scim_requires_external_id", "description": "SCIM identities require stable external IDs.", "condition": {"operation": "scim_sync", "external_id_present": False}, "effect": {"decision": "deny", "reason": "scim_external_id_required", "required_action": "attach_external_id"}},
	{"name": "scim_deprovisioning_required", "description": "SCIM user removal must deprovision sessions and mappings.", "condition": {"operation": "scim_deprovision", "deprovisioning_completed": False}, "effect": {"decision": "deny", "reason": "scim_deprovisioning_required", "required_action": "complete_deprovisioning"}},
	{"name": "claim_mapping_requires_source", "description": "Claim mappings require a source claim.", "condition": {"operation": "add_claim_mapping", "source_claim_present": False}, "effect": {"decision": "deny", "reason": "source_claim_required", "required_action": "set_source_claim"}},
	{"name": "claim_mapping_requires_target", "description": "Claim mappings require a target claim.", "condition": {"operation": "add_claim_mapping", "target_claim_present": False}, "effect": {"decision": "deny", "reason": "target_claim_required", "required_action": "set_target_claim"}},
	{"name": "claim_mapping_review_required", "description": "Claim mappings require review before activation.", "condition": {"operation": "add_claim_mapping", "claim_mapping_reviewed": False}, "effect": {"decision": "deny", "reason": "claim_mapping_review_required", "required_action": "review_claim_mapping"}},
	{"name": "sensitive_claim_mapping_requires_privacy_review", "description": "Sensitive claims require privacy review.", "condition": {"operation": "add_claim_mapping", "sensitive_claim": True, "privacy_review_recorded": False}, "effect": {"decision": "require_review", "reason": "sensitive_claim_privacy_review_required", "required_action": "record_privacy_review"}},
	{"name": "privileged_federation_requires_mfa", "description": "Privileged federation sessions require MFA.", "condition": {"session_privilege": "privileged", "mfa_completed": False}, "effect": {"decision": "deny", "reason": "privileged_mfa_required", "required_action": "complete_mfa"}},
	{"name": "session_requires_active_provider", "description": "Sessions require an active provider.", "condition": {"operation": "issue_session", "provider_active": False}, "effect": {"decision": "deny", "reason": "provider_not_active", "required_action": "activate_provider"}},
	{"name": "session_duration_within_limit", "description": "Federated sessions must stay within tenant duration limits.", "condition": {"operation": "issue_session", "session_hours_gt": 12}, "effect": {"decision": "deny", "reason": "session_duration_exceeds_limit", "required_action": "reduce_session_duration"}},
	{"name": "high_risk_session_requires_reauth", "description": "High-risk federation sessions require reauthentication.", "condition": {"operation": "issue_session", "risk_score_gt": 0.7, "reauth_completed": False}, "effect": {"decision": "require_review", "reason": "high_risk_reauth_required", "required_action": "complete_reauth"}},
	{"name": "session_revocation_requires_reason", "description": "Session revocation requires a reason.", "condition": {"operation": "revoke_session", "reason_present": False}, "effect": {"decision": "deny", "reason": "session_revocation_reason_required", "required_action": "record_revocation_reason"}},
	{"name": "certificate_requires_provider", "description": "Certificates require a tenant-local provider.", "condition": {"operation": "register_certificate", "provider_present": False}, "effect": {"decision": "deny", "reason": "certificate_provider_required", "required_action": "select_provider"}},
	{"name": "certificate_requires_key", "description": "Certificates require a key identifier.", "condition": {"operation": "register_certificate", "key_present": False}, "effect": {"decision": "deny", "reason": "certificate_key_required", "required_action": "attach_key"}},
	{"name": "certificate_expiry_requires_review", "description": "Expiring certificates require rotation review.", "condition": {"operation": "health_report", "expiring_certificate_count_gt": 0, "rotation_review_recorded": False}, "effect": {"decision": "require_review", "reason": "certificate_rotation_review_required", "required_action": "review_certificate_rotation"}},
	{"name": "certificate_rotation_requires_new_key", "description": "Certificate rotation requires a new key.", "condition": {"operation": "rotate_certificate", "new_key_present": False}, "effect": {"decision": "deny", "reason": "new_certificate_key_required", "required_action": "attach_new_key"}},
	{"name": "stale_metadata_requires_refresh", "description": "Stale provider metadata requires refresh or review.", "condition": {"metadata_age_hours_gt": 24, "metadata_refresh_completed": False}, "effect": {"decision": "require_review", "reason": "metadata_refresh_required", "required_action": "refresh_provider_metadata"}},
	{"name": "provider_disable_requires_reason", "description": "Provider disablement requires a reason.", "condition": {"operation": "disable_provider", "reason_present": False}, "effect": {"decision": "deny", "reason": "provider_disable_reason_required", "required_action": "record_disable_reason"}},
	{"name": "provider_disable_revokes_sessions", "description": "Provider disablement requires active sessions to be revoked.", "condition": {"operation": "disable_provider", "active_sessions_revoked": False}, "effect": {"decision": "deny", "reason": "active_sessions_must_be_revoked", "required_action": "revoke_provider_sessions"}},
	{"name": "review_requires_independent_reviewer", "description": "Federation reviews require an independent reviewer.", "condition": {"operation": "decide_review", "reviewer_same_as_requester": True}, "effect": {"decision": "deny", "reason": "independent_federation_review_required", "required_action": "route_to_independent_reviewer"}},
	{"name": "review_decision_requires_notes", "description": "Federation review decisions require notes.", "condition": {"operation": "decide_review", "notes_present": False}, "effect": {"decision": "deny", "reason": "federation_review_notes_required", "required_action": "record_review_notes"}},
	{"name": "duplicate_pending_review_blocked", "description": "Duplicate pending federation reviews are blocked.", "condition": {"operation": "request_review", "pending_review_exists": True}, "effect": {"decision": "deny", "reason": "federation_review_already_pending", "required_action": "complete_existing_review"}},
	{"name": "batch_federation_mutation_requires_bytewax", "description": "Batch federation mutations must use Bytewax event streams.", "condition": {"operation": "batch_federation_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
	{"name": "cross_tenant_federation_access_denied", "description": "Federation records may not cross tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_federation_access_denied", "required_action": "use_tenant_local_context"}},
	{"name": "federation_state_change_requires_audit", "description": "Federation state changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "federation_audit_event_required", "required_action": "record_federation_audit_event"}},
]


UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/idfd/dashboard", "component": "IDFDDashboard", "permission": "idfd:view", "nav_group": "Overview"},
	{"name": "providers", "path": "/idfd/providers", "component": "FederationProviders", "permission": "idfd:manage_providers", "nav_group": "Providers"},
	{"name": "protocols", "path": "/idfd/protocols", "component": "ProtocolWorkbench", "permission": "idfd:manage_providers", "nav_group": "Providers"},
	{"name": "mappings", "path": "/idfd/mappings", "component": "IdentityMappings", "permission": "idfd:manage_mappings", "nav_group": "Mappings"},
	{"name": "sessions", "path": "/idfd/sessions", "component": "FederatedSessions", "permission": "idfd:view", "nav_group": "Operations"},
	{"name": "certificates", "path": "/idfd/certificates", "component": "CertificateCenter", "permission": "idfd:rotate_keys", "nav_group": "Security"},
	{"name": "scim", "path": "/idfd/scim", "component": "SCIMDirectory", "permission": "idfd:manage_providers", "nav_group": "Directory"},
	{"name": "risk", "path": "/idfd/risk", "component": "FederationRiskConsole", "permission": "idfd:view", "nav_group": "Operations"},
	{"name": "reviews", "path": "/idfd/reviews", "component": "FederationReviewQueue", "permission": "idfd:review", "nav_group": "Governance"},
	{"name": "audit", "path": "/idfd/audit", "component": "FederationAudit", "permission": "idfd:view", "nav_group": "Governance"},
	{"name": "settings", "path": "/idfd/settings", "component": "IDFDSettings", "permission": "idfd:admin", "nav_group": "Administration"},
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
		"density": "compact",
	},
	"components": {
		"provider_grid": {"icon": "network", "status_indicator": "provider-pill", "risk_style": "metadata-band"},
		"protocol_panel": {"visual": "protocol-tabs", "highlight": "crypto-chip"},
		"mapping_table": {"visual": "claim-map", "status_style": "review-chip"},
		"session_monitor": {"visual": "session-list", "status_style": "risk-chip"},
		"certificate_timeline": {"visual": "rotation-timeline", "status_style": "expiry-chip"},
		"scim_directory": {"visual": "directory-tree", "status_style": "sync-chip"},
		"risk_console": {"visual": "risk-lanes", "status_style": "reauth-chip"},
		"review_queue": {"visual": "decision-lane", "status_style": "review-chip"},
		"audit_timeline": {"visual": "event-timeline", "status_style": "identity-chip"},
	},
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
			"requires_theme": True,
		},
		"theme": deepcopy(THEME),
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
