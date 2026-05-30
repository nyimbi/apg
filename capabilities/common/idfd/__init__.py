"""APG Identity Federation (IDFD) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract

__version__ = "1.0.0"
__capability_id__ = "idfd"
__capability_name__ = "Identity Federation"
__apg_dependencies__ = ["auth", "mfau", "encr"]

capability_metadata: dict[str, Any] = {
	"name": "idfd",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Enterprise SSO federation across SAML, OIDC, SCIM, session, mapping, and certificate governance",
	"category": "security_compliance",
	"subcategory": "identity_federation",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": [
		"federated_sso",
		"saml_identity_provider",
		"oidc_broker",
		"ldap_federation",
		"scim_directory_sync",
		"identity_mapping",
		"federated_sessions",
		"certificate_rotation",
		"federation_reviews",
		"federation_audit",
	],
	"permissions": ["idfd:view", "idfd:manage_providers", "idfd:manage_mappings", "idfd:rotate_keys", "idfd:review", "idfd:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register IDFD with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "idfd",
		"aliases": ["identity_federation", "sso", "federated_identity"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["audl", "secu", "mten", "ztna", "keym", "moni", "cach"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"federated_sso": "Broker tenant-scoped SSO across identity providers",
			"saml_identity_provider": "Manage SAML metadata, assertions, signing, and encryption",
			"oidc_broker": "Manage OIDC clients, redirect allowlists, scopes, and sessions",
			"ldap_federation": "Govern LDAP federation through TLS and owner controls",
			"scim_directory_sync": "Synchronize and deprovision federated identities with SCIM guardrails",
			"identity_mapping": "Map external identities, groups, and claims to APG principals",
			"federated_sessions": "Issue, revoke, and inspect tenant-scoped federated sessions",
			"federation_reviews": "Route stale metadata, sensitive claims, high-risk sessions, and certificate rotations to review",
			"federation_audit": "Audit provider, mapping, session, certificate, and health lifecycle events",
			"capability_rules": "Evaluate deterministic federation-governance rules",
			"visual_theming": "Apply federation-console theme tokens and components"
		},
		"endpoints": {
			"status": "/idfd/api/v1/status",
			"providers": "/idfd/api/v1/providers",
			"protocols": "/idfd/api/v1/protocols",
			"mappings": "/idfd/api/v1/mappings",
			"sessions": "/idfd/api/v1/sessions",
			"certificates": "/idfd/api/v1/certificates",
			"scim": "/idfd/api/v1/scim",
			"risk": "/idfd/api/v1/risk",
			"reviews": "/idfd/api/v1/reviews",
			"audit": "/idfd/api/v1/audit"
		},
		"adapters": contract["configuration"]["adapters"],
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get IDFD capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = ["capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
