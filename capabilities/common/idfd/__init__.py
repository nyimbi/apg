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
	"provides": ["federated_sso", "saml_identity_provider", "oidc_broker", "identity_mapping", "certificate_rotation"],
	"permissions": ["idfd:view", "idfd:manage_providers", "idfd:manage_mappings", "idfd:rotate_keys", "idfd:admin"]
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
		"optional_dependencies": ["audl", "secu", "mten", "ztna"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"federated_sso": "Broker tenant-scoped SSO across identity providers",
			"saml_identity_provider": "Manage SAML metadata, assertions, signing, and encryption",
			"oidc_broker": "Manage OIDC clients, redirect allowlists, scopes, and sessions",
			"identity_mapping": "Map external identities, groups, and claims to APG principals",
			"capability_rules": "Evaluate deterministic federation-governance rules",
			"visual_theming": "Apply federation-console theme tokens and components"
		},
		"endpoints": {
			"providers": "/idfd/api/v1/providers",
			"protocols": "/idfd/api/v1/protocols",
			"mappings": "/idfd/api/v1/mappings",
			"sessions": "/idfd/api/v1/sessions",
			"certificates": "/idfd/api/v1/certificates"
		},
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
