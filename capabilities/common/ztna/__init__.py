"""APG Zero Trust Network Access (ZTNA) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract

__version__ = "1.0.0"
__capability_id__ = "ztna"
__capability_name__ = "Zero Trust Network Access"
__apg_dependencies__ = ["auth", "secu", "mfau", "moni", "audl", "idfd", "anom", "mqeb", "cach"]

capability_metadata: dict[str, Any] = {
	"name": "ztna",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Identity, device, resource, posture, and risk-aware zero-trust access control",
	"category": "security_compliance",
	"subcategory": "zero_trust_access",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": [
		"zero_trust_policies",
		"identity_context",
		"device_posture",
		"resource_access_broker",
		"continuous_verification",
		"risk_based_access",
		"session_governance",
		"access_reviews",
		"zero_trust_audit",
		"zero_trust_agent_composition",
		"bytewax_lifecycle_batches",
	],
	"permissions": ["ztna:view", "ztna:manage_policies", "ztna:approve_access", "ztna:manage_devices", "ztna:review", "ztna:audit", "ztna:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register ZTNA with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "ztna",
		"aliases": ["zero_trust_network_access", "zero_trust", "risk_based_access"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": [],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"capabilities": {
			"zero_trust_policies": "Bind access decisions to user, device, resource, risk, and tenant policy",
			"identity_context": "Track verified, federated, privileged, and suspended identity context",
			"device_posture": "Evaluate managed device health, attestation, patch, and compliance signals",
			"resource_access_broker": "Broker least-privilege application and service access",
			"continuous_verification": "Continuously re-check risk and revoke sessions when context changes",
			"session_governance": "Start, reevaluate, close, and audit governed resource sessions",
			"access_reviews": "Route high-risk and privileged access through explicit review",
			"zero_trust_audit": "Record access, session, policy, identity, device, and resource audit events",
			"zero_trust_agent_composition": "Register provider-neutral AI agents for governed zero-trust review and lifecycle scopes",
			"bytewax_lifecycle_batches": "Validate zero-trust lifecycle mutation batches through Bytewax stream metadata",
			"capability_rules": "Evaluate deterministic zero-trust access rules",
			"visual_theming": "Apply zero-trust operations theme tokens and components"
		},
		"endpoints": {
			"status": "/ztna/api/v1/status",
			"policies": "/ztna/api/v1/policies",
			"identities": "/ztna/api/v1/identities",
			"devices": "/ztna/api/v1/devices",
			"resources": "/ztna/api/v1/resources",
			"access": "/ztna/api/v1/access",
			"sessions": "/ztna/api/v1/sessions",
			"risk": "/ztna/api/v1/risk",
			"reviews": "/ztna/api/v1/reviews",
			"agents": "/ztna/api/v1/agents",
			"lifecycle": "/ztna/api/v1/lifecycle",
			"audit": "/ztna/api/v1/audit"
		},
		"adapters": contract["configuration"]["adapters"],
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get ZTNA capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = ["capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
