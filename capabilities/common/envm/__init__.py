"""APG Environment Management capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import (
	SUPPORTED_ENVM_AGENT_ROLES,
	SUPPORTED_ENVM_AGENT_RUNTIMES,
	SUPPORTED_STAGES,
	evaluate_capability_rules,
	get_capability_contract,
	streaming_manifest,
)
from .models import EnvmAgent
from .service import EnvmService

__version__ = "1.0.0"
__capability_id__ = "envm"
__capability_name__ = "Environment Management"
__apg_dependencies__ = ["auth", "conf", "audl", "depl", "keym", "moni"]

capability_metadata: dict[str, Any] = {
	"name": "envm",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant-aware environment inventory, promotion, drift, secrets, policy, AI-agent review, and multi-environment governance",
	"category": "infrastructure_operations",
	"subcategory": "environment_management",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": get_capability_contract()["provides"],
	"permissions": ["envm:view", "envm:manage_environments", "envm:promote", "envm:manage_secrets", "envm:govern", "envm:admin"],
	"streaming": streaming_manifest(),
}


def register_capability() -> dict[str, Any]:
	"""Register ENVM with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "envm",
		"aliases": ["environment_management", "environments", "environment_governance"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": contract["requires"],
		"optional_dependencies": ["cicd", "secu"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"environment_inventory": "Register tenant environments, regions, stages, ownership, and status",
			"environment_promotion": "Promote configuration and releases across governed paths",
			"configuration_drift": "Detect and remediate drift between declared and observed state",
			"secret_scopes": "Manage environment-scoped secret references and access policies",
			"envm_agents": "Register AI agents for environment, promotion, drift, secret, and policy review",
			"capability_rules": "Evaluate deterministic environment-governance rules",
			"visual_theming": "Apply environment-management theme tokens and components",
		},
		"endpoints": {
			"environments": "/envm/api/v1/environments",
			"promotion": "/envm/api/v1/promotion",
			"drift": "/envm/api/v1/drift",
			"secrets": "/envm/api/v1/secrets",
			"policies": "/envm/api/v1/policies",
			"agents": "/envm/api/v1/agents",
		},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"streaming": contract["streaming"],
		"permissions": capability_metadata["permissions"],
	}


def get_capability_info() -> dict[str, Any]:
	"""Get ENVM capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = [
	"EnvmAgent",
	"EnvmService",
	"SUPPORTED_ENVM_AGENT_ROLES",
	"SUPPORTED_ENVM_AGENT_RUNTIMES",
	"SUPPORTED_STAGES",
	"capability_metadata",
	"evaluate_capability_rules",
	"get_capability_contract",
	"get_capability_info",
	"register_capability",
	"streaming_manifest",
	"__apg_dependencies__",
	"__capability_id__",
	"__capability_name__",
	"__version__",
]
