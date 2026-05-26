"""APG Deployment Management (DEPL) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract

__version__ = "1.0.0"
__capability_id__ = "depl"
__capability_name__ = "Deployment Management"
__apg_dependencies__ = ["logt", "moni", "hlth"]

capability_metadata: dict[str, Any] = {
	"name": "depl",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant-aware deployment plans, releases, rollout strategies, health gates, rollback, and deployment audit",
	"category": "infrastructure_operations",
	"subcategory": "deployment_management",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["release_management", "deployment_rollouts", "health_gates", "rollback_control", "deployment_audit"],
	"permissions": ["depl:view", "depl:plan", "depl:deploy", "depl:rollback", "depl:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register DEPL with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "depl",
		"aliases": ["deployment", "release_management", "rollouts"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["cicd", "envm", "ntfy", "comp"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"release_management": "Manage release versions, manifests, approvals, and deployment evidence",
			"deployment_rollouts": "Run blue-green, canary, rolling, and manual deployment strategies",
			"health_gates": "Block or pause releases based on health, monitoring, and trace evidence",
			"rollback_control": "Execute governed rollback and remediation procedures",
			"capability_rules": "Evaluate deterministic deployment-governance rules",
			"visual_theming": "Apply deployment-operations theme tokens and components"
		},
		"endpoints": {"releases": "/depl/api/v1/releases", "deployments": "/depl/api/v1/deployments", "rollouts": "/depl/api/v1/rollouts", "health": "/depl/api/v1/health-gates", "rollback": "/depl/api/v1/rollback"},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get DEPL capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = ["capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
