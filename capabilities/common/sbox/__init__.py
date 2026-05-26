"""APG Sandbox/Testing Environment capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract

__version__ = "1.0.0"
__capability_id__ = "sbox"
__capability_name__ = "Sandbox/Testing Environment"
__apg_dependencies__ = ["plgn", "secu", "envm"]

capability_metadata: dict[str, Any] = {
	"name": "sbox",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant sandbox environments, isolated test runs, synthetic datasets, safety policy, and experiment audit trails",
	"category": "platform",
	"subcategory": "sandboxing",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["sandbox_registry", "isolation_profiles", "test_runs", "synthetic_datasets", "safety_policy"],
	"permissions": ["sbox:view", "sbox:create", "sbox:run_tests", "sbox:manage_policy", "sbox:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register SBOX with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "sbox",
		"aliases": ["sandbox", "testing-environment", "safe-experiments"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["cicd", "depl", "logt", "agnt"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"sandbox_registry": "Create and govern tenant sandboxes, templates, owners, and lifecycles",
			"isolation_profiles": "Apply network, data, secret, and runtime isolation policies",
			"test_runs": "Execute safe experiments, plugin tests, and integration checks",
			"synthetic_datasets": "Manage sanitized or generated datasets for test execution",
			"capability_rules": "Evaluate deterministic sandbox-governance rules",
			"visual_theming": "Apply sandbox operations theme tokens and components"
		},
		"endpoints": {"sandboxes": "/sbox/api/v1/sandboxes", "templates": "/sbox/api/v1/templates", "runs": "/sbox/api/v1/runs", "datasets": "/sbox/api/v1/datasets", "policies": "/sbox/api/v1/policies"},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get SBOX capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = ["capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
