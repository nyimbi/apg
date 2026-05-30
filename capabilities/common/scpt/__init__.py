"""APG Custom Scripting Engine (SCPT) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract

__version__ = "1.0.0"
__capability_id__ = "scpt"
__capability_name__ = "Custom Scripting Engine"
__apg_dependencies__ = ["wflo", "secu", "auth", "audl"]

capability_metadata: dict[str, Any] = {
	"name": "scpt",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant-aware custom scripts, secure sandboxes, workflow extensions, package policies, scripting agents, and execution governance",
	"category": "workflow_automation",
	"subcategory": "custom_scripting",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["script_registry", "secure_sandbox", "workflow_extensions", "package_policy", "script_execution", "scripting_agents", "script_governance"],
	"permissions": ["scpt:view", "scpt:write", "scpt:execute", "scpt:approve", "scpt:audit", "scpt:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register SCPT with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "scpt",
		"aliases": ["scripting", "custom_scripting", "script_engine"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["ncod", "schd", "aicr", "moni", "them"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"script_registry": "Version, review, publish, and retire tenant-scoped scripts",
			"secure_sandbox": "Run scripts inside constrained sandboxes with resource and network policy",
			"workflow_extensions": "Attach scripts to workflow steps, triggers, and scheduled jobs",
			"package_policy": "Control allowed packages, secrets, imports, and runtime permissions",
			"scripting_agents": "Register scoped AI scripting assistants for authoring, review, policy advice, tests, and runtime triage",
			"script_governance": "Govern review, publication, retirement, execution evidence, and Bytewax event policy",
			"capability_rules": "Evaluate deterministic scripting-governance rules",
			"visual_theming": "Apply script-workbench theme tokens and components"
		},
		"endpoints": {
			"scripts": "/scpt/api/v1/scripts",
			"executions": "/scpt/api/v1/executions",
			"sandboxes": "/scpt/api/v1/sandboxes",
			"packages": "/scpt/api/v1/packages",
			"approvals": "/scpt/api/v1/approvals",
			"agents": "/scpt/api/v1/agents",
			"audit": "/scpt/api/v1/audit"
		},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"streaming": contract["streaming"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get SCPT capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = ["capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
