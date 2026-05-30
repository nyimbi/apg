"""APG Shutdown and Lifecycle Control capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import (
	SUPPORTED_SHDN_AGENT_ROLES,
	SUPPORTED_SHDN_AGENT_RUNTIMES,
	evaluate_capability_rules,
	get_capability_contract,
	streaming_manifest,
)
from .models import ShdnAgentRecord
from .service import ShdnService

__version__ = "1.0.0"
__capability_id__ = "shdn"
__capability_name__ = "Shutdown and Lifecycle Control"
__apg_dependencies__ = ["moni", "hlth", "bkup", "audl", "envm"]

capability_metadata: dict[str, Any] = {
	"name": "shdn",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant service lifecycle, graceful shutdown, restart orchestration, backup gates, and operational safety controls",
	"category": "platform",
	"subcategory": "lifecycle_control",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["service_lifecycle", "shutdown_orchestration", "restart_plans", "backup_gates", "operational_safety", "shdn_agents"],
	"permissions": ["shdn:view", "shdn:plan", "shdn:execute", "shdn:approve", "shdn:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register SHDN with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "shdn",
		"aliases": ["shutdown", "lifecycle-control", "service-lifecycle"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["depl", "logt", "cicd"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"service_lifecycle": "Manage service start, pause, drain, restart, shutdown, and retirement states",
			"shutdown_orchestration": "Coordinate graceful shutdown workflows with dependency and health gates",
			"restart_plans": "Define restart windows, sequencing, rollback, and recovery expectations",
			"backup_gates": "Require backup and restore evidence before destructive lifecycle operations",
			"shdn_agents": "Compose governed AI agents into lifecycle planning, dependency review, shutdown review, recovery review, approval review, and audit review lanes",
			"capability_rules": "Evaluate deterministic lifecycle-governance rules",
			"visual_theming": "Apply lifecycle control theme tokens and components"
		},
		"endpoints": {"services": "/shdn/api/v1/services", "plans": "/shdn/api/v1/plans", "executions": "/shdn/api/v1/executions", "approvals": "/shdn/api/v1/approvals", "recovery": "/shdn/api/v1/recovery", "agents": "/shdn/api/v1/agents", "policy": "/shdn/api/v1/policy"},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"streaming": streaming_manifest(),
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get SHDN capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = [
	"SUPPORTED_SHDN_AGENT_ROLES",
	"SUPPORTED_SHDN_AGENT_RUNTIMES",
	"ShdnAgentRecord",
	"ShdnService",
	"capability_metadata",
	"register_capability",
	"get_capability_info",
	"get_capability_contract",
	"evaluate_capability_rules",
	"streaming_manifest",
	"__version__",
	"__capability_id__",
	"__capability_name__",
	"__apg_dependencies__",
]
