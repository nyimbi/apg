"""APG Platform Foundation capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import (
	SUPPORTED_PLFD_AGENT_ROLES,
	SUPPORTED_PLFD_AGENT_RUNTIMES,
	evaluate_capability_rules,
	get_capability_contract,
	streaming_manifest,
)
from .models import PlfdAgent
from .service import PlfdService

__version__ = "1.0.0"
__capability_id__ = "plfd"
__capability_name__ = "Platform Foundation"
__apg_dependencies__ = ["conf", "mten", "auth", "audl"]

capability_metadata: dict[str, Any] = {
	"name": "plfd",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Platform baseline services, dependency posture, configuration policy, operational readiness, and foundation governance",
	"category": "platform",
	"subcategory": "foundation",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["foundation_registry", "dependency_posture", "configuration_baselines", "readiness_gates", "platform_governance", "plfd_agents"],
	"permissions": ["plfd:view", "plfd:manage_services", "plfd:manage_baselines", "plfd:approve_changes", "plfd:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register PLFD with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "plfd",
		"aliases": ["platform-foundation", "foundation", "platform-core"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["moni", "hlth", "regy", "secu", "plgn"],
		"provides": contract["provides"],
		"requires": contract["requires"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"foundation_registry": "Track platform foundation services, owners, tiers, readiness, and lifecycle state",
			"dependency_posture": "Validate baseline dependency health and core service availability",
			"configuration_baselines": "Manage required configuration, tenant, auth, and audit baselines",
			"readiness_gates": "Gate platform changes on health, monitoring, security, and rollback evidence",
			"plfd_agents": "Register scoped AI foundation agents for service, dependency, baseline, readiness, change, and security review",
			"capability_rules": "Evaluate deterministic platform-foundation rules",
			"event_streaming": "Emit platform foundation lifecycle events through Bytewax",
			"visual_theming": "Apply platform foundation theme tokens and components"
		},
		"endpoints": {"services": "/plfd/api/v1/services", "dependencies": "/plfd/api/v1/dependencies", "baselines": "/plfd/api/v1/baselines", "readiness": "/plfd/api/v1/readiness", "changes": "/plfd/api/v1/changes", "agents": "/plfd/api/v1/agents", "audit": "/plfd/api/v1/audit"},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"streaming": contract["streaming"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get PLFD capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = [
	"PlfdAgent",
	"PlfdService",
	"SUPPORTED_PLFD_AGENT_ROLES",
	"SUPPORTED_PLFD_AGENT_RUNTIMES",
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
