"""APG IoT Device Integration capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import (
	SUPPORTED_IOTD_AGENT_ROLES,
	SUPPORTED_IOTD_AGENT_RUNTIMES,
	evaluate_capability_rules,
	get_capability_contract,
	streaming_manifest,
)
from .models import IotdAgent
from .service import IotdService

__version__ = "1.0.0"
__capability_id__ = "iotd"
__capability_name__ = "IoT Device Integration"
__apg_dependencies__ = ["auth", "encr", "audl", "conf"]

capability_metadata: dict[str, Any] = {
	"name": "iotd",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant IoT device identity, telemetry ingestion, command governance, firmware lifecycle, and secure edge connectivity",
	"category": "advanced_infrastructure",
	"subcategory": "iot",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["device_registry", "telemetry_ingestion", "command_dispatch", "firmware_lifecycle", "device_security", "device_health", "iotd_agents"],
	"permissions": ["iotd:view", "iotd:register", "iotd:command", "iotd:manage_firmware", "iotd:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register IOTD with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "iotd",
		"aliases": ["iot", "devices", "device-integration"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["edge", "dtwn", "logt", "moni"],
		"provides": contract["provides"],
		"requires": contract["requires"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"device_registry": "Register device identities, certificates, owners, and lifecycle state",
			"telemetry_ingestion": "Ingest encrypted telemetry streams through APG event infrastructure",
			"command_dispatch": "Dispatch governed device commands with approval and audit trails",
			"firmware_lifecycle": "Track signed firmware artifacts, rollout windows, and rollback policy",
			"device_health": "Summarize stale devices, online state, pending commands, and firmware risk",
			"iotd_agents": "Register scoped AI device-operations agents for fleet, telemetry, command, firmware, and security work",
			"capability_rules": "Evaluate deterministic IoT device-governance rules",
			"event_streaming": "Emit IoT lifecycle events through Bytewax",
			"visual_theming": "Apply IoT operations theme tokens and components"
		},
		"endpoints": {"devices": "/iotd/api/v1/devices", "telemetry": "/iotd/api/v1/telemetry", "commands": "/iotd/api/v1/commands", "firmware": "/iotd/api/v1/firmware", "security": "/iotd/api/v1/security", "agents": "/iotd/api/v1/agents", "audit": "/iotd/api/v1/audit"},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"streaming": contract["streaming"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get IOTD capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = [
	"IotdAgent",
	"IotdService",
	"SUPPORTED_IOTD_AGENT_ROLES",
	"SUPPORTED_IOTD_AGENT_RUNTIMES",
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
