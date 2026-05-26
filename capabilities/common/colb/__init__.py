"""APG Collaboration Tools (COLB) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract

__version__ = "1.0.0"
__capability_id__ = "colb"
__capability_name__ = "Collaboration Tools"
__capability_code__ = "REAL_TIME_COLLABORATION"
__apg_dependencies__ = ["chat", "ntfy", "auth"]
__composition_keywords__ = ["requires_real_time_collaboration", "integrates_with_real_time_collaboration", "uses_real_time_collaboration"]

capability_metadata: dict[str, Any] = {
	"name": "colb",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Real-time collaborative workspaces, shared documents, presence, annotations, sessions, and protocol orchestration",
	"category": "collaboration_communication",
	"subcategory": "collaboration",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["collaborative_workspaces", "shared_sessions", "presence_sync", "annotation_threads", "realtime_protocols"],
	"permissions": ["colb:view", "colb:create_workspace", "colb:collaborate", "colb:manage_sessions", "colb:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register COLB with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "colb",
		"aliases": ["collaboration", "real_time_collaboration", "shared_workspaces"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["audl", "mqeb", "vidc", "wflo"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"collaborative_workspaces": "Create tenant-scoped shared spaces with membership and policy controls",
			"shared_sessions": "Coordinate realtime sessions, cursors, edits, and protocol connections",
			"presence_sync": "Synchronize availability, participants, and activity across collaboration tools",
			"annotation_threads": "Attach threaded comments, decisions, and review actions to work artifacts",
			"capability_rules": "Evaluate deterministic collaboration-governance rules",
			"visual_theming": "Apply collaboration-workspace theme tokens and components"
		},
		"endpoints": {
			"workspaces": "/colb/api/v1/workspaces",
			"sessions": "/colb/api/v1/sessions",
			"presence": "/colb/api/v1/presence",
			"annotations": "/colb/api/v1/annotations",
			"protocols": "/colb/api/v1/protocols"
		},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get COLB capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["composition_keywords"] = __composition_keywords__
	info["contract"] = get_capability_contract()
	return info


__all__ = ["capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__capability_code__", "__apg_dependencies__", "__composition_keywords__"]
