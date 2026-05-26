"""APG Video Conferencing (VIDC) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract

__version__ = "1.0.0"
__capability_id__ = "vidc"
__capability_name__ = "Video Conferencing"
__apg_dependencies__ = ["colb", "mqeb", "cvsn"]

capability_metadata: dict[str, Any] = {
	"name": "vidc",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant-aware video meetings, rooms, recordings, captions, moderation, and collaboration integrations",
	"category": "collaboration_communication",
	"subcategory": "video_conferencing",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["video_meetings", "meeting_rooms", "recordings", "live_captions", "meeting_moderation"],
	"permissions": ["vidc:view", "vidc:schedule", "vidc:join", "vidc:moderate", "vidc:manage_recordings", "vidc:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register VIDC with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "vidc",
		"aliases": ["video_conferencing", "meetings", "video_meetings"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["ntfy", "nlpc", "audl", "auth"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"video_meetings": "Create and join tenant-scoped realtime video meetings",
			"meeting_rooms": "Manage rooms, lobbies, participants, and access policies",
			"recordings": "Capture, encrypt, retain, and audit meeting recordings",
			"live_captions": "Generate captions and transcript artifacts with policy controls",
			"capability_rules": "Evaluate deterministic video-conferencing rules",
			"visual_theming": "Apply meeting-room theme tokens and components"
		},
		"endpoints": {
			"meetings": "/vidc/api/v1/meetings",
			"rooms": "/vidc/api/v1/rooms",
			"participants": "/vidc/api/v1/participants",
			"recordings": "/vidc/api/v1/recordings",
			"captions": "/vidc/api/v1/captions"
		},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get VIDC capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = ["capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
