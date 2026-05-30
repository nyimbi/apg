"""APG Chat and Messaging (CHAT) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .service import ChatService

__version__ = "1.0.0"
__capability_id__ = "chat"
__capability_name__ = "Chat and Messaging"
__apg_dependencies__ = ["ntfy", "mqeb", "auth"]

capability_metadata: dict[str, Any] = {
	"name": "chat",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant-aware direct messaging, rooms, moderation, retention, AI-agent participation, realtime delivery, and collaboration hooks",
	"category": "collaboration_communication",
	"subcategory": "chat",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["direct_messages", "team_rooms", "message_moderation", "presence", "message_retention", "agent_participants", "delivery_audit"],
	"permissions": ["chat:view", "chat:send", "chat:manage_rooms", "chat:moderate", "chat:audit", "chat:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register CHAT with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "chat",
		"aliases": ["chat_messaging", "messaging", "team_chat"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["audl", "nlpc", "colb", "mten", "secu", "cach"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"direct_messages": "Send tenant-scoped one-to-one and small-group messages",
			"team_rooms": "Manage channels, rooms, membership, and presence",
			"message_moderation": "Moderate content, attachments, retention, and policy actions",
			"presence": "Track online, typing, receipt, and availability state",
			"agent_participants": "Register scoped AI agents as disclosed chat participants",
			"delivery_audit": "Audit rooms, messages, moderation decisions, and state changes",
			"capability_rules": "Evaluate deterministic chat-governance rules",
			"visual_theming": "Apply team-chat theme tokens and components"
		},
		"endpoints": {
			"rooms": "/chat/api/v1/rooms",
			"messages": "/chat/api/v1/messages",
			"presence": "/chat/api/v1/presence",
			"moderation": "/chat/api/v1/moderation",
			"retention": "/chat/api/v1/retention",
			"agents": "/chat/api/v1/agents",
			"audit": "/chat/api/v1/audit"
		},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"adapters": contract["configuration"]["adapters"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get CHAT capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = ["ChatService", "capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
