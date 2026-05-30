"""APG Help and Knowledge Base (HELP) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract

__version__ = "1.0.0"
__capability_id__ = "help"
__capability_name__ = "Help and Knowledge Base"
__apg_dependencies__ = ["ragn", "srch", "nlpc"]

capability_metadata: dict[str, Any] = {
	"name": "help",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant-aware help center, source registry, knowledge articles, cited answers, localization, curation, audit, and support analytics",
	"category": "collaboration_communication",
	"subcategory": "help_knowledge_base",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["help_center", "source_registry", "knowledge_articles", "assisted_answers", "article_localization", "content_curation", "support_analytics"],
	"permissions": ["help:view", "help:ask", "help:edit_articles", "help:publish", "help:audit", "help:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register HELP with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "help",
		"aliases": ["help_center", "knowledge_base", "support_knowledge"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["auth", "audl", "chat", "ntfy", "them"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"help_center": "Expose tenant-scoped help center navigation and search",
			"source_registry": "Register and approve governed source references for help content",
			"knowledge_articles": "Create, curate, localize, and publish knowledge articles",
			"assisted_answers": "Generate cited answers from approved help sources",
			"article_localization": "Manage translated article variants with fallback locale controls",
			"content_curation": "Review article quality, feedback, freshness, and ownership",
			"capability_rules": "Evaluate deterministic help-center governance rules",
			"visual_theming": "Apply support-knowledge theme tokens and components"
		},
		"endpoints": {
			"sources": "/help/api/v1/sources",
			"articles": "/help/api/v1/articles",
			"search": "/help/api/v1/search",
			"answers": "/help/api/v1/answers",
			"feedback": "/help/api/v1/feedback",
			"localization": "/help/api/v1/localization",
			"curation": "/help/api/v1/curation",
			"audit": "/help/api/v1/audit"
		},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get HELP capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = ["capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
