"""APG Website Builder capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import (
	SUPPORTED_WSBL_AGENT_ROLES,
	SUPPORTED_WSBL_AGENT_RUNTIMES,
	evaluate_capability_rules,
	event_stream_name,
	get_capability_contract,
	streaming_manifest,
)
from .service import WsblService
from .website_runtime import WebsiteAgentRecord

__version__ = "1.0.0"
__capability_id__ = "wsbl"
__capability_name__ = "Website Builder"
__apg_dependencies__ = ["them", "auth", "ncod", "accs", "cons"]

capability_metadata: dict[str, Any] = {
	"name": "wsbl",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant site composition, page editing, component governance, publishing workflows, and privacy-aware public experiences",
	"category": "experience",
	"subcategory": "website_builder",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["site_management", "page_composition", "component_library", "publishing_workflows", "site_theming", "wsbl_agents"],
	"permissions": ["wsbl:view", "wsbl:build", "wsbl:publish", "wsbl:manage_sites", "wsbl:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register WSBL with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "wsbl",
		"aliases": ["website-builder", "site-builder", "pages"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["i18n", "mchn"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"site_management": "Manage tenant sites, ownership, domains, environments, and lifecycle",
			"page_composition": "Build pages from governed no-code sections and content blocks",
			"component_library": "Curate custom components with review and reuse controls",
			"publishing_workflows": "Approve, schedule, publish, rollback, and audit site changes",
			"review_evidence": "Persist review-required components, review-required publish requests, denied publish attempts, and policy audit evidence",
			"capability_rules": "Evaluate deterministic site-governance rules",
			"visual_theming": "Apply site-builder theme tokens and page components",
			"wsbl_agents": "Govern site, component, accessibility, privacy, SEO, and publish review agents"
		},
		"endpoints": {
			"sites": "/wsbl/api/v1/sites",
			"pages": "/wsbl/api/v1/pages",
			"components": "/wsbl/api/v1/components",
			"publishing": "/wsbl/api/v1/publishing",
			"analytics": "/wsbl/api/v1/analytics",
			"agents": "/wsbl/api/v1/agents",
			"pending_reviews": "/wsbl/api/v1/pending-reviews",
			"policy": "/wsbl/api/v1/policy"
		},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"streaming": contract["streaming"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get WSBL capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = [
	"SUPPORTED_WSBL_AGENT_ROLES",
	"SUPPORTED_WSBL_AGENT_RUNTIMES",
	"WebsiteAgentRecord",
	"WsblService",
	"capability_metadata",
	"event_stream_name",
	"evaluate_capability_rules",
	"get_capability_contract",
	"get_capability_info",
	"register_capability",
	"streaming_manifest",
	"__version__",
	"__capability_id__",
	"__capability_name__",
	"__apg_dependencies__",
]
