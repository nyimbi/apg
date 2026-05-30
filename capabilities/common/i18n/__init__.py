"""APG Internationalization (I18N) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import (
	AFRICAN_LANGUAGE_CODES,
	SUPPORTED_I18N_AGENT_ROLES,
	SUPPORTED_I18N_AGENT_RUNTIMES,
	SUPPORTED_LANGUAGE_CODES,
	evaluate_capability_rules,
	get_capability_contract,
	streaming_manifest,
)
from .models import I18nAgent
from .service import I18nService

__version__ = "1.0.0"
__capability_id__ = "i18n"
__capability_name__ = "Internationalization"
__apg_dependencies__ = ["conf", "nlpc", "auth"]

capability_metadata: dict[str, Any] = {
	"name": "i18n",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant-aware localization, translations, locale policy, content fallback, and language-governance services",
	"category": "specialized_ai_analytics",
	"subcategory": "internationalization",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["locale_management", "translation_memory", "content_localization", "language_fallbacks", "regional_formatting"],
	"permissions": ["i18n:view", "i18n:translate", "i18n:manage_locales", "i18n:publish", "i18n:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register I18N with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "i18n",
		"aliases": ["internationalization", "localization", "translation"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["audl", "mchn", "help", "them"],
		"provides": contract["provides"],
		"requires": contract["requires"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"locale_management": "Configure tenant locales, regional formats, timezones, and fallback chains",
			"translation_memory": "Manage approved translations and glossary-aware reuse",
			"content_localization": "Localize UI, help, notification, and workflow content",
			"language_fallbacks": "Resolve missing translations through governed fallback policy",
			"language_policy": "Govern supported languages, coverage gates, fallback policy, and regional formats",
			"i18n_agents": "Register scoped AI localization agents for translation, review, glossary, and release tasks",
			"capability_rules": "Evaluate deterministic localization-governance rules",
			"event_streaming": "Emit localization lifecycle events through Bytewax",
			"visual_theming": "Apply localization-workbench theme tokens and components"
		},
		"endpoints": {
			"locales": "/i18n/api/v1/locales",
			"translations": "/i18n/api/v1/translations",
			"glossaries": "/i18n/api/v1/glossaries",
			"publishing": "/i18n/api/v1/publishing",
			"coverage": "/i18n/api/v1/coverage",
			"agents": "/i18n/api/v1/agents",
			"audit": "/i18n/api/v1/audit"
		},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"streaming": contract["streaming"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get I18N capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = [
	"AFRICAN_LANGUAGE_CODES",
	"I18nAgent",
	"I18nService",
	"SUPPORTED_I18N_AGENT_ROLES",
	"SUPPORTED_I18N_AGENT_RUNTIMES",
	"SUPPORTED_LANGUAGE_CODES",
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
