"""Executable capability contract for APG Data Synchronisation."""
from __future__ import annotations
from typing import Any

CAPABILITY_ID = "int_dsy"
CAPABILITY_NAME = "Data Synchronisation"
CAPABILITY_VERSION = "1.0.0"
CAPABILITY_DOMAIN = "int"
CAPABILITY_DESCRIPTION = (
    "No-code data sync configuration: bidirectional sync between APG capabilities "
    "and external systems, CDC (change data capture), conflict resolution, "
    "field mapping, and sync scheduling."
)

SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["sync_admin", "sync_viewer", "integration_developer"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"sync": {
		"default_batch_size": 500,
		"default_frequency_minutes": 15,
		"conflict_resolution": "source_wins",  # source_wins | target_wins | newest | manual
		"cdc_enabled": True,
	},
	"governance": {"require_tenant_context": True, "audit_events": True},
}

PROVIDES = [
	"sync_configuration", "bidirectional_sync", "change_data_capture",
	"field_mapping", "conflict_resolution", "sync_monitoring", "sync_scheduling",
]
REQUIRES = ["auth", "audl", "ntfy", "int_esb"]
PUBLISHES = ["sync.completed", "sync.conflict_detected", "sync.failed"]
SUBSCRIBES = []

UI_ROUTES = [
	{"name": "syncs", "path": "/int-dsy/syncs", "component": "DsySyncList", "permission": "int_dsy:view", "nav_group": "Sync Jobs"},
	{"name": "mappings", "path": "/int-dsy/mappings", "component": "DsyFieldMappings", "permission": "int_dsy:design", "nav_group": "Configuration"},
	{"name": "history", "path": "/int-dsy/history", "component": "DsySyncHistory", "permission": "int_dsy:view", "nav_group": "Monitoring"},
	{"name": "settings", "path": "/int-dsy/settings", "component": "DsySettings", "permission": "int_dsy:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "int_dsy_theme",
	"tokens": {
		"color.primary": "#0F4C75", "color.accent": "#1B98E0",
		"surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF",
		"text.primary": "#111827", "border.radius": "8px", "density": "compact",
	},
}


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	return {
		"id": CAPABILITY_ID, "name": CAPABILITY_NAME, "version": CAPABILITY_VERSION,
		"domain": CAPABILITY_DOMAIN, "description": CAPABILITY_DESCRIPTION,
		"provides": PROVIDES, "requires": REQUIRES, "publishes": PUBLISHES,
		"subscribes": SUBSCRIBES, "ui_routes": UI_ROUTES, "theme": THEME,
		"configuration": DEFAULT_CONFIGURATION,
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	if not context.get("tenant_context_present"):
		return {"decision": "deny", "matched_rules": ["tenant_required"], "actions": [{"type": "deny", "reason": "missing_tenant_context"}]}
	return {"decision": "allow", "matched_rules": [], "actions": []}
