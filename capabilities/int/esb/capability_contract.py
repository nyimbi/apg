"""Executable capability contract for APG Enterprise Service Bus."""
from __future__ import annotations
from typing import Any

CAPABILITY_ID = "int_esb"
CAPABILITY_NAME = "Enterprise Service Bus"
CAPABILITY_VERSION = "1.0.0"
CAPABILITY_DOMAIN = "int"
CAPABILITY_DESCRIPTION = (
    "Integration flow designer with drag-drop BPMN-style connector wiring, "
    "message transformation, routing, error handling and retry. Built over "
    "NATS JetStream and Temporal for durable delivery. Mulesoft/Boomi equivalent."
)

SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["integration_developer", "integration_admin", "integration_viewer"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"flows": {
		"max_concurrent_flows": 100,
		"default_retry_attempts": 3,
		"default_timeout_seconds": 30,
		"dead_letter_enabled": True,
	},
	"transformations": {
		"supported_formats": ["json", "xml", "csv", "avro"],
		"jmespath_enabled": True,
		"jinja2_templates_enabled": True,
	},
	"governance": {"require_tenant_context": True, "audit_events": True},
}

PROVIDES = [
	"integration_flow_management", "message_routing", "data_transformation",
	"connector_orchestration", "dead_letter_management", "flow_monitoring",
	"error_handling", "retry_management",
]
REQUIRES = ["auth", "audl", "ntfy", "common_nats", "common_temporal"]
PUBLISHES = ["flow.started", "flow.completed", "flow.failed", "message.dead_lettered"]
SUBSCRIBES = []

UI_ROUTES = [
	{"name": "flows", "path": "/int-esb/flows", "component": "EsbFlowDesigner", "permission": "int_esb:design", "nav_group": "Flows"},
	{"name": "monitoring", "path": "/int-esb/monitoring", "component": "EsbMonitoring", "permission": "int_esb:view", "nav_group": "Monitoring"},
	{"name": "dead_letters", "path": "/int-esb/dead-letters", "component": "EsbDeadLetters", "permission": "int_esb:view", "nav_group": "Operations"},
	{"name": "settings", "path": "/int-esb/settings", "component": "EsbSettings", "permission": "int_esb:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "int_esb_theme",
	"tokens": {
		"color.primary": "#0F172A", "color.accent": "#38BDF8",
		"surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF",
		"text.primary": "#111827", "border.radius": "8px", "density": "compact",
	},
}


def get_capability_contract() -> dict[str, Any]:
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
