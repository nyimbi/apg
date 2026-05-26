"""Executable capability contract for APG IoT Device Integration."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"devices": {"device_identity_required": True, "owner_required": True, "certificate_rotation_days": 90, "fleet_grouping_enabled": True},
	"telemetry": {"event_bus_required": True, "encryption_required": True, "schema_validation_required": True, "offline_buffering_enabled": True},
	"commands": {"command_approval_required": True, "dangerous_command_review_required": True, "command_audit_required": True, "ack_timeout_seconds": 30},
	"governance": {"require_tenant_context": True, "firmware_signature_required": True, "stale_device_review_days": 30, "device_rbac_required": True},
	"ui": {"enable_device_console": True, "enable_telemetry_monitor": True, "enable_command_center": True, "enable_firmware_manager": True},
	"theme": {"default_theme": "iotd_device_ops", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "devices", "telemetry", "commands", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["devices", "telemetry", "commands", "governance", "ui", "theme"]} | {"tenant_id": {"type": "string", "minLength": 1}}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All IoT operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "device_requires_identity", "description": "Devices require provisioned identity.", "condition": {"operation": "register_device", "device_identity_present": False}, "effect": {"decision": "deny", "reason": "device_identity_required", "required_action": "provision_device_identity"}},
	{"name": "telemetry_requires_encryption", "description": "Telemetry ingestion requires encryption.", "condition": {"operation": "ingest_telemetry", "encryption_applied": False}, "effect": {"decision": "deny", "reason": "telemetry_encryption_required", "required_action": "encrypt_telemetry"}},
	{"name": "dangerous_command_requires_approval", "description": "Dangerous device commands require approval.", "condition": {"dangerous_command": True, "approval_recorded": False}, "effect": {"decision": "deny", "reason": "command_approval_required", "required_action": "record_command_approval"}},
	{"name": "firmware_requires_signature", "description": "Firmware updates require signed artifacts.", "condition": {"operation": "deploy_firmware", "firmware_signature_verified": False}, "effect": {"decision": "deny", "reason": "firmware_signature_required", "required_action": "verify_firmware_signature"}},
	{"name": "stale_device_requires_review", "description": "Stale devices require review.", "condition": {"last_seen_days_gt": 30, "stale_device_reviewed": False}, "effect": {"decision": "require_review", "reason": "stale_device_review_required", "required_action": "review_device_status"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/iotd/dashboard", "component": "IOTDDashboard", "permission": "iotd:view", "nav_group": "Overview"},
	{"name": "devices", "path": "/iotd/devices", "component": "DeviceRegistry", "permission": "iotd:register", "nav_group": "Devices"},
	{"name": "telemetry", "path": "/iotd/telemetry", "component": "TelemetryMonitor", "permission": "iotd:view", "nav_group": "Telemetry"},
	{"name": "commands", "path": "/iotd/commands", "component": "CommandCenter", "permission": "iotd:command", "nav_group": "Control"},
	{"name": "firmware", "path": "/iotd/firmware", "component": "FirmwareManager", "permission": "iotd:manage_firmware", "nav_group": "Lifecycle"},
	{"name": "security", "path": "/iotd/security", "component": "DeviceSecurity", "permission": "iotd:admin", "nav_group": "Security"},
	{"name": "rules", "path": "/iotd/rules", "component": "DeviceRules", "permission": "iotd:admin", "nav_group": "Governance"},
	{"name": "settings", "path": "/iotd/settings", "component": "IOTDSettings", "permission": "iotd:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "iotd_device_ops",
	"tokens": {"color.primary": "#22543D", "color.accent": "#2C5282", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {"device_card": {"icon": "radio", "status_indicator": "fleet-pill", "risk_style": "connectivity-band"}, "telemetry_stream": {"visual": "signal-table", "highlight": "schema-chip"}, "command_center": {"visual": "approval-console", "status_style": "ack-chip"}, "firmware_manager": {"visual": "rollout-lanes", "status_style": "signature-chip"}}
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "iotd", "display_name": "IoT Device Integration", "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "flask_appbuilder", "view_module": "views.py", "api_prefix": "/iotd/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME)}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	matched: list[str] = []
	actions: list[dict[str, Any]] = []
	decision = "allow"
	for rule in RULES:
		if _matches(rule["condition"], context):
			matched.append(rule["name"])
			effect = rule["effect"]
			actions.append(effect)
			if effect["decision"] == "deny":
				decision = "deny"
			elif effect["decision"] == "require_review" and decision != "deny":
				decision = "require_review"
	return {"decision": decision, "matched_rules": matched, "actions": actions, "context": context}


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_lt"):
			if not context.get(key[:-3], 0) < expected:
				return False
		elif key.endswith("_gt"):
			if not context.get(key[:-3], 0) > expected:
				return False
		elif context.get(key) != expected:
			return False
	return True


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
	for key, value in source.items():
		if isinstance(value, dict) and isinstance(target.get(key), dict):
			_deep_merge(target[key], value)
		else:
			target[key] = value
