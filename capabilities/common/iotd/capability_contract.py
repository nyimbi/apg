"""Executable capability contract for APG IoT Device Integration."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any


SUPPORTED_IOTD_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_IOTD_AGENT_ROLES = [
	"fleet_operator",
	"telemetry_reviewer",
	"command_reviewer",
	"firmware_reviewer",
	"security_reviewer",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"devices": {
		"device_identity_required": True,
		"owner_required": True,
		"certificate_required": True,
		"certificate_rotation_days": 90,
		"fleet_grouping_enabled": True,
		"tenant_isolation_required": True,
	},
	"telemetry": {
		"event_stream": "bytewax",
		"event_stream_required": True,
		"encryption_required": True,
		"schema_validation_required": True,
		"offline_buffering_enabled": True,
		"required_fields": ["timestamp"],
	},
	"commands": {
		"command_approval_required": True,
		"dangerous_command_review_required": True,
		"command_audit_required": True,
		"command_name_required": True,
		"ack_timeout_seconds": 30,
	},
	"firmware": {
		"signature_required": True,
		"artifact_uri_required": True,
		"rollout_device_validation_required": True,
		"rollback_supported": True,
	},
	"iotd_agents": {
		"agent_assist_enabled": True,
		"agent_registration_required": True,
		"agent_runtime_required": True,
		"agent_role_required": True,
		"agent_scope_required": True,
		"agent_contribution_disclosure_required": True,
		"supported_runtimes": SUPPORTED_IOTD_AGENT_RUNTIMES,
		"allowed_roles": SUPPORTED_IOTD_AGENT_ROLES,
	},
	"governance": {
		"require_tenant_context": True,
		"firmware_signature_required": True,
		"stale_device_review_days": 30,
		"device_rbac_required": True,
		"state_change_audit_required": True,
		"batch_event_stream": "bytewax",
	},
	"observability": {
		"audit_required": True,
		"device_health_metrics_required": True,
		"telemetry_metrics_required": True,
		"command_metrics_required": True,
		"agent_activity_required": True,
		"event_stream": "bytewax",
	},
	"adapters": {
		"generated_app_runtime": "service.IotdService",
		"api_helpers": "api.py",
		"view_models": "views.py",
		"event_stream": "bytewax",
		"identity": "auth",
		"encryption": "encr",
		"audit_sink": "audl",
		"configuration": "conf",
		"edge_runtime": "edge",
		"digital_twin": "dtwn",
		"logs": "logt",
		"monitoring": "moni",
	},
	"ui": {
		"enable_device_console": True,
		"enable_telemetry_monitor": True,
		"enable_command_center": True,
		"enable_firmware_manager": True,
		"enable_agent_panel": True,
		"enable_audit": True,
		"enable_health": True,
	},
	"theme": {
		"default_theme": "iotd_device_ops",
		"allow_tenant_overrides": True,
	},
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"devices",
		"telemetry",
		"commands",
		"firmware",
		"iotd_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {
		key: {"type": "object"}
		for key in [
			"devices",
			"telemetry",
			"commands",
			"firmware",
			"iotd_agents",
			"governance",
			"observability",
			"adapters",
			"ui",
			"theme",
		]
	}
	| {"tenant_id": {"type": "string", "minLength": 1}},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All IoT operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "device_requires_identity", "description": "Devices require provisioned identity.", "condition": {"operation": "register_device", "device_identity_present": False}, "effect": {"decision": "deny", "reason": "device_identity_required", "required_action": "provision_device_identity"}},
	{"name": "device_requires_owner", "description": "Devices require an accountable owner.", "condition": {"operation": "register_device", "device_owner_present": False}, "effect": {"decision": "deny", "reason": "device_owner_required", "required_action": "assign_device_owner"}},
	{"name": "device_requires_certificate", "description": "Devices require certificate identity.", "condition": {"operation": "register_device", "certificate_present": False}, "effect": {"decision": "deny", "reason": "device_certificate_required", "required_action": "attach_device_certificate"}},
	{"name": "telemetry_requires_bytewax_stream", "description": "Telemetry ingestion requires Bytewax event streams.", "condition": {"operation": "ingest_telemetry", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
	{"name": "telemetry_requires_encryption", "description": "Telemetry ingestion requires encryption.", "condition": {"operation": "ingest_telemetry", "encryption_applied": False}, "effect": {"decision": "deny", "reason": "telemetry_encryption_required", "required_action": "encrypt_telemetry"}},
	{"name": "telemetry_requires_schema", "description": "Telemetry ingestion requires schema validation.", "condition": {"operation": "ingest_telemetry", "schema_valid": False}, "effect": {"decision": "deny", "reason": "telemetry_schema_invalid", "required_action": "fix_telemetry_schema"}},
	{"name": "command_requires_name", "description": "Commands require a command name.", "condition": {"operation": "dispatch_command", "command_name_present": False}, "effect": {"decision": "deny", "reason": "command_name_required", "required_action": "set_command_name"}},
	{"name": "dangerous_command_requires_approval", "description": "Dangerous device commands require approval.", "condition": {"dangerous_command": True, "approval_recorded": False}, "effect": {"decision": "deny", "reason": "command_approval_required", "required_action": "record_command_approval"}},
	{"name": "firmware_requires_signature", "description": "Firmware updates require signed artifacts.", "condition": {"operation": "register_firmware", "firmware_signature_verified": False}, "effect": {"decision": "deny", "reason": "firmware_signature_required", "required_action": "verify_firmware_signature"}},
	{"name": "firmware_requires_artifact", "description": "Firmware registration requires an artifact URI.", "condition": {"operation": "register_firmware", "artifact_uri_present": False}, "effect": {"decision": "deny", "reason": "firmware_artifact_required", "required_action": "attach_firmware_artifact"}},
	{"name": "deployment_requires_devices", "description": "Firmware deployment requires target devices.", "condition": {"operation": "deploy_firmware", "deployment_device_count_lte": 0}, "effect": {"decision": "deny", "reason": "deployment_devices_required", "required_action": "select_deployment_devices"}},
	{"name": "stale_device_requires_review", "description": "Stale devices require review.", "condition": {"last_seen_days_gt": 30, "stale_device_reviewed": False}, "effect": {"decision": "require_review", "reason": "stale_device_review_required", "required_action": "review_device_status"}},
	{"name": "iotd_agent_requires_registration", "description": "AI IoT agents must be registered.", "condition": {"iotd_agent_present": True, "agent_registered": False}, "effect": {"decision": "deny", "reason": "iotd_agent_registration_required", "required_action": "register_iotd_agent"}},
	{"name": "iotd_agent_runtime_supported", "description": "AI IoT agents must use a supported runtime.", "condition": {"iotd_agent_present": True, "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "iotd_agent_runtime_not_supported", "required_action": "choose_supported_iotd_agent_runtime"}},
	{"name": "iotd_agent_role_supported", "description": "AI IoT agents must use a supported role.", "condition": {"iotd_agent_present": True, "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "iotd_agent_role_not_supported", "required_action": "choose_supported_iotd_agent_role"}},
	{"name": "iotd_agent_requires_scope", "description": "AI IoT agents require explicit scope.", "condition": {"iotd_agent_present": True, "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "iotd_agent_scope_required", "required_action": "set_iotd_agent_scope"}},
	{"name": "iotd_agent_requires_disclosure", "description": "AI IoT-agent contributions require disclosure.", "condition": {"iotd_agent_present": True, "agent_contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "iotd_agent_disclosure_required", "required_action": "disclose_iotd_agent"}},
	{"name": "iotd_state_change_requires_audit", "description": "IoT lifecycle state changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "iotd_audit_event_required", "required_action": "record_iotd_audit_event"}},
	{"name": "batch_iot_mutation_requires_bytewax", "description": "Batch IoT mutations must use Bytewax event streams.", "condition": {"requested_operation": "batch_iot_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/iotd/dashboard", "component": "IOTDDashboard", "permission": "iotd:view", "nav_group": "Overview"},
	{"name": "devices", "path": "/iotd/devices", "component": "DeviceRegistry", "permission": "iotd:register", "nav_group": "Devices"},
	{"name": "telemetry", "path": "/iotd/telemetry", "component": "TelemetryMonitor", "permission": "iotd:view", "nav_group": "Telemetry"},
	{"name": "commands", "path": "/iotd/commands", "component": "CommandCenter", "permission": "iotd:command", "nav_group": "Control"},
	{"name": "firmware", "path": "/iotd/firmware", "component": "FirmwareManager", "permission": "iotd:manage_firmware", "nav_group": "Lifecycle"},
	{"name": "agents", "path": "/iotd/agents", "component": "IOTDAgentPanel", "permission": "iotd:admin", "nav_group": "Operations"},
	{"name": "health", "path": "/iotd/health", "component": "DeviceHealth", "permission": "iotd:view", "nav_group": "Operations"},
	{"name": "security", "path": "/iotd/security", "component": "DeviceSecurity", "permission": "iotd:admin", "nav_group": "Security"},
	{"name": "rules", "path": "/iotd/rules", "component": "DeviceRules", "permission": "iotd:admin", "nav_group": "Governance"},
	{"name": "audit", "path": "/iotd/audit", "component": "DeviceAuditTrail", "permission": "iotd:admin", "nav_group": "Governance"},
	{"name": "settings", "path": "/iotd/settings", "component": "IOTDSettings", "permission": "iotd:admin", "nav_group": "Administration"},
]

THEME: dict[str, Any] = {
	"name": "iotd_device_ops",
	"tokens": {
		"color.primary": "#22543D",
		"color.accent": "#2C5282",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F7F8FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact",
	},
	"components": {
		"device_card": {"icon": "radio", "status_indicator": "fleet-pill", "risk_style": "connectivity-band"},
		"telemetry_stream": {"visual": "signal-table", "highlight": "schema-chip"},
		"command_center": {"visual": "approval-console", "status_style": "ack-chip"},
		"firmware_manager": {"visual": "rollout-lanes", "status_style": "signature-chip"},
		"agent_panel": {"visual": "agent-roster", "status_style": "scope-chip"},
		"health_dashboard": {"visual": "health-grid", "status_style": "freshness-chip"},
		"audit_timeline": {"visual": "event-timeline", "status_style": "device-chip"},
	},
}


def streaming_manifest() -> dict[str, Any]:
	return {
		"processor": "bytewax",
		"topic": "apg.iotd.lifecycle",
		"state": ["devices", "telemetry", "commands", "firmware", "deployments", "health_reports", "iotd_agents", "audit_events"],
		"events": [
			"iotd_device_registered",
			"iotd_telemetry_ingested",
			"iotd_command_dispatched",
			"iotd_command_acknowledged",
			"iotd_firmware_registered",
			"iotd_firmware_deployed",
			"iotd_agent_registered",
		],
		"batch_mutation_guardrail": "batch_iot_mutation_requires_bytewax",
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "iotd",
		"display_name": "IoT Device Integration",
		"version": "1.0.0",
		"provides": [
			"device_registry",
			"telemetry_ingestion",
			"command_dispatch",
			"firmware_lifecycle",
			"device_security",
			"device_health",
			"iotd_agents",
		],
		"requires": ["auth", "encr", "audl", "conf"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/iotd/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True,
		},
		"theme": deepcopy(THEME),
		"streaming": streaming_manifest(),
	}


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
		if key.endswith("_lte"):
			actual = context.get(key[:-4])
			if not isinstance(actual, Number) or not actual <= expected:
				return False
		elif key.endswith("_lt"):
			actual = context.get(key[:-3])
			if not isinstance(actual, Number) or not actual < expected:
				return False
		elif key.endswith("_gt"):
			actual = context.get(key[:-3])
			if not isinstance(actual, Number) or not actual > expected:
				return False
		elif key.endswith("_gte"):
			actual = context.get(key[:-4])
			if not isinstance(actual, Number) or not actual >= expected:
				return False
		elif key.endswith("_ne"):
			actual = context.get(key[:-3])
			if actual is None or actual == expected:
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
