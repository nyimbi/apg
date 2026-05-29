"""UI metadata helpers for the APG IoT Device Integration capability."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import IotdService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(service: IotdService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or IotdService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"routes": capability_routes(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}


def device_console_model(service: IotdService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"route": "/iotd/devices",
		"devices": service.list_devices(tenant_id),
		"stale_devices": service.stale_device_queue(tenant_id),
		"actions": ["register_device", "quarantine_device", "retire_device"],
	}


def telemetry_monitor_model(service: IotdService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"route": "/iotd/telemetry",
		"events": service.list_telemetry(tenant_id),
		"columns": ["device_id", "schema_name", "encrypted", "event_bus", "received_at"],
	}


def command_center_model(service: IotdService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"route": "/iotd/commands",
		"commands": service.list_commands(tenant_id),
		"actions": ["dispatch_command", "acknowledge_command"],
	}


def firmware_manager_model(service: IotdService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"route": "/iotd/firmware",
		"firmware": service.list_firmware(tenant_id),
		"deployments": service.list_deployments(tenant_id),
		"actions": ["register_firmware", "deploy_firmware"],
	}


def security_model(service: IotdService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"route": "/iotd/security",
		"devices": service.list_devices(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
		"guardrails": [rule["name"] for rule in service.describe(tenant_id)["rule_engine"]["rules"]],
	}


def rules_model(service: IotdService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"route": "/iotd/rules",
		"rules": service.describe(tenant_id)["rule_engine"]["rules"],
		"health_reports": service.list_health_reports(tenant_id),
	}
