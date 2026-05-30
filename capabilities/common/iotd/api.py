"""API helpers for the APG IoT Device Integration capability."""

from __future__ import annotations

from typing import Any

from .service import IotdService


SERVICE = IotdService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"device_count": summary["device_count"],
		"telemetry_event_count": summary["telemetry_event_count"],
		"command_count": summary["command_count"],
	}


def register_device(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_device(
		device_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		device_key=str(payload.get("device_key") or ""),
		owner_id=str(payload.get("owner_id") or ""),
		certificate_id=str(payload.get("certificate_id") or ""),
		fleet_id=str(payload.get("fleet_id") or "default"),
		status=str(payload.get("status") or "provisioned"),
		last_seen_days=float(payload.get("last_seen_days") or 0),
		stale_device_reviewed=bool(payload.get("stale_device_reviewed", True)),
		metadata=dict(payload.get("metadata") or {}),
	)


def ingest_telemetry(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.ingest_telemetry(
		event_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		device_id=str(payload["device_id"]),
		schema_name=str(payload.get("schema_name") or "default"),
		payload=dict(payload.get("payload") or {}),
		encryption_applied=bool(payload.get("encryption_applied", True)),
		event_bus=str(payload.get("event_bus") or "bytewax"),
		required_fields=list(payload.get("required_fields") or ["timestamp"]),
	)


def dispatch_command(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.dispatch_command(
		command_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		device_id=str(payload["device_id"]),
		command=str(payload["command"]),
		parameters=dict(payload.get("parameters") or {}),
		dangerous=bool(payload.get("dangerous", False)),
		approval_id=payload.get("approval_id"),
		approval_recorded=payload.get("approval_recorded"),
	)


def acknowledge_command(command_id: str, tenant_id: str = "default", ack_message: str = "acknowledged") -> dict[str, Any]:
	return SERVICE.acknowledge_command(command_id, tenant_id, ack_message)


def register_firmware(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_firmware(
		firmware_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		version=str(payload["version"]),
		artifact_uri=str(payload["artifact_uri"]),
		signature_id=str(payload.get("signature_id") or ""),
		firmware_signature_verified=bool(payload.get("firmware_signature_verified", True)),
	)


def deploy_firmware(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.deploy_firmware(
		deployment_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		firmware_id=str(payload["firmware_id"]),
		fleet_id=str(payload.get("fleet_id") or "default"),
		device_ids=list(payload.get("device_ids") or []),
	)


def health_report(report_id: str, tenant_id: str = "default") -> dict[str, Any]:
	return SERVICE.health_report(report_id, tenant_id)


def register_iotd_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_iotd_agent(
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		runtime=str(payload["runtime"]),
		role=str(payload["role"]),
		scope=str(payload["scope"]),
		contribution_disclosed=bool(payload.get("contribution_disclosed", True)),
		agent_id=payload.get("id"),
	)


def validate_batch_iot_mutation(event_stream: str) -> dict[str, Any]:
	return SERVICE.validate_batch_iot_mutation(event_stream)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_record(
		record_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		metadata=dict(payload.get("metadata") or {}),
		status=str(payload.get("status") or "provisioned"),
	)


def list_records(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_records(tenant_id)


def list_iotd_agents(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_iotd_agents(tenant_id)


def list_audit_events(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_audit_events(tenant_id)


def dashboard_summary(tenant_id: str = "default") -> dict[str, Any]:
	return SERVICE.dashboard_summary(tenant_id)
