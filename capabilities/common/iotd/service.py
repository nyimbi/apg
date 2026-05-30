"""Executable service layer for APG IoT Device Integration."""

from __future__ import annotations

from itertools import count
from typing import Any

from .capability_contract import (
	DEFAULT_CONFIGURATION,
	SUPPORTED_IOTD_AGENT_ROLES,
	SUPPORTED_IOTD_AGENT_RUNTIMES,
	evaluate_capability_rules,
	get_capability_contract,
)
from .device_runtime import DeviceFreshnessInspector, DeviceHealthInspector, TelemetrySchemaValidator, iso_days_ago
from .models import (
	CommandStatus,
	DeviceAuditEvent,
	DeviceCommand,
	DeviceHealthReport,
	DeviceIdentity,
	DeviceStatus,
	FirmwareArtifact,
	FirmwareDeployment,
	FirmwareStatus,
	IotdAgent,
	TelemetryEvent,
	utc_now_iso,
)


class IotdService:
	"""Tenant-aware IoT device, telemetry, command, firmware, and health runtime."""

	def __init__(self) -> None:
		self._devices: dict[str, DeviceIdentity] = {}
		self._telemetry: dict[str, TelemetryEvent] = {}
		self._commands: dict[str, DeviceCommand] = {}
		self._firmware: dict[str, FirmwareArtifact] = {}
		self._deployments: dict[str, FirmwareDeployment] = {}
		self._audit_events: dict[str, DeviceAuditEvent] = {}
		self._health_reports: dict[str, DeviceHealthReport] = {}
		self._agents: dict[str, IotdAgent] = {}
		self._counter = count(1)
		self._schema_validator = TelemetrySchemaValidator()
		self._freshness = DeviceFreshnessInspector()
		self._health = DeviceHealthInspector()

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_device(
		self,
		device_id: str,
		tenant_id: str,
		device_key: str,
		owner_id: str,
		certificate_id: str,
		fleet_id: str = "default",
		status: str = DeviceStatus.PROVISIONED.value,
		last_seen_days: int | float = 0,
		stale_device_reviewed: bool = True,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._enforce_iot_policy(
			tenant_id=tenant_id,
			operation="register_device",
			device_identity_present=bool(device_key),
			device_owner_present=bool(owner_id),
			certificate_present=bool(certificate_id),
			last_seen_days=float(last_seen_days),
			stale_device_reviewed=stale_device_reviewed,
		)
		device = DeviceIdentity(
			id=device_id,
			tenant_id=tenant_id,
			device_key=device_key,
			owner_id=owner_id,
			certificate_id=certificate_id,
			fleet_id=fleet_id,
			status=DeviceStatus(status),
			last_seen_at=iso_days_ago(last_seen_days),
			metadata=dict(metadata or {}),
		)
		self._devices[_state_key(tenant_id, device_id)] = device
		self._audit(tenant_id, "device_registered", device_id=device_id, reason=fleet_id)
		return device.to_dict()

	def ingest_telemetry(
		self,
		event_id: str,
		tenant_id: str,
		device_id: str,
		schema_name: str,
		payload: dict[str, Any],
		encryption_applied: bool = True,
		event_bus: str = "bytewax",
		required_fields: list[str] | None = None,
	) -> dict[str, Any]:
		device = self._require_device(device_id, tenant_id)
		schema = self._schema_validator.validate(payload, required_fields)
		self._enforce_iot_policy(
			tenant_id=tenant_id,
			operation="ingest_telemetry",
			event_stream=event_bus,
			encryption_applied=encryption_applied,
			schema_valid=bool(schema["valid"]),
		)
		event = TelemetryEvent(
			id=event_id,
			tenant_id=tenant_id,
			device_id=device_id,
			schema_name=schema_name,
			payload=dict(payload),
			encrypted=encryption_applied,
			event_bus=event_bus,
		)
		device.status = DeviceStatus.ONLINE
		device.last_seen_at = utc_now_iso()
		self._telemetry[_state_key(tenant_id, event_id)] = event
		self._audit(tenant_id, "telemetry_ingested", device_id=device_id, reason=schema_name)
		return event.to_dict()

	def dispatch_command(
		self,
		command_id: str,
		tenant_id: str,
		device_id: str,
		command: str,
		parameters: dict[str, Any] | None = None,
		dangerous: bool = False,
		approval_id: str | None = None,
		approval_recorded: bool | None = None,
	) -> dict[str, Any]:
		self._require_device(device_id, tenant_id)
		approved = bool(approval_id) if approval_recorded is None else approval_recorded
		self._enforce_iot_policy(
			tenant_id=tenant_id,
			operation="dispatch_command",
			command_name_present=bool(command),
			dangerous_command=dangerous,
			approval_recorded=approved,
		)
		device_command = DeviceCommand(
			id=command_id,
			tenant_id=tenant_id,
			device_id=device_id,
			command=command,
			parameters=dict(parameters or {}),
			dangerous=dangerous,
			approval_id=approval_id,
			status=CommandStatus.DISPATCHED,
		)
		self._commands[_state_key(tenant_id, command_id)] = device_command
		self._audit(tenant_id, "command_dispatched", device_id=device_id, reason=command)
		return device_command.to_dict()

	def acknowledge_command(
		self,
		command_id: str,
		tenant_id: str,
		ack_message: str = "acknowledged",
	) -> dict[str, Any]:
		command = self._require_command(command_id, tenant_id)
		command.status = CommandStatus.ACKNOWLEDGED
		command.acknowledged_at = utc_now_iso()
		command.ack_message = ack_message
		self._audit(tenant_id, "command_acknowledged", device_id=command.device_id, reason=command_id)
		return command.to_dict()

	def register_firmware(
		self,
		firmware_id: str,
		tenant_id: str,
		version: str,
		artifact_uri: str,
		signature_id: str,
		firmware_signature_verified: bool = True,
	) -> dict[str, Any]:
		self._enforce_iot_policy(
			tenant_id=tenant_id,
			operation="register_firmware",
			firmware_signature_verified=firmware_signature_verified,
			artifact_uri_present=bool(artifact_uri),
		)
		artifact = FirmwareArtifact(
			id=firmware_id,
			tenant_id=tenant_id,
			version=version,
			artifact_uri=artifact_uri,
			signature_id=signature_id,
			signature_verified=firmware_signature_verified,
		)
		self._firmware[_state_key(tenant_id, firmware_id)] = artifact
		self._audit(tenant_id, "firmware_registered", reason=version)
		return artifact.to_dict()

	def deploy_firmware(
		self,
		deployment_id: str,
		tenant_id: str,
		firmware_id: str,
		fleet_id: str,
		device_ids: list[str],
	) -> dict[str, Any]:
		artifact = self._require_firmware(firmware_id, tenant_id)
		if not artifact.signature_verified:
			raise PermissionError("firmware_signature_required")
		self._enforce_iot_policy(
			tenant_id=tenant_id,
			operation="deploy_firmware",
			deployment_device_count=len(device_ids),
		)
		for device_id in device_ids:
			self._require_device(device_id, tenant_id)
		artifact.status = FirmwareStatus.DEPLOYED
		deployment = FirmwareDeployment(
			id=deployment_id,
			tenant_id=tenant_id,
			firmware_id=firmware_id,
			fleet_id=fleet_id,
			device_ids=list(device_ids),
		)
		self._deployments[_state_key(tenant_id, deployment_id)] = deployment
		self._audit(tenant_id, "firmware_deployed", reason=deployment_id)
		return deployment.to_dict()

	def health_report(self, report_id: str, tenant_id: str) -> dict[str, Any]:
		report = self._health.summarize(
			report_id=report_id,
			tenant_id=tenant_id,
			devices=list(self._devices.values()),
			commands=list(self._commands.values()),
			firmware=list(self._firmware.values()),
			stale_device_review_days=int(DEFAULT_CONFIGURATION["governance"]["stale_device_review_days"]),
		)
		self._health_reports[_state_key(tenant_id, report_id)] = report
		self._audit(tenant_id, "health_report_generated", reason=report_id)
		return report.to_dict()

	def register_iotd_agent(
		self,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		contribution_disclosed: bool = True,
		agent_id: str | None = None,
	) -> dict[str, Any]:
		normalized_runtime = _normalize_token(runtime)
		normalized_role = _normalize_token(role)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"iotd_agent_present": True,
			"agent_registered": True,
			"agent_runtime_supported": normalized_runtime in SUPPORTED_IOTD_AGENT_RUNTIMES,
			"agent_role_supported": normalized_role in SUPPORTED_IOTD_AGENT_ROLES,
			"agent_scope_present": bool(scope),
			"agent_contribution_disclosed": contribution_disclosed,
		})
		if result["decision"] == "deny":
			raise PermissionError(_reasons(result) or "iot_policy_blocked")
		agent = IotdAgent(
			id=agent_id or f"iotd-agent-{len(self._agents) + 1:06d}",
			tenant_id=tenant_id,
			name=name,
			runtime=normalized_runtime,
			role=normalized_role,
			scope=scope,
			contribution_disclosed=contribution_disclosed,
		)
		self._agents[_state_key(tenant_id, agent.id)] = agent
		self._audit(tenant_id, "iotd_agent_registered", decision=result["decision"], reason=agent.id)
		return agent.to_dict()

	def validate_batch_iot_mutation(self, event_stream: str) -> dict[str, Any]:
		return self.evaluate({
			"tenant_context_present": True,
			"requested_operation": "batch_iot_mutation",
			"event_stream": event_stream,
		})

	def stale_device_queue(self, tenant_id: str) -> list[dict[str, Any]]:
		stale = self._freshness.stale_devices(
			list(self._devices.values()),
			tenant_id,
			int(DEFAULT_CONFIGURATION["governance"]["stale_device_review_days"]),
		)
		return [device.to_dict() for device in sorted(stale, key=lambda item: item.id)]

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "provisioned",
	) -> dict[str, Any]:
		"""Compatibility helper for generated package probes."""
		data = dict(metadata or {})
		return self.register_device(
			device_id=record_id,
			tenant_id=tenant_id,
			device_key=str(data.get("device_key") or f"device-{record_id}"),
			owner_id=str(data.get("owner_id") or "system"),
			certificate_id=str(data.get("certificate_id") or "cert-default"),
			fleet_id=str(data.get("fleet_id") or "default"),
			status=status,
			metadata=data,
		)

	def list_devices(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._devices, tenant_id)

	def list_telemetry(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._telemetry, tenant_id)

	def list_commands(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._commands, tenant_id)

	def list_firmware(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._firmware, tenant_id)

	def list_deployments(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._deployments, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._audit_events, tenant_id)

	def list_health_reports(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._health_reports, tenant_id)

	def list_iotd_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._agents, tenant_id)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_devices(tenant_id)

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		stale = self.stale_device_queue(tenant_id)
		return {
			"tenant_id": tenant_id,
			"device_count": len(self.list_devices(tenant_id)),
			"telemetry_event_count": len(self.list_telemetry(tenant_id)),
			"command_count": len(self.list_commands(tenant_id)),
			"firmware_count": len(self.list_firmware(tenant_id)),
			"stale_device_count": len(stale),
			"iotd_agent_count": len(self.list_iotd_agents(tenant_id)),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
			"routes": len(self.describe(tenant_id)["ui"]["routes"]),
			"theme": self.describe(tenant_id)["theme"]["name"],
			"streaming": self.describe(tenant_id)["streaming"],
		}

	def _enforce_iot_policy(self, tenant_id: str, **context: Any) -> dict[str, Any]:
		result = self.evaluate({"tenant_context_present": bool(tenant_id), **context})
		if result["decision"] != "allow":
			raise PermissionError(_reasons(result) or "iot_policy_blocked")
		return result

	def _require_device(self, device_id: str, tenant_id: str) -> DeviceIdentity:
		device = self._devices.get(_state_key(tenant_id, device_id))
		if device is None or device.tenant_id != tenant_id:
			raise KeyError("device_missing")
		return device

	def _require_command(self, command_id: str, tenant_id: str) -> DeviceCommand:
		command = self._commands.get(_state_key(tenant_id, command_id))
		if command is None or command.tenant_id != tenant_id:
			raise KeyError("command_missing")
		return command

	def _require_firmware(self, firmware_id: str, tenant_id: str) -> FirmwareArtifact:
		artifact = self._firmware.get(_state_key(tenant_id, firmware_id))
		if artifact is None or artifact.tenant_id != tenant_id:
			raise KeyError("firmware_missing")
		return artifact

	def _audit(self, tenant_id: str, event_type: str, device_id: str | None = None, decision: str = "allow", reason: str = "") -> None:
		if not DEFAULT_CONFIGURATION["commands"]["command_audit_required"]:
			return
		event_id = f"audit-{next(self._counter)}"
		self._audit_events[event_id] = DeviceAuditEvent(
			id=event_id,
			tenant_id=tenant_id,
			event_type=event_type,
			device_id=device_id,
			decision=decision,
			reason=reason,
		)

	def _list(self, records: dict[str, Any], tenant_id: str | None = None) -> list[dict[str, Any]]:
		values = list(records.values())
		if tenant_id is not None:
			values = [record for record in values if record.tenant_id == tenant_id]
		return [record.to_dict() for record in sorted(values, key=lambda item: item.id)]


def _normalize_token(value: str) -> str:
	return value.strip().lower().replace("-", "_").replace(" ", "_")


def _state_key(tenant_id: str, item_id: str) -> str:
	return f"{tenant_id}:{item_id}"


def _reasons(result: dict[str, Any]) -> str:
	return ", ".join(action.get("reason", "iot_policy_blocked") for action in result["actions"])
