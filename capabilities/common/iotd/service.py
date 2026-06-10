"""Executable service layer for APG IoT Device Integration."""

from __future__ import annotations

import statistics
from datetime import datetime, timezone
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


from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
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
		self._thresholds: dict[str, dict[str, Any]] = {}
		self._heartbeats: dict[str, list[dict[str, Any]]] = {}
		self._commissioning: dict[str, dict[str, Any]] = {}
		self._firmware_schedules: dict[str, dict[str, Any]] = {}
		self._counter = count(1)
		self._schema_validator = TelemetrySchemaValidator()
		self._freshness = DeviceFreshnessInspector()
		self._health = DeviceHealthInspector()

	# ------------------------------------------------------------------ existing

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
		# check thresholds for all numeric readings
		for metric, value in payload.items():
			if isinstance(value, (int, float)):
				threshold_key = _state_key(tenant_id, f"{device_id}:{metric}")
				if threshold_key in self._thresholds:
					self._evaluate_threshold(tenant_id, device_id, metric, float(value), self._thresholds[threshold_key])
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

	# ------------------------------------------------------------------ new methods

	def device_heartbeat(
		self,
		device_id: str,
		tenant_id: str,
		timestamp: str,
		status: str = "online",
		signal_strength: float | None = None,
		battery_level: float | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Record a heartbeat from a device, updating last-seen and health indicators."""
		device = self._require_device(device_id, tenant_id)
		assert bool(timestamp), "heartbeat timestamp required"
		device.status = DeviceStatus.ONLINE if status == "online" else DeviceStatus(status)
		device.last_seen_at = timestamp
		heartbeat = {
			"device_id": device_id,
			"tenant_id": tenant_id,
			"timestamp": timestamp,
			"status": status,
			"signal_strength": signal_strength,
			"battery_level": battery_level,
			"metadata": dict(metadata or {}),
			"recorded_at": utc_now_iso(),
		}
		key = _state_key(tenant_id, device_id)
		self._heartbeats.setdefault(key, []).append(heartbeat)
		# alert on low battery
		if battery_level is not None and battery_level < 0.1:
			self._audit(tenant_id, "low_battery_alert", device_id=device_id, reason=f"battery={battery_level:.2%}")
		self._audit(tenant_id, "device_heartbeat", device_id=device_id, reason=timestamp)
		return heartbeat

	def set_threshold(
		self,
		device_id: str,
		tenant_id: str,
		metric: str,
		min_val: float | None,
		max_val: float | None,
		alert_level: str = "warning",
	) -> dict[str, Any]:
		"""Define an alerting threshold for a named metric on a specific device."""
		self._require_device(device_id, tenant_id)
		assert bool(metric), "metric name required"
		assert alert_level in {"info", "warning", "critical"}, f"invalid alert_level: {alert_level}"
		assert min_val is not None or max_val is not None, "at least one of min_val or max_val must be set"
		threshold = {
			"device_id": device_id,
			"tenant_id": tenant_id,
			"metric": metric,
			"min_val": min_val,
			"max_val": max_val,
			"alert_level": alert_level,
			"created_at": utc_now_iso(),
		}
		self._thresholds[_state_key(tenant_id, f"{device_id}:{metric}")] = threshold
		self._audit(tenant_id, "threshold_set", device_id=device_id, reason=f"{metric}:[{min_val},{max_val}]")
		return threshold

	def threshold_alert(
		self,
		device_id: str,
		tenant_id: str,
		metric: str,
		value: float,
		threshold: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Evaluate a metric reading against its threshold; emit an alert audit event if breached."""
		self._require_device(device_id, tenant_id)
		th = threshold or self._thresholds.get(_state_key(tenant_id, f"{device_id}:{metric}"))
		assert th is not None, f"no threshold configured for {metric} on {device_id}"
		breached, direction = self._evaluate_threshold(tenant_id, device_id, metric, value, th)
		alert = {
			"device_id": device_id,
			"tenant_id": tenant_id,
			"metric": metric,
			"value": value,
			"threshold": th,
			"breached": breached,
			"direction": direction,
			"alert_level": th.get("alert_level", "warning") if breached else "none",
			"evaluated_at": utc_now_iso(),
		}
		return alert

	def device_firmware_update(
		self,
		device_id: str,
		tenant_id: str,
		version: str,
		schedule: str,
		approved_by: str | None = None,
	) -> dict[str, Any]:
		"""Schedule a firmware update for a specific device at a given time."""
		device = self._require_device(device_id, tenant_id)
		assert bool(version), "firmware version required"
		assert bool(schedule), "schedule timestamp required"
		schedule_record = {
			"device_id": device_id,
			"tenant_id": tenant_id,
			"version": version,
			"schedule": schedule,
			"approved_by": approved_by,
			"status": "scheduled",
			"created_at": utc_now_iso(),
		}
		sched_key = _state_key(tenant_id, f"{device_id}:fw:{version}")
		self._firmware_schedules[sched_key] = schedule_record
		device.metadata["pending_firmware"] = version
		self._audit(tenant_id, "firmware_update_scheduled", device_id=device_id, reason=f"v{version}@{schedule}")
		return schedule_record

	def device_commissioning(
		self,
		device_id: str,
		tenant_id: str,
		installation_site: str,
		commissioned_by: str,
		notes: str = "",
		geolocation: dict[str, float] | None = None,
	) -> dict[str, Any]:
		"""Mark a device as fully commissioned at an installation site."""
		device = self._require_device(device_id, tenant_id)
		assert bool(installation_site), "installation_site required"
		assert bool(commissioned_by), "commissioned_by required"
		record = {
			"device_id": device_id,
			"tenant_id": tenant_id,
			"installation_site": installation_site,
			"commissioned_by": commissioned_by,
			"notes": notes,
			"geolocation": dict(geolocation or {}),
			"commissioned_at": utc_now_iso(),
			"status": "commissioned",
		}
		self._commissioning[_state_key(tenant_id, device_id)] = record
		device.status = DeviceStatus.ONLINE
		device.metadata["installation_site"] = installation_site
		self._audit(tenant_id, "device_commissioned", device_id=device_id, reason=installation_site)
		return record

	def bulk_telemetry_ingest(
		self,
		tenant_id: str,
		device_readings: list[dict[str, Any]],
		event_bus: str = "bytewax",
		encryption_applied: bool = True,
	) -> dict[str, Any]:
		"""Ingest telemetry for multiple devices in a single batch call.

		Each entry in device_readings must contain: event_id, device_id,
		schema_name, payload. Returns a summary with per-device outcome.
		"""
		assert bool(device_readings), "device_readings must be non-empty"
		results: list[dict[str, Any]] = []
		failed: list[dict[str, Any]] = []
		for reading in device_readings:
			event_id = str(reading.get("event_id") or "")
			device_id = str(reading.get("device_id") or "")
			schema_name = str(reading.get("schema_name") or "default")
			payload = dict(reading.get("payload") or {})
			try:
				result = self.ingest_telemetry(
					event_id=event_id,
					tenant_id=tenant_id,
					device_id=device_id,
					schema_name=schema_name,
					payload=payload,
					encryption_applied=encryption_applied,
					event_bus=event_bus,
				)
				results.append({"event_id": event_id, "device_id": device_id, "status": "ok", "result": result})
			except Exception as exc:
				failed.append({"event_id": event_id, "device_id": device_id, "status": "error", "reason": str(exc)})
		summary = {
			"tenant_id": tenant_id,
			"total": len(device_readings),
			"succeeded": len(results),
			"failed": len(failed),
			"results": results,
			"failures": failed,
			"processor": event_bus,
			"ingested_at": utc_now_iso(),
		}
		self._audit(tenant_id, "bulk_telemetry_ingested", reason=f"{len(results)}/{len(device_readings)} ok")
		return summary

	def iot_analytics(
		self,
		tenant_id: str,
		device_group: str,
		period: str,
	) -> dict[str, Any]:
		"""Compute descriptive analytics for a fleet/group over a reporting period."""
		devices = [
			d for d in self._devices.values()
			if d.tenant_id == tenant_id and (device_group == "all" or d.fleet_id == device_group)
		]
		telemetry = [
			t for t in self._telemetry.values()
			if t.tenant_id == tenant_id and any(d.id == t.device_id for d in devices)
		]
		online_count = sum(1 for d in devices if d.status == DeviceStatus.ONLINE)
		offline_count = sum(1 for d in devices if d.status != DeviceStatus.ONLINE)
		# gather all numeric scalar payloads
		all_values: list[float] = []
		for t in telemetry:
			for v in t.payload.values():
				if isinstance(v, (int, float)):
					all_values.append(float(v))
		analytics: dict[str, Any] = {
			"tenant_id": tenant_id,
			"device_group": device_group,
			"period": period,
			"device_count": len(devices),
			"online_count": online_count,
			"offline_count": offline_count,
			"telemetry_event_count": len(telemetry),
			"avg_payload_value": round(statistics.mean(all_values), 4) if all_values else None,
			"stddev_payload_value": round(statistics.stdev(all_values), 4) if len(all_values) > 1 else None,
			"max_payload_value": max(all_values) if all_values else None,
			"min_payload_value": min(all_values) if all_values else None,
			"threshold_count": len([k for k in self._thresholds if k.startswith(f"{tenant_id}:")]),
			"computed_at": utc_now_iso(),
		}
		self._audit(tenant_id, "iot_analytics_computed", reason=f"group={device_group} period={period}")
		return analytics

	def fleet_health_dashboard(self, tenant_id: str) -> dict[str, Any]:
		"""Return an aggregated fleet-level health view for all devices in a tenant."""
		devices = [d for d in self._devices.values() if d.tenant_id == tenant_id]
		fleets: dict[str, dict[str, int]] = {}
		for d in devices:
			fleet = d.fleet_id or "default"
			fleets.setdefault(fleet, {"total": 0, "online": 0, "offline": 0, "provisioned": 0})
			fleets[fleet]["total"] += 1
			if d.status == DeviceStatus.ONLINE:
				fleets[fleet]["online"] += 1
			elif d.status == DeviceStatus.PROVISIONED:
				fleets[fleet]["provisioned"] += 1
			else:
				fleets[fleet]["offline"] += 1
		stale = self.stale_device_queue(tenant_id)
		threshold_breach_events = [
			e for e in self._audit_events.values()
			if e.tenant_id == tenant_id and "threshold" in e.event_type
		]
		commissioned = [
			r for r in self._commissioning.values()
			if r["tenant_id"] == tenant_id and r["status"] == "commissioned"
		]
		return {
			"tenant_id": tenant_id,
			"total_devices": len(devices),
			"online_devices": sum(1 for d in devices if d.status == DeviceStatus.ONLINE),
			"stale_devices": len(stale),
			"commissioned_devices": len(commissioned),
			"fleet_breakdown": fleets,
			"threshold_breach_count": len(threshold_breach_events),
			"pending_firmware_schedules": len(self._firmware_schedules),
			"active_deployments": len([dep for dep in self._deployments.values() if dep.tenant_id == tenant_id]),
			"audit_event_count": len([e for e in self._audit_events.values() if e.tenant_id == tenant_id]),
			"generated_at": utc_now_iso(),
		}

	# ------------------------------------------------------------------ list / compat

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

	def list_thresholds(self, tenant_id: str) -> list[dict[str, Any]]:
		return [t for t in self._thresholds.values() if t["tenant_id"] == tenant_id]

	def list_heartbeats(self, tenant_id: str, device_id: str) -> list[dict[str, Any]]:
		return list(self._heartbeats.get(_state_key(tenant_id, device_id), []))

	def list_commissioning(self, tenant_id: str) -> list[dict[str, Any]]:
		return [r for r in self._commissioning.values() if r["tenant_id"] == tenant_id]

	def device_command(
		self,
		command_id: str,
		tenant_id: str,
		device_id: str,
		command: str,
		parameters: dict[str, Any] | None = None,
		dangerous: bool = False,
		approval_id: str | None = None,
	) -> dict[str, Any]:
		"""Dispatch a named command to a device (thin wrapper over dispatch_command)."""
		return self.dispatch_command(
			command_id=command_id,
			tenant_id=tenant_id,
			device_id=device_id,
			command=command,
			parameters=parameters,
			dangerous=dangerous,
			approval_id=approval_id,
		)

	def firmware_ota(
		self,
		device_id: str,
		tenant_id: str,
		version: str,
		schedule: str,
		approved_by: str | None = None,
	) -> dict[str, Any]:
		"""Schedule an OTA firmware update for a device (alias for device_firmware_update)."""
		return self.device_firmware_update(
			device_id=device_id,
			tenant_id=tenant_id,
			version=version,
			schedule=schedule,
			approved_by=approved_by,
		)

	def data_subscribe(
		self,
		tenant_id: str,
		device_id: str,
		metrics: list[str],
		callback_ref: str,
		subscriber_id: str = "system",
	) -> dict[str, Any]:
		"""Register a subscription for real-time metric streams from a device."""
		self._require_device(device_id, tenant_id)
		assert bool(metrics), "metrics list required"
		assert bool(callback_ref), "callback_ref required"
		sub_id = _state_key(tenant_id, f"{device_id}:sub:{subscriber_id}")
		record = {
			"subscription_id": sub_id,
			"device_id": device_id,
			"tenant_id": tenant_id,
			"metrics": list(metrics),
			"callback_ref": callback_ref,
			"subscriber_id": subscriber_id,
			"status": "active",
			"subscribed_at": utc_now_iso(),
		}
		self._audit(tenant_id, "data_subscribed", device_id=device_id, reason=f"metrics={','.join(metrics)}")
		return record

	def alert_threshold(
		self,
		device_id: str,
		tenant_id: str,
		metric: str,
		min_val: float | None,
		max_val: float | None,
		alert_level: str = "warning",
	) -> dict[str, Any]:
		"""Set an alert threshold on a device metric (alias for set_threshold)."""
		return self.set_threshold(
			device_id=device_id,
			tenant_id=tenant_id,
			metric=metric,
			min_val=min_val,
			max_val=max_val,
			alert_level=alert_level,
		)

	def device_group(
		self,
		tenant_id: str,
		group_name: str,
		device_ids: list[str],
		owner_id: str = "system",
	) -> dict[str, Any]:
		"""Create a logical device group and validate all member devices exist."""
		assert bool(device_ids), "device_ids required"
		for did in device_ids:
			self._require_device(did, tenant_id)
		record = {
			"group_id": _state_key(tenant_id, group_name),
			"tenant_id": tenant_id,
			"group_name": group_name,
			"device_ids": list(device_ids),
			"device_count": len(device_ids),
			"owner_id": owner_id,
			"created_at": utc_now_iso(),
		}
		self._audit(tenant_id, "device_group_created", reason=f"group={group_name} n={len(device_ids)}")
		return record

	def twin_sync(
		self,
		device_id: str,
		tenant_id: str,
		desired_state: dict[str, Any],
		synced_by: str = "system",
	) -> dict[str, Any]:
		"""Sync desired state to a device digital twin and report delta."""
		device = self._require_device(device_id, tenant_id)
		current_state = device.metadata.get("twin_state", {})
		delta = {k: v for k, v in desired_state.items() if current_state.get(k) != v}
		device.metadata["twin_state"] = {**current_state, **desired_state}
		record = {
			"device_id": device_id,
			"tenant_id": tenant_id,
			"desired_state": desired_state,
			"delta_keys": list(delta.keys()),
			"delta_count": len(delta),
			"synced_by": synced_by,
			"synced_at": utc_now_iso(),
		}
		self._audit(tenant_id, "twin_synced", device_id=device_id, reason=f"delta={len(delta)}")
		return record

	def protocol_bridge(
		self,
		tenant_id: str,
		bridge_id: str,
		source_protocol: str,
		target_protocol: str,
		device_ids: list[str],
		configured_by: str = "system",
	) -> dict[str, Any]:
		"""Configure a protocol bridge translating between IoT protocols (MQTT↔HTTP, etc.)."""
		valid = {"mqtt", "http", "coap", "amqp", "websocket", "modbus", "opcua"}
		assert source_protocol in valid, f"unsupported source_protocol: {source_protocol}"
		assert target_protocol in valid, f"unsupported target_protocol: {target_protocol}"
		for did in device_ids:
			self._require_device(did, tenant_id)
		record = {
			"bridge_id": bridge_id,
			"tenant_id": tenant_id,
			"source_protocol": source_protocol,
			"target_protocol": target_protocol,
			"device_ids": list(device_ids),
			"status": "active",
			"configured_by": configured_by,
			"configured_at": utc_now_iso(),
		}
		self._audit(tenant_id, "protocol_bridge_configured", reason=f"{source_protocol}→{target_protocol}")
		return record

	def offline_buffer(
		self,
		device_id: str,
		tenant_id: str,
		max_events: int = 1000,
		flush_on_reconnect: bool = True,
		configured_by: str = "system",
	) -> dict[str, Any]:
		"""Configure offline event buffering for a device that loses connectivity."""
		self._require_device(device_id, tenant_id)
		assert max_events > 0, "max_events must be positive"
		record = {
			"device_id": device_id,
			"tenant_id": tenant_id,
			"max_events": max_events,
			"flush_on_reconnect": flush_on_reconnect,
			"current_buffer_size": 0,
			"status": "active",
			"configured_by": configured_by,
			"configured_at": utc_now_iso(),
		}
		self._audit(tenant_id, "offline_buffer_configured", device_id=device_id, reason=f"max={max_events}")
		return record

	def device_analytics(
		self,
		tenant_id: str,
		device_id: str,
		period: str = "all",
	) -> dict[str, Any]:
		"""Return per-device telemetry and command analytics."""
		device = self._require_device(device_id, tenant_id)
		telemetry = [t for t in self._telemetry.values() if t.tenant_id == tenant_id and t.device_id == device_id]
		commands = [c for c in self._commands.values() if c.tenant_id == tenant_id and c.device_id == device_id]
		heartbeats = list(self._heartbeats.get(_state_key(tenant_id, device_id), []))
		thresholds = [t for t in self._thresholds.values() if t["tenant_id"] == tenant_id and t["device_id"] == device_id]
		return {
			"device_id": device_id,
			"tenant_id": tenant_id,
			"period": period,
			"status": device.status.value,
			"fleet_id": device.fleet_id,
			"telemetry_event_count": len(telemetry),
			"command_count": len(commands),
			"heartbeat_count": len(heartbeats),
			"threshold_count": len(thresholds),
			"last_seen_at": device.last_seen_at,
			"computed_at": utc_now_iso(),
		}

	def decommission(
		self,
		device_id: str,
		tenant_id: str,
		reason: str,
		decommissioned_by: str,
	) -> dict[str, Any]:
		"""Decommission a device: mark it offline, clear pending firmware, archive data."""
		device = self._require_device(device_id, tenant_id)
		assert bool(reason), "reason required"
		assert bool(decommissioned_by), "decommissioned_by required"
		from .models import DeviceStatus as _DS
		device.status = _DS.OFFLINE if hasattr(_DS, "OFFLINE") else DeviceStatus.PROVISIONED
		device.metadata["decommission_reason"] = reason
		device.metadata["decommissioned_by"] = decommissioned_by
		# clear pending firmware schedules
		keys_to_remove = [k for k in self._firmware_schedules if k.startswith(_state_key(tenant_id, device_id))]
		for k in keys_to_remove:
			del self._firmware_schedules[k]
		record = {
			"device_id": device_id,
			"tenant_id": tenant_id,
			"reason": reason,
			"decommissioned_by": decommissioned_by,
			"firmware_schedules_cleared": len(keys_to_remove),
			"status": "decommissioned",
			"decommissioned_at": utc_now_iso(),
		}
		self._audit(tenant_id, "device_decommissioned", device_id=device_id, reason=reason)
		return record

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
			"threshold_count": len(self.list_thresholds(tenant_id)),
			"commissioned_count": len(self.list_commissioning(tenant_id)),
			"routes": len(self.describe(tenant_id)["ui"]["routes"]),
			"theme": self.describe(tenant_id)["theme"]["name"],
			"streaming": self.describe(tenant_id)["streaming"],
		}

	# ------------------------------------------------------------------ internals

	def _evaluate_threshold(
		self,
		tenant_id: str,
		device_id: str,
		metric: str,
		value: float,
		threshold: dict[str, Any],
	) -> tuple[bool, str]:
		min_val: float | None = threshold.get("min_val")
		max_val: float | None = threshold.get("max_val")
		alert_level: str = threshold.get("alert_level", "warning")
		breached = False
		direction = "none"
		if min_val is not None and value < min_val:
			breached = True
			direction = "below_min"
		elif max_val is not None and value > max_val:
			breached = True
			direction = "above_max"
		if breached:
			self._audit(
				tenant_id,
				f"threshold_breach_{alert_level}",
				device_id=device_id,
				reason=f"{metric}={value} {direction}",
			)
		return breached, direction

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
