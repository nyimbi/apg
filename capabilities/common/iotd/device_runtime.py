"""Device runtime helpers for the APG IOTD capability."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from .models import CommandStatus, DeviceCommand, DeviceHealthReport, DeviceIdentity, DeviceStatus, FirmwareArtifact


def iso_days_ago(days: int | float, now: datetime | None = None) -> str:
	now = now or datetime.now(timezone.utc)
	return (now - timedelta(days=float(days))).isoformat()


def parse_iso_timestamp(value: str) -> datetime:
	parsed = datetime.fromisoformat(value)
	if parsed.tzinfo is None:
		parsed = parsed.replace(tzinfo=timezone.utc)
	return parsed


class TelemetrySchemaValidator:
	"""Small deterministic validator for dependency-light telemetry probes."""

	def validate(self, payload: dict[str, Any], required_fields: list[str] | None = None) -> dict[str, Any]:
		required = required_fields or ["timestamp"]
		missing = [field for field in required if field not in payload]
		return {"valid": not missing, "missing_fields": missing}


class DeviceFreshnessInspector:
	"""Find stale devices and compute last-seen ages."""

	def last_seen_days(self, device: DeviceIdentity, now: datetime | None = None) -> float:
		now = now or datetime.now(timezone.utc)
		try:
			last_seen = parse_iso_timestamp(device.last_seen_at)
		except ValueError:
			return 0.0
		return max(0.0, (now - last_seen).total_seconds() / 86400)

	def stale_devices(
		self,
		devices: list[DeviceIdentity],
		tenant_id: str,
		threshold_days: int,
		now: datetime | None = None,
	) -> list[DeviceIdentity]:
		return [
			device
			for device in devices
			if device.tenant_id == tenant_id
			and device.status != DeviceStatus.RETIRED
			and self.last_seen_days(device, now) > threshold_days
		]


class DeviceHealthInspector:
	"""Summarize device operations health for dashboards and reviews."""

	def __init__(self) -> None:
		self._freshness = DeviceFreshnessInspector()

	def summarize(
		self,
		report_id: str,
		tenant_id: str,
		devices: list[DeviceIdentity],
		commands: list[DeviceCommand],
		firmware: list[FirmwareArtifact],
		stale_device_review_days: int,
	) -> DeviceHealthReport:
		tenant_devices = [device for device in devices if device.tenant_id == tenant_id]
		pending = [
			command
			for command in commands
			if command.tenant_id == tenant_id and command.status in {CommandStatus.QUEUED, CommandStatus.DISPATCHED}
		]
		unsigned = [
			artifact
			for artifact in firmware
			if artifact.tenant_id == tenant_id and not artifact.signature_verified
		]
		return DeviceHealthReport(
			id=report_id,
			tenant_id=tenant_id,
			online_device_count=len([device for device in tenant_devices if device.status == DeviceStatus.ONLINE]),
			offline_device_count=len([device for device in tenant_devices if device.status == DeviceStatus.OFFLINE]),
			stale_device_count=len(self._freshness.stale_devices(tenant_devices, tenant_id, stale_device_review_days)),
			pending_command_count=len(pending),
			unsigned_firmware_count=len(unsigned),
		)
