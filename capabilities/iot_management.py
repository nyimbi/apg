"""IoT management capability facade for APG integrations."""

from __future__ import annotations

import statistics
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from uuid_extensions import uuid7str


class DeviceType(StrEnum):
	SENSOR = "sensor"
	CAMERA = "camera"
	GATEWAY = "gateway"
	ACTUATOR = "actuator"


class SensorType(StrEnum):
	TEMPERATURE = "temperature"
	HUMIDITY = "humidity"
	MOTION = "motion"
	PRESSURE = "pressure"


class ConnectionType(StrEnum):
	WIFI = "wifi"
	ETHERNET = "ethernet"
	CELLULAR = "cellular"
	BLUETOOTH = "bluetooth"


class DeviceStatus(StrEnum):
	ONLINE = "online"
	OFFLINE = "offline"
	ERROR = "error"


class IoTManagementCapability:
	"""In-memory IoT device, sensor, command, and alert manager."""

	def __init__(self, config: dict[str, Any] | None = None):
		self.config = config or {}
		self.devices: dict[str, dict[str, Any]] = {}
		self.sensor_readings: list[dict[str, Any]] = []
		self.commands: dict[str, dict[str, Any]] = {}
		self.alert_rules: dict[str, dict[str, Any]] = {}

	def get_capability_info(self) -> dict[str, Any]:
		return {
			"name": "iot_management",
			"features": ["device_registry", "sensor_data", "commands", "alerts"],
			"supported_devices": [item.value for item in DeviceType],
			"supported_sensors": [item.value for item in SensorType],
			"connection_types": [item.value for item in ConnectionType],
		}

	async def register_device(self, device_data: dict[str, Any]) -> dict[str, Any]:
		device_id = device_data.get("device_id") or uuid7str()
		device = dict(device_data)
		device["device_id"] = device_id
		device.setdefault("status", DeviceStatus.ONLINE.value)
		device.setdefault("registered_at", datetime.utcnow().isoformat())
		self.devices[device_id] = device
		return {"success": True, "device_id": device_id, "device": device}

	async def list_devices(self) -> dict[str, Any]:
		return {
			"success": True,
			"devices": list(self.devices.values()),
			"count": len(self.devices),
		}

	async def get_device_info(self, device_id: str) -> dict[str, Any]:
		device = self.devices.get(device_id)
		if not device:
			return {"success": False, "error": "device_not_found"}
		return {"success": True, "device": device}

	async def record_sensor_data(self, reading: dict[str, Any]) -> dict[str, Any]:
		if not reading.get("sensor_id") or not isinstance(reading.get("value"), (int, float, bool)):
			return {"success": False, "error": "invalid_sensor_data"}
		record = dict(reading)
		record["id"] = uuid7str()
		record["timestamp"] = datetime.utcnow()
		self.sensor_readings.append(record)
		return {"success": True, "reading_id": record["id"]}

	async def get_sensor_readings(
		self,
		device_id: str,
		sensor_type: str | None = None,
		hours_back: int = 24,
	) -> dict[str, Any]:
		cutoff = datetime.utcnow() - timedelta(hours=hours_back)
		readings = [
			reading for reading in self.sensor_readings
			if reading["timestamp"] >= cutoff
			and reading.get("sensor_id", "").startswith(device_id)
			and (sensor_type is None or reading.get("sensor_type") == sensor_type)
		]
		return {"success": True, "readings": readings, "count": len(readings)}

	async def get_sensor_statistics(
		self,
		device_id: str,
		sensor_type: str | None = None,
		hours_back: int = 24,
	) -> dict[str, Any]:
		result = await self.get_sensor_readings(device_id, sensor_type, hours_back)
		values = [float(reading["value"]) for reading in result["readings"] if isinstance(reading.get("value"), (int, float))]
		statistics_payload = {}
		if values:
			statistics_payload = {
				"mean": statistics.mean(values),
				"min": min(values),
				"max": max(values),
				"count": len(values),
			}
		return {"success": True, "statistics": statistics_payload}

	async def send_device_command(
		self,
		device_id: str,
		command: str,
		parameters: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		if device_id not in self.devices:
			return {"success": False, "error": "device_not_found"}
		command_id = uuid7str()
		self.commands[command_id] = {
			"command_id": command_id,
			"device_id": device_id,
			"command": command,
			"parameters": parameters or {},
			"status": "sent",
		}
		return {"success": True, "command_id": command_id}

	async def create_alert_rule(self, rule_data: dict[str, Any]) -> dict[str, Any]:
		rule_id = uuid7str()
		rule = dict(rule_data)
		rule["rule_id"] = rule_id
		self.alert_rules[rule_id] = rule
		return {"success": True, "rule_id": rule_id, "rule": rule}


__all__ = [
	"ConnectionType",
	"DeviceStatus",
	"DeviceType",
	"IoTManagementCapability",
	"SensorType",
]
