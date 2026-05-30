"""Domain models for APG IoT Device Integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


class DeviceStatus(str, Enum):
	PROVISIONED = "provisioned"
	ONLINE = "online"
	OFFLINE = "offline"
	QUARANTINED = "quarantined"
	RETIRED = "retired"


class CommandStatus(str, Enum):
	QUEUED = "queued"
	DISPATCHED = "dispatched"
	ACKNOWLEDGED = "acknowledged"
	REJECTED = "rejected"
	TIMED_OUT = "timed_out"


class FirmwareStatus(str, Enum):
	REGISTERED = "registered"
	DEPLOYED = "deployed"
	ROLLED_BACK = "rolled_back"


@dataclass
class DeviceIdentity:
	id: str
	tenant_id: str
	device_key: str
	owner_id: str
	fleet_id: str = "default"
	certificate_id: str = ""
	status: DeviceStatus = DeviceStatus.PROVISIONED
	registered_at: str = field(default_factory=utc_now_iso)
	last_seen_at: str = field(default_factory=utc_now_iso)
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"device_key": self.device_key,
			"owner_id": self.owner_id,
			"fleet_id": self.fleet_id,
			"certificate_id": self.certificate_id,
			"status": self.status.value,
			"registered_at": self.registered_at,
			"last_seen_at": self.last_seen_at,
			"metadata": dict(self.metadata),
		}


@dataclass
class TelemetryEvent:
	id: str
	tenant_id: str
	device_id: str
	schema_name: str
	payload: dict[str, Any]
	encrypted: bool = True
	event_bus: str = "mqeb"
	accepted: bool = True
	received_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"device_id": self.device_id,
			"schema_name": self.schema_name,
			"payload": dict(self.payload),
			"encrypted": self.encrypted,
			"event_bus": self.event_bus,
			"accepted": self.accepted,
			"received_at": self.received_at,
		}


@dataclass
class DeviceCommand:
	id: str
	tenant_id: str
	device_id: str
	command: str
	parameters: dict[str, Any] = field(default_factory=dict)
	dangerous: bool = False
	approval_id: str | None = None
	status: CommandStatus = CommandStatus.QUEUED
	dispatched_at: str = field(default_factory=utc_now_iso)
	acknowledged_at: str | None = None
	ack_message: str = ""

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"device_id": self.device_id,
			"command": self.command,
			"parameters": dict(self.parameters),
			"dangerous": self.dangerous,
			"approval_id": self.approval_id,
			"status": self.status.value,
			"dispatched_at": self.dispatched_at,
			"acknowledged_at": self.acknowledged_at,
			"ack_message": self.ack_message,
		}


@dataclass
class FirmwareArtifact:
	id: str
	tenant_id: str
	version: str
	artifact_uri: str
	signature_id: str
	signature_verified: bool = True
	status: FirmwareStatus = FirmwareStatus.REGISTERED
	registered_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"version": self.version,
			"artifact_uri": self.artifact_uri,
			"signature_id": self.signature_id,
			"signature_verified": self.signature_verified,
			"status": self.status.value,
			"registered_at": self.registered_at,
		}


@dataclass
class FirmwareDeployment:
	id: str
	tenant_id: str
	firmware_id: str
	fleet_id: str
	device_ids: list[str] = field(default_factory=list)
	status: FirmwareStatus = FirmwareStatus.DEPLOYED
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"firmware_id": self.firmware_id,
			"fleet_id": self.fleet_id,
			"device_ids": list(self.device_ids),
			"status": self.status.value,
			"created_at": self.created_at,
		}


@dataclass
class DeviceAuditEvent:
	id: str
	tenant_id: str
	event_type: str
	device_id: str | None = None
	decision: str = "allow"
	reason: str = ""
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"event_type": self.event_type,
			"device_id": self.device_id,
			"decision": self.decision,
			"reason": self.reason,
			"created_at": self.created_at,
		}


@dataclass
class DeviceHealthReport:
	id: str
	tenant_id: str
	online_device_count: int = 0
	offline_device_count: int = 0
	stale_device_count: int = 0
	pending_command_count: int = 0
	unsigned_firmware_count: int = 0
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"online_device_count": self.online_device_count,
			"offline_device_count": self.offline_device_count,
			"stale_device_count": self.stale_device_count,
			"pending_command_count": self.pending_command_count,
			"unsigned_firmware_count": self.unsigned_firmware_count,
			"created_at": self.created_at,
		}


@dataclass
class IotdAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str
	registered: bool = True
	contribution_disclosed: bool = True
	status: str = "active"
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"runtime": self.runtime,
			"role": self.role,
			"scope": self.scope,
			"registered": self.registered,
			"contribution_disclosed": self.contribution_disclosed,
			"status": self.status,
			"created_at": self.created_at,
		}
