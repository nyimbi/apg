"""Pydantic v2 models for APG Data Synchronisation."""
from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field

try:
	from situ_cloudevents._uuid7 import uuid7str  # type: ignore[import]
except ImportError:
	from uuid6 import uuid7  # type: ignore[import]
	def uuid7str() -> str:
		return str(uuid7())


class SyncDirection(str, Enum):
	SOURCE_TO_TARGET = "source_to_target"
	TARGET_TO_SOURCE = "target_to_source"
	BIDIRECTIONAL = "bidirectional"


class SyncStatus(str, Enum):
	IDLE = "idle"
	RUNNING = "running"
	ERROR = "error"
	PAUSED = "paused"


class DsyFieldMapping(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	source_field: str
	target_field: str
	transform: str = ""  # optional JMESPath transform
	required: bool = False


class DsySyncConfig(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str
	source_capability: str
	source_entity: str
	target_capability: str
	target_entity: str
	direction: SyncDirection = SyncDirection.BIDIRECTIONAL
	field_mappings: list[DsyFieldMapping] = Field(default_factory=list)
	frequency_minutes: int = 15
	batch_size: int = 500
	conflict_resolution: str = "source_wins"
	status: SyncStatus = SyncStatus.IDLE
	enabled: bool = True
	last_sync_at: datetime | None = None
	last_sync_records: int = 0
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DsySyncRun(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	sync_config_id: str
	status: str = "running"
	records_processed: int = 0
	records_created: int = 0
	records_updated: int = 0
	records_skipped: int = 0
	conflicts: int = 0
	error_message: str | None = None
	started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	completed_at: datetime | None = None


class DsySyncConflict(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	sync_config_id: str
	entity_id: str
	source_value: dict[str, Any] = Field(default_factory=dict)
	target_value: dict[str, Any] = Field(default_factory=dict)
	resolution: str = "pending"
	resolved_by: str | None = None
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
