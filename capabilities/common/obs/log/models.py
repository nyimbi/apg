"""Pydantic v2 models for Log Aggregation (obs_log)."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field

try:
	from uuid6 import uuid7
	def uuid7str() -> str:
		return str(uuid7())
except ImportError:
	from uuid import uuid4
	def uuid7str() -> str:  # type: ignore[misc]
		return str(uuid4())


LOG_LEVELS = {"TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class LogEntryCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	service_name: str
	level: str = "INFO"
	message: str
	timestamp: str | None = None
	correlation_id: str | None = None
	trace_id: str | None = None
	span_id: str | None = None
	fields: dict[str, Any] = Field(default_factory=dict)
	source_file: str | None = None
	source_line: int | None = None
	logger_name: str | None = None


class LogEntryResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	service_name: str
	level: str
	message: str
	timestamp: str
	correlation_id: str | None = None
	trace_id: str | None = None
	span_id: str | None = None
	fields: dict[str, Any] = Field(default_factory=dict)
	source_file: str | None = None
	source_line: int | None = None
	logger_name: str | None = None
	tenant_id: str
	ingested_at: str


class LogEntryListResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	items: list[LogEntryResponse]
	total: int
	page: int = 1
	page_size: int = 100
	has_more: bool = False


class LogFilterModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	service_name: str | None = None
	level: str | None = None
	min_level: str | None = None
	correlation_id: str | None = None
	trace_id: str | None = None
	start_time: str | None = None
	end_time: str | None = None
	message_contains: str | None = None
	fields_match: dict[str, Any] | None = None
	page: int = 1
	page_size: int = 100


class RetentionPolicyCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	name: str
	service_name: str | None = None
	min_level: str = "DEBUG"
	retention_days: int = Field(ge=1, le=3650, default=30)
	archive_after_days: int | None = None
	delete_after_days: int | None = None
	compress_after_days: int | None = None
	enabled: bool = True


class RetentionPolicyUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	min_level: str | None = None
	retention_days: int | None = None
	archive_after_days: int | None = None
	delete_after_days: int | None = None
	compress_after_days: int | None = None
	enabled: bool | None = None


class RetentionPolicyResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	name: str
	service_name: str | None
	min_level: str
	retention_days: int
	archive_after_days: int | None
	delete_after_days: int | None
	compress_after_days: int | None
	enabled: bool
	tenant_id: str
	created_at: str
	updated_at: str | None = None


class LogLevelOverrideCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	service_name: str
	logger_name: str | None = None
	level: str
	duration_minutes: int | None = Field(default=None, ge=1, le=1440)
	reason: str = ""


class LogLevelOverrideResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	service_name: str
	logger_name: str | None
	level: str
	duration_minutes: int | None
	reason: str
	expires_at: str | None
	active: bool = True
	tenant_id: str
	created_at: str


class LokiExportConfigCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	name: str
	endpoint: str
	tenant_header: str | None = None
	extra_labels: dict[str, str] = Field(default_factory=dict)
	batch_size: int = 1000
	flush_interval_ms: int = 1000
	max_retries: int = 3
	enabled: bool = True


class LokiExportConfigResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	name: str
	endpoint: str
	tenant_header: str | None
	extra_labels: dict[str, str]
	batch_size: int
	flush_interval_ms: int
	max_retries: int
	enabled: bool
	tenant_id: str
	created_at: str


class CorrelationContextCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	correlation_id: str | None = None
	trace_id: str | None = None
	request_id: str | None = None
	user_id: str | None = None
	session_id: str | None = None
	service_name: str
	extra: dict[str, str] = Field(default_factory=dict)


class CorrelationContextResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	correlation_id: str
	trace_id: str | None
	request_id: str | None
	user_id: str | None
	session_id: str | None
	service_name: str
	extra: dict[str, str]
	tenant_id: str
	created_at: str


class AuditEventResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	record_id: str
	record_type: str
	actor: str | None = None
	details: dict[str, Any] = Field(default_factory=dict)
	emitted_at: str
