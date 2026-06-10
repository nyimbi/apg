"""Pydantic v2 models for Distributed Tracing (obs_trc)."""
from __future__ import annotations

from datetime import datetime
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


class SpanCreateModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	trace_id: str | None = None
	parent_span_id: str | None = None
	operation_name: str
	service_name: str
	start_time: datetime | None = None
	tags: dict[str, str] = Field(default_factory=dict)
	baggage: dict[str, str] = Field(default_factory=dict)
	sampled: bool = True
	kind: str = "internal"  # internal|client|server|producer|consumer


class SpanUpdateModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	end_time: datetime | None = None
	status: str | None = None  # ok|error|unset
	status_message: str | None = None
	tags: dict[str, str] | None = None
	logs: list[dict[str, Any]] | None = None
	error: bool | None = None


class SpanResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	trace_id: str
	parent_span_id: str | None = None
	operation_name: str
	service_name: str
	start_time: str
	end_time: str | None = None
	duration_ms: float | None = None
	status: str = "unset"
	status_message: str | None = None
	tags: dict[str, str] = Field(default_factory=dict)
	logs: list[dict[str, Any]] = Field(default_factory=list)
	baggage: dict[str, str] = Field(default_factory=dict)
	sampled: bool = True
	kind: str = "internal"
	error: bool = False
	tenant_id: str
	created_at: str
	updated_at: str | None = None


class TraceResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	root_span_id: str
	service_name: str
	operation_name: str
	start_time: str
	end_time: str | None = None
	duration_ms: float | None = None
	span_count: int = 0
	error_count: int = 0
	status: str = "in_progress"
	tags: dict[str, str] = Field(default_factory=dict)
	tenant_id: str
	created_at: str


class SpanListResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	items: list[SpanResponse]
	total: int
	page: int = 1
	page_size: int = 50


class TraceListResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	items: list[TraceResponse]
	total: int
	page: int = 1
	page_size: int = 50


class SpanFilterModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	trace_id: str | None = None
	service_name: str | None = None
	operation_name: str | None = None
	status: str | None = None
	error_only: bool = False
	start_after: str | None = None
	start_before: str | None = None
	min_duration_ms: float | None = None
	max_duration_ms: float | None = None
	page: int = 1
	page_size: int = 50


class SamplingRuleCreateModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	name: str
	service_name: str | None = None
	operation_pattern: str | None = None
	sample_rate: float = Field(ge=0.0, le=1.0, default=1.0)
	priority: int = 100
	strategy: str = "probabilistic"  # probabilistic|rate_limiting|always_on|always_off


class SamplingRuleResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	name: str
	service_name: str | None = None
	operation_pattern: str | None = None
	sample_rate: float
	priority: int
	strategy: str
	enabled: bool = True
	tenant_id: str
	created_at: str


class ServiceDependencyResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	source_service: str
	target_service: str
	call_count: int = 0
	error_count: int = 0
	avg_latency_ms: float = 0.0
	p99_latency_ms: float = 0.0
	last_seen: str | None = None
	tenant_id: str


class ExportConfigCreateModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	name: str
	exporter_type: str  # jaeger|tempo|otlp|zipkin
	endpoint: str
	headers: dict[str, str] = Field(default_factory=dict)
	batch_size: int = 512
	flush_interval_ms: int = 5000
	enabled: bool = True


class ExportConfigResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	name: str
	exporter_type: str
	endpoint: str
	headers: dict[str, str] = Field(default_factory=dict)
	batch_size: int
	flush_interval_ms: int
	enabled: bool
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
