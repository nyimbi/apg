# Author: Nyimbi Odero
# Company: Datacraft
# Copyright: © 2025
"""Pydantic v2 models for Observability (obs) — top-level umbrella capability.

Covers: TraceSpan, Metric, LogEntry, HealthStatus, SLOConfig, AlertRule.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

try:
	from uuid6 import uuid7

	def uuid7str() -> str:
		return str(uuid7())
except ImportError:
	from uuid import uuid4

	def uuid7str() -> str:  # type: ignore[misc]
		return str(uuid4())


# ---------------------------------------------------------------------------
# TraceSpan
# ---------------------------------------------------------------------------

class TraceSpan(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str, description="Unique span ID (UUID7)")
	trace_id: str = Field(description="Parent trace ID (32-char hex or UUID)")
	parent_span_id: str | None = Field(default=None, description="Parent span ID, None for root spans")
	operation_name: str = Field(description="Name of the traced operation")
	service_name: str = Field(description="Name of the originating service")
	start_time: datetime = Field(description="Span start timestamp (UTC)")
	end_time: datetime | None = Field(default=None, description="Span end timestamp (UTC)")
	duration_ms: float | None = Field(default=None, description="Computed duration in milliseconds")
	status: Literal["ok", "error", "unset"] = Field(default="unset")
	status_message: str | None = None
	kind: Literal["internal", "client", "server", "producer", "consumer"] = "internal"
	tags: dict[str, str] = Field(default_factory=dict)
	logs: list[dict[str, Any]] = Field(default_factory=list)
	baggage: dict[str, str] = Field(default_factory=dict)
	sampled: bool = True
	error: bool = False
	tenant_id: str = Field(default="default")
	created_at: datetime = Field(default_factory=datetime.utcnow)

	@field_validator("operation_name", "service_name")
	@classmethod
	def _non_empty(cls, v: str) -> str:
		if not v.strip():
			raise ValueError("must not be blank")
		return v


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

class Metric(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	name: str = Field(description="Metric name, e.g. http_requests_total")
	metric_type: Literal["counter", "gauge", "histogram", "summary"] = "counter"
	value: float = Field(description="Scalar metric value")
	labels: dict[str, str] = Field(default_factory=dict, description="Prometheus-style label set")
	service_name: str = Field(description="Originating service")
	timestamp: datetime = Field(default_factory=datetime.utcnow)
	unit: str | None = Field(default=None, description="Optional unit, e.g. ms, bytes")
	description: str | None = None
	tenant_id: str = Field(default="default")

	@field_validator("name", "service_name")
	@classmethod
	def _non_empty(cls, v: str) -> str:
		if not v.strip():
			raise ValueError("must not be blank")
		return v


# ---------------------------------------------------------------------------
# LogEntry
# ---------------------------------------------------------------------------

class LogEntry(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	level: Literal["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
	message: str = Field(description="Human-readable log message")
	service_name: str = Field(description="Originating service")
	timestamp: datetime = Field(default_factory=datetime.utcnow)
	# Correlation fields
	trace_id: str | None = None
	span_id: str | None = None
	correlation_id: str | None = None
	# Structured fields
	fields: dict[str, Any] = Field(default_factory=dict)
	logger_name: str | None = None
	source_file: str | None = None
	source_line: int | None = None
	tenant_id: str = Field(default="default")

	@field_validator("message", "service_name")
	@classmethod
	def _non_empty(cls, v: str) -> str:
		if not v.strip():
			raise ValueError("must not be blank")
		return v


# ---------------------------------------------------------------------------
# HealthStatus
# ---------------------------------------------------------------------------

class HealthStatus(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	service_name: str
	status: Literal["healthy", "degraded", "unhealthy", "unknown"] = "unknown"
	checks: dict[str, Any] = Field(default_factory=dict, description="Per-check results keyed by check name")
	message: str | None = None
	version: str | None = None
	uptime_seconds: float | None = None
	checked_at: datetime = Field(default_factory=datetime.utcnow)
	tenant_id: str = Field(default="default")


# ---------------------------------------------------------------------------
# SLOConfig
# ---------------------------------------------------------------------------

class SLOConfig(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	name: str = Field(description="Human-readable SLO name")
	service_name: str
	slo_type: Literal["availability", "latency", "error_rate", "throughput"] = "availability"
	target: float = Field(ge=0.0, le=1.0, description="SLO target as a ratio, e.g. 0.999 for 99.9%")
	window_seconds: int = Field(default=2592000, description="Rolling window in seconds (default 30 days)")
	latency_threshold_ms: float | None = Field(
		default=None, description="For latency SLOs: threshold in milliseconds"
	)
	description: str | None = None
	enabled: bool = True
	tenant_id: str = Field(default="default")
	created_at: datetime = Field(default_factory=datetime.utcnow)

	@field_validator("name", "service_name")
	@classmethod
	def _non_empty(cls, v: str) -> str:
		if not v.strip():
			raise ValueError("must not be blank")
		return v


# ---------------------------------------------------------------------------
# AlertRule
# ---------------------------------------------------------------------------

class AlertRule(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	name: str = Field(description="Alert rule name")
	service_name: str
	severity: Literal["critical", "warning", "info"] = "warning"
	condition: str = Field(description="PromQL-style or freeform condition expression")
	threshold: float = Field(description="Numeric threshold for the condition")
	window_seconds: int = Field(default=300, description="Evaluation window in seconds")
	slo_id: str | None = Field(default=None, description="Optional linked SLO ID")
	notify_channels: list[str] = Field(default_factory=list, description="Notification channel IDs")
	enabled: bool = True
	description: str | None = None
	tenant_id: str = Field(default="default")
	created_at: datetime = Field(default_factory=datetime.utcnow)

	@field_validator("name", "service_name", "condition")
	@classmethod
	def _non_empty(cls, v: str) -> str:
		if not v.strip():
			raise ValueError("must not be blank")
		return v
