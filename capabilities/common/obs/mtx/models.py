"""Pydantic v2 models for Metrics & SLO (obs_mtx)."""
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


class MetricDefinitionCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	name: str
	description: str = ""
	metric_type: str  # counter|gauge|histogram|summary
	unit: str = ""
	labels: list[str] = Field(default_factory=list)
	service_name: str
	namespace: str = "apg"


class MetricDefinitionUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	description: str | None = None
	labels: list[str] | None = None
	enabled: bool | None = None


class MetricDefinitionResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	name: str
	description: str
	metric_type: str
	unit: str
	labels: list[str]
	service_name: str
	namespace: str
	enabled: bool = True
	tenant_id: str
	created_at: str


class MetricDataPointCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	metric_name: str
	value: float
	labels: dict[str, str] = Field(default_factory=dict)
	timestamp: str | None = None
	service_name: str


class MetricDataPointResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	metric_name: str
	value: float
	labels: dict[str, str]
	timestamp: str
	service_name: str
	tenant_id: str


class SLOCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	name: str
	description: str = ""
	service_name: str
	slo_type: str  # availability|latency|error_rate|throughput
	target: float = Field(ge=0.0, le=100.0, description="Target percentage, e.g. 99.9")
	window_days: int = Field(ge=1, le=90, default=30)
	good_query: str = ""
	total_query: str = ""
	latency_threshold_ms: float | None = None


class SLOUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	description: str | None = None
	target: float | None = None
	window_days: int | None = None
	good_query: str | None = None
	total_query: str | None = None
	latency_threshold_ms: float | None = None
	enabled: bool | None = None


class SLOResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	name: str
	description: str
	service_name: str
	slo_type: str
	target: float
	window_days: int
	good_query: str
	total_query: str
	latency_threshold_ms: float | None
	enabled: bool = True
	current_compliance: float | None = None
	error_budget_remaining: float | None = None
	tenant_id: str
	created_at: str
	updated_at: str | None = None


class SLOListResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	items: list[SLOResponse]
	total: int
	page: int = 1
	page_size: int = 50


class BurnRateAlertCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	slo_id: str
	name: str
	short_window_minutes: int = 60
	long_window_minutes: int = 360
	burn_rate_threshold: float = Field(ge=1.0, default=14.4)
	severity: str = "critical"  # critical|warning|info
	notification_channels: list[str] = Field(default_factory=list)


class BurnRateAlertResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	slo_id: str
	name: str
	short_window_minutes: int
	long_window_minutes: int
	burn_rate_threshold: float
	severity: str
	notification_channels: list[str]
	enabled: bool = True
	firing: bool = False
	last_fired_at: str | None = None
	tenant_id: str
	created_at: str


class PrometheusExportConfig(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	endpoint: str = "/metrics"
	port: int = 9090
	scrape_interval_seconds: int = 15
	include_namespaces: list[str] = Field(default_factory=list)
	exclude_labels: list[str] = Field(default_factory=list)
	enabled: bool = True
	tenant_id: str
	created_at: str


class DashboardCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	name: str
	description: str = ""
	service_name: str | None = None
	panels: list[dict[str, Any]] = Field(default_factory=list)
	variables: list[dict[str, Any]] = Field(default_factory=list)
	refresh_interval_seconds: int = 30
	tags: list[str] = Field(default_factory=list)


class DashboardResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	name: str
	description: str
	service_name: str | None
	panels: list[dict[str, Any]]
	variables: list[dict[str, Any]]
	refresh_interval_seconds: int
	tags: list[str]
	tenant_id: str
	created_at: str
	updated_at: str | None = None


class MetricFilterModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	service_name: str | None = None
	metric_type: str | None = None
	namespace: str | None = None
	enabled_only: bool = True
	page: int = 1
	page_size: int = 50


class REDMetricsSummary(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	service_name: str
	window_minutes: int
	request_rate: float = 0.0
	error_rate: float = 0.0
	p50_duration_ms: float = 0.0
	p95_duration_ms: float = 0.0
	p99_duration_ms: float = 0.0
	total_requests: int = 0
	total_errors: int = 0
	tenant_id: str
	computed_at: str


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
