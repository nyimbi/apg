"""APG Monitoring and Observability (MONI) — Pydantic v2 data models.

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from uuid6 import uuid7


def uuid7str() -> str:
	"""Return a UUID7 string suitable for use as a default field factory."""
	return str(uuid7())


# ─── Enums ────────────────────────────────────────────────────────────────────

class MetricType(str, Enum):
	"""Types of monitoring metrics."""
	COUNTER = "counter"
	GAUGE = "gauge"
	HISTOGRAM = "histogram"
	SUMMARY = "summary"
	SET = "set"


class AlertSeverity(str, Enum):
	"""Alert severity levels."""
	CRITICAL = "critical"
	HIGH = "high"
	MEDIUM = "medium"
	LOW = "low"
	INFO = "info"


class AlertStatus(str, Enum):
	"""Alert lifecycle states."""
	ACTIVE = "active"
	ACKNOWLEDGED = "acknowledged"
	RESOLVED = "resolved"
	SUPPRESSED = "suppressed"


class AlertConditionType(str, Enum):
	"""Alert condition types."""
	THRESHOLD = "threshold"
	ANOMALY = "anomaly"
	RATE = "rate"
	ABSENCE = "absence"
	COMPOSITE = "composite"


class DashboardType(str, Enum):
	"""Dashboard personas."""
	EXECUTIVE = "executive"
	OPERATIONAL = "operational"
	DEVELOPER = "developer"
	TENANT = "tenant"
	CUSTOM = "custom"


class DataRetentionPolicy(str, Enum):
	"""Data retention tiers."""
	REAL_TIME = "real_time"       # 1 hour
	SHORT_TERM = "short_term"     # 24 hours
	MEDIUM_TERM = "medium_term"   # 7 days
	LONG_TERM = "long_term"       # 30 days
	ARCHIVE = "archive"           # 1 year


class MonitoringScope(str, Enum):
	"""Monitoring scope for rules and dashboards."""
	GLOBAL = "global"
	TENANT = "tenant"
	APPLICATION = "application"
	SERVICE = "service"
	INFRASTRUCTURE = "infrastructure"


# ─── Validators ───────────────────────────────────────────────────────────────

def _validate_labels(labels: dict[str, str]) -> dict[str, str]:
	"""Validate metric labels — keys and values must be short strings."""
	assert isinstance(labels, dict), "Labels must be a dictionary"
	for key, value in labels.items():
		assert isinstance(key, str) and isinstance(value, str), \
			"Label keys and values must be strings"
		assert len(key) <= 255, f"Label key too long: {key!r}"
		assert len(value) <= 1024, f"Label value too long for key {key!r}"
	return labels


def _validate_alert_condition(condition: str) -> str:
	"""Validate alert condition expression."""
	assert condition.strip(), "Alert condition cannot be empty"
	assert len(condition) <= 2048, "Alert condition too long (max 2048 chars)"
	return condition.strip()


# ─── Models ───────────────────────────────────────────────────────────────────

class MonitoringMetric(BaseModel):
	"""Core monitoring metric for time-series data collection.

	Supports high-cardinality metrics with efficient label-based routing.
	"""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	metric_id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(..., description="APG tenant identifier")
	name: str = Field(..., max_length=255, description="Metric name")

	value: float = Field(..., description="Metric value (any finite float)")
	metric_type: MetricType = Field(default=MetricType.GAUGE)
	unit: str | None = Field(None, max_length=50)

	timestamp: datetime = Field(default_factory=datetime.utcnow)
	interval_seconds: int | None = Field(None, ge=1)

	labels: Annotated[dict[str, str], AfterValidator(_validate_labels)] = Field(
		default_factory=dict,
	)
	source: str = Field(..., max_length=255, description="Source system or component")
	source_type: str = Field(default="unknown", max_length=100)

	quality_score: float = Field(default=1.0, ge=0.0, le=1.0)
	processed: bool = Field(default=False)
	retention_policy: DataRetentionPolicy = Field(default=DataRetentionPolicy.MEDIUM_TERM)

	capability_name: str | None = Field(None, max_length=100)
	correlation_id: str | None = None

	ingestion_latency_ms: float | None = Field(None, ge=0.0)
	processing_time_ms: float | None = Field(None, ge=0.0)

	def is_stale(self, max_age_seconds: int = 300) -> bool:
		"""Return True if metric timestamp exceeds max_age_seconds."""
		assert max_age_seconds > 0, "max_age_seconds must be positive"
		age = (datetime.utcnow() - self.timestamp).total_seconds()
		return age > max_age_seconds

	def get_label_signature(self) -> str:
		"""Stable string signature of labels for grouping/keying."""
		return "|".join(f"{k}={v}" for k, v in sorted(self.labels.items()))

	def to_prometheus_format(self) -> str:
		"""Emit Prometheus exposition format line."""
		labels_str = ",".join(f'{k}="{v}"' for k, v in self.labels.items())
		ms = int(self.timestamp.timestamp() * 1000)
		return f'{self.name}{{{labels_str}}} {self.value} {ms}'


class MonitoringAlert(BaseModel):
	"""Alert with correlation context and escalation tracking."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	alert_id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(...)
	rule_id: str = Field(...)

	name: str = Field(..., max_length=255)
	description: str = Field(default="", max_length=1000)
	severity: AlertSeverity = Field(default=AlertSeverity.MEDIUM)
	status: AlertStatus = Field(default=AlertStatus.ACTIVE)

	message: str = Field(..., max_length=2048)
	summary: str = Field(default="", max_length=500)
	runbook_url: str | None = None

	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	resolved_at: datetime | None = None
	acknowledged_at: datetime | None = None

	correlation_key: str | None = None
	parent_alert_id: str | None = None
	related_alert_ids: list[str] = Field(default_factory=list)

	labels: dict[str, str] = Field(default_factory=dict)
	annotations: dict[str, str] = Field(default_factory=dict)
	source_metric: str | None = None
	source_value: float | None = None
	threshold_value: float | None = None

	assigned_to: str | None = None
	escalation_level: int = Field(default=0, ge=0)
	max_escalation_level: int = Field(default=3, ge=1)
	escalation_interval_minutes: int = Field(default=30, ge=1)

	impact_score: float = Field(default=0.0, ge=0.0, le=1.0)
	affected_services: list[str] = Field(default_factory=list)
	affected_users_count: int = Field(default=0, ge=0)

	notification_sent: bool = False
	audit_logged: bool = False

	def is_active(self) -> bool:
		return self.status == AlertStatus.ACTIVE

	def can_escalate(self) -> bool:
		return self.escalation_level < self.max_escalation_level

	def get_age_minutes(self) -> float:
		return (datetime.utcnow() - self.created_at).total_seconds() / 60

	def should_escalate(self) -> bool:
		if not self.can_escalate() or not self.is_active():
			return False
		return self.get_age_minutes() >= (self.escalation_level + 1) * self.escalation_interval_minutes


class MonitoringRule(BaseModel):
	"""Alert rule configuration with threshold and anomaly detection support."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	rule_id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(...)
	name: str = Field(..., max_length=255)
	description: str = Field(default="", max_length=1000)

	enabled: bool = True
	condition: Annotated[str, AfterValidator(_validate_alert_condition)] = Field(...)
	condition_type: AlertConditionType = Field(default=AlertConditionType.THRESHOLD)

	metric_name: str = Field(..., max_length=255)
	metric_labels: dict[str, str] = Field(default_factory=dict)
	scope: MonitoringScope = Field(default=MonitoringScope.TENANT)

	threshold_value: float | None = None
	threshold_operator: str = Field(default="gt")
	evaluation_window_minutes: int = Field(default=5, ge=1, le=1440)
	evaluation_interval_seconds: int = Field(default=60, ge=10)

	severity: AlertSeverity = Field(default=AlertSeverity.MEDIUM)
	alert_message: str = Field(..., max_length=2048)
	alert_summary: str = Field(default="", max_length=500)
	runbook_url: str | None = None

	escalation_enabled: bool = True
	escalation_interval_minutes: int = Field(default=30, ge=1)
	max_escalation_level: int = Field(default=3, ge=1)

	suppression_enabled: bool = False
	suppression_window_minutes: int = Field(default=60, ge=1)
	correlation_key: str | None = None

	anomaly_detection_enabled: bool = False
	anomaly_sensitivity: float = Field(default=0.8, ge=0.0, le=1.0)
	baseline_period_days: int = Field(default=7, ge=1)

	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str = Field(...)
	last_triggered: datetime | None = None
	trigger_count: int = Field(default=0, ge=0)

	evaluation_time_ms: float = Field(default=0.0, ge=0.0)
	false_positive_rate: float = Field(default=0.0, ge=0.0, le=1.0)
	effectiveness_score: float = Field(default=0.0, ge=0.0, le=1.0)

	def is_due_for_evaluation(self) -> bool:
		"""True if rule has never been triggered or is past its next eval time."""
		if not self.enabled:
			return False
		if not self.last_triggered:
			return True
		next_eval = self.last_triggered + timedelta(seconds=self.evaluation_interval_seconds)
		return datetime.utcnow() >= next_eval

	def get_evaluation_query(self) -> str:
		labels_filter = " AND ".join(f'{k}="{v}"' for k, v in self.metric_labels.items())
		window = f"{self.evaluation_window_minutes}m"
		return f"SELECT {self.metric_name} WHERE {labels_filter} TIMEFRAME {window}"

	def update_performance_stats(self, evaluation_time_ms: float, is_false_positive: bool = False) -> None:
		"""Rolling EWMA update of rule performance statistics."""
		assert evaluation_time_ms >= 0, "evaluation_time_ms must be non-negative"
		self.evaluation_time_ms = (self.evaluation_time_ms * 0.9) + (evaluation_time_ms * 0.1)
		if is_false_positive:
			self.false_positive_rate = (self.false_positive_rate * 0.95) + 0.05
		else:
			self.false_positive_rate = self.false_positive_rate * 0.95
		self.effectiveness_score = 1.0 - self.false_positive_rate
		self.updated_at = datetime.utcnow()


class MonitoringDashboard(BaseModel):
	"""Dashboard configuration with adaptive layout and engagement tracking."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	dashboard_id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(...)
	name: str = Field(..., max_length=255)
	description: str = Field(default="", max_length=1000)

	dashboard_type: DashboardType = Field(default=DashboardType.OPERATIONAL)
	scope: MonitoringScope = Field(default=MonitoringScope.TENANT)
	auto_refresh: bool = True
	refresh_interval_seconds: int = Field(default=30, ge=5, le=3600)

	layout: dict[str, Any] = Field(default_factory=dict)
	widgets: list[dict[str, Any]] = Field(default_factory=list)
	widget_count: int = Field(default=0, ge=0)

	public: bool = False
	shared_with: list[str] = Field(default_factory=list)
	view_permissions: list[str] = Field(default_factory=list)
	edit_permissions: list[str] = Field(default_factory=list)

	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str = Field(...)
	last_viewed: datetime | None = None
	view_count: int = Field(default=0, ge=0)

	cached: bool = False
	cache_ttl_seconds: int = Field(default=300, ge=1)
	preload_data: bool = False

	avg_load_time_ms: float = Field(default=0.0, ge=0.0)
	user_engagement_score: float = Field(default=0.0, ge=0.0, le=1.0)
	popularity_score: float = Field(default=0.0, ge=0.0, le=1.0)

	def add_widget(self, widget_config: dict[str, Any]) -> None:
		assert isinstance(widget_config, dict), "Widget config must be a dict"
		assert "type" in widget_config, "Widget must have a type"
		widget_config["widget_id"] = uuid7str()
		self.widgets.append(widget_config)
		self.widget_count = len(self.widgets)
		self.updated_at = datetime.utcnow()

	def remove_widget(self, widget_id: str) -> bool:
		initial = len(self.widgets)
		self.widgets = [w for w in self.widgets if w.get("widget_id") != widget_id]
		removed = len(self.widgets) < initial
		if removed:
			self.widget_count = len(self.widgets)
			self.updated_at = datetime.utcnow()
		return removed

	def update_view_stats(self, load_time_ms: float) -> None:
		assert load_time_ms >= 0, "load_time_ms must be non-negative"
		self.view_count += 1
		self.last_viewed = datetime.utcnow()
		self.avg_load_time_ms = (self.avg_load_time_ms * 0.9) + (load_time_ms * 0.1)
		days = max((datetime.utcnow() - self.created_at).days, 1)
		self.user_engagement_score = min(self.view_count / days / 10.0, 1.0)


class MonitoringQuery(BaseModel):
	"""Metric query with time range, label filters, and aggregation."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	query_id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(...)

	metric_names: list[str] = Field(...)
	labels: dict[str, str | list[str]] = Field(default_factory=dict)

	start_time: datetime = Field(...)
	end_time: datetime = Field(...)
	step_seconds: int | None = Field(None, ge=1)

	aggregation: str | None = None
	group_by: list[str] = Field(default_factory=list)

	max_results: int = Field(default=1000, ge=1, le=10000)
	include_metadata: bool = False

	timeout_seconds: int = Field(default=30, ge=1, le=300)
	cache_enabled: bool = True

	def validate_time_range(self) -> bool:
		assert self.end_time > self.start_time, "end_time must be after start_time"
		duration = (self.end_time - self.start_time).total_seconds()
		assert duration <= 86400 * 30, "Time range cannot exceed 30 days"
		return True

	def get_duration_seconds(self) -> int:
		return int((self.end_time - self.start_time).total_seconds())

	def generate_query_key(self) -> str:
		parts = [
			"|".join(sorted(self.metric_names)),
			json.dumps(self.labels, sort_keys=True),
			self.start_time.isoformat(),
			self.end_time.isoformat(),
			str(self.step_seconds or ""),
			self.aggregation or "",
			"|".join(sorted(self.group_by)),
		]
		return "|".join(parts)


class MonitoringTarget(BaseModel):
	"""Scrape target configuration with health tracking."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	target_id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(...)
	name: str = Field(..., max_length=255)
	type: str = Field(..., max_length=100)

	endpoint: str = Field(...)
	port: int | None = Field(None, ge=1, le=65535)
	path: str = Field(default="/metrics")
	scheme: str = Field(default="http")

	scrape_interval_seconds: int = Field(default=60, ge=5, le=3600)
	scrape_timeout_seconds: int = Field(default=10, ge=1, le=60)
	enabled: bool = True

	auth_type: str | None = None
	auth_config: dict[str, str] = Field(default_factory=dict)

	static_labels: dict[str, str] = Field(default_factory=dict)
	discovered_labels: dict[str, str] = Field(default_factory=dict)
	metadata: dict[str, Any] = Field(default_factory=dict)

	healthy: bool = True
	last_scrape: datetime | None = None
	scrape_failures: int = Field(default=0, ge=0)
	avg_scrape_duration_ms: float = Field(default=0.0, ge=0.0)

	capability_name: str | None = None
	auto_discovered: bool = False

	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)

	def is_healthy(self) -> bool:
		if not self.healthy or not self.enabled:
			return False
		if self.scrape_failures > 3:
			return False
		if self.last_scrape:
			stale = datetime.utcnow() - timedelta(seconds=self.scrape_interval_seconds * 3)
			if self.last_scrape < stale:
				return False
		return True

	def get_full_endpoint(self) -> str:
		port_part = f":{self.port}" if self.port else ""
		return f"{self.scheme}://{self.endpoint}{port_part}{self.path}"

	def update_scrape_stats(self, success: bool, duration_ms: float) -> None:
		assert duration_ms >= 0, "duration_ms must be non-negative"
		if success:
			self.scrape_failures = 0
			self.last_scrape = datetime.utcnow()
			self.healthy = True
		else:
			self.scrape_failures += 1
			if self.scrape_failures > 3:
				self.healthy = False
		self.avg_scrape_duration_ms = (self.avg_scrape_duration_ms * 0.9) + (duration_ms * 0.1)
		self.updated_at = datetime.utcnow()


# ─── SLO & Error Budget ────────────────────────────────────────────────────────

class SLOStatus(str, Enum):
	"""SLO lifecycle states."""
	ACTIVE = "active"
	BREACHED = "breached"
	PAUSED = "paused"
	RETIRED = "retired"


class SLO(BaseModel):
	"""Service Level Objective definition with error budget tracking."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	slo_id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(...)
	service_name: str = Field(..., max_length=255)
	name: str = Field(..., max_length=255)
	description: str = Field(default="", max_length=1000)

	objective_percent: float = Field(..., ge=0.0, le=100.0, description="e.g. 99.9")
	window_days: int = Field(default=30, ge=1, le=365)
	indicator_metric: str = Field(..., max_length=255)

	status: SLOStatus = Field(default=SLOStatus.ACTIVE)
	current_compliance: float = Field(default=100.0, ge=0.0, le=100.0)
	error_budget_remaining_percent: float = Field(default=100.0, ge=0.0, le=100.0)
	burn_rate: float = Field(default=0.0, ge=0.0)

	owner: str = Field(...)
	notification_route: str = Field(...)

	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str = Field(...)
	is_deleted: bool = False

	def is_breached(self) -> bool:
		return self.current_compliance < self.objective_percent

	def error_budget_minutes(self) -> float:
		"""Total allowed downtime minutes in the window."""
		window_minutes = self.window_days * 24 * 60
		allowed_error_fraction = (100.0 - self.objective_percent) / 100.0
		return window_minutes * allowed_error_fraction

	def remaining_budget_minutes(self) -> float:
		return self.error_budget_minutes() * (self.error_budget_remaining_percent / 100.0)


class HealthCheck(BaseModel):
	"""Health check probe configuration and last result."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	check_id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(...)
	name: str = Field(..., max_length=255)
	service_name: str = Field(..., max_length=255)
	endpoint: str = Field(...)
	method: str = Field(default="GET", max_length=10)
	expected_status: int = Field(default=200, ge=100, le=599)
	timeout_seconds: int = Field(default=5, ge=1, le=60)
	interval_seconds: int = Field(default=30, ge=5)

	healthy: bool = True
	last_checked: datetime | None = None
	last_response_ms: float | None = None
	consecutive_failures: int = Field(default=0, ge=0)

	labels: dict[str, str] = Field(default_factory=dict)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	is_deleted: bool = False


class TraceSpan(BaseModel):
	"""Distributed trace span for correlation and latency analysis."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	span_id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(...)
	trace_id: str = Field(...)
	parent_span_id: str | None = None

	service_name: str = Field(..., max_length=255)
	operation_name: str = Field(..., max_length=255)
	start_time: datetime = Field(...)
	end_time: datetime | None = None
	duration_ms: float | None = Field(None, ge=0.0)

	status: str = Field(default="ok", max_length=50)
	error: bool = False
	error_message: str | None = None

	tags: dict[str, str] = Field(default_factory=dict)
	logs: list[dict[str, Any]] = Field(default_factory=list)

	created_at: datetime = Field(default_factory=datetime.utcnow)
	is_deleted: bool = False

	def calculate_duration(self) -> float | None:
		if self.end_time and self.start_time:
			return (self.end_time - self.start_time).total_seconds() * 1000
		return None


class LogEntry(BaseModel):
	"""Structured log entry with PII status and ingestion governance."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	log_id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(...)
	source_id: str = Field(...)
	service_name: str = Field(..., max_length=255)

	level: str = Field(default="info", max_length=20)
	message: str = Field(..., max_length=8192)
	timestamp: datetime = Field(default_factory=datetime.utcnow)

	trace_id: str | None = None
	span_id: str | None = None
	labels: dict[str, str] = Field(default_factory=dict)

	contains_pii: bool = False
	pii_redacted: bool = True
	structured_data: dict[str, Any] = Field(default_factory=dict)

	created_at: datetime = Field(default_factory=datetime.utcnow)
	is_deleted: bool = False


class AnomalyDetection(BaseModel):
	"""Anomaly detection result for a metric series."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	anomaly_id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(...)
	metric_name: str = Field(..., max_length=255)
	source_id: str = Field(...)

	detected_at: datetime = Field(default_factory=datetime.utcnow)
	anomaly_score: float = Field(..., ge=0.0, le=1.0)
	sensitivity: float = Field(default=0.8, ge=0.0, le=1.0)
	algorithm: str = Field(default="z_score", max_length=64)

	observed_value: float = Field(...)
	expected_value: float = Field(...)
	baseline_mean: float = Field(...)
	baseline_std: float = Field(default=0.0, ge=0.0)

	is_true_positive: bool | None = None
	feedback_note: str | None = None
	labels: dict[str, str] = Field(default_factory=dict)

	created_at: datetime = Field(default_factory=datetime.utcnow)
	is_deleted: bool = False

	def z_score(self) -> float:
		"""Return z-score magnitude of this anomaly."""
		if self.baseline_std <= 0:
			return 0.0
		return abs((self.observed_value - self.baseline_mean) / self.baseline_std)


# ─── Exports ──────────────────────────────────────────────────────────────────

__all__ = [
	"uuid7str",
	"MetricType", "AlertSeverity", "AlertStatus", "AlertConditionType",
	"DashboardType", "DataRetentionPolicy", "MonitoringScope", "SLOStatus",
	"MonitoringMetric", "MonitoringAlert", "MonitoringRule", "MonitoringDashboard",
	"MonitoringQuery", "MonitoringTarget",
	"SLO", "HealthCheck", "TraceSpan", "LogEntry", "AnomalyDetection",
]
