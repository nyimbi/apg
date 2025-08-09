#!/usr/bin/env python3
"""
APG Monitoring and Observability (MONI) - Data Models
Pydantic v2 models following APG coding standards

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

from pydantic import BaseModel, Field, ConfigDict, AfterValidator
from typing import Dict, List, Any, Optional, Union, Annotated
from datetime import datetime, timedelta
from enum import Enum
from uuid_extensions import uuid7str
import json
from decimal import Decimal


class MetricType(str, Enum):
	"""Types of monitoring metrics"""
	COUNTER = "counter"
	GAUGE = "gauge"
	HISTOGRAM = "histogram"
	SUMMARY = "summary"
	SET = "set"


class AlertSeverity(str, Enum):
	"""Alert severity levels"""
	CRITICAL = "critical"
	HIGH = "high"
	MEDIUM = "medium"
	LOW = "low"
	INFO = "info"


class AlertStatus(str, Enum):
	"""Alert status states"""
	ACTIVE = "active"
	ACKNOWLEDGED = "acknowledged"
	RESOLVED = "resolved"
	SUPPRESSED = "suppressed"


class AlertConditionType(str, Enum):
	"""Alert condition types"""
	THRESHOLD = "threshold"
	ANOMALY = "anomaly"
	RATE = "rate"
	ABSENCE = "absence"
	COMPOSITE = "composite"


class DashboardType(str, Enum):
	"""Dashboard types for different use cases"""
	EXECUTIVE = "executive"
	OPERATIONAL = "operational"
	DEVELOPER = "developer"
	TENANT = "tenant"
	CUSTOM = "custom"


class DataRetentionPolicy(str, Enum):
	"""Data retention policies"""
	REAL_TIME = "real_time"		# 1 hour
	SHORT_TERM = "short_term"	# 24 hours
	MEDIUM_TERM = "medium_term"	# 7 days
	LONG_TERM = "long_term"		# 30 days
	ARCHIVE = "archive"			# 1 year


class MonitoringScope(str, Enum):
	"""Monitoring scope for rules and dashboards"""
	GLOBAL = "global"
	TENANT = "tenant"
	APPLICATION = "application"
	SERVICE = "service"
	INFRASTRUCTURE = "infrastructure"


def _validate_positive_number(value: Union[int, float, Decimal]) -> Union[int, float, Decimal]:
	"""Validate that number is positive"""
	assert value > 0, "Value must be positive"
	return value


def _validate_labels(labels: Dict[str, str]) -> Dict[str, str]:
	"""Validate metric labels"""
	assert isinstance(labels, dict), "Labels must be a dictionary"
	for key, value in labels.items():
		assert isinstance(key, str) and isinstance(value, str), "Label keys and values must be strings"
		assert len(key) <= 255, "Label key too long (max 255 chars)"
		assert len(value) <= 1024, "Label value too long (max 1024 chars)"
	return labels


def _validate_alert_condition(condition: str) -> str:
	"""Validate alert condition expression"""
	assert condition.strip(), "Alert condition cannot be empty"
	assert len(condition) <= 2048, "Alert condition too long (max 2048 chars)"
	return condition.strip()


class MonitoringMetric(BaseModel):
	"""
	Core monitoring metric model for time-series data collection
	Supports high-cardinality metrics with efficient storage
	"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	# Core identification
	metric_id: str = Field(default_factory=uuid7str, description="Unique metric identifier")
	tenant_id: str = Field(..., description="APG tenant identifier")
	name: str = Field(..., description="Metric name", max_length=255)
	
	# Metric data
	value: Annotated[float, AfterValidator(_validate_positive_number)] = Field(
		..., description="Metric value"
	)
	metric_type: MetricType = Field(default=MetricType.GAUGE, description="Type of metric")
	unit: Optional[str] = Field(None, description="Unit of measurement", max_length=50)
	
	# Temporal data
	timestamp: datetime = Field(default_factory=datetime.utcnow, description="Metric timestamp")
	interval_seconds: Optional[int] = Field(None, description="Collection interval in seconds", ge=1)
	
	# Labels and metadata
	labels: Annotated[Dict[str, str], AfterValidator(_validate_labels)] = Field(
		default_factory=dict, description="Metric labels for filtering and grouping"
	)
	source: str = Field(..., description="Source system or component", max_length=255)
	source_type: str = Field(default="unknown", description="Type of source", max_length=100)
	
	# Data quality and processing
	quality_score: float = Field(default=1.0, description="Data quality score", ge=0.0, le=1.0)
	processed: bool = Field(default=False, description="Whether metric has been processed")
	retention_policy: DataRetentionPolicy = Field(
		default=DataRetentionPolicy.MEDIUM_TERM, 
		description="Data retention policy"
	)
	
	# APG integration
	capability_name: Optional[str] = Field(None, description="Source APG capability", max_length=100)
	correlation_id: Optional[str] = Field(None, description="Request correlation ID")
	
	# Performance tracking
	ingestion_latency_ms: Optional[float] = Field(None, description="Ingestion latency", ge=0.0)
	processing_time_ms: Optional[float] = Field(None, description="Processing time", ge=0.0)

	def is_stale(self, max_age_seconds: int = 300) -> bool:
		"""Check if metric is stale based on timestamp"""
		assert max_age_seconds > 0, "Max age must be positive"
		age = (datetime.utcnow() - self.timestamp).total_seconds()
		return age > max_age_seconds

	def get_label_signature(self) -> str:
		"""Get consistent label signature for grouping"""
		return "|".join(f"{k}={v}" for k, v in sorted(self.labels.items()))

	def to_prometheus_format(self) -> str:
		"""Convert to Prometheus exposition format"""
		labels_str = ",".join(f'{k}="{v}"' for k, v in self.labels.items())
		return f'{self.name}{{{labels_str}}} {self.value} {int(self.timestamp.timestamp() * 1000)}'


class MonitoringAlert(BaseModel):
	"""
	Intelligent alert model with correlation and context
	Supports smart grouping and escalation management
	"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	# Core identification
	alert_id: str = Field(default_factory=uuid7str, description="Unique alert identifier")
	tenant_id: str = Field(..., description="APG tenant identifier")
	rule_id: str = Field(..., description="ID of the rule that triggered this alert")
	
	# Alert details
	name: str = Field(..., description="Alert name", max_length=255)
	description: str = Field(default="", description="Alert description", max_length=1000)
	severity: AlertSeverity = Field(default=AlertSeverity.MEDIUM, description="Alert severity")
	status: AlertStatus = Field(default=AlertStatus.ACTIVE, description="Alert status")
	
	# Alert data
	message: str = Field(..., description="Alert message", max_length=2048)
	summary: str = Field(default="", description="Brief alert summary", max_length=500)
	runbook_url: Optional[str] = Field(None, description="Link to runbook or documentation")
	
	# Temporal tracking
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Alert creation time")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update time")
	resolved_at: Optional[datetime] = Field(None, description="Resolution time")
	acknowledged_at: Optional[datetime] = Field(None, description="Acknowledgment time")
	
	# Correlation and grouping
	correlation_key: Optional[str] = Field(None, description="Key for alert correlation")
	parent_alert_id: Optional[str] = Field(None, description="Parent alert for grouping")
	related_alert_ids: List[str] = Field(default_factory=list, description="Related alert IDs")
	
	# Context and metadata
	labels: Dict[str, str] = Field(default_factory=dict, description="Alert labels")
	annotations: Dict[str, str] = Field(default_factory=dict, description="Additional annotations")
	source_metric: Optional[str] = Field(None, description="Source metric name")
	source_value: Optional[float] = Field(None, description="Metric value that triggered alert")
	threshold_value: Optional[float] = Field(None, description="Threshold that was exceeded")
	
	# Escalation and routing
	assigned_to: Optional[str] = Field(None, description="Assigned user or team")
	escalation_level: int = Field(default=0, description="Current escalation level", ge=0)
	max_escalation_level: int = Field(default=3, description="Maximum escalation level", ge=1)
	escalation_interval_minutes: int = Field(default=30, description="Escalation interval", ge=1)
	
	# Business impact
	impact_score: float = Field(default=0.0, description="Business impact score", ge=0.0, le=1.0)
	affected_services: List[str] = Field(default_factory=list, description="Affected services")
	affected_users_count: int = Field(default=0, description="Estimated affected users", ge=0)
	
	# APG integration
	notification_sent: bool = Field(default=False, description="Whether notification was sent")
	audit_logged: bool = Field(default=False, description="Whether audit was logged")
	
	def is_active(self) -> bool:
		"""Check if alert is currently active"""
		return self.status == AlertStatus.ACTIVE

	def is_escalated(self) -> bool:
		"""Check if alert has been escalated"""
		return self.escalation_level > 0

	def can_escalate(self) -> bool:
		"""Check if alert can be escalated further"""
		return self.escalation_level < self.max_escalation_level

	def get_age_minutes(self) -> float:
		"""Get alert age in minutes"""
		return (datetime.utcnow() - self.created_at).total_seconds() / 60

	def should_escalate(self) -> bool:
		"""Check if alert should be escalated based on age"""
		if not self.can_escalate() or not self.is_active():
			return False
		age_minutes = self.get_age_minutes()
		return age_minutes >= (self.escalation_level + 1) * self.escalation_interval_minutes


class MonitoringRule(BaseModel):
	"""
	Flexible alert rule configuration with intelligent conditions
	Supports complex expressions and ML-based anomaly detection
	"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	# Core identification
	rule_id: str = Field(default_factory=uuid7str, description="Unique rule identifier")
	tenant_id: str = Field(..., description="APG tenant identifier")
	name: str = Field(..., description="Rule name", max_length=255)
	description: str = Field(default="", description="Rule description", max_length=1000)
	
	# Rule configuration
	enabled: bool = Field(default=True, description="Whether rule is enabled")
	condition: Annotated[str, AfterValidator(_validate_alert_condition)] = Field(
		..., description="Alert condition expression"
	)
	condition_type: AlertConditionType = Field(default=AlertConditionType.THRESHOLD)
	
	# Targeting
	metric_name: str = Field(..., description="Target metric name", max_length=255)
	metric_labels: Dict[str, str] = Field(default_factory=dict, description="Metric label filters")
	scope: MonitoringScope = Field(default=MonitoringScope.TENANT, description="Rule scope")
	
	# Thresholds and parameters
	threshold_value: Optional[float] = Field(None, description="Threshold value for comparison")
	threshold_operator: str = Field(default="gt", description="Comparison operator (gt, lt, eq, etc.)")
	evaluation_window_minutes: int = Field(default=5, description="Evaluation window", ge=1, le=1440)
	evaluation_interval_seconds: int = Field(default=60, description="Evaluation interval", ge=10)
	
	# Alert configuration  
	severity: AlertSeverity = Field(default=AlertSeverity.MEDIUM, description="Alert severity")
	alert_message: str = Field(..., description="Alert message template", max_length=2048)
	alert_summary: str = Field(default="", description="Alert summary template", max_length=500)
	runbook_url: Optional[str] = Field(None, description="Runbook URL")
	
	# Escalation settings
	escalation_enabled: bool = Field(default=True, description="Enable alert escalation")
	escalation_interval_minutes: int = Field(default=30, description="Escalation interval", ge=1)
	max_escalation_level: int = Field(default=3, description="Max escalation level", ge=1)
	
	# Suppression and correlation
	suppression_enabled: bool = Field(default=False, description="Enable alert suppression")
	suppression_window_minutes: int = Field(default=60, description="Suppression window", ge=1)
	correlation_key: Optional[str] = Field(None, description="Alert correlation key")
	
	# ML and anomaly detection
	anomaly_detection_enabled: bool = Field(default=False, description="Enable anomaly detection")
	anomaly_sensitivity: float = Field(default=0.8, description="Anomaly sensitivity", ge=0.0, le=1.0)
	baseline_period_days: int = Field(default=7, description="Baseline period for anomaly detection", ge=1)
	
	# Metadata and tracking
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Rule creation time")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update time")
	created_by: str = Field(..., description="User who created the rule")
	last_triggered: Optional[datetime] = Field(None, description="Last trigger time")
	trigger_count: int = Field(default=0, description="Total trigger count", ge=0)
	
	# Performance metrics
	evaluation_time_ms: float = Field(default=0.0, description="Average evaluation time", ge=0.0)
	false_positive_rate: float = Field(default=0.0, description="False positive rate", ge=0.0, le=1.0)
	effectiveness_score: float = Field(default=0.0, description="Rule effectiveness", ge=0.0, le=1.0)

	def is_due_for_evaluation(self) -> bool:
		"""Check if rule is due for evaluation"""
		if not self.enabled:
			return False
		if not self.last_triggered:
			return True
		next_eval = self.last_triggered + timedelta(seconds=self.evaluation_interval_seconds)
		return datetime.utcnow() >= next_eval

	def get_evaluation_query(self) -> str:
		"""Generate query for metric evaluation"""
		labels_filter = " AND ".join(f'{k}="{v}"' for k, v in self.metric_labels.items())
		window = f"{self.evaluation_window_minutes}m"
		return f"SELECT {self.metric_name} WHERE {labels_filter} TIMEFRAME {window}"

	def update_performance_stats(self, evaluation_time_ms: float, is_false_positive: bool = False) -> None:
		"""Update rule performance statistics"""
		assert evaluation_time_ms >= 0, "Evaluation time must be non-negative"
		
		# Update rolling average of evaluation time
		self.evaluation_time_ms = (self.evaluation_time_ms * 0.9) + (evaluation_time_ms * 0.1)
		
		# Update false positive rate
		if is_false_positive:
			self.false_positive_rate = (self.false_positive_rate * 0.95) + 0.05
		else:
			self.false_positive_rate = self.false_positive_rate * 0.95
		
		# Update effectiveness score (inverse of false positive rate)
		self.effectiveness_score = 1.0 - self.false_positive_rate
		
		self.updated_at = datetime.utcnow()


class MonitoringDashboard(BaseModel):
	"""
	Intelligent dashboard configuration with adaptive layouts
	Supports multiple dashboard types and real-time updates
	"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	# Core identification
	dashboard_id: str = Field(default_factory=uuid7str, description="Unique dashboard identifier")
	tenant_id: str = Field(..., description="APG tenant identifier")
	name: str = Field(..., description="Dashboard name", max_length=255)
	description: str = Field(default="", description="Dashboard description", max_length=1000)
	
	# Dashboard configuration
	dashboard_type: DashboardType = Field(default=DashboardType.OPERATIONAL)
	scope: MonitoringScope = Field(default=MonitoringScope.TENANT)
	auto_refresh: bool = Field(default=True, description="Enable auto-refresh")
	refresh_interval_seconds: int = Field(default=30, description="Refresh interval", ge=5, le=3600)
	
	# Layout and widgets
	layout: Dict[str, Any] = Field(default_factory=dict, description="Dashboard layout configuration")
	widgets: List[Dict[str, Any]] = Field(default_factory=list, description="Dashboard widgets")
	widget_count: int = Field(default=0, description="Number of widgets", ge=0)
	
	# Access control
	public: bool = Field(default=False, description="Whether dashboard is public")
	shared_with: List[str] = Field(default_factory=list, description="Users/teams with access")
	view_permissions: List[str] = Field(default_factory=list, description="View permission roles")
	edit_permissions: List[str] = Field(default_factory=list, description="Edit permission roles")
	
	# Metadata and tracking
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Dashboard creation time")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update time")
	created_by: str = Field(..., description="User who created the dashboard")
	last_viewed: Optional[datetime] = Field(None, description="Last view time")
	view_count: int = Field(default=0, description="Total view count", ge=0)
	
	# Performance optimization
	cached: bool = Field(default=False, description="Whether dashboard data is cached")
	cache_ttl_seconds: int = Field(default=300, description="Cache TTL", ge=1)
	preload_data: bool = Field(default=False, description="Preload dashboard data")
	
	# Analytics and insights
	avg_load_time_ms: float = Field(default=0.0, description="Average load time", ge=0.0)
	user_engagement_score: float = Field(default=0.0, description="User engagement score", ge=0.0, le=1.0)
	popularity_score: float = Field(default=0.0, description="Dashboard popularity", ge=0.0, le=1.0)

	def add_widget(self, widget_config: Dict[str, Any]) -> None:
		"""Add widget to dashboard"""
		assert isinstance(widget_config, dict), "Widget config must be a dictionary"
		assert "type" in widget_config, "Widget must have a type"
		
		widget_config["widget_id"] = uuid7str()
		self.widgets.append(widget_config)
		self.widget_count = len(self.widgets)
		self.updated_at = datetime.utcnow()

	def remove_widget(self, widget_id: str) -> bool:
		"""Remove widget from dashboard"""
		initial_count = len(self.widgets)
		self.widgets = [w for w in self.widgets if w.get("widget_id") != widget_id]
		removed = len(self.widgets) < initial_count
		
		if removed:
			self.widget_count = len(self.widgets)
			self.updated_at = datetime.utcnow()
		
		return removed

	def update_view_stats(self, load_time_ms: float) -> None:
		"""Update dashboard view statistics"""
		assert load_time_ms >= 0, "Load time must be non-negative"
		
		self.view_count += 1
		self.last_viewed = datetime.utcnow()
		
		# Update rolling average of load time
		self.avg_load_time_ms = (self.avg_load_time_ms * 0.9) + (load_time_ms * 0.1)
		
		# Update engagement score based on view frequency
		days_since_created = (datetime.utcnow() - self.created_at).days
		if days_since_created > 0:
			views_per_day = self.view_count / max(days_since_created, 1)
			self.user_engagement_score = min(views_per_day / 10.0, 1.0)


class MonitoringQuery(BaseModel):
	"""
	Flexible query model for metrics retrieval and analysis
	Supports complex filtering, aggregation, and time-based operations
	"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	# Query identification
	query_id: str = Field(default_factory=uuid7str, description="Unique query identifier")
	tenant_id: str = Field(..., description="APG tenant identifier")
	
	# Metric selection
	metric_names: List[str] = Field(..., description="Target metric names")
	labels: Dict[str, Union[str, List[str]]] = Field(
		default_factory=dict, description="Label filters"
	)
	
	# Time range
	start_time: datetime = Field(..., description="Query start time")
	end_time: datetime = Field(..., description="Query end time")
	step_seconds: Optional[int] = Field(None, description="Step size for range queries", ge=1)
	
	# Aggregation
	aggregation: Optional[str] = Field(None, description="Aggregation function (sum, avg, max, min)")
	group_by: List[str] = Field(default_factory=list, description="Group by labels")
	
	# Query options
	max_results: int = Field(default=1000, description="Maximum results to return", ge=1, le=10000)
	include_metadata: bool = Field(default=False, description="Include metric metadata")
	
	# Performance
	timeout_seconds: int = Field(default=30, description="Query timeout", ge=1, le=300)
	cache_enabled: bool = Field(default=True, description="Enable query caching")

	def validate_time_range(self) -> bool:
		"""Validate that time range is sensible"""
		assert self.end_time > self.start_time, "End time must be after start time"
		
		duration = (self.end_time - self.start_time).total_seconds()
		assert duration <= 86400 * 30, "Time range cannot exceed 30 days"  # 30 days max
		
		return True

	def get_duration_seconds(self) -> int:
		"""Get query duration in seconds"""
		return int((self.end_time - self.start_time).total_seconds())

	def generate_query_key(self) -> str:
		"""Generate cache key for the query"""
		key_parts = [
			"|".join(sorted(self.metric_names)),
			json.dumps(self.labels, sort_keys=True),
			self.start_time.isoformat(),
			self.end_time.isoformat(),
			str(self.step_seconds or ""),
			self.aggregation or "",
			"|".join(sorted(self.group_by))
		]
		return "|".join(key_parts)


class MonitoringTarget(BaseModel):
	"""
	Monitoring target configuration for services, hosts, and applications
	Supports auto-discovery and intelligent configuration
	"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	# Core identification
	target_id: str = Field(default_factory=uuid7str, description="Unique target identifier")
	tenant_id: str = Field(..., description="APG tenant identifier")
	name: str = Field(..., description="Target name", max_length=255)
	type: str = Field(..., description="Target type (service, host, application)", max_length=100)
	
	# Target configuration
	endpoint: str = Field(..., description="Monitoring endpoint URL")
	port: Optional[int] = Field(None, description="Target port", ge=1, le=65535)
	path: str = Field(default="/metrics", description="Metrics path")
	scheme: str = Field(default="http", description="URL scheme")
	
	# Collection settings
	scrape_interval_seconds: int = Field(default=60, description="Scrape interval", ge=5, le=3600)
	scrape_timeout_seconds: int = Field(default=10, description="Scrape timeout", ge=1, le=60)
	enabled: bool = Field(default=True, description="Whether target is enabled")
	
	# Authentication
	auth_type: Optional[str] = Field(None, description="Authentication type")
	auth_config: Dict[str, str] = Field(default_factory=dict, description="Authentication config")
	
	# Labels and metadata
	static_labels: Dict[str, str] = Field(default_factory=dict, description="Static labels")
	discovered_labels: Dict[str, str] = Field(default_factory=dict, description="Auto-discovered labels")
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
	
	# Health and performance
	healthy: bool = Field(default=True, description="Target health status")
	last_scrape: Optional[datetime] = Field(None, description="Last successful scrape")
	scrape_failures: int = Field(default=0, description="Consecutive scrape failures", ge=0)
	avg_scrape_duration_ms: float = Field(default=0.0, description="Average scrape duration", ge=0.0)
	
	# APG integration
	capability_name: Optional[str] = Field(None, description="Source APG capability")
	auto_discovered: bool = Field(default=False, description="Whether target was auto-discovered")
	
	# Timestamps
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")

	def is_healthy(self) -> bool:
		"""Check if target is healthy"""
		if not self.healthy or not self.enabled:
			return False
		
		# Consider target unhealthy if too many consecutive failures
		if self.scrape_failures > 3:
			return False
		
		# Consider target stale if no recent scrapes
		if self.last_scrape:
			stale_threshold = datetime.utcnow() - timedelta(
				seconds=self.scrape_interval_seconds * 3
			)
			if self.last_scrape < stale_threshold:
				return False
		
		return True

	def get_full_endpoint(self) -> str:
		"""Get complete endpoint URL"""
		port_part = f":{self.port}" if self.port else ""
		return f"{self.scheme}://{self.endpoint}{port_part}{self.path}"

	def update_scrape_stats(self, success: bool, duration_ms: float) -> None:
		"""Update scrape statistics"""
		assert duration_ms >= 0, "Duration must be non-negative"
		
		if success:
			self.scrape_failures = 0
			self.last_scrape = datetime.utcnow()
			self.healthy = True
		else:
			self.scrape_failures += 1
			if self.scrape_failures > 3:
				self.healthy = False
		
		# Update rolling average of scrape duration
		self.avg_scrape_duration_ms = (self.avg_scrape_duration_ms * 0.9) + (duration_ms * 0.1)
		self.updated_at = datetime.utcnow()


# Export all models
__all__ = [
	'MetricType', 'AlertSeverity', 'AlertStatus', 'AlertConditionType',
	'DashboardType', 'DataRetentionPolicy', 'MonitoringScope',
	'MonitoringMetric', 'MonitoringAlert', 'MonitoringRule',
	'MonitoringDashboard', 'MonitoringQuery', 'MonitoringTarget'
]