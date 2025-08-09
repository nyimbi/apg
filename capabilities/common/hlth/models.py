#!/usr/bin/env python3
"""
APG System Health Management - Pydantic Data Models
Comprehensive health data models with intelligent validation and APG integration

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import statistics
from collections import defaultdict

from pydantic import BaseModel, Field, ConfigDict, AfterValidator, field_validator
from pydantic.functional_validators import AfterValidator
from uuid_extensions import uuid7str
from typing_extensions import Annotated


class HealthStatus(str, Enum):
	"""System health status levels"""
	HEALTHY = "healthy"
	WARNING = "warning"
	CRITICAL = "critical"
	UNKNOWN = "unknown"
	DEGRADED = "degraded"


class HealthSeverity(str, Enum):
	"""Health alert severity levels"""
	LOW = "low"
	MEDIUM = "medium"  
	HIGH = "high"
	CRITICAL = "critical"
	EMERGENCY = "emergency"


class AlertStatus(str, Enum):
	"""Health alert status tracking"""
	ACTIVE = "active"
	ACKNOWLEDGED = "acknowledged"
	RESOLVED = "resolved"
	SUPPRESSED = "suppressed"
	ESCALATED = "escalated"


class RemediationStatus(str, Enum):
	"""Automated remediation action status"""
	PENDING = "pending"
	IN_PROGRESS = "in_progress"
	COMPLETED = "completed"
	FAILED = "failed"
	ROLLED_BACK = "rolled_back"


class HealthDimension(str, Enum):
	"""Multi-dimensional health analysis categories"""
	PERFORMANCE = "performance"
	AVAILABILITY = "availability"
	SECURITY = "security"
	COMPLIANCE = "compliance"
	BUSINESS = "business"
	OPERATIONAL = "operational"


class ComponentType(str, Enum):
	"""System component types for health monitoring"""
	SERVICE = "service"
	DATABASE = "database"
	CACHE = "cache"
	QUEUE = "queue"
	API = "api"
	UI = "ui"
	WORKER = "worker"
	SCHEDULER = "scheduler"
	PROXY = "proxy"
	LOAD_BALANCER = "load_balancer"


class PredictionConfidence(str, Enum):
	"""ML prediction confidence levels"""
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	VERY_HIGH = "very_high"


def _validate_health_score(v: float) -> float:
	"""Validate health score is between 0.0 and 1.0"""
	assert 0.0 <= v <= 1.0, f"Health score must be between 0.0 and 1.0, got {v}"
	return v


def _validate_positive_number(v: Union[int, float]) -> Union[int, float]:
	"""Validate number is positive"""
	assert v >= 0, f"Value must be non-negative, got {v}"
	return v


def _validate_percentage(v: float) -> float:
	"""Validate percentage is between 0.0 and 100.0"""
	assert 0.0 <= v <= 100.0, f"Percentage must be between 0.0 and 100.0, got {v}"
	return v


def _validate_labels(v: Dict[str, str]) -> Dict[str, str]:
	"""Validate health metric labels"""
	assert isinstance(v, dict), "Labels must be a dictionary"
	for key, value in v.items():
		assert isinstance(key, str) and isinstance(value, str), "Labels must be string key-value pairs"
		assert len(key) > 0 and len(value) > 0, "Label keys and values cannot be empty"
	return v


class HealthMetric(BaseModel):
	"""System health metric with business context and ML features"""
	
	model_config = ConfigDict(
		extra='forbid', 
		validate_by_name=True, 
		validate_by_alias=True
	)
	
	# Identity and basic properties
	metric_id: str = Field(default_factory=uuid7str, description="Unique metric identifier")
	tenant_id: str = Field(..., description="APG tenant identifier for multi-tenancy")
	component_id: str = Field(..., description="System component identifier")
	component_type: ComponentType = Field(..., description="Type of monitored component")
	
	# Metric data
	name: str = Field(..., min_length=1, description="Health metric name")
	value: float = Field(..., description="Current metric value")
	unit: str = Field(default="", description="Metric unit of measurement")
	dimension: HealthDimension = Field(..., description="Health dimension category")
	
	# Context and metadata
	timestamp: datetime = Field(default_factory=datetime.utcnow, description="Metric timestamp")
	labels: Annotated[Dict[str, str], AfterValidator(_validate_labels)] = Field(
		default_factory=dict, description="Contextual labels for metric categorization"
	)
	source: str = Field(..., description="Metric data source system")
	source_type: str = Field(default="system", description="Type of metric source")
	
	# Quality and reliability
	quality_score: Annotated[float, AfterValidator(_validate_health_score)] = Field(
		default=1.0, description="Data quality score (0.0-1.0)"
	)
	confidence_level: PredictionConfidence = Field(
		default=PredictionConfidence.HIGH, description="Metric confidence level"
	)
	
	# Business impact
	business_impact: Annotated[float, AfterValidator(_validate_health_score)] = Field(
		default=0.5, description="Business impact weight (0.0-1.0)"
	)
	criticality_score: Annotated[float, AfterValidator(_validate_health_score)] = Field(
		default=0.5, description="Component criticality score (0.0-1.0)"
	)
	
	# APG integration
	capability_name: str = Field(default="hlth", description="Source APG capability")
	correlation_id: str = Field(default="", description="Event correlation identifier")
	
	# Processing metadata
	processed: bool = Field(default=False, description="Whether metric has been processed")
	processed_at: Optional[datetime] = Field(default=None, description="Processing timestamp")
	retention_policy: str = Field(default="30d", description="Data retention policy")
	
	def is_stale(self, max_age_minutes: int = 15) -> bool:
		"""Check if metric is stale based on age"""
		assert max_age_minutes > 0, "max_age_minutes must be positive"
		age = datetime.utcnow() - self.timestamp
		return age > timedelta(minutes=max_age_minutes)
	
	def get_contextual_priority(self) -> float:
		"""Calculate contextual priority based on business impact and criticality"""
		return (self.business_impact * 0.6) + (self.criticality_score * 0.4)
	
	def to_normalized_value(self, baseline_value: float, scale_factor: float = 1.0) -> float:
		"""Normalize metric value against baseline for comparison"""
		assert scale_factor > 0, "scale_factor must be positive"
		if baseline_value == 0:
			return self.value * scale_factor
		return (self.value / baseline_value) * scale_factor


class HealthAlert(BaseModel):
	"""Intelligent health alert with business impact and correlation"""
	
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	# Identity and basic properties  
	alert_id: str = Field(default_factory=uuid7str, description="Unique alert identifier")
	tenant_id: str = Field(..., description="APG tenant identifier")
	rule_id: str = Field(..., description="Health rule that triggered this alert")
	component_id: str = Field(..., description="Affected system component")
	
	# Alert content
	name: str = Field(..., min_length=1, description="Alert title")
	message: str = Field(..., min_length=1, description="Detailed alert message")
	summary: str = Field(default="", description="Brief alert summary")
	description: str = Field(default="", description="Extended alert description")
	
	# Severity and status
	severity: HealthSeverity = Field(..., description="Alert severity level")
	status: AlertStatus = Field(default=AlertStatus.ACTIVE, description="Current alert status")
	health_status: HealthStatus = Field(..., description="Overall health status")
	
	# Timing
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Alert creation time")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update time")
	resolved_at: Optional[datetime] = Field(default=None, description="Alert resolution time")
	acknowledged_at: Optional[datetime] = Field(default=None, description="Alert acknowledgment time")
	
	# Business impact
	business_impact: Annotated[float, AfterValidator(_validate_health_score)] = Field(
		default=0.5, description="Business impact score (0.0-1.0)"
	)
	user_impact_count: Annotated[int, AfterValidator(_validate_positive_number)] = Field(
		default=0, description="Estimated number of affected users"
	)
	revenue_impact: float = Field(default=0.0, description="Estimated revenue impact")
	
	# Source information
	source_metric: str = Field(..., description="Source health metric name")
	source_value: float = Field(..., description="Metric value that triggered alert")
	threshold_value: float = Field(..., description="Threshold that was exceeded")
	threshold_operator: str = Field(..., description="Comparison operator (>, <, =, etc.)")
	
	# Correlation and relationships
	correlation_key: str = Field(default="", description="Alert correlation identifier")
	parent_alert_id: str = Field(default="", description="Parent alert for cascaded alerts")
	related_alert_ids: List[str] = Field(default_factory=list, description="Related alert identifiers")
	
	# Escalation management
	escalation_level: Annotated[int, AfterValidator(_validate_positive_number)] = Field(
		default=0, description="Current escalation level"
	)
	escalation_count: Annotated[int, AfterValidator(_validate_positive_number)] = Field(
		default=0, description="Number of escalations"
	)
	max_escalation_level: Annotated[int, AfterValidator(_validate_positive_number)] = Field(
		default=3, description="Maximum escalation level"
	)
	
	# Response and remediation
	assigned_team: str = Field(default="", description="Assigned response team")
	assigned_user: str = Field(default="", description="Assigned user for resolution")
	runbook_url: str = Field(default="", description="Response runbook URL")
	remediation_actions: List[str] = Field(default_factory=list, description="Available remediation actions")
	
	# Performance tracking
	detection_time_ms: Annotated[float, AfterValidator(_validate_positive_number)] = Field(
		default=0.0, description="Alert detection time in milliseconds"
	)
	notification_time_ms: Annotated[float, AfterValidator(_validate_positive_number)] = Field(
		default=0.0, description="Alert notification time in milliseconds"
	)
	
	# APG integration
	audit_trail: List[Dict[str, Any]] = Field(default_factory=list, description="Alert audit trail")
	notification_channels: List[str] = Field(default_factory=list, description="Notification delivery channels")
	
	def is_active(self) -> bool:
		"""Check if alert is currently active"""
		return self.status == AlertStatus.ACTIVE
	
	def is_escalated(self) -> bool:
		"""Check if alert has been escalated"""
		return self.escalation_level > 0
	
	def can_escalate(self) -> bool:
		"""Check if alert can be escalated further"""
		return self.escalation_level < self.max_escalation_level and self.is_active()
	
	def get_age_minutes(self) -> float:
		"""Get alert age in minutes"""
		age = datetime.utcnow() - self.created_at
		return age.total_seconds() / 60.0
	
	def should_escalate(self, escalation_threshold_minutes: int = 30) -> bool:
		"""Determine if alert should be escalated based on age and current level"""
		assert escalation_threshold_minutes > 0, "escalation_threshold_minutes must be positive"
		if not self.can_escalate():
			return False
		return self.get_age_minutes() > (escalation_threshold_minutes * (self.escalation_level + 1))
	
	def calculate_priority_score(self) -> float:
		"""Calculate alert priority based on severity, business impact, and escalation"""
		severity_weights = {
			HealthSeverity.LOW: 0.2,
			HealthSeverity.MEDIUM: 0.4,
			HealthSeverity.HIGH: 0.7,
			HealthSeverity.CRITICAL: 0.9,
			HealthSeverity.EMERGENCY: 1.0
		}
		severity_score = severity_weights.get(self.severity, 0.5)
		escalation_multiplier = 1.0 + (self.escalation_level * 0.2)
		return (severity_score * 0.4 + self.business_impact * 0.6) * escalation_multiplier


class HealthBaseline(BaseModel):
	"""ML-learned normal operation patterns for health assessment"""
	
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	# Identity
	baseline_id: str = Field(default_factory=uuid7str, description="Unique baseline identifier")
	tenant_id: str = Field(..., description="APG tenant identifier")
	component_id: str = Field(..., description="System component identifier")
	metric_name: str = Field(..., description="Health metric name")
	
	# Statistical baseline data
	mean_value: float = Field(..., description="Statistical mean of baseline data")
	median_value: float = Field(..., description="Statistical median of baseline data")
	std_deviation: float = Field(..., description="Standard deviation of baseline data")
	min_value: float = Field(..., description="Minimum observed value")
	max_value: float = Field(..., description="Maximum observed value")
	percentile_95: float = Field(..., description="95th percentile value")
	percentile_99: float = Field(..., description="99th percentile value")
	
	# Baseline metadata
	sample_count: Annotated[int, AfterValidator(_validate_positive_number)] = Field(
		..., description="Number of samples in baseline"
	)
	confidence_score: Annotated[float, AfterValidator(_validate_health_score)] = Field(
		..., description="Baseline confidence score (0.0-1.0)"
	)
	quality_score: Annotated[float, AfterValidator(_validate_health_score)] = Field(
		default=1.0, description="Baseline data quality score (0.0-1.0)"
	)
	
	# Temporal patterns
	seasonal_patterns: Dict[str, float] = Field(
		default_factory=dict, description="Detected seasonal patterns (hour, day, week)"
	)
	trend_direction: str = Field(default="stable", description="Overall trend direction")
	trend_slope: float = Field(default=0.0, description="Trend slope coefficient")
	
	# Learning metadata
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Baseline creation time")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last baseline update")
	learning_period_days: Annotated[int, AfterValidator(_validate_positive_number)] = Field(
		default=7, description="Learning period in days"
	)
	last_learning_update: datetime = Field(default_factory=datetime.utcnow, description="Last learning update")
	
	# Validation and performance
	accuracy_score: Annotated[float, AfterValidator(_validate_health_score)] = Field(
		default=0.0, description="Baseline prediction accuracy (0.0-1.0)"
	)
	false_positive_rate: Annotated[float, AfterValidator(_validate_percentage)] = Field(
		default=5.0, description="False positive rate percentage"
	)
	
	# Adaptation parameters
	adaptation_rate: Annotated[float, AfterValidator(_validate_health_score)] = Field(
		default=0.1, description="Learning adaptation rate (0.0-1.0)"
	)
	stability_threshold: float = Field(default=0.2, description="Stability threshold for updates")
	
	def is_stale(self, max_age_hours: int = 24) -> bool:
		"""Check if baseline is stale and needs updating"""
		assert max_age_hours > 0, "max_age_hours must be positive"
		age = datetime.utcnow() - self.updated_at
		return age > timedelta(hours=max_age_hours)
	
	def calculate_anomaly_score(self, value: float) -> float:
		"""Calculate anomaly score for a given value against this baseline"""
		if self.std_deviation == 0:
			return 0.0
		z_score = abs(value - self.mean_value) / self.std_deviation
		return min(z_score / 3.0, 1.0)  # Normalize to 0-1 range
	
	def is_anomalous(self, value: float, threshold: float = 2.0) -> bool:
		"""Determine if value is anomalous compared to baseline"""
		assert threshold > 0, "threshold must be positive"
		if self.std_deviation == 0:
			return False
		z_score = abs(value - self.mean_value) / self.std_deviation
		return z_score > threshold
	
	def predict_expected_value(self, context: Dict[str, Any] = None) -> float:
		"""Predict expected value based on baseline and context"""
		base_prediction = self.mean_value
		
		# Apply seasonal adjustments if context provided
		if context and self.seasonal_patterns:
			hour = context.get('hour', 0)
			day_of_week = context.get('day_of_week', 0)
			
			hour_factor = self.seasonal_patterns.get(f'hour_{hour}', 1.0)
			day_factor = self.seasonal_patterns.get(f'day_{day_of_week}', 1.0)
			
			base_prediction *= hour_factor * day_factor
		
		return base_prediction


class HealthRule(BaseModel):
	"""Dynamic health assessment rule with ML optimization"""
	
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	# Identity
	rule_id: str = Field(default_factory=uuid7str, description="Unique rule identifier")
	tenant_id: str = Field(..., description="APG tenant identifier")
	name: str = Field(..., min_length=1, description="Health rule name")
	description: str = Field(default="", description="Rule description")
	
	# Rule configuration
	enabled: bool = Field(default=True, description="Whether rule is active")
	component_id: str = Field(..., description="Target component identifier")
	component_type: ComponentType = Field(..., description="Target component type")
	metric_name: str = Field(..., description="Target health metric name")
	dimension: HealthDimension = Field(..., description="Health dimension")
	
	# Condition definition
	condition_type: str = Field(..., description="Rule condition type (threshold, anomaly, trend)")
	condition_expression: str = Field(..., description="Rule condition expression")
	threshold_value: Optional[float] = Field(default=None, description="Threshold value for comparison")
	threshold_operator: str = Field(default="gt", description="Threshold comparison operator")
	
	# Alert configuration
	alert_severity: HealthSeverity = Field(..., description="Alert severity when rule triggers")
	alert_message_template: str = Field(..., description="Alert message template")
	alert_summary_template: str = Field(default="", description="Alert summary template")
	
	# Timing and evaluation
	evaluation_window_minutes: Annotated[int, AfterValidator(_validate_positive_number)] = Field(
		default=5, description="Evaluation window in minutes"
	)
	evaluation_interval_seconds: Annotated[int, AfterValidator(_validate_positive_number)] = Field(
		default=60, description="Evaluation interval in seconds"
	)
	cooldown_minutes: Annotated[int, AfterValidator(_validate_positive_number)] = Field(
		default=15, description="Cooldown period before re-triggering"
	)
	
	# ML optimization
	auto_tune_enabled: bool = Field(default=True, description="Enable ML-based rule optimization")
	baseline_learning_enabled: bool = Field(default=True, description="Enable baseline learning")
	anomaly_detection_enabled: bool = Field(default=False, description="Enable anomaly detection")
	anomaly_sensitivity: Annotated[float, AfterValidator(_validate_health_score)] = Field(
		default=0.8, description="Anomaly detection sensitivity (0.0-1.0)"
	)
	
	# Performance tracking
	trigger_count: Annotated[int, AfterValidator(_validate_positive_number)] = Field(
		default=0, description="Number of times rule has triggered"
	)
	false_positive_count: Annotated[int, AfterValidator(_validate_positive_number)] = Field(
		default=0, description="Number of false positive alerts"
	)
	effectiveness_score: Annotated[float, AfterValidator(_validate_health_score)] = Field(
		default=0.0, description="Rule effectiveness score (0.0-1.0)"
	)
	last_triggered: Optional[datetime] = Field(default=None, description="Last trigger timestamp")
	
	# Business impact
	business_impact_weight: Annotated[float, AfterValidator(_validate_health_score)] = Field(
		default=0.5, description="Business impact weight (0.0-1.0)"
	)
	sla_impact: bool = Field(default=False, description="Whether rule impacts SLA")
	escalation_enabled: bool = Field(default=True, description="Enable alert escalation")
	
	# Metadata
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Rule creation time")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last rule update")
	created_by: str = Field(default="system", description="Rule creator")
	tags: List[str] = Field(default_factory=list, description="Rule tags for organization")
	
	def is_due_for_evaluation(self) -> bool:
		"""Check if rule is due for evaluation based on interval"""
		if not self.enabled:
			return False
		if not self.last_triggered:
			return True
		time_since_trigger = datetime.utcnow() - self.last_triggered
		return time_since_trigger.total_seconds() >= self.evaluation_interval_seconds
	
	def is_in_cooldown(self) -> bool:
		"""Check if rule is in cooldown period"""
		if not self.last_triggered:
			return False
		time_since_trigger = datetime.utcnow() - self.last_triggered
		return time_since_trigger < timedelta(minutes=self.cooldown_minutes)
	
	def calculate_false_positive_rate(self) -> float:
		"""Calculate false positive rate percentage"""
		if self.trigger_count == 0:
			return 0.0
		return (self.false_positive_count / self.trigger_count) * 100.0
	
	def update_performance_stats(self, was_false_positive: bool = False) -> None:
		"""Update rule performance statistics"""
		self.trigger_count += 1
		if was_false_positive:
			self.false_positive_count += 1
		
		# Update effectiveness score (higher is better, lower false positive rate is better)
		if self.trigger_count > 0:
			fp_rate = self.calculate_false_positive_rate()
			self.effectiveness_score = max(0.0, (100.0 - fp_rate) / 100.0)
		
		self.last_triggered = datetime.utcnow()
		self.updated_at = datetime.utcnow()


class HealthAction(BaseModel):
	"""Automated and manual remediation action for health issues"""
	
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	# Identity
	action_id: str = Field(default_factory=uuid7str, description="Unique action identifier")
	tenant_id: str = Field(..., description="APG tenant identifier")
	alert_id: str = Field(..., description="Related alert identifier")
	component_id: str = Field(..., description="Target component identifier")
	
	# Action definition
	name: str = Field(..., min_length=1, description="Action name")
	description: str = Field(default="", description="Action description")
	action_type: str = Field(..., description="Type of remediation action")
	category: str = Field(default="remediation", description="Action category")
	
	# Execution details
	automated: bool = Field(default=False, description="Whether action is automated")
	status: RemediationStatus = Field(default=RemediationStatus.PENDING, description="Action status")
	command: str = Field(default="", description="Command or script to execute")
	parameters: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")
	
	# Timing
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Action creation time")
	started_at: Optional[datetime] = Field(default=None, description="Action start time")
	completed_at: Optional[datetime] = Field(default=None, description="Action completion time")
	timeout_seconds: Annotated[int, AfterValidator(_validate_positive_number)] = Field(
		default=300, description="Action timeout in seconds"
	)
	
	# Results and tracking
	exit_code: Optional[int] = Field(default=None, description="Action exit code")
	output: str = Field(default="", description="Action output")
	error_message: str = Field(default="", description="Error message if action failed")
	success: bool = Field(default=False, description="Whether action succeeded")
	
	# Risk and safety
	risk_level: str = Field(default="low", description="Risk level of action (low, medium, high)")
	requires_approval: bool = Field(default=False, description="Whether action requires manual approval")
	approved_by: str = Field(default="", description="User who approved action")
	approval_timestamp: Optional[datetime] = Field(default=None, description="Approval timestamp")
	rollback_action_id: str = Field(default="", description="Rollback action identifier")
	
	# Business impact
	estimated_downtime_seconds: Annotated[int, AfterValidator(_validate_positive_number)] = Field(
		default=0, description="Estimated downtime from action"
	)
	business_justification: str = Field(default="", description="Business justification for action")
	
	# Audit and tracking
	executed_by: str = Field(default="system", description="User or system that executed action")
	audit_trail: List[Dict[str, Any]] = Field(default_factory=list, description="Action audit trail")
	
	def is_expired(self) -> bool:
		"""Check if action has expired based on timeout"""
		if not self.started_at:
			return False
		elapsed = datetime.utcnow() - self.started_at
		return elapsed.total_seconds() > self.timeout_seconds
	
	def get_duration_seconds(self) -> float:
		"""Get action duration in seconds"""
		if not self.started_at:
			return 0.0
		end_time = self.completed_at or datetime.utcnow()
		return (end_time - self.started_at).total_seconds()
	
	def can_execute(self) -> bool:
		"""Check if action can be executed"""
		if self.status != RemediationStatus.PENDING:
			return False
		if self.requires_approval and not self.approved_by:
			return False
		return True
	
	def record_execution_start(self, executed_by: str = "system") -> None:
		"""Record action execution start"""
		self.status = RemediationStatus.IN_PROGRESS
		self.started_at = datetime.utcnow()
		self.executed_by = executed_by
		self.audit_trail.append({
			'event': 'execution_started',
			'timestamp': self.started_at.isoformat(),
			'executed_by': executed_by
		})
	
	def record_execution_result(self, success: bool, exit_code: int = 0, 
								output: str = "", error_message: str = "") -> None:
		"""Record action execution result"""
		self.completed_at = datetime.utcnow()
		self.success = success
		self.exit_code = exit_code
		self.output = output
		self.error_message = error_message
		self.status = RemediationStatus.COMPLETED if success else RemediationStatus.FAILED
		
		self.audit_trail.append({
			'event': 'execution_completed',
			'timestamp': self.completed_at.isoformat(),
			'success': success,
			'exit_code': exit_code,
			'duration_seconds': self.get_duration_seconds()
		})


class SystemComponent(BaseModel):
	"""Discoverable system component with dependencies and health tracking"""
	
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	# Identity
	component_id: str = Field(default_factory=uuid7str, description="Unique component identifier")
	tenant_id: str = Field(..., description="APG tenant identifier")
	name: str = Field(..., min_length=1, description="Component name")
	component_type: ComponentType = Field(..., description="Component type")
	
	# Location and connectivity
	host: str = Field(..., description="Host or server name")
	port: Optional[int] = Field(default=None, description="Service port")
	endpoint: str = Field(default="", description="Service endpoint URL")
	namespace: str = Field(default="default", description="Kubernetes namespace or logical grouping")
	
	# Health and status
	health_status: HealthStatus = Field(default=HealthStatus.UNKNOWN, description="Current health status")
	health_score: Annotated[float, AfterValidator(_validate_health_score)] = Field(
		default=0.0, description="Overall health score (0.0-1.0)"
	)
	last_health_check: Optional[datetime] = Field(default=None, description="Last health check timestamp")
	health_check_interval_seconds: Annotated[int, AfterValidator(_validate_positive_number)] = Field(
		default=60, description="Health check interval in seconds"
	)
	
	# Dependencies and relationships
	dependencies: List[str] = Field(default_factory=list, description="List of dependent component IDs")
	dependents: List[str] = Field(default_factory=list, description="List of components that depend on this")
	critical_dependencies: List[str] = Field(default_factory=list, description="Critical dependency component IDs")
	
	# Business context
	business_criticality: Annotated[float, AfterValidator(_validate_health_score)] = Field(
		default=0.5, description="Business criticality score (0.0-1.0)"
	)
	user_facing: bool = Field(default=False, description="Whether component is user-facing")
	sla_tier: str = Field(default="standard", description="SLA tier (basic, standard, premium)")
	
	# Performance baselines
	expected_response_time_ms: Annotated[float, AfterValidator(_validate_positive_number)] = Field(
		default=500.0, description="Expected response time in milliseconds"
	)
	expected_throughput: Annotated[float, AfterValidator(_validate_positive_number)] = Field(
		default=100.0, description="Expected throughput (requests/second)"
	)
	expected_availability: Annotated[float, AfterValidator(_validate_percentage)] = Field(
		default=99.9, description="Expected availability percentage"
	)
	
	# Discovery and metadata
	discovered_at: datetime = Field(default_factory=datetime.utcnow, description="Component discovery time")
	last_seen: datetime = Field(default_factory=datetime.utcnow, description="Last seen timestamp")
	discovery_method: str = Field(default="manual", description="Discovery method used")
	tags: List[str] = Field(default_factory=list, description="Component tags")
	labels: Dict[str, str] = Field(default_factory=dict, description="Component labels")
	
	# Version and configuration
	version: str = Field(default="", description="Component version")
	configuration_hash: str = Field(default="", description="Configuration hash for drift detection")
	last_configuration_update: Optional[datetime] = Field(default=None, description="Last config update")
	
	def is_healthy(self) -> bool:
		"""Check if component is healthy"""
		return self.health_status == HealthStatus.HEALTHY
	
	def is_critical(self) -> bool:
		"""Check if component is business critical"""
		return self.business_criticality >= 0.8
	
	def get_dependency_depth(self, visited: set = None) -> int:
		"""Calculate maximum dependency depth (for cascade analysis)"""
		if visited is None:
			visited = set()
		
		if self.component_id in visited:
			return 0  # Circular dependency protection
		
		visited.add(self.component_id)
		
		if not self.dependencies:
			return 0
		
		# This would need access to other components in a real implementation
		# For now, return based on direct dependency count
		return 1 + len(self.dependencies)
	
	def calculate_blast_radius(self) -> int:
		"""Calculate potential blast radius if this component fails"""
		# Simple calculation based on dependents - would be more sophisticated in practice
		direct_impact = len(self.dependents)
		
		# Weight by criticality
		if self.business_criticality >= 0.8:
			direct_impact *= 2
		elif self.business_criticality >= 0.5:
			direct_impact = int(direct_impact * 1.5)
		
		return direct_impact
	
	def needs_health_check(self) -> bool:
		"""Check if component needs health check based on interval"""
		if not self.last_health_check:
			return True
		
		elapsed = datetime.utcnow() - self.last_health_check
		return elapsed.total_seconds() >= self.health_check_interval_seconds


class HealthReport(BaseModel):
	"""Comprehensive health analysis report with trends and insights"""
	
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	# Identity
	report_id: str = Field(default_factory=uuid7str, description="Unique report identifier")
	tenant_id: str = Field(..., description="APG tenant identifier")
	name: str = Field(..., min_length=1, description="Report name")
	report_type: str = Field(..., description="Report type (executive, technical, compliance)")
	
	# Scope and timing
	scope: List[str] = Field(default_factory=list, description="Component IDs included in report")
	time_period_start: datetime = Field(..., description="Report period start time")
	time_period_end: datetime = Field(..., description="Report period end time")
	generated_at: datetime = Field(default_factory=datetime.utcnow, description="Report generation time")
	
	# Overall health summary
	overall_health_score: Annotated[float, AfterValidator(_validate_health_score)] = Field(
		..., description="Overall system health score (0.0-1.0)"
	)
	health_trend: str = Field(default="stable", description="Health trend (improving, stable, degrading)")
	total_components: Annotated[int, AfterValidator(_validate_positive_number)] = Field(
		..., description="Total number of monitored components"
	)
	healthy_components: Annotated[int, AfterValidator(_validate_positive_number)] = Field(
		..., description="Number of healthy components"
	)
	degraded_components: Annotated[int, AfterValidator(_validate_positive_number)] = Field(
		default=0, description="Number of degraded components"
	)
	critical_components: Annotated[int, AfterValidator(_validate_positive_number)] = Field(
		default=0, description="Number of critical/failed components"
	)
	
	# Alert and incident summary
	total_alerts: Annotated[int, AfterValidator(_validate_positive_number)] = Field(
		default=0, description="Total alerts in period"
	)
	critical_alerts: Annotated[int, AfterValidator(_validate_positive_number)] = Field(
		default=0, description="Critical alerts in period"
	)
	false_positive_alerts: Annotated[int, AfterValidator(_validate_positive_number)] = Field(
		default=0, description="False positive alerts"
	)
	average_resolution_time_minutes: Annotated[float, AfterValidator(_validate_positive_number)] = Field(
		default=0.0, description="Average alert resolution time in minutes"
	)
	
	# Performance metrics
	average_response_time_ms: Annotated[float, AfterValidator(_validate_positive_number)] = Field(
		default=0.0, description="Average system response time in milliseconds"
	)
	availability_percentage: Annotated[float, AfterValidator(_validate_percentage)] = Field(
		default=0.0, description="Overall system availability percentage"
	)
	throughput_requests_per_second: Annotated[float, AfterValidator(_validate_positive_number)] = Field(
		default=0.0, description="System throughput in requests per second"
	)
	
	# Dimensional health scores
	performance_health: Annotated[float, AfterValidator(_validate_health_score)] = Field(
		default=0.0, description="Performance dimension health score"
	)
	security_health: Annotated[float, AfterValidator(_validate_health_score)] = Field(
		default=0.0, description="Security dimension health score"
	)
	compliance_health: Annotated[float, AfterValidator(_validate_health_score)] = Field(
		default=0.0, description="Compliance dimension health score"
	)
	business_health: Annotated[float, AfterValidator(_validate_health_score)] = Field(
		default=0.0, description="Business dimension health score"
	)
	
	# Insights and recommendations
	key_insights: List[str] = Field(default_factory=list, description="Key health insights")
	recommendations: List[str] = Field(default_factory=list, description="Health improvement recommendations")
	risk_areas: List[str] = Field(default_factory=list, description="Identified risk areas")
	improvement_opportunities: List[str] = Field(default_factory=list, description="Improvement opportunities")
	
	# Business impact
	estimated_business_impact: float = Field(default=0.0, description="Estimated business impact of health issues")
	user_impact_count: Annotated[int, AfterValidator(_validate_positive_number)] = Field(
		default=0, description="Estimated number of users impacted"
	)
	sla_compliance: Annotated[float, AfterValidator(_validate_percentage)] = Field(
		default=100.0, description="SLA compliance percentage"
	)
	
	# Trend analysis
	health_score_trend: List[Dict[str, Any]] = Field(
		default_factory=list, description="Health score trend data points"
	)
	alert_volume_trend: List[Dict[str, Any]] = Field(
		default_factory=list, description="Alert volume trend data points"
	)
	performance_trend: List[Dict[str, Any]] = Field(
		default_factory=list, description="Performance trend data points"
	)
	
	# Report metadata
	requested_by: str = Field(default="system", description="User who requested the report")
	report_format: str = Field(default="json", description="Report output format")
	confidentiality: str = Field(default="internal", description="Report confidentiality level")
	
	def get_health_grade(self) -> str:
		"""Get letter grade for overall health score"""
		if self.overall_health_score >= 0.95:
			return "A+"
		elif self.overall_health_score >= 0.9:
			return "A"
		elif self.overall_health_score >= 0.85:
			return "B+"
		elif self.overall_health_score >= 0.8:
			return "B"
		elif self.overall_health_score >= 0.7:
			return "C"
		else:
			return "D"
	
	def get_availability_sla_status(self, sla_target: float = 99.9) -> str:
		"""Get SLA status compared to target"""
		if self.availability_percentage >= sla_target:
			return "MEETING_SLA"
		elif self.availability_percentage >= (sla_target - 0.1):
			return "AT_RISK"
		else:
			return "BELOW_SLA"
	
	def calculate_health_trend_velocity(self) -> float:
		"""Calculate rate of health score change"""
		if len(self.health_score_trend) < 2:
			return 0.0
		
		scores = [point.get('score', 0.0) for point in self.health_score_trend]
		if len(scores) < 2:
			return 0.0
		
		# Simple linear trend calculation
		return (scores[-1] - scores[0]) / len(scores)


# Export all models for easy imports
__all__ = [
	'HealthStatus', 'HealthSeverity', 'AlertStatus', 'RemediationStatus', 
	'HealthDimension', 'ComponentType', 'PredictionConfidence',
	'HealthMetric', 'HealthAlert', 'HealthBaseline', 'HealthRule', 
	'HealthAction', 'SystemComponent', 'HealthReport'
]