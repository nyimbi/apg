"""
APG Behavioral Analytics - Pydantic Models

Enterprise behavioral analytics models with advanced statistical modeling,
anomaly detection, and predictive behavioral analysis capabilities.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator, root_validator
from pydantic import ConfigDict
from uuid_extensions import uuid7str


class EntityType(str, Enum):
	USER = "user"
	SYSTEM = "system"
	SERVICE = "service"
	DEVICE = "device"
	APPLICATION = "application"


class BehaviorType(str, Enum):
	ACCESS_PATTERN = "access_pattern"
	LOGIN_BEHAVIOR = "login_behavior"
	DATA_ACCESS = "data_access"
	COMMUNICATION = "communication"
	SYSTEM_USAGE = "system_usage"
	NETWORK_ACTIVITY = "network_activity"


class AnomalyType(str, Enum):
	STATISTICAL = "statistical"
	TEMPORAL = "temporal"
	VOLUMETRIC = "volumetric"
	PATTERN = "pattern"
	PEER_DEVIATION = "peer_deviation"
	CONTEXTUAL = "contextual"


class RiskLevel(str, Enum):
	CRITICAL = "critical"
	HIGH = "high"
	MEDIUM = "medium"
	LOW = "low"
	MINIMAL = "minimal"


class BaselineStatus(str, Enum):
	ESTABLISHING = "establishing"
	ESTABLISHED = "established"
	UPDATING = "updating"
	EXPIRED = "expired"


class BehavioralProfile(BaseModel):
	"""Comprehensive behavioral profile for users and entities"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	entity_id: str = Field(description="Entity being profiled")
	entity_type: EntityType
	entity_name: str = Field(description="Display name of entity")
	
	department: Optional[str] = None
	role: Optional[str] = None
	location: Optional[str] = None
	
	profile_period_start: datetime = Field(description="Profile period start")
	profile_period_end: datetime = Field(description="Profile period end")
	
	baseline_status: BaselineStatus = BaselineStatus.ESTABLISHING
	baseline_confidence: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	# Temporal patterns
	normal_hours: List[int] = Field(default_factory=list, description="Normal working hours")
	normal_days: List[int] = Field(default_factory=list, description="Normal working days")
	timezone: str = Field(default="UTC")
	
	# Access patterns
	typical_systems: List[str] = Field(default_factory=list)
	typical_applications: List[str] = Field(default_factory=list)
	typical_resources: List[str] = Field(default_factory=list)
	
	# Data access patterns
	data_sensitivity_levels: List[str] = Field(default_factory=list)
	typical_data_volumes: Dict[str, Decimal] = Field(default_factory=dict)
	access_frequencies: Dict[str, int] = Field(default_factory=dict)
	
	# Communication patterns
	communication_frequency: Dict[str, int] = Field(default_factory=dict)
	typical_contacts: List[str] = Field(default_factory=list)
	communication_channels: List[str] = Field(default_factory=list)
	
	# Geographic patterns
	normal_locations: List[Dict[str, Any]] = Field(default_factory=list)
	travel_patterns: Dict[str, Any] = Field(default_factory=dict)
	
	# Device and system patterns
	typical_devices: List[str] = Field(default_factory=list)
	operating_systems: List[str] = Field(default_factory=list)
	browser_patterns: List[str] = Field(default_factory=list)
	
	# Behavioral metrics
	activity_volume: Dict[str, Decimal] = Field(default_factory=dict)
	session_durations: Dict[str, Decimal] = Field(default_factory=dict)
	
	# Statistical baselines
	statistical_baselines: Dict[str, Dict[str, Decimal]] = Field(default_factory=dict)
	
	peer_group_id: Optional[str] = None
	peer_comparison_score: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	last_updated: datetime = Field(default_factory=datetime.utcnow)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class BehavioralBaseline(BaseModel):
	"""Statistical baseline for behavioral analysis"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	profile_id: str = Field(description="Associated behavioral profile")
	
	behavior_type: BehaviorType
	metric_name: str = Field(description="Name of the behavioral metric")
	
	# Statistical measures
	mean_value: Decimal = Field(description="Mean value of the metric")
	median_value: Decimal = Field(description="Median value")
	standard_deviation: Decimal = Field(description="Standard deviation")
	
	# Distribution parameters
	percentile_25: Decimal = Field(description="25th percentile")
	percentile_75: Decimal = Field(description="75th percentile")
	percentile_95: Decimal = Field(description="95th percentile")
	percentile_99: Decimal = Field(description="99th percentile")
	
	# Range values
	min_value: Decimal = Field(description="Minimum observed value")
	max_value: Decimal = Field(description="Maximum observed value")
	
	# Trend analysis
	trend_slope: Optional[Decimal] = None
	seasonality: Optional[Dict[str, Any]] = None
	
	# Time-based patterns
	hourly_patterns: Dict[str, Decimal] = Field(default_factory=dict)
	daily_patterns: Dict[str, Decimal] = Field(default_factory=dict)
	weekly_patterns: Dict[str, Decimal] = Field(default_factory=dict)
	
	# Baseline metadata
	sample_size: int = Field(description="Number of observations")
	confidence_interval: Decimal = Field(default=Decimal('95.0'))
	
	established_date: datetime = Field(default_factory=datetime.utcnow)
	expiry_date: Optional[datetime] = None
	
	is_valid: bool = True
	validation_score: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class BehavioralAnomaly(BaseModel):
	"""Detected behavioral anomaly"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	profile_id: str = Field(description="Associated behavioral profile")
	baseline_id: Optional[str] = None
	
	entity_id: str = Field(description="Entity with anomalous behavior")
	entity_type: EntityType
	
	anomaly_type: AnomalyType
	behavior_type: BehaviorType
	
	# Anomaly details
	metric_name: str = Field(description="Anomalous metric")
	observed_value: Decimal = Field(description="Observed value")
	expected_value: Decimal = Field(description="Expected value from baseline")
	
	deviation_score: Decimal = Field(description="Statistical deviation score", ge=0)
	anomaly_score: Decimal = Field(description="Overall anomaly score", ge=0, le=100)
	
	# Statistical analysis
	z_score: Optional[Decimal] = None
	p_value: Optional[Decimal] = None
	confidence_level: Decimal = Field(default=Decimal('95.0'))
	
	# Context information
	detection_timestamp: datetime = Field(default_factory=datetime.utcnow)
	event_timestamp: datetime = Field(description="When the anomalous behavior occurred")
	
	context_data: Dict[str, Any] = Field(default_factory=dict)
	
	# Risk assessment
	risk_level: RiskLevel = RiskLevel.MEDIUM
	risk_score: Decimal = Field(default=Decimal('50.0'), ge=0, le=100)
	
	# Business context
	business_impact: Dict[str, Any] = Field(default_factory=dict)
	affected_resources: List[str] = Field(default_factory=list)
	
	# Investigation status
	is_investigated: bool = False
	investigation_notes: Optional[str] = None
	investigator: Optional[str] = None
	
	# Resolution
	is_resolved: bool = False
	resolution_type: Optional[str] = None
	resolution_notes: Optional[str] = None
	resolved_at: Optional[datetime] = None
	
	# False positive tracking
	is_false_positive: bool = False
	false_positive_reason: Optional[str] = None
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class PeerGroup(BaseModel):
	"""Peer group for comparative behavioral analysis"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	name: str = Field(description="Peer group name")
	description: str = Field(description="Group description")
	
	# Group criteria
	grouping_criteria: Dict[str, Any] = Field(default_factory=dict)
	department_filter: Optional[str] = None
	role_filter: Optional[str] = None
	location_filter: Optional[str] = None
	
	# Group members
	member_entities: List[str] = Field(default_factory=list)
	member_count: int = 0
	
	# Group statistics
	group_baselines: Dict[str, Dict[str, Decimal]] = Field(default_factory=dict)
	group_patterns: Dict[str, Any] = Field(default_factory=dict)
	
	# Comparative metrics
	variance_threshold: Decimal = Field(default=Decimal('2.0'))
	outlier_threshold: Decimal = Field(default=Decimal('3.0'))
	
	last_updated: datetime = Field(default_factory=datetime.utcnow)
	
	is_active: bool = True
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class RiskAssessment(BaseModel):
	"""Behavioral risk assessment"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	entity_id: str = Field(description="Entity being assessed")
	entity_type: EntityType
	
	assessment_period_start: datetime
	assessment_period_end: datetime
	
	# Overall risk
	overall_risk_score: Decimal = Field(ge=0, le=100)
	risk_level: RiskLevel
	
	# Risk components
	behavioral_risk: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	temporal_risk: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	access_risk: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	peer_deviation_risk: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	# Risk factors
	high_risk_behaviors: List[str] = Field(default_factory=list)
	risk_indicators: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Anomaly summary
	total_anomalies: int = 0
	critical_anomalies: int = 0
	high_risk_anomalies: int = 0
	
	# Trend analysis
	risk_trend: str = Field(default="stable")  # increasing, decreasing, stable
	trend_confidence: Decimal = Field(default=Decimal('50.0'), ge=0, le=100)
	
	# Recommendations
	recommended_actions: List[str] = Field(default_factory=list)
	monitoring_recommendations: List[str] = Field(default_factory=list)
	
	# Business context
	business_impact_assessment: Dict[str, Any] = Field(default_factory=dict)
	compliance_implications: List[str] = Field(default_factory=list)
	
	# Assessment metadata
	assessment_method: str = Field(description="Risk assessment methodology")
	confidence_level: Decimal = Field(default=Decimal('85.0'), ge=0, le=100)
	
	next_assessment_date: Optional[datetime] = None
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class BehavioralMetrics(BaseModel):
	"""Behavioral analytics metrics and KPIs"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	metric_period_start: datetime
	metric_period_end: datetime
	
	# Profile metrics
	total_profiles: int = 0
	active_profiles: int = 0
	profiles_with_baselines: int = 0
	baseline_establishment_rate: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	# Anomaly metrics
	total_anomalies_detected: int = 0
	critical_anomalies: int = 0
	high_risk_anomalies: int = 0
	false_positive_rate: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	# Detection performance
	mean_time_to_detection: Optional[timedelta] = None
	detection_accuracy: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	# Risk assessment metrics
	high_risk_entities: int = 0
	risk_score_distribution: Dict[str, int] = Field(default_factory=dict)
	
	# Peer group metrics
	total_peer_groups: int = 0
	average_group_size: Decimal = Field(default=Decimal('0.0'))
	
	# System performance
	processing_latency: Dict[str, Decimal] = Field(default_factory=dict)
	model_accuracy: Dict[str, Decimal] = Field(default_factory=dict)
	
	# Behavioral trends
	behavior_trends: Dict[str, Any] = Field(default_factory=dict)
	seasonal_patterns: Dict[str, Any] = Field(default_factory=dict)
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class BehavioralAlert(BaseModel):
	"""Behavioral-based security alert"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	anomaly_id: str = Field(description="Associated anomaly ID")
	entity_id: str = Field(description="Entity with anomalous behavior")
	
	alert_type: str = Field(description="Type of behavioral alert")
	title: str = Field(description="Alert title")
	description: str = Field(description="Alert description")
	
	severity: RiskLevel = RiskLevel.MEDIUM
	risk_score: Decimal = Field(ge=0, le=100)
	
	# Alert triggers
	triggering_behaviors: List[str] = Field(default_factory=list)
	threshold_violations: Dict[str, Any] = Field(default_factory=dict)
	
	# Context
	business_context: Dict[str, Any] = Field(default_factory=dict)
	temporal_context: Dict[str, Any] = Field(default_factory=dict)
	
	# Response recommendations
	recommended_actions: List[str] = Field(default_factory=list)
	escalation_criteria: Dict[str, Any] = Field(default_factory=dict)
	
	# Alert status
	is_acknowledged: bool = False
	acknowledged_by: Optional[str] = None
	acknowledged_at: Optional[datetime] = None
	
	is_resolved: bool = False
	resolution_status: Optional[str] = None
	resolved_by: Optional[str] = None
	resolved_at: Optional[datetime] = None
	
	alert_timestamp: datetime = Field(default_factory=datetime.utcnow)
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None