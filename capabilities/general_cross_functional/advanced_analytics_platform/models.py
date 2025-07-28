"""
Advanced Analytics Platform - Comprehensive Pydantic Models

Enterprise-grade data analytics, machine learning, and AI platform models
supporting real-time processing, predictive analytics, and business intelligence.

Copyright © 2025 Datacraft
Author: Nyimbi Odero <nyimbi@gmail.com>
Website: www.datacraft.co.ke
"""

from datetime import datetime, date, time
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Literal
from uuid import UUID

from pydantic import BaseModel, Field, validator, root_validator
from pydantic.config import ConfigDict
from pydantic.types import EmailStr, HttpUrl, Json
from uuid_extensions import uuid7str


class ConfigDict(ConfigDict):
	extra = 'forbid'
	validate_by_name = True
	validate_by_alias = True


# Base Analytics Model
class APAnalyticsBase(BaseModel):
	model_config = ConfigDict()
	
	id: str = Field(default_factory=uuid7str, description="Unique identifier")
	tenant_id: str = Field(..., description="Multi-tenant organization identifier")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	created_by: str = Field(..., description="User ID who created the record")
	updated_by: str = Field(..., description="User ID who last updated the record")
	is_active: bool = Field(default=True, description="Active status flag")
	metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


# Enumeration Types
class APDataSourceType(str, Enum):
	DATABASE = "database"
	FILE_SYSTEM = "file_system"
	API_ENDPOINT = "api_endpoint"
	STREAMING = "streaming"
	CLOUD_STORAGE = "cloud_storage"
	MESSAGE_QUEUE = "message_queue"
	WEB_SCRAPING = "web_scraping"
	IOT_SENSORS = "iot_sensors"
	SOCIAL_MEDIA = "social_media"
	EMAIL_SYSTEM = "email_system"


class APDataFormat(str, Enum):
	JSON = "json"
	CSV = "csv"
	XML = "xml"
	PARQUET = "parquet"
	AVRO = "avro"
	BINARY = "binary"
	TEXT = "text"
	IMAGE = "image"
	VIDEO = "video"
	AUDIO = "audio"


class APProcessingStatus(str, Enum):
	PENDING = "pending"
	RUNNING = "running"
	COMPLETED = "completed"
	FAILED = "failed"
	PAUSED = "paused"
	CANCELLED = "cancelled"
	SCHEDULED = "scheduled"
	RETRYING = "retrying"


class APModelType(str, Enum):
	REGRESSION = "regression"
	CLASSIFICATION = "classification"
	CLUSTERING = "clustering"
	TIME_SERIES = "time_series"
	DEEP_LEARNING = "deep_learning"
	NLP = "nlp"
	COMPUTER_VISION = "computer_vision"
	REINFORCEMENT_LEARNING = "reinforcement_learning"
	ENSEMBLE = "ensemble"
	AUTOML = "automl"


class APVisualizationType(str, Enum):
	LINE_CHART = "line_chart"
	BAR_CHART = "bar_chart"
	PIE_CHART = "pie_chart"
	SCATTER_PLOT = "scatter_plot"
	HEATMAP = "heatmap"
	HISTOGRAM = "histogram"
	BOX_PLOT = "box_plot"
	GAUGE = "gauge"
	MAP = "map"
	SANKEY = "sankey"
	TREEMAP = "treemap"
	NETWORK = "network"


class APAlertSeverity(str, Enum):
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	CRITICAL = "critical"
	INFO = "info"


class APComputeResourceType(str, Enum):
	CPU = "cpu"
	GPU = "gpu"
	TPU = "tpu"
	MEMORY = "memory"
	STORAGE = "storage"
	NETWORK = "network"


# Core Data Source Models
class APDataSourceConnection(APAnalyticsBase):
	"""Data source connection configuration"""
	name: str = Field(..., description="Connection name")
	source_type: APDataSourceType = Field(..., description="Type of data source")
	connection_string: str = Field(..., description="Encrypted connection string")
	authentication: Dict[str, Any] = Field(..., description="Authentication configuration")
	ssl_config: Optional[Dict[str, Any]] = Field(default=None, description="SSL configuration")
	connection_pool_size: int = Field(default=10, description="Connection pool size")
	timeout_seconds: int = Field(default=30, description="Connection timeout")
	retry_attempts: int = Field(default=3, description="Retry attempts on failure")
	health_check_interval: int = Field(default=300, description="Health check interval in seconds")
	compression_enabled: bool = Field(default=True, description="Enable data compression")
	encryption_key: Optional[str] = Field(default=None, description="Encryption key for sensitive data")


class APDataSource(APAnalyticsBase):
	"""Data source definition and configuration"""
	name: str = Field(..., description="Data source name")
	description: Optional[str] = Field(default=None, description="Data source description")
	connection_id: str = Field(..., description="Reference to connection configuration")
	source_schema: Dict[str, Any] = Field(..., description="Data schema definition")
	data_format: APDataFormat = Field(..., description="Data format type")
	refresh_interval: Optional[int] = Field(default=None, description="Auto-refresh interval in seconds")
	data_retention_days: int = Field(default=365, description="Data retention period")
	quality_rules: List[Dict[str, Any]] = Field(default_factory=list, description="Data quality validation rules")
	transformation_rules: List[Dict[str, Any]] = Field(default_factory=list, description="Data transformation rules")
	access_permissions: Dict[str, List[str]] = Field(default_factory=dict, description="Role-based access permissions")
	data_lineage: Dict[str, Any] = Field(default_factory=dict, description="Data lineage tracking")
	privacy_classification: str = Field(default="public", description="Data privacy classification")
	compliance_tags: List[str] = Field(default_factory=list, description="Compliance and regulatory tags")


# Analytics Job and Processing Models
class APAnalyticsJob(APAnalyticsBase):
	"""Analytics job definition and execution tracking"""
	name: str = Field(..., description="Job name")
	description: Optional[str] = Field(default=None, description="Job description")
	job_type: str = Field(..., description="Type of analytics job")
	data_sources: List[str] = Field(..., description="List of data source IDs")
	processing_config: Dict[str, Any] = Field(..., description="Processing configuration")
	schedule_config: Optional[Dict[str, Any]] = Field(default=None, description="Scheduling configuration")
	output_destinations: List[Dict[str, Any]] = Field(default_factory=list, description="Output destinations")
	status: APProcessingStatus = Field(default=APProcessingStatus.PENDING, description="Current job status")
	started_at: Optional[datetime] = Field(default=None, description="Job start timestamp")
	completed_at: Optional[datetime] = Field(default=None, description="Job completion timestamp")
	progress_percentage: float = Field(default=0.0, description="Job progress percentage")
	error_message: Optional[str] = Field(default=None, description="Error message if failed")
	execution_logs: List[Dict[str, Any]] = Field(default_factory=list, description="Execution logs")
	resource_usage: Dict[str, Any] = Field(default_factory=dict, description="Resource usage statistics")
	performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
	notification_config: Dict[str, Any] = Field(default_factory=dict, description="Notification settings")


class APAnalyticsExecution(APAnalyticsBase):
	"""Analytics job execution tracking and monitoring"""
	job_id: str = Field(..., description="Reference to analytics job")
	execution_id: str = Field(default_factory=uuid7str, description="Unique execution identifier")
	status: APProcessingStatus = Field(default=APProcessingStatus.PENDING, description="Execution status")
	started_at: datetime = Field(default_factory=datetime.utcnow, description="Execution start time")
	completed_at: Optional[datetime] = Field(default=None, description="Execution completion time")
	duration_seconds: Optional[float] = Field(default=None, description="Execution duration")
	input_data_volume: int = Field(default=0, description="Input data volume in bytes")
	output_data_volume: int = Field(default=0, description="Output data volume in bytes")
	rows_processed: int = Field(default=0, description="Number of rows processed")
	compute_resources: Dict[str, Any] = Field(default_factory=dict, description="Compute resources used")
	memory_usage_mb: float = Field(default=0.0, description="Peak memory usage in MB")
	cpu_usage_percentage: float = Field(default=0.0, description="Average CPU usage percentage")
	error_details: Optional[Dict[str, Any]] = Field(default=None, description="Error details if failed")
	output_location: Optional[str] = Field(default=None, description="Output data location")
	quality_score: Optional[float] = Field(default=None, description="Data quality score")


# Machine Learning Models
class APMLModel(APAnalyticsBase):
	"""Machine learning model definition and metadata"""
	name: str = Field(..., description="Model name")
	description: Optional[str] = Field(default=None, description="Model description")
	model_type: APModelType = Field(..., description="Type of ML model")
	algorithm: str = Field(..., description="ML algorithm used")
	framework: str = Field(..., description="ML framework (TensorFlow, PyTorch, etc.)")
	version: str = Field(..., description="Model version")
	training_data_sources: List[str] = Field(..., description="Training data source IDs")
	feature_columns: List[str] = Field(default_factory=list, description="Feature column names")
	target_column: Optional[str] = Field(default=None, description="Target column name")
	hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Model hyperparameters")
	training_config: Dict[str, Any] = Field(default_factory=dict, description="Training configuration")
	model_artifact_location: str = Field(..., description="Location of saved model artifacts")
	performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Model performance metrics")
	validation_results: Dict[str, Any] = Field(default_factory=dict, description="Validation results")
	deployment_status: str = Field(default="draft", description="Deployment status")
	deployment_endpoint: Optional[str] = Field(default=None, description="Model serving endpoint")
	serving_config: Dict[str, Any] = Field(default_factory=dict, description="Model serving configuration")
	monitoring_config: Dict[str, Any] = Field(default_factory=dict, description="Model monitoring settings")


class APMLTrainingJob(APAnalyticsBase):
	"""Machine learning training job tracking"""
	model_id: str = Field(..., description="Reference to ML model")
	job_name: str = Field(..., description="Training job name")
	training_data_location: str = Field(..., description="Training data location")
	validation_data_location: Optional[str] = Field(default=None, description="Validation data location")
	test_data_location: Optional[str] = Field(default=None, description="Test data location")
	training_config: Dict[str, Any] = Field(..., description="Training configuration")
	hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Hyperparameters used")
	status: APProcessingStatus = Field(default=APProcessingStatus.PENDING, description="Training status")
	started_at: Optional[datetime] = Field(default=None, description="Training start time")
	completed_at: Optional[datetime] = Field(default=None, description="Training completion time")
	epochs_completed: int = Field(default=0, description="Number of epochs completed")
	total_epochs: int = Field(default=100, description="Total epochs planned")
	current_loss: Optional[float] = Field(default=None, description="Current training loss")
	best_validation_score: Optional[float] = Field(default=None, description="Best validation score achieved")
	early_stopping_triggered: bool = Field(default=False, description="Whether early stopping was triggered")
	resource_allocation: Dict[str, Any] = Field(default_factory=dict, description="Allocated compute resources")
	training_logs: List[Dict[str, Any]] = Field(default_factory=list, description="Training logs")
	model_checkpoints: List[str] = Field(default_factory=list, description="Model checkpoint locations")


# Dashboard and Visualization Models
class APDashboard(APAnalyticsBase):
	"""Analytics dashboard configuration"""
	name: str = Field(..., description="Dashboard name")
	description: Optional[str] = Field(default=None, description="Dashboard description")
	category: str = Field(default="general", description="Dashboard category")
	layout_config: Dict[str, Any] = Field(..., description="Dashboard layout configuration")
	widget_configurations: List[Dict[str, Any]] = Field(default_factory=list, description="Widget configurations")
	data_refresh_interval: int = Field(default=300, description="Data refresh interval in seconds")
	access_permissions: Dict[str, List[str]] = Field(default_factory=dict, description="Access permissions")
	sharing_settings: Dict[str, Any] = Field(default_factory=dict, description="Sharing settings")
	theme_config: Dict[str, Any] = Field(default_factory=dict, description="Theme configuration")
	responsive_config: Dict[str, Any] = Field(default_factory=dict, description="Responsive design settings")
	filters: List[Dict[str, Any]] = Field(default_factory=list, description="Dashboard filters")
	drill_down_config: Dict[str, Any] = Field(default_factory=dict, description="Drill-down configuration")
	export_settings: Dict[str, Any] = Field(default_factory=dict, description="Export settings")
	embedded_config: Optional[Dict[str, Any]] = Field(default=None, description="Embedding configuration")


class APVisualization(APAnalyticsBase):
	"""Individual visualization/chart configuration"""
	dashboard_id: Optional[str] = Field(default=None, description="Parent dashboard ID")
	name: str = Field(..., description="Visualization name")
	description: Optional[str] = Field(default=None, description="Visualization description")
	visualization_type: APVisualizationType = Field(..., description="Type of visualization")
	data_source_id: str = Field(..., description="Data source ID")
	query_config: Dict[str, Any] = Field(..., description="Data query configuration")
	chart_config: Dict[str, Any] = Field(..., description="Chart configuration")
	styling_config: Dict[str, Any] = Field(default_factory=dict, description="Styling configuration")
	interaction_config: Dict[str, Any] = Field(default_factory=dict, description="User interaction settings")
	animation_config: Dict[str, Any] = Field(default_factory=dict, description="Animation settings")
	responsive_settings: Dict[str, Any] = Field(default_factory=dict, description="Responsive design settings")
	refresh_interval: int = Field(default=300, description="Data refresh interval in seconds")
	cache_config: Dict[str, Any] = Field(default_factory=dict, description="Caching configuration")
	alert_thresholds: List[Dict[str, Any]] = Field(default_factory=list, description="Alert thresholds")
	drill_through_config: Optional[Dict[str, Any]] = Field(default=None, description="Drill-through configuration")


# Report Models
class APReport(APAnalyticsBase):
	"""Analytics report definition"""
	name: str = Field(..., description="Report name")
	description: Optional[str] = Field(default=None, description="Report description")
	report_category: str = Field(default="operational", description="Report category")
	template_id: Optional[str] = Field(default=None, description="Report template ID")
	data_sources: List[str] = Field(..., description="Data source IDs")
	report_structure: Dict[str, Any] = Field(..., description="Report structure definition")
	parameters: List[Dict[str, Any]] = Field(default_factory=list, description="Report parameters")
	filters: List[Dict[str, Any]] = Field(default_factory=list, description="Report filters")
	formatting_config: Dict[str, Any] = Field(default_factory=dict, description="Formatting configuration")
	output_formats: List[str] = Field(default_factory=list, description="Supported output formats")
	scheduling_config: Optional[Dict[str, Any]] = Field(default=None, description="Report scheduling")
	distribution_list: List[str] = Field(default_factory=list, description="Distribution email list")
	access_permissions: Dict[str, List[str]] = Field(default_factory=dict, description="Access permissions")
	retention_policy: Dict[str, Any] = Field(default_factory=dict, description="Report retention policy")
	performance_optimization: Dict[str, Any] = Field(default_factory=dict, description="Performance settings")


class APReportExecution(APAnalyticsBase):
	"""Report execution tracking"""
	report_id: str = Field(..., description="Report ID")
	execution_id: str = Field(default_factory=uuid7str, description="Execution ID")
	status: APProcessingStatus = Field(default=APProcessingStatus.PENDING, description="Execution status")
	parameters: Dict[str, Any] = Field(default_factory=dict, description="Execution parameters")
	started_at: datetime = Field(default_factory=datetime.utcnow, description="Execution start time")
	completed_at: Optional[datetime] = Field(default=None, description="Execution completion time")
	output_location: Optional[str] = Field(default=None, description="Report output location")
	output_format: str = Field(default="pdf", description="Report output format")
	file_size_bytes: Optional[int] = Field(default=None, description="Output file size")
	page_count: Optional[int] = Field(default=None, description="Number of pages generated")
	generation_time_seconds: Optional[float] = Field(default=None, description="Report generation time")
	error_message: Optional[str] = Field(default=None, description="Error message if failed")
	quality_metrics: Dict[str, Any] = Field(default_factory=dict, description="Report quality metrics")
	delivery_status: Dict[str, Any] = Field(default_factory=dict, description="Delivery status tracking")


# Alert and Monitoring Models
class APAlert(APAnalyticsBase):
	"""Analytics alert configuration and tracking"""
	name: str = Field(..., description="Alert name")
	description: Optional[str] = Field(default=None, description="Alert description")
	data_source_id: str = Field(..., description="Data source ID to monitor")
	alert_condition: Dict[str, Any] = Field(..., description="Alert trigger condition")
	severity: APAlertSeverity = Field(..., description="Alert severity level")
	threshold_config: Dict[str, Any] = Field(..., description="Threshold configuration")
	evaluation_frequency: int = Field(default=300, description="Evaluation frequency in seconds")
	notification_channels: List[Dict[str, Any]] = Field(..., description="Notification channels")
	escalation_rules: List[Dict[str, Any]] = Field(default_factory=list, description="Escalation rules")
	suppression_rules: List[Dict[str, Any]] = Field(default_factory=list, description="Alert suppression rules")
	is_enabled: bool = Field(default=True, description="Alert enabled status")
	last_evaluated_at: Optional[datetime] = Field(default=None, description="Last evaluation timestamp")
	last_triggered_at: Optional[datetime] = Field(default=None, description="Last trigger timestamp")
	trigger_count: int = Field(default=0, description="Total trigger count")
	false_positive_count: int = Field(default=0, description="False positive count")
	acknowledgment_required: bool = Field(default=False, description="Requires acknowledgment")
	auto_resolution_config: Optional[Dict[str, Any]] = Field(default=None, description="Auto-resolution settings")


class APAlertInstance(APAnalyticsBase):
	"""Individual alert instance/trigger"""
	alert_id: str = Field(..., description="Alert configuration ID")
	instance_id: str = Field(default_factory=uuid7str, description="Alert instance ID")
	triggered_at: datetime = Field(default_factory=datetime.utcnow, description="Alert trigger timestamp")
	severity: APAlertSeverity = Field(..., description="Alert severity")
	trigger_value: Union[float, str, Dict[str, Any]] = Field(..., description="Value that triggered alert")
	threshold_value: Union[float, str, Dict[str, Any]] = Field(..., description="Threshold value")
	message: str = Field(..., description="Alert message")
	context_data: Dict[str, Any] = Field(default_factory=dict, description="Additional context data")
	status: str = Field(default="open", description="Alert status (open, acknowledged, resolved)")
	acknowledged_at: Optional[datetime] = Field(default=None, description="Acknowledgment timestamp")
	acknowledged_by: Optional[str] = Field(default=None, description="User who acknowledged")
	resolved_at: Optional[datetime] = Field(default=None, description="Resolution timestamp")
	resolved_by: Optional[str] = Field(default=None, description="User who resolved")
	resolution_notes: Optional[str] = Field(default=None, description="Resolution notes")
	escalated_at: Optional[datetime] = Field(default=None, description="Escalation timestamp")
	escalation_level: int = Field(default=0, description="Current escalation level")
	notification_log: List[Dict[str, Any]] = Field(default_factory=list, description="Notification delivery log")


# Data Quality and Governance Models
class APDataQualityRule(APAnalyticsBase):
	"""Data quality rule definition"""
	name: str = Field(..., description="Rule name")
	description: Optional[str] = Field(default=None, description="Rule description")
	data_source_id: str = Field(..., description="Data source ID")
	rule_type: str = Field(..., description="Type of quality rule")
	rule_config: Dict[str, Any] = Field(..., description="Rule configuration")
	evaluation_frequency: int = Field(default=3600, description="Evaluation frequency in seconds")
	severity: APAlertSeverity = Field(default=APAlertSeverity.MEDIUM, description="Rule violation severity")
	threshold_config: Dict[str, Any] = Field(default_factory=dict, description="Quality thresholds")
	remediation_actions: List[Dict[str, Any]] = Field(default_factory=list, description="Automated remediation actions")
	is_blocking: bool = Field(default=False, description="Blocks data processing if violated")
	notification_config: Dict[str, Any] = Field(default_factory=dict, description="Notification settings")
	last_evaluated_at: Optional[datetime] = Field(default=None, description="Last evaluation timestamp")
	violation_count: int = Field(default=0, description="Total violation count")
	pass_rate_percentage: float = Field(default=100.0, description="Historical pass rate percentage")


class APDataLineage(APAnalyticsBase):
	"""Data lineage tracking"""
	source_data_id: str = Field(..., description="Source data identifier")
	target_data_id: str = Field(..., description="Target data identifier")
	transformation_id: Optional[str] = Field(default=None, description="Transformation job ID")
	lineage_type: str = Field(..., description="Type of lineage relationship")
	transformation_logic: Optional[Dict[str, Any]] = Field(default=None, description="Transformation logic applied")
	impact_analysis: Dict[str, Any] = Field(default_factory=dict, description="Impact analysis data")
	dependency_level: int = Field(default=1, description="Dependency level in the chain")
	processing_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Processing timestamp")
	data_volume: Optional[int] = Field(default=None, description="Data volume processed")
	quality_score: Optional[float] = Field(default=None, description="Data quality score")
	validation_status: str = Field(default="pending", description="Lineage validation status")
	compliance_tags: List[str] = Field(default_factory=list, description="Compliance tags")


# Feature Store Models
class APFeatureStore(APAnalyticsBase):
	"""Feature store for ML features"""
	name: str = Field(..., description="Feature store name")
	description: Optional[str] = Field(default=None, description="Feature store description")
	feature_groups: List[Dict[str, Any]] = Field(default_factory=list, description="Feature group definitions")
	storage_config: Dict[str, Any] = Field(..., description="Storage configuration")
	serving_config: Dict[str, Any] = Field(..., description="Feature serving configuration")
	version_control: Dict[str, Any] = Field(default_factory=dict, description="Version control settings")
	access_permissions: Dict[str, List[str]] = Field(default_factory=dict, description="Access permissions")
	monitoring_config: Dict[str, Any] = Field(default_factory=dict, description="Monitoring configuration")
	retention_policy: Dict[str, Any] = Field(default_factory=dict, description="Data retention policy")
	backup_config: Dict[str, Any] = Field(default_factory=dict, description="Backup configuration")


class APFeature(APAnalyticsBase):
	"""Individual feature definition"""
	feature_store_id: str = Field(..., description="Feature store ID")
	name: str = Field(..., description="Feature name")
	description: Optional[str] = Field(default=None, description="Feature description")
	data_type: str = Field(..., description="Feature data type")
	feature_group: str = Field(..., description="Feature group name")
	computation_logic: Dict[str, Any] = Field(..., description="Feature computation logic")
	data_sources: List[str] = Field(..., description="Source data identifiers")
	refresh_frequency: int = Field(default=3600, description="Refresh frequency in seconds")
	feature_statistics: Dict[str, Any] = Field(default_factory=dict, description="Feature statistics")
	quality_metrics: Dict[str, Any] = Field(default_factory=dict, description="Feature quality metrics")
	drift_detection_config: Dict[str, Any] = Field(default_factory=dict, description="Drift detection settings")
	version: str = Field(default="1.0", description="Feature version")
	tags: List[str] = Field(default_factory=list, description="Feature tags")
	business_context: Optional[str] = Field(default=None, description="Business context description")


# Compute Resource Management
class APComputeCluster(APAnalyticsBase):
	"""Compute cluster for analytics processing"""
	name: str = Field(..., description="Cluster name")
	description: Optional[str] = Field(default=None, description="Cluster description")
	cluster_type: str = Field(..., description="Type of compute cluster")
	node_configuration: Dict[str, Any] = Field(..., description="Node configuration")
	scaling_config: Dict[str, Any] = Field(..., description="Auto-scaling configuration")
	resource_limits: Dict[str, Any] = Field(..., description="Resource limits")
	networking_config: Dict[str, Any] = Field(default_factory=dict, description="Networking configuration")
	security_config: Dict[str, Any] = Field(default_factory=dict, description="Security configuration")
	monitoring_config: Dict[str, Any] = Field(default_factory=dict, description="Monitoring configuration")
	cost_optimization: Dict[str, Any] = Field(default_factory=dict, description="Cost optimization settings")
	maintenance_windows: List[Dict[str, Any]] = Field(default_factory=list, description="Maintenance windows")
	backup_config: Dict[str, Any] = Field(default_factory=dict, description="Backup configuration")
	current_nodes: int = Field(default=0, description="Current number of nodes")
	target_nodes: int = Field(default=1, description="Target number of nodes")
	status: str = Field(default="inactive", description="Cluster status")


class APResourceUsage(APAnalyticsBase):
	"""Resource usage tracking and monitoring"""
	resource_id: str = Field(..., description="Resource identifier")
	resource_type: APComputeResourceType = Field(..., description="Type of resource")
	usage_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Usage timestamp")
	usage_metrics: Dict[str, Any] = Field(..., description="Resource usage metrics")
	allocation_metrics: Dict[str, Any] = Field(default_factory=dict, description="Resource allocation metrics")
	cost_metrics: Dict[str, Any] = Field(default_factory=dict, description="Cost metrics")
	efficiency_score: Optional[float] = Field(default=None, description="Resource efficiency score")
	optimization_recommendations: List[str] = Field(default_factory=list, description="Optimization recommendations")
	usage_anomalies: List[Dict[str, Any]] = Field(default_factory=list, description="Detected usage anomalies")
	performance_impact: Dict[str, Any] = Field(default_factory=dict, description="Performance impact analysis")


# Advanced Analytics Models
class APPredictiveModel(APAnalyticsBase):
	"""Predictive analytics model configuration"""
	name: str = Field(..., description="Predictive model name")
	description: Optional[str] = Field(default=None, description="Model description")
	prediction_target: str = Field(..., description="What the model predicts")
	model_algorithm: str = Field(..., description="Prediction algorithm used")
	input_features: List[str] = Field(..., description="Input feature names")
	training_data_period: Dict[str, Any] = Field(..., description="Training data time period")
	prediction_horizon: int = Field(..., description="Prediction horizon in time units")
	confidence_thresholds: Dict[str, float] = Field(default_factory=dict, description="Confidence thresholds")
	accuracy_metrics: Dict[str, Any] = Field(default_factory=dict, description="Model accuracy metrics")
	drift_detection: Dict[str, Any] = Field(default_factory=dict, description="Model drift detection settings")
	retraining_schedule: Dict[str, Any] = Field(default_factory=dict, description="Automated retraining schedule")
	prediction_cache_config: Dict[str, Any] = Field(default_factory=dict, description="Prediction caching settings")
	explanation_config: Dict[str, Any] = Field(default_factory=dict, description="Model explainability settings")
	bias_monitoring: Dict[str, Any] = Field(default_factory=dict, description="Bias monitoring configuration")


class APAnomalyDetection(APAnalyticsBase):
	"""Anomaly detection configuration"""
	name: str = Field(..., description="Anomaly detection name")
	description: Optional[str] = Field(default=None, description="Detection description")
	data_source_id: str = Field(..., description="Data source to monitor")
	detection_algorithm: str = Field(..., description="Anomaly detection algorithm")
	algorithm_parameters: Dict[str, Any] = Field(..., description="Algorithm-specific parameters")
	sensitivity_level: float = Field(default=0.95, description="Detection sensitivity (0-1)")
	baseline_period: int = Field(default=30, description="Baseline period in days")
	detection_frequency: int = Field(default=300, description="Detection frequency in seconds")
	seasonal_adjustment: bool = Field(default=True, description="Apply seasonal adjustments")
	multivariate_analysis: bool = Field(default=False, description="Enable multivariate analysis")
	contextual_factors: List[str] = Field(default_factory=list, description="Contextual factors to consider")
	feedback_learning: bool = Field(default=True, description="Enable feedback learning")
	anomaly_scoring: Dict[str, Any] = Field(default_factory=dict, description="Anomaly scoring configuration")
	notification_rules: List[Dict[str, Any]] = Field(default_factory=list, description="Notification rules")
	false_positive_rate: Optional[float] = Field(default=None, description="Historical false positive rate")


# Industry-Specific Models
class APFinancialAnalytics(APAnalyticsBase):
	"""Financial analytics specific configuration"""
	portfolio_id: Optional[str] = Field(default=None, description="Portfolio identifier")
	risk_models: List[Dict[str, Any]] = Field(default_factory=list, description="Risk model configurations")
	compliance_frameworks: List[str] = Field(default_factory=list, description="Regulatory frameworks")
	stress_testing_scenarios: List[Dict[str, Any]] = Field(default_factory=list, description="Stress testing scenarios")
	var_calculation: Dict[str, Any] = Field(default_factory=dict, description="Value at Risk calculation settings")
	credit_risk_models: List[Dict[str, Any]] = Field(default_factory=list, description="Credit risk models")
	market_risk_models: List[Dict[str, Any]] = Field(default_factory=list, description="Market risk models")
	operational_risk_models: List[Dict[str, Any]] = Field(default_factory=list, description="Operational risk models")
	regulatory_reporting: Dict[str, Any] = Field(default_factory=dict, description="Regulatory reporting configuration")
	backtesting_config: Dict[str, Any] = Field(default_factory=dict, description="Model backtesting configuration")


class APHealthcareAnalytics(APAnalyticsBase):
	"""Healthcare analytics specific configuration"""
	patient_cohort_id: Optional[str] = Field(default=None, description="Patient cohort identifier")
	clinical_outcomes: List[str] = Field(default_factory=list, description="Clinical outcomes to track")
	risk_stratification: Dict[str, Any] = Field(default_factory=dict, description="Risk stratification models")
	population_health_metrics: List[str] = Field(default_factory=list, description="Population health metrics")
	quality_measures: List[Dict[str, Any]] = Field(default_factory=list, description="Healthcare quality measures")
	cost_effectiveness_analysis: Dict[str, Any] = Field(default_factory=dict, description="Cost-effectiveness settings")
	clinical_decision_support: Dict[str, Any] = Field(default_factory=dict, description="Clinical decision support rules")
	epidemiological_models: List[Dict[str, Any]] = Field(default_factory=list, description="Epidemiological models")
	privacy_protection: Dict[str, Any] = Field(..., description="Patient privacy protection settings")
	hipaa_compliance: Dict[str, Any] = Field(..., description="HIPAA compliance configuration")


# Validation Methods
@validator('processing_config', pre=True, always=True)
def validate_processing_config(cls, v):
	if not isinstance(v, dict):
		raise ValueError("Processing configuration must be a dictionary")
	return v


@root_validator
def validate_date_consistency(cls, values):
	started_at = values.get('started_at')
	completed_at = values.get('completed_at')
	
	if started_at and completed_at and completed_at < started_at:
		raise ValueError("Completion time cannot be before start time")
	
	return values


@validator('confidence_thresholds')
def validate_confidence_thresholds(cls, v):
	for key, value in v.items():
		if not 0.0 <= value <= 1.0:
			raise ValueError(f"Confidence threshold {key} must be between 0.0 and 1.0")
	return v


# Composite Models for Complex Operations
class APAnalyticsPipeline(APAnalyticsBase):
	"""End-to-end analytics pipeline definition"""
	name: str = Field(..., description="Pipeline name")
	description: Optional[str] = Field(default=None, description="Pipeline description")
	stages: List[Dict[str, Any]] = Field(..., description="Pipeline stages configuration")
	data_flow: Dict[str, Any] = Field(..., description="Data flow definition")
	dependencies: List[str] = Field(default_factory=list, description="Pipeline dependencies")
	trigger_conditions: List[Dict[str, Any]] = Field(default_factory=list, description="Pipeline triggers")
	error_handling: Dict[str, Any] = Field(default_factory=dict, description="Error handling configuration")
	monitoring_config: Dict[str, Any] = Field(default_factory=dict, description="Pipeline monitoring")
	optimization_settings: Dict[str, Any] = Field(default_factory=dict, description="Performance optimization")
	version_control: Dict[str, Any] = Field(default_factory=dict, description="Version control settings")
	deployment_config: Dict[str, Any] = Field(default_factory=dict, description="Deployment configuration")
	rollback_strategy: Dict[str, Any] = Field(default_factory=dict, description="Rollback strategy")


class APBusinessIntelligence(APAnalyticsBase):
	"""Business intelligence solution configuration"""
	solution_name: str = Field(..., description="BI solution name")
	business_domain: str = Field(..., description="Business domain area")
	kpi_definitions: List[Dict[str, Any]] = Field(..., description="KPI definitions")
	metric_hierarchies: Dict[str, Any] = Field(default_factory=dict, description="Metric hierarchies")
	dimensional_model: Dict[str, Any] = Field(..., description="Dimensional model structure")
	cube_configurations: List[Dict[str, Any]] = Field(default_factory=list, description="OLAP cube configurations")
	drill_paths: List[Dict[str, Any]] = Field(default_factory=list, description="Drill-down paths")
	calculated_measures: List[Dict[str, Any]] = Field(default_factory=list, description="Calculated measures")
	time_intelligence: Dict[str, Any] = Field(default_factory=dict, description="Time intelligence functions")
	data_modeling: Dict[str, Any] = Field(default_factory=dict, description="Data modeling configuration")
	performance_tuning: Dict[str, Any] = Field(default_factory=dict, description="Performance tuning settings")
	user_access_matrix: Dict[str, Any] = Field(default_factory=dict, description="User access control matrix")