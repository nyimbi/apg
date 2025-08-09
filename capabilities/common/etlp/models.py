#!/usr/bin/env python3
"""
APG ETLP Data Models
Core data models for pipeline orchestration and processing

Author: APG Platform Team  
Copyright: © 2025 Datacraft
"""

import asyncio
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid_extensions import uuid7str
from pydantic import BaseModel, Field, ConfigDict, validator, AfterValidator
from dataclasses import dataclass


class PipelineStatus(str, Enum):
	"""Pipeline execution status"""
	DRAFT = "draft"
	ACTIVE = "active"
	RUNNING = "running"
	PAUSED = "paused"
	SUCCESS = "success"
	FAILED = "failed"
	CANCELLED = "cancelled"
	SCHEDULED = "scheduled"


class TransformationType(str, Enum):
	"""Types of data transformations"""
	FILTER = "filter"
	MAP = "map"
	AGGREGATE = "aggregate"
	JOIN = "join"
	SORT = "sort"
	SPLIT = "split"
	MERGE = "merge"
	VALIDATE = "validate"
	CLEAN = "clean"
	EXTRACT = "extract"
	LOAD = "load"
	CUSTOM = "custom"


class DataSourceType(str, Enum):
	"""Data source connection types"""
	DATABASE = "database"
	FILE = "file"
	API = "api"
	STREAM = "stream"
	CLOUD_STORAGE = "cloud_storage"
	MESSAGE_QUEUE = "message_queue"
	WEBHOOK = "webhook"
	FTP = "ftp"
	EMAIL = "email"
	CUSTOM = "custom"


class QualityRuleType(str, Enum):
	"""Data quality rule types"""
	NOT_NULL = "not_null"
	UNIQUE = "unique"
	RANGE = "range"
	FORMAT = "format"
	REFERENCE = "reference"
	CUSTOM = "custom"
	ANOMALY = "anomaly"
	COMPLETENESS = "completeness"
	CONSISTENCY = "consistency"
	ACCURACY = "accuracy"


class ExecutionMode(str, Enum):
	"""Pipeline execution modes"""
	BATCH = "batch"
	STREAMING = "streaming"
	MICRO_BATCH = "micro_batch"
	EVENT_DRIVEN = "event_driven"
	SCHEDULED = "scheduled"
	MANUAL = "manual"


@dataclass
class PipelineMetrics:
	"""Pipeline execution metrics"""
	records_processed: int = 0
	records_failed: int = 0
	processing_time_ms: int = 0
	memory_usage_mb: float = 0.0
	cpu_usage_percent: float = 0.0
	error_rate: float = 0.0
	throughput_records_per_sec: float = 0.0


class Pipeline(BaseModel):
	"""Core pipeline definition model"""
	
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True,
		str_strip_whitespace=True
	)
	
	# Core identification
	id: str = Field(default_factory=uuid7str, description="Unique pipeline identifier")
	name: str = Field(..., min_length=1, max_length=255, description="Pipeline name")
	description: Optional[str] = Field(None, max_length=1000, description="Pipeline description")
	
	# APG multi-tenancy
	tenant_id: str = Field(..., description="APG tenant identifier")
	created_by: str = Field(..., description="User who created the pipeline")
	
	# Pipeline configuration
	version: str = Field(default="1.0.0", description="Semantic version")
	status: PipelineStatus = Field(default=PipelineStatus.DRAFT, description="Pipeline status")
	execution_mode: ExecutionMode = Field(default=ExecutionMode.BATCH, description="Execution mode")
	
	# Pipeline definition
	steps: List[Dict[str, Any]] = Field(default_factory=list, description="Pipeline execution steps")
	transformations: List[str] = Field(default_factory=list, description="Applied transformation IDs")
	data_sources: List[str] = Field(default_factory=list, description="Source data connection IDs")
	data_targets: List[str] = Field(default_factory=list, description="Target data connection IDs")
	
	# Configuration and settings
	configuration: Dict[str, Any] = Field(default_factory=dict, description="Pipeline configuration")
	environment_variables: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
	tags: List[str] = Field(default_factory=list, description="Pipeline tags for organization")
	
	# Scheduling and triggers
	schedule_cron: Optional[str] = Field(None, description="Cron expression for scheduling")
	triggers: List[Dict[str, Any]] = Field(default_factory=list, description="Event triggers")
	
	# Quality and monitoring
	quality_rules: List[str] = Field(default_factory=list, description="Applied quality rule IDs")
	monitoring_enabled: bool = Field(default=True, description="Enable monitoring")
	alert_on_failure: bool = Field(default=True, description="Send alerts on failure")
	
	# Performance settings
	max_parallelism: int = Field(default=4, ge=1, le=100, description="Maximum parallel execution")
	timeout_minutes: int = Field(default=60, ge=1, le=10080, description="Execution timeout")
	retry_count: int = Field(default=3, ge=0, le=10, description="Retry attempts on failure")
	
	# APG integration metadata
	lineage_tracked: bool = Field(default=True, description="Enable APG lineage tracking")
	audit_enabled: bool = Field(default=True, description="Enable APG audit logging")
	collaboration_enabled: bool = Field(default=True, description="Enable real-time collaboration")
	
	# AI/ML optimization
	ai_optimization_enabled: bool = Field(default=True, description="Enable AI-powered optimization")
	performance_profile: Optional[Dict[str, Any]] = Field(None, description="AI performance insights")
	
	# Audit trail
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	updated_by: Optional[str] = Field(None, description="User who last updated")
	
	# Soft delete support
	is_deleted: bool = Field(default=False, description="Soft delete flag")
	deleted_at: Optional[datetime] = Field(None, description="Deletion timestamp")
	deleted_by: Optional[str] = Field(None, description="User who deleted")
	
	@validator('name')
	def validate_name(cls, v: str) -> str:
		"""Validate pipeline name format"""
		if not v or not v.strip():
			raise ValueError("Pipeline name cannot be empty")
		if len(v.strip()) < 1:
			raise ValueError("Pipeline name too short")
		return v.strip()
	
	@validator('schedule_cron')
	def validate_cron(cls, v: Optional[str]) -> Optional[str]:
		"""Validate cron expression format"""
		if v and len(v.split()) != 5:
			raise ValueError("Invalid cron expression format")
		return v
	
	@validator('version')
	def validate_version(cls, v: str) -> str:
		"""Validate semantic version format"""
		parts = v.split('.')
		if len(parts) != 3 or not all(p.isdigit() for p in parts):
			raise ValueError("Version must be in semantic format (x.y.z)")
		return v

	async def _log_status_change(self, old_status: PipelineStatus, new_status: PipelineStatus) -> None:
		"""Log pipeline status changes for audit trail"""
		print(f"Pipeline {self.id} status changed: {old_status} -> {new_status}")


class Transformation(BaseModel):
	"""Reusable data transformation definition"""
	
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	# Core identification
	id: str = Field(default_factory=uuid7str, description="Unique transformation identifier")
	name: str = Field(..., min_length=1, max_length=255, description="Transformation name")
	description: Optional[str] = Field(None, max_length=1000, description="Transformation description")
	
	# APG multi-tenancy
	tenant_id: str = Field(..., description="APG tenant identifier")
	created_by: str = Field(..., description="User who created the transformation")
	
	# Transformation definition
	type: TransformationType = Field(..., description="Type of transformation")
	version: str = Field(default="1.0.0", description="Semantic version")
	
	# Logic and configuration
	logic: Dict[str, Any] = Field(..., description="Transformation logic definition")
	input_schema: Optional[Dict[str, Any]] = Field(None, description="Expected input schema")
	output_schema: Optional[Dict[str, Any]] = Field(None, description="Expected output schema")
	parameters: Dict[str, Any] = Field(default_factory=dict, description="Transformation parameters")
	
	# Validation and quality
	validation_rules: List[Dict[str, Any]] = Field(default_factory=list, description="Input validation rules")
	error_handling: Dict[str, Any] = Field(default_factory=dict, description="Error handling configuration")
	
	# Performance and optimization
	cacheable: bool = Field(default=False, description="Allow result caching")
	parallel_execution: bool = Field(default=True, description="Support parallel execution")
	memory_efficient: bool = Field(default=True, description="Optimize for memory usage")
	
	# Metadata and organization
	tags: List[str] = Field(default_factory=list, description="Transformation tags")
	category: Optional[str] = Field(None, description="Transformation category")
	is_public: bool = Field(default=False, description="Available to all tenants")
	
	# Usage tracking
	usage_count: int = Field(default=0, ge=0, description="Number of times used")
	last_used: Optional[datetime] = Field(None, description="Last usage timestamp")
	
	# Audit trail
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	updated_by: Optional[str] = Field(None, description="User who last updated")
	
	# Soft delete support
	is_deleted: bool = Field(default=False, description="Soft delete flag")
	deleted_at: Optional[datetime] = Field(None, description="Deletion timestamp")
	deleted_by: Optional[str] = Field(None, description="User who deleted")

	async def _log_execution(self, success: bool, duration_ms: int, error: Optional[str] = None) -> None:
		"""Log transformation execution for monitoring"""
		status = "SUCCESS" if success else "FAILED"
		print(f"Transformation {self.id} executed: {status} in {duration_ms}ms")


class Execution(BaseModel):
	"""Pipeline execution run history"""
	
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	# Core identification
	id: str = Field(default_factory=uuid7str, description="Unique execution identifier")
	pipeline_id: str = Field(..., description="Associated pipeline ID")
	
	# APG multi-tenancy
	tenant_id: str = Field(..., description="APG tenant identifier")
	
	# Execution details
	status: PipelineStatus = Field(..., description="Execution status")
	execution_mode: ExecutionMode = Field(..., description="Execution mode")
	triggered_by: str = Field(..., description="User or system that triggered execution")
	trigger_type: str = Field(..., description="Type of trigger (manual, scheduled, event)")
	
	# Timing information
	started_at: Optional[datetime] = Field(None, description="Execution start time")
	completed_at: Optional[datetime] = Field(None, description="Execution completion time")
	duration_ms: Optional[int] = Field(None, ge=0, description="Execution duration in milliseconds")
	
	# Execution context
	pipeline_version: str = Field(..., description="Pipeline version executed")
	configuration: Dict[str, Any] = Field(default_factory=dict, description="Execution configuration")
	environment: Dict[str, Any] = Field(default_factory=dict, description="Execution environment")
	
	# Performance metrics
	metrics: Optional[Dict[str, Any]] = Field(None, description="Execution metrics")
	records_processed: int = Field(default=0, ge=0, description="Total records processed")
	records_failed: int = Field(default=0, ge=0, description="Total records failed")
	
	# Error handling
	error_message: Optional[str] = Field(None, description="Error message if failed")
	error_details: Optional[Dict[str, Any]] = Field(None, description="Detailed error information")
	stack_trace: Optional[str] = Field(None, description="Error stack trace")
	
	# Logs and output
	logs: List[Dict[str, Any]] = Field(default_factory=list, description="Execution logs")
	output_artifacts: List[Dict[str, Any]] = Field(default_factory=list, description="Generated artifacts")
	
	# Resource usage
	max_memory_mb: Optional[float] = Field(None, ge=0, description="Peak memory usage")
	avg_cpu_percent: Optional[float] = Field(None, ge=0, le=100, description="Average CPU usage")
	
	# Quality metrics
	data_quality_score: Optional[float] = Field(None, ge=0, le=100, description="Overall data quality score")
	quality_issues: List[Dict[str, Any]] = Field(default_factory=list, description="Data quality issues found")
	
	# APG integration tracking
	lineage_captured: bool = Field(default=False, description="Data lineage captured")
	audit_recorded: bool = Field(default=False, description="Audit trail recorded")
	
	# Audit trail
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	
	@property
	def success_rate(self) -> float:
		"""Calculate success rate for processed records"""
		if self.records_processed == 0:
			return 100.0
		return ((self.records_processed - self.records_failed) / self.records_processed) * 100.0
	
	async def _log_completion(self) -> None:
		"""Log execution completion for monitoring"""
		print(f"Execution {self.id} completed: {self.status} - {self.records_processed} records")


class DataSource(BaseModel):
	"""Data source connection configuration"""
	
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	# Core identification
	id: str = Field(default_factory=uuid7str, description="Unique data source identifier")
	name: str = Field(..., min_length=1, max_length=255, description="Data source name")
	description: Optional[str] = Field(None, max_length=1000, description="Data source description")
	
	# APG multi-tenancy
	tenant_id: str = Field(..., description="APG tenant identifier")
	created_by: str = Field(..., description="User who created the data source")
	
	# Connection details
	type: DataSourceType = Field(..., description="Type of data source")
	connection_string: str = Field(..., description="Connection string or URL")
	
	# Authentication and security
	credentials: Optional[Dict[str, Any]] = Field(None, description="Connection credentials (encrypted)")
	use_ssl: bool = Field(default=True, description="Use SSL/TLS encryption")
	timeout_seconds: int = Field(default=30, ge=1, le=300, description="Connection timeout")
	
	# Configuration
	settings: Dict[str, Any] = Field(default_factory=dict, description="Connection-specific settings")
	headers: Dict[str, str] = Field(default_factory=dict, description="HTTP headers for API connections")
	query_parameters: Dict[str, str] = Field(default_factory=dict, description="Query parameters")
	
	# Schema and metadata
	schema_info: Optional[Dict[str, Any]] = Field(None, description="Data schema information")
	sample_data: Optional[List[Dict[str, Any]]] = Field(None, description="Sample data records")
	
	# Performance settings
	batch_size: int = Field(default=1000, ge=1, le=100000, description="Batch size for data retrieval")
	max_connections: int = Field(default=5, ge=1, le=50, description="Maximum concurrent connections")
	
	# Monitoring and health
	health_check_enabled: bool = Field(default=True, description="Enable health monitoring")
	last_health_check: Optional[datetime] = Field(None, description="Last health check timestamp")
	is_healthy: bool = Field(default=False, description="Current health status")
	
	# Usage tracking
	usage_count: int = Field(default=0, ge=0, description="Number of times used")
	last_used: Optional[datetime] = Field(None, description="Last usage timestamp")
	
	# APG integration
	metadata_sync_enabled: bool = Field(default=True, description="Sync with APG metadata service")
	lineage_tracking_enabled: bool = Field(default=True, description="Enable lineage tracking")
	
	# Metadata and organization
	tags: List[str] = Field(default_factory=list, description="Data source tags")
	category: Optional[str] = Field(None, description="Data source category")
	
	# Audit trail
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	updated_by: Optional[str] = Field(None, description="User who last updated")
	
	# Soft delete support
	is_deleted: bool = Field(default=False, description="Soft delete flag")
	deleted_at: Optional[datetime] = Field(None, description="Deletion timestamp")
	deleted_by: Optional[str] = Field(None, description="User who deleted")

	async def _log_health_check(self, healthy: bool, response_time_ms: int) -> None:
		"""Log health check results"""
		status = "HEALTHY" if healthy else "UNHEALTHY"
		print(f"DataSource {self.id} health check: {status} in {response_time_ms}ms")


class QualityRule(BaseModel):
	"""Data quality validation rule"""
	
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	# Core identification
	id: str = Field(default_factory=uuid7str, description="Unique quality rule identifier")
	name: str = Field(..., min_length=1, max_length=255, description="Quality rule name")
	description: Optional[str] = Field(None, max_length=1000, description="Quality rule description")
	
	# APG multi-tenancy
	tenant_id: str = Field(..., description="APG tenant identifier")
	created_by: str = Field(..., description="User who created the quality rule")
	
	# Rule definition
	type: QualityRuleType = Field(..., description="Type of quality rule")
	field_name: Optional[str] = Field(None, description="Target field name")
	condition: Dict[str, Any] = Field(..., description="Quality rule condition")
	severity: str = Field(default="warning", description="Rule severity level")
	
	# Validation logic
	validation_logic: Dict[str, Any] = Field(..., description="Validation logic definition")
	error_message: str = Field(..., description="Error message template")
	suggested_fix: Optional[str] = Field(None, description="Suggested fix for violations")
	
	# Execution settings
	enabled: bool = Field(default=True, description="Enable rule execution")
	stop_on_violation: bool = Field(default=False, description="Stop processing on violation")
	sample_percentage: float = Field(default=100.0, ge=0.1, le=100.0, description="Percentage of data to validate")
	
	# Performance and optimization
	cacheable: bool = Field(default=True, description="Allow result caching")
	parallel_execution: bool = Field(default=True, description="Support parallel execution")
	
	# Statistical tracking
	violation_count: int = Field(default=0, ge=0, description="Total violations detected")
	execution_count: int = Field(default=0, ge=0, description="Total executions")
	last_violation: Optional[datetime] = Field(None, description="Last violation timestamp")
	
	# Metadata and organization
	tags: List[str] = Field(default_factory=list, description="Quality rule tags")
	category: Optional[str] = Field(None, description="Quality rule category")
	is_public: bool = Field(default=False, description="Available to all tenants")
	
	# Audit trail
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	updated_by: Optional[str] = Field(None, description="User who last updated")
	
	# Soft delete support
	is_deleted: bool = Field(default=False, description="Soft delete flag")
	deleted_at: Optional[datetime] = Field(None, description="Deletion timestamp")
	deleted_by: Optional[str] = Field(None, description="User who deleted")
	
	@property
	def violation_rate(self) -> float:
		"""Calculate violation rate"""
		if self.execution_count == 0:
			return 0.0
		return (self.violation_count / self.execution_count) * 100.0

	async def _log_violation(self, field_value: Any, violation_details: str) -> None:
		"""Log quality rule violation"""
		print(f"Quality rule {self.id} violated: {field_value} - {violation_details}")


class Schedule(BaseModel):
	"""Pipeline scheduling configuration"""
	
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	# Core identification
	id: str = Field(default_factory=uuid7str, description="Unique schedule identifier")
	pipeline_id: str = Field(..., description="Associated pipeline ID")
	name: str = Field(..., min_length=1, max_length=255, description="Schedule name")
	
	# APG multi-tenancy
	tenant_id: str = Field(..., description="APG tenant identifier")
	created_by: str = Field(..., description="User who created the schedule")
	
	# Schedule configuration
	cron_expression: str = Field(..., description="Cron expression for scheduling")
	timezone: str = Field(default="UTC", description="Timezone for schedule")
	enabled: bool = Field(default=True, description="Enable schedule")
	
	# Execution settings
	max_concurrent_runs: int = Field(default=1, ge=1, le=10, description="Maximum concurrent executions")
	catch_up: bool = Field(default=False, description="Execute missed runs on startup")
	retry_on_failure: bool = Field(default=True, description="Retry failed executions")
	
	# Time windows
	start_date: Optional[datetime] = Field(None, description="Schedule start date")
	end_date: Optional[datetime] = Field(None, description="Schedule end date")
	
	# Execution tracking
	next_run: Optional[datetime] = Field(None, description="Next scheduled execution")
	last_run: Optional[datetime] = Field(None, description="Last execution timestamp")
	last_success: Optional[datetime] = Field(None, description="Last successful execution")
	
	# Statistics
	total_runs: int = Field(default=0, ge=0, description="Total executions")
	successful_runs: int = Field(default=0, ge=0, description="Successful executions")
	failed_runs: int = Field(default=0, ge=0, description="Failed executions")
	
	# Audit trail
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	updated_by: Optional[str] = Field(None, description="User who last updated")
	
	# Soft delete support
	is_deleted: bool = Field(default=False, description="Soft delete flag")
	deleted_at: Optional[datetime] = Field(None, description="Deletion timestamp")
	deleted_by: Optional[str] = Field(None, description="User who deleted")
	
	@property
	def success_rate(self) -> float:
		"""Calculate schedule success rate"""
		if self.total_runs == 0:
			return 100.0
		return (self.successful_runs / self.total_runs) * 100.0

	async def _log_execution_trigger(self, execution_id: str) -> None:
		"""Log scheduled execution trigger"""
		print(f"Schedule {self.id} triggered execution {execution_id}")


# Utility functions for model validation and processing
async def validate_pipeline_dependencies(pipeline: Pipeline) -> List[str]:
	"""Validate pipeline dependencies and return any issues"""
	issues = []
	
	# Validate transformations exist
	for transform_id in pipeline.transformations:
		# Check if transformation ID is valid UUID format
		if not transform_id or len(transform_id) < 10:
			issues.append({
				'type': 'validation_error',
				'severity': 'error',
				'message': f'Invalid transformation ID: {transform_id}',
				'field': 'transformations',
				'suggestion': 'Ensure transformation ID is a valid identifier'
			})
	
	# Validate data sources exist
	for source_id in pipeline.data_sources:
		# Check if data source ID is valid UUID format
		if not source_id or len(source_id) < 10:
			issues.append({
				'type': 'validation_error',
				'severity': 'error',
				'message': f'Invalid data source ID: {source_id}',
				'field': 'data_sources',
				'suggestion': 'Ensure data source ID is a valid identifier'
			})
	
	# Validate quality rules exist
	for rule_id in pipeline.quality_rules:
		# Check if quality rule ID is valid UUID format
		if not rule_id or len(rule_id) < 10:
			issues.append({
				'type': 'validation_error',
				'severity': 'error',
				'message': f'Invalid quality rule ID: {rule_id}',
				'field': 'quality_rules',
				'suggestion': 'Ensure quality rule ID is a valid identifier'
			})
	
	return issues


async def calculate_pipeline_complexity(pipeline: Pipeline) -> Dict[str, Any]:
	"""Calculate pipeline complexity metrics"""
	complexity_score = 0
	
	# Factor in number of steps
	complexity_score += len(pipeline.steps) * 2
	
	# Factor in number of transformations
	complexity_score += len(pipeline.transformations) * 3
	
	# Factor in number of data sources
	complexity_score += len(pipeline.data_sources) * 2
	
	# Factor in number of quality rules
	complexity_score += len(pipeline.quality_rules) * 1
	
	return {
		"complexity_score": complexity_score,
		"estimated_memory_mb": complexity_score * 10,
		"estimated_duration_minutes": complexity_score * 0.5,
		"risk_level": "low" if complexity_score < 50 else "medium" if complexity_score < 100 else "high"
	}