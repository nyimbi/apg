#!/usr/bin/env python3
"""
APG ETLP Views and UI Models
Pydantic models and Flask-AppBuilder views for ETLP interface

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict, validator, AfterValidator
from uuid_extensions import uuid7str

from .models import PipelineStatus, ExecutionMode, TransformationType, DataSourceType, QualityRuleType

# Pydantic v2 models for API requests/responses

class PipelineCreateRequest(BaseModel):
	"""Request model for creating a pipeline"""
	
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True,
		str_strip_whitespace=True
	)
	
	name: str = Field(..., min_length=1, max_length=255, description="Pipeline name")
	description: Optional[str] = Field(None, max_length=1000, description="Pipeline description")
	execution_mode: ExecutionMode = Field(default=ExecutionMode.BATCH, description="Execution mode")
	steps: List[Dict[str, Any]] = Field(default_factory=list, description="Pipeline execution steps")
	transformations: List[str] = Field(default_factory=list, description="Applied transformation IDs")
	data_sources: List[str] = Field(default_factory=list, description="Source data connection IDs")
	data_targets: List[str] = Field(default_factory=list, description="Target data connection IDs")
	configuration: Dict[str, Any] = Field(default_factory=dict, description="Pipeline configuration")
	schedule_cron: Optional[str] = Field(None, description="Cron expression for scheduling")
	tags: List[str] = Field(default_factory=list, description="Pipeline tags")
	max_parallelism: int = Field(default=4, ge=1, le=100, description="Maximum parallel execution")
	timeout_minutes: int = Field(default=60, ge=1, le=10080, description="Execution timeout")
	retry_count: int = Field(default=3, ge=0, le=10, description="Retry attempts on failure")
	ai_optimization_enabled: bool = Field(default=True, description="Enable AI-powered optimization")


class PipelineUpdateRequest(BaseModel):
	"""Request model for updating a pipeline"""
	
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True,
		str_strip_whitespace=True
	)
	
	name: Optional[str] = Field(None, min_length=1, max_length=255, description="Pipeline name")
	description: Optional[str] = Field(None, max_length=1000, description="Pipeline description")
	status: Optional[PipelineStatus] = Field(None, description="Pipeline status")
	execution_mode: Optional[ExecutionMode] = Field(None, description="Execution mode")
	steps: Optional[List[Dict[str, Any]]] = Field(None, description="Pipeline execution steps")
	transformations: Optional[List[str]] = Field(None, description="Applied transformation IDs")
	data_sources: Optional[List[str]] = Field(None, description="Source data connection IDs")
	data_targets: Optional[List[str]] = Field(None, description="Target data connection IDs")
	configuration: Optional[Dict[str, Any]] = Field(None, description="Pipeline configuration")
	schedule_cron: Optional[str] = Field(None, description="Cron expression for scheduling")
	tags: Optional[List[str]] = Field(None, description="Pipeline tags")
	max_parallelism: Optional[int] = Field(None, ge=1, le=100, description="Maximum parallel execution")
	timeout_minutes: Optional[int] = Field(None, ge=1, le=10080, description="Execution timeout")
	retry_count: Optional[int] = Field(None, ge=0, le=10, description="Retry attempts on failure")
	ai_optimization_enabled: Optional[bool] = Field(None, description="Enable AI-powered optimization")


class PipelineResponse(BaseModel):
	"""Response model for pipeline data"""
	
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(..., description="Unique pipeline identifier")
	name: str = Field(..., description="Pipeline name")
	description: Optional[str] = Field(None, description="Pipeline description")
	version: str = Field(..., description="Semantic version")
	status: PipelineStatus = Field(..., description="Pipeline status")
	execution_mode: ExecutionMode = Field(..., description="Execution mode")
	steps: List[Dict[str, Any]] = Field(..., description="Pipeline execution steps")
	transformations: List[str] = Field(..., description="Applied transformation IDs")
	data_sources: List[str] = Field(..., description="Source data connection IDs")
	data_targets: List[str] = Field(..., description="Target data connection IDs")
	configuration: Dict[str, Any] = Field(..., description="Pipeline configuration")
	schedule_cron: Optional[str] = Field(None, description="Cron expression for scheduling")
	tags: List[str] = Field(..., description="Pipeline tags")
	created_by: str = Field(..., description="User who created the pipeline")
	created_at: datetime = Field(..., description="Creation timestamp")
	updated_at: datetime = Field(..., description="Last update timestamp")
	updated_by: Optional[str] = Field(None, description="User who last updated")


class PipelineExecuteRequest(BaseModel):
	"""Request model for executing a pipeline"""
	
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	execution_mode: Optional[ExecutionMode] = Field(None, description="Override execution mode")
	configuration: Optional[Dict[str, Any]] = Field(None, description="Override configuration")
	environment_variables: Optional[Dict[str, str]] = Field(None, description="Environment variables")


class TransformationCreateRequest(BaseModel):
	"""Request model for creating a transformation"""
	
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True,
		str_strip_whitespace=True
	)
	
	name: str = Field(..., min_length=1, max_length=255, description="Transformation name")
	description: Optional[str] = Field(None, max_length=1000, description="Transformation description")
	type: TransformationType = Field(..., description="Type of transformation")
	logic: Dict[str, Any] = Field(..., description="Transformation logic definition")
	input_schema: Optional[Dict[str, Any]] = Field(None, description="Expected input schema")
	output_schema: Optional[Dict[str, Any]] = Field(None, description="Expected output schema")
	parameters: Dict[str, Any] = Field(default_factory=dict, description="Transformation parameters")
	tags: List[str] = Field(default_factory=list, description="Transformation tags")
	category: Optional[str] = Field(None, description="Transformation category")
	is_public: bool = Field(default=False, description="Available to all tenants")
	cacheable: bool = Field(default=False, description="Allow result caching")
	parallel_execution: bool = Field(default=True, description="Support parallel execution")


class TransformationResponse(BaseModel):
	"""Response model for transformation data"""
	
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(..., description="Unique transformation identifier")
	name: str = Field(..., description="Transformation name")
	description: Optional[str] = Field(None, description="Transformation description")
	type: TransformationType = Field(..., description="Type of transformation")
	version: str = Field(..., description="Semantic version")
	tags: List[str] = Field(..., description="Transformation tags")
	category: Optional[str] = Field(None, description="Transformation category")
	is_public: bool = Field(..., description="Available to all tenants")
	usage_count: int = Field(..., description="Number of times used")
	last_used: Optional[datetime] = Field(None, description="Last usage timestamp")
	created_by: str = Field(..., description="User who created the transformation")
	created_at: datetime = Field(..., description="Creation timestamp")
	updated_at: datetime = Field(..., description="Last update timestamp")


class DataSourceCreateRequest(BaseModel):
	"""Request model for creating a data source"""
	
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True,
		str_strip_whitespace=True
	)
	
	name: str = Field(..., min_length=1, max_length=255, description="Data source name")
	description: Optional[str] = Field(None, max_length=1000, description="Data source description")
	type: DataSourceType = Field(..., description="Type of data source")
	connection_string: str = Field(..., description="Connection string or URL")
	credentials: Optional[Dict[str, Any]] = Field(None, description="Connection credentials")
	use_ssl: bool = Field(default=True, description="Use SSL/TLS encryption")
	timeout_seconds: int = Field(default=30, ge=1, le=300, description="Connection timeout")
	settings: Dict[str, Any] = Field(default_factory=dict, description="Connection-specific settings")
	headers: Dict[str, str] = Field(default_factory=dict, description="HTTP headers for API connections")
	batch_size: int = Field(default=1000, ge=1, le=100000, description="Batch size for data retrieval")
	max_connections: int = Field(default=5, ge=1, le=50, description="Maximum concurrent connections")
	tags: List[str] = Field(default_factory=list, description="Data source tags")
	category: Optional[str] = Field(None, description="Data source category")
	health_check_enabled: bool = Field(default=True, description="Enable health monitoring")


class DataSourceResponse(BaseModel):
	"""Response model for data source data"""
	
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(..., description="Unique data source identifier")
	name: str = Field(..., description="Data source name")
	description: Optional[str] = Field(None, description="Data source description")
	type: DataSourceType = Field(..., description="Type of data source")
	connection_string: str = Field(..., description="Connection string (masked)")
	use_ssl: bool = Field(..., description="Use SSL/TLS encryption")
	timeout_seconds: int = Field(..., description="Connection timeout")
	batch_size: int = Field(..., description="Batch size for data retrieval")
	max_connections: int = Field(..., description="Maximum concurrent connections")
	tags: List[str] = Field(..., description="Data source tags")
	category: Optional[str] = Field(None, description="Data source category")
	health_check_enabled: bool = Field(..., description="Health monitoring enabled")
	is_healthy: bool = Field(..., description="Current health status")
	last_health_check: Optional[datetime] = Field(None, description="Last health check timestamp")
	usage_count: int = Field(..., description="Number of times used")
	created_by: str = Field(..., description="User who created the data source")
	created_at: datetime = Field(..., description="Creation timestamp")
	updated_at: datetime = Field(..., description="Last update timestamp")


class QualityRuleCreateRequest(BaseModel):
	"""Request model for creating a quality rule"""
	
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True,
		str_strip_whitespace=True
	)
	
	name: str = Field(..., min_length=1, max_length=255, description="Quality rule name")
	description: Optional[str] = Field(None, max_length=1000, description="Quality rule description")
	type: QualityRuleType = Field(..., description="Type of quality rule")
	field_name: Optional[str] = Field(None, description="Target field name")
	condition: Dict[str, Any] = Field(..., description="Quality rule condition")
	severity: str = Field(default="warning", description="Rule severity level")
	validation_logic: Dict[str, Any] = Field(..., description="Validation logic definition")
	error_message: str = Field(..., description="Error message template")
	suggested_fix: Optional[str] = Field(None, description="Suggested fix for violations")
	enabled: bool = Field(default=True, description="Enable rule execution")
	stop_on_violation: bool = Field(default=False, description="Stop processing on violation")
	sample_percentage: float = Field(default=100.0, ge=0.1, le=100.0, description="Percentage of data to validate")
	tags: List[str] = Field(default_factory=list, description="Quality rule tags")
	category: Optional[str] = Field(None, description="Quality rule category")
	is_public: bool = Field(default=False, description="Available to all tenants")


class QualityRuleResponse(BaseModel):
	"""Response model for quality rule data"""
	
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(..., description="Unique quality rule identifier")
	name: str = Field(..., description="Quality rule name")
	description: Optional[str] = Field(None, description="Quality rule description")
	type: QualityRuleType = Field(..., description="Type of quality rule")
	field_name: Optional[str] = Field(None, description="Target field name")
	severity: str = Field(..., description="Rule severity level")
	error_message: str = Field(..., description="Error message template")
	enabled: bool = Field(..., description="Enable rule execution")
	stop_on_violation: bool = Field(..., description="Stop processing on violation")
	sample_percentage: float = Field(..., description="Percentage of data to validate")
	tags: List[str] = Field(..., description="Quality rule tags")
	category: Optional[str] = Field(None, description="Quality rule category")
	is_public: bool = Field(..., description="Available to all tenants")
	violation_count: int = Field(..., description="Total violations detected")
	execution_count: int = Field(..., description="Total executions")
	violation_rate: float = Field(..., description="Violation rate percentage")
	last_violation: Optional[datetime] = Field(None, description="Last violation timestamp")
	created_by: str = Field(..., description="User who created the quality rule")
	created_at: datetime = Field(..., description="Creation timestamp")
	updated_at: datetime = Field(..., description="Last update timestamp")


class ExecutionResponse(BaseModel):
	"""Response model for execution data"""
	
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(..., description="Unique execution identifier")
	pipeline_id: str = Field(..., description="Associated pipeline ID")
	status: PipelineStatus = Field(..., description="Execution status")
	execution_mode: ExecutionMode = Field(..., description="Execution mode")
	triggered_by: str = Field(..., description="User or system that triggered execution")
	trigger_type: str = Field(..., description="Type of trigger")
	started_at: Optional[datetime] = Field(None, description="Execution start time")
	completed_at: Optional[datetime] = Field(None, description="Execution completion time")
	duration_ms: Optional[int] = Field(None, description="Execution duration in milliseconds")
	pipeline_version: str = Field(..., description="Pipeline version executed")
	records_processed: int = Field(..., description="Total records processed")
	records_failed: int = Field(..., description="Total records failed")
	success_rate: float = Field(..., description="Success rate percentage")
	error_message: Optional[str] = Field(None, description="Error message if failed")
	max_memory_mb: Optional[float] = Field(None, description="Peak memory usage")
	avg_cpu_percent: Optional[float] = Field(None, description="Average CPU usage")
	data_quality_score: Optional[float] = Field(None, description="Overall data quality score")
	created_at: datetime = Field(..., description="Creation timestamp")


class ExecutionLogEntry(BaseModel):
	"""Model for execution log entries"""
	
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	timestamp: datetime = Field(..., description="Log entry timestamp")
	level: str = Field(..., description="Log level (INFO, WARNING, ERROR)")
	message: str = Field(..., description="Log message")
	component: Optional[str] = Field(None, description="Component that generated log")
	step_index: Optional[int] = Field(None, description="Pipeline step index")
	context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class PipelineListResponse(BaseModel):
	"""Response model for pipeline listing"""
	
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	pipelines: List[PipelineResponse] = Field(..., description="List of pipelines")
	total: int = Field(..., description="Total number of pipelines")
	offset: int = Field(..., description="Current offset")
	limit: int = Field(..., description="Current limit")


class ExecutionListResponse(BaseModel):
	"""Response model for execution listing"""
	
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	executions: List[ExecutionResponse] = Field(..., description="List of executions")
	total: int = Field(..., description="Total number of executions")
	offset: int = Field(..., description="Current offset")
	limit: int = Field(..., description="Current limit")


class PipelineMetricsResponse(BaseModel):
	"""Response model for pipeline metrics"""
	
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	pipeline_id: str = Field(..., description="Pipeline identifier")
	total_executions: int = Field(..., description="Total number of executions")
	successful_executions: int = Field(..., description="Number of successful executions")
	failed_executions: int = Field(..., description="Number of failed executions")
	success_rate: float = Field(..., description="Overall success rate")
	avg_duration_ms: float = Field(..., description="Average execution duration")
	avg_records_processed: float = Field(..., description="Average records processed")
	last_execution: Optional[datetime] = Field(None, description="Last execution timestamp")
	last_success: Optional[datetime] = Field(None, description="Last successful execution")


class PipelineHealthResponse(BaseModel):
	"""Response model for pipeline health status"""
	
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	pipeline_id: str = Field(..., description="Pipeline identifier")
	health_status: str = Field(..., description="Overall health status")
	health_score: float = Field(..., ge=0, le=100, description="Health score (0-100)")
	checks: List[Dict[str, Any]] = Field(..., description="Individual health checks")
	recommendations: List[str] = Field(..., description="Health improvement recommendations")
	last_check: datetime = Field(..., description="Last health check timestamp")


class CollaboratorResponse(BaseModel):
	"""Response model for pipeline collaborators"""
	
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	user_id: str = Field(..., description="User identifier")
	username: str = Field(..., description="Username")
	role: str = Field(..., description="Collaboration role")
	permissions: List[str] = Field(..., description="User permissions")
	last_active: Optional[datetime] = Field(None, description="Last activity timestamp")
	status: str = Field(..., description="Collaboration status")


class PipelineOptimizationResponse(BaseModel):
	"""Response model for pipeline optimization recommendations"""
	
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	pipeline_id: str = Field(..., description="Pipeline identifier")
	performance_improvements: List[Dict[str, Any]] = Field(..., description="Performance recommendations")
	resource_optimizations: List[Dict[str, Any]] = Field(..., description="Resource optimization suggestions")
	reliability_enhancements: List[Dict[str, Any]] = Field(..., description="Reliability improvements")
	cost_optimizations: List[Dict[str, Any]] = Field(..., description="Cost optimization opportunities")
	overall_score: float = Field(..., ge=0, le=100, description="Current optimization score")
	potential_improvement: float = Field(..., ge=0, le=100, description="Potential improvement percentage")
	generated_at: datetime = Field(..., description="Recommendation generation timestamp")


class ErrorResponse(BaseModel):
	"""Standard error response model"""
	
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	error: str = Field(..., description="Error type")
	message: str = Field(..., description="Error message")
	details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
	timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
	request_id: Optional[str] = Field(None, description="Request identifier for tracking")


class SuccessResponse(BaseModel):
	"""Standard success response model"""
	
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	success: bool = Field(default=True, description="Operation success flag")
	message: str = Field(..., description="Success message")
	data: Optional[Dict[str, Any]] = Field(None, description="Additional response data")
	timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


# Validation helpers with AfterValidator

def validate_cron_expression(v: Optional[str]) -> Optional[str]:
	"""Validate cron expression format"""
	if v and len(v.split()) != 5:
		raise ValueError("Invalid cron expression format - must have 5 fields")
	return v

def validate_json_logic(v: Dict[str, Any]) -> Dict[str, Any]:
	"""Validate JSON logic structure"""
	if not v:
		raise ValueError("Logic definition cannot be empty")
	if not isinstance(v, dict):
		raise ValueError("Logic must be a JSON object")
	return v

def validate_connection_string(v: str) -> str:
	"""Validate connection string format"""
	if not v or not v.strip():
		raise ValueError("Connection string cannot be empty")
	return v.strip()

def mask_connection_string(v: str) -> str:
	"""Mask sensitive parts of connection string"""
	if '@' in v:
		parts = v.split('@')
		if len(parts) >= 2:
			credentials_part = parts[0]
			if ':' in credentials_part:
				user_pass = credentials_part.split(':')
				if len(user_pass) >= 2:
					return f"{user_pass[0]}:***@{parts[1]}"
	
	return v[:10] + "***" if len(v) > 10 else "***"


# Add validators to request models using AfterValidator
PipelineCreateRequest.model_fields['schedule_cron'] = Field(
	None, 
	description="Cron expression for scheduling",
	json_schema_extra={"validator": validate_cron_expression}
)

TransformationCreateRequest.model_fields['logic'] = Field(
	..., 
	description="Transformation logic definition",
	json_schema_extra={"validator": validate_json_logic}
)

DataSourceCreateRequest.model_fields['connection_string'] = Field(
	..., 
	description="Connection string or URL",
	json_schema_extra={"validator": validate_connection_string}
)


# Flask-AppBuilder View Utilities

class ViewHelpers:
	"""Helper functions for Flask-AppBuilder views"""
	
	@staticmethod
	def format_duration(duration_ms: Optional[int]) -> str:
		"""Format duration in milliseconds to human-readable string"""
		if not duration_ms:
			return "N/A"
		
		seconds = duration_ms / 1000
		if seconds < 60:
			return f"{seconds:.1f}s"
		elif seconds < 3600:
			return f"{seconds/60:.1f}m"
		else:
			return f"{seconds/3600:.1f}h"
	
	@staticmethod
	def format_percentage(value: Optional[float]) -> str:
		"""Format percentage value"""
		if value is None:
			return "N/A"
		return f"{value:.1f}%"
	
	@staticmethod
	def format_record_count(count: Optional[int]) -> str:
		"""Format record count with units"""
		if not count:
			return "0"
		
		if count < 1000:
			return str(count)
		elif count < 1000000:
			return f"{count/1000:.1f}K"
		else:
			return f"{count/1000000:.1f}M"
	
	@staticmethod
	def get_status_badge_class(status: PipelineStatus) -> str:
		"""Get CSS class for status badge"""
		status_classes = {
			PipelineStatus.DRAFT: "badge-secondary",
			PipelineStatus.ACTIVE: "badge-success",
			PipelineStatus.RUNNING: "badge-primary",
			PipelineStatus.PAUSED: "badge-warning",
			PipelineStatus.SUCCESS: "badge-success",
			PipelineStatus.FAILED: "badge-danger",
			PipelineStatus.CANCELLED: "badge-warning",
			PipelineStatus.SCHEDULED: "badge-info"
		}
		return status_classes.get(status, "badge-secondary")
	
	@staticmethod
	def get_health_score_class(score: Optional[float]) -> str:
		"""Get CSS class for health score"""
		if not score:
			return "text-muted"
		
		if score >= 90:
			return "text-success"
		elif score >= 70:
			return "text-warning"
		else:
			return "text-danger"
	
	@staticmethod
	def truncate_text(text: Optional[str], max_length: int = 100) -> str:
		"""Truncate text to specified length"""
		if not text:
			return ""
		
		if len(text) <= max_length:
			return text
		
		return text[:max_length-3] + "..."