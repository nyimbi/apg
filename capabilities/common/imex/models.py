"""
APG Import/Export (IMEX) Data Models

Core data models for enterprise import/export operations with APG platform integration.
All models follow APG standards: async, tabs, modern typing, UUID7 IDs, multi-tenancy.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Annotated
from uuid_extensions import uuid7str

from pydantic import BaseModel, Field, ConfigDict, AfterValidator, validator, model_validator
from pydantic import Json


class JobType(str, Enum):
	"""Import/Export job types.

	Defines the different types of data operations that can be performed
	through the APG IMEX capability. Each type represents a distinct
	operational pattern with specific processing requirements.

	Attributes:
		IMPORT: Ingesting data from external sources into APG
		EXPORT: Extracting data from APG to external destinations
		MIGRATION: Moving data between different APG environments
		SYNC: Bidirectional synchronization between APG and external systems
		TRANSFORM: Data transformation operations within APG
	"""
	IMPORT = "import"
	EXPORT = "export"
	MIGRATION = "migration"
	SYNC = "sync"
	TRANSFORM = "transform"


class JobStatus(str, Enum):
	"""Job execution status enumeration.

	Tracks the lifecycle state of import/export jobs through their
	execution pipeline. Status transitions follow a defined workflow
	with appropriate validation and error handling.

	Attributes:
		DRAFT: Job created but not yet scheduled
		SCHEDULED: Job queued for future execution
		QUEUED: Job waiting in execution queue
		RUNNING: Job actively processing data
		PAUSED: Job temporarily suspended
		COMPLETED: Job finished successfully
		FAILED: Job terminated with errors
		CANCELLED: Job manually terminated by user
	"""
	DRAFT = "draft"
	PENDING = "pending"
	SCHEDULED = "scheduled"
	QUEUED = "queued"
	RUNNING = "running"
	PAUSED = "paused"
	COMPLETED = "completed"
	FAILED = "failed"
	CANCELLED = "cancelled"


class DataFormat(str, Enum):
	"""Supported data formats for import/export operations.

	Defines the comprehensive set of data formats that can be processed
	by the APG IMEX capability. Each format has specific parsers and
	validation rules for optimal data processing.

	Attributes:
		CSV: Comma-separated values format
		JSON: JavaScript Object Notation format
		XML: Extensible Markup Language format
		PARQUET: Columnar storage format for analytics
		AVRO: Schema-based binary serialization format
		ORC: Optimized Row Columnar format
		EXCEL: Microsoft Excel spreadsheet format
		YAML: YAML Ain't Markup Language format
		TSV: Tab-separated values format
		JSONL: JSON Lines format (newline-delimited JSON)
		FIXED_WIDTH: Fixed-width column format
	"""
	CSV = "csv"
	JSON = "json"
	XML = "xml"
	PARQUET = "parquet"
	AVRO = "avro"
	ORC = "orc"
	EXCEL = "excel"
	YAML = "yaml"
	TSV = "tsv"
	JSONL = "jsonl"
	FIXED_WIDTH = "fixed_width"


class CompressionType(str, Enum):
	"""Compression algorithms supported for data files.

	Provides compression options to optimize storage and transfer
	of data files during import/export operations. Each compression
	type offers different trade-offs between compression ratio and speed.

	Attributes:
		NONE: No compression applied
		GZIP: GNU zip compression algorithm
		BZIP2: Burrows-Wheeler block compression
		LZ4: Fast compression algorithm
		SNAPPY: Google's fast compression/decompression library
		ZIP: Standard ZIP archive format
		TAR: Tape archive format
	"""
	NONE = "none"
	GZIP = "gzip"
	BZIP2 = "bzip2"
	LZ4 = "lz4"
	SNAPPY = "snappy"
	ZIP = "zip"
	TAR = "tar"


class ValidationLevel(str, Enum):
	"""Data validation strictness levels.

	Defines the rigor of validation applied to data during processing.
	Higher levels provide more thorough validation but may impact
	performance. Custom level allows user-defined validation rules.

	Attributes:
		NONE: No validation performed
		BASIC: Essential format and type validation
		STRICT: Comprehensive validation with business rules
		CUSTOM: User-defined validation criteria
	"""
	NONE = "none"
	BASIC = "basic"
	STRICT = "strict"
	CUSTOM = "custom"


class ProcessingPriority(str, Enum):
	"""Job processing priority levels.

	Determines the execution order and resource allocation for jobs
	in the processing queue. Higher priority jobs are processed first
	and may receive additional system resources.

	Attributes:
		LOW: Background processing, lowest resource allocation
		NORMAL: Standard processing priority
		HIGH: Elevated priority with increased resource allocation
		URGENT: Highest priority, immediate processing
	"""
	LOW = "low"
	NORMAL = "normal"
	HIGH = "high"
	URGENT = "urgent"


class SourceType(str, Enum):
	"""Data source types for import/export operations.

	Defines the various types of data sources that can be connected
	to and processed by the APG IMEX capability. Each source type
	requires specific connection and processing protocols.

	Attributes:
		FILE: Local or mounted file system
		DATABASE: Relational or NoSQL database connection
		API: RESTful or GraphQL API endpoint
		STREAM: Real-time data stream (Bytewax, Kinesis, etc.)
		CLOUD_STORAGE: Cloud storage services (S3, GCS, Azure Blob)
		FTP: File Transfer Protocol server
		SFTP: Secure File Transfer Protocol server
		HTTP: HTTP/HTTPS web endpoint
		WEBSOCKET: WebSocket real-time connection
	"""
	FILE = "file"
	DATABASE = "database"
	API = "api"
	STREAM = "stream"
	CLOUD_STORAGE = "cloud_storage"
	FTP = "ftp"
	SFTP = "sftp"
	HTTP = "http"
	WEBSOCKET = "websocket"


class ErrorHandlingStrategy(str, Enum):
	"""Error handling strategies for data processing.

	Defines how the system responds to errors encountered during
	data processing operations. Each strategy provides different
	trade-offs between data integrity and processing continuity.

	Attributes:
		FAIL_FAST: Stop processing immediately on first error
		SKIP_ERRORS: Skip problematic records and continue processing
		LOG_AND_CONTINUE: Log errors but continue processing
		QUARANTINE: Move problematic records to quarantine for review
		CUSTOM: User-defined error handling logic
	"""
	FAIL_FAST = "fail_fast"
	SKIP_ERRORS = "skip_errors"
	SKIP_AND_CONTINUE = "skip_and_continue"
	LOG_AND_CONTINUE = "log_and_continue"
	QUARANTINE = "quarantine"
	CUSTOM = "custom"


TargetType = SourceType


class ValidationRuleType(str, Enum):
	"""Legacy validation rule type names."""
	REQUIRED = "required"
	RANGE = "range"
	FORMAT = "format"
	PATTERN = "pattern"
	CUSTOM = "custom"


class TransformationType(str, Enum):
	"""Legacy transformation type names."""
	FIELD_MAPPING = "field_mapping"
	FILTER = "filter"
	AGGREGATE = "aggregate"
	JOIN = "join"
	CUSTOM = "custom"


def _validate_positive_int(value: int) -> int:
	"""Validate that an integer value is positive.

	Ensures that numeric configuration values like chunk_size and batch_size
	are positive integers to prevent processing errors and resource issues.

	Args:
		value: The integer value to validate

	Returns:
		The validated positive integer value

	Raises:
		ValueError: If the value is not positive (>0)
	"""
	if value <= 0:
		raise ValueError("Value must be positive")
	return value


def _validate_non_negative_float(value: float) -> float:
	"""Validate that a float value is non-negative.

	Ensures that numeric values like thresholds and percentages
	are non-negative to maintain logical consistency.

	Args:
		value: The float value to validate

	Returns:
		The validated non-negative float value

	Raises:
		ValueError: If the value is negative (<0)
	"""
	if value < 0:
		raise ValueError("Value must be non-negative")
	return value


def _validate_tenant_id(value: str) -> str:
	"""Validate tenant ID format and content.

	Ensures tenant IDs meet APG multi-tenancy requirements for
	proper data isolation and security boundaries.

	Args:
		value: The tenant ID string to validate

	Returns:
		The validated and normalized tenant ID

	Raises:
		ValueError: If the tenant ID is empty or invalid
	"""
	if not value or len(value.strip()) == 0:
		raise ValueError("Tenant ID cannot be empty")
	return value.strip()


class SourceConfig(BaseModel):
	"""Data source configuration for import operations.

	Defines comprehensive configuration for connecting to and reading from
	various data sources. Supports files, databases, APIs, streams, and
	cloud storage with flexible format and connection options.

	Attributes:
		source_type: Type of data source (file, database, API, etc.)
		connection_id: Reference to APG connection capability
		file_path: Path to source file (for file sources)
		database_config: Database connection parameters
		api_config: API endpoint and authentication configuration
		cloud_config: Cloud storage service configuration
		format: Data format (CSV, JSON, Parquet, etc.)
		compression: Compression algorithm applied to data
		encoding: Character encoding (default: utf-8)
		delimiter: Field delimiter for delimited formats
		has_header: Whether first row contains column headers
		skip_rows: Number of rows to skip at start
		chunk_size: Number of records to process per chunk
		timeout_seconds: Connection timeout in seconds
		retry_attempts: Number of retry attempts on failure
		custom_options: Additional format-specific options
	"""
	source_type: SourceType
	connection_id: str | None = None  # Reference to APG conn capability
	file_path: str | None = None
	database_config: dict[str, Any] | None = None
	api_config: dict[str, Any] | None = None
	cloud_config: dict[str, Any] | None = None
	format: DataFormat
	compression: CompressionType = CompressionType.NONE
	encoding: str = "utf-8"
	delimiter: str | None = None
	has_header: bool = True
	skip_rows: int = 0
	chunk_size: Annotated[int, AfterValidator(_validate_positive_int)] = 10000
	timeout_seconds: int = 300
	retry_attempts: int = 3
	custom_options: dict[str, Any] = Field(default_factory=dict)

	model_config = ConfigDict(extra='allow', validate_by_name=True, validate_by_alias=True)

	@model_validator(mode='after')
	def validate_source_location(self) -> 'SourceConfig':
		if self.source_type == SourceType.FILE and not self.file_path:
			raise ValueError("file_path is required for file sources")
		return self


class TargetConfig(BaseModel):
	"""Data target configuration for export operations.

	Defines comprehensive configuration for writing data to various
	target destinations. Supports files, databases, APIs, and cloud
	storage with optimized writing and error handling options.

	Attributes:
		target_type: Type of data target (file, database, API, etc.)
		connection_id: Reference to APG connection capability
		file_path: Path to target file (for file targets)
		database_config: Database connection parameters
		api_config: API endpoint and authentication configuration
		cloud_config: Cloud storage service configuration
		format: Output data format (CSV, JSON, Parquet, etc.)
		compression: Compression algorithm for output data
		encoding: Character encoding (default: utf-8)
		delimiter: Field delimiter for delimited formats
		overwrite_existing: Whether to overwrite existing data
		batch_size: Number of records to write per batch
		timeout_seconds: Connection timeout in seconds
		retry_attempts: Number of retry attempts on failure
		custom_options: Additional format-specific options
	"""
	target_type: SourceType
	connection_id: str | None = None  # Reference to APG conn capability
	file_path: str | None = None
	database_config: dict[str, Any] | None = None
	api_config: dict[str, Any] | None = None
	cloud_config: dict[str, Any] | None = None
	format: DataFormat
	compression: CompressionType = CompressionType.NONE
	encoding: str = "utf-8"
	delimiter: str | None = None
	overwrite_existing: bool = False
	batch_size: Annotated[int, AfterValidator(_validate_positive_int)] = 1000
	timeout_seconds: int = 300
	retry_attempts: int = 3
	custom_options: dict[str, Any] = Field(default_factory=dict)

	model_config = ConfigDict(extra='allow', validate_by_name=True, validate_by_alias=True)


class FieldMapping(BaseModel):
	"""Individual field mapping configuration.

	Defines how a specific field is mapped and transformed between
	source and target schemas. Supports data type conversion,
	transformation logic, and validation rules.

	Attributes:
		source_field: Name of the source field
		target_field: Name of the target field
		data_type: Target data type for conversion
		transformation: Python expression or function for transformation
		default_value: Default value if source field is missing/null
		nullable: Whether the target field can accept null values
		validation_rules: List of validation rules to apply
	"""
	source_field: str
	target_field: str
	data_type: str | None = None
	transformation: str | None = None  # Python expression or function name
	default_value: Any = None
	nullable: bool = True
	validation_rules: list[str] = Field(default_factory=list)

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)


class SchemaMapping(BaseModel):
	"""Schema mapping configuration between source and target.

	Defines comprehensive mapping rules for transforming data
	between different schemas. Supports field mapping, transformation
	scripts, and automatic field matching capabilities.

	Attributes:
		id: Unique identifier for this mapping configuration
		name: Human-readable name for the mapping
		description: Optional description of mapping purpose
		field_mappings: List of individual field mapping rules
		auto_map_similar_fields: Whether to automatically map similar field names
		ignore_extra_fields: Whether to ignore unmapped source fields
		strict_mode: Whether to enforce strict schema validation
		transformation_script: Custom Python script for complex transformations
		created_at: Timestamp when mapping was created
		updated_at: Timestamp when mapping was last updated
	"""
	id: str = Field(default_factory=uuid7str)
	name: str
	description: str | None = None
	field_mappings: list[FieldMapping]
	auto_map_similar_fields: bool = True
	ignore_extra_fields: bool = True
	strict_mode: bool = False
	transformation_script: str | None = None  # Custom Python script for complex transformations
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	created_by: str

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)


class ValidationRule(BaseModel):
	"""Data validation rule configuration.

	Defines individual validation rules that can be applied to data
	during processing. Supports various rule types including format
	validation, range checks, pattern matching, and custom logic.

	Attributes:
		id: Unique identifier for this validation rule
		name: Human-readable name for the rule
		description: Optional description of rule purpose
		rule_type: Type of validation (format, range, pattern, custom)
		field_name: Target field name (None for record-level rules)
		parameters: Configuration parameters for the rule
		error_message: Error message to display when validation fails
		severity: Severity level (warning, error, critical)
		enabled: Whether this rule is active
	"""
	id: str = Field(default_factory=uuid7str)
	name: str = ""
	description: str | None = None
	rule_type: Any  # "format", "range", "pattern", "custom", etc.
	field_name: str | None = None  # None for record-level rules
	parameters: dict[str, Any] = Field(default_factory=dict)
	error_message: str
	severity: str = "error"  # "warning", "error", "critical"
	enabled: bool = True

	model_config = ConfigDict(extra='allow', validate_by_name=True, validate_by_alias=True)

	@model_validator(mode='after')
	def sync_rule_name(self) -> 'ValidationRule':
		if not self.name:
			self.name = self.field_name or getattr(self.rule_type, 'value', str(self.rule_type))
		return self


class TransformationStep(BaseModel):
	"""Data transformation step configuration.

	Defines individual transformation steps in a data processing pipeline.
	Supports filtering, aggregation, joins, and custom transformation
	logic with configurable parameters and execution order.

	Attributes:
		id: Unique identifier for this transformation step
		name: Human-readable name for the step
		description: Optional description of transformation purpose
		step_type: Type of transformation (filter, aggregate, join, custom)
		parameters: Configuration parameters for the transformation
		script: Custom transformation script (Python code)
		enabled: Whether this step is active in the pipeline
		order: Execution order within the transformation pipeline
	"""
	id: str = Field(default_factory=uuid7str)
	name: str = ""
	description: str | None = None
	step_type: Any = ""  # "filter", "aggregate", "join", "custom", etc.
	step_name: str | None = None
	transformation_type: Any = None
	parameters: dict[str, Any] = Field(default_factory=dict)
	script: str | None = None  # Custom transformation script
	enabled: bool = True
	order: int = 0

	model_config = ConfigDict(extra='allow', validate_by_name=True, validate_by_alias=True)

	@model_validator(mode='after')
	def sync_legacy_fields(self) -> 'TransformationStep':
		if self.step_name and not self.name:
			self.name = self.step_name
		if self.transformation_type is not None and not self.step_type:
			self.step_type = self.transformation_type
		if self.step_name is None:
			self.step_name = self.name
		if self.transformation_type is None:
			self.transformation_type = self.step_type
		return self


class ScheduleConfig(BaseModel):
	"""Job scheduling configuration for automated execution.

	Defines when and how frequently jobs should be executed
	automatically. Supports cron-style scheduling with timezone
	awareness, execution limits, and failure handling.

	Attributes:
		enabled: Whether automated scheduling is active
		cron_expression: Cron syntax for schedule definition
		start_date: Earliest execution date (optional)
		end_date: Latest execution date (optional)
		timezone: Timezone for schedule interpretation
		max_runs: Maximum number of executions (optional)
		retry_on_failure: Whether to retry failed executions
		notification_on_success: Send notifications on success
		notification_on_failure: Send notifications on failure
	"""
	enabled: bool = False
	cron_expression: str | None = None
	start_date: datetime | None = None
	end_date: datetime | None = None
	timezone: str = "UTC"
	max_runs: int | None = None
	retry_on_failure: bool = True
	notification_on_success: bool = False
	notification_on_failure: bool = True

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)


class ProcessingMetrics(BaseModel):
	"""Real-time processing metrics for job execution.

	Tracks comprehensive performance and processing statistics
	during job execution. Provides insights into throughput,
	resource usage, error rates, and validation results.

	Attributes:
		records_processed: Total number of records processed
		records_successful: Number of successfully processed records
		records_failed: Number of records that failed processing
		records_skipped: Number of records skipped due to filters
		bytes_processed: Total number of bytes processed
		processing_time_seconds: Total processing time in seconds
		throughput_records_per_second: Record processing rate
		throughput_bytes_per_second: Byte processing rate
		memory_usage_mb: Current memory usage in megabytes
		cpu_usage_percent: Current CPU usage percentage
		error_summary: Error counts by error type
		validation_summary: Validation results by rule type
		last_updated: Timestamp of last metrics update
	"""
	records_processed: int = 0
	records_successful: int = 0
	records_failed: int = 0
	records_skipped: int = 0
	bytes_processed: int = 0
	processing_time_seconds: Annotated[float, AfterValidator(_validate_non_negative_float)] = 0.0
	throughput_records_per_second: Annotated[float, AfterValidator(_validate_non_negative_float)] = 0.0
	throughput_bytes_per_second: Annotated[float, AfterValidator(_validate_non_negative_float)] = 0.0
	memory_usage_mb: Annotated[float, AfterValidator(_validate_non_negative_float)] = 0.0
	cpu_usage_percent: Annotated[float, AfterValidator(_validate_non_negative_float)] = 0.0
	error_summary: dict[str, int] = Field(default_factory=dict)
	validation_summary: dict[str, int] = Field(default_factory=dict)
	last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)


class JobExecution(BaseModel):
	"""Job execution tracking and status management.

	Tracks individual execution instances of import/export jobs.
	Maintains execution state, performance metrics, error details,
	and execution environment information for comprehensive monitoring.

	Attributes:
		id: Unique identifier for this execution instance
		job_id: Reference to the parent ImportExportJob
		execution_number: Sequential execution number for this job
		status: Current execution status
		started_at: Execution start timestamp
		completed_at: Execution completion timestamp
		error_message: High-level error message if execution failed
		error_details: Detailed error information and stack traces
		metrics: Real-time processing metrics
		log_file_path: Path to detailed execution logs
		worker_node: Identifier of worker node executing the job
		execution_config: Runtime configuration used for execution
	"""
	id: str = Field(default_factory=uuid7str)
	job_id: str
	execution_number: int = 1
	status: JobStatus = JobStatus.QUEUED
	started_at: datetime | None = Field(default_factory=lambda: datetime.now(timezone.utc))
	completed_at: datetime | None = None
	started_by: str | None = None
	error_message: str | None = None
	error_details: dict[str, Any] | None = None
	metrics: ProcessingMetrics = Field(default_factory=ProcessingMetrics)
	log_file_path: str | None = None
	worker_node: str | None = None
	execution_config: dict[str, Any] = Field(default_factory=dict)

	model_config = ConfigDict(extra='allow', validate_by_name=True, validate_by_alias=True)


class ImportExportJob(BaseModel):
	"""Main import/export job entity for APG IMEX operations.

	Represents a complete data import/export job with comprehensive
	configuration, status tracking, and APG platform integration.
	Supports complex data transformations, validation, scheduling,
	and multi-tenant execution.

	Attributes:
		id: Unique UUID7 identifier for the job
		tenant_id: Multi-tenant isolation identifier
		name: Human-readable job name
		description: Optional detailed description
		job_type: Type of operation (import, export, migration, etc.)
		priority: Processing priority level

		source_config: Configuration for data source connection
		target_config: Configuration for data target connection
		schema_mapping: Schema transformation rules
		validation_rules: Data validation rules to apply
		transformation_steps: Data transformation pipeline steps
		schedule_config: Automated scheduling configuration

		validation_level: Strictness of data validation
		error_handling: Strategy for handling processing errors
		parallel_processing: Whether to enable parallel execution
		max_workers: Maximum number of parallel workers
		memory_limit_mb: Memory limit for processing in MB
		timeout_minutes: Maximum execution time in minutes

		status: Current job status
		current_execution: Active execution details
		execution_history: List of previous execution IDs
		last_run_at: Timestamp of last execution
		next_run_at: Timestamp of next scheduled execution

		tags: Searchable tags for job organization
		created_by: User who created the job
		created_at: Job creation timestamp
		updated_by: User who last updated the job
		updated_at: Last update timestamp

		etlp_pipeline_id: Reference to APG ETLP pipeline
		audit_trail_id: Reference to APG audit compliance trail
		notification_config: Notification settings for job events
	"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: Annotated[str, AfterValidator(_validate_tenant_id)]
	name: str
	description: str | None = None
	job_type: JobType
	priority: ProcessingPriority = ProcessingPriority.NORMAL

	# Configuration
	source_config: SourceConfig
	target_config: TargetConfig
	schema_mapping: SchemaMapping | None = None
	validation_rules: list[ValidationRule] = Field(default_factory=list)
	transformation_steps: list[TransformationStep] = Field(default_factory=list)
	schedule_config: ScheduleConfig | None = None

	# Processing Options
	validation_level: ValidationLevel = ValidationLevel.BASIC
	error_handling: ErrorHandlingStrategy = ErrorHandlingStrategy.LOG_AND_CONTINUE
	parallel_processing: bool = True
	max_workers: int = 4
	memory_limit_mb: int | None = None
	timeout_minutes: int = 60

	# Status and Tracking
	status: JobStatus = JobStatus.DRAFT
	current_execution: JobExecution | None = None
	execution_history: list[str] = Field(default_factory=list)  # List of execution IDs
	last_run_at: datetime | None = None
	next_run_at: datetime | None = None

	# Metadata
	tags: list[str] = Field(default_factory=list)
	created_by: str = "system"
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_by: str | None = None
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

	# APG Integration
	etlp_pipeline_id: str | None = None  # Reference to APG etlp pipeline
	audit_trail_id: str | None = None    # Reference to APG audit_compliance trail
	notification_config: dict[str, Any] = Field(default_factory=dict)
	metadata: dict[str, Any] = Field(default_factory=dict)

	model_config = ConfigDict(extra='allow', validate_by_name=True, validate_by_alias=True)

	@validator('updated_at', pre=True, always=True)
	def set_updated_at(cls, v):
		return datetime.now(timezone.utc)


class DataQualityReport(BaseModel):
	"""Data quality assessment report for processed data.

	Provides comprehensive analysis of data quality metrics including
	completeness, consistency, accuracy, and anomaly detection.
	Generated automatically during data processing operations.

	Attributes:
		id: Unique identifier for this quality report
		job_id: Reference to the associated ImportExportJob
		execution_id: Reference to the specific job execution

		total_records: Total number of records processed
		valid_records: Number of records passing all validations
		invalid_records: Number of records failing validations
		completeness_score: Data completeness score (0-100)
		consistency_score: Data consistency score (0-100)
		accuracy_score: Data accuracy score (0-100)
		overall_quality_score: Overall quality score (0-100)

		validation_issues: Count of issues by validation rule
		field_quality_scores: Quality scores by field name
		anomalies_detected: List of detected data anomalies
		recommendations: Automated quality improvement suggestions

		generated_at: Report generation timestamp
		generated_by: System or user that generated the report
	"""
	id: str = Field(default_factory=uuid7str)
	job_id: str
	execution_id: str

	# Quality Metrics
	total_records: int
	valid_records: int
	invalid_records: int
	completeness_score: Annotated[float, AfterValidator(_validate_non_negative_float)]
	consistency_score: Annotated[float, AfterValidator(_validate_non_negative_float)]
	accuracy_score: Annotated[float, AfterValidator(_validate_non_negative_float)]
	overall_quality_score: Annotated[float, AfterValidator(_validate_non_negative_float)]

	# Issue Breakdown
	validation_issues: dict[str, int] = Field(default_factory=dict)
	field_quality_scores: dict[str, float] = Field(default_factory=dict)
	anomalies_detected: list[dict[str, Any]] = Field(default_factory=list)
	recommendations: list[str] = Field(default_factory=list)

	# Metadata
	generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	generated_by: str = "system"

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)


class WorkflowStep(BaseModel):
	"""Individual step in a data processing workflow.

	Represents a single operation within a complex data processing
	workflow. Supports dependency management, retry logic, and
	flexible configuration for various step types.

	Attributes:
		id: Unique identifier for this workflow step
		name: Human-readable name for the step
		description: Optional description of step purpose
		step_type: Type of operation (import, export, transform, validate, notify)
		configuration: Step-specific configuration parameters
		dependencies: List of prerequisite step IDs
		enabled: Whether this step is active in the workflow
		retry_count: Number of retry attempts on failure
	"""
	id: str = Field(default_factory=uuid7str)
	name: str
	description: str | None = None
	step_type: str  # "import", "export", "transform", "validate", "notify"
	configuration: dict[str, Any] = Field(default_factory=dict)
	dependencies: list[str] = Field(default_factory=list)  # IDs of prerequisite steps
	enabled: bool = True
	retry_count: int = 0
	timeout_minutes: int = 30

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)


class Workflow(BaseModel):
	"""Data processing workflow definition for complex operations.

	Defines multi-step data processing workflows with dependency
	management, parallel execution, and comprehensive error handling.
	Enables orchestration of complex data operations across multiple steps.

	Attributes:
		id: Unique identifier for this workflow
		tenant_id: Multi-tenant isolation identifier
		name: Human-readable workflow name
		description: Optional detailed description
		version: Workflow version for change management

		steps: List of workflow steps to execute
		schedule_config: Automated scheduling configuration
		parallel_execution: Whether to execute steps in parallel when possible
		error_handling: Strategy for handling step failures

		status: Current workflow status
		last_execution_id: Reference to most recent execution
		execution_history: List of previous execution IDs

		tags: Searchable tags for workflow organization
		created_by: User who created the workflow
		created_at: Workflow creation timestamp
		updated_by: User who last updated the workflow
		updated_at: Last update timestamp
	"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: Annotated[str, AfterValidator(_validate_tenant_id)]
	name: str
	description: str | None = None
	version: str = "1.0.0"

	# Workflow Definition
	steps: list[WorkflowStep]
	schedule_config: ScheduleConfig | None = None
	parallel_execution: bool = False
	error_handling: ErrorHandlingStrategy = ErrorHandlingStrategy.FAIL_FAST

	# Status and Execution
	status: JobStatus = JobStatus.DRAFT
	last_execution_id: str | None = None
	execution_history: list[str] = Field(default_factory=list)

	# Metadata
	tags: list[str] = Field(default_factory=list)
	created_by: str
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_by: str | None = None
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)


class ConnectionTemplate(BaseModel):
	"""Reusable connection template for common data source patterns.

	Provides reusable configuration templates for frequently used
	data connections. Enables standardization and rapid setup
	of common source and target configurations.

	Attributes:
		id: Unique identifier for this connection template
		tenant_id: Multi-tenant isolation identifier
		name: Human-readable template name
	"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: Annotated[str, AfterValidator(_validate_tenant_id)]
	name: str
	description: str | None = None
	category: str  # "database", "cloud_storage", "api", etc.

	# Template Configuration
	source_template: dict[str, Any] = Field(default_factory=dict)
	target_template: dict[str, Any] = Field(default_factory=dict)
	schema_mapping_template: dict[str, Any] | None = None
	validation_template: list[dict[str, Any]] = Field(default_factory=list)

	# Usage Statistics
	usage_count: int = 0
	last_used_at: datetime | None = None

	# Metadata
	is_public: bool = False
	created_by: str
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)


class MonitoringAlert(BaseModel):
	"""System monitoring and alerting configuration.

	Defines alert rules for monitoring system metrics and job performance.
	Supports threshold-based alerting with automated notifications and
	remediation actions for proactive system management.

	Attributes:
		id: Unique identifier for this alert configuration
		tenant_id: Multi-tenant isolation identifier
		name: Human-readable alert name
		description: Optional description of alert purpose

		metric_name: Name of the metric to monitor
		threshold_value: Threshold value that triggers the alert
		comparison_operator: Comparison operator (gt, lt, eq, ne, gte, lte)
		evaluation_window_minutes: Time window for metric evaluation

		notification_channels: List of notification channels to use
		webhook_urls: List of webhook URLs for alert notifications
		auto_remediation_script: Optional script to run for auto-remediation

		enabled: Whether this alert is active
		last_triggered_at: Timestamp of last alert trigger
		trigger_count: Number of times this alert has been triggered

		created_by: User who created the alert
		created_at: Alert creation timestamp
	"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: Annotated[str, AfterValidator(_validate_tenant_id)]
	name: str
	description: str | None = None

	# Alert Conditions
	metric_name: str
	threshold_value: float
	comparison_operator: str  # "gt", "lt", "eq", "ne", "gte", "lte"
	evaluation_window_minutes: int = 5

	# Alert Actions
	notification_channels: list[str] = Field(default_factory=list)
	webhook_urls: list[str] = Field(default_factory=list)
	auto_remediation_script: str | None = None

	# Status
	enabled: bool = True
	last_triggered_at: datetime | None = None
	trigger_count: int = 0

	# Metadata
	created_by: str
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)


# Model registry for APG composition engine
model_registry = {
	"ImportExportJob": ImportExportJob,
	"JobExecution": JobExecution,
	"SourceConfig": SourceConfig,
	"TargetConfig": TargetConfig,
	"SchemaMapping": SchemaMapping,
	"ValidationRule": ValidationRule,
	"TransformationStep": TransformationStep,
	"ProcessingMetrics": ProcessingMetrics,
	"DataQualityReport": DataQualityReport,
	"Workflow": Workflow,
	"WorkflowStep": WorkflowStep,
	"ConnectionTemplate": ConnectionTemplate,
	"MonitoringAlert": MonitoringAlert
}


def _enum_value_str(self: Enum) -> str:
	return str(self.value)


for _enum_class in (
	JobType, JobStatus, DataFormat, CompressionType, ValidationLevel,
	ProcessingPriority, SourceType, ErrorHandlingStrategy,
	ValidationRuleType, TransformationType
):
	_enum_class.__str__ = _enum_value_str

__all__ = [
	# Enums
	"JobType", "JobStatus", "DataFormat", "CompressionType",
	"ValidationLevel", "ProcessingPriority", "SourceType", "TargetType",
	"ErrorHandlingStrategy", "ValidationRuleType", "TransformationType",

	# Core Models
	"ImportExportJob", "JobExecution", "SourceConfig", "TargetConfig",
	"SchemaMapping", "FieldMapping", "ValidationRule", "TransformationStep",
	"ScheduleConfig", "ProcessingMetrics", "DataQualityReport",

	# Workflow Models
	"Workflow", "WorkflowStep",

	# Utility Models
	"ConnectionTemplate", "MonitoringAlert",

	# Registry
	"model_registry"
]
