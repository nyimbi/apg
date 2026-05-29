"""
APG AI Core Framework (aicr) - Data Models

Purpose: Comprehensive data models for AI service orchestration, model lifecycle
         management, and intelligent automation within APG platform ecosystem.
Dependencies: pydantic, typing, datetime, enum
Usage Context: Core AI infrastructure for all APG AI capabilities

This module provides production-grade data models with:
- Multi-tenant AI service isolation and security
- Model lifecycle tracking with version control
- Performance metrics and optimization data
- Integration with APG composition engine
- Comprehensive audit trails and compliance
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Annotated, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict, AfterValidator
from pydantic.types import Json


def uuid7str() -> str:
	"""Generate UUID7 string for identifiers."""
	return str(uuid4())


def _validate_positive_int(value: int) -> int:
	"""Validate that an integer value is positive."""
	if value <= 0:
		raise ValueError("Value must be positive")
	return value


def _validate_non_negative_float(value: float) -> float:
	"""Validate that a float value is non-negative."""
	if value < 0:
		raise ValueError("Value must be non-negative")
	return value


def _validate_tenant_id(value: str) -> str:
	"""Validate tenant ID format for APG multi-tenancy."""
	if not value or len(value.strip()) == 0:
		raise ValueError("Tenant ID cannot be empty")
	return value.strip()


class AIServiceType(str, Enum):
	"""AI service types supported by the framework.

	Defines the different categories of AI services that can be registered
	and orchestrated within the APG AI Core Framework. Each type represents
	a distinct AI capability with specific processing requirements.

	Attributes:
		INFERENCE: Real-time model inference and prediction services
		TRAINING: Model training and fine-tuning services
		PREPROCESSING: Data preprocessing and feature engineering
		POSTPROCESSING: Output processing and result formatting
		EMBEDDING: Vector embedding generation and similarity search
		GENERATION: Content generation and synthesis services
		CLASSIFICATION: Classification and categorization tasks
		DETECTION: Object, anomaly, and pattern detection
		OPTIMIZATION: Model optimization and compression services
		ORCHESTRATION: Multi-model workflow orchestration
	"""
	INFERENCE = "inference"
	TRAINING = "training"
	PREPROCESSING = "preprocessing"
	POSTPROCESSING = "postprocessing"
	EMBEDDING = "embedding"
	GENERATION = "generation"
	CLASSIFICATION = "classification"
	DETECTION = "detection"
	OPTIMIZATION = "optimization"
	ORCHESTRATION = "orchestration"


class AIModelFramework(str, Enum):
	"""AI model frameworks supported by the inference engine.

	Defines the machine learning frameworks that can be executed
	within the AI Core Framework. Each framework has optimized
	execution paths and specific performance characteristics.

	Attributes:
		PYTORCH: PyTorch models with dynamic computation graphs
		TENSORFLOW: TensorFlow models with static graphs
		ONNX: ONNX format for cross-platform compatibility
		OLLAMA: Ollama models for local language model serving
		SCIKIT_LEARN: Scikit-learn models for traditional ML
		XGBOOST: XGBoost models for gradient boosting
		HUGGINGFACE: Hugging Face transformers and models
		CUSTOM: Custom or proprietary model formats
	"""
	PYTORCH = "pytorch"
	TENSORFLOW = "tensorflow"
	ONNX = "onnx"
	OLLAMA = "ollama"
	SCIKIT_LEARN = "scikit_learn"
	XGBOOST = "xgboost"
	HUGGINGFACE = "huggingface"
	CUSTOM = "custom"


class AIServiceStatus(str, Enum):
	"""AI service lifecycle status enumeration.

	Tracks the operational state of AI services throughout their
	lifecycle from deployment to decommissioning. Status transitions
	follow a defined workflow with health monitoring and validation.

	Attributes:
		REGISTERING: Service being registered with the framework
		INITIALIZING: Service startup and initialization in progress
		HEALTHY: Service operational and accepting requests
		DEGRADED: Service operational but with reduced performance
		UNHEALTHY: Service experiencing errors or failures
		MAINTENANCE: Service temporarily offline for maintenance
		SCALING: Service scaling up or down resources
		UPDATING: Service being updated to new version
		DECOMMISSIONING: Service being removed from framework
		FAILED: Service failed and requires intervention
	"""
	REGISTERING = "registering"
	INITIALIZING = "initializing"
	HEALTHY = "healthy"
	DEGRADED = "degraded"
	UNHEALTHY = "unhealthy"
	MAINTENANCE = "maintenance"
	SCALING = "scaling"
	UPDATING = "updating"
	DECOMMISSIONING = "decommissioning"
	FAILED = "failed"


class AIJobPriority(str, Enum):
	"""AI job processing priority levels.

	Determines the execution order and resource allocation for AI jobs
	in the processing queue. Higher priority jobs receive more resources
	and are processed with lower latency.

	Attributes:
		LOW: Background processing with minimal resource allocation
		NORMAL: Standard priority for routine AI operations
		HIGH: Elevated priority for important business operations
		CRITICAL: Highest priority for time-sensitive operations
		REALTIME: Real-time processing with guaranteed latency
	"""
	LOW = "low"
	NORMAL = "normal"
	HIGH = "high"
	CRITICAL = "critical"
	REALTIME = "realtime"


class AIResourceType(str, Enum):
	"""AI resource types for workload optimization.

	Defines the computational resources that can be allocated
	for AI workloads. Each type has specific performance
	characteristics and cost implications.

	Attributes:
		CPU: CPU-based processing for general computation
		GPU: GPU acceleration for parallel processing
		TPU: Tensor Processing Units for ML optimization
		NEUROMORPHIC: Neuromorphic processors for spike-based computing
		EDGE: Edge computing resources for local processing
		CLOUD: Cloud-based elastic computing resources
	"""
	CPU = "cpu"
	GPU = "gpu"
	TPU = "tpu"
	NEUROMORPHIC = "neuromorphic"
	EDGE = "edge"
	CLOUD = "cloud"


class AIModelMetadata(BaseModel):
	"""Comprehensive metadata for AI models.

	Contains detailed information about AI models including performance
	characteristics, resource requirements, and compatibility information
	for optimal deployment and execution planning.

	Attributes:
		model_name: Human-readable name of the model
		model_version: Semantic version of the model
		framework: ML framework used for the model
		model_size_mb: Size of model files in megabytes
		input_shape: Expected input tensor shape
		output_shape: Expected output tensor shape
		data_types: Supported input/output data types
		license: Model license and usage restrictions
		description: Detailed description of model capabilities
		tags: Searchable tags for model discovery
		performance_metrics: Benchmark performance metrics
		resource_requirements: Computational resource needs
		supported_hardware: Compatible hardware accelerators
		optimization_flags: Model optimization settings
	"""
	model_name: str
	model_version: str
	framework: AIModelFramework
	model_size_mb: Annotated[float, AfterValidator(_validate_non_negative_float)]
	input_shape: Optional[list[int]] = None
	output_shape: Optional[list[int]] = None
	data_types: list[str] = Field(default_factory=list)
	license: Optional[str] = None
	description: Optional[str] = None
	tags: list[str] = Field(default_factory=list)
	performance_metrics: dict[str, Any] = Field(default_factory=dict)
	resource_requirements: dict[str, Any] = Field(default_factory=dict)
	supported_hardware: list[str] = Field(default_factory=list)
	optimization_flags: dict[str, Any] = Field(default_factory=dict)

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)


class AIServiceRegistration(BaseModel):
	"""AI service registration for the service registry.

	Defines the registration information for AI services within the
	APG AI Core Framework. Includes service metadata, capabilities,
	and integration requirements for composition engine.

	Attributes:
		id: Unique identifier for the service registration
		tenant_id: Multi-tenant isolation identifier
		service_name: Human-readable service name
		service_type: Type of AI service being registered
		version: Semantic version of the service
		endpoint_url: Base URL for service API endpoints
		health_check_url: URL for service health monitoring
		capabilities: List of AI capabilities provided
		input_formats: Supported input data formats
		output_formats: Supported output data formats
		authentication_required: Whether authentication is required
		rate_limits: Service rate limiting configuration
		sla_requirements: Service level agreement requirements
		dependencies: Other services this service depends on
		metadata: Additional service metadata
		created_at: Service registration timestamp
		updated_at: Last update timestamp
		created_by: User who registered the service
	"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: Annotated[str, AfterValidator(_validate_tenant_id)]
	service_name: str
	service_type: AIServiceType
	version: str
	endpoint_url: str
	health_check_url: str
	capabilities: list[str]
	input_formats: list[str]
	output_formats: list[str]
	authentication_required: bool = True
	rate_limits: dict[str, int] = Field(default_factory=dict)
	sla_requirements: dict[str, Any] = Field(default_factory=dict)
	dependencies: list[str] = Field(default_factory=list)
	metadata: dict[str, Any] = Field(default_factory=dict)
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	created_by: str

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)


class AIModelArtifact(BaseModel):
	"""AI model artifact information and storage details.

	Represents individual model artifacts including files, weights,
	and associated resources. Provides versioning, storage, and
	access control for model components.

	Attributes:
		id: Unique identifier for the model artifact
		model_id: Reference to the parent model registration
		artifact_type: Type of artifact (weights, config, tokenizer, etc.)
		file_path: Storage path for the artifact file
		file_size_bytes: Size of the artifact file in bytes
		checksum: File integrity checksum (SHA-256)
		compression_type: Compression algorithm used
		encryption_enabled: Whether artifact is encrypted
		access_permissions: Access control permissions
		download_url: Secure download URL for the artifact
		metadata: Additional artifact metadata
		created_at: Artifact creation timestamp
		expires_at: Optional expiration timestamp
	"""
	id: str = Field(default_factory=uuid7str)
	model_id: str
	artifact_type: str
	file_path: str
	file_size_bytes: Annotated[int, AfterValidator(_validate_positive_int)]
	checksum: str
	compression_type: Optional[str] = None
	encryption_enabled: bool = False
	access_permissions: list[str] = Field(default_factory=list)
	download_url: Optional[str] = None
	metadata: dict[str, Any] = Field(default_factory=dict)
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	expires_at: Optional[datetime] = None

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)


class AIInferenceRequest(BaseModel):
	"""AI inference request specification.

	Defines the structure for AI inference requests including input data,
	processing parameters, and execution preferences. Supports both
	synchronous and asynchronous inference patterns.

	Attributes:
		id: Unique identifier for the inference request
		tenant_id: Multi-tenant isolation identifier
		service_id: Target AI service for inference
		model_id: Specific model to use for inference
		input_data: Input data for the inference
		parameters: Inference parameters and configuration
		priority: Processing priority for the request
		timeout_seconds: Maximum execution time allowed
		callback_url: URL for async result notification
		metadata: Additional request metadata
		created_at: Request creation timestamp
		requested_by: User who made the request
	"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: Annotated[str, AfterValidator(_validate_tenant_id)]
	service_id: str
	model_id: Optional[str] = None
	input_data: dict[str, Any]
	parameters: dict[str, Any] = Field(default_factory=dict)
	priority: AIJobPriority = AIJobPriority.NORMAL
	timeout_seconds: Annotated[int, AfterValidator(_validate_positive_int)] = 300
	callback_url: Optional[str] = None
	metadata: dict[str, Any] = Field(default_factory=dict)
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	requested_by: str

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)


class AIInferenceResult(BaseModel):
	"""AI inference result with performance metrics.

	Contains the results of AI inference operations including predictions,
	confidence scores, and detailed performance metrics for monitoring
	and optimization purposes.

	Attributes:
		id: Unique identifier for the inference result
		request_id: Reference to the original inference request
		service_id: AI service that processed the request
		model_id: Model used for the inference
		predictions: Primary inference results/predictions
		confidence_scores: Confidence levels for predictions
		probabilities: Probability distributions (for classification)
		explanations: Explainability information for decisions
		performance_metrics: Detailed performance measurements
		processing_time_ms: Total processing time in milliseconds
		queue_time_ms: Time spent waiting in queue
		resource_usage: Computational resources consumed
		status: Result status (success, partial, failed)
		error_message: Error details if processing failed
		warnings: Non-fatal warnings during processing
		metadata: Additional result metadata
		created_at: Result generation timestamp
	"""
	id: str = Field(default_factory=uuid7str)
	request_id: str
	service_id: str
	model_id: Optional[str] = None
	predictions: dict[str, Any]
	confidence_scores: dict[str, float] = Field(default_factory=dict)
	probabilities: dict[str, list[float]] = Field(default_factory=dict)
	explanations: dict[str, Any] = Field(default_factory=dict)
	performance_metrics: dict[str, float] = Field(default_factory=dict)
	processing_time_ms: Annotated[float, AfterValidator(_validate_non_negative_float)]
	queue_time_ms: Annotated[float, AfterValidator(_validate_non_negative_float)]
	resource_usage: dict[str, Any] = Field(default_factory=dict)
	status: str
	error_message: Optional[str] = None
	warnings: list[str] = Field(default_factory=list)
	metadata: dict[str, Any] = Field(default_factory=dict)
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)


class AIServiceHealth(BaseModel):
	"""AI service health status and metrics.

	Comprehensive health information for AI services including operational
	status, performance metrics, and resource utilization for monitoring
	and auto-scaling decisions.

	Attributes:
		service_id: AI service identifier
		status: Current operational status
		last_check: Timestamp of last health check
		response_time_ms: Average response time in milliseconds
		success_rate: Success rate percentage (0-100)
		active_requests: Number of currently processing requests
		queue_length: Number of requests waiting in queue
		cpu_usage_percent: CPU utilization percentage
		memory_usage_percent: Memory utilization percentage
		gpu_usage_percent: GPU utilization percentage (if applicable)
		disk_usage_percent: Disk space utilization percentage
		network_throughput_mbps: Network throughput in Mbps
		error_rate: Error rate percentage (0-100)
		availability_percent: Service availability percentage
		resource_limits: Configured resource limits
		alerts: Active alerts and warnings
		recommendations: Auto-generated optimization recommendations
		metadata: Additional health metadata
	"""
	service_id: str
	status: AIServiceStatus
	last_check: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	response_time_ms: Annotated[float, AfterValidator(_validate_non_negative_float)]
	success_rate: Annotated[float, AfterValidator(_validate_non_negative_float)]
	active_requests: Annotated[int, AfterValidator(_validate_positive_int)]
	queue_length: Annotated[int, AfterValidator(_validate_positive_int)]
	cpu_usage_percent: Annotated[float, AfterValidator(_validate_non_negative_float)]
	memory_usage_percent: Annotated[float, AfterValidator(_validate_non_negative_float)]
	gpu_usage_percent: Optional[Annotated[float, AfterValidator(_validate_non_negative_float)]] = None
	disk_usage_percent: Annotated[float, AfterValidator(_validate_non_negative_float)]
	network_throughput_mbps: Annotated[float, AfterValidator(_validate_non_negative_float)]
	error_rate: Annotated[float, AfterValidator(_validate_non_negative_float)]
	availability_percent: Annotated[float, AfterValidator(_validate_non_negative_float)]
	resource_limits: dict[str, Any] = Field(default_factory=dict)
	alerts: list[str] = Field(default_factory=list)
	recommendations: list[str] = Field(default_factory=list)
	metadata: dict[str, Any] = Field(default_factory=dict)

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)


class AIWorkflowStep(BaseModel):
	"""Individual step in an AI processing workflow.

	Represents a single operation within a complex AI workflow,
	supporting sequential and parallel execution patterns with
	dependency management and error handling.

	Attributes:
		id: Unique identifier for the workflow step
		workflow_id: Reference to the parent workflow
		step_name: Human-readable name for the step
		step_type: Type of AI operation performed
		service_id: AI service to execute the step
		input_mapping: Mapping from workflow data to step inputs
		output_mapping: Mapping from step outputs to workflow data
		parameters: Step-specific parameters and configuration
		dependencies: List of prerequisite step IDs
		parallel_group: Group ID for parallel execution
		retry_count: Number of retry attempts on failure
		timeout_seconds: Maximum execution time for the step
		enabled: Whether this step is active in the workflow
		metadata: Additional step metadata
		created_at: Step creation timestamp
	"""
	id: str = Field(default_factory=uuid7str)
	workflow_id: str
	step_name: str
	step_type: AIServiceType
	service_id: str
	input_mapping: dict[str, str] = Field(default_factory=dict)
	output_mapping: dict[str, str] = Field(default_factory=dict)
	parameters: dict[str, Any] = Field(default_factory=dict)
	dependencies: list[str] = Field(default_factory=list)
	parallel_group: Optional[str] = None
	retry_count: int = 0
	timeout_seconds: Annotated[int, AfterValidator(_validate_positive_int)] = 300
	enabled: bool = True
	metadata: dict[str, Any] = Field(default_factory=dict)
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)


class AIWorkflow(BaseModel):
	"""AI workflow definition for multi-step processing.

	Defines complex AI processing workflows that orchestrate multiple
	AI services in sequential or parallel patterns. Supports conditional
	execution, error handling, and optimization strategies.

	Attributes:
		id: Unique identifier for the workflow
		tenant_id: Multi-tenant isolation identifier
		workflow_name: Human-readable workflow name
		description: Detailed description of workflow purpose
		version: Semantic version of the workflow
		steps: List of workflow steps to execute
		input_schema: JSON schema for workflow inputs
		output_schema: JSON schema for workflow outputs
		execution_mode: Sequential or parallel execution mode
		error_handling: Error handling strategy
		optimization_enabled: Whether to enable automatic optimization
		max_parallel_steps: Maximum number of parallel steps
		total_timeout_seconds: Maximum execution time for entire workflow
		retry_policy: Retry configuration for failed steps
		tags: Searchable tags for workflow organization
		created_by: User who created the workflow
		created_at: Workflow creation timestamp
		updated_at: Last update timestamp
		metadata: Additional workflow metadata
	"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: Annotated[str, AfterValidator(_validate_tenant_id)]
	workflow_name: str
	description: Optional[str] = None
	version: str = "1.0.0"
	steps: list[AIWorkflowStep]
	input_schema: dict[str, Any] = Field(default_factory=dict)
	output_schema: dict[str, Any] = Field(default_factory=dict)
	execution_mode: str = "sequential"
	error_handling: str = "fail_fast"
	optimization_enabled: bool = True
	max_parallel_steps: Annotated[int, AfterValidator(_validate_positive_int)] = 4
	total_timeout_seconds: Annotated[int, AfterValidator(_validate_positive_int)] = 3600
	retry_policy: dict[str, Any] = Field(default_factory=dict)
	tags: list[str] = Field(default_factory=list)
	created_by: str
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	metadata: dict[str, Any] = Field(default_factory=dict)

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)


class AIAuditEvent(BaseModel):
	"""Audit event for AI operations and compliance.

	Comprehensive audit logging for all AI operations within the
	framework, supporting compliance requirements and security
	monitoring with detailed event tracking.

	Attributes:
		id: Unique identifier for the audit event
		tenant_id: Multi-tenant isolation identifier
		event_type: Type of AI operation being audited
		event_action: Specific action performed
		resource_type: Type of resource affected
		resource_id: Identifier of the affected resource
		user_id: User who performed the action
		session_id: Session identifier for the action
		timestamp: Precise timestamp of the event
		source_ip: IP address of the request origin
		user_agent: User agent string of the client
		request_data: Input data for the operation
		response_data: Output data from the operation
		success: Whether the operation succeeded
		error_message: Error details if operation failed
		processing_time_ms: Time taken to process the operation
		resource_usage: Computational resources consumed
		compliance_tags: Tags for regulatory compliance
		risk_level: Risk assessment level for the operation
		metadata: Additional audit metadata
	"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: Annotated[str, AfterValidator(_validate_tenant_id)]
	event_type: str
	event_action: str
	resource_type: str
	resource_id: str
	user_id: str
	session_id: Optional[str] = None
	timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	source_ip: Optional[str] = None
	user_agent: Optional[str] = None
	request_data: dict[str, Any] = Field(default_factory=dict)
	response_data: dict[str, Any] = Field(default_factory=dict)
	success: bool
	error_message: Optional[str] = None
	processing_time_ms: Annotated[float, AfterValidator(_validate_non_negative_float)]
	resource_usage: dict[str, Any] = Field(default_factory=dict)
	compliance_tags: list[str] = Field(default_factory=list)
	risk_level: str = "low"
	metadata: dict[str, Any] = Field(default_factory=dict)

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)


class ModelType(str, Enum):
	"""Legacy AICR model categories used by tests and public docs."""
	CLASSIFICATION = "classification"
	REGRESSION = "regression"
	CLUSTERING = "clustering"
	ANOMALY_DETECTION = "anomaly_detection"
	TIME_SERIES = "time_series"
	NLP = "nlp"
	COMPUTER_VISION = "computer_vision"
	RECOMMENDATION = "recommendation"
	REINFORCEMENT_LEARNING = "reinforcement_learning"


class InferenceStatus(str, Enum):
	"""Legacy inference lifecycle statuses."""
	PENDING = "pending"
	RUNNING = "running"
	COMPLETED = "completed"
	FAILED = "failed"
	CANCELLED = "cancelled"


class PipelineStatus(str, Enum):
	"""Legacy pipeline lifecycle statuses."""
	PENDING = "pending"
	RUNNING = "running"
	COMPLETED = "completed"
	FAILED = "failed"
	PAUSED = "paused"
	CANCELLED = "cancelled"


class MetricType(str, Enum):
	"""Legacy monitoring metric types."""
	COUNTER = "counter"
	GAUGE = "gauge"
	HISTOGRAM = "histogram"
	SUMMARY = "summary"


class AICRCapabilityBase(BaseModel):
	"""Compatibility base model for AICR capability-local records."""
	capability_id: str = Field(default_factory=uuid7str)
	name: str
	description: str
	version: str = "1.0.0"
	is_active: bool = True
	tags: list[str] = Field(default_factory=list)
	metadata: dict[str, Any] = Field(default_factory=dict)
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)


class AICRModel(BaseModel):
	"""Legacy AI model record retained for AICR service/API compatibility."""
	model_id: str = Field(default_factory=uuid7str)
	name: str
	description: str
	model_type: ModelType
	framework: str
	version: str = "1.0.0"
	status: str = "inactive"
	input_schema: dict[str, Any] = Field(default_factory=dict)
	output_schema: dict[str, Any] = Field(default_factory=dict)
	configuration: dict[str, Any] = Field(default_factory=dict)
	performance_metrics: dict[str, Any] = Field(default_factory=dict)
	file_path: Optional[str] = None
	deployment_count: int = 0
	last_inference: Optional[datetime] = None
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	metadata: dict[str, Any] = Field(default_factory=dict)

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)


class AICRInferenceRequest(BaseModel):
	"""Legacy inference request accepted by the AICR facade."""
	request_id: str = Field(default_factory=uuid7str)
	model_id: str
	input_data: dict[str, Any]
	parameters: dict[str, Any] = Field(default_factory=dict)
	output_format: str = "json"
	priority: str = "normal"
	timeout_seconds: int = 30
	metadata: dict[str, Any] = Field(default_factory=dict)
	timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)


class AICRInferenceResponse(BaseModel):
	"""Legacy inference response returned by the AICR facade."""
	response_id: str = Field(default_factory=uuid7str)
	request_id: str
	model_id: str
	status: InferenceStatus
	predictions: Optional[dict[str, Any]] = None
	confidence_scores: list[float] = Field(default_factory=list)
	processing_time_ms: float = 0.0
	error_message: Optional[str] = None
	metadata: dict[str, Any] = Field(default_factory=dict)
	timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)


class AICRPipeline(BaseModel):
	"""Legacy pipeline model used by AICR tests, API, and docs."""
	pipeline_id: str = Field(default_factory=uuid7str)
	name: str
	description: str
	pipeline_type: str
	stages: list[str] = Field(min_length=1)
	configuration: dict[str, Any] = Field(default_factory=dict)
	data_sources: list[str] = Field(default_factory=list)
	schedule: Optional[str] = None
	status: PipelineStatus = PipelineStatus.PENDING
	execution_count: int = 0
	success_rate: float = 0.0
	last_execution: Optional[datetime] = None
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	metadata: dict[str, Any] = Field(default_factory=dict)

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)


class AICRMetric(BaseModel):
	"""Legacy monitoring metric model used by AICR monitoring tests."""
	metric_id: str = Field(default_factory=uuid7str)
	metric_name: str
	metric_type: MetricType
	value: float
	source_component: str
	labels: dict[str, str] = Field(default_factory=dict)
	unit: str = ""
	description: str = ""
	timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	metadata: dict[str, Any] = Field(default_factory=dict)

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)


class AICRServiceRecord(BaseModel):
	"""Tenant-scoped AI service registration used by package governance."""

	id: str
	tenant_id: Annotated[str, AfterValidator(_validate_tenant_id)]
	name: str
	owner: str
	service_type: str = "inference"
	endpoint: str = "local://inference"
	health: str = "healthy"
	model_policy: dict[str, Any] = Field(default_factory=dict)
	status: str = "active"

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)


class AICRInferenceApproval(BaseModel):
	"""Governed approval record for high-risk or large-context inference."""

	id: str
	tenant_id: Annotated[str, AfterValidator(_validate_tenant_id)]
	service_id: str
	requested_by: str
	prompt_summary: str
	context_tokens: int = 0
	workflow_risk: str = "normal"
	decision: str = "pending"
	reviewer: str | None = None
	notes: str | None = None

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)


class AICRGovernanceEvent(BaseModel):
	"""Tenant-scoped evidence event for AICR governance changes."""

	id: str = Field(default_factory=uuid7str)
	tenant_id: Annotated[str, AfterValidator(_validate_tenant_id)]
	event_type: str
	subject_id: str
	message: str
	evidence: dict[str, Any] = Field(default_factory=dict)
	timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)


# Model registry for APG composition engine
model_registry = {
	"AIServiceRegistration": AIServiceRegistration,
	"AIModelMetadata": AIModelMetadata,
	"AIModelArtifact": AIModelArtifact,
	"AIInferenceRequest": AIInferenceRequest,
	"AIInferenceResult": AIInferenceResult,
	"AIServiceHealth": AIServiceHealth,
	"AIWorkflowStep": AIWorkflowStep,
	"AIWorkflow": AIWorkflow,
	"AIAuditEvent": AIAuditEvent,
	"AICRCapabilityBase": AICRCapabilityBase,
	"AICRModel": AICRModel,
	"AICRInferenceRequest": AICRInferenceRequest,
	"AICRInferenceResponse": AICRInferenceResponse,
	"AICRPipeline": AICRPipeline,
	"AICRMetric": AICRMetric,
	"AICRServiceRecord": AICRServiceRecord,
	"AICRInferenceApproval": AICRInferenceApproval,
	"AICRGovernanceEvent": AICRGovernanceEvent,
}

__all__ = [
	# Enums
	"AIServiceType", "AIModelFramework", "AIServiceStatus", "AIJobPriority", "AIResourceType",
	"ModelType", "InferenceStatus", "PipelineStatus", "MetricType",

	# Core Models
	"AIServiceRegistration", "AIModelMetadata", "AIModelArtifact",
	"AIInferenceRequest", "AIInferenceResult", "AIServiceHealth",
	"AICRCapabilityBase", "AICRModel", "AICRInferenceRequest",
	"AICRInferenceResponse", "AICRPipeline", "AICRMetric",
	"AICRServiceRecord", "AICRInferenceApproval", "AICRGovernanceEvent",

	# Workflow Models
	"AIWorkflowStep", "AIWorkflow",

	# Audit and Compliance
	"AIAuditEvent",

	# Registry
	"model_registry"
]
