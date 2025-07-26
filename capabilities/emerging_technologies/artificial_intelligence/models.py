"""
AI Orchestration Models

Database models for AI workflow orchestration, model management,
provider coordination, and performance monitoring.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str
import json

from ..auth_rbac.models import BaseMixin, AuditMixin, Model


def uuid7str():
	"""Generate UUID7 string for consistent ID generation"""
	from uuid_extensions import uuid7
	return str(uuid7())


class AIModel(Model, AuditMixin, BaseMixin):
	"""
	AI model registry with capabilities and performance tracking.
	
	Stores information about available AI models, their capabilities,
	performance metrics, and provider-specific configuration.
	"""
	__tablename__ = 'ai_model'
	
	# Identity
	model_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Model Identity
	name = Column(String(200), nullable=False)
	model_key = Column(String(100), nullable=False, index=True)  # Unique identifier
	version = Column(String(50), default='1.0.0')
	provider = Column(String(50), nullable=False, index=True)  # openai, anthropic, google, azure, custom
	
	# Model Capabilities
	model_type = Column(String(50), nullable=False, index=True)  # text_generation, text_analysis, image_generation, etc.
	capabilities = Column(JSON, default=list)  # List of specific capabilities
	supported_languages = Column(JSON, default=list)  # Supported languages
	max_context_length = Column(Integer, nullable=True)
	max_output_length = Column(Integer, nullable=True)
	
	# Model Configuration
	model_endpoint = Column(String(500), nullable=True)
	api_key_required = Column(Boolean, default=True)
	configuration_schema = Column(JSON, default=dict)  # Expected configuration parameters
	default_parameters = Column(JSON, default=dict)  # Default parameter values
	
	# Performance Characteristics
	average_latency_ms = Column(Float, default=0.0)
	throughput_rpm = Column(Integer, default=0)  # Requests per minute
	cost_per_1k_tokens = Column(Float, nullable=True)
	quality_score = Column(Float, default=0.0)  # 0-100 quality rating
	
	# Availability and Status
	is_active = Column(Boolean, default=True, index=True)
	health_status = Column(String(20), default='unknown', index=True)  # healthy, degraded, unhealthy, unknown
	last_health_check = Column(DateTime, nullable=True)
	
	# Usage Statistics
	total_requests = Column(Integer, default=0)
	successful_requests = Column(Integer, default=0)
	failed_requests = Column(Integer, default=0)
	total_tokens_processed = Column(Integer, default=0)
	total_cost = Column(Float, default=0.0)
	
	# Rate Limiting
	rate_limit_rpm = Column(Integer, nullable=True)  # Requests per minute limit
	rate_limit_tpm = Column(Integer, nullable=True)  # Tokens per minute limit
	current_usage_rpm = Column(Integer, default=0)
	current_usage_tpm = Column(Integer, default=0)
	
	# Relationships
	workflow_steps = relationship("AIWorkflowStep", back_populates="model")
	executions = relationship("AIExecution", back_populates="model")
	
	def __repr__(self):
		return f"<AIModel {self.name} ({self.provider})>"
	
	def is_available(self) -> bool:
		"""Check if model is available for use"""
		return (self.is_active and 
				self.health_status in ['healthy', 'degraded'] and
				not self.is_rate_limited())
	
	def is_rate_limited(self) -> bool:
		"""Check if model is currently rate limited"""
		if self.rate_limit_rpm and self.current_usage_rpm >= self.rate_limit_rpm:
			return True
		if self.rate_limit_tpm and self.current_usage_tpm >= self.rate_limit_tpm:
			return True
		return False
	
	def calculate_success_rate(self) -> float:
		"""Calculate model success rate percentage"""
		total = self.successful_requests + self.failed_requests
		if total == 0:
			return 0.0
		return (self.successful_requests / total) * 100
	
	def update_performance_metrics(self, latency_ms: float, tokens_used: int, 
								   cost: float = None, success: bool = True) -> None:
		"""Update model performance metrics after execution"""
		self.total_requests += 1
		if success:
			self.successful_requests += 1
		else:
			self.failed_requests += 1
		
		self.total_tokens_processed += tokens_used
		
		if cost:
			self.total_cost += cost
		
		# Update average latency (exponential moving average)
		if self.average_latency_ms == 0:
			self.average_latency_ms = latency_ms
		else:
			alpha = 0.1  # Smoothing factor
			self.average_latency_ms = alpha * latency_ms + (1 - alpha) * self.average_latency_ms
	
	def can_handle_request(self, request_tokens: int = None) -> bool:
		"""Check if model can handle a request based on context length and rate limits"""
		if not self.is_available():
			return False
		
		if request_tokens and self.max_context_length and request_tokens > self.max_context_length:
			return False
		
		if self.is_rate_limited():
			return False
		
		return True


class AIWorkflow(Model, AuditMixin, BaseMixin):
	"""
	AI workflow definitions with steps and orchestration logic.
	
	Defines multi-step AI workflows with conditional logic,
	parallel processing, and error handling.
	"""
	__tablename__ = 'ai_workflow'
	
	# Identity
	workflow_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Workflow Identity
	name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	workflow_type = Column(String(50), default='sequential', index=True)  # sequential, parallel, conditional
	version = Column(String(20), default='1.0.0')
	
	# Workflow Configuration
	input_schema = Column(JSON, default=dict)  # Expected input format
	output_schema = Column(JSON, default=dict)  # Expected output format
	timeout_seconds = Column(Integer, default=300)  # 5 minutes default
	retry_policy = Column(JSON, default=dict)  # Retry configuration
	
	# Workflow Status
	is_active = Column(Boolean, default=True, index=True)
	is_template = Column(Boolean, default=False)  # Template workflows for reuse
	
	# Performance Metrics
	total_executions = Column(Integer, default=0)
	successful_executions = Column(Integer, default=0)
	failed_executions = Column(Integer, default=0)
	average_execution_time = Column(Float, default=0.0)  # seconds
	average_cost = Column(Float, default=0.0)
	
	# Usage Statistics
	last_executed = Column(DateTime, nullable=True, index=True)
	popularity_score = Column(Float, default=0.0)  # Based on usage frequency
	
	# Relationships
	steps = relationship("AIWorkflowStep", back_populates="workflow", cascade="all, delete-orphan", order_by="AIWorkflowStep.step_order")
	executions = relationship("AIWorkflowExecution", back_populates="workflow", cascade="all, delete-orphan")
	
	def __repr__(self):
		return f"<AIWorkflow {self.name} v{self.version}>"
	
	def calculate_success_rate(self) -> float:
		"""Calculate workflow success rate percentage"""
		total = self.successful_executions + self.failed_executions
		if total == 0:
			return 0.0
		return (self.successful_executions / total) * 100
	
	def update_execution_metrics(self, execution_time: float, cost: float, success: bool) -> None:
		"""Update workflow metrics after execution"""
		self.total_executions += 1
		self.last_executed = datetime.utcnow()
		
		if success:
			self.successful_executions += 1
		else:
			self.failed_executions += 1
		
		# Update average execution time (exponential moving average)
		if self.average_execution_time == 0:
			self.average_execution_time = execution_time
		else:
			alpha = 0.1
			self.average_execution_time = alpha * execution_time + (1 - alpha) * self.average_execution_time
		
		# Update average cost
		if self.average_cost == 0:
			self.average_cost = cost
		else:
			self.average_cost = alpha * cost + (1 - alpha) * self.average_cost
		
		# Update popularity score based on recent usage
		self.popularity_score = min(100.0, self.popularity_score + 1.0)
	
	def validate_input(self, input_data: Dict[str, Any]) -> List[str]:
		"""Validate input against workflow schema"""
		errors = []
		
		if not self.input_schema:
			return errors
		
		required_fields = self.input_schema.get('required', [])
		for field in required_fields:
			if field not in input_data:
				errors.append(f"Required field '{field}' is missing")
		
		properties = self.input_schema.get('properties', {})
		for field, field_schema in properties.items():
			if field in input_data:
				field_type = field_schema.get('type')
				value = input_data[field]
				
				if field_type == 'string' and not isinstance(value, str):
					errors.append(f"Field '{field}' must be a string")
				elif field_type == 'number' and not isinstance(value, (int, float)):
					errors.append(f"Field '{field}' must be a number")
				elif field_type == 'boolean' and not isinstance(value, bool):
					errors.append(f"Field '{field}' must be a boolean")
				elif field_type == 'array' and not isinstance(value, list):
					errors.append(f"Field '{field}' must be an array")
		
		return errors


class AIWorkflowStep(Model, AuditMixin, BaseMixin):
	"""
	Individual steps within AI workflows.
	
	Defines specific AI operations with model selection,
	prompt configuration, and conditional logic.
	"""
	__tablename__ = 'ai_workflow_step'
	
	# Identity
	step_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	workflow_id = Column(String(36), ForeignKey('ai_workflow.workflow_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Step Configuration
	step_name = Column(String(200), nullable=False)
	step_type = Column(String(50), nullable=False, index=True)  # ai_model, condition, parallel, custom
	step_order = Column(Integer, nullable=False)
	description = Column(Text, nullable=True)
	
	# Model Configuration
	model_id = Column(String(36), ForeignKey('ai_model.model_id'), nullable=True, index=True)
	model_parameters = Column(JSON, default=dict)  # Temperature, max_tokens, etc.
	prompt_template = Column(Text, nullable=True)
	system_prompt = Column(Text, nullable=True)
	
	# Execution Logic
	input_mapping = Column(JSON, default=dict)  # How to map inputs to this step
	output_mapping = Column(JSON, default=dict)  # How to map outputs from this step
	conditions = Column(JSON, default=dict)  # Conditional execution logic
	
	# Error Handling
	retry_count = Column(Integer, default=0)
	retry_delay_seconds = Column(Integer, default=1)
	fallback_models = Column(JSON, default=list)  # Alternative models on failure
	continue_on_error = Column(Boolean, default=False)
	
	# Step Status
	is_active = Column(Boolean, default=True)
	is_parallel = Column(Boolean, default=False)  # Can run in parallel with others
	
	# Performance Metrics
	total_executions = Column(Integer, default=0)
	successful_executions = Column(Integer, default=0)
	failed_executions = Column(Integer, default=0)
	average_execution_time = Column(Float, default=0.0)
	average_tokens_used = Column(Integer, default=0)
	
	# Relationships
	workflow = relationship("AIWorkflow", back_populates="steps")
	model = relationship("AIModel", back_populates="workflow_steps")
	
	def __repr__(self):
		return f"<AIWorkflowStep {self.step_name} (order: {self.step_order})>"
	
	def render_prompt(self, context: Dict[str, Any]) -> str:
		"""Render prompt template with context variables"""
		if not self.prompt_template:
			return ""
		
		try:
			# Simple template substitution - could use Jinja2 or similar
			rendered = self.prompt_template
			for key, value in context.items():
				placeholder = f"{{{{{key}}}}}"
				rendered = rendered.replace(placeholder, str(value))
			return rendered
		except Exception as e:
			raise ValueError(f"Error rendering prompt: {str(e)}")
	
	def should_execute(self, context: Dict[str, Any]) -> bool:
		"""Check if step should execute based on conditions"""
		if not self.is_active:
			return False
		
		if not self.conditions:
			return True
		
		# Evaluate conditions (simplified implementation)
		for condition in self.conditions.get('rules', []):
			field = condition.get('field')
			operator = condition.get('operator')
			value = condition.get('value')
			
			context_value = context.get(field)
			
			if operator == 'equals' and context_value != value:
				return False
			elif operator == 'not_equals' and context_value == value:
				return False
			elif operator == 'contains' and value not in str(context_value):
				return False
			elif operator == 'greater_than' and float(context_value) <= float(value):
				return False
			elif operator == 'less_than' and float(context_value) >= float(value):
				return False
		
		return True
	
	def update_execution_metrics(self, execution_time: float, tokens_used: int, success: bool) -> None:
		"""Update step metrics after execution"""
		self.total_executions += 1
		
		if success:
			self.successful_executions += 1
		else:
			self.failed_executions += 1
		
		# Update averages with exponential moving average
		alpha = 0.1
		
		if self.average_execution_time == 0:
			self.average_execution_time = execution_time
		else:
			self.average_execution_time = alpha * execution_time + (1 - alpha) * self.average_execution_time
		
		if self.average_tokens_used == 0:
			self.average_tokens_used = tokens_used
		else:
			self.average_tokens_used = int(alpha * tokens_used + (1 - alpha) * self.average_tokens_used)


class AIWorkflowExecution(Model, AuditMixin, BaseMixin):
	"""
	Workflow execution instances with state and results.
	
	Tracks individual workflow executions with inputs, outputs,
	step results, and performance metrics.
	"""
	__tablename__ = 'ai_workflow_execution'
	
	# Identity
	execution_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	workflow_id = Column(String(36), ForeignKey('ai_workflow.workflow_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Execution Context
	user_id = Column(String(36), nullable=True, index=True)
	session_id = Column(String(128), nullable=True, index=True)
	correlation_id = Column(String(64), nullable=True, index=True)
	
	# Execution Status
	status = Column(String(20), default='pending', index=True)  # pending, running, completed, failed, cancelled
	started_at = Column(DateTime, nullable=True, index=True)
	completed_at = Column(DateTime, nullable=True, index=True)
	
	# Execution Data
	input_data = Column(JSON, nullable=False)
	output_data = Column(JSON, nullable=True)
	context_data = Column(JSON, default=dict)  # Shared context between steps
	
	# Performance Metrics
	total_duration_seconds = Column(Float, nullable=True)
	total_tokens_used = Column(Integer, default=0)
	total_cost = Column(Float, default=0.0)
	steps_completed = Column(Integer, default=0)
	steps_failed = Column(Integer, default=0)
	
	# Error Information
	error_message = Column(Text, nullable=True)
	error_step_id = Column(String(36), nullable=True)
	error_details = Column(JSON, default=dict)
	
	# Execution Configuration
	execution_config = Column(JSON, default=dict)  # Runtime configuration overrides
	priority = Column(String(20), default='normal')  # low, normal, high, urgent
	
	# Relationships
	workflow = relationship("AIWorkflow", back_populates="executions")
	step_executions = relationship("AIStepExecution", back_populates="workflow_execution", cascade="all, delete-orphan")
	
	def __repr__(self):
		return f"<AIWorkflowExecution {self.execution_id} ({self.status})>"
	
	def is_running(self) -> bool:
		"""Check if execution is currently running"""
		return self.status in ['pending', 'running']
	
	def is_completed(self) -> bool:
		"""Check if execution completed successfully"""
		return self.status == 'completed'
	
	def get_duration(self) -> Optional[float]:
		"""Get execution duration in seconds"""
		if self.started_at and self.completed_at:
			return (self.completed_at - self.started_at).total_seconds()
		elif self.started_at:
			return (datetime.utcnow() - self.started_at).total_seconds()
		return None
	
	def start_execution(self) -> None:
		"""Mark execution as started"""
		self.status = 'running'
		self.started_at = datetime.utcnow()
	
	def complete_execution(self, output_data: Dict[str, Any]) -> None:
		"""Mark execution as completed with results"""
		self.status = 'completed'
		self.completed_at = datetime.utcnow()
		self.output_data = output_data
		
		if self.started_at:
			self.total_duration_seconds = (self.completed_at - self.started_at).total_seconds()
	
	def fail_execution(self, error_message: str, error_step_id: str = None, 
					   error_details: Dict[str, Any] = None) -> None:
		"""Mark execution as failed with error information"""
		self.status = 'failed'
		self.completed_at = datetime.utcnow()
		self.error_message = error_message
		self.error_step_id = error_step_id
		self.error_details = error_details or {}
		
		if self.started_at:
			self.total_duration_seconds = (self.completed_at - self.started_at).total_seconds()


class AIStepExecution(Model, AuditMixin, BaseMixin):
	"""
	Individual step execution results within workflow executions.
	
	Tracks step-level execution details, model responses,
	and performance metrics.
	"""
	__tablename__ = 'ai_step_execution'
	
	# Identity
	step_execution_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	workflow_execution_id = Column(String(36), ForeignKey('ai_workflow_execution.execution_id'), nullable=False, index=True)
	step_id = Column(String(36), ForeignKey('ai_workflow_step.step_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Execution Details
	status = Column(String(20), default='pending', index=True)  # pending, running, completed, failed, skipped
	started_at = Column(DateTime, nullable=True)
	completed_at = Column(DateTime, nullable=True)
	attempt_number = Column(Integer, default=1)
	
	# Model Execution
	model_used = Column(String(100), nullable=True)  # Model that was actually used
	provider_used = Column(String(50), nullable=True)
	model_parameters_used = Column(JSON, default=dict)
	
	# Input/Output
	step_input = Column(JSON, nullable=True)
	step_output = Column(JSON, nullable=True)
	prompt_sent = Column(Text, nullable=True)
	response_received = Column(Text, nullable=True)
	
	# Performance Metrics
	execution_time_ms = Column(Float, nullable=True)
	tokens_used = Column(Integer, default=0)
	cost = Column(Float, default=0.0)
	
	# Provider Response
	provider_request_id = Column(String(200), nullable=True)
	provider_response_metadata = Column(JSON, default=dict)
	
	# Error Information
	error_message = Column(Text, nullable=True)
	error_code = Column(String(50), nullable=True)
	error_details = Column(JSON, default=dict)
	
	# Relationships
	workflow_execution = relationship("AIWorkflowExecution", back_populates="step_executions")
	step = relationship("AIWorkflowStep")
	
	def __repr__(self):
		return f"<AIStepExecution {self.step_execution_id} ({self.status})>"
	
	def start_execution(self, model_used: str, provider_used: str, 
					   step_input: Dict[str, Any], prompt: str = None) -> None:
		"""Mark step execution as started"""
		self.status = 'running'
		self.started_at = datetime.utcnow()
		self.model_used = model_used
		self.provider_used = provider_used
		self.step_input = step_input
		self.prompt_sent = prompt
	
	def complete_execution(self, step_output: Dict[str, Any], response: str = None, 
						   tokens_used: int = 0, cost: float = 0.0,
						   provider_metadata: Dict[str, Any] = None) -> None:
		"""Mark step execution as completed with results"""
		self.status = 'completed'
		self.completed_at = datetime.utcnow()
		self.step_output = step_output
		self.response_received = response
		self.tokens_used = tokens_used
		self.cost = cost
		self.provider_response_metadata = provider_metadata or {}
		
		if self.started_at:
			self.execution_time_ms = (self.completed_at - self.started_at).total_seconds() * 1000
	
	def fail_execution(self, error_message: str, error_code: str = None, 
					   error_details: Dict[str, Any] = None) -> None:
		"""Mark step execution as failed with error information"""
		self.status = 'failed'
		self.completed_at = datetime.utcnow()
		self.error_message = error_message
		self.error_code = error_code
		self.error_details = error_details or {}
		
		if self.started_at:
			self.execution_time_ms = (self.completed_at - self.started_at).total_seconds() * 1000


class AIExecution(Model, AuditMixin, BaseMixin):
	"""
	Direct AI model executions outside of workflows.
	
	Tracks individual AI model calls with inputs, outputs,
	and performance metrics for ad-hoc AI operations.
	"""
	__tablename__ = 'ai_execution'
	
	# Identity
	execution_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Execution Context
	user_id = Column(String(36), nullable=True, index=True)
	session_id = Column(String(128), nullable=True, index=True)
	request_id = Column(String(64), nullable=True, index=True)
	
	# Model Information
	model_id = Column(String(36), ForeignKey('ai_model.model_id'), nullable=False, index=True)
	model_parameters = Column(JSON, default=dict)
	provider_used = Column(String(50), nullable=True)
	
	# Execution Details
	status = Column(String(20), default='pending', index=True)  # pending, running, completed, failed
	execution_type = Column(String(50), nullable=False, index=True)  # text_generation, text_analysis, etc.
	started_at = Column(DateTime, nullable=True, index=True)
	completed_at = Column(DateTime, nullable=True, index=True)
	
	# Input/Output
	input_data = Column(JSON, nullable=False)
	output_data = Column(JSON, nullable=True)
	prompt = Column(Text, nullable=True)
	response = Column(Text, nullable=True)
	
	# Performance Metrics
	execution_time_ms = Column(Float, nullable=True)
	input_tokens = Column(Integer, default=0)
	output_tokens = Column(Integer, default=0)
	total_tokens = Column(Integer, default=0)
	cost = Column(Float, default=0.0)
	
	# Provider Information
	provider_request_id = Column(String(200), nullable=True)
	provider_response_metadata = Column(JSON, default=dict)
	
	# Quality Metrics
	confidence_score = Column(Float, nullable=True)  # 0-1 confidence in result
	user_rating = Column(Integer, nullable=True)  # 1-5 user satisfaction rating
	feedback = Column(Text, nullable=True)
	
	# Error Information
	error_message = Column(Text, nullable=True)
	error_code = Column(String(50), nullable=True)
	error_details = Column(JSON, default=dict)
	
	# Relationships
	model = relationship("AIModel", back_populates="executions")
	
	def __repr__(self):
		return f"<AIExecution {self.execution_id} ({self.execution_type})>"
	
	def start_execution(self) -> None:
		"""Mark execution as started"""
		self.status = 'running'
		self.started_at = datetime.utcnow()
	
	def complete_execution(self, output_data: Dict[str, Any], response: str = None,
						   input_tokens: int = 0, output_tokens: int = 0, cost: float = 0.0,
						   confidence_score: float = None, provider_metadata: Dict[str, Any] = None) -> None:
		"""Mark execution as completed with results"""
		self.status = 'completed'
		self.completed_at = datetime.utcnow()
		self.output_data = output_data
		self.response = response
		self.input_tokens = input_tokens
		self.output_tokens = output_tokens
		self.total_tokens = input_tokens + output_tokens
		self.cost = cost
		self.confidence_score = confidence_score
		self.provider_response_metadata = provider_metadata or {}
		
		if self.started_at:
			self.execution_time_ms = (self.completed_at - self.started_at).total_seconds() * 1000
	
	def fail_execution(self, error_message: str, error_code: str = None,
					   error_details: Dict[str, Any] = None) -> None:
		"""Mark execution as failed with error information"""
		self.status = 'failed'
		self.completed_at = datetime.utcnow()
		self.error_message = error_message
		self.error_code = error_code
		self.error_details = error_details or {}
		
		if self.started_at:
			self.execution_time_ms = (self.completed_at - self.started_at).total_seconds() * 1000
	
	def get_tokens_per_second(self) -> Optional[float]:
		"""Calculate tokens processed per second"""
		if self.execution_time_ms and self.total_tokens:
			return self.total_tokens / (self.execution_time_ms / 1000)
		return None


class AIProvider(Model, AuditMixin, BaseMixin):
	"""
	AI provider configuration and monitoring.
	
	Stores configuration for AI service providers with health
	monitoring, rate limiting, and cost tracking.
	"""
	__tablename__ = 'ai_provider'
	
	# Identity
	provider_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Provider Identity
	name = Column(String(100), nullable=False)
	provider_key = Column(String(50), nullable=False, unique=True)  # openai, anthropic, etc.
	provider_type = Column(String(50), nullable=False)  # api, local, hybrid
	
	# Configuration
	api_endpoint = Column(String(500), nullable=True)
	api_key = Column(String(500), nullable=True)  # Encrypted
	api_version = Column(String(20), nullable=True)
	configuration = Column(JSON, default=dict)  # Provider-specific config
	
	# Provider Status
	is_enabled = Column(Boolean, default=True, index=True)
	is_primary = Column(Boolean, default=False)
	priority = Column(Integer, default=0)  # Lower = higher priority
	
	# Rate Limiting
	requests_per_minute = Column(Integer, nullable=True)
	tokens_per_minute = Column(Integer, nullable=True)
	requests_per_day = Column(Integer, nullable=True)
	current_rpm_usage = Column(Integer, default=0)
	current_tpm_usage = Column(Integer, default=0)
	current_daily_usage = Column(Integer, default=0)
	
	# Health Monitoring
	health_status = Column(String(20), default='unknown', index=True)  # healthy, degraded, unhealthy
	last_health_check = Column(DateTime, nullable=True)
	consecutive_failures = Column(Integer, default=0)
	uptime_percentage = Column(Float, default=0.0)
	
	# Performance Metrics
	total_requests = Column(Integer, default=0)
	successful_requests = Column(Integer, default=0)
	failed_requests = Column(Integer, default=0)
	average_response_time = Column(Float, default=0.0)  # milliseconds
	total_tokens_processed = Column(Integer, default=0)
	total_cost = Column(Float, default=0.0)
	
	# Cost Management
	cost_per_1k_input_tokens = Column(Float, nullable=True)
	cost_per_1k_output_tokens = Column(Float, nullable=True)
	monthly_budget = Column(Float, nullable=True)
	current_monthly_cost = Column(Float, default=0.0)
	
	def __repr__(self):
		return f"<AIProvider {self.name} ({self.provider_key})>"
	
	def is_healthy(self) -> bool:
		"""Check if provider is healthy and available"""
		return (self.is_enabled and 
				self.health_status in ['healthy', 'degraded'] and
				self.consecutive_failures < 3)
	
	def is_rate_limited(self) -> bool:
		"""Check if provider is currently rate limited"""
		if self.requests_per_minute and self.current_rpm_usage >= self.requests_per_minute:
			return True
		if self.tokens_per_minute and self.current_tpm_usage >= self.tokens_per_minute:
			return True
		if self.requests_per_day and self.current_daily_usage >= self.requests_per_day:
			return True
		return False
	
	def is_over_budget(self) -> bool:
		"""Check if provider is over monthly budget"""
		return (self.monthly_budget is not None and 
				self.current_monthly_cost >= self.monthly_budget)
	
	def can_handle_request(self, estimated_tokens: int = None) -> bool:
		"""Check if provider can handle a request"""
		if not self.is_healthy():
			return False
		
		if self.is_rate_limited():
			return False
		
		if self.is_over_budget():
			return False
		
		if estimated_tokens and self.tokens_per_minute:
			if self.current_tpm_usage + estimated_tokens > self.tokens_per_minute:
				return False
		
		return True
	
	def update_usage(self, tokens_used: int, cost: float, success: bool = True) -> None:
		"""Update provider usage metrics"""
		self.total_requests += 1
		self.current_rpm_usage += 1
		self.current_daily_usage += 1
		
		if tokens_used:
			self.total_tokens_processed += tokens_used
			self.current_tpm_usage += tokens_used
		
		if cost:
			self.total_cost += cost
			self.current_monthly_cost += cost
		
		if success:
			self.successful_requests += 1
			self.consecutive_failures = 0
		else:
			self.failed_requests += 1
			self.consecutive_failures += 1
	
	def calculate_success_rate(self) -> float:
		"""Calculate provider success rate percentage"""
		total = self.successful_requests + self.failed_requests
		if total == 0:
			return 0.0
		return (self.successful_requests / total) * 100
	
	def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
		"""Estimate cost for token usage"""
		total_cost = 0.0
		
		if self.cost_per_1k_input_tokens:
			total_cost += (input_tokens / 1000) * self.cost_per_1k_input_tokens
		
		if self.cost_per_1k_output_tokens:
			total_cost += (output_tokens / 1000) * self.cost_per_1k_output_tokens
		
		return total_cost


class AIContext(Model, AuditMixin, BaseMixin):
	"""
	AI conversation context and memory management.
	
	Stores conversation context, user preferences, and session
	state for context-aware AI interactions.
	"""
	__tablename__ = 'ai_context'
	
	# Identity
	context_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Context Scope
	user_id = Column(String(36), nullable=True, index=True)
	session_id = Column(String(128), nullable=True, index=True)
	context_type = Column(String(50), default='conversation', index=True)  # conversation, task, domain
	context_key = Column(String(200), nullable=True, index=True)  # Unique identifier for context
	
	# Context Content
	short_term_memory = Column(JSON, default=list)  # Recent interactions
	long_term_memory = Column(JSON, default=dict)  # Persistent knowledge
	user_preferences = Column(JSON, default=dict)  # User-specific preferences
	domain_knowledge = Column(JSON, default=dict)  # Domain-specific information
	
	# Context Metadata
	context_summary = Column(Text, nullable=True)  # AI-generated summary
	key_entities = Column(JSON, default=list)  # Important entities/concepts
	topics = Column(JSON, default=list)  # Discussion topics
	sentiment = Column(String(20), nullable=True)  # Overall sentiment
	
	# Context Management
	max_memory_items = Column(Integer, default=50)
	auto_summarize = Column(Boolean, default=True)
	context_ttl_hours = Column(Integer, default=168)  # 1 week default
	last_accessed = Column(DateTime, nullable=True, index=True)
	
	# Context Quality
	relevance_score = Column(Float, default=0.0)  # How relevant/useful the context is
	coherence_score = Column(Float, default=0.0)  # How coherent the conversation is
	engagement_level = Column(Float, default=0.0)  # User engagement level
	
	def __repr__(self):
		return f"<AIContext {self.context_id} ({self.context_type})>"
	
	def add_interaction(self, role: str, content: str, metadata: Dict[str, Any] = None) -> None:
		"""Add new interaction to short-term memory"""
		interaction = {
			'role': role,  # user, assistant, system
			'content': content,
			'timestamp': datetime.utcnow().isoformat(),
			'metadata': metadata or {}
		}
		
		if self.short_term_memory is None:
			self.short_term_memory = []
		
		self.short_term_memory.append(interaction)
		
		# Trim memory if it exceeds max items
		if len(self.short_term_memory) > self.max_memory_items:
			# Keep most recent interactions
			self.short_term_memory = self.short_term_memory[-self.max_memory_items:]
		
		self.last_accessed = datetime.utcnow()
	
	def get_context_for_ai(self, max_tokens: int = None) -> List[Dict[str, str]]:
		"""Get formatted context for AI model consumption"""
		context = []
		
		# Add system context if available
		if self.context_summary:
			context.append({
				'role': 'system',
				'content': f"Context: {self.context_summary}"
			})
		
		# Add relevant long-term memory
		if self.long_term_memory:
			relevant_memory = self._get_relevant_memory()
			if relevant_memory:
				context.append({
					'role': 'system',
					'content': f"Relevant background: {relevant_memory}"
				})
		
		# Add short-term memory (recent interactions)
		if self.short_term_memory:
			for interaction in self.short_term_memory[-10:]:  # Last 10 interactions
				context.append({
					'role': interaction['role'],
					'content': interaction['content']
				})
		
		# TODO: Implement token counting and truncation if max_tokens specified
		
		return context
	
	def _get_relevant_memory(self) -> Optional[str]:
		"""Extract relevant information from long-term memory"""
		if not self.long_term_memory:
			return None
		
		# Simple implementation - in practice would use embedding similarity
		relevant_items = []
		
		if 'preferences' in self.long_term_memory:
			relevant_items.append(f"User preferences: {self.long_term_memory['preferences']}")
		
		if 'facts' in self.long_term_memory:
			recent_facts = self.long_term_memory['facts'][-5:]  # Last 5 facts
			if recent_facts:
				relevant_items.append(f"Known facts: {'; '.join(recent_facts)}")
		
		return '; '.join(relevant_items) if relevant_items else None
	
	def update_long_term_memory(self, key: str, value: Any) -> None:
		"""Update long-term memory with new information"""
		if self.long_term_memory is None:
			self.long_term_memory = {}
		
		self.long_term_memory[key] = value
		self.last_accessed = datetime.utcnow()
	
	def is_expired(self) -> bool:
		"""Check if context has expired based on TTL"""
		if not self.last_accessed:
			return False
		
		expiry_time = self.last_accessed + timedelta(hours=self.context_ttl_hours)
		return datetime.utcnow() > expiry_time
	
	def generate_summary(self) -> None:
		"""Generate AI summary of context (placeholder - would use AI model)"""
		if not self.short_term_memory:
			return
		
		# Placeholder implementation - in practice would use AI to summarize
		interaction_count = len(self.short_term_memory)
		recent_topics = list(set(self.topics or []))[:3]
		
		summary_parts = [f"{interaction_count} interactions"]
		if recent_topics:
			summary_parts.append(f"discussing: {', '.join(recent_topics)}")
		if self.sentiment:
			summary_parts.append(f"sentiment: {self.sentiment}")
		
		self.context_summary = "; ".join(summary_parts)