"""
Intelligent Orchestration Models

Database models for workflow orchestration, task automation, decision engines,
and intelligent coordination of digital twin networks.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str
import json

from ..auth_rbac.models import BaseMixin, AuditMixin, Model


class IOWorkflow(Model, AuditMixin, BaseMixin):
	"""
	Intelligent orchestration workflow definition and management.
	
	Represents a workflow that coordinates multiple digital twins
	and automates complex business processes with intelligent decision-making.
	"""
	__tablename__ = 'io_workflow'
	
	# Identity
	workflow_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Workflow Information
	workflow_name = Column(String(200), nullable=False, index=True)
	description = Column(Text, nullable=True)
	workflow_type = Column(String(50), nullable=False, index=True)  # sequential, parallel, dag, event_driven
	category = Column(String(100), nullable=True)  # business process category
	version = Column(String(20), default='1.0.0')
	
	# Workflow Definition
	workflow_definition = Column(JSON, default=dict)  # Complete workflow DAG definition
	task_definitions = Column(JSON, default=list)  # List of task definitions
	task_dependencies = Column(JSON, default=dict)  # Task dependency graph
	data_flow_mapping = Column(JSON, default=dict)  # Data flow between tasks
	
	# Configuration
	input_schema = Column(JSON, default=dict)  # Expected input schema
	output_schema = Column(JSON, default=dict)  # Expected output schema
	configuration = Column(JSON, default=dict)  # Workflow configuration
	environment_variables = Column(JSON, default=dict)  # Environment settings
	
	# Triggers and Scheduling
	trigger_type = Column(String(50), default='manual', index=True)  # manual, scheduled, event, api
	trigger_configuration = Column(JSON, default=dict)  # Trigger settings
	schedule_expression = Column(String(200), nullable=True)  # Cron-like schedule
	event_filters = Column(JSON, default=dict)  # Event trigger filters
	
	# Status and Lifecycle
	status = Column(String(20), default='draft', index=True)  # draft, active, archived, deprecated
	is_enabled = Column(Boolean, default=True)
	is_template = Column(Boolean, default=False)  # Can be used as template
	template_category = Column(String(100), nullable=True)
	
	# Execution Metadata
	total_executions = Column(Integer, default=0)
	successful_executions = Column(Integer, default=0)
	failed_executions = Column(Integer, default=0)
	average_execution_time = Column(Float, nullable=True)  # Average execution time in minutes
	last_execution_at = Column(DateTime, nullable=True)
	
	# Intelligence and Optimization
	ai_optimization_enabled = Column(Boolean, default=False)
	learning_algorithm = Column(String(50), nullable=True)  # reinforcement, supervised, unsupervised
	optimization_metrics = Column(JSON, default=dict)  # Metrics to optimize
	performance_baseline = Column(JSON, default=dict)  # Performance baseline
	
	# Resource Management
	resource_requirements = Column(JSON, default=dict)  # Required compute resources
	estimated_cost = Column(Float, nullable=True)  # Estimated execution cost
	timeout_minutes = Column(Integer, default=60)  # Workflow timeout
	retry_policy = Column(JSON, default=dict)  # Retry configuration
	
	# Access and Security
	created_by = Column(String(36), nullable=False, index=True)
	shared_with = Column(JSON, default=list)  # List of user/group IDs with access
	security_context = Column(JSON, default=dict)  # Security settings
	
	# Quality and Monitoring
	sla_requirements = Column(JSON, default=dict)  # SLA definitions
	monitoring_configuration = Column(JSON, default=dict)  # Monitoring setup
	alerting_rules = Column(JSON, default=list)  # Alert rules
	
	# Relationships
	executions = relationship("IOWorkflowExecution", back_populates="workflow")
	tasks = relationship("IOTask", back_populates="workflow")
	
	def __repr__(self):
		return f"<IOWorkflow {self.workflow_name}>"
	
	def calculate_success_rate(self) -> float:
		"""Calculate workflow success rate"""
		if self.total_executions == 0:
			return 0.0
		return (self.successful_executions / self.total_executions) * 100
	
	def get_average_duration(self) -> float:
		"""Get average execution duration in minutes"""
		return self.average_execution_time or 0.0
	
	def is_ready_for_execution(self) -> bool:
		"""Check if workflow is ready for execution"""
		return (
			self.status == 'active' and
			self.is_enabled and
			bool(self.workflow_definition) and
			bool(self.task_definitions)
		)
	
	def estimate_execution_time(self) -> float:
		"""Estimate execution time based on task definitions"""
		total_time = 0.0
		
		# Simple estimation based on task configurations
		for task_def in self.task_definitions:
			estimated_duration = task_def.get('estimated_duration_minutes', 5.0)
			total_time += estimated_duration
		
		return total_time
	
	def validate_workflow_definition(self) -> Dict[str, Any]:
		"""Validate workflow definition for correctness"""
		issues = []
		warnings = []
		
		# Check for required fields
		if not self.workflow_definition:
			issues.append("Workflow definition is empty")
		
		if not self.task_definitions:
			issues.append("No tasks defined")
		
		# Check for circular dependencies
		if self.task_dependencies:
			# Implementation would check for cycles in dependency graph
			pass
		
		# Check for orphaned tasks
		defined_tasks = {task.get('task_id') for task in self.task_definitions}
		referenced_tasks = set()
		for deps in self.task_dependencies.values():
			referenced_tasks.update(deps)
		
		orphaned = referenced_tasks - defined_tasks
		if orphaned:
			warnings.append(f"Referenced but undefined tasks: {orphaned}")
		
		return {
			'valid': len(issues) == 0,
			'issues': issues,
			'warnings': warnings
		}


class IOWorkflowExecution(Model, AuditMixin, BaseMixin):
	"""
	Individual workflow execution instance with detailed tracking.
	
	Tracks the execution of a workflow instance including status,
	performance metrics, and execution context.
	"""
	__tablename__ = 'io_workflow_execution'
	
	# Identity
	execution_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	workflow_id = Column(String(36), ForeignKey('io_workflow.workflow_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Execution Context
	execution_name = Column(String(200), nullable=True)  # Optional name for this execution
	trigger_type = Column(String(50), nullable=False, index=True)  # manual, scheduled, event, api
	triggered_by = Column(String(36), nullable=True, index=True)  # User or system that triggered
	trigger_context = Column(JSON, default=dict)  # Context that triggered execution
	
	# Input and Output
	input_data = Column(JSON, default=dict)  # Input data for execution
	output_data = Column(JSON, default=dict)  # Output data from execution
	intermediate_data = Column(JSON, default=dict)  # Intermediate data between tasks
	
	# Status and Progress
	status = Column(String(20), default='pending', index=True)  # pending, running, completed, failed, cancelled
	current_task_id = Column(String(36), nullable=True)  # Currently executing task
	completed_tasks = Column(JSON, default=list)  # List of completed task IDs
	failed_tasks = Column(JSON, default=list)  # List of failed task IDs
	progress_percentage = Column(Float, default=0.0)
	
	# Timing
	scheduled_at = Column(DateTime, nullable=True)  # When execution was scheduled
	started_at = Column(DateTime, nullable=True, index=True)
	completed_at = Column(DateTime, nullable=True)
	duration_minutes = Column(Float, nullable=True)
	
	# Performance Metrics
	total_tasks = Column(Integer, default=0)
	completed_task_count = Column(Integer, default=0)
	failed_task_count = Column(Integer, default=0)
	skipped_task_count = Column(Integer, default=0)
	
	# Resource Usage
	cpu_time_seconds = Column(Float, default=0.0)
	memory_usage_mb = Column(Float, default=0.0)
	network_io_bytes = Column(Float, default=0.0)
	storage_io_bytes = Column(Float, default=0.0)
	
	# Error Handling
	error_message = Column(Text, nullable=True)
	error_details = Column(JSON, default=dict)  # Detailed error information
	retry_count = Column(Integer, default=0)
	max_retries = Column(Integer, default=3)
	
	# Quality and SLA
	sla_met = Column(Boolean, nullable=True)  # Whether SLA was met
	quality_score = Column(Float, nullable=True)  # 0-1 quality score
	business_impact = Column(String(20), nullable=True)  # low, medium, high, critical
	
	# Environment and Configuration
	execution_environment = Column(String(50), nullable=True)  # dev, staging, production
	workflow_version = Column(String(20), nullable=True)  # Version of workflow executed
	configuration_snapshot = Column(JSON, default=dict)  # Configuration at execution time
	
	# Cost and Billing
	estimated_cost = Column(Float, nullable=True)
	actual_cost = Column(Float, nullable=True)
	cost_breakdown = Column(JSON, default=dict)  # Detailed cost breakdown
	
	# Relationships
	workflow = relationship("IOWorkflow", back_populates="executions")
	task_executions = relationship("IOTaskExecution", back_populates="workflow_execution")
	
	def __repr__(self):
		return f"<IOWorkflowExecution {self.workflow.workflow_name} - {self.status}>"
	
	def calculate_duration(self) -> Optional[float]:
		"""Calculate execution duration in minutes"""
		if self.started_at and self.completed_at:
			duration = self.completed_at - self.started_at
			self.duration_minutes = duration.total_seconds() / 60
			return self.duration_minutes
		return None
	
	def calculate_progress(self) -> float:
		"""Calculate execution progress percentage"""
		if self.total_tasks == 0:
			return 0.0
		
		self.progress_percentage = (self.completed_task_count / self.total_tasks) * 100
		return self.progress_percentage
	
	def calculate_success_rate(self) -> float:
		"""Calculate task success rate within this execution"""
		total_attempted = self.completed_task_count + self.failed_task_count
		if total_attempted == 0:
			return 0.0
		
		return (self.completed_task_count / total_attempted) * 100
	
	def is_sla_compliant(self) -> bool:
		"""Check if execution meets SLA requirements"""
		if not self.workflow.sla_requirements:
			return True
		
		sla_duration = self.workflow.sla_requirements.get('max_duration_minutes')
		if sla_duration and self.duration_minutes and self.duration_minutes > sla_duration:
			return False
		
		sla_success_rate = self.workflow.sla_requirements.get('min_success_rate', 95.0)
		if self.calculate_success_rate() < sla_success_rate:
			return False
		
		return True
	
	def can_retry(self) -> bool:
		"""Check if execution can be retried"""
		return (
			self.status in ['failed', 'cancelled'] and
			self.retry_count < self.max_retries
		)


class IOTask(Model, AuditMixin, BaseMixin):
	"""
	Individual task definition within a workflow.
	
	Represents a single task that can be executed as part of
	a workflow with specific configuration and dependencies.
	"""
	__tablename__ = 'io_task'
	
	# Identity
	task_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	workflow_id = Column(String(36), ForeignKey('io_workflow.workflow_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Task Information
	task_name = Column(String(200), nullable=False, index=True)
	description = Column(Text, nullable=True)
	task_type = Column(String(50), nullable=False, index=True)  # data_collection, analysis, simulation, etc.
	category = Column(String(100), nullable=True)
	
	# Task Configuration
	configuration = Column(JSON, default=dict)  # Task-specific configuration
	input_mapping = Column(JSON, default=dict)  # How to map workflow data to task inputs
	output_mapping = Column(JSON, default=dict)  # How to map task outputs to workflow data
	parameters = Column(JSON, default=dict)  # Task parameters
	
	# Dependencies and Flow Control
	depends_on = Column(JSON, default=list)  # List of task IDs this task depends on
	condition_expression = Column(Text, nullable=True)  # Condition for task execution
	parallel_group = Column(String(50), nullable=True)  # Group for parallel execution
	
	# Execution Settings
	timeout_minutes = Column(Integer, default=30)
	retry_policy = Column(JSON, default=dict)  # Retry configuration
	error_handling = Column(String(50), default='fail')  # fail, continue, retry
	
	# Resource Requirements
	resource_requirements = Column(JSON, default=dict)  # Required resources
	estimated_duration_minutes = Column(Float, default=5.0)
	priority = Column(Integer, default=5)  # 1-10, higher is more priority
	
	# Digital Twin Integration
	target_twin_ids = Column(JSON, default=list)  # Target digital twin IDs
	twin_operations = Column(JSON, default=list)  # Operations to perform on twins
	data_requirements = Column(JSON, default=dict)  # Required data from twins
	
	# Quality and Validation
	validation_rules = Column(JSON, default=list)  # Output validation rules
	quality_checks = Column(JSON, default=list)  # Quality check definitions
	success_criteria = Column(JSON, default=dict)  # Success criteria
	
	# Monitoring and Alerting
	monitoring_enabled = Column(Boolean, default=True)
	alert_rules = Column(JSON, default=list)  # Alert rules for this task
	metrics_to_collect = Column(JSON, default=list)  # Metrics to collect
	
	# Position and Visual Layout
	position_x = Column(Float, nullable=True)  # X position in visual editor
	position_y = Column(Float, nullable=True)  # Y position in visual editor
	visual_properties = Column(JSON, default=dict)  # Visual properties
	
	# Relationships
	workflow = relationship("IOWorkflow", back_populates="tasks")
	executions = relationship("IOTaskExecution", back_populates="task")
	
	def __repr__(self):
		return f"<IOTask {self.task_name} ({self.task_type})>"
	
	def is_ready_to_execute(self, completed_tasks: List[str]) -> bool:
		"""Check if task is ready to execute based on dependencies"""
		if not self.depends_on:
			return True
		
		# All dependencies must be completed
		return all(dep_id in completed_tasks for dep_id in self.depends_on)
	
	def evaluate_condition(self, execution_context: Dict[str, Any]) -> bool:
		"""Evaluate condition expression for task execution"""
		if not self.condition_expression:
			return True
		
		try:
			# Implementation would safely evaluate condition expression
			# For now, return True as placeholder
			return True
		except Exception:
			return False
	
	def estimate_execution_time(self) -> float:
		"""Get estimated execution time in minutes"""
		return self.estimated_duration_minutes
	
	def validate_configuration(self) -> Dict[str, Any]:
		"""Validate task configuration"""
		issues = []
		warnings = []
		
		# Check required fields based on task type
		if self.task_type == 'api_call' and not self.configuration.get('endpoint'):
			issues.append("API endpoint not configured")
		
		if self.task_type == 'data_collection' and not self.target_twin_ids:
			issues.append("No target digital twins specified")
		
		# Check timeout
		if self.timeout_minutes <= 0:
			issues.append("Timeout must be positive")
		
		return {
			'valid': len(issues) == 0,
			'issues': issues,
			'warnings': warnings
		}


class IOTaskExecution(Model, AuditMixin, BaseMixin):
	"""
	Individual task execution instance within a workflow execution.
	
	Tracks the execution of a single task with detailed metrics,
	logs, and performance data.
	"""
	__tablename__ = 'io_task_execution'
	
	# Identity
	task_execution_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	workflow_execution_id = Column(String(36), ForeignKey('io_workflow_execution.execution_id'), nullable=False, index=True)
	task_id = Column(String(36), ForeignKey('io_task.task_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Execution Context
	execution_order = Column(Integer, nullable=True)  # Order of execution within workflow
	retry_attempt = Column(Integer, default=0)  # Which retry attempt this is
	parent_execution_id = Column(String(36), nullable=True)  # For retry tracking
	
	# Status and Progress
	status = Column(String(20), default='pending', index=True)  # pending, running, completed, failed, skipped, cancelled
	progress_percentage = Column(Float, default=0.0)
	status_message = Column(Text, nullable=True)
	
	# Input and Output
	input_data = Column(JSON, default=dict)  # Input data for task execution
	output_data = Column(JSON, default=dict)  # Output data from task execution
	processed_records = Column(Integer, nullable=True)  # Number of records processed
	
	# Timing
	scheduled_at = Column(DateTime, nullable=True)
	started_at = Column(DateTime, nullable=True, index=True)
	completed_at = Column(DateTime, nullable=True)
	duration_seconds = Column(Float, nullable=True)
	
	# Resource Usage
	cpu_usage_percent = Column(Float, nullable=True)
	memory_usage_mb = Column(Float, nullable=True)
	disk_io_bytes = Column(Float, nullable=True)
	network_io_bytes = Column(Float, nullable=True)
	
	# Digital Twin Interactions
	twins_accessed = Column(JSON, default=list)  # List of twin IDs accessed
	twin_operations_performed = Column(JSON, default=list)  # Operations performed
	data_retrieved_mb = Column(Float, nullable=True)  # Amount of data retrieved
	
	# Quality and Validation
	validation_results = Column(JSON, default=dict)  # Validation check results
	quality_score = Column(Float, nullable=True)  # 0-1 quality score
	quality_issues = Column(JSON, default=list)  # Quality issues found
	
	# Error Handling
	error_occurred = Column(Boolean, default=False)
	error_type = Column(String(100), nullable=True)
	error_message = Column(Text, nullable=True)
	error_details = Column(JSON, default=dict)  # Detailed error information
	stack_trace = Column(Text, nullable=True)
	
	# Logging and Debugging
	execution_log = Column(Text, nullable=True)  # Execution log
	debug_information = Column(JSON, default=dict)  # Debug info
	execution_environment = Column(JSON, default=dict)  # Environment details
	
	# Performance Metrics
	throughput_records_per_second = Column(Float, nullable=True)
	efficiency_score = Column(Float, nullable=True)  # Task efficiency metric
	business_value_generated = Column(Float, nullable=True)  # Business impact
	
	# Cost Tracking
	compute_cost = Column(Float, nullable=True)
	data_transfer_cost = Column(Float, nullable=True)
	total_cost = Column(Float, nullable=True)
	
	# Relationships
	workflow_execution = relationship("IOWorkflowExecution", back_populates="task_executions")
	task = relationship("IOTask", back_populates="executions")
	
	def __repr__(self):
		return f"<IOTaskExecution {self.task.task_name} - {self.status}>"
	
	def calculate_duration(self) -> Optional[float]:
		"""Calculate execution duration in seconds"""
		if self.started_at and self.completed_at:
			duration = self.completed_at - self.started_at
			self.duration_seconds = duration.total_seconds()
			return self.duration_seconds
		return None
	
	def calculate_throughput(self) -> Optional[float]:
		"""Calculate processing throughput"""
		if self.processed_records and self.duration_seconds and self.duration_seconds > 0:
			self.throughput_records_per_second = self.processed_records / self.duration_seconds
			return self.throughput_records_per_second
		return None
	
	def is_successful(self) -> bool:
		"""Check if task execution was successful"""
		return self.status == 'completed' and not self.error_occurred
	
	def calculate_efficiency(self) -> float:
		"""Calculate task execution efficiency"""
		if not self.duration_seconds or not self.task.estimated_duration_minutes:
			return 0.0
		
		estimated_seconds = self.task.estimated_duration_minutes * 60
		efficiency = min(1.0, estimated_seconds / self.duration_seconds)
		self.efficiency_score = efficiency
		return efficiency
	
	def get_performance_summary(self) -> Dict[str, Any]:
		"""Get performance summary for task execution"""
		return {
			'duration_seconds': self.duration_seconds,
			'throughput_rps': self.throughput_records_per_second,
			'efficiency_score': self.efficiency_score,
			'quality_score': self.quality_score,
			'resource_usage': {
				'cpu_percent': self.cpu_usage_percent,
				'memory_mb': self.memory_usage_mb,
				'disk_io_bytes': self.disk_io_bytes,
				'network_io_bytes': self.network_io_bytes
			},
			'cost': self.total_cost,
			'business_value': self.business_value_generated
		}


class IODecisionEngine(Model, AuditMixin, BaseMixin):
	"""
	Intelligent decision engine for workflow orchestration.
	
	Implements AI-driven decision making for dynamic workflow
	routing, resource allocation, and optimization.
	"""
	__tablename__ = 'io_decision_engine'
	
	# Identity
	engine_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Engine Information
	engine_name = Column(String(200), nullable=False, index=True)
	description = Column(Text, nullable=True)
	engine_type = Column(String(50), nullable=False, index=True)  # rule_based, ml_based, hybrid
	domain = Column(String(100), nullable=True)  # Application domain
	
	# Decision Logic
	decision_rules = Column(JSON, default=list)  # List of decision rules
	ml_model_config = Column(JSON, default=dict)  # ML model configuration
	knowledge_base = Column(JSON, default=dict)  # Knowledge base for decisions
	decision_tree = Column(JSON, default=dict)  # Decision tree structure
	
	# Learning and Adaptation
	learning_enabled = Column(Boolean, default=False)
	learning_algorithm = Column(String(50), nullable=True)  # reinforcement, supervised, etc.
	feedback_mechanism = Column(String(50), nullable=True)  # explicit, implicit, automated
	adaptation_rate = Column(Float, default=0.1)  # Rate of adaptation to new data
	
	# Performance Metrics
	total_decisions = Column(Integer, default=0)
	correct_decisions = Column(Integer, default=0)
	decision_confidence_avg = Column(Float, nullable=True)  # Average confidence score
	response_time_avg_ms = Column(Float, nullable=True)  # Average response time
	
	# Configuration
	confidence_threshold = Column(Float, default=0.7)  # Minimum confidence for decisions
	fallback_strategy = Column(String(50), default='default')  # Fallback when confidence low
	context_window_size = Column(Integer, default=100)  # Size of context window
	
	# Status and Lifecycle
	status = Column(String(20), default='active', index=True)  # active, training, inactive
	last_training_at = Column(DateTime, nullable=True)
	last_decision_at = Column(DateTime, nullable=True)
	version = Column(String(20), default='1.0.0')
	
	# Integration
	supported_workflows = Column(JSON, default=list)  # Workflows this engine supports
	integration_endpoints = Column(JSON, default=dict)  # API endpoints for integration
	
	def __repr__(self):
		return f"<IODecisionEngine {self.engine_name}>"
	
	def calculate_accuracy(self) -> float:
		"""Calculate decision accuracy percentage"""
		if self.total_decisions == 0:
			return 0.0
		return (self.correct_decisions / self.total_decisions) * 100
	
	def can_make_decision(self, context: Dict[str, Any]) -> bool:
		"""Check if engine can make decision for given context"""
		return (
			self.status == 'active' and
			bool(self.decision_rules or self.ml_model_config)
		)
	
	def update_performance_metrics(self, decision_correct: bool, confidence: float, response_time_ms: float):
		"""Update performance metrics after a decision"""
		self.total_decisions += 1
		if decision_correct:
			self.correct_decisions += 1
		
		# Update running averages
		if self.decision_confidence_avg is None:
			self.decision_confidence_avg = confidence
		else:
			self.decision_confidence_avg = (
				(self.decision_confidence_avg * (self.total_decisions - 1) + confidence) /
				self.total_decisions
			)
		
		if self.response_time_avg_ms is None:
			self.response_time_avg_ms = response_time_ms
		else:
			self.response_time_avg_ms = (
				(self.response_time_avg_ms * (self.total_decisions - 1) + response_time_ms) /
				self.total_decisions
			)
		
		self.last_decision_at = datetime.utcnow()


class IOOrchestrationRule(Model, AuditMixin, BaseMixin):
	"""
	Orchestration rules for intelligent workflow automation.
	
	Defines rules that govern how workflows are triggered,
	routed, and optimized based on various conditions.
	"""
	__tablename__ = 'io_orchestration_rule'
	
	# Identity
	rule_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Rule Information
	rule_name = Column(String(200), nullable=False, index=True)
	description = Column(Text, nullable=True)
	rule_type = Column(String(50), nullable=False, index=True)  # trigger, routing, optimization, escalation
	category = Column(String(100), nullable=True)
	
	# Rule Definition
	condition_expression = Column(Text, nullable=False)  # Condition when rule applies
	action_definition = Column(JSON, default=dict)  # Action to take when rule matches
	priority = Column(Integer, default=5)  # Rule priority (1-10)
	
	# Scope and Context
	applies_to_workflows = Column(JSON, default=list)  # Workflow IDs this rule applies to
	applies_to_tasks = Column(JSON, default=list)  # Task types this rule applies to
	context_filters = Column(JSON, default=dict)  # Context filters
	
	# Timing and Scheduling
	effective_from = Column(DateTime, nullable=True)
	effective_until = Column(DateTime, nullable=True)
	timezone = Column(String(50), default='UTC')
	schedule_expression = Column(String(200), nullable=True)  # When rule is active
	
	# Status and Lifecycle
	status = Column(String(20), default='active', index=True)  # active, inactive, draft, archived
	is_enabled = Column(Boolean, default=True)
	version = Column(String(20), default='1.0.0')
	
	# Performance and Usage
	execution_count = Column(Integer, default=0)
	success_count = Column(Integer, default=0)
	failure_count = Column(Integer, default=0)
	last_executed_at = Column(DateTime, nullable=True)
	average_execution_time_ms = Column(Float, nullable=True)
	
	# Validation and Testing
	validation_results = Column(JSON, default=dict)  # Rule validation results
	test_cases = Column(JSON, default=list)  # Test cases for rule
	impact_analysis = Column(JSON, default=dict)  # Impact analysis
	
	# Metadata
	created_by = Column(String(36), nullable=False, index=True)
	tags = Column(JSON, default=list)  # Rule tags
	custom_metadata = Column(JSON, default=dict)  # Custom metadata
	
	def __repr__(self):
		return f"<IOOrchestrationRule {self.rule_name}>"
	
	def is_active(self) -> bool:
		"""Check if rule is currently active"""
		if not self.is_enabled or self.status != 'active':
			return False
		
		now = datetime.utcnow()
		
		if self.effective_from and now < self.effective_from:
			return False
		
		if self.effective_until and now > self.effective_until:
			return False
		
		return True
	
	def calculate_success_rate(self) -> float:
		"""Calculate rule success rate"""
		if self.execution_count == 0:
			return 0.0
		return (self.success_count / self.execution_count) * 100
	
	def evaluate_condition(self, context: Dict[str, Any]) -> bool:
		"""Evaluate rule condition against context"""
		try:
			# Implementation would safely evaluate condition expression
			# For now, return True as placeholder
			return True
		except Exception:
			return False
	
	def update_execution_metrics(self, success: bool, execution_time_ms: float):
		"""Update rule execution metrics"""
		self.execution_count += 1
		if success:
			self.success_count += 1
		else:
			self.failure_count += 1
		
		# Update average execution time
		if self.average_execution_time_ms is None:
			self.average_execution_time_ms = execution_time_ms
		else:
			total_time = self.average_execution_time_ms * (self.execution_count - 1)
			self.average_execution_time_ms = (total_time + execution_time_ms) / self.execution_count
		
		self.last_executed_at = datetime.utcnow()