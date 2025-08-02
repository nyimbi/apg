"""
APG Workflow Orchestration Models

Comprehensive data models for workflow orchestration with APG integration,
multi-tenancy, audit trails, and advanced workflow capabilities.

© 2025 Datacraft. All rights reserved.
Author: APG Development Team
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union, Literal
from enum import Enum
from uuid_extensions import uuid7str
from pydantic import BaseModel, Field, ConfigDict, AfterValidator, field_validator
from dataclasses import dataclass
import json

# APG Integration - Real implementations
from typing import TYPE_CHECKING
import structlog

# APG Capability Integrations
try:
	from capabilities.common.auth_rbac.service import AuthRBACService
	from capabilities.common.audit_compliance.service import AuditComplianceService
	APG_INTEGRATION_AVAILABLE = True
except ImportError:
	# Fallback implementations for development/testing
	APG_INTEGRATION_AVAILABLE = False
	
	class AuthRBACService:
		@staticmethod
		async def get_user_permissions(user_id: str, tenant_id: str = None) -> List[str]:
			"""Fallback implementation for user permissions"""
			return ["workflow:read", "workflow:write", "workflow:execute", "workflow:admin"]
		
		@staticmethod
		async def check_permission(user_id: str, permission: str, tenant_id: str = None) -> bool:
			"""Fallback permission check"""
			return True
	
	class AuditComplianceService:
		@staticmethod
		async def log_audit_event(event_type: str, user_id: str, resource_id: str, 
								  action: str, result: str, details: Dict[str, Any] = None,
								  tenant_id: str = None) -> str:
			"""Fallback audit logging"""
			logger = structlog.get_logger(__name__)
			logger.info(f"Audit: {event_type} - {action} by {user_id} on {resource_id}: {result}")
			return uuid7str()

# Initialize services
auth_rbac_service = AuthRBACService()
audit_compliance_service = AuditComplianceService()

class WorkflowStatus(str, Enum):
	"""Comprehensive workflow execution status states."""
	DRAFT = "draft"
	ACTIVE = "active" 
	PAUSED = "paused"
	SUSPENDED = "suspended"
	COMPLETED = "completed"
	FAILED = "failed"
	CANCELLED = "cancelled"
	ARCHIVED = "archived"

class TaskStatus(str, Enum):
	"""Individual task execution status states."""
	PENDING = "pending"
	READY = "ready"
	ASSIGNED = "assigned"
	IN_PROGRESS = "in_progress"
	COMPLETED = "completed"
	FAILED = "failed"
	SKIPPED = "skipped"
	CANCELLED = "cancelled"
	ESCALATED = "escalated"
	EXPIRED = "expired"

class TaskType(str, Enum):
	"""Types of workflow tasks with specific execution patterns."""
	AUTOMATED = "automated"		# System-executed tasks
	HUMAN = "human"				# Human-assigned tasks
	APPROVAL = "approval"		# Approval/decision tasks
	NOTIFICATION = "notification"	# Notification/alert tasks
	INTEGRATION = "integration"	# External system integration
	CONDITIONAL = "conditional"	# Conditional logic gates
	PARALLEL = "parallel"		# Parallel execution container
	SUBPROCESS = "subprocess"	# Sub-workflow execution
	TIMER = "timer"				# Time-based delay tasks
	SCRIPT = "script"			# Custom script execution

class TriggerType(str, Enum):
	"""Workflow trigger mechanisms."""
	MANUAL = "manual"			# Manual user initiation
	SCHEDULED = "scheduled"		# Cron/time-based triggers
	EVENT = "event"				# APG capability events
	API = "api"					# REST/GraphQL API calls
	WEBHOOK = "webhook"			# External webhook calls
	CONDITION = "condition"		# Data condition triggers
	FILE = "file"				# File system events
	EMAIL = "email"				# Email-based triggers

class Priority(int, Enum):
	"""Task and workflow priority levels."""
	LOWEST = 1
	LOW = 3
	NORMAL = 5
	HIGH = 7
	HIGHEST = 9
	CRITICAL = 10

class EscalationAction(str, Enum):
	"""Escalation action types."""
	NOTIFY = "notify"
	REASSIGN = "reassign"
	ESCALATE = "escalate"
	TIMEOUT = "timeout"
	FAIL = "fail"

def validate_cron_expression(expression: str) -> str:
	"""Validate cron expression format."""
	if not expression or not isinstance(expression, str):
		raise ValueError("Cron expression must be a non-empty string")
	
	parts = expression.split()
	if len(parts) not in [5, 6]:  # Standard or with seconds
		raise ValueError("Cron expression must have 5 or 6 parts")
	
	return expression

def validate_json_data(data: Any) -> Any:
	"""Validate JSON serializable data."""
	try:
		json.dumps(data)
		return data
	except (TypeError, ValueError) as e:
		raise ValueError(f"Data must be JSON serializable: {e}")

@dataclass
class WorkflowMetrics:
	"""Workflow execution metrics."""
	total_executions: int = 0
	successful_executions: int = 0
	failed_executions: int = 0
	average_duration_seconds: float = 0.0
	last_execution_at: Optional[datetime] = None
	success_rate: float = 0.0

class WorkflowTemplate(BaseModel):
	"""Reusable workflow template definition."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Unique template identifier")
	name: str = Field(..., min_length=1, max_length=200, description="Template display name")
	description: str = Field(default="", max_length=1000, description="Template description")
	category: str = Field(default="general", max_length=100, description="Template category")
	tags: List[str] = Field(default_factory=list, description="Search tags")
	industry: Optional[str] = Field(None, max_length=100, description="Industry specialization")
	complexity_level: Literal["beginner", "intermediate", "advanced", "expert"] = Field(
		default="intermediate", description="Template complexity level"
	)
	estimated_duration_hours: Optional[float] = Field(None, ge=0, description="Estimated completion time")
	template_data: Dict[str, Any] = Field(
		default_factory=dict, 
		description="Template workflow definition"
	)
	variables: Dict[str, Any] = Field(default_factory=dict, description="Template variables")
	prerequisites: List[str] = Field(default_factory=list, description="Required prerequisites")
	outcomes: List[str] = Field(default_factory=list, description="Expected outcomes")
	created_by: str = Field(..., description="Template creator user ID")
	tenant_id: str = Field(..., description="Tenant identifier")
	is_public: bool = Field(default=False, description="Public template availability")
	is_certified: bool = Field(default=False, description="Certified by APG")
	version: str = Field(default="1.0.0", description="Template version")
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	usage_count: int = Field(default=0, ge=0, description="Number of times used")

class TaskDefinition(BaseModel):
	"""Individual task definition within a workflow."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Unique task identifier")
	name: str = Field(..., min_length=1, max_length=200, description="Task display name")
	description: str = Field(default="", max_length=1000, description="Task description")
	task_type: TaskType = Field(..., description="Type of task execution")
	
	# Assignment and ownership
	assigned_to: Optional[str] = Field(None, description="User ID or role assignment")
	assigned_role: Optional[str] = Field(None, description="Role-based assignment")
	assigned_group: Optional[str] = Field(None, description="Group assignment")
	
	# Scheduling and timing
	estimated_duration_minutes: Optional[int] = Field(None, ge=0, description="Estimated task duration")
	due_date_offset_hours: Optional[int] = Field(None, ge=0, description="Due date offset from start")
	timeout_minutes: Optional[int] = Field(None, ge=1, description="Task timeout period")
	
	# Priority and SLA
	priority: Priority = Field(default=Priority.NORMAL, description="Task priority level")
	sla_hours: Optional[int] = Field(None, ge=1, description="SLA completion time")
	is_critical: bool = Field(default=False, description="Critical path task flag")
	
	# Dependencies and conditions
	dependencies: List[str] = Field(default_factory=list, description="Prerequisite task IDs")
	conditions: List[Dict[str, Any]] = Field(
		default_factory=list, 
		description="Execution conditions"
	)
	skip_conditions: List[Dict[str, Any]] = Field(
		default_factory=list,
		description="Conditions to skip this task"
	)
	
	# Task configuration
	configuration: Dict[str, Any] = Field(
		default_factory=dict,
		description="Task-specific configuration"
	)
	input_parameters: Dict[str, Any] = Field(
		default_factory=dict,
		description="Input parameters"
	)
	output_parameters: List[str] = Field(
		default_factory=list,
		description="Expected output parameter names"
	)
	
	# Error handling and retry
	max_retry_attempts: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
	retry_delay_seconds: int = Field(default=60, ge=1, description="Delay between retries")
	continue_on_failure: bool = Field(default=False, description="Continue workflow on task failure")
	
	# Escalation rules
	escalation_rules: List[Dict[str, Any]] = Field(
		default_factory=list,
		description="Task escalation configuration"
	)
	
	# Notifications
	notification_rules: List[Dict[str, Any]] = Field(
		default_factory=list,
		description="Task notification configuration"
	)
	
	# Audit and metadata
	metadata: Dict[str, Any] = Field(
		default_factory=dict,
		description="Additional task metadata"
	)
	tags: List[str] = Field(default_factory=list, description="Task tags")
	
	# Position in visual designer
	position_x: Optional[float] = Field(None, description="Visual designer X position")
	position_y: Optional[float] = Field(None, description="Visual designer Y position")

class WorkflowTrigger(BaseModel):
	"""Workflow execution trigger configuration."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Unique trigger identifier")
	name: str = Field(..., min_length=1, max_length=200, description="Trigger name")
	trigger_type: TriggerType = Field(..., description="Type of trigger")
	is_enabled: bool = Field(default=True, description="Trigger active status")
	
	# Scheduling configuration
	cron_expression: Optional[str] = Field(
		None, 
		description="Cron expression for scheduled triggers"
	)
	schedule_timezone: str = Field(default="UTC", description="Timezone for scheduling")
	
	# Event-based triggers
	event_source: Optional[str] = Field(None, description="Event source capability")
	event_types: List[str] = Field(default_factory=list, description="Event types to listen for")
	event_filters: Dict[str, Any] = Field(
		default_factory=dict,
		description="Event filtering criteria"
	)
	
	# API/Webhook triggers
	webhook_url: Optional[str] = Field(None, description="Webhook endpoint URL")
	api_endpoints: List[str] = Field(default_factory=list, description="API endpoint paths")
	authentication_required: bool = Field(default=True, description="Require authentication")
	
	# Condition-based triggers
	condition_expression: Optional[str] = Field(None, description="Condition evaluation expression")
	condition_data_source: Optional[str] = Field(None, description="Data source for conditions")
	
	# File-based triggers
	file_patterns: List[str] = Field(default_factory=list, description="File patterns to monitor")
	file_directories: List[str] = Field(default_factory=list, description="Directories to monitor")
	
	# Trigger configuration
	configuration: Dict[str, Any] = Field(
		default_factory=dict,
		description="Trigger-specific configuration"
	)
	
	# Rate limiting
	max_executions_per_hour: Optional[int] = Field(None, ge=1, description="Rate limiting")
	concurrent_execution_limit: Optional[int] = Field(None, ge=1, description="Concurrent executions")
	
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	
	@field_validator('cron_expression')
	@classmethod
	def validate_cron(cls, v: Optional[str]) -> Optional[str]:
		if v is not None:
			return validate_cron_expression(v)
		return v

class Workflow(BaseModel):
	"""Complete workflow definition with APG integration."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	# Core identification
	id: str = Field(default_factory=uuid7str, description="Unique workflow identifier")
	name: str = Field(..., min_length=1, max_length=200, description="Workflow display name")
	description: str = Field(default="", max_length=2000, description="Workflow description")
	version: str = Field(default="1.0.0", description="Workflow version")
	
	# APG multi-tenant integration
	tenant_id: str = Field(..., description="APG tenant identifier")
	created_by: str = Field(..., description="Creator user ID")
	owner_id: str = Field(..., description="Current owner user ID")
	team_id: Optional[str] = Field(None, description="Owning team identifier")
	
	# Workflow definition
	tasks: List[TaskDefinition] = Field(..., min_items=1, description="Workflow tasks")
	triggers: List[WorkflowTrigger] = Field(default_factory=list, description="Workflow triggers")
	
	# Classification and organization
	category: str = Field(default="general", max_length=100, description="Workflow category")
	tags: List[str] = Field(default_factory=list, description="Workflow tags")
	priority: Priority = Field(default=Priority.NORMAL, description="Workflow priority")
	
	# State and lifecycle
	status: WorkflowStatus = Field(default=WorkflowStatus.DRAFT, description="Workflow status")
	is_template: bool = Field(default=False, description="Template workflow flag")
	is_published: bool = Field(default=False, description="Published status")
	is_public: bool = Field(default=False, description="Public visibility")
	
	# Execution configuration
	max_concurrent_instances: int = Field(default=10, ge=1, le=1000, description="Concurrent execution limit")
	default_timeout_hours: Optional[int] = Field(None, ge=1, description="Default workflow timeout")
	auto_retry_failed: bool = Field(default=False, description="Auto-retry failed workflows")
	max_retry_attempts: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
	
	# Variables and parameters
	input_parameters: Dict[str, Any] = Field(
		default_factory=dict,
		description="Workflow input parameters schema"
	)
	output_parameters: Dict[str, Any] = Field(
		default_factory=dict,
		description="Workflow output parameters schema"
	)
	variables: Dict[str, Any] = Field(default_factory=dict, description="Workflow variables")
	
	# SLA and performance
	sla_hours: Optional[int] = Field(None, ge=1, description="Workflow SLA")
	estimated_duration_hours: Optional[float] = Field(None, ge=0, description="Estimated duration")
	
	# Notifications and escalations
	notification_settings: Dict[str, Any] = Field(
		default_factory=dict,
		description="Notification configuration"
	)
	escalation_settings: Dict[str, Any] = Field(
		default_factory=dict,
		description="Escalation configuration"
	)
	
	# APG capability integrations
	required_capabilities: List[str] = Field(
		default_factory=list,
		description="Required APG capabilities"
	)
	integration_settings: Dict[str, Any] = Field(
		default_factory=dict,
		description="APG integration configuration"
	)
	
	# Audit and compliance
	compliance_requirements: List[str] = Field(
		default_factory=list,
		description="Compliance requirements"
	)
	audit_level: Literal["none", "basic", "detailed", "full"] = Field(
		default="basic",
		description="Audit logging level"
	)
	
	# Analytics and monitoring
	metrics: Optional[WorkflowMetrics] = Field(None, description="Workflow metrics")
	monitoring_enabled: bool = Field(default=True, description="Enable monitoring")
	
	# Timestamps and versioning
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	published_at: Optional[datetime] = Field(None, description="Publication timestamp")
	archived_at: Optional[datetime] = Field(None, description="Archive timestamp")
	
	# Soft delete support
	is_deleted: bool = Field(default=False, description="Soft delete flag")
	deleted_at: Optional[datetime] = Field(None, description="Deletion timestamp")
	deleted_by: Optional[str] = Field(None, description="User who deleted")
	
	# Metadata and custom fields
	metadata: Dict[str, Any] = Field(
		default_factory=dict,
		description="Additional workflow metadata"
	)
	
	@field_validator('tasks')
	@classmethod
	def validate_tasks(cls, v: List[TaskDefinition]) -> List[TaskDefinition]:
		"""Validate task definitions and dependencies."""
		if not v:
			raise ValueError("Workflow must have at least one task")
		
		task_ids = {task.id for task in v}
		
		# Validate dependencies reference existing tasks
		for task in v:
			for dep_id in task.dependencies:
				if dep_id not in task_ids:
					raise ValueError(f"Task {task.id} has invalid dependency: {dep_id}")
		
		return v
	
	@field_validator('variables', 'metadata', 'integration_settings')
	@classmethod
	def validate_json_fields(cls, v: Dict[str, Any]) -> Dict[str, Any]:
		return validate_json_data(v)

class WorkflowInstance(BaseModel):
	"""Active workflow execution instance."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	# Core identification
	id: str = Field(default_factory=uuid7str, description="Unique instance identifier")
	workflow_id: str = Field(..., description="Parent workflow ID")
	workflow_version: str = Field(..., description="Executed workflow version")
	
	# APG multi-tenant context
	tenant_id: str = Field(..., description="APG tenant identifier")
	started_by: str = Field(..., description="User who started the instance")
	current_owner: Optional[str] = Field(None, description="Current instance owner")
	
	# Execution state
	status: WorkflowStatus = Field(default=WorkflowStatus.ACTIVE, description="Instance status")
	current_tasks: List[str] = Field(default_factory=list, description="Currently executing task IDs")
	completed_tasks: List[str] = Field(default_factory=list, description="Completed task IDs")
	failed_tasks: List[str] = Field(default_factory=list, description="Failed task IDs")
	skipped_tasks: List[str] = Field(default_factory=list, description="Skipped task IDs")
	
	# Execution context and data
	input_data: Dict[str, Any] = Field(default_factory=dict, description="Instance input data")
	output_data: Dict[str, Any] = Field(default_factory=dict, description="Instance output data")
	variables: Dict[str, Any] = Field(default_factory=dict, description="Runtime variables")
	context: Dict[str, Any] = Field(default_factory=dict, description="Execution context")
	
	# Timing and performance
	started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
	paused_at: Optional[datetime] = Field(None, description="Pause timestamp")
	resumed_at: Optional[datetime] = Field(None, description="Resume timestamp")
	duration_seconds: Optional[float] = Field(None, ge=0, description="Total execution time")
	
	# Progress tracking
	progress_percentage: float = Field(default=0.0, ge=0.0, le=100.0, description="Completion percentage")
	current_step: Optional[str] = Field(None, description="Current execution step")
	total_steps: int = Field(default=0, ge=0, description="Total workflow steps")
	completed_steps: int = Field(default=0, ge=0, description="Completed steps")
	
	# Error handling and retry
	error_message: Optional[str] = Field(None, description="Last error message")
	error_details: Optional[Dict[str, Any]] = Field(None, description="Detailed error information")
	retry_count: int = Field(default=0, ge=0, description="Current retry count")
	max_retries: int = Field(default=3, ge=0, description="Maximum retries allowed")
	
	# SLA and escalation
	sla_deadline: Optional[datetime] = Field(None, description="SLA deadline")
	is_sla_breached: bool = Field(default=False, description="SLA breach flag")
	escalation_level: int = Field(default=0, ge=0, description="Current escalation level")
	escalated_to: Optional[str] = Field(None, description="Escalated to user/role")
	
	# Trigger information
	trigger_source: Optional[str] = Field(None, description="Trigger that started instance")
	trigger_data: Dict[str, Any] = Field(default_factory=dict, description="Trigger payload data")
	
	# Parent/child relationships
	parent_instance_id: Optional[str] = Field(None, description="Parent instance for sub-workflows")
	child_instance_ids: List[str] = Field(default_factory=list, description="Child instance IDs")
	
	# Audit trail
	audit_trail: List[Dict[str, Any]] = Field(
		default_factory=list, 
		description="Detailed audit trail"
	)
	
	# Metadata
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Instance metadata")
	
	@field_validator('input_data', 'output_data', 'variables', 'context', 'metadata')
	@classmethod
	def validate_json_fields(cls, v: Dict[str, Any]) -> Dict[str, Any]:
		return validate_json_data(v)

class TaskExecution(BaseModel):
	"""Individual task execution record."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	# Core identification
	id: str = Field(default_factory=uuid7str, description="Unique execution identifier")
	instance_id: str = Field(..., description="Parent workflow instance ID")
	task_id: str = Field(..., description="Task definition ID")
	task_name: str = Field(..., description="Task name for reference")
	
	# Assignment and ownership
	assigned_to: Optional[str] = Field(None, description="Assigned user ID")
	assigned_role: Optional[str] = Field(None, description="Assigned role")
	assigned_group: Optional[str] = Field(None, description="Assigned group")
	current_assignee: Optional[str] = Field(None, description="Current task assignee")
	
	# Execution state
	status: TaskStatus = Field(default=TaskStatus.PENDING, description="Task execution status")
	priority: Priority = Field(default=Priority.NORMAL, description="Task priority")
	
	# Timing
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	assigned_at: Optional[datetime] = Field(None, description="Assignment timestamp")
	started_at: Optional[datetime] = Field(None, description="Start timestamp")
	completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
	due_date: Optional[datetime] = Field(None, description="Due date")
	duration_seconds: Optional[float] = Field(None, ge=0, description="Execution duration")
	
	# Execution data
	input_data: Dict[str, Any] = Field(default_factory=dict, description="Task input data")
	output_data: Dict[str, Any] = Field(default_factory=dict, description="Task output data")
	result: Dict[str, Any] = Field(default_factory=dict, description="Task execution result")
	
	# Error handling and retry
	error_message: Optional[str] = Field(None, description="Error message if failed")
	error_details: Optional[Dict[str, Any]] = Field(None, description="Detailed error information")
	attempt_number: int = Field(default=1, ge=1, description="Current attempt number")
	max_attempts: int = Field(default=3, ge=1, description="Maximum attempts allowed")
	retry_at: Optional[datetime] = Field(None, description="Next retry timestamp")
	
	# Progress and feedback
	progress_percentage: float = Field(default=0.0, ge=0.0, le=100.0, description="Task progress")
	progress_message: Optional[str] = Field(None, description="Progress status message")
	
	# Human task specifics
	comments: List[Dict[str, Any]] = Field(default_factory=list, description="Task comments")
	attachments: List[Dict[str, Any]] = Field(default_factory=list, description="Task attachments")
	approval_decision: Optional[Literal["approve", "reject", "delegate"]] = Field(
		None, description="Approval task decision"
	)
	approval_reason: Optional[str] = Field(None, description="Approval decision reason")
	
	# Escalation
	escalation_level: int = Field(default=0, ge=0, description="Current escalation level")
	escalated_at: Optional[datetime] = Field(None, description="Escalation timestamp")
	escalated_to: Optional[str] = Field(None, description="Escalated to user/role")
	escalation_reason: Optional[str] = Field(None, description="Escalation reason")
	
	# SLA tracking
	sla_deadline: Optional[datetime] = Field(None, description="SLA deadline")
	is_sla_breached: bool = Field(default=False, description="SLA breach flag")
	sla_breach_time: Optional[datetime] = Field(None, description="SLA breach timestamp")
	
	# Audit information
	created_by: str = Field(..., description="User who created the execution")
	updated_by: Optional[str] = Field(None, description="Last user to update")
	audit_events: List[Dict[str, Any]] = Field(
		default_factory=list,
		description="Audit events for this execution"
	)
	
	# Metadata
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution metadata")
	
	@field_validator('input_data', 'output_data', 'result', 'metadata')
	@classmethod
	def validate_json_fields(cls, v: Dict[str, Any]) -> Dict[str, Any]:
		return validate_json_data(v)

class WorkflowConnector(BaseModel):
	"""External system connector configuration."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	# Core identification
	id: str = Field(default_factory=uuid7str, description="Unique connector identifier")
	name: str = Field(..., min_length=1, max_length=200, description="Connector name")
	description: str = Field(default="", max_length=1000, description="Connector description")
	connector_type: str = Field(..., description="Type of connector")
	
	# APG integration
	tenant_id: str = Field(..., description="APG tenant identifier")
	created_by: str = Field(..., description="Creator user ID")
	
	# Connection configuration
	connection_config: Dict[str, Any] = Field(
		..., 
		description="Connection configuration parameters"
	)
	authentication_config: Dict[str, Any] = Field(
		default_factory=dict,
		description="Authentication configuration"
	)
	
	# Connector state
	is_enabled: bool = Field(default=True, description="Connector enabled status")
	is_validated: bool = Field(default=False, description="Connection validated")
	last_test_at: Optional[datetime] = Field(None, description="Last connection test")
	last_test_result: Optional[str] = Field(None, description="Last test result")
	
	# Rate limiting and quotas
	rate_limit_per_minute: Optional[int] = Field(None, ge=1, description="Rate limit per minute")
	daily_quota: Optional[int] = Field(None, ge=1, description="Daily usage quota")
	current_usage: int = Field(default=0, ge=0, description="Current usage count")
	
	# Error handling
	retry_configuration: Dict[str, Any] = Field(
		default_factory=dict,
		description="Retry configuration"
	)
	timeout_seconds: int = Field(default=30, ge=1, le=300, description="Connection timeout")
	
	# Monitoring and health
	health_check_enabled: bool = Field(default=True, description="Enable health checks")
	health_check_interval_minutes: int = Field(
		default=5, ge=1, le=60, description="Health check interval"
	)
	last_health_check: Optional[datetime] = Field(None, description="Last health check")
	health_status: Literal["healthy", "degraded", "unhealthy", "unknown"] = Field(
		default="unknown", description="Current health status"
	)
	
	# Audit and security
	encryption_enabled: bool = Field(default=True, description="Enable data encryption")
	audit_level: Literal["none", "basic", "detailed"] = Field(
		default="basic", description="Audit level"
	)
	
	# Timestamps
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	
	# Metadata
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Connector metadata")
	
	@field_validator('connection_config', 'authentication_config', 'metadata')
	@classmethod
	def validate_json_fields(cls, v: Dict[str, Any]) -> Dict[str, Any]:
		return validate_json_data(v)

class WorkflowAuditLog(BaseModel):
	"""Comprehensive audit log for workflow operations."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	# Core identification
	id: str = Field(default_factory=uuid7str, description="Unique audit log identifier")
	tenant_id: str = Field(..., description="APG tenant identifier")
	
	# Context identification
	workflow_id: Optional[str] = Field(None, description="Related workflow ID")
	instance_id: Optional[str] = Field(None, description="Related instance ID")
	task_execution_id: Optional[str] = Field(None, description="Related task execution ID")
	
	# Event details
	event_type: str = Field(..., description="Type of audited event")
	event_category: Literal[
		"workflow", "instance", "task", "user", "system", "security", "performance"
	] = Field(..., description="Event category")
	action: str = Field(..., description="Action performed")
	resource_type: str = Field(..., description="Type of resource affected")
	resource_id: str = Field(..., description="ID of affected resource")
	
	# User and session context
	user_id: Optional[str] = Field(None, description="User who performed the action")
	session_id: Optional[str] = Field(None, description="Session identifier")
	ip_address: Optional[str] = Field(None, description="Source IP address")
	user_agent: Optional[str] = Field(None, description="User agent string")
	
	# Event data
	event_data: Dict[str, Any] = Field(default_factory=dict, description="Event payload data")
	previous_values: Optional[Dict[str, Any]] = Field(None, description="Previous values")
	new_values: Optional[Dict[str, Any]] = Field(None, description="New values")
	
	# Result and impact
	result: Literal["success", "failure", "partial"] = Field(..., description="Operation result")
	error_message: Optional[str] = Field(None, description="Error message if failed")
	impact_level: Literal["low", "medium", "high", "critical"] = Field(
		default="low", description="Impact level"
	)
	
	# Compliance and security
	compliance_tags: List[str] = Field(default_factory=list, description="Compliance tags")
	security_classification: Literal["public", "internal", "confidential", "restricted"] = Field(
		default="internal", description="Security classification"
	)
	retention_policy: str = Field(default="standard", description="Data retention policy")
	
	# Timing
	timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	duration_ms: Optional[int] = Field(None, ge=0, description="Operation duration")
	
	# Correlation and tracing
	correlation_id: Optional[str] = Field(None, description="Correlation identifier")
	trace_id: Optional[str] = Field(None, description="Distributed tracing ID")
	parent_span_id: Optional[str] = Field(None, description="Parent span ID")
	
	# Metadata
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
	
	@field_validator('event_data', 'previous_values', 'new_values', 'metadata')
	@classmethod
	def validate_json_fields(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
		if v is not None:
			return validate_json_data(v)
		return v

# Utility functions for logging and assertions
def _log_workflow_operation(operation: str, workflow_id: str, details: Dict[str, Any] = None) -> None:
	"""Log workflow operation for debugging."""
	print(f"[WORKFLOW] {operation}: {workflow_id} - {details or {}}")

def _log_task_execution(operation: str, task_id: str, instance_id: str, details: Dict[str, Any] = None) -> None:
	"""Log task execution for debugging."""
	print(f"[TASK] {operation}: {task_id} in {instance_id} - {details or {}}")

def _log_audit_event(event_type: str, resource_id: str, user_id: str, details: Dict[str, Any] = None) -> None:
	"""Log audit event for compliance."""
	print(f"[AUDIT] {event_type}: {resource_id} by {user_id} - {details or {}}")

# Runtime assertion helpers
def assert_workflow_valid(workflow: Workflow) -> None:
	"""Assert workflow is in valid state."""
	assert workflow.id, "Workflow must have valid ID"
	assert workflow.name, "Workflow must have name"
	assert workflow.tenant_id, "Workflow must have tenant_id"
	assert len(workflow.tasks) > 0, "Workflow must have at least one task"

def assert_instance_active(instance: WorkflowInstance) -> None:
	"""Assert workflow instance is active."""
	assert instance.status == WorkflowStatus.ACTIVE, f"Instance {instance.id} is not active"
	assert instance.workflow_id, "Instance must have workflow_id"

def assert_task_executable(task: TaskDefinition, instance: WorkflowInstance) -> None:
	"""Assert task can be executed in current instance state."""
	assert task.id not in instance.completed_tasks, f"Task {task.id} already completed"
	assert task.id not in instance.failed_tasks, f"Task {task.id} already failed"
	
	# Check dependencies are met
	for dep_id in task.dependencies:
		assert dep_id in instance.completed_tasks, f"Dependency {dep_id} not completed"

__all__ = [
	"WorkflowStatus",
	"TaskStatus", 
	"TaskType",
	"TriggerType",
	"Priority",
	"EscalationAction",
	"WorkflowMetrics",
	"WorkflowTemplate",
	"TaskDefinition",
	"WorkflowTrigger",
	"Workflow",
	"WorkflowInstance",
	"TaskExecution",
	"WorkflowConnector",
	"WorkflowAuditLog",
	"_log_workflow_operation",
	"_log_task_execution", 
	"_log_audit_event",
	"assert_workflow_valid",
	"assert_instance_active",
	"assert_task_executable",
	"validate_cron_expression",
	"validate_json_data"
]