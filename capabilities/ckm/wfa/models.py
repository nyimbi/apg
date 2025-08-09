"""
APG Workflow & Business Process Management - Data Models

Enterprise-grade data models for the APG Workflow & Business Process Management capability.
Implements CLAUDE.md standards with async Python, modern typing, and APG integration.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator, root_validator
from pydantic import ConfigDict
from pydantic.types import EmailStr, PositiveFloat, PositiveInt
from pydantic.functional_validators import AfterValidator
from typing_extensions import Annotated

from uuid_extensions import uuid7str


# =============================================================================
# Configuration and Validation
# =============================================================================

def _log_validation_error(field_name: str, value: Any, error: str) -> str:
	"""Log validation errors with consistent formatting."""
	return f"Validation failed for {field_name}: {value} - {error}"


def validate_non_empty_string(value: str) -> str:
	"""Validate non-empty strings."""
	if not value or not value.strip():
		raise ValueError(_log_validation_error("string", value, "cannot be empty"))
	return value.strip()


def validate_bpmn_element_id(value: str) -> str:
	"""Validate BPMN element IDs."""
	if not value or len(value) < 1 or len(value) > 255:
		raise ValueError(_log_validation_error("bpmn_element_id", value, "must be 1-255 characters"))
	return value.strip()


def validate_process_version(value: str) -> str:
	"""Validate semantic version format."""
	import re
	if not re.match(r'^\d+\.\d+\.\d+$', value):
		raise ValueError(_log_validation_error("process_version", value, "must be semantic version (x.y.z)"))
	return value


def validate_priority_score(value: int) -> int:
	"""Validate priority scores (1-1000)."""
	if value < 1 or value > 1000:
		raise ValueError(_log_validation_error("priority_score", value, "must be between 1 and 1000"))
	return value


def validate_confidence_score(value: float) -> float:
	"""Validate confidence scores (0.0-1.0)."""
	if value < 0.0 or value > 1.0:
		raise ValueError(_log_validation_error("confidence_score", value, "must be between 0.0 and 1.0"))
	return value


def validate_duration_ms(value: int) -> int:
	"""Validate duration in milliseconds."""
	if value < 0:
		raise ValueError(_log_validation_error("duration_ms", value, "cannot be negative"))
	return value


# Type aliases with validation
NonEmptyString = Annotated[str, AfterValidator(validate_non_empty_string)]
BPMNElementId = Annotated[str, AfterValidator(validate_bpmn_element_id)]
ProcessVersion = Annotated[str, AfterValidator(validate_process_version)]
PriorityScore = Annotated[int, AfterValidator(validate_priority_score)]
ConfidenceScore = Annotated[float, AfterValidator(validate_confidence_score)]
DurationMs = Annotated[int, AfterValidator(validate_duration_ms)]


# =============================================================================
# Base Models and Enums
# =============================================================================

class APGBaseModel(BaseModel):
	"""Base model with APG multi-tenant patterns and common fields."""
	
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True,
		str_strip_whitespace=True,
		validate_default=True
	)
	
	# APG Integration Fields
	id: str = Field(default_factory=uuid7str, description="Unique identifier")
	tenant_id: str = Field(..., description="APG tenant identifier")
	
	# Audit Fields (APG audit_compliance integration)
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	created_by: str = Field(..., description="User who created the record")
	updated_by: str = Field(..., description="User who last updated the record")


class APGTenantContext(BaseModel):
	"""APG tenant context for service operations."""
	
	model_config = ConfigDict(extra='forbid')
	
	tenant_id: str
	user_id: str
	session_id: Optional[str] = None
	permissions: List[str] = Field(default_factory=list)
	metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Workflow Status and Type Enumerations
# =============================================================================

class ProcessStatus(str, Enum):
	"""Process definition status enumeration."""
	DRAFT = "draft"
	PUBLISHED = "published"
	ACTIVE = "active"
	DEPRECATED = "deprecated"
	ARCHIVED = "archived"


class InstanceStatus(str, Enum):
	"""Process instance status enumeration."""
	CREATED = "created"
	RUNNING = "running"
	SUSPENDED = "suspended"
	COMPLETED = "completed"
	FAILED = "failed"
	CANCELLED = "cancelled"
	TERMINATED = "terminated"


class TaskStatus(str, Enum):
	"""Task status enumeration."""
	CREATED = "created"
	READY = "ready"
	RESERVED = "reserved"
	IN_PROGRESS = "in_progress"
	SUSPENDED = "suspended"
	COMPLETED = "completed"
	FAILED = "failed"
	OBSOLETE = "obsolete"
	EXITED = "exited"


class TaskPriority(str, Enum):
	"""Task priority enumeration."""
	CRITICAL = "critical"
	HIGH = "high"
	MEDIUM = "medium"
	LOW = "low"


class ActivityType(str, Enum):
	"""BPMN activity type enumeration."""
	START_EVENT = "start_event"
	END_EVENT = "end_event"
	INTERMEDIATE_EVENT = "intermediate_event"
	USER_TASK = "user_task"
	SERVICE_TASK = "service_task"
	SCRIPT_TASK = "script_task"
	BUSINESS_RULE_TASK = "business_rule_task"
	MANUAL_TASK = "manual_task"
	RECEIVE_TASK = "receive_task"
	SEND_TASK = "send_task"
	EXCLUSIVE_GATEWAY = "exclusive_gateway"
	PARALLEL_GATEWAY = "parallel_gateway"
	INCLUSIVE_GATEWAY = "inclusive_gateway"
	EVENT_GATEWAY = "event_gateway"
	SUBPROCESS = "subprocess"
	CALL_ACTIVITY = "call_activity"


class GatewayDirection(str, Enum):
	"""Gateway direction enumeration."""
	UNSPECIFIED = "unspecified"
	CONVERGING = "converging"
	DIVERGING = "diverging"
	MIXED = "mixed"


class EventType(str, Enum):
	"""BPMN event type enumeration."""
	NONE = "none"
	MESSAGE = "message"
	TIMER = "timer"
	ERROR = "error"
	ESCALATION = "escalation"
	CANCEL = "cancel"
	COMPENSATION = "compensation"
	CONDITIONAL = "conditional"
	LINK = "link"
	SIGNAL = "signal"
	MULTIPLE = "multiple"
	PARALLEL_MULTIPLE = "parallel_multiple"
	TERMINATE = "terminate"


class CollaborationRole(str, Enum):
	"""Collaboration role enumeration."""
	PROCESS_OWNER = "process_owner"
	PROCESS_CONTRIBUTOR = "process_contributor"
	PROCESS_REVIEWER = "process_reviewer"
	PROCESS_OBSERVER = "process_observer"
	TASK_COLLABORATOR = "task_collaborator"


class AIServiceType(str, Enum):
	"""AI service type enumeration."""
	PROCESS_OPTIMIZATION = "process_optimization"
	TASK_ROUTING = "task_routing"
	BOTTLENECK_DETECTION = "bottleneck_detection"
	ANOMALY_DETECTION = "anomaly_detection"
	PERFORMANCE_PREDICTION = "performance_prediction"
	RESOURCE_OPTIMIZATION = "resource_optimization"
	DECISION_SUPPORT = "decision_support"


# =============================================================================
# Core Process Models
# =============================================================================

class WBPMProcessDefinition(APGBaseModel):
	"""Main process definition model with BPMN 2.0 support."""
	
	process_key: NonEmptyString = Field(..., description="Unique process key")
	process_name: NonEmptyString = Field(..., description="Human-readable process name")
	process_description: Optional[str] = Field(None, description="Detailed process description")
	process_version: ProcessVersion = Field(default="1.0.0", description="Semantic version")
	process_status: ProcessStatus = Field(default=ProcessStatus.DRAFT, description="Process status")
	
	# BPMN Definition
	bpmn_xml: str = Field(..., description="BPMN 2.0 XML definition")
	bpmn_json: Optional[Dict[str, Any]] = Field(None, description="BPMN 2.0 JSON representation")
	process_variables: List[Dict[str, Any]] = Field(default_factory=list, description="Process variables schema")
	
	# Metadata
	category: Optional[str] = Field(None, description="Process category")
	tags: List[str] = Field(default_factory=list, description="Process tags")
	documentation_url: Optional[str] = Field(None, description="External documentation URL")
	
	# Configuration
	is_executable: bool = Field(default=True, description="Whether process can be executed")
	is_suspended: bool = Field(default=False, description="Whether process is suspended")
	suspension_reason: Optional[str] = Field(None, description="Reason for suspension")
	
	# Version Control
	parent_version_id: Optional[str] = Field(None, description="Parent version ID")
	version_notes: Optional[str] = Field(None, description="Version change notes")
	deployment_time: Optional[datetime] = Field(None, description="Deployment timestamp")
	
	@validator('bpmn_xml')
	def validate_bpmn_xml(cls, v):
		"""Validate BPMN XML format."""
		if not v or not v.strip():
			raise ValueError("BPMN XML cannot be empty")
		# Additional BPMN validation could be added here
		return v.strip()
	
	@validator('tags')
	def validate_tags(cls, v):
		"""Validate and normalize tags."""
		return [tag.strip().lower() for tag in v if tag.strip()]


class WBPMProcessInstance(APGBaseModel):
	"""Process execution instance with comprehensive state management."""
	
	process_id: str = Field(..., description="Process definition ID")
	business_key: Optional[str] = Field(None, description="Business-specific identifier")
	instance_name: Optional[str] = Field(None, description="Human-readable instance name")
	instance_status: InstanceStatus = Field(default=InstanceStatus.CREATED, description="Instance status")
	
	# Execution Context
	process_variables: Dict[str, Any] = Field(default_factory=dict, description="Process variables")
	current_activities: List[str] = Field(default_factory=list, description="Currently active activity IDs")
	suspended_activities: List[str] = Field(default_factory=list, description="Suspended activity IDs")
	
	# Parent/Child Relationships
	parent_instance_id: Optional[str] = Field(None, description="Parent instance ID")
	root_instance_id: Optional[str] = Field(None, description="Root instance ID")
	call_activity_id: Optional[str] = Field(None, description="Call activity that started this instance")
	
	# Timing
	start_time: datetime = Field(default_factory=datetime.utcnow, description="Instance start time")
	end_time: Optional[datetime] = Field(None, description="Instance end time")
	duration_ms: Optional[DurationMs] = Field(None, description="Execution duration in milliseconds")
	
	# Initiator
	initiated_by: str = Field(..., description="User who initiated the instance")
	
	# Error Handling
	last_error_message: Optional[str] = Field(None, description="Last error message")
	error_count: int = Field(default=0, description="Number of errors encountered")
	retry_count: int = Field(default=0, description="Number of retries attempted")
	
	# Priority and SLA
	priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="Instance priority")
	due_date: Optional[datetime] = Field(None, description="Due date for completion")
	
	@validator('end_time')
	def validate_end_time(cls, v, values):
		"""Validate end time is after start time."""
		if v and 'start_time' in values and v < values['start_time']:
			raise ValueError("End time must be after start time")
		return v
	
	@root_validator
	def validate_duration_calculation(cls, values):
		"""Validate duration calculation consistency."""
		start_time = values.get('start_time')
		end_time = values.get('end_time')
		duration_ms = values.get('duration_ms')
		
		if end_time and start_time and duration_ms:
			calculated_duration = int((end_time - start_time).total_seconds() * 1000)
			if abs(calculated_duration - duration_ms) > 1000:  # Allow 1 second tolerance
				raise ValueError("Duration calculation inconsistent with start/end times")
		
		return values


class WBPMProcessActivity(APGBaseModel):
	"""BPMN process activity/element definition."""
	
	process_id: str = Field(..., description="Process definition ID")
	element_id: BPMNElementId = Field(..., description="BPMN element ID")
	element_name: Optional[str] = Field(None, description="Human-readable element name")
	activity_type: ActivityType = Field(..., description="BPMN activity type")
	
	# Configuration
	element_properties: Dict[str, Any] = Field(default_factory=dict, description="BPMN element properties")
	execution_listeners: List[Dict[str, Any]] = Field(default_factory=list, description="Execution listeners")
	task_listeners: List[Dict[str, Any]] = Field(default_factory=list, description="Task listeners")
	
	# Task-specific properties
	assignee: Optional[str] = Field(None, description="Default assignee")
	candidate_users: List[str] = Field(default_factory=list, description="Candidate users")
	candidate_groups: List[str] = Field(default_factory=list, description="Candidate groups")
	form_key: Optional[str] = Field(None, description="Form key for user tasks")
	
	# Service Task properties
	class_name: Optional[str] = Field(None, description="Java class name for service tasks")
	expression: Optional[str] = Field(None, description="Expression for service tasks")
	delegate_expression: Optional[str] = Field(None, description="Delegate expression")
	
	# Gateway properties
	gateway_direction: Optional[GatewayDirection] = Field(None, description="Gateway direction")
	default_flow: Optional[str] = Field(None, description="Default flow for gateways")
	
	# Event properties
	event_type: Optional[EventType] = Field(None, description="Event type")
	event_definition: Optional[Dict[str, Any]] = Field(None, description="Event definition")
	
	# Timing and SLA
	due_date_expression: Optional[str] = Field(None, description="Due date expression")
	follow_up_date_expression: Optional[str] = Field(None, description="Follow-up date expression")


class WBPMProcessFlow(APGBaseModel):
	"""BPMN sequence flow definition."""
	
	process_id: str = Field(..., description="Process definition ID")
	element_id: BPMNElementId = Field(..., description="BPMN flow element ID")
	flow_name: Optional[str] = Field(None, description="Human-readable flow name")
	source_activity_id: str = Field(..., description="Source activity ID")
	target_activity_id: str = Field(..., description="Target activity ID")
	
	# Flow Properties
	condition_expression: Optional[str] = Field(None, description="Flow condition expression")
	is_default_flow: bool = Field(default=False, description="Whether this is a default flow")
	flow_properties: Dict[str, Any] = Field(default_factory=dict, description="Additional flow properties")
	
	@validator('source_activity_id', 'target_activity_id')
	def validate_activity_ids(cls, v):
		"""Validate activity IDs are not empty."""
		if not v or not v.strip():
			raise ValueError("Activity ID cannot be empty")
		return v.strip()
	
	@root_validator
	def validate_no_self_flow(cls, values):
		"""Validate flow doesn't connect activity to itself."""
		source = values.get('source_activity_id')
		target = values.get('target_activity_id')
		if source and target and source == target:
			raise ValueError("Flow cannot connect activity to itself")
		return values


# =============================================================================
# Task Management Models
# =============================================================================

class WBPMTask(APGBaseModel):
	"""Individual workflow task with comprehensive management features."""
	
	process_instance_id: str = Field(..., description="Process instance ID")
	activity_id: str = Field(..., description="Activity definition ID")
	
	# Task Details
	task_name: NonEmptyString = Field(..., description="Task name")
	task_description: Optional[str] = Field(None, description="Task description")
	task_status: TaskStatus = Field(default=TaskStatus.CREATED, description="Task status")
	
	# Assignment
	assignee: Optional[str] = Field(None, description="Assigned user ID")
	owner: Optional[str] = Field(None, description="Task owner")
	delegation_state: Optional[str] = Field(None, description="Delegation state (pending, resolved)")
	
	# Candidate Assignment
	candidate_users: List[str] = Field(default_factory=list, description="Candidate users")
	candidate_groups: List[str] = Field(default_factory=list, description="Candidate groups")
	
	# Task Data
	form_key: Optional[str] = Field(None, description="Form key for task UI")
	task_variables: Dict[str, Any] = Field(default_factory=dict, description="Task-specific variables")
	local_variables: Dict[str, Any] = Field(default_factory=dict, description="Local task variables")
	
	# Timing
	create_time: datetime = Field(default_factory=datetime.utcnow, description="Task creation time")
	claim_time: Optional[datetime] = Field(None, description="Task claim time")
	due_date: Optional[datetime] = Field(None, description="Task due date")
	follow_up_date: Optional[datetime] = Field(None, description="Follow-up date")
	completion_time: Optional[datetime] = Field(None, description="Task completion time")
	
	# Priority and Effort
	priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="Task priority")
	estimated_effort_hours: Optional[Decimal] = Field(None, description="Estimated effort in hours")
	actual_effort_hours: Optional[Decimal] = Field(None, description="Actual effort in hours")
	
	# Parent Task (for subtasks)
	parent_task_id: Optional[str] = Field(None, description="Parent task ID")
	
	# Suspension
	suspension_state: Optional[str] = Field(None, description="Suspension state (active, suspended)")
	suspension_reason: Optional[str] = Field(None, description="Reason for suspension")
	
	@validator('completion_time')
	def validate_completion_time(cls, v, values):
		"""Validate completion time consistency with task status."""
		task_status = values.get('task_status')
		if task_status == TaskStatus.COMPLETED and not v:
			raise ValueError("Completed tasks must have completion time")
		if task_status != TaskStatus.COMPLETED and v:
			raise ValueError("Only completed tasks can have completion time")
		return v
	
	@validator('claim_time')
	def validate_claim_time(cls, v, values):
		"""Validate claim time consistency with assignee."""
		assignee = values.get('assignee')
		if assignee and not v:
			# Allow claim_time to be None initially, will be set when claimed
			pass
		return v


class WBPMTaskHistory(APGBaseModel):
	"""Task history for comprehensive audit trail."""
	
	task_id: str = Field(..., description="Task ID")
	
	# History Details
	action_type: NonEmptyString = Field(..., description="Action type (created, assigned, completed, etc.)")
	action_description: Optional[str] = Field(None, description="Detailed action description")
	
	# State Changes
	old_value: Optional[Dict[str, Any]] = Field(None, description="Previous state")
	new_value: Optional[Dict[str, Any]] = Field(None, description="New state")
	
	# Actor
	performed_by: str = Field(..., description="User who performed the action")
	performed_at: datetime = Field(default_factory=datetime.utcnow, description="Action timestamp")
	
	# Context
	action_context: Dict[str, Any] = Field(default_factory=dict, description="Additional action context")


class WBPMTaskComment(APGBaseModel):
	"""Task comments for collaboration and communication."""
	
	task_id: str = Field(..., description="Task ID")
	
	# Comment Details
	comment_text: NonEmptyString = Field(..., description="Comment text")
	comment_type: str = Field(default="user", description="Comment type (user, system, audit)")
	
	# Reply Thread
	parent_comment_id: Optional[str] = Field(None, description="Parent comment ID for threading")
	thread_level: int = Field(default=0, description="Thread nesting level")
	
	# Attachments
	attachments: List[Dict[str, Any]] = Field(default_factory=list, description="File attachments")
	
	@validator('thread_level')
	def validate_thread_level(cls, v):
		"""Validate thread nesting level."""
		if v < 0 or v > 10:
			raise ValueError("Thread level must be between 0 and 10")
		return v


# =============================================================================
# Template and Collaboration Models
# =============================================================================

class WBPMProcessTemplate(APGBaseModel):
	"""Process template for reusable process definitions."""
	
	template_name: NonEmptyString = Field(..., description="Template name")
	template_description: Optional[str] = Field(None, description="Template description")
	template_category: Optional[str] = Field(None, description="Template category")
	template_tags: List[str] = Field(default_factory=list, description="Template tags")
	
	# Template Content
	bpmn_template: str = Field(..., description="BPMN template XML")
	template_variables: List[Dict[str, Any]] = Field(default_factory=list, description="Template variables")
	configuration_schema: Dict[str, Any] = Field(default_factory=dict, description="Configuration schema")
	
	# Usage and Sharing
	is_public: bool = Field(default=False, description="Whether template is publicly shared")
	usage_count: int = Field(default=0, description="Number of times used")
	rating_average: Decimal = Field(default=Decimal('0.0'), description="Average rating")
	rating_count: int = Field(default=0, description="Number of ratings")
	
	# Versioning
	template_version: ProcessVersion = Field(default="1.0.0", description="Template version")
	parent_template_id: Optional[str] = Field(None, description="Parent template ID")
	
	@validator('rating_average')
	def validate_rating_average(cls, v):
		"""Validate rating average range."""
		if v < 0 or v > 5:
			raise ValueError("Rating average must be between 0.0 and 5.0")
		return v
	
	@validator('usage_count', 'rating_count')
	def validate_counts(cls, v):
		"""Validate count fields are non-negative."""
		if v < 0:
			raise ValueError("Count fields must be non-negative")
		return v


class WBPMCollaborationSession(APGBaseModel):
	"""Real-time collaboration session for process work."""
	
	session_name: NonEmptyString = Field(..., description="Session name")
	session_type: str = Field(..., description="Session type (design, execution, review, analysis)")
	target_process_id: Optional[str] = Field(None, description="Target process ID")
	target_instance_id: Optional[str] = Field(None, description="Target instance ID")
	
	# Session Configuration
	max_participants: int = Field(default=10, description="Maximum participants")
	session_duration_minutes: Optional[int] = Field(None, description="Session duration in minutes")
	conflict_resolution_mode: str = Field(default="last_writer_wins", description="Conflict resolution strategy")
	
	# Session State
	session_status: str = Field(default="active", description="Session status (active, paused, completed)")
	start_time: datetime = Field(default_factory=datetime.utcnow, description="Session start time")
	end_time: Optional[datetime] = Field(None, description="Session end time")
	
	# Host Information
	session_host: str = Field(..., description="Session host user ID")
	
	@validator('max_participants')
	def validate_max_participants(cls, v):
		"""Validate maximum participants range."""
		if v < 1 or v > 100:
			raise ValueError("Maximum participants must be between 1 and 100")
		return v
	
	@validator('session_duration_minutes')
	def validate_session_duration(cls, v):
		"""Validate session duration."""
		if v is not None and v <= 0:
			raise ValueError("Session duration must be positive")
		return v


class WBPMCollaborationParticipant(APGBaseModel):
	"""Collaboration session participant."""
	
	session_id: str = Field(..., description="Session ID")
	user_id: str = Field(..., description="Participant user ID")
	participant_name: NonEmptyString = Field(..., description="Participant display name")
	collaboration_role: CollaborationRole = Field(..., description="Collaboration role")
	
	# Participation State
	join_time: datetime = Field(default_factory=datetime.utcnow, description="Join time")
	leave_time: Optional[datetime] = Field(None, description="Leave time")
	is_active: bool = Field(default=True, description="Whether participant is active")
	last_activity: datetime = Field(default_factory=datetime.utcnow, description="Last activity time")
	
	# Visual Presence
	cursor_position: Optional[Dict[str, Any]] = Field(None, description="Current cursor position")
	selected_elements: List[str] = Field(default_factory=list, description="Selected elements")
	participant_color: Optional[str] = Field(None, description="Participant color (hex)")
	
	# Permissions
	permissions: List[str] = Field(default_factory=list, description="Participant permissions")
	
	@validator('participant_color')
	def validate_participant_color(cls, v):
		"""Validate hex color format."""
		if v and not v.startswith('#') or len(v) != 7:
			raise ValueError("Participant color must be valid hex color (#RRGGBB)")
		return v


# =============================================================================
# Analytics and Monitoring Models
# =============================================================================

class WBPMProcessMetrics(APGBaseModel):
	"""Process performance metrics for analytics."""
	
	process_id: Optional[str] = Field(None, description="Process ID")
	instance_id: Optional[str] = Field(None, description="Instance ID")
	
	# Metric Details
	metric_type: NonEmptyString = Field(..., description="Metric type")
	metric_name: NonEmptyString = Field(..., description="Metric name")
	metric_value: Decimal = Field(..., description="Metric value")
	metric_unit: Optional[str] = Field(None, description="Metric unit")
	
	# Context
	measurement_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Measurement time")
	measurement_context: Dict[str, Any] = Field(default_factory=dict, description="Measurement context")
	
	# Aggregation
	aggregation_period: Optional[str] = Field(None, description="Aggregation period")
	aggregation_level: Optional[str] = Field(None, description="Aggregation level")


class WBPMProcessBottleneck(APGBaseModel):
	"""Identified process bottleneck with analysis."""
	
	process_id: str = Field(..., description="Process ID")
	
	# Bottleneck Details
	bottleneck_activity: NonEmptyString = Field(..., description="Bottleneck activity")
	bottleneck_type: str = Field(..., description="Bottleneck type (resource, time, queue, system)")
	severity: str = Field(..., description="Severity (critical, high, medium, low)")
	
	# Impact Analysis
	impact_score: Decimal = Field(..., description="Impact score (0.00 to 100.00)")
	affected_instances: int = Field(default=0, description="Number of affected instances")
	average_delay_minutes: Optional[Decimal] = Field(None, description="Average delay in minutes")
	
	# Detection Details
	detection_method: Optional[str] = Field(None, description="Detection method")
	detection_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Detection time")
	confidence_score: Optional[ConfidenceScore] = Field(None, description="Detection confidence")
	
	# Resolution
	recommendation: Optional[str] = Field(None, description="Improvement recommendation")
	resolution_status: str = Field(default="open", description="Resolution status")
	resolution_notes: Optional[str] = Field(None, description="Resolution notes")
	resolved_at: Optional[datetime] = Field(None, description="Resolution timestamp")
	resolved_by: Optional[str] = Field(None, description="User who resolved")
	
	@validator('impact_score')
	def validate_impact_score(cls, v):
		"""Validate impact score range."""
		if v < 0 or v > 100:
			raise ValueError("Impact score must be between 0.00 and 100.00")
		return v


# =============================================================================
# AI and Automation Models
# =============================================================================

class WBPMAIRecommendation(APGBaseModel):
	"""AI-generated workflow recommendations."""
	
	recommendation_type: AIServiceType = Field(..., description="Recommendation type")
	target_process_id: Optional[str] = Field(None, description="Target process ID")
	target_instance_id: Optional[str] = Field(None, description="Target instance ID")
	
	# Content
	recommendation_title: NonEmptyString = Field(..., description="Recommendation title")
	recommendation_description: str = Field(..., description="Detailed recommendation")
	implementation_instructions: Optional[str] = Field(None, description="Implementation instructions")
	
	# Analysis
	confidence_score: ConfidenceScore = Field(..., description="Recommendation confidence")
	impact_assessment: Dict[str, Any] = Field(default_factory=dict, description="Impact assessment")
	implementation_effort: str = Field(..., description="Implementation effort (low, medium, high)")
	expected_benefit: Dict[str, Any] = Field(default_factory=dict, description="Expected benefits")
	
	# Lifecycle
	recommendation_status: str = Field(default="pending", description="Status (pending, accepted, rejected, implemented)")
	generated_at: datetime = Field(default_factory=datetime.utcnow, description="Generation timestamp")
	expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")
	reviewed_at: Optional[datetime] = Field(None, description="Review timestamp")
	reviewed_by: Optional[str] = Field(None, description="Reviewer user ID")
	review_notes: Optional[str] = Field(None, description="Review notes")
	
	# Implementation
	implementation_date: Optional[datetime] = Field(None, description="Implementation date")
	implementation_notes: Optional[str] = Field(None, description="Implementation notes")
	success_metrics: Dict[str, Any] = Field(default_factory=dict, description="Success metrics")


class WBPMProcessRule(APGBaseModel):
	"""Business rules for process automation."""
	
	process_id: Optional[str] = Field(None, description="Target process ID")
	
	# Rule Definition
	rule_name: NonEmptyString = Field(..., description="Rule name")
	rule_description: Optional[str] = Field(None, description="Rule description")
	rule_type: str = Field(..., description="Rule type (business_rule, validation_rule, routing_rule, escalation_rule)")
	
	# Rule Logic
	rule_condition: str = Field(..., description="Rule condition expression")
	rule_action: str = Field(..., description="Rule action to execute")
	rule_priority: PriorityScore = Field(default=100, description="Rule priority (1-1000)")
	
	# Rule Context
	applies_to_activities: List[str] = Field(default_factory=list, description="Activities this rule applies to")
	rule_context: Dict[str, Any] = Field(default_factory=dict, description="Rule context")
	
	# Rule State
	is_active: bool = Field(default=True, description="Whether rule is active")
	activation_date: datetime = Field(default_factory=datetime.utcnow, description="Rule activation date")
	deactivation_date: Optional[datetime] = Field(None, description="Rule deactivation date")
	
	# Performance
	execution_count: int = Field(default=0, description="Number of executions")
	last_execution: Optional[datetime] = Field(None, description="Last execution timestamp")
	average_execution_time_ms: Optional[Decimal] = Field(None, description="Average execution time")
	
	@validator('deactivation_date')
	def validate_deactivation_date(cls, v, values):
		"""Validate deactivation date is after activation date."""
		activation_date = values.get('activation_date')
		if v and activation_date and v <= activation_date:
			raise ValueError("Deactivation date must be after activation date")
		return v


# =============================================================================
# Service Response Models
# =============================================================================

class WBPMServiceResponse(BaseModel):
	"""Standard service response wrapper."""
	
	model_config = ConfigDict(extra='forbid')
	
	success: bool = Field(..., description="Whether operation was successful")
	message: str = Field(..., description="Response message")
	data: Optional[Dict[str, Any]] = Field(None, description="Response data")
	errors: List[str] = Field(default_factory=list, description="Error messages")
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class WBPMPagedResponse(BaseModel):
	"""Paged response for list operations."""
	
	model_config = ConfigDict(extra='forbid')
	
	items: List[Dict[str, Any]] = Field(..., description="Response items")
	total_count: int = Field(..., description="Total item count")
	page: int = Field(..., description="Current page number")
	page_size: int = Field(..., description="Page size")
	has_next: bool = Field(..., description="Whether there are more pages")
	has_previous: bool = Field(..., description="Whether there are previous pages")


# =============================================================================
# Configuration Models
# =============================================================================

class WBPMServiceConfig(BaseModel):
	"""Service configuration model."""
	
	model_config = ConfigDict(extra='forbid')
	
	# Database Configuration
	database_url: str = Field(..., description="Database connection URL")
	database_pool_size: int = Field(default=20, description="Database connection pool size")
	database_timeout: int = Field(default=30, description="Database timeout in seconds")
	
	# Cache Configuration
	cache_enabled: bool = Field(default=True, description="Whether caching is enabled")
	cache_url: str = Field(default="redis://localhost:6379", description="Cache connection URL")
	cache_ttl_seconds: int = Field(default=300, description="Default cache TTL")
	
	# Performance Configuration
	max_concurrent_instances: int = Field(default=1000, description="Maximum concurrent process instances")
	max_concurrent_tasks: int = Field(default=5000, description="Maximum concurrent tasks")
	execution_timeout_seconds: int = Field(default=300, description="Default execution timeout")
	
	# Integration Configuration
	apg_auth_service_url: str = Field(..., description="APG auth service URL")
	apg_audit_service_url: str = Field(..., description="APG audit service URL")
	apg_collaboration_service_url: str = Field(..., description="APG collaboration service URL")
	apg_ai_service_url: str = Field(..., description="APG AI service URL")
	
	# Event Bus Configuration
	kafka_brokers: List[str] = Field(default_factory=list, description="Kafka broker URLs")
	kafka_topic_prefix: str = Field(default="wbpm", description="Kafka topic prefix")
	
	# Security Configuration
	encryption_key: str = Field(..., description="Encryption key for sensitive data")
	jwt_secret: str = Field(..., description="JWT signing secret")
	session_timeout_minutes: int = Field(default=60, description="Session timeout in minutes")


# =============================================================================
# Export All Models
# =============================================================================

__all__ = [
	# Base Models
	'APGBaseModel',
	'APGTenantContext',
	
	# Enumerations
	'ProcessStatus',
	'InstanceStatus', 
	'TaskStatus',
	'TaskPriority',
	'ActivityType',
	'GatewayDirection',
	'EventType',
	'CollaborationRole',
	'AIServiceType',
	
	# Core Models
	'WBPMProcessDefinition',
	'WBPMProcessInstance',
	'WBPMProcessActivity',
	'WBPMProcessFlow',
	
	# Task Models
	'WBPMTask',
	'WBPMTaskHistory',
	'WBPMTaskComment',
	
	# Template and Collaboration Models
	'WBPMProcessTemplate',
	'WBPMCollaborationSession',
	'WBPMCollaborationParticipant',
	
	# Analytics Models
	'WBPMProcessMetrics',
	'WBPMProcessBottleneck',
	
	# AI and Automation Models
	'WBPMAIRecommendation',
	'WBPMProcessRule',
	
	# Response Models
	'WBPMServiceResponse',
	'WBPMPagedResponse',
	
	# Configuration
	'WBPMServiceConfig',
]