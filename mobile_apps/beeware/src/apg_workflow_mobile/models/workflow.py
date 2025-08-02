"""
Workflow model and related data structures

© 2025 Datacraft. All rights reserved.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, ConfigDict
from uuid_extensions import uuid7str


class WorkflowStatus(str, Enum):
	"""Workflow status enumeration"""
	DRAFT = "draft"
	ACTIVE = "active"
	PAUSED = "paused"
	COMPLETED = "completed"
	FAILED = "failed"
	CANCELLED = "cancelled"
	ARCHIVED = "archived"


class TriggerType(str, Enum):
	"""Workflow trigger type enumeration"""
	MANUAL = "manual"
	SCHEDULED = "scheduled"
	EVENT = "event"
	API = "api"
	WEBHOOK = "webhook"
	FILE_WATCH = "file_watch"
	DATABASE = "database"
	MESSAGE_QUEUE = "message_queue"


class ScheduleType(str, Enum):
	"""Schedule type enumeration"""
	ONCE = "once"
	RECURRING = "recurring"
	CRON = "cron"
	INTERVAL = "interval"


@dataclass
class WorkflowTrigger:
	"""Workflow trigger configuration"""
	id: str = field(default_factory=uuid7str)
	name: str = ""
	trigger_type: TriggerType = TriggerType.MANUAL
	is_enabled: bool = True
	configuration: Dict[str, Any] = field(default_factory=dict)
	
	# Event-based triggers
	event_type: Optional[str] = None
	event_source: Optional[str] = None
	event_filter: Optional[Dict[str, Any]] = None
	
	# API/Webhook triggers
	endpoint_path: Optional[str] = None
	http_method: str = "POST"
	authentication_required: bool = True
	
	# File watch triggers
	watch_path: Optional[str] = None
	file_pattern: Optional[str] = None
	watch_events: List[str] = field(default_factory=lambda: ["created", "modified"])
	
	# Database triggers
	table_name: Optional[str] = None
	operation: Optional[str] = None  # INSERT, UPDATE, DELETE
	condition: Optional[str] = None
	
	# Message queue triggers
	queue_name: Optional[str] = None
	exchange: Optional[str] = None
	routing_key: Optional[str] = None
	
	created_at: datetime = field(default_factory=datetime.utcnow)
	updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class WorkflowSchedule:
	"""Workflow schedule configuration"""
	id: str = field(default_factory=uuid7str)
	name: str = ""
	schedule_type: ScheduleType = ScheduleType.ONCE
	is_enabled: bool = True
	
	# One-time schedule
	start_at: Optional[datetime] = None
	
	# Recurring schedule
	recurrence_pattern: Optional[str] = None  # daily, weekly, monthly, yearly
	interval_value: int = 1
	interval_unit: str = "days"  # minutes, hours, days, weeks, months
	
	# Cron expression
	cron_expression: Optional[str] = None
	timezone: str = "UTC"
	
	# Schedule limits
	end_at: Optional[datetime] = None
	max_executions: Optional[int] = None
	execution_count: int = 0
	
	# Advanced options
	retry_on_failure: bool = True
	max_retries: int = 3
	retry_delay: int = 300  # seconds
	
	# Status tracking
	last_execution: Optional[datetime] = None
	next_execution: Optional[datetime] = None
	
	created_at: datetime = field(default_factory=datetime.utcnow)
	updated_at: datetime = field(default_factory=datetime.utcnow)


class Workflow(BaseModel):
	"""Workflow model"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	name: str = Field(..., min_length=1, max_length=200)
	description: Optional[str] = None
	version: str = "1.0.0"
	status: WorkflowStatus = WorkflowStatus.DRAFT
	
	# Ownership and permissions
	owner_id: str = Field(..., min_length=1)
	owner_name: Optional[str] = None
	tenant_id: str = Field(..., min_length=1)
	
	# Categorization
	category: Optional[str] = None
	tags: List[str] = Field(default_factory=list)
	priority: int = Field(default=5, ge=1, le=10)
	
	# Workflow definition
	definition: Dict[str, Any] = Field(default_factory=dict)
	variables: Dict[str, Any] = Field(default_factory=dict)
	constants: Dict[str, Any] = Field(default_factory=dict)
	
	# Tasks and steps
	tasks: List[str] = Field(default_factory=list)  # Task IDs
	task_count: int = 0
	completed_tasks: int = 0
	
	# Triggers and scheduling
	triggers: List[WorkflowTrigger] = Field(default_factory=list)
	schedules: List[WorkflowSchedule] = Field(default_factory=list)
	
	# Execution settings
	timeout: Optional[int] = None  # seconds
	max_concurrent_instances: int = 1
	retry_policy: Dict[str, Any] = Field(
		default_factory=lambda: {
			"enabled": True,
			"max_attempts": 3,
			"delay": 60,
			"backoff_multiplier": 2.0,
			"max_delay": 3600
		}
	)
	
	# Permissions and access control
	is_public: bool = False
	shared_with: List[str] = Field(default_factory=list)  # User/Group IDs
	permissions: Dict[str, List[str]] = Field(default_factory=dict)
	
	# Execution statistics
	total_executions: int = 0
	successful_executions: int = 0
	failed_executions: int = 0
	average_duration: Optional[float] = None  # seconds
	last_execution: Optional[datetime] = None
	last_success: Optional[datetime] = None
	last_failure: Optional[datetime] = None
	
	# Metadata
	documentation: Optional[str] = None
	examples: List[Dict[str, Any]] = Field(default_factory=list)
	changelog: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: Optional[str] = None
	updated_by: Optional[str] = None
	
	@property
	def is_active(self) -> bool:
		"""Check if workflow is active"""
		return self.status == WorkflowStatus.ACTIVE
	
	@property
	def is_executable(self) -> bool:
		"""Check if workflow can be executed"""
		return self.status in [WorkflowStatus.ACTIVE, WorkflowStatus.PAUSED]
	
	@property
	def progress_percentage(self) -> float:
		"""Get workflow completion percentage"""
		if self.task_count == 0:
			return 0.0
		return (self.completed_tasks / self.task_count) * 100
	
	@property
	def success_rate(self) -> float:
		"""Get workflow success rate"""
		if self.total_executions == 0:
			return 0.0
		return (self.successful_executions / self.total_executions) * 100
	
	@property
	def has_active_triggers(self) -> bool:
		"""Check if workflow has active triggers"""
		return any(trigger.is_enabled for trigger in self.triggers)
	
	@property
	def has_active_schedules(self) -> bool:
		"""Check if workflow has active schedules"""
		return any(schedule.is_enabled for schedule in self.schedules)
	
	@property
	def next_scheduled_execution(self) -> Optional[datetime]:
		"""Get next scheduled execution time"""
		next_times = [
			schedule.next_execution 
			for schedule in self.schedules 
			if schedule.is_enabled and schedule.next_execution
		]
		return min(next_times) if next_times else None
	
	def add_trigger(self, trigger: WorkflowTrigger):
		"""Add a trigger to the workflow"""
		self.triggers.append(trigger)
		self.updated_at = datetime.utcnow()
	
	def remove_trigger(self, trigger_id: str):
		"""Remove a trigger from the workflow"""
		self.triggers = [t for t in self.triggers if t.id != trigger_id]
		self.updated_at = datetime.utcnow()
	
	def add_schedule(self, schedule: WorkflowSchedule):
		"""Add a schedule to the workflow"""
		self.schedules.append(schedule)
		self.updated_at = datetime.utcnow()
	
	def remove_schedule(self, schedule_id: str):
		"""Remove a schedule from the workflow"""
		self.schedules = [s for s in self.schedules if s.id != schedule_id]
		self.updated_at = datetime.utcnow()
	
	def update_execution_stats(self, success: bool, duration: float):
		"""Update execution statistics"""
		self.total_executions += 1
		self.last_execution = datetime.utcnow()
		
		if success:
			self.successful_executions += 1
			self.last_success = datetime.utcnow()
		else:
			self.failed_executions += 1
			self.last_failure = datetime.utcnow()
		
		# Update average duration
		if self.average_duration is None:
			self.average_duration = duration
		else:
			self.average_duration = (
				(self.average_duration * (self.total_executions - 1) + duration) 
				/ self.total_executions
			)
		
		self.updated_at = datetime.utcnow()
	
	def can_be_executed_by(self, user_id: str, user_permissions: List[str]) -> bool:
		"""Check if user can execute this workflow"""
		# Owner can always execute
		if self.owner_id == user_id:
			return True
		
		# Check if public
		if self.is_public:
			return True
		
		# Check if shared with user
		if user_id in self.shared_with:
			return True
		
		# Check permissions
		execute_perms = self.permissions.get("execute", [])
		return any(perm in user_permissions for perm in execute_perms)
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert workflow to dictionary"""
		return {
			"id": self.id,
			"name": self.name,
			"description": self.description,
			"version": self.version,
			"status": self.status.value,
			"owner_id": self.owner_id,
			"owner_name": self.owner_name,
			"tenant_id": self.tenant_id,
			"category": self.category,
			"tags": self.tags,
			"priority": self.priority,
			"task_count": self.task_count,
			"completed_tasks": self.completed_tasks,
			"progress_percentage": self.progress_percentage,
			"total_executions": self.total_executions,
			"success_rate": self.success_rate,
			"is_active": self.is_active,
			"is_executable": self.is_executable,
			"has_active_triggers": self.has_active_triggers,
			"has_active_schedules": self.has_active_schedules,
			"next_scheduled_execution": (
				self.next_scheduled_execution.isoformat() 
				if self.next_scheduled_execution else None
			),
			"last_execution": (
				self.last_execution.isoformat() 
				if self.last_execution else None
			),
			"created_at": self.created_at.isoformat(),
			"updated_at": self.updated_at.isoformat(),
		}


class WorkflowInstance(BaseModel):
	"""Workflow execution instance"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	workflow_id: str = Field(..., min_length=1)
	workflow_name: Optional[str] = None
	workflow_version: str = "1.0.0"
	
	status: WorkflowStatus = WorkflowStatus.DRAFT
	progress: float = Field(default=0.0, ge=0.0, le=100.0)
	
	# Execution context
	triggered_by: Optional[str] = None  # User ID or system
	trigger_type: TriggerType = TriggerType.MANUAL
	trigger_data: Dict[str, Any] = Field(default_factory=dict)
	
	# Input/Output data
	input_data: Dict[str, Any] = Field(default_factory=dict)
	output_data: Dict[str, Any] = Field(default_factory=dict)
	variables: Dict[str, Any] = Field(default_factory=dict)
	
	# Execution tracking
	started_at: Optional[datetime] = None
	completed_at: Optional[datetime] = None
	duration: Optional[float] = None  # seconds
	
	# Task execution
	current_task_id: Optional[str] = None
	completed_task_ids: List[str] = Field(default_factory=list)
	failed_task_ids: List[str] = Field(default_factory=list)
	
	# Error handling
	error_message: Optional[str] = None
	error_details: Dict[str, Any] = Field(default_factory=dict)
	retry_count: int = 0
	max_retries: int = 3
	
	# Audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	
	@property
	def is_running(self) -> bool:
		"""Check if instance is currently running"""
		return self.status in [WorkflowStatus.ACTIVE]
	
	@property
	def is_completed(self) -> bool:
		"""Check if instance is completed (success or failure)"""
		return self.status in [
			WorkflowStatus.COMPLETED, 
			WorkflowStatus.FAILED, 
			WorkflowStatus.CANCELLED
		]
	
	@property
	def can_be_retried(self) -> bool:
		"""Check if instance can be retried"""
		return (
			self.status == WorkflowStatus.FAILED and 
			self.retry_count < self.max_retries
		)
	
	def start_execution(self, user_id: Optional[str] = None):
		"""Start workflow execution"""
		self.status = WorkflowStatus.ACTIVE
		self.started_at = datetime.utcnow()
		self.triggered_by = user_id
		self.updated_at = datetime.utcnow()
	
	def complete_execution(self, success: bool = True, error: Optional[str] = None):
		"""Complete workflow execution"""
		self.completed_at = datetime.utcnow()
		self.progress = 100.0
		
		if success:
			self.status = WorkflowStatus.COMPLETED
		else:
			self.status = WorkflowStatus.FAILED
			self.error_message = error
		
		if self.started_at:
			self.duration = (self.completed_at - self.started_at).total_seconds()
		
		self.updated_at = datetime.utcnow()
	
	def update_progress(self, progress: float, current_task_id: Optional[str] = None):
		"""Update execution progress"""
		self.progress = max(0.0, min(100.0, progress))
		if current_task_id:
			self.current_task_id = current_task_id
		self.updated_at = datetime.utcnow()
	
	def add_completed_task(self, task_id: str):
		"""Mark task as completed"""
		if task_id not in self.completed_task_ids:
			self.completed_task_ids.append(task_id)
		self.updated_at = datetime.utcnow()
	
	def add_failed_task(self, task_id: str):
		"""Mark task as failed"""
		if task_id not in self.failed_task_ids:
			self.failed_task_ids.append(task_id)
		self.updated_at = datetime.utcnow()
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert instance to dictionary"""
		return {
			"id": self.id,
			"workflow_id": self.workflow_id,
			"workflow_name": self.workflow_name,
			"status": self.status.value,
			"progress": self.progress,
			"triggered_by": self.triggered_by,
			"trigger_type": self.trigger_type.value,
			"started_at": self.started_at.isoformat() if self.started_at else None,
			"completed_at": self.completed_at.isoformat() if self.completed_at else None,
			"duration": self.duration,
			"error_message": self.error_message,
			"retry_count": self.retry_count,
			"is_running": self.is_running,
			"is_completed": self.is_completed,
			"can_be_retried": self.can_be_retried,
			"created_at": self.created_at.isoformat(),
			"updated_at": self.updated_at.isoformat(),
		}