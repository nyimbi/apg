"""
Task model and related data structures

© 2025 Datacraft. All rights reserved.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, ConfigDict
from uuid_extensions import uuid7str


class TaskStatus(str, Enum):
	"""Task status enumeration"""
	PENDING = "pending"
	IN_PROGRESS = "in_progress"
	COMPLETED = "completed"
	FAILED = "failed"
	CANCELLED = "cancelled"
	PAUSED = "paused"
	SKIPPED = "skipped"
	WAITING = "waiting"


class TaskPriority(str, Enum):
	"""Task priority enumeration"""
	LOW = "low"
	NORMAL = "normal"
	HIGH = "high"
	URGENT = "urgent"
	CRITICAL = "critical"


class TaskType(str, Enum):
	"""Task type enumeration"""
	MANUAL = "manual"
	AUTOMATED = "automated"
	API_CALL = "api_call"
	DATA_PROCESSING = "data_processing"
	FILE_OPERATION = "file_operation"
	EMAIL = "email"
	NOTIFICATION = "notification"
	APPROVAL = "approval"
	REVIEW = "review"
	SCRIPT = "script"
	WORKFLOW = "workflow"
	INTEGRATION = "integration"


@dataclass
class TaskAssignment:
	"""Task assignment information"""
	assignee_id: str
	assignee_name: Optional[str] = None
	assignee_email: Optional[str] = None
	assigned_at: datetime = field(default_factory=datetime.utcnow)
	assigned_by: Optional[str] = None
	due_date: Optional[datetime] = None
	is_accepted: bool = False
	accepted_at: Optional[datetime] = None
	
	@property
	def is_overdue(self) -> bool:
		"""Check if task assignment is overdue"""
		if not self.due_date:
			return False
		return datetime.utcnow() > self.due_date
	
	@property
	def days_until_due(self) -> Optional[int]:
		"""Get days until due date"""
		if not self.due_date:
			return None
		delta = self.due_date - datetime.utcnow()
		return delta.days


@dataclass
class TaskComment:
	"""Task comment"""
	id: str = field(default_factory=uuid7str)
	content: str = ""
	author_id: str = ""
	author_name: Optional[str] = None
	created_at: datetime = field(default_factory=datetime.utcnow)
	updated_at: datetime = field(default_factory=datetime.utcnow)
	is_internal: bool = False
	attachments: List[str] = field(default_factory=list)


@dataclass  
class TaskAttachment:
	"""Task attachment"""
	id: str = field(default_factory=uuid7str)
	filename: str = ""
	original_filename: str = ""
	file_size: int = 0
	mime_type: str = ""
	file_path: str = ""
	url: Optional[str] = None
	uploaded_by: str = ""
	uploaded_at: datetime = field(default_factory=datetime.utcnow)
	is_public: bool = False


class Task(BaseModel):
	"""Task model"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	name: str = Field(..., min_length=1, max_length=200)
	description: Optional[str] = None
	task_type: TaskType = TaskType.MANUAL
	status: TaskStatus = TaskStatus.PENDING
	priority: TaskPriority = TaskPriority.NORMAL
	
	# Workflow association
	workflow_id: Optional[str] = None
	workflow_name: Optional[str] = None
	workflow_instance_id: Optional[str] = None
	sequence_number: int = 0
	
	# Assignment and ownership
	owner_id: str = Field(..., min_length=1)
	owner_name: Optional[str] = None
	assignment: Optional[TaskAssignment] = None
	tenant_id: str = Field(..., min_length=1)
	
	# Task configuration
	configuration: Dict[str, Any] = Field(default_factory=dict)
	input_schema: Optional[Dict[str, Any]] = None
	output_schema: Optional[Dict[str, Any]] = None
	
	# Input/Output data
	input_data: Dict[str, Any] = Field(default_factory=dict)
	output_data: Dict[str, Any] = Field(default_factory=dict)
	
	# Execution tracking
	progress: float = Field(default=0.0, ge=0.0, le=100.0)
	started_at: Optional[datetime] = None
	completed_at: Optional[datetime] = None
	duration: Optional[float] = None  # seconds
	
	# Dependencies
	depends_on: List[str] = Field(default_factory=list)  # Task IDs
	blocks: List[str] = Field(default_factory=list)  # Task IDs this blocks
	
	# Timing
	estimated_duration: Optional[int] = None  # minutes
	due_date: Optional[datetime] = None
	reminder_sent: bool = False
	
	# Error handling
	error_message: Optional[str] = None
	error_details: Dict[str, Any] = Field(default_factory=dict)
	retry_count: int = 0
	max_retries: int = 3
	retry_delay: int = 60  # seconds
	
	# Collaboration
	comments: List[TaskComment] = Field(default_factory=list)
	attachments: List[TaskAttachment] = Field(default_factory=list)
	watchers: List[str] = Field(default_factory=list)  # User IDs
	
	# Categorization
	category: Optional[str] = None
	tags: List[str] = Field(default_factory=list)
	labels: Dict[str, str] = Field(default_factory=dict)
	
	# Approval workflow
	requires_approval: bool = False
	approver_id: Optional[str] = None
	approved_at: Optional[datetime] = None
	approval_notes: Optional[str] = None
	
	# Automation
	script_path: Optional[str] = None
	api_endpoint: Optional[str] = None
	automation_config: Dict[str, Any] = Field(default_factory=dict)
	
	# Metrics and tracking
	view_count: int = 0
	last_viewed_at: Optional[datetime] = None
	last_viewed_by: Optional[str] = None
	
	# Audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: Optional[str] = None
	updated_by: Optional[str] = None
	
	@property
	def is_assigned(self) -> bool:
		"""Check if task is assigned"""
		return self.assignment is not None
	
	@property
	def is_overdue(self) -> bool:
		"""Check if task is overdue"""
		if not self.due_date:
			return False
		return datetime.utcnow() > self.due_date and not self.is_completed
	
	@property
	def is_completed(self) -> bool:
		"""Check if task is completed"""
		return self.status in [TaskStatus.COMPLETED, TaskStatus.SKIPPED]
	
	@property
	def is_active(self) -> bool:
		"""Check if task is actively being worked on"""
		return self.status == TaskStatus.IN_PROGRESS
	
	@property
	def is_waiting(self) -> bool:
		"""Check if task is waiting for dependencies"""
		return self.status == TaskStatus.WAITING
	
	@property
	def can_be_started(self) -> bool:
		"""Check if task can be started"""
		return self.status in [TaskStatus.PENDING, TaskStatus.WAITING] and self.dependencies_met
	
	@property
	def dependencies_met(self) -> bool:
		"""Check if all dependencies are met (placeholder for now)"""
		# This would require checking the status of dependent tasks
		# For now, return True
		return True
	
	@property
	def days_until_due(self) -> Optional[int]:
		"""Get days until due date"""
		if not self.due_date:
			return None
		delta = self.due_date - datetime.utcnow()
		return delta.days
	
	@property
	def estimated_completion(self) -> Optional[datetime]:
		"""Estimate completion time based on progress and estimated duration"""
		if not self.estimated_duration or self.progress == 0:
			return None
		
		remaining_progress = 100 - self.progress
		if remaining_progress <= 0:
			return datetime.utcnow()
		
		remaining_time = (remaining_progress / 100) * self.estimated_duration * 60  # Convert to seconds
		return datetime.utcnow() + timedelta(seconds=remaining_time)
	
	@property
	def assignee_name(self) -> Optional[str]:
		"""Get assignee name"""
		return self.assignment.assignee_name if self.assignment else None
	
	@property
	def assignee_id(self) -> Optional[str]:
		"""Get assignee ID"""
		return self.assignment.assignee_id if self.assignment else None
	
	def assign_to(self, user_id: str, user_name: Optional[str] = None, 
				  due_date: Optional[datetime] = None, assigned_by: Optional[str] = None):
		"""Assign task to a user"""
		self.assignment = TaskAssignment(
			assignee_id=user_id,
			assignee_name=user_name,
			due_date=due_date,
			assigned_by=assigned_by
		)
		self.updated_at = datetime.utcnow()
	
	def accept_assignment(self, user_id: str):
		"""Accept task assignment"""
		if self.assignment and self.assignment.assignee_id == user_id:
			self.assignment.is_accepted = True
			self.assignment.accepted_at = datetime.utcnow()
			self.updated_at = datetime.utcnow()
	
	def start_task(self, user_id: Optional[str] = None):
		"""Start task execution"""
		self.status = TaskStatus.IN_PROGRESS
		self.started_at = datetime.utcnow()
		if user_id:
			self.updated_by = user_id
		self.updated_at = datetime.utcnow()
	
	def complete_task(self, output_data: Optional[Dict[str, Any]] = None, 
					 user_id: Optional[str] = None):
		"""Complete task execution"""
		self.status = TaskStatus.COMPLETED
		self.completed_at = datetime.utcnow()
		self.progress = 100.0
		
		if output_data:
			self.output_data.update(output_data)
		
		if self.started_at:
			self.duration = (self.completed_at - self.started_at).total_seconds()
		
		if user_id:
			self.updated_by = user_id
		self.updated_at = datetime.utcnow()
	
	def fail_task(self, error_message: str, error_details: Optional[Dict[str, Any]] = None,
				  user_id: Optional[str] = None):
		"""Mark task as failed"""
		self.status = TaskStatus.FAILED
		self.error_message = error_message
		
		if error_details:
			self.error_details.update(error_details)
		
		if user_id:
			self.updated_by = user_id
		self.updated_at = datetime.utcnow()
	
	def pause_task(self, user_id: Optional[str] = None):
		"""Pause task execution"""
		self.status = TaskStatus.PAUSED
		if user_id:
			self.updated_by = user_id
		self.updated_at = datetime.utcnow()
	
	def resume_task(self, user_id: Optional[str] = None):
		"""Resume task execution"""
		self.status = TaskStatus.IN_PROGRESS
		if user_id:
			self.updated_by = user_id
		self.updated_at = datetime.utcnow()
	
	def cancel_task(self, user_id: Optional[str] = None):
		"""Cancel task execution"""
		self.status = TaskStatus.CANCELLED
		if user_id:
			self.updated_by = user_id
		self.updated_at = datetime.utcnow()
	
	def update_progress(self, progress: float, user_id: Optional[str] = None):
		"""Update task progress"""
		self.progress = max(0.0, min(100.0, progress))
		if user_id:
			self.updated_by = user_id
		self.updated_at = datetime.utcnow()
	
	def add_comment(self, content: str, author_id: str, author_name: Optional[str] = None,
					is_internal: bool = False) -> TaskComment:
		"""Add a comment to the task"""
		comment = TaskComment(
			content=content,
			author_id=author_id,
			author_name=author_name,
			is_internal=is_internal
		)
		self.comments.append(comment)
		self.updated_at = datetime.utcnow()
		return comment
	
	def add_attachment(self, filename: str, file_path: str, uploaded_by: str,
					   file_size: int = 0, mime_type: str = "") -> TaskAttachment:
		"""Add an attachment to the task"""
		attachment = TaskAttachment(
			filename=filename,
			original_filename=filename,
			file_path=file_path,
			file_size=file_size,
			mime_type=mime_type,
			uploaded_by=uploaded_by
		)
		self.attachments.append(attachment)
		self.updated_at = datetime.utcnow()
		return attachment
	
	def add_watcher(self, user_id: str):
		"""Add a user as a watcher"""
		if user_id not in self.watchers:
			self.watchers.append(user_id)
			self.updated_at = datetime.utcnow()
	
	def remove_watcher(self, user_id: str):
		"""Remove a user from watchers"""
		if user_id in self.watchers:
			self.watchers.remove(user_id)
			self.updated_at = datetime.utcnow()
	
	def approve(self, approver_id: str, notes: Optional[str] = None):
		"""Approve the task"""
		self.approver_id = approver_id
		self.approved_at = datetime.utcnow()
		self.approval_notes = notes
		self.updated_at = datetime.utcnow()
	
	def record_view(self, user_id: str):
		"""Record that a user viewed the task"""
		self.view_count += 1
		self.last_viewed_at = datetime.utcnow()
		self.last_viewed_by = user_id
		self.updated_at = datetime.utcnow()
	
	def can_be_viewed_by(self, user_id: str, user_permissions: List[str]) -> bool:
		"""Check if user can view this task"""
		# Owner can always view
		if self.owner_id == user_id:
			return True
		
		# Assignee can view
		if self.assignment and self.assignment.assignee_id == user_id:
			return True
		
		# Watchers can view
		if user_id in self.watchers:
			return True
		
		# Check if user has view permission for this tenant/workflow
		view_perms = ["view_tasks", "view_all_tasks"]
		return any(perm in user_permissions for perm in view_perms)
	
	def can_be_edited_by(self, user_id: str, user_permissions: List[str]) -> bool:
		"""Check if user can edit this task"""
		# Owner can always edit
		if self.owner_id == user_id:
			return True
		
		# Assignee can edit if task is assigned to them
		if self.assignment and self.assignment.assignee_id == user_id:
			return True
		
		# Check if user has edit permission
		edit_perms = ["edit_tasks", "edit_all_tasks"]
		return any(perm in user_permissions for perm in edit_perms)
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert task to dictionary"""
		return {
			"id": self.id,
			"name": self.name,
			"description": self.description,
			"task_type": self.task_type.value,
			"status": self.status.value,
			"priority": self.priority.value,
			"workflow_id": self.workflow_id,
			"workflow_name": self.workflow_name,
			"owner_id": self.owner_id,
			"owner_name": self.owner_name,
			"assignee_id": self.assignee_id,
			"assignee_name": self.assignee_name,
			"progress": self.progress,
			"is_assigned": self.is_assigned,
			"is_overdue": self.is_overdue,
			"is_completed": self.is_completed,
			"is_active": self.is_active,
			"can_be_started": self.can_be_started,
			"days_until_due": self.days_until_due,
			"estimated_completion": (
				self.estimated_completion.isoformat() 
				if self.estimated_completion else None
			),
			"due_date": self.due_date.isoformat() if self.due_date else None,
			"started_at": self.started_at.isoformat() if self.started_at else None,
			"completed_at": self.completed_at.isoformat() if self.completed_at else None,
			"duration": self.duration,
			"error_message": self.error_message,
			"tags": self.tags,
			"category": self.category,
			"created_at": self.created_at.isoformat(),
			"updated_at": self.updated_at.isoformat(),
		}