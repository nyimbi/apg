"""
Notification model and related data structures

© 2025 Datacraft. All rights reserved.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, ConfigDict
import uuid


class NotificationType(str, Enum):
	"""Notification type enumeration"""
	INFO = "info"
	SUCCESS = "success"
	WARNING = "warning"
	ERROR = "error"
	WORKFLOW_UPDATE = "workflow_update"
	TASK_ASSIGNMENT = "task_assignment"
	TASK_COMPLETE = "task_complete"
	TASK_OVERDUE = "task_overdue"
	APPROVAL_REQUEST = "approval_request"
	SYSTEM_ALERT = "system_alert"
	CHAT_MESSAGE = "chat_message"
	FILE_SHARED = "file_shared"
	REMINDER = "reminder"


class NotificationPriority(str, Enum):
	"""Notification priority enumeration"""
	LOW = "low"
	NORMAL = "normal"
	HIGH = "high"
	URGENT = "urgent"
	CRITICAL = "critical"


class NotificationChannel(str, Enum):
	"""Notification delivery channel enumeration"""
	IN_APP = "in_app"
	PUSH = "push"
	EMAIL = "email"
	SMS = "sms"
	SLACK = "slack"
	TEAMS = "teams"
	WEBHOOK = "webhook"


@dataclass
class NotificationAction:
	"""Notification action button"""
	id: str = field(default_factory=lambda: str(uuid.uuid4()))
	label: str = ""
	action_type: str = "navigate"  # navigate, api_call, dismiss
	action_data: Dict[str, Any] = field(default_factory=dict)
	is_primary: bool = False
	is_destructive: bool = False
	
	# Navigation action
	screen: Optional[str] = None
	params: Dict[str, Any] = field(default_factory=dict)
	
	# API call action  
	endpoint: Optional[str] = None
	method: str = "POST"
	payload: Dict[str, Any] = field(default_factory=dict)
	
	# JavaScript action (for web-based notifications)
	javascript: Optional[str] = None


@dataclass
class NotificationAttachment:
	"""Notification attachment"""
	id: str = field(default_factory=lambda: str(uuid.uuid4()))
	type: str = ""  # image, document, link, etc.
	title: Optional[str] = None
	url: Optional[str] = None
	thumbnail_url: Optional[str] = None
	file_size: Optional[int] = None
	mime_type: Optional[str] = None
	metadata: Dict[str, Any] = field(default_factory=dict)


class Notification(BaseModel):
	"""Notification model"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=lambda: str(uuid.uuid4()))
	title: str = Field(..., min_length=1, max_length=200)
	message: str = Field(..., min_length=1, max_length=1000)
	notification_type: NotificationType = NotificationType.INFO
	priority: NotificationPriority = NotificationPriority.NORMAL
	
	# Recipients
	recipient_id: str = Field(..., min_length=1)
	recipient_type: str = "user"  # user, group, role, tenant
	sender_id: Optional[str] = None
	sender_name: Optional[str] = None
	
	# Context and metadata
	context: Dict[str, Any] = Field(default_factory=dict)
	metadata: Dict[str, Any] = Field(default_factory=dict)
	
	# Related entities
	workflow_id: Optional[str] = None
	workflow_instance_id: Optional[str] = None
	task_id: Optional[str] = None
	file_id: Optional[str] = None
	
	# Content and media
	icon: Optional[str] = None
	image_url: Optional[str] = None
	attachments: List[NotificationAttachment] = Field(default_factory=list)
	
	# Actions
	actions: List[NotificationAction] = Field(default_factory=list)
	default_action: Optional[NotificationAction] = None
	
	# Delivery
	channels: List[NotificationChannel] = Field(default_factory=lambda: [NotificationChannel.IN_APP])
	delivery_schedule: Optional[datetime] = None
	expiry_date: Optional[datetime] = None
	
	# Status tracking
	is_read: bool = False
	is_delivered: bool = False
	is_clicked: bool = False
	read_at: Optional[datetime] = None
	delivered_at: Optional[datetime] = None
	clicked_at: Optional[datetime] = None
	
	# Delivery attempts
	delivery_attempts: int = 0
	last_delivery_attempt: Optional[datetime] = None
	delivery_errors: List[str] = Field(default_factory=list)
	
	# Grouping and threading
	group_key: Optional[str] = None
	thread_id: Optional[str] = None
	reply_to_id: Optional[str] = None
	
	# Display preferences
	is_silent: bool = False
	sound: Optional[str] = None
	vibration_pattern: Optional[List[int]] = None
	badge_count: Optional[int] = None
	
	# Tenant context
	tenant_id: str = Field(..., min_length=1)
	
	# Audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: Optional[str] = None
	
	@property
	def age_minutes(self) -> int:
		"""Get notification age in minutes"""
		return int((datetime.utcnow() - self.created_at).total_seconds() / 60)
	
	@property
	def age_hours(self) -> int:
		"""Get notification age in hours"""
		return int(self.age_minutes / 60)
	
	@property
	def is_expired(self) -> bool:
		"""Check if notification is expired"""
		if not self.expiry_date:
			return False
		return datetime.utcnow() > self.expiry_date
	
	@property
	def is_scheduled(self) -> bool:
		"""Check if notification is scheduled for future delivery"""
		if not self.delivery_schedule:
			return False
		return datetime.utcnow() < self.delivery_schedule
	
	@property
	def should_be_delivered(self) -> bool:
		"""Check if notification should be delivered now"""
		if self.is_expired or self.is_delivered:
			return False
		if self.delivery_schedule and datetime.utcnow() < self.delivery_schedule:
			return False
		return True
	
	@property
	def display_title(self) -> str:
		"""Get display title with sender info if available"""
		if self.sender_name:
			return f"{self.sender_name}: {self.title}"
		return self.title
	
	@property
	def priority_score(self) -> int:
		"""Get numeric priority score for sorting"""
		priority_scores = {
			NotificationPriority.LOW: 1,
			NotificationPriority.NORMAL: 2,
			NotificationPriority.HIGH: 3,
			NotificationPriority.URGENT: 4,
			NotificationPriority.CRITICAL: 5
		}
		return priority_scores.get(self.priority, 2)
	
	def mark_as_read(self, user_id: Optional[str] = None):
		"""Mark notification as read"""
		if not self.is_read:
			self.is_read = True
			self.read_at = datetime.utcnow()
			self.updated_at = datetime.utcnow()
	
	def mark_as_clicked(self, user_id: Optional[str] = None):
		"""Mark notification as clicked"""
		if not self.is_clicked:
			self.is_clicked = True
			self.clicked_at = datetime.utcnow()
			self.updated_at = datetime.utcnow()
			
		# Also mark as read when clicked
		self.mark_as_read(user_id)
	
	def mark_as_delivered(self, channel: NotificationChannel):
		"""Mark notification as delivered"""
		if not self.is_delivered:
			self.is_delivered = True
			self.delivered_at = datetime.utcnow()
			self.updated_at = datetime.utcnow()
	
	def add_delivery_error(self, error_message: str, channel: NotificationChannel):
		"""Add delivery error"""
		self.delivery_attempts += 1
		self.last_delivery_attempt = datetime.utcnow()
		self.delivery_errors.append(f"{channel.value}: {error_message}")
		self.updated_at = datetime.utcnow()
	
	def add_action(self, label: str, action_type: str, action_data: Dict[str, Any],
				   is_primary: bool = False) -> NotificationAction:
		"""Add action button to notification"""
		action = NotificationAction(
			label=label,
			action_type=action_type,
			action_data=action_data,
			is_primary=is_primary
		)
		self.actions.append(action)
		self.updated_at = datetime.utcnow()
		return action
	
	def add_attachment(self, attachment_type: str, title: str, url: str,
					   thumbnail_url: Optional[str] = None) -> NotificationAttachment:
		"""Add attachment to notification"""
		attachment = NotificationAttachment(
			type=attachment_type,
			title=title,
			url=url,
			thumbnail_url=thumbnail_url
		)
		self.attachments.append(attachment)
		self.updated_at = datetime.utcnow()
		return attachment
	
	def can_be_delivered_via(self, channel: NotificationChannel) -> bool:
		"""Check if notification can be delivered via specific channel"""
		return channel in self.channels
	
	def should_retry_delivery(self, max_attempts: int = 3) -> bool:
		"""Check if delivery should be retried"""
		return (
			not self.is_delivered and 
			self.delivery_attempts < max_attempts and
			not self.is_expired and
			self.should_be_delivered
		)
	
	def get_retry_delay(self) -> int:
		"""Get delay in seconds before next retry attempt"""
		# Exponential backoff: 60s, 300s, 900s
		base_delay = 60
		return base_delay * (3 ** min(self.delivery_attempts, 3))
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert notification to dictionary"""
		return {
			"id": self.id,
			"title": self.title,
			"message": self.message,
			"notification_type": self.notification_type.value,
			"priority": self.priority.value,
			"priority_score": self.priority_score,
			"recipient_id": self.recipient_id,
			"sender_id": self.sender_id,
			"sender_name": self.sender_name,
			"workflow_id": self.workflow_id,
			"task_id": self.task_id,
			"icon": self.icon,
			"image_url": self.image_url,
			"is_read": self.is_read,
			"is_delivered": self.is_delivered,
			"is_clicked": self.is_clicked,
			"is_expired": self.is_expired,
			"is_scheduled": self.is_scheduled,
			"age_minutes": self.age_minutes,
			"age_hours": self.age_hours,
			"channels": [c.value for c in self.channels],
			"actions": [
				{
					"id": action.id,
					"label": action.label,
					"action_type": action.action_type,
					"is_primary": action.is_primary,
					"action_data": action.action_data
				} for action in self.actions
			],
			"attachments": [
				{
					"id": att.id,
					"type": att.type,
					"title": att.title,
					"url": att.url,
					"thumbnail_url": att.thumbnail_url
				} for att in self.attachments
			],
			"read_at": self.read_at.isoformat() if self.read_at else None,
			"delivered_at": self.delivered_at.isoformat() if self.delivered_at else None,
			"clicked_at": self.clicked_at.isoformat() if self.clicked_at else None,
			"created_at": self.created_at.isoformat(),
			"updated_at": self.updated_at.isoformat(),
		}


class NotificationPreferences(BaseModel):
	"""User notification preferences"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=lambda: str(uuid.uuid4()))
	user_id: str = Field(..., min_length=1)
	tenant_id: str = Field(..., min_length=1)
	
	# Channel preferences
	enable_push_notifications: bool = True
	enable_email_notifications: bool = True
	enable_sms_notifications: bool = False
	enable_in_app_notifications: bool = True
	
	# Type preferences
	workflow_notifications: bool = True
	task_notifications: bool = True
	system_notifications: bool = True
	chat_notifications: bool = True
	reminder_notifications: bool = True
	
	# Timing preferences
	quiet_hours_enabled: bool = False
	quiet_hours_start: Optional[str] = None  # HH:MM format
	quiet_hours_end: Optional[str] = None    # HH:MM format
	timezone: str = "UTC"
	
	# Grouping preferences
	group_similar_notifications: bool = True
	max_notifications_per_group: int = 5
	
	# Priority filtering
	minimum_priority: NotificationPriority = NotificationPriority.NORMAL
	critical_only_during_quiet_hours: bool = True
	
	# Delivery preferences
	batch_notifications: bool = False
	batch_interval_minutes: int = 15
	immediate_for_urgent: bool = True
	
	# Sound and vibration
	enable_sound: bool = True
	enable_vibration: bool = True
	custom_sound: Optional[str] = None
	
	# Audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	
	def is_in_quiet_hours(self, check_time: Optional[datetime] = None) -> bool:
		"""Check if current time is in quiet hours"""
		if not self.quiet_hours_enabled or not self.quiet_hours_start or not self.quiet_hours_end:
			return False
		
		if not check_time:
			check_time = datetime.utcnow()
		
		# Convert to user's timezone
		# For now, just use the time portion
		current_time = check_time.strftime("%H:%M")
		
		return self.quiet_hours_start <= current_time <= self.quiet_hours_end
	
	def should_deliver_notification(self, notification: Notification) -> bool:
		"""Check if notification should be delivered based on preferences"""
		# Check if notification type is enabled
		type_preferences = {
			NotificationType.WORKFLOW_UPDATE: self.workflow_notifications,
			NotificationType.TASK_ASSIGNMENT: self.task_notifications,
			NotificationType.TASK_COMPLETE: self.task_notifications,
			NotificationType.TASK_OVERDUE: self.task_notifications,
			NotificationType.SYSTEM_ALERT: self.system_notifications,
			NotificationType.CHAT_MESSAGE: self.chat_notifications,
			NotificationType.REMINDER: self.reminder_notifications,
		}
		
		if notification.notification_type in type_preferences:
			if not type_preferences[notification.notification_type]:
				return False
		
		# Check priority filtering
		if notification.priority_score < self.minimum_priority.value:
			return False
		
		# Check quiet hours
		if self.is_in_quiet_hours():
			if self.critical_only_during_quiet_hours:
				return notification.priority == NotificationPriority.CRITICAL
			return False
		
		return True
	
	def get_enabled_channels(self, notification: Notification) -> List[NotificationChannel]:
		"""Get enabled channels for notification delivery"""
		enabled_channels = []
		
		if self.enable_in_app_notifications:
			enabled_channels.append(NotificationChannel.IN_APP)
		
		if self.enable_push_notifications:
			enabled_channels.append(NotificationChannel.PUSH)
		
		if self.enable_email_notifications:
			enabled_channels.append(NotificationChannel.EMAIL)
		
		if self.enable_sms_notifications:
			enabled_channels.append(NotificationChannel.SMS)
		
		# Filter by notification's preferred channels
		return [ch for ch in enabled_channels if ch in notification.channels]


@dataclass
class NotificationTemplate:
	"""Template for creating notifications"""
	id: str = field(default_factory=lambda: str(uuid.uuid4()))
	name: str = ""
	notification_type: NotificationType = NotificationType.INFO
	title_template: str = ""
	message_template: str = ""
	icon: Optional[str] = None
	priority: NotificationPriority = NotificationPriority.NORMAL
	channels: List[NotificationChannel] = field(default_factory=lambda: [NotificationChannel.IN_APP])
	actions: List[Dict[str, Any]] = field(default_factory=list)
	metadata: Dict[str, Any] = field(default_factory=dict)
	
	def create_notification(self, recipient_id: str, tenant_id: str, 
						   context: Dict[str, Any], **kwargs) -> Notification:
		"""Create notification from template"""
		# Replace template variables
		title = self._replace_variables(self.title_template, context)
		message = self._replace_variables(self.message_template, context)
		
		# Create actions from template
		actions = []
		for action_template in self.actions:
			action = NotificationAction(
				label=self._replace_variables(action_template["label"], context),
				action_type=action_template["action_type"],
				action_data=action_template.get("action_data", {}),
				is_primary=action_template.get("is_primary", False)
			)
			actions.append(action)
		
		return Notification(
			title=title,
			message=message,
			notification_type=self.notification_type,
			priority=self.priority,
			recipient_id=recipient_id,
			tenant_id=tenant_id,
			icon=self.icon,
			channels=self.channels,
			actions=actions,
			context=context,
			metadata=self.metadata,
			**kwargs
		)
	
	def _replace_variables(self, template: str, context: Dict[str, Any]) -> str:
		"""Replace template variables with context values"""
		result = template
		for key, value in context.items():
			placeholder = f"{{{key}}}"
			if placeholder in result:
				result = result.replace(placeholder, str(value))
		return result