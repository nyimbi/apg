"""
APG Workflow & Business Process Management - Advanced Notification System

Comprehensive notification system supporting multiple channels, smart routing,
and contextual notifications with APG platform integration.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
from collections import defaultdict, deque
import re

from models import (
	APGTenantContext, WBPMServiceResponse, WBPMPagedResponse,
	WBPMProcessInstance, WBPMTask, ProcessStatus, TaskStatus, TaskPriority
)

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Notification Core Classes
# =============================================================================

class NotificationChannel(str, Enum):
	"""Available notification channels."""
	EMAIL = "email"
	SMS = "sms"
	PUSH = "push"
	SLACK = "slack"
	TEAMS = "teams"
	WEBHOOK = "webhook"
	IN_APP = "in_app"
	DASHBOARD = "dashboard"


class NotificationPriority(str, Enum):
	"""Notification priority levels."""
	URGENT = "urgent"
	HIGH = "high"
	NORMAL = "normal"
	LOW = "low"


class NotificationStatus(str, Enum):
	"""Notification delivery status."""
	PENDING = "pending"
	SENT = "sent"
	DELIVERED = "delivered"
	FAILED = "failed"
	ACKNOWLEDGED = "acknowledged"
	EXPIRED = "expired"


class NotificationType(str, Enum):
	"""Types of notifications."""
	TASK_ASSIGNMENT = "task_assignment"
	TASK_OVERDUE = "task_overdue"
	TASK_COMPLETED = "task_completed"
	PROCESS_STARTED = "process_started"
	PROCESS_COMPLETED = "process_completed"
	PROCESS_DELAYED = "process_delayed"
	APPROVAL_REQUEST = "approval_request"
	APPROVAL_GRANTED = "approval_granted"
	APPROVAL_DENIED = "approval_denied"
	ALERT_TRIGGERED = "alert_triggered"
	ESCALATION = "escalation"
	REMINDER = "reminder"
	SYSTEM_MAINTENANCE = "system_maintenance"
	CUSTOM = "custom"


class DeliveryRule(str, Enum):
	"""Notification delivery rules."""
	IMMEDIATE = "immediate"
	BATCH_HOURLY = "batch_hourly"
	BATCH_DAILY = "batch_daily"
	BUSINESS_HOURS_ONLY = "business_hours_only"
	ESCALATION_AFTER_DELAY = "escalation_after_delay"


@dataclass
class NotificationTemplate:
	"""Notification message template."""
	template_id: str = field(default_factory=lambda: f"template_{uuid.uuid4().hex}")
	template_name: str = ""
	notification_type: NotificationType = NotificationType.CUSTOM
	channel: NotificationChannel = NotificationChannel.EMAIL
	subject_template: str = ""
	body_template: str = ""
	variables: List[str] = field(default_factory=list)
	locale: str = "en"
	created_by: str = ""
	created_at: datetime = field(default_factory=datetime.utcnow)
	tenant_id: str = ""


@dataclass
class NotificationPreference:
	"""User notification preferences."""
	preference_id: str = field(default_factory=lambda: f"pref_{uuid.uuid4().hex}")
	user_id: str = ""
	notification_type: NotificationType = NotificationType.CUSTOM
	enabled_channels: List[NotificationChannel] = field(default_factory=list)
	delivery_rule: DeliveryRule = DeliveryRule.IMMEDIATE
	quiet_hours_start: Optional[str] = None  # "22:00"
	quiet_hours_end: Optional[str] = None    # "08:00"
	timezone: str = "UTC"
	max_frequency: Optional[int] = None  # Max notifications per hour
	tenant_id: str = ""


@dataclass
class NotificationMessage:
	"""Individual notification message."""
	message_id: str = field(default_factory=lambda: f"msg_{uuid.uuid4().hex}")
	notification_type: NotificationType = NotificationType.CUSTOM
	channel: NotificationChannel = NotificationChannel.EMAIL
	priority: NotificationPriority = NotificationPriority.NORMAL
	recipient_id: str = ""
	recipient_address: str = ""  # email, phone, webhook URL, etc.
	subject: str = ""
	body: str = ""
	metadata: Dict[str, Any] = field(default_factory=dict)
	process_id: Optional[str] = None
	task_id: Optional[str] = None
	scheduled_at: datetime = field(default_factory=datetime.utcnow)
	sent_at: Optional[datetime] = None
	delivered_at: Optional[datetime] = None
	acknowledged_at: Optional[datetime] = None
	status: NotificationStatus = NotificationStatus.PENDING
	delivery_attempts: int = 0
	max_attempts: int = 3
	expires_at: Optional[datetime] = None
	tenant_id: str = ""


@dataclass
class NotificationBatch:
	"""Batch of notifications for efficient delivery."""
	batch_id: str = field(default_factory=lambda: f"batch_{uuid.uuid4().hex}")
	batch_type: str = "hourly"  # hourly, daily, custom
	recipient_id: str = ""
	channel: NotificationChannel = NotificationChannel.EMAIL
	messages: List[str] = field(default_factory=list)  # message_ids
	created_at: datetime = field(default_factory=datetime.utcnow)
	scheduled_for: datetime = field(default_factory=datetime.utcnow)
	processed_at: Optional[datetime] = None
	tenant_id: str = ""


@dataclass
class EscalationRule:
	"""Escalation rule for notifications."""
	rule_id: str = field(default_factory=lambda: f"escalation_{uuid.uuid4().hex}")
	trigger_notification_type: NotificationType = NotificationType.TASK_OVERDUE
	delay_minutes: int = 60
	escalation_path: List[str] = field(default_factory=list)  # user_ids in order
	escalation_channels: List[NotificationChannel] = field(default_factory=list)
	max_escalations: int = 3
	enabled: bool = True
	tenant_id: str = ""


# =============================================================================
# Notification Template Engine
# =============================================================================

class NotificationTemplateEngine:
	"""Process notification templates with variables."""
	
	def __init__(self):
		self.templates: Dict[str, NotificationTemplate] = {}
		self.default_templates = self._create_default_templates()
		
	def _create_default_templates(self) -> Dict[str, NotificationTemplate]:
		"""Create default notification templates."""
		templates = {}
		
		# Task assignment template
		templates["task_assignment"] = NotificationTemplate(
			template_name="Task Assignment",
			notification_type=NotificationType.TASK_ASSIGNMENT,
			channel=NotificationChannel.EMAIL,
			subject_template="New Task Assigned: {task_name}",
			body_template="""
Hello {recipient_name},

You have been assigned a new task:

Task: {task_name}
Process: {process_name}
Priority: {task_priority}
Due Date: {due_date}
Description: {task_description}

Please log in to the workflow system to view the task details and begin work.

Best regards,
Workflow Management System
			""".strip(),
			variables=["recipient_name", "task_name", "process_name", "task_priority", "due_date", "task_description"]
		)
		
		# Task overdue template
		templates["task_overdue"] = NotificationTemplate(
			template_name="Task Overdue",
			notification_type=NotificationType.TASK_OVERDUE,
			channel=NotificationChannel.EMAIL,
			subject_template="OVERDUE: Task {task_name} requires attention",
			body_template="""
Hello {recipient_name},

The following task is now overdue and requires immediate attention:

Task: {task_name}
Process: {process_name}
Original Due Date: {due_date}
Days Overdue: {days_overdue}

Please complete this task as soon as possible to avoid further delays.

Best regards,
Workflow Management System
			""".strip(),
			variables=["recipient_name", "task_name", "process_name", "due_date", "days_overdue"]
		)
		
		# Process completion template
		templates["process_completed"] = NotificationTemplate(
			template_name="Process Completion",
			notification_type=NotificationType.PROCESS_COMPLETED,
			channel=NotificationChannel.EMAIL,
			subject_template="Process Completed: {process_name}",
			body_template="""
Hello {recipient_name},

The following process has been completed successfully:

Process: {process_name}
Started: {start_date}
Completed: {completion_date}
Duration: {duration}
Final Status: {final_status}

You can view the complete process history in the workflow system.

Best regards,
Workflow Management System
			""".strip(),
			variables=["recipient_name", "process_name", "start_date", "completion_date", "duration", "final_status"]
		)
		
		# Approval request template
		templates["approval_request"] = NotificationTemplate(
			template_name="Approval Request",
			notification_type=NotificationType.APPROVAL_REQUEST,
			channel=NotificationChannel.EMAIL,
			subject_template="Approval Required: {approval_subject}",
			body_template="""
Hello {recipient_name},

Your approval is required for the following:

Subject: {approval_subject}
Process: {process_name}
Requested By: {requester_name}
Request Date: {request_date}
Description: {approval_description}

Please review and approve/deny this request in the workflow system.

Best regards,
Workflow Management System
			""".strip(),
			variables=["recipient_name", "approval_subject", "process_name", "requester_name", "request_date", "approval_description"]
		)
		
		return templates
		
	async def render_template(
		self,
		template_id: str,
		variables: Dict[str, Any],
		context: APGTenantContext
	) -> Tuple[str, str]:
		"""Render notification template with variables."""
		try:
			# Get template
			template = self.templates.get(template_id) or self.default_templates.get(template_id)
			if not template:
				raise ValueError(f"Template not found: {template_id}")
			
			# Render subject and body
			subject = self._render_string(template.subject_template, variables)
			body = self._render_string(template.body_template, variables)
			
			return subject, body
			
		except Exception as e:
			logger.error(f"Error rendering template {template_id}: {e}")
			return f"Notification: {template_id}", f"Error rendering template: {e}"
	
	def _render_string(self, template_string: str, variables: Dict[str, Any]) -> str:
		"""Render template string with variables."""
		try:
			# Simple variable substitution using format
			# In production, consider using Jinja2 for more advanced templating
			rendered = template_string
			
			for key, value in variables.items():
				placeholder = f"{{{key}}}"
				if placeholder in rendered:
					rendered = rendered.replace(placeholder, str(value))
			
			# Handle missing variables
			missing_vars = re.findall(r'\{([^}]+)\}', rendered)
			for var in missing_vars:
				rendered = rendered.replace(f"{{{var}}}", f"[{var}]")
			
			return rendered
			
		except Exception as e:
			logger.error(f"Error rendering string template: {e}")
			return template_string
	
	async def create_template(
		self,
		template_data: Dict[str, Any],
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Create custom notification template."""
		try:
			template = NotificationTemplate(
				template_name=template_data["template_name"],
				notification_type=NotificationType(template_data["notification_type"]),
				channel=NotificationChannel(template_data["channel"]),
				subject_template=template_data["subject_template"],
				body_template=template_data["body_template"],
				variables=template_data.get("variables", []),
				locale=template_data.get("locale", "en"),
				created_by=context.user_id,
				tenant_id=context.tenant_id
			)
			
			self.templates[template.template_id] = template
			
			logger.info(f"Notification template created: {template.template_id}")
			
			return WBPMServiceResponse(
				success=True,
				message="Notification template created successfully",
				data={"template_id": template.template_id}
			)
			
		except Exception as e:
			logger.error(f"Error creating notification template: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to create template: {e}",
				errors=[str(e)]
			)


# =============================================================================
# Notification Channel Handlers
# =============================================================================

class NotificationChannelHandler:
	"""Base class for notification channel handlers."""
	
	def __init__(self, channel: NotificationChannel):
		self.channel = channel
		
	async def send_notification(self, message: NotificationMessage) -> bool:
		"""Send notification through this channel."""
		raise NotImplementedError
	
	async def validate_address(self, address: str) -> bool:
		"""Validate recipient address for this channel."""
		raise NotImplementedError


class EmailChannelHandler(NotificationChannelHandler):
	"""Email notification handler."""
	
	def __init__(self):
		super().__init__(NotificationChannel.EMAIL)
		
	async def send_notification(self, message: NotificationMessage) -> bool:
		"""Send email notification."""
		try:
			# In production, integrate with email service (AWS SES, SendGrid, etc.)
			logger.info(f"Sending email to {message.recipient_address}: {message.subject}")
			
			# Simulate email sending
			await asyncio.sleep(0.1)
			
			# Update message status
			message.sent_at = datetime.utcnow()
			message.status = NotificationStatus.SENT
			
			return True
			
		except Exception as e:
			logger.error(f"Error sending email: {e}")
			message.status = NotificationStatus.FAILED
			return False
	
	async def validate_address(self, address: str) -> bool:
		"""Validate email address."""
		email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
		return bool(re.match(email_pattern, address))


class SMSChannelHandler(NotificationChannelHandler):
	"""SMS notification handler."""
	
	def __init__(self):
		super().__init__(NotificationChannel.SMS)
		
	async def send_notification(self, message: NotificationMessage) -> bool:
		"""Send SMS notification."""
		try:
			# In production, integrate with SMS service (Twilio, AWS SNS, etc.)
			logger.info(f"Sending SMS to {message.recipient_address}: {message.subject}")
			
			# Simulate SMS sending
			await asyncio.sleep(0.1)
			
			message.sent_at = datetime.utcnow()
			message.status = NotificationStatus.SENT
			
			return True
			
		except Exception as e:
			logger.error(f"Error sending SMS: {e}")
			message.status = NotificationStatus.FAILED
			return False
	
	async def validate_address(self, address: str) -> bool:
		"""Validate phone number."""
		phone_pattern = r'^\+?1?[0-9]{10,15}$'
		return bool(re.match(phone_pattern, address.replace('-', '').replace(' ', '')))


class SlackChannelHandler(NotificationChannelHandler):
	"""Slack notification handler."""
	
	def __init__(self):
		super().__init__(NotificationChannel.SLACK)
		
	async def send_notification(self, message: NotificationMessage) -> bool:
		"""Send Slack notification."""
		try:
			# In production, integrate with Slack API
			logger.info(f"Sending Slack message to {message.recipient_address}: {message.subject}")
			
			# Simulate Slack sending
			await asyncio.sleep(0.1)
			
			message.sent_at = datetime.utcnow()
			message.status = NotificationStatus.SENT
			
			return True
			
		except Exception as e:
			logger.error(f"Error sending Slack message: {e}")
			message.status = NotificationStatus.FAILED
			return False
	
	async def validate_address(self, address: str) -> bool:
		"""Validate Slack channel/user."""
		# Slack channels start with # and users with @
		return address.startswith('#') or address.startswith('@') or address.startswith('U')


class WebhookChannelHandler(NotificationChannelHandler):
	"""Webhook notification handler."""
	
	def __init__(self):
		super().__init__(NotificationChannel.WEBHOOK)
		
	async def send_notification(self, message: NotificationMessage) -> bool:
		"""Send webhook notification."""
		try:
			# In production, make HTTP POST to webhook URL
			logger.info(f"Sending webhook to {message.recipient_address}: {message.subject}")
			
			# Simulate webhook sending
			await asyncio.sleep(0.1)
			
			message.sent_at = datetime.utcnow()
			message.status = NotificationStatus.SENT
			
			return True
			
		except Exception as e:
			logger.error(f"Error sending webhook: {e}")
			message.status = NotificationStatus.FAILED
			return False
	
	async def validate_address(self, address: str) -> bool:
		"""Validate webhook URL."""
		url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
		return bool(re.match(url_pattern, address))


class InAppChannelHandler(NotificationChannelHandler):
	"""In-app notification handler."""
	
	def __init__(self):
		super().__init__(NotificationChannel.IN_APP)
		self.user_notifications: Dict[str, List[NotificationMessage]] = defaultdict(list)
		
	async def send_notification(self, message: NotificationMessage) -> bool:
		"""Send in-app notification."""
		try:
			# Store notification for user
			user_id = message.recipient_id
			self.user_notifications[user_id].append(message)
			
			# Keep only recent notifications (last 100)
			if len(self.user_notifications[user_id]) > 100:
				self.user_notifications[user_id] = self.user_notifications[user_id][-50:]
			
			message.sent_at = datetime.utcnow()
			message.status = NotificationStatus.DELIVERED
			
			logger.info(f"In-app notification sent to user {user_id}: {message.subject}")
			
			return True
			
		except Exception as e:
			logger.error(f"Error sending in-app notification: {e}")
			message.status = NotificationStatus.FAILED
			return False
	
	async def validate_address(self, address: str) -> bool:
		"""Validate user ID."""
		return bool(address and len(address) > 0)
	
	async def get_user_notifications(
		self,
		user_id: str,
		unread_only: bool = False
	) -> List[NotificationMessage]:
		"""Get notifications for user."""
		notifications = self.user_notifications.get(user_id, [])
		
		if unread_only:
			notifications = [
				n for n in notifications
				if n.acknowledged_at is None
			]
		
		return sorted(notifications, key=lambda n: n.scheduled_at, reverse=True)


# =============================================================================
# Notification Delivery Engine
# =============================================================================

class NotificationDeliveryEngine:
	"""Core notification delivery engine."""
	
	def __init__(self):
		self.pending_messages: deque = deque()
		self.sent_messages: Dict[str, NotificationMessage] = {}
		self.batch_queue: Dict[str, NotificationBatch] = {}
		self.preferences: Dict[str, NotificationPreference] = {}
		self.escalation_rules: Dict[str, EscalationRule] = {}
		
		# Channel handlers
		self.channel_handlers: Dict[NotificationChannel, NotificationChannelHandler] = {
			NotificationChannel.EMAIL: EmailChannelHandler(),
			NotificationChannel.SMS: SMSChannelHandler(),
			NotificationChannel.SLACK: SlackChannelHandler(),
			NotificationChannel.WEBHOOK: WebhookChannelHandler(),
			NotificationChannel.IN_APP: InAppChannelHandler()
		}
		
		# Delivery worker task
		self.delivery_task: Optional[asyncio.Task] = None
		
	async def start_delivery_worker(self) -> None:
		"""Start background delivery worker."""
		if self.delivery_task is None or self.delivery_task.done():
			self.delivery_task = asyncio.create_task(self._delivery_worker())
			logger.info("Notification delivery worker started")
	
	async def stop_delivery_worker(self) -> None:
		"""Stop background delivery worker."""
		if self.delivery_task and not self.delivery_task.done():
			self.delivery_task.cancel()
			try:
				await self.delivery_task
			except asyncio.CancelledError:
				pass
			logger.info("Notification delivery worker stopped")
	
	async def _delivery_worker(self) -> None:
		"""Background worker for processing notifications."""
		try:
			logger.info("Notification delivery worker started")
			
			while True:
				try:
					# Process pending messages
					await self._process_pending_messages()
					
					# Process batched notifications
					await self._process_batched_notifications()
					
					# Check for escalations
					await self._check_escalations()
					
					# Clean up expired messages
					await self._cleanup_expired_messages()
					
					# Sleep before next cycle
					await asyncio.sleep(10)  # Check every 10 seconds
					
				except asyncio.CancelledError:
					logger.info("Delivery worker cancelled")
					break
				except Exception as e:
					logger.error(f"Error in delivery worker: {e}")
					await asyncio.sleep(30)  # Wait longer on error
					
		except Exception as e:
			logger.error(f"Fatal error in delivery worker: {e}")
	
	async def queue_notification(
		self,
		message: NotificationMessage,
		context: APGTenantContext
	) -> WBPMServiceResponse:
		"""Queue notification for delivery."""
		try:
			message.tenant_id = context.tenant_id
			
			# Check user preferences
			preference = await self._get_user_preference(
				message.recipient_id, message.notification_type, context
			)
			
			if not preference or message.channel not in preference.enabled_channels:
				logger.info(f"Notification filtered by user preferences: {message.message_id}")
				return WBPMServiceResponse(
					success=True,
					message="Notification filtered by user preferences",
					data={"filtered": True}
				)
			
			# Apply delivery rules
			if preference.delivery_rule == DeliveryRule.BATCH_HOURLY:
				await self._add_to_batch(message, "hourly")
			elif preference.delivery_rule == DeliveryRule.BATCH_DAILY:
				await self._add_to_batch(message, "daily")
			elif preference.delivery_rule == DeliveryRule.BUSINESS_HOURS_ONLY:
				if not await self._is_business_hours(preference.timezone):
					message.scheduled_at = await self._next_business_hour(preference.timezone)
				self.pending_messages.append(message)
			else:  # IMMEDIATE
				self.pending_messages.append(message)
			
			logger.info(f"Notification queued: {message.message_id}")
			
			return WBPMServiceResponse(
				success=True,
				message="Notification queued successfully",
				data={"message_id": message.message_id}
			)
			
		except Exception as e:
			logger.error(f"Error queuing notification: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to queue notification: {e}",
				errors=[str(e)]
			)
	
	async def _process_pending_messages(self) -> None:
		"""Process pending notification messages."""
		try:
			messages_to_process = []
			current_time = datetime.utcnow()
			
			# Find messages ready for delivery
			while self.pending_messages:
				message = self.pending_messages.popleft()
				if message.scheduled_at <= current_time:
					messages_to_process.append(message)
				else:
					# Put back and break (assuming queue is sorted)
					self.pending_messages.appendleft(message)
					break
			
			# Process messages
			for message in messages_to_process:
				try:
					await self._deliver_message(message)
				except Exception as e:
					logger.error(f"Error delivering message {message.message_id}: {e}")
					message.delivery_attempts += 1
					
					if message.delivery_attempts < message.max_attempts:
						# Retry later with exponential backoff
						delay_minutes = 2 ** message.delivery_attempts
						message.scheduled_at = current_time + timedelta(minutes=delay_minutes)
						self.pending_messages.append(message)
					else:
						message.status = NotificationStatus.FAILED
						self.sent_messages[message.message_id] = message
						
		except Exception as e:
			logger.error(f"Error processing pending messages: {e}")
	
	async def _deliver_message(self, message: NotificationMessage) -> None:
		"""Deliver individual notification message."""
		try:
			handler = self.channel_handlers.get(message.channel)
			if not handler:
				raise ValueError(f"No handler for channel: {message.channel}")
			
			# Validate recipient address
			if not await handler.validate_address(message.recipient_address):
				raise ValueError(f"Invalid recipient address: {message.recipient_address}")
			
			# Send notification
			success = await handler.send_notification(message)
			
			if success:
				self.sent_messages[message.message_id] = message
				logger.info(f"Message delivered successfully: {message.message_id}")
			else:
				raise Exception("Handler reported delivery failure")
				
		except Exception as e:
			logger.error(f"Error in message delivery: {e}")
			raise
	
	async def _add_to_batch(self, message: NotificationMessage, batch_type: str) -> None:
		"""Add message to batch queue."""
		try:
			batch_key = f"{message.recipient_id}:{message.channel}:{batch_type}"
			
			if batch_key not in self.batch_queue:
				# Calculate next batch time
				if batch_type == "hourly":
					next_hour = datetime.utcnow().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
				else:  # daily
					next_day = datetime.utcnow().replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=1)
					next_hour = next_day
				
				self.batch_queue[batch_key] = NotificationBatch(
					batch_type=batch_type,
					recipient_id=message.recipient_id,
					channel=message.channel,
					scheduled_for=next_hour,
					tenant_id=message.tenant_id
				)
			
			batch = self.batch_queue[batch_key]
			batch.messages.append(message.message_id)
			
			# Store message for later batching
			self.sent_messages[message.message_id] = message
			
		except Exception as e:
			logger.error(f"Error adding to batch: {e}")
	
	async def _process_batched_notifications(self) -> None:
		"""Process batched notifications."""
		try:
			current_time = datetime.utcnow()
			batches_to_process = []
			
			for batch_key, batch in list(self.batch_queue.items()):
				if batch.scheduled_for <= current_time:
					batches_to_process.append((batch_key, batch))
			
			for batch_key, batch in batches_to_process:
				try:
					await self._send_batch_notification(batch)
					del self.batch_queue[batch_key]
				except Exception as e:
					logger.error(f"Error processing batch {batch_key}: {e}")
					
		except Exception as e:
			logger.error(f"Error processing batched notifications: {e}")
	
	async def _send_batch_notification(self, batch: NotificationBatch) -> None:
		"""Send batched notification."""
		try:
			# Collect all messages in batch
			messages = []
			for message_id in batch.messages:
				if message_id in self.sent_messages:
					messages.append(self.sent_messages[message_id])
			
			if not messages:
				return
			
			# Create combined notification
			combined_subject = f"Workflow Summary - {len(messages)} notifications"
			combined_body = self._create_batch_body(messages)
			
			batch_message = NotificationMessage(
				notification_type=NotificationType.CUSTOM,
				channel=batch.channel,
				priority=NotificationPriority.NORMAL,
				recipient_id=batch.recipient_id,
				recipient_address=messages[0].recipient_address,  # Use first message's address
				subject=combined_subject,
				body=combined_body,
				tenant_id=batch.tenant_id
			)
			
			# Send batch message
			await self._deliver_message(batch_message)
			
			# Mark individual messages as sent
			for message in messages:
				message.status = NotificationStatus.SENT
				message.sent_at = datetime.utcnow()
			
			batch.processed_at = datetime.utcnow()
			
		except Exception as e:
			logger.error(f"Error sending batch notification: {e}")
	
	def _create_batch_body(self, messages: List[NotificationMessage]) -> str:
		"""Create combined body for batch notification."""
		try:
			body_parts = ["Here's your workflow summary:\n"]
			
			for i, message in enumerate(messages, 1):
				body_parts.append(f"{i}. {message.subject}")
				if message.body:
					# Take first line of body as preview
					first_line = message.body.split('\n')[0]
					if len(first_line) > 100:
						first_line = first_line[:97] + "..."
					body_parts.append(f"   {first_line}")
				body_parts.append("")
			
			body_parts.append("Please log in to the workflow system for full details.")
			
			return "\n".join(body_parts)
			
		except Exception as e:
			logger.error(f"Error creating batch body: {e}")
			return "Workflow notification batch - see system for details"
	
	async def _get_user_preference(
		self,
		user_id: str,
		notification_type: NotificationType,
		context: APGTenantContext
	) -> Optional[NotificationPreference]:
		"""Get user notification preference."""
		try:
			# Look for specific preference
			pref_key = f"{user_id}:{notification_type}:{context.tenant_id}"
			
			if pref_key not in self.preferences:
				# Create default preference
				self.preferences[pref_key] = NotificationPreference(
					user_id=user_id,
					notification_type=notification_type,
					enabled_channels=[NotificationChannel.EMAIL, NotificationChannel.IN_APP],
					delivery_rule=DeliveryRule.IMMEDIATE,
					tenant_id=context.tenant_id
				)
			
			return self.preferences[pref_key]
			
		except Exception as e:
			logger.error(f"Error getting user preference: {e}")
			return None
	
	async def _is_business_hours(self, timezone: str) -> bool:
		"""Check if current time is within business hours."""
		# Simplified business hours check (9 AM - 5 PM)
		current_time = datetime.utcnow()
		hour = current_time.hour
		return 9 <= hour < 17
	
	async def _next_business_hour(self, timezone: str) -> datetime:
		"""Get next business hour."""
		current_time = datetime.utcnow()
		if current_time.hour >= 17:
			# Next day at 9 AM
			return (current_time + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
		else:
			# Today at 9 AM
			return current_time.replace(hour=9, minute=0, second=0, microsecond=0)
	
	async def _check_escalations(self) -> None:
		"""Check for escalation rules."""
		# Implementation would check for overdue tasks/processes and trigger escalations
		pass
	
	async def _cleanup_expired_messages(self) -> None:
		"""Clean up expired messages."""
		try:
			current_time = datetime.utcnow()
			expired_messages = []
			
			for message_id, message in list(self.sent_messages.items()):
				if message.expires_at and message.expires_at <= current_time:
					expired_messages.append(message_id)
			
			for message_id in expired_messages:
				message = self.sent_messages.pop(message_id)
				message.status = NotificationStatus.EXPIRED
				logger.debug(f"Message expired: {message_id}")
				
		except Exception as e:
			logger.error(f"Error cleaning up expired messages: {e}")


# =============================================================================
# Notification Service
# =============================================================================

class NotificationService:
	"""Main notification service."""
	
	def __init__(self):
		self.template_engine = NotificationTemplateEngine()
		self.delivery_engine = NotificationDeliveryEngine()
		
	async def start(self) -> None:
		"""Start notification service."""
		await self.delivery_engine.start_delivery_worker()
		logger.info("Notification service started")
	
	async def stop(self) -> None:
		"""Stop notification service."""
		await self.delivery_engine.stop_delivery_worker()
		logger.info("Notification service stopped")
	
	async def send_notification(
		self,
		notification_type: NotificationType,
		recipient_id: str,
		recipient_address: str,
		channel: NotificationChannel,
		template_variables: Dict[str, Any],
		context: APGTenantContext,
		priority: NotificationPriority = NotificationPriority.NORMAL,
		process_id: Optional[str] = None,
		task_id: Optional[str] = None
	) -> WBPMServiceResponse:
		"""Send notification using template."""
		try:
			# Render template
			template_id = notification_type.value
			subject, body = await self.template_engine.render_template(
				template_id, template_variables, context
			)
			
			# Create notification message
			message = NotificationMessage(
				notification_type=notification_type,
				channel=channel,
				priority=priority,
				recipient_id=recipient_id,
				recipient_address=recipient_address,
				subject=subject,
				body=body,
				process_id=process_id,
				task_id=task_id,
				tenant_id=context.tenant_id
			)
			
			# Set expiration for urgent messages
			if priority == NotificationPriority.URGENT:
				message.expires_at = datetime.utcnow() + timedelta(hours=24)
			
			# Queue for delivery
			return await self.delivery_engine.queue_notification(message, context)
			
		except Exception as e:
			logger.error(f"Error sending notification: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to send notification: {e}",
				errors=[str(e)]
			)
	
	async def get_user_notifications(
		self,
		user_id: str,
		context: APGTenantContext,
		unread_only: bool = False
	) -> WBPMServiceResponse:
		"""Get in-app notifications for user."""
		try:
			handler = self.delivery_engine.channel_handlers[NotificationChannel.IN_APP]
			notifications = await handler.get_user_notifications(user_id, unread_only)
			
			# Filter by tenant
			filtered_notifications = [
				n for n in notifications if n.tenant_id == context.tenant_id
			]
			
			return WBPMServiceResponse(
				success=True,
				message="Notifications retrieved successfully",
				data={
					"notifications": [n.__dict__ for n in filtered_notifications],
					"total_count": len(filtered_notifications)
				}
			)
			
		except Exception as e:
			logger.error(f"Error getting user notifications: {e}")
			return WBPMServiceResponse(
				success=False,
				message=f"Failed to get notifications: {e}",
				errors=[str(e)]
			)


# =============================================================================
# Service Factory
# =============================================================================

def create_notification_service() -> NotificationService:
	"""Create and configure notification service."""
	service = NotificationService()
	logger.info("Notification service created and configured")
	return service


# Export main classes
__all__ = [
	'NotificationService',
	'NotificationTemplateEngine',
	'NotificationDeliveryEngine',
	'NotificationMessage',
	'NotificationTemplate',
	'NotificationPreference',
	'EscalationRule',
	'NotificationChannel',
	'NotificationPriority',
	'NotificationStatus',
	'NotificationType',
	'DeliveryRule',
	'create_notification_service'
]