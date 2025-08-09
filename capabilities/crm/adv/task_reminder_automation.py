"""
APG Customer Relationship Management - Task and Reminder Automation Module

Advanced automated task creation and reminder system with intelligent scheduling,
notification management, and productivity optimization capabilities.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from decimal import Decimal
from uuid_extensions import uuid7str
import json

from pydantic import BaseModel, Field, validator

from .database import DatabaseManager


logger = logging.getLogger(__name__)


class TaskAutomationType(str, Enum):
	"""Types of task automation"""
	LEAD_FOLLOW_UP = "lead_follow_up"
	OPPORTUNITY_FOLLOW_UP = "opportunity_follow_up"
	MEETING_FOLLOW_UP = "meeting_follow_up"
	EMAIL_FOLLOW_UP = "email_follow_up"
	PIPELINE_STAGE_TASK = "pipeline_stage_task"
	LEAD_NURTURING = "lead_nurturing"
	CONTRACT_REMINDER = "contract_reminder"
	RENEWAL_REMINDER = "renewal_reminder"
	OVERDUE_PAYMENT = "overdue_payment"
	BIRTHDAY_REMINDER = "birthday_reminder"
	ANNIVERSARY_REMINDER = "anniversary_reminder"
	CUSTOM = "custom"


class ReminderType(str, Enum):
	"""Types of reminders"""
	EMAIL = "email"
	SMS = "sms"
	PUSH_NOTIFICATION = "push_notification"
	IN_APP = "in_app"
	DESKTOP = "desktop"
	WEBHOOK = "webhook"
	SLACK = "slack"
	TEAMS = "teams"


class ReminderStatus(str, Enum):
	"""Reminder delivery status"""
	PENDING = "pending"
	SENT = "sent"
	DELIVERED = "delivered"
	FAILED = "failed"
	CANCELLED = "cancelled"
	SNOOZED = "snoozed"


class AutomationTrigger(str, Enum):
	"""Triggers for automated tasks"""
	TIME_BASED = "time_based"
	EVENT_BASED = "event_based"
	RECORD_CREATED = "record_created"
	RECORD_UPDATED = "record_updated"
	FIELD_CHANGED = "field_changed"
	STATUS_CHANGED = "status_changed"
	DATE_REACHED = "date_reached"
	MANUAL = "manual"


class TaskAutomationRule(BaseModel):
	"""Task automation rule configuration"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	
	# Rule details
	name: str = Field(..., min_length=1, max_length=200)
	description: Optional[str] = Field(None, max_length=1000)
	automation_type: TaskAutomationType
	is_active: bool = Field(True, description="Whether rule is active")
	
	# Trigger configuration
	trigger_type: AutomationTrigger
	trigger_conditions: Dict[str, Any] = Field(default_factory=dict)
	trigger_schedule: Optional[str] = Field(None, description="Cron expression for scheduled triggers")
	
	# Task generation settings
	task_template: Dict[str, Any] = Field(default_factory=dict)
	task_priority: str = Field("normal", description="Default task priority")
	task_due_offset_hours: int = Field(24, description="Hours from trigger to due date")
	
	# Assignment rules
	assign_to_owner: bool = Field(True, description="Assign to record owner")
	assign_to_user: Optional[str] = Field(None, description="Specific user to assign")
	assign_to_team: Optional[str] = Field(None, description="Team to assign")
	assignment_logic: Dict[str, Any] = Field(default_factory=dict)
	
	# Conditions and filters
	apply_conditions: Dict[str, Any] = Field(default_factory=dict)
	exclude_conditions: Dict[str, Any] = Field(default_factory=dict)
	
	# Frequency and limits
	max_tasks_per_record: Optional[int] = Field(None, description="Maximum tasks per record")
	cooldown_hours: int = Field(0, description="Hours between task generation")
	
	# Execution tracking
	total_executions: int = Field(0, description="Total times rule executed")
	successful_executions: int = Field(0, description="Successful executions")
	last_executed_at: Optional[datetime] = Field(None, description="Last execution time")
	
	# Metadata
	metadata: Dict[str, Any] = Field(default_factory=dict)
	
	# Audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
	updated_by: str
	version: int = 1


class ReminderRule(BaseModel):
	"""Reminder rule configuration"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	
	# Rule details
	name: str = Field(..., min_length=1, max_length=200)
	description: Optional[str] = Field(None, max_length=1000)
	reminder_type: ReminderType
	is_active: bool = Field(True, description="Whether rule is active")
	
	# Trigger configuration  
	trigger_type: AutomationTrigger
	trigger_conditions: Dict[str, Any] = Field(default_factory=dict)
	
	# Reminder timing
	advance_minutes: List[int] = Field(default_factory=list, description="Minutes before event to remind")
	recurring_interval_minutes: Optional[int] = Field(None, description="Recurring reminder interval")
	max_reminders: int = Field(3, description="Maximum reminder attempts")
	
	# Notification content
	message_template: str = Field(..., description="Reminder message template")
	subject_template: Optional[str] = Field(None, description="Subject template")
	notification_data: Dict[str, Any] = Field(default_factory=dict)
	
	# Delivery settings
	delivery_channels: List[ReminderType] = Field(default_factory=list)
	fallback_channels: List[ReminderType] = Field(default_factory=list)
	
	# Recipient rules
	notify_owner: bool = Field(True, description="Notify record owner")
	notify_assignee: bool = Field(False, description="Notify assignee")
	additional_recipients: List[str] = Field(default_factory=list)
	
	# Conditions
	apply_conditions: Dict[str, Any] = Field(default_factory=dict)
	exclude_conditions: Dict[str, Any] = Field(default_factory=dict)
	
	# Tracking
	total_reminders_sent: int = Field(0, description="Total reminders sent")
	successful_deliveries: int = Field(0, description="Successful deliveries")
	last_executed_at: Optional[datetime] = Field(None, description="Last execution time")
	
	# Metadata
	metadata: Dict[str, Any] = Field(default_factory=dict)
	
	# Audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
	updated_by: str
	version: int = 1


class AutomatedTask(BaseModel):
	"""Automated task record"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	
	# Task details
	title: str = Field(..., description="Task title")
	description: Optional[str] = Field(None, description="Task description")
	priority: str = Field("normal", description="Task priority")
	due_date: Optional[datetime] = Field(None, description="Task due date")
	
	# Automation details
	automation_rule_id: str = Field(..., description="Source automation rule")
	automation_type: TaskAutomationType
	trigger_data: Dict[str, Any] = Field(default_factory=dict)
	
	# Assignment
	assigned_to: str = Field(..., description="Assigned user")
	assigned_team: Optional[str] = Field(None, description="Assigned team")
	
	# CRM relationships
	contact_id: Optional[str] = Field(None, description="Related contact")
	account_id: Optional[str] = Field(None, description="Related account")
	lead_id: Optional[str] = Field(None, description="Related lead")
	opportunity_id: Optional[str] = Field(None, description="Related opportunity")
	
	# Status and completion
	status: str = Field("planned", description="Task status")
	completed_at: Optional[datetime] = Field(None, description="Completion time")
	completed_by: Optional[str] = Field(None, description="User who completed")
	
	# Generated task details
	generated_task_id: Optional[str] = Field(None, description="Generated CRM task ID")
	
	# Metadata
	metadata: Dict[str, Any] = Field(default_factory=dict)
	
	# Audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
	updated_by: str


class ReminderExecution(BaseModel):
	"""Reminder execution record"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	
	# Reminder details
	reminder_rule_id: str = Field(..., description="Source reminder rule")
	reminder_type: ReminderType
	
	# Target details
	target_record_id: str = Field(..., description="Target record ID")
	target_record_type: str = Field(..., description="Target record type")
	recipient: str = Field(..., description="Reminder recipient")
	
	# Delivery details
	delivery_channel: ReminderType
	message_content: str = Field(..., description="Reminder message")
	subject: Optional[str] = Field(None, description="Reminder subject")
	
	# Status and timing
	status: ReminderStatus = ReminderStatus.PENDING
	scheduled_at: datetime = Field(..., description="Scheduled delivery time")
	sent_at: Optional[datetime] = Field(None, description="Actual send time")
	delivered_at: Optional[datetime] = Field(None, description="Delivery confirmation time")
	
	# Attempts and errors
	attempt_count: int = Field(0, description="Delivery attempts")
	max_attempts: int = Field(3, description="Maximum attempts")
	last_error: Optional[str] = Field(None, description="Last error message")
	error_details: Dict[str, Any] = Field(default_factory=dict)
	
	# Response tracking
	acknowledged: bool = Field(False, description="Recipient acknowledged")
	acknowledged_at: Optional[datetime] = Field(None, description="Acknowledgment time")
	snoozed_until: Optional[datetime] = Field(None, description="Snoozed until time")
	
	# Metadata
	metadata: Dict[str, Any] = Field(default_factory=dict)
	
	# Audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class AutomationAnalytics(BaseModel):
	"""Task and reminder automation analytics"""
	tenant_id: str
	analysis_period_start: datetime
	analysis_period_end: datetime
	
	# Task automation metrics
	total_automated_tasks: int = 0
	completed_automated_tasks: int = 0
	overdue_automated_tasks: int = 0
	task_completion_rate: float = 0.0
	average_task_completion_time: float = 0.0
	
	# Reminder metrics
	total_reminders_sent: int = 0
	successful_deliveries: int = 0
	failed_deliveries: int = 0
	reminder_delivery_rate: float = 0.0
	average_response_time: float = 0.0
	
	# Automation rule performance
	most_active_rules: List[Dict[str, Any]] = Field(default_factory=list)
	rule_success_rates: Dict[str, float] = Field(default_factory=dict)
	
	# User productivity impact
	tasks_per_user: Dict[str, int] = Field(default_factory=dict)
	completion_rates_by_user: Dict[str, float] = Field(default_factory=dict)
	productivity_improvement: float = 0.0
	
	# Channel effectiveness
	channel_performance: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
	preferred_channels: List[str] = Field(default_factory=list)
	
	# Timing analysis
	optimal_reminder_times: List[int] = Field(default_factory=list)
	peak_completion_hours: List[int] = Field(default_factory=list)
	
	# CRM impact
	tasks_driving_conversions: int = 0
	revenue_attributed_to_automation: Decimal = Decimal('0')
	
	# Trends
	automation_growth_rate: float = 0.0
	efficiency_trend: float = 0.0
	
	# Analysis metadata
	analyzed_at: datetime = Field(default_factory=datetime.utcnow)
	analysis_version: str = "1.0"


class TaskReminderAutomationEngine:
	"""
	Advanced task and reminder automation system
	
	Provides intelligent automated task creation, reminder scheduling,
	and productivity optimization through smart automation rules.
	"""
	
	def __init__(self, db_manager: DatabaseManager):
		"""
		Initialize task reminder automation engine
		
		Args:
			db_manager: Database manager instance
		"""
		self.db_manager = db_manager
		self._initialized = False
		self._active_rules = {}
		self._reminder_queue = asyncio.Queue()
		self._processing_tasks = []
	
	async def initialize(self):
		"""Initialize the task reminder automation engine"""
		if self._initialized:
			return
		
		logger.info("🔧 Initializing Task Reminder Automation Engine...")
		
		# Ensure database connection
		if not self.db_manager._initialized:
			await self.db_manager.initialize()
		
		# Start background processing
		self._processing_tasks = [
			asyncio.create_task(self._process_reminder_queue()),
			asyncio.create_task(self._process_scheduled_automations()),
			asyncio.create_task(self._cleanup_expired_reminders())
		]
		
		self._initialized = True
		logger.info("✅ Task Reminder Automation Engine initialized successfully")
	
	async def create_task_automation_rule(
		self,
		rule_data: Dict[str, Any],
		tenant_id: str,
		created_by: str
	) -> TaskAutomationRule:
		"""
		Create a new task automation rule
		
		Args:
			rule_data: Automation rule configuration
			tenant_id: Tenant identifier
			created_by: User creating the rule
			
		Returns:
			Created automation rule
		"""
		try:
			logger.info(f"⚙️ Creating task automation rule: {rule_data.get('name')}")
			
			# Add required fields
			rule_data.update({
				'tenant_id': tenant_id,
				'created_by': created_by,
				'updated_by': created_by
			})
			
			# Create rule object
			rule = TaskAutomationRule(**rule_data)
			
			# Store rule in database
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					INSERT INTO crm_task_automation_rules (
						id, tenant_id, name, description, automation_type, is_active,
						trigger_type, trigger_conditions, trigger_schedule,
						task_template, task_priority, task_due_offset_hours,
						assign_to_owner, assign_to_user, assign_to_team, assignment_logic,
						apply_conditions, exclude_conditions, max_tasks_per_record, cooldown_hours,
						total_executions, successful_executions, last_executed_at,
						metadata, created_at, updated_at, created_by, updated_by, version
					) VALUES (
						$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16,
						$17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29
					)
				""", 
				rule.id, rule.tenant_id, rule.name, rule.description, rule.automation_type.value,
				rule.is_active, rule.trigger_type.value, rule.trigger_conditions, rule.trigger_schedule,
				rule.task_template, rule.task_priority, rule.task_due_offset_hours,
				rule.assign_to_owner, rule.assign_to_user, rule.assign_to_team, rule.assignment_logic,
				rule.apply_conditions, rule.exclude_conditions, rule.max_tasks_per_record, rule.cooldown_hours,
				rule.total_executions, rule.successful_executions, rule.last_executed_at,
				rule.metadata, rule.created_at, rule.updated_at, rule.created_by, rule.updated_by, rule.version
				)
			
			# Cache active rule
			if rule.is_active:
				self._active_rules[rule.id] = rule
			
			logger.info(f"✅ Task automation rule created successfully: {rule.id}")
			return rule
			
		except Exception as e:
			logger.error(f"Failed to create task automation rule: {str(e)}", exc_info=True)
			raise
	
	async def create_reminder_rule(
		self,
		rule_data: Dict[str, Any],
		tenant_id: str,
		created_by: str
	) -> ReminderRule:
		"""
		Create a new reminder rule
		
		Args:
			rule_data: Reminder rule configuration
			tenant_id: Tenant identifier
			created_by: User creating the rule
			
		Returns:
			Created reminder rule
		"""
		try:
			logger.info(f"⏰ Creating reminder rule: {rule_data.get('name')}")
			
			# Add required fields
			rule_data.update({
				'tenant_id': tenant_id,
				'created_by': created_by,
				'updated_by': created_by
			})
			
			# Create rule object
			rule = ReminderRule(**rule_data)
			
			# Store rule in database
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					INSERT INTO crm_reminder_rules (
						id, tenant_id, name, description, reminder_type, is_active,
						trigger_type, trigger_conditions, advance_minutes, recurring_interval_minutes, max_reminders,
						message_template, subject_template, notification_data,
						delivery_channels, fallback_channels, notify_owner, notify_assignee, additional_recipients,
						apply_conditions, exclude_conditions,
						total_reminders_sent, successful_deliveries, last_executed_at,
						metadata, created_at, updated_at, created_by, updated_by, version
					) VALUES (
						$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16,
						$17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30
					)
				""", 
				rule.id, rule.tenant_id, rule.name, rule.description, rule.reminder_type.value, rule.is_active,
				rule.trigger_type.value, rule.trigger_conditions, rule.advance_minutes, rule.recurring_interval_minutes, rule.max_reminders,
				rule.message_template, rule.subject_template, rule.notification_data,
				[rt.value for rt in rule.delivery_channels], [rt.value for rt in rule.fallback_channels],
				rule.notify_owner, rule.notify_assignee, rule.additional_recipients,
				rule.apply_conditions, rule.exclude_conditions,
				rule.total_reminders_sent, rule.successful_deliveries, rule.last_executed_at,
				rule.metadata, rule.created_at, rule.updated_at, rule.created_by, rule.updated_by, rule.version
				)
			
			logger.info(f"✅ Reminder rule created successfully: {rule.id}")
			return rule
			
		except Exception as e:
			logger.error(f"Failed to create reminder rule: {str(e)}", exc_info=True)
			raise
	
	async def trigger_task_automation(
		self,
		automation_type: TaskAutomationType,
		trigger_data: Dict[str, Any],
		tenant_id: str,
		triggered_by: str = "system"
	) -> List[AutomatedTask]:
		"""
		Trigger task automation based on event
		
		Args:
			automation_type: Type of automation to trigger
			trigger_data: Data that triggered the automation
			tenant_id: Tenant identifier
			triggered_by: User/system triggering automation
			
		Returns:
			List of created automated tasks
		"""
		try:
			logger.info(f"🤖 Triggering task automation: {automation_type}")
			
			# Get matching automation rules
			matching_rules = await self._get_matching_automation_rules(
				automation_type, trigger_data, tenant_id
			)
			
			created_tasks = []
			
			for rule in matching_rules:
				# Check if conditions are met
				if not await self._evaluate_conditions(rule.apply_conditions, trigger_data):
					continue
				
				# Check exclusion conditions
				if await self._evaluate_conditions(rule.exclude_conditions, trigger_data):
					continue
				
				# Check cooldown period
				if not await self._check_cooldown(rule, trigger_data):
					continue
				
				# Create automated task
				task = await self._create_automated_task(rule, trigger_data, triggered_by)
				if task:
					created_tasks.append(task)
				
				# Update rule execution stats
				await self._update_rule_stats(rule.id, True)
			
			logger.info(f"✅ Created {len(created_tasks)} automated tasks")
			return created_tasks
			
		except Exception as e:
			logger.error(f"Failed to trigger task automation: {str(e)}", exc_info=True)
			raise
	
	async def schedule_reminder(
		self,
		reminder_data: Dict[str, Any],
		tenant_id: str,
		scheduled_by: str = "system"
	) -> ReminderExecution:
		"""
		Schedule a reminder for delivery
		
		Args:
			reminder_data: Reminder configuration
			tenant_id: Tenant identifier
			scheduled_by: User/system scheduling reminder
			
		Returns:
			Created reminder execution record
		"""
		try:
			logger.info(f"⏰ Scheduling reminder for: {reminder_data.get('recipient')}")
			
			# Add required fields
			reminder_data.update({
				'tenant_id': tenant_id,
				'created_by': scheduled_by,
				'updated_by': scheduled_by
			})
			
			# Create reminder execution
			reminder = ReminderExecution(**reminder_data)
			
			# Store in database
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					INSERT INTO crm_reminder_executions (
						id, tenant_id, reminder_rule_id, reminder_type,
						target_record_id, target_record_type, recipient,
						delivery_channel, message_content, subject,
						status, scheduled_at, sent_at, delivered_at,
						attempt_count, max_attempts, last_error, error_details,
						acknowledged, acknowledged_at, snoozed_until,
						metadata, created_at, updated_at
					) VALUES (
						$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
						$15, $16, $17, $18, $19, $20, $21, $22, $23, $24
					)
				""", 
				reminder.id, reminder.tenant_id, reminder.reminder_rule_id, reminder.reminder_type.value,
				reminder.target_record_id, reminder.target_record_type, reminder.recipient,
				reminder.delivery_channel.value, reminder.message_content, reminder.subject,
				reminder.status.value, reminder.scheduled_at, reminder.sent_at, reminder.delivered_at,
				reminder.attempt_count, reminder.max_attempts, reminder.last_error, reminder.error_details,
				reminder.acknowledged, reminder.acknowledged_at, reminder.snoozed_until,
				reminder.metadata, reminder.created_at, reminder.updated_at
				)
			
			# Add to processing queue
			await self._reminder_queue.put(reminder)
			
			logger.info(f"✅ Reminder scheduled successfully: {reminder.id}")
			return reminder
			
		except Exception as e:
			logger.error(f"Failed to schedule reminder: {str(e)}", exc_info=True)
			raise
	
	async def get_automation_analytics(
		self,
		tenant_id: str,
		start_date: datetime,
		end_date: datetime
	) -> AutomationAnalytics:
		"""
		Get comprehensive automation analytics
		
		Args:
			tenant_id: Tenant identifier
			start_date: Analysis period start
			end_date: Analysis period end
			
		Returns:
			Automation analytics data
		"""
		try:
			logger.info(f"📊 Generating automation analytics for tenant: {tenant_id}")
			
			analytics = AutomationAnalytics(
				tenant_id=tenant_id,
				analysis_period_start=start_date,
				analysis_period_end=end_date
			)
			
			async with self.db_manager.get_connection() as conn:
				# Task automation metrics
				task_stats = await conn.fetchrow("""
					SELECT 
						COUNT(*) as total_tasks,
						COUNT(*) FILTER (WHERE status = 'completed') as completed_tasks,
						COUNT(*) FILTER (WHERE status = 'overdue') as overdue_tasks,
						AVG(EXTRACT(EPOCH FROM (completed_at - created_at))/3600) as avg_completion_hours
					FROM crm_automated_tasks
					WHERE tenant_id = $1 AND created_at BETWEEN $2 AND $3
				""", tenant_id, start_date, end_date)
				
				if task_stats:
					analytics.total_automated_tasks = task_stats['total_tasks'] or 0
					analytics.completed_automated_tasks = task_stats['completed_tasks'] or 0
					analytics.overdue_automated_tasks = task_stats['overdue_tasks'] or 0
					analytics.average_task_completion_time = task_stats['avg_completion_hours'] or 0.0
					
					if analytics.total_automated_tasks > 0:
						analytics.task_completion_rate = (analytics.completed_automated_tasks / analytics.total_automated_tasks) * 100
				
				# Reminder metrics
				reminder_stats = await conn.fetchrow("""
					SELECT 
						COUNT(*) as total_reminders,
						COUNT(*) FILTER (WHERE status = 'delivered') as successful_deliveries,
						COUNT(*) FILTER (WHERE status = 'failed') as failed_deliveries,
						AVG(EXTRACT(EPOCH FROM (acknowledged_at - sent_at))/60) as avg_response_minutes
					FROM crm_reminder_executions
					WHERE tenant_id = $1 AND created_at BETWEEN $2 AND $3
				""", tenant_id, start_date, end_date)
				
				if reminder_stats:
					analytics.total_reminders_sent = reminder_stats['total_reminders'] or 0
					analytics.successful_deliveries = reminder_stats['successful_deliveries'] or 0
					analytics.failed_deliveries = reminder_stats['failed_deliveries'] or 0
					analytics.average_response_time = reminder_stats['avg_response_minutes'] or 0.0
					
					if analytics.total_reminders_sent > 0:
						analytics.reminder_delivery_rate = (analytics.successful_deliveries / analytics.total_reminders_sent) * 100
				
				# Most active automation rules
				rule_stats = await conn.fetch("""
					SELECT 
						tar.id, tar.name, tar.automation_type,
						COUNT(at.id) as tasks_created,
						COUNT(at.id) FILTER (WHERE at.status = 'completed') as tasks_completed
					FROM crm_task_automation_rules tar
					LEFT JOIN crm_automated_tasks at ON tar.id = at.automation_rule_id
						AND at.created_at BETWEEN $2 AND $3
					WHERE tar.tenant_id = $1
					GROUP BY tar.id, tar.name, tar.automation_type
					ORDER BY tasks_created DESC
					LIMIT 10
				""", tenant_id, start_date, end_date)
				
				analytics.most_active_rules = [
					{
						'rule_id': row['id'],
						'rule_name': row['name'],
						'automation_type': row['automation_type'],
						'tasks_created': row['tasks_created'],
						'tasks_completed': row['tasks_completed'],
						'success_rate': (row['tasks_completed'] / max(row['tasks_created'], 1)) * 100
					}
					for row in rule_stats
				]
				
				# User productivity metrics
				user_stats = await conn.fetch("""
					SELECT 
						assigned_to,
						COUNT(*) as total_tasks,
						COUNT(*) FILTER (WHERE status = 'completed') as completed_tasks
					FROM crm_automated_tasks
					WHERE tenant_id = $1 AND created_at BETWEEN $2 AND $3
					GROUP BY assigned_to
				""", tenant_id, start_date, end_date)
				
				analytics.tasks_per_user = {row['assigned_to']: row['total_tasks'] for row in user_stats}
				analytics.completion_rates_by_user = {
					row['assigned_to']: (row['completed_tasks'] / max(row['total_tasks'], 1)) * 100 
					for row in user_stats
				}
			
			logger.info(f"✅ Generated analytics for {analytics.total_automated_tasks} automated tasks")
			return analytics
			
		except Exception as e:
			logger.error(f"Failed to generate automation analytics: {str(e)}", exc_info=True)
			raise
	
	# Background processing methods
	
	async def _process_reminder_queue(self):
		"""Process queued reminders"""
		while self._initialized:
			try:
				# Get reminder from queue with timeout
				try:
					reminder = await asyncio.wait_for(self._reminder_queue.get(), timeout=1.0)
				except asyncio.TimeoutError:
					continue
				
				# Check if it's time to send
				if reminder.scheduled_at <= datetime.utcnow():
					await self._send_reminder(reminder)
				else:
					# Re-queue for later
					await self._reminder_queue.put(reminder)
					await asyncio.sleep(60)  # Wait before checking again
				
			except Exception as e:
				logger.error(f"Error processing reminder queue: {str(e)}")
				await asyncio.sleep(5)
	
	async def _process_scheduled_automations(self):
		"""Process scheduled automation rules"""
		while self._initialized:
			try:
				# Check for time-based automation rules every minute
				await asyncio.sleep(60)
				
				async with self.db_manager.get_connection() as conn:
					# Get rules that should be executed
					rule_rows = await conn.fetch("""
						SELECT * FROM crm_task_automation_rules
						WHERE is_active = true 
						AND trigger_type = 'time_based'
						AND trigger_schedule IS NOT NULL
						AND (last_executed_at IS NULL OR last_executed_at < NOW() - INTERVAL '1 hour')
					""")
					
					for rule_row in rule_rows:
						rule = TaskAutomationRule(**dict(rule_row))
						# Check if rule should execute based on cron schedule
						if await self._should_execute_scheduled_rule(rule):
							await self._execute_scheduled_rule(rule)
				
			except Exception as e:
				logger.error(f"Error processing scheduled automations: {str(e)}")
				await asyncio.sleep(30)
	
	async def _cleanup_expired_reminders(self):
		"""Clean up old reminder executions"""
		while self._initialized:
			try:
				# Clean up every hour
				await asyncio.sleep(3600)
				
				async with self.db_manager.get_connection() as conn:
					# Delete old completed/failed reminders (older than 30 days)
					await conn.execute("""
						DELETE FROM crm_reminder_executions
						WHERE status IN ('delivered', 'failed', 'cancelled')
						AND created_at < NOW() - INTERVAL '30 days'
					""")
				
			except Exception as e:
				logger.error(f"Error cleaning up reminders: {str(e)}")
				await asyncio.sleep(300)
	
	# Helper methods
	
	async def _get_matching_automation_rules(
		self,
		automation_type: TaskAutomationType,
		trigger_data: Dict[str, Any],
		tenant_id: str
	) -> List[TaskAutomationRule]:
		"""Get automation rules matching the trigger"""
		try:
			async with self.db_manager.get_connection() as conn:
				rule_rows = await conn.fetch("""
					SELECT * FROM crm_task_automation_rules
					WHERE tenant_id = $1 
					AND automation_type = $2 
					AND is_active = true
					ORDER BY created_at
				""", tenant_id, automation_type.value)
				
				return [TaskAutomationRule(**dict(row)) for row in rule_rows]
				
		except Exception as e:
			logger.error(f"Failed to get matching automation rules: {str(e)}")
			return []
	
	async def _evaluate_conditions(self, conditions: Dict[str, Any], data: Dict[str, Any]) -> bool:
		"""Evaluate if conditions are met"""
		if not conditions:
			return True
		
		# Simple condition evaluation (can be extended)
		for field, expected_value in conditions.items():
			if field not in data or data[field] != expected_value:
				return False
		
		return True
	
	async def _check_cooldown(self, rule: TaskAutomationRule, trigger_data: Dict[str, Any]) -> bool:
		"""Check if rule is in cooldown period"""
		if rule.cooldown_hours <= 0:
			return True
		
		if not rule.last_executed_at:
			return True
		
		cooldown_end = rule.last_executed_at + timedelta(hours=rule.cooldown_hours)
		return datetime.utcnow() >= cooldown_end
	
	async def _create_automated_task(
		self,
		rule: TaskAutomationRule,
		trigger_data: Dict[str, Any],
		triggered_by: str
	) -> Optional[AutomatedTask]:
		"""Create automated task from rule"""
		try:
			# Determine assignee
			assigned_to = await self._determine_assignee(rule, trigger_data)
			if not assigned_to:
				return None
			
			# Build task data
			task_data = {
				'title': self._merge_template(rule.task_template.get('title', 'Automated Task'), trigger_data),
				'description': self._merge_template(rule.task_template.get('description', ''), trigger_data),
				'priority': rule.task_priority,
				'due_date': datetime.utcnow() + timedelta(hours=rule.task_due_offset_hours),
				'automation_rule_id': rule.id,
				'automation_type': rule.automation_type,
				'trigger_data': trigger_data,
				'assigned_to': assigned_to,
				'assigned_team': rule.assign_to_team,
				'tenant_id': rule.tenant_id,
				'created_by': triggered_by,
				'updated_by': triggered_by
			}
			
			# Add CRM relationships from trigger data
			for field in ['contact_id', 'account_id', 'lead_id', 'opportunity_id']:
				if field in trigger_data:
					task_data[field] = trigger_data[field]
			
			# Create task
			task = AutomatedTask(**task_data)
			
			# Store in database
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					INSERT INTO crm_automated_tasks (
						id, tenant_id, title, description, priority, due_date,
						automation_rule_id, automation_type, trigger_data,
						assigned_to, assigned_team, contact_id, account_id, lead_id, opportunity_id,
						status, completed_at, completed_by, generated_task_id,
						metadata, created_at, updated_at, created_by, updated_by
					) VALUES (
						$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
						$16, $17, $18, $19, $20, $21, $22, $23, $24
					)
				""", 
				task.id, task.tenant_id, task.title, task.description, task.priority, task.due_date,
				task.automation_rule_id, task.automation_type.value, task.trigger_data,
				task.assigned_to, task.assigned_team, task.contact_id, task.account_id, task.lead_id, task.opportunity_id,
				task.status, task.completed_at, task.completed_by, task.generated_task_id,
				task.metadata, task.created_at, task.updated_at, task.created_by, task.updated_by
				)
			
			return task
			
		except Exception as e:
			logger.error(f"Failed to create automated task: {str(e)}")
			return None
	
	async def _determine_assignee(self, rule: TaskAutomationRule, trigger_data: Dict[str, Any]) -> Optional[str]:
		"""Determine who to assign the task to"""
		if rule.assign_to_user:
			return rule.assign_to_user
		
		if rule.assign_to_owner and 'owner_id' in trigger_data:
			return trigger_data['owner_id']
		
		if 'created_by' in trigger_data:
			return trigger_data['created_by']
		
		return None
	
	def _merge_template(self, template: str, data: Dict[str, Any]) -> str:
		"""Merge template with data"""
		if not template:
			return ""
		
		# Simple template merging
		for field, value in data.items():
			placeholder = f"{{{{{field}}}}}"
			template = template.replace(placeholder, str(value))
		
		return template
	
	async def _send_reminder(self, reminder: ReminderExecution):
		"""Send a reminder"""
		try:
			logger.info(f"📤 Sending reminder: {reminder.id}")
			
			# Simulate sending reminder (implement actual delivery logic)
			# This would integrate with email, SMS, push notification services
			
			# Update reminder status
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					UPDATE crm_reminder_executions SET
						status = $2,
						sent_at = NOW(),
						attempt_count = attempt_count + 1,
						updated_at = NOW()
					WHERE id = $1
				""", reminder.id, ReminderStatus.SENT.value)
			
			# Simulate delivery confirmation after delay
			await asyncio.sleep(1)
			
			await conn.execute("""
				UPDATE crm_reminder_executions SET
					status = $2,
					delivered_at = NOW(),
					updated_at = NOW()
				WHERE id = $1
			""", reminder.id, ReminderStatus.DELIVERED.value)
			
			logger.info(f"✅ Reminder delivered successfully: {reminder.id}")
			
		except Exception as e:
			logger.error(f"Failed to send reminder {reminder.id}: {str(e)}")
			
			# Update failure status
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					UPDATE crm_reminder_executions SET
						status = $2,
						attempt_count = attempt_count + 1,
						last_error = $3,
						updated_at = NOW()
					WHERE id = $1
				""", reminder.id, ReminderStatus.FAILED.value, str(e))
	
	async def _update_rule_stats(self, rule_id: str, success: bool):
		"""Update automation rule execution statistics"""
		try:
			async with self.db_manager.get_connection() as conn:
				if success:
					await conn.execute("""
						UPDATE crm_task_automation_rules SET
							total_executions = total_executions + 1,
							successful_executions = successful_executions + 1,
							last_executed_at = NOW(),
							updated_at = NOW()
						WHERE id = $1
					""", rule_id)
				else:
					await conn.execute("""
						UPDATE crm_task_automation_rules SET
							total_executions = total_executions + 1,
							last_executed_at = NOW(),
							updated_at = NOW()
						WHERE id = $1
					""", rule_id)
		except Exception as e:
			logger.error(f"Failed to update rule stats: {str(e)}")
	
	async def _should_execute_scheduled_rule(self, rule: TaskAutomationRule) -> bool:
		"""Check if scheduled rule should execute now"""
		# Simple implementation - in production, use proper cron parsing
		return True
	
	async def _execute_scheduled_rule(self, rule: TaskAutomationRule):
		"""Execute a scheduled automation rule"""
		try:
			# Build trigger data for scheduled execution
			trigger_data = {
				'scheduled_execution': True,
				'execution_time': datetime.utcnow().isoformat()
			}
			
			await self.trigger_task_automation(
				rule.automation_type,
				trigger_data,
				rule.tenant_id,
				"scheduler"
			)
			
		except Exception as e:
			logger.error(f"Failed to execute scheduled rule {rule.id}: {str(e)}")
	
	async def shutdown(self):
		"""Shutdown the automation engine"""
		self._initialized = False
		
		# Cancel background tasks
		for task in self._processing_tasks:
			task.cancel()
		
		# Wait for tasks to complete
		await asyncio.gather(*self._processing_tasks, return_exceptions=True)