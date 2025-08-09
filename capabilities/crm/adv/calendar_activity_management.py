"""
APG Customer Relationship Management - Calendar and Activity Management Module

Advanced calendar integration and activity management system with comprehensive
scheduling, meeting coordination, and activity tracking capabilities.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from decimal import Decimal
from uuid_extensions import uuid7str
import json
import pytz
from dateutil.rrule import rrule, DAILY, WEEKLY, MONTHLY, YEARLY

from pydantic import BaseModel, Field, validator

from .database import DatabaseManager


logger = logging.getLogger(__name__)


class CalendarProvider(str, Enum):
	"""Calendar service providers"""
	INTERNAL = "internal"
	GOOGLE = "google"
	OUTLOOK = "outlook"
	EXCHANGE = "exchange"
	CALDAV = "caldav"
	ICAL = "ical"
	CUSTOM = "custom"


class EventType(str, Enum):
	"""Types of calendar events"""
	MEETING = "meeting"
	CALL = "call"
	DEMO = "demo"
	PRESENTATION = "presentation"
	TRAINING = "training"
	FOLLOW_UP = "follow_up"
	DEADLINE = "deadline"
	REMINDER = "reminder"
	TASK = "task"
	APPOINTMENT = "appointment"
	CONFERENCE = "conference"
	WEBINAR = "webinar"
	CUSTOM = "custom"


class EventStatus(str, Enum):
	"""Calendar event status"""
	SCHEDULED = "scheduled"
	CONFIRMED = "confirmed"
	TENTATIVE = "tentative"
	CANCELLED = "cancelled"
	COMPLETED = "completed"
	RESCHEDULED = "rescheduled"
	NO_SHOW = "no_show"
	IN_PROGRESS = "in_progress"


class EventPriority(str, Enum):
	"""Event priority levels"""
	LOW = "low"
	NORMAL = "normal"
	HIGH = "high"
	URGENT = "urgent"


class AttendeeStatus(str, Enum):
	"""Attendee response status"""
	INVITED = "invited"
	ACCEPTED = "accepted"
	DECLINED = "declined"
	TENTATIVE = "tentative"
	NO_RESPONSE = "no_response"


class RecurrenceFrequency(str, Enum):
	"""Recurring event frequency"""
	DAILY = "daily"
	WEEKLY = "weekly"
	MONTHLY = "monthly"
	YEARLY = "yearly"
	CUSTOM = "custom"


class ActivityType(str, Enum):
	"""Types of CRM activities"""
	EMAIL = "email"
	CALL = "call"
	MEETING = "meeting"
	TASK = "task"
	NOTE = "note"
	DOCUMENT = "document"
	PROPOSAL = "proposal"
	CONTRACT = "contract"
	DEMO = "demo"
	FOLLOW_UP = "follow_up"
	SOCIAL_MEDIA = "social_media"
	MARKETING = "marketing"
	SUPPORT = "support"
	CUSTOM = "custom"


class ActivityStatus(str, Enum):
	"""Activity completion status"""
	PLANNED = "planned"
	IN_PROGRESS = "in_progress"
	COMPLETED = "completed"
	CANCELLED = "cancelled"
	OVERDUE = "overdue"
	DEFERRED = "deferred"


class CalendarEventAttendee(BaseModel):
	"""Calendar event attendee"""
	id: str = Field(default_factory=uuid7str)
	event_id: str
	
	# Attendee details
	email: str = Field(..., description="Attendee email address")
	name: Optional[str] = Field(None, description="Attendee name")
	role: str = Field("attendee", description="Attendee role")
	
	# Response status
	status: AttendeeStatus = AttendeeStatus.INVITED
	response_time: Optional[datetime] = Field(None, description="Time of response")
	
	# CRM relationships
	contact_id: Optional[str] = Field(None, description="Associated contact")
	account_id: Optional[str] = Field(None, description="Associated account")
	lead_id: Optional[str] = Field(None, description="Associated lead")
	
	# Settings
	is_organizer: bool = Field(False, description="Whether attendee is organizer")
	is_required: bool = Field(True, description="Whether attendance is required")
	send_notifications: bool = Field(True, description="Send notifications to attendee")
	
	# Metadata
	metadata: Dict[str, Any] = Field(default_factory=dict)


class CalendarEventRecurrence(BaseModel):
	"""Calendar event recurrence settings"""
	id: str = Field(default_factory=uuid7str)
	event_id: str
	
	# Recurrence pattern
	frequency: RecurrenceFrequency
	interval: int = Field(1, description="Interval between occurrences")
	
	# Recurrence rules
	days_of_week: List[int] = Field(default_factory=list, description="Days of week (0=Monday)")
	days_of_month: List[int] = Field(default_factory=list, description="Days of month")
	months_of_year: List[int] = Field(default_factory=list, description="Months of year")
	
	# Recurrence limits
	end_date: Optional[date] = Field(None, description="End date for recurrence")
	occurrence_count: Optional[int] = Field(None, description="Maximum occurrences")
	
	# Exception dates
	exception_dates: List[date] = Field(default_factory=list)
	
	# Custom rule
	custom_rule: Optional[str] = Field(None, description="Custom RRULE string")
	
	# Metadata
	metadata: Dict[str, Any] = Field(default_factory=dict)


class CalendarEvent(BaseModel):
	"""Calendar event record"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	
	# Event identification
	external_id: Optional[str] = Field(None, description="External calendar system ID")
	calendar_provider: CalendarProvider = CalendarProvider.INTERNAL
	provider_metadata: Dict[str, Any] = Field(default_factory=dict)
	
	# Event details
	title: str = Field(..., min_length=1, max_length=500)
	description: Optional[str] = Field(None, max_length=5000)
	location: Optional[str] = Field(None, max_length=500)
	virtual_meeting_url: Optional[str] = Field(None, description="Video conference URL")
	
	# Event type and status
	event_type: EventType = EventType.MEETING
	status: EventStatus = EventStatus.SCHEDULED
	priority: EventPriority = EventPriority.NORMAL
	
	# Timing
	start_time: datetime = Field(..., description="Event start time")
	end_time: datetime = Field(..., description="Event end time")
	timezone: str = Field("UTC", description="Event timezone")
	all_day: bool = Field(False, description="All-day event")
	
	# CRM relationships
	contact_id: Optional[str] = Field(None, description="Primary contact")
	account_id: Optional[str] = Field(None, description="Associated account")
	lead_id: Optional[str] = Field(None, description="Associated lead")
	opportunity_id: Optional[str] = Field(None, description="Associated opportunity")
	campaign_id: Optional[str] = Field(None, description="Associated campaign")
	
	# Attendees and organization
	organizer_email: str = Field(..., description="Event organizer email")
	organizer_name: Optional[str] = Field(None, description="Event organizer name")
	attendees: List[CalendarEventAttendee] = Field(default_factory=list)
	
	# Recurrence
	is_recurring: bool = Field(False, description="Is recurring event")
	recurrence: Optional[CalendarEventRecurrence] = Field(None)
	parent_event_id: Optional[str] = Field(None, description="Parent for recurring instances")
	
	# Notifications and reminders
	reminder_minutes: List[int] = Field(default_factory=list, description="Reminder times in minutes")
	notification_sent: bool = Field(False, description="Whether notifications were sent")
	
	# Activity tracking
	actual_start_time: Optional[datetime] = Field(None, description="Actual start time")
	actual_end_time: Optional[datetime] = Field(None, description="Actual end time")
	actual_duration_minutes: Optional[int] = Field(None, description="Actual duration")
	
	# Meeting notes and outcomes
	meeting_notes: Optional[str] = Field(None, description="Meeting notes")
	meeting_outcomes: List[str] = Field(default_factory=list)
	action_items: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Integration
	sync_with_external: bool = Field(True, description="Sync with external calendar")
	last_synced_at: Optional[datetime] = Field(None, description="Last sync time")
	
	# Metadata
	metadata: Dict[str, Any] = Field(default_factory=dict)
	
	# Audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
	updated_by: str
	version: int = 1
	
	@validator('end_time')
	def end_after_start(cls, v, values):
		if 'start_time' in values and v <= values['start_time']:
			raise ValueError('End time must be after start time')
		return v


class CRMActivity(BaseModel):
	"""CRM activity record"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	
	# Activity details
	title: str = Field(..., min_length=1, max_length=500)
	description: Optional[str] = Field(None, max_length=5000)
	activity_type: ActivityType
	status: ActivityStatus = ActivityStatus.PLANNED
	priority: EventPriority = EventPriority.NORMAL
	
	# Timing
	due_date: Optional[datetime] = Field(None, description="Activity due date")
	completed_at: Optional[datetime] = Field(None, description="Completion time")
	estimated_duration_minutes: Optional[int] = Field(None, description="Estimated duration")
	actual_duration_minutes: Optional[int] = Field(None, description="Actual duration")
	
	# CRM relationships
	contact_id: Optional[str] = Field(None, description="Associated contact")
	account_id: Optional[str] = Field(None, description="Associated account")
	lead_id: Optional[str] = Field(None, description="Associated lead")
	opportunity_id: Optional[str] = Field(None, description="Associated opportunity")
	campaign_id: Optional[str] = Field(None, description="Associated campaign")
	
	# Assignment
	assigned_to: str = Field(..., description="Assigned user")
	assigned_by: Optional[str] = Field(None, description="User who assigned")
	team_id: Optional[str] = Field(None, description="Assigned team")
	
	# Related records
	parent_activity_id: Optional[str] = Field(None, description="Parent activity")
	related_event_id: Optional[str] = Field(None, description="Related calendar event")
	related_email_id: Optional[str] = Field(None, description="Related email")
	
	# Results and outcomes
	outcome: Optional[str] = Field(None, description="Activity outcome")
	notes: Optional[str] = Field(None, description="Activity notes")
	attachments: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Automation
	is_automated: bool = Field(False, description="Created by automation")
	workflow_id: Optional[str] = Field(None, description="Source workflow")
	
	# Tracking
	reminder_sent: bool = Field(False, description="Reminder notification sent")
	overdue_notifications_sent: int = Field(0, description="Overdue notifications count")
	
	# Metadata
	metadata: Dict[str, Any] = Field(default_factory=dict)
	
	# Audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
	updated_by: str
	version: int = 1


class ActivityTemplate(BaseModel):
	"""Template for creating activities"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	
	# Template details
	name: str = Field(..., min_length=1, max_length=200)
	description: Optional[str] = Field(None, max_length=1000)
	category: str = Field("general", description="Template category")
	
	# Default activity settings
	activity_type: ActivityType
	priority: EventPriority = EventPriority.NORMAL
	estimated_duration_minutes: Optional[int] = Field(None)
	
	# Template content
	title_template: str = Field(..., description="Activity title template")
	description_template: Optional[str] = Field(None, description="Activity description template")
	
	# Default assignments
	default_assigned_to: Optional[str] = Field(None, description="Default assignee")
	default_team_id: Optional[str] = Field(None, description="Default team")
	
	# Automation settings
	auto_due_date_days: Optional[int] = Field(None, description="Auto-set due date days from now")
	auto_reminders: List[int] = Field(default_factory=list, description="Auto reminder minutes")
	
	# Usage tracking
	usage_count: int = Field(0, description="Times template used")
	is_active: bool = Field(True, description="Whether template is active")
	
	# Metadata
	metadata: Dict[str, Any] = Field(default_factory=dict)
	
	# Audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
	updated_by: str
	version: int = 1


class CalendarAnalytics(BaseModel):
	"""Calendar and activity analytics"""
	tenant_id: str
	analysis_period_start: datetime
	analysis_period_end: datetime
	
	# Event metrics
	total_events: int = 0
	completed_events: int = 0
	cancelled_events: int = 0
	no_show_events: int = 0
	
	# Meeting metrics
	total_meetings: int = 0
	average_meeting_duration: float = 0.0
	total_meeting_time: int = 0
	meeting_attendance_rate: float = 0.0
	
	# Activity metrics
	total_activities: int = 0
	completed_activities: int = 0
	overdue_activities: int = 0
	average_completion_time: float = 0.0
	
	# Productivity metrics
	events_per_day: float = 0.0
	activities_per_day: float = 0.0
	completion_rate: float = 0.0
	on_time_rate: float = 0.0
	
	# Time analysis
	busiest_hours: Dict[int, int] = Field(default_factory=dict)
	busiest_days: Dict[str, int] = Field(default_factory=dict)
	peak_productivity_hours: List[int] = Field(default_factory=list)
	
	# User performance
	top_performers: List[Dict[str, Any]] = Field(default_factory=list)
	user_metrics: Dict[str, Any] = Field(default_factory=dict)
	
	# CRM impact
	events_with_outcomes: int = 0
	activities_driving_revenue: int = 0
	average_deal_size_impact: Decimal = Decimal('0')
	
	# Trends
	productivity_trend: float = 0.0
	completion_trend: float = 0.0
	engagement_trend: float = 0.0
	
	# Analysis metadata
	analyzed_at: datetime = Field(default_factory=datetime.utcnow)
	analysis_version: str = "1.0"


class CalendarActivityManager:
	"""
	Advanced calendar and activity management system
	
	Provides comprehensive calendar integration, event scheduling,
	activity tracking, and productivity analytics capabilities.
	"""
	
	def __init__(self, db_manager: DatabaseManager):
		"""
		Initialize calendar activity manager
		
		Args:
			db_manager: Database manager instance
		"""
		self.db_manager = db_manager
		self._initialized = False
		self._calendar_integrations = {}
		self._timezone_cache = {}
	
	async def initialize(self):
		"""Initialize the calendar activity manager"""
		if self._initialized:
			return
		
		logger.info("🔧 Initializing Calendar Activity Manager...")
		
		# Ensure database connection
		if not self.db_manager._initialized:
			await self.db_manager.initialize()
		
		# Initialize timezone handling
		self._default_timezone = pytz.UTC
		
		self._initialized = True
		logger.info("✅ Calendar Activity Manager initialized successfully")
	
	async def create_calendar_event(
		self,
		event_data: Dict[str, Any],
		tenant_id: str,
		created_by: str
	) -> CalendarEvent:
		"""
		Create a new calendar event
		
		Args:
			event_data: Event configuration data
			tenant_id: Tenant identifier
			created_by: User creating the event
			
		Returns:
			Created calendar event
		"""
		try:
			logger.info(f"📅 Creating calendar event: {event_data.get('title')}")
			
			# Add required fields
			event_data.update({
				'tenant_id': tenant_id,
				'created_by': created_by,
				'updated_by': created_by
			})
			
			# Set default organizer if not provided
			if not event_data.get('organizer_email'):
				event_data['organizer_email'] = f"{created_by}@example.com"  # Default logic
			
			# Process attendees
			if 'attendees' in event_data:
				processed_attendees = []
				for attendee_data in event_data['attendees']:
					attendee_data['event_id'] = event_data.get('id', uuid7str())
					processed_attendees.append(CalendarEventAttendee(**attendee_data))
				event_data['attendees'] = processed_attendees
			
			# Process recurrence if provided
			if event_data.get('recurrence'):
				recurrence_data = event_data['recurrence']
				recurrence_data['event_id'] = event_data.get('id', uuid7str())
				event_data['recurrence'] = CalendarEventRecurrence(**recurrence_data)
			
			# Create event object
			event = CalendarEvent(**event_data)
			
			# Store event in database
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					INSERT INTO crm_calendar_events (
						id, tenant_id, external_id, calendar_provider, provider_metadata,
						title, description, location, virtual_meeting_url,
						event_type, status, priority, start_time, end_time, timezone, all_day,
						contact_id, account_id, lead_id, opportunity_id, campaign_id,
						organizer_email, organizer_name, is_recurring, parent_event_id,
						reminder_minutes, notification_sent, actual_start_time, actual_end_time,
						actual_duration_minutes, meeting_notes, meeting_outcomes, action_items,
						sync_with_external, last_synced_at, metadata,
						created_at, updated_at, created_by, updated_by, version
					) VALUES (
						$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16,
						$17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30,
						$31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41
					)
				""", 
				event.id, event.tenant_id, event.external_id, event.calendar_provider.value, 
				event.provider_metadata, event.title, event.description, event.location, 
				event.virtual_meeting_url, event.event_type.value, event.status.value, 
				event.priority.value, event.start_time, event.end_time, event.timezone, 
				event.all_day, event.contact_id, event.account_id, event.lead_id, 
				event.opportunity_id, event.campaign_id, event.organizer_email, 
				event.organizer_name, event.is_recurring, event.parent_event_id,
				event.reminder_minutes, event.notification_sent, event.actual_start_time,
				event.actual_end_time, event.actual_duration_minutes, event.meeting_notes,
				event.meeting_outcomes, event.action_items, event.sync_with_external,
				event.last_synced_at, event.metadata, event.created_at, event.updated_at,
				event.created_by, event.updated_by, event.version
				)
				
				# Store attendees
				for attendee in event.attendees:
					await self._store_event_attendee(attendee)
				
				# Store recurrence if present
				if event.recurrence:
					await self._store_event_recurrence(event.recurrence)
			
			logger.info(f"✅ Calendar event created successfully: {event.id}")
			return event
			
		except Exception as e:
			logger.error(f"Failed to create calendar event: {str(e)}", exc_info=True)
			raise
	
	async def create_activity(
		self,
		activity_data: Dict[str, Any],
		tenant_id: str,
		created_by: str
	) -> CRMActivity:
		"""
		Create a new CRM activity
		
		Args:
			activity_data: Activity configuration data
			tenant_id: Tenant identifier
			created_by: User creating the activity
			
		Returns:
			Created CRM activity
		"""
		try:
			logger.info(f"📋 Creating CRM activity: {activity_data.get('title')}")
			
			# Add required fields
			activity_data.update({
				'tenant_id': tenant_id,
				'created_by': created_by,
				'updated_by': created_by
			})
			
			# Set default assignee if not provided
			if not activity_data.get('assigned_to'):
				activity_data['assigned_to'] = created_by
			
			# Create activity object
			activity = CRMActivity(**activity_data)
			
			# Store activity in database
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					INSERT INTO crm_activities (
						id, tenant_id, title, description, activity_type, status, priority,
						due_date, completed_at, estimated_duration_minutes, actual_duration_minutes,
						contact_id, account_id, lead_id, opportunity_id, campaign_id,
						assigned_to, assigned_by, team_id, parent_activity_id,
						related_event_id, related_email_id, outcome, notes, attachments,
						is_automated, workflow_id, reminder_sent, overdue_notifications_sent,
						metadata, created_at, updated_at, created_by, updated_by, version
					) VALUES (
						$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16,
						$17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30,
						$31, $32, $33, $34, $35
					)
				""", 
				activity.id, activity.tenant_id, activity.title, activity.description,
				activity.activity_type.value, activity.status.value, activity.priority.value,
				activity.due_date, activity.completed_at, activity.estimated_duration_minutes,
				activity.actual_duration_minutes, activity.contact_id, activity.account_id,
				activity.lead_id, activity.opportunity_id, activity.campaign_id,
				activity.assigned_to, activity.assigned_by, activity.team_id,
				activity.parent_activity_id, activity.related_event_id, activity.related_email_id,
				activity.outcome, activity.notes, activity.attachments, activity.is_automated,
				activity.workflow_id, activity.reminder_sent, activity.overdue_notifications_sent,
				activity.metadata, activity.created_at, activity.updated_at,
				activity.created_by, activity.updated_by, activity.version
				)
			
			logger.info(f"✅ CRM activity created successfully: {activity.id}")
			return activity
			
		except Exception as e:
			logger.error(f"Failed to create CRM activity: {str(e)}", exc_info=True)
			raise
	
	async def create_activity_from_template(
		self,
		template_id: str,
		activity_data: Dict[str, Any],
		tenant_id: str,
		created_by: str
	) -> CRMActivity:
		"""
		Create activity from template
		
		Args:
			template_id: Activity template identifier
			activity_data: Override data for activity
			tenant_id: Tenant identifier
			created_by: User creating the activity
			
		Returns:
			Created CRM activity
		"""
		try:
			logger.info(f"📋 Creating activity from template: {template_id}")
			
			# Get template
			template = await self._get_activity_template(template_id, tenant_id)
			if not template:
				raise ValueError(f"Activity template not found: {template_id}")
			
			# Merge template with override data
			merged_data = {
				'title': self._merge_template_content(template.title_template, activity_data.get('merge_data', {})),
				'description': self._merge_template_content(template.description_template or '', activity_data.get('merge_data', {})),
				'activity_type': template.activity_type,
				'priority': template.priority,
				'estimated_duration_minutes': template.estimated_duration_minutes,
				'assigned_to': template.default_assigned_to or created_by,
				'team_id': template.default_team_id,
			}
			
			# Auto-set due date if configured
			if template.auto_due_date_days:
				merged_data['due_date'] = datetime.utcnow() + timedelta(days=template.auto_due_date_days)
			
			# Override with provided data
			merged_data.update(activity_data)
			
			# Create activity
			activity = await self.create_activity(merged_data, tenant_id, created_by)
			
			# Update template usage
			await self._update_template_usage(template_id, tenant_id)
			
			return activity
			
		except Exception as e:
			logger.error(f"Failed to create activity from template: {str(e)}", exc_info=True)
			raise
	
	async def complete_activity(
		self,
		activity_id: str,
		tenant_id: str,
		completed_by: str,
		outcome: Optional[str] = None,
		notes: Optional[str] = None
	) -> CRMActivity:
		"""
		Mark activity as completed
		
		Args:
			activity_id: Activity identifier
			tenant_id: Tenant identifier
			completed_by: User completing the activity
			outcome: Activity outcome
			notes: Completion notes
			
		Returns:
			Updated activity
		"""
		try:
			logger.info(f"✅ Completing activity: {activity_id}")
			
			async with self.db_manager.get_connection() as conn:
				# Update activity
				await conn.execute("""
					UPDATE crm_activities SET
						status = $3,
						completed_at = NOW(),
						outcome = $4,
						notes = COALESCE($5, notes),
						updated_at = NOW(),
						updated_by = $6,
						version = version + 1
					WHERE id = $1 AND tenant_id = $2
				""", activity_id, tenant_id, ActivityStatus.COMPLETED.value, 
				outcome, notes, completed_by)
				
				# Get updated activity
				activity_row = await conn.fetchrow("""
					SELECT * FROM crm_activities
					WHERE id = $1 AND tenant_id = $2
				""", activity_id, tenant_id)
				
				if activity_row:
					return CRMActivity(**dict(activity_row))
				else:
					raise ValueError(f"Activity not found: {activity_id}")
			
		except Exception as e:
			logger.error(f"Failed to complete activity: {str(e)}", exc_info=True)
			raise
	
	async def get_calendar_analytics(
		self,
		tenant_id: str,
		start_date: datetime,
		end_date: datetime,
		user_id: Optional[str] = None
	) -> CalendarAnalytics:
		"""
		Get comprehensive calendar and activity analytics
		
		Args:
			tenant_id: Tenant identifier
			start_date: Analysis period start
			end_date: Analysis period end
			user_id: Optional user filter
			
		Returns:
			Calendar analytics data
		"""
		try:
			logger.info(f"📊 Generating calendar analytics for tenant: {tenant_id}")
			
			analytics = CalendarAnalytics(
				tenant_id=tenant_id,
				analysis_period_start=start_date,
				analysis_period_end=end_date
			)
			
			async with self.db_manager.get_connection() as conn:
				# Event metrics
				event_stats = await conn.fetchrow("""
					SELECT 
						COUNT(*) as total_events,
						COUNT(*) FILTER (WHERE status = 'completed') as completed_events,
						COUNT(*) FILTER (WHERE status = 'cancelled') as cancelled_events,
						COUNT(*) FILTER (WHERE status = 'no_show') as no_show_events,
						COUNT(*) FILTER (WHERE event_type = 'meeting') as total_meetings,
						AVG(EXTRACT(EPOCH FROM (end_time - start_time))/60) as avg_duration,
						SUM(EXTRACT(EPOCH FROM (end_time - start_time))/60) as total_time
					FROM crm_calendar_events
					WHERE tenant_id = $1 AND start_time BETWEEN $2 AND $3
					""" + (" AND created_by = $4" if user_id else ""), 
					tenant_id, start_date, end_date, *(([user_id] if user_id else []))
				)
				
				if event_stats:
					analytics.total_events = event_stats['total_events'] or 0
					analytics.completed_events = event_stats['completed_events'] or 0
					analytics.cancelled_events = event_stats['cancelled_events'] or 0
					analytics.no_show_events = event_stats['no_show_events'] or 0
					analytics.total_meetings = event_stats['total_meetings'] or 0
					analytics.average_meeting_duration = event_stats['avg_duration'] or 0.0
					analytics.total_meeting_time = int(event_stats['total_time'] or 0)
				
				# Activity metrics
				activity_stats = await conn.fetchrow("""
					SELECT 
						COUNT(*) as total_activities,
						COUNT(*) FILTER (WHERE status = 'completed') as completed_activities,
						COUNT(*) FILTER (WHERE status = 'overdue') as overdue_activities,
						AVG(EXTRACT(EPOCH FROM (completed_at - created_at))/3600) as avg_completion_hours
					FROM crm_activities
					WHERE tenant_id = $1 AND created_at BETWEEN $2 AND $3
					""" + (" AND assigned_to = $4" if user_id else ""),
					tenant_id, start_date, end_date, *(([user_id] if user_id else []))
				)
				
				if activity_stats:
					analytics.total_activities = activity_stats['total_activities'] or 0
					analytics.completed_activities = activity_stats['completed_activities'] or 0
					analytics.overdue_activities = activity_stats['overdue_activities'] or 0
					analytics.average_completion_time = activity_stats['avg_completion_hours'] or 0.0
				
				# Calculate rates
				period_days = (end_date - start_date).days + 1
				if period_days > 0:
					analytics.events_per_day = analytics.total_events / period_days
					analytics.activities_per_day = analytics.total_activities / period_days
				
				if analytics.total_activities > 0:
					analytics.completion_rate = (analytics.completed_activities / analytics.total_activities) * 100
				
				# Hourly distribution
				hourly_stats = await conn.fetch("""
					SELECT 
						EXTRACT(HOUR FROM start_time) as hour,
						COUNT(*) as count
					FROM crm_calendar_events
					WHERE tenant_id = $1 AND start_time BETWEEN $2 AND $3
					GROUP BY EXTRACT(HOUR FROM start_time)
					ORDER BY hour
				""", tenant_id, start_date, end_date)
				
				analytics.busiest_hours = {
					int(row['hour']): row['count'] for row in hourly_stats
				}
				
				# Daily distribution
				daily_stats = await conn.fetch("""
					SELECT 
						TO_CHAR(start_time, 'Day') as day_name,
						COUNT(*) as count
					FROM crm_calendar_events
					WHERE tenant_id = $1 AND start_time BETWEEN $2 AND $3
					GROUP BY TO_CHAR(start_time, 'Day'), EXTRACT(DOW FROM start_time)
					ORDER BY EXTRACT(DOW FROM start_time)
				""", tenant_id, start_date, end_date)
				
				analytics.busiest_days = {
					row['day_name'].strip(): row['count'] for row in daily_stats
				}
			
			logger.info(f"✅ Generated analytics for {analytics.total_events} events and {analytics.total_activities} activities")
			return analytics
			
		except Exception as e:
			logger.error(f"Failed to generate calendar analytics: {str(e)}", exc_info=True)
			raise
	
	async def get_upcoming_events(
		self,
		tenant_id: str,
		user_id: Optional[str] = None,
		days_ahead: int = 7,
		limit: int = 50
	) -> List[CalendarEvent]:
		"""
		Get upcoming calendar events
		
		Args:
			tenant_id: Tenant identifier
			user_id: Optional user filter
			days_ahead: Days to look ahead
			limit: Maximum events to return
			
		Returns:
			List of upcoming events
		"""
		try:
			async with self.db_manager.get_connection() as conn:
				# Build query
				where_clause = "WHERE tenant_id = $1 AND start_time >= NOW() AND start_time <= NOW() + INTERVAL '%s days'"
				params = [tenant_id, days_ahead]
				
				if user_id:
					where_clause += " AND created_by = $3"
					params.append(user_id)
				
				# Get events
				event_rows = await conn.fetch(f"""
					SELECT * FROM crm_calendar_events
					{where_clause}
					ORDER BY start_time
					LIMIT ${'$3' if not user_id else '$4'}
				""", *params, limit)
				
				events = [CalendarEvent(**dict(row)) for row in event_rows]
				return events
				
		except Exception as e:
			logger.error(f"Failed to get upcoming events: {str(e)}", exc_info=True)
			raise
	
	async def get_overdue_activities(
		self,
		tenant_id: str,
		user_id: Optional[str] = None
	) -> List[CRMActivity]:
		"""
		Get overdue activities
		
		Args:
			tenant_id: Tenant identifier
			user_id: Optional user filter
			
		Returns:
			List of overdue activities
		"""
		try:
			async with self.db_manager.get_connection() as conn:
				# Build query
				where_clause = "WHERE tenant_id = $1 AND due_date < NOW() AND status NOT IN ('completed', 'cancelled')"
				params = [tenant_id]
				
				if user_id:
					where_clause += " AND assigned_to = $2"
					params.append(user_id)
				
				# Get activities
				activity_rows = await conn.fetch(f"""
					SELECT * FROM crm_activities
					{where_clause}
					ORDER BY due_date
				""", *params)
				
				activities = [CRMActivity(**dict(row)) for row in activity_rows]
				return activities
				
		except Exception as e:
			logger.error(f"Failed to get overdue activities: {str(e)}", exc_info=True)
			raise
	
	# Helper methods
	
	def _merge_template_content(self, content: str, merge_data: Dict[str, Any]) -> str:
		"""Merge template content with data"""
		try:
			if not content:
				return ""
			
			# Simple merge - replace {{field}} with values
			for field, value in merge_data.items():
				placeholder = f"{{{{{field}}}}}"
				content = content.replace(placeholder, str(value))
			
			return content
			
		except Exception:
			return content
	
	async def _store_event_attendee(self, attendee: CalendarEventAttendee):
		"""Store event attendee in database"""
		try:
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					INSERT INTO crm_calendar_event_attendees (
						id, event_id, email, name, role, status, response_time,
						contact_id, account_id, lead_id, is_organizer, is_required,
						send_notifications, metadata
					) VALUES (
						$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14
					)
				""", 
				attendee.id, attendee.event_id, attendee.email, attendee.name, 
				attendee.role, attendee.status.value, attendee.response_time,
				attendee.contact_id, attendee.account_id, attendee.lead_id,
				attendee.is_organizer, attendee.is_required, attendee.send_notifications,
				attendee.metadata
				)
		except Exception as e:
			logger.error(f"Failed to store event attendee: {str(e)}")
			raise
	
	async def _store_event_recurrence(self, recurrence: CalendarEventRecurrence):
		"""Store event recurrence in database"""
		try:
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					INSERT INTO crm_calendar_event_recurrence (
						id, event_id, frequency, interval, days_of_week, days_of_month,
						months_of_year, end_date, occurrence_count, exception_dates,
						custom_rule, metadata
					) VALUES (
						$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12
					)
				""", 
				recurrence.id, recurrence.event_id, recurrence.frequency.value, 
				recurrence.interval, recurrence.days_of_week, recurrence.days_of_month,
				recurrence.months_of_year, recurrence.end_date, recurrence.occurrence_count,
				recurrence.exception_dates, recurrence.custom_rule, recurrence.metadata
				)
		except Exception as e:
			logger.error(f"Failed to store event recurrence: {str(e)}")
			raise
	
	async def _get_activity_template(self, template_id: str, tenant_id: str) -> Optional[ActivityTemplate]:
		"""Get activity template by ID"""
		try:
			async with self.db_manager.get_connection() as conn:
				template_row = await conn.fetchrow("""
					SELECT * FROM crm_activity_templates
					WHERE id = $1 AND tenant_id = $2
				""", template_id, tenant_id)
				
				if template_row:
					return ActivityTemplate(**dict(template_row))
				return None
				
		except Exception as e:
			logger.error(f"Failed to get activity template: {str(e)}")
			return None
	
	async def _update_template_usage(self, template_id: str, tenant_id: str):
		"""Update template usage statistics"""
		try:
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					UPDATE crm_activity_templates 
					SET usage_count = usage_count + 1, updated_at = NOW()
					WHERE id = $1 AND tenant_id = $2
				""", template_id, tenant_id)
		except Exception as e:
			logger.error(f"Failed to update template usage: {str(e)}")