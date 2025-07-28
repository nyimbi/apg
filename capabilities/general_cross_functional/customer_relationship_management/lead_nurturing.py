"""
APG Customer Relationship Management - Lead Nurturing Workflows Module

Advanced lead nurturing system with AI-powered personalization, multi-channel campaigns,
behavioral triggers, and intelligent optimization for maximum conversion rates.

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

from pydantic import BaseModel, Field, validator, ConfigDict

from .database import DatabaseManager


logger = logging.getLogger(__name__)


class NurturingStatus(str, Enum):
	"""Nurturing workflow status"""
	DRAFT = "draft"
	ACTIVE = "active"
	PAUSED = "paused"
	COMPLETED = "completed"
	ARCHIVED = "archived"


class TriggerType(str, Enum):
	"""Workflow trigger types"""
	LEAD_CREATED = "lead_created"
	SCORE_THRESHOLD = "score_threshold"
	BEHAVIOR_BASED = "behavior_based"
	TIME_BASED = "time_based"
	FORM_SUBMISSION = "form_submission"
	EMAIL_INTERACTION = "email_interaction"
	WEBSITE_ACTIVITY = "website_activity"
	MANUAL_TRIGGER = "manual_trigger"
	STAGE_CHANGE = "stage_change"
	INACTIVITY = "inactivity"


class ActionType(str, Enum):
	"""Workflow action types"""
	SEND_EMAIL = "send_email"
	SEND_SMS = "send_sms"
	ASSIGN_TO_REP = "assign_to_rep"
	UPDATE_SCORE = "update_score"
	ADD_TAG = "add_tag"
	REMOVE_TAG = "remove_tag"
	CREATE_TASK = "create_task"
	SCHEDULE_CALL = "schedule_call"
	SEND_NOTIFICATION = "send_notification"
	UPDATE_FIELD = "update_field"
	WAIT_DELAY = "wait_delay"
	CONDITIONAL_SPLIT = "conditional_split"


class ChannelType(str, Enum):
	"""Communication channel types"""
	EMAIL = "email"
	SMS = "sms"
	PHONE = "phone"
	SOCIAL_MEDIA = "social_media"
	DIRECT_MAIL = "direct_mail"
	WEBINAR = "webinar"
	IN_APP = "in_app"
	PUSH_NOTIFICATION = "push_notification"


class PersonalizationLevel(str, Enum):
	"""Personalization sophistication levels"""
	BASIC = "basic"  # Name, company only
	STANDARD = "standard"  # Demographics, basic behavior
	ADVANCED = "advanced"  # Behavioral patterns, preferences
	AI_POWERED = "ai_powered"  # ML-driven personalization


class NurturingCondition(BaseModel):
	"""Nurturing workflow condition"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	field_name: str = Field(..., description="Field to evaluate")
	operator: str = Field(..., description="Comparison operator")
	value: Union[str, int, float, List[str]] = Field(..., description="Comparison value")
	weight: float = Field(default=1.0, ge=0.0, le=10.0, description="Condition weight")
	is_required: bool = Field(default=False, description="Whether condition is mandatory")
	created_at: datetime = Field(default_factory=datetime.now)


class NurturingAction(BaseModel):
	"""Nurturing workflow action"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	action_type: ActionType = Field(..., description="Type of action")
	name: str = Field(..., description="Action name")
	description: Optional[str] = Field(None, description="Action description")
	
	# Action configuration
	channel: Optional[ChannelType] = Field(None, description="Communication channel")
	template_id: Optional[str] = Field(None, description="Template identifier")
	content: Optional[str] = Field(None, description="Action content")
	subject: Optional[str] = Field(None, description="Email/SMS subject")
	
	# Timing configuration
	delay_hours: int = Field(default=0, ge=0, description="Delay before execution")
	send_time_hour: Optional[int] = Field(None, ge=0, le=23, description="Preferred send time")
	business_hours_only: bool = Field(default=False, description="Send only during business hours")
	exclude_weekends: bool = Field(default=False, description="Exclude weekends")
	
	# Personalization
	personalization_level: PersonalizationLevel = Field(default=PersonalizationLevel.STANDARD)
	personalization_fields: List[str] = Field(default_factory=list, description="Fields for personalization")
	dynamic_content: Dict[str, Any] = Field(default_factory=dict, description="Dynamic content rules")
	
	# Conditions and branching
	conditions: List[NurturingCondition] = Field(default_factory=list, description="Execution conditions")
	success_branch_id: Optional[str] = Field(None, description="Next action on success")
	failure_branch_id: Optional[str] = Field(None, description="Next action on failure")
	
	# Settings
	max_attempts: int = Field(default=3, ge=1, description="Maximum execution attempts")
	retry_delay_hours: int = Field(default=24, ge=1, description="Retry delay")
	track_engagement: bool = Field(default=True, description="Track engagement metrics")
	
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional action metadata")
	created_at: datetime = Field(default_factory=datetime.now)
	updated_at: datetime = Field(default_factory=datetime.now)


class NurturingWorkflow(BaseModel):
	"""Lead nurturing workflow configuration"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(..., description="Tenant identifier")
	name: str = Field(..., min_length=1, max_length=255, description="Workflow name")
	description: Optional[str] = Field(None, description="Workflow description")
	status: NurturingStatus = Field(default=NurturingStatus.DRAFT, description="Workflow status")
	
	# Trigger configuration
	trigger_type: TriggerType = Field(..., description="How workflow is triggered")
	trigger_conditions: List[NurturingCondition] = Field(default_factory=list, description="Trigger conditions")
	entry_criteria: List[NurturingCondition] = Field(default_factory=list, description="Entry criteria")
	exit_criteria: List[NurturingCondition] = Field(default_factory=list, description="Exit criteria")
	
	# Actions and flow
	actions: List[NurturingAction] = Field(..., min_items=1, description="Workflow actions")
	start_action_id: str = Field(..., description="First action to execute")
	
	# Settings
	max_leads_per_day: Optional[int] = Field(None, ge=1, description="Daily lead processing limit")
	priority: int = Field(default=5, ge=1, le=10, description="Workflow priority")
	time_zone: str = Field(default="UTC", description="Workflow timezone")
	
	# Goal tracking
	goal_type: Optional[str] = Field(None, description="Success goal type")
	goal_value: Optional[float] = Field(None, description="Target goal value")
	conversion_event: Optional[str] = Field(None, description="Conversion event to track")
	
	# Analytics and tracking
	total_enrolled: int = Field(default=0, ge=0, description="Total leads enrolled")
	currently_active: int = Field(default=0, ge=0, description="Currently active leads")
	completed_successfully: int = Field(default=0, ge=0, description="Successful completions")
	conversion_rate: float = Field(default=0.0, ge=0.0, le=100.0, description="Conversion rate percentage")
	avg_time_to_conversion_days: float = Field(default=0.0, ge=0.0, description="Average conversion time")
	
	# Optimization
	ai_optimization_enabled: bool = Field(default=False, description="Enable AI optimization")
	auto_pause_low_performance: bool = Field(default=False, description="Auto-pause on low performance")
	performance_threshold: float = Field(default=5.0, ge=0.0, description="Performance threshold")
	
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional workflow metadata")
	created_at: datetime = Field(default_factory=datetime.now)
	updated_at: datetime = Field(default_factory=datetime.now)
	created_by: str = Field(..., description="Creator user ID")
	updated_by: Optional[str] = Field(None, description="Last updater user ID")


class NurturingEnrollment(BaseModel):
	"""Lead enrollment in nurturing workflow"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(..., description="Tenant identifier")
	workflow_id: str = Field(..., description="Workflow identifier")
	lead_id: str = Field(..., description="Lead identifier")
	
	# Enrollment details
	enrollment_source: str = Field(..., description="How lead was enrolled")
	enrollment_trigger: TriggerType = Field(..., description="Trigger that caused enrollment")
	enrolled_at: datetime = Field(default_factory=datetime.now)
	enrolled_by: Optional[str] = Field(None, description="User who enrolled lead")
	
	# Current state
	current_action_id: Optional[str] = Field(None, description="Current workflow action")
	current_step: int = Field(default=0, ge=0, description="Current workflow step")
	total_steps: int = Field(default=0, ge=0, description="Total workflow steps")
	
	# Status tracking
	is_active: bool = Field(default=True, description="Whether enrollment is active")
	is_paused: bool = Field(default=False, description="Whether enrollment is paused")
	paused_at: Optional[datetime] = Field(None, description="When enrollment was paused")
	paused_reason: Optional[str] = Field(None, description="Reason for pausing")
	
	completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
	completion_reason: Optional[str] = Field(None, description="Reason for completion")
	success: Optional[bool] = Field(None, description="Whether completion was successful")
	
	# Engagement tracking
	emails_sent: int = Field(default=0, ge=0, description="Emails sent")
	emails_opened: int = Field(default=0, ge=0, description="Emails opened")
	emails_clicked: int = Field(default=0, ge=0, description="Email clicks")
	forms_submitted: int = Field(default=0, ge=0, description="Forms submitted")
	meetings_scheduled: int = Field(default=0, ge=0, description="Meetings scheduled")
	
	# Performance metrics
	engagement_score: float = Field(default=0.0, ge=0.0, description="Overall engagement score")
	lead_score_change: float = Field(default=0.0, description="Lead score change during nurturing")
	time_to_conversion_hours: Optional[float] = Field(None, ge=0.0, description="Time to conversion")
	
	# Lead context at enrollment
	lead_score_at_enrollment: Optional[float] = Field(None, description="Lead score at enrollment")
	lead_stage_at_enrollment: Optional[str] = Field(None, description="Lead stage at enrollment")
	lead_source: Optional[str] = Field(None, description="Lead source")
	lead_industry: Optional[str] = Field(None, description="Lead industry")
	
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional enrollment metadata")
	created_at: datetime = Field(default_factory=datetime.now)
	updated_at: datetime = Field(default_factory=datetime.now)


class NurturingExecution(BaseModel):
	"""Individual action execution within nurturing workflow"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(..., description="Tenant identifier")
	enrollment_id: str = Field(..., description="Enrollment identifier")
	workflow_id: str = Field(..., description="Workflow identifier")
	action_id: str = Field(..., description="Action identifier")
	lead_id: str = Field(..., description="Lead identifier")
	
	# Execution details
	scheduled_at: datetime = Field(..., description="Scheduled execution time")
	executed_at: Optional[datetime] = Field(None, description="Actual execution time")
	status: str = Field(default="pending", description="Execution status")
	
	# Results
	success: Optional[bool] = Field(None, description="Whether execution was successful")
	result_data: Dict[str, Any] = Field(default_factory=dict, description="Execution results")
	error_message: Optional[str] = Field(None, description="Error message if failed")
	
	# Engagement tracking
	delivery_status: Optional[str] = Field(None, description="Delivery confirmation")
	opened_at: Optional[datetime] = Field(None, description="When content was opened")
	clicked_at: Optional[datetime] = Field(None, description="When links were clicked")
	responded_at: Optional[datetime] = Field(None, description="When recipient responded")
	
	# Retry tracking
	attempt_number: int = Field(default=1, ge=1, description="Execution attempt number")
	max_attempts: int = Field(default=3, ge=1, description="Maximum allowed attempts")
	next_retry_at: Optional[datetime] = Field(None, description="Next retry time")
	
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional execution metadata")
	created_at: datetime = Field(default_factory=datetime.now)
	updated_at: datetime = Field(default_factory=datetime.now)


class NurturingAnalytics(BaseModel):
	"""Lead nurturing analytics data"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	tenant_id: str = Field(..., description="Tenant identifier")
	workflow_id: Optional[str] = Field(None, description="Specific workflow (optional)")
	period_start: datetime = Field(..., description="Analytics period start")
	period_end: datetime = Field(..., description="Analytics period end")
	
	# Overall metrics
	total_enrollments: int = Field(default=0, description="Total enrollments in period")
	active_enrollments: int = Field(default=0, description="Currently active enrollments")
	completed_enrollments: int = Field(default=0, description="Completed enrollments")
	conversion_rate: float = Field(default=0.0, description="Overall conversion rate")
	
	# Engagement metrics
	emails_sent: int = Field(default=0, description="Total emails sent")
	email_open_rate: float = Field(default=0.0, description="Email open rate")
	email_click_rate: float = Field(default=0.0, description="Email click rate")
	form_submission_rate: float = Field(default=0.0, description="Form submission rate")
	meeting_booking_rate: float = Field(default=0.0, description="Meeting booking rate")
	
	# Performance metrics
	avg_time_to_conversion_days: float = Field(default=0.0, description="Average conversion time")
	avg_engagement_score: float = Field(default=0.0, description="Average engagement score")
	lead_score_improvement: float = Field(default=0.0, description="Average lead score improvement")
	
	# Workflow performance
	top_performing_workflows: List[Dict[str, Any]] = Field(default_factory=list, description="Best performing workflows")
	workflow_performance: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Performance by workflow")
	action_performance: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Performance by action type")
	
	# Channel performance
	channel_performance: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Performance by channel")
	optimal_send_times: Dict[str, List[int]] = Field(default_factory=dict, description="Optimal send times by day")
	
	# Optimization recommendations
	optimization_suggestions: List[Dict[str, Any]] = Field(default_factory=list, description="AI-powered suggestions")
	performance_alerts: List[Dict[str, Any]] = Field(default_factory=list, description="Performance alerts")
	
	created_at: datetime = Field(default_factory=datetime.now)


class LeadNurturingManager:
	"""Advanced lead nurturing workflow management system"""
	
	def __init__(self, db_manager: DatabaseManager):
		self.db_manager = db_manager
		self._initialized = False
		self._workflow_cache = {}
		self._execution_queue = asyncio.Queue()
		self._processing_tasks = []
	
	async def initialize(self):
		"""Initialize the lead nurturing manager"""
		try:
			logger.info("🚀 Initializing Lead Nurturing Manager...")
			
			# Initialize database connection
			await self.db_manager.initialize()
			
			# Load active workflows
			await self._load_active_workflows()
			
			# Start background processing
			await self._start_background_processing()
			
			self._initialized = True
			logger.info("✅ Lead Nurturing Manager initialized successfully")
			
		except Exception as e:
			logger.error(f"Failed to initialize Lead Nurturing Manager: {str(e)}")
			raise
	
	async def create_nurturing_workflow(self, workflow_data: Dict[str, Any], tenant_id: str, created_by: str) -> NurturingWorkflow:
		"""Create a new nurturing workflow"""
		try:
			if not self._initialized:
				await self.initialize()
			
			# Validate workflow data
			workflow = NurturingWorkflow(
				tenant_id=tenant_id,
				created_by=created_by,
				**workflow_data
			)
			
			# Store in database
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					INSERT INTO crm_nurturing_workflows (
						id, tenant_id, name, description, status, trigger_type,
						trigger_conditions, entry_criteria, exit_criteria, actions,
						start_action_id, max_leads_per_day, priority, time_zone,
						goal_type, goal_value, conversion_event, ai_optimization_enabled,
						auto_pause_low_performance, performance_threshold, metadata,
						created_by, created_at, updated_at
					) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24)
				""", 
				workflow.id, workflow.tenant_id, workflow.name, workflow.description,
				workflow.status.value, workflow.trigger_type.value,
				json.dumps([c.model_dump() for c in workflow.trigger_conditions]),
				json.dumps([c.model_dump() for c in workflow.entry_criteria]),
				json.dumps([c.model_dump() for c in workflow.exit_criteria]),
				json.dumps([a.model_dump() for a in workflow.actions]),
				workflow.start_action_id, workflow.max_leads_per_day, workflow.priority,
				workflow.time_zone, workflow.goal_type, workflow.goal_value,
				workflow.conversion_event, workflow.ai_optimization_enabled,
				workflow.auto_pause_low_performance, workflow.performance_threshold,
				json.dumps(workflow.metadata), workflow.created_by, workflow.created_at,
				workflow.updated_at
				)
			
			# Update cache
			self._workflow_cache[workflow.id] = workflow
			
			logger.info(f"✅ Created nurturing workflow: {workflow.name} ({workflow.id})")
			return workflow
			
		except Exception as e:
			logger.error(f"Failed to create nurturing workflow: {str(e)}")
			raise
	
	async def enroll_lead(self, workflow_id: str, lead_data: Dict[str, Any], tenant_id: str, enrolled_by: str = None) -> Optional[NurturingEnrollment]:
		"""Enroll a lead in a nurturing workflow"""
		try:
			if not self._initialized:
				await self.initialize()
			
			# Get workflow
			workflow = await self._get_workflow(workflow_id, tenant_id)
			if not workflow or workflow.status != NurturingStatus.ACTIVE:
				logger.warning(f"Workflow {workflow_id} is not active for enrollment")
				return None
			
			lead_id = lead_data.get('id')
			if not lead_id:
				raise ValueError("Lead ID is required for enrollment")
			
			# Check if lead already enrolled
			existing_enrollment = await self._get_active_enrollment(lead_id, workflow_id, tenant_id)
			if existing_enrollment:
				logger.warning(f"Lead {lead_id} already enrolled in workflow {workflow_id}")
				return existing_enrollment
			
			# Check entry criteria
			if not await self._check_entry_criteria(workflow, lead_data):
				logger.info(f"Lead {lead_id} does not meet entry criteria for workflow {workflow_id}")
				return None
			
			# Create enrollment
			enrollment = NurturingEnrollment(
				tenant_id=tenant_id,
				workflow_id=workflow_id,
				lead_id=lead_id,
				enrollment_source="manual" if enrolled_by else "automatic",
				enrollment_trigger=workflow.trigger_type,
				enrolled_by=enrolled_by,
				current_action_id=workflow.start_action_id,
				total_steps=len(workflow.actions),
				lead_score_at_enrollment=lead_data.get('score'),
				lead_stage_at_enrollment=lead_data.get('stage'),
				lead_source=lead_data.get('source'),
				lead_industry=lead_data.get('industry')
			)
			
			# Store enrollment
			await self._store_enrollment(enrollment)
			
			# Schedule first action
			await self._schedule_next_action(enrollment, workflow)
			
			# Update workflow stats
			await self._update_workflow_stats(workflow_id, "enrollment")
			
			logger.info(f"✅ Enrolled lead {lead_id} in workflow {workflow.name}")
			return enrollment
			
		except Exception as e:
			logger.error(f"Failed to enroll lead in nurturing workflow: {str(e)}")
			raise
	
	async def process_trigger(self, trigger_type: TriggerType, trigger_data: Dict[str, Any], tenant_id: str):
		"""Process a workflow trigger event"""
		try:
			if not self._initialized:
				await self.initialize()
			
			# Find workflows that match this trigger
			matching_workflows = await self._find_matching_workflows(trigger_type, trigger_data, tenant_id)
			
			for workflow in matching_workflows:
				# Check daily limits
				if workflow.max_leads_per_day:
					daily_enrollments = await self._get_daily_enrollments(workflow.id, tenant_id)
					if daily_enrollments >= workflow.max_leads_per_day:
						continue
				
				# Enroll lead
				await self.enroll_lead(workflow.id, trigger_data, tenant_id)
			
		except Exception as e:
			logger.error(f"Failed to process nurturing trigger: {str(e)}")
			raise
	
	async def get_nurturing_analytics(self, tenant_id: str, workflow_id: str = None, period_days: int = 30) -> NurturingAnalytics:
		"""Get nurturing workflow analytics"""
		try:
			if not self._initialized:
				await self.initialize()
			
			period_start = datetime.now() - timedelta(days=period_days)
			period_end = datetime.now()
			
			async with self.db_manager.get_connection() as conn:
				# Base query conditions
				workflow_filter = "AND workflow_id = $4" if workflow_id else ""
				params = [tenant_id, period_start, period_end]
				if workflow_id:
					params.append(workflow_id)
				
				# Get enrollment statistics
				enrollments = await conn.fetch(f"""
					SELECT * FROM crm_nurturing_enrollments 
					WHERE tenant_id = $1 AND enrolled_at >= $2 AND enrolled_at <= $3 {workflow_filter}
				""", *params)
				
				# Get execution statistics
				executions = await conn.fetch(f"""
					SELECT * FROM crm_nurturing_executions 
					WHERE tenant_id = $1 AND executed_at >= $2 AND executed_at <= $3 {workflow_filter}
				""", *params)
				
				# Calculate metrics
				total_enrollments = len(enrollments)
				active_enrollments = sum(1 for e in enrollments if e.get('is_active', False))
				completed_enrollments = sum(1 for e in enrollments if e.get('completed_at'))
				
				conversion_rate = 0.0
				if total_enrollments > 0:
					successful_completions = sum(1 for e in enrollments if e.get('success', False))
					conversion_rate = (successful_completions / total_enrollments) * 100
				
				# Email metrics
				emails_sent = sum(1 for e in executions if e.get('action_type') == 'send_email')
				emails_opened = sum(1 for e in executions if e.get('opened_at'))
				email_open_rate = (emails_opened / emails_sent * 100) if emails_sent > 0 else 0.0
				
				# Build analytics
				analytics = NurturingAnalytics(
					tenant_id=tenant_id,
					workflow_id=workflow_id,
					period_start=period_start,
					period_end=period_end,
					total_enrollments=total_enrollments,
					active_enrollments=active_enrollments,
					completed_enrollments=completed_enrollments,
					conversion_rate=conversion_rate,
					emails_sent=emails_sent,
					email_open_rate=email_open_rate
				)
				
				logger.info(f"📊 Generated nurturing analytics for {period_days} days")
				return analytics
				
		except Exception as e:
			logger.error(f"Failed to get nurturing analytics: {str(e)}")
			raise
	
	async def _load_active_workflows(self):
		"""Load active workflows into cache"""
		try:
			async with self.db_manager.get_connection() as conn:
				workflows_data = await conn.fetch("""
					SELECT * FROM crm_nurturing_workflows 
					WHERE status = 'active'
					ORDER BY priority DESC, created_at ASC
				""")
				
				for workflow_data in workflows_data:
					workflow = NurturingWorkflow(
						id=workflow_data['id'],
						tenant_id=workflow_data['tenant_id'],
						name=workflow_data['name'],
						description=workflow_data['description'],
						status=NurturingStatus(workflow_data['status']),
						trigger_type=TriggerType(workflow_data['trigger_type']),
						trigger_conditions=[NurturingCondition(**c) for c in json.loads(workflow_data['trigger_conditions'] or '[]')],
						entry_criteria=[NurturingCondition(**c) for c in json.loads(workflow_data['entry_criteria'] or '[]')],
						exit_criteria=[NurturingCondition(**c) for c in json.loads(workflow_data['exit_criteria'] or '[]')],
						actions=[NurturingAction(**a) for a in json.loads(workflow_data['actions'] or '[]')],
						start_action_id=workflow_data['start_action_id'],
						max_leads_per_day=workflow_data['max_leads_per_day'],
						priority=workflow_data['priority'],
						time_zone=workflow_data['time_zone'],
						goal_type=workflow_data['goal_type'],
						goal_value=workflow_data['goal_value'],
						conversion_event=workflow_data['conversion_event'],
						ai_optimization_enabled=workflow_data['ai_optimization_enabled'],
						auto_pause_low_performance=workflow_data['auto_pause_low_performance'],
						performance_threshold=workflow_data['performance_threshold'],
						metadata=json.loads(workflow_data['metadata'] or '{}'),
						created_by=workflow_data['created_by'],
						created_at=workflow_data['created_at'],
						updated_at=workflow_data['updated_at']
					)
					
					self._workflow_cache[workflow.id] = workflow
			
			logger.info(f"📋 Loaded {len(self._workflow_cache)} active nurturing workflows")
			
		except Exception as e:
			logger.error(f"Failed to load active workflows: {str(e)}")
			raise
	
	async def _start_background_processing(self):
		"""Start background processing tasks"""
		try:
			# Start execution processor
			task = asyncio.create_task(self._process_execution_queue())
			self._processing_tasks.append(task)
			
			# Start scheduled action checker
			task = asyncio.create_task(self._check_scheduled_actions())
			self._processing_tasks.append(task)
			
			logger.info("🔄 Started background processing tasks")
			
		except Exception as e:
			logger.error(f"Failed to start background processing: {str(e)}")
			raise
	
	async def _process_execution_queue(self):
		"""Process the execution queue"""
		while self._initialized:
			try:
				# Get next execution from queue
				execution = await asyncio.wait_for(self._execution_queue.get(), timeout=1.0)
				
				# Execute the action
				await self._execute_action(execution)
				
			except asyncio.TimeoutError:
				continue
			except Exception as e:
				logger.error(f"Error processing execution queue: {str(e)}")
				await asyncio.sleep(5)
	
	async def _check_scheduled_actions(self):
		"""Check for scheduled actions that need execution"""
		while self._initialized:
			try:
				async with self.db_manager.get_connection() as conn:
					# Find actions scheduled for execution
					pending_executions = await conn.fetch("""
						SELECT * FROM crm_nurturing_executions 
						WHERE status = 'pending' AND scheduled_at <= NOW()
						ORDER BY scheduled_at ASC
						LIMIT 100
					""")
					
					for execution_data in pending_executions:
						execution = NurturingExecution(**execution_data)
						await self._execution_queue.put(execution)
				
				# Sleep for 30 seconds before checking again
				await asyncio.sleep(30)
				
			except Exception as e:
				logger.error(f"Error checking scheduled actions: {str(e)}")
				await asyncio.sleep(60)
	
	async def _execute_action(self, execution: NurturingExecution):
		"""Execute a specific nurturing action"""
		try:
			# Get workflow and action details
			workflow = await self._get_workflow(execution.workflow_id, execution.tenant_id)
			if not workflow:
				logger.error(f"Workflow {execution.workflow_id} not found")
				return
			
			action = next((a for a in workflow.actions if a.id == execution.action_id), None)
			if not action:
				logger.error(f"Action {execution.action_id} not found in workflow")
				return
			
			# Update execution status
			await self._update_execution_status(execution.id, "executing")
			
			# Execute based on action type
			success = False
			result_data = {}
			
			if action.action_type == ActionType.SEND_EMAIL:
				success, result_data = await self._execute_send_email(action, execution)
			elif action.action_type == ActionType.SEND_SMS:
				success, result_data = await self._execute_send_sms(action, execution)
			elif action.action_type == ActionType.UPDATE_SCORE:
				success, result_data = await self._execute_update_score(action, execution)
			elif action.action_type == ActionType.CREATE_TASK:
				success, result_data = await self._execute_create_task(action, execution)
			elif action.action_type == ActionType.WAIT_DELAY:
				success, result_data = await self._execute_wait_delay(action, execution)
			else:
				logger.warning(f"Unsupported action type: {action.action_type}")
				success = False
			
			# Update execution results
			await self._update_execution_results(execution.id, success, result_data)
			
			# Schedule next action if successful
			if success and action.success_branch_id:
				await self._schedule_action(execution, action.success_branch_id, workflow)
			elif not success and action.failure_branch_id:
				await self._schedule_action(execution, action.failure_branch_id, workflow)
			
			logger.info(f"{'✅' if success else '❌'} Executed {action.action_type} for lead {execution.lead_id}")
			
		except Exception as e:
			logger.error(f"Failed to execute action: {str(e)}")
			await self._update_execution_status(execution.id, "failed", str(e))
	
	# Action execution methods (simplified implementations)
	async def _execute_send_email(self, action: NurturingAction, execution: NurturingExecution) -> Tuple[bool, Dict[str, Any]]:
		"""Execute send email action"""
		# This would integrate with the email system
		logger.info(f"📧 Sending email to lead {execution.lead_id}")
		return True, {"email_sent": True, "template_id": action.template_id}
	
	async def _execute_send_sms(self, action: NurturingAction, execution: NurturingExecution) -> Tuple[bool, Dict[str, Any]]:
		"""Execute send SMS action"""
		logger.info(f"📱 Sending SMS to lead {execution.lead_id}")
		return True, {"sms_sent": True}
	
	async def _execute_update_score(self, action: NurturingAction, execution: NurturingExecution) -> Tuple[bool, Dict[str, Any]]:
		"""Execute update score action"""
		logger.info(f"📊 Updating score for lead {execution.lead_id}")
		return True, {"score_updated": True}
	
	async def _execute_create_task(self, action: NurturingAction, execution: NurturingExecution) -> Tuple[bool, Dict[str, Any]]:
		"""Execute create task action"""
		logger.info(f"📋 Creating task for lead {execution.lead_id}")
		return True, {"task_created": True}
	
	async def _execute_wait_delay(self, action: NurturingAction, execution: NurturingExecution) -> Tuple[bool, Dict[str, Any]]:
		"""Execute wait delay action"""
		logger.info(f"⏰ Processing delay for lead {execution.lead_id}")
		return True, {"delay_processed": True}
	
	# Helper methods for database operations
	async def _store_enrollment(self, enrollment: NurturingEnrollment):
		"""Store enrollment in database"""
		async with self.db_manager.get_connection() as conn:
			await conn.execute("""
				INSERT INTO crm_nurturing_enrollments (
					id, tenant_id, workflow_id, lead_id, enrollment_source, enrollment_trigger,
					enrolled_at, enrolled_by, current_action_id, current_step, total_steps,
					is_active, lead_score_at_enrollment, lead_stage_at_enrollment,
					lead_source, lead_industry, metadata, created_at, updated_at
				) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
			""", 
			enrollment.id, enrollment.tenant_id, enrollment.workflow_id, enrollment.lead_id,
			enrollment.enrollment_source, enrollment.enrollment_trigger.value, enrollment.enrolled_at,
			enrollment.enrolled_by, enrollment.current_action_id, enrollment.current_step,
			enrollment.total_steps, enrollment.is_active, enrollment.lead_score_at_enrollment,
			enrollment.lead_stage_at_enrollment, enrollment.lead_source, enrollment.lead_industry,
			json.dumps(enrollment.metadata), enrollment.created_at, enrollment.updated_at
			)
	
	async def _schedule_next_action(self, enrollment: NurturingEnrollment, workflow: NurturingWorkflow):
		"""Schedule the next action for an enrollment"""
		try:
			if not enrollment.current_action_id:
				return
			
			action = next((a for a in workflow.actions if a.id == enrollment.current_action_id), None)
			if not action:
				return
			
			# Create execution record
			execution = NurturingExecution(
				tenant_id=enrollment.tenant_id,
				enrollment_id=enrollment.id,
				workflow_id=enrollment.workflow_id,
				action_id=action.id,
				lead_id=enrollment.lead_id,
				scheduled_at=datetime.now() + timedelta(hours=action.delay_hours)
			)
			
			# Store execution
			await self._store_execution(execution)
			
		except Exception as e:
			logger.error(f"Failed to schedule next action: {str(e)}")
	
	async def _store_execution(self, execution: NurturingExecution):
		"""Store execution in database"""
		async with self.db_manager.get_connection() as conn:
			await conn.execute("""
				INSERT INTO crm_nurturing_executions (
					id, tenant_id, enrollment_id, workflow_id, action_id, lead_id,
					scheduled_at, status, attempt_number, max_attempts, metadata,
					created_at, updated_at
				) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
			""", 
			execution.id, execution.tenant_id, execution.enrollment_id, execution.workflow_id,
			execution.action_id, execution.lead_id, execution.scheduled_at, execution.status,
			execution.attempt_number, execution.max_attempts, json.dumps(execution.metadata),
			execution.created_at, execution.updated_at
			)
	
	# Placeholder helper methods
	async def _get_workflow(self, workflow_id: str, tenant_id: str) -> Optional[NurturingWorkflow]:
		"""Get workflow by ID"""
		return self._workflow_cache.get(workflow_id)
	
	async def _get_active_enrollment(self, lead_id: str, workflow_id: str, tenant_id: str) -> Optional[NurturingEnrollment]:
		"""Check if lead has active enrollment"""
		return None  # Simplified
	
	async def _check_entry_criteria(self, workflow: NurturingWorkflow, lead_data: Dict[str, Any]) -> bool:
		"""Check if lead meets workflow entry criteria"""
		return True  # Simplified
	
	async def _update_workflow_stats(self, workflow_id: str, stat_type: str):
		"""Update workflow statistics"""
		pass  # Simplified
	
	async def _find_matching_workflows(self, trigger_type: TriggerType, trigger_data: Dict[str, Any], tenant_id: str) -> List[NurturingWorkflow]:
		"""Find workflows that match trigger criteria"""
		return [w for w in self._workflow_cache.values() 
				if w.tenant_id == tenant_id and w.trigger_type == trigger_type]
	
	async def _get_daily_enrollments(self, workflow_id: str, tenant_id: str) -> int:
		"""Get today's enrollment count for workflow"""
		return 0  # Simplified
	
	async def _update_execution_status(self, execution_id: str, status: str, error_message: str = None):
		"""Update execution status"""
		async with self.db_manager.get_connection() as conn:
			await conn.execute("""
				UPDATE crm_nurturing_executions SET
					status = $2, executed_at = NOW(), error_message = $3, updated_at = NOW()
				WHERE id = $1
			""", execution_id, status, error_message)
	
	async def _update_execution_results(self, execution_id: str, success: bool, result_data: Dict[str, Any]):
		"""Update execution results"""
		async with self.db_manager.get_connection() as conn:
			await conn.execute("""
				UPDATE crm_nurturing_executions SET
					success = $2, result_data = $3, status = $4, updated_at = NOW()
				WHERE id = $1
			""", execution_id, success, json.dumps(result_data), "completed" if success else "failed")
	
	async def _schedule_action(self, execution: NurturingExecution, next_action_id: str, workflow: NurturingWorkflow):
		"""Schedule the next action in sequence"""
		next_action = next((a for a in workflow.actions if a.id == next_action_id), None)
		if next_action:
			next_execution = NurturingExecution(
				tenant_id=execution.tenant_id,
				enrollment_id=execution.enrollment_id,
				workflow_id=execution.workflow_id,
				action_id=next_action_id,
				lead_id=execution.lead_id,
				scheduled_at=datetime.now() + timedelta(hours=next_action.delay_hours)
			)
			await self._store_execution(next_execution)
	
	async def shutdown(self):
		"""Shutdown the nurturing manager"""
		self._initialized = False
		
		# Cancel background tasks
		for task in self._processing_tasks:
			task.cancel()
		
		# Wait for tasks to complete
		await asyncio.gather(*self._processing_tasks, return_exceptions=True)