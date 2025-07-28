"""
APG Customer Relationship Management - Workflow Automation Engine

Advanced workflow automation system for intelligent business process automation,
task management, approval workflows, and automated sales process optimization.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from enum import Enum
from uuid_extensions import uuid7str
import json

from pydantic import BaseModel, Field, validator

from .models import CRMContact, CRMLead, CRMOpportunity, Priority
from .database import DatabaseManager


logger = logging.getLogger(__name__)


class WorkflowStatus(str, Enum):
	"""Status of workflow instances"""
	ACTIVE = "active"
	INACTIVE = "inactive"
	DRAFT = "draft"
	ARCHIVED = "archived"


class TriggerType(str, Enum):
	"""Types of workflow triggers"""
	MANUAL = "manual"					# Manually triggered
	SCHEDULE = "schedule"				# Time-based trigger
	RECORD_CREATED = "record_created"	# When record is created
	RECORD_UPDATED = "record_updated"	# When record is updated
	FIELD_CHANGED = "field_changed"		# When specific field changes
	STAGE_CHANGED = "stage_changed"		# When opportunity stage changes
	EMAIL_RECEIVED = "email_received"	# When email is received
	FORM_SUBMITTED = "form_submitted"	# When web form is submitted
	SCORE_THRESHOLD = "score_threshold"	# When lead score reaches threshold
	TIME_ELAPSED = "time_elapsed"		# After time period elapses
	CUSTOM = "custom"					# Custom trigger condition


class ActionType(str, Enum):
	"""Types of workflow actions"""
	SEND_EMAIL = "send_email"
	CREATE_TASK = "create_task"
	UPDATE_FIELD = "update_field"
	ASSIGN_OWNER = "assign_owner"
	MOVE_STAGE = "move_stage"
	CREATE_RECORD = "create_record"
	SEND_NOTIFICATION = "send_notification"
	SCHEDULE_FOLLOWUP = "schedule_followup"
	ADD_TO_SEGMENT = "add_to_segment"
	SCORE_LEAD = "score_lead"
	WEBHOOK_CALL = "webhook_call"
	WAIT_DELAY = "wait_delay"
	BRANCH_CONDITION = "branch_condition"
	CUSTOM = "custom"


class ExecutionStatus(str, Enum):
	"""Status of workflow execution"""
	PENDING = "pending"
	RUNNING = "running"
	COMPLETED = "completed"
	FAILED = "failed"
	CANCELLED = "cancelled"
	PAUSED = "paused"


class ConditionOperator(str, Enum):
	"""Operators for condition evaluation"""
	EQUALS = "equals"
	NOT_EQUALS = "not_equals"
	CONTAINS = "contains"
	NOT_CONTAINS = "not_contains"
	GREATER_THAN = "greater_than"
	LESS_THAN = "less_than"
	GREATER_EQUAL = "greater_equal"
	LESS_EQUAL = "less_equal"
	IN = "in"
	NOT_IN = "not_in"
	IS_NULL = "is_null"
	IS_NOT_NULL = "is_not_null"
	REGEX_MATCH = "regex_match"
	CHANGED_FROM = "changed_from"
	CHANGED_TO = "changed_to"


class WorkflowCondition(BaseModel):
	"""Condition for workflow execution"""
	id: str = Field(default_factory=uuid7str)
	field: str = Field(..., description="Field to evaluate")
	operator: ConditionOperator = Field(..., description="Comparison operator")
	value: Union[str, int, float, bool, List[Any], None] = Field(None, description="Value to compare against")
	logical_operator: str = Field("AND", description="Logical operator (AND/OR)")


class WorkflowTrigger(BaseModel):
	"""Workflow trigger configuration"""
	id: str = Field(default_factory=uuid7str)
	trigger_type: TriggerType = Field(..., description="Type of trigger")
	conditions: List[WorkflowCondition] = Field(default_factory=list, description="Trigger conditions")
	
	# Schedule-specific settings
	schedule_expression: Optional[str] = Field(None, description="Cron expression for scheduled triggers")
	schedule_timezone: str = Field("UTC", description="Timezone for scheduled triggers")
	
	# Record-specific settings
	record_type: Optional[str] = Field(None, description="Type of record for record-based triggers")
	fields_to_watch: List[str] = Field(default_factory=list, description="Fields to monitor for changes")
	
	# Custom trigger settings
	custom_code: Optional[str] = Field(None, description="Custom trigger logic")
	
	# Metadata
	metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowAction(BaseModel):
	"""Workflow action configuration"""
	id: str = Field(default_factory=uuid7str)
	name: str = Field(..., description="Action name")
	action_type: ActionType = Field(..., description="Type of action")
	order: int = Field(..., description="Execution order")
	
	# Action parameters
	parameters: Dict[str, Any] = Field(default_factory=dict, description="Action-specific parameters")
	
	# Conditional execution
	conditions: List[WorkflowCondition] = Field(default_factory=list, description="Conditions for action execution")
	
	# Error handling
	retry_count: int = Field(0, description="Number of retries on failure")
	retry_delay_seconds: int = Field(60, description="Delay between retries")
	continue_on_error: bool = Field(False, description="Continue workflow if action fails")
	
	# Branching
	success_actions: List[str] = Field(default_factory=list, description="Actions to execute on success")
	failure_actions: List[str] = Field(default_factory=list, description="Actions to execute on failure")
	
	# Metadata
	metadata: Dict[str, Any] = Field(default_factory=dict)


class Workflow(BaseModel):
	"""Workflow definition"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	
	# Basic information
	name: str = Field(..., min_length=1, max_length=200)
	description: Optional[str] = Field(None, max_length=1000)
	status: WorkflowStatus = WorkflowStatus.DRAFT
	
	# Workflow structure
	triggers: List[WorkflowTrigger] = Field(default_factory=list)
	actions: List[WorkflowAction] = Field(default_factory=list)
	
	# Execution settings
	max_executions: Optional[int] = Field(None, description="Maximum number of executions")
	execution_timeout_minutes: int = Field(60, description="Execution timeout in minutes")
	concurrent_executions: bool = Field(False, description="Allow concurrent executions")
	
	# Targeting
	applies_to_records: List[str] = Field(default_factory=list, description="Record types this workflow applies to")
	target_conditions: List[WorkflowCondition] = Field(default_factory=list, description="Conditions for record targeting")
	
	# Performance tracking
	execution_count: int = Field(0, description="Total number of executions")
	success_count: int = Field(0, description="Number of successful executions")
	failure_count: int = Field(0, description="Number of failed executions")
	last_executed_at: Optional[datetime] = None
	average_execution_time_ms: float = Field(0.0, description="Average execution time")
	
	# Notifications
	notification_emails: List[str] = Field(default_factory=list)
	notify_on_failure: bool = Field(True, description="Send notifications on failure")
	notify_on_success: bool = Field(False, description="Send notifications on success")
	
	# Metadata
	metadata: Dict[str, Any] = Field(default_factory=dict)
	
	# Audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
	updated_by: str
	version: int = 1


class WorkflowExecution(BaseModel):
	"""Workflow execution instance"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	workflow_id: str
	
	# Execution context
	trigger_id: str
	triggered_by: str = Field("system", description="User or system that triggered execution")
	record_id: Optional[str] = Field(None, description="ID of record that triggered workflow")
	record_type: Optional[str] = Field(None, description="Type of record that triggered workflow")
	
	# Execution state
	status: ExecutionStatus = ExecutionStatus.PENDING
	current_action_id: Optional[str] = None
	completed_actions: List[str] = Field(default_factory=list)
	failed_actions: List[str] = Field(default_factory=list)
	
	# Context data
	context_data: Dict[str, Any] = Field(default_factory=dict, description="Execution context variables")
	input_data: Dict[str, Any] = Field(default_factory=dict, description="Input data for execution")
	output_data: Dict[str, Any] = Field(default_factory=dict, description="Output data from execution")
	
	# Timing
	started_at: Optional[datetime] = None
	completed_at: Optional[datetime] = None
	execution_duration_ms: Optional[int] = None
	
	# Error tracking
	error_message: Optional[str] = None
	error_details: Dict[str, Any] = Field(default_factory=dict)
	retry_count: int = Field(0, description="Number of retries attempted")
	
	# Audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class WorkflowAnalytics(BaseModel):
	"""Analytics for workflow performance"""
	workflow_id: str
	workflow_name: str
	tenant_id: str
	
	# Execution metrics
	total_executions: int = 0
	successful_executions: int = 0
	failed_executions: int = 0
	cancelled_executions: int = 0
	
	# Performance metrics
	success_rate: float = 0.0
	average_execution_time_ms: float = 0.0
	median_execution_time_ms: float = 0.0
	fastest_execution_ms: int = 0
	slowest_execution_ms: int = 0
	
	# Time-based metrics
	executions_today: int = 0
	executions_this_week: int = 0
	executions_this_month: int = 0
	
	# Error analysis
	common_errors: List[Dict[str, Any]] = Field(default_factory=list)
	error_rate_by_action: Dict[str, float] = Field(default_factory=dict)
	
	# Trigger analysis
	trigger_frequency: Dict[str, int] = Field(default_factory=dict)
	peak_execution_hours: List[int] = Field(default_factory=list)
	
	# Analysis metadata
	analyzed_at: datetime = Field(default_factory=datetime.utcnow)
	analysis_period_days: int = 30


class WorkflowAutomationEngine:
	"""
	Advanced workflow automation engine
	
	Provides comprehensive workflow creation, execution, monitoring,
	and optimization capabilities for sales process automation.
	"""
	
	def __init__(self, db_manager: DatabaseManager):
		"""
		Initialize workflow automation engine
		
		Args:
			db_manager: Database manager instance
		"""
		self.db_manager = db_manager
		self._initialized = False
		self._action_handlers: Dict[ActionType, Callable] = {}
		self._trigger_handlers: Dict[TriggerType, Callable] = {}
		self._running_executions: Dict[str, WorkflowExecution] = {}
	
	async def initialize(self):
		"""Initialize the workflow automation engine"""
		if self._initialized:
			return
		
		logger.info("🔧 Initializing Workflow Automation Engine...")
		
		# Ensure database connection
		if not self.db_manager._initialized:
			await self.db_manager.initialize()
		
		# Register default action handlers
		await self._register_default_handlers()
		
		# Start background scheduler
		asyncio.create_task(self._background_scheduler())
		
		self._initialized = True
		logger.info("✅ Workflow Automation Engine initialized successfully")
	
	async def create_workflow(
		self,
		workflow_data: Dict[str, Any],
		tenant_id: str,
		created_by: str
	) -> Workflow:
		"""
		Create a new workflow
		
		Args:
			workflow_data: Workflow configuration data
			tenant_id: Tenant identifier
			created_by: User creating the workflow
			
		Returns:
			Created workflow
		"""
		try:
			logger.info(f"🔄 Creating workflow: {workflow_data.get('name')}")
			
			# Add required fields
			workflow_data.update({
				'tenant_id': tenant_id,
				'created_by': created_by,
				'updated_by': created_by
			})
			
			# Create workflow object
			workflow = Workflow(**workflow_data)
			
			# Validate workflow structure
			await self._validate_workflow(workflow)
			
			# Store in database
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					INSERT INTO crm_workflows (
						id, tenant_id, name, description, status, triggers, actions,
						max_executions, execution_timeout_minutes, concurrent_executions,
						applies_to_records, target_conditions, execution_count, success_count,
						failure_count, last_executed_at, average_execution_time_ms,
						notification_emails, notify_on_failure, notify_on_success,
						metadata, created_at, updated_at, created_by, updated_by, version
					) VALUES (
						$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
						$15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26
					)
				""", 
				workflow.id, workflow.tenant_id, workflow.name, workflow.description, workflow.status.value,
				[trigger.model_dump() for trigger in workflow.triggers],
				[action.model_dump() for action in workflow.actions],
				workflow.max_executions, workflow.execution_timeout_minutes, workflow.concurrent_executions,
				workflow.applies_to_records, [condition.model_dump() for condition in workflow.target_conditions],
				workflow.execution_count, workflow.success_count, workflow.failure_count,
				workflow.last_executed_at, workflow.average_execution_time_ms,
				workflow.notification_emails, workflow.notify_on_failure, workflow.notify_on_success,
				workflow.metadata, workflow.created_at, workflow.updated_at,
				workflow.created_by, workflow.updated_by, workflow.version
				)
			
			logger.info(f"✅ Workflow created successfully: {workflow.id}")
			return workflow
			
		except Exception as e:
			logger.error(f"Failed to create workflow: {str(e)}", exc_info=True)
			raise
	
	async def execute_workflow(
		self,
		workflow_id: str,
		tenant_id: str,
		triggered_by: str = "system",
		record_id: Optional[str] = None,
		record_type: Optional[str] = None,
		input_data: Dict[str, Any] = None
	) -> WorkflowExecution:
		"""
		Execute a workflow
		
		Args:
			workflow_id: Workflow identifier
			tenant_id: Tenant identifier
			triggered_by: User or system triggering execution
			record_id: ID of record triggering workflow
			record_type: Type of record triggering workflow
			input_data: Input data for execution
			
		Returns:
			Workflow execution instance
		"""
		try:
			logger.info(f"🚀 Executing workflow: {workflow_id}")
			
			# Get workflow
			workflow = await self.get_workflow(workflow_id, tenant_id)
			if not workflow:
				raise ValueError(f"Workflow not found: {workflow_id}")
			
			if workflow.status != WorkflowStatus.ACTIVE:
				raise ValueError(f"Workflow is not active: {workflow_id}")
			
			# Check execution limits
			if workflow.max_executions and workflow.execution_count >= workflow.max_executions:
				raise ValueError(f"Workflow has reached maximum executions: {workflow.max_executions}")
			
			# Check concurrent executions
			if not workflow.concurrent_executions and workflow_id in self._running_executions:
				raise ValueError(f"Workflow is already running and concurrent executions are disabled")
			
			# Create execution instance
			execution = WorkflowExecution(
				tenant_id=tenant_id,
				workflow_id=workflow_id,
				trigger_id=workflow.triggers[0].id if workflow.triggers else uuid7str(),
				triggered_by=triggered_by,
				record_id=record_id,
				record_type=record_type,
				input_data=input_data or {},
				status=ExecutionStatus.RUNNING,
				started_at=datetime.utcnow()
			)
			
			# Store execution in database
			await self._store_execution(execution)
			
			# Add to running executions
			self._running_executions[execution.id] = execution
			
			# Execute workflow asynchronously
			asyncio.create_task(self._execute_workflow_async(workflow, execution))
			
			logger.info(f"✅ Workflow execution started: {execution.id}")
			return execution
			
		except Exception as e:
			logger.error(f"Failed to execute workflow: {str(e)}", exc_info=True)
			raise
	
	async def get_workflow_analytics(
		self,
		workflow_id: str,
		tenant_id: str,
		period_days: int = 30
	) -> WorkflowAnalytics:
		"""
		Get comprehensive workflow analytics
		
		Args:
			workflow_id: Workflow identifier
			tenant_id: Tenant identifier
			period_days: Analysis period in days
			
		Returns:
			Workflow analytics data
		"""
		try:
			logger.info(f"📊 Generating workflow analytics: {workflow_id}")
			
			# Get workflow
			workflow = await self.get_workflow(workflow_id, tenant_id)
			if not workflow:
				raise ValueError(f"Workflow not found: {workflow_id}")
			
			analytics = WorkflowAnalytics(
				workflow_id=workflow_id,
				workflow_name=workflow.name,
				tenant_id=tenant_id,
				analysis_period_days=period_days
			)
			
			async with self.db_manager.get_connection() as conn:
				# Basic execution statistics
				stats_row = await conn.fetchrow("""
					SELECT 
						COUNT(*) as total_executions,
						COUNT(*) FILTER (WHERE status = 'completed') as successful_executions,
						COUNT(*) FILTER (WHERE status = 'failed') as failed_executions,
						COUNT(*) FILTER (WHERE status = 'cancelled') as cancelled_executions,
						AVG(execution_duration_ms) as avg_duration,
						PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY execution_duration_ms) as median_duration,
						MIN(execution_duration_ms) as min_duration,
						MAX(execution_duration_ms) as max_duration
					FROM crm_workflow_executions 
					WHERE workflow_id = $1 AND tenant_id = $2
					AND created_at >= NOW() - INTERVAL '%s days'
				""", workflow_id, tenant_id, period_days)
				
				if stats_row:
					analytics.total_executions = stats_row['total_executions'] or 0
					analytics.successful_executions = stats_row['successful_executions'] or 0
					analytics.failed_executions = stats_row['failed_executions'] or 0
					analytics.cancelled_executions = stats_row['cancelled_executions'] or 0
					analytics.average_execution_time_ms = float(stats_row['avg_duration'] or 0)
					analytics.median_execution_time_ms = float(stats_row['median_duration'] or 0)
					analytics.fastest_execution_ms = int(stats_row['min_duration'] or 0)
					analytics.slowest_execution_ms = int(stats_row['max_duration'] or 0)
				
				# Calculate success rate
				if analytics.total_executions > 0:
					analytics.success_rate = (analytics.successful_executions / analytics.total_executions) * 100
				
				# Time-based metrics
				time_metrics = await conn.fetchrow("""
					SELECT 
						COUNT(*) FILTER (WHERE created_at >= CURRENT_DATE) as today,
						COUNT(*) FILTER (WHERE created_at >= DATE_TRUNC('week', NOW())) as this_week,
						COUNT(*) FILTER (WHERE created_at >= DATE_TRUNC('month', NOW())) as this_month
					FROM crm_workflow_executions 
					WHERE workflow_id = $1 AND tenant_id = $2
				""", workflow_id, tenant_id)
				
				if time_metrics:
					analytics.executions_today = time_metrics['today'] or 0
					analytics.executions_this_week = time_metrics['this_week'] or 0
					analytics.executions_this_month = time_metrics['this_month'] or 0
				
				# Common errors
				error_rows = await conn.fetch("""
					SELECT 
						error_message,
						COUNT(*) as error_count
					FROM crm_workflow_executions 
					WHERE workflow_id = $1 AND tenant_id = $2
					AND status = 'failed'
					AND error_message IS NOT NULL
					AND created_at >= NOW() - INTERVAL '%s days'
					GROUP BY error_message
					ORDER BY error_count DESC
					LIMIT 10
				""", workflow_id, tenant_id, period_days)
				
				analytics.common_errors = [
					{"error": row['error_message'], "count": row['error_count']}
					for row in error_rows
				]
			
			logger.info(f"✅ Generated analytics for {analytics.total_executions} executions")
			return analytics
			
		except Exception as e:
			logger.error(f"Failed to generate workflow analytics: {str(e)}", exc_info=True)
			raise
	
	async def get_workflow(
		self,
		workflow_id: str,
		tenant_id: str
	) -> Optional[Workflow]:
		"""
		Get workflow by ID
		
		Args:
			workflow_id: Workflow identifier
			tenant_id: Tenant identifier
			
		Returns:
			Workflow if found
		"""
		try:
			async with self.db_manager.get_connection() as conn:
				row = await conn.fetchrow("""
					SELECT * FROM crm_workflows 
					WHERE id = $1 AND tenant_id = $2
				""", workflow_id, tenant_id)
				
				if not row:
					return None
				
				# Convert row to dict and handle nested objects
				workflow_dict = dict(row)
				
				# Parse triggers and actions from JSON
				if workflow_dict['triggers']:
					workflow_dict['triggers'] = [WorkflowTrigger(**trigger) for trigger in workflow_dict['triggers']]
				if workflow_dict['actions']:
					workflow_dict['actions'] = [WorkflowAction(**action) for action in workflow_dict['actions']]
				if workflow_dict['target_conditions']:
					workflow_dict['target_conditions'] = [WorkflowCondition(**condition) for condition in workflow_dict['target_conditions']]
				
				return Workflow(**workflow_dict)
				
		except Exception as e:
			logger.error(f"Failed to get workflow: {str(e)}", exc_info=True)
			raise
	
	async def _execute_workflow_async(
		self,
		workflow: Workflow,
		execution: WorkflowExecution
	):
		"""Execute workflow asynchronously"""
		try:
			# Execute actions in order
			for action in sorted(workflow.actions, key=lambda a: a.order):
				# Check if action should be executed based on conditions
				if not await self._should_execute_action(action, execution):
					continue
				
				# Update current action
				execution.current_action_id = action.id
				await self._update_execution(execution)
				
				# Execute action
				try:
					await self._execute_action(action, execution)
					execution.completed_actions.append(action.id)
				except Exception as e:
					logger.error(f"Action failed: {action.id} - {str(e)}")
					execution.failed_actions.append(action.id)
					
					if not action.continue_on_error:
						raise
			
			# Mark execution as completed
			execution.status = ExecutionStatus.COMPLETED
			execution.completed_at = datetime.utcnow()
			execution.execution_duration_ms = int(
				(execution.completed_at - execution.started_at).total_seconds() * 1000
			)
			
			await self._update_execution_statistics(workflow.id, execution.tenant_id, True, execution.execution_duration_ms)
			
		except Exception as e:
			# Mark execution as failed
			execution.status = ExecutionStatus.FAILED
			execution.error_message = str(e)
			execution.completed_at = datetime.utcnow()
			
			if execution.started_at:
				execution.execution_duration_ms = int(
					(execution.completed_at - execution.started_at).total_seconds() * 1000
				)
			
			await self._update_execution_statistics(workflow.id, execution.tenant_id, False, execution.execution_duration_ms)
			
			logger.error(f"Workflow execution failed: {execution.id} - {str(e)}")
		
		finally:
			# Update execution in database
			await self._update_execution(execution)
			
			# Remove from running executions
			if execution.id in self._running_executions:
				del self._running_executions[execution.id]
	
	async def _register_default_handlers(self):
		"""Register default action and trigger handlers"""
		# Action handlers
		self._action_handlers[ActionType.SEND_EMAIL] = self._handle_send_email
		self._action_handlers[ActionType.CREATE_TASK] = self._handle_create_task
		self._action_handlers[ActionType.UPDATE_FIELD] = self._handle_update_field
		self._action_handlers[ActionType.ASSIGN_OWNER] = self._handle_assign_owner
		self._action_handlers[ActionType.MOVE_STAGE] = self._handle_move_stage
		self._action_handlers[ActionType.CREATE_RECORD] = self._handle_create_record
		self._action_handlers[ActionType.SEND_NOTIFICATION] = self._handle_send_notification
		self._action_handlers[ActionType.SCHEDULE_FOLLOWUP] = self._handle_schedule_followup
		self._action_handlers[ActionType.WAIT_DELAY] = self._handle_wait_delay
		
		# Trigger handlers
		self._trigger_handlers[TriggerType.SCHEDULE] = self._handle_schedule_trigger
		self._trigger_handlers[TriggerType.RECORD_CREATED] = self._handle_record_created_trigger
		self._trigger_handlers[TriggerType.RECORD_UPDATED] = self._handle_record_updated_trigger
		self._trigger_handlers[TriggerType.FIELD_CHANGED] = self._handle_field_changed_trigger
	
	async def _validate_workflow(self, workflow: Workflow):
		"""Validate workflow structure"""
		if not workflow.triggers:
			raise ValueError("Workflow must have at least one trigger")
		
		if not workflow.actions:
			raise ValueError("Workflow must have at least one action")
		
		# Validate action order
		orders = [action.order for action in workflow.actions]
		if len(set(orders)) != len(orders):
			raise ValueError("Action orders must be unique")
		
		# Validate trigger conditions
		for trigger in workflow.triggers:
			if trigger.trigger_type == TriggerType.SCHEDULE and not trigger.schedule_expression:
				raise ValueError("Schedule triggers must have a schedule expression")
	
	async def _should_execute_action(
		self,
		action: WorkflowAction,
		execution: WorkflowExecution
	) -> bool:
		"""Check if action should be executed based on conditions"""
		if not action.conditions:
			return True
		
		for condition in action.conditions:
			if not await self._evaluate_condition(condition, execution):
				return False
		
		return True
	
	async def _evaluate_condition(
		self,
		condition: WorkflowCondition,
		execution: WorkflowExecution
	) -> bool:
		"""Evaluate a single condition"""
		# Get field value from context or record
		field_value = execution.context_data.get(condition.field)
		
		# Apply operator
		if condition.operator == ConditionOperator.EQUALS:
			return field_value == condition.value
		elif condition.operator == ConditionOperator.NOT_EQUALS:
			return field_value != condition.value
		elif condition.operator == ConditionOperator.CONTAINS:
			return condition.value in str(field_value or '')
		elif condition.operator == ConditionOperator.GREATER_THAN:
			return (field_value or 0) > condition.value
		elif condition.operator == ConditionOperator.LESS_THAN:
			return (field_value or 0) < condition.value
		elif condition.operator == ConditionOperator.IS_NULL:
			return field_value is None
		elif condition.operator == ConditionOperator.IS_NOT_NULL:
			return field_value is not None
		
		return True
	
	async def _execute_action(
		self,
		action: WorkflowAction,
		execution: WorkflowExecution
	):
		"""Execute a single action"""
		handler = self._action_handlers.get(action.action_type)
		if not handler:
			raise ValueError(f"No handler for action type: {action.action_type}")
		
		await handler(action, execution)
	
	# Action handlers
	async def _handle_send_email(self, action: WorkflowAction, execution: WorkflowExecution):
		"""Handle send email action"""
		logger.info(f"📧 Sending email: {action.parameters.get('subject', 'No subject')}")
		# Implementation would integrate with email service
	
	async def _handle_create_task(self, action: WorkflowAction, execution: WorkflowExecution):
		"""Handle create task action"""
		logger.info(f"📋 Creating task: {action.parameters.get('title', 'Untitled task')}")
		# Implementation would create task record
	
	async def _handle_update_field(self, action: WorkflowAction, execution: WorkflowExecution):
		"""Handle update field action"""
		field = action.parameters.get('field')
		value = action.parameters.get('value')
		logger.info(f"📝 Updating field {field} to {value}")
		# Implementation would update record field
	
	async def _handle_assign_owner(self, action: WorkflowAction, execution: WorkflowExecution):
		"""Handle assign owner action"""
		owner = action.parameters.get('owner')
		logger.info(f"👤 Assigning owner: {owner}")
		# Implementation would assign record owner
	
	async def _handle_move_stage(self, action: WorkflowAction, execution: WorkflowExecution):
		"""Handle move stage action"""
		stage_id = action.parameters.get('stage_id')
		logger.info(f"🔄 Moving to stage: {stage_id}")
		# Implementation would move opportunity to stage
	
	async def _handle_create_record(self, action: WorkflowAction, execution: WorkflowExecution):
		"""Handle create record action"""
		record_type = action.parameters.get('record_type')
		logger.info(f"📄 Creating record: {record_type}")
		# Implementation would create new record
	
	async def _handle_send_notification(self, action: WorkflowAction, execution: WorkflowExecution):
		"""Handle send notification action"""
		message = action.parameters.get('message')
		logger.info(f"🔔 Sending notification: {message}")
		# Implementation would send notification
	
	async def _handle_schedule_followup(self, action: WorkflowAction, execution: WorkflowExecution):
		"""Handle schedule followup action"""
		delay_days = action.parameters.get('delay_days', 1)
		logger.info(f"⏰ Scheduling followup in {delay_days} days")
		# Implementation would schedule followup task
	
	async def _handle_wait_delay(self, action: WorkflowAction, execution: WorkflowExecution):
		"""Handle wait delay action"""
		delay_seconds = action.parameters.get('delay_seconds', 60)
		logger.info(f"⏸️ Waiting {delay_seconds} seconds")
		await asyncio.sleep(delay_seconds)
	
	# Trigger handlers
	async def _handle_schedule_trigger(self, trigger: WorkflowTrigger):
		"""Handle scheduled trigger"""
		# Implementation would parse cron expression and schedule execution
		pass
	
	async def _handle_record_created_trigger(self, trigger: WorkflowTrigger, record_data: Dict[str, Any]):
		"""Handle record created trigger"""
		# Implementation would check if record matches trigger conditions
		pass
	
	async def _handle_record_updated_trigger(self, trigger: WorkflowTrigger, record_data: Dict[str, Any]):
		"""Handle record updated trigger"""
		# Implementation would check if record update matches trigger conditions
		pass
	
	async def _handle_field_changed_trigger(self, trigger: WorkflowTrigger, old_data: Dict[str, Any], new_data: Dict[str, Any]):
		"""Handle field changed trigger"""
		# Implementation would check if specific field changes match trigger conditions
		pass
	
	async def _background_scheduler(self):
		"""Background task for scheduled workflow execution"""
		while True:
			try:
				# Check for scheduled workflows to execute
				await asyncio.sleep(60)  # Check every minute
				# Implementation would check for scheduled triggers
			except Exception as e:
				logger.error(f"Background scheduler error: {str(e)}")
				await asyncio.sleep(60)
	
	async def _store_execution(self, execution: WorkflowExecution):
		"""Store workflow execution in database"""
		try:
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					INSERT INTO crm_workflow_executions (
						id, tenant_id, workflow_id, trigger_id, triggered_by, record_id, record_type,
						status, current_action_id, completed_actions, failed_actions,
						context_data, input_data, output_data, started_at, completed_at,
						execution_duration_ms, error_message, error_details, retry_count,
						created_at, updated_at
					) VALUES (
						$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
						$15, $16, $17, $18, $19, $20, $21, $22
					)
				""", 
				execution.id, execution.tenant_id, execution.workflow_id, execution.trigger_id,
				execution.triggered_by, execution.record_id, execution.record_type,
				execution.status.value, execution.current_action_id, execution.completed_actions,
				execution.failed_actions, execution.context_data, execution.input_data,
				execution.output_data, execution.started_at, execution.completed_at,
				execution.execution_duration_ms, execution.error_message, execution.error_details,
				execution.retry_count, execution.created_at, execution.updated_at
				)
		except Exception as e:
			logger.error(f"Failed to store workflow execution: {str(e)}")
			raise
	
	async def _update_execution(self, execution: WorkflowExecution):
		"""Update workflow execution in database"""
		try:
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					UPDATE crm_workflow_executions SET
						status = $3, current_action_id = $4, completed_actions = $5,
						failed_actions = $6, context_data = $7, output_data = $8,
						completed_at = $9, execution_duration_ms = $10, error_message = $11,
						error_details = $12, retry_count = $13, updated_at = NOW()
					WHERE id = $1 AND tenant_id = $2
				""", 
				execution.id, execution.tenant_id, execution.status.value,
				execution.current_action_id, execution.completed_actions, execution.failed_actions,
				execution.context_data, execution.output_data, execution.completed_at,
				execution.execution_duration_ms, execution.error_message, execution.error_details,
				execution.retry_count
				)
		except Exception as e:
			logger.error(f"Failed to update workflow execution: {str(e)}")
	
	async def _update_execution_statistics(
		self,
		workflow_id: str,
		tenant_id: str,
		success: bool,
		duration_ms: Optional[int]
	):
		"""Update workflow execution statistics"""
		try:
			async with self.db_manager.get_connection() as conn:
				if success:
					await conn.execute("""
						UPDATE crm_workflows SET
							execution_count = execution_count + 1,
							success_count = success_count + 1,
							last_executed_at = NOW(),
							average_execution_time_ms = 
								CASE 
									WHEN execution_count = 0 THEN $3
									ELSE (average_execution_time_ms * execution_count + $3) / (execution_count + 1)
								END,
							updated_at = NOW()
						WHERE id = $1 AND tenant_id = $2
					""", workflow_id, tenant_id, duration_ms or 0)
				else:
					await conn.execute("""
						UPDATE crm_workflows SET
							execution_count = execution_count + 1,
							failure_count = failure_count + 1,
							last_executed_at = NOW(),
							updated_at = NOW()
						WHERE id = $1 AND tenant_id = $2
					""", workflow_id, tenant_id)
		except Exception as e:
			logger.error(f"Failed to update execution statistics: {str(e)}")