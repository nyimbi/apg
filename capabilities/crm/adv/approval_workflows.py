"""
APG Customer Relationship Management - Approval Workflows Module

Advanced approval workflow system with multi-step approval processes, intelligent routing,
escalation management, and comprehensive audit trails for enterprise-grade governance.

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


class ApprovalType(str, Enum):
	"""Types of approval workflows"""
	OPPORTUNITY_DISCOUNT = "opportunity_discount"
	OPPORTUNITY_CLOSURE = "opportunity_closure"
	CONTRACT_APPROVAL = "contract_approval"
	PRICING_APPROVAL = "pricing_approval"
	REFUND_APPROVAL = "refund_approval"
	CAMPAIGN_APPROVAL = "campaign_approval"
	BUDGET_APPROVAL = "budget_approval"
	EXPENSE_APPROVAL = "expense_approval"
	USER_ACCESS = "user_access"
	DATA_EXPORT = "data_export"
	CUSTOM = "custom"


class ApprovalStatus(str, Enum):
	"""Approval request status"""
	PENDING = "pending"
	IN_REVIEW = "in_review"
	APPROVED = "approved"
	REJECTED = "rejected"
	CANCELLED = "cancelled"
	EXPIRED = "expired"
	ESCALATED = "escalated"


class ApprovalStepStatus(str, Enum):
	"""Individual approval step status"""
	PENDING = "pending"
	APPROVED = "approved"
	REJECTED = "rejected"
	SKIPPED = "skipped"
	ESCALATED = "escalated"


class ApprovalAction(str, Enum):
	"""Available approval actions"""
	APPROVE = "approve"
	REJECT = "reject"
	REQUEST_CHANGES = "request_changes"
	DELEGATE = "delegate"
	ESCALATE = "escalate"
	CANCEL = "cancel"


class EscalationTrigger(str, Enum):
	"""Escalation triggers"""
	TIMEOUT = "timeout"
	REJECTION = "rejection"
	MANUAL = "manual"
	NO_RESPONSE = "no_response"
	THRESHOLD_EXCEEDED = "threshold_exceeded"


class ApprovalWorkflowTemplate(BaseModel):
	"""Approval workflow template configuration"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	
	# Template details
	name: str = Field(..., min_length=1, max_length=200)
	description: Optional[str] = Field(None, max_length=1000)
	approval_type: ApprovalType
	category: str = Field("general", description="Workflow category")
	
	# Workflow configuration
	is_active: bool = Field(True, description="Whether template is active")
	is_sequential: bool = Field(True, description="Sequential vs parallel approval")
	require_all_approvals: bool = Field(True, description="Require all approvers to approve")
	
	# Approval steps configuration
	approval_steps: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Trigger conditions
	trigger_conditions: Dict[str, Any] = Field(default_factory=dict)
	auto_trigger: bool = Field(False, description="Automatically trigger on conditions")
	
	# Timing settings
	default_timeout_hours: int = Field(72, description="Default approval timeout")
	reminder_intervals: List[int] = Field(default_factory=lambda: [24, 48], description="Reminder hours")
	escalation_timeout_hours: int = Field(168, description="Escalation timeout (1 week)")
	
	# Escalation settings
	enable_escalation: bool = Field(True, description="Enable escalation")
	escalation_approvers: List[str] = Field(default_factory=list)
	escalation_conditions: Dict[str, Any] = Field(default_factory=dict)
	
	# Notification settings
	notify_requester: bool = Field(True, description="Notify requester of status changes")
	notify_stakeholders: bool = Field(False, description="Notify stakeholders")
	notification_templates: Dict[str, str] = Field(default_factory=dict)
	
	# Integration settings
	webhook_url: Optional[str] = Field(None, description="Webhook for status updates")
	external_system_integration: Dict[str, Any] = Field(default_factory=dict)
	
	# Usage tracking
	usage_count: int = Field(0, description="Times template used")
	success_rate: float = Field(0.0, description="Approval success rate")
	average_approval_time: float = Field(0.0, description="Average approval time in hours")
	
	# Metadata
	metadata: Dict[str, Any] = Field(default_factory=dict)
	
	# Audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
	updated_by: str
	version: int = 1


class ApprovalRequest(BaseModel):
	"""Approval request record"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	
	# Request details
	title: str = Field(..., min_length=1, max_length=500)
	description: Optional[str] = Field(None, max_length=5000)
	approval_type: ApprovalType
	priority: str = Field("normal", description="Request priority")
	
	# Workflow details
	workflow_template_id: Optional[str] = Field(None, description="Source template")
	is_sequential: bool = Field(True, description="Sequential approval")
	require_all_approvals: bool = Field(True, description="Require all approvals")
	
	# Request context
	requested_by: str = Field(..., description="User requesting approval")
	requested_for: Optional[str] = Field(None, description="User request is for")
	business_justification: Optional[str] = Field(None, description="Business justification")
	
	# Related records
	related_record_id: Optional[str] = Field(None, description="Related record ID")
	related_record_type: Optional[str] = Field(None, description="Related record type")
	contact_id: Optional[str] = Field(None, description="Associated contact")
	account_id: Optional[str] = Field(None, description="Associated account")
	opportunity_id: Optional[str] = Field(None, description="Associated opportunity")
	
	# Request data
	approval_data: Dict[str, Any] = Field(default_factory=dict, description="Data being approved")
	original_values: Dict[str, Any] = Field(default_factory=dict, description="Original values")
	requested_changes: Dict[str, Any] = Field(default_factory=dict, description="Requested changes")
	
	# Status and timing
	status: ApprovalStatus = ApprovalStatus.PENDING
	submitted_at: datetime = Field(default_factory=datetime.utcnow)
	approved_at: Optional[datetime] = Field(None, description="Final approval time")
	rejected_at: Optional[datetime] = Field(None, description="Rejection time")
	expires_at: Optional[datetime] = Field(None, description="Expiration time")
	
	# Approval tracking
	current_step: int = Field(0, description="Current approval step")
	total_steps: int = Field(0, description="Total approval steps")
	completed_steps: int = Field(0, description="Completed steps")
	
	# Final outcome
	final_approver: Optional[str] = Field(None, description="Final approver")
	rejection_reason: Optional[str] = Field(None, description="Rejection reason")
	approval_notes: Optional[str] = Field(None, description="Final approval notes")
	
	# Escalation tracking
	escalated: bool = Field(False, description="Whether request was escalated")
	escalated_at: Optional[datetime] = Field(None, description="Escalation time")
	escalated_by: Optional[str] = Field(None, description="User who escalated")
	escalation_reason: Optional[str] = Field(None, description="Escalation reason")
	
	# Attachments and documents
	attachments: List[Dict[str, Any]] = Field(default_factory=list)
	supporting_documents: List[str] = Field(default_factory=list)
	
	# Metadata
	metadata: Dict[str, Any] = Field(default_factory=dict)
	
	# Audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
	updated_by: str
	version: int = 1


class ApprovalStep(BaseModel):
	"""Individual approval step in workflow"""
	id: str = Field(default_factory=uuid7str)
	approval_request_id: str
	tenant_id: str
	
	# Step details
	step_number: int = Field(..., description="Step sequence number")
	step_name: str = Field(..., description="Step name")
	step_description: Optional[str] = Field(None, description="Step description")
	
	# Approver details
	approver_id: str = Field(..., description="Assigned approver")
	approver_name: str = Field(..., description="Approver display name")
	approver_email: str = Field(..., description="Approver email")
	approver_role: Optional[str] = Field(None, description="Approver role/title")
	
	# Approval settings
	is_required: bool = Field(True, description="Required approval step")
	can_delegate: bool = Field(True, description="Can delegate to others")
	timeout_hours: int = Field(72, description="Step timeout hours")
	
	# Status and timing
	status: ApprovalStepStatus = ApprovalStepStatus.PENDING
	assigned_at: datetime = Field(default_factory=datetime.utcnow)
	responded_at: Optional[datetime] = Field(None, description="Response time")
	completed_at: Optional[datetime] = Field(None, description="Completion time")
	
	# Response details
	action_taken: Optional[ApprovalAction] = Field(None, description="Action taken")
	response_notes: Optional[str] = Field(None, description="Approver notes")
	rejection_reason: Optional[str] = Field(None, description="Rejection reason")
	
	# Delegation details
	delegated_to: Optional[str] = Field(None, description="Delegated to user")
	delegated_at: Optional[datetime] = Field(None, description="Delegation time")
	delegation_reason: Optional[str] = Field(None, description="Delegation reason")
	
	# Escalation details
	escalated: bool = Field(False, description="Step was escalated")
	escalated_at: Optional[datetime] = Field(None, description="Escalation time")
	escalation_trigger: Optional[EscalationTrigger] = Field(None, description="Escalation trigger")
	
	# Reminders
	reminders_sent: int = Field(0, description="Number of reminders sent")
	last_reminder_at: Optional[datetime] = Field(None, description="Last reminder time")
	
	# Metadata
	metadata: Dict[str, Any] = Field(default_factory=dict)
	
	# Audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class ApprovalHistory(BaseModel):
	"""Approval workflow history and audit trail"""
	id: str = Field(default_factory=uuid7str)
	approval_request_id: str
	tenant_id: str
	
	# Event details
	event_type: str = Field(..., description="Type of event")
	event_description: str = Field(..., description="Event description")
	
	# Actor details
	actor_id: str = Field(..., description="User who performed action")
	actor_name: str = Field(..., description="Actor display name")
	actor_role: Optional[str] = Field(None, description="Actor role")
	
	# Event data
	old_status: Optional[str] = Field(None, description="Previous status")
	new_status: Optional[str] = Field(None, description="New status")
	action_data: Dict[str, Any] = Field(default_factory=dict)
	
	# System details
	ip_address: Optional[str] = Field(None, description="IP address")
	user_agent: Optional[str] = Field(None, description="User agent")
	system_info: Dict[str, Any] = Field(default_factory=dict)
	
	# Metadata
	metadata: Dict[str, Any] = Field(default_factory=dict)
	
	# Audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)


class ApprovalAnalytics(BaseModel):
	"""Approval workflow analytics"""
	tenant_id: str
	analysis_period_start: datetime
	analysis_period_end: datetime
	
	# Overall metrics
	total_requests: int = 0
	approved_requests: int = 0
	rejected_requests: int = 0
	pending_requests: int = 0
	expired_requests: int = 0
	
	# Performance metrics
	approval_rate: float = 0.0
	rejection_rate: float = 0.0
	average_approval_time: float = 0.0
	median_approval_time: float = 0.0
	
	# Efficiency metrics
	requests_by_type: Dict[str, int] = Field(default_factory=dict)
	approver_performance: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
	bottleneck_analysis: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Escalation metrics
	escalated_requests: int = 0
	escalation_rate: float = 0.0
	escalation_reasons: Dict[str, int] = Field(default_factory=dict)
	
	# Time analysis
	approval_times_by_type: Dict[str, float] = Field(default_factory=dict)
	fastest_approvals: List[Dict[str, Any]] = Field(default_factory=list)
	slowest_approvals: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Workflow effectiveness
	template_performance: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
	step_completion_rates: Dict[int, float] = Field(default_factory=dict)
	
	# Trends
	approval_volume_trend: List[Dict[str, Any]] = Field(default_factory=list)
	efficiency_trend: float = 0.0
	
	# Analysis metadata
	analyzed_at: datetime = Field(default_factory=datetime.utcnow)
	analysis_version: str = "1.0"


class ApprovalWorkflowEngine:
	"""
	Advanced approval workflow management system
	
	Provides comprehensive approval workflow management with multi-step processes,
	intelligent routing, escalation management, and detailed audit trails.
	"""
	
	def __init__(self, db_manager: DatabaseManager):
		"""
		Initialize approval workflow engine
		
		Args:
			db_manager: Database manager instance
		"""
		self.db_manager = db_manager
		self._initialized = False
		self._active_templates = {}
		self._approval_queue = asyncio.Queue()
		self._processing_tasks = []
	
	async def initialize(self):
		"""Initialize the approval workflow engine"""
		if self._initialized:
			return
		
		logger.info("🔧 Initializing Approval Workflow Engine...")
		
		# Ensure database connection
		if not self.db_manager._initialized:
			await self.db_manager.initialize()
		
		# Start background processing
		self._processing_tasks = [
			asyncio.create_task(self._process_approval_queue()),
			asyncio.create_task(self._process_timeouts_and_escalations()),
			asyncio.create_task(self._send_reminder_notifications())
		]
		
		self._initialized = True
		logger.info("✅ Approval Workflow Engine initialized successfully")
	
	async def create_approval_template(
		self,
		template_data: Dict[str, Any],
		tenant_id: str,
		created_by: str
	) -> ApprovalWorkflowTemplate:
		"""
		Create a new approval workflow template
		
		Args:
			template_data: Template configuration data
			tenant_id: Tenant identifier
			created_by: User creating the template
			
		Returns:
			Created workflow template
		"""
		try:
			logger.info(f"📋 Creating approval workflow template: {template_data.get('name')}")
			
			# Add required fields
			template_data.update({
				'tenant_id': tenant_id,
				'created_by': created_by,
				'updated_by': created_by
			})
			
			# Create template object
			template = ApprovalWorkflowTemplate(**template_data)
			
			# Store template in database
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					INSERT INTO crm_approval_workflow_templates (
						id, tenant_id, name, description, approval_type, category,
						is_active, is_sequential, require_all_approvals, approval_steps,
						trigger_conditions, auto_trigger, default_timeout_hours, reminder_intervals, escalation_timeout_hours,
						enable_escalation, escalation_approvers, escalation_conditions,
						notify_requester, notify_stakeholders, notification_templates,
						webhook_url, external_system_integration,
						usage_count, success_rate, average_approval_time,
						metadata, created_at, updated_at, created_by, updated_by, version
					) VALUES (
						$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
						$16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32
					)
				""", 
				template.id, template.tenant_id, template.name, template.description, 
				template.approval_type.value, template.category, template.is_active, 
				template.is_sequential, template.require_all_approvals, template.approval_steps,
				template.trigger_conditions, template.auto_trigger, template.default_timeout_hours, 
				template.reminder_intervals, template.escalation_timeout_hours,
				template.enable_escalation, template.escalation_approvers, template.escalation_conditions,
				template.notify_requester, template.notify_stakeholders, template.notification_templates,
				template.webhook_url, template.external_system_integration,
				template.usage_count, template.success_rate, template.average_approval_time,
				template.metadata, template.created_at, template.updated_at, 
				template.created_by, template.updated_by, template.version
				)
			
			# Cache active template
			if template.is_active:
				self._active_templates[template.id] = template
			
			logger.info(f"✅ Approval workflow template created successfully: {template.id}")
			return template
			
		except Exception as e:
			logger.error(f"Failed to create approval workflow template: {str(e)}", exc_info=True)
			raise
	
	async def submit_approval_request(
		self,
		request_data: Dict[str, Any],
		tenant_id: str,
		requested_by: str
	) -> ApprovalRequest:
		"""
		Submit a new approval request
		
		Args:
			request_data: Approval request data
			tenant_id: Tenant identifier
			requested_by: User submitting the request
			
		Returns:
			Created approval request
		"""
		try:
			logger.info(f"📝 Submitting approval request: {request_data.get('title')}")
			
			# Add required fields
			request_data.update({
				'tenant_id': tenant_id,
				'requested_by': requested_by,
				'created_by': requested_by,
				'updated_by': requested_by
			})
			
			# Set expiration if not provided
			if not request_data.get('expires_at') and request_data.get('workflow_template_id'):
				template = await self._get_workflow_template(request_data['workflow_template_id'], tenant_id)
				if template:
					request_data['expires_at'] = datetime.utcnow() + timedelta(hours=template.default_timeout_hours)
			
			# Create request object
			request = ApprovalRequest(**request_data)
			
			# Store request in database
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					INSERT INTO crm_approval_requests (
						id, tenant_id, title, description, approval_type, priority,
						workflow_template_id, is_sequential, require_all_approvals,
						requested_by, requested_for, business_justification,
						related_record_id, related_record_type, contact_id, account_id, opportunity_id,
						approval_data, original_values, requested_changes,
						status, submitted_at, approved_at, rejected_at, expires_at,
						current_step, total_steps, completed_steps,
						final_approver, rejection_reason, approval_notes,
						escalated, escalated_at, escalated_by, escalation_reason,
						attachments, supporting_documents,
						metadata, created_at, updated_at, created_by, updated_by, version
					) VALUES (
						$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17,
						$18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32,
						$33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44
					)
				""", 
				request.id, request.tenant_id, request.title, request.description, 
				request.approval_type.value, request.priority, request.workflow_template_id,
				request.is_sequential, request.require_all_approvals, request.requested_by,
				request.requested_for, request.business_justification, request.related_record_id,
				request.related_record_type, request.contact_id, request.account_id, request.opportunity_id,
				request.approval_data, request.original_values, request.requested_changes,
				request.status.value, request.submitted_at, request.approved_at, request.rejected_at, request.expires_at,
				request.current_step, request.total_steps, request.completed_steps,
				request.final_approver, request.rejection_reason, request.approval_notes,
				request.escalated, request.escalated_at, request.escalated_by, request.escalation_reason,
				request.attachments, request.supporting_documents,
				request.metadata, request.created_at, request.updated_at, 
				request.created_by, request.updated_by, request.version
				)
			
			# Create approval steps if template is provided
			if request.workflow_template_id:
				await self._create_approval_steps(request)
			
			# Log submission
			await self._log_approval_event(
				request.id, "request_submitted", f"Approval request submitted: {request.title}",
				requested_by, None, request.status.value, tenant_id
			)
			
			# Add to processing queue
			await self._approval_queue.put(request)
			
			logger.info(f"✅ Approval request submitted successfully: {request.id}")
			return request
			
		except Exception as e:
			logger.error(f"Failed to submit approval request: {str(e)}", exc_info=True)
			raise
	
	async def process_approval_action(
		self,
		request_id: str,
		step_id: str,
		action: ApprovalAction,
		actor_id: str,
		notes: Optional[str],
		tenant_id: str
	) -> ApprovalRequest:
		"""
		Process an approval action (approve, reject, etc.)
		
		Args:
			request_id: Approval request identifier
			step_id: Approval step identifier
			action: Action being taken
			actor_id: User performing the action
			notes: Optional notes
			tenant_id: Tenant identifier
			
		Returns:
			Updated approval request
		"""
		try:
			logger.info(f"⚡ Processing approval action: {action} for request {request_id}")
			
			async with self.db_manager.get_connection() as conn:
				# Get current request and step
				request_row = await conn.fetchrow("""
					SELECT * FROM crm_approval_requests
					WHERE id = $1 AND tenant_id = $2
				""", request_id, tenant_id)
				
				if not request_row:
					raise ValueError(f"Approval request not found: {request_id}")
				
				step_row = await conn.fetchrow("""
					SELECT * FROM crm_approval_steps
					WHERE id = $1 AND approval_request_id = $2
				""", step_id, request_id)
				
				if not step_row:
					raise ValueError(f"Approval step not found: {step_id}")
				
				request = ApprovalRequest(**dict(request_row))
				step = ApprovalStep(**dict(step_row))
				
				# Validate action permissions
				if step.approver_id != actor_id and not await self._can_act_on_behalf(actor_id, step.approver_id, tenant_id):
					raise PermissionError(f"User {actor_id} cannot act on behalf of {step.approver_id}")
				
				# Update step
				step.action_taken = action
				step.response_notes = notes
				step.responded_at = datetime.utcnow()
				step.completed_at = datetime.utcnow()
				
				if action == ApprovalAction.APPROVE:
					step.status = ApprovalStepStatus.APPROVED
				elif action == ApprovalAction.REJECT:
					step.status = ApprovalStepStatus.REJECTED
					step.rejection_reason = notes
				
				# Update step in database
				await conn.execute("""
					UPDATE crm_approval_steps SET
						status = $3, action_taken = $4, response_notes = $5,
						responded_at = $6, completed_at = $7, rejection_reason = $8,
						updated_at = NOW()
					WHERE id = $1 AND approval_request_id = $2
				""", step_id, request_id, step.status.value, action.value, 
				notes, step.responded_at, step.completed_at, step.rejection_reason)
				
				# Update request based on action
				if action == ApprovalAction.APPROVE:
					request = await self._process_step_approval(request, step, actor_id, tenant_id)
				elif action == ApprovalAction.REJECT:
					request = await self._process_step_rejection(request, step, actor_id, tenant_id)
				
				# Log action
				await self._log_approval_event(
					request_id, f"step_{action.value}", 
					f"Step {step.step_number} {action.value}: {notes or 'No notes'}",
					actor_id, request.status.value, request.status.value, tenant_id
				)
			
			logger.info(f"✅ Approval action processed successfully: {action}")
			return request
			
		except Exception as e:
			logger.error(f"Failed to process approval action: {str(e)}", exc_info=True)
			raise
	
	async def get_approval_analytics(
		self,
		tenant_id: str,
		start_date: datetime,
		end_date: datetime,
		filters: Optional[Dict[str, Any]] = None
	) -> ApprovalAnalytics:
		"""
		Get comprehensive approval analytics
		
		Args:
			tenant_id: Tenant identifier
			start_date: Analysis period start
			end_date: Analysis period end
			filters: Additional filters
			
		Returns:
			Approval analytics data
		"""
		try:
			logger.info(f"📊 Generating approval analytics for tenant: {tenant_id}")
			
			analytics = ApprovalAnalytics(
				tenant_id=tenant_id,
				analysis_period_start=start_date,
				analysis_period_end=end_date
			)
			
			async with self.db_manager.get_connection() as conn:
				# Overall metrics
				overall_stats = await conn.fetchrow("""
					SELECT 
						COUNT(*) as total_requests,
						COUNT(*) FILTER (WHERE status = 'approved') as approved_requests,
						COUNT(*) FILTER (WHERE status = 'rejected') as rejected_requests,
						COUNT(*) FILTER (WHERE status = 'pending') as pending_requests,
						COUNT(*) FILTER (WHERE status = 'expired') as expired_requests,
						COUNT(*) FILTER (WHERE escalated = true) as escalated_requests
					FROM crm_approval_requests
					WHERE tenant_id = $1 AND submitted_at BETWEEN $2 AND $3
				""", tenant_id, start_date, end_date)
				
				if overall_stats:
					analytics.total_requests = overall_stats['total_requests'] or 0
					analytics.approved_requests = overall_stats['approved_requests'] or 0
					analytics.rejected_requests = overall_stats['rejected_requests'] or 0
					analytics.pending_requests = overall_stats['pending_requests'] or 0
					analytics.expired_requests = overall_stats['expired_requests'] or 0
					analytics.escalated_requests = overall_stats['escalated_requests'] or 0
					
					# Calculate rates
					if analytics.total_requests > 0:
						analytics.approval_rate = (analytics.approved_requests / analytics.total_requests) * 100
						analytics.rejection_rate = (analytics.rejected_requests / analytics.total_requests) * 100
						analytics.escalation_rate = (analytics.escalated_requests / analytics.total_requests) * 100
				
				# Timing metrics
				timing_stats = await conn.fetchrow("""
					SELECT 
						AVG(EXTRACT(EPOCH FROM (approved_at - submitted_at))/3600) as avg_approval_hours,
						PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (approved_at - submitted_at))/3600) as median_approval_hours
					FROM crm_approval_requests
					WHERE tenant_id = $1 AND submitted_at BETWEEN $2 AND $3
					AND status = 'approved' AND approved_at IS NOT NULL
				""", tenant_id, start_date, end_date)
				
				if timing_stats:
					analytics.average_approval_time = timing_stats['avg_approval_hours'] or 0.0
					analytics.median_approval_time = timing_stats['median_approval_hours'] or 0.0
				
				# Requests by type
				type_stats = await conn.fetch("""
					SELECT approval_type, COUNT(*) as count
					FROM crm_approval_requests
					WHERE tenant_id = $1 AND submitted_at BETWEEN $2 AND $3
					GROUP BY approval_type
					ORDER BY count DESC
				""", tenant_id, start_date, end_date)
				
				analytics.requests_by_type = {row['approval_type']: row['count'] for row in type_stats}
				
				# Approver performance
				approver_stats = await conn.fetch("""
					SELECT 
						s.approver_id,
						s.approver_name,
						COUNT(*) as total_steps,
						COUNT(*) FILTER (WHERE s.status = 'approved') as approved_steps,
						COUNT(*) FILTER (WHERE s.status = 'rejected') as rejected_steps,
						AVG(EXTRACT(EPOCH FROM (s.responded_at - s.assigned_at))/3600) as avg_response_hours
					FROM crm_approval_steps s
					JOIN crm_approval_requests r ON s.approval_request_id = r.id
					WHERE r.tenant_id = $1 AND r.submitted_at BETWEEN $2 AND $3
					AND s.responded_at IS NOT NULL
					GROUP BY s.approver_id, s.approver_name
					ORDER BY total_steps DESC
				""", tenant_id, start_date, end_date)
				
				analytics.approver_performance = {
					row['approver_id']: {
						'name': row['approver_name'],
						'total_steps': row['total_steps'],
						'approved_steps': row['approved_steps'],
						'rejected_steps': row['rejected_steps'],
						'approval_rate': (row['approved_steps'] / max(row['total_steps'], 1)) * 100,
						'avg_response_hours': row['avg_response_hours'] or 0.0
					}
					for row in approver_stats
				}
			
			logger.info(f"✅ Generated analytics for {analytics.total_requests} approval requests")
			return analytics
			
		except Exception as e:
			logger.error(f"Failed to generate approval analytics: {str(e)}", exc_info=True)
			raise
	
	# Background processing methods
	
	async def _process_approval_queue(self):
		"""Process queued approval requests"""
		while self._initialized:
			try:
				# Get request from queue with timeout
				try:
					request = await asyncio.wait_for(self._approval_queue.get(), timeout=1.0)
				except asyncio.TimeoutError:
					continue
				
				# Process the request
				await self._initiate_approval_workflow(request)
				
			except Exception as e:
				logger.error(f"Error processing approval queue: {str(e)}")
				await asyncio.sleep(5)
	
	async def _process_timeouts_and_escalations(self):
		"""Process approval timeouts and escalations"""
		while self._initialized:
			try:
				# Check every 15 minutes
				await asyncio.sleep(900)
				
				async with self.db_manager.get_connection() as conn:
					# Get expired requests
					expired_requests = await conn.fetch("""
						SELECT * FROM crm_approval_requests
						WHERE status = 'pending' AND expires_at < NOW()
					""")
					
					for request_row in expired_requests:
						await self._handle_request_timeout(ApprovalRequest(**dict(request_row)))
					
					# Get steps needing escalation
					escalation_steps = await conn.fetch("""
						SELECT s.*, r.escalation_timeout_hours
						FROM crm_approval_steps s
						JOIN crm_approval_requests r ON s.approval_request_id = r.id
						JOIN crm_approval_workflow_templates t ON r.workflow_template_id = t.id
						WHERE s.status = 'pending' 
						AND t.enable_escalation = true
						AND s.assigned_at < NOW() - (t.escalation_timeout_hours || ' hours')::INTERVAL
						AND s.escalated = false
					""")
					
					for step_row in escalation_steps:
						await self._handle_step_escalation(ApprovalStep(**dict(step_row)))
				
			except Exception as e:
				logger.error(f"Error processing timeouts and escalations: {str(e)}")
				await asyncio.sleep(60)
	
	async def _send_reminder_notifications(self):
		"""Send reminder notifications for pending approvals"""
		while self._initialized:
			try:
				# Check every hour
				await asyncio.sleep(3600)
				
				async with self.db_manager.get_connection() as conn:
					# Get steps needing reminders
					reminder_steps = await conn.fetch("""
						SELECT s.*, r.reminder_intervals
						FROM crm_approval_steps s
						JOIN crm_approval_requests r ON s.approval_request_id = r.id
						JOIN crm_approval_workflow_templates t ON r.workflow_template_id = t.id
						WHERE s.status = 'pending'
						AND (
							s.last_reminder_at IS NULL OR 
							s.last_reminder_at < NOW() - INTERVAL '24 hours'
						)
					""")
					
					for step_row in reminder_steps:
						await self._send_approval_reminder(ApprovalStep(**dict(step_row)))
				
			except Exception as e:
				logger.error(f"Error sending reminder notifications: {str(e)}")
				await asyncio.sleep(300)
	
	# Helper methods
	
	async def _get_workflow_template(self, template_id: str, tenant_id: str) -> Optional[ApprovalWorkflowTemplate]:
		"""Get workflow template by ID"""
		try:
			async with self.db_manager.get_connection() as conn:
				template_row = await conn.fetchrow("""
					SELECT * FROM crm_approval_workflow_templates
					WHERE id = $1 AND tenant_id = $2
				""", template_id, tenant_id)
				
				if template_row:
					return ApprovalWorkflowTemplate(**dict(template_row))
				return None
				
		except Exception as e:
			logger.error(f"Failed to get workflow template: {str(e)}")
			return None
	
	async def _create_approval_steps(self, request: ApprovalRequest):
		"""Create approval steps for a request"""
		try:
			template = await self._get_workflow_template(request.workflow_template_id, request.tenant_id)
			if not template:
				return
			
			async with self.db_manager.get_connection() as conn:
				step_number = 1
				for step_config in template.approval_steps:
					step = ApprovalStep(
						approval_request_id=request.id,
						tenant_id=request.tenant_id,
						step_number=step_number,
						step_name=step_config.get('name', f'Step {step_number}'),
						step_description=step_config.get('description'),
						approver_id=step_config['approver_id'],
						approver_name=step_config.get('approver_name', step_config['approver_id']),
						approver_email=step_config.get('approver_email', f"{step_config['approver_id']}@example.com"),
						approver_role=step_config.get('role'),
						is_required=step_config.get('required', True),
						can_delegate=step_config.get('can_delegate', True),
						timeout_hours=step_config.get('timeout_hours', template.default_timeout_hours)
					)
					
					await conn.execute("""
						INSERT INTO crm_approval_steps (
							id, approval_request_id, tenant_id, step_number, step_name, step_description,
							approver_id, approver_name, approver_email, approver_role,
							is_required, can_delegate, timeout_hours, status, assigned_at,
							responded_at, completed_at, action_taken, response_notes, rejection_reason,
							delegated_to, delegated_at, delegation_reason, escalated, escalated_at, escalation_trigger,
							reminders_sent, last_reminder_at, metadata, created_at, updated_at
						) VALUES (
							$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
							$16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31
						)
					""", 
					step.id, step.approval_request_id, step.tenant_id, step.step_number, 
					step.step_name, step.step_description, step.approver_id, step.approver_name,
					step.approver_email, step.approver_role, step.is_required, step.can_delegate,
					step.timeout_hours, step.status.value, step.assigned_at, step.responded_at,
					step.completed_at, step.action_taken, step.response_notes, step.rejection_reason,
					step.delegated_to, step.delegated_at, step.delegation_reason, step.escalated,
					step.escalated_at, step.escalation_trigger, step.reminders_sent,
					step.last_reminder_at, step.metadata, step.created_at, step.updated_at
					)
					
					step_number += 1
				
				# Update request with step count
				await conn.execute("""
					UPDATE crm_approval_requests SET
						total_steps = $2, updated_at = NOW()
					WHERE id = $1
				""", request.id, len(template.approval_steps))
				
		except Exception as e:
			logger.error(f"Failed to create approval steps: {str(e)}")
			raise
	
	async def _initiate_approval_workflow(self, request: ApprovalRequest):
		"""Initiate approval workflow for a request"""
		try:
			logger.info(f"🚀 Initiating approval workflow for request: {request.id}")
			
			# Update request status
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					UPDATE crm_approval_requests SET
						status = $2, current_step = 1, updated_at = NOW()
					WHERE id = $1
				""", request.id, ApprovalStatus.IN_REVIEW.value)
			
			# Send notifications to first approver(s)
			await self._notify_approvers(request, 1)
			
		except Exception as e:
			logger.error(f"Failed to initiate approval workflow: {str(e)}")
	
	async def _process_step_approval(self, request: ApprovalRequest, step: ApprovalStep, actor_id: str, tenant_id: str) -> ApprovalRequest:
		"""Process step approval and advance workflow"""
		try:
			async with self.db_manager.get_connection() as conn:
				# Update request progress
				await conn.execute("""
					UPDATE crm_approval_requests SET
						completed_steps = completed_steps + 1,
						updated_at = NOW()
					WHERE id = $1
				""", request.id)
				
				# Check if all steps are complete
				remaining_steps = await conn.fetchval("""
					SELECT COUNT(*) FROM crm_approval_steps
					WHERE approval_request_id = $1 AND status = 'pending' AND is_required = true
				""", request.id)
				
				if remaining_steps == 0:
					# All steps approved - approve request
					await conn.execute("""
						UPDATE crm_approval_requests SET
							status = $2, approved_at = NOW(), final_approver = $3,
							updated_at = NOW()
						WHERE id = $1
					""", request.id, ApprovalStatus.APPROVED.value, actor_id)
					
					request.status = ApprovalStatus.APPROVED
					request.approved_at = datetime.utcnow()
					request.final_approver = actor_id
				else:
					# Move to next step
					next_step = step.step_number + 1
					await conn.execute("""
						UPDATE crm_approval_requests SET
							current_step = $2, updated_at = NOW()
						WHERE id = $1
					""", request.id, next_step)
					
					# Notify next approvers
					await self._notify_approvers(request, next_step)
			
			return request
			
		except Exception as e:
			logger.error(f"Failed to process step approval: {str(e)}")
			raise
	
	async def _process_step_rejection(self, request: ApprovalRequest, step: ApprovalStep, actor_id: str, tenant_id: str) -> ApprovalRequest:
		"""Process step rejection"""
		try:
			async with self.db_manager.get_connection() as conn:
				# Reject the entire request
				await conn.execute("""
					UPDATE crm_approval_requests SET
						status = $2, rejected_at = NOW(), final_approver = $3,
						rejection_reason = $4, updated_at = NOW()
					WHERE id = $1
				""", request.id, ApprovalStatus.REJECTED.value, actor_id, step.rejection_reason)
				
				request.status = ApprovalStatus.REJECTED
				request.rejected_at = datetime.utcnow()
				request.final_approver = actor_id
				request.rejection_reason = step.rejection_reason
			
			return request
			
		except Exception as e:
			logger.error(f"Failed to process step rejection: {str(e)}")
			raise
	
	async def _can_act_on_behalf(self, actor_id: str, approver_id: str, tenant_id: str) -> bool:
		"""Check if actor can act on behalf of approver"""
		# Simple implementation - can be extended with delegation rules
		return actor_id == approver_id
	
	async def _log_approval_event(self, request_id: str, event_type: str, description: str, actor_id: str, old_status: Optional[str], new_status: str, tenant_id: str):
		"""Log approval workflow event"""
		try:
			history = ApprovalHistory(
				approval_request_id=request_id,
				tenant_id=tenant_id,
				event_type=event_type,
				event_description=description,
				actor_id=actor_id,
				actor_name=actor_id,  # In real implementation, lookup user name
				old_status=old_status,
				new_status=new_status
			)
			
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					INSERT INTO crm_approval_history (
						id, approval_request_id, tenant_id, event_type, event_description,
						actor_id, actor_name, actor_role, old_status, new_status, action_data,
						ip_address, user_agent, system_info, metadata, created_at
					) VALUES (
						$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16
					)
				""", 
				history.id, history.approval_request_id, history.tenant_id, history.event_type,
				history.event_description, history.actor_id, history.actor_name, history.actor_role,
				history.old_status, history.new_status, history.action_data, history.ip_address,
				history.user_agent, history.system_info, history.metadata, history.created_at
				)
				
		except Exception as e:
			logger.error(f"Failed to log approval event: {str(e)}")
	
	async def _notify_approvers(self, request: ApprovalRequest, step_number: int):
		"""Notify approvers for a specific step"""
		try:
			# Implementation would send notifications via email, etc.
			logger.info(f"📧 Notifying approvers for step {step_number} of request {request.id}")
		except Exception as e:
			logger.error(f"Failed to notify approvers: {str(e)}")
	
	async def _handle_request_timeout(self, request: ApprovalRequest):
		"""Handle request timeout"""
		try:
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					UPDATE crm_approval_requests SET
						status = $2, updated_at = NOW()
					WHERE id = $1
				""", request.id, ApprovalStatus.EXPIRED.value)
				
			await self._log_approval_event(
				request.id, "request_expired", "Request expired due to timeout",
				"system", request.status.value, ApprovalStatus.EXPIRED.value, request.tenant_id
			)
			
		except Exception as e:
			logger.error(f"Failed to handle request timeout: {str(e)}")
	
	async def _handle_step_escalation(self, step: ApprovalStep):
		"""Handle step escalation"""
		try:
			# Mark step as escalated
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					UPDATE crm_approval_steps SET
						escalated = true, escalated_at = NOW(),
						escalation_trigger = $2, updated_at = NOW()
					WHERE id = $1
				""", step.id, EscalationTrigger.TIMEOUT.value)
			
			logger.info(f"⬆️ Escalated approval step: {step.id}")
			
		except Exception as e:
			logger.error(f"Failed to handle step escalation: {str(e)}")
	
	async def _send_approval_reminder(self, step: ApprovalStep):
		"""Send approval reminder"""
		try:
			# Update reminder count
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					UPDATE crm_approval_steps SET
						reminders_sent = reminders_sent + 1,
						last_reminder_at = NOW(),
						updated_at = NOW()
					WHERE id = $1
				""", step.id)
			
			logger.info(f"⏰ Sent approval reminder for step: {step.id}")
			
		except Exception as e:
			logger.error(f"Failed to send approval reminder: {str(e)}")
	
	async def shutdown(self):
		"""Shutdown the approval workflow engine"""
		self._initialized = False
		
		# Cancel background tasks
		for task in self._processing_tasks:
			task.cancel()
		
		# Wait for tasks to complete
		await asyncio.gather(*self._processing_tasks, return_exceptions=True)