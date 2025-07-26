"""
APG Budgeting & Forecasting - Budget Approval Workflows

Enterprise-grade approval workflow engine with flexible department-specific 
approval chains, escalation management, and comprehensive audit integration.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Set, Callable
from decimal import Decimal
from uuid import UUID
import json
import logging
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass

import asyncpg
from pydantic import BaseModel, Field, validator, root_validator
from pydantic import ConfigDict

from .models import (
	APGBaseModel, BFBudgetStatus, BFApprovalStatus,
	PositiveAmount, CurrencyCode, NonEmptyString
)
from .service import APGTenantContext, BFServiceConfig, ServiceResponse, APGServiceBase
from uuid_extensions import uuid7str


# =============================================================================
# Approval Workflow Models
# =============================================================================

class ApprovalAction(str, Enum):
	"""Approval action enumeration."""
	APPROVE = "approve"
	REJECT = "reject"
	REQUEST_CHANGES = "request_changes"
	DELEGATE = "delegate"
	ESCALATE = "escalate"
	WITHDRAW = "withdraw"
	RESUBMIT = "resubmit"


class WorkflowStepType(str, Enum):
	"""Workflow step type enumeration."""
	SEQUENTIAL = "sequential"
	PARALLEL = "parallel"
	CONDITIONAL = "conditional"
	ESCALATION = "escalation"
	NOTIFICATION = "notification"


class EscalationTrigger(str, Enum):
	"""Escalation trigger enumeration."""
	TIMEOUT = "timeout"
	REJECTION = "rejection"
	MANUAL_REQUEST = "manual_request"
	AMOUNT_THRESHOLD = "amount_threshold"
	DEPARTMENT_POLICY = "department_policy"


class WorkflowTemplate(APGBaseModel):
	"""Approval workflow template for different budget types and departments."""
	
	template_name: NonEmptyString = Field(..., max_length=255)
	template_description: Optional[str] = Field(None, max_length=1000)
	
	# Workflow scope and applicability
	applies_to_budget_types: List[str] = Field(..., min_items=1)
	applies_to_departments: List[str] = Field(default_factory=list)
	applies_to_amount_ranges: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Workflow configuration
	is_active: bool = Field(default=True)
	is_default: bool = Field(default=False)
	priority: int = Field(default=1, ge=1, le=10)
	version: int = Field(default=1, ge=1)
	
	# Workflow structure
	workflow_steps: List[Dict[str, Any]] = Field(..., min_items=1)
	approval_requirements: Dict[str, Any] = Field(default_factory=dict)
	escalation_rules: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Timing and SLA configuration
	default_step_timeout_hours: int = Field(default=24, ge=1, le=720)  # 30 days max
	total_workflow_timeout_hours: int = Field(default=120, ge=1, le=2160)  # 90 days max
	escalation_timeout_hours: int = Field(default=72, ge=1, le=168)  # 7 days max
	
	# Notification settings
	notification_settings: Dict[str, Any] = Field(default_factory=dict)
	reminder_intervals: List[int] = Field(default_factory=lambda: [24, 72])  # hours
	
	# Advanced features
	supports_delegation: bool = Field(default=True)
	supports_parallel_approval: bool = Field(default=False)
	requires_unanimous_approval: bool = Field(default=False)
	allows_skip_levels: bool = Field(default=False)
	
	# Conditional logic
	conditional_rules: List[Dict[str, Any]] = Field(default_factory=list)
	dynamic_approver_assignment: bool = Field(default=False)
	
	# Compliance and audit
	requires_reason_for_rejection: bool = Field(default=True)
	requires_reason_for_delegation: bool = Field(default=True)
	audit_all_actions: bool = Field(default=True)
	
	# Integration settings
	integration_webhooks: List[Dict[str, Any]] = Field(default_factory=list)
	external_system_approvals: List[Dict[str, Any]] = Field(default_factory=list)

	@validator('workflow_steps')
	def validate_workflow_steps(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
		"""Validate workflow steps structure."""
		for i, step in enumerate(v):
			if 'step_id' not in step:
				step['step_id'] = f"step_{i+1}"
			if 'step_type' not in step:
				step['step_type'] = WorkflowStepType.SEQUENTIAL.value
			if 'required_approvers' not in step:
				raise ValueError(f"Step {i+1}: required_approvers is missing")
		return v

	@root_validator
	def validate_template_consistency(cls, values: Dict[str, Any]) -> Dict[str, Any]:
		"""Validate workflow template consistency."""
		workflow_steps = values.get('workflow_steps', [])
		total_timeout = values.get('total_workflow_timeout_hours', 120)
		step_timeout = values.get('default_step_timeout_hours', 24)
		
		# Validate timeout consistency
		if len(workflow_steps) * step_timeout > total_timeout:
			raise ValueError("Total workflow timeout must accommodate all steps")
		
		# Validate amount ranges don't overlap
		amount_ranges = values.get('applies_to_amount_ranges', [])
		for i, range1 in enumerate(amount_ranges):
			for j, range2 in enumerate(amount_ranges[i+1:], i+1):
				if cls._ranges_overlap(range1, range2):
					raise ValueError(f"Amount ranges {i+1} and {j+1} overlap")
		
		return values

	@staticmethod
	def _ranges_overlap(range1: Dict[str, Any], range2: Dict[str, Any]) -> bool:
		"""Check if two amount ranges overlap."""
		min1, max1 = range1.get('min_amount', 0), range1.get('max_amount', float('inf'))
		min2, max2 = range2.get('min_amount', 0), range2.get('max_amount', float('inf'))
		return min1 < max2 and min2 < max1


class ApprovalWorkflowInstance(APGBaseModel):
	"""Active approval workflow instance for a specific budget."""
	
	workflow_instance_id: str = Field(default_factory=uuid7str)
	workflow_template_id: str = Field(...)
	budget_id: str = Field(...)
	
	# Workflow state
	current_step_id: str = Field(...)
	current_step_index: int = Field(default=0, ge=0)
	workflow_status: str = Field(default="active", max_length=20)  # active, completed, rejected, cancelled, escalated
	
	# Submission details
	submitted_by: str = Field(...)
	submitted_at: datetime = Field(default_factory=datetime.utcnow)
	submission_notes: Optional[str] = Field(None, max_length=2000)
	submission_attachments: List[str] = Field(default_factory=list)
	
	# Current state
	pending_approvers: List[str] = Field(default_factory=list)
	completed_approvers: List[str] = Field(default_factory=list)
	rejected_by: List[str] = Field(default_factory=list)
	
	# Timing tracking
	workflow_started_at: datetime = Field(default_factory=datetime.utcnow)
	current_step_started_at: datetime = Field(default_factory=datetime.utcnow)
	step_deadline: Optional[datetime] = Field(None)
	workflow_deadline: Optional[datetime] = Field(None)
	
	# Escalation tracking
	escalation_count: int = Field(default=0, ge=0)
	last_escalation_at: Optional[datetime] = Field(None)
	escalated_to: List[str] = Field(default_factory=list)
	
	# Workflow history
	approval_history: List[Dict[str, Any]] = Field(default_factory=list)
	escalation_history: List[Dict[str, Any]] = Field(default_factory=list)
	delegation_history: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Notification tracking
	notifications_sent: List[Dict[str, Any]] = Field(default_factory=list)
	reminder_count: int = Field(default=0, ge=0)
	last_reminder_sent: Optional[datetime] = Field(None)
	
	# Business context
	budget_amount: PositiveAmount = Field(...)
	budget_type: str = Field(...)
	department: str = Field(...)
	fiscal_year: str = Field(...)
	
	# Risk and compliance flags
	high_risk_transaction: bool = Field(default=False)
	requires_external_approval: bool = Field(default=False)
	compliance_checks_completed: bool = Field(default=False)
	fraud_check_status: str = Field(default="pending", max_length=20)
	
	# Performance metrics
	average_response_time_hours: Optional[float] = Field(None, ge=0.0)
	step_completion_times: List[float] = Field(default_factory=list)
	workflow_efficiency_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class ApprovalAction(APGBaseModel):
	"""Individual approval action taken by an approver."""
	
	action_id: str = Field(default_factory=uuid7str)
	workflow_instance_id: str = Field(...)
	step_id: str = Field(...)
	
	# Action details
	action_type: ApprovalAction = Field(...)
	action_taken_by: str = Field(...)
	action_taken_at: datetime = Field(default_factory=datetime.utcnow)
	
	# Decision context
	decision_reason: Optional[str] = Field(None, max_length=2000)
	additional_comments: Optional[str] = Field(None, max_length=1000)
	conditions_or_requirements: List[str] = Field(default_factory=list)
	
	# Delegation details (if applicable)
	delegated_to: Optional[str] = Field(None)
	delegation_reason: Optional[str] = Field(None, max_length=1000)
	delegation_expiry: Optional[datetime] = Field(None)
	
	# Attachments and supporting documents
	supporting_documents: List[str] = Field(default_factory=list)
	external_approvals: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Approval metadata
	approver_role: str = Field(...)
	approver_department: str = Field(...)
	approval_authority_level: str = Field(...)
	
	# Risk assessment
	risk_assessment: Optional[Dict[str, Any]] = Field(None)
	compliance_sign_off: bool = Field(default=False)
	requires_follow_up: bool = Field(default=False)
	
	# System tracking
	client_ip: Optional[str] = Field(None)
	user_agent: Optional[str] = Field(None)
	geo_location: Optional[Dict[str, Any]] = Field(None)
	
	# Verification
	digital_signature: Optional[str] = Field(None)
	two_factor_verified: bool = Field(default=False)
	authentication_method: str = Field(...)

	@validator('decision_reason')
	def validate_decision_reason(cls, v: Optional[str], values: Dict[str, Any]) -> Optional[str]:
		"""Validate decision reason is provided for rejection."""
		action_type = values.get('action_type')
		if action_type in [ApprovalAction.REJECT, ApprovalAction.REQUEST_CHANGES] and not v:
			raise ValueError(f"Decision reason is required for {action_type} actions")
		return v


class WorkflowEscalation(BaseModel):
	"""Workflow escalation record."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	escalation_id: str = Field(default_factory=uuid7str)
	workflow_instance_id: str = Field(...)
	escalation_trigger: EscalationTrigger = Field(...)
	
	# Escalation details
	escalated_from_step: str = Field(...)
	escalated_to_step: str = Field(...)
	escalated_from_approver: str = Field(...)
	escalated_to_approver: str = Field(...)
	
	# Escalation timing
	escalation_triggered_at: datetime = Field(default_factory=datetime.utcnow)
	trigger_condition_met_at: datetime = Field(default_factory=datetime.utcnow)
	escalation_deadline: Optional[datetime] = Field(None)
	
	# Escalation context
	escalation_reason: str = Field(..., max_length=1000)
	urgency_level: str = Field(default="normal", max_length=20)  # low, normal, high, critical
	business_impact: Optional[str] = Field(None, max_length=500)
	
	# Resolution tracking
	is_resolved: bool = Field(default=False)
	resolved_at: Optional[datetime] = Field(None)
	resolution_action: Optional[str] = Field(None)
	resolution_notes: Optional[str] = Field(None)
	
	# Escalation metadata
	escalation_level: int = Field(default=1, ge=1, le=5)
	auto_escalated: bool = Field(default=True)
	escalation_policy_id: Optional[str] = Field(None)


# =============================================================================
# Approval Workflow Service
# =============================================================================

class ApprovalWorkflowService(APGServiceBase):
	"""
	Comprehensive approval workflow service providing flexible,
	department-specific approval chains with escalation management.
	"""
	
	def __init__(self, context: APGTenantContext, config: BFServiceConfig):
		super().__init__(context, config)
		self._workflow_engines: Dict[str, Any] = {}
		self._escalation_handlers: Dict[EscalationTrigger, Callable] = {}
		self._notification_handlers: Dict[str, Callable] = {}
		
		# Initialize workflow components
		self._initialize_workflow_engines()
		self._initialize_escalation_handlers()
		self._initialize_notification_handlers()

	async def create_workflow_template(self, template_data: Dict[str, Any]) -> ServiceResponse:
		"""Create a new approval workflow template."""
		try:
			# Validate permissions
			if not await self._validate_permissions('workflow.create_template'):
				raise PermissionError("Insufficient permissions to create workflow template")
			
			# Inject context data
			template_data.update({
				'tenant_id': self.context.tenant_id,
				'created_by': self.context.user_id,
				'updated_by': self.context.user_id
			})
			
			# Create template model
			template = WorkflowTemplate(**template_data)
			
			# Validate template configuration
			validation_result = await self._validate_template_configuration(template)
			if not validation_result['is_valid']:
				return ServiceResponse(
					success=False,
					message="Workflow template validation failed",
					errors=validation_result['errors']
				)
			
			# Check for conflicts with existing templates
			conflict_check = await self._check_template_conflicts(template)
			if conflict_check['has_conflicts']:
				return ServiceResponse(
					success=False,
					message="Template conflicts with existing templates",
					errors=conflict_check['conflicts'],
					data={'conflicting_templates': conflict_check['conflicting_templates']}
				)
			
			# Start database transaction
			async with self._connection.transaction():
				# Insert template
				template_id = await self._insert_workflow_template(template)
				
				# Create workflow steps
				await self._create_workflow_steps(template_id, template.workflow_steps)
				
				# Create escalation rules
				if template.escalation_rules:
					await self._create_escalation_rules(template_id, template.escalation_rules)
				
				# Set as default if specified
				if template.is_default:
					await self._set_default_template(template_id, template)
				
				# Index template for search
				await self._index_workflow_template(template_id, template)
				
				# Audit template creation
				await self._audit_action('create_workflow_template', 'workflow', template_id,
					new_data=template.dict())
			
			return ServiceResponse(
				success=True,
				message=f"Workflow template '{template.template_name}' created successfully",
				data={
					'template_id': template_id,
					'template_name': template.template_name,
					'applies_to_departments': template.applies_to_departments,
					'workflow_steps_count': len(template.workflow_steps)
				}
			)
			
		except Exception as e:
			return self._handle_service_error(e, 'create_workflow_template')

	async def submit_budget_for_approval(self, budget_id: str, submission_data: Dict[str, Any]) -> ServiceResponse:
		"""Submit a budget for approval workflow processing."""
		try:
			# Validate permissions
			if not await self._validate_permissions('budget.submit_approval', budget_id):
				raise PermissionError("Insufficient permissions to submit budget for approval")
			
			# Get budget and validate submission eligibility
			budget = await self._get_budget(budget_id)
			if not budget:
				raise ValueError("Budget not found")
			
			submission_validation = await self._validate_budget_submission(budget)
			if not submission_validation['is_eligible']:
				return ServiceResponse(
					success=False,
					message="Budget is not eligible for submission",
					errors=submission_validation['errors']
				)
			
			# Determine appropriate workflow template
			workflow_template = await self._determine_workflow_template(budget)
			if not workflow_template:
				raise ValueError("No suitable workflow template found for this budget")
			
			# Create workflow instance
			workflow_instance_data = {
				'workflow_template_id': workflow_template['id'],
				'budget_id': budget_id,
				'submitted_by': self.context.user_id,
				'submission_notes': submission_data.get('notes'),
				'submission_attachments': submission_data.get('attachments', []),
				'budget_amount': budget['total_amount'],
				'budget_type': budget['budget_type'],
				'department': budget['department'],
				'fiscal_year': budget['fiscal_year'],
				'tenant_id': self.context.tenant_id,
				'created_by': self.context.user_id,
				'updated_by': self.context.user_id
			}
			
			workflow_instance = ApprovalWorkflowInstance(**workflow_instance_data)
			
			# Initialize workflow with first step
			first_step = workflow_template['workflow_steps'][0]
			workflow_instance.current_step_id = first_step['step_id']
			workflow_instance.pending_approvers = await self._resolve_step_approvers(first_step, budget)
			
			# Set deadlines
			workflow_instance.step_deadline = datetime.utcnow() + timedelta(
				hours=workflow_template.get('default_step_timeout_hours', 24)
			)
			workflow_instance.workflow_deadline = datetime.utcnow() + timedelta(
				hours=workflow_template.get('total_workflow_timeout_hours', 120)
			)
			
			# Start database transaction
			async with self._connection.transaction():
				# Insert workflow instance
				workflow_id = await self._insert_workflow_instance(workflow_instance)
				
				# Update budget status
				await self._connection.execute("""
					UPDATE budgets 
					SET status = 'submitted',
						workflow_state = 'pending_approval',
						workflow_instance_id = $1,
						updated_at = NOW(),
						updated_by = $2
					WHERE id = $3
				""", workflow_id, self.context.user_id, budget_id)
				
				# Send notifications to initial approvers
				await self._send_approval_notifications(workflow_id, workflow_instance.pending_approvers)
				
				# Schedule escalation monitoring
				await self._schedule_escalation_monitoring(workflow_id)
				
				# Audit workflow submission
				await self._audit_action('submit_approval_workflow', 'budget', budget_id,
					new_data={'workflow_instance_id': workflow_id})
			
			return ServiceResponse(
				success=True,
				message="Budget submitted for approval successfully",
				data={
					'workflow_instance_id': workflow_id,
					'workflow_template': workflow_template['template_name'],
					'pending_approvers': workflow_instance.pending_approvers,
					'estimated_completion': workflow_instance.workflow_deadline.isoformat(),
					'current_step': first_step['step_name'],
					'total_steps': len(workflow_template['workflow_steps'])
				}
			)
			
		except Exception as e:
			return self._handle_service_error(e, 'submit_budget_for_approval')

	async def process_approval_action(self, workflow_instance_id: str, action_data: Dict[str, Any]) -> ServiceResponse:
		"""Process an approval action (approve, reject, delegate, etc.)."""
		try:
			# Validate permissions
			if not await self._validate_permissions('workflow.take_action', workflow_instance_id):
				raise PermissionError("Insufficient permissions to take approval action")
			
			# Get workflow instance
			workflow_instance = await self._get_workflow_instance(workflow_instance_id)
			if not workflow_instance:
				raise ValueError("Workflow instance not found")
			
			# Validate user can take action
			action_validation = await self._validate_approval_action(workflow_instance, action_data)
			if not action_validation['is_valid']:
				return ServiceResponse(
					success=False,
					message="Action validation failed",
					errors=action_validation['errors']
				)
			
			# Create approval action record
			approval_action_data = {
				**action_data,
				'workflow_instance_id': workflow_instance_id,
				'step_id': workflow_instance['current_step_id'],
				'action_taken_by': self.context.user_id,
				'approver_role': action_validation['approver_role'],
				'approver_department': action_validation['approver_department'],
				'approval_authority_level': action_validation['authority_level'],
				'authentication_method': action_data.get('auth_method', 'password'),
				'tenant_id': self.context.tenant_id,
				'created_by': self.context.user_id,
				'updated_by': self.context.user_id
			}
			
			approval_action = ApprovalAction(**approval_action_data)
			
			# Start database transaction
			async with self._connection.transaction():
				# Record the action
				action_id = await self._insert_approval_action(approval_action)
				
				# Process the action based on type
				workflow_update = await self._process_action_by_type(
					workflow_instance, approval_action, action_validation
				)
				
				# Update workflow instance
				await self._update_workflow_instance(workflow_instance_id, workflow_update)
				
				# Handle workflow progression
				progression_result = await self._handle_workflow_progression(
					workflow_instance_id, approval_action.action_type
				)
				
				# Send notifications based on action
				await self._send_action_notifications(workflow_instance_id, approval_action)
				
				# Update budget status if workflow completed
				if progression_result['workflow_completed']:
					await self._update_budget_from_workflow_completion(
						workflow_instance['budget_id'], progression_result['final_status']
					)
				
				# Audit the action
				await self._audit_action('process_approval_action', 'workflow', workflow_instance_id,
					new_data={'action_type': approval_action.action_type, 'action_id': action_id})
			
			return ServiceResponse(
				success=True,
				message=f"Approval action '{approval_action.action_type}' processed successfully",
				data={
					'action_id': action_id,
					'workflow_status': progression_result.get('workflow_status'),
					'next_step': progression_result.get('next_step'),
					'pending_approvers': progression_result.get('pending_approvers', []),
					'workflow_completed': progression_result['workflow_completed'],
					'final_decision': progression_result.get('final_decision')
				}
			)
			
		except Exception as e:
			return self._handle_service_error(e, 'process_approval_action')

	async def handle_workflow_escalation(self, escalation_data: Dict[str, Any]) -> ServiceResponse:
		"""Handle workflow escalation due to timeout or manual request."""
		try:
			workflow_instance_id = escalation_data.get('workflow_instance_id')
			escalation_trigger = EscalationTrigger(escalation_data.get('trigger'))
			
			# Validate permissions
			if not await self._validate_permissions('workflow.escalate', workflow_instance_id):
				raise PermissionError("Insufficient permissions to escalate workflow")
			
			# Get workflow instance and template
			workflow_instance = await self._get_workflow_instance(workflow_instance_id)
			workflow_template = await self._get_workflow_template(workflow_instance['workflow_template_id'])
			
			if not workflow_instance or not workflow_template:
				raise ValueError("Workflow instance or template not found")
			
			# Determine escalation target
			escalation_target = await self._determine_escalation_target(
				workflow_instance, workflow_template, escalation_trigger
			)
			
			if not escalation_target:
				return ServiceResponse(
					success=False,
					message="No escalation target available",
					errors=["Maximum escalation level reached or no escalation rules defined"]
				)
			
			# Create escalation record
			escalation_record_data = {
				'workflow_instance_id': workflow_instance_id,
				'escalation_trigger': escalation_trigger,
				'escalated_from_step': workflow_instance['current_step_id'],
				'escalated_to_step': escalation_target['step_id'],
				'escalated_from_approver': workflow_instance.get('pending_approvers', [''])[0],
				'escalated_to_approver': escalation_target['approver'],
				'escalation_reason': escalation_data.get('reason', f"Escalated due to {escalation_trigger.value}"),
				'urgency_level': escalation_data.get('urgency_level', 'normal'),
				'business_impact': escalation_data.get('business_impact'),
				'escalation_level': workflow_instance.get('escalation_count', 0) + 1,
				'auto_escalated': escalation_data.get('auto_escalated', False)
			}
			
			escalation_record = WorkflowEscalation(**escalation_record_data)
			
			# Start database transaction
			async with self._connection.transaction():
				# Insert escalation record
				escalation_id = await self._insert_escalation_record(escalation_record)
				
				# Update workflow instance with escalation
				escalation_update = {
					'escalation_count': workflow_instance.get('escalation_count', 0) + 1,
					'last_escalation_at': datetime.utcnow(),
					'escalated_to': escalation_target['escalation_chain'],
					'current_step_id': escalation_target['step_id'],
					'pending_approvers': [escalation_target['approver']],
					'step_deadline': datetime.utcnow() + timedelta(
						hours=escalation_target.get('timeout_hours', 24)
					)
				}
				
				await self._update_workflow_instance(workflow_instance_id, escalation_update)
				
				# Send escalation notifications
				await self._send_escalation_notifications(escalation_id, escalation_target)
				
				# Log escalation in workflow history
				await self._add_to_workflow_history(workflow_instance_id, 'escalation', {
					'escalation_id': escalation_id,
					'trigger': escalation_trigger.value,
					'escalated_to': escalation_target['approver']
				})
				
				# Audit escalation
				await self._audit_action('escalate_workflow', 'workflow', workflow_instance_id,
					new_data={'escalation_id': escalation_id, 'trigger': escalation_trigger.value})
			
			return ServiceResponse(
				success=True,
				message="Workflow escalated successfully",
				data={
					'escalation_id': escalation_id,
					'escalated_to': escalation_target['approver'],
					'escalation_level': escalation_record.escalation_level,
					'new_deadline': escalation_update['step_deadline'].isoformat(),
					'urgency_level': escalation_record.urgency_level
				}
			)
			
		except Exception as e:
			return self._handle_service_error(e, 'handle_workflow_escalation')

	async def get_workflow_status(self, workflow_instance_id: str) -> ServiceResponse:
		"""Get comprehensive workflow status and progress information."""
		try:
			# Validate permissions
			if not await self._validate_permissions('workflow.view', workflow_instance_id):
				raise PermissionError("Insufficient permissions to view workflow status")
			
			# Get workflow instance with related data
			workflow_instance = await self._get_workflow_instance_with_details(workflow_instance_id)
			if not workflow_instance:
				raise ValueError("Workflow instance not found")
			
			# Get workflow template
			workflow_template = await self._get_workflow_template(workflow_instance['workflow_template_id'])
			
			# Get approval history
			approval_history = await self._get_approval_history(workflow_instance_id)
			
			# Get escalation history
			escalation_history = await self._get_escalation_history(workflow_instance_id)
			
			# Calculate progress metrics
			progress_metrics = await self._calculate_workflow_progress(workflow_instance, workflow_template)
			
			# Get current step details
			current_step = await self._get_current_step_details(
				workflow_instance['current_step_id'], workflow_template
			)
			
			# Get budget information
			budget = await self._get_budget(workflow_instance['budget_id'])
			
			workflow_status = {
				'workflow_instance_id': workflow_instance_id,
				'workflow_status': workflow_instance['workflow_status'],
				'current_step': current_step,
				'progress': progress_metrics,
				'budget_info': {
					'budget_id': budget['id'],
					'budget_name': budget['budget_name'],
					'total_amount': budget['total_amount'],
					'department': budget['department']
				},
				'timing': {
					'submitted_at': workflow_instance['submitted_at'],
					'current_step_started_at': workflow_instance['current_step_started_at'],
					'step_deadline': workflow_instance.get('step_deadline'),
					'workflow_deadline': workflow_instance.get('workflow_deadline'),
					'days_pending': (datetime.utcnow() - workflow_instance['submitted_at']).days
				},
				'approvers': {
					'pending': workflow_instance.get('pending_approvers', []),
					'completed': workflow_instance.get('completed_approvers', []),
					'rejected_by': workflow_instance.get('rejected_by', [])
				},
				'escalation': {
					'escalation_count': workflow_instance.get('escalation_count', 0),
					'last_escalation_at': workflow_instance.get('last_escalation_at'),
					'escalated_to': workflow_instance.get('escalated_to', [])
				},
				'history': {
					'approvals': approval_history,
					'escalations': escalation_history
				},
				'performance': {
					'average_response_time_hours': workflow_instance.get('average_response_time_hours'),
					'workflow_efficiency_score': workflow_instance.get('workflow_efficiency_score')
				}
			}
			
			return ServiceResponse(
				success=True,
				message="Workflow status retrieved successfully",
				data=workflow_status
			)
			
		except Exception as e:
			return self._handle_service_error(e, 'get_workflow_status')

	# =============================================================================
	# Helper Methods
	# =============================================================================

	def _initialize_workflow_engines(self) -> None:
		"""Initialize workflow processing engines."""
		self._workflow_engines = {
			'sequential': self._process_sequential_workflow,
			'parallel': self._process_parallel_workflow,
			'conditional': self._process_conditional_workflow
		}

	def _initialize_escalation_handlers(self) -> None:
		"""Initialize escalation trigger handlers."""
		self._escalation_handlers = {
			EscalationTrigger.TIMEOUT: self._handle_timeout_escalation,
			EscalationTrigger.REJECTION: self._handle_rejection_escalation,
			EscalationTrigger.MANUAL_REQUEST: self._handle_manual_escalation,
			EscalationTrigger.AMOUNT_THRESHOLD: self._handle_amount_threshold_escalation,
			EscalationTrigger.DEPARTMENT_POLICY: self._handle_department_policy_escalation
		}

	def _initialize_notification_handlers(self) -> None:
		"""Initialize notification handlers."""
		self._notification_handlers = {
			'email': self._send_email_notification,
			'sms': self._send_sms_notification,
			'push': self._send_push_notification,
			'webhook': self._send_webhook_notification
		}

	async def _validate_template_configuration(self, template: WorkflowTemplate) -> Dict[str, Any]:
		"""Validate workflow template configuration."""
		errors = []
		
		# Validate workflow steps
		for i, step in enumerate(template.workflow_steps):
			step_errors = await self._validate_workflow_step(step, i)
			errors.extend(step_errors)
		
		# Validate escalation rules
		for rule in template.escalation_rules:
			if 'trigger' not in rule or 'action' not in rule:
				errors.append("Escalation rules must specify trigger and action")
		
		# Validate amount ranges
		for range_def in template.applies_to_amount_ranges:
			if 'min_amount' not in range_def or 'max_amount' not in range_def:
				errors.append("Amount ranges must specify min_amount and max_amount")
		
		return {
			'is_valid': len(errors) == 0,
			'errors': errors
		}

	async def _validate_workflow_step(self, step: Dict[str, Any], step_index: int) -> List[str]:
		"""Validate individual workflow step configuration."""
		errors = []
		
		required_fields = ['step_id', 'step_type', 'required_approvers']
		for field in required_fields:
			if field not in step:
				errors.append(f"Step {step_index + 1}: Missing required field '{field}'")
		
		# Validate step type
		if step.get('step_type') not in [t.value for t in WorkflowStepType]:
			errors.append(f"Step {step_index + 1}: Invalid step type")
		
		# Validate approvers
		required_approvers = step.get('required_approvers', [])
		if not required_approvers:
			errors.append(f"Step {step_index + 1}: Must specify at least one required approver")
		
		return errors

	async def _check_template_conflicts(self, template: WorkflowTemplate) -> Dict[str, Any]:
		"""Check for conflicts with existing workflow templates."""
		conflicts = []
		conflicting_templates = []
		
		# Check for overlapping scope
		existing_templates = await self._connection.fetch("""
			SELECT id, template_name, applies_to_budget_types, applies_to_departments, 
				   applies_to_amount_ranges, is_default
			FROM workflow_templates
			WHERE tenant_id = $1 AND is_active = TRUE
		""", self.context.tenant_id)
		
		for existing in existing_templates:
			if self._templates_conflict(template, dict(existing)):
				conflicts.append(f"Conflicts with template '{existing['template_name']}'")
				conflicting_templates.append(existing['id'])
		
		return {
			'has_conflicts': len(conflicts) > 0,
			'conflicts': conflicts,
			'conflicting_templates': conflicting_templates
		}

	def _templates_conflict(self, new_template: WorkflowTemplate, existing: Dict[str, Any]) -> bool:
		"""Check if two templates have conflicting scope."""
		# Check budget type overlap
		new_types = set(new_template.applies_to_budget_types)
		existing_types = set(existing.get('applies_to_budget_types', []))
		if new_types & existing_types:
			# Check department overlap
			new_depts = set(new_template.applies_to_departments)
			existing_depts = set(existing.get('applies_to_departments', []))
			if not new_depts or not existing_depts or (new_depts & existing_depts):
				return True
		
		return False

	async def _insert_workflow_template(self, template: WorkflowTemplate) -> str:
		"""Insert workflow template into database."""
		template_dict = template.dict()
		columns = list(template_dict.keys())
		placeholders = [f"${i+1}" for i in range(len(columns))]
		values = list(template_dict.values())
		
		query = f"""
			INSERT INTO workflow_templates ({', '.join(columns)})
			VALUES ({', '.join(placeholders)})
			RETURNING id
		"""
		
		return await self._connection.fetchval(query, *values)

	async def _create_workflow_steps(self, template_id: str, steps: List[Dict[str, Any]]) -> None:
		"""Create workflow steps for template."""
		for step in steps:
			step_data = {
				**step,
				'template_id': template_id,
				'created_by': self.context.user_id
			}
			await self._insert_workflow_step(step_data)

	async def _insert_workflow_step(self, step_data: Dict[str, Any]) -> None:
		"""Insert workflow step into database."""
		columns = list(step_data.keys())
		placeholders = [f"${i+1}" for i in range(len(columns))]
		values = list(step_data.values())
		
		query = f"""
			INSERT INTO workflow_steps ({', '.join(columns)})
			VALUES ({', '.join(placeholders)})
		"""
		
		await self._connection.execute(query, *values)

	async def _determine_workflow_template(self, budget: Dict[str, Any]) -> Optional[Dict[str, Any]]:
		"""Determine the appropriate workflow template for a budget."""
		query = """
			SELECT wt.* FROM workflow_templates wt
			WHERE wt.tenant_id = $1 
			  AND wt.is_active = TRUE
			  AND $2 = ANY(wt.applies_to_budget_types)
			  AND (
			  	array_length(wt.applies_to_departments, 1) IS NULL 
			  	OR $3 = ANY(wt.applies_to_departments)
			  )
			ORDER BY 
				CASE WHEN wt.is_default THEN 1 ELSE 2 END,
				wt.priority ASC
			LIMIT 1
		"""
		
		result = await self._connection.fetchrow(
			query, 
			self.context.tenant_id, 
			budget['budget_type'],
			budget['department']
		)
		
		return dict(result) if result else None

	async def _resolve_step_approvers(self, step: Dict[str, Any], budget: Dict[str, Any]) -> List[str]:
		"""Resolve actual approvers for a workflow step."""
		required_approvers = step.get('required_approvers', [])
		resolved_approvers = []
		
		for approver_spec in required_approvers:
			if isinstance(approver_spec, str):
				if approver_spec.startswith('role:'):
					# Resolve by role
					role = approver_spec[5:]
					role_approvers = await self._get_approvers_by_role(role, budget['department'])
					resolved_approvers.extend(role_approvers)
				elif approver_spec.startswith('user:'):
					# Direct user reference
					user_id = approver_spec[5:]
					resolved_approvers.append(user_id)
			elif isinstance(approver_spec, dict):
				# Complex approver resolution
				spec_approvers = await self._resolve_complex_approver_spec(approver_spec, budget)
				resolved_approvers.extend(spec_approvers)
		
		return list(set(resolved_approvers))  # Remove duplicates

	async def _get_approvers_by_role(self, role: str, department: str) -> List[str]:
		"""Get approvers by role within a department."""
		# This would integrate with APG auth_rbac service
		query = """
			SELECT user_id FROM user_roles ur
			JOIN department_users du ON ur.user_id = du.user_id
			WHERE ur.role_name = $1 AND du.department = $2 AND ur.is_active = TRUE
		"""
		
		results = await self._connection.fetch(query, role, department)
		return [r['user_id'] for r in results]

	async def _insert_workflow_instance(self, instance: ApprovalWorkflowInstance) -> str:
		"""Insert workflow instance into database."""
		instance_dict = instance.dict()
		columns = list(instance_dict.keys())
		placeholders = [f"${i+1}" for i in range(len(columns))]
		values = list(instance_dict.values())
		
		query = f"""
			INSERT INTO approval_workflow_instances ({', '.join(columns)})
			VALUES ({', '.join(placeholders)})
			RETURNING workflow_instance_id
		"""
		
		return await self._connection.fetchval(query, *values)

	async def _send_approval_notifications(self, workflow_id: str, approvers: List[str]) -> None:
		"""Send approval notifications to pending approvers."""
		# This would integrate with APG notification_engine
		self.logger.info(f"Sending approval notifications for workflow {workflow_id} to {len(approvers)} approvers")


# =============================================================================
# Service Factory and Export
# =============================================================================

def create_approval_workflow_service(context: APGTenantContext, config: BFServiceConfig) -> ApprovalWorkflowService:
	"""Factory function to create approval workflow service."""
	return ApprovalWorkflowService(context, config)


# Export approval workflow classes
__all__ = [
	'ApprovalAction',
	'WorkflowStepType',
	'EscalationTrigger',
	'WorkflowTemplate',
	'ApprovalWorkflowInstance',
	'ApprovalAction',
	'WorkflowEscalation',
	'ApprovalWorkflowService',
	'create_approval_workflow_service'
]


def _log_approval_workflow_summary() -> str:
	"""Log summary of approval workflow capabilities."""
	return f"Approval Workflows loaded: {len(__all__)} components with flexible workflow engine and escalation management"