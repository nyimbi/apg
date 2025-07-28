"""
APG Security Orchestration - Pydantic Models

Enterprise security orchestration models with intelligent workflow management,
multi-system integration, and adaptive response capabilities.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator, root_validator
from pydantic import ConfigDict
from uuid_extensions import uuid7str


class PlaybookType(str, Enum):
	INCIDENT_RESPONSE = "incident_response"
	THREAT_HUNTING = "threat_hunting"
	VULNERABILITY_MANAGEMENT = "vulnerability_management"
	COMPLIANCE_CHECK = "compliance_check"
	PREVENTIVE_MAINTENANCE = "preventive_maintenance"
	CUSTOM = "custom"


class WorkflowStatus(str, Enum):
	PENDING = "pending"
	RUNNING = "running"
	PAUSED = "paused"
	COMPLETED = "completed"
	FAILED = "failed"
	CANCELLED = "cancelled"
	TIMEOUT = "timeout"


class ActionType(str, Enum):
	HTTP_REQUEST = "http_request"
	DATABASE_QUERY = "database_query"
	FILE_OPERATION = "file_operation"
	EMAIL_NOTIFICATION = "email_notification"
	SLACK_MESSAGE = "slack_message"
	SYSTEM_COMMAND = "system_command"
	SECURITY_TOOL = "security_tool"
	CUSTOM_SCRIPT = "custom_script"


class ExecutionMode(str, Enum):
	SEQUENTIAL = "sequential"
	PARALLEL = "parallel"
	CONDITIONAL = "conditional"
	LOOP = "loop"


class TriggerType(str, Enum):
	MANUAL = "manual"
	SCHEDULED = "scheduled"
	EVENT_DRIVEN = "event_driven"
	API_CALL = "api_call"
	WEBHOOK = "webhook"
	ALERT_BASED = "alert_based"


class SecurityPlaybook(BaseModel):
	"""Security automation playbook definition"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	name: str = Field(description="Playbook name")
	description: str = Field(description="Playbook description")
	version: str = Field(description="Playbook version")
	
	playbook_type: PlaybookType
	category: str = Field(description="Playbook category")
	tags: List[str] = Field(default_factory=list)
	
	# Workflow definition
	workflow_definition: Dict[str, Any] = Field(default_factory=dict)
	input_parameters: Dict[str, Any] = Field(default_factory=dict)
	output_parameters: Dict[str, Any] = Field(default_factory=dict)
	
	# Actions and steps
	actions: List[Dict[str, Any]] = Field(default_factory=list)
	dependencies: Dict[str, List[str]] = Field(default_factory=dict)
	
	# Triggers
	trigger_conditions: List[Dict[str, Any]] = Field(default_factory=list)
	trigger_type: TriggerType = TriggerType.MANUAL
	
	# Execution configuration
	execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
	timeout_minutes: int = Field(default=60)
	retry_policy: Dict[str, Any] = Field(default_factory=dict)
	
	# Error handling
	error_handling: Dict[str, Any] = Field(default_factory=dict)
	rollback_actions: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Approval requirements
	requires_approval: bool = False
	approval_required_for: List[str] = Field(default_factory=list)
	
	# Usage statistics
	execution_count: int = 0
	success_count: int = 0
	failure_count: int = 0
	average_execution_time: Optional[timedelta] = None
	
	# Validation
	is_validated: bool = False
	validation_results: Dict[str, Any] = Field(default_factory=dict)
	last_tested: Optional[datetime] = None
	
	# Metadata
	created_by: str = Field(description="Playbook creator")
	last_modified_by: Optional[str] = None
	
	is_active: bool = True
	is_published: bool = False
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class WorkflowExecution(BaseModel):
	"""Workflow execution instance and tracking"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	playbook_id: str = Field(description="Associated playbook ID")
	playbook_version: str = Field(description="Playbook version used")
	
	execution_name: str = Field(description="Execution instance name")
	
	# Trigger information
	trigger_type: TriggerType
	trigger_data: Dict[str, Any] = Field(default_factory=dict)
	triggered_by: str = Field(description="User or system that triggered execution")
	
	# Execution parameters
	input_parameters: Dict[str, Any] = Field(default_factory=dict)
	execution_context: Dict[str, Any] = Field(default_factory=dict)
	
	# Status and progress
	status: WorkflowStatus = WorkflowStatus.PENDING
	current_step: Optional[str] = None
	completed_steps: List[str] = Field(default_factory=list)
	failed_steps: List[str] = Field(default_factory=list)
	
	progress_percentage: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	# Timing
	start_time: Optional[datetime] = None
	end_time: Optional[datetime] = None
	execution_duration: Optional[timedelta] = None
	
	# Results
	output_data: Dict[str, Any] = Field(default_factory=dict)
	execution_results: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Error information
	error_details: Optional[Dict[str, Any]] = None
	error_message: Optional[str] = None
	error_stack_trace: Optional[str] = None
	
	# Approval tracking
	approval_required: bool = False
	approval_status: Optional[str] = None
	approved_by: Optional[str] = None
	approved_at: Optional[datetime] = None
	
	# Resource usage
	resource_usage: Dict[str, Any] = Field(default_factory=dict)
	
	# Execution logs
	execution_logs: List[Dict[str, Any]] = Field(default_factory=list)
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class AutomationAction(BaseModel):
	"""Individual automation action within a workflow"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	execution_id: str = Field(description="Parent workflow execution ID")
	action_name: str = Field(description="Action name")
	action_type: ActionType
	
	# Action configuration
	action_config: Dict[str, Any] = Field(default_factory=dict)
	input_data: Dict[str, Any] = Field(default_factory=dict)
	
	# Execution details
	step_number: int = Field(description="Step order in workflow")
	depends_on: List[str] = Field(default_factory=list)
	
	# Status and timing
	status: WorkflowStatus = WorkflowStatus.PENDING
	start_time: Optional[datetime] = None
	end_time: Optional[datetime] = None
	execution_time: Optional[timedelta] = None
	
	# Results
	output_data: Dict[str, Any] = Field(default_factory=dict)
	success: bool = False
	
	# Error handling
	error_message: Optional[str] = None
	retry_count: int = 0
	max_retries: int = 3
	
	# Action metadata
	action_logs: List[str] = Field(default_factory=list)
	resource_impact: Dict[str, Any] = Field(default_factory=dict)
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class ToolIntegration(BaseModel):
	"""Security tool integration configuration"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	tool_name: str = Field(description="Security tool name")
	tool_type: str = Field(description="Tool category")
	vendor: str = Field(description="Tool vendor")
	version: str = Field(description="Tool version")
	
	# Connection configuration
	connection_config: Dict[str, Any] = Field(default_factory=dict)
	authentication_config: Dict[str, Any] = Field(default_factory=dict)
	endpoint_url: str = Field(description="Tool API endpoint")
	
	# Capabilities
	supported_actions: List[str] = Field(default_factory=list)
	supported_data_types: List[str] = Field(default_factory=list)
	
	# Integration metadata
	connector_version: str = Field(description="Connector version")
	last_tested: Optional[datetime] = None
	test_results: Dict[str, Any] = Field(default_factory=dict)
	
	# Status and health
	is_active: bool = True
	is_healthy: bool = False
	health_check_interval: int = Field(default=300, description="Seconds between health checks")
	
	# Usage statistics
	action_count: int = 0
	error_count: int = 0
	average_response_time: Optional[Decimal] = None
	
	# Rate limiting
	rate_limit_config: Dict[str, Any] = Field(default_factory=dict)
	
	# Security
	encryption_enabled: bool = True
	certificate_config: Dict[str, Any] = Field(default_factory=dict)
	
	created_by: str = Field(description="Integration creator")
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class ResponseCoordination(BaseModel):
	"""Multi-team response coordination"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	incident_id: Optional[str] = None
	coordination_name: str = Field(description="Coordination name")
	coordination_type: str = Field(description="Type of coordination")
	
	# Participating teams
	teams_involved: List[str] = Field(default_factory=list)
	team_leads: Dict[str, str] = Field(default_factory=dict)
	
	# Coordination plan
	coordination_plan: Dict[str, Any] = Field(default_factory=dict)
	communication_channels: List[str] = Field(default_factory=list)
	
	# Task allocation
	task_assignments: Dict[str, List[str]] = Field(default_factory=dict)
	dependencies: Dict[str, List[str]] = Field(default_factory=dict)
	
	# Status tracking
	overall_status: WorkflowStatus = WorkflowStatus.PENDING
	team_statuses: Dict[str, str] = Field(default_factory=dict)
	
	# Progress tracking
	milestones: List[Dict[str, Any]] = Field(default_factory=list)
	completed_milestones: List[str] = Field(default_factory=list)
	
	# Communication log
	communications: List[Dict[str, Any]] = Field(default_factory=list)
	decisions_made: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Timing
	start_time: Optional[datetime] = None
	target_completion: Optional[datetime] = None
	actual_completion: Optional[datetime] = None
	
	# Escalation
	escalation_criteria: Dict[str, Any] = Field(default_factory=dict)
	escalated: bool = False
	escalation_reason: Optional[str] = None
	
	coordinator: str = Field(description="Response coordinator")
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class OrchestrationMetrics(BaseModel):
	"""Security orchestration metrics and KPIs"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	metric_period_start: datetime
	metric_period_end: datetime
	
	# Playbook metrics
	total_playbooks: int = 0
	active_playbooks: int = 0
	executed_playbooks: int = 0
	successful_executions: int = 0
	failed_executions: int = 0
	
	# Execution performance
	average_execution_time: Optional[timedelta] = None
	median_execution_time: Optional[timedelta] = None
	success_rate: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	# Automation coverage
	automated_incidents: int = 0
	manual_incidents: int = 0
	automation_rate: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	# Response times
	mean_time_to_response: Optional[timedelta] = None
	mean_time_to_containment: Optional[timedelta] = None
	mean_time_to_resolution: Optional[timedelta] = None
	
	# Tool integration metrics
	total_integrations: int = 0
	healthy_integrations: int = 0
	integration_health_rate: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	# Action metrics
	total_actions_executed: int = 0
	successful_actions: int = 0
	failed_actions: int = 0
	action_success_rate: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	# Resource utilization
	compute_resources_used: Dict[str, Decimal] = Field(default_factory=dict)
	network_bandwidth_used: Decimal = Field(default=Decimal('0.0'))
	
	# Error analysis
	error_categories: Dict[str, int] = Field(default_factory=dict)
	top_failure_reasons: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Workflow patterns
	most_used_playbooks: List[Dict[str, Any]] = Field(default_factory=list)
	peak_execution_hours: List[int] = Field(default_factory=list)
	
	# Team coordination
	coordinated_responses: int = 0
	average_teams_per_response: Decimal = Field(default=Decimal('0.0'))
	coordination_success_rate: Decimal = Field(default=Decimal('0.0'), ge=0, le=100)
	
	# Business impact
	incidents_prevented: int = 0
	time_saved_hours: Decimal = Field(default=Decimal('0.0'))
	cost_savings: Optional[Decimal] = None
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class ApprovalWorkflow(BaseModel):
	"""Approval workflow for sensitive actions"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	execution_id: str = Field(description="Associated workflow execution")
	action_id: Optional[str] = None
	
	# Approval details
	approval_type: str = Field(description="Type of approval required")
	approval_reason: str = Field(description="Reason approval is needed")
	risk_level: str = Field(description="Risk level of action")
	
	# Approval chain
	required_approvers: List[str] = Field(default_factory=list)
	approval_order: List[str] = Field(default_factory=list)
	
	# Current status
	current_approver: Optional[str] = None
	approvals_received: List[Dict[str, Any]] = Field(default_factory=list)
	rejections_received: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Timing
	requested_at: datetime = Field(default_factory=datetime.utcnow)
	approval_deadline: Optional[datetime] = None
	completed_at: Optional[datetime] = None
	
	# Final decision
	final_decision: Optional[str] = None  # approved, rejected, expired
	decision_reason: Optional[str] = None
	
	# Notifications
	notification_sent: bool = False
	reminder_count: int = 0
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None


class WorkflowTemplate(BaseModel):
	"""Reusable workflow template"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(description="Tenant identifier")
	
	template_name: str = Field(description="Template name")
	template_type: PlaybookType
	category: str = Field(description="Template category")
	
	# Template definition
	template_definition: Dict[str, Any] = Field(default_factory=dict)
	parameter_schema: Dict[str, Any] = Field(default_factory=dict)
	
	# Usage information
	usage_count: int = 0
	rating: Optional[Decimal] = None
	reviews: List[Dict[str, Any]] = Field(default_factory=list)
	
	# Template metadata
	is_community_template: bool = False
	is_certified: bool = False
	certification_level: Optional[str] = None
	
	# Versioning
	template_version: str = Field(description="Template version")
	changelog: List[Dict[str, Any]] = Field(default_factory=list)
	
	created_by: str = Field(description="Template creator")
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: Optional[datetime] = None