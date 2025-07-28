"""
Time & Attendance Capability Views

Pydantic v2 views and validation schemas for the revolutionary APG Time & Attendance
capability API endpoints with comprehensive validation and response formatting.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

from datetime import datetime, date, time
from decimal import Decimal
from typing import Dict, List, Any, Optional, Union, Annotated
from uuid import UUID

from pydantic import (
	BaseModel, Field, ConfigDict, AfterValidator, field_validator,
	computed_field, model_validator
)

from .models import (
	TimeEntryStatus, TimeEntryType, AttendanceStatus, BiometricType,
	DeviceType, FraudType, LeaveType, ApprovalStatus, WorkforceType,
	WorkMode, AIAgentType, ProductivityMetric, RemoteWorkStatus,
	_validate_confidence_score, _validate_geolocation
)


# Base Response Models
class APIResponse(BaseModel):
	"""Base API response model"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True,
		str_strip_whitespace=True
	)
	
	success: bool = Field(..., description="Request success status")
	message: str = Field(..., description="Response message")
	timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class SuccessResponse(APIResponse):
	"""Success response with data"""
	data: Optional[Any] = Field(None, description="Response data")


class ErrorResponse(APIResponse):
	"""Error response with details"""
	success: bool = Field(default=False, description="Always false for errors")
	error_code: int = Field(..., description="Error code")
	details: Optional[Any] = Field(None, description="Error details")


class PaginatedResponse(SuccessResponse):
	"""Paginated response model"""
	pagination: Dict[str, Any] = Field(..., description="Pagination metadata")


# Request Models for Time Tracking
class ClockInRequest(BaseModel):
	"""Clock-in request model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	employee_id: str = Field(..., min_length=1, max_length=36, description="Employee identifier")
	device_info: Dict[str, Any] = Field(default_factory=dict, description="Device information")
	location: Optional[Annotated[Dict[str, float], AfterValidator(_validate_geolocation)]] = Field(
		None, description="GPS coordinates"
	)
	biometric_data: Optional[Dict[str, Any]] = Field(None, description="Biometric authentication data")
	notes: Optional[str] = Field(None, max_length=500, description="Optional notes")


class ClockOutRequest(BaseModel):
	"""Clock-out request model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	employee_id: str = Field(..., min_length=1, max_length=36, description="Employee identifier")
	device_info: Dict[str, Any] = Field(default_factory=dict, description="Device information")
	location: Optional[Annotated[Dict[str, float], AfterValidator(_validate_geolocation)]] = Field(
		None, description="GPS coordinates"
	)
	biometric_data: Optional[Dict[str, Any]] = Field(None, description="Biometric authentication data")
	notes: Optional[str] = Field(None, max_length=500, description="Optional notes")


class TimeEntryUpdateRequest(BaseModel):
	"""Time entry update request model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	clock_in: Optional[datetime] = Field(None, description="Clock in time")
	clock_out: Optional[datetime] = Field(None, description="Clock out time")
	entry_type: Optional[TimeEntryType] = Field(None, description="Entry type")
	notes: Optional[str] = Field(None, max_length=1000, description="Entry notes")
	project_assignments: Optional[List[Dict[str, Any]]] = Field(None, description="Project allocations")


# Request Models for Remote Work
class RemoteWorkSessionRequest(BaseModel):
	"""Remote work session start request"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	employee_id: str = Field(..., min_length=1, max_length=36, description="Employee identifier")
	work_mode: WorkMode = Field(..., description="Work mode classification")
	workspace_config: Dict[str, Any] = Field(..., description="Home office configuration")
	timezone: str = Field(default="UTC", max_length=50, description="Worker timezone")
	collaboration_platforms: Optional[List[str]] = Field(None, description="Active platforms")
	
	@field_validator('workspace_config')
	@classmethod
	def validate_workspace_config(cls, v):
		"""Validate workspace configuration"""
		required_fields = ['location', 'equipment']
		for field in required_fields:
			if field not in v:
				raise ValueError(f"workspace_config must contain '{field}' field")
		return v


class ProductivityTrackingRequest(BaseModel):
	"""Remote productivity tracking request"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	employee_id: str = Field(..., min_length=1, max_length=36, description="Employee identifier")
	activity_data: Dict[str, Any] = Field(..., description="Activity and productivity data")
	metric_type: ProductivityMetric = Field(default=ProductivityMetric.TASK_COMPLETION, description="Metric type")
	timestamp: datetime = Field(default_factory=datetime.utcnow, description="Activity timestamp")


# Request Models for AI Agents
class AIAgentRegistrationRequest(BaseModel):
	"""AI agent registration request"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	agent_name: str = Field(..., min_length=1, max_length=100, description="Agent name")
	agent_type: AIAgentType = Field(..., description="Agent type")
	capabilities: List[str] = Field(..., min_items=1, description="Agent capabilities")
	configuration: Dict[str, Any] = Field(..., description="Agent configuration")
	version: str = Field(default="1.0.0", max_length=50, description="Agent version")
	environment: str = Field(default="production", max_length=100, description="Deployment environment")
	cost_per_hour: Optional[Decimal] = Field(None, ge=0, description="Hourly operational cost")
	
	@field_validator('configuration')
	@classmethod
	def validate_configuration(cls, v):
		"""Validate agent configuration"""
		required_fields = ['api_endpoints', 'resource_limits']
		for field in required_fields:
			if field not in v:
				raise ValueError(f"configuration must contain '{field}' field")
		return v


class AIAgentWorkTrackingRequest(BaseModel):
	"""AI agent work tracking request"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	task_data: Dict[str, Any] = Field(..., description="Task completion data")
	resource_consumption: Dict[str, Any] = Field(..., description="Resource usage data")
	timestamp: datetime = Field(default_factory=datetime.utcnow, description="Work timestamp")
	
	@field_validator('resource_consumption')
	@classmethod
	def validate_resource_consumption(cls, v):
		"""Validate resource consumption data"""
		expected_fields = ['cpu_hours', 'memory_gb_hours', 'api_calls']
		for field in expected_fields:
			if field not in v:
				raise ValueError(f"resource_consumption should contain '{field}' field")
		return v


# Request Models for Hybrid Collaboration
class HybridCollaborationRequest(BaseModel):
	"""Hybrid collaboration session request"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	session_name: str = Field(..., min_length=1, max_length=200, description="Session name")
	project_id: str = Field(..., min_length=1, max_length=36, description="Project identifier")
	human_participants: List[str] = Field(..., min_items=1, description="Human participant IDs")
	ai_participants: List[str] = Field(..., min_items=1, description="AI agent IDs")
	session_type: str = Field(default="collaborative_work", max_length=50, description="Session type")
	planned_duration_minutes: int = Field(default=60, ge=1, le=480, description="Planned duration")
	objectives: Optional[List[str]] = Field(None, description="Session objectives")


# Response Models for Time Tracking
class TimeEntryResponse(BaseModel):
	"""Time entry response model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	time_entry_id: str = Field(..., description="Time entry identifier")
	employee_id: str = Field(..., description="Employee identifier")
	entry_date: date = Field(..., description="Entry date")
	clock_in: Optional[datetime] = Field(None, description="Clock in timestamp")
	clock_out: Optional[datetime] = Field(None, description="Clock out timestamp")
	total_hours: Optional[Decimal] = Field(None, description="Total hours worked")
	regular_hours: Optional[Decimal] = Field(None, description="Regular hours")
	overtime_hours: Optional[Decimal] = Field(None, description="Overtime hours")
	entry_type: TimeEntryType = Field(..., description="Entry type")
	status: TimeEntryStatus = Field(..., description="Entry status")
	fraud_score: Annotated[float, AfterValidator(_validate_confidence_score)] = Field(
		default=0.0, description="AI fraud detection score"
	)
	requires_approval: bool = Field(default=False, description="Requires manager approval")
	verification_confidence: Optional[Annotated[float, AfterValidator(_validate_confidence_score)]] = Field(
		None, description="Biometric verification confidence"
	)


class RemoteWorkerResponse(BaseModel):
	"""Remote worker response model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	remote_worker_id: str = Field(..., description="Remote worker identifier")
	employee_id: str = Field(..., description="Employee identifier")
	workspace_id: str = Field(..., description="Workspace identifier")
	work_mode: WorkMode = Field(..., description="Work mode")
	current_activity: RemoteWorkStatus = Field(..., description="Current activity")
	productivity_score: Annotated[float, AfterValidator(_validate_confidence_score)] = Field(
		default=0.0, description="Overall productivity score"
	)
	work_life_balance_score: Annotated[float, AfterValidator(_validate_confidence_score)] = Field(
		default=0.8, description="Work-life balance score"
	)
	active_hours_today: Optional[Decimal] = Field(None, description="Active hours today")
	focus_time_blocks: int = Field(default=0, description="Focus time blocks completed")


class AIAgentResponse(BaseModel):
	"""AI agent response model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	ai_agent_id: str = Field(..., description="AI agent identifier")
	agent_name: str = Field(..., description="Agent name")
	agent_type: AIAgentType = Field(..., description="Agent type")
	capabilities: List[str] = Field(..., description="Agent capabilities")
	health_status: str = Field(..., description="Agent health status")
	performance_score: Annotated[float, AfterValidator(_validate_confidence_score)] = Field(
		default=0.0, description="Overall performance score"
	)
	cost_efficiency_score: Optional[float] = Field(None, description="Cost efficiency score")
	tasks_completed: int = Field(default=0, description="Total tasks completed")
	total_operational_cost: Decimal = Field(default=Decimal('0'), description="Total operational cost")
	uptime_percentage: Annotated[float, AfterValidator(_validate_confidence_score)] = Field(
		default=0.999, description="Agent uptime percentage"
	)


class HybridCollaborationResponse(BaseModel):
	"""Hybrid collaboration response model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	collaboration_id: str = Field(..., description="Collaboration session identifier")
	session_name: str = Field(..., description="Session name")
	project_id: str = Field(..., description="Project identifier")
	human_participants: List[str] = Field(..., description="Human participant IDs")
	ai_participants: List[str] = Field(..., description="AI agent IDs")
	session_lead: str = Field(..., description="Session lead ID")
	start_time: datetime = Field(..., description="Session start time")
	end_time: Optional[datetime] = Field(None, description="Session end time")
	planned_duration_minutes: int = Field(..., description="Planned duration")
	actual_duration_minutes: Optional[int] = Field(None, description="Actual duration")
	efficiency_score: Annotated[float, AfterValidator(_validate_confidence_score)] = Field(
		default=0.8, description="Session efficiency score"
	)
	collaboration_effectiveness: Optional[float] = Field(None, description="Overall effectiveness score")


# Analytics Response Models
class WorkforcePredictionsResponse(BaseModel):
	"""Workforce predictions response model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	analytics_id: str = Field(..., description="Analytics report identifier")
	analysis_type: str = Field(..., description="Type of analysis")
	prediction_period_days: int = Field(..., description="Prediction period")
	model_confidence: Annotated[float, AfterValidator(_validate_confidence_score)] = Field(
		..., description="Model confidence"
	)
	staffing_predictions: Dict[str, Any] = Field(..., description="Staffing predictions")
	cost_optimization: Dict[str, Any] = Field(..., description="Cost optimization recommendations")
	projected_savings: Optional[Decimal] = Field(None, description="Projected cost savings")
	actionable_insights: List[Dict[str, Any]] = Field(..., description="Actionable insights")
	strategic_recommendations: List[str] = Field(..., description="Strategic recommendations")


class ProductivityAnalysisResponse(BaseModel):
	"""Productivity analysis response model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	productivity_score: Annotated[float, AfterValidator(_validate_confidence_score)] = Field(
		..., description="Overall productivity score"
	)
	insights: List[Dict[str, Any]] = Field(..., description="Productivity insights")
	recommendations: List[str] = Field(..., description="Improvement recommendations")
	burnout_risk: str = Field(..., description="Burnout risk level")
	work_life_balance: Annotated[float, AfterValidator(_validate_confidence_score)] = Field(
		..., description="Work-life balance score"
	)
	trend_analysis: Dict[str, Any] = Field(default_factory=dict, description="Productivity trends")


# List Response Models
class TimeEntriesListResponse(PaginatedResponse):
	"""Time entries list response"""
	data: List[TimeEntryResponse] = Field(..., description="Time entries list")


class RemoteWorkersListResponse(PaginatedResponse):
	"""Remote workers list response"""
	data: List[RemoteWorkerResponse] = Field(..., description="Remote workers list")
	summary: Dict[str, Any] = Field(..., description="Summary statistics")


class AIAgentsListResponse(PaginatedResponse):
	"""AI agents list response"""
	data: List[AIAgentResponse] = Field(..., description="AI agents list")
	summary: Dict[str, Any] = Field(..., description="Summary statistics")


# Configuration Response Model
class ConfigurationResponse(BaseModel):
	"""Configuration response model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	environment: str = Field(..., description="Environment type")
	tracking_mode: str = Field(..., description="Time tracking mode")
	features: Dict[str, bool] = Field(..., description="Enabled features")
	performance: Dict[str, Any] = Field(..., description="Performance settings")
	compliance: Dict[str, Any] = Field(..., description="Compliance settings")


# Health Check Response Model
class HealthCheckResponse(BaseModel):
	"""Health check response model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	status: str = Field(..., description="Service health status")
	timestamp: datetime = Field(..., description="Check timestamp")
	version: str = Field(..., description="Service version")
	environment: str = Field(..., description="Environment")
	features: Dict[str, bool] = Field(..., description="Feature availability")
	dependencies: Optional[Dict[str, str]] = Field(None, description="Dependency status")


# Validation Models for Complex Operations
class BiometricAuthenticationRequest(BaseModel):
	"""Biometric authentication request"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	employee_id: str = Field(..., min_length=1, max_length=36, description="Employee identifier")
	biometric_type: BiometricType = Field(..., description="Biometric type")
	template_data: str = Field(..., min_length=32, description="Biometric template data")
	device_info: Dict[str, Any] = Field(..., description="Capture device information")
	quality_threshold: Optional[Annotated[float, AfterValidator(_validate_confidence_score)]] = Field(
		None, description="Quality threshold"
	)


class LeaveRequestSubmission(BaseModel):
	"""Leave request submission model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	employee_id: str = Field(..., min_length=1, max_length=36, description="Employee identifier")
	leave_type: LeaveType = Field(..., description="Type of leave")
	start_date: date = Field(..., description="Leave start date")
	end_date: date = Field(..., description="Leave end date")
	reason: Optional[str] = Field(None, max_length=1000, description="Leave reason")
	is_emergency: bool = Field(default=False, description="Emergency leave flag")
	supporting_documents: Optional[List[str]] = Field(None, description="Supporting document paths")
	
	@model_validator(mode='after')
	def validate_dates(self):
		"""Validate leave dates"""
		if self.end_date < self.start_date:
			raise ValueError("End date must be after start date")
		
		# Validate future dates
		if self.start_date < date.today() and not self.is_emergency:
			raise ValueError("Start date cannot be in the past for non-emergency leave")
		
		return self


class ScheduleOptimizationRequest(BaseModel):
	"""Schedule optimization request"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	schedule_period_weeks: int = Field(default=4, ge=1, le=52, description="Schedule period in weeks")
	department_ids: Optional[List[str]] = Field(None, description="Department IDs to include")
	optimization_goals: List[str] = Field(
		default_factory=lambda: ["cost_minimization", "workload_balancing"],
		description="Optimization objectives"
	)
	constraints: Dict[str, Any] = Field(default_factory=dict, description="Scheduling constraints")
	employee_preferences: Optional[Dict[str, Any]] = Field(None, description="Employee preferences")


# Bulk Operations Models
class BulkTimeEntryUpdate(BaseModel):
	"""Bulk time entry update model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	time_entry_ids: List[str] = Field(..., min_items=1, max_items=100, description="Time entry IDs")
	updates: Dict[str, Any] = Field(..., description="Updates to apply")
	approval_required: bool = Field(default=True, description="Require approval for changes")


class BulkApprovalRequest(BaseModel):
	"""Bulk approval request model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	time_entry_ids: List[str] = Field(..., min_items=1, max_items=50, description="Time entry IDs")
	action: str = Field(..., regex="^(approve|reject)$", description="Approval action")
	comments: Optional[str] = Field(None, max_length=500, description="Approval comments")


# Export all view models
__all__ = [
	# Base responses
	"APIResponse", "SuccessResponse", "ErrorResponse", "PaginatedResponse",
	
	# Request models
	"ClockInRequest", "ClockOutRequest", "TimeEntryUpdateRequest",
	"RemoteWorkSessionRequest", "ProductivityTrackingRequest",
	"AIAgentRegistrationRequest", "AIAgentWorkTrackingRequest",
	"HybridCollaborationRequest", "BiometricAuthenticationRequest",
	"LeaveRequestSubmission", "ScheduleOptimizationRequest",
	"BulkTimeEntryUpdate", "BulkApprovalRequest",
	
	# Response models
	"TimeEntryResponse", "RemoteWorkerResponse", "AIAgentResponse",
	"HybridCollaborationResponse", "WorkforcePredictionsResponse",
	"ProductivityAnalysisResponse", "ConfigurationResponse", "HealthCheckResponse",
	
	# List responses
	"TimeEntriesListResponse", "RemoteWorkersListResponse", "AIAgentsListResponse"
]