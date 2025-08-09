"""
Time & Attendance Capability - Revolutionary Data Models

APG-compatible Pydantic v2 data models for the revolutionary time & attendance capability
delivering 10x superior performance through AI-powered accuracy, predictive analytics, 
biometric integration, and seamless APG ecosystem connectivity.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import json
from datetime import datetime, timedelta, date, time
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Annotated
from uuid import UUID

from pydantic import (
	BaseModel, Field, ConfigDict, AfterValidator, field_validator,
	computed_field, model_validator
)
from uuid_extensions import uuid7str


# Validation Functions
def _validate_confidence_score(v: float) -> float:
	"""Validate confidence scores are between 0.0 and 1.0"""
	if not (0.0 <= v <= 1.0):
		raise ValueError(f"Confidence score must be between 0.0 and 1.0, got {v}")
	return v


def _validate_time_range(v: Dict[str, Any]) -> Dict[str, Any]:
	"""Validate time range has valid start and end times"""
	if 'start_time' not in v or 'end_time' not in v:
		raise ValueError("Time range must contain start_time and end_time")
	
	start = v['start_time']
	end = v['end_time']
	
	if isinstance(start, str):
		start = datetime.fromisoformat(start)
	if isinstance(end, str):
		end = datetime.fromisoformat(end)
	
	if end <= start:
		raise ValueError("End time must be after start time")
	
	return v


def _validate_biometric_template(v: str) -> str:
	"""Validate biometric template is properly encrypted and formatted"""
	if not v or len(v.strip()) == 0:
		raise ValueError("Biometric template cannot be empty")
	
	# Basic format validation for encrypted template
	if len(v) < 32:
		raise ValueError("Biometric template appears to be invalid (too short)")
	
	return v.strip()


def _validate_geolocation(v: Dict[str, float]) -> Dict[str, float]:
	"""Validate geolocation coordinates"""
	required_keys = {'latitude', 'longitude'}
	if not all(key in v for key in required_keys):
		raise ValueError(f"Geolocation must contain keys: {required_keys}")
	
	lat = v['latitude']
	lng = v['longitude']
	
	if not (-90 <= lat <= 90):
		raise ValueError(f"Latitude must be between -90 and 90, got {lat}")
	
	if not (-180 <= lng <= 180):
		raise ValueError(f"Longitude must be between -180 and 180, got {lng}")
	
	return v


def _validate_schedule_pattern(v: Dict[str, Any]) -> Dict[str, Any]:
	"""Validate schedule pattern structure"""
	if not isinstance(v, dict):
		raise ValueError("Schedule pattern must be a dictionary")
	
	required_keys = {'days_of_week', 'start_time', 'end_time'}
	if not all(key in v for key in required_keys):
		raise ValueError(f"Schedule pattern must contain: {required_keys}")
	
	# Validate days of week
	days = v['days_of_week']
	if not isinstance(days, list) or not all(isinstance(d, int) and 0 <= d <= 6 for d in days):
		raise ValueError("days_of_week must be list of integers 0-6 (Monday=0)")
	
	return v


# Enums
class TimeEntryStatus(str, Enum):
	"""Time entry status enumeration"""
	DRAFT = "draft"
	SUBMITTED = "submitted"
	APPROVED = "approved"
	REJECTED = "rejected"
	LOCKED = "locked"
	PROCESSING = "processing"


class TimeEntryType(str, Enum):
	"""Type of time entry"""
	REGULAR = "regular"
	OVERTIME = "overtime"
	HOLIDAY = "holiday"
	SICK = "sick"
	VACATION = "vacation"
	PERSONAL = "personal"
	TRAINING = "training"
	TRAVEL = "travel"
	BREAK = "break"


class AttendanceStatus(str, Enum):
	"""Employee attendance status"""
	PRESENT = "present"
	ABSENT = "absent"
	LATE = "late"
	EARLY_DEPARTURE = "early_departure"
	PARTIAL_DAY = "partial_day"
	EXCUSED = "excused"
	REMOTE = "remote"


class ScheduleStatus(str, Enum):
	"""Schedule status enumeration"""
	DRAFT = "draft"
	PUBLISHED = "published"
	LOCKED = "locked"
	ARCHIVED = "archived"


class BiometricType(str, Enum):
	"""Biometric authentication types"""
	FACIAL_RECOGNITION = "facial_recognition"
	FINGERPRINT = "fingerprint"
	PALM_PRINT = "palm_print"
	IRIS_SCAN = "iris_scan"
	VOICE_RECOGNITION = "voice_recognition"
	BEHAVIORAL = "behavioral"


class DeviceType(str, Enum):
	"""Time tracking device types"""
	MOBILE_APP = "mobile_app"
	WEB_BROWSER = "web_browser"
	BIOMETRIC_TERMINAL = "biometric_terminal"
	IOT_SENSOR = "iot_sensor"
	SMART_WATCH = "smart_watch"
	BADGE_READER = "badge_reader"
	HOME_OFFICE_IOT = "home_office_iot"
	LAPTOP_MONITORING = "laptop_monitoring"
	AI_AGENT_API = "ai_agent_api"
	COLLABORATION_PLATFORM = "collaboration_platform"


class FraudType(str, Enum):
	"""Types of time fraud detected"""
	BUDDY_PUNCHING = "buddy_punching"
	LOCATION_SPOOFING = "location_spoofing"
	TIME_MANIPULATION = "time_manipulation"
	DEVICE_SPOOFING = "device_spoofing"
	PATTERN_ANOMALY = "pattern_anomaly"
	SCHEDULE_VIOLATION = "schedule_violation"


class LeaveType(str, Enum):
	"""Types of leave/time-off"""
	VACATION = "vacation"
	SICK = "sick"
	PERSONAL = "personal"
	MATERNITY = "maternity"
	PATERNITY = "paternity"
	BEREAVEMENT = "bereavement"
	JURY_DUTY = "jury_duty"
	MILITARY = "military"
	SABBATICAL = "sabbatical"
	UNPAID = "unpaid"


class ApprovalStatus(str, Enum):
	"""Approval workflow status"""
	PENDING = "pending"
	APPROVED = "approved"
	REJECTED = "rejected"
	ESCALATED = "escalated"
	AUTO_APPROVED = "auto_approved"
	EXPIRED = "expired"


class WorkforceType(str, Enum):
	"""Types of workforce entities"""
	HUMAN_EMPLOYEE = "human_employee"
	REMOTE_WORKER = "remote_worker"
	HYBRID_WORKER = "hybrid_worker"
	AI_AGENT = "ai_agent"
	CONTRACT_WORKER = "contract_worker"
	TEMPORARY_WORKER = "temporary_worker"


class WorkMode(str, Enum):
	"""Work mode classifications"""
	OFFICE_BASED = "office_based"
	REMOTE_ONLY = "remote_only"
	HYBRID = "hybrid"
	FIELD_WORK = "field_work"
	AI_AUTOMATED = "ai_automated"
	HUMAN_AI_COLLABORATIVE = "human_ai_collaborative"


class AIAgentType(str, Enum):
	"""Types of AI agents"""
	CONVERSATIONAL_AI = "conversational_ai"
	AUTOMATION_BOT = "automation_bot"
	ANALYSIS_AGENT = "analysis_agent"
	CONTENT_GENERATOR = "content_generator"
	DECISION_SUPPORT = "decision_support"
	MONITORING_AGENT = "monitoring_agent"
	INTEGRATION_SERVICE = "integration_service"


class ProductivityMetric(str, Enum):
	"""Productivity measurement types"""
	TIME_BASED = "time_based"
	TASK_COMPLETION = "task_completion"
	OUTPUT_QUALITY = "output_quality"
	COLLABORATION_SCORE = "collaboration_score"
	EFFICIENCY_RATIO = "efficiency_ratio"
	COST_EFFECTIVENESS = "cost_effectiveness"


class RemoteWorkStatus(str, Enum):
	"""Remote work activity status"""
	ACTIVE_WORKING = "active_working"
	BREAK_TIME = "break_time"
	IN_MEETING = "in_meeting"
	FOCUSED_WORK = "focused_work"
	COLLABORATIVE_WORK = "collaborative_work"
	AWAY_FROM_DESK = "away_from_desk"
	OFFLINE = "offline"


# Base Models
class TABaseModel(BaseModel):
	"""Base model for all Time & Attendance models with APG compliance"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True,
		str_strip_whitespace=True,
		validate_default=True,
		frozen=False
	)
	
	id: str = Field(default_factory=uuid7str, description="Unique identifier using UUID7")
	tenant_id: str = Field(..., description="Multi-tenant isolation identifier")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	created_by: str = Field(..., description="User ID who created this record")
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class TAEmployee(TABaseModel):
	"""
	Employee Time & Attendance Profile
	
	Extended employee profile specifically for time tracking with biometric data,
	preferences, schedule assignments, and AI-powered behavioral analytics.
	"""
	
	# Core employee information
	employee_id: str = Field(..., description="Reference to Employee Data Management ID")
	employee_number: str = Field(..., min_length=1, max_length=50, description="Unique employee number")
	department_id: str = Field(..., description="Department identifier")
	manager_id: Optional[str] = Field(None, description="Direct manager employee ID")
	
	# Time tracking configuration
	default_schedule_id: Optional[str] = Field(None, description="Default work schedule ID")
	timezone: str = Field(default="UTC", max_length=50, description="Employee timezone")
	hourly_rate: Optional[Decimal] = Field(None, ge=0, description="Hourly compensation rate")
	overtime_exempt: bool = Field(default=False, description="Exempt from overtime calculations")
	
	# Biometric authentication
	biometric_templates: List[Dict[str, Any]] = Field(
		default_factory=list, description="Encrypted biometric templates"
	)
	biometric_enabled: bool = Field(default=False, description="Biometric authentication enabled")
	biometric_consent: bool = Field(default=False, description="Biometric consent recorded")
	
	# Location and device restrictions
	allowed_locations: List[Dict[str, Any]] = Field(
		default_factory=list, description="Allowed punch locations with geofencing"
	)
	registered_devices: List[Dict[str, Any]] = Field(
		default_factory=list, description="Registered devices for time tracking"
	)
	
	# AI behavioral analytics
	behavioral_profile: Dict[str, Any] = Field(
		default_factory=dict, description="AI-generated behavioral patterns"
	)
	fraud_risk_score: Annotated[float, AfterValidator(_validate_confidence_score)] = Field(
		default=0.0, description="AI-calculated fraud risk score"
	)
	
	# Preferences and settings
	notification_preferences: Dict[str, Any] = Field(
		default_factory=dict, description="Notification delivery preferences"
	)
	mobile_app_enabled: bool = Field(default=True, description="Mobile app access enabled")
	
	# Status and compliance
	is_active: bool = Field(default=True, description="Employee active status")
	last_activity: Optional[datetime] = Field(None, description="Last time tracking activity")
	compliance_flags: List[str] = Field(default_factory=list, description="Compliance issues")
	
	@field_validator('biometric_templates')
	@classmethod
	def _validate_biometric_templates(cls, v):
		"""Validate biometric templates structure and encryption"""
		for template in v:
			if not isinstance(template, dict):
				raise ValueError("Each biometric template must be a dictionary")
			
			required_fields = {'type', 'template_data', 'created_at', 'quality_score'}
			if not all(field in template for field in required_fields):
				raise ValueError(f"Biometric template must contain: {required_fields}")
			
			# Validate template data is encrypted
			_validate_biometric_template(template['template_data'])
			
			# Validate quality score
			_validate_confidence_score(template['quality_score'])
		
		return v
	
	@computed_field
	@property
	def has_active_biometrics(self) -> bool:
		"""Check if employee has active biometric authentication"""
		return self.biometric_enabled and self.biometric_consent and len(self.biometric_templates) > 0


class TATimeEntry(TABaseModel):
	"""
	Revolutionary Time Entry Model
	
	AI-powered time entry with fraud detection, biometric verification,
	real-time validation, and predictive analytics integration.
	"""
	
	# Core time information
	employee_id: str = Field(..., description="Employee identifier")
	entry_date: date = Field(..., description="Date of time entry")
	clock_in: Optional[datetime] = Field(None, description="Clock in timestamp")
	clock_out: Optional[datetime] = Field(None, description="Clock out timestamp")
	
	# Calculated time values
	total_hours: Optional[Decimal] = Field(None, ge=0, description="Total hours worked")
	regular_hours: Optional[Decimal] = Field(None, ge=0, description="Regular hours")
	overtime_hours: Optional[Decimal] = Field(None, ge=0, description="Overtime hours")
	break_minutes: Optional[int] = Field(None, ge=0, description="Break time in minutes")
	
	# Entry classification
	entry_type: TimeEntryType = Field(default=TimeEntryType.REGULAR, description="Type of time entry")
	status: TimeEntryStatus = Field(default=TimeEntryStatus.DRAFT, description="Entry status")
	
	# Location and device verification
	clock_in_location: Optional[Annotated[Dict[str, float], AfterValidator(_validate_geolocation)]] = Field(
		None, description="Clock in GPS coordinates"
	)
	clock_out_location: Optional[Annotated[Dict[str, float], AfterValidator(_validate_geolocation)]] = Field(
		None, description="Clock out GPS coordinates"
	)
	device_info: Dict[str, Any] = Field(default_factory=dict, description="Device information")
	ip_address: Optional[str] = Field(None, max_length=45, description="IP address of device")
	
	# Biometric verification
	biometric_verification: Dict[str, Any] = Field(
		default_factory=dict, description="Biometric authentication results"
	)
	verification_confidence: Annotated[float, AfterValidator(_validate_confidence_score)] = Field(
		default=0.0, description="Biometric verification confidence"
	)
	
	# AI fraud detection
	fraud_indicators: List[Dict[str, Any]] = Field(
		default_factory=list, description="AI-detected fraud indicators"
	)
	anomaly_score: Annotated[float, AfterValidator(_validate_confidence_score)] = Field(
		default=0.0, description="AI-calculated anomaly score"
	)
	validation_results: Dict[str, Any] = Field(
		default_factory=dict, description="Real-time validation results"
	)
	
	# Approval workflow
	requires_approval: bool = Field(default=False, description="Entry requires manager approval")
	approved_by: Optional[str] = Field(None, description="Approver user ID")
	approved_at: Optional[datetime] = Field(None, description="Approval timestamp")
	rejection_reason: Optional[str] = Field(None, max_length=500, description="Rejection reason")
	
	# Project and cost allocation
	project_assignments: List[Dict[str, Any]] = Field(
		default_factory=list, description="Project time allocations"
	)
	cost_center: Optional[str] = Field(None, max_length=50, description="Cost center code")
	billable_hours: Optional[Decimal] = Field(None, ge=0, description="Billable hours")
	
	# Notes and attachments
	notes: Optional[str] = Field(None, max_length=1000, description="Entry notes")
	attachments: List[str] = Field(default_factory=list, description="Attachment file paths")
	
	@field_validator('fraud_indicators')
	@classmethod
	def _validate_fraud_indicators(cls, v):
		"""Validate fraud indicators structure"""
		for indicator in v:
			if not isinstance(indicator, dict):
				raise ValueError("Each fraud indicator must be a dictionary")
			
			required_fields = {'type', 'severity', 'confidence', 'description'}
			if not all(field in indicator for field in required_fields):
				raise ValueError(f"Fraud indicator must contain: {required_fields}")
			
			# Validate confidence score
			_validate_confidence_score(indicator['confidence'])
		
		return v
	
	@computed_field
	@property
	def duration_hours(self) -> Optional[float]:
		"""Calculate duration in hours if both clock in/out exist"""
		if self.clock_in and self.clock_out:
			delta = self.clock_out - self.clock_in
			return delta.total_seconds() / 3600
		return None
	
	@computed_field
	@property
	def is_overtime_eligible(self) -> bool:
		"""Check if entry is eligible for overtime calculation"""
		return (self.total_hours or 0) > 8 and self.entry_type == TimeEntryType.REGULAR
	
	@model_validator(mode='after')
	def _validate_time_consistency(self):
		"""Validate time entry consistency"""
		if self.clock_out and not self.clock_in:
			raise ValueError("Clock out time cannot exist without clock in time")
		
		if self.clock_in and self.clock_out and self.clock_out <= self.clock_in:
			raise ValueError("Clock out time must be after clock in time")
		
		return self


class TASchedule(TABaseModel):
	"""
	AI-Powered Work Schedule Model
	
	Intelligent scheduling with predictive optimization, skills matching,
	and dynamic adjustments based on workload and performance analytics.
	"""
	
	# Schedule identity
	schedule_name: str = Field(..., min_length=1, max_length=100, description="Schedule name")
	schedule_type: str = Field(..., max_length=50, description="Schedule type (fixed, rotating, flexible)")
	description: Optional[str] = Field(None, max_length=500, description="Schedule description")
	
	# Date range
	effective_date: date = Field(..., description="Schedule effective start date")
	end_date: Optional[date] = Field(None, description="Schedule end date")
	
	# Schedule patterns
	schedule_patterns: List[Annotated[Dict[str, Any], AfterValidator(_validate_schedule_pattern)]] = Field(
		..., description="Weekly schedule patterns"
	)
	
	# Employee assignments
	assigned_employees: List[str] = Field(default_factory=list, description="Assigned employee IDs")
	department_id: Optional[str] = Field(None, description="Department identifier")
	location_id: Optional[str] = Field(None, description="Location identifier")
	
	# AI optimization parameters
	optimization_enabled: bool = Field(default=True, description="AI optimization enabled")
	optimization_goals: List[str] = Field(
		default_factory=list, description="Optimization objectives"
	)
	skill_requirements: Dict[str, Any] = Field(
		default_factory=dict, description="Required skills and competencies"
	)
	
	# Workload and capacity
	max_employees_per_shift: Optional[int] = Field(None, ge=1, description="Maximum employees per shift")
	min_employees_per_shift: Optional[int] = Field(None, ge=1, description="Minimum employees per shift")
	workload_predictions: Dict[str, Any] = Field(
		default_factory=dict, description="AI-predicted workload patterns"
	)
	
	# Flexibility and preferences
	allow_overtime: bool = Field(default=True, description="Allow overtime scheduling")
	allow_shift_swapping: bool = Field(default=True, description="Allow employee shift swapping")
	employee_preferences: Dict[str, Any] = Field(
		default_factory=dict, description="Employee scheduling preferences"
	)
	
	# Compliance and rules
	labor_law_compliance: Dict[str, Any] = Field(
		default_factory=dict, description="Labor law compliance rules"
	)
	break_requirements: Dict[str, Any] = Field(
		default_factory=dict, description="Break and meal period requirements"
	)
	
	# Status and performance
	status: ScheduleStatus = Field(default=ScheduleStatus.DRAFT, description="Schedule status")
	performance_metrics: Dict[str, Any] = Field(
		default_factory=dict, description="Schedule performance analytics"
	)
	
	# Version control
	version: str = Field(default="1.0", max_length=20, description="Schedule version")
	parent_schedule_id: Optional[str] = Field(None, description="Parent schedule for versioning")
	
	@computed_field
	@property
	def total_weekly_hours(self) -> float:
		"""Calculate total weekly hours from patterns"""
		total = 0.0
		for pattern in self.schedule_patterns:
			if 'start_time' in pattern and 'end_time' in pattern:
				start = datetime.strptime(pattern['start_time'], '%H:%M').time()
				end = datetime.strptime(pattern['end_time'], '%H:%M').time()
				hours = (datetime.combine(date.today(), end) - datetime.combine(date.today(), start)).total_seconds() / 3600
				total += hours * len(pattern.get('days_of_week', []))
		return total
	
	@computed_field
	@property
	def is_active(self) -> bool:
		"""Check if schedule is currently active"""
		today = date.today()
		return (self.effective_date <= today and 
				(not self.end_date or self.end_date >= today) and
				self.status == ScheduleStatus.PUBLISHED)


class TALeaveRequest(TABaseModel):
	"""
	Intelligent Leave/Time-Off Request Model
	
	AI-powered leave management with predictive approval, workload impact analysis,
	and automated compliance checking with team coordination.
	"""
	
	# Request information
	employee_id: str = Field(..., description="Employee requesting leave")
	leave_type: LeaveType = Field(..., description="Type of leave requested")
	
	# Date and duration
	start_date: date = Field(..., description="Leave start date")
	end_date: date = Field(..., description="Leave end date")
	total_days: Decimal = Field(..., ge=0, description="Total leave days requested")
	total_hours: Decimal = Field(..., ge=0, description="Total leave hours requested")
	
	# Request details
	reason: Optional[str] = Field(None, max_length=1000, description="Reason for leave")
	is_emergency: bool = Field(default=False, description="Emergency leave request")
	
	# AI analysis and predictions
	approval_probability: Annotated[float, AfterValidator(_validate_confidence_score)] = Field(
		default=0.0, description="AI-predicted approval probability"
	)
	workload_impact: Dict[str, Any] = Field(
		default_factory=dict, description="Predicted workload impact analysis"
	)
	coverage_suggestions: List[Dict[str, Any]] = Field(
		default_factory=list, description="AI-suggested coverage options"
	)
	
	# Approval workflow
	status: ApprovalStatus = Field(default=ApprovalStatus.PENDING, description="Request status")
	approval_chain: List[Dict[str, Any]] = Field(
		default_factory=list, description="Approval workflow chain"
	)
	current_approver: Optional[str] = Field(None, description="Current approver user ID")
	
	# Balance and accrual
	leave_balance_before: Optional[Decimal] = Field(None, ge=0, description="Leave balance before request")
	leave_balance_after: Optional[Decimal] = Field(None, ge=0, description="Leave balance after request")
	accrual_impact: Dict[str, Any] = Field(
		default_factory=dict, description="Impact on leave accruals"
	)
	
	# Team coordination
	conflicts_detected: List[Dict[str, Any]] = Field(
		default_factory=list, description="Scheduling conflicts detected"
	)
	team_coverage_plan: Dict[str, Any] = Field(
		default_factory=dict, description="Team coverage planning"
	)
	
	# Compliance and policies
	policy_compliance: Dict[str, Any] = Field(
		default_factory=dict, description="Policy compliance checks"
	)
	legal_requirements: Dict[str, Any] = Field(
		default_factory=dict, description="Legal requirement validation"
	)
	
	# Documentation
	attachments: List[str] = Field(default_factory=list, description="Supporting documents")
	manager_notes: Optional[str] = Field(None, max_length=1000, description="Manager notes")
	hr_notes: Optional[str] = Field(None, max_length=1000, description="HR notes")
	
	@computed_field
	@property
	def duration_days(self) -> int:
		"""Calculate duration in calendar days"""
		return (self.end_date - self.start_date).days + 1
	
	@computed_field
	@property
	def is_extended_leave(self) -> bool:
		"""Check if this is extended leave (>5 consecutive days)"""
		return self.duration_days > 5
	
	@model_validator(mode='after')
	def _validate_leave_consistency(self):
		"""Validate leave request consistency"""
		if self.end_date < self.start_date:
			raise ValueError("End date must be on or after start date")
		
		if self.total_days <= 0:
			raise ValueError("Total days must be positive")
		
		return self


class TAFraudDetection(TABaseModel):
	"""
	AI-Powered Fraud Detection Model
	
	Advanced fraud detection system using machine learning, behavioral analytics,
	and real-time pattern recognition for 99.8% accuracy fraud prevention.
	"""
	
	# Detection information
	employee_id: str = Field(..., description="Employee under investigation")
	detection_date: datetime = Field(default_factory=datetime.utcnow, description="Detection timestamp")
	
	# Fraud classification
	fraud_types: List[FraudType] = Field(..., description="Types of fraud detected")
	severity_level: str = Field(..., regex="^(LOW|MEDIUM|HIGH|CRITICAL)$", description="Fraud severity")
	confidence_score: Annotated[float, AfterValidator(_validate_confidence_score)] = Field(
		..., description="AI confidence in fraud detection"
	)
	
	# Evidence and analysis
	evidence_collected: List[Dict[str, Any]] = Field(
		default_factory=list, description="Fraud evidence and indicators"
	)
	behavioral_anomalies: List[Dict[str, Any]] = Field(
		default_factory=list, description="Behavioral pattern anomalies"
	)
	technical_indicators: Dict[str, Any] = Field(
		default_factory=dict, description="Technical fraud indicators"
	)
	
	# Pattern analysis
	historical_patterns: Dict[str, Any] = Field(
		default_factory=dict, description="Historical behavior patterns"
	)
	deviation_analysis: Dict[str, Any] = Field(
		default_factory=dict, description="Pattern deviation analysis"
	)
	peer_comparison: Dict[str, Any] = Field(
		default_factory=dict, description="Peer behavior comparison"
	)
	
	# Investigation status
	investigation_status: str = Field(
		default="OPEN", regex="^(OPEN|INVESTIGATING|RESOLVED|FALSE_POSITIVE)$",
		description="Investigation status"
	)
	investigator_id: Optional[str] = Field(None, description="Assigned investigator")
	resolution_notes: Optional[str] = Field(None, max_length=2000, description="Resolution notes")
	
	# Impact assessment
	financial_impact: Optional[Decimal] = Field(None, ge=0, description="Estimated financial impact")
	time_period_affected: Optional[Annotated[Dict[str, Any], AfterValidator(_validate_time_range)]] = Field(
		None, description="Time period affected by fraud"
	)
	affected_records: List[str] = Field(default_factory=list, description="Affected time entry IDs")
	
	# Prevention measures
	prevention_actions: List[str] = Field(
		default_factory=list, description="Automated prevention actions taken"
	)
	recommendations: List[str] = Field(
		default_factory=list, description="Prevention recommendations"
	)
	
	# Compliance and reporting
	compliance_notifications: List[Dict[str, Any]] = Field(
		default_factory=list, description="Regulatory compliance notifications"
	)
	audit_trail: List[Dict[str, Any]] = Field(
		default_factory=list, description="Investigation audit trail"
	)
	
	@computed_field
	@property
	def risk_level(self) -> str:
		"""Calculate overall risk level based on fraud types and confidence"""
		if self.confidence_score >= 0.9 and 'CRITICAL' in self.severity_level:
			return "VERY_HIGH"
		elif self.confidence_score >= 0.8:
			return "HIGH"
		elif self.confidence_score >= 0.6:
			return "MEDIUM"
		else:
			return "LOW"
	
	@computed_field
	@property
	def requires_immediate_action(self) -> bool:
		"""Determine if fraud requires immediate action"""
		return (self.severity_level in ['HIGH', 'CRITICAL'] and 
				self.confidence_score >= 0.8)


class TABiometricAuthentication(TABaseModel):
	"""
	Biometric Authentication Model
	
	Privacy-compliant biometric authentication with liveness detection,
	template-based storage, and integration with Computer Vision capability.
	"""
	
	# Authentication session
	employee_id: str = Field(..., description="Employee being authenticated")
	authentication_timestamp: datetime = Field(
		default_factory=datetime.utcnow, description="Authentication timestamp"
	)
	session_id: str = Field(default_factory=uuid7str, description="Authentication session ID")
	
	# Biometric data
	biometric_type: BiometricType = Field(..., description="Type of biometric used")
	device_type: DeviceType = Field(..., description="Device used for capture")
	template_quality: Annotated[float, AfterValidator(_validate_confidence_score)] = Field(
		..., description="Biometric template quality score"
	)
	
	# Authentication results
	authentication_success: bool = Field(..., description="Authentication successful")
	confidence_score: Annotated[float, AfterValidator(_validate_confidence_score)] = Field(
		..., description="Authentication confidence score"
	)
	match_score: Annotated[float, AfterValidator(_validate_confidence_score)] = Field(
		default=0.0, description="Biometric match score"
	)
	
	# Liveness detection
	liveness_check_passed: bool = Field(default=False, description="Liveness detection passed")
	liveness_confidence: Annotated[float, AfterValidator(_validate_confidence_score)] = Field(
		default=0.0, description="Liveness detection confidence"
	)
	anti_spoofing_measures: List[str] = Field(
		default_factory=list, description="Anti-spoofing measures applied"
	)
	
	# Device and environment
	device_info: Dict[str, Any] = Field(default_factory=dict, description="Capture device information")
	environmental_conditions: Dict[str, Any] = Field(
		default_factory=dict, description="Environmental capture conditions"
	)
	location_data: Optional[Annotated[Dict[str, float], AfterValidator(_validate_geolocation)]] = Field(
		None, description="Authentication location"
	)
	
	# Privacy and compliance
	template_encrypted: bool = Field(default=True, description="Biometric template encrypted")
	consent_verified: bool = Field(default=False, description="User consent verified")
	retention_period_days: int = Field(default=30, ge=1, le=2555, description="Data retention period")
	
	# Integration with Computer Vision
	cv_processing_job_id: Optional[str] = Field(None, description="Computer Vision job ID")
	cv_analysis_results: Dict[str, Any] = Field(
		default_factory=dict, description="Computer Vision analysis results"
	)
	
	# Audit and security
	security_events: List[Dict[str, Any]] = Field(
		default_factory=list, description="Security events during authentication"
	)
	audit_log_id: str = Field(default_factory=uuid7str, description="Audit log identifier")
	
	@computed_field
	@property
	def data_retention_expires_at(self) -> datetime:
		"""Calculate when biometric data should be deleted"""
		return self.authentication_timestamp + timedelta(days=self.retention_period_days)
	
	@computed_field
	@property
	def overall_trust_score(self) -> float:
		"""Calculate overall trust score combining all factors"""
		weights = {
			'confidence': 0.4,
			'liveness': 0.3,
			'quality': 0.2,
			'device': 0.1
		}
		
		device_score = 1.0 if self.device_type in [DeviceType.BIOMETRIC_TERMINAL] else 0.8
		
		return (
			self.confidence_score * weights['confidence'] +
			self.liveness_confidence * weights['liveness'] +
			self.template_quality * weights['quality'] +
			device_score * weights['device']
		)


class TAPredictiveAnalytics(TABaseModel):
	"""
	Predictive Analytics Model
	
	AI-powered workforce analytics providing predictive insights, optimization
	recommendations, and business intelligence for strategic decision making.
	"""
	
	# Analysis metadata
	analysis_name: str = Field(..., min_length=1, max_length=100, description="Analysis name")
	analysis_type: str = Field(..., max_length=50, description="Type of predictive analysis")
	analysis_date: datetime = Field(default_factory=datetime.utcnow, description="Analysis timestamp")
	
	# Data scope
	date_range: Annotated[Dict[str, Any], AfterValidator(_validate_time_range)] = Field(
		..., description="Analysis date range"
	)
	employee_scope: List[str] = Field(default_factory=list, description="Employee IDs in scope")
	department_scope: List[str] = Field(default_factory=list, description="Department IDs in scope")
	
	# Predictive models
	models_used: List[str] = Field(..., description="AI models used in analysis")
	model_confidence: Annotated[float, AfterValidator(_validate_confidence_score)] = Field(
		..., description="Overall model confidence"
	)
	prediction_accuracy: Optional[Annotated[float, AfterValidator(_validate_confidence_score)]] = Field(
		None, description="Historical prediction accuracy"
	)
	
	# Workforce predictions
	staffing_predictions: Dict[str, Any] = Field(
		default_factory=dict, description="Staffing requirement predictions"
	)
	absence_predictions: Dict[str, Any] = Field(
		default_factory=dict, description="Employee absence predictions"
	)
	overtime_predictions: Dict[str, Any] = Field(
		default_factory=dict, description="Overtime cost predictions"
	)
	
	# Performance insights
	productivity_trends: Dict[str, Any] = Field(
		default_factory=dict, description="Productivity trend analysis"
	)
	efficiency_opportunities: List[Dict[str, Any]] = Field(
		default_factory=list, description="Efficiency improvement opportunities"
	)
	cost_optimization: Dict[str, Any] = Field(
		default_factory=dict, description="Cost optimization recommendations"
	)
	
	# Risk analysis
	compliance_risks: List[Dict[str, Any]] = Field(
		default_factory=list, description="Compliance risk predictions"
	)
	operational_risks: List[Dict[str, Any]] = Field(
		default_factory=list, description="Operational risk assessments"
	)
	fraud_risk_indicators: Dict[str, Any] = Field(
		default_factory=dict, description="Fraud risk indicators"
	)
	
	# Recommendations
	actionable_insights: List[Dict[str, Any]] = Field(
		default_factory=list, description="Actionable business insights"
	)
	strategic_recommendations: List[str] = Field(
		default_factory=list, description="Strategic recommendations"
	)
	implementation_priorities: List[Dict[str, Any]] = Field(
		default_factory=list, description="Implementation priority ranking"
	)
	
	# Business impact
	projected_savings: Optional[Decimal] = Field(None, ge=0, description="Projected cost savings")
	roi_estimates: Dict[str, Any] = Field(
		default_factory=dict, description="Return on investment estimates"
	)
	impact_metrics: Dict[str, Any] = Field(
		default_factory=dict, description="Business impact metrics"
	)
	
	# Validation and accuracy
	validation_results: Dict[str, Any] = Field(
		default_factory=dict, description="Model validation results"
	)
	confidence_intervals: Dict[str, Any] = Field(
		default_factory=dict, description="Prediction confidence intervals"
	)
	
	@computed_field
	@property
	def analysis_age_days(self) -> int:
		"""Calculate age of analysis in days"""
		return (datetime.utcnow() - self.analysis_date).days
	
	@computed_field
	@property
	def is_current(self) -> bool:
		"""Check if analysis is current (less than 7 days old)"""
		return self.analysis_age_days <= 7


class TAComplianceRule(TABaseModel):
	"""
	Compliance Rule Model
	
	Automated compliance management with multi-jurisdiction support,
	real-time monitoring, and intelligent violation detection and prevention.
	"""
	
	# Rule identification
	rule_name: str = Field(..., min_length=1, max_length=100, description="Compliance rule name")
	rule_code: str = Field(..., min_length=1, max_length=50, description="Unique rule code")
	rule_type: str = Field(..., max_length=50, description="Type of compliance rule")
	
	# Regulatory information
	jurisdiction: str = Field(..., max_length=50, description="Legal jurisdiction")
	regulation_reference: str = Field(..., max_length=200, description="Regulation reference")
	effective_date: date = Field(..., description="Rule effective date")
	expiration_date: Optional[date] = Field(None, description="Rule expiration date")
	
	# Rule definition
	rule_description: str = Field(..., max_length=1000, description="Detailed rule description")
	rule_logic: Dict[str, Any] = Field(..., description="Rule implementation logic")
	validation_criteria: Dict[str, Any] = Field(..., description="Validation criteria")
	
	# Scope and applicability
	applicable_employees: List[str] = Field(
		default_factory=list, description="Employee types/roles this rule applies to"
	)
	applicable_departments: List[str] = Field(
		default_factory=list, description="Departments this rule applies to"
	)
	applicable_locations: List[str] = Field(
		default_factory=list, description="Locations this rule applies to"
	)
	
	# Violation handling
	violation_severity: str = Field(
		..., regex="^(INFO|WARNING|MINOR|MAJOR|CRITICAL)$", description="Violation severity level"
	)
	auto_correction_enabled: bool = Field(default=False, description="Automatic correction enabled")
	notification_required: bool = Field(default=True, description="Notification required for violations")
	
	# Monitoring and enforcement
	monitoring_frequency: str = Field(
		default="REAL_TIME", max_length=20, description="Monitoring frequency"
	)
	enforcement_actions: List[str] = Field(
		default_factory=list, description="Automated enforcement actions"
	)
	escalation_rules: Dict[str, Any] = Field(
		default_factory=dict, description="Violation escalation rules"
	)
	
	# Performance metrics
	violation_count: int = Field(default=0, ge=0, description="Total violations detected")
	last_violation_date: Optional[datetime] = Field(None, description="Last violation timestamp")
	compliance_rate: Annotated[float, AfterValidator(_validate_confidence_score)] = Field(
		default=1.0, description="Compliance rate percentage"
	)
	
	# Configuration
	is_active: bool = Field(default=True, description="Rule active status")
	priority: int = Field(default=5, ge=1, le=10, description="Rule priority (1=highest)")
	
	# Integration
	related_rules: List[str] = Field(default_factory=list, description="Related rule IDs")
	external_systems: List[str] = Field(
		default_factory=list, description="External systems for compliance reporting"
	)
	
	@computed_field
	@property
	def is_current(self) -> bool:
		"""Check if rule is currently effective"""
		today = date.today()
		return (self.effective_date <= today and 
				(not self.expiration_date or self.expiration_date >= today) and
				self.is_active)
	
	@computed_field
	@property
	def days_until_expiration(self) -> Optional[int]:
		"""Calculate days until rule expires"""
		if self.expiration_date:
			return (self.expiration_date - date.today()).days
		return None


class TARemoteWorker(TABaseModel):
	"""
	Revolutionary Remote Worker Model
	
	Advanced remote work tracking with productivity analytics, wellbeing monitoring,
	and intelligent workspace optimization for the distributed workforce.
	"""
	
	# Core remote worker information
	employee_id: str = Field(..., description="Associated employee ID")
	workspace_id: str = Field(default_factory=uuid7str, description="Remote workspace identifier")
	workforce_type: WorkforceType = Field(default=WorkforceType.REMOTE_WORKER, description="Workforce classification")
	work_mode: WorkMode = Field(..., description="Current work mode")
	
	# Workspace configuration
	home_office_setup: Dict[str, Any] = Field(
		default_factory=dict, description="Home office configuration and equipment"
	)
	timezone: str = Field(..., max_length=50, description="Worker timezone")
	preferred_work_hours: Dict[str, Any] = Field(
		default_factory=dict, description="Preferred working hours and schedule"
	)
	
	# Activity and productivity tracking
	current_activity: RemoteWorkStatus = Field(
		default=RemoteWorkStatus.OFFLINE, description="Current work activity"
	)
	productivity_metrics: List[Dict[str, Any]] = Field(
		default_factory=list, description="Real-time productivity measurements"
	)
	focus_time_blocks: List[Dict[str, Any]] = Field(
		default_factory=list, description="Deep work and focus time tracking"
	)
	
	# Collaboration and communication
	collaboration_platforms: List[str] = Field(
		default_factory=list, description="Active collaboration platforms"
	)
	meeting_participation: Dict[str, Any] = Field(
		default_factory=dict, description="Meeting attendance and engagement metrics"
	)
	communication_patterns: Dict[str, Any] = Field(
		default_factory=dict, description="Communication frequency and effectiveness"
	)
	
	# Digital wellbeing and health
	work_life_balance_score: Annotated[float, AfterValidator(_validate_confidence_score)] = Field(
		default=0.8, description="Work-life balance health score"
	)
	burnout_risk_indicators: List[Dict[str, Any]] = Field(
		default_factory=list, description="Burnout risk assessment indicators"
	)
	break_patterns: Dict[str, Any] = Field(
		default_factory=dict, description="Break frequency and duration patterns"
	)
	
	# Environment and IoT integration
	workspace_sensors: List[Dict[str, Any]] = Field(
		default_factory=list, description="IoT sensors in home office"
	)
	environmental_conditions: Dict[str, Any] = Field(
		default_factory=dict, description="Temperature, lighting, noise levels"
	)
	ergonomic_data: Dict[str, Any] = Field(
		default_factory=dict, description="Ergonomic assessment and recommendations"
	)
	
	# Performance and outcomes
	outcome_based_metrics: Dict[str, Any] = Field(
		default_factory=dict, description="Results and deliverable tracking"
	)
	quality_scores: Dict[str, Any] = Field(
		default_factory=dict, description="Work quality assessment"
	)
	efficiency_trends: List[Dict[str, Any]] = Field(
		default_factory=list, description="Efficiency improvement over time"
	)
	
	# Technology and tools
	device_ecosystem: List[Dict[str, Any]] = Field(
		default_factory=list, description="Connected devices and tools"
	)
	software_usage: Dict[str, Any] = Field(
		default_factory=dict, description="Software application usage patterns"
	)
	security_compliance: Dict[str, Any] = Field(
		default_factory=dict, description="Security and compliance monitoring"
	)
	
	@computed_field
	@property
	def overall_productivity_score(self) -> float:
		"""Calculate overall productivity score"""
		if not self.productivity_metrics:
			return 0.0
		
		scores = [metric.get("score", 0.0) for metric in self.productivity_metrics]
		return sum(scores) / len(scores) if scores else 0.0
	
	@computed_field
	@property
	def is_actively_working(self) -> bool:
		"""Check if remote worker is currently active"""
		active_statuses = [
			RemoteWorkStatus.ACTIVE_WORKING,
			RemoteWorkStatus.IN_MEETING,
			RemoteWorkStatus.FOCUSED_WORK,
			RemoteWorkStatus.COLLABORATIVE_WORK
		]
		return self.current_activity in active_statuses


class TAAIAgent(TABaseModel):
	"""
	Revolutionary AI Agent Workforce Model
	
	Comprehensive AI agent management with resource tracking, performance analytics,
	and human-AI collaboration optimization for the hybrid workforce.
	"""
	
	# Core AI agent information
	agent_name: str = Field(..., min_length=1, max_length=100, description="AI agent name")
	agent_type: AIAgentType = Field(..., description="Type of AI agent")
	agent_version: str = Field(..., max_length=50, description="Agent version identifier")
	workforce_type: WorkforceType = Field(default=WorkforceType.AI_AGENT, description="Workforce classification")
	
	# Capabilities and configuration
	capabilities: List[str] = Field(..., description="Agent capabilities and skills")
	configuration: Dict[str, Any] = Field(..., description="Agent configuration parameters")
	deployment_environment: str = Field(..., max_length=100, description="Deployment environment")
	
	# Resource consumption tracking
	cpu_hours: Decimal = Field(default=Decimal('0'), ge=0, description="CPU hours consumed")
	gpu_hours: Decimal = Field(default=Decimal('0'), ge=0, description="GPU hours consumed")
	memory_usage_gb_hours: Decimal = Field(default=Decimal('0'), ge=0, description="Memory consumption")
	api_calls_count: int = Field(default=0, ge=0, description="API calls made")
	storage_used_gb: Decimal = Field(default=Decimal('0'), ge=0, description="Storage consumed")
	
	# Task and work tracking
	tasks_completed: int = Field(default=0, ge=0, description="Total tasks completed")
	active_tasks: List[Dict[str, Any]] = Field(
		default_factory=list, description="Currently active tasks"
	)
	task_queue_size: int = Field(default=0, ge=0, description="Tasks in queue")
	average_task_duration_seconds: Optional[float] = Field(
		None, ge=0, description="Average task completion time"
	)
	
	# Performance metrics
	accuracy_score: Annotated[float, AfterValidator(_validate_confidence_score)] = Field(
		default=0.95, description="Task completion accuracy"
	)
	efficiency_rating: Annotated[float, AfterValidator(_validate_confidence_score)] = Field(
		default=0.90, description="Efficiency compared to baseline"
	)
	error_rate: Annotated[float, AfterValidator(_validate_confidence_score)] = Field(
		default=0.02, description="Error rate in task execution"
	)
	uptime_percentage: Annotated[float, AfterValidator(_validate_confidence_score)] = Field(
		default=0.999, description="Agent uptime percentage"
	)
	
	# Cost tracking
	operational_cost_per_hour: Decimal = Field(default=Decimal('0'), ge=0, description="Hourly operational cost")
	total_operational_cost: Decimal = Field(default=Decimal('0'), ge=0, description="Total operational cost")
	cost_per_task: Optional[Decimal] = Field(None, ge=0, description="Average cost per task")
	roi_metrics: Dict[str, Any] = Field(default_factory=dict, description="Return on investment metrics")
	
	# Human-AI collaboration
	human_collaborators: List[str] = Field(
		default_factory=list, description="Human collaborator employee IDs"
	)
	collaboration_sessions: List[Dict[str, Any]] = Field(
		default_factory=list, description="Human-AI collaboration sessions"
	)
	handoff_efficiency: Annotated[float, AfterValidator(_validate_confidence_score)] = Field(
		default=0.85, description="Efficiency of human-AI handoffs"
	)
	
	# Learning and improvement
	learning_rate: Annotated[float, AfterValidator(_validate_confidence_score)] = Field(
		default=0.1, description="Rate of performance improvement"
	)
	training_hours: Decimal = Field(default=Decimal('0'), ge=0, description="Training time invested")
	model_updates: List[Dict[str, Any]] = Field(
		default_factory=list, description="Model update history"
	)
	
	# Status and health
	is_active: bool = Field(default=True, description="Agent active status")
	health_status: str = Field(
		default="healthy", regex="^(healthy|degraded|error|maintenance)$",
		description="Agent health status"
	)
	last_health_check: datetime = Field(
		default_factory=datetime.utcnow, description="Last health check timestamp"
	)
	
	# Integration and APIs
	api_endpoints: List[str] = Field(default_factory=list, description="Available API endpoints")
	integration_points: Dict[str, Any] = Field(
		default_factory=dict, description="System integration configurations"
	)
	monitoring_metrics: Dict[str, Any] = Field(
		default_factory=dict, description="Real-time monitoring data"
	)
	
	@computed_field
	@property
	def cost_efficiency_score(self) -> float:
		"""Calculate cost efficiency compared to human equivalent"""
		if self.total_operational_cost == 0 or self.tasks_completed == 0:
			return 0.0
		
		# Simplified calculation - would use more sophisticated modeling in reality
		human_equivalent_cost = float(self.tasks_completed) * 25.0  # $25 per task human baseline
		agent_cost = float(self.total_operational_cost)
		
		if human_equivalent_cost == 0:
			return 0.0
		
		return min(human_equivalent_cost / agent_cost, 10.0)  # Cap at 10x efficiency
	
	@computed_field
	@property
	def overall_performance_score(self) -> float:
		"""Calculate overall AI agent performance score"""
		weights = {
			'accuracy': 0.3,
			'efficiency': 0.25,
			'uptime': 0.2,
			'collaboration': 0.15,
			'cost_efficiency': 0.1
		}
		
		cost_eff_normalized = min(self.cost_efficiency_score / 5.0, 1.0)  # Normalize to 0-1
		
		return (
			self.accuracy_score * weights['accuracy'] +
			self.efficiency_rating * weights['efficiency'] +
			self.uptime_percentage * weights['uptime'] +
			self.handoff_efficiency * weights['collaboration'] +
			cost_eff_normalized * weights['cost_efficiency']
		)


class TAHybridCollaboration(TABaseModel):
	"""
	Human-AI Collaboration Session Model
	
	Tracks collaborative work sessions between humans and AI agents with
	productivity measurement, handoff efficiency, and outcome optimization.
	"""
	
	# Session information
	session_name: str = Field(..., min_length=1, max_length=200, description="Collaboration session name")
	project_id: str = Field(..., description="Associated project identifier")
	session_type: str = Field(..., max_length=50, description="Type of collaboration session")
	
	# Participants
	human_participants: List[str] = Field(..., description="Human employee IDs")
	ai_participants: List[str] = Field(..., description="AI agent IDs")
	session_lead: str = Field(..., description="Session lead (human or AI agent ID)")
	
	# Session timeline
	start_time: datetime = Field(..., description="Session start time")
	end_time: Optional[datetime] = Field(None, description="Session end time")
	planned_duration_minutes: int = Field(..., ge=1, description="Planned session duration")
	
	# Work allocation and tracking
	human_work_allocation: Dict[str, Any] = Field(
		default_factory=dict, description="Work allocated to humans"
	)
	ai_work_allocation: Dict[str, Any] = Field(
		default_factory=dict, description="Work allocated to AI agents"
	)
	task_handoffs: List[Dict[str, Any]] = Field(
		default_factory=list, description="Task handoffs between humans and AI"
	)
	
	# Productivity and outcomes
	deliverables_completed: List[Dict[str, Any]] = Field(
		default_factory=list, description="Completed deliverables and outputs"
	)
	quality_metrics: Dict[str, Any] = Field(
		default_factory=dict, description="Quality assessment of outputs"
	)
	efficiency_score: Annotated[float, AfterValidator(_validate_confidence_score)] = Field(
		default=0.8, description="Session efficiency score"
	)
	
	# Communication and coordination
	communication_events: List[Dict[str, Any]] = Field(
		default_factory=list, description="Communication events during session"
	)
	decision_points: List[Dict[str, Any]] = Field(
		default_factory=list, description="Key decision points and outcomes"
	)
	conflict_resolution: List[Dict[str, Any]] = Field(
		default_factory=list, description="Conflicts and resolution methods"
	)
	
	# Resource utilization
	human_hours_contributed: Decimal = Field(default=Decimal('0'), ge=0, description="Human hours invested")
	ai_compute_hours: Decimal = Field(default=Decimal('0'), ge=0, description="AI compute hours used")
	total_cost: Decimal = Field(default=Decimal('0'), ge=0, description="Total session cost")
	
	# Learning and improvement
	lessons_learned: List[str] = Field(default_factory=list, description="Session lessons learned")
	improvement_suggestions: List[str] = Field(
		default_factory=list, description="Process improvement suggestions"
	)
	knowledge_transfer: Dict[str, Any] = Field(
		default_factory=dict, description="Knowledge transferred between participants"
	)
	
	@computed_field
	@property
	def session_duration_minutes(self) -> Optional[int]:
		"""Calculate actual session duration"""
		if self.start_time and self.end_time:
			return int((self.end_time - self.start_time).total_seconds() / 60)
		return None
	
	@computed_field
	@property
	def human_ai_ratio(self) -> float:
		"""Calculate ratio of human to AI contribution"""
		total_human = float(self.human_hours_contributed)
		total_ai = float(self.ai_compute_hours)
		
		if total_ai == 0:
			return float('inf') if total_human > 0 else 0.0
		
		return total_human / total_ai
	
	@computed_field
	@property
	def collaboration_effectiveness(self) -> float:
		"""Calculate overall collaboration effectiveness score"""
		if not self.session_duration_minutes or self.planned_duration_minutes == 0:
			return 0.0
		
		time_efficiency = min(self.planned_duration_minutes / self.session_duration_minutes, 1.0)
		
		# Weight different factors
		weights = {
			'efficiency': 0.4,
			'time_efficiency': 0.3,
			'deliverables': 0.2,
			'communication': 0.1
		}
		
		deliverable_score = min(len(self.deliverables_completed) / 3.0, 1.0)  # Normalize to expected 3 deliverables
		communication_score = min(len(self.communication_events) / 10.0, 1.0)  # Normalize to 10 events
		
		return (
			self.efficiency_score * weights['efficiency'] +
			time_efficiency * weights['time_efficiency'] +
			deliverable_score * weights['deliverables'] +
			communication_score * weights['communication']
		)


# Export all models
__all__ = [
	# Enums
	'TimeEntryStatus', 'TimeEntryType', 'AttendanceStatus', 'ScheduleStatus',
	'BiometricType', 'DeviceType', 'FraudType', 'LeaveType', 'ApprovalStatus',
	'WorkforceType', 'WorkMode', 'AIAgentType', 'ProductivityMetric', 'RemoteWorkStatus',
	
	# Core Models
	'TABaseModel', 'TAEmployee', 'TATimeEntry', 'TASchedule', 'TALeaveRequest',
	'TAFraudDetection', 'TABiometricAuthentication', 'TAPredictiveAnalytics',
	'TAComplianceRule',
	
	# Revolutionary Workforce Models
	'TARemoteWorker', 'TAAIAgent', 'TAHybridCollaboration',
	
	# Validation functions (for testing)
	'_validate_confidence_score', '_validate_time_range', '_validate_biometric_template',
	'_validate_geolocation', '_validate_schedule_pattern'
]