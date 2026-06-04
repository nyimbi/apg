"""
APG Security Framework Data Models

Comprehensive data models for enterprise security controls, threat detection,
and compliance automation following APG coding standards.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from typing import Dict, Any, Optional, List, Set, Union, Annotated
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import json
from uuid_extensions import uuid7str
from pydantic import BaseModel, Field, ConfigDict, AfterValidator, validator

# Import enums from __init__.py
try:
	from . import (
		SecurityLevel, RiskLevel, ThreatType, ComplianceFramework,
		SecurityAction, DeviceTrustLevel
	)
except ImportError:
	from capabilities.common.secu import (
		SecurityLevel, RiskLevel, ThreatType, ComplianceFramework,
		SecurityAction, DeviceTrustLevel
	)

# Validation functions
def validate_ip_address(ip: str) -> str:
	"""Validate IP address format"""
	import ipaddress
	try:
		ipaddress.ip_address(ip)
		return ip
	except ValueError:
		raise ValueError(f"Invalid IP address format: {ip}")

def validate_risk_score(score: float) -> float:
	"""Validate risk score is within valid range"""
	if not 0.0 <= score <= 100.0:
		raise ValueError(f"Risk score must be between 0.0 and 100.0, got {score}")
	return score

def validate_device_fingerprint(fingerprint: str) -> str:
	"""Validate device fingerprint format"""
	if len(fingerprint) < 32:
		raise ValueError("Device fingerprint must be at least 32 characters")
	return fingerprint

def validate_policy_priority(priority: int) -> int:
	"""Validate policy priority range"""
	if not 1 <= priority <= 1000:
		raise ValueError(f"Policy priority must be between 1 and 1000, got {priority}")
	return priority

# Core Security Models
class DeviceContext(BaseModel):
	"""Device context information for security assessment"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Device context ID")
	device_id: str = Field(..., description="Unique device identifier")
	device_type: str = Field(..., description="Device type (desktop, mobile, tablet)")
	os_type: str = Field(..., description="Operating system type")
	os_version: str = Field(..., description="Operating system version")
	browser_type: Optional[str] = Field(default=None, description="Browser type")
	browser_version: Optional[str] = Field(default=None, description="Browser version")
	trust_level: DeviceTrustLevel = Field(default=DeviceTrustLevel.UNKNOWN, description="Device trust level")
	last_seen: datetime = Field(default_factory=datetime.utcnow, description="Last seen timestamp")
	fingerprint: Annotated[str, AfterValidator(validate_device_fingerprint)] = Field(
		..., description="Device fingerprint hash"
	)
	screen_resolution: Optional[str] = Field(default=None, description="Screen resolution")
	timezone: Optional[str] = Field(default=None, description="Device timezone")
	language: Optional[str] = Field(default=None, description="Device language")
	user_agent: Optional[str] = Field(default=None, description="User agent string")
	plugins: List[str] = Field(default_factory=list, description="Browser plugins")
	hardware_info: Dict[str, Any] = Field(default_factory=dict, description="Hardware information")
	security_features: Dict[str, bool] = Field(default_factory=dict, description="Security features enabled")
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)

class NetworkContext(BaseModel):
	"""Network context information for security assessment"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Network context ID")
	ip_address: Annotated[str, AfterValidator(validate_ip_address)] = Field(
		..., description="Client IP address"
	)
	country: Optional[str] = Field(default=None, description="Country of origin")
	region: Optional[str] = Field(default=None, description="Region/state")
	city: Optional[str] = Field(default=None, description="City")
	postal_code: Optional[str] = Field(default=None, description="Postal code")
	latitude: Optional[float] = Field(default=None, description="Latitude coordinate")
	longitude: Optional[float] = Field(default=None, description="Longitude coordinate")
	isp: Optional[str] = Field(default=None, description="Internet service provider")
	organization: Optional[str] = Field(default=None, description="Organization name")
	asn: Optional[str] = Field(default=None, description="Autonomous system number")
	is_vpn: bool = Field(default=False, description="VPN connection detected")
	is_proxy: bool = Field(default=False, description="Proxy connection detected")
	is_tor: bool = Field(default=False, description="Tor network detected")
	is_hosting: bool = Field(default=False, description="Hosting provider detected")
	is_known_malicious: bool = Field(default=False, description="Known malicious IP")
	reputation_score: Annotated[float, AfterValidator(validate_risk_score)] = Field(
		default=0.0, description="IP reputation score (0-100)"
	)
	threat_types: List[str] = Field(default_factory=list, description="Associated threat types")
	last_seen_malicious: Optional[datetime] = Field(default=None, description="Last malicious activity")
	connection_type: Optional[str] = Field(default=None, description="Connection type")
	bandwidth: Optional[int] = Field(default=None, description="Estimated bandwidth")
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)

class BehavioralPattern(BaseModel):
	"""User behavioral pattern analysis"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Behavioral pattern ID")
	user_id: str = Field(..., description="User identifier")
	pattern_type: str = Field(..., description="Type of behavioral pattern")
	baseline_established: bool = Field(default=False, description="Baseline pattern established")
	typical_login_times: List[int] = Field(default_factory=list, description="Typical login hours")
	typical_locations: List[Dict[str, Any]] = Field(default_factory=list, description="Typical locations")
	typical_devices: List[str] = Field(default_factory=list, description="Typical devices")
	access_patterns: Dict[str, Any] = Field(default_factory=dict, description="Access patterns")
	velocity_patterns: Dict[str, float] = Field(default_factory=dict, description="Velocity patterns")
	anomaly_threshold: float = Field(default=0.3, description="Anomaly detection threshold")
	learning_period_days: int = Field(default=30, description="Learning period in days")
	confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Pattern confidence")
	last_updated: datetime = Field(default_factory=datetime.utcnow)
	created_at: datetime = Field(default_factory=datetime.utcnow)

class RiskScore(BaseModel):
	"""Risk assessment score with contributing factors"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Risk score ID")
	overall_score: Annotated[float, AfterValidator(validate_risk_score)] = Field(
		..., description="Overall risk score (0-100)"
	)
	level: RiskLevel = Field(..., description="Risk level classification")
	behavioral_score: Annotated[float, AfterValidator(validate_risk_score)] = Field(
		default=0.0, description="Behavioral risk component"
	)
	device_score: Annotated[float, AfterValidator(validate_risk_score)] = Field(
		default=0.0, description="Device risk component"
	)
	network_score: Annotated[float, AfterValidator(validate_risk_score)] = Field(
		default=0.0, description="Network risk component"
	)
	temporal_score: Annotated[float, AfterValidator(validate_risk_score)] = Field(
		default=0.0, description="Temporal risk component"
	)
	geospatial_score: Annotated[float, AfterValidator(validate_risk_score)] = Field(
		default=0.0, description="Geospatial risk component"
	)
	confidence: Annotated[float, AfterValidator(validate_risk_score)] = Field(
		default=0.0, description="Confidence in assessment"
	)
	factors: List[str] = Field(default_factory=list, description="Contributing risk factors")
	mitigating_factors: List[str] = Field(default_factory=list, description="Risk mitigating factors")
	calculation_method: str = Field(default="weighted_average", description="Calculation method used")
	weights: Dict[str, float] = Field(default_factory=dict, description="Component weights")
	threshold_breaches: List[str] = Field(default_factory=list, description="Threshold breaches")
	calculated_at: datetime = Field(default_factory=datetime.utcnow, description="Calculation timestamp")
	expires_at: Optional[datetime] = Field(default=None, description="Score expiration time")

class ThreatIndicator(BaseModel):
	"""Security threat indicator"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Threat indicator ID")
	threat_type: ThreatType = Field(..., description="Type of threat detected")
	severity: RiskLevel = Field(..., description="Threat severity level")
	confidence: Annotated[float, AfterValidator(validate_risk_score)] = Field(
		..., description="Detection confidence (0-100)"
	)
	source: str = Field(..., description="Detection source system")
	source_reliability: str = Field(default="unknown", description="Source reliability rating")
	title: str = Field(..., description="Threat indicator title")
	description: str = Field(..., description="Detailed threat description")
	indicators: Dict[str, Any] = Field(default_factory=dict, description="Technical indicators")
	iocs: List[str] = Field(default_factory=list, description="Indicators of compromise")
	ttps: List[str] = Field(default_factory=list, description="Tactics, techniques, procedures")
	mitigation: Optional[str] = Field(default=None, description="Mitigation strategy")
	remediation_steps: List[str] = Field(default_factory=list, description="Remediation steps")
	false_positive_likelihood: float = Field(default=0.0, ge=0.0, le=1.0, description="False positive likelihood")
	impact_assessment: Dict[str, Any] = Field(default_factory=dict, description="Impact assessment")
	affected_systems: List[str] = Field(default_factory=list, description="Affected systems")
	kill_chain_phase: Optional[str] = Field(default=None, description="Cyber kill chain phase")
	detected_at: datetime = Field(default_factory=datetime.utcnow, description="Detection timestamp")
	expires_at: Optional[datetime] = Field(default=None, description="Expiration timestamp")
	last_updated: datetime = Field(default_factory=datetime.utcnow)

class SecurityContext(BaseModel):
	"""Complete security context for risk assessment"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Security context ID")
	tenant_id: str = Field(..., description="APG tenant identifier")
	user_id: str = Field(..., description="User identifier")
	session_id: str = Field(..., description="Session identifier")
	request_id: Optional[str] = Field(default=None, description="Request identifier")
	capability_id: Optional[str] = Field(default=None, description="APG capability identifier")
	action: str = Field(..., description="Action being performed")
	resource: Optional[str] = Field(default=None, description="Resource being accessed")
	resource_classification: SecurityLevel = Field(default=SecurityLevel.INTERNAL, description="Resource classification")
	device_context: DeviceContext = Field(..., description="Device information")
	network_context: NetworkContext = Field(..., description="Network information")
	behavioral_context: Optional[BehavioralPattern] = Field(default=None, description="Behavioral patterns")
	risk_score: Optional[RiskScore] = Field(default=None, description="Current risk assessment")
	threat_indicators: List[ThreatIndicator] = Field(default_factory=list, description="Active threats")
	previous_contexts: List[str] = Field(default_factory=list, description="Previous context IDs")
	authentication_factors: List[str] = Field(default_factory=list, description="Authentication factors used")
	authorization_grants: List[str] = Field(default_factory=list, description="Authorization grants")
	security_headers: Dict[str, str] = Field(default_factory=dict, description="Security headers")
	audit_trail: List[Dict[str, Any]] = Field(default_factory=list, description="Security audit trail")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Context creation time")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update time")
	expires_at: Optional[datetime] = Field(default=None, description="Context expiration time")

class ComplianceRequirement(BaseModel):
	"""Individual compliance requirement"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Requirement ID")
	framework: ComplianceFramework = Field(..., description="Compliance framework")
	requirement_id: str = Field(..., description="Framework-specific requirement ID")
	title: str = Field(..., description="Requirement title")
	description: str = Field(..., description="Requirement description")
	category: str = Field(..., description="Requirement category")
	priority: str = Field(default="medium", description="Requirement priority")
	control_objectives: List[str] = Field(default_factory=list, description="Control objectives")
	test_procedures: List[str] = Field(default_factory=list, description="Test procedures")
	evidence_required: List[str] = Field(default_factory=list, description="Required evidence")
	automation_level: str = Field(default="manual", description="Automation level")
	frequency: str = Field(default="annual", description="Assessment frequency")
	responsible_party: str = Field(..., description="Responsible party")
	status: str = Field(default="pending", description="Current status")
	implementation_date: Optional[datetime] = Field(default=None, description="Implementation date")
	last_assessment: Optional[datetime] = Field(default=None, description="Last assessment date")
	next_assessment: Optional[datetime] = Field(default=None, description="Next assessment due")
	created_at: datetime = Field(default_factory=datetime.utcnow)

class ComplianceStatus(BaseModel):
	"""Compliance status tracking"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Compliance status ID")
	tenant_id: str = Field(..., description="APG tenant identifier")
	framework: ComplianceFramework = Field(..., description="Compliance framework")
	status: str = Field(..., description="Overall compliance status")
	score: Annotated[float, AfterValidator(validate_risk_score)] = Field(
		..., description="Compliance score (0-100)"
	)
	requirements_met: int = Field(..., ge=0, description="Requirements satisfied")
	requirements_total: int = Field(..., ge=0, description="Total requirements")
	requirements_partial: int = Field(default=0, ge=0, description="Partially satisfied requirements")
	requirements_not_applicable: int = Field(default=0, ge=0, description="Not applicable requirements")
	violations: List[str] = Field(default_factory=list, description="Current violations")
	gaps: List[str] = Field(default_factory=list, description="Compliance gaps")
	remediation_plan: List[Dict[str, Any]] = Field(default_factory=list, description="Remediation plan")
	risk_rating: RiskLevel = Field(default=RiskLevel.MODERATE, description="Compliance risk rating")
	certification_status: str = Field(default="not_certified", description="Certification status")
	auditor_notes: List[str] = Field(default_factory=list, description="Auditor notes")
	evidence_repository: Dict[str, str] = Field(default_factory=dict, description="Evidence links")
	last_assessment: datetime = Field(default_factory=datetime.utcnow, description="Last assessment time")
	next_assessment: Optional[datetime] = Field(default=None, description="Next assessment due")
	assessor: Optional[str] = Field(default=None, description="Assessor identifier")
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)

class SecurityPolicy(BaseModel):
	"""Security policy definition"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Policy identifier")
	name: str = Field(..., description="Policy name")
	description: str = Field(..., description="Policy description")
	category: str = Field(..., description="Policy category")
	tenant_id: Optional[str] = Field(default=None, description="Tenant-specific policy")
	capability_id: Optional[str] = Field(default=None, description="Capability-specific policy")
	scope: str = Field(default="global", description="Policy scope")
	conditions: Dict[str, Any] = Field(default_factory=dict, description="Policy conditions")
	actions: List[SecurityAction] = Field(default_factory=list, description="Actions to take")
	exceptions: List[Dict[str, Any]] = Field(default_factory=list, description="Policy exceptions")
	priority: Annotated[int, AfterValidator(validate_policy_priority)] = Field(
		default=100, description="Policy priority (1-1000)"
	)
	enabled: bool = Field(default=True, description="Policy enabled status")
	enforcement_mode: str = Field(default="enforce", description="Enforcement mode")
	notification_settings: Dict[str, Any] = Field(default_factory=dict, description="Notification settings")
	approval_required: bool = Field(default=False, description="Approval required for exceptions")
	tags: List[str] = Field(default_factory=list, description="Policy tags")
	version: str = Field(default="1.0.0", description="Policy version")
	change_log: List[Dict[str, Any]] = Field(default_factory=list, description="Policy change log")
	created_by: str = Field(..., description="Policy creator")
	approved_by: Optional[str] = Field(default=None, description="Policy approver")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Update timestamp")
	effective_date: datetime = Field(default_factory=datetime.utcnow, description="Policy effective date")
	expires_at: Optional[datetime] = Field(default=None, description="Policy expiration")
	last_reviewed: Optional[datetime] = Field(default=None, description="Last review date")
	review_frequency: int = Field(default=365, description="Review frequency in days")

class SecurityIncident(BaseModel):
	"""Security incident tracking"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Incident ID")
	tenant_id: str = Field(..., description="APG tenant identifier")
	title: str = Field(..., description="Incident title")
	description: str = Field(..., description="Incident description")
	severity: RiskLevel = Field(..., description="Incident severity")
	status: str = Field(default="open", description="Incident status")
	category: str = Field(..., description="Incident category")
	threat_indicators: List[str] = Field(default_factory=list, description="Related threat indicators")
	affected_systems: List[str] = Field(default_factory=list, description="Affected systems")
	affected_users: List[str] = Field(default_factory=list, description="Affected users")
	impact_assessment: Dict[str, Any] = Field(default_factory=dict, description="Impact assessment")
	timeline: List[Dict[str, Any]] = Field(default_factory=list, description="Incident timeline")
	evidence: List[Dict[str, str]] = Field(default_factory=list, description="Incident evidence")
	containment_actions: List[str] = Field(default_factory=list, description="Containment actions taken")
	remediation_actions: List[str] = Field(default_factory=list, description="Remediation actions")
	lessons_learned: List[str] = Field(default_factory=list, description="Lessons learned")
	assigned_to: Optional[str] = Field(default=None, description="Assigned analyst")
	escalated: bool = Field(default=False, description="Incident escalated")
	false_positive: bool = Field(default=False, description="False positive incident")
	cost_impact: Optional[float] = Field(default=None, description="Financial impact")
	detected_at: datetime = Field(..., description="Detection timestamp")
	reported_at: datetime = Field(default_factory=datetime.utcnow, description="Reporting timestamp")
	resolved_at: Optional[datetime] = Field(default=None, description="Resolution timestamp")
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)

class SecurityMetric(BaseModel):
	"""Security metrics and KPIs"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Metric ID")
	tenant_id: str = Field(..., description="APG tenant identifier")
	metric_name: str = Field(..., description="Metric name")
	metric_type: str = Field(..., description="Metric type")
	category: str = Field(..., description="Metric category")
	value: float = Field(..., description="Metric value")
	unit: str = Field(..., description="Metric unit")
	target_value: Optional[float] = Field(default=None, description="Target value")
	threshold_warning: Optional[float] = Field(default=None, description="Warning threshold")
	threshold_critical: Optional[float] = Field(default=None, description="Critical threshold")
	trend: str = Field(default="stable", description="Metric trend")
	period: str = Field(default="daily", description="Measurement period")
	data_source: str = Field(..., description="Data source")
	calculation_method: str = Field(..., description="Calculation method")
	tags: List[str] = Field(default_factory=list, description="Metric tags")
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
	measured_at: datetime = Field(default_factory=datetime.utcnow, description="Measurement timestamp")
	created_at: datetime = Field(default_factory=datetime.utcnow)

# Export models
__all__ = [
	"DeviceContext",
	"NetworkContext", 
	"BehavioralPattern",
	"RiskScore",
	"ThreatIndicator",
	"SecurityContext",
	"ComplianceRequirement",
	"ComplianceStatus",
	"SecurityPolicy",
	"SecurityIncident",
	"SecurityMetric",
	
	# Validation functions
	"validate_ip_address",
	"validate_risk_score", 
	"validate_device_fingerprint",
	"validate_policy_priority"
]