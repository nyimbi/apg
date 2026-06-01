"""
APG Audit Logging Data Models

Production-grade audit trail data structures with ML-powered analytics, natural language processing,
and immutable blockchain verification. Designed for 10M+ events/second ingestion with
sub-second query response times.

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from typing import Dict, Any, Optional, List, Union, Annotated
from datetime import datetime, timedelta
from enum import Enum, StrEnum
from pathlib import Path
from decimal import Decimal
import hashlib
import json
from uuid_extensions import uuid7str

from pydantic import BaseModel, Field, ConfigDict, AfterValidator, validator
from pydantic.types import SecretStr


def validate_tenant_id(tenant_id: Optional[str]) -> Optional[str]:
	"""Validate tenant ID format for APG multi-tenancy"""
	if tenant_id is None:
		return None
	assert isinstance(tenant_id, str), "Tenant ID must be string"
	assert len(tenant_id) >= 8, "Tenant ID must be at least 8 characters"
	return tenant_id


def validate_event_severity(severity: int) -> int:
	"""Validate event severity score (0-100)"""
	assert isinstance(severity, int), "Severity must be integer"
	assert 0 <= severity <= 100, "Severity must be between 0-100"
	return severity


def validate_risk_score(risk_score: float) -> float:
	"""Validate ML-generated risk score (0.0-1.0)"""
	assert isinstance(risk_score, (int, float)), "Risk score must be numeric"
	assert 0.0 <= risk_score <= 1.0, "Risk score must be between 0.0-1.0"
	return float(risk_score)


def validate_compliance_tags(tags: List[str]) -> List[str]:
	"""Validate compliance framework tags"""
	assert isinstance(tags, list), "Compliance tags must be list"
	valid_frameworks = {"SOX", "GDPR", "HIPAA", "PCI-DSS", "ISO-27001", "SOC-2", "NIST", "CIS"}
	for tag in tags:
		assert isinstance(tag, str), "Compliance tag must be string"
		assert tag in valid_frameworks, f"Invalid compliance framework: {tag}"
	return tags


# Enums for audit logging system

class AuditLevel(StrEnum):
	"""Audit log severity levels"""
	DEBUG = "debug"
	INFO = "info"
	WARNING = "warning"
	ERROR = "error"
	CRITICAL = "critical"


class AuditEventType(StrEnum):
	"""Comprehensive audit event types"""
	# Authentication Events
	USER_LOGIN = "user_login"
	USER_LOGOUT = "user_logout"
	USER_FAILED_LOGIN = "user_failed_login"
	USER_PASSWORD_CHANGE = "user_password_change"
	USER_MFA_ENABLED = "user_mfa_enabled"
	USER_MFA_DISABLED = "user_mfa_disabled"
	
	# Authorization Events  
	PERMISSION_GRANTED = "permission_granted"
	PERMISSION_REVOKED = "permission_revoked"
	ROLE_ASSIGNED = "role_assigned"
	ROLE_REMOVED = "role_removed"
	ACCESS_DENIED = "access_denied"
	
	# Data Operations
	DATA_READ = "data_read"
	DATA_CREATE = "data_create"
	DATA_UPDATE = "data_update"
	DATA_DELETE = "data_delete"
	DATA_EXPORT = "data_export"
	DATA_IMPORT = "data_import"
	
	# System Events
	SYSTEM_START = "system_start"
	SYSTEM_STOP = "system_stop"
	SYSTEM_RESTART = "system_restart"
	CONFIG_CHANGE = "config_change"
	MAINTENANCE_START = "maintenance_start"
	MAINTENANCE_END = "maintenance_end"
	
	# API Events
	API_CALL = "api_call"
	API_ERROR = "api_error"
	API_RATE_LIMIT = "api_rate_limit"
	WEBHOOK_TRIGGERED = "webhook_triggered"
	
	# Security Events
	SECURITY_ALERT = "security_alert"
	INTRUSION_ATTEMPT = "intrusion_attempt"
	MALWARE_DETECTED = "malware_detected"
	VULNERABILITY_FOUND = "vulnerability_found"
	SECURITY_POLICY_VIOLATION = "security_policy_violation"
	
	# Compliance Events
	COMPLIANCE_VIOLATION = "compliance_violation"
	AUDIT_TRAIL_ACCESS = "audit_trail_access"
	EVIDENCE_COLLECTED = "evidence_collected"
	LEGAL_HOLD_APPLIED = "legal_hold_applied"
	LEGAL_HOLD_RELEASED = "legal_hold_released"
	
	# Investigation Events
	INVESTIGATION_CREATED = "investigation_created"
	INVESTIGATION_UPDATED = "investigation_updated"
	INVESTIGATION_CLOSED = "investigation_closed"
	EVIDENCE_ADDED = "evidence_added"
	
	# Custom Events
	CUSTOM_EVENT = "custom_event"


class EventSource(StrEnum):
	"""Sources of audit events within APG platform"""
	APG_CORE = "apg_core"
	AUTH = "auth" 
	MULTI_TENANT = "multi_tenant"
	NOTIFICATIONS = "notifications"
	NLP_CORE = "nlp_core"
	SECURITY = "security"
	COMPLIANCE = "compliance"
	COLLABORATION = "collaboration"
	API_GATEWAY = "api_gateway"
	EXTERNAL_SYSTEM = "external_system"


class ComplianceFramework(StrEnum):
	"""Supported compliance frameworks"""
	SOX = "SOX"
	GDPR = "GDPR"
	HIPAA = "HIPAA"
	PCI_DSS = "PCI-DSS"
	ISO_27001 = "ISO-27001"
	SOC_2 = "SOC-2"
	NIST = "NIST"
	CIS = "CIS"


class InvestigationStatus(StrEnum):
	"""Investigation workflow statuses"""
	OPEN = "open"
	IN_PROGRESS = "in_progress"
	ESCALATED = "escalated"
	RESOLVED = "resolved"
	CLOSED = "closed"
	ARCHIVED = "archived"


class EvidenceType(StrEnum):
	"""Types of digital evidence"""
	LOG_ENTRY = "log_entry"
	DATABASE_RECORD = "database_record"
	FILE_SYSTEM = "file_system"
	NETWORK_TRAFFIC = "network_traffic"
	EMAIL = "email"
	DOCUMENT = "document"
	SCREENSHOT = "screenshot"
	VIDEO_RECORDING = "video_recording"
	BLOCKCHAIN_PROOF = "blockchain_proof"


# Core audit event models

class AuditEvent(BaseModel):
	"""Core audit event with ML-powered enrichment and blockchain verification"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	# Event Identity
	id: str = Field(default_factory=uuid7str, description="Unique event identifier")
	tenant_id: Annotated[Optional[str], AfterValidator(validate_tenant_id)] = Field(None, description="APG tenant identifier")
	timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event timestamp (UTC)")
	correlation_id: Optional[str] = Field(None, description="Event correlation identifier")
	
	# Event Classification
	level: AuditLevel = Field(..., description="Event severity level")
	event_type: AuditEventType = Field(..., description="Structured event type")
	source: EventSource = Field(..., description="APG component source")
	category: str = Field(..., description="Event category")
	subcategory: Optional[str] = Field(None, description="Event subcategory")
	
	# Actor Information (Who)
	user_id: Optional[str] = Field(None, description="User identifier")
	session_id: Optional[str] = Field(None, description="Session identifier")
	service_account: Optional[str] = Field(None, description="Service account identifier")
	actor_type: str = Field(default="user", description="Actor type (user, service, system)")
	actor_display_name: Optional[str] = Field(None, description="Human-readable actor name")
	
	# Action Information (What)
	action: str = Field(..., description="Action performed")
	action_description: Optional[str] = Field(None, description="Human-readable action description")
	operation_id: Optional[str] = Field(None, description="Operation identifier for tracing")
	
	# Resource Information (What was affected)
	resource_type: Optional[str] = Field(None, description="Type of resource")
	resource_id: Optional[str] = Field(None, description="Resource identifier")
	resource_name: Optional[str] = Field(None, description="Human-readable resource name")
	resource_path: Optional[str] = Field(None, description="Resource path or location")
	parent_resource_id: Optional[str] = Field(None, description="Parent resource identifier")
	
	# Context Information (Where, When, How)
	ip_address: Optional[str] = Field(None, description="Client IP address")
	user_agent: Optional[str] = Field(None, description="Client user agent")
	geographic_location: Optional[str] = Field(None, description="Geographic location")
	device_id: Optional[str] = Field(None, description="Device identifier")
	request_id: Optional[str] = Field(None, description="Request identifier")
	
	# Outcome and Performance
	success: bool = Field(default=True, description="Whether action was successful")
	status_code: Optional[int] = Field(None, description="Response status code")
	error_code: Optional[str] = Field(None, description="Error code if failed")
	error_message: Optional[str] = Field(None, description="Error message if failed")
	duration_ms: Optional[int] = Field(None, description="Operation duration in milliseconds")
	
	# ML-Powered Enrichment
	risk_score: Annotated[float, AfterValidator(validate_risk_score)] = Field(default=0.0, description="ML-generated risk score (0.0-1.0)")
	anomaly_score: Annotated[float, AfterValidator(validate_risk_score)] = Field(default=0.0, description="Anomaly detection score (0.0-1.0)")
	threat_indicators: List[str] = Field(default_factory=list, description="Threat intelligence indicators")
	behavioral_tags: List[str] = Field(default_factory=list, description="Behavioral analysis tags")
	
	# Compliance and Governance
	compliance_tags: Annotated[List[str], AfterValidator(validate_compliance_tags)] = Field(default_factory=list, description="Compliance framework tags")
	data_classification: Optional[str] = Field(None, description="Data classification level")
	retention_period_days: int = Field(default=2555, description="Retention period in days (7 years default)")
	legal_hold: bool = Field(default=False, description="Legal hold status")
	
	# Additional Context
	details: Dict[str, Any] = Field(default_factory=dict, description="Additional event details")
	tags: Dict[str, str] = Field(default_factory=dict, description="Custom tags")
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Event metadata")
	
	# Data Integrity
	checksum: Optional[str] = Field(None, description="Event integrity checksum")
	blockchain_hash: Optional[str] = Field(None, description="Blockchain verification hash")
	digital_signature: Optional[str] = Field(None, description="Digital signature")
	
	# APG Platform Integration
	apg_version: Optional[str] = Field(None, description="APG platform version")
	capability_version: Optional[str] = Field(None, description="Capability version")
	
	def model_post_init(self, __context) -> None:
		"""Calculate checksums and hashes after initialization"""
		if not self.checksum:
			self.checksum = self._calculate_checksum()
	
	def _calculate_checksum(self) -> str:
		"""Calculate SHA-256 checksum for integrity verification"""
		# Core event data for checksum (immutable fields only)
		core_data = {
			"id": self.id,
			"timestamp": self.timestamp.isoformat(),
			"tenant_id": self.tenant_id,
			"level": self.level,
			"event_type": self.event_type,
			"source": self.source,
			"user_id": self.user_id,
			"action": self.action,
			"resource_type": self.resource_type,
			"resource_id": self.resource_id,
			"success": self.success
		}
		json_str = json.dumps(core_data, sort_keys=True)
		return hashlib.sha256(json_str.encode()).hexdigest()
	
	def verify_integrity(self) -> bool:
		"""Verify event integrity using checksum"""
		if not self.checksum:
			return False
		return self.checksum == self._calculate_checksum()


class AuditEventBatch(BaseModel):
	"""Batch of audit events for high-throughput ingestion"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	batch_id: str = Field(default_factory=uuid7str, description="Batch identifier")
	tenant_id: Annotated[Optional[str], AfterValidator(validate_tenant_id)] = Field(None, description="APG tenant identifier")
	timestamp: datetime = Field(default_factory=datetime.utcnow, description="Batch timestamp")
	events: List[AuditEvent] = Field(..., description="Audit events in batch", min_length=1, max_length=10000)
	source_system: Optional[str] = Field(None, description="Source system identifier")
	compression: Optional[str] = Field(None, description="Compression algorithm used")
	batch_checksum: Optional[str] = Field(None, description="Batch integrity checksum")
	
	def model_post_init(self, __context) -> None:
		"""Calculate batch checksum after initialization"""
		if not self.batch_checksum:
			self.batch_checksum = self._calculate_batch_checksum()
	
	def _calculate_batch_checksum(self) -> str:
		"""Calculate checksum for entire batch"""
		event_checksums = [event.checksum for event in self.events if event.checksum]
		combined_checksum = "".join(sorted(event_checksums))
		return hashlib.sha256(combined_checksum.encode()).hexdigest()


class ComplianceRule(BaseModel):
	"""Automated compliance rule definition"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Rule identifier")
	tenant_id: Annotated[Optional[str], AfterValidator(validate_tenant_id)] = Field(None, description="APG tenant identifier")
	name: str = Field(..., description="Rule name")
	description: str = Field(..., description="Rule description")
	framework: ComplianceFramework = Field(..., description="Compliance framework")
	
	# Rule Logic
	event_types: List[AuditEventType] = Field(..., description="Event types to monitor")
	conditions: Dict[str, Any] = Field(..., description="Rule conditions (JSON logic)")
	severity: Annotated[int, AfterValidator(validate_event_severity)] = Field(..., description="Violation severity (0-100)")
	
	# Actions
	alert_enabled: bool = Field(default=True, description="Enable alerting")
	auto_remediation: bool = Field(default=False, description="Enable automatic remediation")
	escalation_rules: Dict[str, Any] = Field(default_factory=dict, description="Escalation configuration")
	
	# Metadata
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Rule creation timestamp")
	updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
	created_by: Optional[str] = Field(None, description="Creator user ID")
	is_active: bool = Field(default=True, description="Rule active status")
	
	# Performance
	last_triggered: Optional[datetime] = Field(None, description="Last trigger timestamp")
	trigger_count: int = Field(default=0, description="Total trigger count")
	false_positive_count: int = Field(default=0, description="False positive count")


class AuditLifecycleEvent(BaseModel):
	"""Tenant-scoped immutable audit event for package composition."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str, description="Audit event identifier")
	tenant_id: str = Field(..., description="APG tenant identifier")
	timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event timestamp")
	actor: str = Field(..., description="Actor that performed the action")
	action: str = Field(..., description="Action performed")
	resource_type: str = Field(..., description="Affected resource type")
	resource_id: str = Field(..., description="Affected resource identifier")
	severity: str = Field("info", description="Audit severity")
	contains_pii: bool = Field(False, description="Whether event evidence includes PII")
	immutable: bool = Field(True, description="Whether immutable storage is required")
	checksum: str = Field(..., description="Event integrity checksum")
	details: Dict[str, Any] = Field(default_factory=dict, description="Additional event details")
	policy_decision: str = Field("allow", description="Policy decision that allowed or reviewed the event")
	matched_rules: List[str] = Field(default_factory=list, description="Policy rules matched for this event")
	review_reasons: List[str] = Field(default_factory=list, description="Policy reasons requiring review or blocking")
	audit_evidence: Dict[str, Any] = Field(default_factory=dict, description="Structured guardrail evidence")


class AuditLegalHoldRecord(BaseModel):
	"""Tenant-scoped legal hold lifecycle record."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	id: str = Field(..., description="Legal hold identifier")
	tenant_id: str = Field(..., description="APG tenant identifier")
	scope: Dict[str, Any] = Field(default_factory=dict, description="Hold scope")
	reason: str = Field(..., description="Legal hold reason")
	approver: str = Field(..., description="Approver applying the hold")
	status: str = Field("active", description="active or released")
	applied_at: datetime = Field(default_factory=datetime.utcnow, description="Hold timestamp")
	released_by: Optional[str] = Field(None, description="Actor releasing the hold")
	release_evidence: Optional[str] = Field(None, description="Release evidence")
	released_at: Optional[datetime] = Field(None, description="Release timestamp")


class AuditExportRequest(BaseModel):
	"""Governed audit evidence export request."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	id: str = Field(..., description="Export request identifier")
	tenant_id: str = Field(..., description="APG tenant identifier")
	query: Dict[str, Any] = Field(default_factory=dict, description="Export query")
	requested_by: str = Field(..., description="Requester identity")
	contains_pii: bool = Field(False, description="Whether export includes PII")
	masking_enabled: bool = Field(True, description="Whether PII masking is enabled")
	reason: str = Field(..., description="Export reason")
	decision: str = Field("pending", description="pending, approved, or rejected")
	reviewer: Optional[str] = Field(None, description="Reviewer identity")
	notes: Optional[str] = Field(None, description="Reviewer notes")
	requested_at: datetime = Field(default_factory=datetime.utcnow, description="Request timestamp")
	decided_at: Optional[datetime] = Field(None, description="Decision timestamp")
	policy_decision: str = Field("allow", description="Policy decision for export lifecycle")
	matched_rules: List[str] = Field(default_factory=list, description="Policy rules matched for export lifecycle")
	review_reasons: List[str] = Field(default_factory=list, description="Policy reasons attached to export lifecycle")
	audit_evidence: Dict[str, Any] = Field(default_factory=dict, description="Structured export guardrail evidence")


class AuditPurgeRequest(BaseModel):
	"""Dual-control audit purge request."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	id: str = Field(..., description="Purge request identifier")
	tenant_id: str = Field(..., description="APG tenant identifier")
	scope: Dict[str, Any] = Field(default_factory=dict, description="Purge scope")
	requested_by: str = Field(..., description="Requester identity")
	reason: str = Field(..., description="Purge reason")
	decision: str = Field("pending", description="pending, approved, or rejected")
	reviewer: Optional[str] = Field(None, description="Dual-control reviewer")
	notes: Optional[str] = Field(None, description="Reviewer notes")
	requested_at: datetime = Field(default_factory=datetime.utcnow, description="Request timestamp")
	decided_at: Optional[datetime] = Field(None, description="Decision timestamp")
	policy_decision: str = Field("allow", description="Policy decision for purge lifecycle")
	matched_rules: List[str] = Field(default_factory=list, description="Policy rules matched for purge lifecycle")
	review_reasons: List[str] = Field(default_factory=list, description="Policy reasons attached to purge lifecycle")
	audit_evidence: Dict[str, Any] = Field(default_factory=dict, description="Structured purge guardrail evidence")


class AuditInvestigationRecord(BaseModel):
	"""Investigation lifecycle over audit evidence."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	id: str = Field(..., description="Investigation identifier")
	tenant_id: str = Field(..., description="APG tenant identifier")
	event_ids: List[str] = Field(default_factory=list, description="Tenant audit event IDs")
	owner: str = Field(..., description="Investigation owner")
	priority: str = Field("high", description="Investigation priority")
	status: str = Field("open", description="open or closed")
	opened_at: datetime = Field(default_factory=datetime.utcnow, description="Open timestamp")
	closed_by: Optional[str] = Field(None, description="Actor closing the investigation")
	resolution: Optional[str] = Field(None, description="Resolution summary")
	evidence: Dict[str, Any] = Field(default_factory=dict, description="Resolution evidence")
	closed_at: Optional[datetime] = Field(None, description="Close timestamp")


class AuditAgentRecord(BaseModel):
	"""First-class audit agent registered for tenant-scoped evidence work."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	id: str = Field(..., description="Audit agent identifier")
	tenant_id: str = Field(..., description="APG tenant identifier")
	name: str = Field(..., description="Human-readable agent name")
	runtime: str = Field(..., description="Agent runtime identifier")
	role: str = Field(..., description="AUDL agent role")
	purpose: str = Field(..., description="Agent purpose and operating boundary")
	owner: str = Field(..., description="Accountable human owner")
	human_approval_required: bool = Field(True, description="Whether human approval gates privileged work")
	status: str = Field("active", description="active or disabled")
	configuration: Dict[str, Any] = Field(default_factory=dict, description="Runtime-specific configuration")
	registered_at: datetime = Field(default_factory=datetime.utcnow, description="Registration timestamp")
	policy_decision: str = Field("allow", description="Policy decision for audit agent registration")
	matched_rules: List[str] = Field(default_factory=list, description="Policy rules matched for audit agent registration")
	review_reasons: List[str] = Field(default_factory=list, description="Policy reasons attached to audit agent registration")
	audit_evidence: Dict[str, Any] = Field(default_factory=dict, description="Structured audit agent guardrail evidence")


class AuditBatchEvidence(BaseModel):
	"""Bytewax audit batch validation evidence."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str, description="Batch evidence identifier")
	tenant_id: str = Field(..., description="APG tenant identifier")
	record_count: int = Field(..., description="Number of audit events in the batch")
	event_stream: str = Field(..., description="Requested event stream")
	stream_processing_enabled: bool = Field(True, description="Whether stream processing was enabled")
	status: str = Field("accepted", description="accepted or denied")
	processor: str = Field("bytewax", description="Required lifecycle stream processor")
	policy_decision: str = Field("allow", description="Policy decision for batch validation")
	matched_rules: List[str] = Field(default_factory=list, description="Policy rules matched for batch validation")
	review_reasons: List[str] = Field(default_factory=list, description="Policy reasons attached to batch validation")
	audit_evidence: Dict[str, Any] = Field(default_factory=dict, description="Structured batch guardrail evidence")


class AuditGovernanceEvent(BaseModel):
	"""Tenant-scoped AUDL governance evidence event."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str, description="Governance event identifier")
	tenant_id: str = Field(..., description="APG tenant identifier")
	event_type: str = Field(..., description="Lifecycle event type")
	subject_id: str = Field(..., description="Subject record identifier")
	message: str = Field(..., description="Human-readable event message")
	evidence: Dict[str, Any] = Field(default_factory=dict, description="Structured evidence")
	policy_decision: str = Field("allow", description="Policy decision associated with the governance event")
	matched_rules: List[str] = Field(default_factory=list, description="Policy rules matched for the governance event")
	review_reasons: List[str] = Field(default_factory=list, description="Policy reasons attached to the governance event")
	audit_evidence: Dict[str, Any] = Field(default_factory=dict, description="Structured policy evidence for the governance event")
	timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event timestamp")


# Export all models
__all__ = [
	# Enums
	"AuditLevel", "AuditEventType", "EventSource", "ComplianceFramework", 
	"InvestigationStatus", "EvidenceType",
	
	# Core Models
	"AuditEvent", "AuditEventBatch", "ComplianceRule",
	"AuditLifecycleEvent", "AuditLegalHoldRecord", "AuditExportRequest",
	"AuditPurgeRequest", "AuditInvestigationRecord", "AuditAgentRecord",
	"AuditBatchEvidence", "AuditGovernanceEvent",
	
	# Validators
	"validate_tenant_id", "validate_event_severity", "validate_risk_score",
	"validate_compliance_tags"
]
