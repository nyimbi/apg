"""
APG Audit Logging — Pydantic v2 domain models.

Entities: AuditEvent, AuditTrail, ComplianceReport, RetentionPolicy,
          DataSubjectRequest, EvidencePackage, TamperDetection, AuditQuery

© 2025 Datacraft  www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from uuid6 import uuid7


# ---------------------------------------------------------------------------
# UUID-7 helper
# ---------------------------------------------------------------------------

def uuid7str() -> str:
	"""Return a lexicographically-sortable UUID-7 string."""
	return str(uuid7())


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AuditLevel(StrEnum):
	DEBUG    = "debug"
	INFO     = "info"
	WARNING  = "warning"
	ERROR    = "error"
	CRITICAL = "critical"


class AuditEventType(StrEnum):
	# Authentication
	USER_LOGIN           = "user_login"
	USER_LOGOUT          = "user_logout"
	USER_FAILED_LOGIN    = "user_failed_login"
	USER_PASSWORD_CHANGE = "user_password_change"
	USER_MFA_ENABLED     = "user_mfa_enabled"
	USER_MFA_DISABLED    = "user_mfa_disabled"
	# Authorisation
	PERMISSION_GRANTED   = "permission_granted"
	PERMISSION_REVOKED   = "permission_revoked"
	ROLE_ASSIGNED        = "role_assigned"
	ROLE_REMOVED         = "role_removed"
	ACCESS_DENIED        = "access_denied"
	# Data operations
	DATA_READ            = "data_read"
	DATA_CREATE          = "data_create"
	DATA_UPDATE          = "data_update"
	DATA_DELETE          = "data_delete"
	DATA_EXPORT          = "data_export"
	DATA_IMPORT          = "data_import"
	DATA_ACCESS          = "data_access"      # generic / legacy alias
	# System
	SYSTEM_START         = "system_start"
	SYSTEM_STOP          = "system_stop"
	SYSTEM_RESTART       = "system_restart"
	CONFIG_CHANGE        = "config_change"
	MAINTENANCE_START    = "maintenance_start"
	MAINTENANCE_END      = "maintenance_end"
	# API
	API_CALL             = "api_call"
	API_ERROR            = "api_error"
	API_RATE_LIMIT       = "api_rate_limit"
	WEBHOOK_TRIGGERED    = "webhook_triggered"
	# Security
	SECURITY_ALERT          = "security_alert"
	INTRUSION_ATTEMPT       = "intrusion_attempt"
	MALWARE_DETECTED        = "malware_detected"
	VULNERABILITY_FOUND     = "vulnerability_found"
	SECURITY_POLICY_VIOLATION = "security_policy_violation"
	# Compliance
	COMPLIANCE_VIOLATION    = "compliance_violation"
	AUDIT_TRAIL_ACCESS      = "audit_trail_access"
	EVIDENCE_COLLECTED      = "evidence_collected"
	LEGAL_HOLD_APPLIED      = "legal_hold_applied"
	LEGAL_HOLD_RELEASED     = "legal_hold_released"
	# Investigation
	INVESTIGATION_CREATED   = "investigation_created"
	INVESTIGATION_UPDATED   = "investigation_updated"
	INVESTIGATION_CLOSED    = "investigation_closed"
	EVIDENCE_ADDED          = "evidence_added"
	# Custom
	CUSTOM_EVENT            = "custom_event"


class EventSource(StrEnum):
	APG_CORE        = "apg_core"
	AUTH            = "auth"
	MULTI_TENANT    = "multi_tenant"
	NOTIFICATIONS   = "notifications"
	NLP_CORE        = "nlp_core"
	SECURITY        = "security"
	COMPLIANCE      = "compliance"
	COLLABORATION   = "collaboration"
	API_GATEWAY     = "api_gateway"
	EXTERNAL_SYSTEM = "external_system"


class ComplianceFramework(StrEnum):
	SOX      = "SOX"
	GDPR     = "GDPR"
	HIPAA    = "HIPAA"
	PCI_DSS  = "PCI-DSS"
	ISO_27001= "ISO-27001"
	SOC_2    = "SOC-2"
	NIST     = "NIST"
	CIS      = "CIS"


class RetentionAction(StrEnum):
	ARCHIVE = "archive"
	DELETE  = "delete"
	REVIEW  = "review"


class DSRType(StrEnum):
	"""GDPR / data-subject request type."""
	ACCESS       = "access"        # Art. 15 — right of access
	ERASURE      = "erasure"       # Art. 17 — right to be forgotten
	PORTABILITY  = "portability"   # Art. 20
	RECTIFICATION= "rectification" # Art. 16
	RESTRICTION  = "restriction"   # Art. 18
	OBJECTION    = "objection"     # Art. 21


class DSRStatus(StrEnum):
	PENDING    = "pending"
	IN_REVIEW  = "in_review"
	FULFILLED  = "fulfilled"
	REJECTED   = "rejected"
	PARTIAL    = "partial"


class EvidencePackageStatus(StrEnum):
	ASSEMBLING = "assembling"
	READY      = "ready"
	EXPORTED   = "exported"
	SEALED     = "sealed"


class TamperStatus(StrEnum):
	CLEAN      = "clean"
	SUSPECT    = "suspect"
	CONFIRMED  = "confirmed"


class TrailStatus(StrEnum):
	ACTIVE   = "active"
	CLOSED   = "closed"
	ARCHIVED = "archived"


class ReportStatus(StrEnum):
	PENDING    = "pending"
	GENERATING = "generating"
	READY      = "ready"
	FAILED     = "failed"


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def _validate_risk_score(v: float) -> float:
	assert 0.0 <= v <= 1.0, f"risk/anomaly score must be 0–1, got {v}"
	return float(v)


def _utcnow() -> datetime:
	return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------

class AuditBase(BaseModel):
	"""Common audit columns present on every entity."""
	model_config = ConfigDict(
		extra="forbid",
		validate_by_name=True,
		validate_by_alias=True,
	)

	id:         str      = Field(default_factory=uuid7str)
	tenant_id:  str      = Field(..., description="Tenant identifier — enforced on every query")
	created_at: datetime = Field(default_factory=_utcnow)
	updated_at: datetime = Field(default_factory=_utcnow)
	created_by: str | None = Field(None)
	is_deleted: bool     = Field(False)


# ---------------------------------------------------------------------------
# AuditEvent  (AL_)
# ---------------------------------------------------------------------------

class AuditEventCreate(BaseModel):
	"""Input DTO for logging a single audit event."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id:           str
	level:               AuditLevel
	event_type:          AuditEventType
	source:              EventSource
	category:            str
	subcategory:         str | None     = None
	# Who
	actor_id:            str | None     = None
	actor_type:          str            = "user"
	actor_display_name:  str | None     = None
	session_id:          str | None     = None
	service_account:     str | None     = None
	# What
	action:              str
	action_description:  str | None     = None
	operation_id:        str | None     = None
	# On what
	resource_type:       str | None     = None
	resource_id:         str | None     = None
	resource_name:       str | None     = None
	resource_path:       str | None     = None
	parent_resource_id:  str | None     = None
	# Where / how
	ip_address:          str | None     = None
	user_agent:          str | None     = None
	geographic_location: str | None     = None
	device_id:           str | None     = None
	request_id:          str | None     = None
	correlation_id:      str | None     = None
	# Outcome
	success:             bool           = True
	status_code:         int | None     = None
	error_code:          str | None     = None
	error_message:       str | None     = None
	duration_ms:         int | None     = None
	# Compliance / classification
	compliance_tags:     list[str]      = Field(default_factory=list)
	data_classification: str | None     = None
	retention_days:      int            = 2555   # 7 years default
	legal_hold:          bool           = False
	contains_pii:        bool           = False
	# Freeform context
	details:             dict[str, Any] = Field(default_factory=dict)
	tags:                dict[str, str] = Field(default_factory=dict)


class AuditEventResponse(AuditBase):
	"""Full audit event as returned by the service."""
	level:               AuditLevel
	event_type:          AuditEventType
	source:              EventSource
	category:            str
	subcategory:         str | None     = None
	# Who
	actor_id:            str | None     = None
	actor_type:          str            = "user"
	actor_display_name:  str | None     = None
	session_id:          str | None     = None
	service_account:     str | None     = None
	# What
	action:              str
	action_description:  str | None     = None
	operation_id:        str | None     = None
	# On what
	resource_type:       str | None     = None
	resource_id:         str | None     = None
	resource_name:       str | None     = None
	resource_path:       str | None     = None
	parent_resource_id:  str | None     = None
	# Where / how
	ip_address:          str | None     = None
	user_agent:          str | None     = None
	geographic_location: str | None     = None
	device_id:           str | None     = None
	request_id:          str | None     = None
	correlation_id:      str | None     = None
	# Outcome
	success:             bool           = True
	status_code:         int | None     = None
	error_code:          str | None     = None
	error_message:       str | None     = None
	duration_ms:         int | None     = None
	# Risk / anomaly
	risk_score:    Annotated[float, _validate_risk_score] = 0.0
	anomaly_score: Annotated[float, _validate_risk_score] = 0.0
	threat_indicators: list[str]        = Field(default_factory=list)
	behavioral_tags:   list[str]        = Field(default_factory=list)
	# Compliance / classification
	compliance_tags:     list[str]      = Field(default_factory=list)
	data_classification: str | None     = None
	retention_days:      int            = 2555
	legal_hold:          bool           = False
	contains_pii:        bool           = False
	# Freeform context
	details:             dict[str, Any] = Field(default_factory=dict)
	tags:                dict[str, str] = Field(default_factory=dict)
	# Integrity
	checksum:        str | None = None
	chain_hash:      str | None = None    # HMAC chain linking to previous event
	immutable:       bool       = True

	def verify_integrity(self) -> bool:
		"""Re-derive and compare checksum."""
		return bool(self.checksum) and self.checksum == self._derive_checksum()

	def _derive_checksum(self) -> str:
		payload = json.dumps({
			"id":           self.id,
			"tenant_id":    self.tenant_id,
			"timestamp":    self.created_at.isoformat(),
			"event_type":   self.event_type,
			"actor_id":     self.actor_id,
			"action":       self.action,
			"resource_type":self.resource_type,
			"resource_id":  self.resource_id,
			"success":      self.success,
		}, sort_keys=True)
		return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# AuditTrail  (AT_)
# ---------------------------------------------------------------------------

class AuditTrailCreate(BaseModel):
	"""Create a named trail that groups related events."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id:   str
	name:        str
	description: str | None = None
	tags:        dict[str, str] = Field(default_factory=dict)


class AuditTrailUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	name:        str | None = None
	description: str | None = None
	status:      TrailStatus | None = None
	tags:        dict[str, str] | None = None


class AuditTrailResponse(AuditBase):
	name:        str
	description: str | None         = None
	status:      TrailStatus        = TrailStatus.ACTIVE
	event_count: int                = 0
	tags:        dict[str, str]     = Field(default_factory=dict)
	closed_at:   datetime | None    = None
	closed_by:   str | None         = None


# ---------------------------------------------------------------------------
# ComplianceReport  (CR_)
# ---------------------------------------------------------------------------

class ComplianceReportCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id:               str
	framework:               ComplianceFramework
	period_start:            datetime
	period_end:              datetime
	requested_by:            str
	include_violations:      bool = True
	include_recommendations: bool = True
	export_format:           str  = "json"   # json | pdf | csv


class ComplianceReportResponse(AuditBase):
	framework:               ComplianceFramework
	period_start:            datetime
	period_end:              datetime
	requested_by:            str
	status:                  ReportStatus    = ReportStatus.PENDING
	include_violations:      bool            = True
	include_recommendations: bool            = True
	export_format:           str             = "json"
	violation_count:         int             = 0
	summary:                 dict[str, Any]  = Field(default_factory=dict)
	file_path:               str | None      = None
	completed_at:            datetime | None = None
	error_detail:            str | None      = None


# ---------------------------------------------------------------------------
# RetentionPolicy  (RP_)
# ---------------------------------------------------------------------------

class RetentionPolicyCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id:          str
	name:               str
	description:        str | None          = None
	event_types:        list[AuditEventType]= Field(default_factory=list)
	data_classifications: list[str]         = Field(default_factory=list)
	retain_days:        int                 = Field(..., gt=0)
	archive_after_days: int | None          = None
	action_on_expiry:   RetentionAction     = RetentionAction.ARCHIVE
	is_active:          bool                = True


class RetentionPolicyUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	name:               str | None              = None
	description:        str | None              = None
	event_types:        list[AuditEventType]    | None = None
	data_classifications: list[str]             | None = None
	retain_days:        int | None              = None
	archive_after_days: int | None              = None
	action_on_expiry:   RetentionAction | None  = None
	is_active:          bool | None             = None


class RetentionPolicyResponse(AuditBase):
	name:               str
	description:        str | None          = None
	event_types:        list[AuditEventType]= Field(default_factory=list)
	data_classifications: list[str]         = Field(default_factory=list)
	retain_days:        int
	archive_after_days: int | None          = None
	action_on_expiry:   RetentionAction     = RetentionAction.ARCHIVE
	is_active:          bool                = True
	last_enforced_at:   datetime | None     = None
	events_affected:    int                 = 0


# ---------------------------------------------------------------------------
# DataSubjectRequest  (DS_)
# ---------------------------------------------------------------------------

class DataSubjectRequestCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id:     str
	dsr_type:      DSRType
	subject_id:    str         # the data subject (user/person)
	requested_by:  str
	justification: str
	scope_details: dict[str, Any] = Field(default_factory=dict)


class DataSubjectRequestUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	status:       DSRStatus | None         = None
	reviewer_id:  str | None               = None
	notes:        str | None               = None
	response_data:dict[str, Any] | None    = None


class DataSubjectRequestResponse(AuditBase):
	dsr_type:      DSRType
	subject_id:    str
	requested_by:  str
	justification: str
	scope_details: dict[str, Any]  = Field(default_factory=dict)
	status:        DSRStatus       = DSRStatus.PENDING
	reviewer_id:   str | None      = None
	notes:         str | None      = None
	response_data: dict[str, Any]  = Field(default_factory=dict)
	fulfilled_at:  datetime | None = None
	# For erasure DSRs: audit_impact lists events that *cannot* be erased
	audit_impact:  list[str]       = Field(default_factory=list)


# ---------------------------------------------------------------------------
# EvidencePackage  (EP_)
# ---------------------------------------------------------------------------

class EvidencePackageCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id:      str
	name:           str
	description:    str | None    = None
	event_ids:      list[str]     = Field(default_factory=list)
	trail_ids:      list[str]     = Field(default_factory=list)
	requested_by:   str
	reason:         str
	legal_matter:   str | None    = None
	include_chain:  bool          = True   # include hash-chain proof
	export_format:  str           = "zip"


class EvidencePackageResponse(AuditBase):
	name:           str
	description:    str | None          = None
	event_ids:      list[str]           = Field(default_factory=list)
	trail_ids:      list[str]           = Field(default_factory=list)
	requested_by:   str
	reason:         str
	legal_matter:   str | None          = None
	status:         EvidencePackageStatus = EvidencePackageStatus.ASSEMBLING
	include_chain:  bool                = True
	export_format:  str                 = "zip"
	file_path:      str | None          = None
	file_checksum:  str | None          = None
	event_count:    int                 = 0
	sealed_at:      datetime | None     = None
	sealed_by:      str | None          = None
	chain_of_custody: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# TamperDetection  (TD_)
# ---------------------------------------------------------------------------

class TamperDetectionCreate(BaseModel):
	"""Record a tamper-detection scan result."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id:       str
	scan_type:       str             # "scheduled" | "on-demand" | "triggered"
	scanned_by:      str
	scope_filter:    dict[str, Any]  = Field(default_factory=dict)


class TamperDetectionResponse(AuditBase):
	scan_type:       str
	scanned_by:      str
	scope_filter:    dict[str, Any]  = Field(default_factory=dict)
	status:          TamperStatus    = TamperStatus.CLEAN
	events_scanned:  int             = 0
	events_suspect:  int             = 0
	suspect_ids:     list[str]       = Field(default_factory=list)
	detail:          dict[str, Any]  = Field(default_factory=dict)
	completed_at:    datetime | None = None


# ---------------------------------------------------------------------------
# AuditQuery  (AQ_)
# ---------------------------------------------------------------------------

class AuditQueryCreate(BaseModel):
	"""Persist a structured or natural-language audit query for replay / scheduling."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id:        str
	name:             str | None         = None
	query_type:       str                = "structured"  # "structured" | "nlp" | "sql"
	# Structured filters
	event_types:      list[AuditEventType] = Field(default_factory=list)
	actor_ids:        list[str]           = Field(default_factory=list)
	resource_ids:     list[str]           = Field(default_factory=list)
	sources:          list[EventSource]   = Field(default_factory=list)
	date_start:       datetime | None     = None
	date_end:         datetime | None     = None
	risk_score_min:   float | None        = None
	risk_score_max:   float | None        = None
	compliance_tags:  list[str]           = Field(default_factory=list)
	success:          bool | None         = None
	full_text:        str | None          = None
	# NLP / raw SQL
	nlp_query:        str | None          = None
	raw_sql:          str | None          = None
	# Result control
	limit:            int                 = Field(100, gt=0, le=10_000)
	offset:           int                 = Field(0, ge=0)
	sort_by:          str                 = "created_at"
	sort_desc:        bool                = True
	requested_by:     str


class AuditQueryResponse(AuditBase):
	name:             str | None           = None
	query_type:       str                  = "structured"
	event_types:      list[AuditEventType] = Field(default_factory=list)
	actor_ids:        list[str]            = Field(default_factory=list)
	resource_ids:     list[str]            = Field(default_factory=list)
	sources:          list[EventSource]    = Field(default_factory=list)
	date_start:       datetime | None      = None
	date_end:         datetime | None      = None
	risk_score_min:   float | None         = None
	risk_score_max:   float | None         = None
	compliance_tags:  list[str]            = Field(default_factory=list)
	success:          bool | None          = None
	full_text:        str | None           = None
	nlp_query:        str | None           = None
	raw_sql:          str | None           = None
	limit:            int                  = 100
	offset:           int                  = 0
	sort_by:          str                  = "created_at"
	sort_desc:        bool                 = True
	requested_by:     str
	result_count:     int                  = 0
	executed_at:      datetime | None      = None
	duration_ms:      int | None           = None


# ---------------------------------------------------------------------------
# Search / report aggregation models
# ---------------------------------------------------------------------------

class AuditSearchResult(BaseModel):
	"""Paginated search response."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	query_id:      str | None              = None
	total_count:   int
	events:        list[AuditEventResponse]
	query_ms:      float
	has_more:      bool


class RiskSummary(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id:         str
	period_start:      datetime
	period_end:        datetime
	total_events:      int
	high_risk_count:   int
	anomaly_count:     int
	compliance_violations: int
	by_event_type:     dict[str, int]  = Field(default_factory=dict)
	by_source:         dict[str, int]  = Field(default_factory=dict)
	top_actors:        list[dict[str, Any]] = Field(default_factory=list)


class SIEMEvent(BaseModel):
	"""Flattened event payload for SIEM / syslog export."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	event_id:    str
	tenant_id:   str
	timestamp:   datetime
	level:       AuditLevel
	event_type:  AuditEventType
	source:      EventSource
	actor_id:    str | None
	action:      str
	resource_id: str | None
	success:     bool
	risk_score:  float
	ip_address:  str | None
	checksum:    str | None


__all__ = [
	# helpers
	"uuid7str",
	# enums
	"AuditLevel", "AuditEventType", "EventSource", "ComplianceFramework",
	"RetentionAction", "DSRType", "DSRStatus",
	"EvidencePackageStatus", "TamperStatus", "TrailStatus", "ReportStatus",
	# AuditEvent
	"AuditEventCreate", "AuditEventResponse",
	# AuditTrail
	"AuditTrailCreate", "AuditTrailUpdate", "AuditTrailResponse",
	# ComplianceReport
	"ComplianceReportCreate", "ComplianceReportResponse",
	# RetentionPolicy
	"RetentionPolicyCreate", "RetentionPolicyUpdate", "RetentionPolicyResponse",
	# DataSubjectRequest
	"DataSubjectRequestCreate", "DataSubjectRequestUpdate", "DataSubjectRequestResponse",
	# EvidencePackage
	"EvidencePackageCreate", "EvidencePackageResponse",
	# TamperDetection
	"TamperDetectionCreate", "TamperDetectionResponse",
	# AuditQuery
	"AuditQueryCreate", "AuditQueryResponse",
	# Aggregates
	"AuditSearchResult", "RiskSummary", "SIEMEvent",
]
