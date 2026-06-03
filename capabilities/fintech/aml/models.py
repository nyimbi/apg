"""Pydantic v2 models for APG Anti-Money Laundering capability.

Covers the full AML lifecycle: transaction monitoring, rule evaluation,
alert triage, case management, SAR/CTR filing, watchlist screening,
network analysis, risk segmentation, and regulatory reporting.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from uuid6 import uuid7


def uuid7str() -> str:
	return str(uuid7())


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class _StrEnum(str, Enum):
	"""Base for all AML enums — str(member) returns the value, not the name."""
	def __str__(self) -> str:
		return self.value


class AlertStatus(_StrEnum):
	OPEN = "open"
	UNDER_REVIEW = "under_review"
	ESCALATED = "escalated"
	CASE_OPENED = "case_opened"
	CLOSED = "closed"
	FALSE_POSITIVE = "false_positive"


class AlertSeverity(_StrEnum):
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	CRITICAL = "critical"


class AlertType(_StrEnum):
	LARGE_TRANSACTION = "large_transaction"
	STRUCTURING = "structuring"
	VELOCITY = "velocity"
	ROUND_TRIP = "round_trip"
	LAYERING = "layering"
	SANCTIONS = "sanctions"
	PEP = "pep"
	HIGH_RISK_KYC = "high_risk_kyc"
	MULE_ACCOUNT = "mule_account"
	TRADE_BASED = "trade_based"
	CRYPTO_ASSET = "crypto_asset"
	NFT = "nft"
	CORRESPONDENT = "correspondent"
	TERRORIST_FINANCING = "terrorist_financing"
	AGENT_REVIEW = "agent_review"


class CaseStatus(_StrEnum):
	OPEN = "open"
	UNDER_INVESTIGATION = "under_investigation"
	CONFIRMED_SUSPICIOUS = "confirmed_suspicious"
	SAR_FILED = "sar_filed"
	CLOSED_NO_ACTION = "closed_no_action"
	CLOSED_ACTION_TAKEN = "closed_action_taken"
	REFERRED_TO_LEA = "referred_to_lea"


class CaseType(_StrEnum):
	TRANSACTION_MONITORING = "transaction_monitoring"
	SANCTIONS_ALERT = "sanctions_alert"
	STRUCTURING_ALERT = "structuring_alert"
	MULE_ACCOUNT = "mule_account"
	HIGH_RISK_CUSTOMER = "high_risk_customer"
	TERRORIST_FINANCING = "terrorist_financing"
	TRADE_BASED_ML = "trade_based_ml"
	CRYPTO_ASSET = "crypto_asset"
	NETWORK_ANALYSIS = "network_analysis"
	SUSPICIOUS_ACTIVITY_REPORT = "suspicious_activity_report"


class RuleStatus(_StrEnum):
	ACTIVE = "active"
	INACTIVE = "inactive"
	TESTING = "testing"
	DEPRECATED = "deprecated"


class RuleType(_StrEnum):
	THRESHOLD = "threshold"
	VELOCITY = "velocity"
	PATTERN = "pattern"
	NETWORK = "network"
	WATCHLIST = "watchlist"
	BEHAVIORAL = "behavioral"
	GEOGRAPHIC = "geographic"
	PEER_COMPARISON = "peer_comparison"


class SARStatus(_StrEnum):
	DRAFT = "draft"
	PENDING_APPROVAL = "pending_approval"
	APPROVED = "approved"
	FILED = "filed"
	REJECTED = "rejected"
	AMENDED = "amended"


class CTRStatus(_StrEnum):
	PENDING = "pending"
	FILED = "filed"
	AMENDED = "amended"
	EXEMPT = "exempt"


class WatchlistMatchStatus(_StrEnum):
	PENDING = "pending"
	CONFIRMED = "confirmed"
	FALSE_POSITIVE = "false_positive"
	ESCALATED = "escalated"


class FilingStatus(_StrEnum):
	PENDING = "pending"
	SUBMITTED = "submitted"
	ACKNOWLEDGED = "acknowledged"
	REJECTED = "rejected"
	AMENDED = "amended"


class RiskSegment(_StrEnum):
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	VERY_HIGH = "very_high"
	PROHIBITED = "prohibited"


class TypologyCode(_StrEnum):
	SMURFING = "smurfing"
	ROUND_TRIP = "round_trip"
	LAYERING = "layering"
	TRADE_INVOICE_FRAUD = "trade_invoice_fraud"
	CRYPTO_MIXER = "crypto_mixer"
	NFT_WASH_TRADE = "nft_wash_trade"
	CORRESPONDENT_NESTING = "correspondent_nesting"
	SHELL_COMPANY = "shell_company"
	CASH_INTENSIVE = "cash_intensive"
	TERRORIST_FINANCING = "terrorist_financing"


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------

class AMLBase(BaseModel):
	model_config = ConfigDict(
		extra="forbid",
		validate_by_name=True,
		validate_by_alias=True,
		use_enum_values=True,
	)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
	is_deleted: bool = False


# ---------------------------------------------------------------------------
# TransactionMonitoringRule
# ---------------------------------------------------------------------------

class RuleCondition(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	field: str
	operator: str  # gt, lt, gte, lte, eq, in, contains, regex
	value: Any
	currency: str | None = None


class TransactionMonitoringRuleCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: str
	created_by: str
	name: str
	description: str
	rule_type: RuleType
	conditions: list[RuleCondition]
	alert_type: AlertType
	severity: AlertSeverity
	lookback_days: int = 30
	min_occurrences: int = 1
	score_weight: float = Field(default=1.0, ge=0.0, le=10.0)
	jurisdictions: list[str] = Field(default_factory=list)
	enabled: bool = True
	metadata: dict[str, Any] = Field(default_factory=dict)


class TransactionMonitoringRuleUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	name: str | None = None
	description: str | None = None
	conditions: list[RuleCondition] | None = None
	severity: AlertSeverity | None = None
	lookback_days: int | None = None
	min_occurrences: int | None = None
	score_weight: float | None = None
	enabled: bool | None = None
	status: RuleStatus | None = None
	metadata: dict[str, Any] | None = None


class TransactionMonitoringRuleResponse(AMLBase):
	name: str
	description: str
	rule_type: RuleType
	conditions: list[RuleCondition]
	alert_type: AlertType
	severity: AlertSeverity
	lookback_days: int
	min_occurrences: int
	score_weight: float
	jurisdictions: list[str]
	enabled: bool
	status: RuleStatus = RuleStatus.ACTIVE
	hit_count: int = 0
	false_positive_rate: float = 0.0
	metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# AMLAlert
# ---------------------------------------------------------------------------

class AMLAlertCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: str
	created_by: str
	alert_type: AlertType
	severity: AlertSeverity
	subject_reference: str
	kyc_profile_id: str | None = None
	rule_id: str | None = None
	transaction_ids: list[str] = Field(default_factory=list)
	evidence_references: list[str] = Field(default_factory=list)
	risk_score: int = Field(default=0, ge=0, le=100)
	typology_codes: list[TypologyCode] = Field(default_factory=list)
	amount: float | None = None
	currency: str | None = None
	narrative: str = ""
	metadata: dict[str, Any] = Field(default_factory=dict)


class AMLAlertUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	severity: AlertSeverity | None = None
	status: AlertStatus | None = None
	disposition: str | None = None
	reviewer_id: str | None = None
	narrative: str | None = None
	risk_score: int | None = None
	metadata: dict[str, Any] | None = None


class AMLAlertResponse(AMLBase):
	alert_type: AlertType
	severity: AlertSeverity
	status: AlertStatus = AlertStatus.OPEN
	subject_reference: str
	kyc_profile_id: str | None = None
	rule_id: str | None = None
	transaction_ids: list[str] = Field(default_factory=list)
	evidence_references: list[str] = Field(default_factory=list)
	risk_score: int = 0
	typology_codes: list[TypologyCode] = Field(default_factory=list)
	amount: float | None = None
	currency: str | None = None
	narrative: str = ""
	disposition: str = ""
	reviewer_id: str | None = None
	case_id: str | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# AMLCase
# ---------------------------------------------------------------------------

class AMLCaseCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: str
	created_by: str
	alert_id: str
	case_type: CaseType
	investigator_id: str
	subject_reference: str
	priority: int = Field(default=3, ge=1, le=5)
	evidence_references: list[str] = Field(default_factory=list)
	notes: str = ""
	due_date: datetime | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)


class AMLCaseUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	status: CaseStatus | None = None
	investigator_id: str | None = None
	priority: int | None = None
	evidence_references: list[str] | None = None
	notes: str | None = None
	due_date: datetime | None = None
	metadata: dict[str, Any] | None = None


class AMLCaseResponse(AMLBase):
	alert_id: str
	case_type: CaseType
	investigator_id: str
	subject_reference: str
	status: CaseStatus = CaseStatus.OPEN
	priority: int = 3
	evidence_references: list[str] = Field(default_factory=list)
	notes: str = ""
	due_date: datetime | None = None
	sar_id: str | None = None
	ctr_id: str | None = None
	closed_at: datetime | None = None
	closed_by: str | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# InvestigationNote
# ---------------------------------------------------------------------------

class InvestigationNoteCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: str
	created_by: str
	case_id: str
	body: str
	is_privileged: bool = False
	attachments: list[str] = Field(default_factory=list)


class InvestigationNoteResponse(AMLBase):
	case_id: str
	body: str
	is_privileged: bool = False
	attachments: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# SAR — Suspicious Activity Report
# ---------------------------------------------------------------------------

class SARCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: str
	created_by: str
	case_id: str
	subject_reference: str
	subject_name: str
	subject_dob: str | None = None
	subject_tin: str | None = None
	subject_address: str = ""
	jurisdiction: str
	filing_institution: str
	narrative: str
	suspicious_activity_start: datetime
	suspicious_activity_end: datetime
	total_amount: float
	currency: str
	transaction_ids: list[str] = Field(default_factory=list)
	evidence_references: list[str] = Field(default_factory=list)
	typology_codes: list[TypologyCode] = Field(default_factory=list)
	metadata: dict[str, Any] = Field(default_factory=dict)


class SARUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	narrative: str | None = None
	status: SARStatus | None = None
	approved_by: str | None = None
	filing_reference: str | None = None
	rejection_reason: str | None = None
	metadata: dict[str, Any] | None = None


class SARResponse(AMLBase):
	case_id: str
	subject_reference: str
	subject_name: str
	subject_dob: str | None = None
	subject_tin: str | None = None
	subject_address: str = ""
	jurisdiction: str
	filing_institution: str
	narrative: str
	suspicious_activity_start: datetime
	suspicious_activity_end: datetime
	total_amount: float
	currency: str
	transaction_ids: list[str] = Field(default_factory=list)
	evidence_references: list[str] = Field(default_factory=list)
	typology_codes: list[TypologyCode] = Field(default_factory=list)
	status: SARStatus = SARStatus.DRAFT
	approved_by: str | None = None
	approved_at: datetime | None = None
	filing_reference: str | None = None
	filed_at: datetime | None = None
	rejection_reason: str | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# CTR — Currency Transaction Report
# ---------------------------------------------------------------------------

class CTRCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: str
	created_by: str
	transaction_id: str
	subject_reference: str
	subject_name: str
	subject_id_number: str | None = None
	amount: float
	currency: str
	transaction_date: datetime
	transaction_type: str
	branch_id: str | None = None
	jurisdiction: str
	filing_institution: str
	metadata: dict[str, Any] = Field(default_factory=dict)


class CTRResponse(AMLBase):
	transaction_id: str
	subject_reference: str
	subject_name: str
	subject_id_number: str | None = None
	amount: float
	currency: str
	transaction_date: datetime
	transaction_type: str
	branch_id: str | None = None
	jurisdiction: str
	filing_institution: str
	status: CTRStatus = CTRStatus.PENDING
	filing_reference: str | None = None
	filed_at: datetime | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# WatchlistMatch
# ---------------------------------------------------------------------------

class WatchlistMatchCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: str
	created_by: str
	subject_reference: str
	subject_name: str
	list_name: str  # OFAC SDN, EU Consolidated, UN Consolidated, PEP, etc.
	list_entry_id: str
	match_score: float = Field(ge=0.0, le=1.0)
	match_fields: list[str] = Field(default_factory=list)
	matched_name: str = ""
	matched_dob: str | None = None
	matched_nationality: str | None = None
	kyc_profile_id: str | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)


class WatchlistMatchResponse(AMLBase):
	subject_reference: str
	subject_name: str
	list_name: str
	list_entry_id: str
	match_score: float
	match_fields: list[str]
	matched_name: str = ""
	matched_dob: str | None = None
	matched_nationality: str | None = None
	kyc_profile_id: str | None = None
	status: WatchlistMatchStatus = WatchlistMatchStatus.PENDING
	reviewer_id: str | None = None
	reviewed_at: datetime | None = None
	alert_id: str | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# RegulatoryFiling
# ---------------------------------------------------------------------------

class RegulatoryFilingCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: str
	created_by: str
	filing_type: str  # SAR, CTR, STR, etc.
	jurisdiction: str
	regulator: str
	reference_id: str  # SAR or CTR id
	period_start: datetime
	period_end: datetime
	filing_institution: str
	metadata: dict[str, Any] = Field(default_factory=dict)


class RegulatoryFilingResponse(AMLBase):
	filing_type: str
	jurisdiction: str
	regulator: str
	reference_id: str
	period_start: datetime
	period_end: datetime
	filing_institution: str
	status: FilingStatus = FilingStatus.PENDING
	submission_reference: str | None = None
	submitted_at: datetime | None = None
	acknowledged_at: datetime | None = None
	rejection_reason: str | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# RiskSegmentRecord
# ---------------------------------------------------------------------------

class RiskSegmentCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: str
	created_by: str
	subject_reference: str
	kyc_profile_id: str | None = None
	segment: RiskSegment
	risk_score: int = Field(ge=0, le=100)
	contributing_factors: list[str] = Field(default_factory=list)
	effective_date: datetime = Field(default_factory=datetime.utcnow)
	review_date: datetime | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)


class RiskSegmentResponse(AMLBase):
	subject_reference: str
	kyc_profile_id: str | None = None
	segment: RiskSegment
	risk_score: int
	contributing_factors: list[str]
	effective_date: datetime
	review_date: datetime | None = None
	previous_segment: RiskSegment | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Report / Aggregation models
# ---------------------------------------------------------------------------

class AlertSummary(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	total: int = 0
	open: int = 0
	under_review: int = 0
	escalated: int = 0
	closed: int = 0
	false_positive: int = 0
	critical: int = 0
	high: int = 0
	medium: int = 0
	low: int = 0
	by_type: dict[str, int] = Field(default_factory=dict)


class CaseSummary(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	total: int = 0
	open: int = 0
	under_investigation: int = 0
	confirmed_suspicious: int = 0
	sar_filed: int = 0
	closed: int = 0
	avg_days_to_close: float = 0.0
	overdue: int = 0


class RegulatoryReportRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: str
	jurisdiction: str
	period_start: datetime
	period_end: datetime
	report_type: str = "sar_ctr_summary"


class RegulatoryReportResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	tenant_id: str
	jurisdiction: str
	period_start: datetime
	period_end: datetime
	report_type: str
	sar_count: int = 0
	ctr_count: int = 0
	alert_count: int = 0
	case_count: int = 0
	total_suspicious_amount: float = 0.0
	currency: str = "USD"
	generated_at: datetime = Field(default_factory=datetime.utcnow)
	details: list[dict[str, Any]] = Field(default_factory=list)


class NetworkAnalysisResult(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	subject_reference: str
	tenant_id: str
	analysed_at: datetime = Field(default_factory=datetime.utcnow)
	counterparty_count: int = 0
	high_risk_counterparties: list[str] = Field(default_factory=list)
	transaction_count: int = 0
	total_sent: float = 0.0
	total_received: float = 0.0
	round_trip_detected: bool = False
	layering_detected: bool = False
	network_risk_score: int = 0
	clusters: list[dict[str, Any]] = Field(default_factory=list)
	typology_flags: list[TypologyCode] = Field(default_factory=list)


class PatternDetectionResult(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)
	subject_reference: str
	tenant_id: str
	lookback_days: int
	analysed_at: datetime = Field(default_factory=datetime.utcnow)
	structuring_detected: bool = False
	smurfing_detected: bool = False
	velocity_anomaly: bool = False
	round_trip_detected: bool = False
	layering_detected: bool = False
	patterns: list[dict[str, Any]] = Field(default_factory=list)
	risk_delta: int = 0
	recommended_action: str = "no_action"
