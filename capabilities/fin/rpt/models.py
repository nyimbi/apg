"""
APG Financial Reporting — Domain Models
© 2025 Datacraft. Author: Nyimbi Odero

Pydantic v2 request/response models plus lightweight in-memory dataclasses
used by the service layer. No SQLAlchemy dependency required for the runtime.

Covers all entities: Report, ReportDefinition, ReportSchedule, ReportOutput,
FinancialStatement, ConsolidationGroup, SegmentReport, XBRLTag,
RegulatorySubmission — plus KPI, Commentary, DrillDown, and ComparisonPeriods.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ── uuid7 shim ────────────────────────────────────────────────────────────────
try:
	from uuid6 import uuid7

	def uuid7str() -> str:
		return str(uuid7())

except ImportError:
	import uuid

	def uuid7str() -> str:  # type: ignore[misc]
		return str(uuid.uuid4())


# ── Status Enums ──────────────────────────────────────────────────────────────

class ReportStatus(str, Enum):
	DRAFT       = "draft"
	PENDING     = "pending"
	GENERATING  = "generating"
	GENERATED   = "generated"
	REVIEWING   = "reviewing"
	APPROVED    = "approved"
	PUBLISHED   = "published"
	DISTRIBUTED = "distributed"
	ARCHIVED    = "archived"
	FAILED      = "failed"
	CANCELLED   = "cancelled"


class StatementType(str, Enum):
	BALANCE_SHEET      = "balance_sheet"
	INCOME_STATEMENT   = "income_statement"
	CASH_FLOW          = "cash_flow"
	EQUITY_STATEMENT   = "equity_statement"
	MANAGEMENT_REPORT  = "management_report"
	SEGMENT_REPORT     = "segment_report"
	CONSOLIDATED       = "consolidated"
	REGULATORY         = "regulatory"
	XBRL               = "xbrl"
	CUSTOM             = "custom"


class AccountingStandard(str, Enum):
	IFRS       = "ifrs"
	US_GAAP    = "us_gaap"
	LOCAL_GAAP = "local_gaap"
	MANAGEMENT = "management"
	REGULATORY = "regulatory"


class ConsolidationMethod(str, Enum):
	FULL          = "full"
	PROPORTIONAL  = "proportional"
	EQUITY        = "equity"
	NONE          = "none"


class OutputFormat(str, Enum):
	PDF  = "pdf"
	XLSX = "xlsx"
	HTML = "html"
	JSON = "json"
	XBRL = "xbrl"
	CSV  = "csv"


class PeriodType(str, Enum):
	DAILY       = "daily"
	WEEKLY      = "weekly"
	MONTHLY     = "monthly"
	QUARTERLY   = "quarterly"
	SEMI_ANNUAL = "semi_annual"
	ANNUAL      = "annual"
	CUSTOM      = "custom"


class DisclosureType(str, Enum):
	ACCOUNTING_POLICY    = "accounting_policy"
	SIGNIFICANT_ESTIMATE = "significant_estimate"
	CONTINGENT_LIABILITY = "contingent_liability"
	RELATED_PARTY        = "related_party"
	SEGMENT              = "segment"
	REGULATORY           = "regulatory"
	RISK                 = "risk"
	OTHER                = "other"


class XBRLTaxonomy(str, Enum):
	IFRS_FULL = "ifrs-full"
	US_GAAP   = "us-gaap"
	GLF       = "glf"
	ESRS      = "esrs"
	CUSTOM    = "custom"


class FilingJurisdiction(str, Enum):
	SEC    = "sec"
	FCA    = "fca"
	ESMA   = "esma"
	CMA    = "cma"
	NSE    = "nse"
	JSE    = "jse"
	CUSTOM = "custom"


class AgentRole(str, Enum):
	STATEMENT_REVIEWER       = "statement_reviewer"
	CONSOLIDATION_REVIEWER   = "consolidation_reviewer"
	DISCLOSURE_REVIEWER      = "disclosure_reviewer"
	DISTRIBUTION_REVIEWER    = "distribution_reviewer"
	VARIANCE_NARRATIVE_REVIEWER = "variance_narrative_reviewer"
	CLOSE_REPORTING_REVIEWER = "close_reporting_reviewer"
	XBRL_TAGGER              = "xbrl_tagger"
	REGULATORY_PREPARER      = "regulatory_preparer"


class AgentRuntime(str, Enum):
	CODEX       = "codex"
	CLAUDE_CODE = "claude_code"
	OPENCODE    = "opencode"
	PI          = "pi"


class NarrativeSignificance(str, Enum):
	LOW    = "low"
	MEDIUM = "medium"
	HIGH   = "high"


class KPIStatus(str, Enum):
	OK      = "ok"
	WARNING = "warning"
	ALERT   = "alert"


class ScheduleFrequency(str, Enum):
	DAILY     = "daily"
	WEEKLY    = "weekly"
	MONTHLY   = "monthly"
	QUARTERLY = "quarterly"
	ANNUAL    = "annual"
	ON_DEMAND = "on_demand"


# ── Shared config ─────────────────────────────────────────────────────────────

_CFG = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)


# ── Base model ────────────────────────────────────────────────────────────────

class APGBase(BaseModel):
	"""Common audit fields for every APG entity."""
	model_config = _CFG

	id:         str      = Field(default_factory=uuid7str)
	tenant_id:  str
	created_at: str      = Field(default_factory=lambda: datetime.utcnow().isoformat())
	updated_at: str      = Field(default_factory=lambda: datetime.utcnow().isoformat())
	created_by: str      = "system"
	is_deleted: bool     = False


# ══════════════════════════════════════════════════════════════════════════════
# ReportDefinition — the master template for a statement
# ══════════════════════════════════════════════════════════════════════════════

class ReportDefinitionCreate(BaseModel):
	"""Request to create a report template / definition."""
	model_config = _CFG

	id:                  str              = Field(default_factory=uuid7str)
	tenant_id:           str
	name:                str              = Field(min_length=1, max_length=200)
	statement_type:      StatementType
	accounting_standard: AccountingStandard = AccountingStandard.IFRS
	description:         str | None       = None
	owner:               str
	currency_code:       str              = "USD"
	comparative_periods: int              = Field(default=1, ge=0, le=5)
	created_by:          str              = "system"


class ReportDefinitionUpdate(BaseModel):
	"""Partial-update payload for a report definition."""
	model_config = _CFG

	name:                str | None              = None
	description:         str | None              = None
	accounting_standard: AccountingStandard | None = None
	currency_code:       str | None              = None
	comparative_periods: int | None              = None


class ReportDefinitionResponse(APGBase):
	"""Full response for a report definition."""
	name:                str
	statement_type:      StatementType
	accounting_standard: AccountingStandard
	description:         str | None
	owner:               str
	currency_code:       str
	comparative_periods: int
	line_count:          int
	status:              ReportStatus


# ══════════════════════════════════════════════════════════════════════════════
# ReportSchedule
# ══════════════════════════════════════════════════════════════════════════════

class ReportScheduleCreate(BaseModel):
	model_config = _CFG

	id:            str            = Field(default_factory=uuid7str)
	tenant_id:     str
	definition_id: str
	name:          str            = Field(min_length=1, max_length=200)
	period_type:   PeriodType
	frequency:     ScheduleFrequency = ScheduleFrequency.MONTHLY
	output_format: OutputFormat   = OutputFormat.PDF
	recipients:    list[str]      = Field(default_factory=list)
	auto_publish:  bool           = False
	enabled:       bool           = True
	created_by:    str            = "system"


class ReportScheduleUpdate(BaseModel):
	model_config = _CFG

	name:          str | None   = None
	frequency:     ScheduleFrequency | None = None
	output_format: OutputFormat | None = None
	recipients:    list[str] | None = None
	auto_publish:  bool | None  = None
	enabled:       bool | None  = None


class ReportScheduleResponse(APGBase):
	definition_id: str
	name:          str
	period_type:   PeriodType
	frequency:     ScheduleFrequency
	output_format: OutputFormat
	recipients:    list[str]
	auto_publish:  bool
	enabled:       bool
	last_run_at:   str | None
	next_run_at:   str | None
	status:        ReportStatus


# ══════════════════════════════════════════════════════════════════════════════
# ReportOutput
# ══════════════════════════════════════════════════════════════════════════════

class ReportOutputCreate(BaseModel):
	model_config = _CFG

	id:               str          = Field(default_factory=uuid7str)
	tenant_id:        str
	generation_id:    str
	output_format:    OutputFormat
	file_name:        str
	file_path:        str | None   = None
	file_size_bytes:  int | None   = None
	checksum_sha256:  str | None   = None
	created_by:       str          = "system"


class ReportOutputResponse(APGBase):
	generation_id:   str
	output_format:   OutputFormat
	file_name:       str
	file_path:       str | None
	file_size_bytes: int | None
	checksum_sha256: str | None
	status:          str


# ══════════════════════════════════════════════════════════════════════════════
# FinancialStatement
# ══════════════════════════════════════════════════════════════════════════════

class FinancialStatementCreate(BaseModel):
	model_config = _CFG

	id:                   str              = Field(default_factory=uuid7str)
	tenant_id:            str
	generation_id:        str
	period_id:            str
	statement_type:       StatementType
	title:                str              = Field(min_length=1, max_length=300)
	as_of_date:           str
	currency_code:        str              = "USD"
	reporting_entity:     str
	accounting_standard:  AccountingStandard = AccountingStandard.IFRS
	balance_check_passed: bool             = True
	approved_by:          str
	narrative_reviewed_by: str
	total_assets:         float | None     = None
	total_liabilities:    float | None     = None
	total_equity:         float | None     = None
	total_revenue:        float | None     = None
	net_income:           float | None     = None
	statement_data:       dict[str, Any]   = Field(default_factory=dict)
	created_by:           str              = "system"


class FinancialStatementUpdate(BaseModel):
	model_config = _CFG

	title:                str | None  = None
	balance_check_passed: bool | None = None
	approved_by:          str | None  = None
	narrative_reviewed_by: str | None = None
	total_assets:         float | None = None
	total_liabilities:    float | None = None
	total_equity:         float | None = None
	total_revenue:        float | None = None
	net_income:           float | None = None


class FinancialStatementResponse(APGBase):
	generation_id:        str
	period_id:            str
	statement_type:       StatementType
	title:                str
	as_of_date:           str
	currency_code:        str
	reporting_entity:     str
	accounting_standard:  AccountingStandard
	balance_check_passed: bool
	approved_by:          str
	narrative_reviewed_by: str
	is_final:             bool
	is_published:         bool
	total_assets:         float | None
	total_liabilities:    float | None
	total_equity:         float | None
	total_revenue:        float | None
	net_income:           float | None
	status:               ReportStatus


# ══════════════════════════════════════════════════════════════════════════════
# ConsolidationGroup
# ══════════════════════════════════════════════════════════════════════════════

class ConsolidationGroupCreate(BaseModel):
	model_config = _CFG

	id:                     str                = Field(default_factory=uuid7str)
	tenant_id:              str
	parent_entity:          str
	subsidiary_entity:      str
	method:                 ConsolidationMethod
	ownership_percent:      float              = Field(ge=0.0, le=100.0)
	functional_currency:    str                = "USD"
	reporting_currency:     str                = "USD"
	elimination_reviewed_by: str | None        = None
	effective_from:         str | None         = None
	effective_to:           str | None         = None
	created_by:             str                = "system"


class ConsolidationGroupUpdate(BaseModel):
	model_config = _CFG

	method:                 ConsolidationMethod | None = None
	ownership_percent:      float | None               = None
	elimination_reviewed_by: str | None                = None
	effective_to:           str | None                 = None


class ConsolidationGroupResponse(APGBase):
	parent_entity:          str
	subsidiary_entity:      str
	method:                 ConsolidationMethod
	ownership_percent:      float
	functional_currency:    str
	reporting_currency:     str
	elimination_reviewed_by: str | None
	effective_from:         str | None
	effective_to:           str | None
	status:                 str


# ══════════════════════════════════════════════════════════════════════════════
# SegmentReport
# ══════════════════════════════════════════════════════════════════════════════

class SegmentReportCreate(BaseModel):
	model_config = _CFG

	id:                  str              = Field(default_factory=uuid7str)
	tenant_id:           str
	generation_id:       str
	segment_name:        str              = Field(min_length=1, max_length=200)
	segment_code:        str
	revenue:             float            = 0.0
	operating_profit:    float            = 0.0
	total_assets:        float            = 0.0
	capital_expenditure: float            = 0.0
	depreciation:        float            = 0.0
	employee_count:      int              = 0
	period_id:           str
	accounting_standard: AccountingStandard = AccountingStandard.IFRS
	created_by:          str              = "system"


class SegmentReportUpdate(BaseModel):
	model_config = _CFG

	revenue:             float | None = None
	operating_profit:    float | None = None
	total_assets:        float | None = None
	capital_expenditure: float | None = None
	depreciation:        float | None = None
	employee_count:      int | None   = None


class SegmentReportResponse(APGBase):
	generation_id:       str
	segment_name:        str
	segment_code:        str
	revenue:             float
	operating_profit:    float
	total_assets:        float
	capital_expenditure: float
	depreciation:        float
	employee_count:      int
	period_id:           str
	accounting_standard: AccountingStandard
	status:              str


# ══════════════════════════════════════════════════════════════════════════════
# XBRLTag
# ══════════════════════════════════════════════════════════════════════════════

class XBRLTagCreate(BaseModel):
	model_config = _CFG

	id:            str          = Field(default_factory=uuid7str)
	tenant_id:     str
	statement_id:  str
	taxonomy:      XBRLTaxonomy
	element_name:  str          = Field(min_length=1, max_length=300)
	element_value: str
	context_ref:   str
	unit_ref:      str | None   = None
	decimals:      int | None   = None
	period_start:  str | None   = None
	period_end:    str | None   = None
	instant_date:  str | None   = None
	created_by:    str          = "system"


class XBRLTagUpdate(BaseModel):
	model_config = _CFG

	element_value: str | None = None
	context_ref:   str | None = None
	unit_ref:      str | None = None
	decimals:      int | None = None


class XBRLTagResponse(APGBase):
	statement_id:  str
	taxonomy:      XBRLTaxonomy
	element_name:  str
	element_value: str
	context_ref:   str
	unit_ref:      str | None
	decimals:      int | None
	period_start:  str | None
	period_end:    str | None
	instant_date:  str | None
	status:        str


# ══════════════════════════════════════════════════════════════════════════════
# RegulatorySubmission
# ══════════════════════════════════════════════════════════════════════════════

class RegulatorySubmissionCreate(BaseModel):
	model_config = _CFG

	id:                   str                = Field(default_factory=uuid7str)
	tenant_id:            str
	statement_id:         str
	jurisdiction:         FilingJurisdiction
	form_type:            str                = Field(min_length=1, max_length=100)
	filing_deadline:      str
	prepared_by:          str
	reviewed_by:          str | None         = None
	submission_reference: str | None         = None
	notes:                str | None         = None
	created_by:           str                = "system"


class RegulatorySubmissionUpdate(BaseModel):
	model_config = _CFG

	reviewed_by:          str | None = None
	submission_reference: str | None = None
	notes:                str | None = None
	submitted_at:         str | None = None


class RegulatorySubmissionResponse(APGBase):
	statement_id:         str
	jurisdiction:         FilingJurisdiction
	form_type:            str
	filing_deadline:      str
	prepared_by:          str
	reviewed_by:          str | None
	submission_reference: str | None
	submitted_at:         str | None
	notes:                str | None
	status:               str


# ══════════════════════════════════════════════════════════════════════════════
# Report (generation run)
# ══════════════════════════════════════════════════════════════════════════════

class ReportCreate(BaseModel):
	"""A generation-run: ties a definition + period + output_format together."""
	model_config = _CFG

	id:                  str          = Field(default_factory=uuid7str)
	tenant_id:           str
	definition_id:       str
	period_id:           str
	output_format:       OutputFormat  = OutputFormat.PDF
	data_quality_score:  float        = Field(default=1.0, ge=0.0, le=1.0)
	quality_reviewed_by: str | None   = None
	generation_type:     str          = "standard"
	created_by:          str          = "system"


class ReportUpdate(BaseModel):
	model_config = _CFG

	data_quality_score:  float | None = None
	quality_reviewed_by: str | None   = None
	status:              ReportStatus | None = None


class ReportResponse(APGBase):
	definition_id:       str
	period_id:           str
	output_format:       OutputFormat
	data_quality_score:  float
	quality_reviewed_by: str | None
	generation_type:     str
	status:              ReportStatus
	warning_count:       int
	error_count:         int
	start_time:          str | None
	end_time:            str | None


# ══════════════════════════════════════════════════════════════════════════════
# KPI and Analytics response models
# ══════════════════════════════════════════════════════════════════════════════

class KPIResult(BaseModel):
	"""Single KPI metric result."""
	model_config = _CFG

	name:               str
	value:              float
	unit:               str
	prior_period_value: float | None = None
	change_pct:         float | None = None
	benchmark:          float | None = None
	status:             KPIStatus    = KPIStatus.OK
	sparkline:          list[float]  = Field(default_factory=list)


class KPIDashboardResponse(BaseModel):
	model_config = _CFG

	tenant_id:     str
	as_of_date:    str
	currency_code: str
	kpis:          list[KPIResult]
	generated_at:  str


class CommentaryLine(BaseModel):
	model_config = _CFG

	section:          str
	narrative:        str
	variance_driver:  str | None          = None
	significance:     NarrativeSignificance = NarrativeSignificance.MEDIUM


class AutomatedCommentaryResponse(BaseModel):
	model_config = _CFG

	tenant_id:    str
	statement_id: str
	period_id:    str
	commentary:   list[CommentaryLine]
	generated_at: str
	model_used:   str = "rule_based"


class DrillDownRequest(BaseModel):
	model_config = _CFG

	tenant_id:    str
	statement_id: str
	line_code:    str
	dimension:    str | None = None   # cost_centre | department | product | region
	period_id:    str | None = None


class DrillDownResponse(BaseModel):
	model_config = _CFG

	tenant_id:    str
	statement_id: str
	line_code:    str
	dimension:    str | None
	rows:         list[dict[str, Any]]
	total:        float
	currency_code: str


class ComparisonPeriodRow(BaseModel):
	model_config = _CFG

	label:      str
	current:    float
	prior:      float
	change_abs: float
	change_pct: float


class ComparisonPeriodsResponse(BaseModel):
	model_config = _CFG

	tenant_id:      str
	statement_id:   str
	current_period: str
	prior_period:   str
	currency_code:  str
	rows:           list[ComparisonPeriodRow]
	generated_at:   str


class NarrativeReportRequest(BaseModel):
	model_config = _CFG

	tenant_id:    str
	statement_id: str
	period_id:    str
	audience:     str = "board"   # board | audit_committee | management | regulatory


class NarrativeReportResponse(BaseModel):
	model_config = _CFG

	tenant_id:    str
	statement_id: str
	period_id:    str
	audience:     str
	sections:     list[dict[str, str]]
	generated_at: str


# ══════════════════════════════════════════════════════════════════════════════
# Lightweight in-memory dataclasses (service layer)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ReportDefinition:
	id:                  str
	tenant_id:           str
	name:                str
	statement_type:      str
	accounting_standard: str
	description:         str | None
	owner:               str
	currency_code:       str
	comparative_periods: int
	created_by:          str     = "system"
	line_count:          int     = 0
	status:              str     = "draft"
	is_deleted:          bool    = False
	created_at:          str     = ""
	updated_at:          str     = ""

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ReportLine:
	id:             str
	tenant_id:      str
	definition_id:  str
	line_code:      str
	label:          str
	account_mapping: str
	sort_order:     int
	line_type:      str          = "detail"
	formula:        str | None   = None
	sign_reversal:  bool         = False
	indent_level:   int          = 0
	bold:           bool         = False
	note_reference: str | None   = None
	created_by:     str          = "system"
	status:         str          = "active"
	is_deleted:     bool         = False
	created_at:     str          = ""
	updated_at:     str          = ""

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ReportPeriod:
	id:           str
	tenant_id:    str
	period_code:  str
	name:         str
	period_type:  str
	fiscal_year:  int
	start_date:   str
	end_date:     str
	created_by:   str   = "system"
	is_current:   bool  = False
	is_closed:    bool  = False
	is_deleted:   bool  = False
	status:       str   = "open"
	created_at:   str   = ""
	updated_at:   str   = ""

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class Report:
	"""A single generation run (template + period → outputs)."""
	id:                  str
	tenant_id:           str
	definition_id:       str
	period_id:           str
	output_format:       str
	data_quality_score:  float
	quality_reviewed_by: str | None
	generation_type:     str        = "standard"
	created_by:          str        = "system"
	status:              str        = "generated"
	warning_count:       int        = 0
	error_count:         int        = 0
	is_deleted:          bool       = False
	start_time:          str        = ""
	end_time:            str        = ""
	created_at:          str        = ""
	updated_at:          str        = ""

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class FinancialStatement:
	id:                    str
	tenant_id:             str
	generation_id:         str
	period_id:             str
	statement_type:        str
	title:                 str
	as_of_date:            str
	currency_code:         str
	reporting_entity:      str
	accounting_standard:   str
	balance_check_passed:  bool
	approved_by:           str
	narrative_reviewed_by: str
	created_by:            str                = "system"
	is_final:              bool               = False
	is_published:          bool               = False
	is_deleted:            bool               = False
	total_assets:          float | None       = None
	total_liabilities:     float | None       = None
	total_equity:          float | None       = None
	total_revenue:         float | None       = None
	net_income:            float | None       = None
	statement_data:        dict[str, Any]     = field(default_factory=dict)
	status:                str                = "published"
	created_at:            str                = ""
	updated_at:            str                = ""

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ConsolidationGroup:
	id:                      str
	tenant_id:               str
	parent_entity:           str
	subsidiary_entity:       str
	method:                  str
	ownership_percent:       float
	functional_currency:     str
	reporting_currency:      str
	elimination_reviewed_by: str | None
	effective_from:          str | None
	effective_to:            str | None   = None
	created_by:              str          = "system"
	is_deleted:              bool         = False
	status:                  str          = "reviewed"
	created_at:              str          = ""
	updated_at:              str          = ""

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class SegmentReport:
	id:                  str
	tenant_id:           str
	generation_id:       str
	segment_name:        str
	segment_code:        str
	revenue:             float
	operating_profit:    float
	total_assets:        float
	capital_expenditure: float
	depreciation:        float
	employee_count:      int
	period_id:           str
	accounting_standard: str
	created_by:          str   = "system"
	is_deleted:          bool  = False
	status:              str   = "draft"
	created_at:          str   = ""
	updated_at:          str   = ""

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class XBRLTag:
	id:            str
	tenant_id:     str
	statement_id:  str
	taxonomy:      str
	element_name:  str
	element_value: str
	context_ref:   str
	unit_ref:      str | None
	decimals:      int | None
	period_start:  str | None
	period_end:    str | None
	instant_date:  str | None
	created_by:    str   = "system"
	is_deleted:    bool  = False
	status:        str   = "tagged"
	created_at:    str   = ""
	updated_at:    str   = ""

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class RegulatorySubmission:
	id:                   str
	tenant_id:            str
	statement_id:         str
	jurisdiction:         str
	form_type:            str
	filing_deadline:      str
	prepared_by:          str
	reviewed_by:          str | None
	submission_reference: str | None
	submitted_at:         str | None
	notes:                str | None
	created_by:           str   = "system"
	is_deleted:           bool  = False
	status:               str   = "draft"
	created_at:           str   = ""
	updated_at:           str   = ""

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ReportSchedule:
	id:            str
	tenant_id:     str
	definition_id: str
	name:          str
	period_type:   str
	frequency:     str
	output_format: str
	recipients:    list[str]
	auto_publish:  bool
	enabled:       bool
	created_by:    str        = "system"
	last_run_at:   str | None = None
	next_run_at:   str | None = None
	is_deleted:    bool       = False
	status:        str        = "active"
	created_at:    str        = ""
	updated_at:    str        = ""

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ReportOutput:
	id:              str
	tenant_id:       str
	generation_id:   str
	output_format:   str
	file_name:       str
	file_path:       str | None
	file_size_bytes: int | None
	checksum_sha256: str | None   = None
	created_by:      str          = "system"
	is_deleted:      bool         = False
	status:          str          = "ready"
	created_at:      str          = ""
	updated_at:      str          = ""

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class DisclosureRecord:
	id:                   str
	tenant_id:            str
	statement_id:         str
	disclosure_type:      str
	title:                str
	content:              str
	owner:                str
	reviewed_by:          str
	regulation_framework: str | None = None
	risk_level:           str | None = None
	created_by:           str        = "system"
	is_deleted:           bool       = False
	status:               str        = "reviewed"
	created_at:           str        = ""
	updated_at:           str        = ""

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ReportDistribution:
	id:              str
	tenant_id:       str
	statement_id:    str
	recipients:      list[str]
	output_format:   str
	delivery_method: str   = "email"
	created_by:      str   = "system"
	is_deleted:      bool  = False
	status:          str   = "distributed"
	distributed_at:  str   = ""
	created_at:      str   = ""
	updated_at:      str   = ""

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class RPTAgent:
	id:           str
	tenant_id:    str
	name:         str
	runtime:      str
	role:         str
	instructions: str
	created_by:   str   = "system"
	is_deleted:   bool  = False
	status:       str   = "active"
	created_at:   str   = ""
	updated_at:   str   = ""

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
