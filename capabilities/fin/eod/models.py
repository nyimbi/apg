"""Pydantic v2 models for APG EOD/BOD Processing Engine."""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ── Enumerations ──────────────────────────────────────────────────────────────

class JobStatus(str, Enum):
	PENDING    = "pending"
	RUNNING    = "running"
	COMPLETED  = "completed"
	FAILED     = "failed"
	SKIPPED    = "skipped"
	CANCELLED  = "cancelled"
	RETRYING   = "retrying"


class EODStatus(str, Enum):
	NOT_STARTED = "not_started"
	IN_PROGRESS = "in_progress"
	COMPLETED   = "completed"
	FAILED      = "failed"
	CANCELLED   = "cancelled"
	PARTIAL     = "partial"       # some jobs failed, others succeeded


class EODJobType(str, Enum):
	PRE_VALIDATION         = "pre_eod_validations"
	INTEREST_ACCRUAL       = "interest_accrual_batch"
	FEE_POSTING            = "fee_posting_batch"
	DORMANCY_CHECK         = "dormancy_check_batch"
	TERM_DEPOSIT_MATURITY  = "term_deposit_maturity_batch"
	LOAN_REPAYMENT         = "loan_repayment_batch"
	STANDING_ORDER         = "standing_order_batch"
	FX_REVALUATION         = "fx_revaluation"
	PERIOD_CLOSE           = "period_close"
	REPORTS_GENERATION     = "eod_reports_generation"
	BOD_PERIOD_OPEN        = "bod_period_open"
	BOD_FLOAT_CLEAR        = "bod_float_clear"


class ExceptionSeverity(str, Enum):
	INFO     = "info"
	WARNING  = "warning"
	ERROR    = "error"
	CRITICAL = "critical"


# ── Processing statistics ─────────────────────────────────────────────────────

class ProcessingStats(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	accounts_processed:  int            = 0
	accounts_skipped:    int            = 0
	accounts_failed:     int            = 0
	transactions_posted: int            = 0
	total_amount:        str            = "0"    # Decimal as string for JSON safety
	currency:            str            = "KES"
	duration_seconds:    float          = 0.0
	error_count:         int            = 0
	warning_count:       int            = 0
	extra:               dict[str, Any] = Field(default_factory=dict)

	@property
	def total_amount_decimal(self) -> Decimal:
		return Decimal(self.total_amount)


# ── Individual job result ─────────────────────────────────────────────────────

class JobResult(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	job_name:       str
	job_type:       EODJobType
	status:         JobStatus
	tenant_id:      str
	processing_date: str              # ISO date string YYYY-MM-DD
	started_at:     str | None        = None
	completed_at:   str | None        = None
	duration_seconds: float           = 0.0
	stats:          ProcessingStats   = Field(default_factory=ProcessingStats)
	errors:         list[str]         = Field(default_factory=list)
	warnings:       list[str]         = Field(default_factory=list)
	idempotency_key: str | None       = None
	was_cached:     bool              = False   # True if returned from idempotency cache
	attempt:        int               = 1
	dry_run:        bool              = False
	metadata:       dict[str, Any]    = Field(default_factory=dict)


# ── Full EOD run result ───────────────────────────────────────────────────────

class EODResult(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	run_id:          str
	tenant_id:       str
	eod_date:        str                         # ISO date YYYY-MM-DD
	status:          EODStatus
	started_at:      str | None                  = None
	completed_at:    str | None                  = None
	duration_seconds: float                      = 0.0
	jobs:            list[JobResult]             = Field(default_factory=list)
	jobs_completed:  int                         = 0
	jobs_failed:     int                         = 0
	jobs_skipped:    int                         = 0
	total_transactions: int                      = 0
	total_amount_posted: str                     = "0"
	is_month_end:    bool                        = False
	is_year_end:     bool                        = False
	dry_run:         bool                        = False
	errors:          list[str]                   = Field(default_factory=list)
	warnings:        list[str]                   = Field(default_factory=list)
	blocker_count:   int                         = 0
	idempotency_key: str | None                  = None
	was_cached:      bool                        = False


# ── BOD result ────────────────────────────────────────────────────────────────

class BODResult(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	run_id:          str
	tenant_id:       str
	bod_date:        str
	status:          EODStatus
	started_at:      str | None       = None
	completed_at:    str | None       = None
	duration_seconds: float           = 0.0
	period_opened:   bool             = False
	float_cleared:   bool             = False
	float_amount:    str              = "0"
	errors:          list[str]        = Field(default_factory=list)
	warnings:        list[str]        = Field(default_factory=list)


# ── Processing exception ──────────────────────────────────────────────────────

class EODException(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	exception_id:     str
	tenant_id:        str
	processing_date:  str
	job_name:         str
	account_id:       str | None      = None
	transaction_id:   str | None      = None
	severity:         ExceptionSeverity
	error_code:       str
	message:          str
	stack_trace:      str | None      = None
	resolved:         bool            = False
	resolved_at:      str | None      = None
	resolved_by:      str | None      = None
	resolution:       str | None      = None
	created_at:       str
	metadata:         dict[str, Any]  = Field(default_factory=dict)


# ── EOD Report ────────────────────────────────────────────────────────────────

class EODReport(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	report_id:              str
	tenant_id:              str
	processing_date:        str
	generated_at:           str
	eod_status:             EODStatus
	total_accounts:         int            = 0
	interest_accruals:      int            = 0
	interest_amount:        str            = "0"
	fees_posted:            int            = 0
	fees_amount:            str            = "0"
	dormant_accounts:       int            = 0
	maturities_processed:   int            = 0
	maturity_amount:        str            = "0"
	loan_repayments:        int            = 0
	loan_amount:            str            = "0"
	standing_orders:        int            = 0
	standing_order_amount:  str            = "0"
	fx_revaluations:        int            = 0
	fx_gain_loss:           str            = "0"
	exceptions_count:       int            = 0
	exceptions_resolved:    int            = 0
	suspense_balance:       str            = "0"
	unposted_entries:       int            = 0
	job_results:            list[JobResult] = Field(default_factory=list)
	narrative:              str            = ""
	currency:               str            = "KES"
	is_month_end:           bool           = False
	is_year_end:            bool           = False
	metadata:               dict[str, Any] = Field(default_factory=dict)


# ── Scheduled job ─────────────────────────────────────────────────────────────

class ScheduledJob(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	schedule_id:      str
	tenant_id:        str
	job_name:         str
	scheduled_time:   str
	parameters:       dict[str, Any]  = Field(default_factory=dict)
	created_at:       str
	status:           JobStatus       = JobStatus.PENDING
	executed_at:      str | None      = None


# ── EOD metrics ───────────────────────────────────────────────────────────────

class EODMetrics(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	tenant_id:               str
	period_days:             int
	total_runs:              int            = 0
	successful_runs:         int            = 0
	failed_runs:             int            = 0
	partial_runs:            int            = 0
	avg_duration_seconds:    float          = 0.0
	max_duration_seconds:    float          = 0.0
	min_duration_seconds:    float          = 0.0
	avg_exceptions_per_run:  float          = 0.0
	total_transactions:      int            = 0
	error_rate:              float          = 0.0  # 0.0-1.0
	job_error_rates:         dict[str, float] = Field(default_factory=dict)
	daily_durations:         list[dict[str, Any]] = Field(default_factory=list)


# ── Prerequisite check result ─────────────────────────────────────────────────

class PrerequisiteCheck(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	tenant_id:     str
	eod_date:      str
	ready:         bool
	blockers:      list[str]          = Field(default_factory=list)
	warnings:      list[str]          = Field(default_factory=list)
	checked_at:    str                = ""


# ── In-memory stores (dataclasses for mutability) ─────────────────────────────

@dataclass
class _EODRunRecord:
	"""Internal mutable run record — converted to EODResult on read."""
	run_id:          str
	tenant_id:       str
	eod_date:        str
	status:          EODStatus
	started_at:      str | None       = None
	completed_at:    str | None       = None
	dry_run:         bool             = False
	is_month_end:    bool             = False
	is_year_end:     bool             = False
	job_results:     list[JobResult]  = dc_field(default_factory=list)
	errors:          list[str]        = dc_field(default_factory=list)
	warnings:        list[str]        = dc_field(default_factory=list)
	idempotency_key: str | None       = None
	was_cached:      bool             = False

	def to_eod_result(self) -> EODResult:
		completed = [j for j in self.job_results if j.status == JobStatus.COMPLETED]
		failed    = [j for j in self.job_results if j.status == JobStatus.FAILED]
		skipped   = [j for j in self.job_results if j.status == JobStatus.SKIPPED]
		total_tx  = sum(j.stats.transactions_posted for j in self.job_results)
		duration  = 0.0
		if self.started_at and self.completed_at:
			try:
				s = datetime.fromisoformat(self.started_at)
				e = datetime.fromisoformat(self.completed_at)
				duration = (e - s).total_seconds()
			except ValueError:
				pass
		return EODResult(
			run_id=self.run_id,
			tenant_id=self.tenant_id,
			eod_date=self.eod_date,
			status=self.status,
			started_at=self.started_at,
			completed_at=self.completed_at,
			duration_seconds=duration,
			jobs=list(self.job_results),
			jobs_completed=len(completed),
			jobs_failed=len(failed),
			jobs_skipped=len(skipped),
			total_transactions=total_tx,
			is_month_end=self.is_month_end,
			is_year_end=self.is_year_end,
			dry_run=self.dry_run,
			errors=list(self.errors),
			warnings=list(self.warnings),
			idempotency_key=self.idempotency_key,
			was_cached=self.was_cached,
		)
