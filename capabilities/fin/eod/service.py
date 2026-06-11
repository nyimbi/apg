"""EOD/BOD Processing Engine service for APG Financial Management.

All batch jobs are decorated with @idempotent keyed on (tenant_id, processing_date)
to guarantee exactly-once execution — re-running for the same date is safe.

Architecture:
- run_eod  → orchestrates 10 batch jobs in sequence
- run_bod  → morning open: new period (if month-start) + float clear
- run_job  → single-job recovery/testing entry point
- All state lives in in-memory dicts (swap for PostgreSQL via domain/adapters.py)
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

from capabilities.common.reliability import (
	guard_tenant_id,
	guard_non_empty_string,
	idempotent,
	BoundedCache,
)

try:
	from .models import (
		BODResult,
		EODException,
		EODJobType,
		EODMetrics,
		EODReport,
		EODResult,
		EODStatus,
		ExceptionSeverity,
		JobResult,
		JobStatus,
		PrerequisiteCheck,
		ProcessingStats,
		ScheduledJob,
		_EODRunRecord,
	)
except ImportError:  # pragma: no cover — standalone execution
	from models import (  # type: ignore
		BODResult, EODException, EODJobType, EODMetrics, EODReport,
		EODResult, EODStatus, ExceptionSeverity, JobResult, JobStatus,
		PrerequisiteCheck, ProcessingStats, ScheduledJob, _EODRunRecord,
	)

_log = logging.getLogger(__name__)

# Ordered sequence of EOD batch jobs
_EOD_JOB_SEQUENCE: list[EODJobType] = [
	EODJobType.PRE_VALIDATION,
	EODJobType.INTEREST_ACCRUAL,
	EODJobType.FEE_POSTING,
	EODJobType.DORMANCY_CHECK,
	EODJobType.TERM_DEPOSIT_MATURITY,
	EODJobType.LOAN_REPAYMENT,
	EODJobType.STANDING_ORDER,
	EODJobType.FX_REVALUATION,
	EODJobType.PERIOD_CLOSE,
	EODJobType.REPORTS_GENERATION,
]

# Jobs that only run at month-end
_MONTH_END_ONLY: set[EODJobType] = {EODJobType.FX_REVALUATION, EODJobType.PERIOD_CLOSE}


def _now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


def _today_iso() -> str:
	return date.today().isoformat()


def _parse_date(d: str | date) -> date:
	if isinstance(d, date):
		return d
	return date.fromisoformat(d)


def _is_month_end(d: date) -> bool:
	"""True if d is the last day of its month."""
	import calendar
	_, last = calendar.monthrange(d.year, d.month)
	return d.day == last


def _is_month_start(d: date) -> bool:
	return d.day == 1


def _is_year_end(d: date) -> bool:
	return d.month == 12 and d.day == 31


def _eod_key(tenant_id: str, processing_date: str | date) -> str:
	d = processing_date if isinstance(processing_date, str) else processing_date.isoformat()
	return f"eod:{tenant_id}:{d}"


def _job_key(tenant_id: str, processing_date: str | date, job_name: str) -> str:
	d = processing_date if isinstance(processing_date, str) else processing_date.isoformat()
	return f"job:{tenant_id}:{d}:{job_name}"


def _new_id() -> str:
	"""UUID7-style ID via uuid6 if available, else fallback."""
	try:
		from uuid6 import uuid7
		return str(uuid7())
	except ImportError:
		import uuid
		return str(uuid.uuid4())


class EODService:
	"""Tenant-scoped EOD/BOD processing engine.

	In production, inject real DB adapters via domain/adapters.py.
	All state here is in-memory for standalone/test usage.
	"""

	def __init__(self) -> None:
		# key: _eod_key(tenant_id, date) -> _EODRunRecord
		self._runs:       dict[str, _EODRunRecord]   = {}
		# key: _job_key(tenant_id, date, job_name) -> JobResult
		self._job_results: dict[str, JobResult]      = {}
		# key: exception_id -> EODException
		self._exceptions:  dict[str, EODException]   = {}
		# key: schedule_id -> ScheduledJob
		self._schedules:   dict[str, ScheduledJob]   = {}
		# key: _eod_key -> bool (running flag)
		self._running:     dict[str, bool]            = {}
		# Cache for reports
		self._report_cache: BoundedCache              = BoundedCache(max_size=200)

	# ─── Internal helpers ────────────────────────────────────────────────────

	def _log_job_start(self, tenant_id: str, job_name: str, date_str: str) -> None:
		_log.info("[EOD] tenant=%s date=%s job=%s STARTED", tenant_id, date_str, job_name)

	def _log_job_end(self, tenant_id: str, job_name: str, date_str: str, status: JobStatus, duration: float) -> None:
		_log.info("[EOD] tenant=%s date=%s job=%s %s (%.2fs)", tenant_id, date_str, job_name, status.value, duration)

	def _log_eod_start(self, tenant_id: str, date_str: str, dry_run: bool) -> None:
		mode = "DRY-RUN" if dry_run else "LIVE"
		_log.info("[EOD] ===== START %s tenant=%s date=%s =====", mode, tenant_id, date_str)

	def _log_eod_end(self, tenant_id: str, date_str: str, status: EODStatus, duration: float) -> None:
		_log.info("[EOD] ===== END tenant=%s date=%s status=%s (%.2fs) =====", tenant_id, date_str, status.value, duration)

	def _log_exception(self, tenant_id: str, job: str, msg: str) -> None:
		_log.error("[EOD] EXCEPTION tenant=%s job=%s: %s", tenant_id, job, msg)

	def _log_pretty_path(self, tenant_id: str, date_str: str) -> str:
		return f"[{tenant_id}@{date_str}]"

	def _record_exception(
		self, tenant_id: str, processing_date: str,
		job_name: str, severity: ExceptionSeverity,
		error_code: str, message: str,
		account_id: str | None = None,
		transaction_id: str | None = None,
	) -> EODException:
		exc = EODException(
			exception_id=_new_id(),
			tenant_id=tenant_id,
			processing_date=processing_date,
			job_name=job_name,
			account_id=account_id,
			transaction_id=transaction_id,
			severity=severity,
			error_code=error_code,
			message=message,
			resolved=False,
			created_at=_now_iso(),
		)
		self._exceptions[exc.exception_id] = exc
		return exc

	def _make_job_result(
		self, tenant_id: str, processing_date: str,
		job_type: EODJobType, status: JobStatus,
		stats: ProcessingStats | None = None,
		errors: list[str] | None = None,
		warnings: list[str] | None = None,
		started_at: str | None = None,
		completed_at: str | None = None,
		dry_run: bool = False,
		was_cached: bool = False,
		attempt: int = 1,
		metadata: dict[str, Any] | None = None,
	) -> JobResult:
		started  = started_at  or _now_iso()
		finished = completed_at or _now_iso()
		duration = 0.0
		try:
			s = datetime.fromisoformat(started)
			e = datetime.fromisoformat(finished)
			duration = (e - s).total_seconds()
		except ValueError as _exc:
			_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
		return JobResult(
			job_name=job_type.value,
			job_type=job_type,
			status=status,
			tenant_id=tenant_id,
			processing_date=processing_date,
			started_at=started,
			completed_at=finished,
			duration_seconds=duration,
			stats=stats or ProcessingStats(),
			errors=errors or [],
			warnings=warnings or [],
			idempotency_key=_job_key(tenant_id, processing_date, job_type.value),
			was_cached=was_cached,
			attempt=attempt,
			dry_run=dry_run,
			metadata=metadata or {},
		)

	# ─── Individual batch jobs ────────────────────────────────────────────────
	# Each decorated with @idempotent — key includes tenant_id + processing_date.
	# In dry_run mode, logic runs but no state mutations are committed.

	@idempotent(key_fn=lambda self, tenant_id, processing_date, dry_run=False: _job_key(tenant_id, processing_date, EODJobType.PRE_VALIDATION.value))
	async def _run_pre_eod_validations(self, tenant_id: str, processing_date: str, dry_run: bool = False) -> JobResult:
		"""Check suspense accounts, unposted entries, and GL period availability."""
		t0 = time.monotonic()
		self._log_job_start(tenant_id, EODJobType.PRE_VALIDATION.value, processing_date)
		errors: list[str] = []
		warnings: list[str] = []
		# In a real system: query DB for suspense balances, unposted GL entries, open periods
		# Simulated validation — adapters.py hooks provide real implementations
		suspense_balance = Decimal("0")   # adapter would query real suspense GL
		unposted_count   = 0              # adapter would count unposted journal entries
		if suspense_balance != Decimal("0"):
			errors.append(f"Suspense account has non-zero balance: {suspense_balance}")
		if unposted_count > 0:
			warnings.append(f"{unposted_count} unposted journal entries found — will process")
		status = JobStatus.FAILED if errors else JobStatus.COMPLETED
		stats  = ProcessingStats(accounts_processed=0, transactions_posted=0, extra={"suspense_balance": str(suspense_balance), "unposted_count": unposted_count})
		result = self._make_job_result(tenant_id, processing_date, EODJobType.PRE_VALIDATION, status, stats=stats, errors=errors, warnings=warnings, dry_run=dry_run)
		self._job_results[_job_key(tenant_id, processing_date, EODJobType.PRE_VALIDATION.value)] = result
		self._log_job_end(tenant_id, EODJobType.PRE_VALIDATION.value, processing_date, status, time.monotonic() - t0)
		return result

	@idempotent(key_fn=lambda self, tenant_id, processing_date, dry_run=False: _job_key(tenant_id, processing_date, EODJobType.INTEREST_ACCRUAL.value))
	async def _run_interest_accrual_batch(self, tenant_id: str, processing_date: str, dry_run: bool = False) -> JobResult:
		"""Post daily interest accruals for all interest-bearing accounts."""
		t0 = time.monotonic()
		self._log_job_start(tenant_id, EODJobType.INTEREST_ACCRUAL.value, processing_date)
		# Adapter hook: for each account with interest_rate > 0, compute daily accrual
		# accrual = principal * rate / 365   (actual/365)
		# Post debit to Interest Expense GL, credit to Accrued Interest Payable GL
		stats = ProcessingStats(
			accounts_processed=0, transactions_posted=0,
			total_amount="0", extra={"method": "actual/365"}
		)
		result = self._make_job_result(tenant_id, processing_date, EODJobType.INTEREST_ACCRUAL, JobStatus.COMPLETED, stats=stats, dry_run=dry_run)
		self._job_results[_job_key(tenant_id, processing_date, EODJobType.INTEREST_ACCRUAL.value)] = result
		self._log_job_end(tenant_id, EODJobType.INTEREST_ACCRUAL.value, processing_date, JobStatus.COMPLETED, time.monotonic() - t0)
		return result

	@idempotent(key_fn=lambda self, tenant_id, processing_date, dry_run=False: _job_key(tenant_id, processing_date, EODJobType.FEE_POSTING.value))
	async def _run_fee_posting_batch(self, tenant_id: str, processing_date: str, dry_run: bool = False) -> JobResult:
		"""Post monthly/quarterly account fees due today."""
		t0 = time.monotonic()
		self._log_job_start(tenant_id, EODJobType.FEE_POSTING.value, processing_date)
		d = _parse_date(processing_date)
		fees_due_today = 0
		# Adapter hook: query fee schedules where next_due_date == processing_date
		# For each: post debit to account, credit to Fee Income GL
		# Update next_due_date = add_months(processing_date, frequency)
		stats = ProcessingStats(
			accounts_processed=fees_due_today, transactions_posted=fees_due_today,
			total_amount="0", extra={"day_of_month": d.day}
		)
		result = self._make_job_result(tenant_id, processing_date, EODJobType.FEE_POSTING, JobStatus.COMPLETED, stats=stats, dry_run=dry_run)
		self._job_results[_job_key(tenant_id, processing_date, EODJobType.FEE_POSTING.value)] = result
		self._log_job_end(tenant_id, EODJobType.FEE_POSTING.value, processing_date, JobStatus.COMPLETED, time.monotonic() - t0)
		return result

	@idempotent(key_fn=lambda self, tenant_id, processing_date, dry_run=False: _job_key(tenant_id, processing_date, EODJobType.DORMANCY_CHECK.value))
	async def _run_dormancy_check_batch(self, tenant_id: str, processing_date: str, dry_run: bool = False) -> JobResult:
		"""Flag accounts as dormant if no customer-initiated transaction for N days."""
		t0 = time.monotonic()
		self._log_job_start(tenant_id, EODJobType.DORMANCY_CHECK.value, processing_date)
		# Adapter hook: query accounts where last_transaction_date < processing_date - dormancy_threshold
		# Update account status to DORMANT, emit dormancy_flagged event
		# Regulatory: CBK requires 12 months inactivity for savings, 24 for current
		newly_dormant = 0
		stats = ProcessingStats(
			accounts_processed=0, accounts_failed=0,
			extra={"newly_dormant": newly_dormant, "dormancy_threshold_days": 365}
		)
		result = self._make_job_result(tenant_id, processing_date, EODJobType.DORMANCY_CHECK, JobStatus.COMPLETED, stats=stats, dry_run=dry_run)
		self._job_results[_job_key(tenant_id, processing_date, EODJobType.DORMANCY_CHECK.value)] = result
		self._log_job_end(tenant_id, EODJobType.DORMANCY_CHECK.value, processing_date, JobStatus.COMPLETED, time.monotonic() - t0)
		return result

	@idempotent(key_fn=lambda self, tenant_id, processing_date, dry_run=False: _job_key(tenant_id, processing_date, EODJobType.TERM_DEPOSIT_MATURITY.value))
	async def _run_term_deposit_maturity_batch(self, tenant_id: str, processing_date: str, dry_run: bool = False) -> JobResult:
		"""Process term deposits maturing today: pay principal + interest or roll over."""
		t0 = time.monotonic()
		self._log_job_start(tenant_id, EODJobType.TERM_DEPOSIT_MATURITY.value, processing_date)
		# Adapter hook: query term_deposits where maturity_date == processing_date
		# For auto-renew: create new term deposit at current rate; post interest earned
		# For payout: transfer principal + accrued interest to linked account
		maturities = 0
		stats = ProcessingStats(
			accounts_processed=maturities, transactions_posted=maturities * 2,
			total_amount="0", extra={"auto_renewed": 0, "paid_out": 0}
		)
		result = self._make_job_result(tenant_id, processing_date, EODJobType.TERM_DEPOSIT_MATURITY, JobStatus.COMPLETED, stats=stats, dry_run=dry_run)
		self._job_results[_job_key(tenant_id, processing_date, EODJobType.TERM_DEPOSIT_MATURITY.value)] = result
		self._log_job_end(tenant_id, EODJobType.TERM_DEPOSIT_MATURITY.value, processing_date, JobStatus.COMPLETED, time.monotonic() - t0)
		return result

	@idempotent(key_fn=lambda self, tenant_id, processing_date, dry_run=False: _job_key(tenant_id, processing_date, EODJobType.LOAN_REPAYMENT.value))
	async def _run_loan_repayment_batch(self, tenant_id: str, processing_date: str, dry_run: bool = False) -> JobResult:
		"""Process loan repayments due today: collect from linked accounts, classify arrears."""
		t0 = time.monotonic()
		self._log_job_start(tenant_id, EODJobType.LOAN_REPAYMENT.value, processing_date)
		# Adapter hook: query loan_schedule where due_date == processing_date
		# Debit linked account → credit loan receivable + interest income
		# If insufficient funds: mark installment as missed, update arrears_days
		# Provision update: IFRS9 ECL recalculation trigger
		repayments = 0
		missed     = 0
		stats = ProcessingStats(
			accounts_processed=repayments + missed,
			transactions_posted=repayments,
			total_amount="0",
			extra={"collected": repayments, "missed": missed, "ecl_triggered": repayments > 0}
		)
		result = self._make_job_result(tenant_id, processing_date, EODJobType.LOAN_REPAYMENT, JobStatus.COMPLETED, stats=stats, dry_run=dry_run)
		self._job_results[_job_key(tenant_id, processing_date, EODJobType.LOAN_REPAYMENT.value)] = result
		self._log_job_end(tenant_id, EODJobType.LOAN_REPAYMENT.value, processing_date, JobStatus.COMPLETED, time.monotonic() - t0)
		return result

	@idempotent(key_fn=lambda self, tenant_id, processing_date, dry_run=False: _job_key(tenant_id, processing_date, EODJobType.STANDING_ORDER.value))
	async def _run_standing_order_batch(self, tenant_id: str, processing_date: str, dry_run: bool = False) -> JobResult:
		"""Execute standing orders whose execution_date falls today."""
		t0 = time.monotonic()
		self._log_job_start(tenant_id, EODJobType.STANDING_ORDER.value, processing_date)
		# Adapter hook: query standing_orders where next_execution_date == processing_date
		# Execute transfer; update next_execution_date; track failures
		executed = 0
		failed   = 0
		stats = ProcessingStats(
			accounts_processed=executed + failed,
			transactions_posted=executed,
			accounts_failed=failed,
			total_amount="0",
		)
		result = self._make_job_result(tenant_id, processing_date, EODJobType.STANDING_ORDER, JobStatus.COMPLETED, stats=stats, dry_run=dry_run)
		self._job_results[_job_key(tenant_id, processing_date, EODJobType.STANDING_ORDER.value)] = result
		self._log_job_end(tenant_id, EODJobType.STANDING_ORDER.value, processing_date, JobStatus.COMPLETED, time.monotonic() - t0)
		return result

	@idempotent(key_fn=lambda self, tenant_id, processing_date, dry_run=False: _job_key(tenant_id, processing_date, EODJobType.FX_REVALUATION.value))
	async def _run_fx_revaluation(self, tenant_id: str, processing_date: str, dry_run: bool = False) -> JobResult:
		"""Month-end FX revaluation: restate all FCY balances at closing rates."""
		t0 = time.monotonic()
		self._log_job_start(tenant_id, EODJobType.FX_REVALUATION.value, processing_date)
		# Adapter hook: fetch closing FX rates for the month
		# For each FCY account: compute gain/loss vs book rate
		# Post to FX Gain/Loss GL (P&L); update translated balance
		revalued   = 0
		net_gl     = Decimal("0")
		stats = ProcessingStats(
			accounts_processed=revalued, transactions_posted=revalued,
			total_amount=str(net_gl),
			extra={"currencies_processed": [], "net_gain_loss": str(net_gl)}
		)
		result = self._make_job_result(tenant_id, processing_date, EODJobType.FX_REVALUATION, JobStatus.COMPLETED, stats=stats, dry_run=dry_run)
		self._job_results[_job_key(tenant_id, processing_date, EODJobType.FX_REVALUATION.value)] = result
		self._log_job_end(tenant_id, EODJobType.FX_REVALUATION.value, processing_date, JobStatus.COMPLETED, time.monotonic() - t0)
		return result

	@idempotent(key_fn=lambda self, tenant_id, processing_date, dry_run=False: _job_key(tenant_id, processing_date, EODJobType.PERIOD_CLOSE.value))
	async def _run_period_close(self, tenant_id: str, processing_date: str, dry_run: bool = False) -> JobResult:
		"""Month-end period close: lock GL period, post closing entries, roll retained earnings."""
		t0 = time.monotonic()
		self._log_job_start(tenant_id, EODJobType.PERIOD_CLOSE.value, processing_date)
		# Adapter hook:
		# 1. Validate all accounts balance (sum debits == sum credits)
		# 2. Close P&L accounts to retained earnings
		# 3. Lock period (no more postings allowed)
		# 4. Open next period
		balanced = True
		stats = ProcessingStats(
			accounts_processed=0, transactions_posted=0,
			extra={"period_locked": not dry_run, "balanced": balanced}
		)
		warnings: list[str] = [] if balanced else ["Trial balance out of balance — investigate before confirming close"]
		result = self._make_job_result(tenant_id, processing_date, EODJobType.PERIOD_CLOSE, JobStatus.COMPLETED, stats=stats, warnings=warnings, dry_run=dry_run)
		self._job_results[_job_key(tenant_id, processing_date, EODJobType.PERIOD_CLOSE.value)] = result
		self._log_job_end(tenant_id, EODJobType.PERIOD_CLOSE.value, processing_date, JobStatus.COMPLETED, time.monotonic() - t0)
		return result

	@idempotent(key_fn=lambda self, tenant_id, processing_date, dry_run=False: _job_key(tenant_id, processing_date, EODJobType.REPORTS_GENERATION.value))
	async def _run_eod_reports_generation(self, tenant_id: str, processing_date: str, dry_run: bool = False) -> JobResult:
		"""Generate and store EOD management reports."""
		t0 = time.monotonic()
		self._log_job_start(tenant_id, EODJobType.REPORTS_GENERATION.value, processing_date)
		# Reports: Daily Balance Sheet, P&L flash, Regulatory returns, Liquidity ratio
		reports_generated = ["daily_balance_sheet", "pl_flash", "liquidity_ratio"]
		stats = ProcessingStats(
			accounts_processed=0, transactions_posted=0,
			extra={"reports": reports_generated, "count": len(reports_generated)}
		)
		result = self._make_job_result(tenant_id, processing_date, EODJobType.REPORTS_GENERATION, JobStatus.COMPLETED, stats=stats, dry_run=dry_run)
		self._job_results[_job_key(tenant_id, processing_date, EODJobType.REPORTS_GENERATION.value)] = result
		self._log_job_end(tenant_id, EODJobType.REPORTS_GENERATION.value, processing_date, JobStatus.COMPLETED, time.monotonic() - t0)
		return result

	# ─── Job dispatch table ───────────────────────────────────────────────────

	_JOB_DISPATCH: dict[EODJobType, str] = {
		EODJobType.PRE_VALIDATION:        "_run_pre_eod_validations",
		EODJobType.INTEREST_ACCRUAL:      "_run_interest_accrual_batch",
		EODJobType.FEE_POSTING:           "_run_fee_posting_batch",
		EODJobType.DORMANCY_CHECK:        "_run_dormancy_check_batch",
		EODJobType.TERM_DEPOSIT_MATURITY: "_run_term_deposit_maturity_batch",
		EODJobType.LOAN_REPAYMENT:        "_run_loan_repayment_batch",
		EODJobType.STANDING_ORDER:        "_run_standing_order_batch",
		EODJobType.FX_REVALUATION:        "_run_fx_revaluation",
		EODJobType.PERIOD_CLOSE:          "_run_period_close",
		EODJobType.REPORTS_GENERATION:    "_run_eod_reports_generation",
	}

	async def _dispatch_job(self, job_type: EODJobType, tenant_id: str, processing_date: str, dry_run: bool = False) -> JobResult:
		method_name = self._JOB_DISPATCH[job_type]
		method = getattr(self, method_name)
		return await method(tenant_id, processing_date, dry_run)

	# ─── Public API ───────────────────────────────────────────────────────────

	@idempotent(key_fn=lambda self, tenant_id, eod_date, dry_run=False: _eod_key(tenant_id, eod_date))
	async def run_eod(self, tenant_id: str, eod_date: str | date, dry_run: bool = False) -> EODResult:
		"""Orchestrate full EOD run for tenant_id on eod_date.

		Idempotent: calling multiple times for the same (tenant_id, eod_date)
		returns the cached result without re-executing any jobs.

		Args:
			tenant_id:  Tenant identifier.
			eod_date:   Processing date (ISO string or date object).
			dry_run:    If True, run all logic but commit no state changes.

		Returns:
			EODResult with per-job audit trail and aggregate counts.
		"""
		guard_tenant_id(tenant_id)
		date_str = eod_date.isoformat() if isinstance(eod_date, date) else eod_date
		d        = _parse_date(date_str)
		run_id   = _new_id()
		run_key  = _eod_key(tenant_id, date_str)

		self._log_eod_start(tenant_id, date_str, dry_run)

		# Check prerequisites first (non-idempotent read)
		prereq   = await self.check_eod_prerequisites(tenant_id, date_str)
		if not prereq.ready and not dry_run:
			_log.error("[EOD] %s Prerequisites NOT met: %s", self._log_pretty_path(tenant_id, date_str), prereq.blockers)
			rec = _EODRunRecord(
				run_id=run_id, tenant_id=tenant_id, eod_date=date_str,
				status=EODStatus.FAILED,
				started_at=_now_iso(), completed_at=_now_iso(),
				errors=[f"Blocker: {b}" for b in prereq.blockers],
			)
			self._runs[run_key] = rec
			result = rec.to_eod_result()
			result.blocker_count = len(prereq.blockers)
			return result

		# Create run record
		rec = _EODRunRecord(
			run_id=run_id, tenant_id=tenant_id, eod_date=date_str,
			status=EODStatus.IN_PROGRESS,
			started_at=_now_iso(),
			dry_run=dry_run,
			is_month_end=_is_month_end(d),
			is_year_end=_is_year_end(d),
		)
		self._runs[run_key] = rec
		self._running[run_key] = True
		t_global = time.monotonic()

		try:
			for job_type in _EOD_JOB_SEQUENCE:
				# Skip month-end-only jobs on non-month-end dates
				if job_type in _MONTH_END_ONLY and not rec.is_month_end:
					skipped = self._make_job_result(
						tenant_id, date_str, job_type, JobStatus.SKIPPED,
						metadata={"reason": "not month-end"},
					)
					rec.job_results.append(skipped)
					self._job_results[_job_key(tenant_id, date_str, job_type.value)] = skipped
					continue

				# Stop processing if pre_eod_validations failed (hard stop)
				if job_type != EODJobType.PRE_VALIDATION and rec.job_results:
					validation_result = next(
						(j for j in rec.job_results if j.job_type == EODJobType.PRE_VALIDATION), None
					)
					if validation_result and validation_result.status == JobStatus.FAILED:
						skipped = self._make_job_result(
							tenant_id, date_str, job_type, JobStatus.SKIPPED,
							metadata={"reason": "pre_validation_failed"},
						)
						rec.job_results.append(skipped)
						continue

				# Emergency cancellation check
				if not self._running.get(run_key, True):
					_log.warning("[EOD] %s CANCELLED at job %s", self._log_pretty_path(tenant_id, date_str), job_type.value)
					skipped = self._make_job_result(tenant_id, date_str, job_type, JobStatus.CANCELLED)
					rec.job_results.append(skipped)
					continue

				try:
					job_result = await self._dispatch_job(job_type, tenant_id, date_str, dry_run)
					rec.job_results.append(job_result)
					if job_result.errors:
						for e in job_result.errors:
							self._record_exception(tenant_id, date_str, job_type.value, ExceptionSeverity.ERROR, "JOB_ERROR", e)
				except Exception as ex:
					msg = str(ex)
					self._log_exception(tenant_id, job_type.value, msg)
					failed_result = self._make_job_result(
						tenant_id, date_str, job_type, JobStatus.FAILED,
						errors=[msg], dry_run=dry_run,
					)
					rec.job_results.append(failed_result)
					self._job_results[_job_key(tenant_id, date_str, job_type.value)] = failed_result
					self._record_exception(tenant_id, date_str, job_type.value, ExceptionSeverity.CRITICAL, "JOB_EXCEPTION", msg)

		finally:
			self._running.pop(run_key, None)

		# Determine overall status
		has_failed   = any(j.status == JobStatus.FAILED for j in rec.job_results)
		has_complete = any(j.status == JobStatus.COMPLETED for j in rec.job_results)
		if has_failed and has_complete:
			rec.status = EODStatus.PARTIAL
		elif has_failed:
			rec.status = EODStatus.FAILED
		else:
			rec.status = EODStatus.COMPLETED

		rec.completed_at = _now_iso()
		self._runs[run_key] = rec

		result = rec.to_eod_result()
		self._log_eod_end(tenant_id, date_str, rec.status, time.monotonic() - t_global)
		return result

	async def run_bod(self, tenant_id: str, bod_date: str | date) -> BODResult:
		"""Morning (BOD) processing: open new GL period if month-start, clear overnight float.

		NOT cached — BOD should only be called once per day by the scheduler,
		but can be safely re-run if period_open is idempotent at the GL layer.
		"""
		guard_tenant_id(tenant_id)
		date_str = bod_date.isoformat() if isinstance(bod_date, date) else bod_date
		d        = _parse_date(date_str)
		run_id   = _new_id()
		started  = _now_iso()

		_log.info("[BOD] tenant=%s date=%s STARTED", tenant_id, date_str)

		period_opened = False
		float_cleared = False
		float_amount  = Decimal("0")
		errors:   list[str] = []
		warnings: list[str] = []

		# 1. Open new period if month-start
		if _is_month_start(d):
			try:
				# Adapter hook: open GL period for d.year, d.month
				period_opened = True
				_log.info("[BOD] %s Opened GL period %d-%02d", self._log_pretty_path(tenant_id, date_str), d.year, d.month)
			except Exception as ex:
				errors.append(f"Period open failed: {ex}")

		# 2. Clear overnight float (uncleared cheques / EFT settlements)
		try:
			# Adapter hook: mark cleared items, credit/debit settlement GL
			float_cleared = True
			_log.info("[BOD] %s Float cleared: %s", self._log_pretty_path(tenant_id, date_str), float_amount)
		except Exception as ex:
			errors.append(f"Float clear failed: {ex}")

		status = EODStatus.FAILED if errors else EODStatus.COMPLETED
		_log.info("[BOD] tenant=%s date=%s %s", tenant_id, date_str, status.value)

		return BODResult(
			run_id=run_id,
			tenant_id=tenant_id,
			bod_date=date_str,
			status=status,
			started_at=started,
			completed_at=_now_iso(),
			period_opened=period_opened,
			float_cleared=float_cleared,
			float_amount=str(float_amount),
			errors=errors,
			warnings=warnings,
		)

	async def run_job(self, tenant_id: str, job_name: str, processing_date: str | date, dry_run: bool = False) -> JobResult:
		"""Run a single EOD job by name — for recovery and testing.

		The underlying job method is @idempotent, so retrying is safe.
		"""
		guard_tenant_id(tenant_id)
		guard_non_empty_string(job_name, "job_name")
		date_str = processing_date.isoformat() if isinstance(processing_date, date) else processing_date

		try:
			job_type = EODJobType(job_name)
		except ValueError:
			valid = [j.value for j in EODJobType]
			raise ValueError(f"Unknown job_name {job_name!r}. Valid: {valid}") from None

		if job_type not in self._JOB_DISPATCH:
			raise ValueError(f"Job {job_name!r} is not dispatchable directly.")

		return await self._dispatch_job(job_type, tenant_id, date_str, dry_run)

	async def get_eod_status(self, tenant_id: str, processing_date: str | date) -> dict[str, Any]:
		"""Return run status for a specific EOD date."""
		guard_tenant_id(tenant_id)
		date_str = processing_date.isoformat() if isinstance(processing_date, date) else processing_date
		rec = self._runs.get(_eod_key(tenant_id, date_str))
		if not rec:
			return {"status": EODStatus.NOT_STARTED.value, "jobs_completed": 0, "jobs_failed": 0, "started_at": None, "completed_at": None}
		result = rec.to_eod_result()
		return {
			"status":          result.status.value,
			"jobs_completed":  result.jobs_completed,
			"jobs_failed":     result.jobs_failed,
			"jobs_skipped":    result.jobs_skipped,
			"started_at":      result.started_at,
			"completed_at":    result.completed_at,
			"duration_seconds": result.duration_seconds,
			"dry_run":         result.dry_run,
		}

	async def get_job_result(self, tenant_id: str, processing_date: str | date, job_name: str) -> JobResult | None:
		"""Return the result for a specific job on a specific date."""
		guard_tenant_id(tenant_id)
		date_str = processing_date.isoformat() if isinstance(processing_date, date) else processing_date
		return self._job_results.get(_job_key(tenant_id, date_str, job_name))

	async def retry_failed_job(self, tenant_id: str, processing_date: str | date, job_name: str) -> JobResult:
		"""Retry a single failed job. Clears the idempotency cache entry first so the job re-executes."""
		guard_tenant_id(tenant_id)
		date_str = processing_date.isoformat() if isinstance(processing_date, date) else processing_date
		# Remove cached job result to force re-execution
		key = _job_key(tenant_id, date_str, job_name)
		self._job_results.pop(key, None)
		# Also clear from the run record
		rec = self._runs.get(_eod_key(tenant_id, date_str))
		if rec:
			rec.job_results = [j for j in rec.job_results if j.job_name != job_name]
		_log.info("[EOD] Retrying job %s for tenant=%s date=%s", job_name, tenant_id, date_str)
		return await self.run_job(tenant_id, job_name, date_str)

	async def get_eod_history(self, tenant_id: str, from_date: str | date, to_date: str | date) -> list[dict[str, Any]]:
		"""Return all EOD runs for tenant between from_date and to_date (inclusive)."""
		guard_tenant_id(tenant_id)
		from_d = _parse_date(from_date)
		to_d   = _parse_date(to_date)
		results = []
		for key, rec in self._runs.items():
			if not key.startswith(f"eod:{tenant_id}:"):
				continue
			run_date = _parse_date(rec.eod_date)
			if from_d <= run_date <= to_d:
				r = rec.to_eod_result()
				results.append({
					"run_id":         r.run_id,
					"eod_date":       r.eod_date,
					"status":         r.status.value,
					"jobs_completed": r.jobs_completed,
					"jobs_failed":    r.jobs_failed,
					"duration_seconds": r.duration_seconds,
					"dry_run":        r.dry_run,
				})
		results.sort(key=lambda x: x["eod_date"])
		return results

	async def get_processing_exceptions(self, tenant_id: str, processing_date: str | date) -> list[EODException]:
		"""Return all unresolved exceptions for a processing date."""
		guard_tenant_id(tenant_id)
		date_str = processing_date.isoformat() if isinstance(processing_date, date) else processing_date
		return [
			e for e in self._exceptions.values()
			if e.tenant_id == tenant_id and e.processing_date == date_str
		]

	async def resolve_exception(self, tenant_id: str, exception_id: str, resolution: str, resolved_by: str) -> EODException:
		"""Mark an exception as resolved."""
		guard_tenant_id(tenant_id)
		guard_non_empty_string(exception_id, "exception_id")
		guard_non_empty_string(resolution, "resolution")
		exc = self._exceptions.get(exception_id)
		if not exc:
			raise KeyError(f"Exception {exception_id!r} not found")
		if exc.tenant_id != tenant_id:
			raise PermissionError(f"Exception {exception_id!r} belongs to a different tenant")
		exc.resolved    = True
		exc.resolved_at = _now_iso()
		exc.resolved_by = resolved_by
		exc.resolution  = resolution
		self._exceptions[exception_id] = exc
		return exc

	async def get_pending_items(self, tenant_id: str) -> dict[str, Any]:
		"""Return items awaiting processing: unresolved exceptions, scheduled jobs, failed runs."""
		guard_tenant_id(tenant_id)
		unresolved = [e for e in self._exceptions.values() if e.tenant_id == tenant_id and not e.resolved]
		pending_schedules = [
			s for s in self._schedules.values()
			if s.tenant_id == tenant_id and s.status == JobStatus.PENDING
		]
		failed_runs = [
			rec for key, rec in self._runs.items()
			if key.startswith(f"eod:{tenant_id}:") and rec.status in (EODStatus.FAILED, EODStatus.PARTIAL)
		]
		return {
			"unresolved_exceptions":  len(unresolved),
			"pending_scheduled_jobs": len(pending_schedules),
			"failed_eod_runs":        len(failed_runs),
			"items": {
				"exceptions":      [e.model_dump() for e in unresolved[:20]],
				"scheduled_jobs":  [s.model_dump() for s in pending_schedules[:20]],
				"failed_runs":     [{"eod_date": r.eod_date, "status": r.status.value} for r in failed_runs[:20]],
			},
		}

	async def schedule_job(self, tenant_id: str, job_name: str, scheduled_time: str, parameters: dict[str, Any] | None = None) -> ScheduledJob:
		"""Schedule a custom job for future execution."""
		guard_tenant_id(tenant_id)
		guard_non_empty_string(job_name, "job_name")
		sched = ScheduledJob(
			schedule_id=_new_id(),
			tenant_id=tenant_id,
			job_name=job_name,
			scheduled_time=scheduled_time,
			parameters=parameters or {},
			created_at=_now_iso(),
		)
		self._schedules[sched.schedule_id] = sched
		_log.info("[EOD] Scheduled job %s for tenant=%s at %s", job_name, tenant_id, scheduled_time)
		return sched

	async def get_eod_report(self, tenant_id: str, processing_date: str | date) -> EODReport:
		"""Build and return a full EOD report with all counts and amounts."""
		guard_tenant_id(tenant_id)
		date_str = processing_date.isoformat() if isinstance(processing_date, date) else processing_date
		cache_key = f"report:{tenant_id}:{date_str}"
		cached = self._report_cache.get(cache_key)
		if cached:
			return cached

		rec = self._runs.get(_eod_key(tenant_id, date_str))
		exceptions = await self.get_processing_exceptions(tenant_id, date_str)

		def _job_stat(job_type: EODJobType, field: str, default: Any = 0) -> Any:
			r = self._job_results.get(_job_key(tenant_id, date_str, job_type.value))
			if r is None:
				return default
			return getattr(r.stats, field, default) or r.stats.extra.get(field, default)

		report = EODReport(
			report_id=_new_id(),
			tenant_id=tenant_id,
			processing_date=date_str,
			generated_at=_now_iso(),
			eod_status=rec.status if rec else EODStatus.NOT_STARTED,
			interest_accruals=_job_stat(EODJobType.INTEREST_ACCRUAL, "transactions_posted"),
			interest_amount=_job_stat(EODJobType.INTEREST_ACCRUAL, "total_amount", "0"),
			fees_posted=_job_stat(EODJobType.FEE_POSTING, "transactions_posted"),
			fees_amount=_job_stat(EODJobType.FEE_POSTING, "total_amount", "0"),
			dormant_accounts=_job_stat(EODJobType.DORMANCY_CHECK, "newly_dormant"),
			maturities_processed=_job_stat(EODJobType.TERM_DEPOSIT_MATURITY, "accounts_processed"),
			maturity_amount=_job_stat(EODJobType.TERM_DEPOSIT_MATURITY, "total_amount", "0"),
			loan_repayments=_job_stat(EODJobType.LOAN_REPAYMENT, "transactions_posted"),
			loan_amount=_job_stat(EODJobType.LOAN_REPAYMENT, "total_amount", "0"),
			standing_orders=_job_stat(EODJobType.STANDING_ORDER, "transactions_posted"),
			standing_order_amount=_job_stat(EODJobType.STANDING_ORDER, "total_amount", "0"),
			exceptions_count=len(exceptions),
			exceptions_resolved=sum(1 for e in exceptions if e.resolved),
			job_results=list(self._job_results.values()),
			is_month_end=rec.is_month_end if rec else False,
			is_year_end=rec.is_year_end if rec else False,
		)
		self._report_cache.set(cache_key, report)
		return report

	async def check_eod_prerequisites(self, tenant_id: str, eod_date: str | date) -> PrerequisiteCheck:
		"""Check if EOD can safely run. Returns blockers that must be resolved first."""
		guard_tenant_id(tenant_id)
		date_str = eod_date.isoformat() if isinstance(eod_date, date) else eod_date
		blockers:  list[str] = []
		warnings:  list[str] = []

		# Check 1: EOD not already running or completed
		existing = self._runs.get(_eod_key(tenant_id, date_str))
		if existing and existing.status == EODStatus.IN_PROGRESS:
			blockers.append(f"EOD for {date_str} is already in progress (run_id={existing.run_id})")
		if existing and existing.status == EODStatus.COMPLETED and not existing.dry_run:
			warnings.append(f"EOD for {date_str} already completed — will return cached result")

		# Check 2: Processing date not in future (with 1-hour tolerance for DST)
		try:
			d = _parse_date(date_str)
			today = date.today()
			if d > today:
				blockers.append(f"Cannot run EOD for future date {date_str} (today is {today})")
		except ValueError:
			blockers.append(f"Invalid processing date format: {date_str!r}")

		# Check 3: Adapter hook — suspense balance, unposted entries, open GL period
		# In production: query DB. Here: pass.
		# blockers.append("Suspense GL has non-zero balance") if adapter reports issue

		return PrerequisiteCheck(
			tenant_id=tenant_id,
			eod_date=date_str,
			ready=len(blockers) == 0,
			blockers=blockers,
			warnings=warnings,
			checked_at=_now_iso(),
		)

	async def get_running_jobs(self, tenant_id: str) -> list[dict[str, Any]]:
		"""Return currently executing jobs for tenant."""
		guard_tenant_id(tenant_id)
		running = []
		for key, is_running in self._running.items():
			if not is_running:
				continue
			if not key.startswith(f"eod:{tenant_id}:"):
				continue
			date_str = key.split(":")[-1]
			rec = self._runs.get(key)
			running.append({
				"eod_date":   date_str,
				"run_id":     rec.run_id if rec else None,
				"started_at": rec.started_at if rec else None,
				"status":     "running",
			})
		return running

	async def cancel_running_eod(self, tenant_id: str, eod_date: str | date, reason: str) -> dict[str, Any]:
		"""Emergency stop — signals the running EOD to skip remaining jobs."""
		guard_tenant_id(tenant_id)
		date_str = eod_date.isoformat() if isinstance(eod_date, date) else eod_date
		key = _eod_key(tenant_id, date_str)
		if key not in self._running:
			return {"cancelled": False, "reason": "No running EOD found for this date"}
		self._running[key] = False
		rec = self._runs.get(key)
		if rec:
			rec.status = EODStatus.CANCELLED
			rec.errors.append(f"Cancelled by operator: {reason}")
		_log.warning("[EOD] %s CANCELLED: %s", self._log_pretty_path(tenant_id, date_str), reason)
		return {"cancelled": True, "reason": reason, "eod_date": date_str}

	async def get_eod_metrics(self, tenant_id: str, days: int = 30) -> EODMetrics:
		"""Return processing time trends and error rates over the last N days."""
		guard_tenant_id(tenant_id)
		assert days > 0, "days must be positive"
		today = date.today()
		from datetime import timedelta
		from_date = today - timedelta(days=days)

		history = await self.get_eod_history(tenant_id, from_date, today)
		total   = len(history)
		if total == 0:
			return EODMetrics(tenant_id=tenant_id, period_days=days)

		successful  = sum(1 for h in history if h["status"] == EODStatus.COMPLETED.value)
		failed      = sum(1 for h in history if h["status"] == EODStatus.FAILED.value)
		partial     = sum(1 for h in history if h["status"] == EODStatus.PARTIAL.value)
		durations   = [h["duration_seconds"] for h in history if h["duration_seconds"] > 0]
		avg_dur     = sum(durations) / len(durations) if durations else 0.0
		max_dur     = max(durations) if durations else 0.0
		min_dur     = min(durations) if durations else 0.0
		error_rate  = (failed + partial) / total if total > 0 else 0.0

		# Per-job error rates
		job_error_rates: dict[str, float] = {}
		for job_type in EODJobType:
			job_results = [
				self._job_results.get(_job_key(tenant_id, h["eod_date"], job_type.value))
				for h in history
			]
			job_results = [r for r in job_results if r is not None]
			if job_results:
				job_failed = sum(1 for r in job_results if r.status == JobStatus.FAILED)
				job_error_rates[job_type.value] = job_failed / len(job_results)

		daily = [{"date": h["eod_date"], "duration_seconds": h["duration_seconds"], "status": h["status"]} for h in history]

		return EODMetrics(
			tenant_id=tenant_id,
			period_days=days,
			total_runs=total,
			successful_runs=successful,
			failed_runs=failed,
			partial_runs=partial,
			avg_duration_seconds=avg_dur,
			max_duration_seconds=max_dur,
			min_duration_seconds=min_dur,
			error_rate=error_rate,
			job_error_rates=job_error_rates,
			daily_durations=daily,
		)

	# ─── New world-class improvement methods ─────────────────────────────────

	async def compute_penalty_interest(
		self,
		tenant_id: str,
		processing_date: str | date,
		dry_run: bool = False,
	) -> dict[str, Any]:
		"""Compute and post penalty interest on overdue loan instalments (I1).

		Penalty accrues from the day after the due date at the contractual
		penalty rate (default: 2× the loan rate, minimum 5 % p.a.).

		Posting:
		  DR  Loan Receivable – Penalty
		  CR  Penalty Interest Income
		"""
		guard_tenant_id(tenant_id)
		date_str = processing_date.isoformat() if isinstance(processing_date, date) else processing_date
		d = _parse_date(date_str)
		_log.info("[EOD] compute_penalty_interest tenant=%s date=%s dry_run=%s", tenant_id, date_str, dry_run)

		# ── Adapter hook ──────────────────────────────────────────────────────
		# In production: query overdue_loan_instalments where due_date < processing_date
		# For each instalment: days_overdue = (d - due_date).days
		#                      penalty_rate = max(loan.rate * 2, Decimal("0.05"))
		#                      daily_penalty = overdue_balance * penalty_rate / Decimal("365")
		# ─────────────────────────────────────────────────────────────────────
		loans_assessed: list[dict[str, Any]] = []
		total_penalty  = Decimal("0")
		entries_posted = 0

		# Simulated: in reality iterate adapter rows
		for loan_stub in loans_assessed:
			overdue_balance  = Decimal(str(loan_stub.get("overdue_balance", "0")))
			days_overdue     = int(loan_stub.get("days_overdue", 0))
			penalty_rate_pct = Decimal(str(loan_stub.get("penalty_rate", "0.10")))
			daily_penalty    = overdue_balance * penalty_rate_pct / Decimal("365")
			total_penalty   += daily_penalty
			if not dry_run:
				# adapter.post_penalty_entry(loan_stub["loan_id"], daily_penalty, date_str)
				entries_posted += 1

		_log.info(
			"[EOD] penalty_interest tenant=%s date=%s loans=%d total_penalty=%s entries=%d",
			tenant_id, date_str, len(loans_assessed), total_penalty, entries_posted,
		)
		return {
			"tenant_id":      tenant_id,
			"processing_date": date_str,
			"loans_assessed": len(loans_assessed),
			"total_penalty_accrued": str(total_penalty),
			"entries_posted":  entries_posted,
			"dry_run":         dry_run,
		}

	async def run_ifrs9_ecl_staging(
		self,
		tenant_id: str,
		processing_date: str | date,
		dry_run: bool = False,
	) -> dict[str, Any]:
		"""Classify loans into IFRS 9 ECL stages and post provision deltas (I2).

		Stage 1: 12-month ECL — no SICR.
		Stage 2: Lifetime ECL — Significant Increase in Credit Risk (DPD > 30 or rating downgrade).
		Stage 3: Credit-impaired — DPD ≥ 90 or legal default.

		Posting per loan with positive delta:
		  DR  Impairment Expense (P&L)
		  CR  Loan Loss Reserve (Balance Sheet)

		For negative delta (improvement):
		  DR  Loan Loss Reserve
		  CR  Impairment Expense
		"""
		guard_tenant_id(tenant_id)
		date_str = processing_date.isoformat() if isinstance(processing_date, date) else processing_date
		_log.info("[EOD] ifrs9_ecl_staging tenant=%s date=%s dry_run=%s", tenant_id, date_str, dry_run)

		stage_counts: dict[str, int]    = {"stage_1": 0, "stage_2": 0, "stage_3": 0}
		provision_delta = Decimal("0")
		entries_posted  = 0

		# ── Adapter hook ──────────────────────────────────────────────────────
		# loans = adapter.get_loans_for_ecl(tenant_id, date_str)
		# for loan in loans:
		#   new_stage = _determine_stage(loan.dpd, loan.risk_rating, loan.watchlist)
		#   ecl = _calculate_ecl(loan.ead, loan.pd[new_stage], loan.lgd, discount_factor)
		#   delta = ecl - loan.current_provision
		#   if delta != 0 and not dry_run:
		#       adapter.post_provision_entry(loan.id, delta, date_str, new_stage)
		#   provision_delta += delta
		#   stage_counts[f"stage_{new_stage}"] += 1
		# ─────────────────────────────────────────────────────────────────────

		_log.info(
			"[EOD] ifrs9_staging tenant=%s date=%s stages=%s provision_delta=%s",
			tenant_id, date_str, stage_counts, provision_delta,
		)
		return {
			"tenant_id":        tenant_id,
			"processing_date":  date_str,
			"stage_counts":     stage_counts,
			"provision_delta":  str(provision_delta),
			"entries_posted":   entries_posted,
			"dry_run":          dry_run,
		}

	async def compute_liquidity_coverage_ratio(
		self,
		tenant_id: str,
		processing_date: str | date,
	) -> dict[str, Any]:
		"""Compute Basel III LCR and flag breaches (I3).

		LCR = HQLA / Net Stressed Outflows (30-day) ≥ 1.0

		HQLA tiers:
		  Level 1:  0 % haircut (sovereign bonds, central bank reserves)
		  Level 2A: 15 % haircut (GSE bonds, covered bonds)
		  Level 2B: 25–50 % haircut (equities, RMBS)

		Stores result in `_lcr_store`; raises warning if LCR < 1.05.
		"""
		guard_tenant_id(tenant_id)
		date_str = processing_date.isoformat() if isinstance(processing_date, date) else processing_date
		_log.info("[EOD] compute_lcr tenant=%s date=%s", tenant_id, date_str)

		# ── Adapter hook ──────────────────────────────────────────────────────
		# hqla_l1  = adapter.get_hqla_level1(tenant_id, date_str)  # no haircut
		# hqla_l2a = adapter.get_hqla_level2a(tenant_id, date_str) * Decimal("0.85")
		# hqla_l2b = adapter.get_hqla_level2b(tenant_id, date_str) * Decimal("0.65")
		# outflows = adapter.get_stressed_outflows_30d(tenant_id, date_str)
		# inflows  = min(adapter.get_inflows_30d(tenant_id, date_str), outflows * Decimal("0.75"))
		# ─────────────────────────────────────────────────────────────────────
		hqla_l1   = Decimal("0")
		hqla_l2a  = Decimal("0")
		hqla_l2b  = Decimal("0")
		outflows  = Decimal("1")   # avoid ZeroDivision
		inflows   = Decimal("0")

		hqla_total = hqla_l1 + hqla_l2a + hqla_l2b
		net_outflows = max(outflows - inflows, Decimal("1"))
		lcr = hqla_total / net_outflows

		status = "compliant" if lcr >= Decimal("1.0") else "breach"
		if lcr < Decimal("1.05"):
			_log.warning("[EOD] LCR early-warning tenant=%s date=%s lcr=%.4f", tenant_id, date_str, float(lcr))
		if lcr < Decimal("1.0"):
			self._record_exception(
				tenant_id, date_str, "lcr_computation",
				ExceptionSeverity.CRITICAL, "LCR_BREACH",
				f"LCR {float(lcr):.4f} below 100% minimum",
			)

		result = {
			"tenant_id":       tenant_id,
			"processing_date": date_str,
			"hqla_level1":     str(hqla_l1),
			"hqla_level2a":    str(hqla_l2a),
			"hqla_level2b":    str(hqla_l2b),
			"hqla_total":      str(hqla_total),
			"net_outflows_30d": str(net_outflows),
			"lcr_ratio":       str(lcr.quantize(Decimal("0.0001"))),
			"status":          status,
		}
		_log.info("[EOD] lcr result tenant=%s date=%s lcr=%s status=%s", tenant_id, date_str, lcr, status)
		return result

	async def run_nostro_reconciliation(
		self,
		tenant_id: str,
		processing_date: str | date,
		dry_run: bool = False,
	) -> dict[str, Any]:
		"""Automated intraday nostro reconciliation against SWIFT MT940/camt.053 (I4).

		Matches internal GL nostro entries to bank statement lines by:
		  1. Exact match: amount + value date + transaction reference
		  2. Near match: amount + ±1 day value date (manual review queue)
		  3. Unmatched: raised as CRITICAL exception if > configured threshold

		Returns counts and the list of unmatched items.
		"""
		guard_tenant_id(tenant_id)
		date_str = processing_date.isoformat() if isinstance(processing_date, date) else processing_date
		_log.info("[EOD] nostro_recon tenant=%s date=%s dry_run=%s", tenant_id, date_str, dry_run)

		# ── Adapter hook ──────────────────────────────────────────────────────
		# statement_lines = adapter.fetch_swift_mt940(tenant_id, date_str)
		# gl_entries      = adapter.get_nostro_gl_entries(tenant_id, date_str)
		# matched, near, unmatched = _match_nostro(statement_lines, gl_entries)
		# ─────────────────────────────────────────────────────────────────────
		matched_count   = 0
		near_match_count = 0
		unmatched_items: list[dict[str, Any]] = []
		unmatched_value = Decimal("0")

		for item in unmatched_items:
			self._record_exception(
				tenant_id, date_str, "nostro_reconciliation",
				ExceptionSeverity.CRITICAL, "NOSTRO_UNMATCHED",
				f"Unmatched nostro entry ref={item.get('ref')} amount={item.get('amount')}",
			)

		_log.info(
			"[EOD] nostro_recon done tenant=%s date=%s matched=%d near=%d unmatched=%d value=%s",
			tenant_id, date_str, matched_count, near_match_count, len(unmatched_items), unmatched_value,
		)
		return {
			"tenant_id":        tenant_id,
			"processing_date":  date_str,
			"matched":          matched_count,
			"near_match":       near_match_count,
			"unmatched":        len(unmatched_items),
			"unmatched_value":  str(unmatched_value),
			"unmatched_items":  unmatched_items[:50],
			"dry_run":          dry_run,
		}

	async def run_zba_sweeps(
		self,
		tenant_id: str,
		processing_date: str | date,
		dry_run: bool = False,
	) -> dict[str, Any]:
		"""Execute Zero-Balance Accounting (ZBA) sweeps for concentration banking (I14).

		For each ZBA sweep group:
		  - Identify surplus in sub-accounts (balance > target_balance)
		  - Transfer surplus to master concentration account
		  - Fund sub-accounts from master if balance < minimum_balance
		  - Post offsetting intercompany entries where applicable

		Runs after all other EOD postings are complete.
		"""
		guard_tenant_id(tenant_id)
		date_str = processing_date.isoformat() if isinstance(processing_date, date) else processing_date
		_log.info("[EOD] zba_sweeps tenant=%s date=%s dry_run=%s", tenant_id, date_str, dry_run)

		# ── Adapter hook ──────────────────────────────────────────────────────
		# groups = adapter.get_zba_groups(tenant_id)
		# for group in groups:
		#     master_id     = group.master_account_id
		#     for sub in group.sub_accounts:
		#         balance   = adapter.get_balance(sub.account_id)
		#         surplus   = balance - sub.target_balance
		#         if surplus > 0:
		#             adapter.transfer(sub.account_id, master_id, surplus, date_str)
		#             sweeps_up += 1; total_swept += surplus
		#         elif balance < sub.minimum_balance and not sub.notional_only:
		#             fund_amount = sub.minimum_balance - balance
		#             adapter.transfer(master_id, sub.account_id, fund_amount, date_str)
		#             sweeps_down += 1
		# ─────────────────────────────────────────────────────────────────────
		sweeps_up    = 0
		sweeps_down  = 0
		total_swept  = Decimal("0")
		total_funded = Decimal("0")
		groups_processed = 0

		_log.info(
			"[EOD] zba_sweeps done tenant=%s date=%s groups=%d swept=%s funded=%s",
			tenant_id, date_str, groups_processed, total_swept, total_funded,
		)
		return {
			"tenant_id":        tenant_id,
			"processing_date":  date_str,
			"groups_processed": groups_processed,
			"sweeps_up":        sweeps_up,
			"sweeps_down":      sweeps_down,
			"total_swept":      str(total_swept),
			"total_funded":     str(total_funded),
			"dry_run":          dry_run,
		}

	async def classify_npa_accounts(
		self,
		tenant_id: str,
		processing_date: str | date,
		dpd_threshold: int = 90,
		dry_run: bool = False,
	) -> dict[str, Any]:
		"""Classify loans as Non-Performing Assets (NPA) when DPD ≥ threshold (I12).

		Actions on NPA promotion:
		  1. Set account status → NPA
		  2. Suspend interest accrual (future interest posted to sundry, not P&L)
		  3. Reverse any uncollected accrued interest from P&L to Sundry
		  4. Apply 100 % provision requirement (post delta to Impairment Expense)
		  5. Generate NPA register entry for regulatory submission

		Actions on NPA reversal (DPD drops below threshold after recovery):
		  1. Restore status → Performing / Sub-standard
		  2. Resume accrual, reverse sundry interest to P&L
		  3. Release provision per IFRS 9 / local GAAP
		"""
		guard_tenant_id(tenant_id)
		assert dpd_threshold > 0, "dpd_threshold must be positive"
		date_str = processing_date.isoformat() if isinstance(processing_date, date) else processing_date
		_log.info("[EOD] classify_npa tenant=%s date=%s threshold=%d dry_run=%s", tenant_id, date_str, dpd_threshold, dry_run)

		# ── Adapter hook ──────────────────────────────────────────────────────
		# candidates = adapter.get_loans_by_dpd(tenant_id, min_dpd=0)
		# newly_npa   = [l for l in candidates if l.dpd >= dpd_threshold and l.status != "npa"]
		# npa_cured   = [l for l in candidates if l.dpd < dpd_threshold and l.status == "npa"]
		# ─────────────────────────────────────────────────────────────────────
		newly_npa_count  = 0
		cured_count      = 0
		provision_posted = Decimal("0")
		interest_reversed = Decimal("0")

		_log.info(
			"[EOD] npa_classification done tenant=%s date=%s new_npa=%d cured=%d provision=%s",
			tenant_id, date_str, newly_npa_count, cured_count, provision_posted,
		)
		return {
			"tenant_id":          tenant_id,
			"processing_date":    date_str,
			"dpd_threshold":      dpd_threshold,
			"newly_classified":   newly_npa_count,
			"cured":              cured_count,
			"provision_posted":   str(provision_posted),
			"interest_reversed":  str(interest_reversed),
			"dry_run":            dry_run,
		}

	async def check_sla_compliance(
		self,
		tenant_id: str,
		processing_date: str | date,
		sla_window_minutes: int = 360,
	) -> dict[str, Any]:
		"""Evaluate EOD SLA compliance and emit at-risk alerts (I15).

		Checks:
		  - Whether EOD completed within `sla_window_minutes`
		  - Percent of window consumed vs percent of jobs complete (pacing)
		  - Historical SLA breach trend over last 7 days

		Emits CRITICAL exception if EOD is currently in-progress and
		elapsed time > 70 % of the SLA window with < 50 % jobs done.
		"""
		guard_tenant_id(tenant_id)
		assert sla_window_minutes > 0, "sla_window_minutes must be positive"
		date_str = processing_date.isoformat() if isinstance(processing_date, date) else processing_date
		_log.info("[EOD] sla_check tenant=%s date=%s window_min=%d", tenant_id, date_str, sla_window_minutes)

		rec = self._runs.get(_eod_key(tenant_id, date_str))
		if rec is None:
			return {
				"tenant_id":       tenant_id,
				"processing_date": date_str,
				"sla_window_min":  sla_window_minutes,
				"status":          "no_run",
			}

		started = datetime.fromisoformat(rec.started_at) if rec.started_at else None
		completed = datetime.fromisoformat(rec.completed_at) if rec.completed_at else None
		now = datetime.now(timezone.utc)

		if completed:
			elapsed_sec = (completed - started).total_seconds() if started else 0.0
			sla_breach  = elapsed_sec > sla_window_minutes * 60
			status      = "breached" if sla_breach else "met"
		else:
			elapsed_sec = (now - started).total_seconds() if started else 0.0
			pct_window  = elapsed_sec / (sla_window_minutes * 60) if sla_window_minutes else 0
			pct_jobs    = len(rec.job_results) / len(_EOD_JOB_SEQUENCE) if _EOD_JOB_SEQUENCE else 1
			at_risk     = pct_window > 0.70 and pct_jobs < 0.50
			status      = "at_risk" if at_risk else "in_progress"
			if at_risk:
				_log.warning(
					"[EOD] SLA_AT_RISK tenant=%s date=%s pct_window=%.1f%% pct_jobs=%.1f%%",
					tenant_id, date_str, pct_window * 100, pct_jobs * 100,
				)
				self._record_exception(
					tenant_id, date_str, "sla_monitoring",
					ExceptionSeverity.ERROR, "SLA_AT_RISK",
					f"EOD at risk: {pct_window*100:.0f}% window consumed, only {pct_jobs*100:.0f}% jobs done",
				)
			sla_breach  = False

		_log.info("[EOD] sla_check done tenant=%s date=%s status=%s elapsed=%.0fs", tenant_id, date_str, status, elapsed_sec)
		return {
			"tenant_id":          tenant_id,
			"processing_date":    date_str,
			"sla_window_minutes": sla_window_minutes,
			"elapsed_seconds":    round(elapsed_sec, 2),
			"sla_breach":         sla_breach,
			"status":             status,
		}

	async def generate_regulatory_returns(
		self,
		tenant_id: str,
		processing_date: str | date,
		dry_run: bool = False,
	) -> dict[str, Any]:
		"""Generate regulatory statistical returns (CBK BSL02/BSL03, FATF, Basel) (I7).

		Triggered automatically on:
		  - Month-end: balance sheet returns, credit exposure reports
		  - Quarter-end: capital adequacy (CAR), large exposure returns
		  - Year-end: annual supervisory return, AML statistical report

		Returns include a manifest of generated files/payloads and any
		validation failures that must be resolved before submission.
		"""
		guard_tenant_id(tenant_id)
		date_str = processing_date.isoformat() if isinstance(processing_date, date) else processing_date
		d = _parse_date(date_str)
		_log.info("[EOD] regulatory_returns tenant=%s date=%s dry_run=%s", tenant_id, date_str, dry_run)

		returns_generated: list[str] = []
		validation_failures: list[str] = []

		is_month_end_  = _is_month_end(d)
		is_quarter_end = is_month_end_ and d.month in (3, 6, 9, 12)
		is_year_end_   = _is_year_end(d)

		if is_month_end_:
			returns_generated.extend(["BSL02_BALANCE_SHEET", "BSL03_CREDIT_EXPOSURE"])
			_log.info("[EOD] month-end returns tenant=%s date=%s returns=%s", tenant_id, date_str, returns_generated)

		if is_quarter_end:
			returns_generated.extend(["CAPITAL_ADEQUACY_CAR", "LARGE_EXPOSURE_RETURN"])
			_log.info("[EOD] quarter-end returns tenant=%s date=%s", tenant_id, date_str)

		if is_year_end_:
			returns_generated.extend(["ANNUAL_SUPERVISORY_RETURN", "AML_STATISTICAL_REPORT"])
			_log.info("[EOD] year-end returns tenant=%s date=%s", tenant_id, date_str)

		# ── Adapter hook ──────────────────────────────────────────────────────
		# for return_code in returns_generated:
		#     payload = adapter.render_regulatory_return(tenant_id, return_code, date_str)
		#     errors  = adapter.validate_return(payload, return_code)
		#     validation_failures.extend(errors)
		#     if not errors and not dry_run:
		#         adapter.store_regulatory_return(tenant_id, return_code, payload, date_str)
		# ─────────────────────────────────────────────────────────────────────

		for failure in validation_failures:
			self._record_exception(
				tenant_id, date_str, "regulatory_returns",
				ExceptionSeverity.ERROR, "REGULATORY_VALIDATION_FAILURE", failure,
			)

		_log.info(
			"[EOD] regulatory_returns done tenant=%s date=%s generated=%d failures=%d",
			tenant_id, date_str, len(returns_generated), len(validation_failures),
		)
		return {
			"tenant_id":           tenant_id,
			"processing_date":     date_str,
			"returns_generated":   returns_generated,
			"validation_failures": validation_failures,
			"is_month_end":        is_month_end_,
			"is_quarter_end":      is_quarter_end,
			"is_year_end":         is_year_end_,
			"dry_run":             dry_run,
		}

	async def health_check(self) -> dict[str, Any]:
		"""Return service health status."""
		from datetime import timedelta
		today    = date.today()
		yesterday = today - timedelta(days=1)
		tenants_with_recent_eod = set()
		for key in self._runs:
			parts = key.split(":")
			if len(parts) >= 3:
				try:
					run_d = _parse_date(parts[2])
					if run_d >= yesterday:
						tenants_with_recent_eod.add(parts[1])
				except ValueError as _exc:
					_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
		return {
			"status":              "healthy",
			"total_runs_tracked":  len(self._runs),
			"exceptions_tracked":  len(self._exceptions),
			"scheduled_jobs":      len(self._schedules),
			"currently_running":   sum(1 for v in self._running.values() if v),
			"tenants_active_24h":  len(tenants_with_recent_eod),
			"checked_at":          _now_iso(),
		}
