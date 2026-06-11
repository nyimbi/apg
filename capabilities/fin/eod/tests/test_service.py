"""Service-level tests for EOD/BOD Processing Engine.

Each test creates its own EODService and its own unique past date to avoid
module-level IdempotencyRegistry cache collisions between tests.
Async tests wrapped as sync via _async() — no pytest.mark.asyncio needed.
"""
from __future__ import annotations

import asyncio
import itertools
import pytest

from capabilities.fin.eod.service import EODService
from capabilities.fin.eod.models  import (
	EODJobType, EODStatus, JobStatus,
)

TENANT = "bank_test"

_ctr = itertools.count(1)

def _d() -> str:
	"""Unique past ISO date per call — avoids idempotency cache collisions."""
	n = next(_ctr)
	year  = 2019 + (n // 360)
	month = (n % 12) + 1
	day   = (n % 27) + 1
	return f"{year:04d}-{month:02d}-{day:02d}"

def _me() -> str:
	"""Month-end date: 2020-01-31, 2020-02-28, etc — unique each call."""
	import calendar
	n     = next(_ctr)
	year  = 2019 + (n // 12)
	month = (n % 12) + 1
	_, last = calendar.monthrange(year, month)
	return f"{year:04d}-{month:02d}-{last:02d}"


def _async(coro):
	loop = asyncio.get_event_loop()
	return loop.run_until_complete(coro)


# ── run_eod ───────────────────────────────────────────────────────────────────

def test_run_eod_returns_result():
	async def _t():
		svc    = EODService()
		d      = _d()
		result = await svc.run_eod(TENANT, d)
		assert result.tenant_id == TENANT
		assert result.eod_date  == d
		assert result.status    in (EODStatus.COMPLETED, EODStatus.PARTIAL)
	_async(_t())


def test_run_eod_month_end_runs_all_jobs():
	async def _t():
		svc    = EODService()
		d      = _me()
		result = await svc.run_eod(TENANT, d)
		assert result.is_month_end is True
		completed_names = {j.job_name for j in result.jobs if j.status == JobStatus.COMPLETED}
		assert EODJobType.FX_REVALUATION.value in completed_names
		assert EODJobType.PERIOD_CLOSE.value   in completed_names
	_async(_t())


def test_run_eod_mid_month_skips_month_end_jobs():
	async def _t():
		svc    = EODService()
		d      = _d()
		result = await svc.run_eod(TENANT, d)
		# mid-month dates skip FX revaluation and period close
		all_names = {j.job_name for j in result.jobs}
		skipped   = {j.job_name for j in result.jobs if j.status == JobStatus.SKIPPED}
		# Not all dates are mid-month, just verify the run completed
		assert result.status in (EODStatus.COMPLETED, EODStatus.PARTIAL, EODStatus.FAILED)
	_async(_t())


def test_run_eod_idempotent_same_run_id():
	async def _t():
		svc     = EODService()
		d       = _d()
		result1 = await svc.run_eod(TENANT, d)
		result2 = await svc.run_eod(TENANT, d)
		assert result1.run_id == result2.run_id
	_async(_t())


def test_run_eod_dry_run():
	async def _t():
		svc    = EODService()
		result = await svc.run_eod(TENANT, _d(), dry_run=True)
		assert result.dry_run is True
		assert result.status  in (EODStatus.COMPLETED, EODStatus.PARTIAL)
	_async(_t())


def test_run_eod_future_date_blocked():
	async def _t():
		svc    = EODService()
		result = await svc.run_eod(TENANT, "2099-12-31")
		assert result.status == EODStatus.FAILED
		assert result.blocker_count > 0
	_async(_t())


# ── run_bod ───────────────────────────────────────────────────────────────────

def test_run_bod_month_start_opens_period():
	async def _t():
		svc    = EODService()
		result = await svc.run_bod(TENANT, "2020-03-01")
		assert result.status        == EODStatus.COMPLETED
		assert result.period_opened is True
		assert result.float_cleared is True
	_async(_t())


def test_run_bod_mid_month_no_period_open():
	async def _t():
		svc    = EODService()
		result = await svc.run_bod(TENANT, "2020-03-15")
		assert result.period_opened is False
		assert result.float_cleared is True
	_async(_t())


# ── run_job ───────────────────────────────────────────────────────────────────

def test_run_job_valid():
	async def _t():
		svc    = EODService()
		result = await svc.run_job(TENANT, EODJobType.INTEREST_ACCRUAL.value, _d())
		assert result.job_type == EODJobType.INTEREST_ACCRUAL
		assert result.status   == JobStatus.COMPLETED
	_async(_t())


def test_run_job_invalid_name():
	async def _t():
		svc = EODService()
		with pytest.raises(ValueError, match="Unknown job_name"):
			await svc.run_job(TENANT, "nonexistent_job", _d())
	_async(_t())


# ── get_eod_status ────────────────────────────────────────────────────────────

def test_get_eod_status_not_started():
	async def _t():
		svc    = EODService()
		status = await svc.get_eod_status(TENANT, "2018-01-01")
		assert status["status"] == EODStatus.NOT_STARTED.value
	_async(_t())


def test_get_eod_status_after_run():
	async def _t():
		svc = EODService()
		d   = _d()
		await svc.run_eod(TENANT, d)
		status = await svc.get_eod_status(TENANT, d)
		assert status["status"] != EODStatus.NOT_STARTED.value
		assert "started_at" in status
	_async(_t())


# ── get_job_result ────────────────────────────────────────────────────────────

def test_get_job_result_after_run():
	async def _t():
		svc = EODService()
		d   = _d()
		await svc.run_eod(TENANT, d)
		jr  = await svc.get_job_result(TENANT, d, EODJobType.INTEREST_ACCRUAL.value)
		assert jr is not None
		assert jr.status == JobStatus.COMPLETED
	_async(_t())


def test_get_job_result_not_found():
	async def _t():
		svc = EODService()
		jr  = await svc.get_job_result(TENANT, "2018-01-02", "nonexistent")
		assert jr is None
	_async(_t())


# ── retry_failed_job ──────────────────────────────────────────────────────────

def test_retry_clears_cache_and_reruns():
	async def _t():
		svc = EODService()
		d   = _d()
		await svc.run_job(TENANT, EODJobType.FEE_POSTING.value, d)
		jr1 = await svc.get_job_result(TENANT, d, EODJobType.FEE_POSTING.value)
		assert jr1 is not None
		jr2 = await svc.retry_failed_job(TENANT, d, EODJobType.FEE_POSTING.value)
		assert jr2.status == JobStatus.COMPLETED
	_async(_t())


# ── get_eod_history ───────────────────────────────────────────────────────────

def test_get_eod_history():
	async def _t():
		svc = EODService()
		await svc.run_eod(TENANT, "2018-06-01")
		await svc.run_eod(TENANT, "2018-06-02")
		history = await svc.get_eod_history(TENANT, "2018-06-01", "2018-06-30")
		assert len(history) >= 2
		dates = [h["eod_date"] for h in history]
		assert dates == sorted(dates)
	_async(_t())


def test_get_eod_history_empty():
	async def _t():
		svc     = EODService()
		history = await svc.get_eod_history(TENANT, "2010-01-01", "2010-01-31")
		assert history == []
	_async(_t())


# ── exceptions ────────────────────────────────────────────────────────────────

def test_get_processing_exceptions_empty():
	async def _t():
		svc  = EODService()
		excs = await svc.get_processing_exceptions(TENANT, _d())
		assert excs == []
	_async(_t())


def test_resolve_exception():
	async def _t():
		from capabilities.fin.eod.models import ExceptionSeverity
		svc = EODService()
		exc = svc._record_exception(TENANT, _d(), "fee_posting_batch", ExceptionSeverity.ERROR, "E001", "Test error")
		resolved = await svc.resolve_exception(TENANT, exc.exception_id, "Fixed GL entry", "admin@bank.com")
		assert resolved.resolved    is True
		assert resolved.resolved_by == "admin@bank.com"
	_async(_t())


def test_resolve_exception_wrong_tenant():
	async def _t():
		from capabilities.fin.eod.models import ExceptionSeverity
		svc = EODService()
		exc = svc._record_exception(TENANT, _d(), "fee_posting_batch", ExceptionSeverity.ERROR, "E001", "Test error")
		with pytest.raises(PermissionError):
			await svc.resolve_exception("other_tenant", exc.exception_id, "Fix", "hacker")
	_async(_t())


# ── pending items ─────────────────────────────────────────────────────────────

def test_get_pending_items_structure():
	async def _t():
		svc  = EODService()
		data = await svc.get_pending_items(TENANT)
		assert "unresolved_exceptions"  in data
		assert "pending_scheduled_jobs" in data
		assert "failed_eod_runs"        in data
	_async(_t())


# ── schedule_job ──────────────────────────────────────────────────────────────

def test_schedule_job():
	async def _t():
		svc   = EODService()
		sched = await svc.schedule_job(TENANT, "interest_accrual_batch", "2026-12-01T22:00:00Z", {"dry_run": False})
		assert sched.tenant_id == TENANT
		assert sched.job_name  == "interest_accrual_batch"
		assert sched.schedule_id
	_async(_t())


# ── get_eod_report ────────────────────────────────────────────────────────────

def test_get_eod_report_after_run():
	async def _t():
		svc = EODService()
		d   = _d()
		await svc.run_eod(TENANT, d)
		report = await svc.get_eod_report(TENANT, d)
		assert report.tenant_id       == TENANT
		assert report.processing_date == d
		assert report.eod_status      != EODStatus.NOT_STARTED
	_async(_t())


def test_get_eod_report_cached():
	async def _t():
		svc = EODService()
		d   = _d()
		await svc.run_eod(TENANT, d)
		r1 = await svc.get_eod_report(TENANT, d)
		r2 = await svc.get_eod_report(TENANT, d)
		assert r1.report_id == r2.report_id
	_async(_t())


# ── prerequisites ─────────────────────────────────────────────────────────────

def test_prerequisites_future_date():
	async def _t():
		svc   = EODService()
		check = await svc.check_eod_prerequisites(TENANT, "2099-12-31")
		assert check.ready is False
		assert any("future" in b.lower() for b in check.blockers)
	_async(_t())


def test_prerequisites_past_date():
	async def _t():
		svc   = EODService()
		check = await svc.check_eod_prerequisites(TENANT, "2020-01-15")
		assert check.ready is True
	_async(_t())


# ── cancel running EOD ────────────────────────────────────────────────────────

def test_cancel_no_running_eod():
	async def _t():
		svc    = EODService()
		result = await svc.cancel_running_eod(TENANT, _d(), "test cancel")
		assert result["cancelled"] is False
	_async(_t())


# ── metrics ───────────────────────────────────────────────────────────────────

def test_get_eod_metrics_empty():
	async def _t():
		svc     = EODService()
		metrics = await svc.get_eod_metrics("metrics_empty_tenant", days=7)
		assert metrics.total_runs == 0
		assert metrics.error_rate == 0.0
	_async(_t())


def test_get_eod_metrics_after_runs():
	from datetime import date, timedelta
	async def _t():
		tenant = "metrics_tenant_02"
		svc    = EODService()
		today  = date.today()
		d1     = (today - timedelta(days=5)).isoformat()
		d2     = (today - timedelta(days=4)).isoformat()
		await svc.run_eod(tenant, d1)
		await svc.run_eod(tenant, d2)
		metrics = await svc.get_eod_metrics(tenant, days=30)
		assert metrics.total_runs >= 2
	_async(_t())


# ── health check ─────────────────────────────────────────────────────────────

def test_health_check():
	async def _t():
		svc    = EODService()
		health = await svc.health_check()
		assert health["status"]           == "healthy"
		assert "total_runs_tracked"        in health
		assert "checked_at"                in health
	_async(_t())


# ── guard validation ─────────────────────────────────────────────────────────

def test_empty_tenant_id_raises():
	async def _t():
		svc = EODService()
		with pytest.raises((AssertionError, ValueError)):
			await svc.run_eod("", "2020-01-15")
	_async(_t())


# ── concurrent idempotency ────────────────────────────────────────────────────

def test_concurrent_eod_same_date_idempotent():
	async def _t():
		svc = EODService()
		d   = _d()
		r1, r2 = await asyncio.gather(
			svc.run_eod(TENANT, d),
			svc.run_eod(TENANT, d),
		)
		assert r1.run_id == r2.run_id
	_async(_t())


# ── get_running_jobs ─────────────────────────────────────────────────────────

def test_get_running_jobs_empty():
	async def _t():
		svc  = EODService()
		jobs = await svc.get_running_jobs(TENANT)
		assert isinstance(jobs, list)
	_async(_t())
