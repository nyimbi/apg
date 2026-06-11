"""Unit tests for EOD/BOD Pydantic models."""
from __future__ import annotations

import asyncio
from decimal import Decimal

from capabilities.fin.eod.models import (
	EODException, EODJobType, EODReport, EODResult, EODStatus,
	ExceptionSeverity, JobResult, JobStatus, PrerequisiteCheck,
	ProcessingStats, _EODRunRecord,
)


def test_processing_stats_defaults():
	s = ProcessingStats()
	assert s.accounts_processed == 0
	assert s.total_amount == "0"
	assert s.total_amount_decimal == Decimal("0")


def test_processing_stats_decimal_round_trip():
	s = ProcessingStats(total_amount="1234567.89")
	assert s.total_amount_decimal == Decimal("1234567.89")


def test_job_result_construction():
	jr = JobResult(
		job_name="interest_accrual_batch",
		job_type=EODJobType.INTEREST_ACCRUAL,
		status=JobStatus.COMPLETED,
		tenant_id="t1",
		processing_date="2026-06-11",
		started_at="2026-06-11T22:00:00+00:00",
		completed_at="2026-06-11T22:01:30+00:00",
	)
	assert jr.duration_seconds == 0.0   # computed only by service
	assert jr.was_cached is False
	assert jr.dry_run is False


def test_eod_result_construction():
	result = EODResult(
		run_id="run1", tenant_id="t1", eod_date="2026-06-11",
		status=EODStatus.COMPLETED,
	)
	assert result.jobs_completed == 0
	assert result.total_transactions == 0


def test_eod_run_record_to_eod_result():
	jr = JobResult(
		job_name="fee_posting_batch", job_type=EODJobType.FEE_POSTING,
		status=JobStatus.COMPLETED, tenant_id="t1", processing_date="2026-06-11",
		stats=ProcessingStats(transactions_posted=5),
	)
	rec = _EODRunRecord(
		run_id="r1", tenant_id="t1", eod_date="2026-06-11",
		status=EODStatus.COMPLETED,
		started_at="2026-06-11T22:00:00+00:00",
		completed_at="2026-06-11T22:05:00+00:00",
		job_results=[jr],
	)
	result = rec.to_eod_result()
	assert result.jobs_completed == 1
	assert result.total_transactions == 5
	assert result.duration_seconds == pytest.approx(300.0, abs=1.0)


def test_eod_exception_resolve_fields():
	exc = EODException(
		exception_id="exc1", tenant_id="t1", processing_date="2026-06-11",
		job_name="fee_posting_batch", severity=ExceptionSeverity.ERROR,
		error_code="FEE_001", message="Insufficient funds",
		created_at="2026-06-11T22:01:00+00:00",
	)
	assert exc.resolved is False
	assert exc.resolved_at is None


def test_prerequisite_check_ready():
	p = PrerequisiteCheck(tenant_id="t1", eod_date="2026-06-11", ready=True)
	assert p.ready is True
	assert p.blockers == []


def test_prerequisite_check_blocked():
	p = PrerequisiteCheck(
		tenant_id="t1", eod_date="2026-06-11", ready=False,
		blockers=["Suspense account non-zero: 5000"],
	)
	assert p.ready is False
	assert len(p.blockers) == 1


def test_eod_report_construction():
	r = EODReport(
		report_id="rep1", tenant_id="t1", processing_date="2026-06-11",
		generated_at="2026-06-11T23:00:00+00:00", eod_status=EODStatus.COMPLETED,
	)
	assert r.interest_accruals == 0
	assert r.exceptions_count == 0


def test_job_status_enum_values():
	assert JobStatus.COMPLETED.value == "completed"
	assert JobStatus.FAILED.value    == "failed"
	assert JobStatus.SKIPPED.value   == "skipped"


def test_eod_job_type_enum_count():
	# 10 EOD jobs + 2 BOD jobs = 12
	assert len(EODJobType) == 12


import pytest  # noqa: E402 — needed for approx
