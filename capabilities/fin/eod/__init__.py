"""APG EOD/BOD Processing Engine.

Standalone package: ``pip install apg-fin-eod``

Quick start::

    import asyncio
    from capabilities.fin.eod import EODService

    svc = EODService()
    result = asyncio.run(svc.run_eod("tenant_abc", "2026-06-11"))
    print(result.status, result.jobs_completed)

Capability ID : fin_eod
Provides      : eod_processing, bod_processing, batch_job_recovery,
                eod_reporting, exception_management
"""
from __future__ import annotations

__version__       = "1.0.0"
__package_name__  = "apg-fin-eod"
__capability_id__ = "fin_eod"

from .service import EODService
from .models  import (
	EODResult, BODResult, JobResult, EODException,
	EODReport, EODMetrics, PrerequisiteCheck, ProcessingStats,
	JobStatus, EODStatus, EODJobType, ExceptionSeverity,
)

__all__ = [
	"__version__", "__capability_id__",
	"EODService",
	"EODResult", "BODResult", "JobResult", "EODException",
	"EODReport", "EODMetrics", "PrerequisiteCheck", "ProcessingStats",
	"JobStatus", "EODStatus", "EODJobType", "ExceptionSeverity",
]
