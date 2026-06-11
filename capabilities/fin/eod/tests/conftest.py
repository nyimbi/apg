"""Shared fixtures for EOD tests."""
from __future__ import annotations

import asyncio
import pytest
import sys
import os

# Allow imports from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))

from capabilities.fin.eod.service import EODService
from capabilities.fin.eod.models  import EODJobType, JobStatus, EODStatus


TENANT = "test_tenant_01"
DATE   = "2026-05-31"   # month-end date for full test coverage
DATE_MID = "2026-05-15" # mid-month — FX + period_close should be skipped


@pytest.fixture
def svc() -> EODService:
	return EODService()


@pytest.fixture
def loop():
	loop = asyncio.new_event_loop()
	yield loop
	loop.close()
