"""Pytest configuration and shared fixtures for APG Digital Payments tests."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from decimal import Decimal

import pytest

# Repo root so that relative imports in service.py work via capabilities package
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def svc():
	"""Return a fresh in-memory DigitalPaymentsService for each test."""
	from service import DigitalPaymentsService
	return DigitalPaymentsService(tenant_id="test-tenant", actor_id="test-actor")


@pytest.fixture
def svc2():
	"""Second tenant service for cross-tenant isolation tests."""
	from service import DigitalPaymentsService
	return DigitalPaymentsService(tenant_id="other-tenant", actor_id="other-actor")


@pytest.fixture
def loop():
	"""Event loop for async tests."""
	loop = asyncio.new_event_loop()
	yield loop
	loop.close()


def run(coro, loop=None):
	"""Helper: run a coroutine synchronously in tests."""
	if loop:
		return loop.run_until_complete(coro)
	return asyncio.run(coro)
