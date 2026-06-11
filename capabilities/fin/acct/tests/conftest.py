"""Test fixtures for fin.acct capability."""

from __future__ import annotations

import asyncio
import sys
import os
from decimal import Decimal

import pytest

# Ensure capabilities root is on path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
if _ROOT not in sys.path:
	sys.path.insert(0, _ROOT)

from capabilities.fin.acct.service import BankAccountService
from capabilities.fin.acct.models import AccountStatus, AccountType, TransactionType


@pytest.fixture
def svc():
	return BankAccountService(tenant_id="test-tenant", user_id="test-user")


@pytest.fixture
def tenant():
	return "test-tenant"


@pytest.fixture
def customer():
	return "cust-001"


def run(coro):
	"""Run coroutine synchronously in tests."""
	loop = asyncio.get_event_loop()
	return loop.run_until_complete(coro)
