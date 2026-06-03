"""Shared pytest fixtures for AML tests."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

# Ensure the AML package is importable from this test directory
PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
	sys.path.insert(0, str(PACKAGE_DIR))


@pytest.fixture
def tenant_id() -> str:
	return "test-tenant"


@pytest.fixture
def actor_id() -> str:
	return "test-actor"


@pytest.fixture
def aml_service(tenant_id: str, actor_id: str):
	"""Fresh AMLService instance per test."""
	from service import AMLService  # type: ignore
	return AMLService(tenant_id=tenant_id, actor_id=actor_id)


@pytest.fixture
def legacy_service():
	"""Fresh AntiMoneyLaunderingService (sync legacy API) per test."""
	from service import AntiMoneyLaunderingService  # type: ignore
	return AntiMoneyLaunderingService()


@pytest.fixture
def sample_txn() -> dict:
	return {
		"id": "txn-fixture-1",
		"subject_reference": "customer-abc",
		"kyc_profile_id": "kyc-abc",
		"amount": 500.0,
		"currency": "USD",
		"source_capability": "fintech_payments",
		"source_reference": "pay-001",
		"risk_score": 20,
		"sender_account": "acc-001",
		"receiver_account": "acc-002",
	}


@pytest.fixture
def large_txn() -> dict:
	return {
		"id": "txn-large-1",
		"subject_reference": "customer-xyz",
		"kyc_profile_id": "kyc-xyz",
		"amount": 15_000.0,
		"currency": "USD",
		"source_capability": "fintech_payments",
		"source_reference": "pay-002",
		"risk_score": 55,
		"sender_account": "acc-003",
		"receiver_account": "acc-004",
		"sanctions_hit": False,
	}
