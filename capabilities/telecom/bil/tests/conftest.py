"""Shared fixtures for telecom/bil tests.

Uses importlib.util to load modules directly — avoids relative import issues
when running tests from outside the installed package.

© 2025 Datacraft. All rights reserved.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
	sys.path.insert(0, str(PACKAGE_DIR))


def _load(name: str, path: Path) -> Any:
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec and spec.loader
	mod = importlib.util.module_from_spec(spec)
	sys.modules[name] = mod
	spec.loader.exec_module(mod)  # type: ignore[union-attr]
	return mod


# Pre-load capability_contract first (service depends on it)
_cc = _load("capability_contract_bil", PACKAGE_DIR / "capability_contract.py")
sys.modules["apg_telecom_bil.capability_contract"] = _cc  # type: ignore[assignment]

# Load models
_models = _load("models_bil", PACKAGE_DIR / "models.py")
sys.modules["apg_telecom_bil.models"] = _models  # type: ignore[assignment]


@pytest.fixture(scope="session")
def ServiceClass():
	"""Return TelecomBillingService class (loaded once per session)."""
	svc_mod = _load("service_bil_session", PACKAGE_DIR / "service.py")
	return svc_mod.TelecomBillingService


@pytest.fixture
def svc(ServiceClass):
	"""Fresh in-memory service scoped to tenant-test."""
	return ServiceClass(tenant_id="tenant-test", actor_id="test-actor")


@pytest.fixture
def svc_b(ServiceClass):
	"""Second tenant service for isolation tests."""
	return ServiceClass(tenant_id="tenant-b", actor_id="test-actor-b")


@pytest.fixture
def seeded_svc(ServiceClass):
	"""Service pre-seeded with a cycle, invoice, payment, and CDR."""
	s = ServiceClass(tenant_id="tenant-seed", actor_id="seed-actor")
	s.create_bill_cycle(
		cycle_id="cyc-seed",
		cycle_type="monthly",
		cutoff_date="2026-05-31",
		start_date="2026-05-01",
		end_date="2026-05-31",
	)
	s.generate_invoice(
		invoice_id="inv-seed",
		customer_id="cust-seed",
		cycle_id="cyc-seed",
		total_amount=1000.0,
		currency="KES",
		due_date="2026-06-15",
	)
	s.record_cdr(
		cdr_id="cdr-seed",
		source="MSC-01",
		mediation_status="raw",
		msisdn="+254700000001",
		duration_seconds=120,
		data_volume_bytes=0,
		recorded_at="2026-05-10T12:00:00Z",
	)
	s.record_charge(
		charge_id="chg-seed",
		customer_id="cust-seed",
		charge_type="recurring",
		rating_type="flat_rate",
		amount=1000.0,
		currency="KES",
		tax_amount=160.0,
		cdr_id="cdr-seed",
	)
	return s
