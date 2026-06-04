"""
Pytest configuration and fixtures for APG Digital Lending tests.

© 2025 Datacraft. All rights reserved.
"""

from __future__ import annotations

import sys
import os
from datetime import date, timedelta
from typing import Any

import pytest

# Ensure the package root is importable when running from the tests/ directory
_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_root not in sys.path:
	sys.path.insert(0, _pkg_root)


# ---------------------------------------------------------------------------
# Service fixture — fresh instance per test
# ---------------------------------------------------------------------------

@pytest.fixture
def svc():
	"""Fresh LendingService with no data."""
	from service import LendingService
	return LendingService()


@pytest.fixture
def tenant_id() -> str:
	return "test_tenant"


@pytest.fixture
def actor_id() -> str:
	return "test_actor"


# ---------------------------------------------------------------------------
# Seed data factories
# ---------------------------------------------------------------------------

@pytest.fixture
def seeded_svc(svc, tenant_id):
	"""Service with one product + borrower registered."""
	svc.register_product(
		product_id="TERM01",
		tenant_id=tenant_id,
		name="Standard Term Loan",
		owner_id="admin",
		product_type="term_loan",
		currency="KES",
		min_amount=10_000,
		max_amount=1_000_000,
		min_term_days=30,
		max_term_days=1_800,
		annual_rate=0.18,
		repayment_frequency="monthly",
	)
	svc.onboard_borrower(
		borrower_id="B001",
		tenant_id=tenant_id,
		customer_reference="CUST001",
		kyc_profile_id="KYC001",
		country="KE",
		income_evidence_id="INC001",
		consent_reference="CONSENT001",
	)
	return svc


@pytest.fixture
def app_id(seeded_svc, tenant_id) -> str:
	"""Submitted application ID."""
	result = seeded_svc.submit_application(
		application_id="APP001",
		tenant_id=tenant_id,
		borrower_id="B001",
		product_id="TERM01",
		requested_amount=100_000,
		purpose="business",
		affordability_reference="AFF001",
		bank_statement_reference="BS001",
		aml_reference="AML001",
		fraud_reference="FRAUD001",
		behavior_evidence_reference="BEH001",
		human_review="UW_ALPHA",
	)
	return result["id"]


@pytest.fixture
def approved_app_id(seeded_svc, app_id) -> str:
	"""Application with approved underwriting decision."""
	seeded_svc.underwriting_decision(
		application_id=app_id,
		decision="approve",
		conditions=[],
		underwriter_id="UW001",
	)
	return app_id


@pytest.fixture
def loan_id(seeded_svc, approved_app_id) -> str:
	"""Disbursed active loan ID."""
	result = seeded_svc.disburse_loan(
		loan_id="LOAN001",
		application_id=approved_app_id,
		bank_account="KE000123456789",
		disbursement_date=(date.today() - timedelta(days=60)).isoformat(),
	)
	return result["loan_id"]


# ---------------------------------------------------------------------------
# Flask test client
# ---------------------------------------------------------------------------

@pytest.fixture
def flask_app():
	"""Minimal Flask app with lending blueprint registered."""
	from api import create_app
	app = create_app()
	app.config["TESTING"] = True
	return app


@pytest.fixture
def client(flask_app):
	"""Flask test client."""
	return flask_app.test_client()
