"""
Flask Blueprint API tests for APG Digital Lending.

Run: cd capabilities/fintech/lending && python -m pytest tests/test_api.py -vxs
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pytest
from datetime import date, timedelta


@pytest.fixture
def app():
	from api import create_app, _SERVICE
	# Reset service state for test isolation
	from service import LendingService
	import api as api_module
	api_module._SERVICE = LendingService()
	flask_app = create_app()
	flask_app.config["TESTING"] = True
	return flask_app


@pytest.fixture
def client(app):
	return app.test_client()


def _json(response):
	return json.loads(response.data)


def _seed(client):
	"""Register product + borrower, return IDs."""
	# Register product via direct service access
	import api as api_module
	svc = api_module._SERVICE
	svc.register_product(
		product_id="TERM01", tenant_id="t1", name="Term Loan", owner_id="admin",
		product_type="term_loan", currency="KES",
		min_amount=5_000, max_amount=1_000_000,
		min_term_days=30, max_term_days=1_800,
		annual_rate=0.18, repayment_frequency="monthly",
	)
	svc.onboard_borrower(
		borrower_id="B001", tenant_id="t1", customer_reference="CUST001",
		kyc_profile_id="KYC001", country="KE",
		income_evidence_id="INC001", consent_reference="CONSENT001",
	)
	return svc


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health(client):
	r = client.get("/api/v1/lending/health")
	assert r.status_code == 200
	data = _json(r)
	assert data["data"]["capability"] == "fintech_lending"


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------

def test_list_products_empty(client):
	r = client.get("/api/v1/lending/products")
	assert r.status_code == 200
	data = _json(r)
	assert data["data"]["items"] == []
	assert data["data"]["total"] == 0


def test_list_products_after_seed(client):
	_seed(client)
	r = client.get("/api/v1/lending/products")
	assert r.status_code == 200
	data = _json(r)
	assert data["data"]["total"] == 1
	assert data["data"]["items"][0]["id"] == "TERM01"


def test_get_product(client):
	_seed(client)
	r = client.get("/api/v1/lending/products/TERM01")
	assert r.status_code == 200
	data = _json(r)
	assert data["data"]["name"] == "Term Loan"


def test_get_product_not_found(client):
	r = client.get("/api/v1/lending/products/NONEXISTENT")
	assert r.status_code == 404


def test_product_performance(client):
	_seed(client)
	r = client.get("/api/v1/lending/products/TERM01/performance")
	assert r.status_code == 200


# ---------------------------------------------------------------------------
# Borrower onboarding
# ---------------------------------------------------------------------------

def test_onboard_borrower(client):
	r = client.post(
		"/api/v1/lending/borrowers",
		json={
			"tenant_id": "t1",
			"customer_reference": "CUST_NEW",
			"kyc_profile_id": "KYC_NEW",
			"country": "KE",
			"income_evidence_id": "INC_NEW",
			"consent_reference": "CON_NEW",
		},
	)
	assert r.status_code == 201
	data = _json(r)
	assert "id" in data["data"]


# ---------------------------------------------------------------------------
# Applications
# ---------------------------------------------------------------------------

def test_list_applications_empty(client):
	r = client.get("/api/v1/lending/applications", headers={"X-Tenant-Id": "t1"})
	assert r.status_code == 200
	data = _json(r)
	assert data["data"]["total"] == 0


def test_submit_application(client):
	_seed(client)
	r = client.post(
		"/api/v1/lending/applications",
		json={
			"tenant_id": "t1",
			"created_by": "agent",
			"borrower_id": "B001",
			"product_id": "TERM01",
			"requested_amount": 100_000,
			"requested_tenor_months": 12,
			"purpose": "working_capital",
			"income_source": "employed",
			"monthly_income": 80_000,
			"kyc_ref": "KYC001",
			"aml_ref": "AML001",
			"fraud_ref": "FRAUD001",
			"bank_statement_ref": "BS001",
		},
	)
	assert r.status_code == 201
	data = _json(r)
	assert data["data"]["status"] == "submitted"
	return data["data"]["id"]


def test_get_application_not_found(client):
	r = client.get("/api/v1/lending/applications/NONEXISTENT")
	assert r.status_code == 404


def test_application_analytics(client):
	r = client.get("/api/v1/lending/applications/analytics")
	assert r.status_code == 200
	data = _json(r)
	assert "total_applications" in data["data"]


# ---------------------------------------------------------------------------
# Credit assessment
# ---------------------------------------------------------------------------

def test_credit_score_missing_customer(client):
	r = client.post("/api/v1/lending/credit/score", json={})
	assert r.status_code == 400


def test_credit_score(client):
	_seed(client)
	r = client.post("/api/v1/lending/credit/score", json={"customer_id": "CUST001"})
	assert r.status_code == 200
	data = _json(r)
	assert 300 <= data["data"]["score"] <= 850


def test_bureau_check(client):
	_seed(client)
	r = client.post("/api/v1/lending/credit/bureau-check", json={
		"customer_id": "CUST001",
		"id_number": "12345678",
		"country": "KE",
	})
	assert r.status_code == 200


def test_income_verify(client):
	_seed(client)
	r = client.post("/api/v1/lending/credit/income-verify", json={
		"customer_id": "CUST001",
		"income_source": "employed",
		"stated_amount": 80_000,
		"docs": ["payslip.pdf"],
	})
	assert r.status_code == 200
	data = _json(r)
	assert "verified" in data["data"]


def test_dsr_calculation(client):
	_seed(client)
	import api as api_module
	api_module._SERVICE.income_verification("CUST001", "employed", 80_000, ["payslip.pdf"])
	r = client.post("/api/v1/lending/credit/dsr", json={
		"customer_id": "CUST001",
		"amount": 50_000,
		"annual_rate": 0.18,
		"tenor_months": 12,
	})
	assert r.status_code == 200
	data = _json(r)
	assert "dsr" in data["data"]


# ---------------------------------------------------------------------------
# Amortisation calculator
# ---------------------------------------------------------------------------

def test_amortisation_calc(client):
	r = client.post("/api/v1/lending/amortisation", json={
		"principal": 100_000,
		"annual_rate": 0.18,
		"tenor_months": 12,
		"start_date": date.today().isoformat(),
	})
	assert r.status_code == 200
	data = _json(r)
	assert len(data["data"]["installments"]) == 12
	assert data["data"]["principal"] == 100_000


def test_amortisation_invalid(client):
	r = client.post("/api/v1/lending/amortisation", json={
		"principal": -100,
		"annual_rate": 0.18,
		"tenor_months": 12,
		"start_date": date.today().isoformat(),
	})
	assert r.status_code == 400


# ---------------------------------------------------------------------------
# Full loan lifecycle via API
# ---------------------------------------------------------------------------

def _full_lifecycle(client):
	"""Helper: seed → apply → underwrite → disburse → return loan_id."""
	svc = _seed(client)

	# Submit application
	r = client.post("/api/v1/lending/applications", json={
		"tenant_id": "t1", "created_by": "agent",
		"borrower_id": "B001", "product_id": "TERM01",
		"requested_amount": 100_000, "requested_tenor_months": 12,
		"purpose": "working_capital", "income_source": "employed",
		"monthly_income": 80_000, "kyc_ref": "KYC001",
		"aml_ref": "AML001", "fraud_ref": "FRAUD001", "bank_statement_ref": "BS001",
	})
	assert r.status_code == 201
	app_id = _json(r)["data"]["id"]

	# Approve
	r = client.post(f"/api/v1/lending/applications/{app_id}/underwrite", json={
		"decision": "approve", "conditions": [], "underwriter_id": "UW001",
	})
	assert r.status_code == 200

	# Disburse
	r = client.post("/api/v1/lending/loans/disburse", json={
		"application_id": app_id,
		"bank_account": "KE0001234567",
		"disbursement_date": (date.today() - timedelta(days=30)).isoformat(),
	})
	assert r.status_code == 201
	loan_id = _json(r)["data"]["loan_id"]
	return loan_id, svc


def test_full_lifecycle_disburse(client):
	loan_id, _ = _full_lifecycle(client)
	assert loan_id is not None


def test_get_loan(client):
	loan_id, _ = _full_lifecycle(client)
	r = client.get(f"/api/v1/lending/loans/{loan_id}")
	assert r.status_code == 200
	data = _json(r)
	assert data["data"]["loan_id"] == loan_id


def test_loan_statement(client):
	loan_id, _ = _full_lifecycle(client)
	r = client.get(f"/api/v1/lending/loans/{loan_id}/statement")
	assert r.status_code == 200
	data = _json(r)
	assert "repayments" in data["data"]


def test_loan_schedule(client):
	loan_id, _ = _full_lifecycle(client)
	r = client.get(f"/api/v1/lending/loans/{loan_id}/schedule")
	assert r.status_code == 200
	data = _json(r)
	assert len(data["data"]["installments"]) > 0


def test_loan_dpd(client):
	loan_id, _ = _full_lifecycle(client)
	r = client.get(f"/api/v1/lending/loans/{loan_id}/dpd")
	assert r.status_code == 200
	data = _json(r)
	assert "max_dpd" in data["data"]


def test_process_repayment(client):
	loan_id, _ = _full_lifecycle(client)
	r = client.post(f"/api/v1/lending/loans/{loan_id}/repay", json={
		"tenant_id": "t1", "created_by": "agent",
		"amount": 10_000,
		"payment_date": date.today().isoformat(),
		"payment_method": "mobile_money",
		"reference": "MPESA001",
	})
	assert r.status_code == 200
	data = _json(r)
	assert data["data"]["payment_amount"] == 10_000


def test_early_settlement(client):
	loan_id, _ = _full_lifecycle(client)
	r = client.get(
		f"/api/v1/lending/loans/{loan_id}/early-settlement",
		query_string={"settlement_date": (date.today() + timedelta(days=1)).isoformat()},
	)
	assert r.status_code == 200
	data = _json(r)
	assert data["data"]["total_settlement_amount"] > 0


def test_add_fee(client):
	loan_id, _ = _full_lifecycle(client)
	r = client.post(f"/api/v1/lending/loans/{loan_id}/fee", json={
		"fee_type": "late_payment_penalty",
		"amount": 500,
		"reason": "30 DPD",
	})
	assert r.status_code == 200
	data = _json(r)
	assert data["data"]["fee"]["amount"] == 500


def test_demand_notice(client):
	loan_id, _ = _full_lifecycle(client)
	r = client.get(f"/api/v1/lending/loans/{loan_id}/demand-notice", query_string={"level": 1})
	assert r.status_code == 200
	data = _json(r)
	assert data["data"]["level"] == 1


def test_assign_collector(client):
	loan_id, _ = _full_lifecycle(client)
	r = client.post(f"/api/v1/lending/loans/{loan_id}/assign-collector", json={
		"collector_id": "COLL001",
	})
	assert r.status_code == 200


def test_close_loan_cancelled(client):
	loan_id, _ = _full_lifecycle(client)
	r = client.post(f"/api/v1/lending/loans/{loan_id}/close", json={"reason": "cancelled"})
	assert r.status_code == 200
	data = _json(r)
	assert data["data"]["status"] == "closed"


# ---------------------------------------------------------------------------
# Portfolio reports
# ---------------------------------------------------------------------------

def test_portfolio_report(client):
	r = client.get("/api/v1/lending/reports/portfolio")
	assert r.status_code == 200
	data = _json(r)
	assert "total_book" in data["data"]


def test_delinquency_report(client):
	r = client.get("/api/v1/lending/reports/delinquency")
	assert r.status_code == 200
	data = _json(r)
	assert "npl_ratio" in data["data"]


def test_ifrs9_report(client):
	r = client.get("/api/v1/lending/reports/ifrs9")
	assert r.status_code == 200
	data = _json(r)
	assert "total_ecl" in data["data"]


def test_vintage_report(client):
	r = client.get("/api/v1/lending/reports/vintage")
	assert r.status_code == 200
	data = _json(r)
	assert "cohorts" in data["data"]


def test_concentration_report(client):
	r = client.get("/api/v1/lending/reports/concentration")
	assert r.status_code == 200


def test_stress_test_report(client):
	r = client.post("/api/v1/lending/reports/stress-test", json={
		"scenarios": [
			{"name": "mild", "additional_default_rate": 0.05, "lgd": 0.40},
		],
	})
	assert r.status_code == 200
	data = _json(r)
	assert len(data["data"]["scenarios"]) == 1


def test_dashboard(client):
	r = client.get("/api/v1/lending/dashboard", headers={"X-Tenant-Id": "t1"})
	assert r.status_code == 200
	data = _json(r)
	assert "portfolio" in data["data"]


def test_collection_performance(client):
	r = client.get("/api/v1/lending/reports/collection-performance")
	assert r.status_code == 200


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------

def test_pagination(client):
	_seed(client)
	import api as api_module
	svc = api_module._SERVICE
	# Create 25 applications to test pagination
	for i in range(3):
		svc.submit_application(
			application_id=f"APP{i:03d}", tenant_id="t1",
			borrower_id="B001", product_id="TERM01",
			requested_amount=50_000, purpose="working_capital",
			affordability_reference="AFF",
			bank_statement_reference="BS",
			aml_reference="AML", fraud_reference="FRAUD",
			behavior_evidence_reference="BEH", human_review="UW",
		)
	r = client.get("/api/v1/lending/applications?page=1&page_size=2")
	assert r.status_code == 200
	data = _json(r)
	assert data["data"]["page"] == 1
	assert data["data"]["page_size"] == 2
	assert data["data"]["total"] == 3
	assert len(data["data"]["items"]) == 2
