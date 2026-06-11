"""API integration tests for SACCO GL using Flask test client."""
from __future__ import annotations

import json
from datetime import date

import pytest

from flask import Flask
from capabilities.fintech.sacco.gl.api import bp, _svc


@pytest.fixture()
def app():
	a = Flask(__name__)
	a.register_blueprint(bp)
	return a


@pytest.fixture()
def client(app):
	return app.test_client()


@pytest.fixture(autouse=True)
def fresh_svc():
	"""Reset service state between tests."""
	_svc._accounts.clear()
	_svc._journals.clear()
	_svc._periods.clear()
	_svc._subsidiary.clear()
	yield


HEADERS = {"X-Tenant-ID": "api_test", "Content-Type": "application/json"}
TODAY = date.today().isoformat()


def _init(client):
	return client.post("/api/fintech/sacco/gl/coa/init", headers=HEADERS)


def test_health(client):
	r = client.get("/api/fintech/sacco/gl/health")
	assert r.status_code == 200
	data = r.get_json()
	assert data["status"] == "healthy"


def test_init_coa(client):
	r = _init(client)
	assert r.status_code == 200
	data = r.get_json()
	assert data["total_accounts"] == 30
	assert "1001" in data["created"]


def test_init_coa_idempotent(client):
	_init(client)
	r = _init(client)
	assert r.status_code == 200
	assert r.get_json()["created"] == []


def test_post_deposit(client):
	_init(client)
	body = {"member_id": "M001", "account_type": "FOSA", "amount": "5000", "channel": "cash"}
	r = client.post("/api/fintech/sacco/gl/deposits", json=body, headers=HEADERS)
	assert r.status_code == 201
	assert r.get_json()["transaction_type"] == "member_deposit"


def test_post_deposit_missing_field(client):
	_init(client)
	r = client.post("/api/fintech/sacco/gl/deposits", json={"member_id": "M001"}, headers=HEADERS)
	assert r.status_code == 400


def test_post_disbursement(client):
	_init(client)
	body = {"loan_id": "LN001", "amount": "50000", "loan_type": "BOSA"}
	r = client.post("/api/fintech/sacco/gl/loans/disbursements", json=body, headers=HEADERS)
	assert r.status_code == 201


def test_post_repayment(client):
	_init(client)
	# Disburse first
	client.post("/api/fintech/sacco/gl/loans/disbursements",
		json={"loan_id": "LN001", "amount": "10000", "loan_type": "BOSA", "disbursement_channel": "cash"},
		headers=HEADERS)
	body = {"loan_id": "LN001", "principal": "1000", "interest": "150", "penalty": "0"}
	r = client.post("/api/fintech/sacco/gl/loans/repayments", json=body, headers=HEADERS)
	assert r.status_code == 201


def test_post_interest(client):
	_init(client)
	body = {"account_id": "ACC001", "amount": "500", "period": "2025-01", "account_type": "BOSA"}
	r = client.post("/api/fintech/sacco/gl/interest", json=body, headers=HEADERS)
	assert r.status_code == 201


def test_post_share_purchase(client):
	_init(client)
	body = {"member_id": "M001", "amount": "2000", "channel": "cash"}
	r = client.post("/api/fintech/sacco/gl/shares", json=body, headers=HEADERS)
	assert r.status_code == 201


def test_post_withdrawal(client):
	_init(client)
	client.post("/api/fintech/sacco/gl/deposits",
		json={"member_id": "M001", "account_type": "FOSA", "amount": "5000"},
		headers=HEADERS)
	body = {"member_id": "M001", "amount": "1000", "account_type": "FOSA"}
	r = client.post("/api/fintech/sacco/gl/withdrawals", json=body, headers=HEADERS)
	assert r.status_code == 201


def test_post_provision(client):
	_init(client)
	client.post("/api/fintech/sacco/gl/loans/disbursements",
		json={"loan_id": "LN001", "amount": "5000", "loan_type": "BOSA", "disbursement_channel": "cash"},
		headers=HEADERS)
	body = {"loan_id": "LN001", "provision_amount": "2500"}
	r = client.post("/api/fintech/sacco/gl/provisions", json=body, headers=HEADERS)
	assert r.status_code == 201


def test_post_write_off(client):
	_init(client)
	client.post("/api/fintech/sacco/gl/loans/disbursements",
		json={"loan_id": "LN001", "amount": "5000", "loan_type": "BOSA", "disbursement_channel": "cash"},
		headers=HEADERS)
	client.post("/api/fintech/sacco/gl/provisions",
		json={"loan_id": "LN001", "provision_amount": "5000"},
		headers=HEADERS)
	body = {"loan_id": "LN001", "amount": "5000", "loan_type": "BOSA"}
	r = client.post("/api/fintech/sacco/gl/write-offs", json=body, headers=HEADERS)
	assert r.status_code == 201


def test_account_balance(client):
	_init(client)
	client.post("/api/fintech/sacco/gl/deposits",
		json={"member_id": "M001", "account_type": "FOSA", "amount": "7000"},
		headers=HEADERS)
	r = client.get("/api/fintech/sacco/gl/accounts/1001/balance", headers=HEADERS)
	assert r.status_code == 200
	data = r.get_json()
	assert data["balance"] == "7000"


def test_account_balance_not_found(client):
	_init(client)
	r = client.get("/api/fintech/sacco/gl/accounts/9999/balance", headers=HEADERS)
	assert r.status_code == 404


def test_trial_balance(client):
	_init(client)
	client.post("/api/fintech/sacco/gl/deposits",
		json={"member_id": "M001", "account_type": "FOSA", "amount": "1000"},
		headers=HEADERS)
	r = client.get(f"/api/fintech/sacco/gl/trial-balance?as_of_date={TODAY}", headers=HEADERS)
	assert r.status_code == 200
	data = r.get_json()
	assert data["count"] == 30


def test_balance_sheet(client):
	_init(client)
	r = client.get(f"/api/fintech/sacco/gl/balance-sheet?as_of_date={TODAY}", headers=HEADERS)
	assert r.status_code == 200
	data = r.get_json()
	assert "total_assets" in data
	assert data["is_balanced"] is True


def test_income_statement(client):
	_init(client)
	r = client.get(f"/api/fintech/sacco/gl/income-statement?from_date=2000-01-01&to_date={TODAY}", headers=HEADERS)
	assert r.status_code == 200
	data = r.get_json()
	assert "surplus_deficit" in data


def test_income_statement_missing_params(client):
	_init(client)
	r = client.get("/api/fintech/sacco/gl/income-statement", headers=HEADERS)
	assert r.status_code == 400


def test_validate_double_entry(client):
	_init(client)
	client.post("/api/fintech/sacco/gl/deposits",
		json={"member_id": "M001", "account_type": "FOSA", "amount": "1000"},
		headers=HEADERS)
	r = client.get("/api/fintech/sacco/gl/validate", headers=HEADERS)
	assert r.status_code == 200
	assert r.get_json()["balanced"] is True


def test_gl_summary(client):
	_init(client)
	r = client.get("/api/fintech/sacco/gl/summary", headers=HEADERS)
	assert r.status_code == 200
	assert "total_assets" in r.get_json()


def test_open_period(client):
	_init(client)
	body = {"year": 2025, "month": 3}
	r = client.post("/api/fintech/sacco/gl/periods/open", json=body, headers=HEADERS)
	assert r.status_code == 200


def test_close_period(client):
	_init(client)
	client.post("/api/fintech/sacco/gl/periods/open", json={"year": 2024, "month": 6}, headers=HEADERS)
	r = client.post("/api/fintech/sacco/gl/periods/close",
		json={"year": 2024, "month": 6, "closed_by": "admin"},
		headers=HEADERS)
	assert r.status_code == 200
	assert r.get_json()["status"] == "closed"


def test_period_status(client):
	_init(client)
	client.post("/api/fintech/sacco/gl/periods/open", json={"year": 2025, "month": 1}, headers=HEADERS)
	r = client.get("/api/fintech/sacco/gl/periods/2025/1", headers=HEADERS)
	assert r.status_code == 200


def test_reconciliation(client):
	_init(client)
	client.post("/api/fintech/sacco/gl/deposits",
		json={"member_id": "M001", "account_type": "FOSA", "amount": "5000"},
		headers=HEADERS)
	r = client.get(f"/api/fintech/sacco/gl/reconciliation?as_of_date={TODAY}", headers=HEADERS)
	assert r.status_code == 200
	data = r.get_json()
	assert "reconciled" in data


def test_journal_entries_requires_dates(client):
	_init(client)
	r = client.get("/api/fintech/sacco/gl/journal-entries", headers=HEADERS)
	assert r.status_code == 400


def test_journal_entries_with_dates(client):
	_init(client)
	client.post("/api/fintech/sacco/gl/deposits",
		json={"member_id": "M001", "account_type": "FOSA", "amount": "100"},
		headers=HEADERS)
	r = client.get(f"/api/fintech/sacco/gl/journal-entries?from_date=2000-01-01&to_date={TODAY}",
		headers=HEADERS)
	assert r.status_code == 200
	assert r.get_json()["count"] >= 1
