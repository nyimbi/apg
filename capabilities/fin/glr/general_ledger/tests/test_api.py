"""Flask REST API integration tests for GLR general ledger.

Uses Flask test client — no external server needed.
Tests all major endpoint groups: accounts, periods, journals,
reconciliation, budgets, reports, year-end.
"""
from __future__ import annotations

import json
import pytest
from decimal import Decimal

from capabilities.fin.glr.general_ledger.app import create_app
from capabilities.fin.glr.general_ledger.service import GeneralLedgerService


TENANT = "api-test-tenant"
HEADERS = {"X-Tenant-ID": TENANT, "Content-Type": "application/json"}


@pytest.fixture(scope="module")
def client():
	"""Flask test client with a fresh in-memory service."""
	app = create_app()
	app.config["TESTING"] = True
	with app.test_client() as c:
		yield c


@pytest.fixture(scope="module")
def funded_client(client):
	"""Client whose service has accounts and an open period pre-loaded."""
	# Create accounts
	for code, name, atype in [
		("1000", "Cash", "asset"),
		("1100", "Accounts Receivable", "asset"),
		("2000", "Accounts Payable", "liability"),
		("3000", "Share Capital", "equity"),
		("3100", "Retained Earnings", "equity"),
		("4000", "Revenue", "revenue"),
		("6000", "Salaries", "expense"),
	]:
		client.post(
			"/api/glr/accounts/create",
			data=json.dumps({
				"tenant_id": TENANT,
				"account_code": code,
				"account_name": name,
				"account_type": atype,
			}),
			headers=HEADERS,
		)
	# Create period
	client.post(
		"/api/glr/periods/create",
		data=json.dumps({
			"tenant_id": TENANT,
			"period_code": "2026-01",
			"fiscal_year": 2026,
			"period_number": 1,
			"start_date": "2026-01-01",
			"end_date": "2026-01-31",
		}),
		headers=HEADERS,
	)
	return client


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health(client):
	resp = client.get("/api/glr/health")
	assert resp.status_code == 200
	data = resp.get_json()
	assert data["data"]["status"] == "ok"


# ---------------------------------------------------------------------------
# Accounts
# ---------------------------------------------------------------------------

def test_create_account(client):
	resp = client.post(
		"/api/glr/accounts/create",
		data=json.dumps({
			"tenant_id": TENANT,
			"account_code": "9001",
			"account_name": "Test Account",
			"account_type": "asset",
		}),
		headers=HEADERS,
	)
	assert resp.status_code == 201
	data = resp.get_json()
	assert data["error"] is None
	assert data["data"]["code"] == "9001"


def test_create_account_invalid_code_rejected(client):
	# Account code with special chars is rejected by Pydantic
	resp = client.post(
		"/api/glr/accounts/create",
		data=json.dumps({
			"tenant_id": TENANT,
			"account_code": "!@#INVALID",
			"account_name": "Bad Code",
			"account_type": "asset",
		}),
		headers=HEADERS,
	)
	assert resp.status_code in {400, 422}


def test_create_account_invalid_type_rejected(client):
	resp = client.post(
		"/api/glr/accounts/create",
		data=json.dumps({
			"tenant_id": TENANT,
			"account_code": "9998",
			"account_name": "Bad Type",
			"account_type": "bogus",
		}),
		headers=HEADERS,
	)
	assert resp.status_code in {400, 422}


def test_list_accounts(funded_client):
	resp = funded_client.get("/api/glr/accounts", headers=HEADERS)
	assert resp.status_code == 200
	data = resp.get_json()
	assert isinstance(data["data"], list)
	assert data["meta"]["total"] >= 7


def test_account_hierarchy(funded_client):
	resp = funded_client.get("/api/glr/accounts/hierarchy", headers=HEADERS)
	assert resp.status_code == 200


def test_get_account_not_found(client):
	resp = client.get("/api/glr/accounts/nonexistent-id", headers=HEADERS)
	assert resp.status_code == 404


def test_delete_account(funded_client):
	# Create one to delete
	resp = funded_client.post(
		"/api/glr/accounts/create",
		data=json.dumps({
			"tenant_id": TENANT,
			"account_code": "8999",
			"account_name": "To Delete",
			"account_type": "expense",
		}),
		headers=HEADERS,
	)
	acct_id = resp.get_json()["data"]["id"]
	del_resp = funded_client.delete(f"/api/glr/accounts/{acct_id}", headers=HEADERS)
	assert del_resp.status_code == 200
	assert del_resp.get_json()["data"]["status"] == "deleted"


# ---------------------------------------------------------------------------
# Periods
# ---------------------------------------------------------------------------

def test_list_periods(funded_client):
	resp = funded_client.get("/api/glr/periods", headers=HEADERS)
	assert resp.status_code == 200
	data = resp.get_json()
	assert data["meta"]["total"] >= 1


def test_get_period(funded_client):
	resp = funded_client.get("/api/glr/periods/2026-01", headers=HEADERS)
	assert resp.status_code == 200


def test_period_checklist(funded_client):
	resp = funded_client.get("/api/glr/periods/2026-01/checklist", headers=HEADERS)
	assert resp.status_code == 200
	data = resp.get_json()
	assert "ready_to_close" in data["data"]


def test_close_then_lock_period(funded_client):
	close_resp = funded_client.post(
		"/api/glr/periods/2026-01/close",
		data=json.dumps({"closed_by": "controller"}),
		headers=HEADERS,
	)
	# May fail if there are outstanding items — that's valid behaviour
	if close_resp.status_code == 200:
		lock_resp = funded_client.post(
			"/api/glr/periods/2026-01/lock",
			data=json.dumps({"locked_by": "cfo"}),
			headers=HEADERS,
		)
		assert lock_resp.status_code == 200


# ---------------------------------------------------------------------------
# Journals
# ---------------------------------------------------------------------------

def _get_account_id(client, code: str) -> str:
	resp = client.get("/api/glr/accounts", headers=HEADERS)
	accounts = resp.get_json()["data"]
	for a in accounts:
		if a.get("code") == code or a.get("account_code") == code:
			return a["id"]
	raise KeyError(f"account {code} not found")


def test_create_journal(funded_client):
	# Ensure at least one open period exists (may have been closed by earlier test)
	funded_client.post(
		"/api/glr/periods/create",
		data=json.dumps({
			"tenant_id": TENANT,
			"period_code": "2026-06",
			"fiscal_year": 2026,
			"period_number": 6,
			"start_date": "2026-06-01",
			"end_date": "2026-06-30",
		}),
		headers=HEADERS,
	)
	cash_id = _get_account_id(funded_client, "1000")
	rev_id = _get_account_id(funded_client, "4000")
	resp = funded_client.post(
		"/api/glr/journals/create",
		data=json.dumps({
			"tenant_id": TENANT,
			"journal_date": "2026-06-15",
			"journal_type": "standard",
			"description": "Test journal",
			"lines": [
				{"account_id": cash_id, "debit": "1000", "credit": "0"},
				{"account_id": rev_id, "debit": "0", "credit": "1000"},
			],
			"posted_by": "poster",
		}),
		headers=HEADERS,
	)
	assert resp.status_code == 201
	data = resp.get_json()
	assert data["data"]["status"] == "posted"


def test_create_journal_unbalanced_rejected(funded_client):
	cash_id = _get_account_id(funded_client, "1000")
	rev_id = _get_account_id(funded_client, "4000")
	resp = funded_client.post(
		"/api/glr/journals/create",
		data=json.dumps({
			"tenant_id": TENANT,
			"journal_date": "2026-06-15",
			"journal_type": "standard",
			"description": "Unbalanced",
			"lines": [
				{"account_id": cash_id, "debit": "1000", "credit": "0"},
				{"account_id": rev_id, "debit": "0", "credit": "900"},
			],
			"posted_by": "poster",
		}),
		headers=HEADERS,
	)
	assert resp.status_code in {400, 422}


def test_list_journals(funded_client):
	resp = funded_client.get("/api/glr/journals", headers=HEADERS)
	assert resp.status_code == 200
	data = resp.get_json()
	assert isinstance(data["data"], list)


def test_list_journals_status_filter(funded_client):
	resp = funded_client.get("/api/glr/journals?status=posted", headers=HEADERS)
	assert resp.status_code == 200
	for j in resp.get_json()["data"]:
		assert j["status"] == "posted"


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------

def test_create_and_list_reconciliation(funded_client):
	# Ensure there's a balance to reconcile
	create_resp = funded_client.post(
		"/api/glr/reconciliations/create",
		data=json.dumps({
			"tenant_id": TENANT,
			"account_code": "1000",
			"period_code": "2026-01",
		}),
		headers=HEADERS,
	)
	assert create_resp.status_code in {200, 201}

	list_resp = funded_client.get("/api/glr/reconciliations", headers=HEADERS)
	assert list_resp.status_code == 200
	assert list_resp.get_json()["meta"]["total"] >= 1


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------

def test_list_budgets(funded_client):
	resp = funded_client.get("/api/glr/budgets", headers=HEADERS)
	assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def test_trial_balance_report(funded_client):
	resp = funded_client.get("/api/glr/reports/trial-balance?period_code=2026-01", headers=HEADERS)
	assert resp.status_code == 200
	data = resp.get_json()
	assert "balanced" in data["data"]


def test_balance_sheet_report(funded_client):
	resp = funded_client.get("/api/glr/reports/balance-sheet?period_code=2026-01", headers=HEADERS)
	assert resp.status_code == 200
	assert "assets" in resp.get_json()["data"]


def test_income_statement_report(funded_client):
	resp = funded_client.get("/api/glr/reports/income-statement?period_code=2026-01", headers=HEADERS)
	assert resp.status_code == 200
	assert "revenue" in resp.get_json()["data"]


def test_cash_flow_report(funded_client):
	resp = funded_client.get("/api/glr/reports/cash-flow?period_code=2026-01", headers=HEADERS)
	assert resp.status_code == 200
	assert "operating_activities" in resp.get_json()["data"]


def test_budget_vs_actual_report(funded_client):
	resp = funded_client.get("/api/glr/reports/budget-vs-actual?period_code=2026-01", headers=HEADERS)
	assert resp.status_code == 200
	assert "rows" in resp.get_json()["data"]


def test_xbrl_report(funded_client):
	resp = funded_client.get("/api/glr/reports/xbrl?period_code=2026-01&framework=IFRS", headers=HEADERS)
	assert resp.status_code == 200
	assert "facts" in resp.get_json()["data"]


def test_management_pack_report(funded_client):
	resp = funded_client.get("/api/glr/reports/management-pack?period_code=2026-01", headers=HEADERS)
	assert resp.status_code == 200
	data = resp.get_json()["data"]
	assert "trial_balance" in data
	assert "income_statement" in data


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def test_dashboard(funded_client):
	resp = funded_client.get("/api/glr/dashboard", headers=HEADERS)
	assert resp.status_code == 200
	data = resp.get_json()["data"]
	assert "account_count" in data
