"""API-level tests for SACCO Guarantor Management using Flask test client."""
from __future__ import annotations

import json
from decimal import Decimal

import pytest

from capabilities.fintech.sacco.gua.api import bp, _svc
from flask import Flask


@pytest.fixture
def app():
	a = Flask(__name__)
	a.register_blueprint(bp)
	a.config["TESTING"] = True
	# seed service context
	_svc._requests.clear()
	_svc._guarantees.clear()
	_svc._audit.clear()
	_svc._gl_entries.clear()
	_svc._notices.clear()
	_svc.seed_member("m1", savings=Decimal("100000"), shares=Decimal("40000"))
	_svc.seed_member("m2", savings=Decimal("60000"), shares=Decimal("30000"))
	_svc.seed_loan("ln1", status="active", dpd=0)
	return a


@pytest.fixture
def client(app):
	return app.test_client()


def h(tenant: str = "t1") -> dict:
	return {"X-Tenant-ID": tenant, "Content-Type": "application/json"}


# ── Health ────────────────────────────────────────────────────────────────────

def test_health(client):
	r = client.get("/api/fintech/sacco/gua/health", headers=h())
	assert r.status_code == 200
	data = r.get_json()
	assert data["status"] == "healthy"


# ── Eligibility ───────────────────────────────────────────────────────────────

def test_eligibility_eligible(client):
	r = client.post(
		"/api/fintech/sacco/gua/eligibility",
		headers=h(),
		data=json.dumps({"member_id": "m1", "amount_to_guarantee": "20000"}),
	)
	assert r.status_code == 200
	assert r.get_json()["eligible"] is True


def test_eligibility_missing_field(client):
	r = client.post(
		"/api/fintech/sacco/gua/eligibility",
		headers=h(),
		data=json.dumps({"member_id": "m1"}),
	)
	assert r.status_code == 422


# ── Request flow ──────────────────────────────────────────────────────────────

def test_request_accept_release(client):
	# Request
	r = client.post(
		"/api/fintech/sacco/gua/requests",
		headers=h(),
		data=json.dumps({
			"loan_id": "ln1",
			"guarantor_member_id": "m1",
			"amount_to_guarantee": "20000",
		}),
	)
	assert r.status_code == 201
	req_id = r.get_json()["id"]

	# Accept
	r = client.post(
		f"/api/fintech/sacco/gua/requests/{req_id}/accept",
		headers=h(),
		data=json.dumps({"guarantor_member_id": "m1", "pin_verified": True}),
	)
	assert r.status_code == 200
	gua_id = r.get_json()["id"]

	# Release
	r = client.post(
		f"/api/fintech/sacco/gua/guarantees/{gua_id}/release",
		headers=h(),
		data=json.dumps({"release_reason": "loan_repaid", "released_by": "system"}),
	)
	assert r.status_code == 200
	assert r.get_json()["status"] == "released"


def test_request_decline(client):
	r = client.post(
		"/api/fintech/sacco/gua/requests",
		headers=h(),
		data=json.dumps({
			"loan_id": "ln1",
			"guarantor_member_id": "m1",
			"amount_to_guarantee": "10000",
		}),
	)
	req_id = r.get_json()["id"]
	r = client.post(
		f"/api/fintech/sacco/gua/requests/{req_id}/decline",
		headers=h(),
		data=json.dumps({"guarantor_member_id": "m1", "decline_reason": "Too risky"}),
	)
	assert r.status_code == 200
	assert r.get_json()["status"] == "declined"


def test_request_not_found(client):
	r = client.get("/api/fintech/sacco/gua/requests/nonexistent", headers=h())
	assert r.status_code == 404


# ── Call guarantee ────────────────────────────────────────────────────────────

def test_call_guarantee(client):
	r = client.post(
		"/api/fintech/sacco/gua/requests",
		headers=h(),
		data=json.dumps({
			"loan_id": "ln1",
			"guarantor_member_id": "m1",
			"amount_to_guarantee": "30000",
		}),
	)
	req_id = r.get_json()["id"]
	r = client.post(
		f"/api/fintech/sacco/gua/requests/{req_id}/accept",
		headers=h(),
		data=json.dumps({"guarantor_member_id": "m1", "pin_verified": True}),
	)
	gua_id = r.get_json()["id"]

	r = client.post(
		f"/api/fintech/sacco/gua/guarantees/{gua_id}/call",
		headers=h(),
		data=json.dumps({"amount_called": "10000", "reason": "Default"}),
	)
	assert r.status_code == 200
	data = r.get_json()
	assert data["guarantee"]["status"] == "called"
	assert data["gl_entry"]["debit_account"] == "Guarantor Savings"


# ── Metrics ───────────────────────────────────────────────────────────────────

def test_metrics(client):
	r = client.get("/api/fintech/sacco/gua/metrics", headers=h())
	assert r.status_code == 200
	data = r.get_json()
	assert "total_active_guarantees" in data


# ── Exposure ──────────────────────────────────────────────────────────────────

def test_exposure(client):
	r = client.get("/api/fintech/sacco/gua/exposure/m1", headers=h())
	assert r.status_code == 200
	assert "total_guaranteed" in r.get_json()


# ── Audit ─────────────────────────────────────────────────────────────────────

def test_audit_events(client):
	r = client.get("/api/fintech/sacco/gua/audit", headers=h())
	assert r.status_code == 200
	assert "items" in r.get_json()
