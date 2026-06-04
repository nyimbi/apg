"""REST API tests for telecom/bil using Flask test client.

Uses importlib.util pattern — consistent with existing test_contract.py.

© 2025 Datacraft. All rights reserved.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
	sys.path.insert(0, str(PACKAGE_DIR))


def _load(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec and spec.loader
	mod = importlib.util.module_from_spec(spec)
	sys.modules[name] = mod
	spec.loader.exec_module(mod)  # type: ignore[union-attr]
	return mod


# Pre-load dependencies in order
_cc = _load("_cc_api_test", PACKAGE_DIR / "capability_contract.py")
_svc_mod = _load("_svc_api_test", PACKAGE_DIR / "service.py")
_api_mod = _load("_api_test", PACKAGE_DIR / "api.py")


@pytest.fixture
def app():
	from flask import Flask
	flask_app = Flask(__name__)
	flask_app.config["TESTING"] = True
	flask_app.register_blueprint(_api_mod.bil_api)
	return flask_app


@pytest.fixture
def client(app):
	return app.test_client()


def hdr(tenant="t-api"):
	return {
		"X-Tenant-ID": tenant,
		"X-Actor-ID": "api-test-actor",
		"Content-Type": "application/json",
	}


def post(client, path, body, tenant="t-api"):
	return client.post(path, data=json.dumps(body), headers=hdr(tenant))


def get(client, path, tenant="t-api"):
	return client.get(path, headers=hdr(tenant))


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health(client):
	resp = get(client, "/api/telecom/bil/health")
	assert resp.status_code == 200
	data = json.loads(resp.data)
	assert data["data"]["capability"] == "telecom_bil"


# ---------------------------------------------------------------------------
# CDR
# ---------------------------------------------------------------------------

def test_create_cdr(client):
	body = {
		"cdr_id": "cdr-api-001",
		"source": "MSC-01",
		"mediation_status": "raw",
		"msisdn": "+254712345678",
		"duration_seconds": 60,
		"recorded_at": "2026-05-10T10:00:00Z",
		"policy_attached": True,
	}
	resp = post(client, "/api/telecom/bil/cdrs", body)
	assert resp.status_code == 201
	assert json.loads(resp.data)["status"] == "ok"


def test_list_cdrs(client):
	resp = get(client, "/api/telecom/bil/cdrs")
	assert resp.status_code == 200
	assert "items" in json.loads(resp.data)["data"]


def test_get_cdr_not_found(client):
	resp = get(client, "/api/telecom/bil/cdrs/nonexistent-cdr")
	assert resp.status_code == 404


def test_rate_voice_cdr(client):
	body = {"subscriber_id": "sub-api-001", "duration_seconds": 120,
	        "call_type": "on_net", "cdr_type": "voice", "currency": "KES"}
	resp = post(client, "/api/telecom/bil/cdrs/cdr-api-001/rate", body)
	assert resp.status_code == 200
	assert "total_charge" in json.loads(resp.data)["data"]


def test_rate_data_cdr(client):
	body = {"subscriber_id": "sub-api-001", "data_volume_bytes": 10 * 1024 * 1024, "cdr_type": "data"}
	resp = post(client, "/api/telecom/bil/cdrs/cdr-data-001/rate", body)
	assert resp.status_code == 200


def test_rate_sms_cdr(client):
	body = {"subscriber_id": "sub-api-001", "sms_count": 3, "sms_type": "off_net", "cdr_type": "sms"}
	resp = post(client, "/api/telecom/bil/cdrs/cdr-sms-001/rate", body)
	assert resp.status_code == 200


def test_rate_unsupported_type(client):
	resp = post(client, "/api/telecom/bil/cdrs/cdr-fax/rate", {"subscriber_id": "s", "cdr_type": "fax"})
	assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Invoice
# ---------------------------------------------------------------------------

def test_create_invoice(client):
	post(client, "/api/telecom/bil/cycles", {
		"cycle_id": "cyc-api-001", "cycle_type": "monthly",
		"cutoff_date": "2026-05-31", "start_date": "2026-05-01", "end_date": "2026-05-31",
	})
	resp = post(client, "/api/telecom/bil/invoices", {
		"invoice_id": "inv-api-001", "customer_id": "cust-api-001",
		"cycle_id": "cyc-api-001", "total_amount": 1500.00,
		"currency": "KES", "due_date": "2026-06-15",
	})
	assert resp.status_code == 201


def test_list_invoices(client):
	assert get(client, "/api/telecom/bil/invoices").status_code == 200


def test_get_invoice_not_found(client):
	assert get(client, "/api/telecom/bil/invoices/inv-nonexistent").status_code == 404


def test_approve_invoice(client):
	post(client, "/api/telecom/bil/invoices", {
		"invoice_id": "inv-approve-001", "customer_id": "cust-001",
		"cycle_id": "cyc-1", "total_amount": 1000.0,
		"currency": "KES", "due_date": "2026-06-15",
	})
	resp = post(client, "/api/telecom/bil/invoices/inv-approve-001/approve",
	            {"approval_reference": "AUTH-001"})
	assert resp.status_code == 200
	assert json.loads(resp.data)["data"]["status"] == "approved"


def test_write_off_invoice(client):
	post(client, "/api/telecom/bil/invoices", {
		"invoice_id": "inv-writeoff-001", "customer_id": "cust-001",
		"cycle_id": "cyc-1", "total_amount": 1000.0,
		"currency": "KES", "due_date": "2026-06-15",
	})
	resp = post(client, "/api/telecom/bil/invoices/inv-writeoff-001/write-off",
	            {"approval_reference": "WRITE-OFF-AUTH-001"})
	assert resp.status_code == 200
	assert json.loads(resp.data)["data"]["status"] == "written_off"


def test_cancel_invoice(client):
	post(client, "/api/telecom/bil/invoices", {
		"invoice_id": "inv-cancel-001", "customer_id": "cust-001",
		"cycle_id": "cyc-1", "total_amount": 500.0,
		"currency": "KES", "due_date": "2026-06-15",
	})
	resp = client.delete("/api/telecom/bil/invoices/inv-cancel-001", headers=hdr())
	assert resp.status_code == 200
	assert json.loads(resp.data)["data"]["status"] == "cancelled"


def test_adjust_invoice(client):
	post(client, "/api/telecom/bil/invoices", {
		"invoice_id": "inv-adj-api-001", "customer_id": "cust-001",
		"cycle_id": "cyc-1", "total_amount": 1000.0,
		"currency": "KES", "due_date": "2026-06-15",
	})
	resp = post(client, "/api/telecom/bil/invoices/inv-adj-api-001/adjust", {
		"adjustment_type": "credit", "amount": "50.00", "reason": "Goodwill credit",
	})
	assert resp.status_code == 200
	assert json.loads(resp.data)["data"]["adjustment_type"] == "credit"


# ---------------------------------------------------------------------------
# Payment
# ---------------------------------------------------------------------------

def test_create_payment(client):
	resp = post(client, "/api/telecom/bil/payments", {
		"payment_id": "pay-api-001", "invoice_id": "inv-api-001",
		"payment_method": "mobile_money", "amount": 750.00,
		"currency": "KES", "reference": "MPESA-12345",
		"paid_at": "2026-06-01T10:00:00Z",
	})
	assert resp.status_code == 201


def test_list_payments(client):
	assert get(client, "/api/telecom/bil/payments").status_code == 200


def test_process_payment(client):
	resp = post(client, "/api/telecom/bil/payments/process", {
		"account_id": "acc-proc-001", "amount": "1000.00",
		"payment_method": "mobile_money", "reference": "MPESA-PROC-001",
	})
	assert resp.status_code == 200
	assert json.loads(resp.data)["data"]["status"] == "received"


# ---------------------------------------------------------------------------
# Dispute
# ---------------------------------------------------------------------------

def test_create_dispute(client):
	resp = post(client, "/api/telecom/bil/disputes", {
		"account_id": "acc-disp-api-001", "invoice_id": "inv-api-001",
		"disputed_amount": "200.00", "reason": "Charged for calls I did not make",
	})
	assert resp.status_code == 201
	assert json.loads(resp.data)["data"]["status"] == "open"


def test_list_disputes(client):
	assert get(client, "/api/telecom/bil/disputes").status_code == 200


def test_investigate_dispute(client):
	cr = post(client, "/api/telecom/bil/disputes", {
		"account_id": "acc-inv-001", "invoice_id": "inv-001",
		"disputed_amount": "100.00", "reason": "Test dispute",
	})
	dispute_id = json.loads(cr.data)["data"]["dispute_id"]
	resp = post(client, f"/api/telecom/bil/disputes/{dispute_id}/investigate",
	            {"cdr_analysis": {"finding": "rate was incorrect"}})
	assert resp.status_code == 200
	assert json.loads(resp.data)["data"]["status"] == "under_review"


def test_resolve_dispute(client):
	cr = post(client, "/api/telecom/bil/disputes", {
		"account_id": "acc-res-001", "invoice_id": "inv-001",
		"disputed_amount": "150.00", "reason": "Wrong tariff applied",
	})
	dispute_id = json.loads(cr.data)["data"]["dispute_id"]
	post(client, f"/api/telecom/bil/disputes/{dispute_id}/investigate", {})
	resp = post(client, f"/api/telecom/bil/disputes/{dispute_id}/resolve",
	            {"resolution": "upheld", "credit_amount": "150.00"})
	assert resp.status_code == 200
	assert json.loads(resp.data)["data"]["status"] == "resolved_upheld"


# ---------------------------------------------------------------------------
# Discount
# ---------------------------------------------------------------------------

def test_create_discount(client):
	resp = post(client, "/api/telecom/bil/discounts", {
		"discount_id": "disc-api-001", "customer_id": "cust-disc-001",
		"discount_type": "loyalty", "discount_pct": 15.0,
		"approval_reference": "AUTH-DISC-001",
		"valid_from": "2026-01-01", "valid_to": "2026-12-31",
	})
	assert resp.status_code == 201


def test_discount_over_50_pct_rejected(client):
	resp = post(client, "/api/telecom/bil/discounts", {
		"discount_id": "disc-bad-001", "customer_id": "cust-001",
		"discount_type": "loyalty", "discount_pct": 55.0,
		"approval_reference": "AUTH-001",
		"valid_from": "2026-01-01", "valid_to": "2026-12-31",
	})
	assert resp.status_code in {400, 422}


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def test_report_revenue(client):
	resp = client.get("/api/telecom/bil/reports/revenue?start=2026-05-01&end=2026-05-31",
	                  headers=hdr())
	assert resp.status_code == 200
	assert "total_revenue" in json.loads(resp.data)["data"]


def test_report_arpu(client):
	assert client.get("/api/telecom/bil/reports/arpu?start=2026-05-01&end=2026-05-31",
	                  headers=hdr()).status_code == 200


def test_report_disputes(client):
	assert client.get("/api/telecom/bil/reports/disputes?start=2026-05-01&end=2026-05-31",
	                  headers=hdr()).status_code == 200


def test_report_leakage(client):
	resp = client.get("/api/telecom/bil/reports/leakage?start=2026-05-01&end=2026-05-31",
	                  headers=hdr())
	assert resp.status_code == 200
	assert "leakage_pct" in json.loads(resp.data)["data"]


def test_report_churn(client):
	assert client.get("/api/telecom/bil/reports/churn?start=2026-05-01&end=2026-05-31",
	                  headers=hdr()).status_code == 200


def test_report_dashboard(client):
	resp = get(client, "/api/telecom/bil/reports/dashboard")
	assert resp.status_code == 200
	assert "invoice_count" in json.loads(resp.data)["data"]


# ---------------------------------------------------------------------------
# Tax
# ---------------------------------------------------------------------------

def test_calculate_tax_ke(client):
	resp = post(client, "/api/telecom/bil/tax/calculate", {"amount": "1000.00", "jurisdiction": "KE"})
	assert resp.status_code == 200
	data = json.loads(resp.data)["data"]
	assert data["jurisdiction"] == "KE"
	assert "total_with_tax" in data


def test_calculate_tax_ug(client):
	resp = post(client, "/api/telecom/bil/tax/calculate", {"amount": "1000.00", "jurisdiction": "UG"})
	assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Real-time charging
# ---------------------------------------------------------------------------

def test_realtime_charge_check(client):
	resp = post(client, "/api/telecom/bil/realtime/charge", {
		"subscriber_id": "sub-rt-001", "service_type": "voice", "amount": "50.00",
	})
	assert resp.status_code == 200
	assert "sufficient" in json.loads(resp.data)["data"]


# ---------------------------------------------------------------------------
# Dunning
# ---------------------------------------------------------------------------

def test_dunning_workflow_via_api(client):
	resp = post(client, "/api/telecom/bil/dunning/workflow",
	            {"account_id": "acc-dun-api-001", "dpd_days": 10})
	assert resp.status_code == 200
	assert json.loads(resp.data)["data"]["dunning_step"] == "reminder_2"


def test_suspend_account(client):
	resp = post(client, "/api/telecom/bil/accounts/acc-sus-api/suspend", {"reason": "non_payment"})
	assert resp.status_code == 200
	assert json.loads(resp.data)["data"]["action"] == "suspended"


# ---------------------------------------------------------------------------
# Interconnect
# ---------------------------------------------------------------------------

def test_interconnect_reconcile(client):
	resp = post(client, "/api/telecom/bil/settlements/reconcile", {
		"carrier": "AIRTEL-KE", "period_start": "2026-05-01", "period_end": "2026-05-31",
	})
	assert resp.status_code == 200
	assert "status" in json.loads(resp.data)["data"]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_missing_required_field_returns_error(client):
	resp = post(client, "/api/telecom/bil/payments", {"invoice_id": "inv-1"})
	assert resp.status_code in {400, 422, 500}
