"""REST API tests for APG Tax Administration Flask Blueprint.

Uses Flask test client — no mocks, real service state.
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]
if str(PKG) not in sys.path:
	sys.path.insert(0, str(PKG))


@pytest.fixture
def app():
	from flask import Flask
	from api import tax_bp, _svc

	application = Flask(__name__)
	application.config["TESTING"] = True
	application.register_blueprint(tax_bp)
	# reset service state for each test
	from service import TaxAdministrationService
	import api as _api_mod
	_api_mod._svc = TaxAdministrationService()
	return application


@pytest.fixture
def client(app):
	return app.test_client()


def _headers(tenant: str = "t1") -> dict:
	return {"X-Tenant-ID": tenant, "Content-Type": "application/json"}


def _post(client, url, body, tenant="t1"):
	return client.post(url, data=json.dumps(body), headers=_headers(tenant))


def _get(client, url, tenant="t1", **params):
	qs = "&".join(f"{k}={v}" for k, v in params.items())
	full = f"{url}?{qs}" if qs else url
	return client.get(full, headers=_headers(tenant))


def _register(client, tenant="t1", name="Alice Wanjiku", id_number="ID-001"):
	return _post(client, "/api/v1/tax/taxpayers", {
		"taxpayer_name": name,
		"taxpayer_type": "individual",
		"national_id": id_number,
		"email": "alice@example.com",
		"tax_type": "income_tax",
		"evidence_reference": "reg_ev",
	}, tenant=tenant)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health(client):
	r = client.get("/api/v1/tax/health")
	assert r.status_code == 200
	data = json.loads(r.data)
	assert data["data"]["status"] == "ok"
	assert data["data"]["capability"] == "government_tax"


# ---------------------------------------------------------------------------
# Taxpayers
# ---------------------------------------------------------------------------

def test_register_taxpayer(client):
	r = _register(client)
	assert r.status_code == 201
	data = json.loads(r.data)
	assert data["data"]["taxpayer_name"] == "Alice Wanjiku"
	assert data["data"]["status"] == "pending"


def test_register_taxpayer_missing_name(client):
	r = _post(client, "/api/v1/tax/taxpayers", {
		"taxpayer_type": "individual",
		"evidence_reference": "ev",
	})
	assert r.status_code in (400, 422, 500)


def test_list_taxpayers_empty(client):
	r = _get(client, "/api/v1/tax/taxpayers")
	assert r.status_code == 200
	data = json.loads(r.data)
	assert data["data"] == []
	assert data["meta"]["total"] == 0


def test_list_taxpayers_with_data(client):
	_register(client)
	r = _get(client, "/api/v1/tax/taxpayers")
	data = json.loads(r.data)
	assert data["meta"]["total"] >= 1


def test_get_taxpayer_by_pin(client):
	reg = json.loads(_register(client).data)
	tin = reg["data"]["tax_pin"]
	r = _get(client, f"/api/v1/tax/taxpayers/{tin}")
	assert r.status_code == 200
	data = json.loads(r.data)
	assert data["data"]["tax_pin"] == tin


def test_get_taxpayer_not_found(client):
	r = _get(client, "/api/v1/tax/taxpayers/UNKNOWN-PIN")
	assert r.status_code == 404


def test_verify_tin(client):
	reg = json.loads(_register(client).data)
	tin = reg["data"]["tax_pin"]
	r = _get(client, f"/api/v1/tax/taxpayers/{tin}/verify")
	data = json.loads(r.data)
	assert data["data"]["exists"] is True


def test_search_taxpayer(client):
	_register(client, name="Search Target")
	r = _get(client, "/api/v1/tax/taxpayers", q="Search")
	data = json.loads(r.data)
	assert any("Search Target" in str(t) for t in data["data"])


def test_update_taxpayer(client):
	reg = json.loads(_register(client).data)
	tin = reg["data"]["tax_pin"]
	r = client.put(
		f"/api/v1/tax/taxpayers/{tin}",
		data=json.dumps({"email": "updated@example.com"}),
		headers=_headers(),
	)
	assert r.status_code == 200
	data = json.loads(r.data)
	assert data["data"]["email"] == "updated@example.com"


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------

def test_file_return(client):
	reg = json.loads(_register(client).data)
	tin = reg["data"]["tax_pin"]
	r = _post(client, "/api/v1/tax/returns", {
		"tax_pin": tin,
		"tax_type": "income_tax",
		"period": "2024",
		"gross_income": 1200000,
		"allowable_deductions": 200000,
		"taxable_income": 1000000,
		"tax_liability": 300000,
		"tax_credits": 0,
		"tax_paid": 300000,
		"net_tax_payable": 0,
		"evidence_reference": "ret_ev",
	})
	assert r.status_code == 201
	data = json.loads(r.data)
	assert data["data"]["status"] == "filed"


def test_file_nil_return(client):
	reg = json.loads(_register(client).data)
	tin = reg["data"]["tax_pin"]
	r = _post(client, "/api/v1/tax/returns/nil", {
		"tax_pin": tin,
		"tax_type": "vat",
		"period": "2025-01",
	})
	assert r.status_code == 201
	data = json.loads(r.data)
	assert data["data"]["tax_liability"] == "0"


def test_list_returns_empty(client):
	r = _get(client, "/api/v1/tax/returns")
	data = json.loads(r.data)
	assert data["data"] == []


def test_list_returns_filtered_by_tin(client):
	reg = json.loads(_register(client).data)
	tin = reg["data"]["tax_pin"]
	_post(client, "/api/v1/tax/returns", {
		"tax_pin": tin, "tax_type": "vat", "period": "2025-01",
		"gross_income": 100000, "tax_liability": 16000, "tax_paid": 16000,
		"evidence_reference": "ev",
	})
	r = _get(client, "/api/v1/tax/returns", tin=tin)
	data = json.loads(r.data)
	assert data["meta"]["total"] >= 1


def test_validate_return(client):
	reg = json.loads(_register(client).data)
	tin = reg["data"]["tax_pin"]
	ret = json.loads(_post(client, "/api/v1/tax/returns", {
		"tax_pin": tin, "tax_type": "income_tax", "period": "2024",
		"gross_income": 500000, "allowable_deductions": 50000,
		"taxable_income": 450000, "tax_liability": 100000,
		"tax_credits": 0, "tax_paid": 100000, "net_tax_payable": 0,
		"evidence_reference": "ev",
	}).data)
	ret_id = ret["data"]["id"]
	r = _post(client, f"/api/v1/tax/returns/{ret_id}/validate", {})
	data = json.loads(r.data)
	assert data["data"]["status"] == "valid"


# ---------------------------------------------------------------------------
# Assessments
# ---------------------------------------------------------------------------

def test_create_assessment(client):
	reg = json.loads(_register(client).data)
	tin = reg["data"]["tax_pin"]
	r = _post(client, "/api/v1/tax/assessments", {
		"tax_pin": tin,
		"tax_type": "income_tax",
		"period": "2024",
		"assessed_amount": 150000,
		"reason": "underdeclared",
		"assessment_type": "best_judgement",
		"assessor_id": "officer_1",
	})
	assert r.status_code == 201
	data = json.loads(r.data)
	assert data["data"]["assessed_amount"] == "150000.00"
	assert "debt_id" in data["data"]


def test_list_assessments(client):
	r = _get(client, "/api/v1/tax/assessments")
	assert r.status_code == 200


def test_calc_penalty_interest(client):
	reg = json.loads(_register(client).data)
	tin = reg["data"]["tax_pin"]
	assess = json.loads(_post(client, "/api/v1/tax/assessments", {
		"tax_pin": tin, "tax_type": "income_tax", "period": "2024",
		"assessed_amount": 100000, "reason": "late",
		"assessment_type": "self_assessment",
	}).data)
	assess_id = assess["data"]["id"]
	future = (date.today() + timedelta(days=90)).isoformat()
	r = _post(client, f"/api/v1/tax/assessments/{assess_id}/penalty-interest", {
		"payment_date": future,
	})
	assert r.status_code == 200
	data = json.loads(r.data)
	assert "late_filing_penalty" in data["data"]


# ---------------------------------------------------------------------------
# Payments
# ---------------------------------------------------------------------------

def test_create_payment(client):
	reg = json.loads(_register(client).data)
	tin = reg["data"]["tax_pin"]
	r = _post(client, "/api/v1/tax/payments", {
		"tax_pin": tin,
		"amount": 80000,
		"payment_method": "mobile_money",
		"payment_reference": "MPESA-TEST-001",
		"tax_type": "vat",
		"period": "2025-01",
	})
	assert r.status_code == 201
	data = json.loads(r.data)
	assert data["data"]["amount"] == "80000.00"


def test_list_payments(client):
	r = _get(client, "/api/v1/tax/payments")
	assert r.status_code == 200


# ---------------------------------------------------------------------------
# Debts
# ---------------------------------------------------------------------------

def test_list_debts(client):
	r = _get(client, "/api/v1/tax/debts")
	assert r.status_code == 200


def test_issue_demand_notice(client):
	reg = json.loads(_register(client).data)
	tin = reg["data"]["tax_pin"]
	deadline = (date.today() + timedelta(days=30)).isoformat()
	r = _post(client, "/api/v1/tax/debts/demand-notice", {
		"tax_pin": tin,
		"outstanding_amount": 150000,
		"deadline": deadline,
	})
	assert r.status_code == 201
	data = json.loads(r.data)
	assert data["data"]["notice_number"].startswith("DN-")


def test_debt_collection_action(client):
	reg = json.loads(_register(client).data)
	tin = reg["data"]["tax_pin"]
	r = _post(client, "/api/v1/tax/debts/collection-action", {
		"tax_pin": tin,
		"action_type": "payment_plan",
		"officer_id": "collector_1",
	})
	assert r.status_code == 201


# ---------------------------------------------------------------------------
# Audits
# ---------------------------------------------------------------------------

def test_create_audit(client):
	reg = json.loads(_register(client).data)
	tin = reg["data"]["tax_pin"]
	r = _post(client, "/api/v1/tax/audits", {
		"tax_pin": tin,
		"audit_type": "field_audit",
		"period": "2024",
		"auditor_id": "auditor_1",
		"scope_description": "Full field audit",
	})
	assert r.status_code == 201
	data = json.loads(r.data)
	assert data["data"]["audit_type"] == "field_audit"


def test_list_audits(client):
	r = _get(client, "/api/v1/tax/audits")
	assert r.status_code == 200


def test_record_findings(client):
	reg = json.loads(_register(client).data)
	tin = reg["data"]["tax_pin"]
	audit = json.loads(_post(client, "/api/v1/tax/audits", {
		"tax_pin": tin, "audit_type": "desk_audit", "period": "2024",
		"auditor_id": "auditor_1",
	}).data)
	audit_id = audit["data"]["id"]
	r = _post(client, f"/api/v1/tax/audits/{audit_id}/findings", {
		"findings": [{
			"finding_type": "underpayment",
			"description": "VAT underpaid",
			"additional_tax": 25000,
			"penalty_amount": 1250,
			"interest_amount": 500,
			"evidence_reference": "find_ev",
		}],
	})
	assert r.status_code == 200
	data = json.loads(r.data)
	assert data["data"]["status"] == "in_progress"


def test_close_audit(client):
	reg = json.loads(_register(client).data)
	tin = reg["data"]["tax_pin"]
	audit = json.loads(_post(client, "/api/v1/tax/audits", {
		"tax_pin": tin, "audit_type": "compliance_audit", "period": "2023",
		"auditor_id": "auditor_1",
	}).data)
	audit_id = audit["data"]["id"]
	r = _post(client, f"/api/v1/tax/audits/{audit_id}/close", {
		"outcome": "tax_due",
		"final_tax_due": 50000,
		"penalties": 2500,
	})
	assert r.status_code == 200
	data = json.loads(r.data)
	assert data["data"]["status"] == "finalised"


# ---------------------------------------------------------------------------
# Objections
# ---------------------------------------------------------------------------

def test_create_objection(client):
	reg = json.loads(_register(client).data)
	tin = reg["data"]["tax_pin"]
	assess = json.loads(_post(client, "/api/v1/tax/assessments", {
		"tax_pin": tin, "tax_type": "income_tax", "period": "2024",
		"assessed_amount": 50000, "reason": "test",
		"assessment_type": "best_judgement",
	}).data)
	assess_id = assess["data"]["id"]
	today = date.today().isoformat()
	r = _post(client, "/api/v1/tax/objections", {
		"assessment_id": assess_id,
		"grounds": "Double counted expenses",
		"amount_disputed": 20000,
		"tax_pin": tin,
		"filed_date": today,
	})
	assert r.status_code == 201
	data = json.loads(r.data)
	assert data["data"]["status"] == "submitted"


def test_determine_objection(client):
	reg = json.loads(_register(client).data)
	tin = reg["data"]["tax_pin"]
	assess = json.loads(_post(client, "/api/v1/tax/assessments", {
		"tax_pin": tin, "tax_type": "income_tax", "period": "2023",
		"assessed_amount": 50000, "reason": "test",
		"assessment_type": "best_judgement",
	}).data)
	today = date.today().isoformat()
	obj = json.loads(_post(client, "/api/v1/tax/objections", {
		"assessment_id": assess["data"]["id"],
		"grounds": "Valid grounds",
		"amount_disputed": 20000,
		"filed_date": today,
	}).data)
	obj_id = obj["data"]["id"]
	r = _post(client, f"/api/v1/tax/objections/{obj_id}/determine", {
		"decision": "dismissed",
		"revised_amount": 50000,
		"officer_id": "officer_1",
	})
	assert r.status_code == 200
	data = json.loads(r.data)
	assert data["data"]["status"] == "dismissed"


# ---------------------------------------------------------------------------
# Refunds
# ---------------------------------------------------------------------------

def test_create_refund(client):
	reg = json.loads(_register(client).data)
	tin = reg["data"]["tax_pin"]
	# file a return first
	_post(client, "/api/v1/tax/returns", {
		"tax_pin": tin, "tax_type": "vat", "period": "2025-01",
		"gross_income": 500000, "tax_liability": 80000, "tax_paid": 80000,
		"evidence_reference": "ev",
	})
	r = _post(client, "/api/v1/tax/refunds", {
		"tax_pin": tin,
		"tax_type": "vat",
		"period": "2025-01",
		"claimed_amount": 15000,
		"refund_type": "input_vat_credit",
		"bank_account_number": "1234567890",
		"bank_name": "Equity Bank",
	})
	assert r.status_code == 201
	data = json.loads(r.data)
	assert data["data"]["status"] == "claimed"


def test_approve_refund(client):
	reg = json.loads(_register(client).data)
	tin = reg["data"]["tax_pin"]
	_post(client, "/api/v1/tax/returns", {
		"tax_pin": tin, "tax_type": "vat", "period": "2025-02",
		"gross_income": 300000, "tax_liability": 48000, "tax_paid": 48000,
		"evidence_reference": "ev",
	})
	refund = json.loads(_post(client, "/api/v1/tax/refunds", {
		"tax_pin": tin, "tax_type": "vat", "period": "2025-02",
		"claimed_amount": 10000, "refund_type": "overpayment",
	}).data)
	refund_id = refund["data"]["id"]
	# review
	_post(client, f"/api/v1/tax/refunds/{refund_id}/review", {
		"officer_id": "reviewer_1",
	})
	# approve
	r = _post(client, f"/api/v1/tax/refunds/{refund_id}/approve", {
		"approved_by": "manager_1",
		"payment_method": "bank_transfer",
	})
	assert r.status_code == 200
	data = json.loads(r.data)
	assert data["data"]["status"] == "approved"


# ---------------------------------------------------------------------------
# Clearance Certificates
# ---------------------------------------------------------------------------

def test_request_clearance_no_debt(client):
	reg = json.loads(_register(client).data)
	tin = reg["data"]["tax_pin"]
	r = _post(client, "/api/v1/tax/clearances", {
		"tax_pin": tin,
		"purpose": "government_tender",
		"validity_days": 180,
	})
	assert r.status_code == 201
	data = json.loads(r.data)
	assert data["data"]["status"] == "issued"
	assert data["data"]["certificate_number"].startswith("TCC-")


def test_verify_clearance(client):
	reg = json.loads(_register(client).data)
	tin = reg["data"]["tax_pin"]
	cert = json.loads(_post(client, "/api/v1/tax/clearances", {
		"tax_pin": tin,
		"purpose": "business_license",
	}).data)
	cert_number = cert["data"]["certificate_number"]
	r = _get(client, f"/api/v1/tax/clearances/verify/{cert_number}")
	assert r.status_code == 200
	data = json.loads(r.data)
	assert data["data"]["valid"] is True


def test_clearance_blocked_by_debt(client):
	reg = json.loads(_register(client).data)
	tin = reg["data"]["tax_pin"]
	# create outstanding debt
	_post(client, "/api/v1/tax/assessments", {
		"tax_pin": tin, "tax_type": "income_tax", "period": "2024",
		"assessed_amount": 100000, "reason": "unpaid",
		"assessment_type": "best_judgement",
	})
	r = _post(client, "/api/v1/tax/clearances", {
		"tax_pin": tin, "purpose": "tender",
	})
	# Returns 201 but status is rejected (service makes the decision)
	data = json.loads(r.data)
	assert data["data"]["status"] == "rejected"


# ---------------------------------------------------------------------------
# EOI
# ---------------------------------------------------------------------------

def test_exchange_of_information(client):
	reg = json.loads(_register(client).data)
	tin = reg["data"]["tax_pin"]
	r = _post(client, "/api/v1/tax/eoi", {
		"treaty_partner": "GB",
		"tax_pin": tin,
		"information_requested": "account_balances",
		"urgency": "routine",
	})
	assert r.status_code == 201
	data = json.loads(r.data)
	assert data["data"]["treaty_partner"] == "GB"


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def test_dashboard_report(client):
	r = _get(client, "/api/v1/tax/reports/dashboard")
	assert r.status_code == 200
	data = json.loads(r.data)
	assert "registered_taxpayers" in data["data"]


def test_revenue_report(client):
	r = _get(client, "/api/v1/tax/reports/revenue", period="2024")
	assert r.status_code == 200
	data = json.loads(r.data)
	assert "total_collected" in data["data"]


def test_compliance_report(client):
	r = _get(client, "/api/v1/tax/reports/compliance", period="2024")
	assert r.status_code == 200


def test_delinquency_report(client):
	r = _get(client, "/api/v1/tax/reports/delinquency")
	assert r.status_code == 200
	data = json.loads(r.data)
	assert "aging_buckets" in data["data"]


def test_audit_analytics_report(client):
	r = _get(client, "/api/v1/tax/reports/audits", period="2024")
	assert r.status_code == 200


def test_refund_analytics_report(client):
	r = _get(client, "/api/v1/tax/reports/refunds", period="2025")
	assert r.status_code == 200


# ---------------------------------------------------------------------------
# Pagination & filtering
# ---------------------------------------------------------------------------

def test_pagination_limit_offset(client):
	# Register 5 taxpayers
	for i in range(5):
		_register(client, name=f"Paged Taxpayer {i}", id_number=f"PAG-{i}")
	r = _get(client, "/api/v1/tax/taxpayers", limit=2, offset=0)
	data = json.loads(r.data)
	assert len(data["data"]) <= 2
	assert data["meta"]["total"] >= 5


def test_return_status_filter(client):
	reg = json.loads(_register(client).data)
	tin = reg["data"]["tax_pin"]
	_post(client, "/api/v1/tax/returns", {
		"tax_pin": tin, "tax_type": "vat", "period": "2025-03",
		"gross_income": 100000, "tax_liability": 16000, "tax_paid": 16000,
		"evidence_reference": "ev",
	})
	r = _get(client, "/api/v1/tax/returns", status="filed")
	data = json.loads(r.data)
	assert all(item["status"] == "filed" for item in data["data"])


# ---------------------------------------------------------------------------
# Tenant isolation via headers
# ---------------------------------------------------------------------------

def test_tenant_isolation_via_headers(client):
	# Register in tenant_a
	r_a = _post(client, "/api/v1/tax/taxpayers", {
		"taxpayer_name": "Tenant A Corp",
		"taxpayer_type": "company",
		"national_id": "BRN-A",
		"tax_type": "vat",
		"evidence_reference": "ev",
	}, tenant="tenant_a")
	assert r_a.status_code == 201

	# List from tenant_b — should be empty
	r_b = _get(client, "/api/v1/tax/taxpayers", tenant="tenant_b")
	data_b = json.loads(r_b.data)
	assert data_b["meta"]["total"] == 0
