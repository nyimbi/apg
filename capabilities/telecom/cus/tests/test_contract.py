"""Tests for telecom_cus capability contract and service."""

from __future__ import annotations

import importlib.util
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
	spec.loader.exec_module(mod)
	return mod


def test_contract_shape():
	mod = _load("cc_cus", PACKAGE_DIR / "capability_contract.py")
	c = mod.get_capability_contract("t1")
	assert c["capability"] == "telecom_cus"
	assert c["streaming"]["processor"] == "bytewax"
	assert c["theme"]["tokens"]["border.radius"] == "8px"
	assert "customer_lifecycle_workflow" in c["provides"]
	assert "kyc_workflow" in c["provides"]
	assert len(c["ui"]["routes"]) >= 8
	assert len(c["rule_engine"]["rules"]) >= 20


def test_rule_engine():
	mod = _load("re_cus", PACKAGE_DIR / "capability_contract.py")
	assert mod.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "submit_kyc_document", "document_type_supported": True, "kyc_bypass_scope": True})["decision"] == "deny"
	assert mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "cus_batch", "event_stream": "kafka"})["decision"] == "deny"
	assert mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True})["decision"] == "allow"


def test_customer_lifecycle():
	mod = _load("svc_cus", PACKAGE_DIR / "service.py")
	svc = mod.TelecomCusService()

	cust = svc.create_customer("cust-1", "t1", "individual", "+254700000001", "John Doe", "agent-1")
	doc = svc.submit_kyc_document("doc-1", "t1", "cust-1", "national_id", "ID-123456", "2030-01-01")
	verified = svc.verify_kyc("doc-1", "t1", "kyc-officer")
	plan = svc.activate_plan("plan-1", "t1", "cust-1", "prepaid", "Daily Bundle", "plan-ref-1", "2026-01-01")
	sim = svc.provision_sim("sim-1", "t1", "cust-1", "8964010001234567890", "602010001234567", "+254700000001", "2026-01-01")
	device = svc.register_device("dev-1", "t1", "cust-1", "handset", "356938035643809", "Samsung S24", "2026-01-01")
	case = svc.open_case("case-1", "t1", "cust-1", "billing_query", "Query about data charges", "2026-01-05")
	resolved = svc.update_case_status("case-1", "t1", "resolved", "2026-01-06")
	event = svc.record_lifecycle_event("evt-1", "t1", "cust-1", "plan_changed", "plan-ref-1", "2026-01-01", "agent-1")
	agent = svc.register_agent("agt-1", "t1", "CUS Agent", "codex", "account_manager", "customer management")
	batch = svc.validate_batch("t1", 3)
	summary = svc.dashboard_summary("t1")

	assert cust["customer_type"] == "individual"
	assert verified["status"] == "verified"
	assert plan["plan_type"] == "prepaid"
	assert sim["iccid"] == "8964010001234567890"
	assert device["device_type"] == "handset"
	assert resolved["status"] == "resolved"
	assert event["event_type"] == "plan_changed"
	assert agent["runtime"] == "codex"
	assert batch["processor"] == "bytewax"
	assert summary["customer_count"] == 1
	assert summary["audit_event_count"] >= 9


def test_kyc_rejection():
	mod = _load("svc_kyc_cus", PACKAGE_DIR / "service.py")
	svc = mod.TelecomCusService()
	svc.create_customer("cust-2", "t1", "individual", "+254700000002", "Jane Doe", "agent")
	svc.submit_kyc_document("doc-2", "t1", "cust-2", "passport", "PASS-789", "2030-01-01")
	rejected = svc.reject_kyc("doc-2", "t1")
	assert rejected["status"] == "rejected"


def test_guardrails():
	mod = _load("svc_guard_cus", PACKAGE_DIR / "service.py")
	svc = mod.TelecomCusService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		svc.create_customer("c", "", "individual", "+254", "Name", "agent")
	with pytest.raises(PermissionError, match="customer_type_not_supported"):
		svc.create_customer("c", "t1", "alien", "+254700000001", "Name", "agent")
	with pytest.raises(PermissionError, match="msisdn_required"):
		svc.create_customer("c", "t1", "individual", "", "Name", "agent")
	with pytest.raises(PermissionError, match="kyc_document_type_not_supported"):
		svc.submit_kyc_document("d", "t1", "c1", "tattoo", "ref", None)
	with pytest.raises(PermissionError, match="sim_status_not_supported"):
		svc.update_sim_status("s", "t1", "on_holiday")
	with pytest.raises(PermissionError, match="case_type_not_supported"):
		svc.open_case("c", "t1", "c1", "mystery", "desc", "2026-01-01")


def test_api_and_views():
	api = _load("api_cus", PACKAGE_DIR / "api.py")
	views = _load("views_cus", PACKAGE_DIR / "views.py")

	cust = api.create_customer({"tenant_id": "t-api", "customer_id": "cust-api", "customer_type": "individual", "msisdn": "+254700000099", "name": "API Customer", "created_by": "test"})
	doc = api.submit_kyc_document({"tenant_id": "t-api", "doc_id": "doc-api", "customer_id": "cust-api", "document_type": "national_id", "document_reference": "ID-API-001"})
	case = api.open_case({"tenant_id": "t-api", "case_id": "case-api", "customer_id": "cust-api", "case_type": "billing_query", "description": "Test case"})
	batch = api.validate_batch({"tenant_id": "t-api", "item_count": 1})
	db = views.dashboard_model(api.service(), "t-api")
	c360 = views.customer_360_model(api.service(), "t-api", "cust-api")

	assert cust["msisdn"] == "+254700000099"
	assert doc["document_type"] == "national_id"
	assert case["case_type"] == "billing_query"
	assert batch["processor"] == "bytewax"
	assert db["summary"]["customer_count"] == 1
	assert c360["customer"]["id"] == "cust-api"
	assert len(c360["kyc_documents"]) == 1
