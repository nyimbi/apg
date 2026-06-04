"""Service-level tests for telecom_cus."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
	sys.path.insert(0, str(PACKAGE_DIR))


def _load(name, path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec and spec.loader
	mod = importlib.util.module_from_spec(spec)
	sys.modules[name] = mod
	spec.loader.exec_module(mod)
	return mod


def test_describe_returns_contract():
	mod = _load("svc_desc_cus", PACKAGE_DIR / "service.py")
	svc = mod.TelecomCusService()
	c = svc.describe("t1")
	assert c["capability"] == "telecom_cus"


def test_postpaid_requires_credit_check():
	mod = _load("svc_credit_cus", PACKAGE_DIR / "service.py")
	svc = mod.TelecomCusService()
	svc.create_customer("cust-1", "t1", "individual", "+254700000001", "Jane", "agent")
	with pytest.raises(PermissionError, match="credit_check_required_for_postpaid"):
		svc.activate_plan("plan-1", "t1", "cust-1", "postpaid", "Premium Plan", "ref", "2026-01-01", credit_check_completed=False)
	plan = svc.activate_plan("plan-ok", "t1", "cust-1", "postpaid", "Premium Plan", "ref", "2026-01-01", credit_check_completed=True)
	assert plan["plan_type"] == "postpaid"


def test_prepaid_no_credit_check_needed():
	mod = _load("svc_prepaid_cus", PACKAGE_DIR / "service.py")
	svc = mod.TelecomCusService()
	svc.create_customer("cust-1", "t1", "individual", "+254700000001", "John", "agent")
	plan = svc.activate_plan("plan-prepaid", "t1", "cust-1", "prepaid", "Basic Prepaid", "ref", "2026-01-01", credit_check_completed=False)
	assert plan["plan_type"] == "prepaid"


def test_sim_swap_updates_status():
	mod = _load("svc_sim_cus", PACKAGE_DIR / "service.py")
	svc = mod.TelecomCusService()
	svc.create_customer("cust-1", "t1", "individual", "+254700000001", "John", "agent")
	svc.provision_sim("sim-1", "t1", "cust-1", "8964010001234567890", "602010001234567", "+254700000001", "2026-01-01")
	blocked = svc.update_sim_status("sim-1", "t1", "stolen_blocked")
	assert blocked["status"] == "stolen_blocked"


def test_case_sla_types_all_accepted():
	mod = _load("svc_case_cus", PACKAGE_DIR / "service.py")
	svc = mod.TelecomCusService()
	svc.create_customer("cust-1", "t1", "individual", "+254700000001", "Alice", "agent")
	for i, ctype in enumerate(["complaint", "service_request", "billing_query", "technical_fault"]):
		case = svc.open_case(f"case-{i}", "t1", "cust-1", ctype, f"Test {ctype}", "2026-01-01")
		assert case["case_type"] == ctype


def test_kyc_verify_updates_customer_status():
	mod = _load("svc_kyc_cus", PACKAGE_DIR / "service.py")
	svc = mod.TelecomCusService()
	svc.create_customer("cust-1", "t1", "individual", "+254700000001", "Bob", "agent")
	svc.submit_kyc_document("doc-1", "t1", "cust-1", "national_id", "ID-001")
	svc.verify_kyc("doc-1", "t1", "officer-1")
	customer = svc.customers[("t1", "cust-1")]
	assert customer.kyc_status == "verified"


def test_lifecycle_events_all_types():
	mod = _load("svc_evt_cus", PACKAGE_DIR / "service.py")
	svc = mod.TelecomCusService()
	svc.create_customer("cust-1", "t1", "individual", "+254700000001", "Charlie", "agent")
	for i, etype in enumerate(["plan_changed", "sim_swapped", "number_ported"]):
		evt = svc.record_lifecycle_event(f"evt-{i}", "t1", "cust-1", etype, f"ref-{i}", "2026-01-01", "agent")
		assert evt["event_type"] == etype


def test_multi_tenant_customer_isolation():
	mod = _load("svc_iso_cus", PACKAGE_DIR / "service.py")
	svc = mod.TelecomCusService()
	svc.create_customer("cust-1", "tenant-a", "individual", "+254700000001", "Alice", "agent")
	svc.create_customer("cust-1", "tenant-b", "business", "+254700000002", "Acme Corp", "agent")
	assert svc.customers[("tenant-a", "cust-1")].customer_type == "individual"
	assert svc.customers[("tenant-b", "cust-1")].customer_type == "business"


def test_agent_name_required():
	mod = _load("svc_agt_cus", PACKAGE_DIR / "service.py")
	svc = mod.TelecomCusService()
	with pytest.raises(PermissionError, match="cus_agent_name_required"):
		svc.register_agent("agt", "t1", "", "codex", "account_manager", "scope")
