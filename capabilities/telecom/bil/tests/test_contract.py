"""Tests for telecom_bil capability contract and service."""

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
	mod = _load("cc_bil", PACKAGE_DIR / "capability_contract.py")
	c = mod.get_capability_contract("t1")
	assert c["capability"] == "telecom_bil"
	assert c["streaming"]["processor"] == "bytewax"
	assert c["theme"]["tokens"]["border.radius"] == "8px"
	assert "mediation_workflow" in c["provides"]
	assert len(c["ui"]["routes"]) >= 8
	assert len(c["rule_engine"]["rules"]) >= 20
	assert "comp" in c["requires"]


def test_rule_engine():
	mod = _load("re_bil", PACKAGE_DIR / "capability_contract.py")
	assert mod.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "bil_batch", "event_stream": "redis"})["decision"] == "deny"
	assert mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "apply_discount", "discount_type_supported": True, "approval_present": False})["decision"] == "deny"
	assert mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True})["decision"] == "allow"


def test_billing_lifecycle():
	mod = _load("svc_bil", PACKAGE_DIR / "service.py")
	svc = mod.TelecomBilService()

	cdr = svc.record_cdr("cdr-1", "t1", "msc-1", "normalised", "+254700000001", 120, 0, "2026-01-01T10:00:00")
	charge = svc.record_charge("chg-1", "t1", "cust-1", "usage_based", "tiered", 25.50, "KES", 3.83, cdr["id"])
	cycle = svc.create_bill_cycle("cyc-1", "t1", "monthly", "2026-01-31", "2026-01-01", "2026-01-31")
	invoice = svc.generate_invoice("inv-1", "t1", "cust-1", cycle["id"], 1200.0, "KES", "2026-02-15")
	approved = svc.approve_invoice(invoice["id"], "t1", "approval-ref-1")
	dunning = svc.trigger_dunning("dun-1", "t1", invoice["id"], "reminder_1", "2026-02-16")
	payment = svc.record_payment("pay-1", "t1", invoice["id"], "mobile_money", 1200.0, "KES", "MPESA-REF-123", "2026-02-14")
	discount = svc.apply_discount("disc-1", "t1", "cust-1", "loyalty", 10.0, "approval-disc-1", "2026-01-01", "2026-12-31")
	conv = svc.setup_convergent("conv-1", "t1", "household", "master-1", "child-1,child-2", "KES")
	batch = svc.validate_batch("t1", 5)
	summary = svc.dashboard_summary("t1")

	assert cdr["mediation_status"] == "normalised"
	assert charge["charge_type"] == "usage_based"
	assert cycle["cycle_type"] == "monthly"
	assert approved["status"] == "approved"
	assert dunning["step"] == "reminder_1"
	assert payment["payment_method"] == "mobile_money"
	assert discount["discount_pct"] == 10.0
	assert conv["convergent_mode"] == "household"
	assert batch["processor"] == "bytewax"
	assert summary["invoice_count"] == 1


def test_billing_guardrails():
	mod = _load("svc_guard_bil", PACKAGE_DIR / "service.py")
	svc = mod.TelecomBilService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		svc.record_cdr("c", "", "src", "raw", "msisdn", 0, 0, "")
	with pytest.raises(PermissionError, match="charge_type_not_supported"):
		svc.record_charge("c", "t1", "cust", "unknown_charge", "flat_rate", 10.0, "KES", 1.5)
	with pytest.raises(PermissionError, match="charge_amount_must_be_positive"):
		svc.record_charge("c", "t1", "cust", "recurring", "flat_rate", -5.0, "KES", 0.0)
	with pytest.raises(PermissionError, match="bill_cycle_type_not_supported"):
		svc.create_bill_cycle("c", "t1", "daily_exotic", "cutoff", "start", "end")
	with pytest.raises(PermissionError, match="discount_exceeds_max_allowed"):
		svc.apply_discount("d", "t1", "cust", "loyalty", 75.0, "approval", "2026-01-01", "2026-12-31")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		svc.validate_batch("t1", 1, event_stream="rabbitmq")


def test_tenant_isolation():
	mod = _load("svc_iso_bil", PACKAGE_DIR / "service.py")
	svc = mod.TelecomBilService()

	svc.record_cdr("cdr-1", "tenant-a", "src-a", "raw", "+254700000001", 60, 0, "")
	svc.record_cdr("cdr-1", "tenant-b", "src-b", "raw", "+254700000002", 120, 0, "")

	assert svc.dashboard_summary("tenant-a")["cdr_count"] == 1
	assert svc.dashboard_summary("tenant-b")["cdr_count"] == 1
	assert svc.cdrs[("tenant-a", "cdr-1")].msisdn == "+254700000001"
	assert svc.cdrs[("tenant-b", "cdr-1")].msisdn == "+254700000002"


def test_api_and_views():
	api = _load("api_bil", PACKAGE_DIR / "api.py")
	views = _load("views_bil", PACKAGE_DIR / "views.py")

	cdr = api.record_cdr({"tenant_id": "t-api", "cdr_id": "cdr-api", "source": "msc", "msisdn": "+254700000099"})
	cycle = api.create_bill_cycle({"tenant_id": "t-api", "cycle_id": "cyc-api", "cycle_type": "monthly", "cutoff_date": "2026-01-31", "start_date": "2026-01-01", "end_date": "2026-01-31"})
	batch = api.validate_batch({"tenant_id": "t-api", "item_count": 2})
	db = views.dashboard_model(api.service(), "t-api")
	mediation = views.mediation_console_model(api.service(), "t-api")

	assert cdr["mediation_status"] == "raw"
	assert cycle["cycle_type"] == "monthly"
	assert batch["processor"] == "bytewax"
	assert db["summary"]["cdr_count"] == 1
	assert len(mediation["cdrs"]) == 1
