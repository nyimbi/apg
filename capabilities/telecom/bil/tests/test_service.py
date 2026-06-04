"""Service-level tests for telecom_bil."""

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
	mod = _load("svc_desc_bil", PACKAGE_DIR / "service.py")
	svc = mod.TelecomBilService()
	c = svc.describe("t1")
	assert c["capability"] == "telecom_bil"


def test_invoice_status_transitions():
	mod = _load("svc_inv_bil", PACKAGE_DIR / "service.py")
	svc = mod.TelecomBilService()
	svc.create_bill_cycle("cyc-1", "t1", "monthly", "2026-01-31", "2026-01-01", "2026-01-31")
	invoice = svc.generate_invoice("inv-1", "t1", "cust-1", "cyc-1", 500.0, "KES", "2026-02-15")
	assert invoice["status"] == "draft"
	approved = svc.approve_invoice("inv-1", "t1", "approval-1")
	assert approved["status"] == "approved"
	written_off = svc.write_off_invoice("inv-1", "t1", "write-off-approval-1")
	assert written_off["status"] == "written_off"


def test_all_payment_methods_accepted():
	mod = _load("svc_pay_bil", PACKAGE_DIR / "service.py")
	svc = mod.TelecomBilService()
	svc.create_bill_cycle("c1", "t1", "monthly", "2026-01-31", "2026-01-01", "2026-01-31")
	svc.generate_invoice("inv-1", "t1", "cust-1", "c1", 100.0, "KES", "2026-02-15")
	for i, method in enumerate(["bank_transfer", "mobile_money", "credit_card"]):
		pay = svc.record_payment(f"pay-{i}", "t1", "inv-1", method, 33.33, "KES", f"ref-{i}", "2026-02-01")
		assert pay["payment_method"] == method


def test_dunning_escalation_chain():
	mod = _load("svc_dun_bil", PACKAGE_DIR / "service.py")
	svc = mod.TelecomBilService()
	svc.create_bill_cycle("c1", "t1", "monthly", "2026-01-31", "2026-01-01", "2026-01-31")
	svc.generate_invoice("inv-1", "t1", "cust-1", "c1", 1000.0, "KES", "2026-02-15")
	for step in ["reminder_1", "reminder_2", "suspension_warning"]:
		d = svc.trigger_dunning(f"dun-{step}", "t1", "inv-1", step, "2026-02-16")
		assert d["step"] == step


def test_discount_max_50_pct_enforced():
	mod = _load("svc_disc_bil", PACKAGE_DIR / "service.py")
	svc = mod.TelecomBilService()
	with pytest.raises(PermissionError, match="discount_exceeds_max_allowed"):
		svc.apply_discount("d1", "t1", "cust-1", "loyalty", 51.0, "approval", "2026-01-01", "2026-12-31")
	disc = svc.apply_discount("d2", "t1", "cust-1", "loyalty", 50.0, "approval", "2026-01-01", "2026-12-31")
	assert disc["discount_pct"] == 50.0


def test_convergent_modes_all_supported():
	mod = _load("svc_conv_bil", PACKAGE_DIR / "service.py")
	svc = mod.TelecomBilService()
	for i, mode in enumerate(["single_bill", "household", "corporate_group"]):
		conv = svc.setup_convergent(f"conv-{i}", "t1", mode, f"master-{i}", f"child-{i}", "KES")
		assert conv["convergent_mode"] == mode


def test_agent_runtime_validated():
	mod = _load("svc_agt_bil", PACKAGE_DIR / "service.py")
	svc = mod.TelecomBilService()
	with pytest.raises(PermissionError, match="bil_agent_runtime_not_supported"):
		svc.register_agent("agt", "t1", "Agent", "gpt4-billing", "invoice_generator", "billing")


def test_multi_tenant_charge_isolation():
	mod = _load("svc_iso_bil", PACKAGE_DIR / "service.py")
	svc = mod.TelecomBilService()
	svc.record_charge("chg-1", "tenant-a", "cust-a", "recurring", "flat_rate", 100.0, "KES", 16.0)
	svc.record_charge("chg-1", "tenant-b", "cust-b", "one_time", "volume", 200.0, "KES", 32.0)
	assert svc.dashboard_summary("tenant-a")["charge_count"] == 1
	assert svc.dashboard_summary("tenant-b")["charge_count"] == 1
	assert svc.charges[("tenant-a", "chg-1")].charge_type == "recurring"
	assert svc.charges[("tenant-b", "chg-1")].charge_type == "one_time"


def test_batch_validates_stream():
	mod = _load("svc_batch_bil", PACKAGE_DIR / "service.py")
	svc = mod.TelecomBilService()
	result = svc.validate_batch("t1", 50)
	assert result["stream"] == "apg.telecom.bil.lifecycle"
	with pytest.raises(ValueError):
		svc.validate_batch("t1", 0)
