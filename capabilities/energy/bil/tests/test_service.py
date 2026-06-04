"""Service tests for energy_bil capability."""

from __future__ import annotations

import importlib.util as _ilu, sys as _sys, os as _os

def _load_cap(name, cap="bil"):
	_cap_dir = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
	key = f"_ecap_bil_{name}"
	if key not in _sys.modules:
		spec = _ilu.spec_from_file_location(key, _os.path.join(_cap_dir, f"{name}.py"))
		mod  = _ilu.module_from_spec(spec)
		_sys.modules[key] = mod
		spec.loader.exec_module(mod)
	_sys.modules[name] = _sys.modules[key]

for _m in ("capability_contract", "models", "service", "views"):
	_load_cap(_m)

import pytest
from service import EnergyBillingService


def make_svc() -> EnergyBillingService:
	return EnergyBillingService()


def _tariff(svc, tid="tar1", tenant="t1", ttype="flat_rate", cclass="residential"):
	t = svc.create_tariff(tid, tenant, f"Tariff-{tid}", ttype, cclass, "2026-01-01", "admin")
	svc.approve_tariff(tid, tenant, "manager@acme.com")
	svc.activate_tariff(tid, tenant)
	return t


def test_create_tariff():
	svc = make_svc()
	t = svc.create_tariff("t1", "ten1", "Flat Rate", "flat_rate", "residential", "2026-01-01", "admin")
	assert t["status"] == "draft"


def test_approve_activate_tariff():
	svc = make_svc()
	svc.create_tariff("t1", "ten1", "TOU", "time_of_use", "small_commercial", "2026-01-01", "admin")
	approved = svc.approve_tariff("t1", "ten1", "manager@acme.com")
	assert approved["approved_by"] == "manager@acme.com"
	activated = svc.activate_tariff("t1", "ten1")
	assert activated["status"] == "active"


def test_activate_without_approval_raises():
	svc = make_svc()
	svc.create_tariff("t1", "ten1", "T", "flat_rate", "residential", "2026-01-01", "admin")
	with pytest.raises(ValueError, match="approved"):
		svc.activate_tariff("t1", "ten1")


def test_unsupported_tariff_type_raises():
	svc = make_svc()
	with pytest.raises(ValueError, match="tariff_type_not_supported"):
		svc.create_tariff("t1", "ten1", "Weird", "exotic_tariff", "residential", "2026-01-01", "admin")


def test_generate_and_issue_bill():
	svc = make_svc()
	_tariff(svc)
	bill = svc.generate_bill(
		"b1", "t1", "cust1", "meter1", "tar1", "monthly",
		"2026-05-01", "2026-05-31", 350.0, 5.0, [], 4200.0, "KES",
	)
	assert bill["status"] == "draft"
	issued = svc.issue_bill("b1", "t1", "2026-06-15")
	assert issued["status"] == "issued"
	assert issued["due_date"] == "2026-06-15"


def test_generate_bill_no_tariff_raises():
	svc = make_svc()
	with pytest.raises(ValueError, match="active_tariff_not_found"):
		svc.generate_bill("b1", "t1", "c1", "m1", "no_tariff", "monthly", "S", "E", 100.0, 0.0, [], 500.0)


def test_record_payment_and_status_update():
	svc = make_svc()
	_tariff(svc)
	svc.generate_bill("b1", "t1", "c1", "m1", "tar1", "monthly", "S", "E", 100.0, 0.0, [], 1000.0)
	svc.issue_bill("b1", "t1", "2026-06-15")
	p = svc.record_payment("p1", "t1", "b1", "c1", "mobile_money", 1000.0, "KES", "tx-001")
	assert p["amount"] == 1000.0
	bill = svc.bills[("t1", "b1")]
	assert bill.status == "paid"


def test_partial_payment():
	svc = make_svc()
	_tariff(svc)
	svc.generate_bill("b1", "t1", "c1", "m1", "tar1", "monthly", "S", "E", 100.0, 0.0, [], 1000.0)
	svc.record_payment("p1", "t1", "b1", "c1", "cash", 400.0, "KES")
	bill = svc.bills[("t1", "b1")]
	assert bill.status == "partially_paid"


def test_write_off_bill():
	svc = make_svc()
	_tariff(svc)
	svc.generate_bill("b1", "t1", "c1", "m1", "tar1", "monthly", "S", "E", 100.0, 0.0, [], 500.0)
	result = svc.write_off_bill("b1", "t1", "manager@acme.com")
	assert result["status"] == "written_off"


def test_write_off_without_approval_raises():
	svc = make_svc()
	_tariff(svc)
	svc.generate_bill("b1", "t1", "c1", "m1", "tar1", "monthly", "S", "E", 100.0, 0.0, [], 500.0)
	with pytest.raises(ValueError, match="bill_write_off_requires_approval"):
		svc.write_off_bill("b1", "t1", "")


def test_issue_credit():
	svc = make_svc()
	credit = svc.issue_credit("cr1", "t1", "c1", "renewable_energy_credit", 150.0, "KES", "2027-01-01", "manager")
	assert credit["credit_type"] == "renewable_energy_credit"
	assert credit["status"] == "active"


def test_open_and_resolve_dispute():
	svc = make_svc()
	_tariff(svc)
	svc.generate_bill("b1", "t1", "c1", "m1", "tar1", "monthly", "S", "E", 100.0, 0.0, [], 500.0)
	dispute = svc.open_dispute("d1", "t1", "b1", "c1", "billing_error", "ev-001")
	assert dispute["status"] == "open"
	resolved = svc.resolve_dispute("d1", "t1", "Adjusted meter read", 50.0)
	assert resolved["status"] == "resolved_accepted"


def test_flag_revenue_issue():
	svc = make_svc()
	flag = svc.flag_revenue_issue("ra1", "t1", "unbilled_energy", "meter-xyz", "meter", 2500.0, "KES")
	assert flag["flag_type"] == "unbilled_energy"
	assert flag["status"] == "open"


def test_dashboard_summary():
	svc = make_svc()
	_tariff(svc)
	svc.generate_bill("b1", "t1", "c1", "m1", "tar1", "monthly", "S", "E", 100.0, 0.0, [], 1000.0)
	svc.record_payment("p1", "t1", "b1", "c1", "mobile_money", 1000.0, "KES")
	summary = svc.dashboard_summary("t1")
	assert summary["total_bills"] == 1
	assert summary["total_collected_amount"] == 1000.0
	assert summary["collection_rate_pct"] == 100.0
