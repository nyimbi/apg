"""Service tests for energy_met capability."""

from __future__ import annotations

import importlib.util as _ilu, sys as _sys, os as _os

def _load_cap(name, cap="met"):
	_cap_dir = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
	key = f"_ecap_met_{name}"
	if key not in _sys.modules:
		spec = _ilu.spec_from_file_location(key, _os.path.join(_cap_dir, f"{name}.py"))
		mod  = _ilu.module_from_spec(spec)
		_sys.modules[key] = mod
		spec.loader.exec_module(mod)
	_sys.modules[name] = _sys.modules[key]

for _m in ("capability_contract", "models", "service", "views"):
	_load_cap(_m)

import pytest
from service import SmartMeteringService


def make_svc() -> SmartMeteringService:
	return SmartMeteringService()


def _meter(svc, mid="m1", tid="t1", cust="cust1"):
	return svc.register_meter(mid, tid, f"SN-{mid}", "smart_meter_electricity", "rf_mesh_900mhz", cust, "loc1", "2023-01-01")


def test_register_meter():
	svc = make_svc()
	m = _meter(svc)
	assert m["id"] == "m1"
	assert m["status"] == "active"


def test_register_meter_bad_type():
	svc = make_svc()
	with pytest.raises(ValueError, match="meter_type_not_supported"):
		svc.register_meter("m2", "t1", "SN2", "analog_meter", "plc_g3", "c1", "loc1", "2023-01-01")


def test_submit_reading():
	svc = make_svc()
	_meter(svc)
	r = svc.submit_reading("r1", "t1", "m1", "active_energy_import", "30min", "2026-06-01T00:00:00Z", "2026-06-01T00:30:00Z", 1.23, "kWh", "valid")
	assert r["value"] == 1.23
	assert r["quality_flag"] == "valid"


def test_submit_reading_inactive_meter_raises():
	svc = make_svc()
	_meter(svc)
	svc.update_meter_status("m1", "t1", "inactive")
	with pytest.raises(ValueError, match="meter_not_active"):
		svc.submit_reading("r1", "t1", "m1", "active_energy_import", "30min", "2026-06-01T00:00:00Z", "2026-06-01T00:30:00Z", 1.0, "kWh", "valid")


def test_report_tamper():
	svc = make_svc()
	_meter(svc)
	t = svc.report_tamper("t1", "t1", "m1", "magnetic_tamper", "evidence_001")
	assert t["tamper_type"] == "magnetic_tamper"
	m = svc.get_meter("t1", "m1")
	assert m["status"] == "tampered"


def test_report_tamper_no_evidence_raises():
	svc = make_svc()
	_meter(svc)
	with pytest.raises(ValueError, match="tamper_evidence_required"):
		svc.report_tamper("t2", "t1", "m1", "cover_open", "")


def test_issue_and_complete_command():
	svc = make_svc()
	_meter(svc)
	cmd = svc.issue_command("c1", "t1", "m1", "on_demand_read", "operator1")
	assert cmd["status"] == "pending"
	ack = svc.acknowledge_command("c1", "t1")
	assert ack["status"] == "acknowledged"
	done = svc.complete_command("c1", "t1")
	assert done["status"] == "executed"


def test_disconnect_without_approval_raises():
	svc = make_svc()
	_meter(svc)
	with pytest.raises(ValueError, match="disconnect_approval_required"):
		svc.issue_command("c2", "t1", "m1", "remote_disconnect", "op1", approved_by="")


def test_create_dr_event():
	svc = make_svc()
	_meter(svc)
	_meter(svc, "m2")
	dr = svc.create_dr_event("dr1", "t1", "direct_load_control", 5.0, "2026-06-01T18:00:00Z", "2026-06-01T19:00:00Z", ["m1", "m2"], "coordinator")
	assert dr["status"] == "active"
	assert dr["participation_count"] == 2


def test_dr_opt_out():
	svc = make_svc()
	_meter(svc)
	svc.create_dr_event("dr1", "t1", "direct_load_control", 5.0, "2026-06-01T18:00Z", "2026-06-01T19:00Z", ["m1"], "coord")
	updated = svc.opt_out_meter("dr1", "t1", "m1")
	assert "m1" in updated["opt_out_meter_ids"]


def test_complete_dr_event():
	svc = make_svc()
	svc.create_dr_event("dr1", "t1", "price_signal", 3.0, "T", "T2", [], "coord")
	completed = svc.complete_dr_event("dr1", "t1", 2.8)
	assert completed["status"] == "completed"
	assert completed["actual_reduction_kw"] == 2.8


def test_set_quality_flag():
	svc = make_svc()
	_meter(svc)
	svc.submit_reading("r1", "t1", "m1", "active_energy_import", "30min", "T", "T2", 1.0, "kWh", "valid")
	flag = svc.set_quality_flag("f1", "t1", "r1", "m1", "estimated", "outage_gap", "validator")
	assert flag["quality_flag"] == "estimated"


def test_head_end_status():
	svc = make_svc()
	he = svc.update_head_end_status("he1", "t1", "AMI-HE-01", "plc_g3", 950, 1000)
	assert he["status"] == "healthy"
	assert he["communication_ratio"] == 0.95


def test_dashboard_summary():
	svc = make_svc()
	_meter(svc)
	_meter(svc, "m2")
	svc.report_tamper("t1", "t1", "m1", "cover_open", "ev1")
	summary = svc.dashboard_summary("t1")
	assert summary["total_meters"] == 2
	assert summary["open_tamper_events"] == 1
