"""Service tests for energy_ren capability."""

from __future__ import annotations

import importlib.util as _ilu, sys as _sys, os as _os

def _load_cap(name, cap="ren"):
	_cap_dir = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
	key = f"_ecap_ren_{name}"
	if key not in _sys.modules:
		spec = _ilu.spec_from_file_location(key, _os.path.join(_cap_dir, f"{name}.py"))
		mod  = _ilu.module_from_spec(spec)
		_sys.modules[key] = mod
		spec.loader.exec_module(mod)
	_sys.modules[name] = _sys.modules[key]

for _m in ("capability_contract", "models", "service", "views"):
	_load_cap(_m)

import pytest
from service import RenewableEnergyService


def make_svc() -> RenewableEnergyService:
	return RenewableEnergyService()


def _asset(svc, aid="a1", tid="t1", rtype="solar_pv_utility", cap=100.0):
	return svc.register_asset(aid, tid, f"Asset-{aid}", rtype, cap, "owner1", "2023-01-01", "loc1")


def test_register_asset():
	svc = make_svc()
	a = _asset(svc)
	assert a["id"] == "a1"
	assert a["status"] == "operating"
	assert a["capacity_mw"] == 100.0


def test_register_asset_unsupported_type():
	svc = make_svc()
	with pytest.raises(ValueError, match="renewable_type_not_supported"):
		svc.register_asset("a2", "t1", "Fusion", "nuclear_fusion", 100.0, "o1", "2025-01-01", "loc")


def test_register_asset_zero_capacity():
	svc = make_svc()
	with pytest.raises(ValueError):
		svc.register_asset("a3", "t1", "Zero", "solar_pv_utility", 0.0, "o1", "2025-01-01", "loc")


def test_update_asset_status():
	svc = make_svc()
	_asset(svc)
	updated = svc.update_asset_status("a1", "t1", "curtailed")
	assert updated["status"] == "curtailed"


def test_record_and_approve_curtailment():
	svc = make_svc()
	_asset(svc)
	c = svc.record_curtailment("c1", "t1", "a1", "grid_congestion", 15.0, "2026-06-01T10:00Z", "2026-06-01T12:00Z", 3000.0, "KES")
	assert c["curtailed_mwh"] == 15.0
	assert c["status"] == "pending"
	approved = svc.approve_curtailment("c1", "t1", "grid_op@acme.com")
	assert approved["status"] == "approved"


def test_curtailment_zero_mwh_raises():
	svc = make_svc()
	_asset(svc)
	with pytest.raises(ValueError, match="curtailed_mwh_must_be_positive"):
		svc.record_curtailment("c2", "t1", "a1", "grid_congestion", 0.0, "T", "T2", 0.0, "KES")


def test_issue_and_retire_rec():
	svc = make_svc()
	_asset(svc)
	rec = svc.issue_rec("r1", "t1", "a1", "renewable_energy_certificate", 500.0, 2026, "RECS-Kenya", "SN-001")
	assert rec["status"] == "issued"
	retired = svc.retire_rec("r1", "t1")
	assert retired["status"] == "retired"


def test_rec_double_issuance_raises():
	svc = make_svc()
	_asset(svc)
	svc.issue_rec("r1", "t1", "a1", "renewable_energy_certificate", 500.0, 2026, "RECS-Kenya")
	with pytest.raises(ValueError, match="rec_already_issued_for_period"):
		svc.issue_rec("r2", "t1", "a1", "renewable_energy_certificate", 200.0, 2026, "RECS-Kenya")


def test_retire_already_retired_rec_raises():
	svc = make_svc()
	_asset(svc)
	svc.issue_rec("r1", "t1", "a1", "renewable_energy_certificate", 500.0, 2026, "REG")
	svc.retire_rec("r1", "t1")
	with pytest.raises(ValueError, match="already retired"):
		svc.retire_rec("r1", "t1")


def test_transfer_rec():
	svc = make_svc()
	_asset(svc)
	svc.issue_rec("r1", "t1", "a1", "solar_renewable_energy_certificate", 100.0, 2026, "SREC-KE")
	transferred = svc.transfer_rec("r1", "t1", "buyer-corp")
	assert transferred["transferred_to"] == "buyer-corp"
	assert transferred["status"] == "transferred"


def test_issue_carbon_credit():
	svc = make_svc()
	_asset(svc)
	cc = svc.issue_carbon_credit("cc1", "t1", "a1", "gold_standard", 250.0, 2026, "Gold Standard", "VER-2026-001")
	assert cc["quantity_tco2e"] == 250.0
	assert cc["status"] == "issued"


def test_carbon_credit_no_verification_raises():
	svc = make_svc()
	_asset(svc)
	with pytest.raises(ValueError, match="carbon_credit_verification_required"):
		svc.issue_carbon_credit("cc2", "t1", "a1", "gold_standard", 100.0, 2026, "GS", "")


def test_create_fit():
	svc = make_svc()
	_asset(svc)
	fit = svc.create_fit("f1", "t1", "a1", "fixed_fit", 12.5, "KES", "2026-01-01", "regulator@acme.com")
	assert fit["rate_per_kwh"] == 12.5
	assert fit["status"] == "active"


def test_publish_forecast():
	svc = make_svc()
	_asset(svc)
	forecast = svc.publish_forecast("fc1", "t1", "a1", "generation_output", "24h", "2026-06-02T00:00Z", "2026-06-03T00:00Z", [{"t": "00:00", "mw": 45}], "v1.2", rmse=2.1)
	assert forecast["horizon"] == "24h"


def test_record_performance_metric():
	svc = make_svc()
	_asset(svc)
	m = svc.record_performance_metric("pm1", "t1", "a1", "capacity_factor", "2026-05-01", "2026-05-31", 22.5, "%", 25.0)
	assert m["value"] == 22.5
	assert m["deviation_from_benchmark"] == pytest.approx(-2.5)


def test_dashboard_summary():
	svc = make_svc()
	_asset(svc)
	_asset(svc, "a2", rtype="wind_onshore", cap=80.0)
	svc.record_curtailment("c1", "t1", "a1", "grid_congestion", 5.0, "T", "T2", 1000.0, "KES")
	summary = svc.dashboard_summary("t1")
	assert summary["total_assets"] == 2
	assert summary["total_capacity_mw"] == 180.0
	assert summary["total_curtailed_mwh"] == 5.0
