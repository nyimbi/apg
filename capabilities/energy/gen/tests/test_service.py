"""Service tests for energy_gen capability."""

from __future__ import annotations

import importlib.util as _ilu, sys as _sys, os as _os

def _load_cap(name, cap="gen"):
	_cap_dir = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
	key = f"_ecap_gen_{name}"
	if key not in _sys.modules:
		spec = _ilu.spec_from_file_location(key, _os.path.join(_cap_dir, f"{name}.py"))
		mod  = _ilu.module_from_spec(spec)
		_sys.modules[key] = mod
		spec.loader.exec_module(mod)
	_sys.modules[name] = _sys.modules[key]

for _m in ("capability_contract", "models", "service", "views"):
	_load_cap(_m)

import pytest
from service import GenerationManagementService


def make_svc() -> GenerationManagementService:
	return GenerationManagementService()


def test_describe_returns_contract():
	svc = make_svc()
	contract = svc.describe("acme")
	assert contract["capability"] == "energy_gen"
	assert contract["configuration"]["tenant_id"] == "acme"


def test_register_plant_happy_path():
	svc = make_svc()
	result = svc.register_plant(
		plant_id="p1", tenant_id="t1", name="Coal Plant A",
		plant_type="thermal_coal", fuel_type="coal",
		capacity_mw=300.0, owner_id="owner1",
		commissioning_date="2010-01-01", location_reference="grid_ref_001",
	)
	assert result["id"] == "p1"
	assert result["status"] == "operational"
	assert result["available_mw"] == 300.0


def test_register_plant_unsupported_type_raises():
	svc = make_svc()
	with pytest.raises(ValueError, match="plant_type_not_supported"):
		svc.register_plant(
			plant_id="p2", tenant_id="t1", name="Mystery Plant",
			plant_type="nuclear_fusion", fuel_type="coal",
			capacity_mw=100.0, owner_id="owner1",
			commissioning_date="2025-01-01", location_reference="loc1",
		)


def test_register_plant_zero_capacity_raises():
	svc = make_svc()
	with pytest.raises(ValueError):
		svc.register_plant(
			plant_id="p3", tenant_id="t1", name="Zero Plant",
			plant_type="thermal_gas", fuel_type="natural_gas",
			capacity_mw=0.0, owner_id="owner1",
			commissioning_date="2025-01-01", location_reference="loc1",
		)


def test_list_plants_tenant_isolation():
	svc = make_svc()
	svc.register_plant("p1", "t1", "Plant T1", "solar_pv", "solar", 50.0, "owner1", "2023-01-01", "loc1")
	svc.register_plant("p2", "t2", "Plant T2", "wind_onshore", "wind", 80.0, "owner2", "2023-01-01", "loc2")
	t1_plants = svc.list_plants("t1")
	assert len(t1_plants) == 1
	assert t1_plants[0]["id"] == "p1"


def test_create_and_approve_dispatch_schedule():
	svc = make_svc()
	svc.register_plant("p1", "t1", "Gas Peaker", "gas_peaker", "natural_gas", 200.0, "owner1", "2020-01-01", "loc1")
	schedule = svc.create_dispatch_schedule(
		schedule_id="s1", tenant_id="t1", plant_id="p1",
		dispatch_mode="peaking", scheduled_mw=150.0,
		start_time="2026-06-01T18:00:00Z", end_time="2026-06-01T22:00:00Z",
	)
	assert schedule["status"] == "draft"
	approved = svc.approve_dispatch_schedule("s1", "t1", "dispatcher@acme.com")
	assert approved["status"] == "approved"
	assert approved["approved_by"] == "dispatcher@acme.com"


def test_dispatch_exceeds_capacity_raises():
	svc = make_svc()
	svc.register_plant("p1", "t1", "Small Solar", "solar_pv", "solar", 50.0, "owner1", "2023-01-01", "loc1")
	with pytest.raises(ValueError, match="dispatch_mw_exceeds_capacity"):
		svc.create_dispatch_schedule(
			"s1", "t1", "p1", "baseload", 99.0,
			"2026-06-01T00:00:00Z", "2026-06-02T00:00:00Z",
		)


def test_schedule_and_complete_outage():
	svc = make_svc()
	svc.register_plant("p1", "t1", "Gas Plant", "thermal_gas", "natural_gas", 100.0, "owner1", "2015-01-01", "loc1")
	outage = svc.schedule_outage(
		"o1", "t1", "p1", "planned_maintenance",
		"2026-07-01", "2026-07-05", "Annual inspection", "ev-ref-001",
	)
	assert outage["status"] == "scheduled"
	started = svc.start_outage("o1", "t1")
	assert started["status"] == "in_progress"
	completed = svc.complete_outage("o1", "t1")
	assert completed["status"] == "completed"


def test_outage_not_found_raises():
	svc = make_svc()
	with pytest.raises(KeyError):
		svc.start_outage("nonexistent", "t1")


def test_calculate_kpi():
	svc = make_svc()
	svc.register_plant("p1", "t1", "Wind Farm", "wind_onshore", "wind", 80.0, "o1", "2021-01-01", "loc1")
	kpi = svc.calculate_kpi(
		"k1", "t1", "p1", "capacity_factor", "monthly",
		"2026-05-01", "2026-05-31", 38.5, "%",
	)
	assert kpi["kpi_type"] == "capacity_factor"
	assert kpi["value"] == 38.5


def test_create_capacity_plan():
	svc = make_svc()
	plan = svc.create_capacity_plan(
		"cp1", "t1", "10-Year Plan", 10, 2026,
		500.0, 200.0, 600.0, 15.0, "planner@acme.com",
	)
	assert plan["horizon_years"] == 10
	assert plan["status"] == "draft"


def test_capacity_plan_invalid_horizon_raises():
	svc = make_svc()
	with pytest.raises(ValueError):
		svc.create_capacity_plan("cp2", "t1", "Bad Plan", 25, 2026, 0, 0, 0, 0, "user")


def test_fuel_stock_update_and_low_alert():
	svc = make_svc()
	svc.register_plant("p1", "t1", "Coal Plant", "thermal_coal", "coal", 200.0, "o1", "2010-01-01", "loc1")
	svc.update_fuel_stock("fs1", "t1", "p1", "coal", 5000.0, "tonnes", 3.0, "supplier-A")
	alerts = svc.get_low_fuel_alerts("t1")
	assert len(alerts) == 1
	assert alerts[0]["days_of_supply"] == 3.0


def test_register_agent_happy_path():
	svc = make_svc()
	agent = svc.register_agent("a1", "t1", "DispatchBot", "codex", "dispatch_optimizer")
	assert agent["runtime"] == "codex"
	assert agent["role"] == "dispatch_optimizer"


def test_register_agent_bad_runtime_raises():
	svc = make_svc()
	with pytest.raises(ValueError, match="agent_runtime_not_supported"):
		svc.register_agent("a2", "t1", "BadBot", "gpt_99", "dispatch_optimizer")


def test_dashboard_summary():
	svc = make_svc()
	svc.register_plant("p1", "t1", "Gas", "thermal_gas", "natural_gas", 200.0, "o1", "2015-01-01", "loc1")
	summary = svc.dashboard_summary("t1")
	assert summary["total_plants"] == 1
	assert summary["total_capacity_mw"] == 200.0
	assert summary["active_outages"] == 0


def test_audit_events_recorded():
	svc = make_svc()
	svc.register_plant("p1", "t1", "Hydro", "hydro", "water", 150.0, "o1", "2008-01-01", "loc1")
	assert len(svc.audit_events) >= 1
	assert svc.audit_events[-1].event_type == "plant_registered"
