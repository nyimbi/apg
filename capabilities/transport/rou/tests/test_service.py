"""Service tests for transport_rou (Route Optimisation)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _load(mod_name: str, filename: str):
	path = PACKAGE_DIR / filename
	spec = importlib.util.spec_from_file_location(mod_name, path)
	assert spec and spec.loader
	mod = importlib.util.module_from_spec(spec)
	sys.modules[mod_name] = mod
	spec.loader.exec_module(mod)
	return mod

_cc = _load("_contract2_rou", "capability_contract.py")
import sys as _sys
_sys.modules["capability_contract"] = _cc
_models_mod = _load("_models_rou", "models.py")
_sys.modules["models"] = _models_mod
_svc_mod = _load("_service_rou", "service.py")
RouteOptimisationService = _svc_mod.RouteOptimisationService

def test_plan_route():
	svc = RouteOptimisationService()
	r = svc.plan_route("r1", "t1", "multi_stop", "Nairobi Depot", "Mombasa Port", "v1", "road", 5, 480.0, 360)
	assert r["route_type"] == "multi_stop"
	assert r["stop_count"] == 5


def test_route_missing_origin():
	svc = RouteOptimisationService()
	with pytest.raises(PermissionError, match="origin_required"):
		svc.plan_route("r1", "t1", "single_stop", "", "Mombasa", "v1")


def test_route_unvalidated_address():
	svc = RouteOptimisationService()
	with pytest.raises(PermissionError, match="unvalidated_address_dispatch_denied"):
		svc.plan_route("r1", "t1", "single_stop", "NBO", "MSA", "v1", address_validated=False)


def test_capacity_violation_blocked():
	svc = RouteOptimisationService()
	with pytest.raises(PermissionError, match="vehicle_capacity_constraint_violated"):
		svc.plan_route("r1", "t1", "single_stop", "NBO", "MSA", "v1", capacity_constraint_violated=True)


def test_add_stop():
	svc = RouteOptimisationService()
	svc.plan_route("r1", "t1", "multi_stop", "NBO", "MSA", "v1", stop_count=3)
	s = svc.add_route_stop("s1", "t1", "r1", 1, "-1.29,36.82", "Nairobi CBD", "2026-06-01T08:00:00Z", "2026-06-01T09:00:00Z", 30)
	assert s["sequence"] == 1


def test_add_constraint():
	svc = RouteOptimisationService()
	c = svc.add_constraint("c1", "t1", "r1", "time_window", '{"start": "08:00", "end": "17:00"}')
	assert c["constraint_type"] == "time_window"


def test_record_traffic_event():
	svc = RouteOptimisationService()
	svc.plan_route("r1", "t1", "single_stop", "NBO", "MSA", "v1")
	t = svc.record_traffic_event("te1", "t1", "here_maps", "r1", 45, "2026-06-01T10:00:00Z", "accident")
	assert t["delay_minutes"] == 45


def test_trigger_reroute():
	svc = RouteOptimisationService()
	svc.plan_route("r1", "t1", "single_stop", "NBO", "MSA", "v1")
	svc.plan_route("r2", "t1", "single_stop", "NBO_ALT", "MSA", "v1")
	rr = svc.trigger_reroute("rr1", "t1", "r1", "r2", "traffic_incident", "2026-06-01T10:15:00Z", 25.0)
	assert rr["trigger"] == "traffic_incident"


def test_reroute_invalid_trigger():
	svc = RouteOptimisationService()
	with pytest.raises(PermissionError, match="rerouting_trigger_not_supported"):
		svc.trigger_reroute("rr1", "t1", "r1", "r2", "alien_abduction", "2026-06-01T10:15:00Z")


def test_multimodal_segment():
	svc = RouteOptimisationService()
	svc.plan_route("r1", "t1", "intermodal", "NBO", "MSA", "v1", "intermodal_road_sea")
	seg = svc.plan_multimodal_segment("seg1", "t1", "r1", "sea", "Mombasa Port", "Dubai Port", "MAERSK", 5760)
	assert seg["transport_mode"] == "sea"


def test_register_agent():
	svc = RouteOptimisationService()
	a = svc.register_route_agent("a1", "t1", "Route Bot", "claude_code", "route_planner", "route planning")
	assert a["role"] == "route_planner"


def test_tenant_isolation():
	svc = RouteOptimisationService()
	svc.plan_route("r1", "t1", "single_stop", "A", "B", "v1")
	svc.plan_route("r1", "t2", "single_stop", "C", "D", "v2")
	assert svc.dashboard_summary("t1")["route_count"] == 1
	assert svc.dashboard_summary("t2")["route_count"] == 1
