"""Service tests for transport_dis (Dispatch Operations)."""

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

_cc = _load("_contract2_dis", "capability_contract.py")
import sys as _sys
_sys.modules["capability_contract"] = _cc
_models_mod = _load("_models_dis", "models.py")
_sys.modules["models"] = _models_mod
_svc_mod = _load("_service_dis", "service.py")
DispatchOperationsService = _svc_mod.DispatchOperationsService

def test_plan_load():
	svc = DispatchOperationsService()
	load = svc.plan_load("l1", "t1", "full_truckload", "v1", 20000.0, 80.0, 3, "cost")
	assert load["load_type"] == "full_truckload"
	assert load["total_weight_kg"] == 20000.0


def test_overload_blocked():
	svc = DispatchOperationsService()
	with pytest.raises(PermissionError, match="overload_dispatch_denied"):
		svc.plan_load("l1", "t1", "full_truckload", "v1", 50000.0, 200.0, 1, "cost")


def test_create_dispatch():
	svc = DispatchOperationsService()
	svc.plan_load("l1", "t1", "full_truckload", "v1", 20000.0, 80.0, 3, "cost")
	d = svc.create_dispatch("d1", "t1", "l1", "v1", "dr1", "r1")
	assert d["status"] == "planned"


def test_dispatch_missing_vehicle():
	svc = DispatchOperationsService()
	with pytest.raises(PermissionError, match="vehicle_required"):
		svc.create_dispatch("d1", "t1", "l1", "", "dr1", "r1")


def test_assign_driver():
	svc = DispatchOperationsService()
	svc.create_dispatch("d1", "t1", "l1", "v1", "dr1", "r1")
	a = svc.assign_driver("a1", "t1", "d1", "dr1", "primary", "2026-06-01T07:00:00Z", 10.0)
	assert a["assignment_type"] == "primary"


def test_driver_hours_check():
	svc = DispatchOperationsService()
	with pytest.raises(PermissionError, match="driver_hours_exceeded"):
		svc.assign_driver("a1", "t1", "d1", "dr1", "primary", "2026-06-01T07:00:00Z", -1.0)


def test_update_dispatch_status():
	svc = DispatchOperationsService()
	svc.plan_load("l1", "t1", "full_truckload", "v1", 20000.0, 80.0, 3, "cost")
	svc.create_dispatch("d1", "t1", "l1", "v1", "dr1", "r1")
	d = svc.update_dispatch_status("d1", "t1", "dispatched", "2026-06-01T08:00:00Z")
	assert d["status"] == "dispatched"


def test_raise_exception():
	svc = DispatchOperationsService()
	svc.plan_load("l1", "t1", "full_truckload", "v1", 20000.0, 80.0, 3, "cost")
	svc.create_dispatch("d1", "t1", "l1", "v1", "dr1", "r1")
	exc = svc.raise_exception("e1", "t1", "d1", "traffic_delay", "2026-06-01T09:00:00Z")
	assert exc["exception_type"] == "traffic_delay"
	assert exc["resolved_at"] is None


def test_resolve_exception():
	svc = DispatchOperationsService()
	svc.plan_load("l1", "t1", "full_truckload", "v1", 20000.0, 80.0, 3, "cost")
	svc.create_dispatch("d1", "t1", "l1", "v1", "dr1", "r1")
	svc.raise_exception("e1", "t1", "d1", "traffic_delay", "2026-06-01T09:00:00Z")
	exc = svc.resolve_exception("e1", "t1", "2026-06-01T10:00:00Z", "Route cleared")
	assert exc["resolved_at"] == "2026-06-01T10:00:00Z"


def test_tracking_update():
	svc = DispatchOperationsService()
	u = svc.update_tracking("u1", "t1", "d1", "waypoint", "-1.29,36.82", "2026-06-01T09:30:00Z", 45)
	assert u["update_type"] == "waypoint"


def test_register_agent():
	svc = DispatchOperationsService()
	a = svc.register_dispatch_agent("a1", "t1", "Dispatch Bot", "codex", "load_planner", "load planning scope")
	assert a["role"] == "load_planner"


def test_dashboard_summary():
	svc = DispatchOperationsService()
	svc.plan_load("l1", "t1", "full_truckload", "v1", 20000.0, 80.0, 3, "cost")
	svc.create_dispatch("d1", "t1", "l1", "v1", "dr1", "r1")
	summary = svc.dashboard_summary("t1")
	assert summary["load_plan_count"] == 1
	assert summary["dispatch_count"] == 1
