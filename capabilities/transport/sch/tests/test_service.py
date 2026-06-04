"""Service tests for transport_sch (Transport Scheduling)."""

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

_cc = _load("_contract2_sch", "capability_contract.py")
import sys as _sys
_sys.modules["capability_contract"] = _cc
_models_mod = _load("_models_sch", "models.py")
_sys.modules["models"] = _models_mod
_svc_mod = _load("_service_sch", "service.py")
TransportSchedulingService = _svc_mod.TransportSchedulingService

def test_create_schedule():
	svc = TransportSchedulingService()
	s = svc.create_schedule("sc1", "t1", "load_schedule", "2026-06-01", "2026-06-07", "balanced", "planner-1")
	assert s["schedule_type"] == "load_schedule"
	assert s["status"] == "draft"


def test_publish_schedule_clean():
	svc = TransportSchedulingService()
	svc.create_schedule("sc1", "t1", "load_schedule", "2026-06-01", "2026-06-07", "balanced", "planner-1")
	s = svc.publish_schedule("sc1", "t1")
	assert s["status"] == "published"


def test_publish_blocked_with_conflicts():
	svc = TransportSchedulingService()
	svc.create_schedule("sc1", "t1", "load_schedule", "2026-06-01", "2026-06-07", "balanced", "planner-1")
	svc.record_conflict("cf1", "t1", "sc1", "double_booking", "v1", "2026-06-01T07:00:00Z")
	with pytest.raises(PermissionError, match="unresolved_conflicts_block_publish"):
		svc.publish_schedule("sc1", "t1")


def test_publish_after_resolving_conflict():
	svc = TransportSchedulingService()
	svc.create_schedule("sc1", "t1", "load_schedule", "2026-06-01", "2026-06-07", "balanced", "planner-1")
	svc.record_conflict("cf1", "t1", "sc1", "double_booking", "v1", "2026-06-01T07:00:00Z")
	svc.resolve_conflict("cf1", "t1", "2026-06-01T08:00:00Z", "Vehicle reallocated")
	s = svc.publish_schedule("sc1", "t1")
	assert s["status"] == "published"


def test_create_shift():
	svc = TransportSchedulingService()
	svc.create_schedule("sc1", "t1", "driver_shift", "2026-06-01", "2026-06-07", "balanced", "planner-1")
	sh = svc.create_shift("sh1", "t1", "sc1", "dr1", "day_shift", "2026-06-01T06:00:00Z", "2026-06-01T16:00:00Z", 10.0)
	assert sh["shift_type"] == "day_shift"


def test_shift_hours_breach():
	svc = TransportSchedulingService()
	with pytest.raises(PermissionError, match="driver_hours_breach_denied"):
		svc.create_shift("sh1", "t1", "sc1", "dr1", "day_shift", "2026-06-01T06:00:00Z", "2026-06-01T16:00:00Z", 10.0, driver_hours_compliant=False)


def test_assign_vehicle():
	svc = TransportSchedulingService()
	svc.create_schedule("sc1", "t1", "vehicle_assignment", "2026-06-01", "2026-06-07", "balanced", "p1")
	va = svc.assign_vehicle("va1", "t1", "sc1", "v1", "r1", "2026-06-01T06:00:00Z", "2026-06-01T18:00:00Z")
	assert va["vehicle_id"] == "v1"


def test_double_booking_blocked():
	svc = TransportSchedulingService()
	with pytest.raises(PermissionError, match="double_booking_denied"):
		svc.assign_vehicle("va1", "t1", "sc1", "v1", "r1", "2026-06-01T06:00:00Z", "2026-06-01T18:00:00Z", double_booking_detected=True)


def test_create_charter():
	svc = TransportSchedulingService()
	svc.create_schedule("sc1", "t1", "charter", "2026-06-01", "2026-06-01", "balanced", "p1")
	ch = svc.create_charter("ch1", "t1", "sc1", "school_charter", "school-1", "v1", "dr1", "School A", "Park B", "2026-06-01", customer_confirmed=True)
	assert ch["charter_type"] == "school_charter"


def test_charter_without_confirmation():
	svc = TransportSchedulingService()
	with pytest.raises(PermissionError, match="customer_confirmation_required"):
		svc.create_charter("ch1", "t1", "sc1", "corporate_charter", "corp-1", "v1", "dr1", "HQ", "Airport", "2026-06-01", customer_confirmed=False)


def test_register_agent():
	svc = TransportSchedulingService()
	a = svc.register_scheduling_agent("a1", "t1", "Schedule Bot", "codex", "schedule_planner", "scheduling scope")
	assert a["role"] == "schedule_planner"


def test_dashboard_summary():
	svc = TransportSchedulingService()
	svc.create_schedule("sc1", "t1", "load_schedule", "2026-06-01", "2026-06-07", "balanced", "p1")
	svc.record_conflict("cf1", "t1", "sc1", "double_booking", "v1", "2026-06-01T07:00:00Z")
	summary = svc.dashboard_summary("t1")
	assert summary["schedule_count"] == 1
	assert summary["open_conflict_count"] == 1
