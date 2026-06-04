"""Service tests for transport_mai (Vehicle Maintenance)."""

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

_cc = _load("_contract2_mai", "capability_contract.py")
import sys as _sys
_sys.modules["capability_contract"] = _cc
_models_mod = _load("_models_mai", "models.py")
_sys.modules["models"] = _models_mod
_svc_mod = _load("_service_mai", "service.py")
VehicleMaintenanceService = _svc_mod.VehicleMaintenanceService

def test_create_job():
	svc = VehicleMaintenanceService()
	job = svc.create_job("j1", "t1", "v1", "preventive", "high", "tech-1", "in_house", 4.0, "JC-001")
	assert job["maintenance_type"] == "preventive"
	assert job["status"] == "scheduled"


def test_job_missing_vehicle():
	svc = VehicleMaintenanceService()
	with pytest.raises(PermissionError, match="vehicle_required"):
		svc.create_job("j1", "t1", "", "preventive", "high", "tech-1", "in_house", 4.0, "JC-001")


def test_update_job_status():
	svc = VehicleMaintenanceService()
	svc.create_job("j1", "t1", "v1", "preventive", "medium", "tech-1", "in_house", 4.0, "JC-001")
	j = svc.update_job_status("j1", "t1", "in_progress", 2.5)
	assert j["status"] == "in_progress"
	assert j["actual_hours"] == 2.5


def test_dispatch_check_expired_mot():
	svc = VehicleMaintenanceService()
	with pytest.raises(PermissionError, match="expired_mot_dispatch_denied"):
		svc.dispatch_vehicle_check("v1", "t1", mot_expired=True)


def test_dispatch_check_unsafe():
	svc = VehicleMaintenanceService()
	with pytest.raises(PermissionError, match="unsafe_vehicle_dispatch_denied"):
		svc.dispatch_vehicle_check("v1", "t1", vehicle_safe=False)


def test_dispatch_check_passes():
	svc = VehicleMaintenanceService()
	r = svc.dispatch_vehicle_check("v1", "t1")
	assert r["dispatch_cleared"] is True


def test_order_parts():
	svc = VehicleMaintenanceService()
	svc.create_job("j1", "t1", "v1", "corrective", "high", "tech-1", "in_house", 6.0, "JC-002")
	p = svc.order_parts("o1", "t1", "j1", "brakes", "BP-001", "Brake pads", 4, "supp-1", "2026-06-01")
	assert p["part_number"] == "BP-001"
	assert p["quantity"] == 4


def test_conduct_inspection():
	svc = VehicleMaintenanceService()
	i = svc.conduct_inspection("ins1", "t1", "v1", "pre_trip", "insp-1", "2026-06-01T06:00:00Z", False, "SIG-BASE64", True)
	assert i["passed"] is True
	assert i["defects_found"] is False


def test_inspection_requires_signature():
	svc = VehicleMaintenanceService()
	with pytest.raises(PermissionError, match="digital_signature_required"):
		svc.conduct_inspection("ins1", "t1", "v1", "pre_trip", "insp-1", "2026-06-01T06:00:00Z", False, "", True)


def test_issue_roadworthiness():
	svc = VehicleMaintenanceService()
	r = svc.issue_roadworthiness("rw1", "t1", "v1", "mot_uk", "MOT-2026-001", "2026-06-01", "2027-06-01", "DVSA")
	assert r["certificate_number"] == "MOT-2026-001"


def test_create_schedule():
	svc = VehicleMaintenanceService()
	s = svc.create_schedule("sc1", "t1", "v1", "preventive", "2026-09-01", interval_km=10000)
	assert s["interval_km"] == 10000


def test_register_agent():
	svc = VehicleMaintenanceService()
	a = svc.register_maintenance_agent("a1", "t1", "Maintenance Bot", "codex", "maintenance_scheduler", "scheduling scope")
	assert a["role"] == "maintenance_scheduler"


def test_dashboard_summary():
	svc = VehicleMaintenanceService()
	svc.create_job("j1", "t1", "v1", "preventive", "medium", "tech-1", "in_house", 4.0, "JC-001")
	svc.create_job("j2", "t1", "v2", "corrective", "high", "tech-2", "in_house", 8.0, "JC-002")
	svc.update_job_status("j1", "t1", "completed")
	summary = svc.dashboard_summary("t1")
	assert summary["job_count"] == 2
	assert summary["open_job_count"] == 1
