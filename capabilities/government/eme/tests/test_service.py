"""Service layer tests for APG Emergency Management."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
	sys.path.insert(0, str(PACKAGE_DIR))


def _load(name: str, path: Path):
	"""Load module by path, always (re)registering deps under bare names for fallback imports."""
	# Always overwrite bare-name slots so this capability's deps win even in a multi-cap test run
	for dep in ('capability_contract', 'models'):
		dep_path = PACKAGE_DIR / f"{dep}.py"
		if dep_path.exists():
			dep_spec = importlib.util.spec_from_file_location(f"{name}__{dep}", dep_path)
			dep_mod = importlib.util.module_from_spec(dep_spec)
			sys.modules[f"{name}__{dep}"] = dep_mod
			sys.modules[dep] = dep_mod  # overwrite bare name each time
			dep_spec.loader.exec_module(dep_mod)
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec and spec.loader
	mod = importlib.util.module_from_spec(spec)
	sys.modules[name] = mod
	spec.loader.exec_module(mod)
	return mod


def test_full_emergency_lifecycle():
	svc = _load("svc_eme", PACKAGE_DIR / "service.py").EmergencyManagementService()
	incident = svc.declare_incident("inc1", "t1", "natural_disaster", "major", "loc-nairobi", "commander-1", "Flooding in Westlands", "incident-ev")
	assert incident["severity"] == "major"
	resource = svc.mobilise_resource("res1", "t1", "inc1", "personnel", 50, "persons", "kenya-red-cross", "res-ev")
	assert resource["quantity"] == 50
	agency = svc.activate_agency("ag1", "t1", "inc1", "ngos", "Kenya Red Cross", "krc-contact", "search_and_rescue")
	assert agency["agency_type"] == "ngos"
	eoc = svc.update_eoc("eoc1", "t1", "inc1", "full_activation", "unified_command", "cs-directive-001", "eoc-ev", authorised=True)
	assert eoc["eoc_status"] == "full_activation"
	sitrep = svc.file_sitrep("sr1", "t1", "inc1", "H+6", "author-1", "400 persons displaced", "sr-ev")
	assert sitrep["period"] == "H+6"
	aar = svc.record_aar("aar1", "t1", "inc1", "reviewer-1", "Early warning delayed by 2 hours", "Improve EWS integration", "aar-ev")
	assert "Early warning" in aar["lessons_learned"]
	summary = svc.dashboard_summary("t1")
	assert summary["incident_count"] == 1
	assert summary["aar_count"] == 1


def test_missing_commander_denied():
	svc = _load("svc_eme_cmd", PACKAGE_DIR / "service.py").EmergencyManagementService()
	with pytest.raises(PermissionError, match="incident_commander_required"):
		svc.declare_incident("inc1", "t1", "natural_disaster", "minor", "loc", "", "desc", "ev")


def test_unsupported_incident_type_denied():
	svc = _load("svc_eme_type", PACKAGE_DIR / "service.py").EmergencyManagementService()
	with pytest.raises(PermissionError, match="incident_type_not_supported"):
		svc.declare_incident("inc1", "t1", "alien_invasion", "major", "loc", "commander", "desc", "ev")


def test_unauthorised_eoc_activation_denied():
	svc = _load("svc_eme_eoc", PACKAGE_DIR / "service.py").EmergencyManagementService()
	svc.declare_incident("inc1", "t1", "natural_disaster", "major", "loc", "commander", "desc", "ev")
	with pytest.raises(PermissionError, match="unauthorised_eoc_activation_denied"):
		svc.update_eoc("eoc1", "t1", "inc1", "full_activation", "unified_command", "authority-ref", "ev", authorised=False)


def test_aar_requires_lessons():
	svc = _load("svc_eme_aar", PACKAGE_DIR / "service.py").EmergencyManagementService()
	svc.declare_incident("inc1", "t1", "natural_disaster", "minor", "loc", "commander", "desc", "ev")
	with pytest.raises(PermissionError, match="lessons_required"):
		svc.record_aar("aar1", "t1", "inc1", "reviewer-1", "", "recommendations", "aar-ev")


def test_phase_transition():
	svc = _load("svc_eme_phase", PACKAGE_DIR / "service.py").EmergencyManagementService()
	svc.declare_incident("inc1", "t1", "security_threat", "serious", "loc", "commander-1", "desc", "ev")
	incident = svc.transition_phase("inc1", "t1", "response")
	assert incident["phase"] == "response"


def test_resource_mobilisation():
	svc = _load("svc_eme_res", PACKAGE_DIR / "service.py").EmergencyManagementService()
	svc.declare_incident("inc1", "t1", "industrial_accident", "moderate", "loc", "cmdr", "desc", "ev")
	resource = svc.mobilise_resource("r1", "t1", "inc1", "medical_supplies", 100, "units", "MOH", "ev")
	assert resource["resource_type"] == "medical_supplies"


def test_agent_registration():
	svc = _load("svc_eme_agent", PACKAGE_DIR / "service.py").EmergencyManagementService()
	agent = svc.register_agent("ag1", "t1", "Incident Coordinator", "claude_code", "incident_coordinator", "coordination scope")
	assert agent["role"] == "incident_coordinator"


def test_batch_requires_bytewax():
	svc = _load("svc_eme_batch", PACKAGE_DIR / "service.py").EmergencyManagementService()
	result = svc.validate_batch("t1", 5)
	assert result["processor"] == "bytewax"
	with pytest.raises(PermissionError):
		svc.validate_batch("t1", 5, event_stream="kafka")


def test_tenant_isolation():
	svc = _load("svc_eme_iso", PACKAGE_DIR / "service.py").EmergencyManagementService()
	svc.declare_incident("inc1", "ta", "natural_disaster", "minor", "loc-a", "cmdr-a", "desc-a", "ev")
	svc.declare_incident("inc1", "tb", "security_threat", "major", "loc-b", "cmdr-b", "desc-b", "ev")
	assert svc.dashboard_summary("ta")["incident_count"] == 1
	assert svc.dashboard_summary("tb")["incident_count"] == 1
