"""Service layer tests for APG Electoral & Civil Registration."""

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


def test_full_electoral_lifecycle():
	svc = _load("svc_ele", PACKAGE_DIR / "service.py").ElectoralService()
	station = svc.assign_polling_station("ps1", "t1", "ordinary", "Polling Centre A", "Nairobi Central", "loc-ref-1", 500, "officer-1", "station-ev")
	assert station["station_type"] == "ordinary"
	election = svc.create_election("el1", "t1", "parliamentary", "2027 General Election", "2027-08-09", "2027-06-01", "Nairobi Central")
	assert election["election_type"] == "parliamentary"
	registration = svc.register_voter("reg1", "t1", "voter", "ID-001", "bio-ref-001", "Nairobi Central", "ps1", "reg-ev")
	assert registration["deduplication_status"] == "verified"
	result = svc.collate_result("res1", "t1", "el1", "ps1", "candidate-1", 350, 10, "officer-1", "form-34a")
	assert result["votes_cast"] == 350
	civil = svc.register_civil_event("ce1", "t1", "birth", "child-1", "registrar-1", "witness-1", "2025-06-01", "birth-cert")
	assert civil["registration_type"] == "birth"
	summary = svc.dashboard_summary("t1")
	assert summary["registration_count"] == 1
	assert summary["result_count"] == 1


def test_duplicate_voter_denied():
	svc = _load("svc_ele_dup", PACKAGE_DIR / "service.py").ElectoralService()
	svc.register_voter("reg1", "t1", "voter", "ID-DUP", "bio-1", "const-1", "ps1", "ev")
	with pytest.raises(PermissionError, match="deduplication_required"):
		svc.register_voter("reg2", "t1", "voter", "ID-DUP", "bio-2", "const-1", "ps1", "ev")


def test_missing_biometric_denied():
	svc = _load("svc_ele_bio", PACKAGE_DIR / "service.py").ElectoralService()
	with pytest.raises(PermissionError, match="biometric_required"):
		svc.register_voter("reg1", "t1", "voter", "ID-001", "", "const-1", "ps1", "ev")


def test_missing_national_id_denied():
	svc = _load("svc_ele_nid", PACKAGE_DIR / "service.py").ElectoralService()
	with pytest.raises(PermissionError, match="national_id_required"):
		svc.register_voter("reg1", "t1", "voter", "", "bio-ref", "const-1", "ps1", "ev")


def test_result_without_presiding_officer_denied():
	svc = _load("svc_ele_res", PACKAGE_DIR / "service.py").ElectoralService()
	svc.assign_polling_station("ps1", "t1", "ordinary", "PS1", "const-1", "loc", 300, "officer-1", "ev")
	svc.create_election("el1", "t1", "parliamentary", "Election 2027", "2027-08-09", "2027-06-01", "const-1")
	with pytest.raises(PermissionError, match="presiding_officer_required"):
		svc.collate_result("res1", "t1", "el1", "ps1", "c1", 100, 5, "", "form-34a")


def test_deduplication_record():
	svc = _load("svc_ele_dedup", PACKAGE_DIR / "service.py").ElectoralService()
	svc.register_voter("reg1", "t1", "voter", "ID-001", "bio-1", "const-1", "ps1", "ev")
	dedup = svc.run_deduplication("dd1", "t1", "reg1", "biometric_fingerprint", 0.12, False)
	assert dedup["duplicate_detected"] is False


def test_civil_event_registration():
	svc = _load("svc_ele_civil", PACKAGE_DIR / "service.py").ElectoralService()
	event = svc.register_civil_event("ce1", "t1", "marriage", "couple-1", "registrar-1", "witness-1", "2025-01-01", "marriage-cert")
	assert event["registration_type"] == "marriage"


def test_agent_lifecycle():
	svc = _load("svc_ele_agent", PACKAGE_DIR / "service.py").ElectoralService()
	agent = svc.register_agent("ag1", "t1", "Registration Officer Bot", "codex", "registration_officer", "voter registration scope")
	assert agent["role"] == "registration_officer"


def test_batch_requires_bytewax():
	svc = _load("svc_ele_batch", PACKAGE_DIR / "service.py").ElectoralService()
	result = svc.validate_batch("t1", 100)
	assert result["processor"] == "bytewax"
