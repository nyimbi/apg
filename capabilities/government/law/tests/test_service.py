"""Service layer tests for APG Law Enforcement & Justice."""

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


def test_full_law_enforcement_lifecycle():
	svc = _load("svc_law", PACKAGE_DIR / "service.py").LawEnforcementService()
	incident = svc.report_incident("inc1", "t1", "theft", "OB-2025-001", "officer-1", "loc-nairobi", "complainant-1", "Phone stolen", "incident-ev")
	assert incident["ob_number"] == "OB-2025-001"
	docket = svc.open_docket("dok1", "t1", "inc1", "detective-1", "DKT-2025-001", "2025-06-01", "docket-ev")
	assert docket["status"] == "open"
	evidence = svc.log_evidence("ev1", "t1", "dok1", "digital", "Suspect phone", "custodian-1", "exhibit-A", "exhibit-store")
	assert evidence["evidence_type"] == "digital"
	custody = svc.record_custody_action("ca1", "t1", "ev1", "transferred", "lab-tech-1", "exhibit-store", "forensics-lab", "transfer-ev")
	assert custody["custody_action"] == "transferred"
	assert svc.evidence[("t1", "ev1")].current_location == "forensics-lab"
	hearing = svc.schedule_hearing("hr1", "t1", "dok1", "magistrates_court", "mention", "COURT-001", "2025-07-01", "magistrate-1")
	assert hearing["hearing_type"] == "mention"
	prosecution = svc.record_prosecution("pro1", "t1", "dok1", "DPP-2025-001", "charges_filed", "Theft contrary to section 275", "prosecutor-1", "prosecution-ev")
	assert prosecution["prosecution_status"] == "charges_filed"
	summary = svc.dashboard_summary("t1")
	assert summary["incident_count"] == 1
	assert summary["evidence_count"] == 1


def test_missing_ob_number_denied():
	svc = _load("svc_law_ob", PACKAGE_DIR / "service.py").LawEnforcementService()
	with pytest.raises(PermissionError, match="ob_number_required"):
		svc.report_incident("inc1", "t1", "theft", "", "officer-1", "loc", "complainant", "desc", "ev")


def test_unsupported_incident_type_denied():
	svc = _load("svc_law_type", PACKAGE_DIR / "service.py").LawEnforcementService()
	with pytest.raises(PermissionError, match="incident_type_not_supported"):
		svc.report_incident("inc1", "t1", "jaywalking", "OB-001", "officer-1", "loc", "complainant", "desc", "ev")


def test_chain_of_custody_breach_denied():
	svc = _load("svc_law_coc", PACKAGE_DIR / "service.py").LawEnforcementService()
	with pytest.raises(PermissionError, match="chain_of_custody_breach_denied"):
		svc.record_custody_action("ca1", "t1", "missing-evidence", "transferred", "actor", "loc-a", "loc-b", "ev")


def test_prosecution_without_dpp_denied():
	svc = _load("svc_law_dpp", PACKAGE_DIR / "service.py").LawEnforcementService()
	svc.report_incident("inc1", "t1", "fraud", "OB-001", "officer-1", "loc", "comp", "desc", "ev")
	svc.open_docket("dok1", "t1", "inc1", "detective-1", "DKT-001", "2025-01-01", "ev")
	with pytest.raises(PermissionError, match="dpp_reference_required"):
		svc.record_prosecution("pro1", "t1", "dok1", "", "charges_filed", "Fraud", "prosecutor-1", "ev")


def test_evidence_type_unsupported_denied():
	svc = _load("svc_law_evtype", PACKAGE_DIR / "service.py").LawEnforcementService()
	svc.report_incident("inc1", "t1", "theft", "OB-001", "officer-1", "loc", "comp", "desc", "ev")
	svc.open_docket("dok1", "t1", "inc1", "detective-1", "DKT-001", "2025-01-01", "ev")
	with pytest.raises(PermissionError, match="evidence_type_not_supported"):
		svc.log_evidence("ev1", "t1", "dok1", "voodoo", "Witch doctor ingredients", "custodian-1", "exhibit-Z", "store")


def test_agent_registration():
	svc = _load("svc_law_agent", PACKAGE_DIR / "service.py").LawEnforcementService()
	agent = svc.register_agent("ag1", "t1", "Docket Manager", "codex", "docket_manager", "docket operations")
	assert agent["role"] == "docket_manager"


def test_batch_requires_bytewax():
	svc = _load("svc_law_batch", PACKAGE_DIR / "service.py").LawEnforcementService()
	result = svc.validate_batch("t1", 3)
	assert result["processor"] == "bytewax"


def test_tenant_isolation():
	svc = _load("svc_law_iso", PACKAGE_DIR / "service.py").LawEnforcementService()
	svc.report_incident("inc1", "ta", "theft", "OB-A", "officer-a", "loc-a", "comp-a", "d", "ev")
	svc.report_incident("inc1", "tb", "assault", "OB-B", "officer-b", "loc-b", "comp-b", "d", "ev")
	assert svc.dashboard_summary("ta")["incident_count"] == 1
	assert svc.dashboard_summary("tb")["incident_count"] == 1
