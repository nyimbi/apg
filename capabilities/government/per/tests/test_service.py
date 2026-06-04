"""Service layer tests for APG Permits Management."""

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


def test_full_permit_lifecycle():
	svc = _load("svc_per", PACKAGE_DIR / "service.py").PermitsManagementService()
	application = svc.submit_application("app1", "t1", "building", "applicant-1", "SITE-001", "app-ev", fee_paid=True)
	assert application["permit_type"] == "building"
	permit = svc.issue_permit("per1", "t1", "app1", "building", "BP-2025-001", "applicant-1", "SITE-001", "2025-01-01", "2026-12-31", "per-ev")
	assert permit["permit_number"] == "BP-2025-001"
	condition = svc.record_condition("con1", "t1", "per1", "pre_commencement", "Submit structural drawings", "2025-02-01", "applicant-1", "cond-ev")
	assert condition["condition_type"] == "pre_commencement"
	inspection = svc.schedule_inspection("ins1", "t1", "per1", "foundation", "inspector-1", "2025-03-01", "ins-ev")
	assert inspection["inspection_type"] == "foundation"
	svc.record_inspection_outcome("ins1", "t1", "pass", "Foundation meets specifications")
	compliance = svc.record_compliance("comp1", "t1", "per1", "compliant", "officer-1", "2025-04-01", "All conditions met", "comp-ev")
	assert compliance["compliance_status"] == "compliant"
	summary = svc.dashboard_summary("t1")
	assert summary["permit_count"] == 1
	assert summary["inspection_count"] == 1


def test_application_fee_required():
	svc = _load("svc_per_fee", PACKAGE_DIR / "service.py").PermitsManagementService()
	with pytest.raises(PermissionError, match="application_fee_required"):
		svc.submit_application("app1", "t1", "building", "a1", "SITE-001", "ev", fee_paid=False)


def test_permit_without_approved_application_denied():
	svc = _load("svc_per_app", PACKAGE_DIR / "service.py").PermitsManagementService()
	with pytest.raises(PermissionError, match="approved_application_required"):
		svc.issue_permit("per1", "t1", "missing-app", "building", "BP-001", "a1", "SITE-001", "2025-01-01", "2026-12-31", "ev")


def test_duplicate_permit_denied():
	svc = _load("svc_per_dup", PACKAGE_DIR / "service.py").PermitsManagementService()
	svc.submit_application("app1", "t1", "building", "a1", "SITE-001", "ev", fee_paid=True)
	svc.issue_permit("per1", "t1", "app1", "building", "BP-001", "a1", "SITE-001", "2025-01-01", "2026-12-31", "ev")
	svc.submit_application("app2", "t1", "building", "a1", "SITE-001", "ev", fee_paid=True)
	with pytest.raises(PermissionError, match="duplicate_permit_denied"):
		svc.issue_permit("per2", "t1", "app2", "building", "BP-002", "a1", "SITE-001", "2025-01-01", "2026-12-31", "ev")


def test_unsupported_condition_type_denied():
	svc = _load("svc_per_cond", PACKAGE_DIR / "service.py").PermitsManagementService()
	svc.submit_application("app1", "t1", "building", "a1", "SITE-001", "ev", fee_paid=True)
	svc.issue_permit("per1", "t1", "app1", "building", "BP-001", "a1", "SITE-001", "2025-01-01", "2026-12-31", "ev")
	with pytest.raises(PermissionError, match="condition_type_not_supported"):
		svc.record_condition("c1", "t1", "per1", "some_unknown_condition", "Do something", "2025-02-01", "a1", "ev")


def test_enforcement_action():
	svc = _load("svc_per_enf", PACKAGE_DIR / "service.py").PermitsManagementService()
	svc.submit_application("app1", "t1", "building", "a1", "SITE-001", "ev", fee_paid=True)
	svc.issue_permit("per1", "t1", "app1", "building", "BP-001", "a1", "SITE-001", "2025-01-01", "2026-12-31", "ev")
	comp = svc.record_compliance("comp1", "t1", "per1", "major_breach", "officer-1", "2025-03-01", "Unauthorized extension", "ev")
	enforcement = svc.initiate_enforcement("enf1", "t1", "per1", "comp1", "stop_work_order", "officer-1", "Stop work immediately", "enf-ev")
	assert enforcement["action_type"] == "stop_work_order"


def test_agent_registration():
	svc = _load("svc_per_agent", PACKAGE_DIR / "service.py").PermitsManagementService()
	agent = svc.register_agent("ag1", "t1", "Permit Assessor", "codex", "permit_assessor", "assessment scope")
	assert agent["role"] == "permit_assessor"


def test_batch_requires_bytewax():
	svc = _load("svc_per_batch", PACKAGE_DIR / "service.py").PermitsManagementService()
	result = svc.validate_batch("t1", 5)
	assert result["processor"] == "bytewax"
