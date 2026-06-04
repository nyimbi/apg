"""Service layer tests for APG Licensing & Permits."""

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


def test_full_licensing_lifecycle():
	svc = _load("svc_lic", PACKAGE_DIR / "service.py").LicensingService()
	application = svc.submit_application("app1", "t1", "business", "applicant-1", "BRS-001", "app-ev", fee_paid=True)
	assert application["status"] == "submitted"
	fee = svc.collect_fee("fee1", "t1", "app1", "application_fee", 2000.0, "KES", "RCP-001")
	assert fee["paid"] is True
	licence = svc.issue_licence("lic1", "t1", "app1", "business", "BL-2025-001", "applicant-1", "2025-01-01", "2025-12-31", "lic-ev")
	assert licence["licence_number"] == "BL-2025-001"
	inspection = svc.schedule_inspection("ins1", "t1", "lic1", "renewal", "inspector-1", "2025-06-01", "ins-ev")
	assert inspection["outcome"] == "pending"
	svc.record_inspection_outcome("ins1", "t1", "pass", "All requirements met")
	renewal = svc.renew_licence("ren1", "t1", "lic1", "standard", "2026-12-31", "ren-ev", renewal_fee_paid=True)
	assert renewal["new_expiry_date"] == "2026-12-31"
	summary = svc.dashboard_summary("t1")
	assert summary["licence_count"] == 1
	assert summary["inspection_count"] == 1


def test_application_fee_required():
	svc = _load("svc_lic_fee", PACKAGE_DIR / "service.py").LicensingService()
	with pytest.raises(PermissionError, match="application_fee_required"):
		svc.submit_application("app1", "t1", "business", "applicant-1", "BRS-001", "app-ev", fee_paid=False)


def test_duplicate_licence_denied():
	svc = _load("svc_lic_dup", PACKAGE_DIR / "service.py").LicensingService()
	svc.submit_application("app1", "t1", "business", "applicant-1", "BRS-001", "ev", fee_paid=True)
	svc.issue_licence("lic1", "t1", "app1", "business", "BL-001", "applicant-1", "2025-01-01", "2025-12-31", "ev")
	svc.submit_application("app2", "t1", "business", "applicant-1", "BRS-001", "ev", fee_paid=True)
	with pytest.raises(PermissionError, match="duplicate_licence_denied"):
		svc.issue_licence("lic2", "t1", "app2", "business", "BL-002", "applicant-1", "2025-01-01", "2025-12-31", "ev")


def test_renewal_blocked_by_failed_inspection():
	svc = _load("svc_lic_fail", PACKAGE_DIR / "service.py").LicensingService()
	svc.submit_application("app1", "t1", "business", "a1", "BRS", "ev", fee_paid=True)
	svc.issue_licence("lic1", "t1", "app1", "business", "BL-001", "a1", "2025-01-01", "2025-12-31", "ev")
	svc.schedule_inspection("ins1", "t1", "lic1", "renewal", "inspector-1", "2025-06-01", "ev")
	svc.record_inspection_outcome("ins1", "t1", "fail", "Multiple violations found")
	with pytest.raises(PermissionError, match="inspection_fail_blocks_renewal"):
		svc.renew_licence("ren1", "t1", "lic1", "standard", "2026-12-31", "ev", renewal_fee_paid=True)


def test_revocation_notice_required():
	svc = _load("svc_lic_rev", PACKAGE_DIR / "service.py").LicensingService()
	svc.submit_application("app1", "t1", "business", "a1", "BRS", "ev", fee_paid=True)
	svc.issue_licence("lic1", "t1", "app1", "business", "BL-001", "a1", "2025-01-01", "2025-12-31", "ev")
	with pytest.raises(PermissionError, match="notice_period_required"):
		svc.revoke_licence("rev1", "t1", "lic1", "fraud", "approval-ref", "ev", notice_served=False)


def test_revocation_after_notice():
	svc = _load("svc_lic_revok", PACKAGE_DIR / "service.py").LicensingService()
	svc.submit_application("app1", "t1", "business", "a1", "BRS", "ev", fee_paid=True)
	svc.issue_licence("lic1", "t1", "app1", "business", "BL-001", "a1", "2025-01-01", "2025-12-31", "ev")
	revocation = svc.revoke_licence("rev1", "t1", "lic1", "fraud", "approval-ref", "rev-ev", notice_served=True)
	assert revocation["notice_served"] is True
	assert svc.licences[("t1", "lic1")].status == "revoked"


def test_agent_registration():
	svc = _load("svc_lic_agent", PACKAGE_DIR / "service.py").LicensingService()
	agent = svc.register_agent("ag1", "t1", "Renewal Notifier", "claude_code", "renewal_notifier", "renewal notifications")
	assert agent["role"] == "renewal_notifier"


def test_batch_requires_bytewax():
	svc = _load("svc_lic_batch", PACKAGE_DIR / "service.py").LicensingService()
	result = svc.validate_batch("t1", 5)
	assert result["processor"] == "bytewax"
