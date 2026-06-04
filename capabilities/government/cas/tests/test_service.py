"""Service layer tests for APG Case Management."""

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


def test_full_case_lifecycle():
	svc = _load("svc_cas", PACKAGE_DIR / "service.py").CaseManagementService()
	case = svc.open_case("c1", "t1", "complaint", "online_portal", "citizen-1", "high", "Water supply cut", "Details here", "case-evidence")
	assert case["case_type"] == "complaint"
	assignment = svc.assign_case("a1", "t1", "c1", "officer", "officer-1", "supervisor-1", "assign-ev")
	assert assignment["assignee_id"] == "officer-1"
	sla = svc.set_sla("sla1", "t1", "c1", "urgent", "2025-12-31")
	assert sla["sla_category"] == "urgent"
	outcome = svc.record_outcome("o1", "t1", "c1", "resolved_satisfied", "Issue resolved", "approval-ref", "outcome-ev")
	assert outcome["outcome_type"] == "resolved_satisfied"
	notification = svc.send_notification("n1", "t1", "c1", "sms", "citizen-1", "Your case has been resolved")
	assert notification["sent"] is True
	summary = svc.dashboard_summary("t1")
	assert summary["case_count"] == 1
	assert summary["outcome_count"] == 1


def test_tenant_isolation():
	svc = _load("svc_cas_iso", PACKAGE_DIR / "service.py").CaseManagementService()
	svc.open_case("c1", "ta", "complaint", "online_portal", "c-a", "low", "s", "d", "ev")
	svc.open_case("c1", "tb", "enquiry", "telephone", "c-b", "medium", "s", "d", "ev")
	assert svc.dashboard_summary("ta")["case_count"] == 1
	assert svc.dashboard_summary("tb")["case_count"] == 1


def test_missing_tenant_denied():
	svc = _load("svc_cas_auth", PACKAGE_DIR / "service.py").CaseManagementService()
	with pytest.raises(PermissionError, match="tenant_context_required"):
		svc.open_case("c1", "", "complaint", "online_portal", "citizen", "low", "s", "d", "ev")


def test_unsupported_case_type_denied():
	svc = _load("svc_cas_type", PACKAGE_DIR / "service.py").CaseManagementService()
	with pytest.raises(PermissionError, match="case_type_not_supported"):
		svc.open_case("c1", "t1", "unknown_type", "online_portal", "c1", "low", "s", "d", "ev")


def test_outcome_without_approval_denied():
	svc = _load("svc_cas_out", PACKAGE_DIR / "service.py").CaseManagementService()
	svc.open_case("c1", "t1", "complaint", "online_portal", "c1", "low", "s", "d", "ev")
	with pytest.raises(PermissionError, match="outcome_approval_required"):
		svc.record_outcome("o1", "t1", "c1", "resolved_satisfied", "resolved", "", "ev")


def test_escalation_requires_supervisor():
	svc = _load("svc_cas_esc", PACKAGE_DIR / "service.py").CaseManagementService()
	svc.open_case("c1", "t1", "complaint", "online_portal", "c1", "high", "s", "d", "ev")
	with pytest.raises(PermissionError, match="supervisor_required"):
		svc.escalate_case("e1", "t1", "c1", "sla_breach", "team-lead", "", "ev")


def test_review_requires_reviewer():
	svc = _load("svc_cas_rev", PACKAGE_DIR / "service.py").CaseManagementService()
	with pytest.raises(PermissionError, match="reviewer_required"):
		svc.record_review("r1", "t1", "ref1", "", "approved", "ev")


def test_agent_lifecycle():
	svc = _load("svc_cas_agent", PACKAGE_DIR / "service.py").CaseManagementService()
	agent = svc.register_agent("ag1", "t1", "Case Router", "claude_code", "case_router", "routing scope")
	assert agent["role"] == "case_router"


def test_batch_requires_bytewax():
	svc = _load("svc_cas_batch", PACKAGE_DIR / "service.py").CaseManagementService()
	result = svc.validate_batch("t1", 3)
	assert result["processor"] == "bytewax"
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		svc.validate_batch("t1", 3, event_stream="rabbitmq")


def test_notification_unsupported_type_denied():
	svc = _load("svc_cas_notif", PACKAGE_DIR / "service.py").CaseManagementService()
	svc.open_case("c1", "t1", "complaint", "online_portal", "c1", "low", "s", "d", "ev")
	with pytest.raises(PermissionError, match="notification_type_not_supported"):
		svc.send_notification("n1", "t1", "c1", "carrier_pigeon", "citizen-1", "msg")
