"""Service tests for PPM Time & Expense Management (tex)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
	sys.path.insert(0, str(PACKAGE_DIR))


def _load(name: str, path: Path):
	# Evict generic module names that other capabilities may have cached,
	# then prepend this capability's directory so fallback imports resolve correctly.
	_pkg = str(path.parent)
	for _key in ("capability_contract", "models", "service"):
		sys.modules.pop(_key, None)
	if _pkg not in sys.path:
		sys.path.insert(0, _pkg)
	else:
		sys.path.remove(_pkg)
		sys.path.insert(0, _pkg)
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec and spec.loader
	mod = importlib.util.module_from_spec(spec)
	sys.modules[name] = mod
	spec.loader.exec_module(mod)
	return mod


def _svc():
	mod = _load(f"svc_ppm_tex_{id(object())}", PACKAGE_DIR / "service.py")
	return mod.TimeExpenseService()


def test_full_time_expense_lifecycle():
	svc = _svc()
	ts = svc.submit_timesheet("ts-1", "t1", "res-1", "proj-1", "weekly", "2026-W01", "submitted", "res-1", "mgr-1")
	entry = svc.record_time_entry("te-1", "t1", "ts-1", "proj-1", "task-1", "regular", "billable", 8.0, "2026-01-05", "Backend dev")
	ts_approval = svc.approve_timesheet("ta-1", "t1", "ts-1", "mgr-1", "approved", "Looks good", "evidence-ta")
	expense = svc.submit_expense("ex-1", "t1", "res-1", "proj-1", "travel_airfare", "USD", 20.0, "not_required", "2026-01-06", "Flight to client", "approval-ex", "evidence-ex")
	exp_approval = svc.approve_expense("ea-1", "t1", "ex-1", "mgr-1", "approved", "Approved", "evidence-ea")
	reimb = svc.process_reimbursement("rm-1", "t1", "ex-1", "res-1", "payroll", 20.0, "USD", "approval-rm", "2026-01-31")
	rate = svc.set_billing_rate("br-1", "t1", "res-1", "proj-1", "standard", 200.0, "USD", "2026-01-01", "approval-br")
	agent = svc.register_agent("ag-1", "t1", "TEX Bot", "claude_code", "timesheet_reviewer", "time and expense operations")

	assert ts["period_type"] == "weekly"
	assert entry["hours"] == 8.0
	assert entry["billable_status"] == "billable"
	assert ts_approval["status"] == "approved"
	assert expense["category"] == "travel_airfare"
	assert expense["amount"] == 20.0
	assert exp_approval["status"] == "approved"
	assert reimb["method"] == "payroll"
	assert rate["rate_type"] == "standard"
	assert agent["role"] == "timesheet_reviewer"


def test_timesheet_status_updated_on_approval():
	svc = _svc()
	svc.submit_timesheet("ts-s", "t1", "res-1", "proj-1", "weekly", "2026-W02", "submitted", "res-1", "mgr")
	svc.approve_timesheet("ta-s", "t1", "ts-s", "mgr", "approved", "", "ev")
	ts = svc.get_timesheet("ts-s", "t1")
	assert ts["status"] == "approved"


def test_billable_hours_summary():
	svc = _svc()
	svc.submit_timesheet("ts-bh", "t1", "res-1", "proj-bh", "weekly", "W01", "submitted", "res-1", "mgr")
	svc.record_time_entry("te-b1", "t1", "ts-bh", "proj-bh", "t1", "regular", "billable", 6.0, "2026-01-05", "")
	svc.record_time_entry("te-b2", "t1", "ts-bh", "proj-bh", "t2", "admin", "non_billable", 2.0, "2026-01-05", "")
	summary = svc.billable_hours_summary("t1", "proj-bh")
	assert summary["billable_hours"] == 6.0
	assert summary["non_billable_hours"] == 2.0
	assert summary["total_hours"] == 8.0


def test_duplicate_expense_rejected():
	svc = _svc()
	svc.submit_expense("ex-d1", "t1", "res-1", "proj-1", "meals_entertainment", "USD", 30.0, "uploaded", "2026-01-10", "Lunch", "ap", "ev")
	with pytest.raises(PermissionError, match="duplicate_expense_submission_denied"):
		svc.submit_expense("ex-d2", "t1", "res-1", "proj-1", "meals_entertainment", "USD", 30.0, "uploaded", "2026-01-10", "Lunch again", "ap", "ev")


def test_tenant_isolation():
	svc = _svc()
	svc.submit_timesheet("ts-a", "tenant-a", "r1", "p1", "weekly", "W01", "submitted", "r1", "mgr")
	svc.submit_timesheet("ts-a", "tenant-b", "r1", "p1", "weekly", "W01", "submitted", "r1", "mgr")
	assert svc.dashboard_summary("tenant-a")["timesheet_count"] == 1
	assert svc.dashboard_summary("tenant-b")["timesheet_count"] == 1


def test_guardrail_timesheet_requires_project():
	svc = _svc()
	with pytest.raises(PermissionError, match="timesheet_submission_requires_project"):
		svc.submit_timesheet("ts", "t1", "res-1", "", "weekly", "W01", "submitted", "res-1", "mgr")


def test_guardrail_negative_hours():
	svc = _svc()
	svc.submit_timesheet("ts-nh", "t1", "res-1", "proj-1", "weekly", "W01", "submitted", "res-1", "mgr")
	with pytest.raises(PermissionError, match="time_entry_hours_must_be_positive"):
		svc.record_time_entry("te-nh", "t1", "ts-nh", "proj-1", "t", "regular", "billable", -1.0, "2026-01-05", "")


def test_guardrail_expense_requires_receipt_above_threshold():
	svc = _svc()
	# Amount > 25 USD threshold, receipt still pending
	with pytest.raises(PermissionError, match="expense_above_threshold_requires_receipt"):
		svc.submit_expense("ex-r", "t1", "res-1", "proj-1", "meals_entertainment", "USD", 50.0, "pending_upload", "2026-01-05", "Expensive dinner", "ap", "ev")


def test_guardrail_reimbursement_requires_approval():
	svc = _svc()
	with pytest.raises(PermissionError, match="personal_expense_reimbursement_requires_approval"):
		svc.process_reimbursement("rm", "t1", "ex-1", "res-1", "payroll", 100.0, "USD", "", "2026-01-31")


def test_batch_requires_bytewax():
	svc = _svc()
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		svc.validate_batch("t1", 3, event_stream="sqs")


def test_api_layer():
	mod = _load("api_ppm_tex", PACKAGE_DIR / "api.py")
	ts = mod.submit_timesheet({"tenant_id": "api-t", "timesheet_id": "api-ts", "resource_id": "r1", "project_id": "p1", "submitted_by": "r1", "reviewer_id": "mgr"})
	dashboard = mod.dashboard({"tenant_id": "api-t"})
	assert ts["period_type"] == "weekly"
	assert dashboard["timesheet_count"] == 1
