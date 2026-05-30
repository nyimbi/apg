"""Executable HCM Time and Attendance capability package tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import pytest

from capabilities.capability_contract_registry import validate_contract_shape


PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
	sys.path.insert(0, str(PACKAGE_DIR))


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def _build_lifecycle(service):
	policy = service.create_time_policy("policy-1", "tenant-test", "Standard Workweek", "Africa/Nairobi", ["mon", "tue", "wed", "thu", "fri"], 40)
	schedule = service.create_schedule("schedule-1", "tenant-test", "employee-1", policy["id"], "fixed", "2026-06-01", "2026-06-30")
	shift = service.create_shift("shift-1", "tenant-test", schedule["id"], "2026-06-01", "08:00", "17:00")
	entry = service.record_time_entry("entry-1", "tenant-test", "employee-1", shift["id"], "regular", "mobile", "2026-06-01T08:00:00+03:00", "2026-06-01T17:00:00+03:00", "device-1")
	break_record = service.record_break("break-1", "tenant-test", entry["id"], "meal", "12:00", "13:00")
	timesheet = service.submit_timesheet("timesheet-1", "tenant-test", "employee-1", "2026-06-01", "2026-06-07", [entry["id"]], "employee-1")
	approved = service.approve_timesheet(timesheet["id"], "tenant-test", "manager-1")
	leave = service.request_leave("leave-1", "tenant-test", "employee-1", "vacation", "2026-06-10", "2026-06-12", "annual leave")
	exception = service.record_exception("exception-1", "tenant-test", "employee-1", "late_arrival", "high", "late arrival", "owner-1", entry["id"])
	export = service.create_payroll_export("export-1", "tenant-test", "2026-06-01", "2026-06-07", [timesheet["id"]], "payroll-1")
	agent = service.register_attendance_agent("tenant-test", "Attendance Reviewer", "codex", "attendance_reviewer", "review attendance exceptions")
	return {"policy": policy, "schedule": schedule, "shift": shift, "entry": entry, "break": break_record, "timesheet": timesheet, "approved": approved, "leave": leave, "exception": exception, "export": export, "agent": agent}


def test_contract_shape_streaming_routes_and_agents_are_valid():
	module = _load_module("contract_attendance", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "tat_time_attendance"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "attendance_agents" in contract["provides"]
	assert "/hcm/time-attendance/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"


def test_rule_engine_blocks_missing_context_non_bytewax_and_review_gaps():
	module = _load_module("rules_attendance", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "attendance_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "record_time_entry", "geofence_verified": False, "review_recorded": False})["matched_rules"] == ["geofence_requires_review"]
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "agent_action", "privileged_action": True, "human_approved": False})["decision"] == "require_review"


def test_service_executes_attendance_lifecycle():
	service_module = _load_module("service_attendance", PACKAGE_DIR / "service.py")
	service = service_module.TimeAttendanceLifecycleService()
	records = _build_lifecycle(service)
	summary = service.dashboard_summary("tenant-test")

	assert records["policy"]["timezone"] == "Africa/Nairobi"
	assert records["schedule"]["schedule_type"] == "fixed"
	assert records["shift"]["status"] == "planned"
	assert records["entry"]["hours"] == 9.0
	assert records["break"]["break_type"] == "meal"
	assert records["approved"]["status"] == "approved"
	assert records["leave"]["status"] == "requested"
	assert records["exception"]["owner_id"] == "owner-1"
	assert records["export"]["processor"] == "bytewax"
	assert records["agent"]["role"] == "attendance_reviewer"
	assert summary["time_entry_count"] == 1
	assert summary["audit_event_count"] == 11
	assert summary["streaming"]["processor"] == "bytewax"


def test_service_guardrails_reject_invalid_actions():
	service_module = _load_module("guardrail_service_attendance", PACKAGE_DIR / "service.py")
	service = service_module.TimeAttendanceLifecycleService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.create_time_policy("policy", "", "Policy", "UTC", ["mon"], 40)
	with pytest.raises(PermissionError, match="time_policy_name_required"):
		service.create_time_policy("policy", "tenant-test", "", "UTC", ["mon"], 40)
	policy = service.create_time_policy("policy", "tenant-test", "Policy", "UTC", ["mon"], 40)
	with pytest.raises(PermissionError, match="schedule_type_not_supported"):
		service.create_schedule("schedule", "tenant-test", "employee", policy["id"], "daily", "2026-06-01", "2026-06-30")
	schedule = service.create_schedule("schedule", "tenant-test", "employee", policy["id"], "fixed", "2026-06-01", "2026-06-30")
	shift = service.create_shift("shift", "tenant-test", schedule["id"], "2026-06-01", "08:00", "17:00")
	with pytest.raises(PermissionError, match="attendance_device_required"):
		service.record_time_entry("entry", "tenant-test", "employee", shift["id"], "regular", "mobile", "2026-06-01T08:00:00+00:00")
	with pytest.raises(PermissionError, match="geofence_review_required"):
		service.record_time_entry("entry", "tenant-test", "employee", shift["id"], "regular", "mobile", "2026-06-01T08:00:00+00:00", device_id="device", geofence_verified=False)
	entry = service.record_time_entry("entry", "tenant-test", "employee", shift["id"], "regular", "mobile", "2026-06-01T08:00:00+00:00", "2026-06-01T17:00:00+00:00", "device")
	timesheet = service.submit_timesheet("timesheet", "tenant-test", "employee", "2026-06-01", "2026-06-07", [entry["id"]], "employee")
	with pytest.raises(PermissionError, match="timesheets_must_be_approved"):
		service.create_payroll_export("export", "tenant-test", "2026-06-01", "2026-06-07", [timesheet["id"]], "payroll")
	service.approve_timesheet(timesheet["id"], "tenant-test", "manager")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.create_payroll_export("export", "tenant-test", "2026-06-01", "2026-06-07", [timesheet["id"]], "payroll", "queue")
	with pytest.raises(PermissionError, match="attendance_agent_runtime_not_supported"):
		service.register_attendance_agent("tenant-test", "Agent", "unsupported", "attendance_reviewer", "review")
	with pytest.raises(PermissionError, match="attendance_exception_owner_required"):
		service.record_exception("exception", "tenant-test", "employee", "late_arrival", "high", "late")


def test_agents_batch_api_views_and_app_are_executable():
	api_module = _load_module("api_attendance", PACKAGE_DIR / "api.py")
	views = _load_module("views_attendance", PACKAGE_DIR / "views.py")
	app = _load_module("app_attendance", PACKAGE_DIR / "app.py")

	policy = api_module.create_time_policy({"tenant_id": "tenant-api", "id": "policy-api", "name": "API Policy", "timezone": "UTC", "workweek": ["mon"], "overtime_threshold_hours": 40})
	agent = api_module.register_attendance_agent({"tenant_id": "tenant-api", "name": "Compliance Reviewer", "runtime": "claude_code", "role": "compliance_reviewer"})
	batch = api_module.service().validate_batch("tenant-api", 2)
	model = views.policy_model(api_module.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert policy["id"] == "policy-api"
	assert agent["role"] == "compliance_reviewer"
	assert batch["processor"] == "bytewax"
	assert model["records"][0]["name"] == "API Policy"
	assert self_test["passed"] is True
	assert semantic["capabilities"]["tat_time_attendance"]["screens"]["agents"]["route"] == "/hcm/time-attendance/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_attendance", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["tat_time_attendance"]["streaming"]["processor"] == "bytewax"
