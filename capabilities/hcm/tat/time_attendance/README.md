# HCM Time and Attendance

Time and Attendance is the APG capability packet for work policies, schedules, shifts, time entries, breaks, timesheets, leave requests, attendance exceptions, payroll exports, and attendance-focused AI agents.

The packet is executable without optional web, database, biometric, location, or payroll adapters. Those adapters can attach at composition time through the capability contract.

## What It Provides

- `time_policy_lifecycle` for tenant work rules, workweeks, timezones, and overtime thresholds.
- `work_schedule_lifecycle` and `shift_lifecycle` for fixed, flexible, rotating, compressed, and remote work planning.
- `time_entry_lifecycle` and `break_lifecycle` for web, mobile, kiosk, biometric, API, and import time capture.
- `timesheet_lifecycle` with submission, approval, and nonnegative hour guardrails.
- `leave_request_lifecycle` with paid, unpaid, sick, parental, bereavement, and public holiday request handling.
- `attendance_exception_workflow` for missing clock-out, late arrival, early departure, overtime, geofence, biometric, and duplicate-entry exceptions.
- `attendance_payroll_export` with approved-timesheet gates and Bytewax stream metadata.
- `attendance_agents` for Codex, Claude Code, OpenCode, and Pi-based review agents with human approval gates.
- APG Python UI routes, compact theme tokens, deterministic rules, semantic metadata, and publish-plan evidence.

## Runtime Surface

- `capability_contract.py` defines configuration, UI routes, theme, provided and required capabilities, deterministic rules, and Bytewax event metadata.
- `service.py` implements the dependency-light lifecycle service.
- `api.py` exposes small function wrappers that composed applications can bind to web, queue, or CLI adapters.
- `views.py` provides dashboard, workbench, rule, settings, and agent view models.
- `app.py` exposes `semantic_model()`, `component_manifest()`, and `self_test()` for APG packaging.

## Example

```python
from capabilities.hcm.tat.time_attendance.service import TimeAttendanceLifecycleService

service = TimeAttendanceLifecycleService()
policy = service.create_time_policy(
	"policy-1",
	"tenant-a",
	"Standard Workweek",
	"Africa/Nairobi",
	["mon", "tue", "wed", "thu", "fri"],
	40,
)
schedule = service.create_schedule("schedule-1", "tenant-a", "employee-1", policy["id"], "fixed", "2026-06-01", "2026-06-30")
shift = service.create_shift("shift-1", "tenant-a", schedule["id"], "2026-06-01", "08:00", "17:00")
entry = service.record_time_entry(
	"entry-1",
	"tenant-a",
	"employee-1",
	shift["id"],
	"regular",
	"mobile",
	"2026-06-01T08:00:00+03:00",
	"2026-06-01T17:00:00+03:00",
	device_id="device-1",
)
timesheet = service.submit_timesheet("timesheet-1", "tenant-a", "employee-1", "2026-06-01", "2026-06-07", [entry["id"]], "employee-1")
service.approve_timesheet(timesheet["id"], "tenant-a", "manager-1")
```

## Composition Notes

The capability requires authorization, audit, notification, composition configuration, workflow, employee profile, payroll period, device registry, location policy, and privacy policy capabilities. The dependency-light service records the lifecycle locally; production applications should attach durable stores, identity, policy, device, location, payroll, and audit adapters at the APG composition layer.

All batch and export metadata uses Bytewax. Non-Bytewax batch routing is rejected by the executable rules.

## Verification

Run the focused package checks when changing this capability:

```bash
./.venv/bin/python -m py_compile capabilities/hcm/tat/time_attendance/__init__.py capabilities/hcm/tat/time_attendance/capability_contract.py capabilities/hcm/tat/time_attendance/service.py capabilities/hcm/tat/time_attendance/api.py capabilities/hcm/tat/time_attendance/views.py capabilities/hcm/tat/time_attendance/app.py capabilities/hcm/tat/time_attendance/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/hcm/tat/time_attendance/tests/test_package_contract.py
./.venv/bin/apg capabilities publish-plan capabilities/hcm/tat/time_attendance --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/hcm/tat/time_attendance --json
```
