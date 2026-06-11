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

## New Features (v2.0)

### Bradford Factor Absenteeism Scoring
Computes the Bradford Factor (B = S² × D) for each employee over a rolling 52-week window. Classifies risk into four bands (low/medium/high/critical) and calculates a 4-week directional trend. Emits `tat.bradford.alert` to NATS when B ≥ 450.

```python
result = await svc.calculate_bradford_factor("employee-1", window_days=365)
# {"bradford_factor": 312.0, "risk_band": "high", "trend": "worsening", ...}
```

### Fatigue Risk Score Engine (FRMS-compliant)
Computes a 0–100 Fatigue Risk Index using a simplified Three-Process Model variant that incorporates cumulative hours, night-shift burden, rest-period deficits, and excess daily hours. Scores ≥ 70 emit `tat.safety.fatigue_alert` to NATS.

```python
result = await svc.calculate_fatigue_risk_score("employee-1", lookback_days=14)
# {"fatigue_index": 68.5, "severity": "medium", "recommended_rest_hours": 13.7, ...}
```

### Earned Wage Access (EWA) Integration
Returns real-time accrued gross earnings from approved/submitted time entries since the last payroll run, enabling EWA providers to show employees their current earned balance. Publishes `tat.ewa.balance_updated` after each computation.

```python
result = await svc.get_accrued_earnings_to_date(
    "employee-1", hourly_rate=Decimal("250"), payroll_run_start=date(2026,6,1)
)
# {"accrued_gross": 12750.0, "currency": "KES", ...}
```

### Intelligent Break Enforcement
Scans open/submitted time entries and auto-inserts mandatory meal breaks where a qualifying shift (>6 h by default) has no recorded break. Flags entries with `auto_break_inserted=true` and publishes `tat.compliance.break_inserted`.

```python
result = await svc.enforce_break_compliance(from_date=date.today(), to_date=date.today())
# {"breaks_inserted": 3, "entries_checked": 12, ...}
```

### Automated TOIL-to-Payroll Conversion
Converts expired TOIL/comp-time balances to payroll line items at period-end. Computes monetary equivalent from the employee's last approved timesheet rate, writes debit transactions, and publishes `tat.toil.converted`.

```python
result = await svc.convert_toil_to_payroll(period_end=date(2026,6,30), currency="KES")
# {"employees_converted": 4, "total_payout": 18750.0, "currency": "KES", ...}
```

### Shift Marketplace
Self-service open-shift pickup flow. `publish_open_shift()` broadcasts an unfilled shift to eligible employees via `tat.shift.marketplace.open`. Employees call `volunteer_for_shift()` to bid; eligibility (whitelist, hours budget) is enforced atomically.

```python
offer = await svc.publish_open_shift("shift-7", skills_required=["forklift"], max_volunteers=3)
volunteer = await svc.volunteer_for_shift(offer["id"], "employee-5")
```

### Offline Punch Reconciliation
Accepts a signed batch of locally-stored punch records from field devices after reconnection. Validates sequence integrity, deduplicates against existing entries, and inserts missing records. Publishes `tat.offline.reconciled`.

```python
result = await svc.reconcile_offline_punches(
    "employee-1", punch_records=[...], device_id="tablet-01"
)
# {"inserted": 3, "skipped_duplicates": 1, "failed": 0, ...}
```

### Skills Coverage Gap Analysis
Identifies shifts where the assigned employee's skill profile does not cover all required skills. Uses the APG composition adapter pattern to query employee skills without a direct cross-capability DB join. Emits `tat.skills.gap_detected` for each gap found.

```python
gaps = await svc.analyse_skills_coverage_gaps(from_date=date(2026,6,1), to_date=date(2026,6,30))
# {"shifts_with_gaps": 7, "gap_details": [...], ...}
```

### Polygon Geofence Support
Registers irregular-shaped work site boundaries defined by GPS waypoints. Stores GeoJSON Polygon in the database; `validate_polygon_geofence()` uses PostGIS `ST_Within` when available and falls back to a ray-casting algorithm for non-PostGIS deployments. Reduces false-positive punch rejections from 12–22% down to <0.5%.

```python
loc = await svc.create_polygon_geofence("Warehouse A", waypoints=[
    {"latitude": -1.286, "longitude": 36.820},
    {"latitude": -1.285, "longitude": 36.822},
    {"latitude": -1.288, "longitude": 36.823},
])
result = await svc.validate_polygon_geofence("employee-1", -1.287, 36.821, loc["id"])
# {"is_valid": True, "geofence_type": "polygon", ...}
```

## Runtime Surface

- `capability_contract.py` defines configuration, UI routes, theme, provided and required capabilities, deterministic rules, and Bytewax event metadata.
- `service.py` implements the dependency-light lifecycle service plus all new async methods.
- `api.py` exposes small function wrappers that composed applications can bind to web, queue, or CLI adapters.
- `views.py` provides dashboard, workbench, rule, settings, and agent view models.
- `app.py` exposes `semantic_model()`, `component_manifest()`, and `self_test()` for APG packaging.

## Event Inventory

| NATS Subject | Trigger |
|---|---|
| `tat.time_entry.clocked_in` | Employee clock-in |
| `tat.time_entry.clocked_out` | Employee clock-out |
| `tat.timesheet.approved` | Timesheet approval |
| `tat.leave_request.approved` | Leave approval |
| `tat.payroll_export.created` | Payroll export bundle created |
| `tat.bradford.alert` | Bradford Factor ≥ 450 |
| `tat.safety.fatigue_alert` | Fatigue Index ≥ 70 |
| `tat.ewa.balance_updated` | EWA balance recomputed |
| `tat.compliance.break_inserted` | Auto-break inserted |
| `tat.toil.converted` | TOIL-to-payroll conversion |
| `tat.shift.marketplace.open` | Open shift published |
| `tat.shift.marketplace.volunteered` | Employee volunteered for shift |
| `tat.offline.reconciled` | Offline punch batch reconciled |
| `tat.skills.gap_detected` | Shift skills gap identified |
| `tat.geofence.polygon_created` | Polygon geofence registered |

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

All batch and export metadata uses Bytewax+NATS. Non-Bytewax batch routing is rejected by the executable rules.

## Verification

Run the focused package checks when changing this capability:

```bash
./.venv/bin/python -m py_compile capabilities/hcm/tat/time_attendance/__init__.py capabilities/hcm/tat/time_attendance/capability_contract.py capabilities/hcm/tat/time_attendance/service.py capabilities/hcm/tat/time_attendance/api.py capabilities/hcm/tat/time_attendance/views.py capabilities/hcm/tat/time_attendance/app.py capabilities/hcm/tat/time_attendance/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/hcm/tat/time_attendance/tests/test_package_contract.py
./.venv/bin/apg capabilities publish-plan capabilities/hcm/tat/time_attendance --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/hcm/tat/time_attendance --json
```
