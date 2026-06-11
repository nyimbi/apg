# Time and Attendance — User Guide

**Capability ID**: `tat_time_attendance` | **Domain**: `hcm` | **Version**: `2.2.0`
**Copyright**: © 2025 Datacraft | **Author**: Nyimbi Odero

---

## Overview

Time and Attendance (TAT) is the APG capability packet for work policies, schedules, shifts, time entries, breaks, timesheets, leave requests, attendance exceptions, payroll exports, and attendance AI agents. It operates in two modes:

- **In-memory mode** (no DB session) — suitable for unit tests and capability sandbox validation.
- **DB mode** (asyncpg/SQLAlchemy async session injected) — full production lifecycle with NATS event emission and Bytewax stream integration.

---

## Installation

```bash
pip install apg-hcm-time-attendance
```

---

## Quick Start

```python
import asyncio
from capabilities.hcm.tat.time_attendance.service import TimeAttendanceService
from decimal import Decimal
from datetime import date

async def main():
    # In-memory mode — no database required
    svc = TimeAttendanceService()

    # Clock in
    entry = await svc.clock_in("employee-1", "tenant-acme")
    print(entry.clock_in)

    # Clock out
    closed = await svc.clock_out("employee-1", "tenant-acme")
    print(closed.total_hours)

    # Request leave
    leave = await svc.process_leave_request(
        "employee-1", "tenant-acme",
        leave_type="vacation",
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 5),
    )
    print(leave.status)

asyncio.run(main())
```

---

## Configuration

### Time Policy

A time policy defines the rules governing work and overtime for a tenant or department.

```python
policy = await svc.create_time_policy(
    name="Standard 40h",
    timezone="Africa/Nairobi",
    workweek=["Mon","Tue","Wed","Thu","Fri"],
    overtime_threshold_daily=8.0,
    overtime_threshold_weekly=40.0,
    double_time_threshold_daily=12.0,
    overtime_multiplier=1.5,
    holiday_pay_multiplier=2.0,
    min_rest_between_shifts_h=11.0,
    max_consecutive_days=6,
    max_weekly_hours=48.0,
    toil_enabled=True,
    flexi_core_start=time(10, 0),
    flexi_core_end=time(16, 0),
)
```

---

## Core Operations

### Clocking In and Out (DB mode)

```python
# DB-mode clock-in
entry = await svc._db_clock_in(
    employee_id="emp-001",
    shift_id="shift-abc",
    method="biometric",
    device_id="terminal-01",
    latitude=-1.286,
    longitude=36.820,
    biometric_confidence=0.97,
)

# DB-mode clock-out
closed = await svc._db_clock_out(entry["id"], latitude=-1.286, longitude=36.820)
```

### Breaks

```python
brk = await svc.record_break(
    time_entry_id=entry["id"],
    break_type="meal",
    break_start=datetime(2026, 6, 11, 12, 0, tzinfo=UTC),
    break_end=datetime(2026, 6, 11, 12, 30, tzinfo=UTC),
    is_paid=False,
)
```

### Timesheet Lifecycle

```python
ts = await svc.process_timesheet("emp-001", date(2026,6,1), date(2026,6,7), hourly_rate=Decimal("350"))
await svc.submit_timesheet(ts["id"])
await svc.approve_timesheet(ts["id"])  # manager action
```

### Leave Management

```python
request = await svc.request_leave(
    employee_id="emp-001",
    leave_type="sick",
    start_date=date(2026, 6, 15),
    end_date=date(2026, 6, 17),
    reason="Influenza — doctor confirmed",
    medical_cert_attached=True,
)
await svc.approve_leave_request(request["id"])
```

### Overtime

```python
breakdown = await svc.calculate_overtime(
    employee_id="emp-001",
    period_start=date(2026, 6, 1),
    period_end=date(2026, 6, 7),
    policy_id=policy["id"],
)
# {"regular_hours": 40.0, "overtime_hours": 5.5, "double_time_hours": 0.0, ...}
```

---

## Advanced Features

### Bradford Factor Scoring

The Bradford Factor (B = S² × D) is a leading indicator of disengagement risk. Use it in your HR review cycle or as a trigger for early-intervention conversations.

```python
bf = await svc.calculate_bradford_factor("emp-001", window_days=365)
print(bf["bradford_factor"], bf["risk_band"], bf["trend"])
# 180.0  medium  stable
```

Risk bands:

| Score | Band |
|-------|------|
| < 100 | low |
| 100–199 | medium |
| 200–449 | high |
| ≥ 450 | critical — NATS alert emitted |

### Fatigue Risk Score

The Fatigue Risk Index (0–100) incorporates cumulative hours, night-shift burden, rest-period shortfalls, and excess daily hours. Scores ≥ 70 automatically emit `tat.safety.fatigue_alert` to NATS.

```python
frm = await svc.calculate_fatigue_risk_score("emp-001", lookback_days=14)
print(frm["fatigue_index"], frm["severity"])
# 72.3  high
```

Typical use: run nightly for all employees with night shifts and surface critical cases in the safety dashboard.

### Earned Wage Access (EWA)

Expose accrued earnings to an EWA provider or your payroll module in real time.

```python
ewa = await svc.get_accrued_earnings_to_date(
    employee_id="emp-001",
    hourly_rate=Decimal("350"),
    payroll_run_start=date(2026, 6, 1),
    currency="KES",
)
print(ewa["accrued_gross"])  # 17150.0
```

Integrate with Wagestream, DailyPay, or a custom internal EWA API by subscribing to the `tat.ewa.balance_updated` NATS subject.

### Automatic Break Compliance

Scan all open or submitted entries for a date range and auto-insert mandatory meal breaks where missing. Complies with EU Working Time Directive, Kenya Employment Act, and OSHA rest rules.

```python
result = await svc.enforce_break_compliance(
    from_date=date.today(),
    to_date=date.today(),
    break_threshold_hours=6.0,
    min_break_minutes=30,
)
print(result["breaks_inserted"])
```

### TOIL-to-Payroll Conversion

At period-end, convert expired TOIL/comp-time balances to cash payroll line items automatically.

```python
conversions = await svc.convert_toil_to_payroll(period_end=date(2026, 6, 30), currency="KES")
print(conversions["total_payout"], conversions["employees_converted"])
```

### Shift Marketplace

Open unfilled shifts to volunteer pickup without manager phone calls.

```python
# Manager publishes an open shift
offer = await svc.publish_open_shift(
    shift_id="shift-999",
    skills_required=["forklift_operator"],
    max_volunteers=3,
)

# Employee volunteers
bid = await svc.volunteer_for_shift(offer["id"], "emp-042")
```

Employees subscribed to `tat.shift.marketplace.open` on NATS receive instant notification.

### Offline Punch Reconciliation

Field devices in low-connectivity areas submit punch batches on reconnection.

```python
result = await svc.reconcile_offline_punches(
    employee_id="emp-001",
    device_id="rugged-tablet-03",
    punch_records=[
        {"clock_in": "2026-06-10T06:00:00Z", "clock_out": "2026-06-10T14:00:00Z", "sequence_no": 1, "hmac": "abc..."},
        {"clock_in": "2026-06-11T06:00:00Z", "sequence_no": 2, "hmac": "def..."},
    ],
)
print(result["inserted"], result["skipped_duplicates"])
```

### Skills Coverage Gap Analysis

Identify shifts where the assigned employee's skills do not cover all the shift's requirements. Uses the APG composition adapter — no direct cross-capability SQL join.

```python
gaps = await svc.analyse_skills_coverage_gaps(
    from_date=date(2026, 6, 1),
    to_date=date(2026, 6, 30),
    department_id="dept-warehouse",
)
for g in gaps["gap_details"]:
    print(g["shift_date"], g["employee_id"], g["gap_skills"], g["coverage_pct"])
```

### Polygon Geofence

Define irregular site boundaries (warehouses, campuses, construction zones) with GPS waypoints. Validation uses PostGIS `ST_Within` where available; falls back to a ray-casting algorithm.

```python
# Create polygon geofence
loc = await svc.create_polygon_geofence(
    name="Warehouse A",
    waypoints=[
        {"latitude": -1.2860, "longitude": 36.8200},
        {"latitude": -1.2855, "longitude": 36.8220},
        {"latitude": -1.2880, "longitude": 36.8230},
        {"latitude": -1.2890, "longitude": 36.8205},
    ],
    timezone="Africa/Nairobi",
)

# Validate a clock-in punch
check = await svc.validate_polygon_geofence("emp-001", -1.2870, 36.8210, loc["id"])
print(check["is_valid"], check["geofence_type"])
# True  polygon
```

---

## Shift Scheduling

### Creating Schedules and Shifts

```python
schedule = await svc.create_shift_schedule(
    policy_id=policy["id"],
    schedule_name="Day Shift",
    schedule_type="fixed",
    effective_date=date(2026, 6, 1),
    patterns=[{"days_of_week": [0,1,2,3,4], "start_time": "08:00", "end_time": "17:00"}],
    allow_overtime=True,
)

# Auto-generate a full roster
roster = await svc.roster_generation(
    schedule_id=schedule["id"],
    period_start=date(2026, 7, 1),
    period_end=date(2026, 7, 31),
    employee_ids=["emp-001", "emp-002", "emp-003"],
)
```

### Shift Swap

```python
swap = await svc.shift_swap_request(
    requester_shift_id="shift-101",
    target_shift_id="shift-202",
    reason="Family commitment",
)
await svc.approve_shift_swap(swap["id"])
```

---

## Biometric Device Sync

Batch-import biometric punch records from hardware terminals.

```python
sync_log = await svc.biometric_device_sync(
    device_id="terminal-hq-01",
    raw_records=[
        {"employee_id": "emp-001", "clock_in": "2026-06-11T08:03:22Z", "biometric_confidence": 0.96},
        {"employee_id": "emp-002", "clock_in": "2026-06-11T08:07:44Z", "clock_out": "2026-06-11T17:01:12Z", "biometric_confidence": 0.98},
    ],
)
print(sync_log["records_created"], sync_log["records_skipped"])
```

---

## Compliance

### Real-Time Compliance Monitoring

The compliance engine integrates with NATS via Bytewax. Subscribe to `tat.compliance.violation` to receive real-time alerts for:

- Daily hours > 16 (DAILY_MAX_HOURS)
- Missing break for shift > 6 h (MINIMUM_BREAK) — auto-corrected
- Overtime without approval (OVERTIME_APPROVAL) — flagged for approval
- Rest-period shortfall < 11 h between shifts

```python
report = await svc.enforce_compliance_rules("tenant-acme")
print(report["compliance_score"], report["violations_detected"])
```

### Exception Management

```python
exc = await svc.record_exception(
    employee_id="emp-001",
    exception_type="missing_clock_out",
    severity="medium",
    description="Employee did not clock out on 2026-06-10",
)
await svc.resolve_exception(exc["id"], "Clock-out recorded manually by supervisor")
```

---

## Payroll Export

```python
export = await svc.create_payroll_export(
    period_start=date(2026, 6, 1),
    period_end=date(2026, 6, 30),
    timesheet_ids=["ts-001", "ts-002", "ts-003"],
    notes="June 2026 payroll run",
)
# Bytewax stream processor picks up the export record via NATS
```

---

## Reports

```python
# Daily summary
report = await svc.generate_attendance_report(
    report_type="daily_summary",
    from_date=date(2026, 6, 1),
    to_date=date(2026, 6, 30),
)

# Overtime by employee
ot = await svc.generate_attendance_report("overtime_report", date(2026,6,1), date(2026,6,30))

# Leave usage
lv = await svc.generate_attendance_report("leave_usage", date(2026,6,1), date(2026,6,30))

# Exception report
ex = await svc.generate_attendance_report("exception_report", date(2026,6,1), date(2026,6,30))
```

---

## NATS Event Reference

All events use subject pattern `tat.<entity>.<action>` and carry `tenant_id`, `actor_id`, and `occurred_at` in the envelope.

| Subject | Key Payload Fields |
|---|---|
| `tat.time_entry.clocked_in` | `entry_id`, `employee_id`, `clock_in` |
| `tat.time_entry.clocked_out` | `entry_id`, `employee_id`, `total_hours`, `overtime_hours` |
| `tat.timesheet.approved` | `timesheet_id`, `approved_by` |
| `tat.leave_request.approved` | `request_id`, `approved_by` |
| `tat.payroll_export.created` | `export_id`, `total_hours`, `timesheet_count` |
| `tat.bradford.alert` | `employee_id`, `bradford_factor`, `risk_band` |
| `tat.safety.fatigue_alert` | `employee_id`, `fatigue_index`, `severity` |
| `tat.ewa.balance_updated` | `employee_id`, `accrued_gross`, `currency` |
| `tat.compliance.break_inserted` | `entry_id`, `employee_id`, `break_minutes` |
| `tat.toil.converted` | `period_end`, `employee_count`, `total_payout`, `currency` |
| `tat.shift.marketplace.open` | `marketplace_id`, `shift_id`, `expires_at`, `skills_required` |
| `tat.shift.marketplace.volunteered` | `volunteer_id`, `marketplace_id`, `employee_id` |
| `tat.offline.reconciled` | `employee_id`, `device_id`, `inserted`, `skipped`, `failed` |
| `tat.skills.gap_detected` | `shift_id`, `employee_id`, `gap_skills`, `coverage_pct` |
| `tat.geofence.polygon_created` | `location_id`, `name`, `waypoint_count`, `bounding_radius_metres` |
| `tat.device.synced` | `device_id`, `log_id`, `created`, `skipped` |

---

## Testing

```bash
# Unit tests (no DB required)
uv run pytest -vxs capabilities/hcm/tat/time_attendance/tests/ci/

# Integration tests (requires PostgreSQL)
uv run pytest -vxs capabilities/hcm/tat/time_attendance/tests/integration/

# Full package contract check
uv run pytest -q capabilities/hcm/tat/time_attendance/tests/test_package_contract.py
```

---

## Verification Commands

```bash
./.venv/bin/python -m py_compile \
  capabilities/hcm/tat/time_attendance/__init__.py \
  capabilities/hcm/tat/time_attendance/capability_contract.py \
  capabilities/hcm/tat/time_attendance/service.py \
  capabilities/hcm/tat/time_attendance/api.py \
  capabilities/hcm/tat/time_attendance/views.py \
  capabilities/hcm/tat/time_attendance/app.py

./.venv/bin/pytest -q capabilities/hcm/tat/time_attendance/tests/test_package_contract.py
./.venv/bin/apg capabilities publish-plan capabilities/hcm/tat/time_attendance --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/hcm/tat/time_attendance --json
```

---

## Composition Dependencies

| Capability | Use |
|---|---|
| `auth` | Permission enforcement on every API call |
| `audl` | Immutable audit log entries |
| `mten` | Tenant isolation and row-level security |
| `ntfy` | Manager and employee notifications |
| `conf` | Capability configuration management |
| `empl` | Employee profile and contract data |
| `payr` | Payroll period and rate data for pay calculation |
| `devr` | Device registry for biometric terminal management |
| `locp` | Location and geofence policy management |
| `skil` | Skills & competency profiles (used by gap analysis) |

---

## Data Model Prefixes

All SQLAlchemy models and database tables use the `tat_` prefix to avoid cross-capability name collisions:

`tat_time_policy`, `tat_shift_schedule`, `tat_shift`, `tat_time_entry`, `tat_break`,
`tat_timesheet`, `tat_leave_policy`, `tat_leave_entitlement`, `tat_leave_request`,
`tat_overtime_request`, `tat_attendance_exception`, `tat_public_holiday`,
`tat_geofence_location`, `tat_comp_time`, `tat_roster`, `tat_payroll_export`,
`tat_shift_swap_request`, `tat_attendance_device`, `tat_biometric_sync_log`,
`tat_flexitime_balance`, `tat_shift_marketplace`, `tat_shift_volunteer`.
