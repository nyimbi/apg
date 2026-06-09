# Time and Attendance Tracking

**Capability ID**: `tat_time_attendance` | **Domain**: `hcm` | **Version**: `2.2.0`

## Description

Time and Attendance is the APG capability packet for work policies, schedules, shifts, time entries, breaks, timesheets, leave requests, attendance exceptions, payroll exports, and attendance-focused AI agents.

## Installation

```bash
pip install apg-hcm-time_attendance
```

## Provides

- `time_policy_lifecycle`
- `work_schedule_lifecycle`
- `shift_lifecycle`
- `time_entry_lifecycle`
- `break_lifecycle`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/hcm/time-attendance/dashboard` | `tat_time_attendance:view` | Overview |
| `/hcm/time-attendance/policies` | `tat_time_attendance:manage_policies` | Setup |
| `/hcm/time-attendance/overtime-rules` | `tat_time_attendance:manage_policies` | Setup |
| `/hcm/time-attendance/schedules` | `tat_time_attendance:manage_schedules` | Planning |
| `/hcm/time-attendance/shifts` | `tat_time_attendance:manage_schedules` | Planning |
| `/hcm/time-attendance/time-entries` | `tat_time_attendance:record_time` | Operations |
| `/hcm/time-attendance/timesheets` | `tat_time_attendance:approve` | Operations |
| `/hcm/time-attendance/overtime` | `tat_time_attendance:approve_overtime` | Operations |

## Key Service Methods

- `_fetch_one()`
- `_fetch_many()`
- `_soft_delete()`
- `create_time_policy()`
- `get_time_policy()`
- `list_time_policies()`
- `update_time_policy()`
- `delete_time_policy()`
- `create_shift_schedule()`
- `get_shift_schedule()`

_(See `service.py` for complete API.)_

## Interoperability

`tat_time_attendance` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use tat_time_attendance;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `TAT_TIME_ATTENDANCE_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
