# Transport Scheduling — User Guide

**Capability ID**: `transport_sch` | **Domain**: `transport` | **Version**: `1.1.0`
**© 2025 Datacraft** | www.datacraft.co.ke

---

## Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quickstart](#quickstart)
4. [Core Concepts](#core-concepts)
5. [Schedule Lifecycle](#schedule-lifecycle)
6. [Driver Shift Planning](#driver-shift-planning)
7. [Vehicle Assignment](#vehicle-assignment)
8. [Charter Management](#charter-management)
9. [Conflict Detection and Resolution](#conflict-detection-and-resolution)
10. [Compliance Monitoring](#compliance-monitoring)
11. [Analytics and Reporting](#analytics-and-reporting)
12. [Disruption Management](#disruption-management)
13. [Advanced: Versioning and Rollback](#advanced-versioning-and-rollback)
14. [Advanced: Driver Preferences and Wellbeing](#advanced-driver-preferences-and-wellbeing)
15. [Advanced: Dynamic Charter Pricing](#advanced-dynamic-charter-pricing)
16. [Advanced: GTFS Export](#advanced-gtfs-export)
17. [Advanced: Notification Escalation](#advanced-notification-escalation)
18. [Advanced: Compliance Audit Pack](#advanced-compliance-audit-pack)
19. [Advanced: Multi-Schedule Capacity Summary](#advanced-multi-schedule-capacity-summary)
20. [Policy Rules Reference](#policy-rules-reference)
21. [Streaming Events](#streaming-events)
22. [Composability](#composability)
23. [Configuration Reference](#configuration-reference)

---

## Overview

`transport_sch` is the central scheduling runtime for the APG transport domain. It manages:

- **Trip schedules** — draft → published lifecycle with hard conflict gates
- **Driver rosters** — shift creation, HOS (hours-of-service) validation, tachograph compliance
- **Vehicle assignments** — double-booking detection at assignment time
- **Charter bookings** — school, corporate, tourist, medical, airport transfers
- **Conflict detection** — automated scanning, manual recording, and resolution tracking
- **Analytics** — KPI dashboards, SLA reports, capacity gap analysis, compliance packs

All operations are tenant-scoped. Cross-tenant access is denied by the policy engine.

---

## Installation

```bash
pip install apg-transport-sch
```

Or in development:

```bash
cd capabilities/transport/sch
pip install -e .
```

---

## Quickstart

```python
import asyncio
from apg_transport_sch import TransportSchedulingService

svc = TransportSchedulingService(tenant_id="acme", actor_id="ops@acme.com")

async def main():
    # Create a weekly route schedule
    result = await svc.create_schedule_async(
        service_type="load_schedule",
        routes=[
            {"route_id": "R001", "origin": "Nairobi", "destination": "Mombasa"},
            {"route_id": "R002", "origin": "Nairobi", "destination": "Kisumu"},
        ],
        frequency="daily",
        start_date="2026-07-01",
        end_date="2026-07-31",
        created_by="ops@acme.com",
        tenant_id="acme",
    )
    schedule_id = result["schedule"]["id"]
    print(f"Created schedule: {schedule_id}")

    # Plan driver shifts
    plan = await svc.driver_shift_planning(
        drivers=[
            {"driver_id": "D001", "available_hours": 9.0, "preferred_shift": "day_shift"},
            {"driver_id": "D002", "available_hours": 8.5, "preferred_shift": "day_shift"},
        ],
        shifts=[
            {"shift_id": "SHF-001", "shift_type": "day_shift", "start": "2026-07-01T06:00:00", "end": "2026-07-01T14:00:00", "hours": 8.0},
            {"shift_id": "SHF-002", "shift_type": "day_shift", "start": "2026-07-01T06:00:00", "end": "2026-07-01T14:00:00", "hours": 8.0},
        ],
        constraints={"max_hours_per_day": 9.0, "tacho_required": True},
        schedule_id=schedule_id,
        tenant_id="acme",
    )
    print(f"Assigned {plan['shifts_assigned']} shifts, {len(plan['violations'])} violations")

    # Check and publish
    pub = await svc.schedule_publish(schedule_id, notify_drivers=True, tenant_id="acme")
    print(f"Published: {pub['schedule']['status']}, notifications: {pub['notifications_sent']}")

asyncio.run(main())
```

---

## Core Concepts

### Tenant Isolation
Every operation carries a `tenant_id`. The policy engine denies any request without a tenant context and blocks cross-tenant writes. Instantiate the service with the tenant ID for the session, or pass it explicitly per call.

### Policy Rules
Before any write operation, `_enforce()` evaluates the request against the capability contract rules. A `PermissionError` is raised with the denial reason if any rule matches. See [Policy Rules Reference](#policy-rules-reference) for the full list.

### Audit Trail
Every state mutation emits an audit event via `_audit()`. Events are accumulated in `service.audit_events` and include `tenant_id`, `event_type`, `reference_id`, and `processor: bytewax`. In production these flow onto the `apg.transport.scheduling.lifecycle` Bytewax stream.

---

## Schedule Lifecycle

Schedules follow this state machine:

```
draft → published → in_progress → completed
                 ↘ cancelled
                 ↘ rescheduled
                 ↘ on_hold
```

### Creating a Schedule

```python
# Synchronous
svc.create_schedule(
    schedule_id="SCH-001",
    tenant_id="acme",
    schedule_type="load_schedule",   # must be in SUPPORTED_SCHEDULE_TYPES
    start_date="2026-07-01",
    end_date="2026-07-31",
    optimisation_mode="balanced",
    created_by="ops@acme.com",
)

# Async (creates schedule + vehicle stubs for multiple routes)
await svc.create_schedule_async(
    service_type="load_schedule",
    routes=[{"route_id": "R001", "origin": "A", "destination": "B"}],
    frequency="daily",
    tenant_id="acme",
)
```

### Publishing a Schedule

Publication is hard-blocked if any conflict remains unresolved. Use the async `schedule_publish` method for the full flow (conflict check + publish + driver notifications):

```python
result = await svc.schedule_publish("SCH-001", notify_drivers=True, tenant_id="acme")
```

If conflicts exist, `schedule_publish` raises `ValueError` with details. Resolve all conflicts first using `resolve_conflict(...)`.

---

## Driver Shift Planning

### Supported Shift Types
`day_shift`, `night_shift`, `split_shift`, `rest_day`, `on_call`, `overtime`, `bank_holiday`

### Regulatory Limits (EU EC 561/2006)
| Limit | Value |
|-------|-------|
| Max daily driving | 9.0 h |
| Max weekly driving | 56.0 h |
| Max fortnightly | 90.0 h |
| Min daily rest | 11.0 h |

### Bulk Shift Planning

```python
plan = await svc.driver_shift_planning(
    drivers=[{"driver_id": "D001", "available_hours": 9.0, "preferred_shift": "day_shift"}],
    shifts=[{"shift_id": "SHF-001", "shift_type": "day_shift",
             "start": "2026-07-01T06:00:00", "end": "2026-07-01T14:00:00", "hours": 8.0}],
    constraints={"max_hours_per_day": 9.0, "tacho_required": True},
    schedule_id="SCH-001",
    tenant_id="acme",
)
# plan["violations"] lists shifts that couldn't be assigned and why
```

### Shift Swap Approval

```python
swap = await svc.shift_swap_approve(
    shift_id="SHF-001",
    requesting_driver_id="D001",
    replacement_driver_id="D002",
    approved_by="supervisor@acme.com",
    tenant_id="acme",
)
```

### Cancelling a Shift

```python
result = await svc.cancel_shift("SHF-001", reason="driver_sick", tenant_id="acme")
```

---

## Vehicle Assignment

### Assigning Vehicles

```python
result = await svc.vehicle_assignment(
    schedule_id="SCH-001",
    vehicles=[
        {"vehicle_id": "VEH-001", "route_id": "R001", "from": "2026-07-01", "until": "2026-07-31"},
        {"vehicle_id": "VEH-002", "route_id": "R002", "from": "2026-07-01", "until": "2026-07-31"},
    ],
    tenant_id="acme",
)
# result["double_booking_warnings"] lists any vehicles already assigned
```

Double-booking is detected at assignment time and flagged. The policy rule `double_booking_denied` will deny the assignment if `double_booking_detected=True` is passed explicitly. In the async bulk path, double-booking warnings are returned without denial to allow ops review.

---

## Charter Management

### Supported Charter Types
`school_charter`, `corporate_charter`, `event_charter`, `tourist_charter`, `airport_transfer`, `medical_transport`, `funeral_transport`

### Creating a Charter Booking

```python
booking = await svc.charter_booking(
    client_id="CLIENT-001",
    origin="Nairobi CBD",
    destination="JKIA",
    date="2026-07-15",
    vehicle_type="minibus",
    distance_km=25.0,
    driver_id="D001",
    tenant_id="acme",
)
# booking["total_cost_usd"] = distance × rate + 10% fuel surcharge
```

Charter rates (USD/km): minibus 1.20, bus 2.50, luxury_coach 4.00, sedan 0.80, suv 1.10, truck 3.20.

### Dynamic Charter Pricing

For surge pricing with demand index and lead-time adjustments:

```python
price = await svc.charter_dynamic_price(
    vehicle_type="bus",
    distance_km=200.0,
    lead_time_days=1,
    demand_index=1.3,
    vehicle_utilisation_pct=80.0,
    tenant_id="acme",
)
# price["total_price"] applies demand, lead-time, and utilisation multipliers
```

---

## Conflict Detection and Resolution

### Automated Detection

```python
check = await svc.schedule_conflict_check("SCH-001", tenant_id="acme")
# check["new_conflicts_detected"] — count of conflicts found
# check["publishable"] — True only when open conflict count == 0
```

Detects: driver double-booking (same driver in multiple shifts), vehicle double-booking (same vehicle in multiple assignments).

### Recording a Conflict Manually

```python
svc.record_conflict(
    conflict_id="CFT-001",
    tenant_id="acme",
    schedule_id="SCH-001",
    conflict_type="driver_hours_breach",
    resource_id="D001",
    detected_at="2026-07-01T08:00:00",
)
```

### Resolving a Conflict

```python
svc.resolve_conflict(
    conflict_id="CFT-001",
    tenant_id="acme",
    resolved_at="2026-07-01T10:00:00",
    resolution_notes="Shifted driver to afternoon slot",
)
```

---

## Compliance Monitoring

### Driver HOS Compliance

```python
report = await svc.driver_hours_compliance("D001", "2026-07", tenant_id="acme")
# report["compliant"] — True/False
# report["violations"] — list of violation descriptions
```

### Tachograph Compliance Report

```python
report = await svc.tachograph_compliance_report("D001", "2026-07", tenant_id="acme")
# report["compliance_rate_pct"] — % tacho-compliant shifts
```

### Full Compliance Audit Pack

For regulatory submission:

```python
pack = await svc.compliance_audit_pack("2026-Q2", tenant_id="acme")
# Includes: HOS rates, tacho rates, open conflicts, charter confirmations, audit event log
```

---

## Analytics and Reporting

### Dashboard Summary

```python
svc.dashboard_summary("acme")
```

### KPI Summary Card

```python
await svc.schedule_kpi_summary(tenant_id="acme")
```

### Period Analytics

```python
await svc.schedule_analytics("2026-07", tenant_id="acme")
await svc.schedule_analytics_detail("2026-07", tenant_id="acme")
```

### Capacity Planning

```python
plan = await svc.capacity_planning(
    "2026-Q3",
    demand_forecast={"trips_per_day": 50, "peak_vehicles_needed": 20, "peak_drivers_needed": 25},
    tenant_id="acme",
)
# plan["capacity_sufficient"], plan["vehicle_gap"], plan["driver_gap"], plan["recommendations"]
```

### Passenger Load Forecast

```python
forecast = await svc.passenger_load_forecast(
    "SCH-001",
    historical_load_avg=85.0,
    growth_rate_pct=7.0,
    horizon_weeks=8,
    tenant_id="acme",
)
```

### Deviation Alerts

```python
alert = await svc.schedule_deviation_alert(
    "SCH-001",
    actual_departure="2026-07-01T06:15:00",
    planned_departure="2026-07-01T06:00:00",
    threshold_minutes=10,
    tenant_id="acme",
)
# alert["alert_raised"] — True when |deviation| >= threshold
# alert["severity"] — "low" | "medium" | "high"
```

### SLA On-Time Performance

```python
sla = await svc.schedule_sla_report("SCH-001", on_time_threshold_minutes=5, tenant_id="acme")
# sla["on_time_pct"], sla["p50_deviation_minutes"], sla["p95_deviation_minutes"]
```

### Schedule Comparison

```python
comparison = await svc.schedule_compare("SCH-001", "SCH-002", tenant_id="acme")
```

### Multi-Schedule Capacity Summary

```python
summary = await svc.multi_schedule_capacity_summary(
    ["SCH-001", "SCH-002", "SCH-003"],
    tenant_id="acme",
)
# summary["totals"] — fleet-wide aggregates
# summary["breakdown"] — per-schedule breakdown
```

---

## Disruption Management

```python
disruption = await svc.schedule_disruption_management(
    "DIS-001",
    disruption_type="vehicle_breakdown",
    affected_schedule_id="SCH-001",
    severity="high",
    tenant_id="acme",
)
# disruption["mitigation_actions"] — recommended ops actions
# disruption["conflicts_raised"] — count of conflicts automatically raised
```

Supported disruption types: `vehicle_breakdown`, `driver_unavailable`, `traffic_incident`, `weather_event`.

---

## Advanced: Versioning and Rollback

Snapshot a schedule before making changes, then roll back if needed:

```python
# Before any mutation — snapshot current state
snapshot = await svc.schedule_version_snapshot("SCH-001", changed_by="ops@acme.com", tenant_id="acme")
v = snapshot["version"]  # e.g. 1

# Make changes...
svc.publish_schedule("SCH-001", "acme")

# Roll back to version 1 if something went wrong
rolled = await svc.schedule_rollback("SCH-001", version=v, rolled_back_by="ops@acme.com", tenant_id="acme")
```

---

## Advanced: Driver Preferences and Wellbeing

### Recording Preferences

```python
await svc.driver_preference_update(
    "D001",
    preferences={
        "preferred_shift_type": "day_shift",
        "max_consecutive_days": 5,
        "requested_days_off": ["2026-07-04", "2026-07-05"],
        "max_weekly_hours": 45.0,
        "preferred_routes": ["R001", "R002"],
    },
    tenant_id="acme",
)
```

### Wellbeing Score

```python
score = await svc.driver_wellbeing_score("D001", tenant_id="acme")
# score["wellbeing_score"] — 0–100
# score["rating"] — "green" (>=80) | "amber" (60–79) | "red" (<60)
# score["components"] — breakdown of overtime, rest, utilisation, variety
```

---

## Advanced: Dynamic Charter Pricing

```python
price = await svc.charter_dynamic_price(
    vehicle_type="luxury_coach",
    distance_km=500.0,
    lead_time_days=0,        # same-day — 1.3x lead factor
    demand_index=1.5,        # peak demand
    vehicle_utilisation_pct=90.0,  # fleet near-full — 0.95 util factor
    tenant_id="acme",
)
# price["total_price"] reflects all multipliers + 10% fuel surcharge
```

---

## Advanced: GTFS Export

Export a published schedule as a GTFS Static feed:

```python
gtfs = await svc.gtfs_export(
    "SCH-001",
    agency_name="Acme Transit",
    agency_url="https://acme.ke",
    tenant_id="acme",
)
# gtfs["agency"], ["routes"], ["trips"], ["stop_times"] — GTFS feed records
# gtfs["export_ref"] — download path
```

Only published schedules may be exported; raises `ValueError` for other statuses.

---

## Advanced: Notification Escalation

Define a multi-step escalation ladder for unacknowledged notifications:

```python
ladder = await svc.notification_escalation_ladder(
    "NTF-001",
    escalation_steps=[
        {"delay_minutes": 5,  "channel": "sms",      "recipient_id": "D001"},
        {"delay_minutes": 15, "channel": "call",     "recipient_id": "supervisor@acme.com"},
        {"delay_minutes": 30, "channel": "incident", "recipient_id": "ops_manager@acme.com"},
    ],
    tenant_id="acme",
)
# ladder["steps"] contains step IDs and statuses for the escalation poller
```

---

## Advanced: Compliance Audit Pack

```python
pack = await svc.compliance_audit_pack("2026-H1", tenant_id="acme")
# Covers: HOS compliance %, tacho compliance %, open conflicts,
#         charter confirmation %, audit event distribution by type
```

---

## Advanced: Multi-Schedule Capacity Summary

```python
summary = await svc.multi_schedule_capacity_summary(
    ["SCH-001", "SCH-002", "SCH-003"],
    tenant_id="acme",
)
# summary["totals"]["unique_drivers"], ["open_conflicts"], ["vehicles"]
```

---

## Policy Rules Reference

| Rule Name | Triggered When | Effect |
|-----------|----------------|--------|
| `tenant_context_required` | `tenant_context_present=False` | deny |
| `scheduling_write_requires_policy` | write without `policy_attached=True` | deny |
| `schedule_type_supported` | unknown schedule type | deny |
| `shift_type_supported` | unknown shift type | deny |
| `driver_hours_breach_denied` | `driver_hours_compliant=False` | deny |
| `tacho_compliance_required` | `tacho_compliant=False` | deny |
| `double_booking_denied` | `double_booking_detected=True` | deny |
| `charter_type_supported` | unknown charter type | deny |
| `charter_customer_confirmation_required` | `customer_confirmed=False` | deny |
| `publish_blocked_on_conflict` | `unresolved_conflicts_present=True` | deny |
| `cross_tenant_schedule_denied` | `cross_tenant_access=True` | deny |
| `scheduling_agent_runtime_supported` | unknown agent runtime | deny |
| `scheduling_agent_role_supported` | unknown agent role | deny |
| `privileged_scheduling_agent_action_requires_human_approval` | privileged scope without approval | deny |
| `charter_vehicle_inspection_required` | `vehicle_inspected=False` on dispatch | deny |

---

## Streaming Events

All events are published to `apg.transport.scheduling.lifecycle` via Bytewax:

| Event | Trigger |
|-------|---------|
| `schedule_created` | `create_schedule` |
| `schedule_published` | `publish_schedule` |
| `shift_assigned` | `create_shift` |
| `vehicle_assigned` | `assign_vehicle` |
| `charter_confirmed` | `create_charter` |
| `conflict_detected` | `record_conflict` |
| `conflict_resolved` | `resolve_conflict` |
| `schedule_optimised` | `schedule_optimise_ml` |
| `scheduling_agent_registered` | `register_scheduling_agent` |
| `schedule_disruption_logged` | `schedule_disruption_management` |
| `capacity_plan_created` | `capacity_planning` |
| `schedule_deviation_alert_raised` | `schedule_deviation_alert` |
| `shift_swap_approved` | `shift_swap_approve` |
| `gtfs_export_generated` | `gtfs_export` |
| `compliance_audit_pack_generated` | `compliance_audit_pack` |
| `schedule_version_snapshot_created` | `schedule_version_snapshot` |
| `schedule_rolled_back` | `schedule_rollback` |
| `driver_preference_updated` | `driver_preference_update` |
| `charter_dynamic_price_calculated` | `charter_dynamic_price` |
| `notification_escalation_ladder_registered` | `notification_escalation_ladder` |

---

## Composability

| Downstream Capability | What `transport_sch` Provides |
|-----------------------|-------------------------------|
| `transport_dis` | Driver and vehicle availability windows |
| `transport_fle` | Vehicle utilisation data from charter/assignment records |
| `transport_mai` | Consumes maintenance blackout windows as schedule blocks |
| `transport_rou` | Route IDs referenced in vehicle assignments |
| `ntfy` | Notification events (schedule_published, conflict_alert) |

Reference in APG composition files:

```apg
use transport_sch;
```

---

## Configuration Reference

All keys are tenant-scoped and set via the `conf` capability or `TRANSPORT_SCH_*` env vars.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `schedules.advance_planning_days` | int | 14 | How far ahead to plan |
| `schedules.auto_publish_enabled` | bool | false | Auto-publish draft schedules |
| `shifts.max_daily_hours` | float | 10 | Per-shift hour cap |
| `shifts.max_weekly_hours` | float | 56 | Weekly hour cap |
| `shifts.tacho_compliance_enabled` | bool | true | Enforce tacho check |
| `shifts.break_rules_enforced` | bool | true | Enforce mandatory breaks |
| `charters.customer_confirmation_required` | bool | true | Block unconfirmed charters |
| `charters.vehicle_inspection_required` | bool | true | Require pre-departure inspection |
| `optimisation.default_mode` | str | balanced | Default optimisation objective |
| `optimisation.auto_optimise_on_publish` | bool | true | Run optimisation on publish |
| `conflicts.block_publish_on_conflict` | bool | true | Hard-block publish with open conflicts |
| `conflicts.auto_detection_enabled` | bool | true | Auto-scan on schedule operations |
| `notifications.advance_notice_hours` | int | 24 | Advance notice for shift reminders |
| `agents.human_approval_required_for_privileged_actions` | bool | true | Gate privileged agent actions |
