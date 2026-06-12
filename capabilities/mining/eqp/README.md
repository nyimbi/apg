# Equipment & Plant Management

## Overview
Manages the full lifecycle of mining fleet and processing plant equipment including registration, dispatch, maintenance work orders, preventive maintenance scheduling, pre-shift inspections, fuel consumption tracking, fault reporting, and fleet KPI reporting. Enforces equipment availability guardrails: breakdown equipment cannot be dispatched, operators must hold valid licences, and pre-shift inspections must pass before daily dispatch.

## Capability ID
`mining_eqp`

## Provides
| Service | Description |
|---|---|
| fleet_register_management | Equipment registration with unique asset numbers and lifecycle tracking |
| equipment_lifecycle_tracking | Commissioned → active → standby → decommissioned state management |
| maintenance_work_order_workflow | WO creation, approval, execution, and equipment status restoration |
| preventive_maintenance_scheduling | PM schedule attachment and trigger-based WO creation |
| equipment_dispatch_management | Dispatch with pre-shift inspection and operator licence checks |
| fuel_consumption_tracking | Fuel docket recording with variance flagging |
| equipment_kpi_reporting | Physical availability, utilisation, MTBF, MTTR |
| pre_shift_inspection_workflow | Pre-shift checklist submission with fail → MAINTENANCE auto-transition |
| fault_and_defect_management | Fault reporting with critical → BREAKDOWN auto-escalation |
| tyre_management | Tyre hours and rotation tracking via component records |

## Requires
| Capability | Reason |
|---|---|
| auth | User authentication |
| audl | Audit trail for all maintenance and dispatch events |
| mten | Multi-tenancy isolation |
| conf | Runtime configuration |
| ntfy | Breakdown alerts and PM schedule notifications |
| wflo | Work order approval workflows |
| moni | Real-time equipment availability monitoring |
| schd | PM schedule trigger integration |
| mqeb | Event streaming for dispatch and KPI dashboards |

## Configuration
| Key | Default | Description |
|---|---|---|
| dispatch.pre_shift_inspection_required | true | Inspection pass required before dispatch |
| dispatch.operator_license_check_required | true | Operator must hold valid licence |
| maintenance.work_order_approval_required | true | WO must be approved before execution |
| fuel.fuel_docket_required | true | Docket number required for all fuel records |
| fuel.fuel_variance_alert_threshold_pct | 10 | Variance from rolling average triggers flag |
| kpis.availability_target_pct | 85 | Physical availability KPI target |

## API Routes
| Path | Method | Description | Permission |
|---|---|---|---|
| /api/mining-eqp/fleet | GET/POST | List fleet/register equipment | mining_eqp:view/write |
| /api/mining-eqp/fleet/:id | GET/PUT | Get/update equipment | mining_eqp:view/write |
| /api/mining-eqp/fleet/:id/decommission | POST | Decommission equipment | mining_eqp:write |
| /api/mining-eqp/fleet/:id/dispatch | POST | Dispatch to mine area | mining_eqp:dispatch |
| /api/mining-eqp/maintenance | GET/POST | List/create work orders | mining_eqp:view/maintenance |
| /api/mining-eqp/maintenance/:id/approve | POST | Approve work order | mining_eqp:maintenance |
| /api/mining-eqp/maintenance/:id/complete | PUT | Complete work order | mining_eqp:maintenance |
| /api/mining-eqp/inspections | GET/POST | List/submit inspections | mining_eqp:view/write |
| /api/mining-eqp/fuel | GET/POST | List/record fuel dockets | mining_eqp:view/write |
| /api/mining-eqp/faults | GET/POST | List/report faults | mining_eqp:view/write |
| /api/mining-eqp/faults/:id/resolve | POST | Resolve a fault | mining_eqp:write |
| /api/mining-eqp/kpis | GET | Fleet KPI summary | mining_eqp:reports |

## Business Rules
| Rule | Condition | Effect |
|---|---|---|
| breakdown_dispatch_denied | Equipment in BREAKDOWN status | DENY dispatch |
| unlicensed_operator_dispatch_denied | Operator lacks licence | DENY dispatch |
| pre_shift_inspection_required | No passing inspection today | DENY dispatch |
| work_order_approval_required | Execute without approval | DENY |
| critical_fault_work_order_required | Critical fault without WO | DENY |
| delete_active_equipment_denied | Delete active equipment | DENY — decommission first |
| pm_schedule_required_for_commissioning | Commission without PM schedule | DENY |
| negative_fuel_quantity_denied | Negative fuel quantity | DENY |
| decommission_requires_approval | Decommission without approval | DENY |
| asset_number_unique | Duplicate asset number | DENY |

## Data Models
| Model | Key Fields |
|---|---|
| EquipmentCreate/Response | asset_number, equipment_class, make, model, lifecycle_status, dispatch_status, total_operating_hours |
| MaintenanceWorkOrderCreate/Response | maintenance_type, equipment_id, priority, approved_by, actual_hours, total_cost |
| InspectionCreate/Response | inspection_type, inspector_id, items[], overall_result, faults_found, work_order_raised |
| FuelDocketCreate/Response | fuel_type, quantity_litres, docket_number, total_cost, variance_flag |
| EquipmentFaultCreate/Response | severity, component, description, resolved, work_order_id |

## Streaming Events
- `equipment_commissioned` / `equipment_decommissioned`
- `work_order_created` / `work_order_completed`
- `equipment_breakdown_recorded`
- `equipment_dispatched`
- `fuel_docket_recorded`
- `pre_shift_inspection_submitted`
- `fault_reported` / `fault_resolved`
- `kpi_threshold_breached`

## Edge Cases Handled
- Critical fault resolution only restores AVAILABLE status if no other active critical faults remain
- Pre-shift inspection check is date-scoped to current UTC day; yesterday's pass does not qualify
- Failed inspection auto-transitions equipment to MAINTENANCE and blocks dispatch
- Fuel variance is computed against rolling average of last N dockets for same equipment
- Decommissioning equipment in BREAKDOWN status allowed (equipment must be stopped anyway)
- Duplicate asset numbers rejected atomically before any state mutation

## Composability Notes
- Breakdown events feed `mining_saf` incident and hazard workflows
- Equipment hours feed `mining_pro` shift delay and availability calculations
- Fuel consumption feeds cost accounting via financial integration
- PM schedules integrate with `schd` calendar for resource planning
- Dispatch status consumed by `mining_pro` for shift resource tracking

## World-Class Enhancements (v2.0)

1. **Predictive Failure / RUL Engine** — EWMA per sensor with Z-score drift triggers `rul_alert` before failure; cuts unplanned downtime 20–35 %.
2. **Shift-Based Availability Accounting** — `ShiftPattern` enum aligns PA/MA/utilisation to scheduled hours (IOGP/VDMA compliant), not calendar hours.
3. **MTBF/MTTR Trending with Confidence Intervals** — rolling 3- and 6-month time series with 95 % bootstrap CIs; separates signal from noise.
4. **Automated PM Escalation Rules Engine** — `escalate_overdue_pm()` auto-raises HIGH-priority WO, downgrades availability, and fires `pm_overdue_escalated` event.
5. **Digital Twin State Sync** — `EquipmentTwinState` model (GPS, RPM, payload, tyre pressure, coolant temp) wired to `mqeb` for live map visualisation.
6. **Tyre Life Cycle Management** — `TyreRecord` with TKPH calculation; `fit_tyre()`, `remove_tyre()`, `tyre_rotation()`, `tyre_life_report()`.
7. **GET Cost-per-Tonne Tracking** — correlates bucket-tooth replacements with `mining_pro` production records; surfaces fleet-vs-OEM benchmark KPI.
8. **Operator Performance Profiling** — per-operator breakdown rate, fuel over-consumption, pre-start pass rate, and dispatch-to-park duration.
9. **Spare Parts Inventory Integration** — `check_parts_availability()` validates WO parts against `invt` before `IN_PROGRESS`; emits `purchase_order_trigger` on back-order.
10. **NPV Life Cycle Cost Analysis** — full WACC-discounted NPV model replaces 60 %-ratio rule; outputs ranked replacement queue with payback period.
11. **Regulatory Compliance Matrix** — `ComplianceCertificate` model; `list_expiring_certificates()` and `compliance_matrix_report()`; auto-blocks dispatch on expiry.
12. **Fuel Anti-Fraud Detection** — cross-checks tank capacity, interval plausibility, duplicate dockets, and price tolerance; routes `fuel_fraud_alert` to security/finance.
13. **Mine Planning Schedule Integration** — `sync_dispatch_schedule()` consumes `ShiftRoster` from `schd` and produces a `DispatchPlan` with pre-populated inspections.
14. **Event-Sourced Audit Trail with Replay** — append-only `_event_log` + `replay_from_events()` for full regulatory audit trail and time-travel debugging.
15. **Multi-Site Fleet Transfer** — `TransferOrder` with `IN_TRANSIT` lock; `initiate_transfer()`, `confirm_receipt()`, `list_inter_site_transfers()`.

## New Methods

### `condition_monitoring` — sensor anomaly detection

```python
svc = EquipmentService(tenant_id="site_a")
result = await svc.condition_monitoring(
    asset_id="<uuid>",
    sensor_readings={"temperature_c": 115, "vibration_mm_s": 9.4, "oil_viscosity_cst": 102},
    monitoring_type="vibration",
    recorded_by="auto_daq",
)
# result["alert"] == True; result["anomalies"] lists breached sensors
# Routes _log_warn and persists to _condition_monitoring store
```

### `equipment_analytics` — fleet KPI rollup by period

```python
report = await svc.equipment_analytics(period="2026-05")
# Returns: active_fleet_count, breakdown_events, total_breakdown_hours,
#          total_repair_cost, fleet_pa_pct, top_failure_modes (top-5),
#          condition_monitoring_alerts, pm_completed
print(f"Fleet PA: {report['fleet_pa_pct']}%  Top fault: {report['top_failure_modes'][0]}")
```

### `replacement_recommendation` — retain / monitor / replace decision

```python
rec = await svc.replacement_recommendation(asset_id="<uuid>")
# rec["decision"] ∈ {"retain", "monitor", "replace"}
# rec["repair_cost_ratio"] — cumulative repairs / replacement_value
# rec["rationale"] — human-readable explanation for CFO / fleet planner
if rec["decision"] == "replace":
    await procurement.raise_capital_request(asset_number=rec["asset_number"])
```
