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
