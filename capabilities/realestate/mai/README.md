# Facilities Maintenance

## Overview
Full CAFM-grade maintenance management: asset register with lifecycle tracking, preventive maintenance (PPM) schedules with automatic next-due calculation, corrective and emergency work orders with SLA deadline enforcement, contractor management with insurance validation, statutory inspection tracking, defect management, and SLA compliance dashboards.

## Capability ID
`realestate_mai`

## Provides
- `preventive_maintenance_scheduling`: PPM schedules at 9 frequency tiers with auto next-due
- `work_order_management`: P1–P5 priority work orders with SLA deadlines
- `contractor_management`: Insurance-validated contractor panel with type classification
- `asset_lifecycle_tracking`: 6-phase lifecycle from new to decommissioned
- `cafm_integration_bridge`: Sync adapters for 8 CAFM platforms
- `sla_monitoring`: Response and resolution SLA with breach escalation
- `inspection_management`: Statutory and periodic inspections with overdue alerting
- `defect_tracking`: Severity-graded defects linked to inspections and work orders
- `maintenance_cost_management`: Cost line capture against work orders
- `compliance_maintenance_reporting`: Statutory compliance schedule status

## Requires
| Capability | Reason |
|-----------|--------|
| `auth` | Contractor assignment authority |
| `audl` | Work order and PPM completion audit |
| `mten` | Multi-tenant isolation |
| `conf` | SLA thresholds and frequency configuration |
| `ntfy` | SLA breach, P1, statutory overdue alerts |
| `wflo` | Work order approval for large costs |
| `schd` | Generate PPM schedule forward dates |
| `comp` | Statutory compliance tracking |
| `mqeb` | Publish maintenance events |
| `moni` | Real-time SLA breach monitoring |

## Configuration
| Key | Default | Description |
|-----|---------|-------------|
| `sla.breach_alert_threshold_pct` | 80 | % of SLA used before warning |
| `ppm.auto_generate_advance_days` | 30 | Days ahead to generate PPM work orders |
| `contractors.insurance_required` | true | Block uninsured contractor assignment |
| `cafm.sync_enabled` | false | Enable CAFM sync |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|-----------|
| `/realestate/mai/assets` | GET/POST | Asset register | `assets` |
| `/realestate/mai/assets/end-of-life` | GET | EOL assets | `assets` |
| `/realestate/mai/assets/<id>/condition` | PATCH | Update condition score | `assets` |
| `/realestate/mai/assets/condition/below-threshold` | GET | Assets below condition threshold | `assets` |
| `/realestate/mai/ppm` | GET/POST | PPM schedules | `ppm` |
| `/realestate/mai/ppm/overdue` | GET | Overdue PPMs | `ppm` |
| `/realestate/mai/work-orders` | GET/POST | Work orders | `work_orders` |
| `/realestate/mai/work-orders/<id>/assign` | POST | Assign contractor | `work_orders` |
| `/realestate/mai/work-orders/<id>/close` | POST | Close (verified) | `work_orders` |
| `/realestate/mai/work-orders/<id>/checkin` | POST | Technician check-in | `work_orders` |
| `/realestate/mai/work-orders/<id>/checkout` | POST | Technician check-out | `work_orders` |
| `/realestate/mai/work-orders/near-sla-breach` | GET | WOs near SLA breach | `work_orders` |
| `/realestate/mai/contractors` | GET/POST | Contractor registry | `contractors` |
| `/realestate/mai/contractors/<id>/scorecard` | GET | Performance scorecard | `contractors` |
| `/realestate/mai/contractors/league-table` | GET | Ranked contractor table | `contractors` |
| `/realestate/mai/inspections` | POST | Create inspection | `inspections` |
| `/realestate/mai/inspections/overdue` | GET | Overdue inspections | `inspections` |
| `/realestate/mai/defects` | GET/POST | Defect tracker | `defects` |
| `/realestate/mai/sla` | GET | SLA dashboard | `sla` |
| `/realestate/mai/compliance/certificates` | GET/POST | Statutory certificates | `compliance` |
| `/realestate/mai/compliance/expiring` | GET | Expiring certificates | `compliance` |
| `/realestate/mai/compliance/properties/<id>` | GET | Property compliance status | `compliance` |
| `/realestate/mai/budgets` | POST | Set maintenance budget | `budgets` |
| `/realestate/mai/budgets/variance` | GET | Budget vs actual | `budgets` |
| `/realestate/mai/analytics/portfolio/benchmark` | GET | Portfolio benchmarking | `analytics` |
| `/realestate/mai/analytics/reactive-patterns` | GET | Reactive failure patterns | `analytics` |
| `/realestate/mai/escalation/policies` | POST | Create escalation policy | `escalations` |
| `/realestate/mai/escalation/process` | POST | Run escalation tick | `escalations` |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| `decommissioned_asset_work_order_denied` | asset decommissioned | deny |
| `p1_work_order_requires_immediate_assignment` | P1, no contractor | deny |
| `contractor_without_insurance_denied` | no valid insurance | deny |
| `sla_breach_requires_escalation` | breached, not escalated | deny |
| `work_order_completion_requires_verification` | not verified | deny |
| `statutory_inspection_overdue_triggers_alert` | overdue, no alert | deny |
| `cafm_integration_requires_configuration` | not configured | deny |

## Data Models
- `AssetCreate/Response` — asset with category, lifecycle phase, warranty, replacement cost
- `PpmScheduleCreate/Response` — frequency-based schedule with completion counter
- `WorkOrderCreate/Response` — typed work order with SLA deadlines and cost lines
- `MaintenanceContractorCreate/Response` — contractor with insurance expiry and performance stats
- `SlaCreate/Response` — SLA definition by priority with compliance rate tracking
- `InspectionCreate/Response` — inspection with type, findings, and linked defects
- `DefectCreate/Response` — severity-graded defect with photo evidence links

## Streaming Events
- `work_order_raised`, `work_order_assigned`, `work_order_completed`, `work_order_overdue`
- `ppm_schedule_generated`, `ppm_completed`, `ppm_overdue`
- `asset_registered`, `asset_status_changed`, `asset_end_of_life_alert`
- `inspection_completed`, `defect_raised`, `defect_resolved`
- `sla_breach_detected`, `contractor_registered`

## Edge Cases Handled
- P1 work orders enforce immediate contractor assignment at creation time
- Decommissioned assets block new work order creation
- PPM next-due calculation handles all 9 frequencies including `biennial`
- Uninsured contractor assignment rejected even if contractor is registered
- Statutory inspection overdue forces an alert event before allowing status update
- Work order close requires `verification_complete = True`, not just status update
- SLA deadlines calculated from priority at creation time (not assignment time)

## Composability Notes
- Asset register links to `realestate_prm` properties
- Maintenance costs post to `realestate_acc` property ledger
- Contractor registry shared with `realestate_con` contractor management
- Inspection findings create defects that may generate work orders
- Statutory certificates auto-schedule renewal inspections 60 days before expiry
- Portfolio benchmarking consumes `cost_per_sqm`, PPM, defect, and SLA metrics across properties
- Escalation policies evaluated per scheduler tick via `process_escalations()`

## New Service Methods (v1.1)

| Method | Category | Description |
|--------|----------|-------------|
| `update_asset_condition_score()` | Asset Intelligence | Set 0–100 condition score; auto-escalates lifecycle phase |
| `get_assets_below_condition_threshold()` | Asset Intelligence | Filter assets with score below threshold |
| `get_work_orders_near_sla_breach()` | SLA Management | WOs with ≥N% of SLA elapsed, sorted by urgency |
| `compute_contractor_scorecard()` | Contractor Performance | Rolling FTFR, resolution hours, breach rate per contractor |
| `get_contractor_league_table()` | Contractor Performance | Ranked contractor list by composite performance score |
| `set_maintenance_budget()` | Cost Management | Set property/year budget |
| `get_budget_vs_actual()` | Cost Management | Budget vs committed vs actual with variance |
| `checkin_work_order()` | Field Operations | GPS-stamped technician arrival; starts resolution clock |
| `checkout_work_order()` | Field Operations | Technician departure with elapsed time |
| `detect_reactive_patterns()` | Failure Analysis | Assets with repeated corrective WOs in rolling window |
| `register_compliance_certificate()` | Statutory Compliance | Register certificate with auto-renewal scheduling |
| `get_expiring_certificates()` | Statutory Compliance | Certificates expiring within N days |
| `get_property_compliance_status()` | Statutory Compliance | Full compliance snapshot per property |
| `benchmark_portfolio()` | Portfolio Analytics | Cross-property ranking with percentile positions |
| `create_escalation_policy()` | Escalation | Define multi-level escalation chain by priority |
| `process_escalations()` | Escalation | Advance due escalation levels; designed for scheduler tick |
