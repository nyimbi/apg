# Vehicle Maintenance

## Overview
The Vehicle Maintenance capability manages preventive and corrective maintenance job scheduling, workshop bay allocation, parts inventory and ordering, warranty tracking, vehicle inspections with digital signature capture, and roadworthiness certificate management. It enforces pre-dispatch safety checks and blocks operation of expired-MOT or unsafe vehicles.

Version 2.0 adds world-class enhancements: real odometer-linked predictive alerts, breakdown SLA tracking, auto-reorder parts workflows, technician workload balancing, defect resolution with root-cause classification, compliance calendar, fleet TCO reporting, warranty claim filing, labour utilisation analytics, supplier scorecards, and parts receipt capture.

## Capability ID
`transport_mai`

## Provides
- preventive_maintenance_schedule_workflow: Interval-based (km/days) PM scheduling with odometer-linked due dates
- workshop_management_workflow: Bay allocation, technician workload balancing, and capacity management
- parts_inventory_workflow: Parts ordering, receiving, auto-reorder, stock management, and supplier scorecards
- warranty_tracking_workflow: Manufacturer and extended warranty tracking and structured claim filing
- roadworthiness_compliance_workflow: MOT, NCOP, TÜV, and similar certificate issuance with compliance calendar
- breakdown_management_workflow: Breakdown event capture, SLA tracking, and automated emergency job creation
- fleet_analytics_workflow: Vehicle health scoring, fleet TCO reporting, and labour utilisation analytics

## Requires
- auth, audl, mten, conf: Core platform services
- ntfy: MOT expiry, breakdown, and job status notifications
- wflo: Maintenance job state machine and warranty claim workflow
- moni: Workshop capacity and parts stock monitoring
- comp: Regulatory compliance (roadworthiness standards)
- mqeb: Event streaming
- schd: Job scheduling integration

## Configuration

| Key | Description | Default |
|-----|-------------|---------|
| roadworthiness.fail_dispatch_on_expired | Block dispatch if expired | true |
| inspections.digital_signature_required | Digital sign mandatory | true |
| parts.reorder_alerts_enabled | Low stock alerts | true |
| workshop.technician_skill_matching | Match skills to jobs | true |

## API Routes

| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /transport-maintenance/jobs | GET | Maintenance jobs | transport_mai:jobs |
| /transport-maintenance/workshop | GET | Workshop allocations | transport_mai:workshop |
| /transport-maintenance/parts | GET | Parts inventory | transport_mai:parts |
| /transport-maintenance/warranty | GET | Warranty records | transport_mai:warranty |
| /transport-maintenance/inspections | GET | Inspection records | transport_mai:inspections |
| /transport-maintenance/roadworthiness | GET | Roadworthiness certs | transport_mai:compliance |
| /transport-maintenance/breakdown | POST | Log breakdown event | transport_mai:jobs_write |
| /transport-maintenance/compliance-calendar | GET | Compliance deadline calendar | transport_mai:compliance |
| /transport-maintenance/fleet-tco | GET | Fleet TCO report | transport_mai:reports |
| /transport-maintenance/supplier-scorecard | GET | Supplier performance | transport_mai:parts |

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| expired_mot_dispatch_denied | MOT expired | deny |
| unsafe_vehicle_dispatch_denied | Vehicle unsafe | deny |
| inspection_digital_signature_required | No digital signature | deny |
| parts_quantity_positive | Quantity <= 0 | deny |
| cross_tenant_maintenance_denied | Cross-tenant write | deny |
| warranty_claim_requires_active_warranty | Warranty expired | deny |
| breakdown_sla_breach | SLA target exceeded | escalate |

## Data Models
- MaintenanceJob: id, vehicle_id, maintenance_type, status, priority, technician_id, estimated_hours, job_card_ref
- WorkshopAllocation: id, workshop_type, location, bay_number, job_id
- PartsOrder: id, job_id, parts_category, part_number, quantity, supplier_id, received_at
- WarrantyRecord: id, vehicle_id, warranty_type, provider, start_date, expiry_date, claim_ref
- VehicleInspection: id, vehicle_id, inspection_type, defects_found, digital_signature, passed
- RoadworthinessRecord: id, vehicle_id, standard, certificate_number, expires_at

## Service Methods (v2)

### Core (synchronous)
- `create_job()`, `update_job_status()`, `dispatch_vehicle_check()`
- `allocate_workshop()`, `order_parts()`, `record_warranty()`
- `conduct_inspection()`, `issue_roadworthiness()`
- `create_maintenance_schedule()`, `register_maintenance_agent()`
- `list_jobs()`, `list_schedules()`, `dashboard_summary()`

### Async — Original
- `schedule_service()`, `log_defect()`, `create_work_order()`, `complete_work_order()`
- `parts_inventory_check()`, `tyre_management()`, `roadworthiness_check()`
- `maintenance_history()`, `predictive_maintenance_alert()`, `cost_per_km()`
- `vehicle_health_score()`, `fleet_health_overview()`, `bulk_schedule_services()`
- `export_maintenance_data()`, `health_check()`, `warranty_expiry_check()`
- `close_job()`, `performance_kpi()`, `compliance_check()`, `predictive_maintenance()`
- `integration_external()`, `cost_analysis()`, `exception_handling()`, `bulk_operation()`
- `reporting_export()`, `customer_notification()`, `analytics_dashboard()`

### Async — New (v2)
- `record_odometer_reading(vehicle_id, km)` — time-stamped odometer history
- `get_technician_workload(technician_id)` — open jobs, backlog hours, utilisation
- `log_breakdown_event(vehicle_id, location, breakdown_type, sla_minutes)` — SLA-tracked breakdown
- `check_sla_breaches()` — scan for unresolved SLA breaches fleet-wide
- `set_parts_reorder_threshold(part_number, min_qty, reorder_qty, supplier_id)` — stock rules
- `trigger_reorder_if_low()` — auto-issue orders for low-stock parts
- `resolve_defect(defect_id, resolution_notes, root_cause_category, resolved_by)` — close defect loop
- `defect_recurrence_report(vehicle_id)` — systemic fault detection by root cause
- `get_compliance_calendar(days_ahead)` — unified MOT/inspection deadline timeline
- `fleet_tco_report(period, replacement_threshold_usd)` — TCO with replacement candidates
- `file_warranty_claim(warranty_id, job_id, defect_description, evidence_refs)` — structured claim filing
- `labour_utilisation_report(technician_id, period)` — billable hours and efficiency
- `record_parts_receipt(order_id, received_qty, quality_ok)` — receipt capture for scorecards
- `supplier_scorecard(supplier_id, period)` — on-time %, fill rate %, defect rate %, composite score

## Streaming Events
- maintenance_job_created, maintenance_job_completed, parts_ordered, warranty_claimed
- inspection_completed, roadworthiness_certificate_issued, maintenance_schedule_generated
- breakdown_event_logged, parts_auto_reorder_triggered, defect_resolved
- warranty_claim_filed, odometer_reading_recorded, parts_order_received

## Edge Cases Handled
- MOT expiry check is mandatory pre-dispatch — no bypass available
- Digital signature is required for all inspection types including pre/post-trip checks
- Unsafe vehicle flag independently blocks dispatch regardless of MOT status
- Parts quantity of zero is rejected at the rule engine
- Warranty claims validate expiry before filing
- Defect resolution requires explicit root-cause classification
- `resolve_defect` must be called before `roadworthiness_check` will clear a vehicle with prior defects

## Composability Notes
Receives vehicle IDs from `transport_fle`. Maintenance schedules feed into `transport_sch` for planned downtime. Parts reorder notifications are routed through `ntfy`. Roadworthiness certificates are validated by `transport_dis` pre-dispatch. Warranty claims route through `wflo` for approval. Breakdown events can trigger `transport_dis` fleet reassignment.

---

## World-Class Enhancements (v2.0)

- **I1.** Vehicle Maintenance — World-Class Improvement Plan
- **I2.** Technician Skill Matching & Workload Balancing
- **I3.** Real Odometer-Linked Service Due Dates
- **I4.** Breakdown Event Pipeline with SLA Tracking
- **I5.** Parts Reorder Automation with Minimum Stock Rules
- **I6.** Digital Twin Vehicle State Machine
- **I7.** Warranty Claim Auto-Filing Workflow
- **I8.** Maintenance Cost Ledger with Actual Parts Pricing
- **I9.** Multi-Vehicle Bulk Inspection Campaigns
- **I10.** Predictive Failure Model Integration
- **I11.** Labour Time Tracking with Technician Clock-In/Clock-Out
- **I12.** Supplier Performance Scorecard
- **I13.** Compliance Calendar with Automated Reminders
- **I14.** Job Dependency Graph for Complex Repairs
- **I15.** Fleet-Wide TCO (Total Cost of Ownership) Report

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
