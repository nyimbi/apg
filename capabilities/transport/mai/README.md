# Vehicle Maintenance

## Overview
The Vehicle Maintenance capability manages preventive and corrective maintenance job scheduling, workshop bay allocation, parts inventory and ordering, warranty tracking, vehicle inspections with digital signature capture, and roadworthiness certificate management. It enforces pre-dispatch safety checks and blocks operation of expired-MOT or unsafe vehicles.

## Capability ID
`transport_mai`

## Provides
- preventive_maintenance_schedule_workflow: Interval-based (km/days) PM scheduling
- workshop_management_workflow: Bay allocation and capacity management
- parts_inventory_workflow: Parts ordering, receiving, and stock management
- warranty_tracking_workflow: Manufacturer and extended warranty claim management
- roadworthiness_compliance_workflow: MOT, NCOP, TÜV, and similar certificate issuance

## Requires
- auth, audl, mten, conf: Core platform services
- ntfy: MOT expiry and job status notifications
- wflo: Maintenance job state machine
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

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| expired_mot_dispatch_denied | MOT expired | deny |
| unsafe_vehicle_dispatch_denied | Vehicle unsafe | deny |
| inspection_digital_signature_required | No digital signature | deny |
| parts_quantity_positive | Quantity <= 0 | deny |
| cross_tenant_maintenance_denied | Cross-tenant write | deny |

## Data Models
- MaintenanceJob: id, vehicle_id, maintenance_type, status, priority, technician_id, estimated_hours, job_card_ref
- WorkshopAllocation: id, workshop_type, location, bay_number, job_id
- PartsOrder: id, job_id, parts_category, part_number, quantity, supplier_id
- WarrantyRecord: id, vehicle_id, warranty_type, provider, start_date, expiry_date
- VehicleInspection: id, vehicle_id, inspection_type, defects_found, digital_signature, passed
- RoadworthinessRecord: id, vehicle_id, standard, certificate_number, expires_at

## Streaming Events
- maintenance_job_created, maintenance_job_completed, parts_ordered, warranty_claimed
- inspection_completed, roadworthiness_certificate_issued, maintenance_schedule_generated

## Edge Cases Handled
- MOT expiry check is mandatory pre-dispatch — no bypass available
- Digital signature is required for all inspection types including pre/post-trip checks
- Unsafe vehicle flag independently blocks dispatch regardless of MOT status
- Parts quantity of zero is rejected at the rule engine
- Warranty claims flow through the `wflo` approval workflow

## Composability Notes
Receives vehicle IDs from `transport_fle`. Maintenance schedules feed into `transport_sch` for planned downtime. Parts reorder notifications are routed through `ntfy`. Roadworthiness certificates are validated by `transport_dis` pre-dispatch.
