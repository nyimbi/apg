# Medical Device Management

## Overview
Medical device lifecycle management covering device inventory with FDA UDI tracking, preventive and corrective maintenance scheduling with work orders, calibration record management, and adverse event reporting. Enforces UDI requirements for Class II/III devices, blocks use of recalled or calibration-overdue devices, and automatically escalates serious adverse events.

## Capability ID
`healthcare_dev`

## Provides
- device_inventory_management: Register and track medical devices by type, class, location, and status
- maintenance_schedule_management: Schedule preventive, corrective, calibration, and inspection work orders
- calibration_record_tracking: Record calibration results with certificate references and auto-update device status
- fda_udi_tracking: Track FDA Unique Device Identifiers (UDI) in GS1, HIBCC, and ICCBBA formats
- adverse_event_reporting: Report device malfunctions, patient injuries, and near-misses with severity tracking
- work_order_management: Work order lifecycle with technician assignment and completion documentation
- device_lifecycle_management: Full lifecycle from active to recalled, retired, or out-of-service
- regulatory_submission_support: FDA MDR warning for serious adverse events requiring 510(k) reporting

## Requires
- auth: Role-based access for biomedical engineers and clinical staff
- audl: Audit trail for all device modifications and adverse events
- mten: Multi-tenant isolation
- conf: Tenant-specific maintenance schedules and calibration intervals
- ntfy: Alerts for overdue calibration, recalls, and open serious adverse events
- wflo: Work order approval and adverse event investigation workflows
- comp: Regulatory compliance tracking for FDA requirements
- schd: Scheduled preventive maintenance reminders
- moni: Device availability and downtime monitoring
- mqeb: Event emission for downstream analytics

## Configuration

| Key | Description |
|-----|-------------|
| devices.udi_required_for_class_ii_iii | Require UDI for Class II and Class III devices |
| calibration.certificate_required | Block calibration record without certificate reference |
| calibration.overdue_alert_days | Days before due date to trigger overdue alert (default: 7) |
| adverse_events.fda_mdr_reporting_threshold | Severity threshold for FDA MDR warning (default: serious) |

## API Routes

| Method | Path | Description | Permission |
|--------|------|-------------|------------|
| GET | /api/healthcare/dev/inventory | List devices | healthcare_dev:inventory |
| POST | /api/healthcare/dev/inventory | Register device | healthcare_dev:inventory_write |
| GET | /api/healthcare/dev/inventory/<id> | Device detail | healthcare_dev:inventory |
| PUT | /api/healthcare/dev/inventory/<id>/status | Update status | healthcare_dev:inventory_write |
| GET | /api/healthcare/dev/maintenance | List maintenance | healthcare_dev:maintenance |
| POST | /api/healthcare/dev/maintenance | Schedule maintenance | healthcare_dev:maintenance |
| POST | /api/healthcare/dev/maintenance/<id>/complete | Complete work order | healthcare_dev:maintenance |
| GET | /api/healthcare/dev/calibration | Calibration records | healthcare_dev:calibration |
| POST | /api/healthcare/dev/calibration | Record calibration | healthcare_dev:calibration |
| GET | /api/healthcare/dev/adverse-events | List adverse events | healthcare_dev:adverse_events |
| POST | /api/healthcare/dev/adverse-events | Report adverse event | healthcare_dev:adverse_events_write |
| POST | /api/healthcare/dev/adverse-events/<id>/close | Close event | healthcare_dev:adverse_events_write |

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| udi_required_for_class_ii_iii | device_class_requires_udi=True, udi_present=False | deny |
| recalled_device_use_denied | operation=assign_device, device_status=recalled | deny |
| calibration_overdue_blocks_use | operation=assign_device, calibration_status=overdue | deny |
| out_of_service_device_not_assignable | operation=assign_device, device_status=out_of_service | deny |
| retired_device_not_modifiable | operation=update_device, device_status=retired | deny |
| calibration_certificate_required | operation=record_calibration, certificate_present=False | deny |
| serious_adverse_event_requires_fda_report | severity=serious, fda_mdr_initiated=False | warn |

## Data Models
- DeviceCreate/Response: device_type, device_class, serial_number, udi, status, calibration_status
- MaintenanceScheduleCreate/Response: maintenance_type, scheduled_date, estimated_hours, work_order_id, completed_at
- CalibrationRecordCreate/Response: calibration_date, next_due_date, certificate_reference, result
- AdverseEventCreate/Response: event_type, severity, patient_id, fda_mdr_reference, root_cause, corrective_action

## Streaming Events
- device_registered, device_status_changed
- maintenance_scheduled, work_order_completed
- calibration_recorded, calibration_overdue
- adverse_event_reported, device_recalled

## Edge Cases Handled
- Class II/III device registration without UDI is hard denied at rule layer
- Serious adverse events automatically move the device to in_maintenance status
- Calibration records update device.calibration_status and last_calibrated_at atomically
- Recalled and out-of-service devices cannot be assigned regardless of other conditions

## Composability Notes
Device adverse events feed into `healthcare_reg` for FDA MDR tracking. Calibration records are consumed by `healthcare_lab` instrument management. Maintenance schedules integrate with `schd` for PM reminders and with `healthcare_cli` for workflow task creation.
