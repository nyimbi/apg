# Medical Device Management

## Overview
Medical device lifecycle management covering device inventory with FDA UDI tracking, preventive and corrective maintenance scheduling with work orders, calibration record management, adverse event reporting, chain-of-custody assignment, device loan management, decontamination/sterility tracking, fleet benchmarking, manufacturer quality scorecards, warranty lifecycle alerts, multi-jurisdiction regulatory profiles, and tamper-evident NATS-backed audit replay. Enforces UDI requirements for Class II/III devices, blocks use of recalled or calibration-overdue devices, automatically escalates serious adverse events, and supports 21 CFR Part 11, ISO 13485, EU MDR 2017/745, UKCA, HC CMDR, and TGA compliance profiles.

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
- chain_of_custody_tracking: Shift-based device assignment and release with policy enforcement
- device_loan_management: Loan tracking with automatic re-qualification scheduling on return
- decontamination_record_tracking: Sterilisation cycle records with SAL classification for reusable devices
- fleet_benchmarking: Z-score outlier detection against device-type fleet averages
- manufacturer_quality_scorecard: Composite quality score per manufacturer for procurement and CAPA evidence
- warranty_lifecycle_alerts: Horizon-based warranty expiry alerts with cost tier classification
- multi_jurisdiction_regulatory_profiles: FDA, EU MDR, UKCA, Health Canada, and TGA profile overlays
- durable_audit_replay: HMAC-signed audit event streaming via NATS JetStream with tamper detection

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
- mqeb: Event emission for downstream analytics (NATS JetStream via bytewax pipeline)

## Configuration

| Key | Description |
|-----|-------------|
| devices.udi_required_for_class_ii_iii | Require UDI for Class II and Class III devices |
| calibration.certificate_required | Block calibration record without certificate reference |
| calibration.overdue_alert_days | Days before due date to trigger overdue alert (default: 7) |
| adverse_events.fda_mdr_reporting_threshold | Severity threshold for FDA MDR warning (default: serious) |
| audit.hmac_key_env | Env var holding HMAC-SHA256 signing key (default: AUDIT_HMAC_KEY) |
| streaming.nats_url_env | Env var for NATS JetStream URL (default: NATS_URL) |

## Environment Variables

| Variable | Description |
|----------|-------------|
| NATS_URL | NATS JetStream server URL for durable audit event streaming |
| AUDIT_HMAC_KEY | HMAC-SHA256 key for audit event tamper detection |
| OLLAMA_BASE_URL | Ollama server URL for ML-based anomaly detection |

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
| POST | /api/healthcare/dev/assignments | Assign device to staff | healthcare_dev:assignment_write |
| POST | /api/healthcare/dev/assignments/<id>/release | Release device | healthcare_dev:assignment_write |
| POST | /api/healthcare/dev/loans | Create device loan | healthcare_dev:loan_write |
| POST | /api/healthcare/dev/loans/<id>/return | Return loaned device | healthcare_dev:loan_write |
| POST | /api/healthcare/dev/decontamination | Record decontamination cycle | healthcare_dev:decontamination_write |
| GET | /api/healthcare/dev/benchmarks | Fleet benchmark by device type | healthcare_dev:analytics |
| GET | /api/healthcare/dev/manufacturer-scorecard | Manufacturer quality scorecard | healthcare_dev:analytics |
| GET | /api/healthcare/dev/warranty-alerts | Warranty expiry alerts | healthcare_dev:inventory |
| GET | /api/healthcare/dev/regulatory-profile/<jurisdiction> | Regulatory profile | healthcare_dev:compliance |
| POST | /api/healthcare/dev/audit/publish | Publish signed audit event | healthcare_dev:audit_write |
| GET | /api/healthcare/dev/audit/replay | Replay audit events by time window | healthcare_dev:audit |

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
| loaned_device_status_check | operation=create_device_loan, device_status not in {active,available} | deny |
| loan_return_triggers_requalification | operation=return_device_from_loan | schedule inspection + calibration |
| audit_events_hmac_signed | operation=publish_audit_event | HMAC-SHA256 signature attached |

## Data Models
- DeviceCreate/Response: device_type, device_class, serial_number, udi, status, calibration_status, warranty_expiry
- MaintenanceScheduleCreate/Response: maintenance_type, scheduled_date, estimated_hours, work_order_id, completed_at
- CalibrationRecordCreate/Response: calibration_date, next_due_date, certificate_reference, result
- AdverseEventCreate/Response: event_type, severity, patient_id, fda_mdr_reference, root_cause, corrective_action
- DeviceLoan (dict): borrower_org, loan_start, loan_end, contact, status, actual_return_at
- DecontaminationRecord (dict): cycle_type, steriliser_id, cycle_number, result, sal_classification, biological_indicator
- DeviceAssignment (dict): assignee_id, shift_id, location, status, released_at, condition_at_release

## Streaming Events (NATS JetStream subjects)
- `apg.healthcare.dev.lifecycle` — device_registered, device_status_changed, device_recalled
- `apg.healthcare.dev.lifecycle` — maintenance_scheduled, work_order_completed
- `apg.healthcare.dev.lifecycle` — calibration_recorded, calibration_overdue
- `apg.healthcare.dev.lifecycle` — adverse_event_reported, device_loaned, device_returned
- `apg.healthcare.dev.audit.<tenant_id>` — all audit events (HMAC-signed, MaxAge 7 years)
- `apg.healthcare.dev.shadow.<device_id>` — device shadow delta events (planned)

## Edge Cases Handled
- Class II/III device registration without UDI is hard denied at rule layer
- Serious adverse events automatically move the device to in_maintenance status
- Calibration records update device.calibration_status and last_calibrated_at atomically
- Recalled and out-of-service devices cannot be assigned regardless of other conditions
- Devices with active overdue calibration are blocked at assignment, not just reporting
- Loan creation fails if device is not in active or available status
- Loan return sets status to in_maintenance and flags requalification_required
- Decontamination cycles with SAL below 10^-6 are classified as high_level_disinfection only
- Audit event replay verifies HMAC signatures per-event and flags tampered records
- Fleet benchmarks with stddev=0 (all identical scores) return z_score=0 without division error

## New Methods Added (v1.1)

| Method | Description |
|--------|-------------|
| assign_device | Assign device to staff member for a shift with policy enforcement |
| release_device | Release device from assignment and record end-of-shift condition |
| warranty_expiry_alerts | Devices with warranty expiring within N days with cost tier |
| record_decontamination | Sterilisation/decontamination cycle with SAL classification |
| fleet_benchmark | Z-score comparison of devices against fleet mean for a metric |
| manufacturer_quality_scorecard | Composite quality score per manufacturer |
| regulatory_profile | Multi-jurisdiction regulatory rule profile overlay |
| publish_audit_event | HMAC-signed audit event publication to NATS JetStream |
| replay_audit_events | Time-windowed audit event retrieval with tamper detection |
| create_device_loan | Loan record with automatic return inspection scheduling |
| return_device_from_loan | Return processing with requalification trigger |

## Composability Notes
Device adverse events feed into `healthcare_reg` for FDA MDR tracking. Calibration records are consumed by `healthcare_lab` instrument management. Maintenance schedules integrate with `schd` for PM reminders and with `healthcare_cli` for workflow task creation. Manufacturer scorecards compose with procurement capabilities for vendor management. Audit replay integrates with `comp` for regulatory submission evidence packages.

---

## World-Class Enhancements (v2.0)

- **I1.** Medical Device Management — World-Class Improvements
- **I2.** Real-Time Telemetry Streaming via NATS JetStream
- **I3.** Predictive Maintenance via Ollama-Served Time-Series Models
- **I4.** UDI Barcode / QR Scanner Integration
- **I5.** Regulatory Submission Workflow Engine
- **I6.** Multi-Jurisdiction Regulatory Profile Overlay
- **I7.** Device Certificate & Documentation Vault
- **I8.** Automated Recall Impact Analysis
- **I9.** IoT Device Shadow / Digital Twin State
- **I10.** Lease & Loan Asset Tracking
- **I11.** Warranty & Contract Lifecycle Alerts
- **I12.** Shift-Based Device Assignment & Chain of Custody
- **I13.** Decontamination & Sterility Tracking
- **I14.** Comparative Benchmarking Against Fleet Averages
- **I15.** Supplier & Manufacturer Quality Scorecard

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
