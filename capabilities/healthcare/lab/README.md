# Laboratory Information System

## Overview
Full-featured LIS capability providing lab order management, specimen tracking with chain of custody, result entry and verification, critical value alerting with mandatory acknowledgement, QC management with Westgard rule evaluation, and instrument status tracking. Critical value workflow blocks result release until notification is confirmed.

## Capability ID
`healthcare_lab`

## Provides
- lab_order_management: STAT/ASAP/routine order lifecycle from pending through resulted and verified
- specimen_tracking: Chain-of-custody tracking with barcode assignment and rejection reason documentation
- result_entry_verification: Preliminary-to-final result workflow with reference range flagging (H/HH/L/LL)
- critical_value_alerting: Automatic critical flag detection with mandatory notification and acknowledgement
- qc_management: Westgard rule evaluation (1-3s, 1-2s) with automatic QC hold on failure
- instrument_management: Instrument registry with status lifecycle and calibration tracking
- lis_integration: Event stream for downstream EMR, pharmacy, and analytics consumers
- reference_range_evaluation: Numeric result comparison against configurable reference intervals
- lab_reporting: Result report generation for FHIR DiagnosticReport export

## Requires
- auth: PHI access authorization for result data
- audl: Audit trail for all order/result/QC operations
- mten: Multi-tenant isolation
- conf: Tenant-specific configuration
- ntfy: Critical value notifications to clinicians
- wflo: Verification and QC review approval workflows
- moni: Instrument availability and turnaround time monitoring
- mqeb: Result event emission to EMR and analytics

## Configuration

| Key | Description |
|-----|-------------|
| orders.stat_turnaround_minutes | Target turnaround for STAT orders (default: 60) |
| results.critical_value_notification_required | Block result verify until notification sent |
| qc.westgard_rules_enabled | Enable 1-3s/1-2s Westgard evaluation |
| qc.qc_frequency_hours | Required QC frequency (default: 8h) |
| specimens.chain_of_custody_required | Chain of custody tracking for all specimens |

## API Routes

| Method | Path | Description | Permission |
|--------|------|-------------|------------|
| GET | /api/healthcare/lab/orders | List orders | healthcare_lab:orders |
| POST | /api/healthcare/lab/orders | Create order | healthcare_lab:orders_write |
| GET | /api/healthcare/lab/orders/<id> | Order detail | healthcare_lab:orders |
| POST | /api/healthcare/lab/orders/<id>/cancel | Cancel order | healthcare_lab:orders_write |
| GET | /api/healthcare/lab/specimens | List specimens | healthcare_lab:specimens |
| POST | /api/healthcare/lab/specimens | Collect specimen | healthcare_lab:specimens |
| POST | /api/healthcare/lab/specimens/<id>/reject | Reject specimen | healthcare_lab:specimens |
| POST | /api/healthcare/lab/specimens/<id>/receive | Receive specimen | healthcare_lab:specimens |
| GET | /api/healthcare/lab/results | List results | healthcare_lab:results |
| POST | /api/healthcare/lab/results | Enter result | healthcare_lab:results_write |
| POST | /api/healthcare/lab/results/<id>/verify | Verify result | healthcare_lab:results_write |
| GET | /api/healthcare/lab/critical-values | List critical values | healthcare_lab:critical_values |
| POST | /api/healthcare/lab/critical-values | Notify critical value | healthcare_lab:critical_values |
| POST | /api/healthcare/lab/critical-values/<id>/acknowledge | Acknowledge | healthcare_lab:critical_values |
| GET | /api/healthcare/lab/qc | List QC runs | healthcare_lab:qc |
| POST | /api/healthcare/lab/qc | Run QC | healthcare_lab:qc |
| GET | /api/healthcare/lab/instruments | List instruments | healthcare_lab:instruments |
| POST | /api/healthcare/lab/instruments | Register instrument | healthcare_lab:instruments |
| PUT | /api/healthcare/lab/instruments/<id>/status | Update status | healthcare_lab:instruments |

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| cross_tenant_result_access_denied | cross_tenant_access=True | deny |
| critical_value_notification_required | operation=verify_result, critical_value=True, notification_sent=False | deny |
| critical_value_acknowledgement_required | operation=close_critical_value, acknowledgement_present=False | deny |
| qc_hold_blocks_result_release | operation=verify_result, instrument_qc_status=qc_hold | deny |
| specimen_rejection_reason_required | operation=reject_specimen, rejection_reason_present=False | deny |
| cancelled_order_not_collectable | operation=collect_specimen, order_status=cancelled | deny |
| result_amendment_requires_original | operation=amend_result, original_result_present=False | deny |
| stat_order_turnaround_warning | operation=verify_result, stat_order_overdue=True | warn |

## Data Models
- LabOrderCreate/Response: test_code, category, priority, ordered_by, specimen_type, status
- SpecimenCreate/Response: specimen_type, barcode, chain-of-custody fields, rejection_reason
- LabResultCreate/Response: analyte, value, unit, reference range, abnormal_flag, critical_value, amendment_of
- CriticalValueNotification: result_id, severity, notified_to, acknowledged_by, acknowledged_at
- QCRunCreate/Response: instrument_id, measured/target/sd, z_score, westgard_violations, status
- InstrumentCreate/Response: model, serial_number, test_categories, status, last_calibrated_at

## Streaming Events
- order_created, order_cancelled
- specimen_collected, specimen_rejected
- result_entered, result_verified, result_amended
- critical_value_flagged, critical_value_acknowledged
- qc_run_completed, instrument_status_changed

## Edge Cases Handled
- Critical value detection uses 1.5× reference range as panic threshold (HH/LL flags)
- QC failure on 1-3s Westgard rule automatically puts instrument on QC hold
- Specimens collected for cancelled orders are blocked at the rule layer
- Result verification requires prior critical value notification if critical flag is set
- Amendment creates a new result linked to original; original is preserved read-only

## Composability Notes
Orders originate from `healthcare_emr` encounters. Results feed back into EMR as DiagnosticReport FHIR resources. Critical values trigger `ntfy` notifications that also appear in `healthcare_cli` clinical alerts. Quality indicators in `healthcare_ana` consume aggregated result data.
