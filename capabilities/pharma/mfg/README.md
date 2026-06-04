# Pharmaceutical Manufacturing

## Overview
Manages pharmaceutical manufacturing operations from batch record creation through equipment qualification, yield management, deviation handling, line clearance, raw material management, and QP batch release. Enforces GMP compliance, electronic batch records, QP release signatures, and equipment qualification requirements at every production step.

## Capability ID
`pharma_mfg`

## Provides
- batch_record_management_workflow: Electronic batch record (EBR) creation, execution, and QP sign-off
- manufacturing_execution_workflow: Line assignment, start, and completion with clearance enforcement
- equipment_qualification_workflow: IQ/OQ/PQ lifecycle with requalification scheduling
- yield_management_workflow: Step-by-step yield recording, reconciliation, and variance investigation
- deviation_management_workflow: GMP deviation capture, severity classification, and investigation
- gmp_compliance_workflow: Framework registration, self-inspection, and change control integration
- material_management_workflow: Vendor-qualified raw material receipt, QC release, and dispensing
- line_clearance_workflow: Pre-batch cleaning verification and line clearance certification
- cleaning_validation_workflow: Cleaning procedure validation and periodic review
- qp_release_workflow: Qualified Person electronic sign-off and batch certification

## Requires
| Capability | Reason |
|------------|--------|
| auth | Role-based access including QP release authority |
| audl | 21 CFR Part 11 compliant EBR audit trail |
| mten | Site-level manufacturing data isolation |
| conf | GMP framework and deviation threshold configuration |
| ntfy | Critical deviation and yield variance notifications |
| wflo | QP release and CAPA approval workflow |
| comp | GMP compliance enforcement |
| moni | Environmental and equipment monitoring |
| schd | Calibration and maintenance scheduling |
| mqeb | Event streaming for batch and deviation events |

## Configuration
| Key | Description | Default |
|-----|-------------|---------|
| yield_management.yield_variance_threshold_pct | Yield variance alert threshold | 2.0 |
| yield_management.investigation_trigger_pct | Variance requiring investigation | 5.0 |
| equipment.requalification_trigger_months | Equipment requalification cycle | 12 |
| deviations.reporting_timeline_hours.critical | Critical deviation reporting deadline | 24 |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /pharma-mfg/api/v1/batches | POST | Create batch record | pharma_mfg:batches |
| /pharma-mfg/api/v1/batches/<id>/start | POST | Start batch on line | pharma_mfg:batches |
| /pharma-mfg/api/v1/batches/<id>/release | POST | QP release batch | pharma_mfg:ebr |
| /pharma-mfg/api/v1/equipment/<id>/qualify | POST | Record qualification | pharma_mfg:qualification |
| /pharma-mfg/api/v1/deviations | POST | Raise deviation | pharma_mfg:deviations |
| /pharma-mfg/api/v1/yield/reconcile/<batch_id> | POST | Reconcile batch yield | pharma_mfg:yield |
| /pharma-mfg/api/v1/lines/<id>/clear | POST | Clear production line | pharma_mfg:lines |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| batch_master_formula_required | Batch created without master formula | Deny — attach master formula |
| qp_release_required | Batch released without QP signature | Deny — obtain QP signature |
| equipment_qualification_required | Equipment used without qualification | Deny — complete qualification |
| line_clearance_required | Batch started without line clearance | Deny — complete clearance |
| deviation_investigation_required | Deviation closed without investigation | Deny — complete investigation |
| yield_reconciliation_required | Batch closed without yield reconciliation | Deny — reconcile yield |
| material_incoming_qc_required | Material released without incoming QC | Deny — complete incoming QC |

## Data Models
- BatchRecord: batch_number, manufacturing_type, master_formula_reference, status, qp_release_reference, yield_percentage
- Equipment: equipment_id, equipment_type, status, iq_reference, oq_reference, pq_reference, requalification_due
- EquipmentQualification: qualification_type, protocol_reference, report_reference, next_requalification
- ManufacturingDeviation: deviation_number, deviation_type, severity, gmp_impact, root_cause, capa_reference
- YieldRecord: yield_type, step_name, theoretical_quantity, actual_quantity, variance_pct, investigation_required
- ProductionLine: line_code, status, cleaning_status, current_batch_id
- RawMaterial: material_code, lot_number, status, incoming_qc_reference, expiry_date

## Streaming Events
- batch_started, batch_completed, batch_released, batch_rejected
- equipment_qualified, equipment_out_of_service
- deviation_raised, deviation_closed
- yield_reconciled, yield_variance_exceeded
- gmp_deviation_critical, line_clearance_completed, qp_release_signed

## Edge Cases Handled
- Equipment with expired calibration blocks batch start even if IQ/OQ/PQ is current
- Yield variance above 5% triggers mandatory investigation flag independent of batch status
- QP release requires both QP signature reference and electronic signature reference; one is not sufficient
- Line clearance status resets to dirty on batch start and must be re-verified before next batch
- Critical deviations trigger 24-hour reporting even when the batch is subsequently rejected

## Composability Notes
Released batches feed `pharma_dis` for dispatch. Deviation records integrate with `pharma_qms` CAPA workflow. Equipment qualification references are cited in `pharma_qms` validation lifecycle. Batch genealogy data supports `pharma_dis` recall management.
