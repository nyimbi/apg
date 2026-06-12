# Laboratory Information System

## Overview
Full-featured LIS capability providing lab order management, specimen tracking with chain of custody, result entry and verification, critical value alerting with mandatory acknowledgement, QC management with Westgard rule evaluation, instrument status tracking, FHIR R4 export, auto-reflex test ordering, multi-instrument specimen routing, accreditation compliance scoring, and consent-gated result release.

## Capability ID
`healthcare_lab`

## Provides
- `lab_order_management`: STAT/ASAP/routine/reflex order lifecycle from pending through resulted and verified
- `specimen_tracking`: Chain-of-custody tracking with barcode assignment, rejection reason documentation, and viability scoring
- `result_entry_verification`: Preliminary-to-final result workflow with reference range flagging (H/HH/L/LL)
- `critical_value_alerting`: Automatic critical flag detection with mandatory notification, read-back confirmation, and escalation
- `qc_management`: Westgard rule evaluation (1-3s, 1-2s, R-4s) with automatic QC hold on failure
- `instrument_management`: Instrument registry with status lifecycle, calibration tracking, and HL7 v2 message ingestion
- `lis_integration`: Event stream for downstream EMR, pharmacy, and analytics consumers via NATS subjects
- `reference_range_evaluation`: Numeric result comparison against configurable demographic-stratified reference intervals
- `lab_reporting`: Result report generation, FHIR R4 DiagnosticReport bundle export, and accreditation compliance scorecard
- `reflex_ordering`: Auto-reflex test ordering engine triggered by configurable result conditions
- `specimen_routing`: Multi-instrument load-balanced routing with weighted round-robin and queue-depth awareness
- `consent_management`: Consent-gated result release for sensitive categories (genetics, HIV, substance abuse, reproductive, mental health)

## Requires
- `auth`: PHI access authorization for result data
- `audl`: Audit trail for all order/result/QC operations
- `mten`: Multi-tenant isolation
- `conf`: Tenant-specific configuration
- `ntfy`: Critical value notifications to clinicians
- `wflo`: Verification and QC review approval workflows
- `moni`: Instrument availability and turnaround time monitoring
- `mqeb`: Result event emission to EMR and analytics (NATS/bytewax)

## Configuration

| Key | Description |
|-----|-------------|
| `orders.stat_turnaround_minutes` | Target turnaround for STAT orders (default: 60) |
| `results.critical_value_notification_required` | Block result verify until notification sent |
| `qc.westgard_rules_enabled` | Enable 1-3s/1-2s/R-4s Westgard evaluation |
| `qc.qc_frequency_hours` | Required QC frequency (default: 8h) |
| `specimens.chain_of_custody_required` | Chain of custody tracking for all specimens |
| `reflex.rules_enabled` | Enable auto-reflex test ordering engine |
| `routing.default_max_queue` | Per-instrument default queue limit for routing (default: 100) |
| `consent.gated_categories` | Test categories requiring explicit patient consent before release |
| `compliance.standard` | Accreditation standard for scorecard (CAP / CLIA / ISO_15189 / SANAS) |

## API Routes

| Method | Path | Description | Permission |
|--------|------|-------------|------------|
| GET | `/api/healthcare/lab/orders` | List orders | `healthcare_lab:orders` |
| POST | `/api/healthcare/lab/orders` | Create order | `healthcare_lab:orders_write` |
| GET | `/api/healthcare/lab/orders/<id>` | Order detail | `healthcare_lab:orders` |
| POST | `/api/healthcare/lab/orders/<id>/cancel` | Cancel order | `healthcare_lab:orders_write` |
| POST | `/api/healthcare/lab/orders/<id>/hold` | Place order on hold | `healthcare_lab:orders_write` |
| POST | `/api/healthcare/lab/orders/<id>/unhold` | Release order from hold | `healthcare_lab:orders_write` |
| GET | `/api/healthcare/lab/orders/<id>/fhir` | Export FHIR R4 bundle | `healthcare_lab:fhir_export` |
| GET | `/api/healthcare/lab/specimens` | List specimens | `healthcare_lab:specimens` |
| POST | `/api/healthcare/lab/specimens` | Collect specimen | `healthcare_lab:specimens_write` |
| POST | `/api/healthcare/lab/specimens/<id>/reject` | Reject specimen | `healthcare_lab:specimens_write` |
| POST | `/api/healthcare/lab/specimens/<id>/receive` | Receive specimen | `healthcare_lab:specimens_write` |
| POST | `/api/healthcare/lab/specimens/<id>/route` | Route specimen to instrument | `healthcare_lab:routing` |
| GET | `/api/healthcare/lab/specimens/<id>/viability` | Assess specimen viability | `healthcare_lab:specimens` |
| GET | `/api/healthcare/lab/specimens/<id>/custody` | Full custody chain | `healthcare_lab:specimens` |
| GET | `/api/healthcare/lab/results` | List results | `healthcare_lab:results` |
| POST | `/api/healthcare/lab/results` | Enter result | `healthcare_lab:results_write` |
| POST | `/api/healthcare/lab/results/<id>/verify` | Verify result | `healthcare_lab:results_write` |
| POST | `/api/healthcare/lab/results/<id>/validate` | Validate result | `healthcare_lab:results_write` |
| POST | `/api/healthcare/lab/results/<id>/release` | Release validated result | `healthcare_lab:results_write` |
| POST | `/api/healthcare/lab/results/<id>/amend` | Amend released result | `healthcare_lab:results_write` |
| GET | `/api/healthcare/lab/critical-values` | List critical values | `healthcare_lab:critical_values` |
| POST | `/api/healthcare/lab/critical-values` | Notify critical value | `healthcare_lab:critical_values_write` |
| POST | `/api/healthcare/lab/critical-values/<id>/acknowledge` | Acknowledge notification | `healthcare_lab:critical_values_write` |
| GET | `/api/healthcare/lab/qc` | List QC runs | `healthcare_lab:qc` |
| POST | `/api/healthcare/lab/qc` | Run QC | `healthcare_lab:qc_write` |
| GET | `/api/healthcare/lab/qc/summary` | QC pass/fail summary | `healthcare_lab:qc` |
| GET | `/api/healthcare/lab/instruments` | List instruments | `healthcare_lab:instruments` |
| POST | `/api/healthcare/lab/instruments` | Register instrument | `healthcare_lab:instruments_write` |
| PUT | `/api/healthcare/lab/instruments/<id>/status` | Update status | `healthcare_lab:instruments_write` |
| POST | `/api/healthcare/lab/instruments/<id>/calibrate` | Record calibration | `healthcare_lab:instruments_write` |
| POST | `/api/healthcare/lab/instruments/<id>/message` | Ingest analyser HL7/ASTM message | `healthcare_lab:instruments_write` |
| GET | `/api/healthcare/lab/reference-ranges` | List reference ranges | `healthcare_lab:reference_ranges` |
| POST | `/api/healthcare/lab/reference-ranges` | Create reference range | `healthcare_lab:reference_ranges_write` |
| GET | `/api/healthcare/lab/tests` | Test catalogue | `healthcare_lab:tests` |
| POST | `/api/healthcare/lab/tests` | Add test to catalogue | `healthcare_lab:tests_write` |
| GET | `/api/healthcare/lab/referrals` | List external referrals | `healthcare_lab:referrals` |
| POST | `/api/healthcare/lab/referrals` | Create referral | `healthcare_lab:referrals_write` |
| GET | `/api/healthcare/lab/reports/workload` | Workload report | `healthcare_lab:reports` |
| GET | `/api/healthcare/lab/reports/tat` | TAT monitoring report | `healthcare_lab:reports` |
| GET | `/api/healthcare/lab/reports/compliance` | Accreditation compliance scorecard | `healthcare_lab:reports` |
| GET | `/api/healthcare/lab/reports/critical-values` | Critical value compliance report | `healthcare_lab:reports` |
| GET | `/api/healthcare/lab/reports/rejections` | Specimen rejection report | `healthcare_lab:reports` |
| GET | `/api/healthcare/lab/reports/lab/<id>` | Full patient lab report | `healthcare_lab:reports` |
| POST | `/api/healthcare/lab/reflex-rules` | Configure auto-reflex rule | `healthcare_lab:reflex_rules` |
| POST | `/api/healthcare/lab/routing/weights` | Configure routing weights | `healthcare_lab:routing` |
| POST | `/api/healthcare/lab/consent` | Record patient consent | `healthcare_lab:consent_write` |
| GET | `/api/healthcare/lab/consent/<patient_id>/<category>` | Check consent status | `healthcare_lab:consent` |
| GET | `/api/healthcare/lab/audit` | Audit event log | `healthcare_lab:audit` |
| GET | `/api/healthcare/lab/audit/verify` | Verify audit chain integrity | `healthcare_lab:audit` |
| GET | `/api/healthcare/lab/dashboard` | Lab dashboard summary | `healthcare_lab:view` |

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| `cross_tenant_result_access_denied` | `cross_tenant_access=True` | deny |
| `critical_value_notification_required` | `operation=verify_result, critical_value=True, notification_sent=False` | deny |
| `critical_value_acknowledgement_required` | `operation=close_critical_value, acknowledgement_present=False` | deny |
| `qc_hold_blocks_result_release` | `operation=verify_result, instrument_qc_status=qc_hold` | deny |
| `specimen_rejection_reason_required` | `operation=reject_specimen, rejection_reason_present=False` | deny |
| `cancelled_order_not_collectable` | `operation=collect_specimen, order_status=cancelled` | deny |
| `result_amendment_requires_original` | `operation=amend_result, original_result_present=False` | deny |
| `consent_required_for_sensitive_release` | `operation=release_result, test_category in CONSENT_GATED, consent_absent=True` | deny |
| `stat_order_turnaround_warning` | `operation=verify_result, stat_order_overdue=True` | warn |

## Data Models

- `LabOrderCreate/Response`: test_code, category, priority, ordered_by, specimen_type, status
- `SpecimenCreate/Response`: specimen_type, barcode, chain-of-custody fields, rejection_reason
- `LabResultCreate/Response`: analyte, value, unit, reference range, abnormal_flag, critical_value, amendment_of
- `CriticalValueNotification`: result_id, severity, notified_to, acknowledged_by, acknowledged_at, escalated
- `QCRunCreate/Response`: instrument_id, measured/target/sd, z_score, westgard_violations, status
- `InstrumentCreate/Response`: model, serial_number, test_categories, status, last_calibrated_at
- `ReferenceRangeCreate/Response`: test_code, analyte, low, high, critical_low, critical_high, age/sex stratification
- `LabTestCreate/Response`: test_code, loinc_code, cpt_code, snomed_code, turnaround_minutes, price
- `ExternalReferralCreate/Response`: reference_lab_name, tracking_number, expected_tat_hours, status

## Streaming Events (NATS subjects)

| Subject | Description |
|---------|-------------|
| `lab.order.state_changed` | Every order status transition |
| `lab.specimen.collected` | New specimen collected |
| `lab.specimen.rejected` | Specimen rejected with reason |
| `lab.result.entered` | Preliminary result entered |
| `lab.result.verified` | Result verified/released |
| `lab.result.amended` | Result corrected |
| `lab.critical.pending.<tenant>.<result_id>` | Critical value awaiting acknowledgement |
| `lab.critical.escalated` | Critical value escalated beyond 60 min SLA |
| `lab.qc.completed` | QC run finished |
| `lab.instrument.qc_hold` | Instrument placed on QC hold |
| `lab.routing.assigned` | Specimen routed to instrument |
| `lab.reflex.triggered` | Auto-reflex order created |
| `lab.tat.at_risk` | Order projected to breach TAT SLA |

## Edge Cases Handled

- Critical value detection uses 1.5× reference range as panic threshold (HH/LL flags)
- QC failure on 1-3s Westgard rule automatically puts instrument on QC hold
- Specimens collected for cancelled orders are blocked at the rule layer
- Result verification requires prior critical value notification if critical flag is set
- Amendment creates a new result linked to original; original is preserved read-only
- Reflex rules evaluate after every `enter_result`; circular reflex chains are not checked (configure with care)
- Specimen viability scoring accounts for refrigerated/frozen transport via CLSI EP25 stability multipliers
- Consent records are time-limited; expired consent is treated as absent for sensitive category result release
- Routing falls back gracefully when all instruments are on QC hold or offline

## Composability Notes

Orders originate from `healthcare_emr` encounters. Results feed back into EMR as FHIR R4 DiagnosticReport resources via `export_fhir_diagnostic_report`. Critical values trigger `ntfy` notifications that also appear in `healthcare_cli` clinical alerts. Quality indicators in `healthcare_ana` consume aggregated result data. Accreditation scorecard exports integrate with `healthcare_qms` quality management system. Consent records reference `healthcare_consent` capability for cross-capability consent propagation.

---

## World-Class Enhancements (v2.0)

Fifteen targeted improvements over baseline implementation:

- **I1. Predictive TAT Alerting via NATS + Bayesian Estimation** [Real-time analytics / streaming]
- **I2. Continuous Delta-Check History with Configurable Per-Analyte Thresholds** [Clinical safety]
- **I3. FHIR R4 DiagnosticReport Serialisation** [Interoperability]
- **I4. Automated Westgard EWMA / CUSUM Statistical Process Control** [Quality management]
- **I5. Specimen Viability Scoring with Real-Time Degradation Modelling** [Pre-analytical quality]
- **I6. Auto-Reflex Test Ordering Engine** [Workflow automation]
- **I7. HL7 v2 Bidirectional LIS Interface (ORM/ORU Full Cycle)** [LIS integration]
- **I8. Regulatory Reporting Pack (CAP/CLIA/ISO 15189 Compliance Scorecard)** [Regulatory compliance]
- **I9. NATS-backed Real-Time Critical Value Escalation Ladder** [Patient safety / alerting]
- **I10. Genomic and Molecular Test Result Support (VCF/HGVS Notation)** [Precision medicine]
- **I11. Instrument Predictive Maintenance via Anomaly Detection** [Operations / uptime]
- **I12. Audit-Trail Immutability via Append-Only Event Log with Cryptographic Hashing** [Security / compliance]
- **I13. Multi-Laboratory Network Specimen Routing and Load Balancing** [Operations / scalability]
- **I14. Patient Report Portal with SMART on FHIR Launch** [Patient engagement]
- **I15. Consent-Gated Result Release for Genetic and Sensitive Tests** [Privacy / compliance]

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
