# Laboratory Information System — User Guide

**Capability ID**: `healthcare_lab` | **Domain**: `healthcare` | **Version**: `2.0.0`

## Overview

`healthcare_lab` is a full-featured Laboratory Information System (LIS) capability for the APG platform. It covers the complete laboratory workflow from order entry through specimen collection, analytical processing, result verification, and FHIR R4 report export. It includes accreditation compliance scoring, auto-reflex test ordering, multi-instrument load-balanced routing, and consent-gated release of sensitive results.

## Installation

```bash
pip install apg-healthcare-lab
```

## Quick Start

```python
from apg_healthcare_lab.service import LaboratoryInformationService
from apg_healthcare_lab.models import LabOrderCreate, SpecimenCreate, LabResultCreate

svc = LaboratoryInformationService()

# 1. Create an order
order = await svc.create_order(LabOrderCreate(
    tenant_id="hosp-001",
    patient_id="pat-123",
    encounter_id="enc-456",
    test_code="CBC",
    test_name="Complete Blood Count",
    test_category="haematology",
    collection_priority="stat",
    ordered_by="dr-smith",
    specimen_type="whole_blood",
    created_by="dr-smith",
))

# 2. Collect specimen
spec = await svc.collect_specimen(SpecimenCreate(
    tenant_id="hosp-001",
    order_id=order.id,
    patient_id="pat-123",
    specimen_type="whole_blood",
    collected_by="nurse-jones",
    collection_site="ward-3b",
    collection_volume_ml=3.0,
    created_by="nurse-jones",
))

# 3. Enter result
result = await svc.enter_result(LabResultCreate(
    tenant_id="hosp-001",
    order_id=order.id,
    specimen_id=spec.id,
    analyte="Hb",
    value=6.2,
    unit="g/dL",
    reference_low=12.0,
    reference_high=17.5,
    result_status="preliminary",
    instrument_id="analyser-001",
    performed_by="tech-kim",
    created_by="tech-kim",
))
# result.abnormal_flag == "LL"  → critical low haemoglobin
```

## Core Workflows

### Order Lifecycle

```
PENDING → RECEIVED → COLLECTED → IN_PROGRESS → RESULTED → VERIFIED → FINAL
                                                         ↓
                                                     ON_HOLD (hold_order)
                                                     CANCELLED (cancel_order)
```

All state transitions are audited. Use `hold_order` / `unhold_order` for orders awaiting clinical clarification.

### Specimen Chain of Custody

Every specimen builds an append-only custody chain from collection through processing:

```python
# Label at point of care
await svc.label_specimen(tenant_id, spec.id, barcode="L20240601-001", tube_type="EDTA", labelled_by="nurse-jones")

# Track transfer to main lab
await svc.track_specimen_chain_of_custody(
    tenant_id, spec.id,
    from_location="ward-3b", to_location="main-lab",
    transferred_by="porter-ali", transport_condition="ambient",
)

# Full chain
chain = await svc.get_custody_chain(tenant_id, spec.id)
```

Valid `transport_condition` values: `ambient | refrigerated | frozen | dry_ice`

### Specimen Viability Scoring

Before processing, check whether time-in-transit has compromised analyte stability:

```python
viability = await svc.assess_specimen_viability(
    tenant_id, spec.id,
    test_codes=["K", "Glucose", "Creatinine"],
)
# viability["viability_score"]  0–100
# viability["risk_analytes"]    ["K"]  ← potassium near stability limit
# viability["recommended_action"]  "process_immediately"
```

Stability windows follow CLSI EP25. Refrigerated transport extends windows by 2.5×; frozen by 10×.

### Result Entry and Verification

```python
# Validate result
validated = await svc.validate_result(tenant_id, result.id, validated_by="lab-sci-park")

# Release to clinician (HL7_ORU | API_push | print | portal | fax)
released = await svc.release_result(tenant_id, result.id, released_by="lab-sci-park", release_method="portal")

# Amend an already-released result (original preserved)
amendment = await svc.result_amend(
    tenant_id, result.id,
    amended_value=7.1,
    amendment_reason="transcription_error_corrected",
    amended_by="lab-sci-park",
)
```

### Critical Values

```python
# Notify ordering clinician (mandatory within 60 min)
notif = await svc.alert_critical_value(
    tenant_id, result.id,
    analyte="Hb", value=6.2, unit="g/dL",
    severity="critical",
    notified_to="dr-smith", notified_by="tech-kim",
    notification_method="phone", read_back_confirmed=True,
)

# Acknowledge (clinician confirmation)
ack = await svc.acknowledge_critical_value(tenant_id, notif.id, acknowledged_by="dr-smith")

# SLA compliance report
report = await svc.generate_critical_value_report(tenant_id, date_from="2026-01-01")
```

Critical values exceeding 60-minute SLA are flagged `escalated=True` and published to `lab.critical.escalated` on NATS.

### Delta Checking

```python
check = await svc.delta_check(
    tenant_id, patient_id="pat-123",
    test_code="K",  # tight threshold analyte (≤15%)
    new_result=6.8,
    delta_threshold_pct=25.0,  # overridden to 15% for K
)
# check["delta_exceeded"] == True → "hold_for_review"
```

Tight-threshold analytes (K, Na, Hb, Plt, WBC): threshold capped at 15% regardless of `delta_threshold_pct`.

### Auto-Reflex Test Ordering

Configure rules at setup time:

```python
# TSH abnormal → automatically order free T4
await svc.configure_reflex_rule(
    tenant_id,
    trigger_test_code="TSH",
    condition="abnormal",
    threshold=0.0,          # unused for "abnormal" condition
    reflex_test_code="FT4",
    reflex_test_name="Free T4",
    reflex_priority="routine",
    configured_by="lab-director",
)

# Creatinine > 2.0 mg/dL → order eGFR
await svc.configure_reflex_rule(
    tenant_id,
    trigger_test_code="Creatinine",
    condition="gt",
    threshold=2.0,
    reflex_test_code="eGFR",
    reflex_test_name="Estimated GFR",
    reflex_priority="asap",
    configured_by="lab-director",
)
```

Reflex rules are evaluated automatically inside `enter_result`. Triggered rules emit `lab.reflex.triggered` events to NATS.

### QC Management

```python
# Run QC material — Westgard rules evaluated automatically
qc = await svc.qc_material_run(
    tenant_id, analyser_id="cobas-001",
    qc_level="L2",
    measured_value=5.12,
    expected_range={"mean": 5.0, "sd": 0.15},
    performed_by="tech-kim",
    test_code="K",
)
# qc["status"] == "passed" | "warning" | "failed"
# On "failed": instrument automatically placed on QC hold

# Record corrective action
action = await svc.qc_failure_action(
    tenant_id, qc_run_id=qc["id"],
    corrective_action="recalibrate",
    performed_by="tech-kim",
)

# QC summary across all instruments
summary = await svc.generate_qc_summary(tenant_id)
```

Westgard rules evaluated: `1-2s` (warning), `1-3s` (rejection), `R-4s` (rejection).

### Instrument Management

```python
inst = await svc.register_instrument(InstrumentCreate(
    tenant_id=tenant_id,
    name="Cobas c702",
    model="cobas c702",
    serial_number="SN-20240101",
    manufacturer="Roche",
    test_categories=["chemistry", "immunology"],
    location="main-lab-bay-2",
    created_by="biomedical-eng",
))

# Record calibration
cal = await svc.record_calibration(
    tenant_id, inst.id,
    calibrated_by="biomedical-eng",
    notes="Annual NIST-traceable calibration",
    pass_fail=True,
)

# Ingest HL7 v2 result message from instrument
msg = await svc.interface_analyser(
    tenant_id, inst.id,
    protocol="hl7_v2",
    message_type="ORU_R01",
    raw_payload="MSH|...\nOBX|1|NM|K^Potassium||5.1|mmol/L|3.5-5.0||||F",
)
```

### Multi-Instrument Specimen Routing

```python
# Configure routing weights
await svc.configure_routing_weights(
    tenant_id, test_code="CBC",
    weights=[
        {"instrument_id": "xn-3000-01", "weight": 2.0, "max_queue": 50},
        {"instrument_id": "xn-3000-02", "weight": 1.0, "max_queue": 50},
    ],
)

# Route a specimen
routing = await svc.route_specimen(tenant_id, spec.id, test_code="CBC")
# routing["selected_instrument_id"]  → weighted selection based on queue depth
```

Falls back to any eligible instrument if configured instruments are full. Returns `routing_failed=True` if no instrument is available.

### FHIR R4 Export

```python
bundle = await svc.export_fhir_diagnostic_report(tenant_id, order_id=order.id)
# FHIR Bundle (type=collection) containing:
#   DiagnosticReport + Observation[] + Communication[] for critical values
```

LOINC codes sourced from `LabTestResponse.loinc_code`. SNOMED status mapping: preliminary→33694004, final→36998000, corrected→397963008.

### Accreditation Compliance Scorecard

```python
scorecard = await svc.generate_compliance_scorecard(
    tenant_id,
    period="2026-Q1",
    standard="ISO_15189",  # CAP | CLIA | ISO_15189 | SANAS
)
# scorecard["overall"]  "PASS" | "FAIL"
# scorecard["criteria"]["critical_value_sla"]["actual_compliance_pct"]
```

Criteria evaluated:

| Criterion | Target |
|-----------|--------|
| QC frequency | ≤ 8 h between runs per instrument |
| Critical value SLA | ≥ 95% notifications within 60 min |
| Specimen rejection rate | ≤ 2% |
| Proficiency testing | ≥ 80% satisfactory EQA scores |
| STAT TAT 90th percentile | ≤ 60 min |
| Delta check utilisation | ≥ 90% of results checked |

### Consent-Gated Result Release

```python
# Record patient consent before releasing sensitive results
consent = await svc.record_patient_consent(
    tenant_id,
    patient_id="pat-123",
    test_categories=["genetics", "hiv"],
    consented_by="counsellor-chen",
    expiry_date=datetime(2027, 6, 1),
    consent_method="written",
)

# Check consent before release
status = await svc.check_consent(tenant_id, "pat-123", "genetics")
# status["has_consent"]  True | False
# status["reason"]        "valid_consent_on_record"

# release_result will raise PolicyViolationError for gated categories without consent
```

Consent-gated categories: `genetics | hiv | substance_abuse | reproductive | mental_health`

### Audit Trail

```python
events = await svc.get_audit_events(tenant_id, event_type="result_verified", limit=50)

# Verify audit chain integrity (cryptographic hash chain)
integrity = await svc.verify_audit_chain(tenant_id)
# integrity["valid"]  True | False
# integrity["first_break_at"]  None | int (index of first tampered entry)
```

## External Referrals

```python
# Refer to external lab with courier tracking
referral = await svc.refer_to_external_lab(
    tenant_id, spec.id,
    external_lab="PathCare Reference Laboratory",
    courier="DHL",
    tracking_number="DHL-JNB-001",
    test_requested="BRCA1_BRCA2_Panel",
    expected_tat_days=7,
)

# Receive result back from external lab
ext_result = await svc.receive_external_result(
    tenant_id, referral["id"],
    result_data={"analyte": "BRCA1", "variant": "c.5266dupC", "interpretation": "Pathogenic"},
    verified_by="lab-sci-park",
)
```

## Dashboard and Reporting

```python
# Real-time dashboard summary
dashboard = await svc.dashboard_summary(tenant_id)

# Workload report (staffing and reagent planning)
workload = await svc.lab_workload_report(tenant_id, period="2026-05", by_analyser=True)

# TAT monitoring report
tat = await svc.tat_monitoring(tenant_id, period="2026-05", by_analyser=True)

# Full patient lab report (JSON/HTML/PDF)
report = await svc.generate_lab_report(tenant_id, order_id=order.id, fmt="json")
```

## Key Service Methods

### Order Management
- `create_order(payload)` — Place a new order
- `receive_lab_order(tenant_id, order_id, specimen_requirements, received_by)` — Acknowledge receipt
- `cancel_order(tenant_id, order_id, reason)` — Cancel with documented reason
- `hold_order(tenant_id, order_id, reason)` / `unhold_order(...)` — Hold management
- `update_order(tenant_id, order_id, payload)` — Partial update
- `get_order(tenant_id, order_id)` / `list_orders(...)` — Retrieval

### Specimen Management
- `collect_specimen(payload)` — Record collection
- `label_specimen(...)` — Assign barcode
- `track_specimen_chain_of_custody(...)` — Record custody transfer
- `reject_specimen(...)` — Reject with documented reason
- `receive_specimen(...)` — Mark received at lab
- `assess_specimen_viability(...)` — CLSI EP25 viability scoring
- `route_specimen(...)` — Load-balanced instrument routing
- `get_custody_chain(...)` — Full custody log

### Result Management
- `enter_result(payload)` — Enter preliminary result (triggers reflex rules)
- `process_test(...)` — Raw analyser result entry
- `validate_result(...)` — Validate for release
- `verify_result(...)` — Final verification
- `release_result(...)` — Release to clinician/portal
- `result_amend(...)` — Amend with audit trail
- `delta_check(...)` — Compare to patient history
- `update_result(...)` / `get_result(...)` / `list_results(...)`

### Critical Values
- `alert_critical_value(...)` — Create alert record
- `acknowledge_critical_value(...)` — Record acknowledgement
- `list_critical_values(...)` / `get_critical_value(...)` / `create_critical_value(...)`
- `critical_value_alert(...)` — Full SLA-tracked notification record

### QC Management
- `run_qc(payload)` — Record structured QC run
- `qc_material_run(...)` — Multi-rule Westgard evaluation
- `qc_failure_action(...)` — Record corrective action
- `generate_qc_summary(tenant_id)` — Per-instrument pass rates
- `list_qc_runs(...)` / `get_qc_run(...)` / `update_qc_run(...)`
- `external_proficiency_testing(...)` — EQA participation record

### Instrument Management
- `register_instrument(payload)` — Add to registry
- `update_instrument_status(...)` — Status lifecycle
- `record_calibration(...)` — Log calibration event
- `interface_analyser(...)` — Ingest HL7/ASTM message
- `list_instruments(...)` / `get_instrument(...)` / `update_instrument(...)`

### Reference Ranges
- `create_reference_range(payload)` — Add demographic-stratified range
- `validate_reference_range(...)` — Classify value against best-matching range
- `update_reference_range(...)` / `delete_reference_range(...)` / `list_reference_ranges(...)`

### Test Catalogue
- `create_test(payload)` — Add test with LOINC/CPT/SNOMED codes
- `update_test(...)` / `delete_test(...)` / `get_test(...)` / `list_tests(...)`

### Reflex Rules
- `configure_reflex_rule(...)` — Define auto-reflex trigger
- `evaluate_reflex_rules(...)` — Evaluate rules for a result (called internally)

### Routing
- `configure_routing_weights(...)` — Set per-instrument weights and queue limits
- `route_specimen(...)` — Select instrument for a specimen+test

### Consent
- `record_patient_consent(...)` — Store time-limited consent record
- `check_consent(...)` — Validate consent before sensitive result release

### External Referrals
- `create_referral(payload)` / `get_referral(...)` / `list_referrals(...)` / `update_referral(...)`
- `refer_to_external_lab(...)` — Refer with courier tracking
- `receive_external_result(...)` — Import result from external lab

### Reporting
- `generate_lab_report(...)` — Full order report (JSON)
- `export_fhir_diagnostic_report(...)` — FHIR R4 DiagnosticReport bundle
- `generate_compliance_scorecard(...)` — Accreditation compliance (CAP/CLIA/ISO 15189/SANAS)
- `generate_critical_value_report(...)` — SLA compliance report
- `generate_rejection_report(...)` — Pre-analytical quality report
- `lab_workload_report(...)` — Workload and productivity
- `tat_monitoring(...)` — Turnaround time statistics
- `dashboard_summary(...)` — Real-time dashboard

### Audit
- `get_audit_events(...)` — Query audit log
- `verify_audit_chain(...)` — Cryptographic integrity check

## Interoperability

```apg
use healthcare_lab;
```

Results emit to NATS subjects for consumption by:
- `healthcare_emr` — FHIR DiagnosticReport import
- `healthcare_ana` — Analytics and trend dashboards
- `healthcare_qms` — Quality management system
- `ntfy` — Critical value and TAT breach notifications

## Configuration

All keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed `HEALTHCARE_LAB_`:

```bash
HEALTHCARE_LAB_STAT_TURNAROUND_MINUTES=60
HEALTHCARE_LAB_CRITICAL_VALUE_SLA_MINUTES=60
HEALTHCARE_LAB_QC_FREQUENCY_HOURS=8
HEALTHCARE_LAB_ROUTING_DEFAULT_MAX_QUEUE=100
```

## Further Reading

- `service.py` — Complete business logic implementation (2,600+ lines)
- `models.py` — Pydantic v2 data models
- `api.py` — Flask-AppBuilder REST API endpoints
- `views.py` — FAB views and UI schemas
- `domain/rules.py` — Business rule definitions
- `domain/calculations.py` — Statistical helpers (Westgard, delta, pass rates)
- `domain/adapters.py` — HL7/FHIR format adapters
- `capability_contract.py` — Policy evaluation engine
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 proposed enhancements with competitor benchmarks
- `README.md` — Quick API reference
