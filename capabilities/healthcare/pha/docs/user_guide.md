# Pharmacy Management — User Guide
© 2025 Datacraft | Author: Nyimbi Odero

## Contents
1. [Overview](#1-overview)
2. [Installation & Setup](#2-installation--setup)
3. [Formulary Management](#3-formulary-management)
4. [Prescription Workflow](#4-prescription-workflow)
5. [Dispensing Workflow](#5-dispensing-workflow)
6. [Allergy Cross-Check](#6-allergy-cross-check)
7. [Barcode Product Verification](#7-barcode-product-verification)
8. [Drug Interaction Checking](#8-drug-interaction-checking)
9. [Controlled Substances](#9-controlled-substances)
10. [Inventory Management](#10-inventory-management)
11. [Drug Recall Management](#11-drug-recall-management)
12. [Cold Chain Monitoring](#12-cold-chain-monitoring)
13. [Medication Adherence Scoring](#13-medication-adherence-scoring)
14. [Adverse Drug Event Reporting](#14-adverse-drug-event-reporting)
15. [Prior Authorization](#15-prior-authorization)
16. [Dispense Label Generation](#16-dispense-label-generation)
17. [FHIR R4 Export](#17-fhir-r4-export)
18. [Pharmacist Queue Metrics](#18-pharmacist-queue-metrics)
19. [Reports](#19-reports)
20. [Policy Rules Reference](#20-policy-rules-reference)
21. [Error Reference](#21-error-reference)

---

## 1. Overview

`healthcare_pha` is the APG pharmacy management runtime. It enforces patient-safety
policy rules at every dispense gate, maintains tenant-isolated drug knowledge, and
exposes an async Python service layer consumed by Flask-AppBuilder API blueprints.

Key safety invariants:
- A prescription must be pharmacist-verified before any dispense order transitions
  to `dispensed`.
- Contraindicated drug interactions are a hard block; no override path exists.
- Controlled substance waste requires dual-witness signature.
- Anaphylactic allergy matches block dispense unconditionally.
- Scanned NDC mismatch or expired lot blocks dispense before inventory deduction.

All operations are tenant-scoped; cross-tenant data access is structurally impossible.

---

## 2. Installation & Setup

```bash
# From the APG workspace root
uv pip install -e capabilities/healthcare/pha
```

Initialise the service (in-process; swap for a persistent backend in production):

```python
from capabilities.healthcare.pha.service import PharmacyManagementService
svc = PharmacyManagementService()
```

Tenant ID is passed on every call. Use your organisation's tenant slug, e.g.
`"nairobi_general"`.

---

## 3. Formulary Management

### Add a drug

```python
from capabilities.healthcare.pha.models import DrugCreate

drug = await svc.add_drug_to_formulary(DrugCreate(
    tenant_id="t1",
    drug_name="Amoxicillin",
    generic_name="amoxicillin",
    ndc_code="00093-4155-01",
    rxnorm_code="723",
    drug_type="generic",
    drug_schedule="non_controlled",
    dosage_form="capsule",
    strength="500",
    unit="mg",
    manufacturer="Teva",
    formulary_status="preferred",
    created_by="pharmacist_01",
))
```

### Mark as LASA

```python
await svc.mark_drug_lasa(
    tenant_id="t1",
    drug_id=drug.id,
    lasa_pair="Amoxil",
    alert_type="look_alike",
)
```

### Formulary review (P&T committee)

```python
await svc.formulary_review(
    tenant_id="t1",
    drug_id=drug.id,
    review_type="annual_review",
    recommendation="maintain",
    reviewed_by="dr_smith",
    clinical_rationale="Effective first-line; cost-effective generic available.",
    cost_data={"unit_cost_usd": 0.12, "annual_spend_usd": 4800},
)
```

---

## 4. Prescription Workflow

```python
from capabilities.healthcare.pha.models import PrescriptionCreate

rx = await svc.create_prescription(PrescriptionCreate(
    tenant_id="t1",
    patient_id="pat_001",
    prescriber_id="dr_001",
    prescriber_npi="1234567890",
    drug_id=drug.id,
    drug_name="Amoxicillin",
    dosage_form="capsule",
    strength="500",
    quantity=21,
    unit="capsule",
    days_supply=7,
    sig="Take one capsule three times daily with food",
    refills_authorized=0,
    refills_remaining=0,
    diagnosis_icd10="J06.9",
    is_controlled=False,
    electronic=True,
    created_by="dr_001",
))
```

Cancel a prescription:

```python
await svc.cancel_prescription(tenant_id="t1", rx_id=rx.id, actor_id="pharmacist_01")
```

---

## 5. Dispensing Workflow

The full dispense lifecycle: create → verify → dispense → pickup.

### Step 1 — Allergy and interaction pre-checks (recommended)

```python
# Allergy check (see section 6)
allergy_report = await svc.check_allergy_cross_reactivity(
    tenant_id="t1", drug_id=drug.id,
    patient_allergies=[{"allergen": "penicillin", "severity": "moderate", "reaction_type": "rash"}],
)
assert not allergy_report["hard_block"], "Dispense blocked: allergy"

# Interaction check
interaction_report = await svc.check_drug_interactions_at_dispense(
    tenant_id="t1", prescription_id=rx.id,
    patient_current_drugs=["drug_id_metformin"],
)
assert interaction_report["dispense_safe"], "Dispense blocked: interaction"
```

### Step 2 — Pharmacist verification

```python
verification = await svc.pharmacist_verification(
    tenant_id="t1",
    prescription_id=rx.id,
    pharmacist_id="pharmacist_01",
    clinical_notes="Dose appropriate. No interactions. Patient counselled.",
)
assert verification["ready_to_dispense"]
```

### Step 3 — Create dispense order

```python
from capabilities.healthcare.pha.models import DispenseOrderCreate

order = await svc.create_dispense_order(DispenseOrderCreate(
    tenant_id="t1",
    patient_id="pat_001",
    drug_id=drug.id,
    prescription_id=rx.id,
    quantity=21,
    unit="capsule",
    pharmacist_verified=True,
    formulary_status="preferred",
    interaction_severity="none",
    drug_inventory_status="in_stock",
    prior_auth_approved=True,
    formulary_override_present=False,
    step_therapy_completed=True,
    created_by="pharmacist_01",
))
```

### Step 4 — Barcode scan verification (see section 7)

```python
from datetime import datetime
scan = await svc.scan_and_verify_product(
    tenant_id="t1", order_id=order.id,
    scanned_ndc="00093-4155-01", scanned_lot="LOT2025A",
    scanned_expiry=datetime(2026, 12, 31), scanned_by="pharmacist_01",
)
assert scan["scan_verified"], "Product scan failed"
```

### Step 5 — Dispense

```python
await svc.dispense(tenant_id="t1", order_id=order.id)
```

### Step 6 — Patient pickup

```python
await svc.mark_picked_up(tenant_id="t1", order_id=order.id)
```

---

## 6. Allergy Cross-Check

Check a drug against a patient's allergy profile before dispensing. Checks direct
allergen name match and known cross-reactive drug class pairs.

```python
report = await svc.check_allergy_cross_reactivity(
    tenant_id="t1",
    drug_id=drug.id,
    patient_allergies=[
        {"allergen": "penicillin", "severity": "anaphylactic", "reaction_type": "anaphylaxis"},
        {"allergen": "aspirin", "severity": "moderate", "reaction_type": "urticaria"},
    ],
)

# report keys:
# hard_block (bool) — anaphylactic match; must not dispense
# dispense_safe (bool)
# pharmacist_review_required (bool)
# matches (list) — each match has: allergen, match_type, severity, action
```

**When `hard_block=True`**, the dispense workflow must not proceed.
A pharmacist may choose an alternative drug.

---

## 7. Barcode Product Verification

Call `scan_and_verify_product` between the dispense order creation and the final
`dispense()` call. Pass the NDC, lot, and expiry decoded from the physical product
barcode (GS1-128 or DataMatrix).

```python
scan = await svc.scan_and_verify_product(
    tenant_id="t1",
    order_id=order.id,
    scanned_ndc="00093-4155-01",
    scanned_lot="LOT2025A",
    scanned_expiry=datetime(2026, 12, 31),
    scanned_by="pharmacist_01",
)

# scan keys:
# scan_verified (bool)
# hard_block (bool)    — NDC mismatch or expired product
# near_expiry_warning  — expires within 7 days; pharmacist acknowledgement required
# mismatches (list)    — human-readable mismatch descriptions
```

---

## 8. Drug Interaction Checking

### Record an interaction pair

```python
from capabilities.healthcare.pha.models import DrugInteractionCreate

await svc.record_interaction(DrugInteractionCreate(
    tenant_id="t1",
    drug_a_id="drug_warfarin",
    drug_b_id="drug_aspirin",
    severity="major",
    mechanism="Additive anticoagulation effect",
    clinical_effect="Increased bleeding risk",
    management="Monitor INR closely; reduce warfarin dose if INR elevated",
    evidence_source="Lexicomp 2025",
    created_by="pharmacist_01",
))
```

### Check at point of dispense

```python
report = await svc.check_drug_interactions_at_dispense(
    tenant_id="t1",
    prescription_id=rx.id,
    patient_current_drugs=["drug_warfarin", "drug_metformin"],
)
# report["dispense_safe"] is False if any contraindicated interaction found
# report["pharmacist_override_required"] is True for major interactions
```

---

## 9. Controlled Substances

### Log a controlled substance action

```python
from capabilities.healthcare.pha.models import ControlledSubstanceLogCreate

await svc.log_controlled_substance(ControlledSubstanceLogCreate(
    tenant_id="t1",
    drug_id="drug_morphine",
    drug_schedule="schedule_ii",
    action="dispense",
    quantity=10.0,
    unit="ml",
    patient_id="pat_001",
    performed_by="pharmacist_01",
    created_by="pharmacist_01",
))
```

### Waste (requires dual witness)

```python
await svc.log_controlled_substance(ControlledSubstanceLogCreate(
    tenant_id="t1",
    drug_id="drug_morphine",
    drug_schedule="schedule_ii",
    action="waste",
    quantity=2.0,
    unit="ml",
    performed_by="pharmacist_01",
    witness_id="pharmacist_02",   # mandatory for waste
    waste_amount=2.0,
    created_by="pharmacist_01",
))
```

### Narcotics register reconciliation

```python
await svc.narcotics_register_reconciliation(
    tenant_id="t1",
    period="2026-05-01/2026-05-31",
    reconciled_by="pharmacist_01",
    witness_id="pharmacist_02",
)
```

---

## 10. Inventory Management

### Add an inventory lot

```python
from capabilities.healthcare.pha.models import InventoryItemCreate
from datetime import datetime

item = await svc.add_inventory_item(InventoryItemCreate(
    tenant_id="t1",
    drug_id=drug.id,
    lot_number="LOT2025A",
    quantity_on_hand=500.0,
    unit="capsule",
    expiry_date=datetime(2026, 12, 31),
    location="Shelf-A3",
    created_by="pharmacist_01",
))
```

### Check expiry dates

```python
alerts = await svc.track_lot_expiry(tenant_id="t1", threshold_days=30)
# Returns list sorted by days_remaining ascending
# alert["alert_severity"] is "critical" for <= 7 days
```

### Reorder point check

```python
report = await svc.reorder_point_check(
    tenant_id="t1",
    drug_id=drug.id,
    reorder_point=50.0,
    reorder_quantity=200.0,
)
# report["action_required"] == True when stock is at or below reorder_point
```

### Automated reorder

```python
result = await svc.automated_reorder(tenant_id="t1")
# Creates ReorderRequest records for all drugs at/below their reorder point
# result["reorders_triggered"] — count of new requests created
```

---

## 11. Drug Recall Management

Process an FDA recall notice. All matching inventory lots are immediately quarantined
and pending/verified dispense orders for affected drugs are put on hold.

```python
recall = await svc.process_drug_recall(
    tenant_id="t1",
    ndc_code="00093-4155-01",
    recall_class="Class_I",
    reason="Potential microbial contamination",
    lot_numbers=["LOT2025A", "LOT2025B"],  # None = all lots
    initiated_by="fda",
)

# recall keys:
# inventory_items_quarantined (int)
# dispense_orders_halted (int)
# patient_notification_required (bool) — True for Class_I
```

---

## 12. Cold Chain Monitoring

```python
await svc.cold_chain_record(
    tenant_id="t1",
    drug_id="drug_insulin",
    temperature_log=[
        {"timestamp": "2026-06-01T08:00:00Z", "temperature_celsius": 4.2, "location": "Fridge-1"},
        {"timestamp": "2026-06-01T12:00:00Z", "temperature_celsius": 8.9, "location": "Fridge-1"},
    ],
    recorded_by="pharmacist_01",
    storage_requirement="2-8C",
)
# Any reading outside valid range triggers an excursion record and quarantine flag
```

---

## 13. Medication Adherence Scoring

Compute PDC (Proportion of Days Covered) and MPR (Medication Possession Ratio) from
dispense history. 180-day lookback by default.

```python
score = await svc.compute_adherence_score(
    tenant_id="t1",
    patient_id="pat_001",
    drug_id=drug.id,
    lookback_days=180,
)

# score["pdc"]                 — 0.0–1.0; ≥ 0.80 is adherent
# score["adherence_category"]  — adherent | partially_adherent | non_adherent
# score["gap_days"]            — total days without medication supply
# score["adherence_alert"]     — True when PDC < 0.80
```

---

## 14. Adverse Drug Event Reporting

```python
ade = await svc.record_adverse_drug_event(
    tenant_id="t1",
    drug_id=drug.id,
    patient_id="pat_001",
    reaction_type="Stevens-Johnson Syndrome",
    severity="severe",
    onset_date=datetime(2026, 6, 1),
    outcome="recovering",
    reported_by="pharmacist_01",
    prescription_id=rx.id,
    causality="probable",
    narrative="Patient developed SJS 5 days after initiating amoxicillin.",
)

# ade["is_serious"] — True for severe/life_threatening/fatal
# ade["medwatch_required"] — True for serious ADEs
# ade["regulatory_submission_status"] — "pending" for serious ADEs
```

Serious ADEs trigger an `serious_ade_expedited_reporting_required` audit event and
a WARNING log entry.

---

## 15. Prior Authorization

```python
from capabilities.healthcare.pha.models import PriorAuthCreate

pa = await svc.request_prior_auth(PriorAuthCreate(
    tenant_id="t1",
    patient_id="pat_001",
    drug_id=drug.id,
    prescription_id=rx.id,
    insurance_id="ins_abc",
    diagnosis_icd10="J06.9",
    requested_by="pharmacist_01",
    clinical_justification="First-line antibiotic for community-acquired URI.",
    created_by="pharmacist_01",
))

# Approve
await svc.approve_prior_auth(tenant_id="t1", pa_id=pa.id, decision_by="ins_reviewer", expires_in_days=90)

# Deny
await svc.deny_prior_auth(tenant_id="t1", pa_id=pa.id, decision_by="ins_reviewer", denial_reason="Not medically necessary")
```

---

## 16. Dispense Label Generation

Generate a structured label payload after the dispense order is verified.

```python
label = await svc.generate_dispense_label(
    tenant_id="t1",
    order_id=order.id,
    lot_number="LOT2025A",
    expiry_date=datetime(2026, 12, 31),
)

# label["drug_display_name"]   — tall-man lettering applied for LASA drugs
# label["barcode_gs1_128"]     — GS1-128 barcode string (01/17/10 application identifiers)
# label["is_lasa"]             — bool; true triggers highlighted label template
# label["auxiliary_labels"]    — list of printed advisory statements
# label["refrigeration_required"] — bool
```

Render via your label template engine (Jinja2, ReportLab, ZPL) using the dict output.

---

## 17. FHIR R4 Export

Export any dispense order as a FHIR R4 MedicationDispense resource.

```python
fhir_resource = await svc.to_fhir_medication_dispense(
    tenant_id="t1",
    order_id=order.id,
)

# fhir_resource is a Python dict that serialises directly to valid FHIR JSON
import json
print(json.dumps(fhir_resource, indent=2))
```

The resource includes: `resourceType`, `id`, `status`, `medicationCodeableConcept`
(RxNorm coded), `subject` (Patient reference), `performer`, `quantity`, and
`whenHandedOver`. The `meta.source` element identifies the originating tenant.

---

## 18. Pharmacist Queue Metrics

Get real-time workload analytics for a pharmacist or the full pharmacy.

```python
metrics = await svc.pharmacist_queue_metrics(
    tenant_id="t1",
    pharmacist_id="pharmacist_01",  # omit for tenant-wide aggregate
)

# metrics["queue_depth"]                — dict of status → count
# metrics["mean_verification_minutes"] — avg receipt-to-verify turnaround
# metrics["p95_verification_minutes"]  — 95th percentile (needs ≥ 20 samples)
# metrics["sla_breach_count"]          — orders exceeding 15-minute SLA
# metrics["counselling_rate"]          — fraction of dispensed orders counselled
```

---

## 19. Reports

### Dispensing summary

```python
from datetime import datetime
report = await svc.dispensing_summary_report(
    tenant_id="t1",
    period_start=datetime(2026, 5, 1),
    period_end=datetime(2026, 5, 31),
)
# report.total_dispenses, top_drugs, counselling_completion_rate
```

### Inventory valuation

```python
report = await svc.inventory_valuation_report(tenant_id="t1")
# report.total_value, expiring_within_30_days, below_reorder_point
```

### Narcotics audit

```python
report = await svc.narcotics_audit_report(
    tenant_id="t1",
    period_start=datetime(2026, 5, 1),
    period_end=datetime(2026, 5, 31),
)
# report.discrepancies_found, witness_compliance_rate
```

### Cold chain compliance

```python
report = await svc.cold_chain_report(
    tenant_id="t1",
    period_start=datetime(2026, 5, 1),
    period_end=datetime(2026, 5, 31),
)
# report.compliance_rate, excursions, affected_drugs
```

---

## 20. Policy Rules Reference

| Rule | Trigger | Effect |
|------|---------|--------|
| `contraindicated_dispense_denied` | dispense + interaction_severity=contraindicated | hard deny |
| `pharmacist_verification_required` | dispense + pharmacist_verified=False | hard deny |
| `recalled_drug_dispense_denied` | dispense + inventory_status=recalled | hard deny |
| `expired_drug_dispense_denied` | dispense + inventory_status=expired | hard deny |
| `controlled_substance_dual_witness_required` | waste + dual_witness_present=False | hard deny |
| `prior_auth_required_for_non_formulary` | dispense + formulary=prior_auth + auth=False | hard deny |
| `non_formulary_requires_override` | dispense + formulary=non_formulary + override=False | hard deny |
| `step_therapy_required` | dispense + formulary=step_therapy + completed=False | hard deny |
| `allergy_anaphylactic_hard_block` | allergy check + severity=anaphylactic | hard block |
| `product_scan_ndc_mismatch` | scan + NDC mismatch | hard block |
| `product_scan_expired_lot` | scan + expiry <= now | hard block |

All hard-deny rules raise `PolicyViolationError` with a machine-readable reason string.

---

## 21. Error Reference

| Exception | Cause | Resolution |
|-----------|-------|------------|
| `PolicyViolationError("pharmacist_verification_required_before_dispense")` | `dispense()` called before verification | Call `pharmacist_verification()` first |
| `PolicyViolationError("cannot_cancel_completed_dispense_order")` | Cancel on dispensed/picked_up order | Only pending/verified orders can be cancelled |
| `PolicyViolationError("only_pending_reorders_can_be_submitted")` | `submit_reorder()` on non-pending request | Check `status` before submitting |
| `PolicyViolationError("narrow_therapeutic_index_drug substitution blocked")` | Substitution for NTI drug without AB rating | Provide clinician override or AB-rated generic |
| `AssertionError("witness_id required for controlled substance dispense")` | CS dispense without witness | Always pass `witness_id` for scheduled drugs |
| `KeyError("dispense order <id> not found")` | Scan or FHIR export on missing order | Verify `order_id` exists for the tenant |
