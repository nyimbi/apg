# Pharmacy Management — World-Class Improvement Plan
© 2025 Datacraft | Author: Nyimbi Odero

## Overview
15 high-impact improvements that move `healthcare_pha` from a well-structured
in-process store toward a production-grade, safety-critical pharmacy runtime.
Ordered by patient-safety impact descending.

---

## 1. Real-Time Drug Interaction Engine with CDS Hooks
**Current state:** Interaction checks query only locally recorded pairs.
**Improvement:** Integrate CDS Hooks (SMART on FHIR) against a live drug knowledge
base (OpenFDA, DrugBank, or RxNorm API). On every `create_dispense_order` call,
fire a CDS Hooks `order-sign` card request to a configurable remote service.
Return structured cards (info / warning / critical / hard-stop) and surface them
in the dispense API response. Cache results with a 24-hour TTL keyed by drug-pair
hash to reduce latency.

**Safety impact:** Catches interactions not yet manually entered into the tenant KB.

---

## 2. Allergy Cross-Check at Dispense
**Current state:** `verify_prescription` marks allergy check as "pass" unconditionally.
**Improvement:** Accept a patient allergy profile (allergen list + reaction severity)
from `healthcare_emr` via an async call. Run cross-checks against drug active
ingredients, excipients, and drug-class membership (e.g., penicillin class). Block
dispense if cross-reactive allergen is detected; surface pharmacist override flow
for mild/moderate reactions.

**Safety impact:** Direct prevention of anaphylactic dispensing errors.

---

## 3. Barcode / 2D Scan Verification at Point of Dispense
**Current state:** No physical product verification step.
**Improvement:** Add `scan_and_verify_product` method that accepts a scanned NDC/GS1
DataMatrix payload and validates it against the dispense order's expected drug, lot,
and expiry before the order transitions to `dispensed`. Mismatches are a hard block;
near-expiry lots (< 7 days) raise a pharmacist acknowledgement gate. Integrate with
`add_audit_scan_event` to build a tamper-evident scan chain.

**Safety impact:** Eliminates wrong-drug and expired-lot dispensing errors.

---

## 4. Medication Adherence Scoring & Refill Prediction
**Current state:** Dispense history is stored but not analysed.
**Improvement:** Add `compute_adherence_score` (MPR/PDC calculation from dispense
history), `predict_refill_date` (expected pickup based on days_supply × fill
history), and `generate_adherence_alert` (patient or prescriber notification when
PDC < 0.8). Persist scores in a `_adherence_scores` store keyed by
`(tenant_id, patient_id, drug_id)`. Feed downstream to `healthcare_ana`.

**Safety impact:** Reduces treatment failure caused by non-adherence.

---

## 5. Dose Range Checking with Weight/Age/Renal Adjustment
**Current state:** Dose appropriateness check is a hard-coded "pass".
**Improvement:** Accept patient weight (kg), age, and renal function (eGFR, CrCl)
as optional context. Implement `check_dose_range` that compares prescribed dose to
dosing tables (loaded from a configurable YAML/JSON knowledge file per drug). Flag
supra-therapeutic doses as "major" and sub-therapeutic doses as "warning". Renal
and hepatic adjusters are applied via a pluggable `DoseAdjuster` protocol.

**Safety impact:** Prevents dosing errors at prescription intake.

---

## 6. Drug Recall Management with Active Lot Quarantine
**Current state:** Inventory items can be set to "recalled" status manually.
**Improvement:** Add `process_fda_recall` that accepts a recall notice (NDC, lot
range, recall class I/II/III) and automatically: (a) quarantines matching inventory
lots, (b) halts in-flight dispense orders for those lots, (c) triggers patient
notifications for already-dispensed recalled lots, (d) generates a recall audit
trail. Integrate with FDA Enforcement API (OpenFDA) for automated recall feed.

**Safety impact:** Prevents dispensing or continued use of recalled drugs.

---

## 7. Automated Perpetual Inventory with Bin-Level Tracking
**Current state:** Inventory is lot-level with manual counts.
**Improvement:** Add bin/location hierarchy (`pharmacy_bin`, `aisle`, `shelf`) to
`InventoryItemResponse`. Implement `perpetual_inventory_update` that adjusts
`quantity_on_hand` on every dispense, return, and waste event atomically (optimistic
locking via `version` field). Add `bin_utilisation_report` and cycle-count scheduler
that selects high-velocity items for daily sub-count.

**Safety impact:** Reduces stock-out risk; enables real-time DEA balance tracking.

---

## 8. Medication Label Generation with ISMP Safe-Label Formatting
**Current state:** No label generation.
**Improvement:** Add `generate_dispense_label` that produces a structured label
payload (Pydantic model) containing: drug name (tall-man for LASA), strength, SIG in
plain language, prescriber, patient, lot, expiry, barcode (GS1-128), ISMP hazardous
drug indicator, and refrigeration/light-sensitivity icons. Output as JSON renderable
by a label template engine. Flag LASA drugs with a distinct visual marker field.

**Safety impact:** ISMP-aligned labels reduce administration errors downstream.

---

## 9. Step-Therapy Pathway Engine
**Current state:** Step-therapy completion is a boolean flag passed by the caller.
**Improvement:** Model step-therapy pathways as ordered drug sequences stored in
`_step_therapy_pathways`. Add `record_step_therapy_trial`, `check_step_completion`,
and `get_required_steps` methods. `check_step_completion` verifies that all
prerequisite drugs have been dispensed for the required duration before allowing the
target drug. Feed completion status to the dispense policy rule engine automatically.

**Safety impact:** Ensures clinical protocol compliance; reduces inappropriate
advanced therapy use.

---

## 10. Pharmacovigilance / Adverse Drug Event Reporting
**Current state:** No ADR capture.
**Improvement:** Add `record_adverse_drug_event` with structured fields:
`drug_id`, `patient_id`, `reaction_type`, `severity` (using MedDRA terminology),
`onset_date`, `outcome`, `causality_assessment` (WHO scale). Implement
`generate_fda_medwatch_report` to produce a structured MedWatch 3500 export.
Forward serious ADRs (death/hospitalisation/congenital anomaly) immediately to a
configurable pharmacovigilance endpoint via async HTTP.

**Safety impact:** Closes the pharmacovigilance feedback loop; enables signal detection.

---

## 11. Compounding Pharmacy Workflow
**Current state:** Compounded drugs exist as a drug type but have no workflow.
**Improvement:** Add `create_compounding_order` with fields: base_drug_ids, final
concentration, volume, sterility_testing_required, beyond_use_date (BUD) calculation
per USP 795/797/800. Add `record_compounding_QC` and `release_compounded_batch`.
Enforce clean-room personnel certification checks before batch release. Track BUD as
a hard expiry on the resulting inventory item.

**Safety impact:** Prevents release of out-of-spec compounded preparations.

---

## 12. Concurrent Event Sourcing Architecture
**Current state:** In-process dicts with last-write-wins semantics.
**Improvement:** Replace mutable dicts with an append-only event journal
(`PharmacyEvent` CloudEvent schema). Each mutating method appends an event and
derives current state by replaying. Projections (current drug state, inventory
totals) are computed by event-sourced read models refreshed via `mqeb` event bus.
This enables audit-quality event history, optimistic concurrency, and replay-based
bug diagnosis.

**Reliability impact:** Eliminates lost-update bugs; enables time-travel queries.

---

## 13. Insurance Claims Integration (NCPDP D.0 / Reject Codes)
**Current state:** No claims or benefit adjudication.
**Improvement:** Add `submit_claim` (NCPDP D.0 transaction model) and
`process_claim_response` (reject code mapping to human-readable reason +
recommended action). Implement real-time benefit check (`rtbc_check`) returning
patient copay, remaining deductible, and formulary tier before dispensing. Reject
codes 70 (product not covered) and 75 (prior auth required) automatically trigger
PA workflow creation.

**Financial impact:** Reduces rejected claims; eliminates manual prior-auth initiation.

---

## 14. Pharmacist Workload & Queue Analytics
**Current state:** Dashboard shows aggregate counts only.
**Improvement:** Add `pharmacist_queue_metrics` returning per-pharmacist turn-around
time (prescription receipt → verification → dispense), queue depth at each stage,
and SLA breach flags (> 15 min verification time). Add `shift_performance_report`
aggregating QC error rate, intervention rate, counselling completion rate, and
counselling language distribution. Use `statistics.mean`/`stdev` for outlier
detection; no external ML dependency required.

**Operational impact:** Identifies bottlenecks and enables workload balancing.

---

## 15. FHIR R4 MedicationDispense Export
**Current state:** All data is internal; no interoperability surface.
**Improvement:** Add `to_fhir_medication_dispense` that maps a `DispenseOrderResponse`
to a FHIR R4 `MedicationDispense` resource (JSON). Map fields: status, medication
(RxNorm coded), subject (patient reference), performer (pharmacist), quantity, dosage
instruction (from SIG), whenHandedOver. Expose via a read-only FHIR endpoint
`GET /fhir/R4/MedicationDispense/<id>`. Enable bulk export via
`GET /fhir/R4/MedicationDispense?patient=<id>` for EMR reconciliation.

**Interoperability impact:** Enables seamless medication data exchange with EMR,
HIE, and payer systems without bespoke ETL.
