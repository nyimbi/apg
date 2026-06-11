# Land Registry — World-Class Improvements

## Overview

15 improvements drawn from best-in-class land administration systems: NLIMS (Kenya), Torrens
title systems (Australia/New Zealand), England & Wales HM Land Registry, Singapore SLA,
Rwanda NLDIMS, and FIG/UN-Habitat benchmarks.

---

### I1. Parcel Subdivision and Amalgamation
**Category:** Cadastral Operations
**Justification:** Every production registry must support subdividing a parent parcel into
child parcels (e.g. 10 ha → two 5 ha plots) and merging adjacent parcels. Without it the
cadastre becomes stale as estate development proceeds.
**Implementation:** `subdivide_parcel(parent_id, child_parcels[])` validates child areas sum
≤ parent area, creates child parcel records, marks parent `subdivided`, and emits audit
events. Enforces encumbrance propagation and survey reference attachment.
**Competitor Reference:** Kenya NLIMS subdivision module; Torrens sub-lot creation in NSW
Land Registry Services.

---

### I2. Stamp Duty and Transfer Tax Computation
**Category:** Revenue & Compliance
**Justification:** Every land transfer in Kenya triggers Stamp Duty (4% residential, 2%
agricultural per Stamp Duty Act Cap 480) and CGT. Registrars cannot complete a transfer
without a duty payment receipt. Missing this forces external tax lookups and manual
reconciliation.
**Implementation:** `compute_stamp_duty(transfer_id, consideration_kes, land_use)` returns
breakdown: `stamp_duty_kes`, `cgt_kes`, `registration_fee_kes`, `total_payable_kes` using
`Decimal` arithmetic and tiered rate tables. `record_duty_payment(transfer_id, ...)` validates
payment and marks transfer `duty_paid`.
**Competitor Reference:** Kenya Revenue Authority iTax integration in NLIMS; UK SDLT
calculator in HM Land Registry portal.

---

### I3. Geospatial Boundary Management (GeoJSON)
**Category:** Spatial Data
**Justification:** Coordinates stored as freeform dicts are not queryable. International
standards (ISO 19125, OGC) mandate GeoJSON for polygon boundaries. Boundary overlaps are the
primary source of title fraud and ownership disputes in Kenya (NLC 2023).
**Implementation:** `set_parcel_boundary(parcel_id, geojson_polygon)` validates geometry
(closed ring, ≥4 points), checks for overlap via bounding-box pre-filter, stores normalised
GeoJSON, and emits `boundary_set`. `get_parcel_boundary(parcel_id)` returns stored geometry.
**Competitor Reference:** Rwanda NLDIMS GIS layer; Singapore SLA OneMap polygon registry.

---

### I4. Title History / Chain of Ownership
**Category:** Provenance
**Justification:** Legal due diligence requires the full ownership chain (folio history). Without
it advocates cannot satisfy the 12-year limitation period check required under the Land
Registration Act 2012, s.25. Current service only tracks the current owner.
**Implementation:** `get_title_chain(parcel_id)` traverses transfers in chronological order,
assembles `[{from_owner, to_owner, consideration_kes, date, instrument}]`, prepends the
original issue record, and returns the full chain with a SHA-256 integrity hash.
**Competitor Reference:** England & Wales HM Land Registry "title register" folio; Torrens
indefeasibility chain.

---

### I5. Lease Management and Renewal Workflow
**Category:** Tenure Administration
**Justification:** Leasehold is the dominant tenure in Nairobi (>60% of titled parcels, NLC
2022). Leases expire; the service stored `lease_term_years` but had no lifecycle — no expiry
calculation, no renewal notice, no extension record.
**Implementation:** `register_lease(title_id, lessee_id, start_date, term_years,
annual_rent_kes)` calculates `expiry_date`. `renew_lease(lease_id, extension_years,
new_rent_kes)` extends expiry and recalculates total rent. All monetary fields use `Decimal`.
**Competitor Reference:** Kenya NLC lease management; Dubai DLD lease registration system.

---

### I6. Caution and Restriction Workflow
**Category:** Risk & Fraud Prevention
**Justification:** Cautions are widely abused in Kenya to stall fraudulent transfers. NLC
reported 14,000 active cautions in 2023. A structured lifecycle (lodge → confirm/withdraw →
expire) with automatic 60-day expiry is required by LRA 2012, s.71–73.
**Implementation:** `lodge_caution(title_id, grounds, expiry_days=60)` creates the caution.
`confirm_caution(caution_id, court_order_ref)` upgrades to permanent restriction.
`withdraw_caution(caution_id, reason)` terminates it. `expire_stale_cautions()` bulk-expires
overdue lodged cautions.
**Competitor Reference:** HM Land Registry Form CT1 caution workflow; Kenya LRA §71.

---

### I7. Bulk Import / Systematic Registration
**Category:** Onboarding at Scale
**Justification:** Kenya's systematic land registration programme (SLRP) targets 6 million
unregistered parcels. Batch import via list is the only viable onboarding path; one-by-one
registration is operationally impossible.
**Implementation:** `bulk_register_parcels(parcels: list[dict]) -> BulkResult` processes in
batches, wraps each in try/except, returns `{succeeded, failed, total, success_count,
error_count}`. Validates duplicates within the batch before persisting.
**Competitor Reference:** Rwanda NLDIMS systematic demarcation batch loader; FAO VGGT bulk
registration guidance.

---

### I8. Title Rectification (Error Correction)
**Category:** Data Integrity
**Justification:** LRA 2012, s.86 provides a formal rectification procedure for titles with
errors (wrong area, mis-spelt names, incorrect parcel reference). The original `update_title`
overwrote data silently. Rectification must carry a registrar authority reference.
**Implementation:** `rectify_title(title_id, corrections, authority_reference, rectified_by)`
snapshots the pre-rectification record in audit events, applies only allowed fields, and emits
`title_rectified` with a full before/after diff.
**Competitor Reference:** Torrens rectification under NSW RPA s.136; HM Land Registry AP1
form rectification.

---

### I9. Land Use Change / Rezoning Workflow
**Category:** Planning Integration
**Justification:** Land use changes require county planning approval in Kenya (Physical and
Land Use Planning Act 2019). The original `update_parcel` silently accepted `land_use`
changes, enabling illegal conversions.
**Implementation:** `apply_land_use_change(parcel_id, proposed_use, planning_ref,
applicant_id)` creates a `pending` record. `approve_land_use_change(change_id, approved_by,
conditions)` validates the planning reference and updates the parcel.
**Competitor Reference:** Kenya Physical Planning department integration; Singapore URA
rezoning gateway.

---

### I10. Stamp Duty Exemptions Registry
**Category:** Revenue Accuracy
**Justification:** Kenya Stamp Duty Act provides exemptions for first-time homebuyers,
government, charities, and inheritance. Without exemption tracking, duty refunds and disputes
clog the courts.
**Implementation:** `register_exemption(transfer_id, exemption_type, statutory_basis,
supporting_docs_ref, granted_by)` creates an exemption record that allows zero-duty payments
in `record_duty_payment`. Valid types: `first_time_buyer`, `government`, `ngo`, `inheritance`,
`court_order`.
**Competitor Reference:** KRA iTax exemption codes; HMRC SDLT exemption schedule.

---

### I11. Property Rates Ledger and Arrears Tracking
**Category:** Revenue Management
**Justification:** County governments collect land rates under the Rating Act (Cap 267). The
original `assess_land_rates` created an assessment but had no payment tracking, arrears
accumulation, or penalty interest. Counties lose billions annually due to uncollected rates
(OCOB 2023).
**Implementation:** `record_rates_payment(assessment_id, amount_paid_kes, payment_date,
receipt_number)` tracks partial/full payments. `compute_rates_arrears(parcel_id, as_of_date)`
calculates outstanding principal plus 2%/month statutory penalty. All arithmetic uses
`Decimal`.
**Competitor Reference:** Nairobi City County e-rates portal; Cape Town rates management
system.

---

### I12. Survey Plan Registry and Surveyor Licensing
**Category:** Professional Oversight
**Justification:** Every new parcel, subdivision, or boundary dispute requires a licensed
surveyor's plan deposited with the Director of Surveys (Survey Act Cap 299). The service had
`surveyors`/`survey_plans` dicts but no methods.
**Implementation:** `register_surveyor(surveyor_id, name, licence_number, expiry_date)`
registers the professional. `deposit_survey_plan(parcel_id, surveyor_id, plan_number, ...)
validates licence currency before accepting the plan. `list_survey_plans(parcel_id)` supports
cadastral verification.
**Competitor Reference:** Kenya Survey Act §33; Singapore SLA Integrated Survey System.

---

### I13. Spousal Consent and Co-ownership Enforcement
**Category:** Gender Equity
**Justification:** LRA 2012, s.93 and Matrimonial Property Act 2013, s.12 require spousal
consent for transfer of matrimonial property. The World Bank Land Governance Assessment (2023)
identified spousal consent bypass as the leading cause of gender-based land dispossession in
Kenya.
**Implementation:** `register_spousal_consent(title_id, spouse_id, ...)` creates a consent
record. `flag_matrimonial_property(title_id, reason)` marks the title; `initiate_transfer`
blocks if no consent record exists for flagged titles.
**Competitor Reference:** Kenya LRA §93; Rwanda co-ownership provisions; Tanzania Land Act
§161.

---

### I14. Dispute Escalation and Tribunal Integration
**Category:** Conflict Resolution
**Justification:** Adjudication decisions are appealed to the Land Dispute Tribunal and
Environment & Land Court. The original `decide_adjudication` was terminal. Kenya's land courts
had 68,000 pending cases in 2023 largely because registries cannot track appeal status
(Judiciary Annual Report).
**Implementation:** `escalate_adjudication(adjudication_id, escalation_type, tribunal_ref,
grounds)` links to the original record. `record_tribunal_decision(escalation_id, decision,
judgement_ref)` updates both escalation and adjudication. `list_escalations` surfaces the
pipeline.
**Competitor Reference:** Kenya ELC integration; Trinidad & Tobago Land Tribunal API.

---

### I15. Title Certificate Generation (PDF Metadata)
**Category:** Document Issuance
**Justification:** The end product of land registration is a title deed document. The service
emitted data records but had no document generation step. Digital title certificates are
mandated under the NLIMS rollout.
**Implementation:** `generate_title_certificate(title_id, generated_by)` assembles a
structured `certificate_payload` containing all title metadata, parcel details, active
encumbrances, a QR-code seed (SHA-256 of title_id + tenant_id), and a digital signature
placeholder. Pure data assembly — no PDF library dependency.
**Competitor Reference:** Kenya NLIMS digital title; NSW LRS eDeed; UK HM Land Registry
digital title document.
