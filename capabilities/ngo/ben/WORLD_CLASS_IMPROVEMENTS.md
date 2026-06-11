# Beneficiary Registry — World-Class Improvements

## Overview

Fifteen improvements that elevate `ngo_ben` from a basic registry into a humanitarian-grade beneficiary intelligence platform. Each targets a real gap versus commercial and open-source competitors operating in the humanitarian/social-protection space.

---

### I1. AI-Powered Risk Trajectory Prediction
**Category**: AI/ML
**Justification**: Static vulnerability snapshots miss deteriorating households until a crisis point. Predicting trajectory — not just current state — allows proactive programme targeting, cutting response lag from weeks to days.
**Implementation**: Maintain a rolling time-series of assessment scores per beneficiary and compute a linear trend coefficient; flag households whose composite score is increasing faster than 5 points per 30-day window as "deteriorating" with a projected future score.
**Competitive reference**: UNHCR PRIMES — predictive flagging for at-risk persons of concern

---

### I2. Household Graph Linkage
**Category**: Feature
**Justification**: Households share shocks; treating members as isolated individuals produces duplicate transfers and misses aggregate vulnerability. Household-level aggregation is standard in CGAP/WFP social-protection systems.
**Implementation**: Store a `household_id` on each beneficiary, maintain a `_households` mapping that rolls up member scores and active enrolments, and expose `get_household_summary()` returning aggregate exposure and total programme transfers.
**Competitive reference**: WFP SCOPE — household-centric registration with inter-member linkage

---

### I3. Biometric Hash De-duplication
**Category**: Security
**Justification**: National-ID-based dedup fails when IDs are forged, absent, or reused. Storing a SHA-256 hash of a biometric template enables ghost-beneficiary detection without storing raw biometric data — compliant with GDPR Art. 9 pseudonymisation.
**Implementation**: Accept an optional `biometric_hash` (hex string) on registration; index by hash in `_biometric_index`; raise `duplicate_biometric` on collision before persisting.
**Competitive reference**: Simprints — biometric dedup for humanitarian cash transfers

---

### I4. Consent & Data Minimisation Ledger
**Category**: Compliance
**Justification**: Kenya Data Protection Act 2019, GDPR, and USAID/DFID donor frameworks require explicit consent records, purpose limitation, and right-to-erasure. Absent consent records trigger funding suspension during audits — a material operational risk.
**Implementation**: `record_consent(beneficiary_id, purpose, consent_text, channel)` creates an immutable content-hashed record; `withdraw_consent` triggers cascading soft-purge of data collected under that purpose; consent status gates all bulk data exports.
**Competitive reference**: Salesforce NPSP — per-contact consent with purpose fields

---

### I5. Configurable Weighted Vulnerability Scoring
**Category**: AI/ML
**Justification**: Equal-weight scoring ignores programme-specific sensitivity. A food-security programme should weight food_security 3x more heavily than an education programme. Configurable pillar weights make the score actionable rather than generic.
**Implementation**: Accept a `weights: dict[str, float]` parameter in `create_vulnerability_assessment()`; normalise weights to sum to 1.0; persist both the raw score and the weight configuration used so comparisons across assessment versions remain valid.
**Competitive reference**: IPA Poverty Probability Index (PPI) — instrument-specific weighted scoring

---

### I6. Grievance & Redress Tracking
**Category**: Feature
**Justification**: Beneficiaries who cannot raise complaints are 3x more likely to drop out of programmes. An integrated grievance trail satisfies SPHERE humanitarian standards and provides auditable evidence for donor reporting.
**Implementation**: `raise_grievance(beneficiary_id, category, description, raised_by)` creates a record with SLA deadline derived from severity; `resolve_grievance()` closes with resolution note and SLA elapsed time; `list_open_grievances(days_overdue)` flags SLA breaches.
**Competitive reference**: CommCare (Dimagi) — integrated case/grievance management for social programmes

---

### I7. Multi-Currency Transfer Ledger with FX Snapshot
**Category**: Feature
**Justification**: Programmes funded in USD/EUR disburse in KES/UGX/TZS. Storing only local-currency amounts makes donor reporting inaccurate when exchange rates move; capturing the FX rate at creation time enables both local and source-currency reporting from a single ledger.
**Implementation**: Extend `create_transfer` with `source_currency`, `source_amount: Decimal`, `fx_rate: Decimal`; compute `local_amount = source_amount * fx_rate`; `programme_reach_summary` returns `total_transferred_local` and `total_transferred_source` aggregated correctly.
**Competitive reference**: Red Rose — multi-currency ledger for NGO cash transfer programmes

---

### I8. Recertification & Enrolment Expiry Management
**Category**: Compliance
**Justification**: USAID and ECHO frameworks require periodic re-verification of beneficiary eligibility. Without automated expiry tracking, programmes overpay ineligible beneficiaries and accumulate compliance findings that risk programme closure.
**Implementation**: Enrolments gain `valid_until` and `recertification_due` fields; `list_recertification_due(days_ahead)` returns expiring enrolments; `recertify_enrolment()` re-runs eligibility and either extends or terminates the enrolment.
**Competitive reference**: UN OCHA ReliefWeb — case renewal workflow with annual re-targeting requirement

---

### I9. Batch Disbursement with Maker-Checker Approval
**Category**: Feature
**Justification**: Manual one-by-one transfer approvals bottleneck monthly payment runs. A batch record with two-person authorisation (maker ≠ checker) enables bulk M-Pesa API submission, single-event audit trail, and payment-file reconciliation.
**Implementation**: `create_disbursement_batch(transfer_ids, batch_reference, submitted_by)` creates a batch in `pending_approval`; `approve_batch(batch_id, approved_by)` enforces approver != submitter; `mark_batch_processed()` bulk-confirms all constituent transfers atomically.
**Competitive reference**: Segovia/ThitsaWorks — batch M-Pesa disbursement for NGOs

---

### I10. Longitudinal Outcome Tracking
**Category**: Feature
**Justification**: Donors require evidence of impact, not just output counts. Tracking outcomes (school attendance, income, nutrition) over time against the vulnerability baseline enables difference-in-difference impact calculations satisfying DFID and Mastercard Foundation reporting standards.
**Implementation**: `record_outcome(beneficiary_id, outcome_type, value, measured_at, measured_by)` stores a time-series record; `outcome_trajectory()` returns sorted series with baseline delta; `programme_impact_report()` aggregates mean baseline-to-endline deltas per outcome type.
**Competitive reference**: GiveDirectly — longitudinal panel data collection for impact evidence

---

### I11. Predictive Attrition Scoring
**Category**: AI/ML
**Justification**: Programme dropout costs NGOs $150–400 per beneficiary in re-enrolment and catch-up transfers. Predicting dropout risk from assessment trends, transfer gaps, and grievance history enables proactive retention outreach.
**Implementation**: `predict_attrition_risk(beneficiary_id)` scores 0–1 from: days since last transfer, vulnerability trend direction, open grievances count, and assessment age; returns `{risk_score, risk_tier, key_factors}`; pure logistic scoring on feature deltas — no external model required.
**Competitive reference**: Mercy Corps — internal beneficiary retention scoring model

---

### I12. Offline-First Sync Protocol
**Category**: UX
**Justification**: 60–80% of field data collection happens in areas with no internet connectivity. A sync protocol that queues mutations and resolves conflicts on reconnect enables field staff to work uninterrupted while maintaining data integrity.
**Implementation**: `export_sync_bundle(last_sync_at)` returns a compact newline-delimited JSON delta of records changed since the timestamp; `apply_sync_bundle(bundle)` idempotently applies mutations using `updated_at` as conflict tiebreaker; bundle is streamable for low-bandwidth connections.
**Competitive reference**: KoBoToolbox — offline data collection with delta-sync

---

### I13. Programme Eligibility Rules Engine
**Category**: Feature
**Justification**: Field officers applying inconsistent targeting judgements cause 30–50% targeting errors in community-level programmes. A declarative rules engine enforces criteria uniformly and produces an auditable eligibility record per beneficiary.
**Implementation**: Store per-programme rules as JSON predicate trees; `evaluate_eligibility(beneficiary_id, programme_id)` walks the tree against beneficiary fields and returns `{eligible, reasons, score}`; predicates support threshold, range, set-membership, and logical AND/OR operators.
**Competitive reference**: WFP SCOPE — configurable targeting criteria with eligibility audit trail

---

### I14. Duplicate-Merge Workflow
**Category**: Feature
**Justification**: Detection without resolution is incomplete. Caseworkers need to merge two duplicate records into one canonical record, preserving history from both sides. Without this, duplicate flags accumulate without consequence.
**Implementation**: `merge_beneficiaries(primary_id, duplicate_id, merged_by)` re-parents all enrolments, assessments, and transfers from the duplicate to the primary, soft-deletes the duplicate with `status="merged"`, and emits a `beneficiary_merged` audit event with full field diff.
**Competitive reference**: Salesforce Health Cloud — patient/member dedup and merge with lineage preservation

---

### I15. Exit Outcome Classification
**Category**: Compliance
**Justification**: Donors and impact evaluators need to know whether beneficiaries exit having achieved self-reliance or because they dropped out or died. Exit reasons tied to outcome codes enable graduation-model reporting (BRAC, CGAP).
**Implementation**: Extend `exit_beneficiary()` to accept a structured `exit_outcome` from an enum (`graduated`, `relocated`, `deceased`, `dropout`, `ineligible`, `transferred_out`); `programme_graduation_report()` returns outcome-code breakdown per programme.
**Competitive reference**: BRAC Ultra-Poor Graduation Programme — structured exit-outcome classification for impact reporting
